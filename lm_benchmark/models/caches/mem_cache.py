# Copyright 2023 Amirkeivan Mohtashami, Martin Jaggi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import torch

from .cache import LMCache, LMCacheStorage

class MemLMCacheStorage(LMCacheStorage):

    def __init__(self, config, layer):
        super().__init__(config, layer)
        n_embd_per_head = config.n_embd // config.n_head
        landmark_selection = getattr(config, 'landmark_selection', 'positional')
        self.is_content_aware = landmark_selection != 'positional'
        # max_block_size keeps buffer shape consistent across modes; positional uses
        # all max_block_size slots (50 content + 1 inserted landmark); content-aware
        # uses only the first block_size slots and pads the trailing slot with zeros.
        self.max_block_size = config.mem_cache_freq + 1
        self.block_size = config.mem_cache_freq if self.is_content_aware else (config.mem_cache_freq + 1)
        self.register_buffer("cache_mem_k", torch.empty((config.batch_size, config.mem_cache_size, self.max_block_size, config.n_head, n_embd_per_head)), persistent=False)
        self.register_buffer("cache_mem_v", torch.empty((config.batch_size, config.mem_cache_size, self.max_block_size, config.n_head, n_embd_per_head)), persistent=False)
        # Per-block original within-block landmark position. For positional mode this
        # is always block_size - 1 (and never read); for content-aware it records the
        # pre-permutation index so RoPE can be applied with the correct PE position.
        self.register_buffer("cache_mem_landmark_pos", torch.zeros((config.batch_size, config.mem_cache_size), dtype=torch.long), persistent=False)
        self.register_buffer("last_incomplete_k", torch.empty((config.batch_size, config.n_head, self.max_block_size, n_embd_per_head)), persistent=False)
        self.register_buffer("last_incomplete_v", torch.empty((config.batch_size, config.n_head, self.max_block_size, n_embd_per_head)), persistent=False)
        self.register_buffer("last_incomplete_ismem", torch.empty((config.batch_size, self.max_block_size), dtype=torch.bool), persistent=False)
        self.last_incomplete_len = 0
        self.cache_iter = 0
        self.cache_size = 0
        self.clear_state()

    def clear_state(self):
        self.last_incomplete_len = 0
        self.cache_iter = 0
        self.cache_size = 0
        if hasattr(self, 'cache_mem_landmark_pos'):
            self.cache_mem_landmark_pos.zero_()

    def retrieve_for_query(self, q, cache_context, pos_emb_closure, start_index):
        B, nh, T, hs = q.size()
        block_size = self.block_size
        last_incomplete_k = pos_emb_closure.adapt_keys(self.last_incomplete_k[:B, :, :self.last_incomplete_len], start_index=start_index - self.last_incomplete_len)
        att_incomplete = (q @ last_incomplete_k.transpose(-2, -1)) * (1.0 / math.sqrt(last_incomplete_k.size(-1)))
        last_incomplete_v = self.last_incomplete_v[:B, :, :self.last_incomplete_len].unsqueeze(2).expand(B, nh, T, self.last_incomplete_len, hs)
        last_incomplete_mem = self.last_incomplete_ismem[:B, :self.last_incomplete_len]

        if self.cache_size == 0 or self.config.cache_topk == 0:
            return att_incomplete, {'v': last_incomplete_v.clone(), 'is_mem': last_incomplete_mem.clone()}

        top_k = self.config.cache_topk
        # The landmark slot in storage is always block_size - 1 (last meaningful slot).
        # For positional, that is mem_cache_freq (the inserted token slot). For
        # content-aware it is mem_cache_freq - 1 because the swap moved the landmark
        # there. Slots from block_size .. max_block_size-1 are zero padding for the
        # content-aware case and are never read.
        k_with_cached_mem = self.cache_mem_k[:B, :self.cache_size, block_size - 1].view(B, -1, nh, hs).transpose(1, 2)
        mem_indices = torch.cat((
            torch.arange(k_with_cached_mem.shape[2] - self.cache_iter, k_with_cached_mem.shape[2], device=q.device),
            torch.arange(0, k_with_cached_mem.shape[2] - self.cache_iter, device=q.device),
        )).unsqueeze(0).expand(B, -1)

        mem_indices = torch.where(mem_indices >= (k_with_cached_mem.shape[2] - top_k), mem_indices - (k_with_cached_mem.shape[2] - top_k) + 1, 0)

        if self.is_content_aware:
            # Use the original within-block landmark position so RoPE on the
            # landmark key matches what the model would have seen pre-permutation.
            lm_pos_for_blocks = self.cache_mem_landmark_pos[:B].gather(1, mem_indices)
            mem_token_indices = mem_indices * block_size + lm_pos_for_blocks
        else:
            mem_token_indices = mem_indices * block_size + (block_size - 1)

        k_with_cached_mem = pos_emb_closure.adapt_keys(k_with_cached_mem, indices=mem_token_indices.unsqueeze(1).expand(B, nh, -1))
        mem_att = (q @ k_with_cached_mem.transpose(-2, -1)) * (1.0 / math.sqrt(k_with_cached_mem.size(-1)))
        mem_att = torch.nn.functional.softmax(mem_att, dim=-1)
        if self.config.cache_selection_method == "max_over_heads":
            mem_att = mem_att.amax(dim=1).unsqueeze(1).expand(B, nh, T, -1)
        elif self.config.cache_selection_method == "per_token_and_head":
            pass
        elif self.config.cache_selection_method == "max_over_tokens":
            mem_att = mem_att.amax(dim=2, keepdim=True).expand(B, nh, T, -1)
        else:
            raise NotImplementedError

        mem_att = mem_att.topk(dim=-1,k=top_k)[1]
        if top_k > 1:
            mem_att = torch.where(
                mem_att < self.cache_iter,
                (k_with_cached_mem.shape[2] - self.cache_iter) + mem_att,
                mem_att - self.cache_iter)
            mem_att = mem_att.sort(dim=-1)[0]
            mem_att = torch.where(
                mem_att >= (k_with_cached_mem.shape[2] - self.cache_iter),
                mem_att - (k_with_cached_mem.shape[2] - self.cache_iter),
                mem_att + self.cache_iter)

        mem_att_or = mem_att
        # Use block_size (not max_block_size) so content-aware retrieval excludes the
        # padding slot. For positional, block_size == max_block_size so this is a no-op.
        mem_att = mem_att.permute(0, 2, 3, 1).view(B, T, top_k, 1, nh, 1).expand(B, T, top_k, block_size, nh, hs)
        mem_k = self.cache_mem_k[:B, :, :block_size].unsqueeze(1).expand(B, T, -1, -1, -1, -1).take_along_dim(mem_att, dim=2).view(B, T, -1, nh, hs)
        mem_k = mem_k.permute((0, 3, 1, 2, 4))
        mem_v = self.cache_mem_v[:B, :, :block_size].unsqueeze(1).expand(B, T, -1, -1, -1, -1).take_along_dim(mem_att, dim=2).view(B, T, -1, nh, hs)
        mem_v = mem_v.permute((0, 3, 1, 2, 4))

        chosen_indices = mem_indices.view(B, 1, 1, -1).expand(B, nh, T, -1).take_along_dim(mem_att_or, dim=-1)

        if self.is_content_aware:
            # Build per-(B, nh, T, top_k, slot) within-block PE indices that undo the
            # storage swap: physical slot block_size-1 holds the landmark whose
            # original position was lm_pos; physical slot lm_pos holds the displaced
            # token whose original position was block_size-1; all other slots map to
            # themselves. The swap is its own inverse, so the formula matches the
            # one used in store_in_cache.
            chosen_lm_pos = self.cache_mem_landmark_pos[:B].view(B, 1, 1, -1).expand(B, nh, T, -1).take_along_dim(mem_att_or, dim=-1)
            slot_arange = torch.arange(block_size, device=q.device).view(1, 1, 1, 1, block_size)
            chosen_lm_pos_exp = chosen_lm_pos.unsqueeze(-1)
            last_slot = block_size - 1
            within_block_pos = torch.where(
                slot_arange == last_slot,
                chosen_lm_pos_exp.expand(B, nh, T, top_k, block_size),
                torch.where(
                    slot_arange == chosen_lm_pos_exp,
                    torch.full((B, nh, T, top_k, block_size), last_slot, dtype=torch.long, device=q.device),
                    slot_arange.expand(B, nh, T, top_k, block_size),
                ),
            )
            block_offset = ((chosen_indices > 0) * block_size).unsqueeze(-1)
            k_indices = (block_offset + within_block_pos).view(B, nh, T, -1)
        else:
            k_indices = torch.arange(0, block_size * top_k, device=q.device)
            k_indices = (((chosen_indices > 0) * block_size).unsqueeze(-1) + k_indices.view(1, 1, 1, top_k, -1)).view(B, nh, T, -1)

        # is_mem stays a flat (B, top_k * block_size) mask with True only at the last
        # slot of each block; the swap guarantees the landmark is physically there.
        is_mem = torch.cat((q.new_zeros((B, top_k, block_size - 1), dtype=torch.bool), q.new_ones((B, top_k, 1), dtype=torch.bool)), dim=-1).view(B, -1)

        mem_k = pos_emb_closure.adapt_keys(mem_k.reshape(B, nh, -1, hs), indices=k_indices.reshape(B, nh, -1)).view(B, nh, T, -1, hs)
        att_k = (q.unsqueeze(3) @ mem_k.transpose(-2, -1)).squeeze(3) * (1.0 / math.sqrt(mem_k.size(-1)))
        att_k = pos_emb_closure.adapt_attention_before_softmax(att_k, start_query_index=start_index, k_indices=k_indices)

        att_prefix = torch.cat((att_k, att_incomplete), dim=-1)
        v_prefix = torch.cat((mem_v, last_incomplete_v), dim=-2)
        is_mem_prefix = torch.cat((is_mem, last_incomplete_mem), dim=-1)

        return att_prefix, {'v': v_prefix, 'is_mem': is_mem_prefix}

    def store_in_cache(self, keys, values_dict, **kwargs):

        B, nh, T, hs = keys.size()
        k = torch.cat((self.last_incomplete_k[:B, :, :self.last_incomplete_len], keys), dim=-2)
        v = torch.cat((self.last_incomplete_v[:B, :, :self.last_incomplete_len], values_dict['v']), dim=-2)
        is_mem = torch.cat((self.last_incomplete_ismem[:B, :self.last_incomplete_len], values_dict['is_mem']), dim=-1)
        B, nh, T, hs = k.size()

        block_size = self.block_size
        max_block_size = self.max_block_size
        incomplete_len = T % block_size
        full_len = T - incomplete_len
        k, incomplete_k = torch.split(k, (full_len, incomplete_len), dim=-2)
        v, incomplete_v = torch.split(v, (full_len, incomplete_len), dim=-2)
        is_mem, incomplete_ismem = torch.split(is_mem, (full_len, incomplete_len), dim=-1)
        self.last_incomplete_k[:B, :, :incomplete_len].copy_(incomplete_k)
        self.last_incomplete_v[:B, :, :incomplete_len].copy_(incomplete_v)
        self.last_incomplete_ismem[:B, :incomplete_len].copy_(incomplete_ismem)
        self.last_incomplete_len = incomplete_len
        T = full_len
        assert T % block_size == 0
        is_mem_for_cache = is_mem.view(B, -1, block_size)

        added_size = is_mem_for_cache.shape[1]
        k_for_cache = k.transpose(1, 2).view(B, added_size, block_size, nh, hs)
        v_for_cache = v.transpose(1, 2).view(B, added_size, block_size, nh, hs)

        if self.is_content_aware:
            assert is_mem_for_cache.sum(dim=-1).eq(1).all(), \
                "Each block must have exactly one landmark for content-aware mode"
            lm_pos = is_mem_for_cache.int().argmax(dim=-1)  # (B, num_blocks)

            # Permutation that swaps landmark slot with last slot:
            # destination slot i gets source slot perm[i].
            slot_arange = torch.arange(block_size, device=keys.device).view(1, 1, block_size)
            lm_pos_exp = lm_pos.unsqueeze(-1)
            last_slot = block_size - 1
            perm = torch.where(
                slot_arange == last_slot,
                lm_pos_exp.expand(B, added_size, block_size),
                torch.where(
                    slot_arange == lm_pos_exp,
                    torch.full((B, added_size, block_size), last_slot, dtype=torch.long, device=keys.device),
                    slot_arange.expand(B, added_size, block_size),
                ),
            )

            perm_kv = perm.view(B, added_size, block_size, 1, 1).expand(-1, -1, -1, nh, hs)
            k_for_cache = k_for_cache.gather(2, perm_kv)
            v_for_cache = v_for_cache.gather(2, perm_kv)
            is_mem_for_cache = is_mem_for_cache.gather(2, perm)

            # Post-swap invariant: landmark is physically at the last slot.
            assert is_mem_for_cache[..., -1].all()
            assert not is_mem_for_cache[..., :-1].any()

            if block_size < max_block_size:
                pad_size = max_block_size - block_size
                pad_kv = k_for_cache.new_zeros((B, added_size, pad_size, nh, hs))
                k_for_cache = torch.cat((k_for_cache, pad_kv), dim=2)
                v_for_cache = torch.cat((v_for_cache, v_for_cache.new_zeros((B, added_size, pad_size, nh, hs))), dim=2)
        else:
            assert is_mem_for_cache[..., -1].all()
            assert not is_mem_for_cache[..., :-1].any()
            lm_pos = torch.full((B, added_size), block_size - 1, dtype=torch.long, device=keys.device)

        is_mem_for_cache = is_mem_for_cache[:, -self.cache_mem_k.shape[1]:]
        k_for_cache = k_for_cache[:, -self.cache_mem_k.shape[1]:]
        v_for_cache = v_for_cache[:, -self.cache_mem_k.shape[1]:]
        lm_pos = lm_pos[:, -self.cache_mem_k.shape[1]:]
        self.cache_iter = (self.cache_iter + added_size - is_mem_for_cache.shape[1]) % self.cache_mem_k.shape[1]
        self.cache_size += added_size - is_mem_for_cache.shape[1]
        added_size = is_mem_for_cache.shape[1]

        if self.cache_iter + added_size >= self.cache_mem_k.shape[1]:
            next_iter = (self.cache_iter + added_size) - self.cache_mem_k.shape[1]
            rem = (self.cache_mem_k.shape[1] - self.cache_iter)
            self.cache_mem_k[:B, :next_iter].copy_(k_for_cache[:, rem:])
            self.cache_mem_k[:B, self.cache_iter:].copy_(k_for_cache[:, :rem])
            self.cache_mem_v[:B, :next_iter].copy_(v_for_cache[:, rem:])
            self.cache_mem_v[:B, self.cache_iter:].copy_(v_for_cache[:, :rem])
            self.cache_mem_landmark_pos[:B, :next_iter].copy_(lm_pos[:, rem:])
            self.cache_mem_landmark_pos[:B, self.cache_iter:].copy_(lm_pos[:, :rem])
        else:
            next_iter = self.cache_iter + added_size
            self.cache_mem_k[:B, self.cache_iter:next_iter].copy_(k_for_cache)
            self.cache_mem_v[:B, self.cache_iter:next_iter].copy_(v_for_cache)
            self.cache_mem_landmark_pos[:B, self.cache_iter:next_iter].copy_(lm_pos)

        self.cache_iter = next_iter
        self.cache_size += added_size


class MemLMCacheContext(object):
    def __init__(self):
        self.group_prob = None


class MemLMCache(LMCache):

    def __init__(self, config):
        super().__init__(config)
        self.last_incomplete_len = 0
        self.total_len = 0
        landmark_selection = getattr(config, 'landmark_selection', 'positional')
        self.is_content_aware = landmark_selection != 'positional'
        self.max_block_size = config.mem_cache_freq + 1
        self.block_size = config.mem_cache_freq if self.is_content_aware else (config.mem_cache_freq + 1)

    def get_cache_storage(self):
        return MemLMCacheStorage

    def get_context_class(self):
        return MemLMCacheContext

    def forward(self, x):
        if self.is_content_aware:
            # Pass-through: do not insert any landmark token. The model selects
            # landmarks from existing tokens via norm/centrality, and the storage
            # layer permutes them to a fixed slot at write time.
            previous_incomplete_len = self.last_incomplete_len
            B, T = x.size()
            prev_total_len = self.total_len
            self.total_len += T
            self.last_incomplete_len = (previous_incomplete_len + T) % self.config.mem_cache_freq
            start_index = min(prev_total_len // self.config.mem_cache_freq, (self.config.cache_topk + 1)) * self.config.mem_cache_freq + previous_incomplete_len
            return x, start_index, self.context_class()

        previous_incomplete_len = self.last_incomplete_len
        B, T = x.size()
        incomplete_placeholder = x.new_full((B, previous_incomplete_len), -1)
        x = torch.cat((incomplete_placeholder, x), dim=-1)
        B, T = x.size()
        incomplete_len = T % self.config.mem_cache_freq
        full_len = T - incomplete_len
        mem_x, incomplete_x = torch.split(x, (full_len, incomplete_len), dim=-1)
        mem_x = mem_x.view(B, -1, self.config.mem_cache_freq)
        mem_x = torch.cat((mem_x, mem_x.new_full((mem_x.shape[0], mem_x.shape[1], 1), self.config.landmark_id)), dim=-1)
        x = torch.cat((mem_x.view(B, -1), incomplete_x), dim=-1)[:, previous_incomplete_len:]
        self.last_incomplete_len = incomplete_len
        prev_total_len = self.total_len
        self.total_len += x.shape[1]
        start_index = min(prev_total_len // (self.config.mem_cache_freq + 1), (self.config.cache_topk + 1)) * (self.config.mem_cache_freq + 1) + previous_incomplete_len
        return x, start_index, self.context_class()

    def get_final_logits(self, x):
        if self.is_content_aware:
            # No tokens were inserted, so nothing to strip.
            return x
        B, T, C = x.size()
        incomplete_len = self.last_incomplete_len
        T_with_mem = T - incomplete_len
        if T_with_mem <= 0:
            incomplete_len = T
            T_with_mem = 0
            x, incomplete = torch.split(x, (0, T), dim=1)
            previous_incomplete_len = -T_with_mem
        else:
            x, incomplete = torch.split(x, (T_with_mem, incomplete_len), dim=1)
            previous_incomplete_len = (self.config.mem_cache_freq + 1 - T_with_mem % (self.config.mem_cache_freq + 1)) % (self.config.mem_cache_freq + 1)
        incomplete_placeholder = x.new_full((B, previous_incomplete_len, C), -1)
        x = torch.cat((incomplete_placeholder, x), dim=1).view(B, -1, self.config.mem_cache_freq + 1, C)
        x = x[:, :, :-1].reshape(B, -1, C)[:, previous_incomplete_len:]
        return torch.cat((x, incomplete), dim=1)

    def clear_state(self):
        super().clear_state()
        self.last_incomplete_len = 0
        self.total_len = 0
