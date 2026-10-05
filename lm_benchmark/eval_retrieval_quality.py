#!/usr/bin/env python3
"""Landmark Retrieval Quality Experiment"""
import os
import sys
import json
import torch
import tiktoken
import numpy as np
import csv
from collections import Counter
from pathlib import Path
sys.path.insert(0, os.path.expanduser("~/landformer/LandFORMER/lm_benchmark"))
import config
import models

def load_model(checkpoint_dir, device):
    checkpoint_path = checkpoint_dir
    checkpoint_filename = "ckpt.pt"
    if os.path.isfile(checkpoint_path):
        checkpoint_path, checkpoint_filename = os.path.split(checkpoint_path)
    with open(os.path.join(checkpoint_path, "summary.json")) as f:
        summary = json.load(f)
    args_dict = summary["args"]
    class Args:
        pass
    model_args = Args()
    for k, v in args_dict.items():
        setattr(model_args, k, v)
    model_args.device = torch.device(device)
    model_args.dtype = torch.bfloat16
    if not hasattr(model_args, 'use_cache'):
        model_args.use_cache = False
    if not hasattr(model_args, 'lm_cache'):
        model_args.lm_cache = 'none'
    model = models.make_model_from_args(model_args).to(model_args.device)
    ckpt = torch.load(os.path.join(checkpoint_path, checkpoint_filename), map_location="cpu")
    model.load_state_dict(
        {k: v for k, v in ckpt["model"].items() if "attn.bias" not in k and "wpe" not in k},
        strict=False
    )
    model.eval()
    return model, model_args

def tokenize_book_with_landmarks(text, tokenizer, landmark_id=50260, mem_freq=50):
    raw_ids = tokenizer.encode(text)
    result = []
    block_token_ids = []
    current_block = []
    for tok in raw_ids:
        current_block.append(tok)
        result.append(tok)
        if len(current_block) == mem_freq:
            result.append(landmark_id)
            block_token_ids.append(current_block)
            current_block = []
    return result, block_token_ids

def extract_landmark_hidden_states(model, token_ids, model_args, landmark_id=50260):
    device = model_args.device
    seq_len = model_args.sequence_length
    all_landmark_states = []
    for start in range(0, len(token_ids) - seq_len + 1, seq_len):
        chunk = token_ids[start:start + seq_len]
        x = torch.tensor([chunk], dtype=torch.long, device=device)
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=model_args.dtype):
            idx, pos_emb_closure = model.transformer.wpe(x)
            is_mem = idx == landmark_id
            tok_emb = model.transformer.wte(idx)
            landmark_selection = getattr(model_args, 'landmark_selection', 'positional')
            if landmark_selection != 'positional':
                tok_emb = model._replace_landmark_embeddings(tok_emb, is_mem, landmark_selection, model_args.mem_freq)
            h = pos_emb_closure.adapt_model_input(tok_emb, start_index=0)
            h = model.transformer.drop(h)
            for block in model.transformer.h:
                h, _ = block(h, is_mem, pos_emb_closure, None, start_index=0)
            h = model.transformer.ln_f(h)
            landmark_positions = is_mem[0].nonzero(as_tuple=True)[0]
            for pos in landmark_positions:
                all_landmark_states.append(h[0, pos].cpu().float().numpy())
    return np.array(all_landmark_states) if all_landmark_states else np.array([])

def compute_content_similarity(block_token_ids, min_count_threshold=5):
    n_blocks = len(block_token_ids)
    token_block_count = Counter()
    block_token_sets = []
    for block in block_token_ids:
        unique_tokens = set(block)
        block_token_sets.append(unique_tokens)
        for tok in unique_tokens:
            token_block_count[tok] += 1
    rare_block_sets = []
    for bset in block_token_sets:
        rare = {t for t in bset if token_block_count[t] < min_count_threshold}
        rare_block_sets.append(rare)
    sim_matrix = np.zeros((n_blocks, n_blocks))
    for i in range(n_blocks):
        for j in range(i + 1, n_blocks):
            if len(rare_block_sets[i]) == 0 and len(rare_block_sets[j]) == 0:
                sim = 0.0
            else:
                intersection = len(rare_block_sets[i] & rare_block_sets[j])
                union = len(rare_block_sets[i] | rare_block_sets[j])
                sim = intersection / union if union > 0 else 0.0
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim
    return sim_matrix

def compute_landmark_similarity(landmark_states):
    norms = np.linalg.norm(landmark_states, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normalized = landmark_states / norms
    return normalized @ normalized.T

def compute_recall_at_k(landmark_sim, content_sim, k_values=[1, 5, 10, 20]):
    n = landmark_sim.shape[0]
    results = {k: [] for k in k_values}
    for i in range(n):
        lm_scores = landmark_sim[i].copy()
        ct_scores = content_sim[i].copy()
        lm_scores[i] = -1
        ct_scores[i] = -1
        if ct_scores.max() <= 0:
            continue
        for k in k_values:
            if k >= n:
                continue
            top_k_landmark = set(np.argsort(lm_scores)[-k:])
            top_k_content = set(np.argsort(ct_scores)[-k:])
            if len(top_k_content) == 0:
                continue
            overlap = len(top_k_landmark & top_k_content)
            recall = overlap / min(k, len(top_k_content))
            results[k].append(recall)
    return {k: np.mean(v) if v else 0.0 for k, v in results.items()}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_dir", type=str, default=os.path.expanduser("~/landformer/LandFORMER/data/test"))
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_books", type=int, default=20)
    parser.add_argument("--min_tokens", type=int, default=10000)
    parser.add_argument("--max_blocks", type=int, default=500)
    args = parser.parse_args()
    tokenizer = tiktoken.get_encoding("gpt2")
    test_dir = Path(args.test_dir)
    books = []
    for f in sorted(test_dir.glob("*.txt")):
        with open(f) as fh:
            text = fh.read()
        n_tokens = len(tokenizer.encode(text))
        if n_tokens >= args.min_tokens:
            books.append((f, text))
        if len(books) >= args.max_books:
            break
    print(f"Selected {len(books)} books for evaluation")
    print(f"Loading model from: {args.checkpoint}")
    model, model_args = load_model(args.checkpoint, args.device)
    strategy = getattr(model_args, 'landmark_selection', 'positional')
    print(f"Strategy: {strategy}")
    k_values = [1, 5, 10, 20]
    all_recalls = {k: [] for k in k_values}
    book_results = []
    for book_idx, (book_path, text) in enumerate(books):
        print(f"\n[{book_idx+1}/{len(books)}] Processing {book_path.name}...")
        token_ids, block_token_ids = tokenize_book_with_landmarks(text, tokenizer, landmark_id=model_args.landmark_id, mem_freq=model_args.mem_freq)
        if len(block_token_ids) > args.max_blocks:
            block_token_ids = block_token_ids[:args.max_blocks]
            n_tokens_needed = args.max_blocks * (model_args.mem_freq + 1)
            token_ids = token_ids[:n_tokens_needed]
        n_blocks = len(block_token_ids)
        print(f"  Blocks: {n_blocks}")
        landmark_states = extract_landmark_hidden_states(model, token_ids, model_args, landmark_id=model_args.landmark_id)
        if landmark_states.size == 0:
            print(f"  Skipping (no landmarks extracted)")
            continue
        n_extracted = landmark_states.shape[0]
        n_blocks = min(n_blocks, n_extracted)
        block_token_ids = block_token_ids[:n_blocks]
        landmark_states = landmark_states[:n_blocks]
        print(f"  Extracted landmark states: {n_extracted}, using: {n_blocks}")
        if n_blocks < 30:
            print(f"  Skipping (too few blocks)")
            continue
        landmark_sim = compute_landmark_similarity(landmark_states)
        content_sim = compute_content_similarity(block_token_ids)
        recalls = compute_recall_at_k(landmark_sim, content_sim, k_values)
        for k in k_values:
            all_recalls[k].append(recalls[k])
        book_results.append({'book': book_path.name, 'n_blocks': n_blocks, **{f'recall@{k}': recalls[k] for k in k_values}})
        print(f"  Recall@1={recalls[1]:.3f}  @5={recalls[5]:.3f}  @10={recalls[10]:.3f}  @20={recalls[20]:.3f}")
    print(f"\n{'='*60}")
    print(f"LANDMARK RETRIEVAL QUALITY - {strategy.upper()}")
    print(f"{'='*60}")
    print(f"Books evaluated: {len(book_results)}")
    print(f"{'Metric':<12} {'Mean':>8} {'Std':>8}")
    print('-' * 30)
    for k in k_values:
        vals = all_recalls[k]
        if vals:
            print(f"Recall@{k:<5} {np.mean(vals):>7.3f} {np.std(vals):>7.3f}")
    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        with open(args.output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['book', 'n_blocks'] + [f'recall@{k}' for k in k_values])
            writer.writeheader()
            writer.writerows(book_results)
            writer.writerow({'book': 'MEAN', 'n_blocks': 0, **{f'recall@{k}': np.mean(all_recalls[k]) for k in k_values}})
            writer.writerow({'book': 'STD', 'n_blocks': 0, **{f'recall@{k}': np.std(all_recalls[k]) for k in k_values}})
        print(f"\nResults saved to {args.output_csv}")

if __name__ == "__main__":
    main()
