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

import numpy as np
import torch
import os

def apply_add_mem_tokens(mem_id, tokens_filename, freq, start_idx, end_idx):
    tokens = np.memmap(tokens_filename, dtype=np.uint16, mode='r')
    print(f"Processing {start_idx}-{end_idx}")
    tokens_with_mem = []
    for t_idx in range(start_idx, end_idx):
        t = tokens[t_idx]
        tokens_with_mem.append(t)
        if freq is not None and t_idx % freq == freq - 1:
            tokens_with_mem.append(mem_id)
    return tokens_with_mem

def add_mem_tokens(landmark_id, data, mem_freq, output_file=None):
    """Add landmark tokens - memory efficient, writes chunks to disk"""
    print(f"\nAdding landmark tokens (memory-efficient)")
    print(f"Input: {len(data):,} tokens")
    print(f"Landmark ID: {landmark_id}, Frequency: every {mem_freq} tokens")
    
    chunk_size = 10_000_000
    
    if output_file is not None:
        # Write directly to disk in chunks
        with open(output_file, 'wb') as f:
            for start in range(0, len(data), chunk_size):
                end = min(start + chunk_size, len(data))
                print(f"Processing {start:,} - {end:,}", flush=True)
                
                chunk = np.array(data[start:end])
                result = []
                for i in range(0, len(chunk), mem_freq):
                    block_end = min(i + mem_freq, len(chunk))
                    result.extend(chunk[i:block_end].tolist())
                    if block_end < len(chunk):
                        result.append(landmark_id)
                
                chunk_arr = np.array(result, dtype=np.uint16)
                chunk_arr.tofile(f)
                del result, chunk_arr, chunk
        
        print(f"Completed! Written to {output_file}\n", flush=True)
        return None
    else:
        # Small data (validation) - keep in memory
        result = []
        for start in range(0, len(data), chunk_size):
            end = min(start + chunk_size, len(data))
            print(f"Processing {start:,} - {end:,}", flush=True)
            chunk = np.array(data[start:end])
            for i in range(0, len(chunk), mem_freq):
                block_end = min(i + mem_freq, len(chunk))
                result.extend(chunk[i:block_end].tolist())
                if block_end < len(chunk):
                    result.append(landmark_id)
        
        print(f"Completed! Output: {len(result):,} tokens\n", flush=True)
        return np.array(result, dtype=np.uint16)
