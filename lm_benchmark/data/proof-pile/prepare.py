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

import json
import os

from datasets import load_dataset
import numpy as np
import tiktoken
from tqdm import tqdm


dir_path = os.path.dirname(os.path.realpath(__file__))

# Try the new way of loading datasets (bypassing deprecated dataset scripts)
try:
    # Try to load from parquet files directly via Hugging Face Hub
    from huggingface_hub import hf_hub_download
    import pandas as pd
    
    print("Downloading proof-pile data from Hugging Face Hub...")
    # Download arxiv split parquet file
    arxiv_file = hf_hub_download(repo_id="hoskinson-center/proof-pile", 
                                   filename="data/arxiv-0.parquet",
                                   repo_type="dataset")
    df = pd.read_parquet(arxiv_file)
    
    # Convert to dataset format
    dataset = {'arxiv': type('Dataset', (), {'__iter__': lambda self: iter([{'text': row['text'], 'meta': row.get('meta', '{}')} for _, row in df.iterrows()])})}
except Exception as e:
    print(f"Failed to load from Hub: {e}")
    print("Falling back to load_dataset...")
    try:
        dataset = load_dataset("hoskinson-center/proof-pile", cache_dir=os.path.join(dir_path, "cache"))
    except RuntimeError as e2:
        print(f"ERROR: Could not load proof-pile dataset: {e2}")
        print("Proof-pile dataset loader is deprecated in newer HuggingFace versions.")
        print("Please download the dataset manually or use PG19 instead.")
        exit(1)


num_proc = 16
arxiv = dataset.filter(lambda x: json.loads(x['meta']).get('config', None) == "arxiv", num_proc=num_proc)


enc = tiktoken.get_encoding("gpt2")
def process(example):
    ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
    ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
    # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
    out = {'ids': ids, 'len': len(ids)}
    return out

# tokenize the dataset
tokenized = arxiv.map(
    process,
    remove_columns=['text'],
    desc="tokenizing the splits",
    num_proc=num_proc,
)


for split, dset in tokenized.items():
    arr_len = np.sum(dset['len'])
    filename = os.path.join(dir_path, f'{split}.bin')
    dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
    total_batches = 1024

    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
        # Batch together samples for faster write
        batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
        arr_batch = np.concatenate(batch['ids'])
        # Write into mmap
        arr[idx : idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)
    arr.flush()
