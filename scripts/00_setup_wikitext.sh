#!/bin/bash
# Setup WikiText-2 for testing

echo "Setting up WikiText-2..."

python3 - << 'EOF'
from datasets import load_dataset
import numpy as np
import tiktoken
import os

enc = tiktoken.get_encoding('gpt2')
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')

def tokenize_split(split_name):
    tokens = []
    for item in dataset[split_name]:
        if item['text'].strip():
            tokens.extend(enc.encode_ordinary(item['text']))
    return np.array(tokens, dtype=np.uint16)

os.makedirs('data/wikitext2', exist_ok=True)
tokenize_split('train').tofile('data/wikitext2/train.bin')
tokenize_split('validation').tofile('data/wikitext2/val.bin')
print('WikiText-2 ready!')
EOF

echo "Done! Data saved to data/wikitext2/"