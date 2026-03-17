#!/bin/bash
# Test Contribution #1 (memory augmentation) on WikiText-2

echo "Training with pre-computed keys (Contribution #1)..."

python lm_benchmark/main.py \
    --config_format rotary \
    --model landmark \
    --n_embd 512 \
    --n_head 8 \
    --n_layer 6 \
    --batch_size 4 \
    --sequence_length 512 \
    --mem_freq 50 \
    --mem_cache_freq 50 \
    --mem_cache_size 80 \
    --dataset wikitext2 \
    --iterations 500 \
    --eval_freq 100 \
    --save_checkpoint_freq 500 \
    --use_cache \
    --lm_cache mem \
    --use_precomputed_keys \
    --device cuda:0 \
    --results_base_folder ./exps/wikitext2_contrib1

echo "Contribution #1 training complete!"