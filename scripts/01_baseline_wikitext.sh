#!/bin/bash
# Test baseline (no optimization) on WikiText-2

echo "Training baseline model on WikiText-2..."

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
    --dataset wikitext2 \
    --iterations 500 \
    --eval_freq 100 \
    --save_checkpoint_freq 500 \
    --use_cache \
    --mem_cache_size 80 \
    --lm_cache mem \
    --device cuda:0 \
    --results_base_folder ./exps/wikitext2_baseline

echo "Baseline training complete!"