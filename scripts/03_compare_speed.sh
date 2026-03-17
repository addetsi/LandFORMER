#!/bin/bash
# Compare inference speed: Baseline vs Contribution #1

echo "================================"
echo "Testing BASELINE inference speed"
echo "================================"

python lm_benchmark/eval.py \
    --checkpoint ./exps/wikitext2_baseline/wikitext2/landmark/landmark_lr0.002_memfreq50_bs4x4_seqlen512/iterations=500_eval_freq=100_results_base_folder=./exps/wikitext2_baseline_save_checkpoint_freq=500_dataset=wikitext2_n_layer=6_n_embd=512_use_cache=True_lm_cache=mem_mem_cache_size=80/mem_cache_freq=50_seed=2/ckpt_500.pt \
    --dataset wikitext2 \
    --eval_seq_length 2048 \
    --use_cache \
    --lm_cache mem \
    --mem_cache_freq 50 \
    --mem_cache_size 80 \
    --no_compile \
    --device cuda:0

echo ""
echo "========================================"
echo "Testing CONTRIBUTION #1 inference speed"
echo "========================================"

python lm_benchmark/eval.py \
    --checkpoint ./exps/wikitext2_contrib1/wikitext2/landmark/landmark_lr0.002_memfreq50_bs4x4_seqlen512/iterations=500_eval_freq=100_results_base_folder=./exps/wikitext2_contrib1_save_checkpoint_freq=500_dataset=wikitext2_n_layer=6_n_embd=512_use_cache=True_lm_cache=mem_mem_cache_size=80/mem_cache_freq=50_use_precomputed_keys=True_seed=2/ckpt_500.pt \
    --dataset wikitext2 \
    --eval_seq_length 2048 \
    --use_cache \
    --lm_cache mem \
    --mem_cache_freq 50 \
    --mem_cache_size 80 \
    --no_compile \
    --use_precomputed_keys \
    --device cuda:0

echo ""
echo "================================"
echo "Comparison complete!"
echo "================================"