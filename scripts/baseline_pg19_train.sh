#!/bin/bash
#
# PG-19 Baseline Training (Paper Replication)
# Target: Perplexity ~14.72 on validation
#

echo "========================================"
echo "PG-19 BASELINE TRAINING"
echo "Started: $(date)"
echo "========================================"

cd /home/s4251938/landformer/LandFORMER/lm_benchmark

# Exact paper hyperparameters
python3 main.py \
    --config_format rotary \
    --model landmark \
    --dataset pg19 \
    --n_embd 1024 \
    --n_head 8 \
    --n_layer 12 \
    --batch_size 16 \
    --sequence_length 512 \
    --acc_steps 8 \
    --mem_freq 50 \
    --positional_encoder rotary \
    --softmax_func mem_opt \
    --iterations 240000 \
    --lr 0.002 \
    --warmup_percent 0.02 \
    --weight_decay 0.001 \
    --dropout 0.0 \
    --scheduler cos \
    --eval_freq 2000 \
    --save_checkpoint_freq 20000 \
    --device cuda:0 \
    --results_base_folder /local/s4251938/checkpoints/pg19_baseline \
    --no_compile \
    2>&1 | tee /local/s4251938/checkpoints/pg19_baseline/training.log

echo "========================================"
echo "Training completed: $(date)"
echo "========================================"

