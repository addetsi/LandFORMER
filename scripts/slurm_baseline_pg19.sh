#!/bin/bash
#SBATCH --job-name=pg19_baseline
#SBATCH --output=/local/s4251938/logs/baseline_%j.out
#SBATCH --error=/local/s4251938/logs/baseline_%j.err
#SBATCH --partition=L40s_staff
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=120:00:00
#SBATCH --mem=64G

echo "========================================
"
echo "Job started: $(date)"
echo "Running on: $(hostname)"
echo "GPU info:"
nvidia-smi
echo "========================================
"

# Activate environment
conda activate /local/s4251938/pg19_env

# Create log directory
mkdir -p /local/s4251938/logs

# Navigate to code
cd /home/s4251938/landformer/LandFORMER/lm_benchmark

# Run training
python main.py \
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
    --no_compile

echo "========================================
"
echo "Job completed: $(date)"
echo "========================================
"
