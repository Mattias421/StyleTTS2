#!/bin/bash
#SBATCH --job-name=styletts2
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --output=Models/logs/styletts2.out
#SBATCH --partition=gpu,gpu-h100,gpu-h100-nvl
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1

module load eSpeak-NG
module load libsndfile
module load OpenBLAS
ml CUDA/12.4.0

source .venv/bin/activate

if [[ "$2" == cde ]]; then
  accelerate launch --mixed_precision=fp16 --num_processes=1 train_finetune_cde_accelerate.py --config_path $1
else
  accelerate launch --mixed_precision=fp16 --num_processes=1 train_finetune_accelerate.py --config_path $1
fi
