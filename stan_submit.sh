#!/bin/bash
#SBATCH --job-name=styletts2
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --output=Models/logs/styletts2.out
#SBATCH --partition=gpu,gpu-h100,gpu-h100-nvl
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1

module load eSpeak-NG
module load libsndfile
module load OpenBLAS
ml CUDA/12.4.0

source .venv/bin/activate

python train_finetune.py -p $1
