#!/bin/bash
#SBATCH --job-name=smalstyletts2
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=90G
#SBATCH --output=Models/logs/styletts2.out
#SBATCH --partition=gpu,gpu-h100,gpu-h100-nvl
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config-stem-or-path> [cde]" >&2
  exit 1
fi

config_path=$1
config_path=${config_path%.yml}
config_path=${config_path%.yaml}

module load eSpeak-NG
module load libsndfile
module load OpenBLAS
ml CUDA/12.4.0

source .venv/bin/activate

export HF_HOME=/mnt/parscratch/users/acq22mc/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_DISABLE_XET=1
mkdir -p "$HF_HOME"

export SSL_CERT_FILE
SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
export REQUESTS_CA_BUNDLE=$SSL_CERT_FILE
export CURL_CA_BUNDLE=$SSL_CERT_FILE

if [[ "${2:-}" == cde ]]; then
  accelerate launch --mixed_precision=fp16 --num_processes=1 train_finetune_cde_accelerate.py --config_path "${config_path}.yml"
  accelerate launch --mixed_precision=fp16 --num_processes=1 train_finetune_cde_accelerate.py --config_path "${config_path}_slm.yml"
else
  accelerate launch --mixed_precision=fp16 --num_processes=1 train_finetune_accelerate.py --config_path "${config_path}.yml"
  accelerate launch --mixed_precision=fp16 --num_processes=1 train_finetune_accelerate.py --config_path "${config_path}_slm.yml"
fi
