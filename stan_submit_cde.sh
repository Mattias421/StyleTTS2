#!/bin/bash
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --partition=gpu,gpu-h100,gpu-h100-nvl
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1

set -euo pipefail

config_path=$1

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

python train_finetune_cde.py -p "$config_path"
