#!/bin/bash
#SBATCH --job-name=exp53-te
#SBATCH --array=0-3
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=90G
#SBATCH --partition=gpu,gpu-h100,gpu-h100-nvl
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-exp53-%A_%a.out

set -euo pipefail

config_stems=(
  dt050_l1
  dt050_l2
  dt100_l1
  dt100_l2
)

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
config_path="Configs/exp53_te_unidigpu/${config_stems[$SLURM_ARRAY_TASK_ID]}"

cd "$repo_root"

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

accelerate launch --mixed_precision=fp16 --num_processes=1 \
  train_finetune_cde_te_accelerate.py --config_path "${config_path}.yml"
accelerate launch --mixed_precision=fp16 --num_processes=1 \
  train_finetune_cde_te_accelerate.py --config_path "${config_path}_slm.yml"
