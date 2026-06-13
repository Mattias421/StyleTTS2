#!/bin/bash
#SBATCH --job-name=exp51-bidi
#SBATCH --array=0-3
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --partition=gpu,gpu-h100,gpu-h100-nvl
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-exp51-%A_%a.out

set -euo pipefail

config_stems=(
  dt020
  dt020_l4
  dt100
  dt100_l4
)

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
config_path="Configs/exp51_bidigpu/${config_stems[$SLURM_ARRAY_TASK_ID]}"

cd "$repo_root"
bash stan_submit.sh "$config_path"
