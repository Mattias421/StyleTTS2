#!/bin/bash
#SBATCH --job-name=exp52-unidi
#SBATCH --array=0-5
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --partition=gpu,gpu-h100,gpu-h100-nvl
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-exp52-%A_%a.out

set -euo pipefail

config_stems=(
  dt050_l1
  dt050
  dt050_l4
  dt100_l1
  dt100
  dt100_l4
)

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
config_path="Configs/exp52_unidigpu/${config_stems[$SLURM_ARRAY_TASK_ID]}"

cd "$repo_root"
bash stan_submit.sh "$config_path" cde
