#!/bin/bash
#SBATCH --job-name=exp51-bidi
#SBATCH --array=0-3
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --partition=gpu,gpu-h100,gpu-h100-nvl
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-exp51-%A_%a.out

config_stems=(
  dt020
  dt020_l4
  dt100
  dt100_l4
)

config_path="Configs/exp51_bidigpu/${config_stems[$SLURM_ARRAY_TASK_ID]}"

cd $EXP/StyleTTS2
bash stan_submit.sh "$config_path" cde
