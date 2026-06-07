#!/usr/bin/env bash

set -u -o pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
accel="$repo_root/.venv/bin/accelerate"
issue_root="/exp/exp4/acq22mc/symphony-jobs/CDE-71/data_split_exp1/mimas"
gpu_id="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$issue_root"

run_one() {
  local name="$1"
  local config_rel="$2"
  local train_script="$3"
  local run_dir="$issue_root/$name"
  local command_line="CUDA_VISIBLE_DEVICES=$gpu_id $accel launch --num_processes 1 --mixed_precision=fp16 $train_script -p $config_rel"

  mkdir -p "$run_dir"
  printf '%s\n' "$command_line" >"$run_dir/command.txt"

  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] start $name"
  set +e
  (
    cd "$repo_root" &&
      CUDA_VISIBLE_DEVICES="$gpu_id" "$accel" launch --num_processes 1 --mixed_precision=fp16 "$train_script" -p "$config_rel"
  ) >"$run_dir/run.log" 2>&1
  status=$?
  printf '%s\n' "$status" >"$run_dir/exit.code"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] end $name exit=$status"
}

configs=(
  "esd_small_data_split_exp1_base_1m:Configs/data_split_exp1/esd_small_data_split_exp1_base_1m.yml:train_finetune_accelerate.py"
  "esd_small_data_split_exp1_base_5m:Configs/data_split_exp1/esd_small_data_split_exp1_base_5m.yml:train_finetune_accelerate.py"
  "esd_small_data_split_exp1_base_15m:Configs/data_split_exp1/esd_small_data_split_exp1_base_15m.yml:train_finetune_accelerate.py"
  "esd_small_data_split_exp1_base_30m:Configs/data_split_exp1/esd_small_data_split_exp1_base_30m.yml:train_finetune_accelerate.py"
  "esd_small_data_split_exp1_base_60m:Configs/data_split_exp1/esd_small_data_split_exp1_base_60m.yml:train_finetune_accelerate.py"
)

for entry in "${configs[@]}"; do
  IFS=: read -r name config_rel train_script <<<"$entry"
  run_one "$name" "$config_rel" "$train_script"
done
