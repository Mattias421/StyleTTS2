#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/exp/exp4/acq22mc/StyleTTS2/.venv/bin/python}"
accel="${ACCELERATE_BIN:-/exp/exp4/acq22mc/StyleTTS2/.venv/bin/accelerate}"
issue_root="/exp/exp4/acq22mc/symphony-jobs/CDE-71/data_split_exp1/mimas"
driver_lib="${NVIDIA_DRIVER_LIB:-$issue_root/driver-535.161.07/extracted}"
gpu_id="${CUDA_VISIBLE_DEVICES:-0}"
selected_splits=",${SPLITS:-1m,5m,15m,30m,60m},"

[[ "$(hostname)" == "mimas" ]]
[[ "$gpu_id" == "0" || "$gpu_id" == "3" ]]
test -x "$python_bin"
test -x "$accel"
test -f "$driver_lib/libcuda.so.1"
export LD_LIBRARY_PATH="$driver_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$python_bin" -c 'import accelerate, torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1'

mkdir -p "$issue_root"

run_one() {
  local name="$1"
  local config_rel="$2"
  local train_script="$3"
  local run_dir="$issue_root/$name"
  local config_path="$repo_root/$config_rel"
  local expected_checkpoint
  local command_line="CUDA_VISIBLE_DEVICES=$gpu_id $accel launch --num_processes 1 --mixed_precision=fp16 $train_script -p $config_rel"

  expected_checkpoint="$("$python_bin" - "$config_path" <<'PY'
import pathlib
import sys
import yaml

config = yaml.safe_load(open(sys.argv[1]))
epochs = config["epochs"]
assert epochs < config["loss_params"]["joint_epoch"]
assert "data_split_exp1/mimas/base/" in config["log_dir"]
for path in (
    config["pretrained_model"],
    config["F0_path"],
    config["ASR_config"],
    config["ASR_path"],
    config["PLBERT_dir"],
    config["data_params"]["train_data"],
    config["data_params"]["val_data"],
    config["data_params"]["OOD_data"],
):
    assert pathlib.Path(path).exists(), path
print(pathlib.Path(config["log_dir"]) / f"epoch_2nd_{epochs - 1:05d}.pth")
PY
)"
  mkdir -p "$run_dir"
  if [[ -f "$run_dir/exit.code" ]] && [[ "$(<"$run_dir/exit.code")" == "0" ]] && [[ -f "$expected_checkpoint" ]]; then
    echo "completed run exists: $name"
    return
  fi
  if [[ -f "$run_dir/pid" ]] && kill -0 "$(<"$run_dir/pid")" 2>/dev/null; then
    echo "live run exists: $name" >&2
    return 1
  fi
  rm -f "$run_dir/exit.code" "$run_dir/end_time_utc"
  printf '%s\n' "$command_line" >"$run_dir/command.txt"
  hostname >"$run_dir/hostname"
  printf '%s\n' "$gpu_id" >"$run_dir/gpu"
  printf '%s\n' "$config_path" >"$run_dir/config_path"
  printf '%s\n' "$expected_checkpoint" >"$run_dir/expected_checkpoint"
  date -u +%Y-%m-%dT%H:%M:%SZ >"$run_dir/start_time_utc"

  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] start $name"
  set +e
  (
    cd "$repo_root" &&
      exec env CUDA_VISIBLE_DEVICES="$gpu_id" "$accel" launch --num_processes 1 --mixed_precision=fp16 "$train_script" -p "$config_rel"
  ) >"$run_dir/run.log" 2>&1 &
  pid=$!
  printf '%s\n' "$pid" >"$run_dir/pid"
  wait "$pid"
  status=$?
  set -e
  printf '%s\n' "$status" >"$run_dir/exit.code"
  date -u +%Y-%m-%dT%H:%M:%SZ >"$run_dir/end_time_utc"
  if [[ "$status" == "0" ]] && [[ ! -f "$expected_checkpoint" ]]; then
    echo "missing expected checkpoint: $expected_checkpoint" >>"$run_dir/run.log"
    printf '1\n' >"$run_dir/exit.code"
    status=1
  fi
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] end $name exit=$status"
  return "$status"
}

configs=(
  "1m:esd_small_data_split_exp1_base_1m:Configs/data_split_exp1/esd_small_data_split_exp1_base_1m.yml:train_finetune_accelerate.py"
  "5m:esd_small_data_split_exp1_base_5m:Configs/data_split_exp1/esd_small_data_split_exp1_base_5m.yml:train_finetune_accelerate.py"
  "15m:esd_small_data_split_exp1_base_15m:Configs/data_split_exp1/esd_small_data_split_exp1_base_15m.yml:train_finetune_accelerate.py"
  "30m:esd_small_data_split_exp1_base_30m:Configs/data_split_exp1/esd_small_data_split_exp1_base_30m.yml:train_finetune_accelerate.py"
  "60m:esd_small_data_split_exp1_base_60m:Configs/data_split_exp1/esd_small_data_split_exp1_base_60m.yml:train_finetune_accelerate.py"
)

for entry in "${configs[@]}"; do
  IFS=: read -r split name config_rel train_script <<<"$entry"
  [[ "$selected_splits" == *",$split,"* ]] || continue
  run_one "$name" "$config_rel" "$train_script"
done
