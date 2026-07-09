#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <base|cde> <gpu-index>" >&2
  exit 2
fi

kind=$1
gpu=$2
issue=CDE-71
host=$(hostname -s)
source_dir=$(pwd)

case "$host:$kind" in
  mimas:base)
    experiment_root=/exp/exp4/acq22mc
    data_root=/store/store4/data
    environment_root=/exp/exp4/acq22mc/StyleTTS2
    ;;
  phoebe:cde)
    experiment_root=/exp/exp8/acq22mc
    data_root=/store/store2/data
    environment_root=/exp/exp8/acq22mc/StyleTTS2
    ;;
  *)
    echo "Refusing unsupported host/model pairing: $host/$kind" >&2
    exit 2
    ;;
esac

job_root="$experiment_root/symphony-jobs/$issue/data_split_exp1/$host"
train_root="$job_root"
eval_root="$job_root/eval"
accelerate="$environment_root/.venv/bin/accelerate"
python="$environment_root/.venv/bin/python"

test -d "$data_root/ESD"
test -x "$accelerate"
test -x "$python"
test -f "$source_dir/text_to_speech.py"
test -f "$source_dir/eval_tts.py"
test -f "$source_dir/Data/val_list_esd.txt"
test -f "$source_dir/Data/ref_val_list_esd.txt"
mkdir -p "$eval_root"

"$accelerate" --help >/dev/null
"$python" -c 'import accelerate; print(f"accelerate {accelerate.__version__}")'
CUDA_VISIBLE_DEVICES="$gpu" "$python" -c \
  'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1; print(torch.cuda.get_device_name(0))'

declare -A final_epoch=(
  [1m]=00149
  [5m]=00029
  [15m]=00009
  [30m]=00004
  [60m]=00002
)

wait_for_training() {
  local evidence_dir=$1
  local checkpoint=$2

  while [[ ! -f "$evidence_dir/exit.code" ]]; do
    printf '[%s] waiting for %s/exit.code\n' "$(date -u +%FT%TZ)" "$evidence_dir"
    sleep 300
  done
  if [[ $(<"$evidence_dir/exit.code") != 0 ]]; then
    echo "Training failed: $evidence_dir/exit.code=$(<"$evidence_dir/exit.code")" >&2
    return 1
  fi
  test -s "$checkpoint"
}

run_one() {
  local split=$1
  local run="esd_small_data_split_exp1_${kind}_${split}"
  local model_config="$source_dir/Configs/data_split_exp1/$run.yml"
  local checkpoint="$train_root/$kind/$run/epoch_2nd_${final_epoch[$split]}.pth"
  local evidence_dir="$train_root/$run"
  local run_dir="$eval_root/$run"
  local samples_dir="$run_dir/samples"
  local config="$run_dir/tts.yml"

  mkdir -p "$run_dir"
  wait_for_training "$evidence_dir" "$checkpoint"
  test -f "$model_config"

  cat >"$config" <<EOF
model_config: $model_config
checkpoint: $checkpoint
root_path: $data_root
split: val
text_list: $source_dir/Data/val_list_esd.txt
ref_list: $source_dir/Data/ref_val_list_esd.txt
output_dir: $samples_dir
device: cuda
sample_rate: 24000
alpha: 0.3
beta: 0.7
diffusion_steps: 5
embedding_scale: 1.0
seed: 0
resume: true
EOF

  printf '%s\n' \
    "hostname=$host" \
    "gpu=$gpu" \
    "source_dir=$source_dir" \
    "model_config=$model_config" \
    "checkpoint=$checkpoint" \
    "samples_dir=$samples_dir" \
    "CUDA_VISIBLE_DEVICES=$gpu TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 $accelerate launch --num_processes 1 text_to_speech.py -p $config" \
    >"$run_dir/command.txt"
  date -u +%FT%TZ >"$run_dir/start_time_utc.txt"

  set +e
  (
    cd "$source_dir"
    CUDA_VISIBLE_DEVICES="$gpu" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
      "$accelerate" launch --num_processes 1 text_to_speech.py -p "$config"
  ) >"$run_dir/synthesis.log" 2>&1
  local status=$?
  set -e
  printf '%s\n' "$status" >"$run_dir/synthesis.exit.code"
  if [[ $status -ne 0 ]]; then
    return "$status"
  fi

  set +e
  (
    cd "$source_dir"
    "$python" eval_tts.py "$samples_dir" \
      --root-path "$data_root" \
      --split val \
      --ground-truth-dir "$eval_root/ground_truth_val"
  ) >"$run_dir/evaluation.log" 2>&1
  status=$?
  set -e
  printf '%s\n' "$status" >"$run_dir/evaluation.exit.code"
  date -u +%FT%TZ >"$run_dir/end_time_utc.txt"
  return "$status"
}

for split in 1m 5m 15m 30m 60m; do
  run_one "$split"
done
