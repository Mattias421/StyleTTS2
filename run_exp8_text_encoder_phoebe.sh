#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$script_dir"

source .venv/bin/activate

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export HF_HOME=${HF_HOME:-/store/store4/acq22mc/.cache/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export HF_HUB_DISABLE_XET=1
mkdir -p "$HF_HOME"

if python -c "import certifi" >/dev/null 2>&1; then
  export SSL_CERT_FILE
  SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
  export REQUESTS_CA_BUNDLE=$SSL_CERT_FILE
  export CURL_CA_BUNDLE=$SSL_CERT_FILE
fi

variants=(
  dt050_l1
  dt050_l2
  dt100_l1
  dt100_l2
)

IFS=',' read -r -a gpus <<< "${GPUS:-0,1}"
log_dir="Models/exp8_text_encoder/run_logs"
mkdir -p "$log_dir"

run_variant() {
  local gpu="$1"
  local variant="$2"
  local config_path="Configs/exp8_text_encoder/${variant}"
  local log_path="${log_dir}/${variant}.log"

  {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting ${variant} on GPU ${gpu}"
    CUDA_VISIBLE_DEVICES="$gpu" accelerate launch --mixed_precision=fp16 --num_processes=1 \
      train_finetune_cde_te_accelerate.py --config_path "${config_path}.yml"
    CUDA_VISIBLE_DEVICES="$gpu" accelerate launch --mixed_precision=fp16 --num_processes=1 \
      train_finetune_cde_te_accelerate.py --config_path "${config_path}_slm.yml"
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] finished ${variant}"
  } >"$log_path" 2>&1
}

run_worker() {
  local worker_index="$1"
  local gpu="${gpus[$worker_index]}"
  local index

  for index in "${!variants[@]}"; do
    if (( index % ${#gpus[@]} == worker_index )); then
      run_variant "$gpu" "${variants[$index]}"
    fi
  done
}

status=0
for worker_index in "${!gpus[@]}"; do
  run_worker "$worker_index" &
done

for job in $(jobs -p); do
  wait "$job" || status=1
done

exit "$status"
