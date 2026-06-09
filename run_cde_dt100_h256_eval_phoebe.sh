#!/usr/bin/env bash
set -u

cd /exp/exp8/acq22mc/StyleTTS2 || exit 1

PYTHON=".venv/bin/python"
LOG_DIR="Models/cde_dt100_hlayers/eval_logs"
mkdir -p "$LOG_DIR"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

run_variant() {
    local gpu="$1"
    local variant="$2"
    local cfg="Configs/cde_dt100_hlayers/tts_${variant}.yml"
    local sample_dir="Models/cde_dt100_hlayers/${variant}/samples"
    local log_path="${LOG_DIR}/${variant}.log"

    {
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting ${variant} on GPU ${gpu}"
        local wav_count=0
        local manifest_count=0
        if [ -d "$sample_dir" ]; then
            wav_count=$(find "$sample_dir" -maxdepth 1 -name '*.wav' | wc -l)
        fi
        if [ -f "$sample_dir/manifest.jsonl" ]; then
            manifest_count=$(wc -l < "$sample_dir/manifest.jsonl")
        fi

        if [ "$wav_count" -eq 2000 ] && [ "$manifest_count" -eq 2000 ]; then
            echo "generation skipped; found 2000 wavs and manifest rows"
        else
            CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" text_to_speech.py -p "$cfg"
            local status=$?
            echo "generation exited ${status}"
            if [ "$status" -ne 0 ]; then
                return "$status"
            fi
        fi

        if [ -f "$sample_dir/eval_tts_results.txt" ]; then
            echo "evaluation skipped; results already exist"
            return 0
        fi

        "$PYTHON" eval_tts.py "$sample_dir" \
            --root-path /store/store2/data \
            --split val \
            --nj 16
        local status=$?
        echo "evaluation exited ${status}"
        return "$status"
    } >"$log_path" 2>&1
}

run_variant 0 esd_small_cde_dt100_h256_l2 &&
    run_variant 0 esd_small_cde_dt100_h256_l6 &
pid0=$!

run_variant 1 esd_small_cde_dt100_h256_l4 &
pid1=$!

status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
exit "$status"
