#!/bin/bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)

mkdir -p Models/logs

for config_stem in dt050 dt050_l1 dt050_l4; do
  sbatch \
    --job-name="exp51-bidi-${config_stem}" \
    --output="Models/logs/exp51-bidi-${config_stem}-%j.out" \
    stan_submit.sh "Configs/exp51_bidigpu/${config_stem}" cde
done
