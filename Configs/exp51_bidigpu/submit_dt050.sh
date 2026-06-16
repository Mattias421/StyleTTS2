#!/bin/bash
set -euo pipefail


mkdir -p Models/logs

sbatch \
  --job-name=exp51-bidi-dt050 \
  --output=Models/logs/exp51-bidi-dt050-%j.out \
  stan_submit.sh Configs/exp51_bidigpu/dt050 cde
