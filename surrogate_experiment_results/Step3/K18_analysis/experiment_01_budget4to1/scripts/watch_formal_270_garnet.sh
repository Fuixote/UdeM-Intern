#!/usr/bin/env bash
set -euo pipefail

cd /local1/fuweik/UdeM-Intern
source configs/runtime/garnet.env

exp="surrogate_experiment_results/Step3/K18_analysis/experiment_01_budget4to1"

python scripts/experiment_notify.py \
  --project "K18-E1 formal 270 full-epoch run" \
  --subject "K18-E1 formal 270 finished" \
  --session k18_e1_formal_270 \
  --result-dir "${exp}/results/formal_270_full_epoch_20260626" \
  --log-dir "${exp}/results/logs" \
  --interval 60
