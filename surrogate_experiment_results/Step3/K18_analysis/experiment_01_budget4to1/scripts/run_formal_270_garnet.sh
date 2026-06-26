#!/usr/bin/env bash
set -euo pipefail

cd /local1/fuweik/UdeM-Intern
source configs/runtime/garnet.env

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

exp="surrogate_experiment_results/Step3/K18_analysis/experiment_01_budget4to1"
output_root="${exp}/results/formal_270_full_epoch_20260626"

echo "[formal-270] wrapper start $(date -Is)"
echo "[formal-270] output_root ${output_root}"
echo "[formal-270] workers normal=${NORMAL_WORKERS:-16} long=${LONG_WORKERS:-4}"
echo "[formal-270] thread limits OMP=${OMP_NUM_THREADS} MKL=${MKL_NUM_THREADS} OPENBLAS=${OPENBLAS_NUM_THREADS} NUMEXPR=${NUMEXPR_NUM_THREADS}"

python "${exp}/scripts/launch_formal_k18_sample_size_jobs.py" \
  --output-root "${output_root}" \
  --normal-workers "${NORMAL_WORKERS:-16}" \
  --long-workers "${LONG_WORKERS:-4}" \
  --monitor-interval "${MONITOR_INTERVAL:-60}"

echo "[formal-270] wrapper end $(date -Is)"
