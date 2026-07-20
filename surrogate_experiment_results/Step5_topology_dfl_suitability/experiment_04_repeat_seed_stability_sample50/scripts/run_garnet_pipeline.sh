#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/local1/fuweik/UdeM-Intern"
EXPERIMENT_REL="surrogate_experiment_results/Step5_topology_dfl_suitability/experiment_04_repeat_seed_stability_sample50"
EXP1_REL="surrogate_experiment_results/Step5_topology_dfl_suitability/experiment_01_weak_label_seed42_sample50"
OUTPUT_REL="${EXPERIMENT_REL}/results/repeat_seed120"
LOG_DIR="${OUTPUT_REL}/logs"
RESULT_DIR="${OUTPUT_REL}/results"
ARTIFACT_SHARDS="${ARTIFACT_SHARDS:-8}"

cd "${PROJECT_ROOT}"
source configs/runtime/garnet.env
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

mkdir -p "${LOG_DIR}" "${RESULT_DIR}"

write_phase() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$1" > "${OUTPUT_REL}/pipeline_status.txt"
  printf '[exp4-pipeline] %s %s\n' "$(date --iso-8601=seconds)" "$1"
}

write_phase "artifact_build_started shards=${ARTIFACT_SHARDS}"
pids=()
for ((shard=0; shard<ARTIFACT_SHARDS; shard++)); do
  python "${EXPERIMENT_REL}/scripts/build_repeat_seed_artifacts.py" \
    --shard-index "${shard}" \
    --shard-count "${ARTIFACT_SHARDS}" \
    > "${LOG_DIR}/artifact_shard_${shard}.log" 2>&1 &
  pids+=("$!")
done

artifact_failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    artifact_failed=1
  fi
done
if [[ "${artifact_failed}" -ne 0 ]]; then
  write_phase "artifact_build_failed"
  exit 1
fi

write_phase "artifact_audit_started"
python "${EXPERIMENT_REL}/scripts/audit_repeat_seed_artifacts.py" \
  > "${LOG_DIR}/artifact_audit.log" 2>&1

write_phase "strict_plan_started"
python "${EXPERIMENT_REL}/scripts/plan_repeat_seed_jobs.py" \
  > "${LOG_DIR}/plan.log" 2>&1

JOBS_CSV="${OUTPUT_REL}/plans/repeat_seed120_jobs.csv"
LAUNCHER="${EXP1_REL}/scripts/launch_weak_label_jobs.py"

write_phase "launcher_preview_started"
python "${LAUNCHER}" \
  --jobs-csv "${JOBS_CSV}" \
  --output-root "${OUTPUT_REL}" \
  --expected-job-count 120 \
  --normal-workers 20 \
  --long-workers 0 \
  --monitor-interval 60 \
  --require-hostname garnet \
  > "${LOG_DIR}/launcher_preview.log" 2>&1

write_phase "training_started jobs=120 workers=20"
python "${LAUNCHER}" \
  --jobs-csv "${JOBS_CSV}" \
  --output-root "${OUTPUT_REL}" \
  --expected-job-count 120 \
  --normal-workers 20 \
  --long-workers 0 \
  --monitor-interval 60 \
  --require-hostname garnet \
  --execute \
  > "${LOG_DIR}/launcher.log" 2>&1

write_phase "review_started"
python "${EXPERIMENT_REL}/scripts/review_repeat_seed_results.py" \
  > "${LOG_DIR}/review.log" 2>&1

write_phase "complete"
