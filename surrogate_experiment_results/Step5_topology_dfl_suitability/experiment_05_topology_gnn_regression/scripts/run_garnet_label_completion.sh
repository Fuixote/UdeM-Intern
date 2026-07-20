#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/local1/fuweik/UdeM-Intern"
EXP4_REL="surrogate_experiment_results/Step5_topology_dfl_suitability/experiment_04_repeat_seed_stability_sample50"
EXP5_REL="surrogate_experiment_results/Step5_topology_dfl_suitability/experiment_05_topology_gnn_regression"
OUTPUT_REL="${EXP5_REL}/results/multiseed_completion1880"
TOPOLOGIES="${EXP5_REL}/configs/multiseed_label_completion940.csv"
FORMAL_SUMMARY="surrogate_experiment_results/Step5_topology_dfl_suitability/experiment_03_formal_continuous_label_seed42_sample50/results/formal1000/results/weak_label_topology_summary.csv"
EXP4_REPEAT_LABELS="${EXP4_REL}/results/repeat_seed120/results/repeat_seed_labels_long.csv"
RESULT_DIR="${OUTPUT_REL}/results"
LOG_DIR="${OUTPUT_REL}/logs"
ARTIFACT_SHARDS="${ARTIFACT_SHARDS:-24}"
WORKERS="${WORKERS:-24}"

cd "${PROJECT_ROOT}"
source configs/runtime/garnet.env
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
mkdir -p "${RESULT_DIR}" "${LOG_DIR}"

write_phase() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$1" > "${OUTPUT_REL}/pipeline_status.txt"
  printf '[exp5-label-completion] %s %s\n' "$(date --iso-8601=seconds)" "$1"
}

write_phase "artifact_build_started topologies=940 bundles=1880 shards=${ARTIFACT_SHARDS}"
pids=()
for ((shard=0; shard<ARTIFACT_SHARDS; shard++)); do
  python "${EXP4_REL}/scripts/build_repeat_seed_artifacts.py" \
    --topologies-csv "${TOPOLOGIES}" \
    --output-root "${OUTPUT_REL}" \
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
python "${EXP4_REL}/scripts/audit_repeat_seed_artifacts.py" \
  --topologies-csv "${TOPOLOGIES}" \
  --output-root "${OUTPUT_REL}" \
  --audit-output "${RESULT_DIR}/completion_artifact_audit.json" \
  --test-verification manifest \
  > "${LOG_DIR}/artifact_audit.log" 2>&1

write_phase "strict_plan_started jobs=1880"
python "${EXP4_REL}/scripts/plan_repeat_seed_jobs.py" \
  --topologies-csv "${TOPOLOGIES}" \
  --artifact-root "${OUTPUT_REL}" \
  --job-output-root "${OUTPUT_REL}" \
  --expected-job-count 1880 \
  --plan-output "${OUTPUT_REL}/plans/multiseed_completion1880_plan.json" \
  --jobs-csv-output "${OUTPUT_REL}/plans/multiseed_completion1880_jobs.csv" \
  > "${LOG_DIR}/plan.log" 2>&1

JOBS_CSV="${OUTPUT_REL}/plans/multiseed_completion1880_jobs.csv"
LAUNCHER="${EXP4_REL}/../experiment_01_weak_label_seed42_sample50/scripts/launch_weak_label_jobs.py"

write_phase "launcher_preview_started jobs=1880 workers=${WORKERS}"
python "${LAUNCHER}" \
  --jobs-csv "${JOBS_CSV}" \
  --output-root "${OUTPUT_REL}" \
  --expected-job-count 1880 \
  --normal-workers "${WORKERS}" \
  --long-workers 0 \
  --monitor-interval 60 \
  --require-hostname garnet \
  > "${LOG_DIR}/launcher_preview.log" 2>&1

write_phase "training_started jobs=1880 workers=${WORKERS}"
python "${LAUNCHER}" \
  --jobs-csv "${JOBS_CSV}" \
  --output-root "${OUTPUT_REL}" \
  --expected-job-count 1880 \
  --normal-workers "${WORKERS}" \
  --long-workers 0 \
  --monitor-interval 60 \
  --require-hostname garnet \
  --execute \
  > "${LOG_DIR}/launcher.log" 2>&1

write_phase "completion_review_started"
python "${EXP5_REL}/scripts/review_multiseed_completion_results.py" \
  --topologies-csv "${TOPOLOGIES}" \
  --output-root "${OUTPUT_REL}" \
  > "${LOG_DIR}/review.log" 2>&1

write_phase "formal_target_build_started"
python "${EXP5_REL}/scripts/build_multiseed_targets.py" \
  --formal-summary "${FORMAL_SUMMARY}" \
  --repeat-labels "${EXP4_REPEAT_LABELS}" \
  --repeat-labels "${RESULT_DIR}/completion_repeat_labels.csv" \
  --output "${RESULT_DIR}/multiseed_targets.csv" \
  --audit-output "${RESULT_DIR}/multiseed_targets.audit.json" \
  --missing-topologies-output "${RESULT_DIR}/missing_after_completion.csv" \
  --require-complete \
  > "${LOG_DIR}/target_build.log" 2>&1

write_phase "formal_graph_build_started"
python "${EXP5_REL}/scripts/build_incidence_dataset.py" \
  --formal-summary "${FORMAL_SUMMARY}" \
  --target-table "${RESULT_DIR}/multiseed_targets.csv" \
  --require-formal-targets \
  --output "${RESULT_DIR}/formal_topology_incidence_graphs.jsonl" \
  --audit-output "${RESULT_DIR}/formal_topology_incidence_graphs.audit.json" \
  > "${LOG_DIR}/formal_graph_build.log" 2>&1

write_phase "complete_gnn_not_started"
