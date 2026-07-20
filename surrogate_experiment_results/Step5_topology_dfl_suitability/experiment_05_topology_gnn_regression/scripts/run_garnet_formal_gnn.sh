#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/local1/fuweik/UdeM-Intern"
EXP_REL="surrogate_experiment_results/Step5_topology_dfl_suitability/experiment_05_topology_gnn_regression"
OUTPUT_REL="${EXP_REL}/results/formal_three_seed/gnn_formal15"
GRAPH_JSONL="${EXP_REL}/results/multiseed_completion1880/results/formal_topology_incidence_graphs.jsonl"
FOLDS_CSV="${EXP_REL}/results/formal_three_seed/splits/folds.csv"
PLAN_JSON="${OUTPUT_REL}/plans/formal_gnn15_plan.json"
JOBS_CSV="${OUTPUT_REL}/plans/formal_gnn15_jobs.csv"
LOG_DIR="${OUTPUT_REL}/logs"
WORKERS="${WORKERS:-3}"

cd "${PROJECT_ROOT}"
source configs/runtime/garnet.env
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export PYTHONUNBUFFERED=1
mkdir -p "${LOG_DIR}"

write_phase() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$1" > "${OUTPUT_REL}/pipeline_status.txt"
  printf '[exp5-formal-gnn] %s %s\n' "$(date --iso-8601=seconds)" "$1"
}

write_phase "strict_plan_started jobs=15 workers=${WORKERS} threads_per_job=4"
python "${EXP_REL}/scripts/plan_formal_gnn_jobs.py" \
  --graph-jsonl "${GRAPH_JSONL}" \
  --folds "${FOLDS_CSV}" \
  --output-root "${OUTPUT_REL}" \
  --plan-output "${PLAN_JSON}" \
  --jobs-csv-output "${JOBS_CSV}" \
  --expected-job-count 15 \
  --max-epochs 500 \
  --early-stop-patience 30 \
  --early-stop-min-delta 0.0001 \
  --threads 4 \
  > "${LOG_DIR}/plan.log" 2>&1

write_phase "launcher_preview_started jobs=15 workers=${WORKERS}"
python "${EXP_REL}/scripts/launch_formal_gnn_jobs.py" \
  --jobs-csv "${JOBS_CSV}" \
  --output-root "${OUTPUT_REL}" \
  --expected-job-count 15 \
  --workers "${WORKERS}" \
  --monitor-interval 15 \
  --require-hostname garnet \
  > "${LOG_DIR}/launcher_preview.log" 2>&1

write_phase "training_started jobs=15 workers=${WORKERS} threads_per_job=4"
python "${EXP_REL}/scripts/launch_formal_gnn_jobs.py" \
  --jobs-csv "${JOBS_CSV}" \
  --output-root "${OUTPUT_REL}" \
  --expected-job-count 15 \
  --workers "${WORKERS}" \
  --monitor-interval 15 \
  --require-hostname garnet \
  --execute \
  > "${LOG_DIR}/launcher.log" 2>&1

write_phase "review_started jobs=15 expected_predictions=3000 expected_ensemble=1000"
python "${EXP_REL}/scripts/review_formal_gnn_results.py" \
  --jobs-csv "${JOBS_CSV}" \
  --output-root "${OUTPUT_REL}" \
  --folds "${FOLDS_CSV}" \
  --expected-job-count 15 \
  > "${LOG_DIR}/review.log" 2>&1

write_phase "complete_formal_gnn_review_passed"
