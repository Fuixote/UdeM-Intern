#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/local1/fuweik/UdeM-Intern"
EXP_REL="surrogate_experiment_results/Step5_topology_dfl_suitability/experiment_06_hurdle_gnn"
EXP5_REL="surrogate_experiment_results/Step5_topology_dfl_suitability/experiment_05_topology_gnn_regression"
OUTPUT_REL="${EXP_REL}/results/formal_seed42_fivefold"
GRAPH_JSONL="${EXP5_REL}/results/multiseed_completion1880/results/formal_topology_incidence_graphs.jsonl"
FOLDS_CSV="${EXP5_REL}/results/formal_three_seed/splits/folds.csv"
CLASSIFIER_CSV="${OUTPUT_REL}/plans/classifier_jobs.csv"
REGRESSOR_CSV="${OUTPUT_REL}/plans/regressor_jobs.csv"
LOG_DIR="${OUTPUT_REL}/logs"
WORKERS="${WORKERS:-1}"

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
  printf '[exp6-hurdle-seed42] %s %s\n' "$(date --iso-8601=seconds)" "$1"
}

write_phase "plan_started folds=0,1,2,3,4 seed=42 classifier_jobs=5 regressor_jobs=50"
python "${EXP_REL}/scripts/plan_hurdle_jobs.py" \
  --graph-jsonl "${GRAPH_JSONL}" \
  --folds "${FOLDS_CSV}" \
  --output-root "${OUTPUT_REL}" \
  --seed 42 \
  --expected-classifier-jobs 5 \
  --expected-regressor-jobs 50 \
  --max-epochs 300 \
  --early-stop-patience 30 \
  --threads 4 \
  > "${LOG_DIR}/plan.log" 2>&1

write_phase "classifier_preview_started jobs=5 workers=${WORKERS}"
python "${EXP_REL}/scripts/launch_hurdle_jobs.py" \
  --jobs-csv "${CLASSIFIER_CSV}" \
  --output-root "${OUTPUT_REL}" \
  --expected-job-count 5 \
  --workers "${WORKERS}" \
  --monitor-interval 15 \
  --require-hostname garnet \
  > "${LOG_DIR}/classifier_preview.log" 2>&1

write_phase "classifier_training_started jobs=5 workers=${WORKERS}"
python "${EXP_REL}/scripts/launch_hurdle_jobs.py" \
  --jobs-csv "${CLASSIFIER_CSV}" \
  --output-root "${OUTPUT_REL}" \
  --expected-job-count 5 \
  --workers "${WORKERS}" \
  --monitor-interval 15 \
  --require-hostname garnet \
  --execute \
  > "${LOG_DIR}/classifier_launcher.log" 2>&1

write_phase "regressor_preview_started jobs=50 workers=${WORKERS}"
python "${EXP_REL}/scripts/launch_hurdle_jobs.py" \
  --jobs-csv "${REGRESSOR_CSV}" \
  --output-root "${OUTPUT_REL}" \
  --expected-job-count 50 \
  --workers "${WORKERS}" \
  --monitor-interval 15 \
  --require-hostname garnet \
  --require-dependencies \
  > "${LOG_DIR}/regressor_preview.log" 2>&1

write_phase "regressor_training_started jobs=50 workers=${WORKERS}"
python "${EXP_REL}/scripts/launch_hurdle_jobs.py" \
  --jobs-csv "${REGRESSOR_CSV}" \
  --output-root "${OUTPUT_REL}" \
  --expected-job-count 50 \
  --workers "${WORKERS}" \
  --monitor-interval 15 \
  --require-hostname garnet \
  --require-dependencies \
  --execute \
  > "${LOG_DIR}/regressor_launcher.log" 2>&1

write_phase "review_started classifier_jobs=5 regressor_jobs=50"
python "${EXP_REL}/scripts/review_hurdle_results.py" \
  --classifier-jobs "${CLASSIFIER_CSV}" \
  --regressor-jobs "${REGRESSOR_CSV}" \
  --output-root "${OUTPUT_REL}" \
  --expected-classifier-jobs 5 \
  --expected-regressor-jobs 50 \
  > "${LOG_DIR}/review.log" 2>&1

write_phase "complete_hurdle_seed42_fivefold_review_passed"
