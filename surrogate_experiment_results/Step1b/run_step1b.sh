#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_PYTHON="/home/weikang/miniconda3/envs/KEPs/bin/python"
PYTHON_BIN="${KEP_PYTHON:-$DEFAULT_PYTHON}"

DATA_DIR="${STEP1B_DATA_DIR:-dataset/processed/step1_noisy_linear_sigma010}"
SPLIT_SEED="${STEP1B_SPLIT_SEED:-42}"
SUBSET_SEED="${STEP1B_SUBSET_SEED:-42}"
THETA_SEED="${STEP1B_THETA_SEED:-42}"
GUROBI_SEED="${STEP1B_GUROBI_SEED:-42}"
TRAIN_POOL_SIZE="${STEP1B_TRAIN_POOL_SIZE:-1200}"
VAL_SIZE="${STEP1B_VAL_SIZE:-400}"
TEST_SIZE="${STEP1B_TEST_SIZE:-400}"
TRAIN_SIZE="${STEP1B_TRAIN_SIZE:-50}"
N_EPOCHS_2STAGE="${STEP1B_2STAGE_N_EPOCHS:-100}"
N_EPOCHS_E2E="${STEP1B_E2E_N_EPOCHS:-100}"
LR_2STAGE="${STEP1B_2STAGE_LR:-0.05}"
LR_E2E="${STEP1B_E2E_LR:-0.1}"
FY_EPSILON="${STEP1B_FY_EPSILON:-1.0}"
FY_M="${STEP1B_FY_M:-4}"
METRIC_STRIDE="${STEP1B_METRIC_STRIDE:-1}"
THETA_INIT="${STEP1B_THETA_INIT:-}"
OUTPUT_ROOT="${STEP1B_OUTPUT_ROOT:-results/step1b_runs}"
SPLIT_PATH="${STEP1B_SPLIT_PATH:-results/step1b_splits/master_split_seed=$SPLIT_SEED.json}"
RUN_DIR="${STEP1B_OUTPUT_DIR:-$OUTPUT_ROOT/train_size=$TRAIN_SIZE/seed=$THETA_SEED}"

echo "Step1b root: $ROOT_DIR"
echo "Python: $PYTHON_BIN"
echo "Data: $DATA_DIR"
echo "Split: $SPLIT_PATH"
echo "Run dir: $RUN_DIR"
echo "master split train_pool=$TRAIN_POOL_SIZE val=$VAL_SIZE test=$TEST_SIZE split_seed=$SPLIT_SEED"
echo "train_size=$TRAIN_SIZE subset_seed=$SUBSET_SEED theta_seed=$THETA_SEED"
echo "2stage epochs=$N_EPOCHS_2STAGE lr=$LR_2STAGE"
echo "e2e epochs=$N_EPOCHS_E2E lr=$LR_E2E fy_epsilon=$FY_EPSILON M=$FY_M metric_stride=$METRIC_STRIDE"
echo "theta_init=${THETA_INIT:-seeded random}"

"$PYTHON_BIN" "$SCRIPT_DIR/split_dataset.py" \
  --data_dir "$DATA_DIR" \
  --split_path "$SPLIT_PATH" \
  --train_pool_size "$TRAIN_POOL_SIZE" \
  --val_size "$VAL_SIZE" \
  --test_size "$TEST_SIZE" \
  --seed "$SPLIT_SEED" \
  --reuse_if_exists

COMMON_THETA_ARGS=()
if [[ -n "$THETA_INIT" ]]; then
  read -r -a THETA_INIT_ARGS <<< "$THETA_INIT"
  COMMON_THETA_ARGS=(--theta_init "${THETA_INIT_ARGS[@]}")
fi

MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}" \
"$PYTHON_BIN" "$SCRIPT_DIR/train_2stage.py" \
  --split_path "$SPLIT_PATH" \
  --out_dir "$RUN_DIR" \
  --train_size "$TRAIN_SIZE" \
  --subset_seed "$SUBSET_SEED" \
  --theta_seed "$THETA_SEED" \
  --n_epochs "$N_EPOCHS_2STAGE" \
  --lr "$LR_2STAGE" \
  --plot \
  "${COMMON_THETA_ARGS[@]}"

MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}" \
"$PYTHON_BIN" "$SCRIPT_DIR/train_end2end.py" \
  --split_path "$SPLIT_PATH" \
  --out_dir "$RUN_DIR" \
  --train_size "$TRAIN_SIZE" \
  --subset_seed "$SUBSET_SEED" \
  --theta_seed "$THETA_SEED" \
  --gurobi_seed "$GUROBI_SEED" \
  --n_epochs "$N_EPOCHS_E2E" \
  --lr "$LR_E2E" \
  --fy_epsilon "$FY_EPSILON" \
  --fy_M "$FY_M" \
  --metric_stride "$METRIC_STRIDE" \
  --plot \
  "${COMMON_THETA_ARGS[@]}"

"$PYTHON_BIN" "$SCRIPT_DIR/evaluate_models.py" \
  --split_path "$SPLIT_PATH" \
  --out_dir "$RUN_DIR" \
  --gurobi_seed "$GUROBI_SEED" \
  --weights \
  "$RUN_DIR/model_weights/2stage_best_by_validation_mse_loss.npz" \
  "$RUN_DIR/model_weights/e2e_best_by_validation_decision_gap.npz"

echo
echo "Step1b run complete:"
echo "  $RUN_DIR/train_subset.json"
echo "  $RUN_DIR/model_weights/2stage_best_by_validation_mse_loss.npz"
echo "  $RUN_DIR/model_weights/e2e_best_by_validation_decision_gap.npz"
echo "  $RUN_DIR/metrics/2stage_loss_curve.csv"
echo "  $RUN_DIR/metrics/e2e_loss_curve.csv"
echo "  $RUN_DIR/metrics/test_summary.csv"
echo "  $RUN_DIR/plots/2stage_mse_loss.png"
echo "  $RUN_DIR/plots/e2e_fy_loss.png"
