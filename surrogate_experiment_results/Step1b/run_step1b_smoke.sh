#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_PYTHON="/home/weikang/miniconda3/envs/KEPs/bin/python"
PYTHON_BIN="${KEP_PYTHON:-$DEFAULT_PYTHON}"

DATA_DIR="${STEP1B_DATA_DIR:-dataset/processed/step1_noisy_linear_sigma010}"
TRAIN_SIZE="${STEP1B_TRAIN_SIZE:-2}"
VAL_SIZE="${STEP1B_VAL_SIZE:-2}"
TEST_SIZE="${STEP1B_TEST_SIZE:-2}"
N_EPOCHS="${STEP1B_N_EPOCHS:-1}"
FY_EPSILON="${STEP1B_FY_EPSILON:-1.0}"
FY_M="${STEP1B_FY_M:-2}"
LR_FY="${STEP1B_LR_FY:-0.1}"
SEED="${STEP1B_SEED:-42}"
CHECKPOINT_STRIDE="${STEP1B_CHECKPOINT_STRIDE:-1}"
METHODS="${STEP1B_METHODS:-mse fy_warm}"
CHECKPOINT_RULES="${STEP1B_CHECKPOINT_RULES:-validation_decision_gap}"
OUT_DIR="${STEP1B_OUTPUT_DIR:-results/step1b_runs/smoke/seed=$SEED-train=$TRAIN_SIZE-val=$VAL_SIZE-test=$TEST_SIZE}"

echo "Step1b smoke root: $ROOT_DIR"
echo "Python: $PYTHON_BIN"
echo "Data: $DATA_DIR"
echo "Output: $OUT_DIR"
echo "split train=$TRAIN_SIZE val=$VAL_SIZE test=$TEST_SIZE seed=$SEED"
echo "FY epochs=$N_EPOCHS epsilon=$FY_EPSILON M=$FY_M lr=$LR_FY"
echo "methods=$METHODS"
echo "checkpoint_rules=$CHECKPOINT_RULES"

"$PYTHON_BIN" "$SCRIPT_DIR/generalization_experiment.py" \
  --data_dir "$DATA_DIR" \
  --out_dir "$OUT_DIR" \
  --train_size "$TRAIN_SIZE" \
  --val_size "$VAL_SIZE" \
  --test_size "$TEST_SIZE" \
  --seed "$SEED" \
  --n_epochs "$N_EPOCHS" \
  --lr_fy "$LR_FY" \
  --fy_epsilon "$FY_EPSILON" \
  --fy_M "$FY_M" \
  --checkpoint_stride "$CHECKPOINT_STRIDE" \
  --methods $METHODS \
  --checkpoint_rules $CHECKPOINT_RULES

echo
echo "Step1b smoke complete. Key outputs:"
echo "  $OUT_DIR/config.json"
echo "  $OUT_DIR/split.json"
echo "  $OUT_DIR/metrics/validation_trajectory_metrics.csv"
echo "  $OUT_DIR/metrics/test_summary.csv"
echo "  $OUT_DIR/metrics/test_per_graph.csv"
