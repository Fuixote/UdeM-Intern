#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_PYTHON="/home/weikang/miniconda3/envs/KEPs/bin/python"
PYTHON_BIN="${KEP_PYTHON:-$DEFAULT_PYTHON}"

DATA_DIR="${STEP1C_DATA_DIR:-dataset/processed/step1_noisy_linear_sigma010}"
DEFAULT_VALIDATION_DATA_DIR="dataset/processed/step1_noisy_linear_sigma010_validation2000_seed20260519"
VALIDATION_DATA_DIR="${STEP1C_VALIDATION_DATA_DIR-$DEFAULT_VALIDATION_DATA_DIR}"
SPLIT_SEED="${STEP1C_SPLIT_SEED:-42}"
SUBSET_SEED="${STEP1C_SUBSET_SEED:-42}"
THETA_SEED="${STEP1C_THETA_SEED:-42}"
GUROBI_SEED="${STEP1C_GUROBI_SEED:-42}"
TRAIN_POOL_SIZE="${STEP1C_TRAIN_POOL_SIZE:-1200}"
VAL_SIZE="${STEP1C_VAL_SIZE:-400}"
TEST_SIZE="${STEP1C_TEST_SIZE:-400}"
TRAIN_SIZE="${STEP1C_TRAIN_SIZE:-50}"
N_EPOCHS_2STAGE="${STEP1C_2STAGE_N_EPOCHS:-100}"
N_EPOCHS_SPOPLUS="${STEP1C_SPOPLUS_N_EPOCHS:-100}"
LR_2STAGE="${STEP1C_2STAGE_LR:-0.05}"
LR_SPOPLUS="${STEP1C_SPOPLUS_LR:-0.1}"
SPOPLUS_NORMALIZE_LOSS="${STEP1C_SPOPLUS_NORMALIZE_LOSS:-0}"
SPOPLUS_GRAD_CLIP="${STEP1C_SPOPLUS_GRAD_CLIP:-}"
SPOPLUS_WEIGHT_DECAY="${STEP1C_SPOPLUS_WEIGHT_DECAY:-0.0}"
METRIC_STRIDE="${STEP1C_METRIC_STRIDE:-1}"
SPOPLUS_EARLY_STOP_METRIC="${STEP1C_SPOPLUS_EARLY_STOP_METRIC:-}"
SPOPLUS_EARLY_STOP_PATIENCE="${STEP1C_SPOPLUS_EARLY_STOP_PATIENCE:-0}"
SPOPLUS_EARLY_STOP_MIN_DELTA="${STEP1C_SPOPLUS_EARLY_STOP_MIN_DELTA:-0.0}"
TRAIN_GRAPH_LIMIT="${STEP1C_TRAIN_GRAPH_LIMIT:-}"
VALIDATION_LIMIT="${STEP1C_VALIDATION_LIMIT:-}"
TEST_LIMIT="${STEP1C_TEST_LIMIT:-}"
THETA_INIT="${STEP1C_THETA_INIT:-}"
OUTPUT_ROOT="${STEP1C_OUTPUT_ROOT:-results/step1c_runs}"
SPLIT_PATH="${STEP1C_SPLIT_PATH:-results/step1b_splits/master_split_seed=$SPLIT_SEED.json}"
BOOTSTRAP_SAMPLES="${STEP1C_BOOTSTRAP_SAMPLES:-1000}"
BOOTSTRAP_SEED="${STEP1C_BOOTSTRAP_SEED:-42}"

EARLY_STOP_SUFFIX=""
EARLY_STOP_ARGS=()
if [[ -n "$SPOPLUS_EARLY_STOP_METRIC" ]]; then
  EARLY_STOP_SUFFIX="_earlystop=${SPOPLUS_EARLY_STOP_METRIC}_patience=${SPOPLUS_EARLY_STOP_PATIENCE}_mindelta=${SPOPLUS_EARLY_STOP_MIN_DELTA}"
  EARLY_STOP_ARGS=(
    --early_stop_metric "$SPOPLUS_EARLY_STOP_METRIC"
    --early_stop_patience "$SPOPLUS_EARLY_STOP_PATIENCE"
    --early_stop_min_delta "$SPOPLUS_EARLY_STOP_MIN_DELTA"
  )
fi

SPOPLUS_ARGS=()
LIMIT_ARGS=()
if [[ "$SPOPLUS_NORMALIZE_LOSS" == "1" || "$SPOPLUS_NORMALIZE_LOSS" == "true" || "$SPOPLUS_NORMALIZE_LOSS" == "TRUE" ]]; then
  SPOPLUS_ARGS+=(--normalize_loss)
fi
if [[ -n "$SPOPLUS_GRAD_CLIP" ]]; then
  SPOPLUS_ARGS+=(--grad_clip "$SPOPLUS_GRAD_CLIP")
fi
if [[ -n "$TRAIN_GRAPH_LIMIT" ]]; then
  LIMIT_ARGS+=(--train_graph_limit "$TRAIN_GRAPH_LIMIT")
fi
if [[ -n "$VALIDATION_LIMIT" ]]; then
  LIMIT_ARGS+=(--validation_limit "$VALIDATION_LIMIT")
fi

TEST_LIMIT_ARGS=()
if [[ -n "$TEST_LIMIT" ]]; then
  TEST_LIMIT_ARGS+=(--test_limit "$TEST_LIMIT")
fi

VALIDATION_SUFFIX=""
VALIDATION_ARGS=()
if [[ -n "$VALIDATION_DATA_DIR" ]]; then
  VALIDATION_TAG="$(basename "$VALIDATION_DATA_DIR")"
  VALIDATION_SUFFIX="_val=${VALIDATION_TAG}"
  VALIDATION_ARGS=(--validation_data_dir "$VALIDATION_DATA_DIR")
fi

DEFAULT_RUN_DIR="$OUTPUT_ROOT/train_size=$TRAIN_SIZE/split_seed=$SPLIT_SEED/subset_seed=$SUBSET_SEED/theta_seed=$THETA_SEED/spoplus_epochs=${N_EPOCHS_SPOPLUS}_stride=${METRIC_STRIDE}${EARLY_STOP_SUFFIX}${VALIDATION_SUFFIX}"
RUN_DIR="${STEP1C_OUTPUT_DIR:-$DEFAULT_RUN_DIR}"

echo "Step1c root: $ROOT_DIR"
echo "Python: $PYTHON_BIN"
echo "Data: $DATA_DIR"
echo "Validation data override: ${VALIDATION_DATA_DIR:-split validation}"
echo "Split: $SPLIT_PATH"
echo "Run dir: $RUN_DIR"
echo "master split train_pool=$TRAIN_POOL_SIZE val=$VAL_SIZE test=$TEST_SIZE split_seed=$SPLIT_SEED"
echo "train_size=$TRAIN_SIZE subset_seed=$SUBSET_SEED theta_seed=$THETA_SEED"
echo "2stage epochs=$N_EPOCHS_2STAGE lr=$LR_2STAGE"
echo "spoplus epochs=$N_EPOCHS_SPOPLUS lr=$LR_SPOPLUS normalize_loss=$SPOPLUS_NORMALIZE_LOSS grad_clip=${SPOPLUS_GRAD_CLIP:-none} weight_decay=$SPOPLUS_WEIGHT_DECAY metric_stride=$METRIC_STRIDE"
echo "spoplus early_stop_metric=${SPOPLUS_EARLY_STOP_METRIC:-disabled} patience=$SPOPLUS_EARLY_STOP_PATIENCE min_delta=$SPOPLUS_EARLY_STOP_MIN_DELTA"
echo "limits train_graph_limit=${TRAIN_GRAPH_LIMIT:-none} validation_limit=${VALIDATION_LIMIT:-none} test_limit=${TEST_LIMIT:-none}"
echo "theta_init=${THETA_INIT:-seeded random}"
echo "bootstrap_samples=$BOOTSTRAP_SAMPLES bootstrap_seed=$BOOTSTRAP_SEED"

"$PYTHON_BIN" - "$RUN_DIR/run_config.json" <<PY
import json
import sys
from pathlib import Path

config = {
    "data_dir": "$DATA_DIR",
    "validation_data_dir": "$VALIDATION_DATA_DIR" or None,
    "split_path": "$SPLIT_PATH",
    "run_dir": "$RUN_DIR",
    "split_seed": int("$SPLIT_SEED"),
    "subset_seed": int("$SUBSET_SEED"),
    "theta_seed": int("$THETA_SEED"),
    "gurobi_seed": int("$GUROBI_SEED"),
    "train_pool_size": int("$TRAIN_POOL_SIZE"),
    "validation_size": int("$VAL_SIZE"),
    "test_size": int("$TEST_SIZE"),
    "train_size": int("$TRAIN_SIZE"),
    "n_epochs_2stage": int("$N_EPOCHS_2STAGE"),
    "n_epochs_spoplus": int("$N_EPOCHS_SPOPLUS"),
    "lr_2stage": float("$LR_2STAGE"),
    "lr_spoplus": float("$LR_SPOPLUS"),
    "spoplus_normalize_loss": "$SPOPLUS_NORMALIZE_LOSS".lower() in {"1", "true", "yes"},
    "spoplus_grad_clip": float("$SPOPLUS_GRAD_CLIP") if "$SPOPLUS_GRAD_CLIP" else None,
    "spoplus_weight_decay": float("$SPOPLUS_WEIGHT_DECAY"),
    "metric_stride": int("$METRIC_STRIDE"),
    "early_stop_metric": "$SPOPLUS_EARLY_STOP_METRIC" or None,
    "early_stop_patience": int("$SPOPLUS_EARLY_STOP_PATIENCE"),
    "early_stop_min_delta": float("$SPOPLUS_EARLY_STOP_MIN_DELTA"),
    "train_graph_limit": int("$TRAIN_GRAPH_LIMIT") if "$TRAIN_GRAPH_LIMIT" else None,
    "validation_limit": int("$VALIDATION_LIMIT") if "$VALIDATION_LIMIT" else None,
    "test_limit": int("$TEST_LIMIT") if "$TEST_LIMIT" else None,
    "theta_init": "$THETA_INIT" or None,
    "bootstrap_samples": int("$BOOTSTRAP_SAMPLES"),
    "bootstrap_seed": int("$BOOTSTRAP_SEED"),
}
path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(config, indent=2), encoding="utf-8")
PY

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
  "${VALIDATION_ARGS[@]}" \
  --out_dir "$RUN_DIR" \
  --train_size "$TRAIN_SIZE" \
  --subset_seed "$SUBSET_SEED" \
  --theta_seed "$THETA_SEED" \
  --n_epochs "$N_EPOCHS_2STAGE" \
  --lr "$LR_2STAGE" \
  "${LIMIT_ARGS[@]}" \
  --plot \
  "${COMMON_THETA_ARGS[@]}"

MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}" \
"$PYTHON_BIN" "$SCRIPT_DIR/train_spoplus.py" \
  --split_path "$SPLIT_PATH" \
  "${VALIDATION_ARGS[@]}" \
  --out_dir "$RUN_DIR" \
  --train_size "$TRAIN_SIZE" \
  --subset_seed "$SUBSET_SEED" \
  --theta_seed "$THETA_SEED" \
  --gurobi_seed "$GUROBI_SEED" \
  --n_epochs "$N_EPOCHS_SPOPLUS" \
  --lr "$LR_SPOPLUS" \
  --metric_stride "$METRIC_STRIDE" \
  --weight_decay "$SPOPLUS_WEIGHT_DECAY" \
  "${EARLY_STOP_ARGS[@]}" \
  "${SPOPLUS_ARGS[@]}" \
  "${LIMIT_ARGS[@]}" \
  --plot \
  "${COMMON_THETA_ARGS[@]}"

"$PYTHON_BIN" "$SCRIPT_DIR/evaluate_models.py" \
  --split_path "$SPLIT_PATH" \
  --out_dir "$RUN_DIR" \
  --gurobi_seed "$GUROBI_SEED" \
  --bootstrap_samples "$BOOTSTRAP_SAMPLES" \
  --bootstrap_seed "$BOOTSTRAP_SEED" \
  "${TEST_LIMIT_ARGS[@]}" \
  --weights \
  "$RUN_DIR/model_weights/2stage_best_by_validation_mse_loss.npz" \
  "$RUN_DIR/model_weights/spoplus_best_by_validation_decision_gap.npz" \
  "$RUN_DIR/model_weights/spoplus_best_by_validation_spoplus_loss.npz"

echo
echo "Step1c run complete:"
echo "  $RUN_DIR/train_subset.json"
echo "  $RUN_DIR/model_weights/2stage_best_by_validation_mse_loss.npz"
echo "  $RUN_DIR/model_weights/spoplus_best_by_validation_decision_gap.npz"
echo "  $RUN_DIR/model_weights/spoplus_best_by_validation_spoplus_loss.npz"
echo "  $RUN_DIR/run_config.json"
echo "  $RUN_DIR/metrics/2stage_loss_curve.csv"
echo "  $RUN_DIR/metrics/spoplus_loss_curve.csv"
echo "  $RUN_DIR/metrics/early_stopping.json"
echo "  $RUN_DIR/metrics/test_summary.csv"
echo "  $RUN_DIR/plots/2stage_mse_loss.png"
echo "  $RUN_DIR/plots/spoplus_loss.png"
