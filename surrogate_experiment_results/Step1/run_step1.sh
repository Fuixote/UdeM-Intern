#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_PYTHON="/home/weikang/miniconda3/envs/KEPs/bin/python"
PYTHON_BIN="${KEP_PYTHON:-$DEFAULT_PYTHON}"

DATA_DIR="${STEP1_DATA_DIR:-dataset/processed/clean_linear_dataset}"
N_TOTAL="${STEP1_N_TOTAL:-100}"
N_EPOCHS="${STEP1_N_EPOCHS:-1000}"
LR_MSE="${STEP1_LR_MSE:-0.05}"
LR_FY="${STEP1_LR_FY:-0.1}"
FY_EPSILON="${STEP1_FY_EPSILON:-1.0}"
FY_M="${STEP1_FY_M:-16}"
SEED="${STEP1_SEED:-42}"
GRID_SIZE="${STEP1_GRID_SIZE:-25}"
N_MILESTONES="${STEP1_N_MILESTONES:-5}"
STEP1_SCRIPT_DIR="surrogate_experiment_results/Step1"
STEP1_OUT_DIR="${STEP1_OUTPUT_DIR:-$STEP1_SCRIPT_DIR/epsilon=$FY_EPSILON}"
STEP1_OUT_ABS_DIR="$ROOT_DIR/$STEP1_OUT_DIR"

mkdir -p "$STEP1_OUT_DIR"

echo "Step1 root: $ROOT_DIR"
echo "Python: $PYTHON_BIN"
echo "Data: $DATA_DIR"
echo "Output: $STEP1_OUT_DIR"
echo "n_total=$N_TOTAL n_epochs=$N_EPOCHS fy_epsilon=$FY_EPSILON fy_M=$FY_M seed=$SEED"

echo
echo "[1/6] Generate MSE and FY parameter trajectories"
"$PYTHON_BIN" "$STEP1_SCRIPT_DIR/Step1.py" \
  --data_dir "$DATA_DIR" \
  --out_dir "$STEP1_OUT_DIR" \
  --n_total "$N_TOTAL" \
  --n_epochs "$N_EPOCHS" \
  --lr_mse "$LR_MSE" \
  --lr_fy "$LR_FY" \
  --fy_epsilon "$FY_EPSILON" \
  --fy_M "$FY_M" \
  --seed "$SEED"

echo
echo "[2/6] Append True Regret to MSE trajectory"
"$PYTHON_BIN" "$STEP1_SCRIPT_DIR/add_true_regret_to_trajectory.py" \
  --data_dir "$DATA_DIR" \
  --traj_path "$STEP1_OUT_DIR/trajectory_mse.npy" \
  --out_path "$STEP1_OUT_DIR/trajectory_mse_with_regret.npy" \
  --n_total "$N_TOTAL" \
  --seed "$SEED"

echo
echo "[3/6] Append FY Loss to FY trajectory"
"$PYTHON_BIN" "$STEP1_SCRIPT_DIR/add_FY_loss_to_trajectory.py" \
  --data_dir "$DATA_DIR" \
  --traj_path "$STEP1_OUT_DIR/trajectory_fy.npy" \
  --out_path "$STEP1_OUT_DIR/trajectory_fy_with_fy_loss.npy" \
  --n_total "$N_TOTAL" \
  --fy_epsilon "$FY_EPSILON" \
  --fy_M "$FY_M" \
  --seed "$SEED"

echo
echo "[4/6] Append True Regret to FY trajectory with FY Loss"
"$PYTHON_BIN" "$STEP1_SCRIPT_DIR/add_true_regret_to_trajectory.py" \
  --data_dir "$DATA_DIR" \
  --traj_path "$STEP1_OUT_DIR/trajectory_fy_with_fy_loss.npy" \
  --out_path "$STEP1_OUT_DIR/trajectory_fy_with_fy_loss_and_regret.npy" \
  --n_total "$N_TOTAL" \
  --seed "$SEED"

echo
echo "[5/6] Plot 2D True Regret landscape and trajectories"
"$PYTHON_BIN" "$STEP1_SCRIPT_DIR/plot_trajectories_2D.py" \
  --data_dir "$DATA_DIR" \
  --mse_traj_path "$STEP1_OUT_ABS_DIR/trajectory_mse_with_regret.npy" \
  --fy_traj_path "$STEP1_OUT_ABS_DIR/trajectory_fy_with_fy_loss_and_regret.npy" \
  --out_dir "$STEP1_OUT_DIR" \
  --n_total "$N_TOTAL" \
  --grid_size "$GRID_SIZE" \
  --seed "$SEED" \
  --n_milestones "$N_MILESTONES"

echo
echo "[6/6] Plot 3D trajectory metrics"
"$PYTHON_BIN" "$STEP1_SCRIPT_DIR/plot_trajectories_3D.py" \
  --mse_path "$STEP1_OUT_ABS_DIR/trajectory_mse_with_regret.npy" \
  --fy_path "$STEP1_OUT_ABS_DIR/trajectory_fy_with_fy_loss_and_regret.npy" \
  --out_path "$STEP1_OUT_ABS_DIR/trajectory_3d_metrics.png"

echo
echo "Step1 complete. Outputs:"
echo "  $STEP1_OUT_DIR/trajectory_mse.npy"
echo "  $STEP1_OUT_DIR/trajectory_fy.npy"
echo "  $STEP1_OUT_DIR/trajectory_mse_with_regret.npy"
echo "  $STEP1_OUT_DIR/trajectory_fy_with_fy_loss.npy"
echo "  $STEP1_OUT_DIR/trajectory_fy_with_fy_loss_and_regret.npy"
echo "  $STEP1_OUT_DIR/trajectory_contour.png"
echo "  $STEP1_OUT_DIR/trajectory_3d_metrics.png"
