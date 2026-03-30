#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

DEFAULT_PYTHON="/home/weikang/miniconda3/envs/KEPs/bin/python"
PYTHON_BIN="${KEP_PYTHON:-$DEFAULT_PYTHON}"
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage:
  ./run_experiment.sh [--dry-run] <command> [args...]

Commands:
  data-generate [generator args...]
      Run 0-data-generation.py with any extra generator arguments.

  data-process [processing args...]
      Run 1-data-processing.py with any extra processing arguments.
      Example: ./run_experiment.sh data-process --all

  2stg-gnn
      Run stage-1 GNN training, then stage-2 Gurobi solving on the latest checkpoint.

  2stg-reg
      Run stage-1 MLP regression training, then stage-2 Gurobi solving on the latest checkpoint.

  dfl-gnn [dfl args...]
      Run end-to-end GNN (Fenchel-Young / perturbed optimizer) training.

  dfl-reg [dfl args...]
      Run end-to-end MLP (Fenchel-Young / perturbed optimizer) training.

  oracle [optional_model_path] [solver args...]
      Run the ground-truth oracle solver. If a model checkpoint is supplied, its
      test_files.txt is copied for fair comparison. If omitted, the latest
      two-stage GNN checkpoint is used when available.

  evaluate [evaluation args...]
      Run 4-evaulation.py. Example: ./run_experiment.sh evaluate --full_eval

  app
      Launch the Flask visualization app.

  help
      Print this help text.
EOF
}

log() {
    echo "[run_experiment] $*"
}

run_cmd() {
    log "+ $*"
    if [ "$DRY_RUN" -eq 0 ]; then
        "$@"
    fi
}

run_python() {
    if [ ! -x "$PYTHON_BIN" ] && [ "$PYTHON_BIN" = "$DEFAULT_PYTHON" ]; then
        echo "Error: expected Python interpreter not found at $PYTHON_BIN" >&2
        echo "Set KEP_PYTHON=/path/to/python if your environment lives elsewhere." >&2
        exit 1
    fi
    run_cmd "$PYTHON_BIN" "$@"
}

latest_result_dir() {
    local prefix="$1"
    ls -td "results/${prefix}"* 2>/dev/null | head -n 1 || true
}

latest_checkpoint() {
    local prefix="$1"
    local filename="$2"
    local latest_dir
    latest_dir="$(latest_result_dir "$prefix")"
    if [ -z "$latest_dir" ]; then
        return 1
    fi
    local checkpoint="${latest_dir}/${filename}"
    if [ ! -f "$checkpoint" ]; then
        return 1
    fi
    printf '%s\n' "$checkpoint"
}

resolve_oracle_reference_model() {
    local checkpoint
    for prefix in "2stg_Gnn_" "2stg_Reg_" "dfl_Gnn_" "dfl_Reg_"; do
        if checkpoint="$(latest_checkpoint "$prefix" "best_stage1_model_real.pth" 2>/dev/null)"; then
            printf '%s\n' "$checkpoint"
            return 0
        fi
        if checkpoint="$(latest_checkpoint "$prefix" "best_dfl_model.pth" 2>/dev/null)"; then
            printf '%s\n' "$checkpoint"
            return 0
        fi
        if checkpoint="$(latest_checkpoint "$prefix" "best_dfl_reg_model.pth" 2>/dev/null)"; then
            printf '%s\n' "$checkpoint"
            return 0
        fi
    done
    return 1
}

run_two_stage() {
    local train_script="$1"
    local prefix="$2"
    run_python "$train_script"
    local model_path
    model_path="$(latest_checkpoint "$prefix" "best_stage1_model_real.pth")"
    log "Using latest checkpoint: $model_path"
    run_python 3-stage2-solver-gurobi.py --model_path "$model_path"
}

if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=1
    shift
fi

COMMAND="${1:-help}"
if [ $# -gt 0 ]; then
    shift
fi

case "$COMMAND" in
    data-generate)
        run_python 0-data-generation.py "$@"
        ;;
    data-process)
        run_python 1-data-processing.py "$@"
        ;;
    2stg-gnn)
        run_two_stage 2-stage1-training-GNN.py "2stg_Gnn_"
        ;;
    2stg-reg)
        run_two_stage 2-stage1-training-Reg.py "2stg_Reg_"
        ;;
    dfl-gnn)
        run_python 2-end2end-GNN.py "$@"
        ;;
    dfl-reg)
        run_python 2-end2end-Reg.py "$@"
        ;;
    oracle)
        oracle_args=(--gt_mode)
        if [ $# -gt 0 ] && [[ "$1" == *.pth ]]; then
            oracle_args+=(--model_path "$1")
            shift
        elif model_path="$(resolve_oracle_reference_model 2>/dev/null)"; then
            oracle_args+=(--model_path "$model_path")
            log "Oracle will reuse test split from: $model_path"
        fi
        oracle_args+=("$@")
        run_python 3-stage2-solver-gurobi.py "${oracle_args[@]}"
        ;;
    evaluate)
        run_python 4-evaulation.py "$@"
        ;;
    app)
        run_python app.py
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        echo "Error: unknown command '$COMMAND'" >&2
        echo >&2
        usage >&2
        exit 1
        ;;
esac
