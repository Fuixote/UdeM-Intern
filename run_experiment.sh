#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

DEFAULT_PYTHON="/home/weikang/miniconda3/envs/KEPs/bin/python"
PYTHON_BIN="${KEP_PYTHON:-$DEFAULT_PYTHON}"
RAW_DATA_DIR="${KEP_RAW_DATA_DIR:-dataset/raw}"
PROCESSED_DATA_DIR="${KEP_DATA_DIR:-dataset/processed}"
RESULTS_ROOT="${KEP_RESULTS_DIR:-results}"
SOLUTIONS_ROOT="${KEP_SOLUTIONS_DIR:-solutions}"
STAGE1_EPOCHS=""
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage:
  ./run_experiment.sh [--dry-run] <command> [common options] [command args...]

Common options:
  --solver <hybrid|cf|pief>
      Explicitly choose the solver/formulation for `2stg-*`, `dfl-*`, and `oracle`.
      If omitted, `hybrid` is used by default.

  --data_dir <path>
      Explicitly set processed graph data root/batch. This replaces relying on
      `KEP_DATA_DIR` and is forwarded to all downstream Python entry points.

  --raw_data_dir <path>
      Explicitly set raw data root. This replaces relying on `KEP_RAW_DATA_DIR`.

  --results_root <path>
      Explicitly set results root. This replaces relying on `KEP_RESULTS_DIR`.

  --solutions_root <path>
      Explicitly set solutions root. This replaces relying on `KEP_SOLUTIONS_DIR`.

  --python <path>
      Explicitly set the Python interpreter. This replaces relying on `KEP_PYTHON`.

  --epochs <int>
      Set the number of stage-1 training epochs for `2stg-reg` and `2stg-lr`.

Commands:
  data-generate [generator args...]
      Run 0-data-generation.py with any extra generator arguments.

  data-process [processing args...]
      Run 1-data-processing.py with any extra processing arguments.
      With no raw batch directory, scan dataset/raw and repair missing processed batches.
      Example: ./run_experiment.sh data-process --raw_data_dir dataset/raw --data_dir dataset/processed
      Example: ./run_experiment.sh data-process dataset/raw/<batch_name> dataset/processed --all

  2stg-gnn
      Run stage-1 GNN training, then stage-2 hybrid (CF-cycle + PIEF-chain)
      solving on the latest checkpoint.
      Example: ./run_experiment.sh 2stg-gnn --data_dir dataset/processed/<batch_name>
      Example: ./run_experiment.sh 2stg-gnn --solver cf --data_dir dataset/processed/<batch_name>

  2stg-reg
      Run stage-1 Reg training (MLP tabular regression), then stage-2 hybrid
      (CF-cycle + PIEF-chain) solving on the latest checkpoint.

  2stg-lr
      Run stage-1 LR training (linear tabular regression), then stage-2 hybrid
      (CF-cycle + PIEF-chain) solving on the latest checkpoint.

  dfl-gnn [--pretrain_PATH <2stg_gnn_checkpoint>] [dfl args...]
      Run end-to-end GNN training with the hybrid formulation by default.
      If --pretrain_PATH is omitted, train from scratch with a deterministic 60/20/20 split.
      Example: ./run_experiment.sh dfl-gnn --data_dir dataset/processed/<batch_name> --pretrain_PATH results/2stg_Gnn_<timestamp>/best_stage1_model_real.pth
      Example: ./run_experiment.sh dfl-gnn --data_dir dataset/processed/<batch_name>

  dfl-reg [--pretrain_PATH <2stg_reg_checkpoint>] [dfl args...]
      Run end-to-end MLP training with the hybrid formulation by default.
      If --pretrain_PATH is omitted, train from scratch with a deterministic 60/20/20 split.
      Example: ./run_experiment.sh dfl-reg --data_dir dataset/processed/<batch_name> --pretrain_PATH results/2stg_Reg_<timestamp>/best_stage1_model_real.pth
      Example: ./run_experiment.sh dfl-reg --data_dir dataset/processed/<batch_name>

  dfl-lr [--pretrain_PATH <2stg_lr_checkpoint>] [dfl args...]
      Run end-to-end linear regression training with the hybrid formulation by
      default. If --pretrain_PATH is omitted, train from scratch with a
      deterministic 60/20/20 split.

  2stg-gnn-cf
      Run stage-1 GNN training, then stage-2 CF Gurobi solving on the latest checkpoint.

  2stg-reg-cf
      Run stage-1 Reg training (MLP tabular regression), then stage-2 CF
      Gurobi solving on the latest checkpoint.

  2stg-lr-cf
      Run stage-1 LR training (linear tabular regression), then stage-2 CF
      Gurobi solving on the latest checkpoint.

  2stg-gnn-pief
      Run stage-1 GNN training, then stage-2 dual-PIEF solving on the latest checkpoint.

  2stg-reg-pief
      Run stage-1 Reg training (MLP tabular regression), then stage-2
      dual-PIEF solving on the latest checkpoint.

  2stg-lr-pief
      Run stage-1 LR training (linear tabular regression), then stage-2
      dual-PIEF solving on the latest checkpoint.

  2stg-gnn-hybrid
      Run stage-1 GNN training, then stage-2 CF-cycle + PIEF-chain solving.

  2stg-reg-hybrid
      Run stage-1 Reg training (MLP tabular regression), then stage-2 CF-cycle
      + PIEF-chain solving.

  2stg-lr-hybrid
      Run stage-1 LR training (linear tabular regression), then stage-2
      CF-cycle + PIEF-chain solving.

  dfl-gnn-cf [--pretrain_PATH <2stg_gnn_checkpoint>] [dfl args...]
      Run end-to-end GNN (Fenchel-Young / perturbed optimizer) training with
      the CF formulation and an explicit warm-start checkpoint. The resulting
      dfl_Gnn_cf_* folder names
      will include the source 2stg_Gnn timestamp for easier comparison.

  dfl-reg-cf [--pretrain_PATH <2stg_reg_checkpoint>] [dfl args...]
      Run end-to-end MLP (Fenchel-Young / perturbed optimizer) training with
      the CF formulation and an explicit warm-start checkpoint. The resulting
      dfl_Reg_cf_* folder names
      will include the source 2stg_Reg timestamp for easier comparison.

  dfl-lr-cf [--pretrain_PATH <2stg_lr_checkpoint>] [dfl args...]
      Run end-to-end linear regression training with the CF formulation and an
      explicit warm-start checkpoint.

  dfl-gnn-pief [--pretrain_PATH <2stg_gnn_checkpoint>] [dfl args...]
      Run end-to-end GNN training with the dual-PIEF formulation.

  dfl-reg-pief [--pretrain_PATH <2stg_reg_checkpoint>] [dfl args...]
      Run end-to-end MLP training with the dual-PIEF formulation.

  dfl-lr-pief [--pretrain_PATH <2stg_lr_checkpoint>] [dfl args...]
      Run end-to-end linear regression training with the dual-PIEF formulation.

  dfl-gnn-hybrid [--pretrain_PATH <2stg_gnn_checkpoint>] [dfl args...]
      Run end-to-end GNN training with CF-cycle + PIEF-chain.

  dfl-reg-hybrid [--pretrain_PATH <2stg_reg_checkpoint>] [dfl args...]
      Run end-to-end MLP training with CF-cycle + PIEF-chain.

  dfl-lr-hybrid [--pretrain_PATH <2stg_lr_checkpoint>] [dfl args...]
      Run end-to-end linear regression training with CF-cycle + PIEF-chain.

  oracle [optional_model_path] [solver args...]
      Run the ground-truth oracle solver. If a model checkpoint is supplied, its
      test_files.txt is copied for fair comparison. If omitted, the latest
      two-stage GNN checkpoint is used when available. By default this uses the
      hybrid solver.

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
    run_cmd env \
        KEP_PYTHON="$PYTHON_BIN" \
        KEP_RAW_DATA_DIR="$RAW_DATA_DIR" \
        KEP_DATA_DIR="$PROCESSED_DATA_DIR" \
        KEP_RESULTS_DIR="$RESULTS_ROOT" \
        KEP_SOLUTIONS_DIR="$SOLUTIONS_ROOT" \
        "$PYTHON_BIN" "$@"
}

stage1_epoch_args() {
    if [ -n "$STAGE1_EPOCHS" ]; then
        printf '%s\0' --epochs "$STAGE1_EPOCHS"
    fi
}

normalize_data_process_args() {
    local has_target=0
    local positional_count=0
    local arg

    for arg in "$@"; do
        case "$arg" in
            --all|--file)
                has_target=1
                ;;
            -*)
                ;;
            *)
                positional_count=$((positional_count + 1))
                ;;
        esac
    done

    if [ "$has_target" -eq 0 ] && [ "$positional_count" -gt 0 ]; then
        printf '%s\0' "$@" "--all"
    else
        printf '%s\0' "$@"
    fi
}

resolve_processed_data_dir() {
    local requested_dir="${1:-$PROCESSED_DATA_DIR}"

    if compgen -G "${requested_dir}/G-*.json" > /dev/null; then
        printf '%s\n' "$requested_dir"
        return 0
    fi

    local latest_batch=""
    latest_batch="$(
        find "$requested_dir" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' 2>/dev/null \
        | grep -E '^[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{6}(__.+)?$' \
        | sort -r \
        | head -n 1
    )"

    if [ -n "$latest_batch" ] && compgen -G "${requested_dir}/${latest_batch}/G-*.json" > /dev/null; then
        echo "[run_experiment] No top-level processed graphs found in $requested_dir; using latest batch ${requested_dir}/${latest_batch}" >&2
        printf '%s\n' "${requested_dir}/${latest_batch}"
        return 0
    fi

    printf '%s\n' "$requested_dir"
}

latest_result_dir() {
    local prefix="$1"
    ls -td "${RESULTS_ROOT}/${prefix}"* 2>/dev/null | head -n 1 || true
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
    for prefix in "2stg_Gnn_" "2stg_Reg_" "2stg_LR_" "dfl_Gnn_" "dfl_Reg_" "dfl_LR_"; do
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

require_option_value() {
    local option_name="$1"
    if [ $# -lt 2 ] || [ -z "${2:-}" ]; then
        echo "Error: $option_name requires a value" >&2
        exit 1
    fi
}

validate_solver_name() {
    case "$1" in
        hybrid|cf|pief)
            ;;
        *)
            echo "Error: unsupported solver '$1' (expected one of: hybrid, cf, pief)" >&2
            exit 1
            ;;
    esac
}

parse_common_cli_args() {
    COMMAND_ARGS=()

    while [ $# -gt 0 ]; do
        case "$1" in
            --solver)
                require_option_value "$1" "${2:-}"
                REQUESTED_SOLVER="$2"
                validate_solver_name "$REQUESTED_SOLVER"
                shift 2
                ;;
            --data_dir)
                require_option_value "$1" "${2:-}"
                PROCESSED_DATA_DIR="$2"
                shift 2
                ;;
            --raw_data_dir)
                require_option_value "$1" "${2:-}"
                RAW_DATA_DIR="$2"
                shift 2
                ;;
            --results_root)
                require_option_value "$1" "${2:-}"
                RESULTS_ROOT="$2"
                shift 2
                ;;
            --solutions_root)
                require_option_value "$1" "${2:-}"
                SOLUTIONS_ROOT="$2"
                shift 2
                ;;
            --python)
                require_option_value "$1" "${2:-}"
                PYTHON_BIN="$2"
                shift 2
                ;;
            --epochs)
                require_option_value "$1" "${2:-}"
                STAGE1_EPOCHS="$2"
                shift 2
                ;;
            *)
                COMMAND_ARGS+=("$1")
                shift
                ;;
        esac
    done
}

resolve_command_solver() {
    local default_solver="$1"
    local alias_solver="${2:-}"

    if [ -n "${REQUESTED_SOLVER:-}" ] && [ -n "$alias_solver" ] && [ "$REQUESTED_SOLVER" != "$alias_solver" ]; then
        echo "Error: conflicting solver selection ('$REQUESTED_SOLVER' vs command alias '$alias_solver')" >&2
        exit 1
    fi

    local resolved_solver="${REQUESTED_SOLVER:-${alias_solver:-$default_solver}}"
    validate_solver_name "$resolved_solver"
    printf '%s\n' "$resolved_solver"
}

stage2_solver_entrypoint() {
    case "$1" in
        hybrid)
            printf '%s\n' "formulations/hybrid/stage2_solver.py"
            ;;
        cf)
            printf '%s\n' "formulations/cf/stage2_solver.py"
            ;;
        pief)
            printf '%s\n' "formulations/pief/stage2_solver.py"
            ;;
    esac
}

dfl_gnn_entrypoint() {
    case "$1" in
        hybrid)
            printf '%s\n' "formulations/hybrid/end2end_gnn.py"
            ;;
        cf)
            printf '%s\n' "formulations/cf/end2end_gnn.py"
            ;;
        pief)
            printf '%s\n' "formulations/pief/end2end_gnn.py"
            ;;
    esac
}

dfl_reg_entrypoint() {
    case "$1" in
        hybrid)
            printf '%s\n' "formulations/hybrid/end2end_reg.py"
            ;;
        cf)
            printf '%s\n' "formulations/cf/end2end_reg.py"
            ;;
        pief)
            printf '%s\n' "formulations/pief/end2end_reg.py"
            ;;
    esac
}

if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=1
    shift
fi

COMMAND="${1:-help}"
if [ $# -gt 0 ]; then
    shift
fi

REQUESTED_SOLVER=""
ALIAS_SOLVER=""

case "$COMMAND" in
    2stg-gnn-cf|2stg-reg-cf|2stg-lr-cf|dfl-gnn-cf|dfl-reg-cf|dfl-lr-cf)
        ALIAS_SOLVER="cf"
        COMMAND="${COMMAND%-cf}"
        ;;
    2stg-gnn-pief|2stg-reg-pief|2stg-lr-pief|dfl-gnn-pief|dfl-reg-pief|dfl-lr-pief)
        ALIAS_SOLVER="pief"
        COMMAND="${COMMAND%-pief}"
        ;;
    2stg-gnn-hybrid|2stg-reg-hybrid|2stg-lr-hybrid|dfl-gnn-hybrid|dfl-reg-hybrid|dfl-lr-hybrid)
        ALIAS_SOLVER="hybrid"
        COMMAND="${COMMAND%-hybrid}"
        ;;
    2stg-gnn-cf-piefchain)
        ALIAS_SOLVER="hybrid"
        COMMAND="2stg-gnn"
        ;;
    2stg-reg-cf-piefchain)
        ALIAS_SOLVER="hybrid"
        COMMAND="2stg-reg"
        ;;
    2stg-lr-cf-piefchain)
        ALIAS_SOLVER="hybrid"
        COMMAND="2stg-lr"
        ;;
    dfl-gnn-cf-piefchain)
        ALIAS_SOLVER="hybrid"
        COMMAND="dfl-gnn"
        ;;
    dfl-reg-cf-piefchain)
        ALIAS_SOLVER="hybrid"
        COMMAND="dfl-reg"
        ;;
    dfl-lr-cf-piefchain)
        ALIAS_SOLVER="hybrid"
        COMMAND="dfl-lr"
        ;;
esac

COMMAND_ARGS=()
parse_common_cli_args "$@"
set -- "${COMMAND_ARGS[@]}"

case "$COMMAND" in
    data-generate)
        run_python 0-data-generation.py "$@"
        ;;
    data-process)
        mapfile -d '' -t data_process_args < <(normalize_data_process_args "$@")
        run_python 1-data-processing.py "${data_process_args[@]}"
        ;;
    2stg-gnn)
        solver_name="$(resolve_command_solver "hybrid" "$ALIAS_SOLVER")"
        stage2_script="$(stage2_solver_entrypoint "$solver_name")"
        resolved_processed_dir="$(resolve_processed_data_dir "$PROCESSED_DATA_DIR")"
        run_python 2-stage1-training-GNN.py --data_dir "$resolved_processed_dir" --results_root "$RESULTS_ROOT"
        model_path="$(latest_checkpoint "2stg_Gnn_" "best_stage1_model_real.pth")"
        log "Using latest checkpoint: $model_path"
        run_python "$stage2_script" --model_path "$model_path" --data_dir "$resolved_processed_dir" --results_root "$RESULTS_ROOT" --solutions_root "$SOLUTIONS_ROOT"
        ;;
    2stg-reg)
        solver_name="$(resolve_command_solver "hybrid" "$ALIAS_SOLVER")"
        stage2_script="$(stage2_solver_entrypoint "$solver_name")"
        resolved_processed_dir="$(resolve_processed_data_dir "$PROCESSED_DATA_DIR")"
        mapfile -d '' -t stage1_args < <(stage1_epoch_args)
        run_python 2-stage1-training-Reg.py --data_dir "$resolved_processed_dir" --results_root "$RESULTS_ROOT" "${stage1_args[@]}"
        model_path="$(latest_checkpoint "2stg_Reg_" "best_stage1_model_real.pth")"
        log "Using latest checkpoint: $model_path"
        run_python "$stage2_script" --model_path "$model_path" --data_dir "$resolved_processed_dir" --results_root "$RESULTS_ROOT" --solutions_root "$SOLUTIONS_ROOT"
        ;;
    2stg-lr)
        solver_name="$(resolve_command_solver "hybrid" "$ALIAS_SOLVER")"
        stage2_script="$(stage2_solver_entrypoint "$solver_name")"
        resolved_processed_dir="$(resolve_processed_data_dir "$PROCESSED_DATA_DIR")"
        mapfile -d '' -t stage1_args < <(stage1_epoch_args)
        run_python 2-stage1-training-LR.py --data_dir "$resolved_processed_dir" --results_root "$RESULTS_ROOT" "${stage1_args[@]}"
        model_path="$(latest_checkpoint "2stg_LR_" "best_stage1_model_real.pth")"
        log "Using latest checkpoint: $model_path"
        run_python "$stage2_script" --model_path "$model_path" --data_dir "$resolved_processed_dir" --results_root "$RESULTS_ROOT" --solutions_root "$SOLUTIONS_ROOT"
        ;;
    dfl-gnn)
        solver_name="$(resolve_command_solver "hybrid" "$ALIAS_SOLVER")"
        dfl_script="$(dfl_gnn_entrypoint "$solver_name")"
        run_python "$dfl_script" --data_dir "$PROCESSED_DATA_DIR" --results_root "$RESULTS_ROOT" --solutions_root "$SOLUTIONS_ROOT" "$@"
        ;;
    dfl-reg)
        solver_name="$(resolve_command_solver "hybrid" "$ALIAS_SOLVER")"
        dfl_script="$(dfl_reg_entrypoint "$solver_name")"
        run_python "$dfl_script" --data_dir "$PROCESSED_DATA_DIR" --results_root "$RESULTS_ROOT" --solutions_root "$SOLUTIONS_ROOT" "$@"
        ;;
    dfl-lr)
        solver_name="$(resolve_command_solver "hybrid" "$ALIAS_SOLVER")"
        dfl_script="$(dfl_reg_entrypoint "$solver_name")"
        run_python "$dfl_script" --model_family lr --data_dir "$PROCESSED_DATA_DIR" --results_root "$RESULTS_ROOT" --solutions_root "$SOLUTIONS_ROOT" "$@"
        ;;
    oracle)
        solver_name="$(resolve_command_solver "hybrid" "$ALIAS_SOLVER")"
        stage2_script="$(stage2_solver_entrypoint "$solver_name")"
        oracle_args=(--gt_mode)
        if [ $# -gt 0 ] && [[ "$1" == *.pth ]]; then
            oracle_args+=(--model_path "$1")
            shift
        elif model_path="$(resolve_oracle_reference_model 2>/dev/null)"; then
            oracle_args+=(--model_path "$model_path")
            log "Oracle will reuse test split from: $model_path"
        fi
        oracle_args+=(--data_dir "$PROCESSED_DATA_DIR" --results_root "$RESULTS_ROOT" --solutions_root "$SOLUTIONS_ROOT")
        oracle_args+=("$@")
        run_python "$stage2_script" "${oracle_args[@]}"
        ;;
    evaluate)
        run_python 4-evaulation.py --sol_dir "$SOLUTIONS_ROOT" --data_dir "$PROCESSED_DATA_DIR" --results_root "$RESULTS_ROOT" "$@"
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
