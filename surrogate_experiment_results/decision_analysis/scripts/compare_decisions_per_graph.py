#!/usr/bin/env python3
"""Replay KEP decisions for selected Step2 resampling seeds."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
STEP1C_DIR = PROJECT_ROOT / "surrogate_experiment_results" / "Step1c"
DEFAULT_SELECTED_SEEDS = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "decision_analysis"
    / "results"
    / "selected_case_seeds.csv"
)
DEFAULT_RUN_ROOT = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "Step2_resampling"
    / "phase1_runs"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "decision_analysis"
    / "results"
    / "per_graph_decision_comparison.csv"
)
DEFAULT_DATASET_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "step2b_poly_d8_main2000_seed20260523"
)
DEFAULT_SPLIT_PATH = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "Step2_resampling"
    / "splits"
    / "step2b_poly_d8"
    / "master_split_seed=42.json"
)
WEIGHT_FILENAMES = (
    "2stage_best_by_validation_mse_loss.npz",
    "spoplus_best_by_validation_decision_gap.npz",
    "spoplus_best_by_validation_spoplus_loss.npz",
)
OUTPUT_METHOD_LABELS = {"2stage_val_mse", "spoplus_val_spoplus_loss"}
DEFAULT_GAP_TOLERANCE = 1e-4
DEFAULT_NORMALIZED_GAP_TOLERANCE = 1e-6
CSV_FIELDS = [
    "regime",
    "case_type",
    "subset_seed",
    "graph_id",
    "method_label",
    "method",
    "selection_metric",
    "selected_epoch",
    "theta_1",
    "theta_2",
    "optimal_obj",
    "achieved_obj",
    "decision_gap",
    "normalized_gap",
    "num_edges",
    "num_edges_opt",
    "num_edges_pred",
    "same_solution_as_opt",
    "edge_jaccard_with_opt",
    "edge_hamming_with_opt",
    "edge_overlap_count",
    "existing_gap",
    "existing_normalized_gap",
    "gap_abs_diff",
    "normalized_gap_abs_diff",
    "model_path",
]


def normalized_gap(gap: float, optimal_obj: float, epsilon: float = 1e-9) -> float:
    return float(gap) / (abs(float(optimal_obj)) + float(epsilon))


def metric_diffs_within_tolerance(
    gap_diff: float,
    normalized_gap_diff: float,
    gap_tolerance: float = DEFAULT_GAP_TOLERANCE,
    normalized_gap_tolerance: float = DEFAULT_NORMALIZED_GAP_TOLERANCE,
) -> bool:
    return (
        float(gap_diff) <= float(gap_tolerance)
        and float(normalized_gap_diff) <= float(normalized_gap_tolerance)
    )


def solution_overlap_metrics(y_pred, y_opt) -> dict[str, float | int | bool]:
    pred = np.asarray(y_pred, dtype=float) > 0.5
    opt = np.asarray(y_opt, dtype=float) > 0.5
    if pred.shape != opt.shape:
        raise ValueError(f"Shape mismatch: y_pred={pred.shape}, y_opt={opt.shape}")

    intersection = int(np.sum(pred & opt))
    union = int(np.sum(pred | opt))
    return {
        "num_edges_opt": int(np.sum(opt)),
        "num_edges_pred": int(np.sum(pred)),
        "same_solution_as_opt": bool(np.array_equal(pred, opt)),
        "edge_jaccard_with_opt": 1.0 if union == 0 else float(intersection / union),
        "edge_hamming_with_opt": float(np.mean(pred != opt)) if len(pred) else 0.0,
        "edge_overlap_count": intersection,
    }


def resolve_run_dir(run_root: str | Path, regime: str, subset_seed: int) -> Path:
    return Path(run_root) / regime / f"subset_seed={int(subset_seed)}"


def ensure_step1c_imports():
    if str(STEP1C_DIR) not in sys.path:
        sys.path.insert(0, str(STEP1C_DIR))
    import step1c_common as common
    from evaluate_models import load_model_weight
    from split_dataset import list_graph_files, make_master_split, read_json

    return common, load_model_weight, list_graph_files, make_master_split, read_json


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_selected_cases(path: str | Path, regime: str) -> list[dict[str, str]]:
    rows = [row for row in read_csv_rows(path) if row["regime"] == regime]
    if not rows:
        raise ValueError(f"No selected case seeds found for regime={regime} in {path}")
    return rows


def load_existing_metrics(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing existing per-graph metrics: {path}")
    rows = read_csv_rows(path)
    return {
        (row["method"], row["selection_metric"], row["graph"]): row
        for row in rows
    }


def method_label(model: dict[str, object]) -> str:
    if model["method"] == "2stage" and model["selection_metric"] == "validation_mse_loss":
        return "2stage_val_mse"
    if (
        model["method"] == "spoplus"
        and model["selection_metric"] == "validation_decision_gap"
    ):
        return "spoplus_val_decision_gap"
    if (
        model["method"] == "spoplus"
        and model["selection_metric"] == "validation_spoplus_loss"
    ):
        return "spoplus_val_spoplus_loss"
    return f"{model['method']}_{model['selection_metric']}"


def load_models(run_dir: Path, load_model_weight) -> list[dict[str, object]]:
    weights_dir = run_dir / "model_weights"
    weight_paths = [weights_dir / filename for filename in WEIGHT_FILENAMES]
    missing = [path for path in weight_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model weights:\n" + "\n".join(str(path) for path in missing)
        )
    return [load_model_weight(path) for path in weight_paths]


def load_or_make_split_entries(
    split_path: Path,
    dataset_dir: Path,
    split_seed: int,
    train_pool_size: int,
    validation_size: int,
    test_size: int,
):
    _, _, list_graph_files, make_master_split, read_json = ensure_step1c_imports()
    if split_path.exists():
        split = read_json(split_path)
    else:
        split = make_master_split(
            list_graph_files(dataset_dir),
            train_pool_size=train_pool_size,
            val_size=validation_size,
            test_size=test_size,
            seed=split_seed,
        )
    return split["test"]


def resolve_graph_path(entry: dict[str, object], dataset_dir: Path) -> Path:
    path = Path(str(entry["path"]))
    if path.exists():
        return path
    fallback = dataset_dir / path.name
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Could not resolve graph path from split entry: {entry}")


def filter_split_entries_by_graphs(
    entries: list[dict[str, object]], graph_names: set[str] | None
) -> list[dict[str, object]]:
    if not graph_names:
        return entries
    wanted = {str(name) for name in graph_names}
    filtered = [
        entry for entry in entries if Path(str(entry["path"])).name in wanted
    ]
    missing = sorted(wanted - {Path(str(entry["path"])).name for entry in filtered})
    if missing:
        raise ValueError(f"Requested graph names were not in split: {missing}")
    return filtered


def write_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def replay_model_on_record(step1a, model: dict[str, object], record: dict[str, object]):
    theta = np.asarray(model["theta"], dtype=float)
    w_hat = np.asarray(record["X"], dtype=float) @ theta
    y_pred = np.asarray(step1a.solve_once(w_hat, record["graph"], None), dtype=float)
    y_opt = np.asarray(record["y_optimal"], dtype=float)
    w_true = np.asarray(record["w_true"], dtype=float)
    optimal_obj = float(np.dot(w_true, y_opt))
    achieved_obj = float(np.dot(w_true, y_pred))
    gap = optimal_obj - achieved_obj
    return {
        "w_hat": w_hat,
        "y_pred": y_pred,
        "y_opt": y_opt,
        "optimal_obj": optimal_obj,
        "achieved_obj": achieved_obj,
        "decision_gap": gap,
        "normalized_gap": normalized_gap(gap, optimal_obj),
    }


def replay_decisions(args) -> list[dict[str, object]]:
    common, load_model_weight, _, _, _ = ensure_step1c_imports()
    import gurobipy as gp

    selected_cases = load_selected_cases(args.selected_seeds, args.regime)
    test_entries = load_or_make_split_entries(
        split_path=args.split_path,
        dataset_dir=args.dataset_dir,
        split_seed=args.split_seed,
        train_pool_size=args.train_pool_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
    )
    if args.graph_limit is not None:
        test_entries = test_entries[: args.graph_limit]
    test_entries = filter_split_entries_by_graphs(
        test_entries,
        set(args.graphs) if args.graphs else None,
    )
    graph_paths = [resolve_graph_path(entry, args.dataset_dir) for entry in test_entries]

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Seed", args.gurobi_seed)
    env.start()

    rows: list[dict[str, object]] = []
    try:
        step1a = common.load_step1a_module()

        for case in selected_cases:
            subset_seed = int(case["subset_seed"])
            run_dir = resolve_run_dir(args.run_root, args.regime, subset_seed)
            existing = load_existing_metrics(run_dir / "metrics" / "test_per_graph.csv")
            models = load_models(run_dir, load_model_weight)
            print(f"Replaying seed={subset_seed} run_dir={run_dir}", flush=True)
            records = []
            try:
                print(f"  loading heldout graphs: n={len(graph_paths)}", flush=True)
                records = common.load_graph_records(graph_paths, env)
                for model in models:
                    label = method_label(model)
                    for record in records:
                        replay = replay_model_on_record(step1a, model, record)
                        if label not in OUTPUT_METHOD_LABELS:
                            continue
                        overlap = solution_overlap_metrics(replay["y_pred"], replay["y_opt"])
                        key = (
                            str(model["method"]),
                            str(model["selection_metric"]),
                            str(record["filename"]),
                        )
                        existing_row = existing.get(key)
                        if existing_row is None:
                            raise KeyError(f"Missing existing metric row: {key}")
                        gap_diff = abs(
                            float(replay["decision_gap"]) - float(existing_row["gap"])
                        )
                        norm_diff = abs(
                            float(replay["normalized_gap"])
                            - float(existing_row["normalized_gap"])
                        )
                        if args.require_existing_match and not metric_diffs_within_tolerance(
                            gap_diff=gap_diff,
                            normalized_gap_diff=norm_diff,
                            gap_tolerance=args.gap_tolerance,
                            normalized_gap_tolerance=args.normalized_gap_tolerance,
                        ):
                            raise AssertionError(
                                "Replay metric mismatch "
                                f"seed={subset_seed} graph={record['filename']} "
                                f"method={label} gap_diff={gap_diff} norm_diff={norm_diff}"
                            )
                        theta = np.asarray(model["theta"], dtype=float)
                        rows.append(
                            {
                                "regime": args.regime,
                                "case_type": case["case_type"],
                                "subset_seed": subset_seed,
                                "graph_id": record["filename"],
                                "method_label": label,
                                "method": model["method"],
                                "selection_metric": model["selection_metric"],
                                "selected_epoch": int(model["selected_epoch"]),
                                "theta_1": float(theta[0]),
                                "theta_2": float(theta[1]),
                                "optimal_obj": replay["optimal_obj"],
                                "achieved_obj": replay["achieved_obj"],
                                "decision_gap": replay["decision_gap"],
                                "normalized_gap": replay["normalized_gap"],
                                "num_edges": len(replay["y_opt"]),
                                **overlap,
                                "existing_gap": float(existing_row["gap"]),
                                "existing_normalized_gap": float(
                                    existing_row["normalized_gap"]
                                ),
                                "gap_abs_diff": gap_diff,
                                "normalized_gap_abs_diff": norm_diff,
                                "model_path": model["path"],
                            }
                        )
            finally:
                common.dispose_graph_records(records)
    finally:
        env.dispose()
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Replay per-graph KEP decisions for selected case seeds."
    )
    parser.add_argument("--selected-seeds", type=Path, default=DEFAULT_SELECTED_SEEDS)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--regime", default="step2b_poly_d8")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--train-pool-size", type=int, default=1200)
    parser.add_argument("--validation-size", type=int, default=400)
    parser.add_argument("--test-size", type=int, default=400)
    parser.add_argument("--gurobi-seed", type=int, default=42)
    parser.add_argument("--graph-limit", type=int)
    parser.add_argument("--graphs", nargs="+", help="Optional graph filenames to replay.")
    parser.add_argument("--gap-tolerance", type=float, default=DEFAULT_GAP_TOLERANCE)
    parser.add_argument(
        "--normalized-gap-tolerance",
        type=float,
        default=DEFAULT_NORMALIZED_GAP_TOLERANCE,
    )
    parser.add_argument(
        "--no-require-existing-match",
        action="store_false",
        dest="require_existing_match",
        help="Write replay rows even if they differ from existing test_per_graph.csv.",
    )
    parser.set_defaults(require_existing_match=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    rows = replay_decisions(args)
    write_csv(args.output, rows)
    max_gap = max((float(row["gap_abs_diff"]) for row in rows), default=0.0)
    max_norm = max((float(row["normalized_gap_abs_diff"]) for row in rows), default=0.0)
    print(f"Saved {len(rows)} rows to {args.output}")
    print(f"Max replay diff: gap={max_gap:.6g} normalized_gap={max_norm:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
