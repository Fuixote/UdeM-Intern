#!/usr/bin/env python3
"""Analyze whether edge-level prediction errors are decision-critical."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import compare_decisions_per_graph as replay


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EDGE_OUTPUT = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "decision_analysis"
    / "results"
    / "edge_error_criticality.csv"
)
DEFAULT_SUMMARY_OUTPUT = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "decision_analysis"
    / "results"
    / "graph_level_edge_criticality_summary.csv"
)
EDGE_FIELDS = [
    "regime",
    "case_type",
    "subset_seed",
    "graph_id",
    "edge_id",
    "src",
    "dst",
    "w_true",
    "w_hat_2stage",
    "w_hat_spoplus",
    "abs_err_2stage",
    "abs_err_spoplus",
    "rank_err_2stage",
    "rank_err_spoplus",
    "in_opt",
    "in_2stage",
    "in_spoplus",
    "in_any_selected",
    "in_2stage_symdiff",
    "in_spoplus_symdiff",
    "utility",
    "recipient_cPRA",
]
SUMMARY_FIELDS = [
    "regime",
    "case_type",
    "subset_seed",
    "graph_id",
    "method_label",
    "method",
    "selection_metric",
    "num_edges",
    "num_top_error_edges",
    "mse_all_edges",
    "mse_edges_in_opt",
    "mse_edges_in_pred",
    "mse_edges_in_symdiff",
    "mse_edges_not_selected",
    "top10_error_edges_in_opt_rate",
    "top10_error_edges_in_pred_rate",
    "top10_error_edges_in_symdiff_rate",
]


def descending_error_ranks(errors) -> np.ndarray:
    values = np.asarray(errors, dtype=float)
    order = np.argsort(-values, kind="mergesort")
    ranks = np.empty(len(values), dtype=int)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def binary_mask(values) -> np.ndarray:
    return np.asarray(values, dtype=float) > 0.5


def edge_membership_flags(y_opt, y_2stage, y_spoplus) -> dict[str, np.ndarray]:
    opt = binary_mask(y_opt)
    two_stage = binary_mask(y_2stage)
    spoplus = binary_mask(y_spoplus)
    if opt.shape != two_stage.shape or opt.shape != spoplus.shape:
        raise ValueError(
            f"Shape mismatch: opt={opt.shape}, 2stage={two_stage.shape}, "
            f"spoplus={spoplus.shape}"
        )
    return {
        "in_opt": opt,
        "in_2stage": two_stage,
        "in_spoplus": spoplus,
        "in_any_selected": opt | two_stage | spoplus,
        "in_2stage_symdiff": opt ^ two_stage,
        "in_spoplus_symdiff": opt ^ spoplus,
    }


def safe_masked_mean(values, mask) -> float:
    values = np.asarray(values, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(values[mask]))


def edge_index_arrays(graph: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    edge_index = graph["edge_index"]
    if hasattr(edge_index, "detach"):
        edge_index = edge_index.detach().cpu().numpy()
    edge_index = np.asarray(edge_index, dtype=int)
    return edge_index[0], edge_index[1]


def graph_feature_array(graph: dict[str, object], name: str) -> np.ndarray:
    values = graph["features"][name]
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=float)


def method_flag_names(method_label: str) -> tuple[str, str]:
    if method_label == "2stage_val_mse":
        return "in_2stage", "in_2stage_symdiff"
    if method_label == "spoplus_val_spoplus_loss":
        return "in_spoplus", "in_spoplus_symdiff"
    raise ValueError(f"Unsupported method label for edge criticality: {method_label}")


def summarize_method_edges(
    abs_errors,
    flags: dict[str, np.ndarray],
    method_label: str,
    top_k: int = 10,
) -> dict[str, float | int]:
    abs_errors = np.asarray(abs_errors, dtype=float)
    squared_errors = abs_errors ** 2
    pred_flag, symdiff_flag = method_flag_names(method_label)
    top_n = min(int(top_k), len(abs_errors))
    ranks = descending_error_ranks(abs_errors)
    top_mask = ranks <= top_n
    not_selected = ~(flags["in_opt"] | flags[pred_flag])
    return {
        "num_edges": int(len(abs_errors)),
        "num_top_error_edges": int(top_n),
        "mse_all_edges": float(np.mean(squared_errors)) if len(abs_errors) else float("nan"),
        "mse_edges_in_opt": safe_masked_mean(squared_errors, flags["in_opt"]),
        "mse_edges_in_pred": safe_masked_mean(squared_errors, flags[pred_flag]),
        "mse_edges_in_symdiff": safe_masked_mean(squared_errors, flags[symdiff_flag]),
        "mse_edges_not_selected": safe_masked_mean(squared_errors, not_selected),
        "top10_error_edges_in_opt_rate": safe_masked_mean(flags["in_opt"], top_mask),
        "top10_error_edges_in_pred_rate": safe_masked_mean(flags[pred_flag], top_mask),
        "top10_error_edges_in_symdiff_rate": safe_masked_mean(
            flags[symdiff_flag], top_mask
        ),
    }


def write_csv(path: str | Path, rows: list[dict[str, object]], fieldnames) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze_edge_criticality(args) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    common, load_model_weight, _, _, _ = replay.ensure_step1c_imports()
    import gurobipy as gp

    selected_cases = replay.load_selected_cases(args.selected_seeds, args.regime)
    test_entries = replay.load_or_make_split_entries(
        split_path=args.split_path,
        dataset_dir=args.dataset_dir,
        split_seed=args.split_seed,
        train_pool_size=args.train_pool_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
    )
    if args.graph_limit is not None:
        test_entries = test_entries[: args.graph_limit]
    test_entries = replay.filter_split_entries_by_graphs(
        test_entries,
        set(args.graphs) if args.graphs else None,
    )
    graph_paths = [
        replay.resolve_graph_path(entry, args.dataset_dir) for entry in test_entries
    ]

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Seed", args.gurobi_seed)
    env.start()

    edge_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    try:
        step1a = common.load_step1a_module()

        for case in selected_cases:
            subset_seed = int(case["subset_seed"])
            run_dir = replay.resolve_run_dir(args.run_root, args.regime, subset_seed)
            models = replay.load_models(run_dir, load_model_weight)
            print(f"Analyzing seed={subset_seed} run_dir={run_dir}", flush=True)
            records = []
            try:
                print(f"  loading heldout graphs: n={len(graph_paths)}", flush=True)
                records = common.load_graph_records(graph_paths, env)
                for record in records:
                    by_label = {
                        replay.method_label(model): (
                            model,
                            replay.replay_model_on_record(step1a, model, record),
                        )
                        for model in models
                    }
                    two_stage = by_label["2stage_val_mse"][1]
                    spoplus = by_label["spoplus_val_spoplus_loss"][1]
                    y_opt = two_stage["y_opt"]
                    flags = edge_membership_flags(
                        y_opt,
                        two_stage["y_pred"],
                        spoplus["y_pred"],
                    )
                    w_true = np.asarray(record["w_true"], dtype=float)
                    abs_err_2stage = np.abs(two_stage["w_hat"] - w_true)
                    abs_err_spoplus = np.abs(spoplus["w_hat"] - w_true)
                    rank_2stage = descending_error_ranks(abs_err_2stage)
                    rank_spoplus = descending_error_ranks(abs_err_spoplus)
                    src, dst = edge_index_arrays(record["graph"])
                    utility = graph_feature_array(record["graph"], "utility")
                    recipient_cpra = graph_feature_array(record["graph"], "recipient_cPRA")

                    for label, abs_errors in [
                        ("2stage_val_mse", abs_err_2stage),
                        ("spoplus_val_spoplus_loss", abs_err_spoplus),
                    ]:
                        model, _ = by_label[label]
                        summary_rows.append(
                            {
                                "regime": args.regime,
                                "case_type": case["case_type"],
                                "subset_seed": subset_seed,
                                "graph_id": record["filename"],
                                "method_label": label,
                                "method": model["method"],
                                "selection_metric": model["selection_metric"],
                                **summarize_method_edges(
                                    abs_errors,
                                    flags,
                                    method_label=label,
                                    top_k=args.top_k,
                                ),
                            }
                        )

                    for edge_id in range(len(w_true)):
                        edge_rows.append(
                            {
                                "regime": args.regime,
                                "case_type": case["case_type"],
                                "subset_seed": subset_seed,
                                "graph_id": record["filename"],
                                "edge_id": edge_id,
                                "src": int(src[edge_id]),
                                "dst": int(dst[edge_id]),
                                "w_true": float(w_true[edge_id]),
                                "w_hat_2stage": float(two_stage["w_hat"][edge_id]),
                                "w_hat_spoplus": float(spoplus["w_hat"][edge_id]),
                                "abs_err_2stage": float(abs_err_2stage[edge_id]),
                                "abs_err_spoplus": float(abs_err_spoplus[edge_id]),
                                "rank_err_2stage": int(rank_2stage[edge_id]),
                                "rank_err_spoplus": int(rank_spoplus[edge_id]),
                                "in_opt": bool(flags["in_opt"][edge_id]),
                                "in_2stage": bool(flags["in_2stage"][edge_id]),
                                "in_spoplus": bool(flags["in_spoplus"][edge_id]),
                                "in_any_selected": bool(flags["in_any_selected"][edge_id]),
                                "in_2stage_symdiff": bool(
                                    flags["in_2stage_symdiff"][edge_id]
                                ),
                                "in_spoplus_symdiff": bool(
                                    flags["in_spoplus_symdiff"][edge_id]
                                ),
                                "utility": float(utility[edge_id]),
                                "recipient_cPRA": float(recipient_cpra[edge_id]),
                            }
                        )
            finally:
                common.dispose_graph_records(records)
    finally:
        env.dispose()
    return edge_rows, summary_rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Analyze edge-level prediction error criticality."
    )
    parser.add_argument("--selected-seeds", type=Path, default=replay.DEFAULT_SELECTED_SEEDS)
    parser.add_argument("--run-root", type=Path, default=replay.DEFAULT_RUN_ROOT)
    parser.add_argument("--regime", default="step2b_poly_d8")
    parser.add_argument("--dataset-dir", type=Path, default=replay.DEFAULT_DATASET_DIR)
    parser.add_argument("--split-path", type=Path, default=replay.DEFAULT_SPLIT_PATH)
    parser.add_argument("--edge-output", type=Path, default=DEFAULT_EDGE_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--train-pool-size", type=int, default=1200)
    parser.add_argument("--validation-size", type=int, default=400)
    parser.add_argument("--test-size", type=int, default=400)
    parser.add_argument("--gurobi-seed", type=int, default=42)
    parser.add_argument("--graph-limit", type=int)
    parser.add_argument("--graphs", nargs="+", help="Optional graph filenames to analyze.")
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    edge_rows, summary_rows = analyze_edge_criticality(args)
    write_csv(args.edge_output, edge_rows, EDGE_FIELDS)
    write_csv(args.summary_output, summary_rows, SUMMARY_FIELDS)
    print(f"Saved {len(edge_rows)} edge rows to {args.edge_output}")
    print(f"Saved {len(summary_rows)} summary rows to {args.summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
