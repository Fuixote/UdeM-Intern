#!/usr/bin/env python3
"""Compute best and second-best distinct KEP solutions induced by trained methods.

This script extends the existing decision-analysis replay path:

  For each selected subset seed, heldout graph, and method:
    1. Load trained theta.
    2. Compute predicted edge weights: w_hat = X @ theta.
    3. Solve KEP under w_hat to obtain the method's best predicted solution.
    4. Add a temporary no-good cut to exclude the current MIP assignment.
    5. Re-solve until a second distinct edge-selection solution is found.
    6. Evaluate both predicted solutions under true weights w_true against the oracle y_opt.

The no-good cut is added directly to the cached Gurobi model and removed afterward, so
this script does not require modifying Step1a / backend solver source files.

Default output:
  surrogate_experiment_results/decision_analysis/results/second_best_gap_comparison.csv
  surrogate_experiment_results/decision_analysis/results/second_best_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent

# Make sibling script imports robust when this file is run from repo root.
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compare_decisions_per_graph import (  # noqa: E402
    DEFAULT_DATASET_DIR,
    DEFAULT_RUN_ROOT,
    DEFAULT_SELECTED_SEEDS,
    DEFAULT_SPLIT_PATH,
    ensure_step1c_imports,
    filter_split_entries_by_graphs,
    load_models,
    load_or_make_split_entries,
    load_selected_cases,
    method_label,
    normalized_gap,
    resolve_graph_path,
    resolve_run_dir,
)


DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "decision_analysis"
    / "results"
    / "second_best_gap_comparison.csv"
)
DEFAULT_SUMMARY_OUTPUT = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "decision_analysis"
    / "results"
    / "second_best_summary.csv"
)

DEFAULT_METHOD_LABELS = ("2stage_val_mse", "spoplus_val_spoplus_loss")


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
    "solution_rank",
    "cut_attempt",
    "solver_status",
    "solver_status_name",
    "solver_obj_predicted",
    "predicted_obj",
    "true_obj",
    "oracle_obj",
    "gap_to_oracle",
    "normalized_gap_to_oracle",
    "predicted_margin_from_best",
    "true_obj_diff_from_rank1",
    "num_edges",
    "edge_count",
    "same_solution_as_oracle",
    "edge_jaccard_with_oracle",
    "edge_hamming_with_oracle",
    "edge_overlap_count_with_oracle",
    "same_solution_as_rank1",
    "edge_jaccard_with_rank1",
    "edge_hamming_with_rank1",
    "edge_overlap_count_with_rank1",
    "solution_edge_signature",
    "model_path",
]

SUMMARY_FIELDS = [
    "regime",
    "method_label",
    "solution_rank",
    "row_count",
    "mean_gap_to_oracle",
    "median_gap_to_oracle",
    "mean_normalized_gap_to_oracle",
    "median_normalized_gap_to_oracle",
    "exact_oracle_rate",
    "near_1pct_rate",
    "near_5pct_rate",
    "mean_predicted_margin_from_best",
    "median_predicted_margin_from_best",
    "mean_edge_jaccard_with_oracle",
    "median_edge_jaccard_with_oracle",
]


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def finite_mean(values: list[float]) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return float("nan")
    return float(sum(clean) / len(clean))


def finite_median(values: list[float]) -> float:
    clean = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not clean:
        return float("nan")
    n = len(clean)
    mid = n // 2
    if n % 2:
        return float(clean[mid])
    return float(0.5 * (clean[mid - 1] + clean[mid]))


def status_name(status: int) -> str:
    try:
        from gurobipy import GRB
    except Exception:
        return str(status)

    mapping = {
        GRB.LOADED: "LOADED",
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.CUTOFF: "CUTOFF",
        GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
        GRB.NODE_LIMIT: "NODE_LIMIT",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INPROGRESS: "INPROGRESS",
        GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
    }
    return mapping.get(status, str(status))


def edge_signature(y: np.ndarray) -> str:
    selected = np.flatnonzero(np.asarray(y, dtype=float) > 0.5)
    return "|".join(str(int(idx)) for idx in selected)


def edge_overlap_metrics(y: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    left = np.asarray(y, dtype=float) > 0.5
    right = np.asarray(reference, dtype=float) > 0.5
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch: y={left.shape}, reference={right.shape}")

    intersection = int(np.sum(left & right))
    union = int(np.sum(left | right))
    return {
        "same_solution": bool(np.array_equal(left, right)),
        "edge_jaccard": 1.0 if union == 0 else float(intersection / union),
        "edge_hamming": float(np.mean(left != right)) if len(left) else 0.0,
        "edge_overlap_count": intersection,
    }


def iter_binary_vars(solver) -> list[tuple[tuple[Any, ...], Any]]:
    """Return all binary decision variables in the cached hybrid KEP model.

    Keys are script-level identifiers, not Gurobi names:
      ("cycle", cycle_idx)
      ("chain", edge_idx, position)
    """
    variables: list[tuple[tuple[Any, ...], Any]] = []

    for idx in range(len(solver.cycle_candidates)):
        variables.append((("cycle", int(idx)), solver.cycle_vars[idx]))

    for edge_idx, position in solver.valid_chain_keys:
        variables.append(
            (("chain", int(edge_idx), int(position)), solver.chain_vars[edge_idx, position])
        )

    return variables


def selected_mip_keys_from_current_solution(solver) -> set[tuple[Any, ...]]:
    selected: set[tuple[Any, ...]] = set()
    for key, var in iter_binary_vars(solver):
        if float(var.X) > 0.5:
            selected.add(key)
    return selected


def add_no_good_cut(solver, selected_keys: set[tuple[Any, ...]], name: str):
    """Exclude exactly the current MIP binary assignment.

    Cut:
        sum_{j: x_j=1} (1 - x_j) + sum_{j: x_j=0} x_j >= 1

    This forces the next solution to differ in at least one cycle/chain-position
    variable. We still check edge-selection distinctness outside this function,
    because two different internal chain assignments can theoretically map to
    the same selected edge set.
    """
    import gurobipy as gp

    binary_vars = iter_binary_vars(solver)
    if not binary_vars:
        return None

    expr_terms = []
    for key, var in binary_vars:
        if key in selected_keys:
            expr_terms.append(1.0 - var)
        else:
            expr_terms.append(var)

    cut = solver.model.addConstr(gp.quicksum(expr_terms) >= 1.0, name=name)
    solver.model.update()
    return cut


def set_cached_solver_objective(solver, weights: np.ndarray) -> None:
    weights = np.asarray(weights, dtype=float)

    for idx, candidate in enumerate(solver.cycle_candidates):
        solver.cycle_vars[idx].Obj = float(
            sum(weights[int(edge_idx)] for edge_idx in candidate["edges"])
        )

    for edge_idx, position in solver.valid_chain_keys:
        solver.chain_vars[edge_idx, position].Obj = float(weights[int(edge_idx)])

    solver.model.update()


def decode_current_edge_selection(solver) -> np.ndarray:
    selected_edges: list[int] = []

    selected_cycle_indices = [
        idx
        for idx in range(len(solver.cycle_candidates))
        if float(solver.cycle_vars[idx].X) > 0.5
    ]
    selected_chain_keys = [
        (int(edge_idx), int(position))
        for edge_idx, position in solver.valid_chain_keys
        if float(solver.chain_vars[edge_idx, position].X) > 0.5
    ]

    for idx in selected_cycle_indices:
        selected_edges.extend(int(edge_idx) for edge_idx in solver.cycle_candidates[idx]["edges"])

    selected_edges.extend(int(edge_idx) for edge_idx, _ in selected_chain_keys)

    y = np.zeros(int(solver.num_edges), dtype=float)
    for edge_idx in selected_edges:
        y[edge_idx] = 1.0
    return y


def solve_cached_model_with_current_cuts(
    solver,
    weights: np.ndarray,
    reset_before_solve: bool = True,
) -> dict[str, Any]:
    """Solve the cached KEP model under current objective and current cuts."""
    from gurobipy import GRB

    set_cached_solver_objective(solver, weights)

    if reset_before_solve:
        solver.model.reset()

    solver.model.optimize()

    status = int(solver.model.status)
    sol_count = int(solver.model.SolCount)

    if status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SOLUTION_LIMIT, GRB.SUBOPTIMAL) and sol_count > 0:
        y = decode_current_edge_selection(solver)
        mip_keys = selected_mip_keys_from_current_solution(solver)
        solver_obj = float(solver.model.ObjVal)
        return {
            "found": True,
            "status": status,
            "status_name": status_name(status),
            "solver_obj": solver_obj,
            "y": y,
            "mip_keys": mip_keys,
        }

    return {
        "found": False,
        "status": status,
        "status_name": status_name(status),
        "solver_obj": float("nan"),
        "y": None,
        "mip_keys": set(),
    }


def solve_top_distinct_edge_solutions(
    solver,
    weights: np.ndarray,
    max_solutions: int = 2,
    max_cut_attempts: int = 20,
    reset_before_solve: bool = True,
) -> list[dict[str, Any]]:
    """Return top distinct edge-selection solutions under predicted weights.

    The no-good cut excludes MIP assignments. Since the same edge set may sometimes
    arise from a different internal assignment, this function keeps adding cuts until
    it finds a new edge signature or exhausts max_cut_attempts.
    """
    cuts = []
    solutions: list[dict[str, Any]] = []
    seen_edge_signatures: set[str] = set()

    try:
        for attempt in range(1, max_cut_attempts + 1):
            if len(solutions) >= max_solutions:
                break

            result = solve_cached_model_with_current_cuts(
                solver,
                weights=weights,
                reset_before_solve=reset_before_solve,
            )
            if not result["found"]:
                break

            y = np.asarray(result["y"], dtype=float)
            signature = edge_signature(y)

            if signature not in seen_edge_signatures:
                result = dict(result)
                result["solution_rank"] = len(solutions) + 1
                result["cut_attempt"] = attempt
                result["solution_edge_signature"] = signature
                solutions.append(result)
                seen_edge_signatures.add(signature)

            cut = add_no_good_cut(
                solver,
                selected_keys=result["mip_keys"],
                name=f"tmp_second_best_nogood_attempt_{attempt}",
            )
            if cut is None:
                break
            cuts.append(cut)

    finally:
        if cuts:
            solver.model.remove(cuts)
            solver.model.update()
        if reset_before_solve:
            solver.model.reset()

    return solutions


def rows_for_model_record(
    *,
    args,
    case: dict[str, str],
    model: dict[str, Any],
    label: str,
    record: dict[str, Any],
) -> list[dict[str, Any]]:
    theta = np.asarray(model["theta"], dtype=float)
    w_hat = np.asarray(record["X"], dtype=float) @ theta
    w_true = np.asarray(record["w_true"], dtype=float)
    y_oracle = np.asarray(record["y_optimal"], dtype=float)

    solver = record["graph"]["cached_solver"]

    solutions = solve_top_distinct_edge_solutions(
        solver=solver,
        weights=w_hat,
        max_solutions=args.max_solutions,
        max_cut_attempts=args.max_cut_attempts,
        reset_before_solve=not args.no_reset_before_solve,
    )

    if not solutions:
        return []

    y_rank1 = np.asarray(solutions[0]["y"], dtype=float)
    pred_obj_rank1 = float(np.dot(w_hat, y_rank1))
    true_obj_rank1 = float(np.dot(w_true, y_rank1))
    oracle_obj = float(np.dot(w_true, y_oracle))

    rows: list[dict[str, Any]] = []
    for solution in solutions:
        y = np.asarray(solution["y"], dtype=float)

        predicted_obj = float(np.dot(w_hat, y))
        true_obj = float(np.dot(w_true, y))
        gap = float(oracle_obj - true_obj)
        norm_gap = normalized_gap(gap, oracle_obj)

        oracle_overlap = edge_overlap_metrics(y, y_oracle)
        rank1_overlap = edge_overlap_metrics(y, y_rank1)

        rows.append(
            {
                "regime": args.regime,
                "case_type": case.get("case_type", ""),
                "subset_seed": int(case["subset_seed"]),
                "graph_id": record["filename"],
                "method_label": label,
                "method": model["method"],
                "selection_metric": model["selection_metric"],
                "selected_epoch": int(model["selected_epoch"]),
                "theta_1": float(theta[0]),
                "theta_2": float(theta[1]),
                "solution_rank": int(solution["solution_rank"]),
                "cut_attempt": int(solution["cut_attempt"]),
                "solver_status": int(solution["status"]),
                "solver_status_name": solution["status_name"],
                "solver_obj_predicted": float(solution["solver_obj"]),
                "predicted_obj": predicted_obj,
                "true_obj": true_obj,
                "oracle_obj": oracle_obj,
                "gap_to_oracle": gap,
                "normalized_gap_to_oracle": norm_gap,
                "predicted_margin_from_best": float(pred_obj_rank1 - predicted_obj),
                "true_obj_diff_from_rank1": float(true_obj_rank1 - true_obj),
                "num_edges": int(len(y)),
                "edge_count": int(np.sum(y > 0.5)),
                "same_solution_as_oracle": bool(oracle_overlap["same_solution"]),
                "edge_jaccard_with_oracle": float(oracle_overlap["edge_jaccard"]),
                "edge_hamming_with_oracle": float(oracle_overlap["edge_hamming"]),
                "edge_overlap_count_with_oracle": int(oracle_overlap["edge_overlap_count"]),
                "same_solution_as_rank1": bool(rank1_overlap["same_solution"]),
                "edge_jaccard_with_rank1": float(rank1_overlap["edge_jaccard"]),
                "edge_hamming_with_rank1": float(rank1_overlap["edge_hamming"]),
                "edge_overlap_count_with_rank1": int(rank1_overlap["edge_overlap_count"]),
                "solution_edge_signature": solution["solution_edge_signature"],
                "model_path": str(model["path"]),
            }
        )

    return rows


def compute_second_best_rows(args) -> list[dict[str, Any]]:
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

    output_rows: list[dict[str, Any]] = []

    try:
        for case_idx, case in enumerate(selected_cases, start=1):
            subset_seed = int(case["subset_seed"])
            run_dir = resolve_run_dir(args.run_root, args.regime, subset_seed)
            models = load_models(run_dir, load_model_weight)

            print(
                f"[{case_idx}/{len(selected_cases)}] seed={subset_seed} "
                f"case_type={case.get('case_type', '')} graphs={len(graph_paths)}",
                flush=True,
            )

            records = []
            try:
                records = common.load_graph_records(graph_paths, env)

                for model in models:
                    label = method_label(model)
                    if label not in set(args.method_labels):
                        continue

                    print(f"  method={label}", flush=True)

                    for graph_idx, record in enumerate(records, start=1):
                        if graph_idx % args.progress_every == 0 or graph_idx == len(records):
                            print(
                                f"    graph {graph_idx}/{len(records)} {record['filename']}",
                                flush=True,
                            )

                        output_rows.extend(
                            rows_for_model_record(
                                args=args,
                                case=case,
                                model=model,
                                label=label,
                                record=record,
                            )
                        )

            finally:
                common.dispose_graph_records(records)

    finally:
        env.dispose()

    return output_rows


def summarize_second_best_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        key = (
            str(row["regime"]),
            str(row["method_label"]),
            int(row["solution_rank"]),
        )
        grouped[key].append(row)

    summary_rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        regime, method, rank = key
        group = grouped[key]

        gaps = [float(row["gap_to_oracle"]) for row in group]
        norm_gaps = [float(row["normalized_gap_to_oracle"]) for row in group]
        margins = [float(row["predicted_margin_from_best"]) for row in group]
        jaccards = [float(row["edge_jaccard_with_oracle"]) for row in group]

        summary_rows.append(
            {
                "regime": regime,
                "method_label": method,
                "solution_rank": rank,
                "row_count": len(group),
                "mean_gap_to_oracle": finite_mean(gaps),
                "median_gap_to_oracle": finite_median(gaps),
                "mean_normalized_gap_to_oracle": finite_mean(norm_gaps),
                "median_normalized_gap_to_oracle": finite_median(norm_gaps),
                "exact_oracle_rate": finite_mean(
                    [1.0 if str(row["same_solution_as_oracle"]) == "True" else 0.0 for row in group]
                ),
                "near_1pct_rate": finite_mean([1.0 if value < 0.01 else 0.0 for value in norm_gaps]),
                "near_5pct_rate": finite_mean([1.0 if value < 0.05 else 0.0 for value in norm_gaps]),
                "mean_predicted_margin_from_best": finite_mean(margins),
                "median_predicted_margin_from_best": finite_median(margins),
                "mean_edge_jaccard_with_oracle": finite_mean(jaccards),
                "median_edge_jaccard_with_oracle": finite_median(jaccards),
            }
        )

    return summary_rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compute best and second-best distinct KEP solutions for 2stage/SPO+."
    )

    parser.add_argument("--selected-seeds", type=Path, default=DEFAULT_SELECTED_SEEDS)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--regime", default="step2b_poly_d8")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)

    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--train-pool-size", type=int, default=1200)
    parser.add_argument("--validation-size", type=int, default=400)
    parser.add_argument("--test-size", type=int, default=400)
    parser.add_argument("--gurobi-seed", type=int, default=42)

    parser.add_argument(
        "--method-labels",
        nargs="+",
        default=list(DEFAULT_METHOD_LABELS),
        help=(
            "Method labels to include. Defaults to "
            "2stage_val_mse spoplus_val_spoplus_loss."
        ),
    )
    parser.add_argument("--max-solutions", type=int, default=2)
    parser.add_argument(
        "--max-cut-attempts",
        type=int,
        default=20,
        help=(
            "Maximum MIP no-good-cut attempts per graph/method. "
            "Increase if rank-2 edge solution is not found due to duplicate edge sets."
        ),
    )
    parser.add_argument(
        "--no-reset-before-solve",
        action="store_true",
        help="Do not call Gurobi model.reset() before each solve.",
    )

    parser.add_argument("--graph-limit", type=int)
    parser.add_argument(
        "--graphs",
        nargs="+",
        help="Optional graph filenames to evaluate, e.g. G-696.json G-392.json.",
    )
    parser.add_argument("--progress-every", type=int, default=50)

    args = parser.parse_args(argv)

    if args.max_solutions < 1:
        raise ValueError("--max-solutions must be >= 1")
    if args.max_cut_attempts < args.max_solutions:
        raise ValueError("--max-cut-attempts should be >= --max-solutions")

    return args


def main(argv=None) -> int:
    args = parse_args(argv)

    rows = compute_second_best_rows(args)
    write_csv(args.output, rows, CSV_FIELDS)

    summary_rows = summarize_second_best_rows(rows)
    write_csv(args.summary_output, summary_rows, SUMMARY_FIELDS)

    print(f"Saved {len(rows)} solution rows to {args.output}")
    print(f"Saved {len(summary_rows)} summary rows to {args.summary_output}")

    if rows:
        by_rank = defaultdict(int)
        for row in rows:
            by_rank[int(row["solution_rank"])] += 1
        print("Rows by solution_rank:")
        for rank in sorted(by_rank):
            print(f"  rank {rank}: {by_rank[rank]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())