#!/usr/bin/env python3
"""Enumerate true-label top-M KEP solutions for selected Step2c graphs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compare_decisions_per_graph import (  # noqa: E402
    DEFAULT_SPLIT_PATH,
    ensure_step1c_imports,
    filter_split_entries_by_graphs,
    load_or_make_split_entries,
    normalized_gap,
    resolve_graph_path,
)
from compute_second_best_solutions import (  # noqa: E402
    edge_overlap_metrics,
    solve_top_distinct_edge_solutions,
    write_csv,
)


DEFAULT_REGIME = "step2c_poly_d8_mult_eps050"
DEFAULT_DATASET_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "step2c_poly_d8_mult_eps050_main2000_seed20260523"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "decision_analysis"
    / "results"
    / "step2c_mechanism_dissection"
)
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "step2c_selected_graphs_true_top50_oracle_landscape.csv"
DEFAULT_SUMMARY_OUTPUT = (
    DEFAULT_OUTPUT_DIR / "step2c_selected_graphs_true_top50_oracle_landscape_summary.csv"
)
DEFAULT_GRAPHS = (
    "G-392.json",
    "G-1285.json",
    "G-1560.json",
    "G-1169.json",
    "G-1449.json",
    "G-1657.json",
    "G-191.json",
    "G-142.json",
    "G-946.json",
    "G-14.json",
    "G-163.json",
    "G-552.json",
    "G-1110.json",
    "G-178.json",
    "G-1206.json",
    "G-1308.json",
)


CSV_FIELDS = [
    "regime",
    "max_cycle",
    "max_chain",
    "graph_id",
    "solution_rank",
    "cut_attempt",
    "solver_status",
    "solver_status_name",
    "solver_obj_true",
    "true_obj",
    "oracle_obj",
    "gap_to_oracle",
    "normalized_gap_to_oracle",
    "num_vertices",
    "num_edges",
    "density",
    "in_degree_mean",
    "out_degree_mean",
    "in_degree_gini",
    "out_degree_gini",
    "reciprocal_edge_count",
    "num_2cycles",
    "num_3cycles",
    "largest_scc_fraction",
    "number_of_sccs",
    "edge_count",
    "num_cycle_candidates",
    "num_2cycle_candidates",
    "num_3cycle_candidates",
    "num_chain_candidates",
    "same_solution_as_oracle",
    "edge_jaccard_with_oracle",
    "edge_hamming_with_oracle",
    "edge_overlap_count_with_oracle",
    "solution_edge_signature",
]

SUMMARY_FIELDS = [
    "regime",
    "max_cycle",
    "max_chain",
    "graph_id",
    "requested_top_m",
    "observed_top_m",
    "num_vertices",
    "num_edges",
    "density",
    "in_degree_mean",
    "out_degree_mean",
    "in_degree_gini",
    "out_degree_gini",
    "reciprocal_edge_count",
    "num_2cycles",
    "num_3cycles",
    "largest_scc_fraction",
    "number_of_sccs",
    "num_cycle_candidates",
    "num_2cycle_candidates",
    "num_3cycle_candidates",
    "num_chain_candidates",
    "oracle_objective",
    "oracle_second_best_gap_pct",
    "num_observed_solutions_within_1pct",
    "num_observed_solutions_within_5pct",
    "num_observed_solutions_within_10pct",
    "near_oracle_jaccard_mean",
    "near_oracle_jaccard_min",
]


def finite_mean(values: list[float]) -> float:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return float("nan")
    return float(sum(clean) / len(clean))


def gini(values: list[float]) -> float:
    clean = sorted(float(value) for value in values)
    if not clean:
        return 0.0
    total = sum(clean)
    if abs(total) < 1e-12:
        return 0.0
    n = len(clean)
    weighted = sum((idx + 1) * value for idx, value in enumerate(clean))
    return float((2.0 * weighted) / (n * total) - (n + 1.0) / n)


def graph_edges(graph_json: dict[str, Any]) -> set[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()
    for source, vertex in graph_json.get("data", {}).items():
        for match in vertex.get("matches", []) or []:
            recipient = str(match.get("recipient", ""))
            if recipient:
                edges.add((str(source), recipient))
    return edges


def count_directed_3cycles(edges: set[tuple[str, str]]) -> int:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for source, target in edges:
        if source != target:
            adjacency[source].add(target)

    count = 0
    for source, targets in adjacency.items():
        for middle in targets:
            if middle == source:
                continue
            for target in adjacency.get(middle, set()):
                if target in (source, middle):
                    continue
                if source in adjacency.get(target, set()):
                    count += 1
    return int(count // 3)


def strongly_connected_components(vertices: set[str], edges: set[tuple[str, str]]) -> list[list[str]]:
    adjacency: dict[str, list[str]] = {vertex: [] for vertex in vertices}
    for source, target in edges:
        adjacency.setdefault(source, []).append(target)
        adjacency.setdefault(target, [])

    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    components: list[list[str]] = []

    def strongconnect(vertex: str) -> None:
        nonlocal index
        indices[vertex] = index
        lowlink[vertex] = index
        index += 1
        stack.append(vertex)
        on_stack.add(vertex)

        for target in adjacency.get(vertex, []):
            if target not in indices:
                strongconnect(target)
                lowlink[vertex] = min(lowlink[vertex], lowlink[target])
            elif target in on_stack:
                lowlink[vertex] = min(lowlink[vertex], indices[target])

        if lowlink[vertex] == indices[vertex]:
            component: list[str] = []
            while True:
                item = stack.pop()
                on_stack.remove(item)
                component.append(item)
                if item == vertex:
                    break
            components.append(component)

    for vertex in sorted(vertices):
        if vertex not in indices:
            strongconnect(vertex)

    return components


def compute_raw_topology_metrics(graph_json: dict[str, Any]) -> dict[str, Any]:
    data = graph_json.get("data", {})
    edges = graph_edges(graph_json)
    vertices = {str(vertex) for vertex in data.keys()}
    for source, target in edges:
        vertices.add(source)
        vertices.add(target)

    metadata_vertices = graph_json.get("metadata", {}).get("total_vertices")
    num_vertices = int(metadata_vertices) if metadata_vertices is not None else len(vertices)
    num_edges = len(edges)
    density = 0.0 if num_vertices <= 1 else float(num_edges / (num_vertices * (num_vertices - 1)))

    out_degree = {vertex: 0 for vertex in vertices}
    in_degree = {vertex: 0 for vertex in vertices}
    for source, target in edges:
        out_degree[source] = out_degree.get(source, 0) + 1
        in_degree[target] = in_degree.get(target, 0) + 1
        in_degree.setdefault(source, in_degree.get(source, 0))
        out_degree.setdefault(target, out_degree.get(target, 0))

    reciprocal_edge_count = sum(1 for source, target in edges if (target, source) in edges)
    components = strongly_connected_components(vertices, edges)
    largest_scc = max((len(component) for component in components), default=0)

    return {
        "num_vertices": num_vertices,
        "num_edges": num_edges,
        "density": density,
        "in_degree_mean": finite_mean([float(value) for value in in_degree.values()]),
        "out_degree_mean": finite_mean([float(value) for value in out_degree.values()]),
        "in_degree_gini": gini([float(value) for value in in_degree.values()]),
        "out_degree_gini": gini([float(value) for value in out_degree.values()]),
        "reciprocal_edge_count": int(reciprocal_edge_count),
        "num_2cycles": int(reciprocal_edge_count // 2),
        "num_3cycles": count_directed_3cycles(edges),
        "largest_scc_fraction": 0.0 if num_vertices == 0 else float(largest_scc / num_vertices),
        "number_of_sccs": len(components),
    }


def cycle_candidate_counts(solver) -> dict[str, int]:
    length_counts = defaultdict(int)
    for candidate in solver.cycle_candidates:
        length_counts[len(candidate.get("edges", []))] += 1
    return {
        "num_cycle_candidates": int(len(solver.cycle_candidates)),
        "num_2cycle_candidates": int(length_counts.get(2, 0)),
        "num_3cycle_candidates": int(length_counts.get(3, 0)),
        "num_chain_candidates": int(len(solver.valid_chain_keys)),
    }


def rows_for_record(args, graph_path: Path, record: dict[str, Any]) -> list[dict[str, Any]]:
    graph_json = json.loads(graph_path.read_text(encoding="utf-8"))
    topology = compute_raw_topology_metrics(graph_json)

    w_true = np.asarray(record["w_true"], dtype=float)
    y_oracle = np.asarray(record["y_optimal"], dtype=float)
    oracle_obj = float(np.dot(w_true, y_oracle))
    solver = record["graph"]["cached_solver"]
    candidate_counts = cycle_candidate_counts(solver)

    solutions = solve_top_distinct_edge_solutions(
        solver=solver,
        weights=w_true,
        max_solutions=args.max_solutions,
        max_cut_attempts=args.max_cut_attempts,
        reset_before_solve=not args.no_reset_before_solve,
    )

    rows: list[dict[str, Any]] = []
    for solution in solutions:
        y = np.asarray(solution["y"], dtype=float)
        true_obj = float(np.dot(w_true, y))
        gap = float(oracle_obj - true_obj)
        oracle_overlap = edge_overlap_metrics(y, y_oracle)

        rows.append(
            {
                "regime": args.regime,
                "max_cycle": int(args.max_cycle),
                "max_chain": int(args.max_chain),
                "graph_id": record["filename"],
                "solution_rank": int(solution["solution_rank"]),
                "cut_attempt": int(solution["cut_attempt"]),
                "solver_status": int(solution["status"]),
                "solver_status_name": solution["status_name"],
                "solver_obj_true": float(solution["solver_obj"]),
                "true_obj": true_obj,
                "oracle_obj": oracle_obj,
                "gap_to_oracle": gap,
                "normalized_gap_to_oracle": normalized_gap(gap, oracle_obj),
                **topology,
                "edge_count": int(np.sum(y > 0.5)),
                **candidate_counts,
                "same_solution_as_oracle": bool(oracle_overlap["same_solution"]),
                "edge_jaccard_with_oracle": float(oracle_overlap["edge_jaccard"]),
                "edge_hamming_with_oracle": float(oracle_overlap["edge_hamming"]),
                "edge_overlap_count_with_oracle": int(oracle_overlap["edge_overlap_count"]),
                "solution_edge_signature": solution["solution_edge_signature"],
            }
        )

    return rows


def read_float(row: dict[str, Any], key: str) -> float:
    return float(row.get(key, "nan"))


def summarize_oracle_landscape_rows(
    rows: list[dict[str, Any]],
    top_m: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["graph_id"])].append(row)

    summary_rows: list[dict[str, Any]] = []
    passthrough = [
        "regime",
        "max_cycle",
        "max_chain",
        "num_vertices",
        "num_edges",
        "density",
        "in_degree_mean",
        "out_degree_mean",
        "in_degree_gini",
        "out_degree_gini",
        "reciprocal_edge_count",
        "num_2cycles",
        "num_3cycles",
        "largest_scc_fraction",
        "number_of_sccs",
        "num_cycle_candidates",
        "num_2cycle_candidates",
        "num_3cycle_candidates",
        "num_chain_candidates",
    ]

    for graph_id in sorted(grouped):
        group = sorted(grouped[graph_id], key=lambda row: int(float(row["solution_rank"])))
        first = group[0]
        gaps = [read_float(row, "normalized_gap_to_oracle") for row in group]
        near_5 = [
            row
            for row in group
            if read_float(row, "normalized_gap_to_oracle") <= 0.05 + 1e-12
        ]
        second_best_gap = (
            100.0 * read_float(group[1], "normalized_gap_to_oracle")
            if len(group) >= 2
            else float("nan")
        )

        summary = {
            "graph_id": graph_id,
            "requested_top_m": int(top_m),
            "observed_top_m": len(group),
            "oracle_objective": read_float(first, "oracle_obj"),
            "oracle_second_best_gap_pct": second_best_gap,
            "num_observed_solutions_within_1pct": sum(1 for gap in gaps if gap <= 0.01 + 1e-12),
            "num_observed_solutions_within_5pct": sum(1 for gap in gaps if gap <= 0.05 + 1e-12),
            "num_observed_solutions_within_10pct": sum(1 for gap in gaps if gap <= 0.10 + 1e-12),
            "near_oracle_jaccard_mean": finite_mean(
                [read_float(row, "edge_jaccard_with_oracle") for row in near_5]
            ),
            "near_oracle_jaccard_min": (
                min(read_float(row, "edge_jaccard_with_oracle") for row in near_5)
                if near_5
                else float("nan")
            ),
        }
        for key in passthrough:
            summary[key] = first.get(key, "")
        summary_rows.append(summary)

    return summary_rows


def compute_oracle_landscape_rows(args) -> list[dict[str, Any]]:
    common, _, _, _, _ = ensure_step1c_imports()

    import gurobipy as gp

    test_entries = load_or_make_split_entries(
        split_path=args.split_path,
        dataset_dir=args.dataset_dir,
        split_seed=args.split_seed,
        train_pool_size=args.train_pool_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
    )
    test_entries = filter_split_entries_by_graphs(test_entries, set(args.graphs))
    graph_paths = [resolve_graph_path(entry, args.dataset_dir) for entry in test_entries]

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Seed", args.gurobi_seed)
    env.start()

    output_rows: list[dict[str, Any]] = []
    try:
        records = []
        try:
            records = common.load_graph_records(
                graph_paths,
                env,
                max_cycle=args.max_cycle,
                max_chain=args.max_chain,
            )
            graph_path_by_name = {path.name: path for path in graph_paths}
            for idx, record in enumerate(records, start=1):
                print(f"[{idx}/{len(records)}] graph={record['filename']}", flush=True)
                graph_path = graph_path_by_name[record["filename"]]
                output_rows.extend(rows_for_record(args, graph_path, record))
        finally:
            common.dispose_graph_records(records)
    finally:
        env.dispose()

    return output_rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Enumerate true-label top-M oracle-landscape solutions for Step2c graphs."
    )
    parser.add_argument("--regime", default=DEFAULT_REGIME)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--split-path",
        type=Path,
        default=PROJECT_ROOT
        / "surrogate_experiment_results"
        / "Step2_resampling"
        / "splits"
        / DEFAULT_REGIME
        / "master_split_seed=42.json",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--graphs", nargs="+", default=list(DEFAULT_GRAPHS))
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--train-pool-size", type=int, default=1200)
    parser.add_argument("--validation-size", type=int, default=400)
    parser.add_argument("--test-size", type=int, default=400)
    parser.add_argument("--gurobi-seed", type=int, default=42)
    parser.add_argument("--max-cycle", type=int, default=3)
    parser.add_argument("--max-chain", type=int, default=4)
    parser.add_argument("--max-solutions", type=int, default=50)
    parser.add_argument("--max-cut-attempts", type=int, default=200)
    parser.add_argument("--no-reset-before-solve", action="store_true")
    args = parser.parse_args(argv)

    if args.max_solutions < 1:
        raise ValueError("--max-solutions must be >= 1")
    if args.max_cut_attempts < args.max_solutions:
        raise ValueError("--max-cut-attempts should be >= --max-solutions")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    rows = compute_oracle_landscape_rows(args)
    write_csv(args.output, rows, CSV_FIELDS)
    summary_rows = summarize_oracle_landscape_rows(rows, top_m=args.max_solutions)
    write_csv(args.summary_output, summary_rows, SUMMARY_FIELDS)
    print(f"Saved {len(rows)} oracle-landscape rows to {args.output}")
    print(f"Saved {len(summary_rows)} oracle-landscape summary rows to {args.summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
