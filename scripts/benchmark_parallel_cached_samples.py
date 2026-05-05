#!/usr/bin/env python
"""Benchmark serial vs parallel cached Gurobi solves for KEP graphs.

This is a standalone diagnostic script. It does not change the end-to-end
training code. The benchmark mirrors the FY training inner loop: one fixed
graph, M perturbed edge-weight vectors, and repeated solves with cached hybrid
Gurobi models. By default it runs several graphs so invoking this file directly
produces a small multi-graph benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import gurobipy as gp
import numpy as np
from gurobipy import GurobiError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment_config import PROCESSED_DATA_DIR, RESULTS_ROOT, resolve_path
from formulations.common.backend_utils import infer_ndd_mask
from formulations.hybrid.backend import CachedHybridKepModel
from model.graph_utils import parse_json_to_dfl_data
from model.model_structure import DEFAULT_Y_SCALE
from scripts.benchmark_hybrid_model_reuse_on_dataset import (
    cycle_candidates_only,
    make_weight_vectors,
    select_data_dir,
)


SUMMARY_FIELDS = [
    "graph",
    "num_nodes",
    "num_edges",
    "num_cycles",
    "m_samples",
    "parallel_copies",
    "serial_build_time",
    "serial_solve_time",
    "serial_total_time",
    "parallel_build_time",
    "parallel_solve_wall_time",
    "parallel_total_time",
    "speedup_solve_only",
    "speedup_including_build",
    "max_obj_diff",
    "mean_obj_diff",
    "obj_mismatches",
    "selection_mismatches",
]

PER_SOLVE_FIELDS = [
    "graph",
    "parallel_copies",
    "sample_idx",
    "worker_idx",
    "serial_time",
    "parallel_time",
    "serial_obj",
    "parallel_obj",
    "obj_diff",
    "selection_mismatch",
]


@dataclass
class SolverBundle:
    env: gp.Env
    solver: CachedHybridKepModel

    def dispose(self) -> None:
        self.solver.dispose()
        self.env.dispose()


def parse_copy_counts(raw: str) -> list[int]:
    counts = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError(f"parallel copy counts must be positive, got {value}")
        counts.append(value)
    if not counts:
        raise ValueError("at least one parallel copy count is required")
    return counts


def parse_args() -> argparse.Namespace:
    default_output_dir = (
        Path(RESULTS_ROOT)
        / "hybrid_parallel_cached_samples_benchmark"
        / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    )
    parser = argparse.ArgumentParser(
        description=(
            "Compare M=16 serial cached Gurobi solves with parallel cached "
            "solver-copy pools on KEP graphs."
        )
    )
    parser.add_argument("--data_dir", type=str, default=str(PROCESSED_DATA_DIR))
    parser.add_argument("--graph_index", type=int, default=0)
    parser.add_argument("--num_graphs", type=int, default=6)
    parser.add_argument("--graph_name", type=str, default=None)
    parser.add_argument("--m_samples", type=int, default=16)
    parser.add_argument("--parallel_copies", type=str, default="4,8,16")
    parser.add_argument("--max_cycle", type=int, default=3)
    parser.add_argument("--max_chain", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--time_limit", type=float, default=None)
    parser.add_argument("--reset_before_solve", action="store_true")
    parser.add_argument("--output_dir", type=str, default=str(resolve_path(default_output_dir)))
    return parser.parse_args()


def select_graph_paths(
    data_dir: Path,
    graph_name: str | None,
    graph_index: int,
    num_graphs: int,
) -> list[Path]:
    graph_files = sorted(data_dir.glob("G-*.json"))
    if not graph_files:
        raise SystemExit(f"No processed KEP graph files found under {data_dir}")

    if graph_name:
        graph_path = data_dir / graph_name
        if graph_path not in graph_files:
            raise SystemExit(f"Graph {graph_name} was not found under {data_dir}")
        return [graph_path]

    if num_graphs <= 0:
        raise SystemExit("--num_graphs must be positive")
    if graph_index < 0 or graph_index >= len(graph_files):
        raise SystemExit(f"--graph_index {graph_index} is out of range for {len(graph_files)} graph files")

    selected = graph_files[graph_index : graph_index + num_graphs]
    if not selected:
        raise SystemExit("No graph files selected for benchmark")
    return selected


def load_graph_from_path(graph_path: Path, args: argparse.Namespace):
    data = parse_json_to_dfl_data(
        graph_path,
        max_cycle=args.max_cycle,
        max_chain=args.max_chain,
        label_scale=DEFAULT_Y_SCALE,
    )
    return data


def create_env(seed: int, threads: int) -> gp.Env:
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Seed", seed)
    env.setParam("Threads", threads)
    env.start()
    return env


def create_solver_bundle(data, args: argparse.Namespace, name: str) -> SolverBundle:
    env = create_env(args.seed, args.threads)
    node_is_ndd = infer_ndd_mask(data.x)
    solver = CachedHybridKepModel(
        edge_index=data.edge_index,
        is_ndd_mask=node_is_ndd,
        num_nodes=int(data.num_nodes_custom[0].item()),
        cycle_candidates=cycle_candidates_only(data.candidates),
        max_chain=args.max_chain,
        env=env,
        time_limit=args.time_limit,
        threads=args.threads,
        name=name,
    )
    return SolverBundle(env=env, solver=solver)


def split_indexed_items(items: Iterable, num_chunks: int) -> list[list[tuple[int, object]]]:
    chunks = [[] for _ in range(num_chunks)]
    for idx, item in enumerate(items):
        chunks[idx % num_chunks].append((idx, item))
    return chunks


def solve_serial(solver: CachedHybridKepModel, weight_vectors: np.ndarray, args: argparse.Namespace):
    results = []
    solve_times = []
    for weights in weight_vectors:
        t0 = time.perf_counter()
        result = solver.solve(
            weights,
            time_limit=args.time_limit,
            reset_before_solve=args.reset_before_solve,
        )
        solve_times.append(time.perf_counter() - t0)
        results.append(result)
    return results, solve_times


def solve_chunk(
    worker_idx: int,
    solver: CachedHybridKepModel,
    indexed_weights: list[tuple[int, np.ndarray]],
    args: argparse.Namespace,
) -> list[tuple[int, int, dict, float]]:
    rows = []
    for sample_idx, weights in indexed_weights:
        t0 = time.perf_counter()
        result = solver.solve(
            weights,
            time_limit=args.time_limit,
            reset_before_solve=args.reset_before_solve,
        )
        rows.append((sample_idx, worker_idx, result, time.perf_counter() - t0))
    return rows


def solve_parallel(
    bundles: list[SolverBundle],
    weight_vectors: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[dict], list[float], list[int], float]:
    chunks = split_indexed_items(weight_vectors, len(bundles))
    results = [None] * len(weight_vectors)
    solve_times = [0.0] * len(weight_vectors)
    worker_indices = [-1] * len(weight_vectors)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(bundles)) as executor:
        futures = [
            executor.submit(solve_chunk, worker_idx, bundle.solver, chunks[worker_idx], args)
            for worker_idx, bundle in enumerate(bundles)
            if chunks[worker_idx]
        ]
        for future in as_completed(futures):
            for sample_idx, worker_idx, result, solve_time in future.result():
                results[sample_idx] = result
                solve_times[sample_idx] = solve_time
                worker_indices[sample_idx] = worker_idx
    wall_time = time.perf_counter() - t0

    return results, solve_times, worker_indices, wall_time


def compare_results(serial_results: list[dict], candidate_results: list[dict], tolerance: float = 1e-6) -> dict:
    obj_diffs = []
    selection_mismatches = 0
    for serial, candidate in zip(serial_results, candidate_results):
        obj_diff = abs(float(serial["objective"]) - float(candidate["objective"]))
        obj_diffs.append(obj_diff)
        selection_mismatches += int(
            not np.array_equal(serial["edge_selection"], candidate["edge_selection"])
        )

    return {
        "max_obj_diff": float(max(obj_diffs)) if obj_diffs else 0.0,
        "mean_obj_diff": float(np.mean(obj_diffs)) if obj_diffs else 0.0,
        "obj_mismatches": int(sum(diff > tolerance for diff in obj_diffs)),
        "selection_mismatches": int(selection_mismatches),
    }


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(output_dir: Path, summary_rows: list[dict], per_solve_rows: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "parallel_cached_samples_summary.csv", summary_rows, SUMMARY_FIELDS)
    write_csv(output_dir / "parallel_cached_samples_per_solve.csv", per_solve_rows, PER_SOLVE_FIELDS)
    with open(output_dir / "parallel_cached_samples_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, indent=2)


def print_summary(row: dict) -> None:
    print(f"\nGraph {row['graph']} | parallel copies: {row['parallel_copies']}")
    print(f"  serial build time       : {row['serial_build_time']:.4f} s")
    print(f"  serial solve time       : {row['serial_solve_time']:.4f} s")
    print(f"  parallel build time     : {row['parallel_build_time']:.4f} s")
    print(f"  parallel solve wall time: {row['parallel_solve_wall_time']:.4f} s")
    print(f"  speedup solve only      : {row['speedup_solve_only']:.2f}x")
    print(f"  speedup incl. build     : {row['speedup_including_build']:.2f}x")
    print(f"  max objective diff      : {row['max_obj_diff']:.6e}")
    print(f"  objective mismatches    : {row['obj_mismatches']}")
    print(f"  selection mismatches    : {row['selection_mismatches']}")


def benchmark_one_graph(
    graph_path: Path,
    args: argparse.Namespace,
    copy_counts: list[int],
) -> tuple[list[dict], list[dict]]:
    data = load_graph_from_path(graph_path, args)
    graph = graph_path.name
    weight_vectors = make_weight_vectors(
        data,
        num_weight_vectors=args.m_samples,
        rng=np.random.default_rng(args.seed),
    )

    print(f"\n--- Graph: {graph_path} ---")

    serial_bundle = None
    parallel_bundles = []
    summary_rows = []
    per_solve_rows = []
    try:
        t0 = time.perf_counter()
        serial_bundle = create_solver_bundle(data, args, name=f"serial_cached_{graph}")
        serial_build_time = time.perf_counter() - t0
        serial_results, serial_times = solve_serial(serial_bundle.solver, weight_vectors, args)
        serial_solve_time = float(sum(serial_times))
        serial_total_time = serial_build_time + serial_solve_time

        graph_metadata = {
            "graph": graph,
            "num_nodes": int(data.num_nodes_custom[0].item()),
            "num_edges": int(data.edge_index.shape[1]),
            "num_cycles": len(cycle_candidates_only(data.candidates)),
            "m_samples": int(args.m_samples),
            "serial_build_time": float(serial_build_time),
            "serial_solve_time": float(serial_solve_time),
            "serial_total_time": float(serial_total_time),
        }

        for copies in copy_counts:
            for bundle in parallel_bundles:
                bundle.dispose()
            parallel_bundles = []

            t_build = time.perf_counter()
            parallel_bundles = [
                create_solver_bundle(data, args, name=f"parallel_cached_{copies}_{idx}_{graph}")
                for idx in range(copies)
            ]
            parallel_build_time = time.perf_counter() - t_build
            parallel_results, parallel_times, worker_indices, parallel_wall_time = solve_parallel(
                parallel_bundles,
                weight_vectors,
                args,
            )
            comparison = compare_results(serial_results, parallel_results)

            parallel_total_time = parallel_build_time + parallel_wall_time
            row = {
                **graph_metadata,
                "parallel_copies": int(copies),
                "parallel_build_time": float(parallel_build_time),
                "parallel_solve_wall_time": float(parallel_wall_time),
                "parallel_total_time": float(parallel_total_time),
                "speedup_solve_only": (
                    serial_solve_time / parallel_wall_time if parallel_wall_time > 0 else float("inf")
                ),
                "speedup_including_build": (
                    serial_total_time / parallel_total_time if parallel_total_time > 0 else float("inf")
                ),
                **comparison,
            }
            summary_rows.append(row)
            print_summary(row)

            for sample_idx, (serial, parallel) in enumerate(zip(serial_results, parallel_results)):
                obj_diff = abs(float(serial["objective"]) - float(parallel["objective"]))
                per_solve_rows.append(
                    {
                        "graph": graph,
                        "parallel_copies": int(copies),
                        "sample_idx": int(sample_idx),
                        "worker_idx": int(worker_indices[sample_idx]),
                        "serial_time": float(serial_times[sample_idx]),
                        "parallel_time": float(parallel_times[sample_idx]),
                        "serial_obj": float(serial["objective"]),
                        "parallel_obj": float(parallel["objective"]),
                        "obj_diff": float(obj_diff),
                        "selection_mismatch": bool(
                            not np.array_equal(serial["edge_selection"], parallel["edge_selection"])
                        ),
                    }
                )

    except GurobiError as exc:
        print(f"Gurobi unavailable: {exc}")
        raise
    finally:
        if serial_bundle is not None:
            serial_bundle.dispose()
        for bundle in parallel_bundles:
            bundle.dispose()

    return summary_rows, per_solve_rows


def main() -> int:
    args = parse_args()
    if args.m_samples <= 0:
        raise SystemExit("--m_samples must be positive")
    copy_counts = parse_copy_counts(args.parallel_copies)
    output_dir = resolve_path(args.output_dir)
    data_dir = select_data_dir(args.data_dir)
    graph_paths = select_graph_paths(data_dir, args.graph_name, args.graph_index, args.num_graphs)

    print("===== Parallel Cached Hybrid Solver Benchmark =====")
    print(f"Data dir               : {data_dir}")
    print(f"Graph start index      : {args.graph_index}")
    print(f"Graphs selected        : {len(graph_paths)}")
    print(f"M samples              : {args.m_samples}")
    print(f"Parallel copy counts   : {copy_counts}")
    print(f"Threads per Gurobi model: {args.threads}")
    print(f"Output dir             : {output_dir}")

    summary_rows = []
    per_solve_rows = []
    try:
        for graph_path in graph_paths:
            graph_summary_rows, graph_per_solve_rows = benchmark_one_graph(graph_path, args, copy_counts)
            summary_rows.extend(graph_summary_rows)
            per_solve_rows.extend(graph_per_solve_rows)
    except GurobiError:
        return 0

    write_outputs(output_dir, summary_rows, per_solve_rows)
    print(f"\nOutputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
