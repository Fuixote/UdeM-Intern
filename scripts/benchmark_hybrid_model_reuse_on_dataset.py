#!/usr/bin/env python
import argparse
import csv
import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import sys
import time
from datetime import datetime
from pathlib import Path

import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
from gurobipy import GurobiError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment_config import PROCESSED_DATA_DIR, RESULTS_ROOT, resolve_path
from formulations.common.backend_utils import infer_ndd_mask
from formulations.hybrid.backend import CachedHybridKepModel, solve_cf_cycle_pief_chain
from model.graph_utils import load_graph_dataset, parse_json_to_dfl_data
from model.model_structure import DEFAULT_Y_SCALE


SUMMARY_FIELDS = [
    "graph",
    "num_nodes",
    "num_edges",
    "num_cycles",
    "num_weight_vectors",
    "rebuild_total_time",
    "reuse_build_time",
    "reuse_solve_time",
    "reuse_total_time",
    "speedup_including_build",
    "speedup_excluding_build",
    "max_obj_diff",
    "mean_obj_diff",
    "obj_mismatches",
    "selection_mismatches",
]

PER_SOLVE_FIELDS = [
    "graph",
    "solve_idx",
    "rebuild_time",
    "reuse_time",
    "rebuild_obj",
    "reuse_obj",
    "obj_diff",
    "rebuild_status",
    "reuse_status",
    "selection_mismatch",
]


def parse_args():
    default_output_dir = Path(RESULTS_ROOT) / "hybrid_gurobi_reuse_benchmark" / datetime.now().strftime(
        "%Y-%m-%d_%H%M%S"
    )
    parser = argparse.ArgumentParser(description="Benchmark hybrid Gurobi model rebuild vs model reuse on KEP data.")
    parser.add_argument("--data_dir", type=str, default=str(PROCESSED_DATA_DIR))
    parser.add_argument("--num_graphs", type=int, default=5)
    parser.add_argument("--num_weight_vectors", type=int, default=100)
    parser.add_argument("--max_cycle", type=int, default=3)
    parser.add_argument("--max_chain", type=int, default=4)
    parser.add_argument("--noise_scale", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--time_limit", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default=str(resolve_path(default_output_dir)))
    parser.add_argument("--reset_before_solve", action="store_true")
    return parser.parse_args()


def cycle_candidates_only(candidates):
    if candidates and isinstance(candidates[0], list):
        candidates = candidates[0]
    return [candidate for candidate in candidates if candidate.get("type") == "cycle"]


def make_weight_vectors(data, num_weight_vectors, noise_scale, rng):
    base_weights = data.y.detach().cpu().numpy().reshape(-1).astype(np.float32) * float(DEFAULT_Y_SCALE)
    if base_weights.size == 0:
        return np.empty((num_weight_vectors, 0), dtype=np.float32)
    if np.allclose(base_weights, 0.0):
        base_weights = data.edge_attr[:, 0].detach().cpu().numpy().reshape(-1).astype(np.float32)
        base_weights = np.maximum(1e-6, base_weights)

    noise = rng.normal(0.0, noise_scale, size=(num_weight_vectors, base_weights.shape[0])).astype(np.float32)
    weights = np.maximum(0.0, base_weights.reshape(1, -1) * (1.0 + noise))
    return weights.astype(np.float32, copy=False)


def select_data_dir(data_dir):
    data_dir = resolve_path(data_dir)
    if list(data_dir.glob("G-*.json")):
        return data_dir

    populated_subdirs = []
    if data_dir.is_dir():
        for child in data_dir.iterdir():
            if child.is_dir():
                graph_count = len(list(child.glob("G-*.json")))
                if graph_count > 0:
                    populated_subdirs.append((graph_count, child.name, child))
    if not populated_subdirs:
        return data_dir

    populated_subdirs.sort(key=lambda item: (-item[0], item[1]))
    return populated_subdirs[0][2]


def load_benchmark_dataset(args):
    data_dir = select_data_dir(args.data_dir)
    if data_dir != resolve_path(args.data_dir):
        print(f"Hybrid benchmark loading no top-level graph files in {resolve_path(args.data_dir)}; using {data_dir}.")
    return load_graph_dataset(
        str(data_dir),
        lambda path: parse_json_to_dfl_data(
            path,
            max_cycle=args.max_cycle,
            max_chain=args.max_chain,
            label_scale=DEFAULT_Y_SCALE,
        ),
        log_prefix="Hybrid benchmark loading",
    )


def create_gurobi_env(args):
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Seed", args.seed)
    env.setParam("Threads", args.threads)
    env.start()
    return env


def graph_filename(data):
    filename = getattr(data, "filename", "graph")
    if isinstance(filename, (list, tuple)):
        filename = filename[0]
    return str(filename)


def benchmark_one_graph(data, args, rng, env):
    graph = graph_filename(data)
    node_is_ndd = infer_ndd_mask(data.x)
    num_nodes = int(data.num_nodes_custom[0].item())
    num_edges = int(data.edge_index.shape[1])
    cycle_candidates = cycle_candidates_only(data.candidates)
    weight_vectors = make_weight_vectors(
        data,
        num_weight_vectors=args.num_weight_vectors,
        noise_scale=args.noise_scale,
        rng=rng,
    )

    rebuild_results = []
    rebuild_times = []
    for weights in weight_vectors:
        t0 = time.perf_counter()
        result = solve_cf_cycle_pief_chain(
            weights=weights,
            edge_index=data.edge_index,
            is_ndd_mask=node_is_ndd,
            num_nodes=num_nodes,
            cycle_candidates=cycle_candidates,
            max_chain=args.max_chain,
            env=env,
            time_limit=args.time_limit,
        )
        rebuild_times.append(time.perf_counter() - t0)
        rebuild_results.append(result)

    t_build0 = time.perf_counter()
    cached_model = CachedHybridKepModel(
        edge_index=data.edge_index,
        is_ndd_mask=node_is_ndd,
        num_nodes=num_nodes,
        cycle_candidates=cycle_candidates,
        max_chain=args.max_chain,
        env=env,
        time_limit=args.time_limit,
        threads=args.threads,
        name=f"cached_hybrid_{graph}",
    )
    reuse_build_time = time.perf_counter() - t_build0

    reuse_results = []
    reuse_times = []
    try:
        for weights in weight_vectors:
            t0 = time.perf_counter()
            result = cached_model.solve(
                weights,
                time_limit=args.time_limit,
                reset_before_solve=args.reset_before_solve,
            )
            reuse_times.append(time.perf_counter() - t0)
            reuse_results.append(result)
    finally:
        cached_model.dispose()

    per_solve_rows = []
    obj_diffs = []
    selection_mismatches = 0
    for solve_idx, (rebuild, reuse) in enumerate(zip(rebuild_results, reuse_results)):
        obj_diff = abs(float(rebuild["objective"]) - float(reuse["objective"]))
        selection_mismatch = not np.array_equal(rebuild["edge_selection"], reuse["edge_selection"])
        obj_diffs.append(obj_diff)
        selection_mismatches += int(selection_mismatch)
        per_solve_rows.append(
            {
                "graph": graph,
                "solve_idx": solve_idx,
                "rebuild_time": rebuild_times[solve_idx],
                "reuse_time": reuse_times[solve_idx],
                "rebuild_obj": float(rebuild["objective"]),
                "reuse_obj": float(reuse["objective"]),
                "obj_diff": obj_diff,
                "rebuild_status": int(rebuild["status"]),
                "reuse_status": int(reuse["status"]),
                "selection_mismatch": bool(selection_mismatch),
            }
        )

    rebuild_total_time = float(sum(rebuild_times))
    reuse_solve_time = float(sum(reuse_times))
    reuse_total_time = reuse_build_time + reuse_solve_time
    summary_row = {
        "graph": graph,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_cycles": len(cycle_candidates),
        "num_weight_vectors": int(len(weight_vectors)),
        "rebuild_total_time": rebuild_total_time,
        "reuse_build_time": float(reuse_build_time),
        "reuse_solve_time": reuse_solve_time,
        "reuse_total_time": float(reuse_total_time),
        "speedup_including_build": rebuild_total_time / reuse_total_time if reuse_total_time > 0 else float("inf"),
        "speedup_excluding_build": rebuild_total_time / reuse_solve_time if reuse_solve_time > 0 else float("inf"),
        "max_obj_diff": float(max(obj_diffs)) if obj_diffs else 0.0,
        "mean_obj_diff": float(np.mean(obj_diffs)) if obj_diffs else 0.0,
        "obj_mismatches": int(sum(diff > 1e-6 for diff in obj_diffs)),
        "selection_mismatches": int(selection_mismatches),
    }
    return summary_row, per_solve_rows


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def average_by_solve_idx(per_solve_rows):
    by_idx = {}
    for row in per_solve_rows:
        bucket = by_idx.setdefault(int(row["solve_idx"]), {"rebuild": [], "reuse": []})
        bucket["rebuild"].append(float(row["rebuild_time"]))
        bucket["reuse"].append(float(row["reuse_time"]))

    solve_indices = sorted(by_idx)
    rebuild = [float(np.mean(by_idx[idx]["rebuild"])) for idx in solve_indices]
    reuse = [float(np.mean(by_idx[idx]["reuse"])) for idx in solve_indices]
    return solve_indices, rebuild, reuse


def plot_per_solve_times(per_solve_rows, output_dir):
    solve_indices, rebuild, reuse = average_by_solve_idx(per_solve_rows)
    plt.figure(figsize=(8, 4.5))
    plt.plot(solve_indices, rebuild, label="Rebuild hybrid model", linewidth=2)
    plt.plot(solve_indices, reuse, label="Reuse cached hybrid model", linewidth=2)
    plt.xlabel("Solve index")
    plt.ylabel("Seconds")
    plt.title("Hybrid per-solve time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "per_solve_time_curve.png", dpi=180)
    plt.close()


def plot_total_times(summary_rows, output_dir):
    rebuild_total = sum(float(row["rebuild_total_time"]) for row in summary_rows)
    reuse_total = sum(float(row["reuse_total_time"]) for row in summary_rows)
    reuse_solve = sum(float(row["reuse_solve_time"]) for row in summary_rows)

    plt.figure(figsize=(7, 4.5))
    labels = ["Rebuild total", "Reuse incl. build", "Reuse solves only"]
    values = [rebuild_total, reuse_total, reuse_solve]
    plt.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B"])
    plt.ylabel("Seconds")
    plt.title("Hybrid total solve time")
    plt.tight_layout()
    plt.savefig(output_dir / "total_time_bar.png", dpi=180)
    plt.close()


def plot_speedups(summary_rows, output_dir):
    x = np.arange(len(summary_rows))
    width = 0.38
    labels = [row["graph"] for row in summary_rows]
    incl = [float(row["speedup_including_build"]) for row in summary_rows]
    excl = [float(row["speedup_excluding_build"]) for row in summary_rows]

    plt.figure(figsize=(max(8, len(labels) * 1.2), 4.8))
    plt.bar(x - width / 2, incl, width, label="Including build")
    plt.bar(x + width / 2, excl, width, label="Excluding build")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Speedup")
    plt.title("Hybrid model reuse speedup by graph")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "speedup_by_graph.png", dpi=180)
    plt.close()


def save_outputs(summary_rows, per_solve_rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "hybrid_reuse_summary.csv", summary_rows, SUMMARY_FIELDS)
    write_csv(output_dir / "hybrid_reuse_per_solve.csv", per_solve_rows, PER_SOLVE_FIELDS)
    if per_solve_rows:
        plot_per_solve_times(per_solve_rows, output_dir)
    if summary_rows:
        plot_total_times(summary_rows, output_dir)
        plot_speedups(summary_rows, output_dir)


def print_graph_report(summary_row):
    print(f"\nGraph {summary_row['graph']}")
    print(f"  nodes                 : {summary_row['num_nodes']}")
    print(f"  edges                 : {summary_row['num_edges']}")
    print(f"  cycle candidates      : {summary_row['num_cycles']}")
    print(f"  rebuild total time    : {summary_row['rebuild_total_time']:.4f} s")
    print(f"  reuse build time      : {summary_row['reuse_build_time']:.4f} s")
    print(f"  reuse solve time      : {summary_row['reuse_solve_time']:.4f} s")
    print(f"  reuse total time      : {summary_row['reuse_total_time']:.4f} s")
    print(f"  speedup incl. build   : {summary_row['speedup_including_build']:.2f}x")
    print(f"  speedup excl. build   : {summary_row['speedup_excluding_build']:.2f}x")
    print(f"  max objective diff    : {summary_row['max_obj_diff']:.6e}")
    print(f"  objective mismatches  : {summary_row['obj_mismatches']}")
    print(f"  selection mismatches  : {summary_row['selection_mismatches']}")


def print_overall_report(summary_rows, output_dir):
    rebuild_total = sum(float(row["rebuild_total_time"]) for row in summary_rows)
    reuse_total = sum(float(row["reuse_total_time"]) for row in summary_rows)
    reuse_solve = sum(float(row["reuse_solve_time"]) for row in summary_rows)
    max_obj_diff = max(float(row["max_obj_diff"]) for row in summary_rows) if summary_rows else 0.0

    print("\nOverall")
    print(f"  rebuild total time    : {rebuild_total:.4f} s")
    print(f"  reuse total time      : {reuse_total:.4f} s")
    print(f"  reuse solve time      : {reuse_solve:.4f} s")
    print(f"  speedup incl. build   : {rebuild_total / reuse_total if reuse_total > 0 else float('inf'):.2f}x")
    print(f"  speedup excl. build   : {rebuild_total / reuse_solve if reuse_solve > 0 else float('inf'):.2f}x")
    print(f"  max objective diff    : {max_obj_diff:.6e}")
    print(f"  outputs               : {output_dir}")


def main():
    args = parse_args()
    output_dir = resolve_path(args.output_dir)

    print("===== Hybrid Gurobi Model Reuse Benchmark =====")
    print(f"Data dir             : {resolve_path(args.data_dir)}")
    print(f"Graphs requested     : {args.num_graphs}")
    print(f"Weight vectors/graph : {args.num_weight_vectors}")
    print(f"Max chain            : {args.max_chain}")
    print(f"Noise scale          : {args.noise_scale}")
    print(f"Output dir           : {output_dir}")

    dataset = load_benchmark_dataset(args)[: args.num_graphs]
    if not dataset:
        raise SystemExit("No processed KEP graphs found for benchmark.")
    print(f"Graphs benchmarked   : {len(dataset)}")

    rng = np.random.default_rng(args.seed)
    env = None
    summary_rows = []
    per_solve_rows = []
    try:
        env = create_gurobi_env(args)
        for data in dataset:
            summary_row, graph_per_solve_rows = benchmark_one_graph(data, args, rng, env)
            summary_rows.append(summary_row)
            per_solve_rows.extend(graph_per_solve_rows)
            print_graph_report(summary_row)
    except GurobiError as exc:
        print(f"Gurobi unavailable: {exc}")
        return 0
    finally:
        if env is not None:
            env.dispose()

    save_outputs(summary_rows, per_solve_rows, output_dir)
    print_overall_report(summary_rows, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
