#!/usr/bin/env python
"""Demo: random 50-node KEP graph, rebuild every solve vs reuse one model."""
import argparse, sys, time
from pathlib import Path
import gurobipy as gp, numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from formulations.hybrid.backend import CachedHybridKepModel, solve_cf_cycle_pief_chain

def random_graph(num_nodes, num_edges, num_ndds, seed):
    rng = np.random.default_rng(seed); edges = set()
    while len(edges) < num_edges:
        s = int(rng.integers(0, num_nodes)); d = int(rng.integers(num_ndds, num_nodes))
        if s != d: edges.add((s, d))
    is_ndd = np.zeros(num_nodes, dtype=bool); is_ndd[:num_ndds] = True
    return {"edge_index": np.array(sorted(edges), dtype=np.int64).T, "is_ndd": is_ndd, "num_nodes": num_nodes, "cycles": []}
def random_weights(num_vectors, num_edges, seed): return np.random.default_rng(seed).integers(0, 101, size=(num_vectors, num_edges), dtype=np.int32)
def benchmark_graph(graph, weights, max_chain, env):
    edge_index, is_ndd, n, cycles = graph["edge_index"], graph["is_ndd"], graph["num_nodes"], graph["cycles"]
    t = time.perf_counter()
    for w in weights: solve_cf_cycle_pief_chain(w, edge_index, is_ndd, n, cycles, max_chain=max_chain, env=env)
    rebuild = time.perf_counter() - t
    t = time.perf_counter(); model = CachedHybridKepModel(edge_index, is_ndd, n, cycles, max_chain=max_chain, env=env)
    build = time.perf_counter() - t
    try:
        t = time.perf_counter()
        for w in weights: model.solve(w)
        reuse = time.perf_counter() - t
    finally:
        model.dispose()
    return {"rebuild_seconds": rebuild, "build_seconds": build, "reuse_seconds": reuse, "speedup": rebuild / reuse}
def main():
    p = argparse.ArgumentParser(); p.add_argument("--num_nodes", type=int, default=50)
    p.add_argument("--num_edges", type=int, default=300); p.add_argument("--num_ndds", type=int, default=5)
    p.add_argument("--num_weights", type=int, default=100); p.add_argument("--max_chain", type=int, default=4)
    p.add_argument("--seed", type=int, default=42); args = p.parse_args()
    graph = random_graph(args.num_nodes, args.num_edges, args.num_ndds, args.seed)
    weights = random_weights(args.num_weights, graph["edge_index"].shape[1], args.seed + 1)
    env = gp.Env(empty=True); env.setParam("OutputFlag", 0); env.start()
    try:
        r = benchmark_graph(graph, weights, args.max_chain, env)
        print(f"random graph: nodes={args.num_nodes}, edges={args.num_edges}, NDDs={args.num_ndds}")
        print(f"{args.num_weights} random integer objectives in [0,100]")
        print(f"  rebuild every time : {r['rebuild_seconds']:.4f}s")
        print(f"  reuse model solves : {r['reuse_seconds']:.4f}s (+ build {r['build_seconds']:.4f}s)")
        print(f"  solve speedup      : {r['speedup']:.2f}x")
    finally:
        env.dispose()
if __name__ == "__main__": main()
