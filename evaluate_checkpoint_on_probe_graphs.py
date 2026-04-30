"""
Evaluate a stage-1 linear checkpoint on the exact graph set used by a probe plot.

This is intended to compare a trained two-stage checkpoint against the
linear_probe_landscape.py metrics on the same graphs.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import gurobipy as gp
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from formulations.common.backend_utils import infer_ndd_mask
from formulations.hybrid.backend import solve_cf_cycle_pief_chain
from model.graph_utils import parse_json_to_dfl_data


def load_graph(graph_path, max_cycle=3, max_chain=4):
    data = parse_json_to_dfl_data(
        graph_path,
        max_cycle=max_cycle,
        max_chain=max_chain,
        label_scale=1.0,
    )
    src, dst = data.edge_index
    return {
        "utility": data.edge_attr[:, 0].numpy(),
        "recipient_cPRA": data.x[dst, 1].numpy(),
        "source_donor_age": data.x[src, 7].numpy(),
        "w_true": data.y.numpy(),
        "edge_index": data.edge_index,
        "cycle_candidates": [c for c in data.candidates if c["type"] == "cycle"],
        "node_is_ndd": infer_ndd_mask(data.x),
        "num_nodes": data.num_nodes_custom[0].item(),
        "filename": data.filename,
    }


def feature_matrix(graph, feature_mode):
    if feature_mode == "utility_cpra":
        return np.stack([graph["utility"], graph["recipient_cPRA"]], axis=1)
    if feature_mode == "lr_small":
        return np.stack(
            [graph["utility"], graph["recipient_cPRA"], graph["source_donor_age"]],
            axis=1,
        )
    raise ValueError(f"Unsupported checkpoint feature mode for this probe evaluator: {feature_mode}")


def solve_once(weights, graph, env):
    result = solve_cf_cycle_pief_chain(
        weights=weights,
        edge_index=graph["edge_index"],
        is_ndd_mask=graph["node_is_ndd"],
        num_nodes=graph["num_nodes"],
        cycle_candidates=graph["cycle_candidates"],
        env=env,
    )
    return result["edge_selection"]


def checkpoint_linear_params(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        config = dict(checkpoint.get("config", {}))
    else:
        state_dict = checkpoint
        config = {}

    weight = state_dict["net.0.weight"].detach().cpu().numpy().reshape(-1)
    bias_tensor = state_dict.get("net.0.bias")
    bias = float(bias_tensor.detach().cpu().numpy().reshape(-1)[0]) if bias_tensor is not None else 0.0
    feature_mode = config.get("FEATURE_MODE")
    if feature_mode is None:
        feature_mode = "utility_cpra" if len(weight) == 2 else "lr_small"
    return weight.astype(float), bias, feature_mode, config


def main():
    parser = argparse.ArgumentParser(description="Evaluate a linear checkpoint on probe plot graphs.")
    parser.add_argument("--checkpoint", required=True, help="Path to best_stage1_model_real.pth")
    parser.add_argument(
        "--metrics_json",
        default="plot_results/epsilon=0.2/utility_cpra/metrics.json",
        help="Probe metrics.json containing graph_filenames and reference metrics",
    )
    parser.add_argument(
        "--data_dir",
        default="dataset/processed/2026-04-17_135607",
        help="Processed graph directory containing G-*.json files",
    )
    args = parser.parse_args()

    metrics = json.load(open(args.metrics_json))
    graph_names = metrics["graph_filenames"]
    theta, bias, feature_mode, config = checkpoint_linear_params(args.checkpoint)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Checkpoint feature_mode: {feature_mode}")
    print(f"Linear weights: {np.array2string(theta, precision=6)}")
    print(f"Linear bias: {bias:.6f}")
    print(f"Graphs from metrics: {len(graph_names)}")
    print(
        "Reference plot metrics: "
        f"R_mse={metrics.get('R_mse'):.6f}, "
        f"R_oracle={metrics.get('R_oracle'):.6f}, "
        f"R_fy={metrics.get('R_fy'):.6f}"
    )

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Seed", 42)
    env.start()

    total_regret = 0.0
    total_optimal = 0.0
    total_achieved = 0.0
    try:
        for name in graph_names:
            graph = load_graph(os.path.join(args.data_dir, name))
            X = feature_matrix(graph, feature_mode)
            w_hat = X @ theta + bias
            y_pred = solve_once(w_hat, graph, env)
            y_opt = solve_once(graph["w_true"], graph, env)
            achieved = float(np.dot(graph["w_true"], y_pred))
            optimal = float(np.dot(graph["w_true"], y_opt))
            total_achieved += achieved
            total_optimal += optimal
            total_regret += optimal - achieved
    finally:
        env.dispose()

    n = len(graph_names)
    print("\nCheckpoint on probe graph set:")
    print(f"Avg Optimal  = {total_optimal / n:.6f}")
    print(f"Avg Achieved = {total_achieved / n:.6f}")
    print(f"Avg Regret   = {total_regret / n:.6f}")
    print(f"Gap vs plot R_oracle = {total_regret / n - metrics.get('R_oracle'):.6f}")
    print(f"Gap vs plot R_mse    = {total_regret / n - metrics.get('R_mse'):.6f}")


if __name__ == "__main__":
    main()
