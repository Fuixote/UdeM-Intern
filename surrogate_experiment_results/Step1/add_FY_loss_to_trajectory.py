"""
Append average Fenchel-Young surrogate loss to a saved 2D parameter trajectory.

Default input:
    trajectory_fy.npy                shape (n_epochs + 1, 2)

Default output:
    trajectory_fy_with_fy_loss.npy   shape (n_epochs + 1, 3)

Columns for the default output:
    theta_1, theta_2, fy_loss

The loss matches the objective whose gradient is used by Step1.py::grad_fy:

    L_FY(theta) ~= E_z[max_y <w_hat(theta) + z, y>] - <w_hat(theta), y*>

up to terms that are constant with respect to theta.
"""

import argparse
import glob
import os
import random
import sys
from pathlib import Path

import gurobipy as gp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from linear_probe_landscape import (  # noqa: E402
    CachedHybridKepModel,
    feature_matrix,
    load_graph,
    make_antithetic_perturbations,
    solve_once,
)


PROBE = {
    "feature_names": ["utility", "recipient_cPRA"],
    "feature_labels": [r"$\theta_1$ (utility weight)", r"$\theta_2$ (cPRA weight)"],
}


def load_graphs(data_dir, n_total, seed=42, env=None):
    files = sorted(glob.glob(f"{data_dir}/G-*.json"))
    chosen = random.Random(seed).sample(files, min(n_total, len(files)))
    graphs = []
    for i, fp in enumerate(chosen):
        g = load_graph(fp)
        g["cached_solver"] = CachedHybridKepModel(g, env)
        y_optimal = solve_once(g["w_true"], g, env)
        graphs.append(
            {
                "X": feature_matrix(g, PROBE),
                "w_true": g["w_true"],
                "y_optimal": y_optimal,
                "graph": g,
            }
        )
        if (i + 1) % 20 == 0 or i + 1 == len(chosen):
            print(f"  loaded {i + 1}/{len(chosen)}", flush=True)
    return graphs


def make_perturbations_by_graph(graphs, eps_abs, M, seed):
    rng = np.random.RandomState(seed)
    return [
        make_antithetic_perturbations(rng, M, gd["w_true"].shape[0], eps_abs)
        for gd in graphs
    ]


def average_fy_loss(theta, graphs, perturbations_by_graph, env):
    total = 0.0
    for gd, perturbations in zip(graphs, perturbations_by_graph):
        w_hat = (gd["X"] @ theta).astype(np.float32)
        target_score = np.dot(w_hat, gd["y_optimal"])
        loss_sum = 0.0
        for z in perturbations:
            perturbed_weights = w_hat + z
            y_perturbed = solve_once(perturbed_weights, gd["graph"], env)
            loss_sum += np.dot(perturbed_weights, y_perturbed) - target_score
        total += loss_sum / len(perturbations)
    return total / len(graphs)


def append_fy_loss(trajectory, graphs, perturbations_by_graph, env):
    losses = np.empty(trajectory.shape[0], dtype=np.float64)
    for idx, row in enumerate(trajectory):
        theta = row[:2]
        losses[idx] = average_fy_loss(theta, graphs, perturbations_by_graph, env)
        if (idx + 1) % 25 == 0 or idx + 1 == len(trajectory):
            print(
                f"  evaluated {idx + 1}/{len(trajectory)}  "
                f"theta={np.round(theta, 4)}  fy_loss={losses[idx]:.6f}",
                flush=True,
            )
    return np.column_stack([trajectory, losses])


def main():
    here = Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default=str(ROOT / "dataset/processed/clean_linear_dataset"),
    )
    parser.add_argument("--traj_path", default=str(here / "trajectory_fy.npy"))
    parser.add_argument(
        "--out_path",
        default=str(here / "trajectory_fy_with_fy_loss.npy"),
    )
    parser.add_argument("--n_total", type=int, default=100)
    parser.add_argument("--fy_epsilon", type=float, default=0.2)
    parser.add_argument("--fy_M", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    trajectory = np.load(args.traj_path)
    if trajectory.ndim != 2 or trajectory.shape[1] < 2:
        raise ValueError(
            f"Expected trajectory shape (n, >=2), got {trajectory.shape} from {args.traj_path}"
        )
    if args.fy_M <= 0:
        raise ValueError("--fy_M must be positive")

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Seed", args.seed)
    env.start()

    try:
        print(f"Loaded trajectory: {args.traj_path} shape={trajectory.shape}")
        print(f"Loading graphs from {args.data_dir} with n_total={args.n_total}")
        graphs = load_graphs(args.data_dir, args.n_total, seed=args.seed, env=env)

        print(
            "Computing average FY loss for each trajectory point "
            f"with epsilon={args.fy_epsilon}, M={args.fy_M}, seed={args.seed} ..."
        )
        perturbations_by_graph = make_perturbations_by_graph(
            graphs, args.fy_epsilon, args.fy_M, args.seed
        )
        with_loss = append_fy_loss(trajectory, graphs, perturbations_by_graph, env)
        np.save(args.out_path, with_loss)

        print(f"Saved: {args.out_path} shape={with_loss.shape}")
        print(f"Final row: {np.round(with_loss[-1], 6)}")
    finally:
        env.dispose()


if __name__ == "__main__":
    main()
