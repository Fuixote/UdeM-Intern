"""
Append average True Regret to a saved parameter trajectory.

Default input:
    trajectory_mse.npy             shape (n_epochs + 1, 2)

Default output:
    trajectory_mse_with_regret.npy shape (n_epochs + 1, 3)

Columns for the default output:
    theta_1, theta_2, true_regret

If the input has extra columns, they are preserved and True Regret is appended.
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
    compute_regret,
    feature_matrix,
    load_graph,
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


def average_true_regret(theta, graphs, env):
    total = 0.0
    for gd in graphs:
        w_hat = gd["X"] @ theta
        y_pred = solve_once(w_hat, gd["graph"], env)
        total += compute_regret(y_pred, gd["y_optimal"], gd["w_true"])
    return total / len(graphs)


def append_true_regret(trajectory, graphs, env):
    regrets = np.empty(trajectory.shape[0], dtype=np.float64)
    for idx, row in enumerate(trajectory):
        theta = row[:2]
        regrets[idx] = average_true_regret(theta, graphs, env)
        if (idx + 1) % 25 == 0 or idx + 1 == len(trajectory):
            print(
                f"  evaluated {idx + 1}/{len(trajectory)}  "
                f"theta={np.round(theta, 4)}  regret={regrets[idx]:.6f}",
                flush=True,
            )
    return np.column_stack([trajectory, regrets])


def main():
    here = Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default=str(ROOT / "dataset/processed/clean_linear_dataset"),
    )
    parser.add_argument("--traj_path", default=str(here / "trajectory_mse.npy"))
    parser.add_argument(
        "--out_path",
        default=str(here / "trajectory_mse_with_regret.npy"),
    )
    parser.add_argument("--n_total", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    trajectory = np.load(args.traj_path)
    if trajectory.ndim != 2 or trajectory.shape[1] < 2:
        raise ValueError(
            f"Expected trajectory shape (n, >=2), got {trajectory.shape} from {args.traj_path}"
        )

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Seed", args.seed)
    env.start()

    try:
        print(f"Loaded trajectory: {args.traj_path} shape={trajectory.shape}")
        print(f"Loading graphs from {args.data_dir} with n_total={args.n_total}")
        graphs = load_graphs(args.data_dir, args.n_total, seed=args.seed, env=env)

        print("Computing average True Regret for each trajectory point ...")
        with_regret = append_true_regret(trajectory, graphs, env)
        np.save(args.out_path, with_regret)

        print(f"Saved: {args.out_path} shape={with_regret.shape}")
        print(f"Final row: {np.round(with_regret[-1], 6)}")
    finally:
        env.dispose()


if __name__ == "__main__":
    main()
