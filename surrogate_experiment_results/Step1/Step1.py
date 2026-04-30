"""
Step 1 — 记录 2-stage (MSE) 和 end-to-end (FY) 训练过程中每个 epoch 的 θ 参数

输出:
    trajectory_mse.npy  shape (n_epochs+1, 2)  — MSE 每 epoch 的 (θ_1, θ_2)
    trajectory_fy.npy   shape (n_epochs+1, 2)  — FY  每 epoch 的 (θ_1, θ_2)
"""

import argparse
import os
import sys
import glob
import random
from pathlib import Path

import numpy as np
import gurobipy as gp

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from linear_probe_landscape import (
    load_graph,
    CachedHybridKepModel,
    solve_once,
    make_antithetic_perturbations,
    feature_matrix,
)

PROBE = {
    "feature_names": ["utility", "recipient_cPRA"],
    "feature_labels": [r"$\theta_1$ (utility weight)", r"$\theta_2$ (cPRA weight)"],
}


# ─────────────────────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────────────────────

def load_graphs(data_dir, n_total, seed=42, env=None):
    files = sorted(glob.glob(f"{data_dir}/G-*.json"))
    chosen = random.Random(seed).sample(files, min(n_total, len(files)))
    graphs = []
    for i, fp in enumerate(chosen):
        g = load_graph(fp)
        g["cached_solver"] = CachedHybridKepModel(g, env)
        y_opt = solve_once(g["w_true"], g, env)
        X = feature_matrix(g, PROBE)
        graphs.append({"X": X, "w_true": g["w_true"], "y_optimal": y_opt, "graph": g})
        if (i + 1) % 20 == 0:
            print(f"  loaded {i+1}/{len(chosen)}", flush=True)
    return graphs


def compute_ols(graphs):
    X = np.vstack([g["X"] for g in graphs])
    w = np.concatenate([g["w_true"] for g in graphs])
    theta, *_ = np.linalg.lstsq(X, w, rcond=None)
    return theta


# ─────────────────────────────────────────────────────────────
# 梯度
# ─────────────────────────────────────────────────────────────

def grad_mse(graphs, theta):
    g = np.zeros(2)
    for d in graphs:
        g += d["X"].T @ (d["X"] @ theta - d["w_true"])
    return g / len(graphs)


def grad_fy(graphs, theta, eps_abs, M, rng, env):
    g = np.zeros(2)
    for d in graphs:
        w_hat = (d["X"] @ theta).astype(np.float32)
        perturbs = make_antithetic_perturbations(rng, M, len(w_hat), eps_abs)
        y_bar = np.mean(
            [solve_once(w_hat + z, d["graph"], env) for z in perturbs], axis=0
        )
        g += d["X"].T @ (y_bar - d["y_optimal"])
    return g / len(graphs)


# ─────────────────────────────────────────────────────────────
# Adam
# ─────────────────────────────────────────────────────────────

class Adam:
    def __init__(self, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = np.zeros(2)
        self.v = np.zeros(2)
        self.t = 0

    def step(self, theta, grad):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad ** 2
        mh = self.m / (1 - self.b1 ** self.t)
        vh = self.v / (1 - self.b2 ** self.t)
        return theta - self.lr * mh / (np.sqrt(vh) + self.eps)


# ─────────────────────────────────────────────────────────────
# 训练
# ─────────────────────────────────────────────────────────────

def run_mse(graphs, theta_init, n_epochs, lr):
    opt = Adam(lr)
    theta = theta_init.copy()
    traj = [theta.copy()]
    for ep in range(n_epochs):
        theta = opt.step(theta, grad_mse(graphs, theta))
        traj.append(theta.copy())
        if (ep + 1) % 5 == 0:
            loss = np.mean([np.mean((d["X"] @ theta - d["w_true"]) ** 2) for d in graphs])
            print(f"  [MSE] ep {ep+1:>4}  θ={np.round(theta, 3)}  L={loss:.4f}")
    return np.array(traj)


def run_fy(graphs, theta_init, n_epochs, lr, eps_abs, M, seed, env):
    opt = Adam(lr)
    rng = np.random.RandomState(seed)
    theta = theta_init.copy()
    traj = [theta.copy()]
    for ep in range(n_epochs):
        theta = opt.step(theta, grad_fy(graphs, theta, eps_abs, M, rng, env))
        traj.append(theta.copy())
        print(f"  [FY]  ep {ep+1:>4}  θ={np.round(theta, 3)}")
    return np.array(traj)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
        default=str(ROOT / "dataset/processed/clean_linear_dataset"))
    parser.add_argument("--out_dir",    default=str(Path(__file__).parent))
    parser.add_argument("--n_total",    type=int,   default=100)
    parser.add_argument("--n_epochs",   type=int,   default=1000)
    parser.add_argument("--lr_mse",     type=float, default=0.05)
    parser.add_argument("--lr_fy",      type=float, default=0.1)
    parser.add_argument("--fy_epsilon", type=float, default=0.2)
    parser.add_argument("--fy_M",       type=int,   default=16)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--theta_init", type=float, nargs=2, default=None,
        help="e.g. --theta_init 2.0 1.0")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Seed", args.seed)
    env.start()

    try:
        print("Loading graphs …")
        graphs = load_graphs(args.data_dir, args.n_total, seed=args.seed, env=env)
        print(f"  total={len(graphs)}")

        theta_ols = compute_ols(graphs)
        print(f"  OLS θ = {np.round(theta_ols, 4)}")

        rng0 = np.random.RandomState(args.seed)
        theta_init = (np.array(args.theta_init, dtype=float)
                      if args.theta_init is not None
                      else rng0.uniform(0.5, 3.5, size=2))
        print(f"  θ_init = {np.round(theta_init, 4)}")

        eps_abs = args.fy_epsilon
        print(f"  FY ε = {eps_abs}  (fixed absolute, noise ~ N(0, ε))")

        print(f"\nMSE training ({args.n_epochs} epochs) …")
        traj_mse = run_mse(graphs, theta_init, args.n_epochs, args.lr_mse)

        print(f"\nFY training ({args.n_epochs} epochs) …")
        traj_fy = run_fy(graphs, theta_init, args.n_epochs, args.lr_fy,
                         eps_abs, args.fy_M, args.seed, env)

        out_mse = os.path.join(args.out_dir, "trajectory_mse.npy")
        out_fy  = os.path.join(args.out_dir, "trajectory_fy.npy")
        np.save(out_mse, traj_mse)
        np.save(out_fy,  traj_fy)

        print(f"\nMSE trajectory saved → {out_mse}  shape={traj_mse.shape}")
        print(f"FY  trajectory saved → {out_fy}   shape={traj_fy.shape}")
        print(f"\nMSE final θ : {np.round(traj_mse[-1], 4)}")
        print(f"FY  final θ : {np.round(traj_fy[-1],  4)}")

    finally:
        env.dispose()


if __name__ == "__main__":
    main()
