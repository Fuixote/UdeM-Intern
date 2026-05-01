"""
可视化 MSE 和 FY 训练轨迹在 True Regret 等高线上的运动

读取:
    trajectory_mse_with_regret.npy  shape (n_epochs+1, 3), columns:
        theta_1, theta_2, true_regret
    trajectory_fy_with_fy_loss_and_regret.npy  shape (n_epochs+1, 4), columns:
        theta_1, theta_2, fy_loss, true_regret

输出:
    trajectory_contour.png
    true_regret_surface.npz
"""

import os
import sys
import glob
import random
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from linear_probe_landscape import (
    load_graph,
    CachedHybridKepModel,
    solve_once,
    feature_matrix,
    regret_landscape,
)

import gurobipy as gp

PROBE = {
    "feature_names": ["utility", "recipient_cPRA"],
    "feature_labels": [r"$\theta_1$ (utility weight)", r"$\theta_2$ (cPRA weight)"],
}
TRUE_THETA = np.array([10.0, 5.0])


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
        graphs.append({"graph": g, "X": X, "w_true": g["w_true"], "y_optimal": y_opt})
        if (i + 1) % 20 == 0:
            print(f"  loaded {i+1}/{len(chosen)}", flush=True)
    return graphs


# ─────────────────────────────────────────────────────────────
# 绘图
# ─────────────────────────────────────────────────────────────

def load_trajectory(path, name):
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"{name} trajectory must have shape (n, >=2), got {arr.shape}")
    traj = arr[:, :2]
    metrics = {}
    if name == "MSE":
        if arr.shape[1] >= 3:
            metrics["true_regret"] = arr[:, 2]
        else:
            print(f"Warning: {name} has no True Regret column: {path}")
    elif name == "FY":
        if arr.shape[1] >= 3:
            metrics["fy_loss"] = arr[:, 2]
        if arr.shape[1] >= 4:
            metrics["true_regret"] = arr[:, 3]
        else:
            print(f"Warning: {name} has no True Regret column: {path}")
    return traj, metrics, arr.shape


def resolve_path(path, base_dir):
    if os.path.isabs(path):
        return path
    if os.path.exists(path):
        return path
    return os.path.join(base_dir, path)


def draw_contour_with_traj(ax, fig, T1, T2, R, traj, metrics, cmap_name, title, xl, yl,
                           show_ratio_line=True, n_milestones=5):
    """在 True Regret 等高线上叠加单条轨迹。"""
    # 景观
    lo, hi = R.min(), R.min() + (R.max() - R.min()) * 0.6
    Rc = np.clip(R, lo, hi)
    cf = ax.contourf(T1, T2, Rc, levels=40, cmap="viridis")
    ax.contour(T1, T2, Rc, levels=12, colors="white", linewidths=0.3, alpha=0.4)
    cb = fig.colorbar(cf, ax=ax, shrink=0.82)
    cb.set_label("True Regret", fontsize=8)

    # 最优方向射线 θ_1/θ_2 = 2
    if show_ratio_line:
        t1_max = float(T1.max())
        t2_max = float(T2.max())
        # 射线：从原点出发，ratio=2，画到图的边界
        end_t1 = min(t1_max, t2_max * 2)
        end_t2 = end_t1 / 2
        ax.plot([0, end_t1], [0, end_t2], "--", color="white",
                lw=1.4, alpha=0.7, label=r"optimal ray ($\theta_1/\theta_2=2$)")

    # 轨迹：颜色由浅到深表示 epoch 进展
    cmap = plt.get_cmap(cmap_name)
    n = len(traj)
    for i in range(n - 1):
        c = cmap(0.2 + 0.8 * i / max(n - 2, 1))
        ax.plot(traj[i:i+2, 0], traj[i:i+2, 1], color=c, lw=1.6, zorder=5)

    # 里程碑标记（等间距 epoch）
    milestone_idx = np.linspace(0, n - 1, n_milestones + 2, dtype=int)[1:-1]
    for idx in milestone_idx:
        c = cmap(0.2 + 0.8 * idx / max(n - 2, 1))
        ax.plot(*traj[idx], "o", color=c, ms=5, zorder=6,
                markeredgecolor="white", markeredgewidth=0.8)
        ax.annotate(f"ep{idx}", traj[idx], fontsize=6, color="white",
                    xytext=(4, 4), textcoords="offset points")

    # 起点 / 终点
    ax.plot(*traj[0],  "o", color=cmap(0.2),  ms=10, zorder=8,
            markeredgecolor="white", markeredgewidth=1.5, label=f"start {np.round(traj[0], 2)}")
    final_label = f"final {np.round(traj[-1], 2)}"
    if "fy_loss" in metrics:
        final_label += f", FY loss={metrics['fy_loss'][-1]:.4g}"
    if "true_regret" in metrics:
        final_label += f", regret={metrics['true_regret'][-1]:.4g}"
    ax.plot(*traj[-1], "*", color=cmap(1.0),   ms=14, zorder=8,
            markeredgecolor="white", markeredgewidth=1.2,
            label=final_label)

    # 真实参数
    ax.plot(*TRUE_THETA, "D", color="lime", ms=10, zorder=9,
            markeredgecolor="black", markeredgewidth=1.2,
            label=r"$\theta^*=[10,5]$")

    ax.set_xlabel(xl, fontsize=9)
    ax.set_ylabel(yl, fontsize=9)
    ax.set_title(title, fontsize=12, pad=8)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.85)


def plot(traj_mse, metrics_mse, traj_fy, metrics_fy, T1, T2, R,
         n_epochs, out_path, n_milestones):
    xl, yl = PROBE["feature_labels"]
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 7))

    draw_contour_with_traj(
        ax_l, fig, T1, T2, R, traj_mse, metrics_mse, "Blues_r",
        f"MSE (2-stage)  —  {n_epochs} epochs", xl, yl,
        n_milestones=n_milestones,
    )
    draw_contour_with_traj(
        ax_r, fig, T1, T2, R, traj_fy, metrics_fy, "Oranges_r",
        f"FY (end-to-end)  —  {n_epochs} epochs", xl, yl,
        n_milestones=n_milestones,
    )

    fig.suptitle(
        r"True Regret landscape  ·  $\hat{w}_e = \theta_1 u_e + \theta_2 c_e$"
        f"  ·  init={np.round(traj_mse[0], 2)}",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    here = Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
        default=str(ROOT / "dataset/processed/clean_linear_dataset"))
    parser.add_argument("--traj_dir",   default=str(here))
    parser.add_argument("--mse_traj_path",
        default=str(here / "trajectory_mse_with_regret.npy"))
    parser.add_argument("--fy_traj_path",
        default=str(here / "trajectory_fy_with_fy_loss_and_regret.npy"))
    parser.add_argument("--out_dir",    default=str(here))
    parser.add_argument("--n_total",    type=int,   default=100)
    parser.add_argument("--grid_size",  type=int,   default=25)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--n_milestones", type=int, default=5,
        help="轨迹上标注的里程碑数量")
    args = parser.parse_args()

    # 读取轨迹
    mse_path = args.mse_traj_path
    fy_path = args.fy_traj_path
    mse_path = resolve_path(mse_path, args.traj_dir)
    fy_path = resolve_path(fy_path, args.traj_dir)

    traj_mse, metrics_mse, mse_shape = load_trajectory(mse_path, "MSE")
    traj_fy, metrics_fy, fy_shape = load_trajectory(fy_path, "FY")
    n_epochs = len(traj_mse) - 1
    print(f"Loaded trajectories: MSE raw {mse_shape}, FY raw {fy_shape}")
    if "true_regret" in metrics_mse:
        regret_mse = metrics_mse["true_regret"]
        print(
            "Loaded MSE True Regret column: "
            f"start={regret_mse[0]:.6f}, final={regret_mse[-1]:.6f}, "
            f"min={regret_mse.min():.6f}, max={regret_mse.max():.6f}"
        )
    if "fy_loss" in metrics_fy:
        fy_loss = metrics_fy["fy_loss"]
        print(
            "Loaded FY Loss column: "
            f"start={fy_loss[0]:.6f}, final={fy_loss[-1]:.6f}, "
            f"min={fy_loss.min():.6f}, max={fy_loss.max():.6f}"
        )
    if "true_regret" in metrics_fy:
        regret_fy = metrics_fy["true_regret"]
        print(
            "Loaded FY True Regret column: "
            f"start={regret_fy[0]:.6f}, final={regret_fy[-1]:.6f}, "
            f"min={regret_fy.min():.6f}, max={regret_fy.max():.6f}"
        )

    # 网格：覆盖两条轨迹 + 真实参数，留出边距
    all_pts = np.vstack([traj_mse, traj_fy, TRUE_THETA[None]])
    pad = 1.5
    t1g = np.linspace(max(0.0, all_pts[:, 0].min() - pad),
                      all_pts[:, 0].max() + pad, args.grid_size)
    t2g = np.linspace(max(0.0, all_pts[:, 1].min() - pad),
                      all_pts[:, 1].max() + pad, args.grid_size)
    print(f"Grid: θ_1 ∈ [{t1g[0]:.2f}, {t1g[-1]:.2f}]  "
          f"θ_2 ∈ [{t2g[0]:.2f}, {t2g[-1]:.2f}]")

    # Gurobi
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Seed", args.seed)
    env.start()

    try:
        print("Loading graphs …")
        graphs = load_graphs(args.data_dir, args.n_total, seed=args.seed, env=env)
        print(f"  total={len(graphs)}")

        print("Computing True Regret landscape …")
        T1, T2, R = regret_landscape(graphs, t1g, t2g, env)
        surface_path = os.path.join(args.out_dir, "true_regret_surface.npz")
        np.savez_compressed(
            surface_path,
            T1=T1,
            T2=T2,
            true_regret=R,
            theta_1_grid=t1g,
            theta_2_grid=t2g,
            n_total=args.n_total,
            seed=args.seed,
            grid_size=args.grid_size,
        )
        print(f"Saved: {surface_path}")

        out_path = os.path.join(args.out_dir, "trajectory_contour.png")
        plot(
            traj_mse, metrics_mse, traj_fy, metrics_fy,
            T1, T2, R, n_epochs, out_path, args.n_milestones
        )

    finally:
        env.dispose()


if __name__ == "__main__":
    main()
