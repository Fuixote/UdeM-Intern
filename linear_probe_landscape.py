"""
Linear Probe Landscape Experiment
==================================
Visualize MSE, true Regret, and Fenchel-Young loss landscapes over a 2D
parameter space (theta_1, theta_2) for the linear model:

    w_hat_e = theta_1 * utility_e + theta_2 * cPRA_e

This demonstrates that the MSE-optimal parameters differ from the
decision-optimal parameters, even when model capacity is identical.
"""

import argparse
import os
import sys
import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.graph_utils import parse_json_to_dfl_data
from formulations.hybrid.backend import solve_cf_cycle_pief_chain
from formulations.common.backend_utils import infer_ndd_mask


# ─────────────────────────────────────────────
# Graph loading & feature extraction
# ─────────────────────────────────────────────

def load_graph(graph_path, max_cycle=3, max_chain=4):
    """Load a single graph and return all needed components."""
    data = parse_json_to_dfl_data(graph_path, max_cycle=max_cycle, max_chain=max_chain, label_scale=1.0)
    # label_scale=1.0 → data.y is in original scale (not divided by 25)

    src, dst = data.edge_index
    utility = data.edge_attr[:, 0].numpy()        # shape [E]
    cpra = data.x[dst, 1].numpy()                 # recipient cPRA, shape [E]
    w_true = data.y.numpy()                        # ground truth weights, shape [E]

    X = np.stack([utility, cpra], axis=1)          # [E, 2]

    cycle_candidates = [c for c in data.candidates if c["type"] == "cycle"]
    node_is_ndd = infer_ndd_mask(data.x)
    num_nodes = data.num_nodes_custom[0].item()

    return {
        "X": X,
        "w_true": w_true,
        "edge_index": data.edge_index,
        "cycle_candidates": cycle_candidates,
        "node_is_ndd": node_is_ndd,
        "num_nodes": num_nodes,
        "num_edges": data.num_edges,
        "filename": data.filename,
    }


# ─────────────────────────────────────────────
# Solver wrapper
# ─────────────────────────────────────────────

def solve_once(w_hat, graph, env):
    """Solve KEP with predicted weights, return edge selection vector."""
    result = solve_cf_cycle_pief_chain(
        weights=w_hat,
        edge_index=graph["edge_index"],
        is_ndd_mask=graph["node_is_ndd"],
        num_nodes=graph["num_nodes"],
        cycle_candidates=graph["cycle_candidates"],
        env=env,
    )
    return result["edge_selection"]


def compute_regret(y_pred, y_optimal, w_true):
    """Regret = optimal_obj - achieved_obj under true weights."""
    achieved = np.dot(w_true, y_pred)
    optimal = np.dot(w_true, y_optimal)
    return optimal - achieved


# ─────────────────────────────────────────────
# Landscape computations
# ─────────────────────────────────────────────

def mse_landscape(graphs_data, theta1_grid, theta2_grid):
    """Compute MSE(theta) averaged over all graphs."""
    T1, T2 = np.meshgrid(theta1_grid, theta2_grid)
    mse = np.zeros_like(T1)
    for gd in graphs_data:
        X, w_true = gd["X"], gd["w_true"]
        for i in range(T1.shape[0]):
            for j in range(T1.shape[1]):
                w_hat = X[:, 0] * T1[i, j] + X[:, 1] * T2[i, j]
                mse[i, j] += np.mean((w_hat - w_true) ** 2)
    mse /= len(graphs_data)
    return T1, T2, mse


def regret_landscape(graphs_data, theta1_grid, theta2_grid, env):
    """Compute average true Regret(theta) over all graphs."""
    T1, T2 = np.meshgrid(theta1_grid, theta2_grid)
    regret = np.zeros_like(T1)
    total = T1.size
    for idx in range(total):
        i, j = divmod(idx, T1.shape[1])
        for gd in graphs_data:
            w_hat = gd["X"][:, 0] * T1[i, j] + gd["X"][:, 1] * T2[i, j]
            y_pred = solve_once(w_hat, gd["graph"], env)
            regret[i, j] += compute_regret(y_pred, gd["y_optimal"], gd["w_true"])
        if (idx + 1) % 50 == 0 or idx + 1 == total:
            print(f"  Regret grid: {idx + 1}/{total}", flush=True)
    regret /= len(graphs_data)
    return T1, T2, regret


def smoothed_regret_landscape(graphs_data, theta1_grid, theta2_grid, env,
                              epsilon=0.2, M=8, theta_mse=None):
    """Compute smoothed regret: the true regret of the FY-expected solution.

    At each grid point, solve M perturbed ILPs, average their edge
    selections, then evaluate regret under true weights.  This is
    equivalent to the regret landscape convolved with the perturbation
    kernel, which is what FY loss actually smooths in practice.

    epsilon is applied at a fixed absolute scale calibrated to the
    typical weight magnitude at the MSE optimum, so smoothing is
    consistent across the grid.
    """
    # Calibrate epsilon to the MSE-optimal weight scale
    if theta_mse is not None:
        ref_stds = []
        for gd in graphs_data:
            w_ref = gd["X"][:, 0] * theta_mse[0] + gd["X"][:, 1] * theta_mse[1]
            ref_stds.append(np.std(w_ref))
        eps_abs = epsilon * np.mean(ref_stds)
    else:
        eps_abs = epsilon

    T1, T2 = np.meshgrid(theta1_grid, theta2_grid)
    sm_regret = np.zeros_like(T1)
    total = T1.size
    rng = np.random.RandomState(42)

    for idx in range(total):
        i, j = divmod(idx, T1.shape[1])
        for gd in graphs_data:
            X, w_true = gd["X"], gd["w_true"]
            w_hat = X[:, 0] * T1[i, j] + X[:, 1] * T2[i, j]
            y_optimal = gd["y_optimal"]

            y_soft = np.zeros_like(w_hat)
            for _ in range(M):
                z = rng.normal(0, eps_abs, size=w_hat.shape).astype(np.float32)
                y_m = solve_once(w_hat + z, gd["graph"], env)
                y_soft += y_m
            y_soft /= M

            sm_regret[i, j] += compute_regret(y_soft, y_optimal, w_true)

        if (idx + 1) % 25 == 0 or idx + 1 == total:
            print(f"  Smoothed regret grid: {idx + 1}/{total}", flush=True)

    sm_regret /= len(graphs_data)
    return T1, T2, sm_regret


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────

def plot_landscapes(mse_data, regret_data, fy_data, markers, output_path,
                    regret_at_markers=None, num_graphs=0):
    """Plot 1x3 contour subplots with marked optima."""
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    titles = [
        r"$\mathcal{L}_{\mathrm{MSE}}(\theta)$",
        r"$\mathcal{R}(\theta)$ (True Regret)",
        r"$\tilde{\mathcal{R}}_\varepsilon(\theta)$ (Smoothed Regret)",
    ]
    datasets = [mse_data, regret_data, fy_data]

    for ax, (T1, T2, Z), title in zip(axes, datasets, titles):
        z_min = Z.min()
        z_range = Z.max() - z_min
        # Use tighter clip to show fine structure near minima
        clip_hi = z_min + z_range * 0.3
        Z_clipped = np.clip(Z, z_min, clip_hi)
        levels = np.linspace(z_min, clip_hi, 35)
        cf = ax.contourf(T1, T2, Z_clipped, levels=levels, cmap="viridis", extend="max")
        ax.contour(T1, T2, Z_clipped, levels=12, colors="white", linewidths=0.4, alpha=0.6)
        fig.colorbar(cf, ax=ax, shrink=0.8)

        for k, (name, theta, color, marker_shape) in enumerate(markers):
            ax.plot(theta[0], theta[1], marker=marker_shape, color=color,
                    markersize=14, markeredgecolor="black", markeredgewidth=1.5,
                    label=name, zorder=10)
            # Add regret annotation on the regret panel
            if regret_at_markers is not None and title.startswith(r"$\mathcal{R}"):
                offset = [(15, 15), (-15, -20), (15, -20)][k % 3]
                ax.annotate(
                    f"R={regret_at_markers[k]:.2f}",
                    xy=(theta[0], theta[1]),
                    xytext=offset,
                    textcoords="offset points",
                    fontsize=9, fontweight="bold", color=color,
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.7),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
                )

        ax.set_xlabel(r"$\theta_1$ (utility weight)", fontsize=12)
        ax.set_ylabel(r"$\theta_2$ (cPRA weight)", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=9, loc="upper right",
                  framealpha=0.9, edgecolor="gray")

    fig.suptitle(
        r"Loss Landscapes: $\hat{w}_e = \theta_1 \cdot u_e + \theta_2 \cdot c_e$ (no bias)"
        + (f"\naveraged over {num_graphs} graphs" if num_graphs else ""),
        fontsize=15, y=1.03,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved landscape plot to: {output_path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def select_multiple_graphs(data_dir, test_files_path=None, n=10, percentile=10):
    """Select n graphs around the given edge-count percentile."""
    if test_files_path and os.path.isfile(test_files_path):
        with open(test_files_path) as f:
            filenames = [line.strip() for line in f if line.strip()]
        paths = [os.path.join(data_dir, fn) for fn in filenames]
        paths = [p for p in paths if os.path.isfile(p)]
    else:
        paths = sorted(glob.glob(os.path.join(data_dir, "G-*.json")))

    if not paths:
        raise FileNotFoundError(f"No graph files found in {data_dir}")

    edge_counts = []
    for p in paths:
        with open(p) as f:
            content = json.load(f)
        n_edges = sum(len(node.get("matches", [])) for node in content["data"].values())
        edge_counts.append((n_edges, p))

    edge_counts.sort(key=lambda x: x[0])
    target_idx = max(0, min(len(edge_counts) - 1, int(len(edge_counts) * percentile / 100)))
    # Take n graphs centered around the target percentile
    start = max(0, target_idx - n // 2)
    end = min(len(edge_counts), start + n)
    start = max(0, end - n)
    selected = edge_counts[start:end]

    for ne, p in selected:
        print(f"  Selected: {os.path.basename(p)} ({ne} edges)")
    return [p for _, p in selected]


def select_representative_graph(data_dir, test_files_path=None, percentile=10):
    """Pick a graph at the given edge-count percentile from the test set.

    A smaller graph (low percentile) produces a regret landscape with more
    visible transitions because each edge matters more to the ILP solution.
    """
    if test_files_path and os.path.isfile(test_files_path):
        with open(test_files_path) as f:
            filenames = [line.strip() for line in f if line.strip()]
        paths = [os.path.join(data_dir, fn) for fn in filenames]
        paths = [p for p in paths if os.path.isfile(p)]
    else:
        paths = sorted(glob.glob(os.path.join(data_dir, "G-*.json")))

    if not paths:
        raise FileNotFoundError(f"No graph files found in {data_dir}")

    # Estimate edge counts quickly
    edge_counts = []
    for p in paths:
        with open(p) as f:
            content = json.load(f)
        n_edges = sum(len(node.get("matches", [])) for node in content["data"].values())
        edge_counts.append((n_edges, p))

    edge_counts.sort(key=lambda x: x[0])
    target_idx = max(0, min(len(edge_counts) - 1, int(len(edge_counts) * percentile / 100)))
    chosen = edge_counts[target_idx]
    print(f"Selected graph: {os.path.basename(chosen[1])} ({chosen[0]} edges, "
          f"p{percentile} of {len(paths)} graphs)")
    return chosen[1]


def main():
    parser = argparse.ArgumentParser(description="Linear Probe Landscape Experiment")
    parser.add_argument("--graph_path", type=str, default=None,
                        help="Path to a specific G-*.json graph file")
    parser.add_argument("--data_dir", type=str,
                        default="dataset/processed/2026-04-17_135607",
                        help="Processed data directory (used to auto-select a graph)")
    parser.add_argument("--test_files", type=str, default=None,
                        help="Path to test_files.txt to select from test set only")
    parser.add_argument("--grid_size", type=int, default=25,
                        help="Grid resolution for regret and FY landscapes")
    parser.add_argument("--mse_grid_size", type=int, default=100,
                        help="Grid resolution for MSE landscape")
    parser.add_argument("--fy_M", type=int, default=8,
                        help="Number of perturbation samples for FY loss")
    parser.add_argument("--fy_epsilon", type=float, default=0.2,
                        help="Perturbation scale for FY loss")
    parser.add_argument("--output", type=str, default="results/linear_probe_landscape.png",
                        help="Output image path")
    parser.add_argument("--margin", type=float, default=0.5,
                        help="Grid margin factor around MSE optimum")
    parser.add_argument("--percentile", type=int, default=10,
                        help="Edge-count percentile for graph selection (lower = smaller graph)")
    parser.add_argument("--num_graphs", type=int, default=10,
                        help="Number of graphs to average over")
    args = parser.parse_args()

    # 1. Load graphs
    if args.graph_path:
        graph_paths = [args.graph_path]
    else:
        test_files = args.test_files
        if test_files is None:
            candidate = "results/2stg_LR_2026-04-17_144841/test_files.txt"
            if os.path.isfile(candidate):
                test_files = candidate
        graph_paths = select_multiple_graphs(args.data_dir, test_files,
                                             n=args.num_graphs,
                                             percentile=args.percentile)

    graphs_data = []
    all_X = []
    all_w = []
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Seed", 42)
    env.start()

    try:
        for gp_path in graph_paths:
            g = load_graph(gp_path)
            y_opt = solve_once(g["w_true"], g, env)
            graphs_data.append({
                "X": g["X"],
                "w_true": g["w_true"],
                "graph": g,
                "y_optimal": y_opt,
            })
            all_X.append(g["X"])
            all_w.append(g["w_true"])

        print(f"Loaded {len(graphs_data)} graphs")

        # 2. Pooled MSE optimum (no bias)
        X_pool = np.concatenate(all_X, axis=0)
        w_pool = np.concatenate(all_w, axis=0)
        theta_mse = np.linalg.lstsq(X_pool, w_pool, rcond=None)[0]
        print(f"Pooled MSE optimum (OLS, no bias): theta_1={theta_mse[0]:.4f}, theta_2={theta_mse[1]:.4f}")

        # 3. Determine grid range
        # Symmetric around origin-to-MSE direction, wide enough to see
        # angular regret transitions.  The regret landscape is constant
        # along rays (positive scaling doesn't change ILP solution), so
        # the interesting variation is *angular*, not radial.
        t1_center, t2_center = theta_mse
        radius = np.sqrt(t1_center ** 2 + t2_center ** 2)
        pad = radius * args.margin

        t1_lo = min(-pad, t1_center - radius * 0.5)
        t1_hi = t1_center + pad
        t2_lo = min(t2_center - pad, -pad)
        t2_hi = max(pad, t2_center + radius * 0.5)

        print(f"Grid range: theta_1 in [{t1_lo:.2f}, {t1_hi:.2f}], theta_2 in [{t2_lo:.2f}, {t2_hi:.2f}]")

        # 4. Compute landscapes
        print("\nComputing MSE landscape...")
        mse_t1 = np.linspace(t1_lo, t1_hi, args.mse_grid_size)
        mse_t2 = np.linspace(t2_lo, t2_hi, args.mse_grid_size)
        mse_data = mse_landscape(graphs_data, mse_t1, mse_t2)

        coarse_t1 = np.linspace(t1_lo, t1_hi, args.grid_size)
        coarse_t2 = np.linspace(t2_lo, t2_hi, args.grid_size)

        print(f"Computing Regret landscape ({args.grid_size}x{args.grid_size} x {len(graphs_data)} graphs)...")
        regret_data = regret_landscape(graphs_data, coarse_t1, coarse_t2, env)

        print(f"Computing smoothed regret landscape ({args.grid_size}x{args.grid_size} x {len(graphs_data)} graphs x M={args.fy_M})...")
        fy_data = smoothed_regret_landscape(graphs_data, coarse_t1, coarse_t2, env,
                                            epsilon=args.fy_epsilon, M=args.fy_M,
                                            theta_mse=theta_mse)

        # 5. Find grid-optimal points
        T1_r, T2_r, R = regret_data
        min_idx_r = np.unravel_index(np.argmin(R), R.shape)
        theta_oracle = np.array([T1_r[min_idx_r], T2_r[min_idx_r]])

        T1_f, T2_f, F = fy_data
        min_idx_f = np.unravel_index(np.argmin(F), F.shape)
        theta_fy = np.array([T1_f[min_idx_f], T2_f[min_idx_f]])

        # Regret at each marked point (nearest grid point)
        def regret_at(theta):
            ii = np.argmin(np.abs(coarse_t2 - theta[1]))
            jj = np.argmin(np.abs(coarse_t1 - theta[0]))
            return R[ii, jj]

        r_mse = regret_at(theta_mse)
        r_oracle = R[min_idx_r]
        r_fy = regret_at(theta_fy)

        print(f"\nResults:")
        print(f"  theta*_2s   (MSE min)     : ({theta_mse[0]:.4f}, {theta_mse[1]:.4f})")
        print(f"  theta*_oracle (Regret min) : ({theta_oracle[0]:.4f}, {theta_oracle[1]:.4f})")
        print(f"  theta*_e2e  (FY min)       : ({theta_fy[0]:.4f}, {theta_fy[1]:.4f})")
        print(f"  Avg Regret at theta*_2s    : {r_mse:.4f}")
        print(f"  Avg Regret at theta*_oracle: {r_oracle:.4f}")
        print(f"  Avg Regret at theta*_e2e   : {r_fy:.4f}")
        print(f"  Regret range on grid       : [{R.min():.4f}, {R.max():.4f}]")

        # 6. Plot
        markers = [
            (r"$\theta^*_{\mathrm{2s}}$ (MSE)", theta_mse, "red", "*"),
            (r"$\theta^*_{\mathrm{oracle}}$ (Regret)", theta_oracle, "lime", "D"),
            (r"$\theta^*_{\mathrm{e2e}}$ (Smoothed)", theta_fy, "cyan", "o"),
        ]
        regret_at_markers = [r_mse, r_oracle, r_fy]

        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        plot_landscapes(mse_data, regret_data, fy_data, markers, args.output,
                        regret_at_markers=regret_at_markers,
                        num_graphs=len(graphs_data))

    finally:
        env.dispose()


if __name__ == "__main__":
    main()
