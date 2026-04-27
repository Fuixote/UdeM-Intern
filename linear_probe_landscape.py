"""
Linear Probe Landscape Experiment
==================================
Visualize true Regret and Fenchel-Young surrogate loss landscapes over a 2D
slice of a 3D parameter space (theta_1, theta_2, theta_3) for the linear model:

    w_hat_e = theta_1 * utility_e
            + theta_2 * recipient_cPRA_e
            + theta_3 * source_donor_age_e

This demonstrates that the FY-optimal parameters differ from the
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

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.graph_utils import parse_json_to_dfl_data
from formulations.hybrid.backend import solve_cf_cycle_pief_chain
from formulations.common.backend_utils import infer_ndd_mask

FEATURE_NAMES = [
    "utility",
    "recipient_cPRA",
    "source_donor_age",
]

FEATURE_LABELS = [
    r"$\theta_1$ (utility weight)",
    r"$\theta_2$ (recipient cPRA weight)",
    r"$\theta_3$ (source donor age weight)",
]

PAIRWISE_SLICES = [
    (0, 1),
    (0, 2),
    (1, 2),
]


# ─────────────────────────────────────────────
# Graph loading & feature extraction
# ─────────────────────────────────────────────

def load_graph(graph_path, max_cycle=3, max_chain=4):
    """Load a single graph and return all needed components."""
    data = parse_json_to_dfl_data(graph_path, max_cycle=max_cycle, max_chain=max_chain, label_scale=1.0)
    # label_scale=1.0 → data.y is in original scale (not divided by 25)

    src, dst = data.edge_index
    utility = data.edge_attr[:, 0].numpy()             # shape [E]
    recipient_cpra = data.x[dst, 1].numpy()            # recipient cPRA, shape [E]
    source_donor_age = data.x[src, 7].numpy()          # source donor age / 100, shape [E]
    w_true = data.y.numpy()                            # ground truth weights, shape [E]

    X = np.stack([utility, recipient_cpra, source_donor_age], axis=1)  # [E, 3]

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

def slice_theta(theta_base, dim_x, dim_y, x_value, y_value):
    """Return a 3D theta with two slice dimensions overwritten."""
    theta = np.array(theta_base, dtype=float, copy=True)
    theta[dim_x] = x_value
    theta[dim_y] = y_value
    return theta


def make_antithetic_perturbations(rng, M, num_edges, eps_abs):
    """Create fixed perturbations with paired +/- samples to reduce MC noise."""
    if M <= 0:
        raise ValueError("M must be positive for FY loss")
    half = M // 2
    samples = rng.normal(0, eps_abs, size=(half, num_edges)).astype(np.float32)
    perturbations = [samples, -samples]
    if M % 2:
        perturbations.append(np.zeros((1, num_edges), dtype=np.float32))
    return np.concatenate(perturbations, axis=0)


def regret_landscape(graphs_data, theta_x_grid, theta_y_grid, env,
                     theta_base, dims):
    """Compute average true Regret(theta) over a 2D slice of 3D theta."""
    dim_x, dim_y = dims
    T1, T2 = np.meshgrid(theta_x_grid, theta_y_grid)
    regret = np.zeros_like(T1)
    total = T1.size
    for idx in range(total):
        i, j = divmod(idx, T1.shape[1])
        theta = slice_theta(theta_base, dim_x, dim_y, T1[i, j], T2[i, j])
        for gd in graphs_data:
            w_hat = gd["X"] @ theta
            y_pred = solve_once(w_hat, gd["graph"], env)
            regret[i, j] += compute_regret(y_pred, gd["y_optimal"], gd["w_true"])
        if (idx + 1) % 50 == 0 or idx + 1 == total:
            print(f"  Regret grid: {idx + 1}/{total}", flush=True)
    regret /= len(graphs_data)
    return T1, T2, regret


def fy_loss_landscape(graphs_data, theta_x_grid, theta_y_grid, env,
                      theta_base, dims, epsilon=0.2, M=8, theta_mse=None):
    """Compute the Fenchel-Young surrogate loss landscape over a 2D slice.

    At each grid point, approximates the perturbed FY loss:

        L_FY(theta) ≈ (1/M) sum_m [
            max_y <w_hat(theta) + z_m, y> - <w_hat(theta), y*>
        ]

    where y_perturbed_m = argmax_{y in Y} (w_hat + epsilon * z_m)^T y,
    z_m ~ N(0, I), and y* is the true-weight optimal solution. The omitted
    conjugate term is constant with respect to theta, so it does not affect
    the landscape minimizer. With common perturbation samples this finite-M
    estimate is convex in theta; the perturbation expectation gives the
    smooth FY surface.
    """
    # Calibrate epsilon to the MSE-optimal weight scale
    if theta_mse is not None:
        ref_stds = []
        for gd in graphs_data:
            w_ref = gd["X"] @ theta_mse
            ref_stds.append(np.std(w_ref))
        eps_abs = epsilon * np.mean(ref_stds)
    else:
        eps_abs = epsilon

    dim_x, dim_y = dims
    T1, T2 = np.meshgrid(theta_x_grid, theta_y_grid)
    fy_loss = np.zeros_like(T1)
    total = T1.size
    rng = np.random.RandomState(42)
    perturbations_by_graph = [
        make_antithetic_perturbations(rng, M, gd["w_true"].shape[0], eps_abs)
        for gd in graphs_data
    ]

    for idx in range(total):
        i, j = divmod(idx, T1.shape[1])
        theta = slice_theta(theta_base, dim_x, dim_y, T1[i, j], T2[i, j])
        for gd, perturbations in zip(graphs_data, perturbations_by_graph):
            X = gd["X"]
            w_hat = X @ theta
            y_optimal = gd["y_optimal"]

            # FY loss, up to a theta-independent constant:
            # E_z[max_y <w_hat + z, y>] - <w_hat, y*>.
            loss_sum = 0.0
            target_score = np.dot(w_hat, y_optimal)
            for z in perturbations:
                perturbed_weights = w_hat + z
                y_perturbed = solve_once(perturbed_weights, gd["graph"], env)
                loss_sum += np.dot(perturbed_weights, y_perturbed) - target_score
            fy_loss[i, j] += loss_sum / M

        if (idx + 1) % 25 == 0 or idx + 1 == total:
            print(f"  FY loss grid: {idx + 1}/{total}", flush=True)

    fy_loss /= len(graphs_data)
    return T1, T2, fy_loss


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────

def clipped_surface(data):
    """Return clipped surface data and clipping bounds for plotting."""
    T1, T2, Z = data
    z_min = Z.min()
    z_range = Z.max() - z_min
    clip_hi = z_min + z_range * 0.3 if z_range > 0 else z_min + 1.0
    Z_clipped = np.clip(Z, z_min, clip_hi)
    return T1, T2, Z_clipped, z_min, clip_hi


def clipped_contour(ax, fig, data, title, markers, x_label, y_label,
                    regret_at_markers=None):
    """Draw one clipped contour panel with shared marker styling."""
    T1, T2, Z_clipped, z_min, clip_hi = clipped_surface(data)
    levels = np.linspace(z_min, clip_hi, 35)
    cf = ax.contourf(T1, T2, Z_clipped, levels=levels, cmap="viridis", extend="max")
    ax.contour(T1, T2, Z_clipped, levels=12, colors="white", linewidths=0.4, alpha=0.6)
    fig.colorbar(cf, ax=ax, shrink=0.82)

    for k, (name, theta_xy, color, marker_shape) in enumerate(markers):
        ax.plot(theta_xy[0], theta_xy[1], marker=marker_shape, color=color,
                markersize=12, markeredgecolor="black", markeredgewidth=1.3,
                label=name, zorder=10)
        if regret_at_markers is not None:
            offset = [(12, 12), (-16, -20), (12, -20)][k % 3]
            ax.annotate(
                f"R={regret_at_markers[k]:.2f}",
                xy=(theta_xy[0], theta_xy[1]),
                xytext=offset,
                textcoords="offset points",
                fontsize=8, fontweight="bold", color=color,
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.7),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
            )

    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_title(title, fontsize=12, pad=8)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9, edgecolor="gray")


def clipped_3d_surface(ax, fig, data, title, markers, x_label, y_label):
    """Draw one clipped 3D surface panel with marker stems."""
    T1, T2, Z_clipped, z_min, clip_hi = clipped_surface(data)
    surf = ax.plot_surface(
        T1, T2, Z_clipped,
        cmap="viridis",
        linewidth=0,
        antialiased=True,
        alpha=0.85,
    )
    fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.08)

    for name, theta_xy, color, marker_shape in markers:
        ii = np.argmin(np.abs(T2[:, 0] - theta_xy[1]))
        jj = np.argmin(np.abs(T1[0, :] - theta_xy[0]))
        z_surface = Z_clipped[ii, jj]
        z_top = clip_hi + (clip_hi - z_min) * 0.08
        ax.plot(
            [theta_xy[0], theta_xy[0]],
            [theta_xy[1], theta_xy[1]],
            [z_surface, z_top],
            color=color,
            linewidth=2,
            zorder=5,
        )
        ax.scatter(
            theta_xy[0], theta_xy[1], z_top,
            marker=marker_shape,
            color=color,
            s=110,
            edgecolors="black",
            linewidths=1.1,
            zorder=6,
            label=name,
        )

    ax.set_xlabel(x_label, fontsize=9, labelpad=6)
    ax.set_ylabel(y_label, fontsize=9, labelpad=6)
    ax.set_zlabel("Loss (clipped)", fontsize=9, labelpad=4)
    ax.set_title(title, fontsize=11, pad=10)
    ax.view_init(elev=28, azim=-50)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.85)


def plot_slice_landscape(result, output_path, num_graphs=0, graph_label=None):
    """Plot one feature-pair slice as 2x2: contours on top, 3D surfaces below."""
    fig = plt.figure(figsize=(16, 14))
    axes = [
        fig.add_subplot(2, 2, 1),
        fig.add_subplot(2, 2, 2),
        fig.add_subplot(2, 2, 3, projection="3d"),
        fig.add_subplot(2, 2, 4, projection="3d"),
    ]

    dim_x, dim_y = result["dims"]
    fixed_dim = result["fixed_dim"]
    fixed_value = result["theta_base"][fixed_dim]
    markers = [
        (r"$\theta^*_{\mathrm{2s}}$  (MSE/OLS)", result["theta_mse_xy"], "red", "*"),
        (r"$\theta^*_{\mathrm{oracle}}$ (slice Regret min)", result["theta_oracle_xy"], "lime", "D"),
        (r"$\theta^*_{\mathrm{e2e}}$  (slice FY min)", result["theta_fy_xy"], "cyan", "o"),
    ]
    pair_title = (
        f"{FEATURE_NAMES[dim_x]} vs {FEATURE_NAMES[dim_y]} "
        f"(fixed {FEATURE_NAMES[fixed_dim]}={fixed_value:.4f})"
    )

    clipped_contour(
        axes[0],
        fig,
        result["regret_data"],
        "True Regret — contour",
        markers,
        FEATURE_LABELS[dim_x],
        FEATURE_LABELS[dim_y],
        regret_at_markers=result["regret_at_markers"],
    )
    clipped_contour(
        axes[1],
        fig,
        result["fy_data"],
        "Fenchel-Young Surrogate — contour",
        markers,
        FEATURE_LABELS[dim_x],
        FEATURE_LABELS[dim_y],
    )
    clipped_3d_surface(
        axes[2],
        fig,
        result["regret_data"],
        "True Regret — 3D surface",
        markers,
        FEATURE_LABELS[dim_x],
        FEATURE_LABELS[dim_y],
    )
    clipped_3d_surface(
        axes[3],
        fig,
        result["fy_data"],
        "Fenchel-Young Surrogate — 3D surface",
        markers,
        FEATURE_LABELS[dim_x],
        FEATURE_LABELS[dim_y],
    )

    fig.suptitle(
        pair_title + "\n"
        r"$\hat{w}_e = \theta_1 u_e + \theta_2 c_e + \theta_3 a_e$"
        + (f"\n{graph_label}" if graph_label else (f"\naveraged over {num_graphs} graphs" if num_graphs else "")),
        fontsize=15,
        y=1.005,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved landscape plot to: {output_path}")
    plt.close(fig)


def output_path_for_slice(base_output, dims):
    """Append the feature-pair name before the output extension."""
    base = Path(base_output)
    suffix = base.suffix or ".png"
    stem = base.stem if base.suffix else base.name
    pair_name = f"{FEATURE_NAMES[dims[0]]}_{FEATURE_NAMES[dims[1]]}"
    return str(base.with_name(f"{stem}_{pair_name}{suffix}"))


def epsilon_dir_name(epsilon):
    """Return a stable directory name such as epsilon=0.2."""
    return f"epsilon={float(epsilon):.6g}"


def output_base_for_epsilon(output_name, epsilon, plot_root="plot_results"):
    """Place this run's plots under plot_results/epsilon=<value>/."""
    base = Path(output_name)
    filename = base.name or "linear_probe_landscape.png"
    return Path(plot_root) / epsilon_dir_name(epsilon) / filename


# ─────────────────────────────────────────────
# Graph selection helpers
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
    start = max(0, target_idx - n // 2)
    end = min(len(edge_counts), start + n)
    start = max(0, end - n)
    selected = edge_counts[start:end]

    for ne, p in selected:
        print(f"  Selected: {os.path.basename(p)} ({ne} edges)")
    return [p for _, p in selected]


def select_representative_graph(data_dir, test_files_path=None, percentile=10):
    """Pick a graph at the given edge-count percentile from the test set."""
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
    chosen = edge_counts[target_idx]
    print(f"Selected graph: {os.path.basename(chosen[1])} ({chosen[0]} edges, "
          f"p{percentile} of {len(paths)} graphs)")
    return chosen[1]


def select_graph_by_id(data_dir, graph_id, test_files_path=None):
    """Select one graph by numeric id, e.g. graph_id=42 -> G-42.json."""
    filename = f"G-{int(graph_id)}.json"
    path = os.path.join(data_dir, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Requested graph not found: {path}")

    if test_files_path and os.path.isfile(test_files_path):
        with open(test_files_path) as f:
            allowed = {line.strip() for line in f if line.strip()}
        if filename not in allowed:
            print(f"Warning: {filename} is not listed in {test_files_path}; using it anyway.")

    with open(path) as f:
        content = json.load(f)
    n_edges = sum(len(node.get("matches", [])) for node in content["data"].values())
    print(f"Selected graph: {filename} ({n_edges} edges)")
    return path


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

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
                        help="Grid resolution for Regret and FY landscapes")
    parser.add_argument("--fy_M", type=int, default=8,
                        help="Number of perturbation samples for FY loss")
    parser.add_argument("--fy_epsilon", type=float, default=0.2,
                        help="Perturbation scale for FY loss (relative to weight std)")
    parser.add_argument("--output", type=str, default="linear_probe_landscape.png",
                        help="Output image filename; saved under plot_results/epsilon=<fy_epsilon>/")
    parser.add_argument("--plot_root", type=str, default="plot_results",
                        help="Root directory for epsilon-grouped landscape plots")
    parser.add_argument("--margin", type=float, default=0.5,
                        help="Grid half-width factor around the 3D MSE optimum")
    parser.add_argument("--percentile", type=int, default=10,
                        help="Edge-count percentile for graph selection (lower = smaller graph)")
    parser.add_argument("--graph_id", type=int, default=42,
                        help="Graph id to use by default, e.g. 42 selects G-42.json")
    parser.add_argument("--num_graphs", type=int, default=1,
                        help="Number of graphs to average over. Use 1 for the selected single graph.")
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
        if args.num_graphs <= 1:
            graph_paths = [select_graph_by_id(args.data_dir, args.graph_id, test_files)]
        else:
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
        graph_label = None
        if len(graphs_data) == 1:
            graph_label = f"single graph: {graphs_data[0]['graph']['filename']}"

        # 2. 3-feature OLS solution — used for grid anchoring and theta*_2s marker
        X_pool = np.concatenate(all_X, axis=0)
        w_pool = np.concatenate(all_w, axis=0)
        theta_mse = np.linalg.lstsq(X_pool, w_pool, rcond=None)[0]
        print("OLS anchor (theta*_2s):")
        for name, value in zip(FEATURE_NAMES, theta_mse):
            print(f"  {name}: {value:.4f}")

        # 3. Determine per-parameter grid ranges centered on the OLS solution
        radius = np.linalg.norm(theta_mse)
        pad = max(radius * args.margin, 1e-6)
        theta_grids = []
        for k, center in enumerate(theta_mse):
            lo = center - pad
            hi = center + pad
            theta_grids.append(np.linspace(lo, hi, args.grid_size))
            print(f"Grid range: {FEATURE_NAMES[k]} theta_{k + 1} in [{lo:.2f}, {hi:.2f}]")

        # 4. Compute pairwise 2D slices through the 3D parameter space
        slice_results = []
        for dims in PAIRWISE_SLICES:
            dim_x, dim_y = dims
            fixed_dim = next(dim for dim in range(len(FEATURE_NAMES)) if dim not in dims)

            theta_x_grid = theta_grids[dim_x]
            theta_y_grid = theta_grids[dim_y]

            print(
                f"\nSlice: {FEATURE_NAMES[dim_x]} vs {FEATURE_NAMES[dim_y]} "
                f"(fixed {FEATURE_NAMES[fixed_dim]}={theta_mse[fixed_dim]:.4f})"
            )
            print(f"Computing True Regret landscape "
                  f"({args.grid_size}x{args.grid_size} x {len(graphs_data)} graphs)...")
            regret_data = regret_landscape(
                graphs_data, theta_x_grid, theta_y_grid, env,
                theta_base=theta_mse, dims=dims,
            )

            print(f"Computing FY Loss landscape "
                  f"({args.grid_size}x{args.grid_size} x {len(graphs_data)} graphs x M={args.fy_M})...")
            fy_data = fy_loss_landscape(
                graphs_data, theta_x_grid, theta_y_grid, env,
                theta_base=theta_mse, dims=dims,
                epsilon=args.fy_epsilon, M=args.fy_M,
                theta_mse=theta_mse,
            )

            # 5. Find slice-optimal points
            T1_r, T2_r, R = regret_data
            min_idx_r = np.unravel_index(np.argmin(R), R.shape)
            theta_oracle = np.array(theta_mse, copy=True)
            theta_oracle[dim_x] = T1_r[min_idx_r]
            theta_oracle[dim_y] = T2_r[min_idx_r]

            T1_f, T2_f, F = fy_data
            min_idx_f = np.unravel_index(np.argmin(F), F.shape)
            theta_fy = np.array(theta_mse, copy=True)
            theta_fy[dim_x] = T1_f[min_idx_f]
            theta_fy[dim_y] = T2_f[min_idx_f]

            # Regret at each marked point (nearest grid point in this slice)
            def regret_at(theta):
                ii = np.argmin(np.abs(theta_y_grid - theta[dim_y]))
                jj = np.argmin(np.abs(theta_x_grid - theta[dim_x]))
                return R[ii, jj]

            r_mse = regret_at(theta_mse)
            r_oracle = R[min_idx_r]
            r_fy = regret_at(theta_fy)

            print("Slice results:")
            print(f"  theta*_2s    (OLS/MSE anchor) : {np.array2string(theta_mse, precision=4)}")
            print(f"  theta*_oracle (Regret min)    : {np.array2string(theta_oracle, precision=4)}")
            print(f"  theta*_e2e   (FY min)         : {np.array2string(theta_fy, precision=4)}")
            regret_label = "Regret" if len(graphs_data) == 1 else "Avg Regret"
            print(f"  {regret_label} at theta*_2s       : {r_mse:.4f}")
            print(f"  {regret_label} at theta*_oracle   : {r_oracle:.4f}")
            print(f"  {regret_label} at theta*_e2e      : {r_fy:.4f}")
            print(f"  Regret range on slice         : [{R.min():.4f}, {R.max():.4f}]")

            slice_results.append({
                "dims": dims,
                "fixed_dim": fixed_dim,
                "theta_base": theta_mse,
                "regret_data": regret_data,
                "fy_data": fy_data,
                "theta_mse_xy": np.array([theta_mse[dim_x], theta_mse[dim_y]]),
                "theta_oracle_xy": np.array([theta_oracle[dim_x], theta_oracle[dim_y]]),
                "theta_fy_xy": np.array([theta_fy[dim_x], theta_fy[dim_y]]),
                "regret_at_markers": [r_mse, r_oracle, r_fy],
            })

        # 6. Plot each pairwise slice as its own 2x2 figure
        output_base = output_base_for_epsilon(args.output, args.fy_epsilon, args.plot_root)
        output_base.parent.mkdir(parents=True, exist_ok=True)
        print(f"Plot outputs will be saved under: {output_base.parent}")
        for result in slice_results:
            slice_output = output_path_for_slice(output_base, result["dims"])
            plot_slice_landscape(result, slice_output, num_graphs=len(graphs_data), graph_label=graph_label)

    finally:
        env.dispose()


if __name__ == "__main__":
    main()
