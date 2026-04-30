"""
Linear Probe Landscape Experiment
==================================
Visualize true Regret and Fenchel-Young surrogate loss landscapes over the full
2D parameter space (theta_1, theta_2) for the utility-plus-cPRA probe:

    w_hat_e = theta_1 * utility_e + theta_2 * recipient_cPRA_e

This demonstrates that the FY-optimal parameters differ from the
decision-optimal parameters, even when model capacity is identical.
"""

import argparse
import csv
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
from formulations.common.backend_utils import (
    build_edge_lists,
    build_pief_chain_keys,
    edge_selection_array,
    infer_ndd_mask,
    to_numpy_edge_index,
    to_numpy_weights,
)

PROBES = {
    "utility_cpra": {
        "feature_names": ["utility", "recipient_cPRA"],
        "feature_labels": [
            r"$\theta_1$ (utility weight)",
            r"$\theta_2$ (recipient cPRA weight)",
        ],
        "title": "utility + cPRA probe",
        "formula": r"$\hat{w}_e = \theta_1 u_e + \theta_2 c_e$",
        "suffix": "utility_cpra",
    },
    "utility_intercept": {
        "feature_names": ["utility", "intercept"],
        "feature_labels": [
            r"$\theta_1$ (utility weight)",
            r"$\theta_2$ (intercept)",
        ],
        "title": "utility + intercept probe",
        "formula": r"$\hat{w}_e = \theta_1 u_e + \theta_2$",
        "suffix": "utility_intercept",
    },
}


# ─────────────────────────────────────────────
# Graph loading & feature extraction
# ─────────────────────────────────────────────

def load_graph(graph_path, max_cycle=3, max_chain=4):
    """Load a single graph and return all needed components."""
    data = parse_json_to_dfl_data(graph_path, max_cycle=max_cycle, max_chain=max_chain, label_scale=1.0)
    # label_scale=1.0 → data.y is in original scale (not divided by 25)

    _, dst = data.edge_index
    utility = data.edge_attr[:, 0].numpy()             # shape [E]
    recipient_cpra = data.x[dst, 1].numpy()            # recipient cPRA, shape [E]
    w_true = data.y.numpy()                            # ground truth weights, shape [E]

    cycle_candidates = [c for c in data.candidates if c["type"] == "cycle"]
    node_is_ndd = infer_ndd_mask(data.x)
    num_nodes = data.num_nodes_custom[0].item()

    return {
        "features": {
            "utility": utility,
            "recipient_cPRA": recipient_cpra,
            "intercept": np.ones_like(utility),
        },
        "w_true": w_true,
        "edge_index": data.edge_index,
        "cycle_candidates": cycle_candidates,
        "node_is_ndd": node_is_ndd,
        "num_nodes": num_nodes,
        "num_edges": data.num_edges,
        "filename": data.filename,
    }


def feature_matrix(graph_data, probe):
    """Build the [E, 2] feature matrix for the requested probe."""
    return np.stack(
        [graph_data["features"][name] for name in probe["feature_names"]],
        axis=1,
    )


# ─────────────────────────────────────────────
# Solver wrapper
# ─────────────────────────────────────────────

class CachedHybridKepModel:
    """Pre-built hybrid KEP model whose objective can be updated cheaply."""

    def __init__(self, graph, env, max_chain=4):
        self.graph = graph
        self.max_chain = max_chain
        self.edge_index_np = to_numpy_edge_index(graph["edge_index"])
        self.src, self.dst, self.outgoing, self.incoming = build_edge_lists(
            self.edge_index_np, graph["num_nodes"]
        )
        self.num_edges = len(self.src)
        self.cycle_candidates = graph["cycle_candidates"]
        self.pair_nodes = [
            node_idx for node_idx in range(graph["num_nodes"])
            if not bool(graph["node_is_ndd"][node_idx])
        ]
        self.ndd_nodes = [
            node_idx for node_idx in range(graph["num_nodes"])
            if bool(graph["node_is_ndd"][node_idx])
        ]
        self.valid_chain_keys = build_pief_chain_keys(
            src=self.src,
            dst=self.dst,
            outgoing=self.outgoing,
            is_ndd_mask=graph["node_is_ndd"],
            max_chain=max_chain,
        )

        self.chain_incoming = {}
        self.chain_outgoing = {}
        for edge_idx, position in self.valid_chain_keys:
            self.chain_incoming.setdefault((int(self.dst[edge_idx]), position), []).append((edge_idx, position))
            self.chain_outgoing.setdefault((int(self.src[edge_idx]), position), []).append((edge_idx, position))

        self.model = gp.Model(f"cached_{graph['filename']}", env=env)
        self.model.Params.OutputFlag = 0
        self.model.Params.Threads = 1
        self.cycle_vars = self.model.addVars(len(self.cycle_candidates), vtype=GRB.BINARY, name="cycle")
        self.chain_vars = self.model.addVars(self.valid_chain_keys, vtype=GRB.BINARY, name="chain")

        self.model.setObjective(
            gp.quicksum(0.0 * self.cycle_vars[idx] for idx in range(len(self.cycle_candidates)))
            + gp.quicksum(0.0 * self.chain_vars[key] for key in self.valid_chain_keys),
            GRB.MAXIMIZE,
        )

        for pair_node in self.pair_nodes:
            cycle_usage = gp.quicksum(
                self.cycle_vars[idx]
                for idx, candidate in enumerate(self.cycle_candidates)
                if pair_node in candidate["nodes"]
            )
            chain_usage = gp.quicksum(
                self.chain_vars[edge_idx, position]
                for edge_idx, position in self.valid_chain_keys
                if int(self.dst[edge_idx]) == pair_node
            )
            self.model.addConstr(cycle_usage + chain_usage <= 1, name=f"pair_once_{pair_node}")

        for ndd_node in self.ndd_nodes:
            self.model.addConstr(
                gp.quicksum(
                    self.chain_vars[edge_idx, 1]
                    for edge_idx, position in self.valid_chain_keys
                    if position == 1 and int(self.src[edge_idx]) == ndd_node
                )
                <= 1,
                name=f"ndd_once_{ndd_node}",
            )

        for pair_node in self.pair_nodes:
            for position in range(2, max_chain + 1):
                outgoing_at_position = gp.quicksum(
                    self.chain_vars[edge_idx, position]
                    for edge_idx, _ in self.chain_outgoing.get((pair_node, position), [])
                )
                incoming_previous = gp.quicksum(
                    self.chain_vars[edge_idx, position - 1]
                    for edge_idx, _ in self.chain_incoming.get((pair_node, position - 1), [])
                )
                self.model.addConstr(
                    outgoing_at_position <= incoming_previous,
                    name=f"chain_flow_{pair_node}_{position}",
                )
        self.model.update()

    def solve(self, weights):
        weights = to_numpy_weights(weights)
        for idx, candidate in enumerate(self.cycle_candidates):
            self.cycle_vars[idx].Obj = float(sum(weights[edge_idx] for edge_idx in candidate["edges"]))
        for edge_idx, position in self.valid_chain_keys:
            self.chain_vars[edge_idx, position].Obj = float(weights[edge_idx])

        self.model.optimize()

        selected_cycle_indices = []
        selected_chain_keys = []
        if self.model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and self.model.SolCount > 0:
            selected_cycle_indices = [
                idx for idx in range(len(self.cycle_candidates)) if self.cycle_vars[idx].X > 0.5
            ]
            selected_chain_keys = [
                (edge_idx, position)
                for edge_idx, position in self.valid_chain_keys
                if self.chain_vars[edge_idx, position].X > 0.5
            ]

        selected_edges = []
        for idx in selected_cycle_indices:
            selected_edges.extend(self.cycle_candidates[idx]["edges"])
        selected_edges.extend(edge_idx for edge_idx, _ in selected_chain_keys)
        return edge_selection_array(self.num_edges, selected_edges)

    def dispose(self):
        self.model.dispose()


def solve_once(w_hat, graph, env):
    """Solve KEP with predicted weights, return edge selection vector."""
    if graph.get("cached_solver") is not None:
        return graph["cached_solver"].solve(w_hat)
    result = solve_cf_cycle_pief_chain(
        weights=w_hat,
        edge_index=graph["edge_index"],
        is_ndd_mask=graph["node_is_ndd"],
        num_nodes=graph["num_nodes"],
        cycle_candidates=graph["cycle_candidates"],
        env=env,
    )
    return result["edge_selection"]


def assert_cached_solver_matches_original(graph, env):
    """Validate cached model reuse against the original one-shot backend.

    Compares objective values rather than exact edge selections: when edge
    weights have ties, multiple optimal solutions exist and different Gurobi
    model instances may select different ones.  Objective equality is the
    correct equivalence criterion.
    """
    if graph.get("cached_solver") is None:
        raise ValueError("Cached solver was not attached to graph")
    weights_to_check = [
        graph["w_true"],
        graph["features"]["utility"],
        graph["w_true"] + np.linspace(-0.25, 0.25, graph["num_edges"]),
    ]
    for check_idx, weights in enumerate(weights_to_check, start=1):
        w = to_numpy_weights(weights)
        reference_sel = solve_cf_cycle_pief_chain(
            weights=w,
            edge_index=graph["edge_index"],
            is_ndd_mask=graph["node_is_ndd"],
            num_nodes=graph["num_nodes"],
            cycle_candidates=graph["cycle_candidates"],
            env=env,
        )["edge_selection"]
        cached_sel = graph["cached_solver"].solve(w)
        ref_obj = float(np.dot(w, reference_sel))
        cached_obj = float(np.dot(w, cached_sel))
        if abs(ref_obj - cached_obj) > 1e-6 * (abs(ref_obj) + 1.0):
            raise AssertionError(
                f"Cached solver objective mismatch on {graph['filename']} check {check_idx}: "
                f"reference={ref_obj:.6f}, cached={cached_obj:.6f}"
            )


def compute_regret(y_pred, y_optimal, w_true):
    """Regret = optimal_obj - achieved_obj under true weights."""
    achieved = np.dot(w_true, y_pred)
    optimal = np.dot(w_true, y_optimal)
    return optimal - achieved


# ─────────────────────────────────────────────
# Landscape computations
# ─────────────────────────────────────────────

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


def regret_landscape(graphs_data, theta_x_grid, theta_y_grid, env):
    """Compute average true Regret(theta) over the full 2D probe space."""
    T1, T2 = np.meshgrid(theta_x_grid, theta_y_grid)
    regret = np.zeros_like(T1)
    total = T1.size
    for idx in range(total):
        i, j = divmod(idx, T1.shape[1])
        theta = np.array([T1[i, j], T2[i, j]])
        for gd in graphs_data:
            w_hat = gd["X"] @ theta
            y_pred = solve_once(w_hat, gd["graph"], env)
            regret[i, j] += compute_regret(y_pred, gd["y_optimal"], gd["w_true"])
        if (idx + 1) % 50 == 0 or idx + 1 == total:
            print(f"  Regret grid: {idx + 1}/{total}", flush=True)
    regret /= len(graphs_data)
    return T1, T2, regret


def fy_loss_landscape(graphs_data, theta_x_grid, theta_y_grid, env,
                      epsilon=0.2, M=8, theta_mse=None):
    """Compute the Fenchel-Young surrogate loss landscape over the 2D probe.

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
        theta = np.array([T1[i, j], T2[i, j]])
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


def add_xy_padding(ax, T1, T2):
    """Keep boundary minima and marker labels inside the visible axes."""
    x_min, x_max = float(np.min(T1)), float(np.max(T1))
    y_min, y_max = float(np.min(T2)), float(np.max(T2))
    x_pad = max((x_max - x_min) * 0.04, 1e-6)
    y_pad = max((y_max - y_min) * 0.04, 1e-6)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)


def marker_label_offset(theta_xy, T1, T2):
    """Choose an annotation offset that avoids the nearest plot boundary."""
    x_min, x_max = float(np.min(T1)), float(np.max(T1))
    y_min, y_max = float(np.min(T2)), float(np.max(T2))
    x_span = max(x_max - x_min, 1e-12)
    y_span = max(y_max - y_min, 1e-12)
    x_frac = (theta_xy[0] - x_min) / x_span
    y_frac = (theta_xy[1] - y_min) / y_span
    dx = 12 if x_frac < 0.75 else -48
    dy = 14 if y_frac < 0.25 else (-22 if y_frac > 0.75 else 12)
    return dx, dy


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
            ax.annotate(
                f"R={regret_at_markers[k]:.2f}",
                xy=(theta_xy[0], theta_xy[1]),
                xytext=marker_label_offset(theta_xy, T1, T2),
                textcoords="offset points",
                fontsize=8, fontweight="bold", color=color,
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.7),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
            )

    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_title(title, fontsize=12, pad=8)
    add_xy_padding(ax, T1, T2)
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
    add_xy_padding(ax, T1, T2)
    ax.view_init(elev=28, azim=-50)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.85)


def plot_landscape(result, output_path, probe, num_graphs=0, graph_label=None):
    """Plot the full 2D probe landscape as 2x2: contours on top, 3D surfaces below."""
    fig = plt.figure(figsize=(16, 14))
    axes = [
        fig.add_subplot(2, 2, 1),
        fig.add_subplot(2, 2, 2),
        fig.add_subplot(2, 2, 3, projection="3d"),
        fig.add_subplot(2, 2, 4, projection="3d"),
    ]

    markers = [
        (r"$\theta^*_{\mathrm{2s}}$  (MSE/OLS)", result["theta_mse_xy"], "red", "*"),
        (r"$\theta^*_{\mathrm{oracle}}$ (Regret min)", result["theta_oracle_xy"], "lime", "D"),
        (r"$\theta^*_{\mathrm{e2e}}$  (FY min)", result["theta_fy_xy"], "cyan", "o"),
    ]
    clipped_contour(
        axes[0],
        fig,
        result["regret_data"],
        "True Regret — contour",
        markers,
        probe["feature_labels"][0],
        probe["feature_labels"][1],
        regret_at_markers=result["regret_at_markers"],
    )
    clipped_contour(
        axes[1],
        fig,
        result["fy_data"],
        "Fenchel-Young Surrogate — contour",
        markers,
        probe["feature_labels"][0],
        probe["feature_labels"][1],
    )
    clipped_3d_surface(
        axes[2],
        fig,
        result["regret_data"],
        "True Regret — 3D surface",
        markers,
        probe["feature_labels"][0],
        probe["feature_labels"][1],
    )
    clipped_3d_surface(
        axes[3],
        fig,
        result["fy_data"],
        "Fenchel-Young Surrogate — 3D surface",
        markers,
        probe["feature_labels"][0],
        probe["feature_labels"][1],
    )

    fig.suptitle(
        probe["title"] + "\n"
        + probe["formula"]
        + (f"\n{graph_label}" if graph_label else (f"\naveraged over {num_graphs} graphs" if num_graphs else "")),
        fontsize=15,
        y=1.005,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved landscape plot to: {output_path}")
    plt.close(fig)


def plot_axis_summary(result, output_path, probe, axis, num_graphs=0, graph_label=None):
    """Plot 1D projection summaries for one parameter axis."""
    if axis not in (0, 1):
        raise ValueError("axis must be 0 for theta_1 or 1 for theta_2")

    T1_r, T2_r, R = result["regret_data"]
    T1_f, T2_f, F = result["fy_data"]
    theta_grid_r = T1_r[0, :] if axis == 0 else T2_r[:, 0]
    theta_grid_f = T1_f[0, :] if axis == 0 else T2_f[:, 0]
    theta_label = probe["feature_labels"][axis]
    other_label = probe["feature_labels"][1 - axis]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharex=False)
    panels = [
        (axes[0], theta_grid_r, T1_r, T2_r, R, "True Regret"),
        (axes[1], theta_grid_f, T1_f, T2_f, F, "Fenchel-Young Surrogate"),
    ]
    marker_specs = [
        (r"$\theta^*_{\mathrm{2s}}$", result["theta_mse_xy"], "red", "*"),
        (r"$\theta^*_{\mathrm{oracle}}$", result["theta_oracle_xy"], "lime", "D"),
        (r"$\theta^*_{\mathrm{e2e}}$", result["theta_fy_xy"], "cyan", "o"),
    ]

    for ax, theta_grid, T1, T2, Z, title in panels:
        if axis == 0:
            min_over_other = Z.min(axis=0)
            mean_over_other = Z.mean(axis=0)
            fixed_idx = int(np.argmin(np.abs(T2[:, 0] - result["theta_mse_xy"][1])))
            fixed_at_mse = Z[fixed_idx, :]
        else:
            min_over_other = Z.min(axis=1)
            mean_over_other = Z.mean(axis=1)
            fixed_idx = int(np.argmin(np.abs(T1[0, :] - result["theta_mse_xy"][0])))
            fixed_at_mse = Z[:, fixed_idx]

        ax.plot(theta_grid, min_over_other, color="black", linewidth=2.0,
                label=rf"$\min$ over other parameter")
        ax.plot(theta_grid, mean_over_other, color="tab:blue", linewidth=2.0,
                linestyle="--", label="mean over other parameter")
        ax.plot(theta_grid, fixed_at_mse, color="0.45", linewidth=1.6,
                linestyle=":", label=f"slice at MSE {other_label}")

        y_min, y_max = float(np.min(Z)), float(np.max(Z))
        y_pad = max((y_max - y_min) * 0.08, 1e-6)
        for label, theta_xy, color, marker_shape in marker_specs:
            theta_value = theta_xy[axis]
            ii = int(np.argmin(np.abs(T2[:, 0] - theta_xy[1])))
            jj = int(np.argmin(np.abs(T1[0, :] - theta_xy[0])))
            y_value = Z[ii, jj]
            ax.axvline(theta_value, color=color, linewidth=1.2, alpha=0.55)
            ax.plot(theta_value, y_value, marker=marker_shape, color=color,
                    markersize=10, markeredgecolor="black", markeredgewidth=1.0,
                    label=label)

        ax.set_title(title)
        ax.set_xlabel(theta_label)
        ax.set_ylabel("Loss")
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle(
        f"{probe['title']} — {theta_label} summary\n"
        + probe["formula"]
        + (f"\n{graph_label}" if graph_label else (f"\naveraged over {num_graphs} graphs" if num_graphs else "")),
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved axis summary plot to: {output_path}")
    plt.close(fig)


def epsilon_dir_name(epsilon):
    """Return a stable directory name such as epsilon=0.2."""
    return f"epsilon={float(epsilon):.6g}"


def probe_output_dir(plot_root, epsilon, probe_key):
    """Return the output directory for one epsilon/probe pair."""
    return Path(plot_root) / epsilon_dir_name(epsilon) / probe_key


def nearest_grid_value(data, theta_xy):
    """Read the landscape value at the nearest grid point to theta_xy."""
    T1, T2, Z = data
    ii = int(np.argmin(np.abs(T2[:, 0] - theta_xy[1])))
    jj = int(np.argmin(np.abs(T1[0, :] - theta_xy[0])))
    return float(Z[ii, jj])


def as_float_list(values):
    """Convert numpy vectors to JSON-friendly float lists."""
    return [float(x) for x in np.asarray(values).ravel()]


def graph_filenames(base_graphs):
    """Return stable graph filename metadata for this run."""
    return [item["graph"]["filename"] for item in base_graphs]


def build_metrics(result, probe_key, probe, args, base_graphs):
    """Build scalar metadata needed for paper tables and epsilon sweeps."""
    r_mse, r_oracle, r_fy = [float(x) for x in result["regret_at_markers"]]
    f_mse, f_oracle, f_fy = [float(x) for x in result["fy_at_markers"]]
    _, _, R = result["regret_data"]
    _, _, F = result["fy_data"]
    return {
        "epsilon": float(args.fy_epsilon),
        "probe_key": probe_key,
        "feature_names": probe["feature_names"],
        "num_graphs": len(base_graphs),
        "graph_filenames": graph_filenames(base_graphs),
        "grid_size": int(args.grid_size),
        "fy_M": int(args.fy_M),
        "margin": float(args.margin),
        "theta_mse": as_float_list(result["theta_mse_xy"]),
        "theta_oracle": as_float_list(result["theta_oracle_xy"]),
        "theta_fy": as_float_list(result["theta_fy_xy"]),
        "R_mse": r_mse,
        "R_oracle": r_oracle,
        "R_fy": r_fy,
        "delta_R_mse": r_mse - r_oracle,
        "delta_R_fy": r_fy - r_oracle,
        "FY_mse": f_mse,
        "FY_oracle": f_oracle,
        "FY_fy": f_fy,
        "R_min_grid": float(np.min(R)),
        "R_max_grid": float(np.max(R)),
        "FY_min_grid": float(np.min(F)),
        "FY_max_grid": float(np.max(F)),
    }


def save_probe_artifacts(result, output_dir, probe_key, probe, args, base_graphs):
    """Save reproducible arrays and scalar metrics for one probe."""
    output_dir.mkdir(parents=True, exist_ok=True)
    T1_r, T2_r, R = result["regret_data"]
    _, _, F = result["fy_data"]
    npz_path = output_dir / "landscape_data.npz"
    np.savez_compressed(
        npz_path,
        theta1_grid=T1_r[0, :],
        theta2_grid=T2_r[:, 0],
        true_regret=R,
        fy_loss=F,
        theta_mse=result["theta_mse_xy"],
        theta_oracle=result["theta_oracle_xy"],
        theta_fy=result["theta_fy_xy"],
        regret_at_markers=np.asarray(result["regret_at_markers"], dtype=float),
        fy_at_markers=np.asarray(result["fy_at_markers"], dtype=float),
    )

    metrics = build_metrics(result, probe_key, probe, args, base_graphs)
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved landscape data to: {npz_path}")
    print(f"Saved metrics to: {metrics_path}")
    return metrics


def update_epsilon_sweep_csv(metrics, plot_root):
    """Upsert this run into the cross-epsilon metrics table."""
    sweep_dir = Path(plot_root) / "epsilon_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    csv_path = sweep_dir / "epsilon_sweep_metrics.csv"
    fieldnames = [
        "epsilon", "probe_key", "num_graphs", "grid_size", "fy_M",
        "theta_mse_1", "theta_mse_2",
        "theta_oracle_1", "theta_oracle_2",
        "theta_fy_1", "theta_fy_2",
        "R_mse", "R_oracle", "R_fy",
        "delta_R_mse", "delta_R_fy",
        "FY_mse", "FY_oracle", "FY_fy",
        "graph_filenames",
    ]
    row = {
        "epsilon": metrics["epsilon"],
        "probe_key": metrics["probe_key"],
        "num_graphs": metrics["num_graphs"],
        "grid_size": metrics["grid_size"],
        "fy_M": metrics["fy_M"],
        "theta_mse_1": metrics["theta_mse"][0],
        "theta_mse_2": metrics["theta_mse"][1],
        "theta_oracle_1": metrics["theta_oracle"][0],
        "theta_oracle_2": metrics["theta_oracle"][1],
        "theta_fy_1": metrics["theta_fy"][0],
        "theta_fy_2": metrics["theta_fy"][1],
        "R_mse": metrics["R_mse"],
        "R_oracle": metrics["R_oracle"],
        "R_fy": metrics["R_fy"],
        "delta_R_mse": metrics["delta_R_mse"],
        "delta_R_fy": metrics["delta_R_fy"],
        "FY_mse": metrics["FY_mse"],
        "FY_oracle": metrics["FY_oracle"],
        "FY_fy": metrics["FY_fy"],
        "graph_filenames": json.dumps(metrics["graph_filenames"]),
    }

    rows = []
    if csv_path.exists():
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))

    def same_run(existing):
        return (
            str(existing.get("epsilon")) == str(row["epsilon"])
            and existing.get("probe_key") == row["probe_key"]
            and str(existing.get("num_graphs")) == str(row["num_graphs"])
            and str(existing.get("grid_size")) == str(row["grid_size"])
            and str(existing.get("fy_M")) == str(row["fy_M"])
            and existing.get("graph_filenames") == row["graph_filenames"]
        )

    rows = [existing for existing in rows if not same_run(existing)]
    rows.append(row)
    rows.sort(key=lambda r: (r["probe_key"], float(r["epsilon"])))

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated epsilon sweep metrics table: {csv_path}")


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


def compute_probe_result(base_graphs, probe_key, env, args):
    """Compute OLS anchor, true regret, and FY landscapes for one probe."""
    probe = PROBES[probe_key]
    graphs_data = []
    all_X = []
    all_w = []
    for item in base_graphs:
        X = feature_matrix(item["graph"], probe)
        graphs_data.append({
            "X": X,
            "w_true": item["graph"]["w_true"],
            "graph": item["graph"],
            "y_optimal": item["y_optimal"],
        })
        all_X.append(X)
        all_w.append(item["graph"]["w_true"])

    # Utility probe OLS solution, used for grid anchoring and theta*_2s marker.
    X_pool = np.concatenate(all_X, axis=0)
    w_pool = np.concatenate(all_w, axis=0)
    theta_mse = np.linalg.lstsq(X_pool, w_pool, rcond=None)[0]
    print(f"\nProbe: {probe['title']}")
    print("OLS anchor (theta*_2s):")
    for name, value in zip(probe["feature_names"], theta_mse):
        print(f"  {name}: {value:.4f}")

    # Determine per-parameter grid ranges centered on the OLS solution.
    radius = np.linalg.norm(theta_mse)
    pad = max(radius * args.margin, 1e-6)
    theta_grids = []
    for k, center in enumerate(theta_mse):
        lo = center - pad
        hi = center + pad
        theta_grids.append(np.linspace(lo, hi, args.grid_size))
        print(f"Grid range: {probe['feature_names'][k]} theta_{k + 1} in [{lo:.2f}, {hi:.2f}]")

    theta_x_grid = theta_grids[0]
    theta_y_grid = theta_grids[1]

    print(f"Computing {probe['title']} True Regret landscape "
          f"({args.grid_size}x{args.grid_size} x {len(graphs_data)} graphs)...")
    regret_data = regret_landscape(graphs_data, theta_x_grid, theta_y_grid, env)

    print(f"Computing {probe['title']} FY Loss landscape "
          f"({args.grid_size}x{args.grid_size} x {len(graphs_data)} graphs x M={args.fy_M}, "
          f"epsilon={args.fy_epsilon})...")
    fy_data = fy_loss_landscape(
        graphs_data, theta_x_grid, theta_y_grid, env,
        epsilon=args.fy_epsilon, M=args.fy_M,
        theta_mse=theta_mse,
    )

    T1_r, T2_r, R = regret_data
    min_idx_r = np.unravel_index(np.argmin(R), R.shape)
    theta_oracle = np.array([T1_r[min_idx_r], T2_r[min_idx_r]])

    T1_f, T2_f, F = fy_data
    min_idx_f = np.unravel_index(np.argmin(F), F.shape)
    theta_fy = np.array([T1_f[min_idx_f], T2_f[min_idx_f]])

    r_mse = nearest_grid_value(regret_data, theta_mse)
    r_oracle = float(R[min_idx_r])
    r_fy = nearest_grid_value(regret_data, theta_fy)
    f_mse = nearest_grid_value(fy_data, theta_mse)
    f_oracle = nearest_grid_value(fy_data, theta_oracle)
    f_fy = float(F[min_idx_f])

    print("Full 2D probe results:")
    print(f"  theta*_2s    (OLS/MSE anchor) : {np.array2string(theta_mse, precision=4)}")
    print(f"  theta*_oracle (Regret min)    : {np.array2string(theta_oracle, precision=4)}")
    print(f"  theta*_e2e   (FY min)         : {np.array2string(theta_fy, precision=4)}")
    regret_label = "Regret" if len(graphs_data) == 1 else "Avg Regret"
    print(f"  {regret_label} at theta*_2s       : {r_mse:.4f}")
    print(f"  {regret_label} at theta*_oracle   : {r_oracle:.4f}")
    print(f"  {regret_label} at theta*_e2e      : {r_fy:.4f}")
    print(f"  Regret range on full 2D grid  : [{R.min():.4f}, {R.max():.4f}]")
    print(f"  FY range on full 2D grid      : [{F.min():.4f}, {F.max():.4f}]")

    return {
        "regret_data": regret_data,
        "fy_data": fy_data,
        "theta_mse_xy": theta_mse,
        "theta_oracle_xy": theta_oracle,
        "theta_fy_xy": theta_fy,
        "regret_at_markers": [r_mse, r_oracle, r_fy],
        "fy_at_markers": [f_mse, f_oracle, f_fy],
    }


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
    parser.add_argument("--plot_root", type=str, default="plot_results",
                        help="Root directory for epsilon-grouped landscape plots")
    parser.add_argument("--margin", type=float, default=0.5,
                        help="Grid half-width factor around the 2D MSE optimum")
    parser.add_argument("--percentile", type=int, default=10,
                        help="Edge-count percentile for graph selection (lower = smaller graph)")
    parser.add_argument("--graph_id", type=int, default=42,
                        help="Graph id to use by default, e.g. 42 selects G-42.json")
    parser.add_argument("--num_graphs", type=int, default=1,
                        help="Number of graphs to average over. Use 1 for the selected single graph.")
    parser.add_argument("--include_intercept_summaries", action="store_true",
                        help="Also save utility+intercept summaries as an appendix/sanity check")
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

    base_graphs = []
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Seed", 42)
    env.start()

    try:
        for gp_path in graph_paths:
            g = load_graph(gp_path)
            g["cached_solver"] = CachedHybridKepModel(g, env)
            assert_cached_solver_matches_original(g, env)
            y_opt = solve_once(g["w_true"], g, env)
            base_graphs.append({"graph": g, "y_optimal": y_opt})

        print(f"Loaded {len(base_graphs)} graphs with cached Gurobi models")
        graph_label = None
        if len(base_graphs) == 1:
            graph_label = f"single graph: {base_graphs[0]['graph']['filename']}"

        cpra_probe = PROBES["utility_cpra"]
        cpra_result = compute_probe_result(base_graphs, "utility_cpra", env, args)
        cpra_dir = probe_output_dir(args.plot_root, args.fy_epsilon, "utility_cpra")
        cpra_dir.mkdir(parents=True, exist_ok=True)
        print(f"Utility+cPRA outputs will be saved under: {cpra_dir}")
        plot_landscape(cpra_result, cpra_dir / "linear_probe_landscape.png", cpra_probe,
                       num_graphs=len(base_graphs), graph_label=graph_label)
        plot_axis_summary(
            cpra_result,
            cpra_dir / "theta1_summary.png",
            cpra_probe,
            axis=0,
            num_graphs=len(base_graphs),
            graph_label=graph_label,
        )
        plot_axis_summary(
            cpra_result,
            cpra_dir / "theta2_summary.png",
            cpra_probe,
            axis=1,
            num_graphs=len(base_graphs),
            graph_label=graph_label,
        )
        cpra_metrics = save_probe_artifacts(
            cpra_result, cpra_dir, "utility_cpra", cpra_probe, args, base_graphs
        )
        update_epsilon_sweep_csv(cpra_metrics, args.plot_root)

        if args.include_intercept_summaries:
            intercept_probe = PROBES["utility_intercept"]
            intercept_result = compute_probe_result(base_graphs, "utility_intercept", env, args)
            intercept_dir = probe_output_dir(args.plot_root, args.fy_epsilon, "utility_intercept")
            intercept_dir.mkdir(parents=True, exist_ok=True)
            print(f"Utility+intercept outputs will be saved under: {intercept_dir}")
            plot_axis_summary(
                intercept_result,
                intercept_dir / "theta1_summary.png",
                intercept_probe,
                axis=0,
                num_graphs=len(base_graphs),
                graph_label=graph_label,
            )
            plot_axis_summary(
                intercept_result,
                intercept_dir / "theta2_summary.png",
                intercept_probe,
                axis=1,
                num_graphs=len(base_graphs),
                graph_label=graph_label,
            )
            intercept_metrics = save_probe_artifacts(
                intercept_result, intercept_dir, "utility_intercept",
                intercept_probe, args, base_graphs
            )
            update_epsilon_sweep_csv(intercept_metrics, args.plot_root)
        else:
            print("Skipped utility+intercept summaries. Use --include_intercept_summaries to save them.")

    finally:
        for item in base_graphs:
            cached_solver = item["graph"].get("cached_solver")
            if cached_solver is not None:
                cached_solver.dispose()
        env.dispose()


if __name__ == "__main__":
    main()
