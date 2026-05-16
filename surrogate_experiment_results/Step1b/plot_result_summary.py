"""Create presentation plots from the current Step1b result archive."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

METHOD_LABELS = {
    ("2stage", "validation_mse_loss"): "2stage (val MSE)",
    ("e2e", "validation_decision_gap"): "e2e (val gap)",
    ("e2e", "validation_fy_loss"): "e2e (val FY)",
}

METHOD_ORDER = [
    ("2stage", "validation_mse_loss"),
    ("e2e", "validation_decision_gap"),
    ("e2e", "validation_fy_loss"),
]

METHOD_COLORS = {
    ("2stage", "validation_mse_loss"): "#2b2b2b",
    ("e2e", "validation_decision_gap"): "#1f77b4",
    ("e2e", "validation_fy_loss"): "#d55e00",
}

METHOD_MARKERS = {
    ("2stage", "validation_mse_loss"): "o",
    ("e2e", "validation_decision_gap"): "s",
    ("e2e", "validation_fy_loss"): "^",
}

DATASETS = [
    {
        "key": "heldout400",
        "summary_file": "test_summary.csv",
        "per_graph_file": "test_per_graph.csv",
        "epoch_output": "heldout400_epoch_diagnostics",
        "calibration_output": "heldout400_reward_calibration",
        "title": "Held-out noisy-linear test",
        "subtitle": "400 graphs",
        "graph_source": "split_test",
    },
    {
        "key": "unseen1000",
        "summary_file": "unseen_test_summary.csv",
        "per_graph_file": "unseen_test_per_graph.csv",
        "epoch_output": "unseen1000_epoch_diagnostics",
        "calibration_output": "unseen1000_reward_calibration",
        "title": "Unseen noisy-linear test",
        "subtitle": "1000 graphs",
        "graph_source": "dataset_dir",
        "dataset_dir": PROJECT_ROOT
        / "dataset"
        / "processed"
        / "step1_noisy_linear_sigma010_unseen_test1000_seed20260513",
    },
    {
        "key": "realistic2000",
        "summary_file": "realistic_unseen_test_summary.csv",
        "per_graph_file": "realistic_unseen_test_per_graph.csv",
        "epoch_output": "realistic2000_epoch_diagnostics",
        "calibration_output": "realistic2000_reward_calibration",
        "title": "Realistic-synthetic stress test",
        "subtitle": "2000 graphs",
        "graph_source": "dataset_dir",
        "dataset_dir": PROJECT_ROOT / "dataset" / "processed" / "realistic_synthetic_dataset",
    },
]

DEFAULT_SPLIT_PATH = PROJECT_ROOT / "results" / "step1b_splits" / "master_split_seed=42.json"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: str | None) -> float:
    if value is None or value == "":
        return math.nan
    return float(value)


def available_train_sizes(results_root: Path) -> list[int]:
    sizes = []
    for child in results_root.iterdir():
        if child.is_dir() and child.name.startswith("train_size="):
            try:
                sizes.append(int(child.name.split("=", 1)[1]))
            except ValueError:
                continue
    return sorted(sizes)


def load_summary_rows(results_root: Path, train_sizes: list[int]) -> dict[str, list[dict[str, str]]]:
    rows_by_dataset: dict[str, list[dict[str, str]]] = {dataset["key"]: [] for dataset in DATASETS}
    for train_size in train_sizes:
        metrics_dir = results_root / f"train_size={train_size}" / "metrics"
        for dataset in DATASETS:
            path = metrics_dir / dataset["summary_file"]
            if not path.exists():
                continue
            for row in read_csv(path):
                row = dict(row)
                row["train_size"] = row.get("train_size") or str(train_size)
                row["dataset_key"] = dataset["key"]
                rows_by_dataset[dataset["key"]].append(row)
    return rows_by_dataset


def row_key(row: dict[str, str]) -> tuple[str, str]:
    return row["method"], row["selection_metric"]


def style_axes(ax):
    ax.grid(True, alpha=0.24, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    fig.savefig(out_path.with_suffix(".pdf"))


def train_size_colors(train_sizes):
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("viridis")
    if len(train_sizes) == 1:
        return {train_sizes[0]: cmap(0.55)}
    return {
        size: cmap(0.12 + 0.76 * idx / (len(train_sizes) - 1))
        for idx, size in enumerate(train_sizes)
    }


def plot_metric_panels(rows_by_dataset, train_sizes, metric, ylabel, title, out_path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharex=True, constrained_layout=True)
    x_positions = np.arange(len(train_sizes))
    x_by_size = {size: idx for idx, size in enumerate(train_sizes)}

    for ax, dataset in zip(axes, DATASETS):
        rows = rows_by_dataset[dataset["key"]]
        for method_key in METHOD_ORDER:
            method_rows = [row for row in rows if row_key(row) == method_key]
            method_rows.sort(key=lambda row: int(row["train_size"]))
            xs = [x_by_size[int(row["train_size"])] for row in method_rows]
            ys = [as_float(row[metric]) for row in method_rows]
            if not xs:
                continue
            ax.plot(
                xs,
                ys,
                label=METHOD_LABELS[method_key],
                color=METHOD_COLORS[method_key],
                marker=METHOD_MARKERS[method_key],
                linewidth=2.0,
                markersize=6,
            )
        ax.set_title(f"{dataset['title']}\n{dataset['subtitle']}", fontsize=11)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(size) for size in train_sizes])
        ax.set_xlabel("Training graphs")
        style_axes(ax)

    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle(title, y=1.18, fontsize=14)
    save_figure(fig, out_path)
    plt.close(fig)


def plot_direct_method_performance_bars(rows_by_dataset, train_sizes, out_path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharex=True, constrained_layout=True)
    x_positions = np.arange(len(train_sizes))
    width = 0.24
    offsets = [-width, 0.0, width]

    for ax, dataset in zip(axes, DATASETS):
        rows = rows_by_dataset[dataset["key"]]
        for offset, method_key in zip(offsets, METHOD_ORDER):
            values = []
            for size in train_sizes:
                candidates = [
                    row
                    for row in rows
                    if row_key(row) == method_key and int(row["train_size"]) == size
                ]
                if not candidates:
                    values.append(math.nan)
                    continue
                row = candidates[0]
                values.append(as_float(row["test_mean_normalized_gap"]))

            ax.bar(
                x_positions + offset,
                values,
                width=width,
                color=METHOD_COLORS[method_key],
                label=METHOD_LABELS[method_key],
                alpha=0.86,
            )
        ax.set_ylim(bottom=0.0)
        ax.set_title(f"{dataset['title']}\n{dataset['subtitle']}", fontsize=11)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(size) for size in train_sizes])
        ax.set_xlabel("Training graphs")
        style_axes(ax)

    axes[0].set_ylabel("Mean normalized decision gap (lower is better)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Direct Method Performance by Training Set Size", y=1.18, fontsize=14)
    save_figure(fig, out_path)
    plt.close(fig)


def plot_parameter_endpoints(rows_by_dataset, train_sizes, out_path):
    import matplotlib.pyplot as plt

    rows = rows_by_dataset["heldout400"]
    fig, ax = plt.subplots(figsize=(7.0, 5.6), constrained_layout=True)
    train_colors = {
        train_sizes[0]: "#0072b2",
        train_sizes[1]: "#009e73" if len(train_sizes) > 1 else "#009e73",
        train_sizes[2]: "#cc79a7" if len(train_sizes) > 2 else "#cc79a7",
    }

    for method_key in METHOD_ORDER:
        method_rows = [row for row in rows if row_key(row) == method_key]
        method_rows.sort(key=lambda row: int(row["train_size"]))
        xs = [as_float(row["theta_1"]) for row in method_rows]
        ys = [as_float(row["theta_2"]) for row in method_rows]
        ax.plot(
            xs,
            ys,
            color=METHOD_COLORS[method_key],
            linewidth=1.2,
            alpha=0.5,
        )
        for row in method_rows:
            size = int(row["train_size"])
            ax.scatter(
                as_float(row["theta_1"]),
                as_float(row["theta_2"]),
                color=train_colors.get(size, "#555555"),
                marker=METHOD_MARKERS[method_key],
                s=82,
                edgecolor="#222222",
                linewidth=0.7,
                label=f"{METHOD_LABELS[method_key]}, n={size}",
            )

    ax.scatter(10.0, 5.0, marker="*", s=210, color="#f0c808", edgecolor="#222222", linewidth=0.8)
    ax.text(10.05, 5.05, "clean-signal coefficient [10,5]", fontsize=9, va="bottom")
    ax.set_xlabel(r"$\theta_1$ utility coefficient")
    ax.set_ylabel(r"$\theta_2$ cPRA coefficient")
    ax.set_title("Selected Model Parameters")
    style_axes(ax)

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), frameon=False, fontsize=8, ncols=1, loc="best")
    save_figure(fig, out_path)
    plt.close(fig)


def plot_theta_trajectories(results_root, train_sizes, rows_by_dataset, out_path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(train_sizes), figsize=(4.7 * len(train_sizes), 4.4), constrained_layout=True)
    if len(train_sizes) == 1:
        axes = [axes]
    rows = rows_by_dataset["heldout400"]

    for ax, size in zip(axes, train_sizes):
        traj_dir = results_root / f"train_size={size}" / "trajectories"
        t2 = np.load(traj_dir / "trajectory_2stage.npy")
        te = np.load(traj_dir / "trajectory_e2e.npy")
        ax.plot(t2[:, 0], t2[:, 1], color=METHOD_COLORS[("2stage", "validation_mse_loss")], linewidth=1.8, label="2stage trajectory")
        ax.plot(te[:, 0], te[:, 1], color=METHOD_COLORS[("e2e", "validation_decision_gap")], linewidth=1.8, label="e2e trajectory")
        ax.scatter(t2[0, 0], t2[0, 1], color="#777777", s=42, marker="x", label="shared init")
        ax.scatter(10.0, 5.0, marker="*", s=170, color="#f0c808", edgecolor="#222222", linewidth=0.8, label="[10,5]")

        for method_key in METHOD_ORDER:
            selected = [
                row
                for row in rows
                if row_key(row) == method_key and int(row["train_size"]) == size
            ]
            if not selected:
                continue
            row = selected[0]
            ax.scatter(
                as_float(row["theta_1"]),
                as_float(row["theta_2"]),
                marker=METHOD_MARKERS[method_key],
                color=METHOD_COLORS[method_key],
                edgecolor="#222222",
                linewidth=0.7,
                s=78,
                label=METHOD_LABELS[method_key],
            )

        ax.set_title(f"train_size={size}")
        ax.set_xlabel(r"$\theta_1$")
        ax.set_ylabel(r"$\theta_2$")
        style_axes(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="upper center", ncols=4, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Training Trajectories in Parameter Space", y=1.18, fontsize=14)
    save_figure(fig, out_path)
    plt.close(fig)


def plot_loss_curves(results_root, train_sizes, csv_name, columns, ylabel, title, out_path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(train_sizes), figsize=(4.7 * len(train_sizes), 4.1), sharey=False, constrained_layout=True)
    if len(train_sizes) == 1:
        axes = [axes]

    for ax, size in zip(axes, train_sizes):
        rows = read_csv(results_root / f"train_size={size}" / "metrics" / csv_name)
        epochs = [int(row["epoch"]) for row in rows]
        for label, column, color, marker in columns:
            values = [as_float(row[column]) for row in rows]
            ax.plot(epochs, values, label=label, color=color, marker=marker, markevery=max(1, len(epochs) // 10), linewidth=1.8, markersize=4)
        ax.set_title(f"train_size={size}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        style_axes(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=len(columns), frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle(title, y=1.18, fontsize=14)
    save_figure(fig, out_path)
    plt.close(fig)


def plot_combined_training_curves(results_root, train_sizes, out_path):
    import matplotlib.pyplot as plt

    row_specs = [
        {
            "title": "2stage reward fitting",
            "csv_name": "2stage_loss_curve.csv",
            "ylabel": "MSE loss",
            "columns": [
                ("train MSE", "train_mse_loss", "#2b2b2b", "o"),
                ("validation MSE", "validation_mse_loss", "#0072b2", "s"),
            ],
        },
        {
            "title": "e2e FY surrogate",
            "csv_name": "e2e_loss_curve.csv",
            "ylabel": "FY objective",
            "columns": [
                ("train FY objective", "train_fy_loss", "#2b2b2b", "o"),
                ("validation FY objective", "validation_fy_loss", "#d55e00", "s"),
            ],
        },
        {
            "title": "e2e downstream metric",
            "csv_name": "e2e_loss_curve.csv",
            "ylabel": "Decision gap",
            "columns": [
                ("train decision gap", "train_decision_gap", "#2b2b2b", "o"),
                ("validation decision gap", "validation_decision_gap", "#1f77b4", "s"),
            ],
        },
    ]

    fig, axes = plt.subplots(
        len(row_specs),
        len(train_sizes),
        figsize=(4.35 * len(train_sizes), 8.8),
        sharex=False,
        sharey=False,
        constrained_layout=True,
    )
    if len(train_sizes) == 1:
        axes = axes[:, np.newaxis]

    for row_idx, spec in enumerate(row_specs):
        for col_idx, size in enumerate(train_sizes):
            ax = axes[row_idx, col_idx]
            rows = read_csv(results_root / f"train_size={size}" / "metrics" / spec["csv_name"])
            epochs = [int(row["epoch"]) for row in rows]
            for label, column, color, marker in spec["columns"]:
                values = [as_float(row[column]) for row in rows]
                ax.plot(
                    epochs,
                    values,
                    label=label,
                    color=color,
                    linewidth=1.8,
                )

            if row_idx == 0:
                ax.set_title(f"train_size={size}")
            if col_idx == 0:
                ax.set_ylabel(f"{spec['title']}\n{spec['ylabel']}")
                ax.legend(frameon=False, fontsize=8, loc="best")
            if row_idx == len(row_specs) - 1:
                ax.set_xlabel("Epoch")
            style_axes(ax)

    fig.suptitle("Training and Validation Curves", y=1.02, fontsize=14)
    save_figure(fig, out_path)
    plt.close(fig)


def plot_e2e_metric_tradeoff(results_root, train_sizes, out_path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(train_sizes), figsize=(4.7 * len(train_sizes), 4.2), constrained_layout=True)
    if len(train_sizes) == 1:
        axes = [axes]

    for ax, size in zip(axes, train_sizes):
        rows = read_csv(results_root / f"train_size={size}" / "metrics" / "e2e_loss_curve.csv")
        xs = [as_float(row["validation_fy_loss"]) for row in rows]
        ys = [as_float(row["validation_decision_gap"]) for row in rows]
        epochs = [int(row["epoch"]) for row in rows]
        sc = ax.scatter(xs, ys, c=epochs, cmap="viridis", s=28, edgecolor="none")
        best_fy = min(rows, key=lambda row: as_float(row["validation_fy_loss"]))
        best_gap = min(rows, key=lambda row: as_float(row["validation_decision_gap"]))
        ax.scatter(as_float(best_fy["validation_fy_loss"]), as_float(best_fy["validation_decision_gap"]), marker="^", s=100, color=METHOD_COLORS[("e2e", "validation_fy_loss")], edgecolor="#222222", linewidth=0.7, label="best val FY")
        ax.scatter(as_float(best_gap["validation_fy_loss"]), as_float(best_gap["validation_decision_gap"]), marker="s", s=90, color=METHOD_COLORS[("e2e", "validation_decision_gap")], edgecolor="#222222", linewidth=0.7, label="best val gap")
        ax.set_title(f"train_size={size}")
        ax.set_xlabel("Validation FY objective")
        ax.set_ylabel("Validation decision gap")
        style_axes(ax)

    fig.colorbar(sc, ax=axes, label="Epoch", shrink=0.88)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=2, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("E2E Validation FY Objective vs Decision Gap", y=1.18, fontsize=14)
    save_figure(fig, out_path)
    plt.close(fig)


def graph_sort_key(path: Path):
    import re

    match = re.fullmatch(r"G-(\d+)\.json", path.name)
    if match:
        return (0, int(match.group(1)))
    return (1, path.name)


def graph_paths_for_epoch_dataset(dataset, split_path: Path):
    if dataset["graph_source"] == "split_test":
        from split_dataset import read_json

        split = read_json(split_path)
        return [entry["path"] for entry in split["test"]]
    if dataset["graph_source"] == "dataset_dir":
        dataset_dir = Path(dataset["dataset_dir"])
        paths = sorted(dataset_dir.glob("G-*.json"), key=graph_sort_key)
        if not paths:
            raise FileNotFoundError(f"No G-*.json files found in {dataset_dir}")
        return paths
    raise ValueError(f"Unknown graph source: {dataset['graph_source']}")


def epoch_diagnostics_csv_path(results_root: Path, train_size: int, dataset) -> Path:
    return (
        results_root
        / f"train_size={train_size}"
        / "metrics"
        / f"{dataset['epoch_output']}.csv"
    )


def summarize_test_evaluations(evaluations):
    gaps = np.asarray([row["gap"] for row in evaluations], dtype=float)
    normalized = np.asarray([row["normalized_gap"] for row in evaluations], dtype=float)
    ratios = np.asarray([row["ratio"] for row in evaluations], dtype=float)
    return {
        "test_mean_decision_gap": float(np.mean(gaps)),
        "test_mean_normalized_gap": float(np.mean(normalized)),
        "test_median_normalized_gap": float(np.median(normalized)),
        "test_mean_achieved_oracle_ratio": float(np.nanmean(ratios)),
    }


def compute_epoch_diagnostics_for_train_size(results_root, train_size, dataset, graphs, env):
    import step1b_common as common

    metrics_dir = results_root / f"train_size={train_size}" / "metrics"
    e2e_rows = read_csv(metrics_dir / "e2e_loss_curve.csv")
    output_rows = []
    for row_idx, row in enumerate(e2e_rows, start=1):
        theta = np.asarray([as_float(row["theta_1"]), as_float(row["theta_2"])], dtype=float)
        evaluations = common.evaluate_theta(theta, graphs, env)
        output_rows.append(
            {
                "epoch": int(row["epoch"]),
                "theta_1": float(theta[0]),
                "theta_2": float(theta[1]),
                "train_fy_loss": as_float(row["train_fy_loss"]),
                "validation_fy_loss": as_float(row["validation_fy_loss"]),
                "train_decision_gap": as_float(row["train_decision_gap"]),
                "validation_decision_gap": as_float(row["validation_decision_gap"]),
                **summarize_test_evaluations(evaluations),
            }
        )
        if row_idx == 1 or row_idx == len(e2e_rows) or row_idx % 10 == 0:
            print(
                f"  {dataset['key']} train_size={train_size}: "
                f"evaluated {row_idx}/{len(e2e_rows)} epochs",
                flush=True,
            )

    fieldnames = [
        "epoch",
        "theta_1",
        "theta_2",
        "train_fy_loss",
        "validation_fy_loss",
        "train_decision_gap",
        "validation_decision_gap",
        "test_mean_decision_gap",
        "test_mean_normalized_gap",
        "test_median_normalized_gap",
        "test_mean_achieved_oracle_ratio",
    ]
    output_path = epoch_diagnostics_csv_path(results_root, train_size, dataset)
    write_csv(output_path, output_rows, fieldnames=fieldnames)
    print(f"Saved {output_path}", flush=True)


def ensure_epoch_diagnostics(results_root, train_sizes, dataset, split_path, gurobi_seed, force=False):
    missing_sizes = [
        size
        for size in train_sizes
        if force or not epoch_diagnostics_csv_path(results_root, size, dataset).exists()
    ]
    if not missing_sizes:
        return

    import gurobipy as gp
    import step1b_common as common

    graph_paths = graph_paths_for_epoch_dataset(dataset, split_path)
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Seed", gurobi_seed)
    env.start()
    graphs = []
    try:
        print(
            f"Loading {dataset['key']} graphs for epoch diagnostics: "
            f"n={len(graph_paths)}",
            flush=True,
        )
        graphs = common.load_graph_records(graph_paths, env)
        for size in missing_sizes:
            compute_epoch_diagnostics_for_train_size(results_root, size, dataset, graphs, env)
    finally:
        common.dispose_graph_records(graphs)
        env.dispose()


def plot_epoch_diagnostics_dataset(results_root, train_sizes, dataset, out_path):
    import matplotlib.pyplot as plt

    panel_specs = [
        ("Train FY objective", "train_fy_loss"),
        ("Validation FY objective", "validation_fy_loss"),
        ("Validation decision gap", "validation_decision_gap"),
        ("Test decision gap", "test_mean_decision_gap"),
    ]
    colors = train_size_colors(train_sizes)
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.2), constrained_layout=True)
    axes = axes.ravel()

    for ax, (title, column) in zip(axes, panel_specs):
        for size in train_sizes:
            rows = read_csv(epoch_diagnostics_csv_path(results_root, size, dataset))
            epochs = [int(row["epoch"]) for row in rows]
            values = [as_float(row[column]) for row in rows]
            ax.plot(
                epochs,
                values,
                label=f"n={size}",
                color=colors[size],
                linewidth=1.9,
            )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        style_axes(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=len(train_sizes), frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle(
        f"{dataset['title']} Epoch Diagnostics ({dataset['subtitle']})",
        y=1.11,
        fontsize=14,
    )
    save_figure(fig, out_path)
    plt.close(fig)


def per_graph_csv_path(results_root: Path, train_size: int, dataset) -> Path:
    return results_root / f"train_size={train_size}" / "metrics" / dataset["per_graph_file"]


def plot_per_graph_gap_boxplots_dataset(results_root, train_sizes, dataset, out_path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1,
        len(train_sizes),
        figsize=(4.1 * len(train_sizes), 4.4),
        sharey=True,
        constrained_layout=True,
    )
    if len(train_sizes) == 1:
        axes = [axes]

    short_labels = {
        ("2stage", "validation_mse_loss"): "2stage\nval MSE",
        ("e2e", "validation_decision_gap"): "e2e\nval gap",
        ("e2e", "validation_fy_loss"): "e2e\nval FY",
    }

    for ax, size in zip(axes, train_sizes):
        path = per_graph_csv_path(results_root, size, dataset)
        if not path.exists():
            ax.set_title(f"train_size={size}\nmissing per-graph CSV")
            ax.axis("off")
            continue

        rows = read_csv(path)
        data = []
        colors = []
        tick_labels = []
        for method_key in METHOD_ORDER:
            values = [
                as_float(row["normalized_gap"])
                for row in rows
                if row_key(row) == method_key and row.get("normalized_gap") not in (None, "")
            ]
            values = [value for value in values if math.isfinite(value)]
            data.append(values)
            colors.append(METHOD_COLORS[method_key])
            tick_labels.append(short_labels[method_key])

        box = ax.boxplot(
            data,
            patch_artist=True,
            showmeans=True,
            showfliers=False,
            meanprops={
                "marker": "D",
                "markerfacecolor": "#ffffff",
                "markeredgecolor": "#222222",
                "markersize": 4.2,
            },
            medianprops={"color": "#111111", "linewidth": 1.2},
            whiskerprops={"color": "#555555", "linewidth": 1.0},
            capprops={"color": "#555555", "linewidth": 1.0},
        )
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.72)
            patch.set_edgecolor("#333333")

        ax.set_title(f"train_size={size}")
        ax.set_xticks(range(1, len(tick_labels) + 1))
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel("Selected checkpoint")
        style_axes(ax)

    axes[0].set_ylabel("Per-graph normalized decision gap")
    fig.suptitle(
        f"{dataset['title']} Per-graph Gap Distribution ({dataset['subtitle']})",
        y=1.08,
        fontsize=14,
    )
    save_figure(fig, out_path)
    plt.close(fig)


MODEL_WEIGHT_FILENAMES = {
    ("2stage", "validation_mse_loss"): "2stage_best_by_validation_mse_loss.npz",
    ("e2e", "validation_decision_gap"): "e2e_best_by_validation_decision_gap.npz",
    ("e2e", "validation_fy_loss"): "e2e_best_by_validation_fy_loss.npz",
}


def resolve_project_path(path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return PROJECT_ROOT / resolved


def model_weight_path(results_root: Path, train_size: int, method_key) -> Path:
    return (
        results_root
        / f"train_size={train_size}"
        / "model_weights"
        / MODEL_WEIGHT_FILENAMES[method_key]
    )


def load_theta(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=True) as data:
        return np.asarray(data["theta"], dtype=float)


def reward_calibration_csv_path(results_root: Path, train_size: int, dataset) -> Path:
    return (
        results_root
        / f"train_size={train_size}"
        / "metrics"
        / f"{dataset['calibration_output']}.csv"
    )


def calibration_bins(predicted, observed, n_bins):
    predicted = np.asarray(predicted, dtype=float)
    observed = np.asarray(observed, dtype=float)
    finite = np.isfinite(predicted) & np.isfinite(observed)
    predicted = predicted[finite]
    observed = observed[finite]
    if len(predicted) == 0:
        return []

    order = np.argsort(predicted)
    splits = np.array_split(order, min(n_bins, len(order)))
    rows = []
    for bin_index, indices in enumerate(splits):
        if len(indices) == 0:
            continue
        bin_pred = predicted[indices]
        bin_obs = observed[indices]
        rows.append(
            {
                "bin_index": bin_index,
                "bin_count": int(len(indices)),
                "mean_predicted_reward": float(np.mean(bin_pred)),
                "mean_synthetic_label_reward": float(np.mean(bin_obs)),
                "min_predicted_reward": float(np.min(bin_pred)),
                "max_predicted_reward": float(np.max(bin_pred)),
                "calibration_error": float(abs(np.mean(bin_obs) - np.mean(bin_pred))),
            }
        )
    return rows


def compute_reward_calibration_for_train_size(results_root, train_size, dataset, graph_paths, n_bins):
    import step1b_common as common

    step1a = common.load_step1a_module()
    model_thetas = {
        method_key: load_theta(model_weight_path(results_root, train_size, method_key))
        for method_key in METHOD_ORDER
    }
    predicted_by_method = {method_key: [] for method_key in METHOD_ORDER}
    observed_by_method = {method_key: [] for method_key in METHOD_ORDER}

    for path in graph_paths:
        graph = step1a.load_graph(resolve_project_path(path))
        features = step1a.feature_matrix(graph, common.PROBE)
        synthetic_weights = np.asarray(graph["w_true"], dtype=float)
        for method_key, theta in model_thetas.items():
            predicted_by_method[method_key].append(features @ theta)
            observed_by_method[method_key].append(synthetic_weights)

    output_rows = []
    for method_key in METHOD_ORDER:
        predicted = np.concatenate(predicted_by_method[method_key])
        observed = np.concatenate(observed_by_method[method_key])
        for row in calibration_bins(predicted, observed, n_bins):
            output_rows.append(
                {
                    "dataset_key": dataset["key"],
                    "train_size": train_size,
                    "method": method_key[0],
                    "selection_metric": method_key[1],
                    **row,
                }
            )

    fieldnames = [
        "dataset_key",
        "train_size",
        "method",
        "selection_metric",
        "bin_index",
        "bin_count",
        "mean_predicted_reward",
        "mean_synthetic_label_reward",
        "min_predicted_reward",
        "max_predicted_reward",
        "calibration_error",
    ]
    output_path = reward_calibration_csv_path(results_root, train_size, dataset)
    write_csv(output_path, output_rows, fieldnames=fieldnames)
    print(f"Saved {output_path}", flush=True)


def ensure_reward_calibration(results_root, train_sizes, dataset, split_path, n_bins, force=False):
    missing_sizes = [
        size
        for size in train_sizes
        if force or not reward_calibration_csv_path(results_root, size, dataset).exists()
    ]
    if not missing_sizes:
        return

    graph_paths = graph_paths_for_epoch_dataset(dataset, split_path)
    print(
        f"Computing {dataset['key']} reward calibration: "
        f"graphs={len(graph_paths)}, bins={n_bins}",
        flush=True,
    )
    for size in missing_sizes:
        compute_reward_calibration_for_train_size(
            results_root,
            size,
            dataset,
            graph_paths,
            n_bins,
        )


def plot_reward_calibration_dataset(results_root, train_sizes, dataset, out_path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1,
        len(train_sizes),
        figsize=(4.25 * len(train_sizes), 4.35),
        sharex=False,
        sharey=False,
        constrained_layout=True,
    )
    if len(train_sizes) == 1:
        axes = [axes]

    for ax, size in zip(axes, train_sizes):
        path = reward_calibration_csv_path(results_root, size, dataset)
        rows = read_csv(path)
        panel_values = []
        for method_key in METHOD_ORDER:
            method_rows = [
                row for row in rows
                if row_key(row) == method_key
            ]
            method_rows.sort(key=lambda row: int(row["bin_index"]))
            if not method_rows:
                continue
            xs = [as_float(row["mean_predicted_reward"]) for row in method_rows]
            ys = [as_float(row["mean_synthetic_label_reward"]) for row in method_rows]
            panel_values.extend(xs)
            panel_values.extend(ys)
            ax.plot(
                xs,
                ys,
                color=METHOD_COLORS[method_key],
                marker=METHOD_MARKERS[method_key],
                linewidth=1.8,
                markersize=4.5,
                label=METHOD_LABELS[method_key],
            )

        if panel_values:
            finite_values = [value for value in panel_values if math.isfinite(value)]
            lower = min(finite_values)
            upper = max(finite_values)
            pad = 0.05 * (upper - lower) if upper > lower else 1.0
            lower -= pad
            upper += pad
            ax.plot([lower, upper], [lower, upper], color="#888888", linestyle="--", linewidth=1.0)
            ax.set_xlim(lower, upper)
            ax.set_ylim(lower, upper)

        ax.set_title(f"train_size={size}")
        ax.set_xlabel("Mean predicted reward per bin")
        ax.set_ylabel("Mean synthetic label reward per bin")
        style_axes(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=3, frameon=False, bbox_to_anchor=(0.5, 1.07))
    fig.suptitle(
        f"{dataset['title']} Reward Calibration ({dataset['subtitle']})",
        y=1.13,
        fontsize=14,
    )
    save_figure(fig, out_path)
    plt.close(fig)


def write_manifest(out_dir: Path, written: list[Path]):
    lines = [
        "# Step1b Plot Results",
        "",
        "Generated from `remote_results/formal_M16_2stage500_e2e500_s10`.",
        "",
        "Each PNG also has a matching PDF file.",
        "",
    ]
    for path in written:
        lines.append(f"- `{path.name}`")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Plot Step1b summary figures.")
    parser.add_argument(
        "--results_root",
        default="surrogate_experiment_results/Step1b/remote_results/formal_M16_2stage500_e2e500_s10",
    )
    parser.add_argument(
        "--out_dir",
        default="surrogate_experiment_results/Step1b/plot_results",
    )
    parser.add_argument(
        "--split_path",
        default=str(DEFAULT_SPLIT_PATH),
        help="Step1b split file used for the held-out 400-graph test set.",
    )
    parser.add_argument("--gurobi_seed", type=int, default=42)
    parser.add_argument(
        "--force_epoch_diagnostics",
        action="store_true",
        help="Recompute cached *_epoch_diagnostics.csv files before plotting.",
    )
    parser.add_argument(
        "--skip_epoch_diagnostics",
        action="store_true",
        help="Skip the three epoch-diagnostics plots and any missing-cache computation.",
    )
    parser.add_argument(
        "--calibration_bins",
        type=int,
        default=12,
        help="Number of quantile bins for reward calibration plots.",
    )
    parser.add_argument(
        "--force_reward_calibration",
        action="store_true",
        help="Recompute cached *_reward_calibration.csv files before plotting.",
    )
    parser.add_argument(
        "--skip_reward_calibration",
        action="store_true",
        help="Skip reward calibration plots and any missing-cache computation.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 14,
            "savefig.bbox": "tight",
        }
    )

    args = parse_args(argv)
    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    split_path = Path(args.split_path)
    train_sizes = available_train_sizes(results_root)
    if not train_sizes:
        raise ValueError(f"No train_size directories found in {results_root}")

    rows_by_dataset = load_summary_rows(results_root, train_sizes)
    written = []

    output_specs = [
        (
            "01_normalized_decision_gap_by_dataset.png",
            lambda path: plot_metric_panels(
                rows_by_dataset,
                train_sizes,
                "test_mean_normalized_gap",
                "Mean normalized decision gap",
                "Held-out Decision Quality Across Training Set Sizes",
                path,
            ),
        ),
        (
            "02_raw_decision_gap_by_dataset.png",
            lambda path: plot_metric_panels(
                rows_by_dataset,
                train_sizes,
                "test_mean_decision_gap",
                "Mean oracle objective gap",
                "Raw Synthetic-label Decision Gap",
                path,
            ),
        ),
        (
            "03_achieved_oracle_ratio_by_dataset.png",
            lambda path: plot_metric_panels(
                rows_by_dataset,
                train_sizes,
                "test_mean_achieved_oracle_ratio",
                "Mean achieved / oracle objective",
                "Achieved Oracle Objective Ratio",
                path,
            ),
        ),
        (
            "04_paired_improvement_over_2stage.png",
            lambda path: plot_direct_method_performance_bars(rows_by_dataset, train_sizes, path),
        ),
        (
            "05_selected_parameter_endpoints.png",
            lambda path: plot_parameter_endpoints(rows_by_dataset, train_sizes, path),
        ),
        (
            "06_parameter_trajectories.png",
            lambda path: plot_theta_trajectories(results_root, train_sizes, rows_by_dataset, path),
        ),
        (
            "07_08_09_training_curves_combined.png",
            lambda path: plot_combined_training_curves(results_root, train_sizes, path),
        ),
        (
            "07_2stage_mse_loss_curves.png",
            lambda path: plot_loss_curves(
                results_root,
                train_sizes,
                "2stage_loss_curve.csv",
                [
                    ("train MSE", "train_mse_loss", "#2b2b2b", "o"),
                    ("validation MSE", "validation_mse_loss", "#0072b2", "s"),
                ],
                "MSE loss",
                "2stage Reward-fitting Loss Curves",
                path,
            ),
        ),
        (
            "08_e2e_fy_objective_curves.png",
            lambda path: plot_loss_curves(
                results_root,
                train_sizes,
                "e2e_loss_curve.csv",
                [
                    ("train FY objective", "train_fy_loss", "#2b2b2b", "o"),
                    ("validation FY objective", "validation_fy_loss", "#d55e00", "s"),
                ],
                "Perturbed FY objective",
                "E2E FY Objective Curves",
                path,
            ),
        ),
        (
            "09_e2e_decision_gap_curves.png",
            lambda path: plot_loss_curves(
                results_root,
                train_sizes,
                "e2e_loss_curve.csv",
                [
                    ("train decision gap", "train_decision_gap", "#2b2b2b", "o"),
                    ("validation decision gap", "validation_decision_gap", "#1f77b4", "s"),
                ],
                "Synthetic-label decision gap",
                "E2E Decision-gap Curves",
                path,
            ),
        ),
        (
            "10_e2e_validation_tradeoff.png",
            lambda path: plot_e2e_metric_tradeoff(results_root, train_sizes, path),
        ),
    ]

    for index, dataset in enumerate(DATASETS, start=11):
        output_specs.append(
            (
                f"{index:02d}_{dataset['key']}_gap_boxplots.png",
                lambda path, dataset=dataset: plot_per_graph_gap_boxplots_dataset(
                    results_root,
                    train_sizes,
                    dataset,
                    path,
                ),
            )
        )

    if not args.skip_epoch_diagnostics:
        for dataset in DATASETS:
            output_specs.append(
                (
                    f"{dataset['epoch_output']}.png",
                    lambda path, dataset=dataset: plot_epoch_diagnostics_dataset(
                        results_root,
                        train_sizes,
                        dataset,
                        path,
                    ),
                )
            )

    if not args.skip_reward_calibration:
        for index, dataset in enumerate(DATASETS, start=14):
            output_specs.append(
                (
                    f"{index:02d}_{dataset['key']}_reward_calibration.png",
                    lambda path, dataset=dataset: plot_reward_calibration_dataset(
                        results_root,
                        train_sizes,
                        dataset,
                        path,
                    ),
                )
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_epoch_diagnostics:
        for dataset in DATASETS:
            ensure_epoch_diagnostics(
                results_root,
                train_sizes,
                dataset,
                split_path=split_path,
                gurobi_seed=args.gurobi_seed,
                force=args.force_epoch_diagnostics,
            )

    if not args.skip_reward_calibration:
        for dataset in DATASETS:
            ensure_reward_calibration(
                results_root,
                train_sizes,
                dataset,
                split_path=split_path,
                n_bins=args.calibration_bins,
                force=args.force_reward_calibration,
            )

    for filename, plotter in output_specs:
        out_path = out_dir / filename
        plotter(out_path)
        written.append(out_path)
        print(f"Saved {out_path}")

    write_manifest(out_dir, written)
    print(f"Saved {out_dir / 'README.md'}")


if __name__ == "__main__":
    main()
