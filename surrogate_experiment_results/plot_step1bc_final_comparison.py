"""Final comparison plots that combine Step1b FY and Step1c SPO+ runs.

The script is intentionally read-only with respect to experiment run
directories. It gathers completed summaries/per-graph outputs from the matched
Step1b/Step1c val2000 protocol and writes publication-facing comparison plots
under ``surrogate_experiment_results/plot_results_step1bc`` by default.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_STEP1B_ROOT = (
    SCRIPT_DIR
    / "Step1b"
    / "remote_results"
    / "formal_M16_2stage500_e2e500_s10_val2000"
)
DEFAULT_STEP1C_ROOT = (
    SCRIPT_DIR
    / "Step1c"
    / "remote_results"
    / "formal_spoplus_ablation_val2000"
)
DEFAULT_OUT_DIR = SCRIPT_DIR / "plot_results_step1bc"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    title: str
    summary_files: tuple[str, ...]
    per_graph_files: tuple[str, ...]


@dataclass(frozen=True)
class ResultSource:
    name: str
    root: Path


METHOD_ORDER = [
    "2stage_val_mse",
    "fy_val_gap",
    "fy_val_fy",
    "spoplus_val_gap",
    "spoplus_val_spoplus",
]

METHOD_LABELS = {
    "2stage_val_mse": "2stage (val MSE)",
    "fy_val_gap": "FY (val gap)",
    "fy_val_fy": "FY (val FY)",
    "spoplus_val_gap": "SPO+ (val gap)",
    "spoplus_val_spoplus": "SPO+ (val SPO+)",
}

METHOD_COLORS = {
    "2stage_val_mse": "#3a3a3a",
    "fy_val_gap": "#1f77b4",
    "fy_val_fy": "#d55e00",
    "spoplus_val_gap": "#009e73",
    "spoplus_val_spoplus": "#cc79a7",
}

METHOD_MARKERS = {
    "2stage_val_mse": "o",
    "fy_val_gap": "s",
    "fy_val_fy": "^",
    "spoplus_val_gap": "D",
    "spoplus_val_spoplus": "v",
}


def default_dataset_specs(unseen_stem: str) -> list[DatasetSpec]:
    return [
        DatasetSpec(
            key="heldout400",
            title="Held-out noisy-linear test\n400 graphs",
            summary_files=("test_summary.csv",),
            per_graph_files=("test_per_graph.csv",),
        ),
        DatasetSpec(
            key=unseen_stem,
            title=f"Unseen noisy-linear test\n{unseen_stem.replace('unseen', '')} graphs",
            summary_files=(f"{unseen_stem}_summary.csv", "unseen_test_summary.csv"),
            per_graph_files=(f"{unseen_stem}_per_graph.csv", "unseen_test_per_graph.csv"),
        ),
    ]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: object) -> float:
    if value is None or value == "":
        return math.nan
    return float(value)


def as_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(float(value))


def available_train_sizes(*roots: Path) -> list[int]:
    sizes = set()
    for root in roots:
        if not root.exists():
            continue
        for child in root.iterdir():
            if child.is_dir() and child.name.startswith("train_size="):
                try:
                    sizes.add(int(child.name.split("=", 1)[1]))
                except ValueError:
                    continue
    return sorted(sizes)


def canonical_method_id(source_name: str, row: dict[str, str]) -> str | None:
    method = row.get("method", "")
    selection = row.get("selection_metric", "")

    if method == "2stage" and selection == "validation_mse_loss":
        return "2stage_val_mse"

    if source_name == "step1b" and method == "e2e":
        if selection == "validation_decision_gap":
            return "fy_val_gap"
        if selection == "validation_fy_loss":
            return "fy_val_fy"

    if source_name == "step1c" and method == "spoplus":
        if selection == "validation_decision_gap":
            return "spoplus_val_gap"
        if selection == "validation_spoplus_loss":
            return "spoplus_val_spoplus"

    return None


def source_priority(source_name: str, method_id: str) -> int:
    if method_id == "2stage_val_mse":
        return 0 if source_name == "step1b" else 1
    if method_id.startswith("fy_"):
        return 0 if source_name == "step1b" else 9
    if method_id.startswith("spoplus_"):
        return 0 if source_name == "step1c" else 9
    return 9


def first_existing_metrics_file(metrics_dir: Path, candidates: tuple[str, ...]) -> Path | None:
    for filename in candidates:
        path = metrics_dir / filename
        if path.exists():
            return path
    return None


def collect_summary_rows(
    sources: list[ResultSource],
    datasets: list[DatasetSpec],
    train_sizes: list[int],
) -> tuple[list[dict[str, str]], list[str]]:
    chosen: dict[tuple[str, int, str], tuple[int, dict[str, str]]] = {}
    warnings: list[str] = []

    for source in sources:
        if not source.root.exists():
            warnings.append(f"Missing source root: {source.root}")
            continue
        for train_size in train_sizes:
            metrics_dir = source.root / f"train_size={train_size}" / "metrics"
            if not metrics_dir.exists():
                continue
            for dataset in datasets:
                path = first_existing_metrics_file(metrics_dir, dataset.summary_files)
                if path is None:
                    continue
                for raw_row in read_csv(path):
                    method_id = canonical_method_id(source.name, raw_row)
                    if method_id is None:
                        continue
                    row = dict(raw_row)
                    row["source"] = source.name
                    row["source_root"] = str(source.root)
                    row["dataset_key"] = dataset.key
                    row["dataset_title"] = dataset.title
                    row["summary_file"] = str(path)
                    row["train_size"] = row.get("train_size") or str(train_size)
                    row["method_id"] = method_id
                    row["method_label"] = METHOD_LABELS[method_id]

                    key = (dataset.key, int(row["train_size"]), method_id)
                    priority = source_priority(source.name, method_id)
                    if key not in chosen or priority < chosen[key][0]:
                        chosen[key] = (priority, row)

    rows = [row for _, row in chosen.values()]
    rows.sort(
        key=lambda row: (
            row["dataset_key"],
            int(row["train_size"]),
            METHOD_ORDER.index(row["method_id"]),
        )
    )
    return rows, warnings


def collect_per_graph_rows(
    sources: list[ResultSource],
    datasets: list[DatasetSpec],
    train_sizes: list[int],
) -> tuple[list[dict[str, str]], list[str]]:
    chosen: dict[tuple[str, int, str], tuple[int, list[dict[str, str]]]] = {}
    warnings: list[str] = []

    for source in sources:
        if not source.root.exists():
            continue
        for train_size in train_sizes:
            metrics_dir = source.root / f"train_size={train_size}" / "metrics"
            if not metrics_dir.exists():
                continue
            for dataset in datasets:
                path = first_existing_metrics_file(metrics_dir, dataset.per_graph_files)
                if path is None:
                    continue
                grouped: dict[str, list[dict[str, str]]] = {}
                for raw_row in read_csv(path):
                    method_id = canonical_method_id(source.name, raw_row)
                    if method_id is None:
                        continue
                    row = dict(raw_row)
                    row["source"] = source.name
                    row["source_root"] = str(source.root)
                    row["dataset_key"] = dataset.key
                    row["dataset_title"] = dataset.title
                    row["per_graph_file"] = str(path)
                    row["train_size"] = row.get("train_size") or str(train_size)
                    row["method_id"] = method_id
                    row["method_label"] = METHOD_LABELS[method_id]
                    grouped.setdefault(method_id, []).append(row)

                for method_id, rows in grouped.items():
                    key = (dataset.key, train_size, method_id)
                    priority = source_priority(source.name, method_id)
                    if key not in chosen or priority < chosen[key][0]:
                        chosen[key] = (priority, rows)

    rows = []
    for _, group_rows in chosen.values():
        rows.extend(group_rows)
    rows.sort(
        key=lambda row: (
            row["dataset_key"],
            int(row["train_size"]),
            METHOD_ORDER.index(row["method_id"]),
            row.get("graph", ""),
        )
    )
    return rows, warnings


def bootstrap_mean_ci(
    values: Iterable[float],
    n_bootstrap: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    clean_values = [float(value) for value in values if math.isfinite(float(value))]
    if not clean_values:
        return math.nan, math.nan, math.nan
    array = np.asarray(clean_values, dtype=float)
    mean_value = float(np.mean(array))
    if array.size == 1 or n_bootstrap <= 0:
        return mean_value, mean_value, mean_value
    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        samples[index] = float(np.mean(array[rng.integers(0, array.size, size=array.size)]))
    alpha = 1.0 - confidence
    low, high = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])
    return mean_value, float(low), float(high)


def per_graph_stats(
    per_graph_rows: list[dict[str, str]],
    n_bootstrap: int,
    seed: int,
) -> dict[tuple[str, int, str], tuple[float, float, float, int]]:
    grouped: dict[tuple[str, int, str], list[float]] = {}
    for row in per_graph_rows:
        value = as_float(row.get("normalized_gap"))
        if not math.isfinite(value):
            continue
        key = (row["dataset_key"], int(row["train_size"]), row["method_id"])
        grouped.setdefault(key, []).append(value)

    stats = {}
    for index, (key, values) in enumerate(sorted(grouped.items())):
        mean_value, low, high = bootstrap_mean_ci(
            values,
            n_bootstrap=n_bootstrap,
            seed=seed + index,
        )
        stats[key] = (mean_value, low, high, len(values))
    return stats


def style_axes(ax) -> None:
    ax.grid(True, alpha=0.24, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")


def plot_mean_normalized_gap(
    summary_rows: list[dict[str, str]],
    per_graph_rows: list[dict[str, str]],
    datasets: list[DatasetSpec],
    train_sizes: list[int],
    out_path: Path,
    n_bootstrap: int,
    seed: int,
) -> None:
    import matplotlib.pyplot as plt

    stats = per_graph_stats(per_graph_rows, n_bootstrap=n_bootstrap, seed=seed)
    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(6.2 * len(datasets), 4.8),
        sharex=True,
    )
    if len(datasets) == 1:
        axes = [axes]

    x_positions = np.arange(len(train_sizes), dtype=float)
    bar_width = 0.12
    method_offsets = np.linspace(-0.30, 0.30, len(METHOD_ORDER))

    for ax, dataset in zip(axes, datasets):
        rows = [row for row in summary_rows if row["dataset_key"] == dataset.key]
        plotted = False
        for method_index, method_id in enumerate(METHOD_ORDER):
            positions = []
            ys = []
            yerr_low = []
            yerr_high = []
            for size_index, train_size in enumerate(train_sizes):
                candidates = [
                    row
                    for row in rows
                    if int(row["train_size"]) == train_size and row["method_id"] == method_id
                ]
                if not candidates:
                    continue
                value = as_float(candidates[0].get("test_mean_normalized_gap"))
                positions.append(x_positions[size_index] + method_offsets[method_index])
                ys.append(value)
                stat_key = (dataset.key, train_size, method_id)
                if stat_key in stats:
                    _, low, high, _ = stats[stat_key]
                    yerr_low.append(max(0.0, value - low))
                    yerr_high.append(max(0.0, high - value))
                else:
                    yerr_low.append(0.0)
                    yerr_high.append(0.0)

            if not positions:
                continue
            plotted = True
            ax.bar(
                positions,
                ys,
                width=bar_width,
                yerr=np.asarray([yerr_low, yerr_high], dtype=float),
                color=METHOD_COLORS[method_id],
                alpha=0.86,
                edgecolor=METHOD_COLORS[method_id],
                linewidth=0.8,
                error_kw={
                    "elinewidth": 1.3,
                    "capsize": 3.0,
                    "capthick": 1.3,
                    "ecolor": METHOD_COLORS[method_id],
                },
                label=METHOD_LABELS[method_id],
            )

        style_axes(ax)
        ax.set_title(dataset.title)
        ax.set_xlabel("Training graphs")
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(size) for size in train_sizes])
        if not plotted:
            ax.text(
                0.5,
                0.5,
                "No data yet",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="#777777",
            )

    axes[0].set_ylabel("Mean normalized decision gap\n(lower is better)")
    fig.suptitle(
        "Step1b FY vs Step1c SPO+ Mean Performance with 95% CI",
        fontsize=14,
        y=0.99,
    )
    handles = []
    labels = []
    seen_labels = set()
    for ax in axes:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if label in seen_labels:
                continue
            handles.append(handle)
            labels.append(label)
            seen_labels.add(label)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.925),
            ncol=5,
            frameon=False,
        )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.82])
    save_figure(fig, out_path)
    plt.close(fig)


def plot_mean_normalized_gap_no_error_bar(
    summary_rows: list[dict[str, str]],
    datasets: list[DatasetSpec],
    train_sizes: list[int],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(6.2 * len(datasets), 4.8),
        sharex=True,
    )
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        rows = [row for row in summary_rows if row["dataset_key"] == dataset.key]
        for method_id in METHOD_ORDER:
            xs = []
            ys = []
            for train_size in train_sizes:
                candidates = [
                    row
                    for row in rows
                    if int(row["train_size"]) == train_size and row["method_id"] == method_id
                ]
                if not candidates:
                    continue
                xs.append(train_size)
                ys.append(as_float(candidates[0].get("test_mean_normalized_gap")))

            if not xs:
                continue
            ax.plot(
                xs,
                ys,
                color=METHOD_COLORS[method_id],
                marker=METHOD_MARKERS[method_id],
                linewidth=2.2,
                markersize=6.0,
                label=METHOD_LABELS[method_id],
            )

        style_axes(ax)
        ax.set_title(dataset.title)
        ax.set_xlabel("Training graphs")
        ax.set_xticks(train_sizes)
        if not rows:
            ax.text(
                0.5,
                0.5,
                "No data yet",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="#777777",
            )

    axes[0].set_ylabel("Mean normalized decision gap\n(lower is better)")
    fig.suptitle(
        "Step1b FY vs Step1c SPO+ Mean Performance",
        fontsize=14,
        y=0.99,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.925),
            ncol=5,
            frameon=False,
        )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.82])
    save_figure(fig, out_path)
    plt.close(fig)


def plot_per_graph_boxplots(
    per_graph_rows: list[dict[str, str]],
    datasets: list[DatasetSpec],
    train_sizes: list[int],
    out_path: Path,
    show_fliers: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(6.4 * len(datasets), 4.8),
        sharey=False,
    )
    if len(datasets) == 1:
        axes = [axes]

    width = 0.13
    method_offsets = np.linspace(-0.28, 0.28, len(METHOD_ORDER))
    x_positions = np.arange(len(train_sizes), dtype=float)

    for ax, dataset in zip(axes, datasets):
        dataset_rows = [row for row in per_graph_rows if row["dataset_key"] == dataset.key]
        plotted = False
        for method_index, method_id in enumerate(METHOD_ORDER):
            values_by_size = []
            positions = []
            for size_index, train_size in enumerate(train_sizes):
                values = [
                    as_float(row.get("normalized_gap"))
                    for row in dataset_rows
                    if int(row["train_size"]) == train_size and row["method_id"] == method_id
                ]
                values = [value for value in values if math.isfinite(value)]
                if not values:
                    continue
                values_by_size.append(values)
                positions.append(x_positions[size_index] + method_offsets[method_index])
            if not values_by_size:
                continue

            plotted = True
            boxes = ax.boxplot(
                values_by_size,
                positions=positions,
                widths=width,
                patch_artist=True,
                showfliers=show_fliers,
                manage_ticks=False,
                medianprops={"color": "white", "linewidth": 1.1},
                whiskerprops={"color": METHOD_COLORS[method_id], "linewidth": 0.9},
                capprops={"color": METHOD_COLORS[method_id], "linewidth": 0.9},
            )
            for patch in boxes["boxes"]:
                patch.set_facecolor(METHOD_COLORS[method_id])
                patch.set_alpha(0.78)
                patch.set_edgecolor(METHOD_COLORS[method_id])

        style_axes(ax)
        ax.set_title(dataset.title)
        ax.set_xlabel("Training graphs")
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(size) for size in train_sizes])
        if not plotted:
            ax.text(
                0.5,
                0.5,
                "No per-graph data yet",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="#777777",
            )

    axes[0].set_ylabel("Per-graph normalized decision gap\n(lower is better)")
    handles = [
        plt.Line2D([0], [0], color=METHOD_COLORS[method_id], lw=6)
        for method_id in METHOD_ORDER
    ]
    fig.suptitle("Per-Graph Decision Gap Distribution", fontsize=14, y=0.99)
    fig.legend(
        handles,
        [METHOD_LABELS[method_id] for method_id in METHOD_ORDER],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=5,
        frameon=False,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.82])
    save_figure(fig, out_path)
    plt.close(fig)


def plot_checkpoint_epochs(
    summary_rows: list[dict[str, str]],
    train_sizes: list[int],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    heldout_rows = [row for row in summary_rows if row["dataset_key"] == "heldout400"]
    fig, ax = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
    for method_id in METHOD_ORDER:
        xs = []
        ys = []
        for train_size in train_sizes:
            candidates = [
                row
                for row in heldout_rows
                if int(row["train_size"]) == train_size and row["method_id"] == method_id
            ]
            if not candidates:
                continue
            epoch = as_int(candidates[0].get("selected_epoch"))
            if epoch is None:
                continue
            xs.append(train_size)
            ys.append(epoch)
        if xs:
            ax.plot(
                xs,
                ys,
                marker=METHOD_MARKERS[method_id],
                color=METHOD_COLORS[method_id],
                linewidth=2.0,
                label=METHOD_LABELS[method_id],
            )
    style_axes(ax)
    ax.set_title("Selected Checkpoint Epochs")
    ax.set_xlabel("Training graphs")
    ax.set_ylabel("Selected epoch")
    ax.set_xticks(train_sizes)
    ax.legend(frameon=False, ncol=2)
    save_figure(fig, out_path)
    plt.close(fig)


def plot_selected_theta(
    summary_rows: list[dict[str, str]],
    train_sizes: list[int],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    heldout_rows = [row for row in summary_rows if row["dataset_key"] == "heldout400"]
    fig, ax = plt.subplots(figsize=(7.2, 5.4), constrained_layout=True)
    marker_sizes = {
        train_size: 42 + 22 * index for index, train_size in enumerate(train_sizes)
    }
    for method_id in METHOD_ORDER:
        xs = []
        ys = []
        sizes = []
        labels = []
        for train_size in train_sizes:
            candidates = [
                row
                for row in heldout_rows
                if int(row["train_size"]) == train_size and row["method_id"] == method_id
            ]
            if not candidates:
                continue
            theta_1 = as_float(candidates[0].get("theta_1"))
            theta_2 = as_float(candidates[0].get("theta_2"))
            if not math.isfinite(theta_1) or not math.isfinite(theta_2):
                continue
            xs.append(theta_1)
            ys.append(theta_2)
            sizes.append(marker_sizes.get(train_size, 64))
            labels.append(train_size)

        if not xs:
            continue
        ax.scatter(
            xs,
            ys,
            s=sizes,
            color=METHOD_COLORS[method_id],
            marker=METHOD_MARKERS[method_id],
            alpha=0.82,
            label=METHOD_LABELS[method_id],
        )
        for theta_1, theta_2, train_size in zip(xs, ys, labels):
            ax.annotate(
                str(train_size),
                (theta_1, theta_2),
                textcoords="offset points",
                xytext=(4, 3),
                fontsize=7,
                color=METHOD_COLORS[method_id],
            )

    ax.scatter(
        [10.0],
        [5.0],
        marker="*",
        s=145,
        color="#f0c419",
        edgecolor="#6b5a00",
        linewidth=0.7,
        label="clean-signal coefficient [10,5]",
        zorder=5,
    )
    style_axes(ax)
    ax.set_title("Selected Linear Probe Parameters")
    ax.set_xlabel(r"$\theta_1$ utility coefficient")
    ax.set_ylabel(r"$\theta_2$ cPRA coefficient")
    ax.legend(frameon=False, fontsize=8)
    save_figure(fig, out_path)
    plt.close(fig)


def read_loss_curve(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return read_csv(path)


def preferred_curve_path(
    sources: list[ResultSource],
    train_size: int,
    filename: str,
    preferred_source: str,
) -> Path | None:
    ordered_sources = sorted(
        sources,
        key=lambda source: 0 if source.name == preferred_source else 1,
    )
    for source in ordered_sources:
        path = source.root / f"train_size={train_size}" / "metrics" / filename
        if path.exists():
            return path
    return None


def plot_loss_curves(
    sources: list[ResultSource],
    train_sizes: list[int],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    curve_specs = [
        {
            "title": "2stage reward fitting",
            "filename": "2stage_loss_curve.csv",
            "preferred_source": "step1b",
            "train_column": "train_mse_loss",
            "validation_column": "validation_mse_loss",
            "ylabel": "MSE loss",
        },
        {
            "title": "FY end-to-end surrogate",
            "filename": "e2e_loss_curve.csv",
            "preferred_source": "step1b",
            "train_column": "train_fy_loss",
            "validation_column": "validation_fy_loss",
            "ylabel": "FY objective",
        },
        {
            "title": "SPO+ end-to-end surrogate",
            "filename": "spoplus_loss_curve.csv",
            "preferred_source": "step1c",
            "train_column": "train_spoplus_loss",
            "validation_column": "validation_spoplus_loss",
            "ylabel": "SPO+ loss",
        },
    ]

    cmap = plt.get_cmap("viridis")
    if len(train_sizes) == 1:
        train_size_colors = {train_sizes[0]: cmap(0.55)}
    else:
        train_size_colors = {
            train_size: cmap(0.12 + 0.76 * idx / (len(train_sizes) - 1))
            for idx, train_size in enumerate(train_sizes)
        }

    fig, axes = plt.subplots(
        1,
        len(curve_specs),
        figsize=(5.2 * len(curve_specs), 4.6),
        sharex=False,
    )
    if len(curve_specs) == 1:
        axes = [axes]

    for ax, spec in zip(axes, curve_specs):
        plotted = False
        for train_size in train_sizes:
            path = preferred_curve_path(
                sources,
                train_size,
                spec["filename"],
                spec["preferred_source"],
            )
            if path is None:
                continue

            rows = read_loss_curve(path)
            epochs = [as_float(row.get("epoch")) for row in rows]
            train_values = [as_float(row.get(spec["train_column"])) for row in rows]
            validation_values = [
                as_float(row.get(spec["validation_column"])) for row in rows
            ]

            train_pairs = [
                (epoch, value)
                for epoch, value in zip(epochs, train_values)
                if math.isfinite(epoch) and math.isfinite(value)
            ]
            validation_pairs = [
                (epoch, value)
                for epoch, value in zip(epochs, validation_values)
                if math.isfinite(epoch) and math.isfinite(value)
            ]
            color = train_size_colors[train_size]
            if train_pairs:
                plotted = True
                ax.plot(
                    [epoch for epoch, _ in train_pairs],
                    [value for _, value in train_pairs],
                    color=color,
                    linestyle="--",
                    linewidth=1.35,
                    alpha=0.78,
                )
            if validation_pairs:
                plotted = True
                ax.plot(
                    [epoch for epoch, _ in validation_pairs],
                    [value for _, value in validation_pairs],
                    color=color,
                    linestyle="-",
                    linewidth=2.0,
                    label=f"n={train_size}",
                )

        style_axes(ax)
        ax.set_title(spec["title"])
        ax.set_xlabel("Epoch")
        ax.set_ylabel(spec["ylabel"])
        if not plotted:
            ax.text(
                0.5,
                0.5,
                "No curve data yet",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="#777777",
            )

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.91),
            ncol=min(4, len(handles)),
            frameon=False,
        )
    fig.suptitle("Training and Validation Loss Curves", fontsize=14, y=0.99)
    fig.text(
        0.5,
        0.02,
        "Dashed = train split; solid = validation split. Loss scales differ across panels.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=[0.0, 0.05, 1.0, 0.82])
    save_figure(fig, out_path)
    plt.close(fig)


def parse_train_sizes(raw: str | None, roots: list[Path]) -> list[int]:
    if raw:
        return [int(item.strip()) for item in raw.split(",") if item.strip()]
    sizes = available_train_sizes(*roots)
    return sizes or [50, 200, 600, 1200]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Combine Step1b FY and Step1c SPO+ result archives into final "
            "comparison plots. Missing train sizes or unseen-test files are skipped."
        )
    )
    parser.add_argument("--step1b_root", default=str(DEFAULT_STEP1B_ROOT))
    parser.add_argument("--step1c_root", default=str(DEFAULT_STEP1C_ROOT))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--train_sizes",
        default=None,
        help="Comma-separated train sizes. Default: union of train_size=* folders.",
    )
    parser.add_argument(
        "--unseen_stem",
        default="unseen10000",
        help=(
            "Output stem for the large unseen test files, e.g. unseen10000 for "
            "metrics/unseen10000_summary.csv and metrics/unseen10000_per_graph.csv."
        ),
    )
    parser.add_argument("--bootstrap_samples", type=int, default=1000)
    parser.add_argument("--bootstrap_seed", type=int, default=42)
    parser.add_argument(
        "--show_fliers",
        action="store_true",
        help="Show per-graph boxplot outliers. Hidden by default for readability.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    step1b_root = Path(args.step1b_root)
    step1c_root = Path(args.step1c_root)
    out_dir = Path(args.out_dir)
    train_sizes = parse_train_sizes(args.train_sizes, [step1b_root, step1c_root])
    datasets = default_dataset_specs(args.unseen_stem)
    sources = [
        ResultSource("step1b", step1b_root),
        ResultSource("step1c", step1c_root),
    ]

    summary_rows, summary_warnings = collect_summary_rows(sources, datasets, train_sizes)
    per_graph_rows, per_graph_warnings = collect_per_graph_rows(
        sources, datasets, train_sizes
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "combined_step1bc_summary.csv", summary_rows)
    write_csv(out_dir / "combined_step1bc_per_graph.csv", per_graph_rows)

    if not summary_rows:
        print("No summary rows found. Check result roots.", file=sys.stderr)
        for warning in summary_warnings + per_graph_warnings:
            print(f"warning: {warning}", file=sys.stderr)
        return 1

    plot_mean_normalized_gap_no_error_bar(
        summary_rows,
        datasets,
        train_sizes,
        out_dir / "figure0_mean_normalized_gap_no_error_bars.png",
    )
    plot_mean_normalized_gap(
        summary_rows,
        per_graph_rows,
        datasets,
        train_sizes,
        out_dir / "figure1_mean_normalized_gap_heldout400_unseen10000.png",
        n_bootstrap=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    plot_per_graph_boxplots(
        per_graph_rows,
        datasets,
        train_sizes,
        out_dir / "figure2_per_graph_gap_boxplots_heldout400_unseen10000.png",
        show_fliers=args.show_fliers,
    )
    plot_checkpoint_epochs(
        summary_rows,
        train_sizes,
        out_dir / "figure3_checkpoint_epochs.png",
    )
    plot_selected_theta(
        summary_rows,
        train_sizes,
        out_dir / "figure4_selected_theta_endpoints.png",
    )
    plot_loss_curves(
        sources,
        train_sizes,
        out_dir / "figure5_loss_curves.png",
    )

    if summary_warnings or per_graph_warnings:
        for warning in summary_warnings + per_graph_warnings:
            print(f"warning: {warning}", file=sys.stderr)

    print(f"summary rows: {len(summary_rows)}")
    print(f"per-graph rows: {len(per_graph_rows)}")
    print(f"wrote: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
