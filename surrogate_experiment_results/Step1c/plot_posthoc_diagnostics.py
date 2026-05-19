"""Post-hoc Step1c diagnostics from cached trajectory and per-graph metrics."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_RESULTS_ROOT = (
    SCRIPT_DIR
    / "remote_results"
    / "formal_spoplus_ablation_val2000"
)
DEFAULT_OUT_DIR = SCRIPT_DIR / "plot_results"

METHOD_ORDER = [
    ("2stage", "validation_mse_loss"),
    ("e2e", "validation_decision_gap"),
    ("e2e", "validation_fy_loss"),
]

METHOD_LABELS = {
    ("2stage", "validation_mse_loss"): "2stage (val MSE)",
    ("e2e", "validation_decision_gap"): "e2e (val gap)",
    ("e2e", "validation_fy_loss"): "e2e (val FY)",
}

METHOD_COLORS = {
    ("2stage", "validation_mse_loss"): "#2b2b2b",
    ("e2e", "validation_decision_gap"): "#1f77b4",
    ("e2e", "validation_fy_loss"): "#d55e00",
}

TRAIN_SIZE_COLORS = {
    50: "#4b2e83",
    200: "#2a6f84",
    600: "#25a970",
    1200: "#a6d82a",
}

EPOCH_DATASETS = [
    {
        "key": "validation",
        "title": "Validation split",
        "csv_name": "e2e_loss_curve.csv",
        "metric": "validation_decision_gap",
    },
    {
        "key": "heldout400",
        "title": "Held-out noisy-linear test",
        "csv_name": "heldout400_epoch_diagnostics.csv",
        "metric": "test_mean_decision_gap",
    },
    {
        "key": "unseen1000",
        "title": "Unseen noisy-linear test",
        "csv_name": "unseen1000_epoch_diagnostics.csv",
        "metric": "test_mean_decision_gap",
    },
    {
        "key": "realistic2000",
        "title": "Realistic-synthetic stress test",
        "csv_name": "realistic2000_epoch_diagnostics.csv",
        "metric": "test_mean_decision_gap",
    },
]

PER_GRAPH_DATASETS = [
    {
        "key": "heldout400",
        "title": "Held-out noisy-linear test",
        "csv_name": "test_per_graph.csv",
    },
    {
        "key": "unseen1000",
        "title": "Unseen noisy-linear test",
        "csv_name": "unseen_test_per_graph.csv",
    },
    {
        "key": "realistic2000",
        "title": "Realistic-synthetic stress test",
        "csv_name": "realistic_unseen_test_per_graph.csv",
    },
]

SELECTION_CRITERIA = [
    ("validation_decision_gap", "selected by val gap", "#1f77b4"),
    ("validation_fy_loss", "selected by val FY", "#d55e00"),
]


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


def as_float(value) -> float:
    if value is None or value == "":
        return math.nan
    return float(value)


def row_key(row: dict[str, str]) -> tuple[str, str]:
    return row["method"], row["selection_metric"]


def metric_values(rows: list[dict[str, str]], metric: str) -> list[float]:
    return [as_float(row.get(metric)) for row in rows]


def best_epoch(rows: list[dict[str, str]], metric: str) -> int:
    finite_rows = [
        row for row in rows
        if math.isfinite(as_float(row.get(metric)))
    ]
    if not finite_rows:
        raise ValueError(f"No finite values for metric {metric}")
    selected = min(finite_rows, key=lambda row: as_float(row[metric]))
    return int(float(selected["epoch"]))


def selection_suboptimality(
    rows: list[dict[str, str]],
    selected_epoch: int,
    metric: str,
) -> float:
    by_epoch = {int(float(row["epoch"])): row for row in rows}
    if selected_epoch not in by_epoch:
        raise ValueError(f"Selected epoch {selected_epoch} is absent from metric rows")
    values = [value for value in metric_values(rows, metric) if math.isfinite(value)]
    if not values:
        raise ValueError(f"No finite values for metric {metric}")
    selected_value = as_float(by_epoch[selected_epoch][metric])
    return selected_value - min(values)


def paired_gap_deltas(
    rows: list[dict[str, str]],
    candidate_key: tuple[str, str],
    baseline_key: tuple[str, str] = ("2stage", "validation_mse_loss"),
    value_column: str = "gap",
) -> list[float]:
    baseline = {
        row["graph"]: as_float(row[value_column])
        for row in rows
        if row_key(row) == baseline_key
    }
    deltas = []
    for row in rows:
        if row_key(row) != candidate_key:
            continue
        graph = row["graph"]
        if graph in baseline:
            deltas.append(baseline[graph] - as_float(row[value_column]))
    return deltas


def nonzero_mean(values, tolerance: float = 1e-12) -> float:
    nonzero = [float(value) for value in values if abs(float(value)) > tolerance]
    if not nonzero:
        return 0.0
    return float(np.mean(nonzero))


def available_train_sizes(results_root: Path) -> list[int]:
    sizes = []
    for child in results_root.iterdir():
        if child.is_dir() and child.name.startswith("train_size="):
            try:
                sizes.append(int(child.name.split("=", 1)[1]))
            except ValueError:
                continue
    return sorted(sizes)


def metrics_dir(results_root: Path, train_size: int) -> Path:
    return results_root / f"train_size={train_size}" / "metrics"


def load_e2e_rows(results_root: Path, train_size: int) -> list[dict[str, str]]:
    return read_csv(metrics_dir(results_root, train_size) / "e2e_loss_curve.csv")


def load_epoch_dataset_rows(results_root: Path, train_size: int, dataset) -> list[dict[str, str]]:
    return read_csv(metrics_dir(results_root, train_size) / dataset["csv_name"])


def load_per_graph_rows(results_root: Path, train_size: int, dataset) -> list[dict[str, str]]:
    rows = read_csv(metrics_dir(results_root, train_size) / dataset["csv_name"])
    output = []
    for row in rows:
        row = dict(row)
        row["train_size"] = str(train_size)
        row["dataset_key"] = dataset["key"]
        output.append(row)
    return output


def style_axes(ax):
    ax.grid(True, alpha=0.24, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    fig.savefig(out_path.with_suffix(".pdf"))


def plot_trajectory_dashboard(results_root: Path, train_size: int, out_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    e2e_rows = load_e2e_rows(results_root, train_size)
    heldout_rows = load_epoch_dataset_rows(results_root, train_size, EPOCH_DATASETS[1])
    unseen_rows = load_epoch_dataset_rows(results_root, train_size, EPOCH_DATASETS[2])
    realistic_rows = load_epoch_dataset_rows(results_root, train_size, EPOCH_DATASETS[3])

    selected_epochs = {
        "val gap": best_epoch(e2e_rows, "validation_decision_gap"),
        "val FY": best_epoch(e2e_rows, "validation_fy_loss"),
        "heldout oracle": best_epoch(heldout_rows, "test_mean_decision_gap"),
        "unseen oracle": best_epoch(unseen_rows, "test_mean_decision_gap"),
        "realistic oracle": best_epoch(realistic_rows, "test_mean_decision_gap"),
    }
    marker_styles = {
        "val gap": ("#1f77b4", "--", 1.4),
        "val FY": ("#d55e00", "--", 1.4),
        "heldout oracle": ("#009e73", ":", 1.1),
        "unseen oracle": ("#6a3d9a", ":", 1.1),
        "realistic oracle": ("#cc79a7", ":", 1.1),
    }

    epochs = [int(float(row["epoch"])) for row in e2e_rows]
    fig, axes = plt.subplots(3, 1, figsize=(10.2, 9.0), sharex=True, constrained_layout=True)

    axes[0].plot(
        epochs,
        metric_values(e2e_rows, "train_fy_loss"),
        label="train FY objective",
        color="#333333",
        linewidth=1.9,
    )
    axes[0].plot(
        epochs,
        metric_values(e2e_rows, "validation_fy_loss"),
        label="validation FY objective",
        color="#d55e00",
        linewidth=1.9,
    )
    axes[0].set_ylabel("FY objective")
    axes[0].set_title("Surrogate objective")

    axes[1].plot(
        epochs,
        metric_values(e2e_rows, "train_decision_gap"),
        label="train decision gap",
        color="#333333",
        linewidth=1.8,
    )
    axes[1].plot(
        epochs,
        metric_values(e2e_rows, "validation_decision_gap"),
        label="validation decision gap",
        color="#1f77b4",
        linewidth=1.8,
    )
    axes[1].plot(
        epochs,
        metric_values(heldout_rows, "test_mean_decision_gap"),
        label="heldout400 test gap",
        color="#009e73",
        linewidth=1.8,
    )
    axes[1].set_ylabel("Decision gap")
    axes[1].set_title("In-distribution decision quality")

    axes[2].plot(
        epochs,
        metric_values(unseen_rows, "test_mean_decision_gap"),
        label="unseen1000 test gap",
        color="#6a3d9a",
        linewidth=1.8,
    )
    axes[2].plot(
        epochs,
        metric_values(realistic_rows, "test_mean_decision_gap"),
        label="realistic2000 test gap",
        color="#cc79a7",
        linewidth=1.8,
    )
    axes[2].set_ylabel("Decision gap")
    axes[2].set_xlabel("Epoch")
    axes[2].set_title("Out-of-split and label-shift robustness")

    for ax in axes:
        for label, epoch in selected_epochs.items():
            color, linestyle, linewidth = marker_styles[label]
            ax.axvline(epoch, color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.75)
        style_axes(ax)
        ax.legend(frameon=False, fontsize=9, loc="best")

    handles = [
        plt.Line2D([0], [0], color=color, linestyle=linestyle, linewidth=linewidth, label=f"{label}: epoch {selected_epochs[label]}")
        for label, (color, linestyle, linewidth) in marker_styles.items()
    ]
    fig.legend(handles=handles, loc="upper center", ncols=3, frameon=False, bbox_to_anchor=(0.5, 1.04), fontsize=9)
    fig.suptitle(f"Step1c Trajectory Diagnostic Dashboard (train_size={train_size})", y=1.08, fontsize=14)

    out_path = out_dir / f"17_trajectory_dashboard_train_size_{train_size}.png"
    save_figure(fig, out_path)
    plt.close(fig)
    return out_path


def compute_selection_suboptimality_rows(results_root: Path, train_sizes: list[int]):
    rows = []
    for train_size in train_sizes:
        e2e_rows = load_e2e_rows(results_root, train_size)
        selected_epochs = {
            criterion: best_epoch(e2e_rows, criterion)
            for criterion, _, _ in SELECTION_CRITERIA
        }
        for dataset in EPOCH_DATASETS:
            metric_rows = load_epoch_dataset_rows(results_root, train_size, dataset)
            metric = dataset["metric"]
            oracle_epoch = best_epoch(metric_rows, metric)
            oracle_value = min(
                value for value in metric_values(metric_rows, metric)
                if math.isfinite(value)
            )
            for criterion, label, _ in SELECTION_CRITERIA:
                selected_epoch = selected_epochs[criterion]
                by_epoch = {int(float(row["epoch"])): row for row in metric_rows}
                selected_value = as_float(by_epoch[selected_epoch][metric])
                rows.append(
                    {
                        "train_size": train_size,
                        "dataset_key": dataset["key"],
                        "dataset_title": dataset["title"],
                        "selection_metric": criterion,
                        "selection_label": label,
                        "selected_epoch": selected_epoch,
                        "oracle_epoch": oracle_epoch,
                        "selected_value": selected_value,
                        "oracle_value": oracle_value,
                        "selection_suboptimality": selected_value - oracle_value,
                    }
                )
    return rows


def plot_selection_suboptimality(rows, train_sizes: list[int], out_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.4), sharex=True, constrained_layout=True)
    axes = axes.ravel()
    width = 0.34
    x_positions = np.arange(len(train_sizes))
    offsets = [-width / 2, width / 2]

    for ax, dataset in zip(axes, EPOCH_DATASETS):
        for offset, (criterion, label, color) in zip(offsets, SELECTION_CRITERIA):
            values = []
            for train_size in train_sizes:
                matches = [
                    row for row in rows
                    if int(row["train_size"]) == train_size
                    and row["dataset_key"] == dataset["key"]
                    and row["selection_metric"] == criterion
                ]
                values.append(float(matches[0]["selection_suboptimality"]) if matches else math.nan)
            ax.bar(x_positions + offset, values, width=width, color=color, alpha=0.86, label=label)
        ax.axhline(0.0, color="#222222", linewidth=0.9)
        ax.set_title(dataset["title"])
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(size) for size in train_sizes])
        ax.set_xlabel("Training graphs")
        ax.set_ylabel("Selection suboptimality")
        style_axes(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=2, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Checkpoint Selection Suboptimality vs Best Evaluated Epoch", y=1.09, fontsize=14)

    out_path = out_dir / "18_selection_suboptimality.png"
    save_figure(fig, out_path)
    plt.close(fig)
    return out_path


def per_graph_summary_rows(results_root: Path, train_sizes: list[int]):
    output_rows = []
    for dataset in PER_GRAPH_DATASETS:
        for train_size in train_sizes:
            rows = load_per_graph_rows(results_root, train_size, dataset)
            for method_key in METHOD_ORDER:
                values = [
                    as_float(row["normalized_gap"])
                    for row in rows
                    if row_key(row) == method_key
                ]
                if not values:
                    continue
                output_rows.append(
                    {
                        "dataset_key": dataset["key"],
                        "train_size": train_size,
                        "method": method_key[0],
                        "selection_metric": method_key[1],
                        "mean_normalized_gap": float(np.mean(values)),
                        "median_normalized_gap": float(np.median(values)),
                        "zero_gap_fraction": float(np.mean(np.isclose(values, 0.0))),
                        "nonzero_mean_normalized_gap": nonzero_mean(values),
                    }
                )
    return output_rows


def plot_per_graph_diagnostics_dataset(results_root: Path, train_sizes: list[int], dataset, out_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    all_rows = []
    for train_size in train_sizes:
        all_rows.extend(load_per_graph_rows(results_root, train_size, dataset))

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.3), constrained_layout=True)
    axes = axes.ravel()

    bins = np.linspace(
        0.0,
        max(as_float(row["normalized_gap"]) for row in all_rows) if all_rows else 1.0,
        35,
    )
    if len(bins) < 2 or bins[-1] == bins[0]:
        bins = np.linspace(0.0, 1.0, 35)

    for method_key in METHOD_ORDER:
        values = [
            as_float(row["normalized_gap"])
            for row in all_rows
            if row_key(row) == method_key
        ]
        axes[0].hist(
            values,
            bins=bins,
            histtype="step",
            linewidth=1.8,
            density=True,
            color=METHOD_COLORS[method_key],
            label=METHOD_LABELS[method_key],
        )
    axes[0].set_title("Normalized gap histogram by method")
    axes[0].set_xlabel("Per-graph normalized decision gap")
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=False, fontsize=8)
    style_axes(axes[0])

    for method_key in METHOD_ORDER[1:]:
        values = []
        for train_size in train_sizes:
            rows = [row for row in all_rows if int(row["train_size"]) == train_size]
            values.extend(
                paired_gap_deltas(rows, method_key, value_column="normalized_gap")
            )
        axes[1].hist(
            values,
            bins=35,
            histtype="stepfilled",
            alpha=0.35,
            color=METHOD_COLORS[method_key],
            label=METHOD_LABELS[method_key],
        )
    axes[1].axvline(0.0, color="#222222", linewidth=1.0)
    axes[1].set_title("Paired improvement over 2stage")
    axes[1].set_xlabel("2stage gap - e2e gap")
    axes[1].set_ylabel("Graph-count")
    axes[1].legend(frameon=False, fontsize=8)
    style_axes(axes[1])

    for train_size in train_sizes:
        rows = [row for row in all_rows if int(row["train_size"]) == train_size]
        gap_by_graph = {
            row["graph"]: as_float(row["normalized_gap"])
            for row in rows
            if row_key(row) == ("e2e", "validation_decision_gap")
        }
        fy_by_graph = {
            row["graph"]: as_float(row["normalized_gap"])
            for row in rows
            if row_key(row) == ("e2e", "validation_fy_loss")
        }
        common_graphs = sorted(set(gap_by_graph) & set(fy_by_graph))
        axes[2].scatter(
            [gap_by_graph[graph] for graph in common_graphs],
            [fy_by_graph[graph] for graph in common_graphs],
            s=10,
            alpha=0.28,
            color=TRAIN_SIZE_COLORS.get(train_size, "#555555"),
            label=f"n={train_size}",
        )
    xlim = axes[2].get_xlim()
    ylim = axes[2].get_ylim()
    lower = min(xlim[0], ylim[0])
    upper = max(xlim[1], ylim[1])
    axes[2].plot([lower, upper], [lower, upper], color="#777777", linestyle="--", linewidth=1.0)
    axes[2].set_xlim(lower, upper)
    axes[2].set_ylim(lower, upper)
    axes[2].set_title("E2E val-gap vs val-FY checkpoints")
    axes[2].set_xlabel("e2e selected by val gap")
    axes[2].set_ylabel("e2e selected by val FY")
    axes[2].legend(frameon=False, fontsize=8)
    style_axes(axes[2])

    width = 0.24
    x_positions = np.arange(len(train_sizes))
    offsets = [-width, 0.0, width]
    for offset, method_key in zip(offsets, METHOD_ORDER):
        values = []
        for train_size in train_sizes:
            rows = [
                row for row in all_rows
                if int(row["train_size"]) == train_size and row_key(row) == method_key
            ]
            values.append(nonzero_mean(as_float(row["normalized_gap"]) for row in rows))
        axes[3].bar(
            x_positions + offset,
            values,
            width=width,
            color=METHOD_COLORS[method_key],
            alpha=0.86,
            label=METHOD_LABELS[method_key],
        )
    axes[3].set_title("Conditional mean on nonzero-gap graphs")
    axes[3].set_xticks(x_positions)
    axes[3].set_xticklabels([str(size) for size in train_sizes])
    axes[3].set_xlabel("Training graphs")
    axes[3].set_ylabel("Mean normalized gap | gap > 0")
    axes[3].legend(frameon=False, fontsize=8)
    style_axes(axes[3])

    fig.suptitle(f"{dataset['title']} Per-graph Diagnostics", y=1.03, fontsize=14)
    out_path = out_dir / f"19_{dataset['key']}_per_graph_diagnostics.png"
    save_figure(fig, out_path)
    plt.close(fig)
    return out_path


def write_manifest_append(out_dir: Path, written: list[Path]):
    manifest = out_dir / "posthoc_diagnostics_README.md"
    lines = [
        "# Step1c Post-hoc Diagnostic Plots",
        "",
        "Generated by `surrogate_experiment_results/Step1c/plot_posthoc_diagnostics.py`.",
        "",
        "These plots are diagnostic only. They do not define checkpoint selection.",
        "",
        "Files:",
        "",
    ]
    for path in written:
        lines.append(f"- `{path.name}`")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Plot Step1c post-hoc diagnostics.")
    parser.add_argument("--results_root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
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
    train_sizes = available_train_sizes(results_root)
    if not train_sizes:
        raise ValueError(f"No train_size directories found in {results_root}")

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for train_size in train_sizes:
        path = plot_trajectory_dashboard(results_root, train_size, out_dir)
        written.append(path)
        print(f"Saved {path}")

    selection_rows = compute_selection_suboptimality_rows(results_root, train_sizes)
    selection_csv = out_dir / "18_selection_suboptimality.csv"
    write_csv(selection_csv, selection_rows)
    path = plot_selection_suboptimality(selection_rows, train_sizes, out_dir)
    written.append(path)
    print(f"Saved {selection_csv}")
    print(f"Saved {path}")

    summary_rows = per_graph_summary_rows(results_root, train_sizes)
    summary_csv = out_dir / "19_per_graph_diagnostics_summary.csv"
    write_csv(summary_csv, summary_rows)
    print(f"Saved {summary_csv}")

    for dataset in PER_GRAPH_DATASETS:
        path = plot_per_graph_diagnostics_dataset(results_root, train_sizes, dataset, out_dir)
        written.append(path)
        print(f"Saved {path}")

    manifest = write_manifest_append(out_dir, written)
    print(f"Saved {manifest}")


if __name__ == "__main__":
    main()
