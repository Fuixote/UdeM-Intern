"""Create presentation plots from the current Step1b result archive."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


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
        "title": "Held-out noisy-linear test",
        "subtitle": "400 graphs",
    },
    {
        "key": "unseen1000",
        "summary_file": "unseen_test_summary.csv",
        "title": "Unseen noisy-linear test",
        "subtitle": "1000 graphs",
    },
    {
        "key": "realistic2000",
        "summary_file": "realistic_unseen_test_summary.csv",
        "title": "Realistic-synthetic stress test",
        "subtitle": "2000 graphs",
    },
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def plot_paired_improvement(rows_by_dataset, train_sizes, out_path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharex=True, constrained_layout=True)
    x_positions = np.arange(len(train_sizes))
    width = 0.34
    e2e_methods = [
        ("e2e", "validation_decision_gap"),
        ("e2e", "validation_fy_loss"),
    ]
    offsets = [-width / 2, width / 2]

    for ax, dataset in zip(axes, DATASETS):
        rows = rows_by_dataset[dataset["key"]]
        for offset, method_key in zip(offsets, e2e_methods):
            values = []
            lows = []
            highs = []
            for size in train_sizes:
                candidates = [
                    row
                    for row in rows
                    if row_key(row) == method_key and int(row["train_size"]) == size
                ]
                if not candidates:
                    values.append(math.nan)
                    lows.append(math.nan)
                    highs.append(math.nan)
                    continue
                row = candidates[0]
                value = as_float(row["paired_mean_improvement_over_2stage"])
                low = as_float(row["paired_mean_improvement_ci_low"])
                high = as_float(row["paired_mean_improvement_ci_high"])
                values.append(value)
                lows.append(value - low if not math.isnan(low) else 0.0)
                highs.append(high - value if not math.isnan(high) else 0.0)

            ax.bar(
                x_positions + offset,
                values,
                width=width,
                color=METHOD_COLORS[method_key],
                label=METHOD_LABELS[method_key],
                alpha=0.86,
            )
            ax.errorbar(
                x_positions + offset,
                values,
                yerr=[lows, highs],
                fmt="none",
                ecolor="#333333",
                elinewidth=1.1,
                capsize=3,
                alpha=0.85,
            )
        ax.axhline(0.0, color="#111111", linewidth=1.0)
        ax.set_title(f"{dataset['title']}\n{dataset['subtitle']}", fontsize=11)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(size) for size in train_sizes])
        ax.set_xlabel("Training graphs")
        style_axes(ax)

    axes[0].set_ylabel("Paired gap improvement over 2stage")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=2, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("E2E Paired Improvement vs 2stage", y=1.18, fontsize=14)
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
            lambda path: plot_paired_improvement(rows_by_dataset, train_sizes, path),
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

    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, plotter in output_specs:
        out_path = out_dir / filename
        plotter(out_path)
        written.append(out_path)
        print(f"Saved {out_path}")

    write_manifest(out_dir, written)
    print(f"Saved {out_dir / 'README.md'}")


if __name__ == "__main__":
    main()
