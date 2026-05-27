#!/usr/bin/env python3
"""Generate Step2 summary and diagnostic figures.

The script is intentionally summary-first: figures 01-08 use root-level summary
tables and dataset diagnostics. Optional per-graph and trajectory figures are
generated only when the local heavy artifacts are present.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D


PRIMARY_CHECKPOINTS = (
    "2stage_val_mse",
    "fy_val_fy_loss",
    "spoplus_val_spoplus_loss",
)

HELDOUT_PRIMARY_COLUMNS = {
    "2stage_val_mse": "2stage_val_mse",
    "fy_val_fy_loss": "fy_val_fy",
    "spoplus_val_spoplus_loss": "spoplus_val_spoplus",
}

CHECKPOINT_ORDER = (
    "2stage_val_mse",
    "fy_val_decision_gap",
    "fy_val_fy_loss",
    "spoplus_val_decision_gap",
    "spoplus_val_spoplus_loss",
)

METHOD_LABELS = {
    "2stage_val_mse": "2stage (val MSE)",
    "fy_val_decision_gap": "FY (val gap)",
    "fy_val_fy_loss": "FY (val FY)",
    "spoplus_val_decision_gap": "SPO+ (val gap)",
    "spoplus_val_spoplus_loss": "SPO+ (val SPO+)",
}

METHOD_COLORS = {
    "2stage_val_mse": "#4c4c4c",
    "fy_val_decision_gap": "#1f77b4",
    "fy_val_fy_loss": "#e07000",
    "spoplus_val_decision_gap": "#009e73",
    "spoplus_val_spoplus_loss": "#cc79a7",
}

TRAIN_SIZES = (50, 200, 600, 1200)
DEGREES = (1, 2, 4, 8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary_csv",
        type=Path,
        default=Path("surrogate_experiment_results/Step2/step2_unseen10000_all_checkpoints_summary.csv"),
    )
    parser.add_argument(
        "--heldout_csv",
        type=Path,
        default=Path("surrogate_experiment_results/Step2/step2_heldout400_primary_summary.csv"),
    )
    parser.add_argument(
        "--label_diagnostics_root",
        type=Path,
        default=Path("dataset/processed"),
    )
    parser.add_argument(
        "--step2_root",
        type=Path,
        default=Path("surrogate_experiment_results/Step2"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("surrogate_experiment_results/Step2/plot_results"),
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        choices=["png", "pdf", "svg"],
        help="Figure formats to write.",
    )
    return parser.parse_args()


def parse_float(value: object, default: float = math.nan) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if text in {"", "-", "None", "nan"}:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def parse_int(value: object, default: int | None = None) -> int | None:
    number = parse_float(value)
    if math.isnan(number):
        return default
    return int(number)


def regime_sort_key(regime: str) -> tuple[int, int, str]:
    if regime.startswith("step2a_"):
        return (0, 0, regime)
    if regime.startswith("step2b_"):
        return (1, extract_degree(regime) or 0, regime)
    if regime.startswith("step2c_"):
        return (2, extract_degree(regime) or 0, regime)
    return (9, 0, regime)


def extract_degree(text: str) -> int | None:
    for part in text.split("_"):
        if part.startswith("d") and part[1:].isdigit():
            return int(part[1:])
    return None


def display_regime(regime: str) -> str:
    if regime.startswith("step2a_"):
        return "Step2a"
    if regime.startswith("step2b_"):
        return f"B d{extract_degree(regime)}"
    if regime.startswith("step2c_"):
        return f"C d{extract_degree(regime)}"
    return regime


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_summary(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in load_csv_rows(path):
        normalized = dict(row)
        normalized["train_size_int"] = parse_int(row.get("train_size"), 0)
        normalized["degree_int"] = parse_int(row.get("degree"))
        normalized["test_mean_normalized_gap_float"] = parse_float(row.get("test_mean_normalized_gap"))
        normalized["test_mean_decision_gap_float"] = parse_float(row.get("test_mean_decision_gap"))
        normalized["paired_mean_improvement_float"] = parse_float(row.get("paired_mean_improvement_over_2stage"))
        normalized["paired_ci_low_float"] = parse_float(row.get("paired_mean_improvement_ci_low"))
        normalized["paired_ci_high_float"] = parse_float(row.get("paired_mean_improvement_ci_high"))
        normalized["selected_epoch_float"] = parse_float(row.get("selected_epoch"))
        normalized["theta_1_float"] = parse_float(row.get("theta_1"))
        normalized["theta_2_float"] = parse_float(row.get("theta_2"))
        theta_1 = normalized["theta_1_float"]
        theta_2 = normalized["theta_2_float"]
        normalized["theta_norm_float"] = (
            math.sqrt(theta_1 * theta_1 + theta_2 * theta_2)
            if not math.isnan(theta_1) and not math.isnan(theta_2)
            else math.nan
        )
        rows.append(normalized)
    return rows


def load_heldout_primary(path: Path) -> list[dict[str, object]]:
    """Convert heldout400 primary wide summary rows into checkpoint-level rows."""
    rows: list[dict[str, object]] = []
    for row in load_csv_rows(path):
        for checkpoint, column in HELDOUT_PRIMARY_COLUMNS.items():
            normalized = dict(row)
            normalized["checkpoint_label"] = checkpoint
            normalized["train_size_int"] = parse_int(row.get("train_size"), 0)
            normalized["degree_int"] = parse_int(row.get("degree"))
            normalized["heldout_mean_normalized_gap_float"] = parse_float(row.get(column))
            rows.append(normalized)
    return rows


def best_primary_checkpoint_by_setting(rows: list[dict[str, object]]) -> dict[tuple[str, int], dict[str, object]]:
    """Return the lowest normalized-gap primary checkpoint for each regime/train size."""
    winners: dict[tuple[str, int], dict[str, object]] = {}
    for row in rows:
        checkpoint = row.get("checkpoint_label")
        if checkpoint not in PRIMARY_CHECKPOINTS:
            continue
        regime = str(row.get("regime"))
        train_size = row.get("train_size_int")
        if not isinstance(train_size, int):
            train_size = parse_int(row.get("train_size"))
        if train_size is None:
            continue
        gap = row.get("test_mean_normalized_gap_float")
        if isinstance(gap, str):
            gap = parse_float(gap)
        if gap is None or math.isnan(float(gap)):
            continue
        key = (regime, int(train_size))
        current = winners.get(key)
        current_gap = current.get("test_mean_normalized_gap_float") if current else math.inf
        if isinstance(current_gap, str):
            current_gap = parse_float(current_gap)
        if current is None or float(gap) < float(current_gap):
            winners[key] = row
    return winners


def split_from_dataset_name(name: str) -> str | None:
    for split in ("main2000", "val2000", "unseen10000"):
        if f"_{split}_" in name:
            return split
    return None


def regime_from_dataset_name(name: str) -> str | None:
    for suffix in ("_main2000_seed", "_val2000_seed", "_unseen10000_seed"):
        if suffix in name:
            return name.split(suffix)[0]
    return None


def load_label_diagnostics(root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(root.glob("step2*/label_diagnostics.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        dataset = data.get("dataset", {})
        name = dataset.get("name") or path.parent.name
        regime = regime_from_dataset_name(name)
        split = split_from_dataset_name(name)
        if not regime or not split:
            continue
        labels = data.get("labels", {}).get("ground_truth_label", {})
        latent = data.get("labels", {}).get("latent_clean_linear_label", {})
        corr = data.get("correlations", {}).get("latent_clean_linear_label_vs_ground_truth_label")
        records.append(
            {
                "path": path,
                "dataset": name,
                "regime": regime,
                "split": split,
                "block": "Step2a" if regime.startswith("step2a") else "Step2b" if regime.startswith("step2b") else "Step2c",
                "degree": extract_degree(regime),
                "graph_count": dataset.get("graph_count"),
                "edge_count": dataset.get("edge_count"),
                "label_mean": parse_float(labels.get("mean")),
                "label_std": parse_float(labels.get("std")),
                "label_fraction_zero": parse_float(labels.get("fraction_zero")),
                "clean_mean": parse_float(latent.get("mean")),
                "clean_std": parse_float(latent.get("std")),
                "clean_corr": parse_float(corr),
            }
        )
    return records


def load_graph_diagnostics(root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(root.glob("step2*/label_graph_diagnostics.csv")):
        dataset = path.parent.name
        regime = regime_from_dataset_name(dataset)
        split = split_from_dataset_name(dataset)
        if not regime or not split:
            continue
        for row in load_csv_rows(path):
            records.append(
                {
                    "regime": regime,
                    "split": split,
                    "graph": row.get("file"),
                    "label_to_clean_mean_ratio": parse_float(row.get("label_to_clean_mean_ratio")),
                    "fraction_zero_label": parse_float(row.get("fraction_zero_label")),
                }
            )
    return records


def group_values(rows: Iterable[dict[str, object]], key_fields: tuple[str, ...], value_field: str) -> dict[tuple[object, ...], list[float]]:
    groups: dict[tuple[object, ...], list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(value_field)
        if isinstance(value, str):
            value = parse_float(value)
        if value is None or math.isnan(float(value)):
            continue
        key = tuple(row.get(field) for field in key_fields)
        groups[key].append(float(value))
    return groups


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else math.nan


def compute_selector_deltas(
    rows: list[dict[str, object]],
    direct_label: str,
    surrogate_label: str,
) -> list[dict[str, object]]:
    groups: dict[tuple[object, object], dict[str, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        label = row.get("checkpoint_label")
        if label in {direct_label, surrogate_label}:
            groups[(row.get("regime"), row.get("train_size"))][str(label)] = row

    deltas: list[dict[str, object]] = []
    for (regime, train_size), pair in groups.items():
        if direct_label not in pair or surrogate_label not in pair:
            continue
        direct_gap = parse_float(pair[direct_label].get("test_mean_normalized_gap"))
        surrogate_gap = parse_float(pair[surrogate_label].get("test_mean_normalized_gap"))
        if math.isnan(direct_gap) or math.isnan(surrogate_gap):
            continue
        deltas.append(
            {
                "regime": regime,
                "train_size": train_size,
                "selector_delta": direct_gap - surrogate_gap,
            }
        )
    return deltas


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, formats: Iterable[str]) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        written.append(str(path))
    plt.close(fig)
    return written


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def plot_label_std_corr(records: list[dict[str, object]], out_dir: Path, formats: list[str]) -> list[str]:
    main = [r for r in records if r["split"] == "main2000"]
    refs = [r for r in main if r["block"] == "Step2a"]
    ref_std = refs[0]["label_std"] if refs else math.nan
    ref_corr = refs[0]["clean_corr"] if refs else math.nan

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
    for panel_idx, (ax, block, title) in enumerate(zip(axes, ("Step2b", "Step2c"), ("Step2b: polynomial", "Step2c: polynomial + noise"))):
        block_rows = sorted([r for r in main if r["block"] == block], key=lambda r: r["degree"] or 0)
        x = [r["degree"] for r in block_rows]
        std = [r["label_std"] for r in block_rows]
        corr = [r["clean_corr"] for r in block_rows]
        ax.plot(x, std, marker="o", color="#4c78a8", label="label std")
        if not math.isnan(ref_std):
            ax.axhline(ref_std, color="#4c78a8", linestyle=":", alpha=0.6, label="Step2a std")
        ax.set_title(title)
        ax.set_xlabel("Polynomial degree")
        ax.set_ylabel("Label std")
        ax.set_xticks(DEGREES)
        ax2 = ax.twinx()
        ax2.plot(x, corr, marker="s", color="#e45756", label="corr(clean, label)")
        if not math.isnan(ref_corr):
            ax2.axhline(ref_corr, color="#e45756", linestyle=":", alpha=0.6, label="Step2a corr")
        ax2.set_ylabel("Clean-label correlation" if panel_idx == 1 else "")
        ax2.set_ylim(0.65, 1.03)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="best", frameon=False)
    fig.suptitle("Step2 Label QA: Degree Increases Label Variance and Misspecification", y=1.02)
    return save_figure(fig, out_dir, "01_label_std_and_corr_by_degree", formats)


def plot_main_unseen_alignment(records: list[dict[str, object]], out_dir: Path, formats: list[str]) -> list[str]:
    by_key: dict[tuple[str, str], dict[str, object]] = {(r["regime"], r["split"]): r for r in records}
    regimes = sorted({r["regime"] for r in records}, key=regime_sort_key)
    stats = [
        ("label_mean", "Label mean"),
        ("label_std", "Label std"),
        ("clean_corr", "Corr(clean, label)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    colors = {"Step2a": "#4c4c4c", "Step2b": "#1f77b4", "Step2c": "#009e73"}
    for ax, (field, title) in zip(axes, stats):
        xs, ys, cs = [], [], []
        for regime in regimes:
            main = by_key.get((regime, "main2000"))
            unseen = by_key.get((regime, "unseen10000"))
            if not main or not unseen:
                continue
            x = main[field]
            y = unseen[field]
            if math.isnan(float(x)) or math.isnan(float(y)):
                continue
            xs.append(float(x))
            ys.append(float(y))
            cs.append(colors.get(str(main["block"]), "#888888"))
            ax.text(float(x), float(y), display_regime(regime), fontsize=8, alpha=0.85)
        ax.scatter(xs, ys, c=cs, s=50, edgecolor="white", linewidth=0.7)
        if xs and ys:
            lo = min(xs + ys)
            hi = max(xs + ys)
            pad = (hi - lo) * 0.08 if hi > lo else 0.1
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#666666", linestyle="--", linewidth=1)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(title)
        ax.set_xlabel("main2000")
        ax.set_ylabel("unseen10000")
    fig.suptitle("Main and Unseen Label Distributions Are Aligned", y=1.02)
    return save_figure(fig, out_dir, "02_main_vs_unseen_label_alignment", formats)


def plot_gap_vs_degree(rows: list[dict[str, object]], out_dir: Path, formats: list[str]) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=False)
    for ax, block, title in zip(axes, ("Step2b", "Step2c"), ("Step2b: noiseless polynomial", "Step2c: multiplicative noise")):
        for checkpoint in PRIMARY_CHECKPOINTS:
            means = []
            for degree in DEGREES:
                values = [
                    row["test_mean_normalized_gap_float"]
                    for row in rows
                    if row["block"] == block
                    and row["degree_int"] == degree
                    and row["checkpoint_label"] == checkpoint
                ]
                means.append(mean(values))
            ax.plot(
                DEGREES,
                means,
                marker="o",
                linewidth=2,
                color=METHOD_COLORS[checkpoint],
                label=METHOD_LABELS[checkpoint],
            )
        ax.set_title(title)
        ax.set_xlabel("Polynomial degree")
        ax.set_xticks(DEGREES)
        ax.set_ylabel("Mean normalized decision gap\n(lower is better)")
        ax.legend(frameon=False)
    fig.suptitle("Unseen10000 Performance vs Label Misspecification", y=1.02)
    return save_figure(fig, out_dir, "03_unseen_normalized_gap_vs_degree", formats)


def plot_paired_improvement_heatmap(rows: list[dict[str, object]], out_dir: Path, formats: list[str]) -> list[str]:
    checkpoints = ("fy_val_fy_loss", "spoplus_val_spoplus_loss", "fy_val_decision_gap", "spoplus_val_decision_gap")
    regimes = sorted({str(r["regime"]) for r in rows}, key=regime_sort_key)
    fig, axes = plt.subplots(2, 2, figsize=(16, 8), sharex=True, sharey=True)
    values_all = [
        r["paired_mean_improvement_float"]
        for r in rows
        if r["checkpoint_label"] in checkpoints and not math.isnan(r["paired_mean_improvement_float"])
    ]
    max_abs = max(abs(min(values_all)), abs(max(values_all))) if values_all else 1.0
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-max_abs, vmax=max_abs)
    image = None
    for ax, checkpoint in zip(axes.ravel(), checkpoints):
        matrix = np.full((len(TRAIN_SIZES), len(regimes)), np.nan)
        for i, train_size in enumerate(TRAIN_SIZES):
            for j, regime in enumerate(regimes):
                matches = [
                    r
                    for r in rows
                    if r["checkpoint_label"] == checkpoint
                    and r["regime"] == regime
                    and r["train_size_int"] == train_size
                ]
                if matches:
                    matrix[i, j] = matches[0]["paired_mean_improvement_float"]
        image = ax.imshow(matrix, aspect="auto", cmap="RdBu", norm=norm)
        ax.set_title(METHOD_LABELS[checkpoint])
        ax.set_yticks(range(len(TRAIN_SIZES)), [str(x) for x in TRAIN_SIZES])
        ax.set_xticks(range(len(regimes)), [display_regime(r) for r in regimes], rotation=45, ha="right")
        ax.set_ylabel("Training graphs")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                if not math.isnan(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color="black")
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82, label="Paired raw-gap improvement over 2stage\n(higher is better)")
    fig.suptitle("Where Decision-Focused Checkpoints Improve Over 2stage", y=0.99)
    return save_figure(fig, out_dir, "04_paired_improvement_heatmap", formats)


def plot_selector_delta(rows: list[dict[str, object]], out_dir: Path, formats: list[str]) -> list[str]:
    fy = compute_selector_deltas(rows, "fy_val_decision_gap", "fy_val_fy_loss")
    spo = compute_selector_deltas(rows, "spoplus_val_decision_gap", "spoplus_val_spoplus_loss")
    regimes = sorted({str(r["regime"]) for r in rows}, key=regime_sort_key)

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), sharey=True)
    for ax, data, title, color in [
        (axes[0], fy, "FY: val FY loss vs val decision gap", METHOD_COLORS["fy_val_fy_loss"]),
        (axes[1], spo, "SPO+: val SPO+ loss vs val decision gap", METHOD_COLORS["spoplus_val_spoplus_loss"]),
    ]:
        by_regime = defaultdict(list)
        for row in data:
            by_regime[row["regime"]].append(row["selector_delta"])
        means = [mean(by_regime[regime]) for regime in regimes]
        ax.bar(range(len(regimes)), means, color=color, alpha=0.8)
        ax.axhline(0, color="#333333", linewidth=1)
        ax.set_xticks(range(len(regimes)), [display_regime(r) for r in regimes], rotation=45, ha="right")
        ax.set_title(title)
        ax.set_ylabel("Mean selector delta on unseen10000\npositive = surrogate selector better")
    fig.suptitle("Checkpoint Selection: Surrogate Loss vs Validation Decision Gap", y=1.02)
    return save_figure(fig, out_dir, "05_selector_delta_fy_spoplus", formats)


def plot_selected_epoch_heatmap(rows: list[dict[str, object]], out_dir: Path, formats: list[str]) -> list[str]:
    checkpoints = ("fy_val_fy_loss", "fy_val_decision_gap", "spoplus_val_spoplus_loss", "spoplus_val_decision_gap")
    regimes = sorted({str(r["regime"]) for r in rows}, key=regime_sort_key)
    fig, axes = plt.subplots(2, 2, figsize=(16, 8), sharex=True, sharey=True)
    image = None
    for ax, checkpoint in zip(axes.ravel(), checkpoints):
        matrix = np.full((len(TRAIN_SIZES), len(regimes)), np.nan)
        for i, train_size in enumerate(TRAIN_SIZES):
            for j, regime in enumerate(regimes):
                matches = [
                    r
                    for r in rows
                    if r["checkpoint_label"] == checkpoint
                    and r["regime"] == regime
                    and r["train_size_int"] == train_size
                ]
                if matches:
                    matrix[i, j] = matches[0]["selected_epoch_float"]
        image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=500)
        ax.set_title(METHOD_LABELS[checkpoint])
        ax.set_yticks(range(len(TRAIN_SIZES)), [str(x) for x in TRAIN_SIZES])
        ax.set_xticks(range(len(regimes)), [display_regime(r) for r in regimes], rotation=45, ha="right")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                if not math.isnan(value):
                    ax.text(j, i, f"{int(value)}", ha="center", va="center", fontsize=7, color="white" if value > 250 else "black")
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82, label="Selected epoch")
    fig.suptitle("Selected Checkpoint Epochs", y=0.99)
    return save_figure(fig, out_dir, "06_selected_epoch_heatmap", formats)


def plot_theta_endpoints(rows: list[dict[str, object]], out_dir: Path, formats: list[str]) -> list[str]:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=False, sharey=False)
    for ax, block in zip(axes, ("Step2a", "Step2b", "Step2c")):
        for checkpoint in PRIMARY_CHECKPOINTS:
            xs = [r["theta_1_float"] for r in rows if r["block"] == block and r["checkpoint_label"] == checkpoint]
            ys = [r["theta_2_float"] for r in rows if r["block"] == block and r["checkpoint_label"] == checkpoint]
            ax.scatter(xs, ys, s=42, alpha=0.75, color=METHOD_COLORS[checkpoint], label=METHOD_LABELS[checkpoint], edgecolor="white", linewidth=0.5)
        ax.scatter([10], [5], marker="*", s=160, color="black", label="clean [10,5]", zorder=5)
        ax.set_title(block)
        ax.set_xlabel("theta_1")
        ax.set_ylabel("theta_2")
        ax.axhline(0, color="#777777", linewidth=0.8, alpha=0.5)
        ax.axvline(0, color="#777777", linewidth=0.8, alpha=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Selected Parameter Endpoints", y=1.12)
    return save_figure(fig, out_dir, "07_theta_endpoints", formats)


def plot_theta_norm_vs_gap(rows: list[dict[str, object]], out_dir: Path, formats: list[str]) -> list[str]:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, block in zip(axes, ("Step2a", "Step2b", "Step2c")):
        for checkpoint in PRIMARY_CHECKPOINTS:
            xs = [r["theta_norm_float"] for r in rows if r["block"] == block and r["checkpoint_label"] == checkpoint]
            ys = [r["test_mean_normalized_gap_float"] for r in rows if r["block"] == block and r["checkpoint_label"] == checkpoint]
            ax.scatter(xs, ys, s=42, alpha=0.75, color=METHOD_COLORS[checkpoint], label=METHOD_LABELS[checkpoint], edgecolor="white", linewidth=0.5)
        ax.set_title(block)
        ax.set_xlabel("||theta||")
        ax.set_ylabel("Mean normalized decision gap\n(lower is better)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Parameter Scale vs Decision Quality", y=1.12)
    return save_figure(fig, out_dir, "08_theta_norm_vs_gap", formats)


def find_per_graph_files(step2_root: Path) -> list[Path]:
    return sorted(step2_root.glob("**/remote_results/**/metrics/unseen10000_per_graph.csv"))


def load_per_graph_subset(step2_root: Path, regime: str, train_size: int) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = defaultdict(dict)
    for path in find_per_graph_files(step2_root):
        text = str(path)
        if f"/{regime}/" not in text or f"/train_size={train_size}/" not in text:
            continue
        for row in load_csv_rows(path):
            label = row.get("checkpoint_label")
            if label not in PRIMARY_CHECKPOINTS:
                continue
            graph = row.get("graph")
            if not graph:
                continue
            result[graph][str(label)] = parse_float(row.get("normalized_gap"))
    return result


def plot_paired_delta_histograms(step2_root: Path, out_dir: Path, formats: list[str]) -> list[str]:
    regimes = ["step2a_additive_rho050", "step2b_poly_d8", "step2c_poly_d8_mult_eps050"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    any_data = False
    for ax, regime in zip(axes, regimes):
        data = load_per_graph_subset(step2_root, regime, 1200)
        for checkpoint in ("fy_val_fy_loss", "spoplus_val_spoplus_loss"):
            deltas = [
                values["2stage_val_mse"] - values[checkpoint]
                for values in data.values()
                if "2stage_val_mse" in values and checkpoint in values
            ]
            if not deltas:
                continue
            any_data = True
            ax.hist(deltas, bins=50, alpha=0.55, density=True, color=METHOD_COLORS[checkpoint], label=METHOD_LABELS[checkpoint])
        ax.axvline(0, color="#333333", linewidth=1)
        ax.set_title(f"{display_regime(regime)}, n=1200")
        ax.set_xlabel("Per-graph normalized-gap improvement over 2stage")
        ax.set_ylabel("Density")
        ax.legend(frameon=False)
    if not any_data:
        plt.close(fig)
        return []
    fig.suptitle("Per-Graph Paired Improvement Distributions", y=1.02)
    return save_figure(fig, out_dir, "09_paired_delta_histograms", formats)


def ecdf(values: list[float]) -> tuple[np.ndarray, np.ndarray]:
    arr = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, len(arr) + 1) / len(arr)
    return arr, y


def plot_tail_gap_ecdf(step2_root: Path, out_dir: Path, formats: list[str]) -> list[str]:
    regimes = ["step2a_additive_rho050", "step2b_poly_d8", "step2c_poly_d8_mult_eps050"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    any_data = False
    for ax, regime in zip(axes, regimes):
        data = load_per_graph_subset(step2_root, regime, 1200)
        for checkpoint in PRIMARY_CHECKPOINTS:
            values = [v[checkpoint] for v in data.values() if checkpoint in v]
            if not values:
                continue
            any_data = True
            x, y = ecdf(values)
            ax.plot(x, y, color=METHOD_COLORS[checkpoint], label=METHOD_LABELS[checkpoint], linewidth=2)
        ax.set_title(f"{display_regime(regime)}, n=1200")
        ax.set_xlabel("Per-graph normalized decision gap")
        ax.set_ylabel("ECDF")
        ax.legend(frameon=False)
    if not any_data:
        plt.close(fig)
        return []
    fig.suptitle("Tail Behavior of Per-Graph Decision Gaps", y=1.02)
    return save_figure(fig, out_dir, "10_tail_gap_ecdf", formats)


def plot_hard_graph_attribution(
    step2_root: Path,
    graph_records: list[dict[str, object]],
    out_dir: Path,
    formats: list[str],
) -> list[str]:
    regimes = ["step2a_additive_rho050", "step2b_poly_d8", "step2c_poly_d8_mult_eps050"]
    graph_index = {
        (str(row["regime"]), str(row["split"]), str(row["graph"])): row
        for row in graph_records
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=False)
    any_data = False
    for ax, regime in zip(axes, regimes):
        per_graph = load_per_graph_subset(step2_root, regime, 1200)
        for checkpoint in ("fy_val_fy_loss", "spoplus_val_spoplus_loss"):
            xs, ys = [], []
            for graph, values in per_graph.items():
                diag = graph_index.get((regime, "unseen10000", graph))
                if not diag or "2stage_val_mse" not in values or checkpoint not in values:
                    continue
                x = diag["label_to_clean_mean_ratio"]
                y = values["2stage_val_mse"] - values[checkpoint]
                if math.isnan(float(x)) or math.isnan(float(y)):
                    continue
                xs.append(float(x))
                ys.append(float(y))
            if not xs:
                continue
            any_data = True
            ax.scatter(
                xs,
                ys,
                s=8,
                alpha=0.16,
                color=METHOD_COLORS[checkpoint],
                label=METHOD_LABELS[checkpoint],
                rasterized=True,
            )
        ax.axhline(0, color="#333333", linewidth=1)
        ax.axvline(1, color="#777777", linestyle="--", linewidth=1, alpha=0.8)
        ax.set_title(f"{display_regime(regime)}, n=1200")
        ax.set_xlabel("Graph label-to-clean mean ratio")
        ax.set_ylabel("Per-graph normalized-gap improvement")
        ax.legend(frameon=False)
    if not any_data:
        plt.close(fig)
        return []
    fig.suptitle("Hard-Graph Attribution: Improvement vs Graph Label Rescaling", y=1.02)
    return save_figure(fig, out_dir, "11_hard_graph_attribution", formats)


def find_loss_curve(step2_root: Path, regime: str, family: str, train_size: int) -> Path | None:
    if family == "fy":
        pattern = f"**/remote_results/{regime}/step1b_fy/**/train_size={train_size}/metrics/e2e_loss_curve.csv"
    else:
        pattern = f"**/remote_results/{regime}/step1c_spoplus/**/train_size={train_size}/metrics/spoplus_loss_curve.csv"
    matches = sorted(step2_root.glob(pattern))
    return matches[0] if matches else None


def load_loss_curve(path: Path | None) -> list[dict[str, object]]:
    if path is None or not path.exists():
        return []
    rows = []
    for row in load_csv_rows(path):
        parsed = dict(row)
        for key, value in row.items():
            parsed[f"{key}_float"] = parse_float(value)
        rows.append(parsed)
    return rows


def plot_training_dashboard(step2_root: Path, out_dir: Path, formats: list[str], regime: str, stem: str, title: str) -> list[str]:
    fy = load_loss_curve(find_loss_curve(step2_root, regime, "fy", 1200))
    spo = load_loss_curve(find_loss_curve(step2_root, regime, "spo", 1200))
    if not fy and not spo:
        return []
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    if fy:
        axes[0, 0].plot([r["epoch_float"] for r in fy], [r["validation_fy_loss_float"] for r in fy], color=METHOD_COLORS["fy_val_fy_loss"], label="FY val FY")
        axes[0, 1].plot([r["epoch_float"] for r in fy], [r["validation_decision_gap_float"] for r in fy], color=METHOD_COLORS["fy_val_fy_loss"], label="FY val gap")
        axes[1, 0].plot([r["epoch_float"] for r in fy], [r["theta_1_float"] for r in fy], color=METHOD_COLORS["fy_val_fy_loss"], linestyle="-", label="FY theta_1")
        axes[1, 1].plot([r["epoch_float"] for r in fy], [r["theta_2_float"] for r in fy], color=METHOD_COLORS["fy_val_fy_loss"], linestyle="-", label="FY theta_2")
    if spo:
        axes[0, 0].plot([r["epoch_float"] for r in spo], [r["validation_spoplus_loss_float"] for r in spo], color=METHOD_COLORS["spoplus_val_spoplus_loss"], label="SPO+ val SPO+")
        axes[0, 1].plot([r["epoch_float"] for r in spo], [r["validation_decision_gap_float"] for r in spo], color=METHOD_COLORS["spoplus_val_spoplus_loss"], label="SPO+ val gap")
        axes[1, 0].plot([r["epoch_float"] for r in spo], [r["theta_1_float"] for r in spo], color=METHOD_COLORS["spoplus_val_spoplus_loss"], linestyle="-", label="SPO+ theta_1")
        axes[1, 1].plot([r["epoch_float"] for r in spo], [r["theta_2_float"] for r in spo], color=METHOD_COLORS["spoplus_val_spoplus_loss"], linestyle="-", label="SPO+ theta_2")

    axes[0, 0].set_title("Validation surrogate objective")
    axes[0, 1].set_title("Validation decision gap")
    axes[1, 0].set_title("theta_1 trajectory")
    axes[1, 1].set_title("theta_2 trajectory")
    for ax in axes.ravel():
        ax.set_xlabel("Epoch")
        ax.legend(frameon=False)
    fig.suptitle(title, y=1.02)
    return save_figure(fig, out_dir, stem, formats)


def plot_heldout_vs_unseen_primary_gap(
    heldout_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    out_dir: Path,
    formats: list[str],
) -> list[str]:
    heldout_index = {
        (str(row["regime"]), int(row["train_size_int"]), str(row["checkpoint_label"])): row
        for row in heldout_rows
        if row.get("checkpoint_label") in PRIMARY_CHECKPOINTS
    }
    block_markers = {"Step2a": "o", "Step2b": "s", "Step2c": "^"}

    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    plotted_x: list[float] = []
    plotted_y: list[float] = []
    for checkpoint in PRIMARY_CHECKPOINTS:
        for block, marker in block_markers.items():
            xs, ys, sizes = [], [], []
            for row in summary_rows:
                if row.get("checkpoint_label") != checkpoint or row.get("block") != block:
                    continue
                key = (str(row["regime"]), int(row["train_size_int"]), checkpoint)
                heldout = heldout_index.get(key)
                if not heldout:
                    continue
                x = heldout["heldout_mean_normalized_gap_float"]
                y = row["test_mean_normalized_gap_float"]
                if math.isnan(float(x)) or math.isnan(float(y)):
                    continue
                xs.append(float(x))
                ys.append(float(y))
                sizes.append(32 + 0.04 * int(row["train_size_int"]))
            if not xs:
                continue
            plotted_x.extend(xs)
            plotted_y.extend(ys)
            ax.scatter(
                xs,
                ys,
                s=sizes,
                marker=marker,
                color=METHOD_COLORS[checkpoint],
                alpha=0.72,
                edgecolor="white",
                linewidth=0.6,
            )
    if plotted_x and plotted_y:
        lo = min(plotted_x + plotted_y)
        hi = max(plotted_x + plotted_y)
        pad = (hi - lo) * 0.08 if hi > lo else 0.001
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#555555", linestyle="--", linewidth=1)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
    method_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=METHOD_COLORS[c], label=METHOD_LABELS[c], markersize=7)
        for c in PRIMARY_CHECKPOINTS
    ]
    block_handles = [
        Line2D([0], [0], marker=m, linestyle="", color="#666666", label=b, markersize=7)
        for b, m in block_markers.items()
    ]
    first_legend = ax.legend(handles=method_handles, loc="upper left", frameon=False, title="Checkpoint")
    ax.add_artist(first_legend)
    ax.legend(handles=block_handles, loc="lower right", frameon=False, title="Regime block")
    ax.set_title("Heldout400 vs Large-Unseen Primary Performance")
    ax.set_xlabel("Heldout400 mean normalized decision gap")
    ax.set_ylabel("Large-unseen mean normalized decision gap")
    return save_figure(fig, out_dir, "15_heldout_vs_unseen_primary_gap", formats)


def plot_best_method_map_unseen10000(rows: list[dict[str, object]], out_dir: Path, formats: list[str]) -> list[str]:
    regimes = sorted({str(r["regime"]) for r in rows}, key=regime_sort_key)
    winners = best_primary_checkpoint_by_setting(rows)
    method_codes = {
        "2stage_val_mse": 0,
        "fy_val_fy_loss": 1,
        "spoplus_val_spoplus_loss": 2,
    }
    method_short = {
        "2stage_val_mse": "2stg",
        "fy_val_fy_loss": "FY",
        "spoplus_val_spoplus_loss": "SPO+",
    }
    cmap = ListedColormap([METHOD_COLORS[c] for c in PRIMARY_CHECKPOINTS])
    matrix = np.full((len(TRAIN_SIZES), len(regimes)), np.nan)
    labels: dict[tuple[int, int], str] = {}
    for i, train_size in enumerate(TRAIN_SIZES):
        for j, regime in enumerate(regimes):
            winner = winners.get((regime, train_size))
            if not winner:
                continue
            checkpoint = str(winner["checkpoint_label"])
            matrix[i, j] = method_codes[checkpoint]
            labels[(i, j)] = method_short[checkpoint]

    fig, ax = plt.subplots(figsize=(13.8, 4.8))
    ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap=cmap, vmin=-0.5, vmax=2.5)
    ax.set_yticks(range(len(TRAIN_SIZES)), [str(x) for x in TRAIN_SIZES])
    ax.set_xticks(range(len(regimes)), [display_regime(r) for r in regimes], rotation=45, ha="right")
    ax.set_ylabel("Training graphs")
    ax.set_title("Best Primary Checkpoint on Large-Unseen Test")
    for (i, j), label in labels.items():
        ax.text(j, i, label, ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    handles = [
        Line2D([0], [0], marker="s", linestyle="", color=METHOD_COLORS[c], label=METHOD_LABELS[c], markersize=9)
        for c in PRIMARY_CHECKPOINTS
    ]
    ax.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.22))
    return save_figure(fig, out_dir, "16_best_method_map_unseen10000", formats)


def plot_gap_vs_train_size_selected_regimes(rows: list[dict[str, object]], out_dir: Path, formats: list[str]) -> list[str]:
    selected_regimes = [
        "step2a_additive_rho050",
        "step2b_poly_d4",
        "step2b_poly_d8",
        "step2c_poly_d4_mult_eps050",
        "step2c_poly_d8_mult_eps050",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.0), sharex=True)
    axes_flat = axes.ravel()
    for ax, regime in zip(axes_flat, selected_regimes):
        for checkpoint in PRIMARY_CHECKPOINTS:
            values = []
            for train_size in TRAIN_SIZES:
                matches = [
                    row
                    for row in rows
                    if row.get("regime") == regime
                    and row.get("checkpoint_label") == checkpoint
                    and row.get("train_size_int") == train_size
                ]
                values.append(matches[0]["test_mean_normalized_gap_float"] if matches else math.nan)
            ax.plot(
                TRAIN_SIZES,
                values,
                marker="o",
                linewidth=2,
                color=METHOD_COLORS[checkpoint],
                label=METHOD_LABELS[checkpoint],
            )
        ax.set_title(display_regime(regime))
        ax.set_xticks(TRAIN_SIZES)
    axes_flat[-1].axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.supxlabel("Training graphs", y=0.03)
    fig.supylabel("Mean normalized decision gap (lower is better)", x=0.02)
    fig.suptitle("Large-Unseen Gap vs Train Size in Representative Regimes", y=1.08)
    return save_figure(fig, out_dir, "17_unseen_gap_vs_train_size_selected_regimes", formats)


def plot_selector_delta_heatmap(rows: list[dict[str, object]], out_dir: Path, formats: list[str]) -> list[str]:
    panels = [
        ("FY", compute_selector_deltas(rows, "fy_val_decision_gap", "fy_val_fy_loss")),
        ("SPO+", compute_selector_deltas(rows, "spoplus_val_decision_gap", "spoplus_val_spoplus_loss")),
    ]
    regimes = sorted({str(r["regime"]) for r in rows}, key=regime_sort_key)
    values_all = [float(row["selector_delta"]) for _, panel_rows in panels for row in panel_rows]
    max_abs = max(abs(min(values_all)), abs(max(values_all))) if values_all else 1.0
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-max_abs, vmax=max_abs)

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.0), sharey=True)
    image = None
    for ax, (title, panel_rows) in zip(axes, panels):
        matrix = np.full((len(TRAIN_SIZES), len(regimes)), np.nan)
        for row in panel_rows:
            regime = str(row["regime"])
            train_size = parse_int(row.get("train_size"))
            if train_size not in TRAIN_SIZES or regime not in regimes:
                continue
            i = TRAIN_SIZES.index(train_size)
            j = regimes.index(regime)
            matrix[i, j] = float(row["selector_delta"])
        image = ax.imshow(matrix, aspect="auto", cmap="PRGn", norm=norm)
        ax.set_title(f"{title}: surrogate-loss selector vs val-gap selector")
        ax.set_yticks(range(len(TRAIN_SIZES)), [str(x) for x in TRAIN_SIZES])
        ax.set_xticks(range(len(regimes)), [display_regime(r) for r in regimes], rotation=45, ha="right")
        ax.set_ylabel("Training graphs")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                if not math.isnan(value):
                    ax.text(j, i, f"{value:.4f}", ha="center", va="center", fontsize=6.5, color="black")
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.78, label="Selector delta on large-unseen normalized gap\npositive = surrogate-loss selector better")
    fig.suptitle("Checkpoint Selection Delta by Regime and Train Size", y=1.02)
    return save_figure(fig, out_dir, "18_selector_delta_heatmap", formats)


def write_manifest(out_dir: Path, written: list[str], skipped: list[str]) -> None:
    manifest = {
        "generated": written,
        "skipped": skipped,
    }
    (out_dir / "plot_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    setup_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = load_summary(args.summary_csv)
    heldout_rows = load_heldout_primary(args.heldout_csv)
    label_records = load_label_diagnostics(args.label_diagnostics_root)
    graph_records = load_graph_diagnostics(args.label_diagnostics_root)

    written: list[str] = []
    skipped: list[str] = []

    plotters = [
        ("01_label_std_and_corr_by_degree", lambda: plot_label_std_corr(label_records, args.out_dir, args.formats)),
        ("02_main_vs_unseen_label_alignment", lambda: plot_main_unseen_alignment(label_records, args.out_dir, args.formats)),
        ("03_unseen_normalized_gap_vs_degree", lambda: plot_gap_vs_degree(summary_rows, args.out_dir, args.formats)),
        ("04_paired_improvement_heatmap", lambda: plot_paired_improvement_heatmap(summary_rows, args.out_dir, args.formats)),
        ("05_selector_delta_fy_spoplus", lambda: plot_selector_delta(summary_rows, args.out_dir, args.formats)),
        ("06_selected_epoch_heatmap", lambda: plot_selected_epoch_heatmap(summary_rows, args.out_dir, args.formats)),
        ("07_theta_endpoints", lambda: plot_theta_endpoints(summary_rows, args.out_dir, args.formats)),
        ("08_theta_norm_vs_gap", lambda: plot_theta_norm_vs_gap(summary_rows, args.out_dir, args.formats)),
        ("09_paired_delta_histograms", lambda: plot_paired_delta_histograms(args.step2_root, args.out_dir, args.formats)),
        ("10_tail_gap_ecdf", lambda: plot_tail_gap_ecdf(args.step2_root, args.out_dir, args.formats)),
        ("11_hard_graph_attribution", lambda: plot_hard_graph_attribution(args.step2_root, graph_records, args.out_dir, args.formats)),
        ("12_training_trajectory_dashboard_step2a", lambda: plot_training_dashboard(args.step2_root, args.out_dir, args.formats, "step2a_additive_rho050", "12_training_trajectory_dashboard_step2a", "Training Trajectory Dashboard: Step2a, n=1200")),
        ("13_training_trajectory_dashboard_step2b_d8", lambda: plot_training_dashboard(args.step2_root, args.out_dir, args.formats, "step2b_poly_d8", "13_training_trajectory_dashboard_step2b_d8", "Training Trajectory Dashboard: Step2b d8, n=1200")),
        ("14_training_trajectory_dashboard_step2c_d8", lambda: plot_training_dashboard(args.step2_root, args.out_dir, args.formats, "step2c_poly_d8_mult_eps050", "14_training_trajectory_dashboard_step2c_d8", "Training Trajectory Dashboard: Step2c d8, n=1200")),
        ("15_heldout_vs_unseen_primary_gap", lambda: plot_heldout_vs_unseen_primary_gap(heldout_rows, summary_rows, args.out_dir, args.formats)),
        ("16_best_method_map_unseen10000", lambda: plot_best_method_map_unseen10000(summary_rows, args.out_dir, args.formats)),
        ("17_unseen_gap_vs_train_size_selected_regimes", lambda: plot_gap_vs_train_size_selected_regimes(summary_rows, args.out_dir, args.formats)),
        ("18_selector_delta_heatmap", lambda: plot_selector_delta_heatmap(summary_rows, args.out_dir, args.formats)),
    ]

    for name, plotter in plotters:
        try:
            outputs = plotter()
        except Exception as exc:  # pragma: no cover - manifest records plotting failures.
            skipped.append(f"{name}: failed with {type(exc).__name__}: {exc}")
            continue
        if outputs:
            print(f"generated {name}")
            written.extend(outputs)
        else:
            print(f"skipped {name}")
            skipped.append(f"{name}: required optional data not found")

    write_manifest(args.out_dir, written, skipped)
    print(f"wrote {len(written)} figure files to {args.out_dir}")
    if skipped:
        print("skipped:")
        for item in skipped:
            print(f"  {item}")


if __name__ == "__main__":
    main()
