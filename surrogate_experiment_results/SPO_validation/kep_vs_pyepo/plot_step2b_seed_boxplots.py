#!/usr/bin/env python3
"""Plot Step2b degree seed-distribution boxplots from Step2 resampling results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

try:
    from .plot_step2b_degree_overlay import METHODS, METRIC_LABELS
except ImportError:  # pragma: no cover - supports direct script execution.
    from plot_step2b_degree_overlay import METHODS, METRIC_LABELS


SCRIPT_DIR = Path(__file__).resolve().parent
SURROGATE_RESULTS_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_PER_SEED_CSV = (
    SURROGATE_RESULTS_ROOT
    / "Step2_resampling"
    / "results"
    / "phase1_heldout400_per_seed.csv"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "step2b_bridge_results" / "plots"
DEFAULT_DEGREES = [1, 2, 4, 8]
DEFAULT_METRIC = "test_mean_normalized_gap"
DEFAULT_SPOPLUS_METHOD_LABEL = "spoplus_val_spoplus_loss"
SOURCE_LABEL = "step2_resampling_heldout400_mirrored_by_bridge"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-seed-csv", type=Path, default=DEFAULT_PER_SEED_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--degrees", nargs="+", type=int, default=DEFAULT_DEGREES)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--subset-seeds", nargs="+", type=int, default=None)
    parser.add_argument("--train-size", type=int, default=50)
    parser.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        choices=["test_mean_normalized_gap", "test_mean_decision_gap", "test_median_normalized_gap"],
    )
    parser.add_argument(
        "--spoplus-method-label",
        choices=["spoplus_val_spoplus_loss", "spoplus_val_decision_gap"],
        default=DEFAULT_SPOPLUS_METHOD_LABEL,
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["png", "pdf", "svg"],
        default=["png"],
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser


def read_csv_rows(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def selected_subset_seeds(args: argparse.Namespace) -> list[int]:
    if args.subset_seeds is not None:
        return sorted(dict.fromkeys(args.subset_seeds))
    return list(range(args.seed_start, args.seed_start + args.seed_count))


def method_label_to_overlay_methods(method_label: str) -> list[str]:
    if method_label == "2stage_val_mse":
        return ["pyepo_lr", "step1c_lr"]
    return ["pyepo_spoplus", "step1c_spoplus"]


def row_is_step2b_target(row: dict, degrees: set[int], subset_seeds: set[int]) -> bool:
    if row.get("block") and row.get("block") != "step2b":
        return False
    if not row.get("regime", "").startswith("step2b"):
        return False
    try:
        degree = int(row["degree"])
        subset_seed = int(row["subset_seed"])
    except (KeyError, TypeError, ValueError):
        return False
    return degree in degrees and subset_seed in subset_seeds


def collect_boxplot_rows(
    per_seed_rows: list[dict],
    degrees: list[int],
    subset_seeds: list[int],
    metric: str,
    train_size: int,
    spoplus_method_label: str,
) -> list[dict]:
    degrees = sorted(dict.fromkeys(int(degree) for degree in degrees))
    subset_seeds = sorted(dict.fromkeys(int(seed) for seed in subset_seeds))
    degree_set = set(degrees)
    seed_set = set(subset_seeds)
    source_method_labels = {"2stage_val_mse", spoplus_method_label}

    by_key: dict[tuple[int, int, str], dict] = {}
    duplicates: list[tuple[int, int, str]] = []
    for row in per_seed_rows:
        method_label = row.get("method_label", "")
        if method_label not in source_method_labels:
            continue
        if not row_is_step2b_target(row, degree_set, seed_set):
            continue
        key = (int(row["degree"]), int(row["subset_seed"]), method_label)
        if key in by_key:
            duplicates.append(key)
        by_key[key] = row
    if duplicates:
        raise ValueError("Duplicate Step2b seed rows: {}".format(sorted(duplicates)))

    missing = [
        (degree, seed, method_label)
        for degree in degrees
        for seed in subset_seeds
        for method_label in sorted(source_method_labels)
        if (degree, seed, method_label) not in by_key
    ]
    if missing:
        preview = ", ".join(map(str, missing[:6]))
        raise ValueError(
            "Missing Step2b seed rows for {} entries; first missing: {}".format(
                len(missing),
                preview,
            )
        )

    label_by_method = {method: label for method, label, _, _, _ in METHODS}
    rows: list[dict] = []
    for degree in degrees:
        for seed in subset_seeds:
            for source_method_label in ["2stage_val_mse", spoplus_method_label]:
                source_row = by_key[(degree, seed, source_method_label)]
                value = float(source_row[metric])
                for method in method_label_to_overlay_methods(source_method_label):
                    rows.append(
                        {
                            "degree": degree,
                            "subset_seed": seed,
                            "method": method,
                            "label": label_by_method[method],
                            "value": value,
                            "metric": metric,
                            "source": SOURCE_LABEL,
                            "train_size": train_size,
                            "regime": source_row.get("regime", ""),
                            "source_method_label": source_method_label,
                            "source_method": source_row.get("method", ""),
                            "selection_metric": source_row.get("selection_metric", ""),
                            "selected_epoch": source_row.get("selected_epoch", ""),
                            "evaluation": "heldout400_resampling",
                        }
                    )
    method_order = {method: idx for idx, (method, _, _, _, _) in enumerate(METHODS)}
    return sorted(
        rows,
        key=lambda row: (int(row["degree"]), int(row["subset_seed"]), method_order[row["method"]]),
    )


def write_boxplot_table(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "degree",
        "subset_seed",
        "method",
        "label",
        "value",
        "metric",
        "source",
        "train_size",
        "regime",
        "source_method_label",
        "source_method",
        "selection_metric",
        "selected_epoch",
        "evaluation",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def style_boxplot(boxplot, color: str) -> None:
    for patch in boxplot["boxes"]:
        patch.set_facecolor(color)
        patch.set_edgecolor("#2f2f2f")
        patch.set_linewidth(0.9)
        patch.set_alpha(0.86)
    for key in ("whiskers", "caps", "medians"):
        for artist in boxplot[key]:
            artist.set_color("#2f2f2f")
            artist.set_linewidth(0.9)
    for flier in boxplot["fliers"]:
        flier.set_marker("o")
        flier.set_markersize(3.0)
        flier.set_markeredgecolor(color)
        flier.set_markerfacecolor("none")
        flier.set_alpha(0.8)


def plot_boxplot_rows(rows: list[dict], output_path: Path, metric: str, dpi: int) -> None:
    if not rows:
        raise ValueError("No boxplot rows to render")

    degrees = sorted({int(row["degree"]) for row in rows})
    degree_positions = {degree: idx + 1 for idx, degree in enumerate(degrees)}
    subset_seeds = sorted({int(row["subset_seed"]) for row in rows})
    train_size = rows[0]["train_size"]
    method_offsets = [-0.27, -0.09, 0.09, 0.27]
    box_width = 0.16

    fig, ax = plt.subplots(figsize=(15.5, 5.8))
    for method_idx, (method, label, color, _, _) in enumerate(METHODS):
        values_by_degree = []
        positions = []
        for degree in degrees:
            values = [
                float(row["value"])
                for row in rows
                if int(row["degree"]) == degree and row["method"] == method
            ]
            values_by_degree.append(values)
            positions.append(degree_positions[degree] + method_offsets[method_idx])
        boxplot = ax.boxplot(
            values_by_degree,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            manage_ticks=False,
        )
        style_boxplot(boxplot, color)

    ax.set_xticks([degree_positions[degree] for degree in degrees])
    ax.set_xticklabels([str(degree) for degree in degrees])
    ax.set_xlabel("Polynomial Degree")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(
        "KEP Step2b: train size = {}, subset seeds = {}-{}".format(
            train_size,
            subset_seeds[0],
            subset_seeds[-1],
        )
    )
    ax.grid(axis="y", color="0.86", linewidth=0.8)
    ax.legend(
        handles=[
            Patch(facecolor=color, edgecolor="#2f2f2f", label=label)
            for _, label, color, _, _ in METHODS
        ],
        loc="upper left",
        ncol=2,
        fontsize=9,
    )
    ax.set_xlim(0.5, len(degrees) + 0.5)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def seed_label(subset_seeds: list[int]) -> str:
    subset_seeds = sorted(dict.fromkeys(subset_seeds))
    if subset_seeds == list(range(subset_seeds[0], subset_seeds[-1] + 1)):
        return "{}-{}".format(subset_seeds[0], subset_seeds[-1])
    return "_".join(str(seed) for seed in subset_seeds)


def output_stem(train_size: int, subset_seeds: list[int], metric: str) -> str:
    return "step2b_degree_boxplot_train_size={}_seeds={}_{}".format(
        train_size,
        seed_label(subset_seeds),
        metric,
    )


def run(args: argparse.Namespace) -> dict:
    per_seed_rows = read_csv_rows(args.per_seed_csv)
    subset_seeds = selected_subset_seeds(args)
    rows = collect_boxplot_rows(
        per_seed_rows,
        degrees=args.degrees,
        subset_seeds=subset_seeds,
        metric=args.metric,
        train_size=args.train_size,
        spoplus_method_label=args.spoplus_method_label,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem(args.train_size, subset_seeds, args.metric)
    csv_path = args.output_dir / "{}.csv".format(stem)
    write_boxplot_table(csv_path, rows)

    plot_paths = []
    for fmt in args.formats:
        plot_path = args.output_dir / "{}.{}".format(stem, fmt)
        plot_boxplot_rows(rows, plot_path, metric=args.metric, dpi=args.dpi)
        plot_paths.append(plot_path)
        print("Saved {}".format(plot_path))
    print("Saved {}".format(csv_path))
    return {"csv": csv_path, "plots": plot_paths, "rows": rows}


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
