#!/usr/bin/env python3
"""Plot Phase 1 heldout400 seed distributions."""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_PLOT_DIR = SCRIPT_DIR / "plot_results"

MAIN_METHOD_LABELS = ("2stage_val_mse", "spoplus_val_spoplus_loss")


@dataclass(frozen=True)
class GapBoxData:
    regimes: list[str]
    labels: list[str]
    mse_values: list[list[float]]
    spoplus_values: list[list[float]]


@dataclass(frozen=True)
class ReductionBoxData:
    regimes: list[str]
    labels: list[str]
    values: list[list[float]]


@dataclass(frozen=True)
class BlockPanelData:
    block: str
    title: str
    degrees: list[int]
    mse_values: list[list[float]]
    spoplus_values: list[list[float]]


@dataclass(frozen=True)
class BlockRelativeReductionData:
    block: str
    title: str
    degrees: list[int]
    values: list[list[float]]


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def regime_sort_key(regime: str) -> tuple[int, int]:
    block_order = 0 if regime.startswith("step2b") else 1
    match = re.search(r"_d(\d+)", regime)
    degree = int(match.group(1)) if match else 999
    return block_order, degree


def regime_label(regime: str) -> str:
    match = re.search(r"_d(\d+)", regime)
    degree = match.group(1) if match else "?"
    if regime.startswith("step2b"):
        return f"Step2b d{degree}"
    return f"Step2c d{degree} eps050"


def regime_block(regime: str) -> str:
    return "step2b" if regime.startswith("step2b") else "step2c"


def regime_degree(regime: str) -> int:
    match = re.search(r"_d(\d+)", regime)
    return int(match.group(1)) if match else 999


def group_main_gap_boxes(rows: list[dict[str, str]]) -> GapBoxData:
    regimes = sorted({row["regime"] for row in rows}, key=regime_sort_key)
    by_regime_method: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        method = row.get("method_label", "")
        if method not in MAIN_METHOD_LABELS:
            continue
        key = (row["regime"], method)
        by_regime_method.setdefault(key, []).append(
            float(row["test_mean_normalized_gap"])
        )

    return GapBoxData(
        regimes=regimes,
        labels=[regime_label(regime) for regime in regimes],
        mse_values=[
            by_regime_method.get((regime, "2stage_val_mse"), []) for regime in regimes
        ],
        spoplus_values=[
            by_regime_method.get((regime, "spoplus_val_spoplus_loss"), [])
            for regime in regimes
        ],
    )


def group_by_block_gap_boxes(rows: list[dict[str, str]]) -> list[BlockPanelData]:
    grouped: dict[tuple[str, int, str], list[float]] = {}
    for row in rows:
        method = row.get("method_label", "")
        if method not in MAIN_METHOD_LABELS:
            continue
        regime = row["regime"]
        key = (regime_block(regime), regime_degree(regime), method)
        grouped.setdefault(key, []).append(float(row["test_mean_normalized_gap"]))

    panels: list[BlockPanelData] = []
    for block, title in [
        ("step2b", "Step2b: noiseless polynomial"),
        ("step2c", "Step2c: multiplicative noise"),
    ]:
        degrees = sorted({degree for b, degree, _ in grouped if b == block})
        panels.append(
            BlockPanelData(
                block=block,
                title=title,
                degrees=degrees,
                mse_values=[
                    grouped.get((block, degree, "2stage_val_mse"), [])
                    for degree in degrees
                ],
                spoplus_values=[
                    grouped.get((block, degree, "spoplus_val_spoplus_loss"), [])
                    for degree in degrees
                ],
            )
        )
    return panels


def relative_reduction_percent(mse_gap: float, spoplus_gap: float) -> float:
    if mse_gap == 0:
        return math.nan
    return 100.0 * (mse_gap - spoplus_gap) / mse_gap


def group_by_block_relative_reduction_boxes(
    rows: list[dict[str, str]],
) -> list[BlockRelativeReductionData]:
    grouped: dict[tuple[str, int], list[float]] = {}
    for row in rows:
        regime = row["regime"]
        key = (regime_block(regime), regime_degree(regime))
        value = relative_reduction_percent(
            float(row["mse_norm_gap"]),
            float(row["spoplus_norm_gap"]),
        )
        if not math.isnan(value):
            grouped.setdefault(key, []).append(value)

    panels: list[BlockRelativeReductionData] = []
    for block, title in [
        ("step2b", "Step2b: noiseless polynomial"),
        ("step2c", "Step2c: multiplicative noise"),
    ]:
        degrees = sorted({degree for b, degree in grouped if b == block})
        panels.append(
            BlockRelativeReductionData(
                block=block,
                title=title,
                degrees=degrees,
                values=[grouped.get((block, degree), []) for degree in degrees],
            )
        )
    return panels


def group_paired_reduction_boxes(rows: list[dict[str, str]]) -> ReductionBoxData:
    regimes = sorted({row["regime"] for row in rows}, key=regime_sort_key)
    by_regime: dict[str, list[float]] = {regime: [] for regime in regimes}
    for row in rows:
        by_regime[row["regime"]].append(float(row["norm_gap_reduction"]))
    return ReductionBoxData(
        regimes=regimes,
        labels=[regime_label(regime) for regime in regimes],
        values=[by_regime[regime] for regime in regimes],
    )


def style_boxplot(boxplot, color: str) -> None:
    for patch in boxplot["boxes"]:
        patch.set_facecolor(color)
        patch.set_edgecolor("#2f2f2f")
        patch.set_linewidth(0.9)
        patch.set_alpha(0.82)
    for key in ("whiskers", "caps", "medians"):
        for artist in boxplot[key]:
            artist.set_color("#2f2f2f")
            artist.set_linewidth(0.9)
    for flier in boxplot["fliers"]:
        flier.set_marker("o")
        flier.set_markersize(2.5)
        flier.set_markeredgecolor(color)
        flier.set_markerfacecolor(color)
        flier.set_alpha(0.45)


def to_percent_boxes(values: list[list[float]]) -> list[list[float]]:
    return [[100.0 * value for value in box] for box in values]


def save_figure(fig, output_stem: Path) -> list[Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [output_stem.with_suffix(".png"), output_stem.with_suffix(".pdf")]
    fig.savefig(paths[0], dpi=220)
    fig.savefig(paths[1])
    return paths


def format_percent_axis(ax) -> None:
    from matplotlib.ticker import FuncFormatter

    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}%"))


def plot_gap_boxplot(data: GapBoxData, output_stem: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    positions = list(range(1, len(data.labels) + 1))
    mse_pos = [pos - 0.18 for pos in positions]
    spo_pos = [pos + 0.18 for pos in positions]

    fig, ax = plt.subplots(figsize=(11.8, 5.6))
    mse = ax.boxplot(
        to_percent_boxes(data.mse_values),
        positions=mse_pos,
        widths=0.28,
        patch_artist=True,
        manage_ticks=False,
    )
    spo = ax.boxplot(
        to_percent_boxes(data.spoplus_values),
        positions=spo_pos,
        widths=0.28,
        patch_artist=True,
        manage_ticks=False,
    )
    style_boxplot(mse, "#4C78A8")
    style_boxplot(spo, "#F58518")

    ax.set_yscale("symlog", linthresh=1e-4)
    ax.axvline(4.5, color="#8a8a8a", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(data.labels, rotation=25, ha="right")
    ax.set_ylabel("Mean normalized decision gap on heldout400 (%)\nsymlog scale")
    ax.set_title("Phase 1 heldout400 decision gap (%) across 50 subset seeds")
    format_percent_axis(ax)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(
        handles=[
            Patch(facecolor="#4C78A8", edgecolor="#2f2f2f", label="2stage val MSE"),
            Patch(facecolor="#F58518", edgecolor="#2f2f2f", label="SPO+ val SPO+ loss"),
        ],
        frameon=False,
        loc="upper left",
    )
    fig.tight_layout()
    paths = save_figure(fig, output_stem)
    plt.close(fig)
    return paths


def plot_reduction_boxplot(data: ReductionBoxData, output_stem: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positions = list(range(1, len(data.labels) + 1))
    fig, ax = plt.subplots(figsize=(11.8, 5.4))
    boxes = ax.boxplot(
        to_percent_boxes(data.values),
        positions=positions,
        widths=0.48,
        patch_artist=True,
        manage_ticks=False,
    )
    style_boxplot(boxes, "#54A24B")

    ax.axhline(0, color="#222222", linewidth=1.0)
    ax.axvline(4.5, color="#8a8a8a", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(data.labels, rotation=25, ha="right")
    ax.set_ylabel("Normalized-gap reduction\npercentage points: 2stage MSE - SPO+")
    ax.set_title("Phase 1 heldout400 paired improvement across 50 subset seeds")
    format_percent_axis(ax)
    ax.grid(axis="y", alpha=0.25)
    ax.text(
        0.01,
        0.98,
        "Positive values mean SPO+ has lower heldout400 normalized gap.",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color="#333333",
    )
    fig.tight_layout()
    paths = save_figure(fig, output_stem)
    plt.close(fig)
    return paths


def plot_by_block_gap_boxplot(
    panels: list[BlockPanelData],
    output_stem: Path,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.4), sharey=False)
    fig.suptitle("Heldout400 Performance vs Label Misspecification", fontsize=16)

    method_offset = 0.13
    box_width = 0.22
    for ax, panel in zip(axes, panels):
        degrees = panel.degrees
        mse_pos = [degree - method_offset for degree in degrees]
        spo_pos = [degree + method_offset for degree in degrees]
        mse = ax.boxplot(
            to_percent_boxes(panel.mse_values),
            positions=mse_pos,
            widths=box_width,
            patch_artist=True,
            manage_ticks=False,
        )
        spo = ax.boxplot(
            to_percent_boxes(panel.spoplus_values),
            positions=spo_pos,
            widths=box_width,
            patch_artist=True,
            manage_ticks=False,
        )
        style_boxplot(mse, "#4D4D4D")
        style_boxplot(spo, "#CC79A7")

        ax.set_title(panel.title, fontsize=14)
        ax.set_xlabel("Polynomial degree")
        ax.set_ylabel("Mean normalized decision gap (%)\n(lower is better)")
        format_percent_axis(ax)
        ax.set_xticks(degrees)
        ax.set_xticklabels([str(degree) for degree in degrees])
        ax.grid(alpha=0.28)
        if degrees:
            ax.set_xlim(min(degrees) - 0.45, max(degrees) + 0.45)
        ax.legend(
            handles=[
                Patch(facecolor="#4D4D4D", edgecolor="#2f2f2f", label="2stage (val MSE)"),
                Patch(facecolor="#CC79A7", edgecolor="#2f2f2f", label="SPO+ (val SPO+)"),
            ],
            frameon=False,
            loc="upper left",
        )

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    paths = save_figure(fig, output_stem)
    plt.close(fig)
    return paths


def plot_by_block_relative_reduction_boxplot(
    panels: list[BlockRelativeReductionData],
    output_stem: Path,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.4), sharey=False)
    fig.suptitle("Heldout400 Relative Improvement vs Label Misspecification", fontsize=16)

    for ax, panel in zip(axes, panels):
        degrees = panel.degrees
        boxes = ax.boxplot(
            panel.values,
            positions=degrees,
            widths=0.36,
            patch_artist=True,
            manage_ticks=False,
        )
        style_boxplot(boxes, "#54A24B")

        ax.axhline(0, color="#222222", linewidth=1.0)
        ax.set_title(panel.title, fontsize=14)
        ax.set_xlabel("Polynomial degree")
        ax.set_ylabel("Relative decision-gap reduction vs 2stage (%)")
        format_percent_axis(ax)
        ax.set_xticks(degrees)
        ax.set_xticklabels([str(degree) for degree in degrees])
        ax.grid(alpha=0.28)
        if degrees:
            ax.set_xlim(min(degrees) - 0.45, max(degrees) + 0.45)
        ax.text(
            0.02,
            0.96,
            "(2stage - SPO+) / 2stage",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            color="#333333",
        )

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    paths = save_figure(fig, output_stem)
    plt.close(fig)
    return paths


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Plot Phase 1 heldout400 boxplots from aggregate CSV files."
    )
    parser.add_argument(
        "--per_seed_csv",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "phase1_heldout400_per_seed.csv",
    )
    parser.add_argument(
        "--paired_csv",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "phase1_heldout400_paired_main.csv",
    )
    parser.add_argument("--plot_dir", type=Path, default=DEFAULT_PLOT_DIR)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    per_seed_rows = read_csv(args.per_seed_csv)
    paired_rows = read_csv(args.paired_csv)
    gap_paths = plot_gap_boxplot(
        group_main_gap_boxes(per_seed_rows),
        args.plot_dir / "phase1_heldout400_gap_boxplot",
    )
    by_block_paths = plot_by_block_gap_boxplot(
        group_by_block_gap_boxes(per_seed_rows),
        args.plot_dir / "phase1_heldout400_by_block_gap_boxplot",
    )
    relative_paths = plot_by_block_relative_reduction_boxplot(
        group_by_block_relative_reduction_boxes(paired_rows),
        args.plot_dir / "phase1_heldout400_by_block_relative_reduction_boxplot",
    )
    reduction_paths = plot_reduction_boxplot(
        group_paired_reduction_boxes(paired_rows),
        args.plot_dir / "phase1_heldout400_paired_reduction_boxplot",
    )
    for path in [*gap_paths, *by_block_paths, *relative_paths, *reduction_paths]:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
