#!/usr/bin/env python3
"""Plot paper-shortest-path summary CSV files."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple


IMPLEMENTATION_LABELS = {
    "ls": "LS",
    "ours-spoplus": "Ours SPO+",
    "pyepo-spoplus": "PyEPO SPO+",
}
IMPLEMENTATION_COLORS = {
    "ls": "#4C78A8",
    "ours-spoplus": "#F58518",
    "pyepo-spoplus": "#54A24B",
}


def read_rows(summary_csv: Path) -> List[Dict[str, str]]:
    with summary_csv.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _group_values(rows: Sequence[Dict[str, str]], implementation: str | None = None):
    grouped: Dict[Tuple[float, int, str], List[float]] = defaultdict(list)
    for row in rows:
        if implementation is not None and row["implementation"] != implementation:
            continue
        key = (
            float(row["noise_half_width"]),
            int(row["degree"]),
            row["implementation"],
        )
        grouped[key].append(float(row["test_norm_spo"]))
    return grouped


def _ordered_implementations(rows: Sequence[Mapping[str, str]]) -> List[str]:
    present = {row["implementation"] for row in rows}
    return [
        value
        for value in ("ls", "ours-spoplus", "pyepo-spoplus")
        if value in present
    ]


def plot_ls_vs_spoplus(rows: Sequence[Dict[str, str]], output_path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import PercentFormatter

    noises = sorted({float(row["noise_half_width"]) for row in rows})
    degrees = sorted({int(row["degree"]) for row in rows})
    implementations = _ordered_implementations(rows)
    grouped = _group_values(rows)
    fig, axes = plt.subplots(
        1,
        len(noises),
        figsize=(6.5 * len(noises), 4.8),
        sharey=True,
        constrained_layout=True,
    )
    if len(noises) == 1:
        axes = [axes]
    for ax, noise in zip(axes, noises):
        positions = []
        values = []
        patch_colors = []
        centers = []
        for degree_idx, degree in enumerate(degrees):
            base = degree_idx * (len(implementations) + 1.25)
            degree_positions = []
            for impl_idx, implementation in enumerate(implementations):
                key = (noise, degree, implementation)
                if key not in grouped:
                    continue
                position = base + impl_idx
                positions.append(position)
                degree_positions.append(position)
                values.append(grouped[key])
                patch_colors.append(IMPLEMENTATION_COLORS.get(implementation, "#777777"))
            if degree_positions:
                centers.append((sum(degree_positions) / len(degree_positions), degree))
        boxes = ax.boxplot(values, positions=positions, patch_artist=True, widths=0.72)
        for patch, color in zip(boxes["boxes"], patch_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        ax.set_title(f"Noise Half-width = {noise:g}")
        ax.set_xlabel("Polynomial Degree")
        ax.set_xticks([center for center, _ in centers])
        ax.set_xticklabels([str(degree) for _, degree in centers])
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Normalized test SPO loss")
    fig.suptitle("SPO Paper Shortest-Path Middle Row: LS vs SPO+", fontsize=13)
    legend_handles = [
        Patch(
            facecolor=IMPLEMENTATION_COLORS[implementation],
            edgecolor="black",
            alpha=0.65,
            label=IMPLEMENTATION_LABELS[implementation],
        )
        for implementation in implementations
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=len(legend_handles), frameon=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _paired_pyepo_ours(rows: Sequence[Dict[str, str]]) -> List[Dict[str, float]]:
    keyed = defaultdict(dict)
    for row in rows:
        if row["implementation"] not in {"ours-spoplus", "pyepo-spoplus"}:
            continue
        key = (
            int(row["trial"]),
            int(row["degree"]),
            float(row["noise_half_width"]),
        )
        keyed[key][row["implementation"]] = float(row["test_norm_spo"])
    pairs = []
    for (trial, degree, noise), values in keyed.items():
        if "pyepo-spoplus" in values and "ours-spoplus" in values:
            pairs.append(
                {
                    "trial": float(trial),
                    "degree": float(degree),
                    "noise_half_width": float(noise),
                    "pyepo": values["pyepo-spoplus"],
                    "ours": values["ours-spoplus"],
                    "difference": values["ours-spoplus"] - values["pyepo-spoplus"],
                }
            )
    return pairs


def plot_pyepo_vs_ours_scatter(rows: Sequence[Dict[str, str]], output_path: Path) -> bool:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    pairs = _paired_pyepo_ours(rows)
    if not pairs:
        return False
    pyepo_values = [item["pyepo"] for item in pairs]
    ours_values = [item["ours"] for item in pairs]
    noise_values = [item["noise_half_width"] for item in pairs]
    lower = min(pyepo_values + ours_values)
    upper = max(pyepo_values + ours_values)
    padding = max((upper - lower) * 0.05, 1e-6)
    fig, ax = plt.subplots(figsize=(5.5, 5.2), constrained_layout=True)
    scatter = ax.scatter(
        pyepo_values,
        ours_values,
        c=noise_values,
        cmap="viridis",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.3,
    )
    ax.plot([lower, upper], [lower, upper], color="#333333", linewidth=1)
    ax.set_xlim(lower - padding, upper + padding)
    ax.set_ylim(lower - padding, upper + padding)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_xlabel("PyEPO SPO+ normalized test SPO loss")
    ax.set_ylabel("Our SPO+ normalized test SPO loss")
    ax.set_title("PyEPO SPO+ vs Ours SPO+")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Noise half-width")
    ax.grid(alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return True


def plot_pyepo_vs_ours_difference(rows: Sequence[Dict[str, str]], output_path: Path) -> bool:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    pairs = _paired_pyepo_ours(rows)
    if not pairs:
        return False
    grouped: Dict[Tuple[float, int], List[float]] = defaultdict(list)
    for pair in pairs:
        grouped[(pair["noise_half_width"], int(pair["degree"]))].append(pair["difference"])
    keys = sorted(grouped)
    values = [grouped[key] for key in keys]
    labels = [f"noise {noise:g}\ndeg {degree}" for noise, degree in keys]
    fig, ax = plt.subplots(figsize=(max(6.0, 0.8 * len(keys)), 4.5), constrained_layout=True)
    ax.axhline(0.0, color="#333333", linewidth=1)
    boxes = ax.boxplot(values, patch_artist=True, widths=0.65)
    for patch in boxes["boxes"]:
        patch.set_facecolor("#F58518")
        patch.set_alpha(0.65)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_xticks(range(1, len(keys) + 1))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Ours - PyEPO normalized test SPO loss")
    ax.set_title("Paired SPO+ Difference by Degree and Noise")
    ax.grid(axis="y", alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return True


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot paper-shortest-path CSV outputs.")
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    rows = read_rows(args.summary_csv)
    output_dir = args.output_dir or args.summary_csv.parent / "plots"
    main_plot = output_dir / "paper_shortest_path_middle_row.png"
    plot_ls_vs_spoplus(rows, main_plot)
    wrote_scatter = plot_pyepo_vs_ours_scatter(
        rows,
        output_dir / "pyepo_vs_ours_spoplus_scatter.png",
    )
    wrote_difference = plot_pyepo_vs_ours_difference(
        rows,
        output_dir / "pyepo_vs_ours_spoplus_difference.png",
    )
    print(f"wrote {main_plot}")
    if wrote_scatter:
        print(f"wrote {output_dir / 'pyepo_vs_ours_spoplus_scatter.png'}")
    if wrote_difference:
        print(f"wrote {output_dir / 'pyepo_vs_ours_spoplus_difference.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
