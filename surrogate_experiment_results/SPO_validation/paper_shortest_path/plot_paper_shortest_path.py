#!/usr/bin/env python3
"""Plot paper-shortest-path summary CSV files."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


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


def plot_ls_vs_spoplus(rows: Sequence[Dict[str, str]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    noises = sorted({float(row["noise_half_width"]) for row in rows})
    degrees = sorted({int(row["degree"]) for row in rows})
    implementations = [
        value
        for value in ("ls", "ours-spoplus", "pyepo-spoplus")
        if value in {row["implementation"] for row in rows}
    ]
    grouped = _group_values(rows)
    fig, axes = plt.subplots(1, len(noises), figsize=(6 * len(noises), 4), sharey=True)
    if len(noises) == 1:
        axes = [axes]
    colors = {
        "ls": "#4C78A8",
        "ours-spoplus": "#F58518",
        "pyepo-spoplus": "#54A24B",
    }
    for ax, noise in zip(axes, noises):
        positions = []
        values = []
        labels = []
        patch_colors = []
        for degree_idx, degree in enumerate(degrees):
            base = degree_idx * (len(implementations) + 1)
            for impl_idx, implementation in enumerate(implementations):
                key = (noise, degree, implementation)
                if key not in grouped:
                    continue
                positions.append(base + impl_idx)
                values.append(grouped[key])
                labels.append(f"{degree}\n{implementation}")
                patch_colors.append(colors.get(implementation, "#777777"))
        boxes = ax.boxplot(values, positions=positions, patch_artist=True, widths=0.75)
        for patch, color in zip(boxes["boxes"], patch_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        ax.set_title(f"Noise half-width = {noise:g}")
        ax.set_xlabel("degree / implementation")
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("normalized test SPO loss")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_pyepo_vs_ours(rows: Sequence[Dict[str, str]], output_path: Path) -> bool:
    import matplotlib.pyplot as plt

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
    pairs = [
        (values["pyepo-spoplus"], values["ours-spoplus"])
        for values in keyed.values()
        if "pyepo-spoplus" in values and "ours-spoplus" in values
    ]
    if not pairs:
        return False
    pyepo_values = [item[0] for item in pairs]
    ours_values = [item[1] for item in pairs]
    lower = min(pyepo_values + ours_values)
    upper = max(pyepo_values + ours_values)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(pyepo_values, ours_values, color="#F58518", alpha=0.75)
    ax.plot([lower, upper], [lower, upper], color="#333333", linewidth=1)
    ax.set_xlabel("PyEPO SPO+ normalized test SPO loss")
    ax.set_ylabel("Our SPO+ normalized test SPO loss")
    ax.grid(alpha=0.25)
    fig.tight_layout()
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
    plot_ls_vs_spoplus(rows, output_dir / "paper_shortest_path_ls_vs_spoplus.png")
    wrote_overlay = plot_pyepo_vs_ours(rows, output_dir / "pyepo_vs_ours_spoplus.png")
    print(f"wrote {output_dir / 'paper_shortest_path_ls_vs_spoplus.png'}")
    if wrote_overlay:
        print(f"wrote {output_dir / 'pyepo_vs_ours_spoplus.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
