#!/usr/bin/env python3
"""Draw the randomized toy-family setup schematic for the AAAI draft."""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
GENERATOR_PATH = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "decision_analysis"
    / "scripts"
    / "run_randomized_property_x_toy_experiments.py"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "paper_script"
    / "aaai27"
    / "figures"
    / "randomized_toy_setup_schematic.png"
)


PANEL_BLUE = "#4C78A8"
PANEL_GREEN = "#59A14F"
PANEL_ORANGE = "#F28E2B"
PANEL_RED = "#E15759"
PANEL_GRAY = "#6B7280"
LIGHT_BLUE = "#D9E8F5"
LIGHT_GREEN = "#DCEED8"
LIGHT_ORANGE = "#FBE5CF"
EDGE_GRAY = "#CBD5E1"
TEXT = "#111827"


def load_generator_defaults() -> dict[str, int]:
    """Read randomized toy-family dimensions from the current generator code."""
    spec = importlib.util.spec_from_file_location("toy_generator", GENERATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load generator from {GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    signature = inspect.signature(module.run_sweep)
    return {
        "packing_blocks": signature.parameters["packing_blocks"].default,
        "packing_choices": signature.parameters["packing_choices"].default,
        "stable_cliques": signature.parameters["stable_cliques"].default,
        "stable_vertices_per_clique": signature.parameters[
            "stable_vertices_per_clique"
        ].default,
        "path_count": signature.parameters["path_count"].default,
        "path_length": signature.parameters["path_length"].default,
    }


def setup_axis(ax, title: str, subtitle: str) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0.0, 1.03, title, transform=ax.transAxes, fontsize=13, weight="bold", color=TEXT)
    ax.text(0.0, 0.975, subtitle, transform=ax.transAxes, fontsize=9.2, color=PANEL_GRAY)


def draw_block(ax, center_x: float, center_y: float, width: float, height: float, chosen: int) -> None:
    block = FancyBboxPatch(
        (center_x - width / 2, center_y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        linewidth=0.8,
        edgecolor=PANEL_BLUE,
        facecolor=LIGHT_BLUE,
    )
    ax.add_patch(block)
    y_offsets = [-0.27, -0.09, 0.09, 0.27]
    for idx, y_offset in enumerate(y_offsets):
        color = PANEL_GREEN if idx == chosen else "white"
        edge = PANEL_GREEN if idx == chosen else PANEL_BLUE
        ax.add_patch(
            Circle(
                (center_x, center_y + y_offset * height),
                0.012,
                facecolor=color,
                edgecolor=edge,
                linewidth=0.9,
            )
        )


def draw_decomposable_packing(ax, blocks: int, choices: int) -> None:
    setup_axis(
        ax,
        "Decomposable packing",
        f"{blocks} independent blocks, {choices} components each",
    )
    xs = [0.11, 0.37, 0.63, 0.89]
    ys = [0.74, 0.50, 0.26]
    chosen_cycle = [0, 2, 1, 3]
    for block_idx in range(blocks):
        row, col = divmod(block_idx, 4)
        draw_block(
            ax,
            xs[col],
            ys[row],
            width=0.13,
            height=0.18,
            chosen=chosen_cycle[block_idx % len(chosen_cycle)],
        )
        ax.text(xs[col], ys[row] - 0.12, f"B{block_idx + 1}", ha="center", fontsize=7, color=PANEL_GRAY)
    ax.text(0.50, 0.065, "feasible solution = one selected component per block", ha="center", fontsize=8.4, color=TEXT)


def draw_clique(ax, center_x: float, center_y: float, radius: float, chosen: int) -> None:
    positions = [
        (center_x, center_y + radius),
        (center_x + radius, center_y),
        (center_x, center_y - radius),
        (center_x - radius, center_y),
    ]
    box = FancyBboxPatch(
        (center_x - radius * 1.55, center_y - radius * 1.55),
        radius * 3.1,
        radius * 3.1,
        boxstyle="round,pad=0.004,rounding_size=0.012",
        linewidth=0.7,
        edgecolor=PANEL_GREEN,
        facecolor=LIGHT_GREEN,
    )
    ax.add_patch(box)
    for left in range(len(positions)):
        for right in range(left + 1, len(positions)):
            ax.plot(
                [positions[left][0], positions[right][0]],
                [positions[left][1], positions[right][1]],
                color=EDGE_GRAY,
                linewidth=0.6,
                zorder=1,
            )
    for idx, (x_pos, y_pos) in enumerate(positions):
        color = PANEL_ORANGE if idx == chosen else "white"
        edge = PANEL_ORANGE if idx == chosen else PANEL_GREEN
        ax.add_patch(
            Circle(
                (x_pos, y_pos),
                0.0115,
                facecolor=color,
                edgecolor=edge,
                linewidth=0.9,
                zorder=2,
            )
        )


def draw_stable_set(ax, cliques: int, vertices_per_clique: int) -> None:
    setup_axis(
        ax,
        "Stable / independent set",
        f"{cliques} disjoint conflict cliques, {vertices_per_clique} vertices each",
    )
    xs = [0.11, 0.37, 0.63, 0.89]
    ys = [0.74, 0.50, 0.26]
    chosen_cycle = [1, 3, 0, 2]
    for clique_idx in range(cliques):
        row, col = divmod(clique_idx, 4)
        draw_clique(
            ax,
            xs[col],
            ys[row],
            radius=0.042,
            chosen=chosen_cycle[clique_idx % len(chosen_cycle)],
        )
        ax.text(xs[col], ys[row] - 0.105, f"C{clique_idx + 1}", ha="center", fontsize=7, color=PANEL_GRAY)
    ax.text(0.50, 0.065, "independent set chooses one non-conflicting vertex per clique", ha="center", fontsize=8.4, color=TEXT)


def draw_parallel_path(ax, path_count: int, path_length: int) -> None:
    setup_axis(
        ax,
        "Parallel shortest path",
        f"{path_count} parallel s-t paths, {path_length} edges each",
    )
    source = (0.06, 0.50)
    target = (0.94, 0.50)
    ax.add_patch(Circle(source, 0.025, facecolor="white", edgecolor=PANEL_ORANGE, linewidth=1.2))
    ax.add_patch(Circle(target, 0.025, facecolor="white", edgecolor=PANEL_ORANGE, linewidth=1.2))
    ax.text(source[0], source[1] - 0.065, "s", ha="center", fontsize=10, weight="bold", color=TEXT)
    ax.text(target[0], target[1] - 0.065, "t", ha="center", fontsize=10, weight="bold", color=TEXT)

    y_positions = [0.76, 0.59, 0.41, 0.24]
    for path_idx, y_pos in enumerate(y_positions[:path_count]):
        highlight = path_idx == 1
        color = PANEL_RED if highlight else PANEL_ORANGE
        line_width = 1.8 if highlight else 1.0
        xs = [source[0]] + [
            source[0] + (target[0] - source[0]) * (step + 1) / path_length
            for step in range(path_length - 1)
        ] + [target[0]]
        ys = [source[1]] + [y_pos] * (path_length - 1) + [target[1]]
        for idx in range(len(xs) - 1):
            ax.plot([xs[idx], xs[idx + 1]], [ys[idx], ys[idx + 1]], color=color, linewidth=line_width, alpha=0.95)
        for idx in range(1, len(xs) - 1):
            ax.add_patch(Circle((xs[idx], ys[idx]), 0.0065, facecolor="white", edgecolor=color, linewidth=0.6))
        ax.text(0.50, y_pos + 0.035, f"P{path_idx + 1}", ha="center", fontsize=7, color=PANEL_GRAY)

    arrow = FancyArrowPatch(
        (0.21, 0.16),
        (0.79, 0.16),
        arrowstyle="<->",
        mutation_scale=10,
        linewidth=1.0,
        color=PANEL_RED,
    )
    ax.add_patch(arrow)
    ax.text(0.50, 0.075, "switching paths changes the whole 12-edge route", ha="center", fontsize=8.4, color=TEXT)


def main() -> int:
    defaults = load_generator_defaults()
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.1), constrained_layout=True)
    draw_decomposable_packing(
        axes[0],
        blocks=defaults["packing_blocks"],
        choices=defaults["packing_choices"],
    )
    draw_stable_set(
        axes[1],
        cliques=defaults["stable_cliques"],
        vertices_per_clique=defaults["stable_vertices_per_clique"],
    )
    draw_parallel_path(
        axes[2],
        path_count=defaults["path_count"],
        path_length=defaults["path_length"],
    )
    fig.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved randomized toy setup schematic to: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
