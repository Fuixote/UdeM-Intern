#!/usr/bin/env python3
"""Create first-pass decision-analysis plots."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = (
    PROJECT_ROOT / "surrogate_experiment_results" / "decision_analysis" / "results"
)
DEFAULT_PLOT_DIR = (
    PROJECT_ROOT / "surrogate_experiment_results" / "decision_analysis" / "plots"
)
METHOD_LABELS = {
    "2stage_val_mse": "2stage",
    "spoplus_val_spoplus_loss": "SPO+",
}
METHOD_COLORS = {
    "2stage_val_mse": "#4c78a8",
    "spoplus_val_spoplus_loss": "#f58518",
}


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def join_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row["regime"],
        str(row["subset_seed"]),
        row["graph_id"],
        row["method_label"],
    )


def join_prediction_mse(
    per_graph_rows: list[dict[str, str]],
    edge_summary_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    summary_by_key = {join_key(row): row for row in edge_summary_rows}
    joined = []
    for row in per_graph_rows:
        summary = summary_by_key.get(join_key(row))
        if summary is None:
            continue
        joined.append(
            {
                **row,
                "prediction_mse": float(summary["mse_all_edges"]),
                "normalized_gap": float(row["normalized_gap"]),
                "edge_jaccard_with_opt": float(
                    row.get("edge_jaccard_with_opt", "nan")
                ),
            }
        )
    return joined


def error_percentile_bin(rank: int, num_edges: int, bin_count: int = 10) -> int:
    if num_edges <= 0:
        raise ValueError("num_edges must be positive")
    if rank <= 0:
        raise ValueError("rank must be positive")
    raw_bin = math.ceil(float(rank) / float(num_edges) * int(bin_count)) - 1
    return max(0, min(int(bin_count) - 1, int(raw_bin)))


def error_percentile_rate_rows(
    edge_rows: list[dict[str, str]],
    bin_count: int = 10,
) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in edge_rows:
        key = (row["regime"], str(row["subset_seed"]), row["graph_id"])
        groups.setdefault(key, []).append(row)

    accum: dict[tuple[str, int], dict[str, float]] = {}
    method_configs = [
        (
            "2stage_val_mse",
            "rank_err_2stage",
            "in_2stage",
            "in_2stage_symdiff",
        ),
        (
            "spoplus_val_spoplus_loss",
            "rank_err_spoplus",
            "in_spoplus",
            "in_spoplus_symdiff",
        ),
    ]
    for rows in groups.values():
        num_edges = len(rows)
        for row in rows:
            for method_label, rank_field, selected_field, symdiff_field in method_configs:
                bin_index = error_percentile_bin(
                    rank=int(row[rank_field]),
                    num_edges=num_edges,
                    bin_count=bin_count,
                )
                acc = accum.setdefault(
                    (method_label, bin_index),
                    {"count": 0.0, "selected": 0.0, "symdiff": 0.0},
                )
                acc["count"] += 1.0
                acc["selected"] += 1.0 if parse_bool(row[selected_field]) else 0.0
                acc["symdiff"] += 1.0 if parse_bool(row[symdiff_field]) else 0.0

    output = []
    for method_label, bin_index in sorted(accum):
        acc = accum[(method_label, bin_index)]
        count = acc["count"]
        output.append(
            {
                "method_label": method_label,
                "bin_index": bin_index,
                "bin_label": f"{bin_index * 10}-{(bin_index + 1) * 10}%",
                "edge_count": int(count),
                "selected_rate": acc["selected"] / count if count else float("nan"),
                "symdiff_rate": acc["symdiff"] / count if count else float("nan"),
            }
        )
    return output


def save_figure(fig, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    return output_path


def plot_mse_vs_gap(joined_rows: list[dict[str, object]], output_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for method_label, label in METHOD_LABELS.items():
        rows = [row for row in joined_rows if row["method_label"] == method_label]
        ax.scatter(
            [float(row["prediction_mse"]) for row in rows],
            [float(row["normalized_gap"]) for row in rows],
            s=16,
            alpha=0.62,
            color=METHOD_COLORS[method_label],
            label=label,
            edgecolors="none",
        )
    ax.set_xlabel("Edge prediction MSE per graph")
    ax.set_ylabel("Normalized decision gap")
    ax.set_title("Prediction error vs decision gap")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, linewidth=0.6)
    return save_figure(fig, output_path)


def plot_solution_overlap_vs_gap(
    per_graph_rows: list[dict[str, str]], output_path: Path
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for method_label, label in METHOD_LABELS.items():
        rows = [row for row in per_graph_rows if row["method_label"] == method_label]
        ax.scatter(
            [float(row["edge_jaccard_with_opt"]) for row in rows],
            [float(row["normalized_gap"]) for row in rows],
            s=16,
            alpha=0.62,
            color=METHOD_COLORS[method_label],
            label=label,
            edgecolors="none",
        )
    ax.set_xlabel("Jaccard(selected edges, oracle edges)")
    ax.set_ylabel("Normalized decision gap")
    ax.set_title("Solution overlap vs decision gap")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, linewidth=0.6)
    return save_figure(fig, output_path)


def plot_error_percentile_rates(
    rate_rows: list[dict[str, object]], output_path: Path
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for method_label, label in METHOD_LABELS.items():
        rows = [row for row in rate_rows if row["method_label"] == method_label]
        xs = [(int(row["bin_index"]) + 0.5) * 10.0 for row in rows]
        ax.plot(
            xs,
            [float(row["selected_rate"]) for row in rows],
            color=METHOD_COLORS[method_label],
            linewidth=1.8,
            marker="o",
            markersize=4,
            label=f"{label} selected",
        )
        ax.plot(
            xs,
            [float(row["symdiff_rate"]) for row in rows],
            color=METHOD_COLORS[method_label],
            linewidth=1.5,
            linestyle="--",
            marker="s",
            markersize=3.5,
            label=f"{label} symdiff",
        )
    ax.set_xlabel("Edge prediction error percentile bin")
    ax.set_ylabel("Rate")
    ax.set_title("High-error edge criticality")
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    ax.grid(alpha=0.25, linewidth=0.6)
    return save_figure(fig, output_path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Plot decision analysis outputs.")
    parser.add_argument(
        "--per-graph",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "per_graph_decision_comparison.csv",
    )
    parser.add_argument(
        "--edge-summary",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "graph_level_edge_criticality_summary.csv",
    )
    parser.add_argument(
        "--edge-rows",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "edge_error_criticality.csv",
    )
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    per_graph_rows = read_csv_rows(args.per_graph)
    edge_summary_rows = read_csv_rows(args.edge_summary)
    edge_rows = read_csv_rows(args.edge_rows)
    joined = join_prediction_mse(per_graph_rows, edge_summary_rows)
    rate_rows = error_percentile_rate_rows(edge_rows)

    paths = [
        plot_mse_vs_gap(joined, args.plot_dir / "mse_vs_normalized_gap.png"),
        plot_solution_overlap_vs_gap(
            per_graph_rows, args.plot_dir / "solution_overlap_vs_gap.png"
        ),
        plot_error_percentile_rates(
            rate_rows, args.plot_dir / "high_error_edge_selected_symdiff_rate.png"
        ),
    ]
    for path in paths:
        print(f"Saved plot: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
