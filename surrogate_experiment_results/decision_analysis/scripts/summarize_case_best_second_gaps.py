#!/usr/bin/env python3
"""Summarize best-vs-second KEP gaps for the selected Case A/B/C examples."""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = (
    PROJECT_ROOT / "surrogate_experiment_results" / "decision_analysis" / "results"
)
DEFAULT_CASE_INDEX = DEFAULT_RESULTS_DIR / "case_studies" / "case_study_index.csv"
DEFAULT_SECOND_BEST = DEFAULT_RESULTS_DIR / "second_best_gap_comparison.csv"
DEFAULT_OUTPUT = DEFAULT_RESULTS_DIR / "case_studies" / "case_best_second_gap_summary.csv"
DEFAULT_SUMMARY_OUTPUT = (
    DEFAULT_RESULTS_DIR / "case_studies" / "case_best_second_gap_by_case_method.csv"
)
DEFAULT_LATEX_OUTPUT = DEFAULT_RESULTS_DIR / "case_studies" / "case_best_second_gap_summary.tex"
DEFAULT_PLOT_OUTPUT = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "decision_analysis"
    / "plots"
    / "case_best_second_gap_summary.png"
)

METHOD_LABELS = ("2stage_val_mse", "spoplus_val_spoplus_loss")
METHOD_DISPLAY = {
    "2stage_val_mse": "2stage",
    "spoplus_val_spoplus_loss": "SPO+",
}
CASE_DISPLAY = {
    "case_a_bad_prediction_irrelevant": "Case A",
    "case_b_different_solution_near_optimal": "Case B",
    "case_c_spoplus_fixes_2stage": "Case C",
}

CASE_BEST_SECOND_FIELDS = [
    "case_label",
    "case_id",
    "case_type",
    "subset_seed",
    "graph_id",
    "method_label",
    "rank1_gap_to_oracle",
    "rank2_gap_to_oracle",
    "rank2_minus_rank1_gap_to_oracle",
    "rank1_normalized_gap",
    "rank2_normalized_gap",
    "rank2_minus_rank1_normalized_gap",
    "rank1_same_oracle",
    "rank2_same_oracle",
    "rank1_jaccard_oracle",
    "rank2_jaccard_oracle",
    "rank2_predicted_margin_from_best",
]

SUMMARY_FIELDS = [
    "case_label",
    "method_label",
    "near_threshold",
    "row_count",
    "mean_rank1_normalized_gap",
    "mean_rank2_normalized_gap",
    "mean_rank2_minus_rank1_normalized_gap",
    "median_rank2_minus_rank1_normalized_gap",
    "rank2_near_threshold_count",
    "rank2_near_threshold_rate",
    "rank2_same_oracle_rate",
    "mean_rank2_predicted_margin_from_best",
]


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def parse_float(value: object) -> float:
    if value is None or str(value) == "":
        return float("nan")
    return float(value)


def finite_values(values: list[float]) -> list[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


def finite_mean(values: list[float]) -> float:
    clean = finite_values(values)
    if not clean:
        return float("nan")
    return float(sum(clean) / len(clean))


def finite_median(values: list[float]) -> float:
    clean = sorted(finite_values(values))
    if not clean:
        return float("nan")
    mid = len(clean) // 2
    if len(clean) % 2:
        return float(clean[mid])
    return float(0.5 * (clean[mid - 1] + clean[mid]))


def case_key(row: dict[str, str]) -> tuple[int, str]:
    return (int(row["subset_seed"]), row["graph_id"])


def second_key(row: dict[str, str]) -> tuple[int, str, str, int]:
    return (
        int(row["subset_seed"]),
        row["graph_id"],
        row["method_label"],
        int(row["solution_rank"]),
    )


def case_sort_key(row: dict[str, Any]) -> tuple[str, str, int]:
    method_rank = METHOD_LABELS.index(row["method_label"]) if row["method_label"] in METHOD_LABELS else 99
    return (str(row["case_label"]), str(row["case_id"]), method_rank)


def build_case_best_second_rows(
    case_rows: list[dict[str, str]],
    second_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    second_by_key = {second_key(row): row for row in second_rows}

    output_rows: list[dict[str, Any]] = []
    for case in case_rows:
        seed, graph_id = case_key(case)
        for method_label in METHOD_LABELS:
            rank1 = second_by_key.get((seed, graph_id, method_label, 1))
            rank2 = second_by_key.get((seed, graph_id, method_label, 2))
            if rank1 is None or rank2 is None:
                continue

            rank1_gap = parse_float(rank1.get("gap_to_oracle"))
            rank2_gap = parse_float(rank2.get("gap_to_oracle"))
            rank1_norm = parse_float(rank1.get("normalized_gap_to_oracle"))
            rank2_norm = parse_float(rank2.get("normalized_gap_to_oracle"))

            output_rows.append(
                {
                    "case_label": case["case_label"],
                    "case_id": case["case_id"],
                    "case_type": case.get("case_type", ""),
                    "subset_seed": seed,
                    "graph_id": graph_id,
                    "method_label": method_label,
                    "rank1_gap_to_oracle": rank1_gap,
                    "rank2_gap_to_oracle": rank2_gap,
                    "rank2_minus_rank1_gap_to_oracle": rank2_gap - rank1_gap,
                    "rank1_normalized_gap": rank1_norm,
                    "rank2_normalized_gap": rank2_norm,
                    "rank2_minus_rank1_normalized_gap": rank2_norm - rank1_norm,
                    "rank1_same_oracle": parse_bool(rank1.get("same_solution_as_oracle")),
                    "rank2_same_oracle": parse_bool(rank2.get("same_solution_as_oracle")),
                    "rank1_jaccard_oracle": parse_float(rank1.get("edge_jaccard_with_oracle")),
                    "rank2_jaccard_oracle": parse_float(rank2.get("edge_jaccard_with_oracle")),
                    "rank2_predicted_margin_from_best": parse_float(
                        rank2.get("predicted_margin_from_best")
                    ),
                }
            )

    return sorted(output_rows, key=case_sort_key)


def build_case_best_second_summary(
    rows: list[dict[str, Any]],
    near_threshold: float = 0.05,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["case_label"], row["method_label"])].append(row)

    summary_rows: list[dict[str, Any]] = []
    for case_label, method_label in sorted(
        grouped,
        key=lambda key: (
            key[0],
            METHOD_LABELS.index(key[1]) if key[1] in METHOD_LABELS else 99,
        ),
    ):
        group = grouped[(case_label, method_label)]
        rank1_norm = [float(row["rank1_normalized_gap"]) for row in group]
        rank2_norm = [float(row["rank2_normalized_gap"]) for row in group]
        deltas = [float(row["rank2_minus_rank1_normalized_gap"]) for row in group]
        margins = [parse_float(row.get("rank2_predicted_margin_from_best")) for row in group]
        rank2_near_count = sum(1 for value in rank2_norm if float(value) <= near_threshold)
        rank2_oracle_values = [
            1.0 if row["rank2_same_oracle"] else 0.0
            for row in group
            if "rank2_same_oracle" in row
        ]
        rank2_oracle_rate = finite_mean(rank2_oracle_values)

        summary_rows.append(
            {
                "case_label": case_label,
                "method_label": method_label,
                "near_threshold": near_threshold,
                "row_count": len(group),
                "mean_rank1_normalized_gap": finite_mean(rank1_norm),
                "mean_rank2_normalized_gap": finite_mean(rank2_norm),
                "mean_rank2_minus_rank1_normalized_gap": finite_mean(deltas),
                "median_rank2_minus_rank1_normalized_gap": finite_median(deltas),
                "rank2_near_threshold_count": rank2_near_count,
                "rank2_near_threshold_rate": rank2_near_count / len(group) if group else float("nan"),
                "rank2_same_oracle_rate": rank2_oracle_rate,
                "mean_rank2_predicted_margin_from_best": finite_mean(margins),
            }
        )

    return summary_rows


def compact_case_id(case_id: str) -> str:
    if case_id.startswith("case_a_"):
        prefix = "A"
    elif case_id.startswith("case_b_"):
        prefix = "B"
    elif case_id.startswith("case_c_"):
        prefix = "C"
    else:
        prefix = case_id
    suffix = case_id.rsplit("_", 1)[-1]
    return f"{prefix}{int(suffix)}" if suffix.isdigit() else case_id


def plot_case_best_second(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    if not rows:
        return

    os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(METHOD_LABELS), figsize=(12.5, 4.8), sharey=True)
    if len(METHOD_LABELS) == 1:
        axes = [axes]

    y_max = max(float(row["rank2_normalized_gap"]) for row in rows)
    y_limit = max(0.02, y_max * 1.18)

    for ax, method_label in zip(axes, METHOD_LABELS):
        method_rows = [row for row in rows if row["method_label"] == method_label]
        x_values = list(range(len(method_rows)))
        labels = [
            f"{compact_case_id(row['case_id'])}\n{row['graph_id'].replace('.json', '')}"
            for row in method_rows
        ]
        rank1 = [float(row["rank1_normalized_gap"]) for row in method_rows]
        rank2 = [float(row["rank2_normalized_gap"]) for row in method_rows]

        ax.plot(x_values, rank1, marker="o", linewidth=1.8, label="rank 1")
        ax.plot(x_values, rank2, marker="s", linewidth=1.8, label="rank 2")
        for x, y1, y2 in zip(x_values, rank1, rank2):
            ax.vlines(x, y1, y2, color="0.75", linewidth=1.0, zorder=0)

        ax.axhline(0.05, color="0.35", linestyle="--", linewidth=1.0, label="5% oracle gap")
        ax.set_title(METHOD_DISPLAY.get(method_label, method_label))
        ax.set_xticks(x_values)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0.0, y_limit)
        ax.grid(axis="y", color="0.9", linewidth=0.8)
        ax.set_xlabel("selected Case A/B/C graphs")

    axes[0].set_ylabel("normalized oracle gap")
    axes[-1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Best vs second-best oracle gaps for selected Case A/B/C graphs", y=1.02)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def latex_escape(value: object) -> str:
    return str(value).replace("_", r"\_")


def write_latex_table(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        r"\begin{tabular}{lllrrrr}",
        r"\toprule",
        r"Case & Graph & Method & Rank-1 gap & Rank-2 gap & $\Delta$ gap & Pred. margin \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    latex_escape(compact_case_id(row["case_id"])),
                    latex_escape(row["graph_id"].replace(".json", "")),
                    latex_escape(METHOD_DISPLAY.get(row["method_label"], row["method_label"])),
                    f"{float(row['rank1_normalized_gap']):.4f}",
                    f"{float(row['rank2_normalized_gap']):.4f}",
                    f"{float(row['rank2_minus_rank1_normalized_gap']):.4f}",
                    f"{float(row['rank2_predicted_margin_from_best']):.3f}",
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Summarize Case A/B/C rank-1 vs rank-2 KEP oracle gaps."
    )
    parser.add_argument("--case-index", type=Path, default=DEFAULT_CASE_INDEX)
    parser.add_argument("--second-best", type=Path, default=DEFAULT_SECOND_BEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--latex-output", type=Path, default=DEFAULT_LATEX_OUTPUT)
    parser.add_argument("--plot-output", type=Path, default=DEFAULT_PLOT_OUTPUT)
    parser.add_argument("--near-threshold", type=float, default=0.05)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    case_rows = read_csv_rows(args.case_index)
    second_rows = read_csv_rows(args.second_best)
    output_rows = build_case_best_second_rows(case_rows, second_rows)
    summary_rows = build_case_best_second_summary(output_rows, args.near_threshold)

    write_csv(args.output, output_rows, CASE_BEST_SECOND_FIELDS)
    write_csv(args.summary_output, summary_rows, SUMMARY_FIELDS)
    write_latex_table(args.latex_output, output_rows)
    if not args.no_plot:
        plot_case_best_second(output_rows, args.plot_output)

    print(f"Saved {len(output_rows)} case best-vs-second rows to {args.output}")
    print(f"Saved {len(summary_rows)} case/method summary rows to {args.summary_output}")
    print(f"Saved LaTeX table to {args.latex_output}")
    if not args.no_plot:
        print(f"Saved plot to {args.plot_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
