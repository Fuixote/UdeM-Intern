#!/usr/bin/env python3
"""Summarize whether decision-critical error explains decision gap."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = (
    PROJECT_ROOT / "surrogate_experiment_results" / "decision_analysis" / "results"
)

PREDICTORS = [
    (
        "mse_all_edges",
        "all-edge MSE",
    ),
    (
        "mse_edges_in_opt",
        "MSE on oracle-selected edges",
    ),
    (
        "mse_edges_in_pred",
        "MSE on method-selected edges",
    ),
    (
        "mse_edges_in_symdiff",
        "MSE on oracle-vs-method symmetric-difference edges",
    ),
    (
        "mse_edges_not_selected",
        "MSE on edges not selected by oracle or method",
    ),
    (
        "top10_error_edges_in_symdiff_rate",
        "rate of top-10 error edges in oracle-vs-method symmetric difference",
    ),
]

OUTPUT_FIELDS = [
    "method_label",
    "method",
    "predictor",
    "predictor_description",
    "n_pairs",
    "pearson_corr_with_normalized_gap",
    "spearman_corr_with_normalized_gap",
    "abs_pearson_corr",
    "abs_spearman_corr",
    "abs_pearson_rank_within_method",
]


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: list[dict[str, object]], fieldnames) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def finite_pairs(xs, ys) -> tuple[list[float], list[float]]:
    pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    return [x for x, _ in pairs], [y for _, y in pairs]


def pearson_corr(xs, ys) -> float:
    clean_xs, clean_ys = finite_pairs(xs, ys)
    n = len(clean_xs)
    if n < 2:
        return float("nan")
    mean_x = sum(clean_xs) / n
    mean_y = sum(clean_ys) / n
    var_x = sum((x - mean_x) ** 2 for x in clean_xs)
    var_y = sum((y - mean_y) ** 2 for y in clean_ys)
    if var_x <= 0.0 or var_y <= 0.0:
        return float("nan")
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(clean_xs, clean_ys))
    return cov / math.sqrt(var_x * var_y)


def average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def spearman_corr(xs, ys) -> float:
    clean_xs, clean_ys = finite_pairs(xs, ys)
    if len(clean_xs) < 2:
        return float("nan")
    return pearson_corr(average_ranks(clean_xs), average_ranks(clean_ys))


def graph_method_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row["regime"],
        str(row["subset_seed"]),
        row["graph_id"],
        row["method_label"],
    )


def summarize_correlations(
    per_graph_rows: list[dict[str, str]],
    edge_summary_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    gap_by_key = {
        graph_method_key(row): parse_float(row["normalized_gap"])
        for row in per_graph_rows
    }
    method_by_label = {
        row["method_label"]: row.get("method", row["method_label"])
        for row in edge_summary_rows
    }
    rows_by_method: dict[str, list[dict[str, str]]] = {}
    for row in edge_summary_rows:
        if graph_method_key(row) in gap_by_key:
            rows_by_method.setdefault(row["method_label"], []).append(row)

    output_rows: list[dict[str, object]] = []
    for method_label in sorted(rows_by_method):
        method_rows = rows_by_method[method_label]
        gaps = [gap_by_key[graph_method_key(row)] for row in method_rows]
        method_output: list[dict[str, object]] = []
        for predictor, description in PREDICTORS:
            xs = [parse_float(row.get(predictor, "")) for row in method_rows]
            clean_xs, clean_gaps = finite_pairs(xs, gaps)
            pearson = pearson_corr(clean_xs, clean_gaps)
            spearman = spearman_corr(clean_xs, clean_gaps)
            method_output.append(
                {
                    "method_label": method_label,
                    "method": method_by_label.get(method_label, method_label),
                    "predictor": predictor,
                    "predictor_description": description,
                    "n_pairs": len(clean_xs),
                    "pearson_corr_with_normalized_gap": pearson,
                    "spearman_corr_with_normalized_gap": spearman,
                    "abs_pearson_corr": abs(pearson) if math.isfinite(pearson) else float("nan"),
                    "abs_spearman_corr": abs(spearman) if math.isfinite(spearman) else float("nan"),
                    "abs_pearson_rank_within_method": "",
                }
            )
        ranked = sorted(
            [
                row
                for row in method_output
                if math.isfinite(float(row["abs_pearson_corr"]))
            ],
            key=lambda row: (-float(row["abs_pearson_corr"]), row["predictor"]),
        )
        for rank, row in enumerate(ranked, start=1):
            row["abs_pearson_rank_within_method"] = rank
        output_rows.extend(method_output)
    return output_rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Correlate all-edge and decision-critical prediction errors with decision gap."
    )
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
        "--output",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "decision_critical_mse_correlations.csv",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    rows = summarize_correlations(
        read_csv_rows(args.per_graph),
        read_csv_rows(args.edge_summary),
    )
    write_csv(args.output, rows, OUTPUT_FIELDS)
    print(f"Saved {len(rows)} correlation rows to {args.output}")
    for row in rows:
        if row["predictor"] in {
            "mse_all_edges",
            "mse_edges_in_symdiff",
            "top10_error_edges_in_symdiff_rate",
        }:
            print(
                f"{row['method_label']} {row['predictor']}: "
                f"pearson={float(row['pearson_corr_with_normalized_gap']):.6g} "
                f"spearman={float(row['spearman_corr_with_normalized_gap']):.6g} "
                f"n={row['n_pairs']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
