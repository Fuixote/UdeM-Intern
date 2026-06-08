#!/usr/bin/env python3
"""Summarize cycle-length sensitivity for best and second-best KEP solutions."""

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
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "decision_analysis"
    / "results"
    / "cycle_sensitivity"
)
DEFAULT_CASE_INDEX = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "decision_analysis"
    / "results"
    / "case_studies"
    / "case_study_index.csv"
)
DEFAULT_PLOTS_DIR = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "decision_analysis"
    / "plots"
    / "cycle_sensitivity"
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

SECOND_BEST_SUMMARY_FIELDS = [
    "regime",
    "max_cycle",
    "max_chain",
    "method_label",
    "solution_rank",
    "row_count",
    "mean_gap_to_oracle",
    "median_gap_to_oracle",
    "mean_normalized_gap_to_oracle",
    "median_normalized_gap_to_oracle",
    "exact_oracle_rate",
    "near_1pct_rate",
    "near_5pct_rate",
    "mean_rank2_minus_rank1_normalized_gap",
    "median_rank2_minus_rank1_normalized_gap",
    "mean_edge_jaccard_with_oracle",
    "median_edge_jaccard_with_oracle",
    "mean_predicted_margin_from_best",
    "median_predicted_margin_from_best",
    "mean_num_cycle_candidates",
    "median_num_cycle_candidates",
    "mean_num_chain_candidates",
    "median_num_chain_candidates",
]

RANK2_CASE_FIELDS = [
    "regime",
    "max_cycle",
    "max_chain",
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
    "rank2_jaccard_rank1",
    "rank2_predicted_margin_from_best",
    "rank2_true_obj_diff_from_rank1",
    "num_cycle_candidates",
    "num_chain_candidates",
]

CASE_SUMMARY_FIELDS = [
    "case_label",
    "max_cycle",
    "method_label",
    "row_count",
    "mean_rank1_normalized_gap",
    "mean_rank2_normalized_gap",
    "median_rank2_normalized_gap",
    "mean_rank2_minus_rank1_normalized_gap",
    "median_rank2_minus_rank1_normalized_gap",
    "near_1pct_rank2_count",
    "rank2_near_1pct_rate",
    "near_5pct_rank2_count",
    "rank2_near_5pct_rate",
]

PAIRED_DELTA_FIELDS = [
    "regime",
    "method_label",
    "case_type",
    "subset_seed",
    "graph_id",
    "baseline_max_cycle",
    "comparison_max_cycle",
    "baseline_rank2_gap_to_oracle",
    "comparison_rank2_gap_to_oracle",
    "delta_rank2_gap_to_oracle",
    "baseline_rank2_normalized_gap",
    "comparison_rank2_normalized_gap",
    "delta_rank2_normalized_gap",
    "baseline_rank2_same_oracle",
    "comparison_rank2_same_oracle",
    "baseline_num_cycle_candidates",
    "comparison_num_cycle_candidates",
]

PAIRED_DELTA_SUMMARY_FIELDS = [
    "comparison_max_cycle",
    "method_label",
    "paired_count",
    "fraction_delta_lt_0",
    "fraction_delta_eq_0",
    "fraction_delta_gt_0",
    "mean_delta_rank2_normalized_gap",
    "median_delta_rank2_normalized_gap",
    "q25_delta_rank2_normalized_gap",
    "q75_delta_rank2_normalized_gap",
    "mean_baseline_rank2_normalized_gap",
    "mean_comparison_rank2_normalized_gap",
    "median_baseline_rank2_normalized_gap",
    "median_comparison_rank2_normalized_gap",
]


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def parse_float(value: object) -> float:
    if value is None or str(value).strip() == "":
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


def finite_quantile(values: list[float], probability: float) -> float:
    clean = sorted(finite_values(values))
    if not clean:
        return float("nan")
    if probability <= 0:
        return float(clean[0])
    if probability >= 1:
        return float(clean[-1])
    position = (len(clean) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(clean[lower])
    weight = position - lower
    return float(clean[lower] * (1.0 - weight) + clean[upper] * weight)


def row_key(row: dict[str, str]) -> tuple[int, int, str, str, int]:
    return (
        int(row["max_cycle"]),
        int(row["subset_seed"]),
        row["graph_id"],
        row["method_label"],
        int(row["solution_rank"]),
    )


def summary_sort_key(row: dict[str, Any]) -> tuple[int, int, int]:
    method_rank = METHOD_LABELS.index(row["method_label"]) if row["method_label"] in METHOD_LABELS else 99
    return (int(row["max_cycle"]), method_rank, int(row["solution_rank"]))


def rank2_case_sort_key(row: dict[str, Any]) -> tuple[int, int, int, str]:
    method_rank = METHOD_LABELS.index(row["method_label"]) if row["method_label"] in METHOD_LABELS else 99
    return (int(row["max_cycle"]), method_rank, int(row["subset_seed"]), str(row["graph_id"]))


def build_rank2_gap_by_case(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_key = {row_key(row): row for row in rows}
    output_rows: list[dict[str, Any]] = []

    rank1_rows = [row for row in rows if int(row["solution_rank"]) == 1]
    for rank1 in rank1_rows:
        cycle = int(rank1["max_cycle"])
        seed = int(rank1["subset_seed"])
        graph_id = rank1["graph_id"]
        method_label = rank1["method_label"]
        rank2 = by_key.get((cycle, seed, graph_id, method_label, 2))
        if rank2 is None:
            continue

        rank1_gap = parse_float(rank1.get("gap_to_oracle"))
        rank2_gap = parse_float(rank2.get("gap_to_oracle"))
        rank1_norm = parse_float(rank1.get("normalized_gap_to_oracle"))
        rank2_norm = parse_float(rank2.get("normalized_gap_to_oracle"))

        output_rows.append(
            {
                "regime": rank1.get("regime", ""),
                "max_cycle": cycle,
                "max_chain": int(rank1.get("max_chain", 0) or 0),
                "case_type": rank1.get("case_type", ""),
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
                "rank2_jaccard_rank1": parse_float(rank2.get("edge_jaccard_with_rank1")),
                "rank2_predicted_margin_from_best": parse_float(
                    rank2.get("predicted_margin_from_best")
                ),
                "rank2_true_obj_diff_from_rank1": parse_float(
                    rank2.get("true_obj_diff_from_rank1")
                ),
                "num_cycle_candidates": parse_float(rank2.get("num_cycle_candidates")),
                "num_chain_candidates": parse_float(rank2.get("num_chain_candidates")),
            }
        )

    return sorted(output_rows, key=rank2_case_sort_key)


def rank2_delta_map(rank2_rows: list[dict[str, Any]]) -> dict[tuple[int, str], list[float]]:
    grouped: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in rank2_rows:
        grouped[(int(row["max_cycle"]), row["method_label"])].append(
            float(row["rank2_minus_rank1_normalized_gap"])
        )
    return grouped


def build_second_best_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row.get("regime", ""),
                int(row["max_cycle"]),
                int(row.get("max_chain", 0) or 0),
                row["method_label"],
                int(row["solution_rank"]),
            )
        ].append(row)

    rank2_deltas = rank2_delta_map(build_rank2_gap_by_case(rows))
    output_rows: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda item: (item[1], METHOD_LABELS.index(item[3]) if item[3] in METHOD_LABELS else 99, item[4])):
        regime, max_cycle, max_chain, method_label, solution_rank = key
        group = grouped[key]
        gaps = [parse_float(row.get("gap_to_oracle")) for row in group]
        norm_gaps = [parse_float(row.get("normalized_gap_to_oracle")) for row in group]
        oracle_flags = [1.0 if parse_bool(row.get("same_solution_as_oracle")) else 0.0 for row in group]
        jaccards = [parse_float(row.get("edge_jaccard_with_oracle")) for row in group]
        margins = [parse_float(row.get("predicted_margin_from_best")) for row in group]
        cycle_counts = [parse_float(row.get("num_cycle_candidates")) for row in group]
        chain_counts = [parse_float(row.get("num_chain_candidates")) for row in group]
        deltas = rank2_deltas.get((max_cycle, method_label), []) if solution_rank == 2 else []

        output_rows.append(
            {
                "regime": regime,
                "max_cycle": max_cycle,
                "max_chain": max_chain,
                "method_label": method_label,
                "solution_rank": solution_rank,
                "row_count": len(group),
                "mean_gap_to_oracle": finite_mean(gaps),
                "median_gap_to_oracle": finite_median(gaps),
                "mean_normalized_gap_to_oracle": finite_mean(norm_gaps),
                "median_normalized_gap_to_oracle": finite_median(norm_gaps),
                "exact_oracle_rate": finite_mean(oracle_flags),
                "near_1pct_rate": sum(1 for value in norm_gaps if value <= 0.01) / len(group),
                "near_5pct_rate": sum(1 for value in norm_gaps if value <= 0.05) / len(group),
                "mean_rank2_minus_rank1_normalized_gap": finite_mean(deltas),
                "median_rank2_minus_rank1_normalized_gap": finite_median(deltas),
                "mean_edge_jaccard_with_oracle": finite_mean(jaccards),
                "median_edge_jaccard_with_oracle": finite_median(jaccards),
                "mean_predicted_margin_from_best": finite_mean(margins),
                "median_predicted_margin_from_best": finite_median(margins),
                "mean_num_cycle_candidates": finite_mean(cycle_counts),
                "median_num_cycle_candidates": finite_median(cycle_counts),
                "mean_num_chain_candidates": finite_mean(chain_counts),
                "median_num_chain_candidates": finite_median(chain_counts),
            }
        )

    return sorted(output_rows, key=summary_sort_key)


def case_key(row: dict[str, str]) -> tuple[int, str]:
    return (int(row["subset_seed"]), row["graph_id"])


def build_case_detail_rows(
    case_rows: list[dict[str, str]],
    rank_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    case_by_key = {case_key(row): row for row in case_rows}
    detail_rows: list[dict[str, Any]] = []
    for row in rank_rows:
        case = case_by_key.get((int(row["subset_seed"]), row["graph_id"]))
        if case is None:
            continue
        enriched = dict(row)
        enriched["case_label"] = case["case_label"]
        enriched["case_id"] = case["case_id"]
        detail_rows.append(enriched)
    return sorted(
        detail_rows,
        key=lambda row: (
            str(row["case_label"]),
            str(row["case_id"]),
            int(row["max_cycle"]),
            METHOD_LABELS.index(row["method_label"]) if row["method_label"] in METHOD_LABELS else 99,
        ),
    )


def build_case_summary(
    case_rows: list[dict[str, str]],
    rank_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    detail_rows = build_case_detail_rows(case_rows, rank_rows)
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in detail_rows:
        grouped[(row["case_label"], int(row["max_cycle"]), row["method_label"])].append(row)

    summary_rows: list[dict[str, Any]] = []
    for key in sorted(
        grouped,
        key=lambda item: (
            item[0],
            item[1],
            METHOD_LABELS.index(item[2]) if item[2] in METHOD_LABELS else 99,
        ),
    ):
        case_label, max_cycle, method_label = key
        group = grouped[key]
        rank1 = [float(row["rank1_normalized_gap"]) for row in group]
        rank2 = [float(row["rank2_normalized_gap"]) for row in group]
        deltas = [float(row["rank2_minus_rank1_normalized_gap"]) for row in group]
        near_1 = sum(1 for value in rank2 if value <= 0.01)
        near_5 = sum(1 for value in rank2 if value <= 0.05)

        summary_rows.append(
            {
                "case_label": case_label,
                "max_cycle": max_cycle,
                "method_label": method_label,
                "row_count": len(group),
                "mean_rank1_normalized_gap": finite_mean(rank1),
                "mean_rank2_normalized_gap": finite_mean(rank2),
                "median_rank2_normalized_gap": finite_median(rank2),
                "mean_rank2_minus_rank1_normalized_gap": finite_mean(deltas),
                "median_rank2_minus_rank1_normalized_gap": finite_median(deltas),
                "near_1pct_rank2_count": near_1,
                "rank2_near_1pct_rate": near_1 / len(group),
                "near_5pct_rank2_count": near_5,
                "rank2_near_5pct_rate": near_5 / len(group),
            }
        )

    return summary_rows


def paired_delta_key(row: dict[str, Any]) -> tuple[int, str, str]:
    return (int(row["subset_seed"]), str(row["graph_id"]), str(row["method_label"]))


def build_rank2_paired_delta_rows(
    rank_rows: list[dict[str, Any]],
    baseline_cycle: int = 3,
) -> list[dict[str, Any]]:
    rows_by_key: dict[tuple[int, str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rank_rows:
        rows_by_key[paired_delta_key(row)][int(row["max_cycle"])] = row

    delta_rows: list[dict[str, Any]] = []
    for key, cycle_rows in rows_by_key.items():
        baseline = cycle_rows.get(baseline_cycle)
        if baseline is None:
            continue
        for comparison_cycle in sorted(cycle for cycle in cycle_rows if cycle != baseline_cycle):
            comparison = cycle_rows[comparison_cycle]
            baseline_gap = parse_float(baseline.get("rank2_gap_to_oracle"))
            comparison_gap = parse_float(comparison.get("rank2_gap_to_oracle"))
            baseline_norm = parse_float(baseline.get("rank2_normalized_gap"))
            comparison_norm = parse_float(comparison.get("rank2_normalized_gap"))

            delta_rows.append(
                {
                    "regime": baseline.get("regime", ""),
                    "method_label": key[2],
                    "case_type": baseline.get("case_type", ""),
                    "subset_seed": key[0],
                    "graph_id": key[1],
                    "baseline_max_cycle": baseline_cycle,
                    "comparison_max_cycle": comparison_cycle,
                    "baseline_rank2_gap_to_oracle": baseline_gap,
                    "comparison_rank2_gap_to_oracle": comparison_gap,
                    "delta_rank2_gap_to_oracle": comparison_gap - baseline_gap,
                    "baseline_rank2_normalized_gap": baseline_norm,
                    "comparison_rank2_normalized_gap": comparison_norm,
                    "delta_rank2_normalized_gap": comparison_norm - baseline_norm,
                    "baseline_rank2_same_oracle": baseline.get("rank2_same_oracle", ""),
                    "comparison_rank2_same_oracle": comparison.get("rank2_same_oracle", ""),
                    "baseline_num_cycle_candidates": parse_float(
                        baseline.get("num_cycle_candidates")
                    ),
                    "comparison_num_cycle_candidates": parse_float(
                        comparison.get("num_cycle_candidates")
                    ),
                }
            )

    return sorted(
        delta_rows,
        key=lambda row: (
            int(row["comparison_max_cycle"]),
            method_sort(row["method_label"]),
            int(row["subset_seed"]),
            str(row["graph_id"]),
        ),
    )


def build_rank2_paired_delta_summary(
    delta_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in delta_rows:
        grouped[(int(row["comparison_max_cycle"]), str(row["method_label"]))].append(row)

    summary_rows: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda item: (item[0], method_sort(item[1]))):
        comparison_cycle, method_label = key
        group = grouped[key]
        deltas = [parse_float(row["delta_rank2_normalized_gap"]) for row in group]
        baseline_values = [parse_float(row["baseline_rank2_normalized_gap"]) for row in group]
        comparison_values = [
            parse_float(row["comparison_rank2_normalized_gap"]) for row in group
        ]
        finite_deltas = finite_values(deltas)
        count = len(finite_deltas)

        summary_rows.append(
            {
                "comparison_max_cycle": comparison_cycle,
                "method_label": method_label,
                "paired_count": count,
                "fraction_delta_lt_0": (
                    sum(1 for value in finite_deltas if value < 0.0) / count
                    if count
                    else float("nan")
                ),
                "fraction_delta_eq_0": (
                    sum(1 for value in finite_deltas if value == 0.0) / count
                    if count
                    else float("nan")
                ),
                "fraction_delta_gt_0": (
                    sum(1 for value in finite_deltas if value > 0.0) / count
                    if count
                    else float("nan")
                ),
                "mean_delta_rank2_normalized_gap": finite_mean(finite_deltas),
                "median_delta_rank2_normalized_gap": finite_median(finite_deltas),
                "q25_delta_rank2_normalized_gap": finite_quantile(finite_deltas, 0.25),
                "q75_delta_rank2_normalized_gap": finite_quantile(finite_deltas, 0.75),
                "mean_baseline_rank2_normalized_gap": finite_mean(baseline_values),
                "mean_comparison_rank2_normalized_gap": finite_mean(comparison_values),
                "median_baseline_rank2_normalized_gap": finite_median(baseline_values),
                "median_comparison_rank2_normalized_gap": finite_median(
                    comparison_values
                ),
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


def method_sort(method_label: str) -> int:
    return METHOD_LABELS.index(method_label) if method_label in METHOD_LABELS else 99


def prepare_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_rank2_gap(summary_rows: list[dict[str, Any]], output_path: str | Path) -> None:
    plt = prepare_matplotlib()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for method_label in METHOD_LABELS:
        rows = [
            row
            for row in summary_rows
            if row["method_label"] == method_label and int(row["solution_rank"]) == 2
        ]
        rows = sorted(rows, key=lambda row: int(row["max_cycle"]))
        ax.plot(
            [int(row["max_cycle"]) for row in rows],
            [float(row["mean_normalized_gap_to_oracle"]) for row in rows],
            marker="o",
            linewidth=1.8,
            label=METHOD_DISPLAY.get(method_label, method_label),
        )
    ax.set_xlabel("maximum cycle length")
    ax.set_ylabel("rank-2 mean normalized oracle gap")
    ax.set_xticks([3, 4, 5])
    ax.grid(axis="y", color="0.9", linewidth=0.8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_near5_rate(summary_rows: list[dict[str, Any]], output_path: str | Path) -> None:
    plt = prepare_matplotlib()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for method_label in METHOD_LABELS:
        rows = [
            row
            for row in summary_rows
            if row["method_label"] == method_label and int(row["solution_rank"]) == 2
        ]
        rows = sorted(rows, key=lambda row: int(row["max_cycle"]))
        ax.plot(
            [int(row["max_cycle"]) for row in rows],
            [100.0 * float(row["near_5pct_rate"]) for row in rows],
            marker="s",
            linewidth=1.8,
            label=METHOD_DISPLAY.get(method_label, method_label),
        )
    ax.set_xlabel("maximum cycle length")
    ax.set_ylabel("rank-2 within 5% oracle rate (%)")
    ax.set_xticks([3, 4, 5])
    ax.set_ylim(0, 100)
    ax.grid(axis="y", color="0.9", linewidth=0.8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_case_cycle_gaps(case_detail_rows: list[dict[str, Any]], output_path: str | Path) -> None:
    if not case_detail_rows:
        return

    plt = prepare_matplotlib()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    case_labels = sorted({row["case_label"] for row in case_detail_rows})
    fig, axes = plt.subplots(
        len(case_labels),
        len(METHOD_LABELS),
        figsize=(11.8, 3.3 * len(case_labels)),
        sharex=True,
        sharey=False,
    )
    if len(case_labels) == 1:
        axes = [axes]

    for row_idx, case_label in enumerate(case_labels):
        for col_idx, method_label in enumerate(METHOD_LABELS):
            ax = axes[row_idx][col_idx]
            rows = [
                row
                for row in case_detail_rows
                if row["case_label"] == case_label and row["method_label"] == method_label
            ]
            by_case_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                by_case_id[row["case_id"]].append(row)
            for case_id, group in sorted(by_case_id.items()):
                group = sorted(group, key=lambda row: int(row["max_cycle"]))
                label_base = f"{compact_case_id(case_id)} {group[0]['graph_id'].replace('.json', '')}"
                x_values = [int(row["max_cycle"]) for row in group]
                ax.plot(
                    x_values,
                    [float(row["rank1_normalized_gap"]) for row in group],
                    marker="o",
                    linewidth=1.4,
                    label=f"{label_base} rank1",
                )
                ax.plot(
                    x_values,
                    [float(row["rank2_normalized_gap"]) for row in group],
                    marker="s",
                    linewidth=1.4,
                    linestyle="--",
                    label=f"{label_base} rank2",
                )
            ax.axhline(0.05, color="0.35", linestyle=":", linewidth=1.0)
            ax.set_title(f"{CASE_DISPLAY.get(case_label, case_label)} / {METHOD_DISPLAY.get(method_label, method_label)}")
            ax.set_xticks([3, 4, 5])
            ax.grid(axis="y", color="0.9", linewidth=0.8)
            if col_idx == 0:
                ax.set_ylabel("normalized oracle gap")
            if row_idx == len(case_labels) - 1:
                ax.set_xlabel("maximum cycle length")
            ax.legend(fontsize=6, loc="best")

    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def input_paths(results_dir: Path, cycles: list[int]) -> list[Path]:
    return [results_dir / f"second_best_gap_maxcycle{cycle}.csv" for cycle in cycles]


def parse_cycles(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Create max-cycle 3/4/5 second-best sensitivity summaries and plots."
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--case-index", type=Path, default=DEFAULT_CASE_INDEX)
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS_DIR)
    parser.add_argument("--cycles", default="3,4,5")
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "cycle_length_second_best_summary.csv",
    )
    parser.add_argument(
        "--case-summary-output",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "cycle_length_case_summary.csv",
    )
    parser.add_argument(
        "--rank2-case-output",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "cycle_length_rank2_gap_by_case.csv",
    )
    parser.add_argument(
        "--paired-delta-output",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "cycle_length_rank2_paired_delta_by_case.csv",
    )
    parser.add_argument(
        "--paired-delta-summary-output",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "cycle_length_rank2_paired_delta_summary.csv",
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cycles = parse_cycles(args.cycles)
    paths = input_paths(args.results_dir, cycles)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required cycle sensitivity CSVs: {}".format(", ".join(missing)))

    rows: list[dict[str, str]] = []
    for path in paths:
        rows.extend(read_csv_rows(path))

    summary_rows = build_second_best_summary(rows)
    rank2_rows = build_rank2_gap_by_case(rows)
    case_rows = read_csv_rows(args.case_index) if args.case_index.exists() else []
    case_detail_rows = build_case_detail_rows(case_rows, rank2_rows)
    case_summary_rows = build_case_summary(case_rows, rank2_rows)
    paired_delta_rows = build_rank2_paired_delta_rows(rank2_rows, baseline_cycle=3)
    paired_delta_summary_rows = build_rank2_paired_delta_summary(paired_delta_rows)

    write_csv(args.summary_output, summary_rows, SECOND_BEST_SUMMARY_FIELDS)
    write_csv(args.rank2_case_output, rank2_rows, RANK2_CASE_FIELDS)
    write_csv(args.case_summary_output, case_summary_rows, CASE_SUMMARY_FIELDS)
    write_csv(args.paired_delta_output, paired_delta_rows, PAIRED_DELTA_FIELDS)
    write_csv(
        args.paired_delta_summary_output,
        paired_delta_summary_rows,
        PAIRED_DELTA_SUMMARY_FIELDS,
    )

    if not args.no_plot:
        plot_rank2_gap(summary_rows, args.plots_dir / "rank2_gap_by_cycle_length.png")
        plot_near5_rate(summary_rows, args.plots_dir / "near5_rate_by_cycle_length.png")
        plot_case_cycle_gaps(
            case_detail_rows,
            args.plots_dir / "case_abc_cycle_length_rank1_rank2_gap.png",
        )

    print(f"Loaded {len(rows)} solution rows from {len(paths)} files")
    print(f"Saved {len(summary_rows)} summary rows to {args.summary_output}")
    print(f"Saved {len(rank2_rows)} rank2 case rows to {args.rank2_case_output}")
    print(f"Saved {len(case_summary_rows)} selected case summary rows to {args.case_summary_output}")
    print(f"Saved {len(paired_delta_rows)} paired delta rows to {args.paired_delta_output}")
    print(
        "Saved {} paired delta summary rows to {}".format(
            len(paired_delta_summary_rows),
            args.paired_delta_summary_output,
        )
    )
    if not args.no_plot:
        print(f"Saved plots to {args.plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
