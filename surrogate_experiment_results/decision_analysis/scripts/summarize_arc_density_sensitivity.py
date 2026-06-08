#!/usr/bin/env python3
"""Summarize arc-density sensitivity second-best solution rows."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "decision_analysis"
    / "results"
    / "density_sensitivity"
)
DEFAULT_INPUT = DEFAULT_RESULTS_DIR / "arc_density_second_best_gap.csv"
DEFAULT_SUMMARY_OUTPUT = DEFAULT_RESULTS_DIR / "arc_density_second_best_summary.csv"
DEFAULT_CASE_SUMMARY_OUTPUT = DEFAULT_RESULTS_DIR / "arc_density_case_summary.csv"
DEFAULT_DELTA_OUTPUT = DEFAULT_RESULTS_DIR / "arc_density_delta_vs_original.csv"
DEFAULT_ORACLE_CHANGE_SUMMARY_OUTPUT = (
    DEFAULT_RESULTS_DIR / "arc_density_oracle_change_summary.csv"
)

METHOD_LABELS = ("2stage_val_mse", "spoplus_val_spoplus_loss")
VARIANT_ORDER = (
    "original",
    "add25pct",
    "add25arcs",
    "remove25arcs",
    "remove25pct",
)

SECOND_BEST_SUMMARY_FIELDS = [
    "density_variant",
    "arc_delta_type",
    "method_label",
    "solution_rank",
    "row_count",
    "mean_original_num_arcs",
    "mean_variant_num_arcs",
    "mean_arc_delta",
    "mean_gap_to_oracle",
    "median_gap_to_oracle",
    "mean_normalized_gap_to_oracle",
    "median_normalized_gap_to_oracle",
    "exact_oracle_rate",
    "near_1pct_rate",
    "near_5pct_rate",
    "solution_changed_vs_original_rate",
    "mean_rank2_minus_rank1_normalized_gap",
    "median_rank2_minus_rank1_normalized_gap",
    "mean_edge_jaccard_with_oracle",
    "median_edge_jaccard_with_oracle",
    "mean_predicted_margin_from_best",
    "median_predicted_margin_from_best",
    "mean_solution_selected_edge_count",
    "mean_num_added_arcs_in_solution",
    "mean_num_removed_arcs_from_original_solution",
    "mean_num_added_arcs_in_oracle",
    "mean_num_removed_arcs_from_original_oracle",
]

CASE_SUMMARY_FIELDS = [
    "case_id",
    "case_label",
    "base_graph_id",
    "subset_seed",
    "density_variant",
    "arc_delta_type",
    "original_num_arcs",
    "variant_num_arcs",
    "arc_delta",
    "added_arc_count",
    "removed_arc_count",
    "method_label",
    "oracle_obj",
    "oracle_obj_delta_vs_original",
    "oracle_obj_relative_delta_vs_original",
    "oracle_solution_changed_vs_original",
    "rank1_true_obj",
    "rank2_true_obj",
    "rank1_gap_to_oracle",
    "rank2_gap_to_oracle",
    "rank2_minus_rank1_gap_to_oracle",
    "rank1_normalized_gap_to_oracle",
    "rank2_normalized_gap_to_oracle",
    "rank2_minus_rank1_normalized_gap",
    "rank1_near_1pct",
    "rank2_near_1pct",
    "rank1_near_5pct",
    "rank2_near_5pct",
    "rank1_solution_changed_vs_original",
    "rank2_solution_changed_vs_original",
    "rank1_same_oracle",
    "rank2_same_oracle",
    "rank1_edge_jaccard_with_oracle",
    "rank2_edge_jaccard_with_oracle",
    "rank2_edge_jaccard_with_rank1",
    "rank1_solution_selected_edge_count",
    "rank2_solution_selected_edge_count",
    "num_added_arcs_in_oracle",
    "num_added_arcs_in_rank1",
    "num_added_arcs_in_rank2",
    "num_removed_arcs_from_original_oracle",
    "num_removed_arcs_from_original_rank1",
    "num_removed_arcs_from_original_rank2",
    "rank1_arc_key_signature",
    "rank2_arc_key_signature",
    "oracle_arc_key_signature",
]

DELTA_FIELDS = [
    "case_id",
    "case_label",
    "base_graph_id",
    "subset_seed",
    "density_variant",
    "arc_delta_type",
    "original_num_arcs",
    "variant_num_arcs",
    "arc_delta",
    "method_label",
    "original_oracle_obj",
    "variant_oracle_obj",
    "delta_oracle_obj",
    "relative_delta_oracle_obj",
    "oracle_solution_changed_vs_original",
    "original_rank1_normalized_gap",
    "variant_rank1_normalized_gap",
    "delta_rank1_normalized_gap",
    "original_rank2_normalized_gap",
    "variant_rank2_normalized_gap",
    "delta_rank2_normalized_gap",
    "original_rank2_minus_rank1_normalized_gap",
    "variant_rank2_minus_rank1_normalized_gap",
    "delta_rank2_minus_rank1_normalized_gap",
    "rank1_solution_changed_vs_original",
    "rank2_solution_changed_vs_original",
    "num_added_arcs_in_oracle",
    "num_added_arcs_in_rank1",
    "num_added_arcs_in_rank2",
    "num_removed_arcs_from_original_oracle",
    "num_removed_arcs_from_original_rank1",
    "num_removed_arcs_from_original_rank2",
]

ORACLE_CHANGE_SUMMARY_FIELDS = [
    "density_variant",
    "arc_delta_type",
    "method_label",
    "graph_count",
    "rank1_pair_count",
    "rank2_pair_count",
    "fraction_oracle_obj_increased",
    "fraction_oracle_obj_unchanged",
    "fraction_oracle_obj_decreased",
    "fraction_oracle_solution_changed",
    "mean_oracle_obj_delta",
    "median_oracle_obj_delta",
    "q25_oracle_obj_delta",
    "q75_oracle_obj_delta",
    "mean_relative_oracle_obj_delta",
    "median_relative_oracle_obj_delta",
    "fraction_rank1_solution_changed",
    "fraction_rank1_solution_unchanged",
    "fraction_rank2_solution_changed",
    "fraction_rank2_solution_unchanged",
    "mean_num_added_arcs_in_oracle",
    "mean_num_added_arcs_in_rank1",
    "mean_num_added_arcs_in_rank2",
    "mean_num_removed_arcs_from_original_oracle",
    "mean_num_removed_arcs_from_original_rank1",
    "mean_num_removed_arcs_from_original_rank2",
]


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_float(value: object) -> float:
    if value is None or str(value).strip() == "":
        return float("nan")
    return float(value)


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


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


def signature_set(value: object) -> set[str]:
    text = str(value or "").strip()
    if not text:
        return set()
    return {token for token in text.split("|") if token}


def solution_key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(row.get("base_graph_id", "")),
        str(row.get("density_variant", "")),
        str(row.get("method_label", "")),
        int(float(row.get("solution_rank", 0))),
    )


def graph_variant_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("base_graph_id", "")), str(row.get("density_variant", "")))


def method_sort(method_label: str) -> int:
    return METHOD_LABELS.index(method_label) if method_label in METHOD_LABELS else 99


def variant_sort(density_variant: str) -> int:
    return VARIANT_ORDER.index(density_variant) if density_variant in VARIANT_ORDER else 99


def build_original_solution_sets(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str, int], set[str]]:
    output: dict[tuple[str, str, int], set[str]] = {}
    for row in rows:
        if str(row.get("density_variant", "")) != "original":
            continue
        output[
            (
                str(row.get("base_graph_id", "")),
                str(row.get("method_label", "")),
                int(float(row.get("solution_rank", 0))),
            )
        ] = signature_set(row.get("solution_arc_key_signature", ""))
    return output


def build_original_oracle(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        if str(row.get("density_variant", "")) != "original":
            continue
        base_graph_id = str(row.get("base_graph_id", ""))
        output.setdefault(
            base_graph_id,
            {
                "oracle_obj": parse_float(row.get("oracle_obj")),
                "oracle_set": signature_set(row.get("oracle_arc_key_signature", "")),
            },
        )
    return output


def row_solution_metrics(
    row: dict[str, Any],
    original_solution_sets: dict[tuple[str, str, int], set[str]],
    original_oracle: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    base_graph_id = str(row.get("base_graph_id", ""))
    method_label = str(row.get("method_label", ""))
    rank = int(float(row.get("solution_rank", 0)))
    density_variant = str(row.get("density_variant", ""))

    added_set = signature_set(row.get("added_arc_keys", ""))
    removed_set = signature_set(row.get("removed_arc_keys", ""))
    solution_set = signature_set(row.get("solution_arc_key_signature", ""))
    oracle_set = signature_set(row.get("oracle_arc_key_signature", ""))
    original_solution_set = original_solution_sets.get((base_graph_id, method_label, rank), set())
    original_oracle_set = original_oracle.get(base_graph_id, {}).get("oracle_set", set())

    if density_variant == "original":
        solution_changed = False
        oracle_changed = False
    else:
        solution_changed = solution_set != original_solution_set
        oracle_changed = oracle_set != original_oracle_set

    return {
        "solution_changed_vs_original": solution_changed,
        "oracle_solution_changed_vs_original": oracle_changed,
        "num_added_arcs_in_solution": len(added_set & solution_set),
        "num_removed_arcs_from_original_solution": len(removed_set & original_solution_set),
        "num_added_arcs_in_oracle": len(added_set & oracle_set),
        "num_removed_arcs_from_original_oracle": len(removed_set & original_oracle_set),
    }


def enriched_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    original_solution_sets = build_original_solution_sets(rows)
    original_oracle = build_original_oracle(rows)
    output: list[dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        enriched.update(row_solution_metrics(row, original_solution_sets, original_oracle))
        output.append(enriched)
    return output


def build_rank_pair_map(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[int, dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        base_graph_id, density_variant, method_label, rank = solution_key(row)
        grouped[(base_graph_id, density_variant, method_label)][rank] = row
    return grouped


def build_case_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = enriched_rows(rows)
    rank_pairs = build_rank_pair_map(rows)
    original_oracle = build_original_oracle(rows)
    output: list[dict[str, Any]] = []

    for key in sorted(
        rank_pairs,
        key=lambda item: (item[0], variant_sort(item[1]), method_sort(item[2])),
    ):
        base_graph_id, density_variant, method_label = key
        pair = rank_pairs[key]
        rank1 = pair.get(1)
        rank2 = pair.get(2)
        if rank1 is None or rank2 is None:
            continue

        original_obj = original_oracle.get(base_graph_id, {}).get("oracle_obj", float("nan"))
        oracle_obj = parse_float(rank1.get("oracle_obj"))
        oracle_delta = oracle_obj - original_obj
        rank1_norm = parse_float(rank1.get("normalized_gap_to_oracle"))
        rank2_norm = parse_float(rank2.get("normalized_gap_to_oracle"))
        rank1_gap = parse_float(rank1.get("gap_to_oracle"))
        rank2_gap = parse_float(rank2.get("gap_to_oracle"))

        output.append(
            {
                "case_id": rank1.get("case_id", ""),
                "case_label": rank1.get("case_label", rank1.get("case_type", "")),
                "base_graph_id": base_graph_id,
                "subset_seed": int(float(rank1.get("subset_seed", 0) or 0)),
                "density_variant": density_variant,
                "arc_delta_type": rank1.get("arc_delta_type", ""),
                "original_num_arcs": int(float(rank1.get("original_num_arcs", 0) or 0)),
                "variant_num_arcs": int(float(rank1.get("variant_num_arcs", 0) or 0)),
                "arc_delta": int(float(rank1.get("arc_delta", 0) or 0)),
                "added_arc_count": int(float(rank1.get("added_arc_count", 0) or 0)),
                "removed_arc_count": int(float(rank1.get("removed_arc_count", 0) or 0)),
                "method_label": method_label,
                "oracle_obj": oracle_obj,
                "oracle_obj_delta_vs_original": oracle_delta,
                "oracle_obj_relative_delta_vs_original": (
                    oracle_delta / abs(original_obj) if original_obj else float("nan")
                ),
                "oracle_solution_changed_vs_original": rank1[
                    "oracle_solution_changed_vs_original"
                ],
                "rank1_true_obj": parse_float(rank1.get("true_obj")),
                "rank2_true_obj": parse_float(rank2.get("true_obj")),
                "rank1_gap_to_oracle": rank1_gap,
                "rank2_gap_to_oracle": rank2_gap,
                "rank2_minus_rank1_gap_to_oracle": rank2_gap - rank1_gap,
                "rank1_normalized_gap_to_oracle": rank1_norm,
                "rank2_normalized_gap_to_oracle": rank2_norm,
                "rank2_minus_rank1_normalized_gap": rank2_norm - rank1_norm,
                "rank1_near_1pct": rank1_norm <= 0.01,
                "rank2_near_1pct": rank2_norm <= 0.01,
                "rank1_near_5pct": rank1_norm <= 0.05,
                "rank2_near_5pct": rank2_norm <= 0.05,
                "rank1_solution_changed_vs_original": rank1[
                    "solution_changed_vs_original"
                ],
                "rank2_solution_changed_vs_original": rank2[
                    "solution_changed_vs_original"
                ],
                "rank1_same_oracle": parse_bool(rank1.get("same_solution_as_oracle")),
                "rank2_same_oracle": parse_bool(rank2.get("same_solution_as_oracle")),
                "rank1_edge_jaccard_with_oracle": parse_float(
                    rank1.get("edge_jaccard_with_oracle")
                ),
                "rank2_edge_jaccard_with_oracle": parse_float(
                    rank2.get("edge_jaccard_with_oracle")
                ),
                "rank2_edge_jaccard_with_rank1": parse_float(
                    rank2.get("edge_jaccard_with_rank1")
                ),
                "rank1_solution_selected_edge_count": parse_float(
                    rank1.get("solution_selected_edge_count", rank1.get("edge_count"))
                ),
                "rank2_solution_selected_edge_count": parse_float(
                    rank2.get("solution_selected_edge_count", rank2.get("edge_count"))
                ),
                "num_added_arcs_in_oracle": rank1["num_added_arcs_in_oracle"],
                "num_added_arcs_in_rank1": rank1["num_added_arcs_in_solution"],
                "num_added_arcs_in_rank2": rank2["num_added_arcs_in_solution"],
                "num_removed_arcs_from_original_oracle": rank1[
                    "num_removed_arcs_from_original_oracle"
                ],
                "num_removed_arcs_from_original_rank1": rank1[
                    "num_removed_arcs_from_original_solution"
                ],
                "num_removed_arcs_from_original_rank2": rank2[
                    "num_removed_arcs_from_original_solution"
                ],
                "rank1_arc_key_signature": rank1.get("solution_arc_key_signature", ""),
                "rank2_arc_key_signature": rank2.get("solution_arc_key_signature", ""),
                "oracle_arc_key_signature": rank1.get("oracle_arc_key_signature", ""),
            }
        )

    return output


def build_second_best_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched = enriched_rows(rows)
    case_summary = build_case_summary(rows)
    rank2_delta_by_variant_method: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in case_summary:
        rank2_delta_by_variant_method[
            (str(row["density_variant"]), str(row["method_label"]))
        ].append(parse_float(row["rank2_minus_rank1_normalized_gap"]))

    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        grouped[
            (
                str(row.get("density_variant", "")),
                str(row.get("arc_delta_type", "")),
                str(row.get("method_label", "")),
                int(float(row.get("solution_rank", 0))),
            )
        ].append(row)

    output: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda item: (variant_sort(item[0]), method_sort(item[2]), item[3])):
        density_variant, arc_delta_type, method_label, rank = key
        group = grouped[key]
        norm_gaps = [parse_float(row.get("normalized_gap_to_oracle")) for row in group]
        gaps = [parse_float(row.get("gap_to_oracle")) for row in group]
        oracle_flags = [1.0 if parse_bool(row.get("same_solution_as_oracle")) else 0.0 for row in group]
        changed_flags = [
            1.0 if row["solution_changed_vs_original"] else 0.0 for row in group
        ]
        rank2_deltas = (
            rank2_delta_by_variant_method.get((density_variant, method_label), [])
            if rank == 2
            else []
        )

        output.append(
            {
                "density_variant": density_variant,
                "arc_delta_type": arc_delta_type,
                "method_label": method_label,
                "solution_rank": rank,
                "row_count": len(group),
                "mean_original_num_arcs": finite_mean(
                    [parse_float(row.get("original_num_arcs")) for row in group]
                ),
                "mean_variant_num_arcs": finite_mean(
                    [parse_float(row.get("variant_num_arcs")) for row in group]
                ),
                "mean_arc_delta": finite_mean(
                    [parse_float(row.get("arc_delta")) for row in group]
                ),
                "mean_gap_to_oracle": finite_mean(gaps),
                "median_gap_to_oracle": finite_median(gaps),
                "mean_normalized_gap_to_oracle": finite_mean(norm_gaps),
                "median_normalized_gap_to_oracle": finite_median(norm_gaps),
                "exact_oracle_rate": finite_mean(oracle_flags),
                "near_1pct_rate": finite_mean(
                    [1.0 if value <= 0.01 else 0.0 for value in norm_gaps]
                ),
                "near_5pct_rate": finite_mean(
                    [1.0 if value <= 0.05 else 0.0 for value in norm_gaps]
                ),
                "solution_changed_vs_original_rate": finite_mean(changed_flags),
                "mean_rank2_minus_rank1_normalized_gap": finite_mean(rank2_deltas),
                "median_rank2_minus_rank1_normalized_gap": finite_median(rank2_deltas),
                "mean_edge_jaccard_with_oracle": finite_mean(
                    [parse_float(row.get("edge_jaccard_with_oracle")) for row in group]
                ),
                "median_edge_jaccard_with_oracle": finite_median(
                    [parse_float(row.get("edge_jaccard_with_oracle")) for row in group]
                ),
                "mean_predicted_margin_from_best": finite_mean(
                    [parse_float(row.get("predicted_margin_from_best")) for row in group]
                ),
                "median_predicted_margin_from_best": finite_median(
                    [parse_float(row.get("predicted_margin_from_best")) for row in group]
                ),
                "mean_solution_selected_edge_count": finite_mean(
                    [
                        parse_float(
                            row.get("solution_selected_edge_count", row.get("edge_count"))
                        )
                        for row in group
                    ]
                ),
                "mean_num_added_arcs_in_solution": finite_mean(
                    [float(row["num_added_arcs_in_solution"]) for row in group]
                ),
                "mean_num_removed_arcs_from_original_solution": finite_mean(
                    [
                        float(row["num_removed_arcs_from_original_solution"])
                        for row in group
                    ]
                ),
                "mean_num_added_arcs_in_oracle": finite_mean(
                    [float(row["num_added_arcs_in_oracle"]) for row in group]
                ),
                "mean_num_removed_arcs_from_original_oracle": finite_mean(
                    [
                        float(row["num_removed_arcs_from_original_oracle"])
                        for row in group
                    ]
                ),
            }
        )

    return output


def build_delta_vs_original(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    case_rows = build_case_summary(rows)
    original_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in case_rows:
        if row["density_variant"] == "original":
            original_by_key[(str(row["base_graph_id"]), str(row["method_label"]))] = row

    output: list[dict[str, Any]] = []
    for row in case_rows:
        original = original_by_key.get((str(row["base_graph_id"]), str(row["method_label"])))
        if original is None:
            continue

        output.append(
            {
                "case_id": row["case_id"],
                "case_label": row["case_label"],
                "base_graph_id": row["base_graph_id"],
                "subset_seed": row["subset_seed"],
                "density_variant": row["density_variant"],
                "arc_delta_type": row["arc_delta_type"],
                "original_num_arcs": row["original_num_arcs"],
                "variant_num_arcs": row["variant_num_arcs"],
                "arc_delta": row["arc_delta"],
                "method_label": row["method_label"],
                "original_oracle_obj": original["oracle_obj"],
                "variant_oracle_obj": row["oracle_obj"],
                "delta_oracle_obj": row["oracle_obj"] - original["oracle_obj"],
                "relative_delta_oracle_obj": (
                    (row["oracle_obj"] - original["oracle_obj"])
                    / abs(original["oracle_obj"])
                    if original["oracle_obj"]
                    else float("nan")
                ),
                "oracle_solution_changed_vs_original": row[
                    "oracle_solution_changed_vs_original"
                ],
                "original_rank1_normalized_gap": original[
                    "rank1_normalized_gap_to_oracle"
                ],
                "variant_rank1_normalized_gap": row["rank1_normalized_gap_to_oracle"],
                "delta_rank1_normalized_gap": row["rank1_normalized_gap_to_oracle"]
                - original["rank1_normalized_gap_to_oracle"],
                "original_rank2_normalized_gap": original[
                    "rank2_normalized_gap_to_oracle"
                ],
                "variant_rank2_normalized_gap": row["rank2_normalized_gap_to_oracle"],
                "delta_rank2_normalized_gap": row["rank2_normalized_gap_to_oracle"]
                - original["rank2_normalized_gap_to_oracle"],
                "original_rank2_minus_rank1_normalized_gap": original[
                    "rank2_minus_rank1_normalized_gap"
                ],
                "variant_rank2_minus_rank1_normalized_gap": row[
                    "rank2_minus_rank1_normalized_gap"
                ],
                "delta_rank2_minus_rank1_normalized_gap": row[
                    "rank2_minus_rank1_normalized_gap"
                ]
                - original["rank2_minus_rank1_normalized_gap"],
                "rank1_solution_changed_vs_original": row[
                    "rank1_solution_changed_vs_original"
                ],
                "rank2_solution_changed_vs_original": row[
                    "rank2_solution_changed_vs_original"
                ],
                "num_added_arcs_in_oracle": row["num_added_arcs_in_oracle"],
                "num_added_arcs_in_rank1": row["num_added_arcs_in_rank1"],
                "num_added_arcs_in_rank2": row["num_added_arcs_in_rank2"],
                "num_removed_arcs_from_original_oracle": row[
                    "num_removed_arcs_from_original_oracle"
                ],
                "num_removed_arcs_from_original_rank1": row[
                    "num_removed_arcs_from_original_rank1"
                ],
                "num_removed_arcs_from_original_rank2": row[
                    "num_removed_arcs_from_original_rank2"
                ],
            }
        )

    return sorted(
        output,
        key=lambda row: (
            str(row["base_graph_id"]),
            variant_sort(str(row["density_variant"])),
            method_sort(str(row["method_label"])),
        ),
    )


def build_oracle_change_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    case_rows = build_case_summary(rows)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in case_rows:
        grouped[
            (
                str(row["density_variant"]),
                str(row["arc_delta_type"]),
                str(row["method_label"]),
            )
        ].append(row)

    output: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda item: (variant_sort(item[0]), method_sort(item[2]))):
        density_variant, arc_delta_type, method_label = key
        group = grouped[key]
        oracle_deltas = [parse_float(row["oracle_obj_delta_vs_original"]) for row in group]
        relative_deltas = [
            parse_float(row["oracle_obj_relative_delta_vs_original"]) for row in group
        ]
        rank1_changed = [
            1.0 if row["rank1_solution_changed_vs_original"] else 0.0 for row in group
        ]
        rank2_changed = [
            1.0 if row["rank2_solution_changed_vs_original"] else 0.0 for row in group
        ]
        oracle_changed = [
            1.0 if row["oracle_solution_changed_vs_original"] else 0.0 for row in group
        ]
        count = len(group)

        output.append(
            {
                "density_variant": density_variant,
                "arc_delta_type": arc_delta_type,
                "method_label": method_label,
                "graph_count": count,
                "rank1_pair_count": count,
                "rank2_pair_count": count,
                "fraction_oracle_obj_increased": (
                    sum(1 for value in oracle_deltas if value > 1e-9) / count
                    if count
                    else float("nan")
                ),
                "fraction_oracle_obj_unchanged": (
                    sum(1 for value in oracle_deltas if abs(value) <= 1e-9) / count
                    if count
                    else float("nan")
                ),
                "fraction_oracle_obj_decreased": (
                    sum(1 for value in oracle_deltas if value < -1e-9) / count
                    if count
                    else float("nan")
                ),
                "fraction_oracle_solution_changed": finite_mean(oracle_changed),
                "mean_oracle_obj_delta": finite_mean(oracle_deltas),
                "median_oracle_obj_delta": finite_median(oracle_deltas),
                "q25_oracle_obj_delta": finite_quantile(oracle_deltas, 0.25),
                "q75_oracle_obj_delta": finite_quantile(oracle_deltas, 0.75),
                "mean_relative_oracle_obj_delta": finite_mean(relative_deltas),
                "median_relative_oracle_obj_delta": finite_median(relative_deltas),
                "fraction_rank1_solution_changed": finite_mean(rank1_changed),
                "fraction_rank1_solution_unchanged": finite_mean(
                    [1.0 - value for value in rank1_changed]
                ),
                "fraction_rank2_solution_changed": finite_mean(rank2_changed),
                "fraction_rank2_solution_unchanged": finite_mean(
                    [1.0 - value for value in rank2_changed]
                ),
                "mean_num_added_arcs_in_oracle": finite_mean(
                    [parse_float(row["num_added_arcs_in_oracle"]) for row in group]
                ),
                "mean_num_added_arcs_in_rank1": finite_mean(
                    [parse_float(row["num_added_arcs_in_rank1"]) for row in group]
                ),
                "mean_num_added_arcs_in_rank2": finite_mean(
                    [parse_float(row["num_added_arcs_in_rank2"]) for row in group]
                ),
                "mean_num_removed_arcs_from_original_oracle": finite_mean(
                    [
                        parse_float(row["num_removed_arcs_from_original_oracle"])
                        for row in group
                    ]
                ),
                "mean_num_removed_arcs_from_original_rank1": finite_mean(
                    [
                        parse_float(row["num_removed_arcs_from_original_rank1"])
                        for row in group
                    ]
                ),
                "mean_num_removed_arcs_from_original_rank2": finite_mean(
                    [
                        parse_float(row["num_removed_arcs_from_original_rank2"])
                        for row in group
                    ]
                ),
            }
        )

    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Summarize arc-density sensitivity second-best solution rows."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument(
        "--case-summary-output",
        type=Path,
        default=DEFAULT_CASE_SUMMARY_OUTPUT,
    )
    parser.add_argument("--delta-output", type=Path, default=DEFAULT_DELTA_OUTPUT)
    parser.add_argument(
        "--oracle-change-summary-output",
        type=Path,
        default=DEFAULT_ORACLE_CHANGE_SUMMARY_OUTPUT,
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    rows = read_csv_rows(args.input)

    second_best_rows = build_second_best_summary(rows)
    case_rows = build_case_summary(rows)
    delta_rows = build_delta_vs_original(rows)
    oracle_change_rows = build_oracle_change_summary(rows)

    write_csv(args.summary_output, second_best_rows, SECOND_BEST_SUMMARY_FIELDS)
    write_csv(args.case_summary_output, case_rows, CASE_SUMMARY_FIELDS)
    write_csv(args.delta_output, delta_rows, DELTA_FIELDS)
    write_csv(
        args.oracle_change_summary_output,
        oracle_change_rows,
        ORACLE_CHANGE_SUMMARY_FIELDS,
    )

    print(f"Loaded {len(rows)} formal density rows from {args.input}")
    print(f"Saved {len(second_best_rows)} rows to {args.summary_output}")
    print(f"Saved {len(case_rows)} rows to {args.case_summary_output}")
    print(f"Saved {len(delta_rows)} rows to {args.delta_output}")
    print(f"Saved {len(oracle_change_rows)} rows to {args.oracle_change_summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
