#!/usr/bin/env python3
"""Analyze observed solution margins and near-optimal alternatives."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = (
    PROJECT_ROOT / "surrogate_experiment_results" / "decision_analysis" / "results"
)

TWO_STAGE_LABEL = "2stage_val_mse"
SPOPLUS_LABEL = "spoplus_val_spoplus_loss"

MARGIN_FIELDS = [
    "regime",
    "case_type",
    "subset_seed",
    "graph_id",
    "optimal_obj",
    "obj_2stage",
    "obj_spoplus",
    "abs_obj_2stage_minus_spoplus",
    "gap_2stage",
    "gap_spoplus",
    "normalized_gap_2stage",
    "normalized_gap_spoplus",
    "normalized_gap_reduction",
    "edge_jaccard_2stage_with_opt",
    "edge_jaccard_spoplus_with_opt",
    "edge_jaccard_2stage_spoplus",
    "same_2stage_spoplus",
    "two_stage_different_solution_near_tie",
    "spoplus_different_solution_near_tie",
    "any_different_solution_near_tie",
    "observed_candidate_count",
    "observed_unique_solution_count",
]

CANDIDATE_FIELDS = [
    "regime",
    "case_type",
    "subset_seed",
    "graph_id",
    "candidate_label",
    "true_obj",
    "objective_gap_from_opt",
    "normalized_gap_from_opt",
    "rank_true_obj",
    "edge_count",
    "edge_jaccard_with_opt",
    "edge_jaccard_with_2stage",
    "edge_jaccard_with_spoplus",
    "same_as_opt",
    "same_as_2stage",
    "same_as_spoplus",
]

SUMMARY_FIELDS = [
    "scope",
    "near_tie_threshold",
    "graph_count",
    "mean_abs_obj_2stage_minus_spoplus",
    "mean_edge_jaccard_2stage_spoplus",
    "two_stage_near_tie_count",
    "two_stage_near_tie_rate",
    "spoplus_near_tie_count",
    "spoplus_near_tie_rate",
    "any_near_tie_count",
    "any_near_tie_rate",
    "same_2stage_spoplus_count",
    "same_2stage_spoplus_rate",
    "observed_unique_solution_count_mean",
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


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def parse_float(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    if value == "":
        return float("nan")
    return float(value)


def graph_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row["regime"], str(row["subset_seed"]), row["graph_id"])


def method_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (*graph_key(row), row["method_label"])


def selected_set(edge_rows: list[dict[str, str]], field: str) -> frozenset[int]:
    return frozenset(int(float(row["edge_id"])) for row in edge_rows if parse_bool(row[field]))


def jaccard(left: set[int] | frozenset[int], right: set[int] | frozenset[int]) -> float:
    union = left | right
    if not union:
        return 1.0
    return float(len(left & right)) / float(len(union))


def solution_overlap_metrics(
    edge_rows: list[dict[str, str]], left_field: str, right_field: str
) -> dict[str, object]:
    left = selected_set(edge_rows, left_field)
    right = selected_set(edge_rows, right_field)
    return {
        "same_solution": left == right,
        "left_count": len(left),
        "right_count": len(right),
        "intersection_count": len(left & right),
        "union_count": len(left | right),
        "edge_jaccard": jaccard(left, right),
    }


def build_paired_graph_rows(
    per_graph_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    by_method = {method_key(row): row for row in per_graph_rows}
    keys = sorted(
        {
            graph_key(row)
            for row in per_graph_rows
            if row.get("method_label") in {TWO_STAGE_LABEL, SPOPLUS_LABEL}
        }
    )
    records = []
    for key in keys:
        two_stage = by_method.get((*key, TWO_STAGE_LABEL))
        spoplus = by_method.get((*key, SPOPLUS_LABEL))
        if two_stage is None or spoplus is None:
            continue
        records.append(
            {
                "key": key,
                "regime": key[0],
                "subset_seed": key[1],
                "graph_id": key[2],
                "case_type": two_stage.get("case_type", ""),
                "two_stage": two_stage,
                "spoplus": spoplus,
            }
        )
    return records


def group_edge_rows(edge_rows: list[dict[str, str]]) -> dict[tuple[str, str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in edge_rows:
        grouped.setdefault(graph_key(row), []).append(row)
    return grouped


def dense_descending_ranks(values: dict[str, float]) -> dict[str, int]:
    unique_values = sorted({value for value in values.values()}, reverse=True)
    rank_by_value = {value: index + 1 for index, value in enumerate(unique_values)}
    return {key: rank_by_value[value] for key, value in values.items()}


def solution_sets(edge_rows: list[dict[str, str]]) -> dict[str, frozenset[int]]:
    return {
        "y_opt": selected_set(edge_rows, "in_opt"),
        "y_2stage": selected_set(edge_rows, "in_2stage"),
        "y_spoplus": selected_set(edge_rows, "in_spoplus"),
    }


def unique_solution_count(solutions: dict[str, frozenset[int]]) -> int:
    return len(set(solutions.values()))


def candidate_rows_for_graph(
    record: dict[str, object],
    edge_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    two_stage = record["two_stage"]
    spoplus = record["spoplus"]
    optimal_obj = parse_float(two_stage, "optimal_obj")
    candidates = {
        "y_opt": optimal_obj,
        "y_2stage": parse_float(two_stage, "achieved_obj"),
        "y_spoplus": parse_float(spoplus, "achieved_obj"),
    }
    ranks = dense_descending_ranks(candidates)
    solutions = solution_sets(edge_rows)
    output = []
    for label in ("y_opt", "y_2stage", "y_spoplus"):
        solution = solutions[label]
        true_obj = candidates[label]
        output.append(
            {
                "regime": record["regime"],
                "case_type": record["case_type"],
                "subset_seed": record["subset_seed"],
                "graph_id": record["graph_id"],
                "candidate_label": label,
                "true_obj": true_obj,
                "objective_gap_from_opt": optimal_obj - true_obj,
                "normalized_gap_from_opt": normalized_gap(optimal_obj - true_obj, optimal_obj),
                "rank_true_obj": ranks[label],
                "edge_count": len(solution),
                "edge_jaccard_with_opt": jaccard(solution, solutions["y_opt"]),
                "edge_jaccard_with_2stage": jaccard(solution, solutions["y_2stage"]),
                "edge_jaccard_with_spoplus": jaccard(solution, solutions["y_spoplus"]),
                "same_as_opt": solution == solutions["y_opt"],
                "same_as_2stage": solution == solutions["y_2stage"],
                "same_as_spoplus": solution == solutions["y_spoplus"],
            }
        )
    return output


def normalized_gap(gap: float, optimal_obj: float) -> float:
    denom = abs(float(optimal_obj))
    if denom <= 0.0:
        return 0.0 if abs(float(gap)) <= 0.0 else float("inf")
    return float(gap) / denom


def mean(values: list[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return float("nan")
    return sum(clean) / len(clean)


def bool_rate(rows: list[dict[str, object]], field: str) -> float:
    if not rows:
        return float("nan")
    return sum(1 for row in rows if bool(row[field])) / len(rows)


def summary_rows_for_margin_rows(
    margin_rows: list[dict[str, object]], near_tie_threshold: float
) -> list[dict[str, object]]:
    graph_count = len(margin_rows)
    two_count = sum(1 for row in margin_rows if bool(row["two_stage_different_solution_near_tie"]))
    spo_count = sum(1 for row in margin_rows if bool(row["spoplus_different_solution_near_tie"]))
    any_count = sum(1 for row in margin_rows if bool(row["any_different_solution_near_tie"]))
    same_count = sum(1 for row in margin_rows if bool(row["same_2stage_spoplus"]))
    return [
        {
            "scope": "all_paired_graphs",
            "near_tie_threshold": near_tie_threshold,
            "graph_count": graph_count,
            "mean_abs_obj_2stage_minus_spoplus": mean(
                [float(row["abs_obj_2stage_minus_spoplus"]) for row in margin_rows]
            ),
            "mean_edge_jaccard_2stage_spoplus": mean(
                [float(row["edge_jaccard_2stage_spoplus"]) for row in margin_rows]
            ),
            "two_stage_near_tie_count": two_count,
            "two_stage_near_tie_rate": two_count / graph_count if graph_count else float("nan"),
            "spoplus_near_tie_count": spo_count,
            "spoplus_near_tie_rate": spo_count / graph_count if graph_count else float("nan"),
            "any_near_tie_count": any_count,
            "any_near_tie_rate": any_count / graph_count if graph_count else float("nan"),
            "same_2stage_spoplus_count": same_count,
            "same_2stage_spoplus_rate": same_count / graph_count if graph_count else float("nan"),
            "observed_unique_solution_count_mean": mean(
                [float(row["observed_unique_solution_count"]) for row in margin_rows]
            ),
        }
    ]


def analyze_margin_near_ties(
    per_graph_rows: list[dict[str, str]],
    edge_rows: list[dict[str, str]],
    near_tie_threshold: float = 0.01,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    records = build_paired_graph_rows(per_graph_rows)
    edge_rows_by_key = group_edge_rows(edge_rows)
    margin_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []

    for record in records:
        rows = edge_rows_by_key.get(record["key"])
        if not rows:
            continue
        two_stage = record["two_stage"]
        spoplus = record["spoplus"]
        overlap = solution_overlap_metrics(rows, "in_2stage", "in_spoplus")
        candidates = candidate_rows_for_graph(record, rows)
        candidate_rows.extend(candidates)
        solutions = solution_sets(rows)

        optimal_obj = parse_float(two_stage, "optimal_obj")
        obj_2stage = parse_float(two_stage, "achieved_obj")
        obj_spoplus = parse_float(spoplus, "achieved_obj")
        norm_gap_2stage = parse_float(two_stage, "normalized_gap")
        norm_gap_spoplus = parse_float(spoplus, "normalized_gap")
        two_near_tie = (
            not parse_bool(two_stage.get("same_solution_as_opt", "False"))
            and norm_gap_2stage < near_tie_threshold
        )
        spo_near_tie = (
            not parse_bool(spoplus.get("same_solution_as_opt", "False"))
            and norm_gap_spoplus < near_tie_threshold
        )
        margin_rows.append(
            {
                "regime": record["regime"],
                "case_type": record["case_type"],
                "subset_seed": record["subset_seed"],
                "graph_id": record["graph_id"],
                "optimal_obj": optimal_obj,
                "obj_2stage": obj_2stage,
                "obj_spoplus": obj_spoplus,
                "abs_obj_2stage_minus_spoplus": abs(obj_2stage - obj_spoplus),
                "gap_2stage": parse_float(two_stage, "decision_gap"),
                "gap_spoplus": parse_float(spoplus, "decision_gap"),
                "normalized_gap_2stage": norm_gap_2stage,
                "normalized_gap_spoplus": norm_gap_spoplus,
                "normalized_gap_reduction": norm_gap_2stage - norm_gap_spoplus,
                "edge_jaccard_2stage_with_opt": parse_float(
                    two_stage, "edge_jaccard_with_opt"
                ),
                "edge_jaccard_spoplus_with_opt": parse_float(
                    spoplus, "edge_jaccard_with_opt"
                ),
                "edge_jaccard_2stage_spoplus": overlap["edge_jaccard"],
                "same_2stage_spoplus": overlap["same_solution"],
                "two_stage_different_solution_near_tie": two_near_tie,
                "spoplus_different_solution_near_tie": spo_near_tie,
                "any_different_solution_near_tie": two_near_tie or spo_near_tie,
                "observed_candidate_count": 3,
                "observed_unique_solution_count": unique_solution_count(solutions),
            }
        )
    summary_rows = summary_rows_for_margin_rows(margin_rows, near_tie_threshold)
    return margin_rows, candidate_rows, summary_rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Analyze margins and observed near-optimal alternative solutions."
    )
    parser.add_argument(
        "--per-graph",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "per_graph_decision_comparison.csv",
    )
    parser.add_argument(
        "--edge-rows",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "edge_error_criticality.csv",
    )
    parser.add_argument(
        "--margin-output",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "margin_near_tie_analysis.csv",
    )
    parser.add_argument(
        "--candidate-output",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "observed_candidate_solution_ranking.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "margin_near_tie_summary.csv",
    )
    parser.add_argument(
        "--near-tie-threshold",
        type=float,
        default=0.01,
        help="Normalized-gap threshold for different-solution near ties.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    margin_rows, candidate_rows, summary_rows = analyze_margin_near_ties(
        read_csv_rows(args.per_graph),
        read_csv_rows(args.edge_rows),
        near_tie_threshold=args.near_tie_threshold,
    )
    write_csv(args.margin_output, margin_rows, MARGIN_FIELDS)
    write_csv(args.candidate_output, candidate_rows, CANDIDATE_FIELDS)
    write_csv(args.summary_output, summary_rows, SUMMARY_FIELDS)
    print(f"Saved {len(margin_rows)} margin rows to {args.margin_output}")
    print(f"Saved {len(candidate_rows)} candidate rows to {args.candidate_output}")
    print(f"Saved {len(summary_rows)} summary rows to {args.summary_output}")
    if summary_rows:
        row = summary_rows[0]
        print(
            "Near-tie rates: "
            f"2stage={float(row['two_stage_near_tie_rate']):.6g}, "
            f"SPO+={float(row['spoplus_near_tie_rate']):.6g}, "
            f"any={float(row['any_near_tie_rate']):.6g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
