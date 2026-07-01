#!/usr/bin/env python3
"""Summarize Step4 rank-reversal target contexts."""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DETAIL_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = DETAIL_DIR.parents[1]
STRUCTURAL_SCRIPTS = PROJECT_ROOT / "surrogate_experiment_results" / "Step4 Topology Structural Atlas" / "scripts"
if str(STRUCTURAL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(STRUCTURAL_SCRIPTS))

import step4_topology_common as common  # noqa: E402


FULL_RUN_DIR = DETAIL_DIR / "results" / "full_8topology_5seed_3size"
DEFAULT_DECISION_ROWS = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "Step4 Decision Overlay"
    / "results"
    / "full_8topology_5seed_3size"
    / "decision_solution_rows.csv"
)
DEFAULT_TARGET_ROWS = FULL_RUN_DIR / "rank_reversal_all_different_contexts.csv"
DEFAULT_CASE_OUTPUT = FULL_RUN_DIR / "rank_reversal_case_summary_by_topology_sample.csv"
DEFAULT_SWITCH_OUTPUT = FULL_RUN_DIR / "candidate_set_switch_summary.csv"

CASE_SUMMARY_FIELDS = [
    "topology_id",
    "sample_size",
    "total_contexts",
    "same_decision_contexts",
    "different_decision_contexts",
    "different_decision_rate",
    "beneficial_reversal_count",
    "harmful_reversal_count",
    "tie_reversal_count",
    "beneficial_rate_total",
    "harmful_rate_total",
    "tie_rate_total",
    "mean_delta_different",
    "mean_abs_delta_different",
    "max_beneficial_delta",
    "max_harmful_abs_delta",
]

SWITCH_SUMMARY_FIELDS = [
    "topology_id",
    "sample_size",
    "two_stage_candidate_ids",
    "spoplus_candidate_ids",
    "case_direction",
    "count",
    "rate_total",
    "mean_delta",
    "mean_abs_delta",
    "max_abs_delta",
    "mean_gap_2stage",
    "mean_gap_spoplus",
    "dominant_oracle_candidate_ids",
    "dominant_oracle_count",
    "example_data_seed",
    "example_test_sample_index",
    "example_graph",
    "example_abs_delta",
]


def decision_context_key(row: dict[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(row["topology_id"]),
        common.parse_int(row["sample_size"]),
        common.parse_int(row["data_seed"]),
        common.parse_int(row["test_sample_index"]),
    )


def target_group_key(row: dict[str, Any]) -> tuple[str, int]:
    return (str(row["topology_id"]), common.parse_int(row["sample_size"]))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def count_total_contexts(decision_rows: list[dict[str, Any]]) -> dict[tuple[str, int], int]:
    grouped: dict[tuple[str, int, int, int], set[str]] = defaultdict(set)
    for row in decision_rows:
        grouped[decision_context_key(row)].add(str(row["solution_source"]))

    totals: dict[tuple[str, int], int] = defaultdict(int)
    for key, roles in grouped.items():
        if {"oracle", "2stage", "spoplus"} <= roles:
            topology_id, sample_size, _data_seed, _sample_index = key
            totals[(topology_id, sample_size)] += 1
    return dict(totals)


def summarize_cases(
    total_contexts: dict[tuple[str, int], int],
    target_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_group: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in target_rows:
        by_group[target_group_key(row)].append(row)

    output: list[dict[str, Any]] = []
    for key in sorted(set(total_contexts) | set(by_group)):
        total = total_contexts.get(key, 0)
        rows = by_group.get(key, [])
        direction_counts = Counter(str(row["case_direction"]) for row in rows)
        deltas = [common.parse_float(row["true_delta_spoplus_minus_2stage"]) for row in rows]
        abs_deltas = [abs(value) for value in deltas]
        beneficial = direction_counts["beneficial_reversal"]
        harmful = direction_counts["harmful_reversal"]
        tie = direction_counts["decision_different_true_tie"]
        positive_deltas = [value for value in deltas if value > 0]
        harmful_abs_deltas = [abs(value) for value in deltas if value < 0]
        topology_id, sample_size = key
        output.append(
            {
                "topology_id": topology_id,
                "sample_size": sample_size,
                "total_contexts": total,
                "same_decision_contexts": total - len(rows),
                "different_decision_contexts": len(rows),
                "different_decision_rate": len(rows) / total if total else 0.0,
                "beneficial_reversal_count": beneficial,
                "harmful_reversal_count": harmful,
                "tie_reversal_count": tie,
                "beneficial_rate_total": beneficial / total if total else 0.0,
                "harmful_rate_total": harmful / total if total else 0.0,
                "tie_rate_total": tie / total if total else 0.0,
                "mean_delta_different": mean(deltas),
                "mean_abs_delta_different": mean(abs_deltas),
                "max_beneficial_delta": max(positive_deltas) if positive_deltas else 0.0,
                "max_harmful_abs_delta": max(harmful_abs_deltas) if harmful_abs_deltas else 0.0,
            }
        )
    return output


def summarize_switches(
    total_contexts: dict[tuple[str, int], int],
    target_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_switch: dict[tuple[str, int, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in target_rows:
        key = (
            str(row["topology_id"]),
            common.parse_int(row["sample_size"]),
            str(row["two_stage_candidate_ids"]),
            str(row["spoplus_candidate_ids"]),
            str(row["case_direction"]),
        )
        by_switch[key].append(row)

    output: list[dict[str, Any]] = []
    for key, rows in by_switch.items():
        topology_id, sample_size, two_ids, spo_ids, direction = key
        total = total_contexts.get((topology_id, sample_size), 0)
        deltas = [common.parse_float(row["true_delta_spoplus_minus_2stage"]) for row in rows]
        abs_deltas = [abs(value) for value in deltas]
        oracle_counts = Counter(str(row["oracle_candidate_ids"]) for row in rows)
        dominant_oracle, dominant_count = oracle_counts.most_common(1)[0]
        example = max(rows, key=lambda row: common.parse_float(row["abs_true_delta"]))
        output.append(
            {
                "topology_id": topology_id,
                "sample_size": sample_size,
                "two_stage_candidate_ids": two_ids,
                "spoplus_candidate_ids": spo_ids,
                "case_direction": direction,
                "count": len(rows),
                "rate_total": len(rows) / total if total else 0.0,
                "mean_delta": mean(deltas),
                "mean_abs_delta": mean(abs_deltas),
                "max_abs_delta": max(abs_deltas) if abs_deltas else 0.0,
                "mean_gap_2stage": mean([common.parse_float(row["gap_2stage"]) for row in rows]),
                "mean_gap_spoplus": mean([common.parse_float(row["gap_spoplus"]) for row in rows]),
                "dominant_oracle_candidate_ids": dominant_oracle,
                "dominant_oracle_count": dominant_count,
                "example_data_seed": common.parse_int(example["data_seed"]),
                "example_test_sample_index": common.parse_int(example["test_sample_index"]),
                "example_graph": example["graph"],
                "example_abs_delta": common.parse_float(example["abs_true_delta"]),
            }
        )

    return sorted(
        output,
        key=lambda row: (
            str(row["topology_id"]),
            common.parse_int(row["sample_size"]),
            -common.parse_int(row["count"]),
            -common.parse_float(row["mean_abs_delta"]),
            str(row["two_stage_candidate_ids"]),
            str(row["spoplus_candidate_ids"]),
        ),
    )


def summarize_reversal_detail(
    decision_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    total_contexts = count_total_contexts(decision_rows)
    return summarize_cases(total_contexts, target_rows), summarize_switches(total_contexts, target_rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-rows", type=Path, default=DEFAULT_DECISION_ROWS)
    parser.add_argument("--target-rows", type=Path, default=DEFAULT_TARGET_ROWS)
    parser.add_argument("--case-output", type=Path, default=DEFAULT_CASE_OUTPUT)
    parser.add_argument("--switch-output", type=Path, default=DEFAULT_SWITCH_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    decision_rows = common.read_csv_rows(args.decision_rows)
    target_rows = common.read_csv_rows(args.target_rows)
    case_rows, switch_rows = summarize_reversal_detail(decision_rows, target_rows)
    common.write_csv(args.case_output, case_rows, CASE_SUMMARY_FIELDS)
    common.write_csv(args.switch_output, switch_rows, SWITCH_SUMMARY_FIELDS)
    print(f"Saved case summary rows: {len(case_rows)} to {args.case_output}")
    print(f"Saved switch summary rows: {len(switch_rows)} to {args.switch_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
