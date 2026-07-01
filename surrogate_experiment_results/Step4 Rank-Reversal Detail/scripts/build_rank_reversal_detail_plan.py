#!/usr/bin/env python3
"""Build target contexts for Step4 rank-reversal detail analysis."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DETAIL_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = DETAIL_DIR.parents[1]
STRUCTURAL_SCRIPTS = PROJECT_ROOT / "surrogate_experiment_results" / "Step4 Topology Structural Atlas" / "scripts"
if str(STRUCTURAL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(STRUCTURAL_SCRIPTS))

import step4_topology_common as common  # noqa: E402


DEFAULT_DECISION_ROWS = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "Step4 Decision Overlay"
    / "results"
    / "decision_solution_rows.csv"
)
DEFAULT_OUTPUT = DETAIL_DIR / "results" / "rank_reversal_target_contexts.csv"

TARGET_FIELDS = [
    "topology_id",
    "data_seed",
    "sample_size",
    "test_sample_index",
    "graph",
    "oracle_candidate_ids",
    "two_stage_candidate_ids",
    "spoplus_candidate_ids",
    "two_stage_only_candidate_ids",
    "spoplus_only_candidate_ids",
    "true_obj_oracle",
    "true_obj_2stage",
    "true_obj_spoplus",
    "gap_2stage",
    "gap_spoplus",
    "true_delta_spoplus_minus_2stage",
    "case_direction",
    "abs_true_delta",
]


def context_key(row: dict[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(row["topology_id"]),
        common.parse_int(row["data_seed"]),
        common.parse_int(row["sample_size"]),
        common.parse_int(row["test_sample_index"]),
    )


def id_set(text: Any) -> set[str]:
    return common.pipe_text_set(text)


def id_text(values: set[str]) -> str:
    return common.pipe_join(sorted(values))


def case_direction(delta: float) -> str:
    if delta > 1e-9:
        return "beneficial_reversal"
    if delta < -1e-9:
        return "harmful_reversal"
    return "decision_different_true_tie"


def build_rank_reversal_targets(
    decision_rows: list[dict[str, Any]],
    *,
    targets_per_topology: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in decision_rows:
        grouped[context_key(row)][str(row["solution_source"])] = row

    candidates: list[dict[str, Any]] = []
    for key in sorted(grouped):
        roles = grouped[key]
        if not {"oracle", "2stage", "spoplus"} <= set(roles):
            continue
        two_ids = id_set(roles["2stage"].get("selected_candidate_ids", ""))
        spo_ids = id_set(roles["spoplus"].get("selected_candidate_ids", ""))
        if two_ids == spo_ids:
            continue
        oracle_ids = id_set(roles["oracle"].get("selected_candidate_ids", ""))
        two_true = common.parse_float(roles["2stage"].get("true_obj"))
        spo_true = common.parse_float(roles["spoplus"].get("true_obj"))
        delta = float(spo_true - two_true)
        topology_id, data_seed, sample_size, sample_index = key
        candidates.append(
            {
                "topology_id": topology_id,
                "data_seed": data_seed,
                "sample_size": sample_size,
                "test_sample_index": sample_index,
                "graph": roles["2stage"].get("graph", f"G-{sample_index:06d}.json"),
                "oracle_candidate_ids": id_text(oracle_ids),
                "two_stage_candidate_ids": id_text(two_ids),
                "spoplus_candidate_ids": id_text(spo_ids),
                "two_stage_only_candidate_ids": id_text(two_ids - spo_ids),
                "spoplus_only_candidate_ids": id_text(spo_ids - two_ids),
                "true_obj_oracle": common.parse_float(roles["oracle"].get("true_obj")),
                "true_obj_2stage": two_true,
                "true_obj_spoplus": spo_true,
                "gap_2stage": common.parse_float(roles["2stage"].get("gap_to_oracle")),
                "gap_spoplus": common.parse_float(roles["spoplus"].get("gap_to_oracle")),
                "true_delta_spoplus_minus_2stage": delta,
                "case_direction": case_direction(delta),
                "abs_true_delta": abs(delta),
            }
        )

    by_topology: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_topology[str(row["topology_id"])].append(row)

    output: list[dict[str, Any]] = []
    for topology_id in sorted(by_topology):
        group = sorted(
            by_topology[topology_id],
            key=lambda row: (
                -float(row["abs_true_delta"]),
                common.parse_int(row["sample_size"]),
                common.parse_int(row["data_seed"]),
                common.parse_int(row["test_sample_index"]),
            ),
        )
        output.extend(group[: int(targets_per_topology)])
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-rows", type=Path, default=DEFAULT_DECISION_ROWS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--targets-per-topology", type=int, default=20)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = common.read_csv_rows(args.decision_rows)
    targets = build_rank_reversal_targets(rows, targets_per_topology=args.targets_per_topology)
    common.write_csv(args.output, targets, TARGET_FIELDS)
    print(f"Saved rank-reversal target contexts: {len(targets)} to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
