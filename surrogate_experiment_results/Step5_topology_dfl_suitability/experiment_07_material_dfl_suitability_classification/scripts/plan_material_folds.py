#!/usr/bin/env python3
"""Create deterministic five-fold assignments stratified by material class."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

import material_common as common


def assign_folds(
    rows: list[dict[str, str]],
    *,
    label_field: str = "primary_label",
    folds: int = 5,
    seed: int = 20260722,
) -> list[dict[str, Any]]:
    if folds < 2:
        raise ValueError("fold count must be at least two")
    topology_ids = [row["topology_id"] for row in rows]
    if len(topology_ids) != len(set(topology_ids)):
        raise ValueError("label topology ids are not unique")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        label = row[label_field]
        if label not in common.CLASS_LABELS:
            raise ValueError(f"unknown label:{row['topology_id']}:{label}")
        grouped[label].append(row)
    fold_sizes = [0] * folds
    output: list[dict[str, Any]] = []
    for label in common.CLASS_LABELS:
        ordered = sorted(
            grouped[label],
            key=lambda row: common.stable_key(seed, label, row["topology_id"]),
        )
        start_fold = min(range(folds), key=lambda fold: (fold_sizes[fold], fold))
        for index, row in enumerate(ordered):
            fold = (start_fold + index) % folds
            fold_sizes[fold] += 1
            output.append(
                {
                    "topology_id": row["topology_id"],
                    "topology_hash": row["topology_hash"],
                    "feasible_set_hash": row["feasible_set_hash"],
                    "fold": fold,
                    "stratification_label_field": label_field,
                    "stratification_label": label,
                    "formal_label_mean_pp": float(row["formal_label_mean_pp"]),
                    "split_seed": seed,
                }
            )
    output.sort(key=lambda row: common.topology_sort_key(str(row["topology_id"])))
    return output


def audit(
    assignments: list[dict[str, Any]],
    *,
    folds: int,
    label_field: str,
) -> dict[str, Any]:
    failures: list[str] = []
    topology_ids = [str(row["topology_id"]) for row in assignments]
    fold_counts = Counter(int(row["fold"]) for row in assignments)
    label_fold_counts: dict[str, Counter[int]] = defaultdict(Counter)
    for row in assignments:
        label_fold_counts[str(row["stratification_label"])][int(row["fold"])] += 1
    if len(assignments) != 1000:
        failures.append(f"assignment_count_mismatch:{len(assignments)}!=1000")
    if len(topology_ids) != len(set(topology_ids)):
        failures.append("assignment_topology_ids_not_unique")
    if set(fold_counts) != set(range(folds)):
        failures.append("fold_ids_incomplete")
    if max(fold_counts.values(), default=0) - min(fold_counts.values(), default=0) > 1:
        failures.append("fold_sizes_not_balanced")
    for label in common.CLASS_LABELS:
        counts = [label_fold_counts[label][fold] for fold in range(folds)]
        if max(counts) - min(counts) > 1:
            failures.append(f"label_not_balanced:{label}")
    return {
        "passed": not failures,
        "status": "success" if not failures else "failed",
        "assignment_count": len(assignments),
        "fold_count": folds,
        "fold_sizes": {str(fold): fold_counts[fold] for fold in range(folds)},
        "stratification_label_field": label_field,
        "label_fold_counts": {
            label: {
                str(fold): label_fold_counts[label][fold]
                for fold in range(folds)
            }
            for label in common.CLASS_LABELS
        },
        "target_used_only_for_split_stratification": True,
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        type=Path,
        default=common.DEFAULT_OUTPUT_ROOT / "labels" / "material_labels.csv",
    )
    parser.add_argument("--label-field", default="primary_label")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument(
        "--output",
        type=Path,
        default=common.DEFAULT_OUTPUT_ROOT / "splits" / "material_folds.csv",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=common.DEFAULT_OUTPUT_ROOT / "splits" / "material_folds.audit.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    assignments = assign_folds(
        common.read_csv(args.labels),
        label_field=args.label_field,
        folds=args.folds,
        seed=args.seed,
    )
    result = audit(assignments, folds=args.folds, label_field=args.label_field)
    common.atomic_write_csv(args.output, assignments)
    common.atomic_write_json(args.audit_output, result)
    print(
        json.dumps(
            {
                **result,
                "fold_output": str(args.output),
                "audit_output": str(args.audit_output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
