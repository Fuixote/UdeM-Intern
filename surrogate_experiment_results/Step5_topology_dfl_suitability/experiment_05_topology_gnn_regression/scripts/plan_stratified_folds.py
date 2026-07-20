#!/usr/bin/env python3
"""Plan deterministic five-fold topology splits stratified by target regime."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import tempfile
from typing import Any

import gnn_data_common as common


def validate_targets(
    rows: list[dict[str, str]],
    *,
    target_field: str,
    require_formal_targets: bool,
) -> None:
    topology_ids = [row.get("topology_id", "") for row in rows]
    failures: list[str] = []
    if len(rows) != 1000:
        failures.append(f"target_count_mismatch:{len(rows)}!=1000")
    if len(topology_ids) != len(set(topology_ids)):
        failures.append("target_topology_ids_not_unique")
    for row in rows:
        topology_id = row.get("topology_id", "missing")
        try:
            float(row.get(target_field, ""))
        except (TypeError, ValueError):
            failures.append(f"target_unavailable:{topology_id}:{target_field}")
        if require_formal_targets and not common.truthy(row.get("formal_label_ready")):
            failures.append(f"formal_target_not_ready:{topology_id}")
    if failures:
        raise ValueError(";".join(failures[:20]))


def target_strata(
    rows: list[dict[str, str]],
    *,
    target_field: str = "normalized_improvement_pp",
) -> dict[str, str]:
    values = {row["topology_id"]: float(row[target_field]) for row in rows}
    positive_material = sorted((value, topology_id) for topology_id, value in values.items() if value > 0.1)
    negative_material = sorted((value, topology_id) for topology_id, value in values.items() if value < -0.1)
    positive_extreme_count = min(len(positive_material), max(5, round(0.1 * len(positive_material))))
    negative_extreme_count = min(len(negative_material), max(5, round(0.1 * len(negative_material))))
    extreme_positive = {topology_id for _, topology_id in positive_material[-positive_extreme_count:]}
    extreme_negative = {topology_id for _, topology_id in negative_material[:negative_extreme_count]}
    output = {}
    for topology_id, value in values.items():
        if topology_id in extreme_positive:
            output[topology_id] = "extreme_positive"
        elif topology_id in extreme_negative:
            output[topology_id] = "extreme_negative"
        elif value > 0.1:
            output[topology_id] = "material_positive"
        elif value < -0.1:
            output[topology_id] = "material_negative"
        elif value > 0:
            output[topology_id] = "small_positive"
        elif value < 0:
            output[topology_id] = "small_negative"
        else:
            output[topology_id] = "exact_zero"
    return output


def assign_folds(
    rows: list[dict[str, str]],
    *,
    folds: int = 5,
    seed: int = 20260720,
    target_field: str = "normalized_improvement_pp",
) -> list[dict[str, Any]]:
    if folds < 2:
        raise ValueError("folds must be at least two")
    strata = target_strata(rows, target_field=target_field)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[strata[row["topology_id"]]].append(row)
    fold_sizes = [0] * folds
    assignments = []
    for stratum in sorted(grouped):
        ordered = sorted(grouped[stratum], key=lambda row: common.stable_key(seed, stratum, row["topology_id"]))
        start_fold = min(range(folds), key=lambda fold: (fold_sizes[fold], fold))
        for index, row in enumerate(ordered):
            fold = (start_fold + index) % folds
            fold_sizes[fold] += 1
            assignments.append({
                "topology_id": row["topology_id"],
                "topology_hash": row["topology_hash"],
                "feasible_set_hash": row["feasible_set_hash"],
                "fold": fold,
                "target_stratum": stratum,
                "target_name": target_field,
                "target_value": float(row[target_field]),
                "normalized_improvement_pp": float(row[target_field]),
                "split_seed": seed,
            })
    assignments.sort(key=lambda row: row["topology_id"])
    return assignments


def audit(
    assignments: list[dict[str, Any]],
    folds: int,
    *,
    target_field: str = "normalized_improvement_pp",
    require_formal_targets: bool = False,
) -> dict[str, Any]:
    failures = []
    topology_ids = [row["topology_id"] for row in assignments]
    fold_counts = Counter(int(row["fold"]) for row in assignments)
    per_stratum = defaultdict(Counter)
    for row in assignments:
        per_stratum[row["target_stratum"]][int(row["fold"])] += 1
    if len(assignments) != 1000:
        failures.append(f"assignment_count_mismatch:{len(assignments)}!=1000")
    if len(topology_ids) != len(set(topology_ids)):
        failures.append("topology_assignments_not_unique")
    if {row.get("target_name") for row in assignments} != {target_field}:
        failures.append("target_name_mismatch")
    if set(fold_counts) != set(range(folds)):
        failures.append("fold_ids_incomplete")
    if max(fold_counts.values(), default=0) - min(fold_counts.values(), default=0) > 1:
        failures.append("fold_sizes_not_balanced")
    for stratum, counts in per_stratum.items():
        filled = [counts[fold] for fold in range(folds)]
        if max(filled) - min(filled) > 1:
            failures.append(f"stratum_not_balanced:{stratum}")
    return {
        "passed": not failures,
        "assignment_count": len(assignments),
        "fold_count": folds,
        "fold_sizes": {str(fold): fold_counts[fold] for fold in range(folds)},
        "stratum_fold_counts": {stratum: {str(fold): counts[fold] for fold in range(folds)} for stratum, counts in sorted(per_stratum.items())},
        "target_values_used_only_for_split_stratification": True,
        "target_field": target_field,
        "formal_targets_required": require_formal_targets,
        "failures": failures,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-summary", type=Path, default=common.DEFAULT_FORMAL_SUMMARY)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--target-field", default="normalized_improvement_pp")
    parser.add_argument("--require-formal-targets", action="store_true")
    parser.add_argument("--output", type=Path, default=common.DEFAULT_OUTPUT_ROOT / "splits" / "folds.csv")
    parser.add_argument("--audit-output", type=Path, default=common.DEFAULT_OUTPUT_ROOT / "splits" / "folds.audit.json")
    args = parser.parse_args()
    rows = common.read_csv(args.formal_summary)
    validate_targets(
        rows,
        target_field=args.target_field,
        require_formal_targets=args.require_formal_targets,
    )
    assignments = assign_folds(
        rows,
        folds=args.folds,
        seed=args.seed,
        target_field=args.target_field,
    )
    result = audit(
        assignments,
        args.folds,
        target_field=args.target_field,
        require_formal_targets=args.require_formal_targets,
    )
    write_csv(args.output, assignments)
    args.audit_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**result, "output": str(args.output), "audit_output": str(args.audit_output)}, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
