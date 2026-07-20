#!/usr/bin/env python3
"""Select a deterministic, label-stratified 60-topology seed-audit panel."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import tempfile
from typing import Any

import repeat_seed_common as common


CATEGORY_COUNTS = {
    "ceiling_zero": 10,
    "nonzero_gap_plateau": 10,
    "small_nonzero": 10,
    "material_positive": 10,
    "material_negative": 10,
    "extreme_positive": 5,
    "extreme_negative": 5,
}


def category_pools(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    pools = {name: [] for name in CATEGORY_COUNTS}
    positives: list[dict[str, str]] = []
    negatives: list[dict[str, str]] = []
    for row in rows:
        value = float(row["normalized_improvement_pp"])
        gap_2stage = float(row["test_normalized_gap_2stage"])
        gap_spoplus = float(row["test_normalized_gap_spoplus"])
        if value == 0.0 and gap_2stage == 0.0 and gap_spoplus == 0.0:
            pools["ceiling_zero"].append(row)
        elif value == 0.0:
            pools["nonzero_gap_plateau"].append(row)
        elif abs(value) <= 0.1:
            pools["small_nonzero"].append(row)
        elif value > 0.1:
            positives.append(row)
        else:
            negatives.append(row)

    positives.sort(key=lambda row: (float(row["normalized_improvement_pp"]), row["topology_id"]))
    negatives.sort(key=lambda row: (float(row["normalized_improvement_pp"]), row["topology_id"]))
    pools["extreme_positive"] = positives[-CATEGORY_COUNTS["extreme_positive"] :]
    pools["material_positive"] = positives[: -CATEGORY_COUNTS["extreme_positive"]]
    pools["extreme_negative"] = negatives[: CATEGORY_COUNTS["extreme_negative"]]
    pools["material_negative"] = negatives[CATEGORY_COUNTS["extreme_negative"] :]
    return pools


def complexity_key(row: dict[str, str]) -> tuple[float, float, float, str]:
    return (
        float(row.get("num_feasible_candidates") or 0),
        float(row.get("candidate_conflict_edges") or 0),
        float(row.get("num_arcs") or 0),
        str(row["topology_id"]),
    )


def evenly_spaced(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    ordered = sorted(rows, key=complexity_key)
    if len(ordered) < count:
        raise ValueError(f"pool has {len(ordered)} rows, needs {count}")
    if count == 1:
        return [ordered[len(ordered) // 2]]
    indices = [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)]
    if len(set(indices)) != count:
        raise AssertionError("evenly spaced selection produced duplicate positions")
    return [ordered[index] for index in indices]


def select_panel(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    pools = category_pools(rows)
    selected: list[dict[str, Any]] = []
    for category, count in CATEGORY_COUNTS.items():
        chosen = evenly_spaced(pools[category], count)
        for within_index, row in enumerate(chosen):
            selected.append(
                {
                    **row,
                    "selection_category": category,
                    "selection_category_index": within_index,
                    "seed42_normalized_improvement_pp": row["normalized_improvement_pp"],
                    "seed42_test_normalized_gap_2stage": row["test_normalized_gap_2stage"],
                    "seed42_test_normalized_gap_spoplus": row["test_normalized_gap_spoplus"],
                    "reference_test_seed": common.REFERENCE_SEED,
                }
            )
    topology_ids = [row["topology_id"] for row in selected]
    if len(selected) != 60 or len(set(topology_ids)) != 60:
        raise AssertionError("selection must contain exactly 60 unique topologies")
    for index, row in enumerate(selected):
        row["repeat_manifest_index"] = index
    return selected


def atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-summary", type=Path, default=common.DEFAULT_FORMAL_SUMMARY)
    parser.add_argument("--output", type=Path, default=common.DEFAULT_SELECTED_TOPOLOGIES)
    parser.add_argument("--audit-output", type=Path, default=common.EXPERIMENT_ROOT / "configs" / "topology_selection_audit.json")
    args = parser.parse_args()
    source_rows = common.read_csv(args.formal_summary)
    selected = select_panel(source_rows)
    atomic_write_csv(args.output, selected)
    counts = {category: sum(row["selection_category"] == category for row in selected) for category in CATEGORY_COUNTS}
    audit = {
        "passed": counts == CATEGORY_COUNTS and len(selected) == 60,
        "formal_summary": common.project_relative(args.formal_summary),
        "source_topology_count": len(source_rows),
        "selected_topology_count": len(selected),
        "category_counts": counts,
        "selection_rule": "category pools followed by deterministic evenly spaced complexity quantiles",
        "topology_ids": [row["topology_id"] for row in selected],
    }
    args.audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
