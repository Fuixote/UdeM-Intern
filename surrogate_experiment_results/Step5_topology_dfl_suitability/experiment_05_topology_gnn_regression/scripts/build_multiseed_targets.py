#!/usr/bin/env python3
"""Build strict three-seed mean targets and label-uncertainty metadata."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import tempfile
from typing import Any

import gnn_data_common as common


EXP4_RESULTS = (
    common.STEP5_ROOT
    / "experiment_04_repeat_seed_stability_sample50"
    / "results"
    / "repeat_seed120"
    / "results"
)
DEFAULT_REPEAT_LABELS = EXP4_RESULTS / "repeat_seed_labels_long.csv"
REQUIRED_SEEDS = (42, 43, 44)


def aggregate_targets(
    formal_rows: list[dict[str, str]],
    repeat_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, Any]]:
    repeat_by_topology: dict[str, dict[int, dict[str, str]]] = {}
    failures: list[str] = []
    for row in repeat_rows:
        topology_id = row["topology_id"]
        seed = int(row["train_seed"])
        seed_rows = repeat_by_topology.setdefault(topology_id, {})
        if seed in seed_rows:
            failures.append(f"duplicate_repeat_label:{topology_id}@{seed}")
        seed_rows[seed] = row

    target_rows: list[dict[str, Any]] = []
    missing_topology_rows: list[dict[str, str]] = []
    for formal in formal_rows:
        topology_id = formal["topology_id"]
        seed_rows = dict(repeat_by_topology.get(topology_id, {}))
        formal_seed42 = float(formal["normalized_improvement_pp"])
        if 42 in seed_rows:
            repeat_seed42 = float(seed_rows[42]["normalized_improvement_pp"])
            if repeat_seed42 != formal_seed42:
                failures.append(f"seed42_value_mismatch:{topology_id}")
            if seed_rows[42].get("test_hash", "") != formal.get("test_hash", ""):
                failures.append(f"seed42_test_hash_mismatch:{topology_id}")
        values = {42: formal_seed42}
        for seed in (43, 44):
            if seed in seed_rows:
                if seed_rows[seed].get("test_hash", "") != formal.get("test_hash", ""):
                    failures.append(f"test_hash_mismatch:{topology_id}@{seed}")
                values[seed] = float(seed_rows[seed]["normalized_improvement_pp"])

        missing_seeds = [seed for seed in REQUIRED_SEEDS if seed not in values]
        complete = not missing_seeds
        ordered_values = [values[seed] for seed in REQUIRED_SEEDS if seed in values]
        target_rows.append(
            {
                "topology_id": topology_id,
                "topology_hash": formal["topology_hash"],
                "feasible_set_hash": formal["feasible_set_hash"],
                "test_hash": formal["test_hash"],
                "seed42_normalized_improvement_pp": values[42],
                "seed43_normalized_improvement_pp": values.get(43, ""),
                "seed44_normalized_improvement_pp": values.get(44, ""),
                "label_seed_count": len(values),
                "required_label_seeds": "42;43;44",
                "missing_label_seeds": ";".join(str(seed) for seed in missing_seeds),
                "available_seed_mean_pp": statistics.fmean(ordered_values),
                "formal_label_mean_pp": statistics.fmean(ordered_values) if complete else "",
                "label_uncertainty_std_pp": statistics.pstdev(ordered_values) if complete else "",
                "uncertainty_ddof": 0 if complete else "",
                "formal_label_ready": complete,
            }
        )
        if not complete:
            missing_topology_rows.append(formal)

    known_ids = {row["topology_id"] for row in formal_rows}
    unexpected = sorted(set(repeat_by_topology) - known_ids)
    if unexpected:
        failures.append(f"unexpected_repeat_topologies:{','.join(unexpected)}")
    complete_count = sum(bool(row["formal_label_ready"]) for row in target_rows)
    audit = {
        "passed": not failures,
        "formal_ready": not failures and complete_count == len(formal_rows),
        "target_policy": "mean_of_normalized_improvement_pp_over_train_seeds_42_43_44",
        "uncertainty_policy": "population_standard_deviation_over_train_seeds_42_43_44_ddof0",
        "required_seeds": list(REQUIRED_SEEDS),
        "topology_count": len(formal_rows),
        "complete_three_seed_topology_count": complete_count,
        "incomplete_topology_count": len(formal_rows) - complete_count,
        "missing_seed_job_count": sum(
            len(str(row["missing_label_seeds"]).split(";"))
            for row in target_rows
            if row["missing_label_seeds"]
        ),
        "failures": failures,
    }
    return target_rows, missing_topology_rows, audit


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and not fields:
        raise ValueError("cannot infer fields for an empty CSV")
    fieldnames = fields or list(rows[0])
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-summary", type=Path, default=common.DEFAULT_FORMAL_SUMMARY)
    parser.add_argument("--repeat-labels", type=Path, default=DEFAULT_REPEAT_LABELS)
    parser.add_argument("--output", type=Path, default=common.DEFAULT_OUTPUT_ROOT / "targets" / "multiseed_targets.csv")
    parser.add_argument("--audit-output", type=Path, default=common.DEFAULT_OUTPUT_ROOT / "targets" / "multiseed_targets.audit.json")
    parser.add_argument(
        "--missing-topologies-output",
        type=Path,
        default=common.EXPERIMENT_ROOT / "configs" / "multiseed_label_completion940.csv",
    )
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    formal_rows = common.read_csv(args.formal_summary)
    repeat_rows = common.read_csv(args.repeat_labels)
    targets, missing_rows, audit = aggregate_targets(formal_rows, repeat_rows)
    write_csv(args.output, targets)
    formal_fields = list(formal_rows[0]) if formal_rows else []
    write_csv(args.missing_topologies_output, missing_rows, formal_fields)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        **audit,
        "output": str(args.output),
        "audit_output": str(args.audit_output),
        "missing_topologies_output": str(args.missing_topologies_output),
    }, indent=2, sort_keys=True))
    if not audit["passed"]:
        return 1
    if args.require_complete and not audit["formal_ready"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
