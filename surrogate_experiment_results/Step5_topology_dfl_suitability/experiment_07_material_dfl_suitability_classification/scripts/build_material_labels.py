#!/usr/bin/env python3
"""Build and audit the primary and confidence-aware Experiment 07 labels."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import material_common as common


LABEL_FIELDS = [
    "topology_id",
    "topology_hash",
    "feasible_set_hash",
    "test_hash",
    *common.SEED_FIELDS,
    "formal_label_mean_pp",
    "label_uncertainty_std_pp",
    "primary_label",
    "primary_label_index",
    "target_is_material_helpful",
    "target_is_material_harmful",
    "material_direction_seed_agreement_count",
    "sign_agreement_passed",
    "sign_agreement_label",
    "bootstrap_ci_low_pp",
    "bootstrap_ci_high_pp",
    "bootstrap_ci_excludes_zero",
    "bootstrap_ci_label",
    "high_variance",
    "confidence_passed",
    "confidence_label",
    "confidence_label_index",
    "confidence_state",
    "uncertainty_reasons",
    "material_threshold_pp",
    "high_variance_std_threshold_pp",
    "bootstrap_alpha",
]


def build(
    source_rows: list[dict[str, str]],
    *,
    material_threshold_pp: float,
    high_variance_std_pp: float,
    bootstrap_alpha: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    failures: list[str] = []
    topology_ids = [row.get("topology_id", "") for row in source_rows]
    if len(source_rows) != 1000:
        failures.append(f"source_count_mismatch:{len(source_rows)}!=1000")
    if len(topology_ids) != len(set(topology_ids)):
        failures.append("source_topology_ids_not_unique")
    output: list[dict[str, Any]] = []
    for row in source_rows:
        topology_id = row.get("topology_id", "missing")
        if str(row.get("formal_label_ready", "")).lower() != "true":
            failures.append(f"formal_label_not_ready:{topology_id}")
            continue
        if row.get("label_seed_count") != "3":
            failures.append(f"seed_count_mismatch:{topology_id}")
            continue
        try:
            output.append(
                common.derive_label_row(
                    row,
                    material_threshold_pp=material_threshold_pp,
                    high_variance_std_pp=high_variance_std_pp,
                    bootstrap_alpha=bootstrap_alpha,
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(str(exc))
    output.sort(key=lambda row: common.topology_sort_key(str(row["topology_id"])))
    if len(output) != len(source_rows):
        failures.append(f"output_count_mismatch:{len(output)}!={len(source_rows)}")
    return output, failures


def audit(
    rows: list[dict[str, Any]],
    failures: list[str],
    *,
    source_path: Path,
    source_sha256: str,
    material_threshold_pp: float,
    high_variance_std_pp: float,
    bootstrap_alpha: float,
) -> dict[str, Any]:
    topology_ids = [str(row["topology_id"]) for row in rows]
    hashes = [str(row["topology_hash"]) for row in rows]
    feasible_hashes = [str(row["feasible_set_hash"]) for row in rows]
    confidence_states = Counter(str(row["confidence_state"]) for row in rows)
    reasons = Counter(
        reason
        for row in rows
        for reason in str(row["uncertainty_reasons"]).split(";")
        if reason
    )
    if len(topology_ids) != len(set(topology_ids)):
        failures.append("output_topology_ids_not_unique")
    if len(hashes) != len(set(hashes)):
        failures.append("topology_hashes_not_unique")
    if len(feasible_hashes) != len(set(feasible_hashes)):
        failures.append("feasible_set_hashes_not_unique")
    if source_sha256 != common.LOCKED_TARGET_SHA256:
        failures.append(f"source_sha256_mismatch:{source_sha256}")
    return {
        "passed": not failures,
        "status": "success" if not failures else "failed",
        "source_path": str(source_path),
        "source_sha256": source_sha256,
        "expected_source_sha256": common.LOCKED_TARGET_SHA256,
        "topology_count": len(rows),
        "material_threshold_pp": material_threshold_pp,
        "threshold_comparison": "strict: helpful if mean > threshold; harmful if mean < -threshold",
        "high_variance_std_threshold_pp": high_variance_std_pp,
        "bootstrap": {
            "method": "exact ordered n=3 nonparametric percentile bootstrap",
            "resample_count": 27,
            "alpha": bootstrap_alpha,
        },
        "primary_label_counts": common.label_counts(rows, "primary_label"),
        "sign_agreement_label_counts": common.label_counts(rows, "sign_agreement_label"),
        "bootstrap_ci_label_counts": common.label_counts(rows, "bootstrap_ci_label"),
        "confidence_label_counts": common.label_counts(rows, "confidence_label"),
        "confidence_state_counts": dict(sorted(confidence_states.items())),
        "uncertainty_reason_counts": dict(sorted(reasons.items())),
        "target_or_uncertainty_used_as_input_feature": False,
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-table", type=Path, default=common.DEFAULT_TARGET_TABLE)
    parser.add_argument("--material-threshold-pp", type=float, default=0.5)
    parser.add_argument("--high-variance-std-pp", type=float, default=0.5)
    parser.add_argument("--bootstrap-alpha", type=float, default=0.05)
    parser.add_argument(
        "--output",
        type=Path,
        default=common.DEFAULT_OUTPUT_ROOT / "labels" / "material_labels.csv",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=common.DEFAULT_OUTPUT_ROOT / "labels" / "material_labels.audit.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_sha256 = common.sha256_file(args.target_table)
    rows, failures = build(
        common.read_csv(args.target_table),
        material_threshold_pp=args.material_threshold_pp,
        high_variance_std_pp=args.high_variance_std_pp,
        bootstrap_alpha=args.bootstrap_alpha,
    )
    result = audit(
        rows,
        failures,
        source_path=args.target_table,
        source_sha256=source_sha256,
        material_threshold_pp=args.material_threshold_pp,
        high_variance_std_pp=args.high_variance_std_pp,
        bootstrap_alpha=args.bootstrap_alpha,
    )
    if rows:
        common.atomic_write_csv(args.output, rows, LABEL_FIELDS)
    common.atomic_write_json(args.audit_output, result)
    print(
        json.dumps(
            {
                **result,
                "label_output": str(args.output),
                "audit_output": str(args.audit_output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
