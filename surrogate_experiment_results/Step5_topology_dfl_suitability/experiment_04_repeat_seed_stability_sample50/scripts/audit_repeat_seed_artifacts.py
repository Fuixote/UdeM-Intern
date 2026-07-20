#!/usr/bin/env python3
"""Audit repeat-seed fit bundles and prove they share the formal fixed test banks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import repeat_seed_common as common


builder = common.base_builder()


def append_once(failures: list[str], message: str) -> None:
    if message not in failures:
        failures.append(message)


def audit_bundle(row: dict[str, str], seed: int, output_root: Path, formal_output_root: Path) -> dict[str, Any]:
    topology_id = row["topology_id"]
    paths = builder.artifact_paths(output_root, regime=common.DEFAULT_REGIME, topology_id=topology_id, data_seed=seed, sample_size=common.SAMPLE_SIZE)
    fixed_test_path, fixed_manifest_path = common.reference_test_paths(topology_id, formal_output_root=formal_output_root)
    failures: list[str] = []
    required = [paths["train_bank"], paths["validation"], paths["eval_manifest"], paths["fit_manifest"], paths["split_manifest"], fixed_test_path, fixed_manifest_path]
    for path in required:
        if not path.is_file():
            append_once(failures, f"missing:{path}")
    if failures:
        return {"topology_id": topology_id, "train_seed": seed, "passed": False, "failures": failures}

    train = builder.common.read_npz_dataset(paths["train_bank"])
    validation = builder.common.read_npz_dataset(paths["validation"])
    test = builder.common.read_npz_dataset(fixed_test_path)
    eval_manifest = common.read_json(paths["eval_manifest"])
    fit_manifest = common.read_json(paths["fit_manifest"])
    split_manifest = common.read_json(paths["split_manifest"])
    test_manifest = common.read_json(fixed_manifest_path)

    train_rows = list(train["manifest"].get("samples", []))
    validation_rows = list(validation["manifest"].get("samples", []))
    test_rows = list(test["manifest"].get("samples", []))
    if len(train_rows) != common.TRAINING_SIZE:
        append_once(failures, "train_size_mismatch")
    if len(validation_rows) != common.VALIDATION_SIZE:
        append_once(failures, "validation_size_mismatch")
    if len(test_rows) != common.TEST_SIZE:
        append_once(failures, "test_size_mismatch")
    for label, manifest in (("train", train["manifest"]), ("validation", validation["manifest"]), ("fit", fit_manifest), ("eval", eval_manifest)):
        if int(manifest.get("train_seed", manifest.get("data_seed", -1))) != seed:
            append_once(failures, f"{label}_train_seed_mismatch")
    if {sample.get("train_seed") for sample in test_rows} - {None}:
        append_once(failures, "test_varies_by_train_seed")

    train_hash = builder.common.sample_manifest_hashes(train_rows)
    validation_hash = builder.common.sample_manifest_hashes(validation_rows)
    test_hash = builder.common.sample_manifest_hashes(test_rows)
    if train["manifest"].get("bank_hash") != train_hash:
        append_once(failures, "train_hash_mismatch")
    if validation["manifest"].get("dataset_hash") != validation_hash:
        append_once(failures, "validation_hash_mismatch")
    if test["manifest"].get("dataset_hash") != test_hash:
        append_once(failures, "test_dataset_hash_mismatch")
    expected_hash = str(row.get("test_hash", ""))
    for label, observed in (
        ("selection", expected_hash),
        ("test_manifest", str(test_manifest.get("test_hash", ""))),
        ("eval", str(eval_manifest.get("test_hash", ""))),
        ("protocol", str(eval_manifest.get("repeat_seed_protocol", {}).get("reference_test_hash", ""))),
        ("npz", test_hash),
    ):
        if observed != expected_hash:
            append_once(failures, f"{label}_fixed_test_hash_mismatch")
    expected_path = common.project_relative(fixed_test_path)
    if str(eval_manifest.get("test_path")) != expected_path:
        append_once(failures, "eval_fixed_test_path_mismatch")
    protocol = eval_manifest.get("repeat_seed_protocol", {})
    if protocol.get("fixed_test_bank") is not True or protocol.get("test_bank_varies_with_train_seed") is not False:
        append_once(failures, "repeat_seed_protocol_mismatch")

    split = split_manifest.get("sample_size_splits", {}).get(str(common.SAMPLE_SIZE), {})
    training_indices = {int(value) for value in split.get("training_indices", [])}
    validation_indices = {int(value) for value in split.get("validation_indices", [])}
    if training_indices & validation_indices:
        append_once(failures, "training_validation_overlap")
    if training_indices | validation_indices != set(range(common.SAMPLE_SIZE)):
        append_once(failures, "training_validation_union_mismatch")
    return {
        "topology_id": topology_id,
        "train_seed": seed,
        "passed": not failures,
        "train_hash": train_hash,
        "validation_hash": validation_hash,
        "test_hash": test_hash,
        "fixed_test_path": expected_path,
        "failures": failures,
    }


def audit_all(rows: list[dict[str, str]], output_root: Path, formal_output_root: Path) -> dict[str, Any]:
    bundles = [audit_bundle(row, seed, output_root, formal_output_root) for row in rows for seed in common.TRAIN_SEEDS]
    failures = [f"{row['topology_id']}@{row['train_seed']}:{','.join(row['failures'])}" for row in bundles if not row["passed"]]
    for row in rows:
        topology_id = row["topology_id"]
        selected = [bundle for bundle in bundles if bundle["topology_id"] == topology_id and bundle["passed"]]
        if len(selected) == len(common.TRAIN_SEEDS):
            if len({bundle["test_hash"] for bundle in selected}) != 1:
                failures.append(f"{topology_id}:test_hash_varies_across_train_seeds")
            if len({bundle["train_hash"] for bundle in selected}) != len(common.TRAIN_SEEDS):
                failures.append(f"{topology_id}:training_hash_did_not_change_across_train_seeds")
    return {
        "passed": not failures,
        "expected_topology_count": len(rows),
        "observed_topology_count": len(rows),
        "train_seeds": list(common.TRAIN_SEEDS),
        "expected_bundle_count": len(rows) * len(common.TRAIN_SEEDS),
        "observed_bundle_count": len(bundles),
        "passed_bundle_count": sum(bundle["passed"] for bundle in bundles),
        "fixed_test_bank": True,
        "failures": failures,
        "bundles": bundles,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topologies-csv", type=Path, default=common.DEFAULT_SELECTED_TOPOLOGIES)
    parser.add_argument("--output-root", type=Path, default=common.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--formal-output-root", type=Path, default=common.DEFAULT_FORMAL_OUTPUT_ROOT)
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args()
    rows = common.read_csv(args.topologies_csv)
    audit = audit_all(rows, args.output_root, args.formal_output_root)
    output = args.audit_output or args.output_root / "results" / "repeat_seed_artifact_audit.json"
    builder.common.atomic_write_json(output, audit)
    print(json.dumps({key: audit[key] for key in ("passed", "observed_topology_count", "observed_bundle_count", "passed_bundle_count", "failures")}, indent=2, sort_keys=True))
    return 0 if audit["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
