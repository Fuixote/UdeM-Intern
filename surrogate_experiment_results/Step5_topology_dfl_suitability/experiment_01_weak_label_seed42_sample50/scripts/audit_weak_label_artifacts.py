#!/usr/bin/env python3
"""Audit Step5 weak-label data bundles before paired jobs are planned."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
STEP3_SCRIPTS = PROJECT_ROOT / "surrogate_experiment_results" / "Step3" / "scripts"
for import_path in (SCRIPT_DIR, STEP3_SCRIPTS):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import build_weak_label_artifacts as builder  # noqa: E402
import fixed_topology_xy_common as common  # noqa: E402


DEFAULT_TOPOLOGIES = EXPERIMENT_ROOT / "configs" / "topologies.locked.csv"
PROVENANCE_FIELDS = (
    "experiment_version",
    "master_label_seed",
    "generator_version",
    "generator_config_hash",
    "topology_hash",
    "arc_order_hash",
    "feasible_set_hash",
)


def _append(failures: list[str], message: str) -> None:
    if message not in failures:
        failures.append(message)


def _read_json(path: Path, failures: list[str], label: str) -> dict[str, Any] | None:
    if not path.is_file():
        _append(failures, f"missing_{label}")
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _append(failures, f"invalid_{label}:{exc}")
        return None
    if not isinstance(value, dict):
        _append(failures, f"invalid_{label}:not_an_object")
        return None
    return value


def _read_npz(path: Path, failures: list[str], label: str) -> dict[str, Any] | None:
    if not path.is_file():
        _append(failures, f"missing_{label}")
        return None
    try:
        return common.read_npz_dataset(path)
    except Exception as exc:  # pragma: no cover - numpy error types vary.
        _append(failures, f"invalid_{label}:{exc}")
        return None


def _check_provenance(
    manifest: dict[str, Any],
    *,
    row: dict[str, str],
    failures: list[str],
    label: str,
) -> None:
    for field in PROVENANCE_FIELDS:
        if manifest.get(field) in (None, ""):
            _append(failures, f"{label}_missing_{field}")
    for field in ("topology_hash", "arc_order_hash", "feasible_set_hash"):
        if str(manifest.get(field, "")) != str(row.get(field, "")):
            _append(failures, f"{label}_{field}_mismatch")


def _check_core_fields(
    manifest: dict[str, Any],
    *,
    topology_id: str,
    regime: str,
    protocol: str,
    failures: list[str],
    label: str,
) -> None:
    expected = {
        "topology_id": str(topology_id),
        "regime": str(regime),
        "protocol": str(protocol),
    }
    for field, value in expected.items():
        if str(manifest.get(field, "")) != value:
            _append(failures, f"{label}_{field}_mismatch")


def audit_bundle(
    row: dict[str, str],
    *,
    output_root: str | Path,
    regime: str,
    protocol: str,
    data_seed: int,
    sample_size: int,
    test_size: int,
) -> dict[str, Any]:
    topology_id = str(row["topology_id"])
    training_size, validation_size = builder.validate_sample_size(sample_size)
    paths = builder.artifact_paths(
        output_root,
        regime=regime,
        topology_id=topology_id,
        data_seed=data_seed,
        sample_size=sample_size,
    )
    failures: list[str] = []
    train_bank = _read_npz(paths["train_bank"], failures, "train_bank")
    validation = _read_npz(paths["validation"], failures, "validation")
    test = _read_npz(paths["test"], failures, "test")
    fit_manifest = _read_json(paths["fit_manifest"], failures, "fit_manifest")
    split_manifest = _read_json(paths["split_manifest"], failures, "split_manifest")
    eval_manifest = _read_json(paths["eval_manifest"], failures, "eval_manifest")
    test_manifest = _read_json(paths["test_manifest"], failures, "test_manifest")

    manifests: list[tuple[str, dict[str, Any]]] = []
    for label, value in (
        ("train_bank", None if train_bank is None else train_bank["manifest"]),
        ("validation", None if validation is None else validation["manifest"]),
        ("test", None if test is None else test["manifest"]),
        ("fit_manifest", fit_manifest),
        ("split_manifest", split_manifest),
        ("eval_manifest", eval_manifest),
        ("test_manifest", test_manifest),
    ):
        if value is not None:
            manifests.append((label, value))
            _check_core_fields(
                value,
                topology_id=topology_id,
                regime=regime,
                protocol=protocol,
                failures=failures,
                label=label,
            )
            _check_provenance(value, row=row, failures=failures, label=label)

    if train_bank is not None:
        manifest = train_bank["manifest"]
        rows = list(manifest.get("samples", []))
        observed_hash = common.sample_manifest_hashes(rows)
        if len(rows) != training_size or int(manifest.get("max_train_size", -1)) != training_size:
            _append(failures, "train_bank_size_mismatch")
        if manifest.get("bank_hash") != observed_hash:
            _append(failures, "train_bank_hash_mismatch")
        if manifest.get("prefix_hashes", {}).get(str(training_size)) != observed_hash:
            _append(failures, "training_prefix_hash_mismatch")
        if manifest.get("split_namespace") != builder.train_namespace(protocol):
            _append(failures, "train_namespace_mismatch")

    if validation is not None:
        manifest = validation["manifest"]
        rows = list(manifest.get("samples", []))
        observed_hash = common.sample_manifest_hashes(rows)
        if len(rows) != validation_size or int(manifest.get("sample_count", -1)) != validation_size:
            _append(failures, "validation_size_mismatch")
        if manifest.get("dataset_hash") != observed_hash:
            _append(failures, "validation_hash_mismatch")
        if manifest.get("split_namespace") != builder.train_namespace(protocol):
            _append(failures, "validation_namespace_mismatch")

    if test is not None:
        manifest = test["manifest"]
        rows = list(manifest.get("samples", []))
        observed_hash = common.sample_manifest_hashes(rows)
        if len(rows) != int(test_size) or int(manifest.get("sample_count", -1)) != int(test_size):
            _append(failures, "test_size_mismatch")
        if manifest.get("dataset_hash") != observed_hash:
            _append(failures, "test_hash_mismatch")
        if manifest.get("split_namespace") != builder.test_namespace(protocol):
            _append(failures, "test_namespace_mismatch")
        if {sample.get("train_seed") for sample in rows} - {None}:
            _append(failures, "test_varies_by_train_seed")

    if fit_manifest is not None:
        fit_rows = list(fit_manifest.get("samples", []))
        if len(fit_rows) != int(sample_size):
            _append(failures, "fit_sample_count_mismatch")
        if fit_manifest.get("fit_bank_hash") != common.sample_manifest_hashes(fit_rows):
            _append(failures, "fit_bank_hash_mismatch")

    split_row: dict[str, Any] = {}
    if split_manifest is not None:
        if split_manifest.get("assignment_rule") != "every_fifth_sample_is_validation":
            _append(failures, "split_assignment_rule_mismatch")
        split_row = split_manifest.get("sample_size_splits", {}).get(str(sample_size), {})
        if not split_row:
            _append(failures, "split_row_missing")
        if int(split_row.get("training_size", -1)) != training_size:
            _append(failures, "split_training_size_mismatch")
        if int(split_row.get("validation_size", -1)) != validation_size:
            _append(failures, "split_validation_size_mismatch")
        training_indices = {int(value) for value in split_row.get("training_indices", [])}
        validation_indices = {int(value) for value in split_row.get("validation_indices", [])}
        if len(training_indices) != training_size:
            _append(failures, "split_training_indices_mismatch")
        if len(validation_indices) != validation_size:
            _append(failures, "split_validation_indices_mismatch")
        if training_indices.intersection(validation_indices):
            _append(failures, "training_validation_overlap")
        if training_indices.union(validation_indices) != set(range(int(sample_size))):
            _append(failures, "training_validation_union_mismatch")

    if eval_manifest is not None:
        expected_values = {
            "data_seed": int(data_seed),
            "sample_size": int(sample_size),
            "training_size": training_size,
            "validation_size": validation_size,
            "trainer_train_size_arg": training_size,
        }
        for field, value in expected_values.items():
            try:
                observed = int(eval_manifest.get(field, -1))
            except (TypeError, ValueError):
                observed = -1
            if observed != value:
                _append(failures, f"eval_{field}_mismatch")
        if validation is not None and eval_manifest.get("validation_hash") != validation["manifest"].get("dataset_hash"):
            _append(failures, "eval_validation_hash_mismatch")
        if test is not None and eval_manifest.get("test_hash") != test["manifest"].get("dataset_hash"):
            _append(failures, "eval_test_hash_mismatch")

    if test_manifest is not None and test is not None:
        if int(test_manifest.get("test_size", -1)) != int(test_size):
            _append(failures, "test_manifest_size_mismatch")
        if test_manifest.get("test_hash") != test["manifest"].get("dataset_hash"):
            _append(failures, "test_manifest_hash_mismatch")

    protocol_records = [manifest.get("step5_protocol") for _, manifest in manifests]
    if any(not isinstance(record, dict) for record in protocol_records):
        _append(failures, "missing_step5_protocol")
    elif len({json.dumps(record, sort_keys=True) for record in protocol_records}) != 1:
        _append(failures, "step5_protocol_drift")

    return {
        "topology_id": topology_id,
        "passed": not failures,
        "failures": failures,
        "data_seed": int(data_seed),
        "sample_size": int(sample_size),
        "training_size": training_size,
        "validation_size": validation_size,
        "test_size": int(test_size),
        "data_dir": str(paths["data_dir"]),
        "test_dir": str(paths["test_dir"]),
    }


def audit_sweep(
    topology_rows: list[dict[str, str]],
    *,
    output_root: str | Path,
    regime: str,
    protocol: str,
    data_seed: int,
    sample_size: int,
    test_size: int,
) -> dict[str, Any]:
    bundle_results = [
        audit_bundle(
            row,
            output_root=output_root,
            regime=regime,
            protocol=protocol,
            data_seed=data_seed,
            sample_size=sample_size,
            test_size=test_size,
        )
        for row in topology_rows
    ]
    failures = [
        f"{result['topology_id']}:{failure}"
        for result in bundle_results
        for failure in result["failures"]
    ]
    return {
        "passed": not failures,
        "failures": failures,
        "topology_count": len(topology_rows),
        "passed_topologies": sum(result["passed"] for result in bundle_results),
        "failed_topologies": sum(not result["passed"] for result in bundle_results),
        "data_seed": int(data_seed),
        "sample_size": int(sample_size),
        "training_size": builder.validate_sample_size(sample_size)[0],
        "validation_size": builder.validate_sample_size(sample_size)[1],
        "test_size": int(test_size),
        "regime": str(regime),
        "protocol": str(protocol),
        "bundles": bundle_results,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topologies-csv", type=Path, default=DEFAULT_TOPOLOGIES)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--regime", default=builder.DEFAULT_REGIME)
    parser.add_argument("--protocol", choices=("screen", "confirm"), default="screen")
    parser.add_argument("--data-seed", type=int, default=builder.DEFAULT_DATA_SEED)
    parser.add_argument("--sample-size", type=int, default=builder.DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--test-size", type=int, default=builder.DEFAULT_TEST_SIZE)
    parser.add_argument("--topology-id", action="append", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = builder.selected_topology_rows(
        args.topologies_csv,
        topology_ids=args.topology_id,
        limit=args.limit,
    )
    result = audit_sweep(
        rows,
        output_root=args.output_root,
        regime=args.regime,
        protocol=args.protocol,
        data_seed=args.data_seed,
        sample_size=args.sample_size,
        test_size=args.test_size,
    )
    output = args.output or Path(args.output_root) / "results" / "weak_label_artifact_audit.json"
    common.atomic_write_json(output, result)
    print(json.dumps({**{key: result[key] for key in ("passed", "topology_count", "passed_topologies", "failed_topologies", "failures")}, "output": str(output)}, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
