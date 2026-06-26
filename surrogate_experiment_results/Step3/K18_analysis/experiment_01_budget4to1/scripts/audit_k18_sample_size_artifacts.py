#!/usr/bin/env python3
"""Audit K18 sample-size 4:1 artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[5]
STEP3_SCRIPTS = PROJECT_ROOT / "surrogate_experiment_results" / "Step3" / "scripts"
if str(STEP3_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(STEP3_SCRIPTS))

import fixed_topology_xy_common as common  # noqa: E402


def _append(failures: list[str], failure: str) -> None:
    if failure not in failures:
        failures.append(failure)


def _resolve_manifest_path(manifest_path: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return manifest_path.parent / path


def _is_subset(left: list[int], right: list[int]) -> bool:
    return set(left).issubset(set(right))


def audit_artifacts(
    *,
    train_bank_path: str | Path,
    split_manifest_path: str | Path,
    eval_manifest_paths: list[str | Path],
) -> dict[str, Any]:
    failures: list[str] = []
    train_bank_path = Path(train_bank_path)
    split_manifest_path = Path(split_manifest_path)
    train_bank = common.read_npz_dataset(train_bank_path)
    train_manifest = train_bank["manifest"]
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))

    sample_sizes = [int(size) for size in split_manifest.get("sample_sizes", [])]
    if sample_sizes != sorted(sample_sizes) or len(sample_sizes) != len(set(sample_sizes)):
        _append(failures, "sample_sizes_not_strictly_increasing")
    if split_manifest.get("assignment_rule") != "every_fifth_sample_is_validation":
        _append(failures, "assignment_rule_mismatch")
    if Path(split_manifest.get("training_bank_path", train_bank_path)) != train_bank_path:
        _append(failures, "training_bank_path_mismatch")

    prefix_hashes = train_manifest.get("prefix_hashes", {})
    sample_size_splits = split_manifest.get("sample_size_splits", {})
    training_prefix_sizes: list[int] = []
    previous_training_indices: list[int] = []
    previous_validation_indices: list[int] = []
    test_hashes: set[str] = set()
    eval_manifest_by_sample_size: dict[int, tuple[Path, dict[str, Any]]] = {}

    for path in eval_manifest_paths:
        manifest_path = Path(path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        eval_manifest_by_sample_size[int(manifest["sample_size"])] = (manifest_path, manifest)

    for sample_size in sample_sizes:
        key = str(sample_size)
        split = sample_size_splits.get(key)
        if split is None:
            _append(failures, f"missing_split_{sample_size}")
            continue
        training_size = int(split.get("training_size", -1))
        validation_size = int(split.get("validation_size", -1))
        training_indices = [int(value) for value in split.get("training_indices", [])]
        validation_indices = [int(value) for value in split.get("validation_indices", [])]
        training_prefix_sizes.append(training_size)

        if training_size != len(training_indices):
            _append(failures, f"training_size_mismatch_{sample_size}")
        if validation_size != len(validation_indices):
            _append(failures, f"validation_size_mismatch_{sample_size}")
        if set(training_indices).intersection(validation_indices):
            _append(failures, f"role_leakage_{sample_size}")
        if previous_training_indices and not _is_subset(previous_training_indices, training_indices):
            _append(failures, f"training_not_nested_{sample_size}")
        if previous_validation_indices and not _is_subset(previous_validation_indices, validation_indices):
            _append(failures, f"validation_not_nested_{sample_size}")
        previous_training_indices = training_indices
        previous_validation_indices = validation_indices

        prefix_hash = prefix_hashes.get(str(training_size))
        if not prefix_hash:
            _append(failures, f"missing_training_prefix_hash_{sample_size}")
        if prefix_hash and split.get("training_hash") != prefix_hash:
            _append(failures, f"training_hash_mismatch_{sample_size}")

        pair = eval_manifest_by_sample_size.get(sample_size)
        if pair is None:
            _append(failures, f"missing_eval_manifest_{sample_size}")
            continue
        eval_manifest_path, eval_manifest = pair
        if int(eval_manifest.get("training_size", -1)) != training_size:
            _append(failures, f"eval_training_size_mismatch_{sample_size}")
        if int(eval_manifest.get("validation_size", -1)) != validation_size:
            _append(failures, f"eval_validation_size_mismatch_{sample_size}")
        if int(eval_manifest.get("trainer_train_size_arg", -1)) != training_size:
            _append(failures, f"trainer_train_size_arg_mismatch_{sample_size}")
        if eval_manifest.get("test_hash"):
            test_hashes.add(str(eval_manifest["test_hash"]))

        validation_path = _resolve_manifest_path(eval_manifest_path, eval_manifest["validation_path"])
        validation_dataset = common.read_npz_dataset(validation_path)
        validation_rows = validation_dataset["manifest"].get("samples", [])
        validation_hash = common.sample_manifest_hashes(validation_rows)
        if validation_dataset["manifest"].get("sample_count") != validation_size:
            _append(failures, f"validation_npz_size_mismatch_{sample_size}")
        if validation_hash != eval_manifest.get("validation_hash"):
            _append(failures, f"validation_hash_mismatch_{sample_size}")
        if split.get("validation_hash") != eval_manifest.get("validation_hash"):
            _append(failures, f"split_eval_validation_hash_mismatch_{sample_size}")

    if len(test_hashes) > 1:
        _append(failures, "test_hash_not_shared_across_sample_sizes")
    return {
        "passed": not failures,
        "failures": failures,
        "sample_size_count": len(sample_sizes),
        "sample_sizes": sample_sizes,
        "training_prefix_sizes": training_prefix_sizes,
        "test_hashes": sorted(test_hashes),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-bank", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = audit_artifacts(
        train_bank_path=args.train_bank,
        split_manifest_path=args.split_manifest,
        eval_manifest_paths=args.eval_manifest,
    )
    if args.output:
        common.atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
