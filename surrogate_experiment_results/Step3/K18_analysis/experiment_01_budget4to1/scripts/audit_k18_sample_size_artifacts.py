#!/usr/bin/env python3
"""Audit K18 sample-size 4:1 artifacts."""

from __future__ import annotations

import argparse
import csv
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


def _manifest_sample_indices(rows: list[dict[str, Any]]) -> list[int]:
    return [int(row["sample_index"]) for row in rows if "sample_index" in row]


def _nonempty_provenance(manifest: dict[str, Any], failures: list[str], prefix: str) -> None:
    for key in (
        "experiment_version",
        "master_label_seed",
        "generator_version",
        "generator_config_hash",
        "topology_hash",
        "arc_order_hash",
        "feasible_set_hash",
    ):
        if manifest.get(key) in (None, ""):
            _append(failures, f"{prefix}_missing_{key}")


def _read_npz_or_failure(path: Path, failures: list[str], failure: str) -> dict[str, Any] | None:
    if not path.exists():
        _append(failures, failure)
        return None
    try:
        return common.read_npz_dataset(path)
    except Exception as exc:  # pragma: no cover - exact numpy error type is not stable.
        _append(failures, f"{failure}: {exc}")
        return None


def audit_artifacts(
    *,
    train_bank_path: str | Path,
    split_manifest_path: str | Path,
    eval_manifest_paths: list[str | Path],
    expected_test_size: int | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    train_bank_path = Path(train_bank_path)
    split_manifest_path = Path(split_manifest_path)
    train_bank = common.read_npz_dataset(train_bank_path)
    train_manifest = train_bank["manifest"]
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    _nonempty_provenance(train_manifest, failures, "train_bank")
    _nonempty_provenance(split_manifest, failures, "split_manifest")

    sample_sizes = [int(size) for size in split_manifest.get("sample_sizes", [])]
    if sample_sizes != sorted(sample_sizes) or len(sample_sizes) != len(set(sample_sizes)):
        _append(failures, "sample_sizes_not_strictly_increasing")
    if split_manifest.get("assignment_rule") != "every_fifth_sample_is_validation":
        _append(failures, "assignment_rule_mismatch")
    if Path(split_manifest.get("training_bank_path", train_bank_path)) != train_bank_path:
        _append(failures, "training_bank_path_mismatch")

    fit_manifest: dict[str, Any] | None = None
    fit_manifest_raw = split_manifest.get("fit_manifest_path")
    if not fit_manifest_raw:
        _append(failures, "fit_manifest_missing")
    else:
        fit_manifest_path = _resolve_manifest_path(split_manifest_path, fit_manifest_raw)
        if not fit_manifest_path.exists():
            _append(failures, "fit_manifest_missing")
        else:
            fit_manifest = json.loads(fit_manifest_path.read_text(encoding="utf-8"))
            _nonempty_provenance(fit_manifest, failures, "fit_manifest")
            fit_rows = list(fit_manifest.get("samples", []))
            computed_fit_hash = common.sample_manifest_hashes(fit_rows)
            if fit_manifest.get("fit_bank_hash") != computed_fit_hash:
                _append(failures, "fit_manifest_hash_mismatch")
            if split_manifest.get("fit_bank_hash") != computed_fit_hash:
                _append(failures, "split_fit_bank_hash_mismatch")
            if int(fit_manifest.get("sample_count", -1)) != len(fit_rows):
                _append(failures, "fit_manifest_sample_count_mismatch")

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
        _nonempty_provenance(eval_manifest, failures, f"eval_manifest_{sample_size}")
        for key in ("topology_id", "regime", "protocol", "data_seed"):
            if eval_manifest.get(key) != split_manifest.get(key):
                _append(failures, f"eval_{key}_mismatch_{sample_size}")
        if int(eval_manifest.get("training_size", -1)) != training_size:
            _append(failures, f"eval_training_size_mismatch_{sample_size}")
        if int(eval_manifest.get("validation_size", -1)) != validation_size:
            _append(failures, f"eval_validation_size_mismatch_{sample_size}")
        if int(eval_manifest.get("trainer_train_size_arg", -1)) != training_size:
            _append(failures, f"trainer_train_size_arg_mismatch_{sample_size}")
        if int(eval_manifest.get("sample_size", -1)) != training_size + validation_size:
            _append(failures, f"sample_size_sum_mismatch_{sample_size}")
        if eval_manifest.get("test_hash"):
            test_hashes.add(str(eval_manifest["test_hash"]))

        validation_path = _resolve_manifest_path(eval_manifest_path, eval_manifest["validation_path"])
        validation_dataset = _read_npz_or_failure(
            validation_path,
            failures,
            f"validation_npz_missing_{sample_size}",
        )
        if validation_dataset is None:
            continue
        _nonempty_provenance(validation_dataset["manifest"], failures, f"validation_npz_{sample_size}")
        validation_rows = validation_dataset["manifest"].get("samples", [])
        validation_hash = common.sample_manifest_hashes(validation_rows)
        if validation_dataset["manifest"].get("sample_count") != validation_size:
            _append(failures, f"validation_npz_size_mismatch_{sample_size}")
        if validation_hash != eval_manifest.get("validation_hash"):
            _append(failures, f"validation_hash_mismatch_{sample_size}")
        if split.get("validation_hash") != eval_manifest.get("validation_hash"):
            _append(failures, f"split_eval_validation_hash_mismatch_{sample_size}")
        if validation_dataset["manifest"].get("split_namespace") != train_manifest.get("split_namespace"):
            _append(failures, f"validation_source_namespace_mismatch_{sample_size}")

        test_path = _resolve_manifest_path(eval_manifest_path, eval_manifest.get("test_path", ""))
        test_dataset = _read_npz_or_failure(test_path, failures, f"test_npz_missing_{sample_size}")
        if test_dataset is not None:
            test_manifest = test_dataset["manifest"]
            _nonempty_provenance(test_manifest, failures, f"test_npz_{sample_size}")
            test_rows = test_manifest.get("samples", [])
            computed_test_hash = common.sample_manifest_hashes(test_rows)
            for key in ("topology_id", "regime", "protocol"):
                if test_manifest.get(key) != eval_manifest.get(key):
                    _append(failures, f"test_{key}_mismatch_{sample_size}")
            sample_topology_ids = {str(row.get("topology_id", "")) for row in test_rows}
            if sample_topology_ids and sample_topology_ids != {str(eval_manifest.get("topology_id", ""))}:
                _append(failures, f"test_sample_topology_id_mismatch_{sample_size}")
            if test_manifest.get("dataset_hash") != computed_test_hash:
                _append(failures, f"test_npz_dataset_hash_mismatch_{sample_size}")
            if computed_test_hash != eval_manifest.get("test_hash"):
                _append(failures, f"test_hash_mismatch_{sample_size}")
            if expected_test_size is not None and int(test_manifest.get("sample_count", -1)) != int(expected_test_size):
                _append(failures, f"test_npz_size_mismatch_{sample_size}")
            expected_test_namespace = f"{eval_manifest.get('protocol', train_manifest.get('protocol'))}_test"
            if test_manifest.get("split_namespace") != expected_test_namespace:
                _append(failures, f"test_namespace_mismatch_{sample_size}")

    if fit_manifest is not None and sample_sizes:
        largest_sample_size = sample_sizes[-1]
        largest_split = sample_size_splits.get(str(largest_sample_size), {})
        training_indices = {int(value) for value in largest_split.get("training_indices", [])}
        validation_indices = {int(value) for value in largest_split.get("validation_indices", [])}
        fit_role_indices = {
            int(row["fit_index"])
            for row in fit_manifest.get("fit_role_rows", [])
            if "fit_index" in row
        }
        if training_indices.intersection(validation_indices):
            _append(failures, "fit_training_validation_overlap")
        if fit_role_indices and training_indices.union(validation_indices) != fit_role_indices:
            _append(failures, "fit_training_validation_union_mismatch")
        train_bank_indices = set(_manifest_sample_indices(train_manifest.get("samples", [])))
        if not training_indices.issubset(train_bank_indices):
            _append(failures, "fit_training_indices_missing_from_train_bank")

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


def audit_plan_jobs(
    jobs: list[dict[str, Any]],
    *,
    expected_topology_count: int | None = None,
    expected_data_seed_count: int | None = None,
    expected_sample_sizes: list[int] | tuple[int, ...] | None = None,
    expected_job_count: int | None = None,
    require_ready: bool = True,
) -> dict[str, Any]:
    failures: list[str] = []
    job_ids = [str(job.get("job_id", "")) for job in jobs]
    if len(job_ids) != len(set(job_ids)):
        _append(failures, "duplicate_job_id")

    topologies = sorted({str(job.get("topology_id", "")) for job in jobs if job.get("topology_id")})
    data_seeds = sorted({int(job.get("data_seed")) for job in jobs if str(job.get("data_seed", "")).strip()})
    sample_sizes = sorted({int(job.get("sample_size")) for job in jobs if str(job.get("sample_size", "")).strip()})
    ready_jobs = [job for job in jobs if str(job.get("status", "")) == "ready"]

    if expected_topology_count is not None and len(topologies) != int(expected_topology_count):
        _append(failures, "topology_count_mismatch")
    if expected_data_seed_count is not None and len(data_seeds) != int(expected_data_seed_count):
        _append(failures, "data_seed_count_mismatch")
    if expected_sample_sizes is not None and sample_sizes != sorted(int(size) for size in expected_sample_sizes):
        _append(failures, "sample_sizes_mismatch")
    if expected_job_count is not None and len(jobs) != int(expected_job_count):
        _append(failures, "job_count_mismatch")
    if require_ready and len(ready_jobs) != len(jobs):
        _append(failures, "not_all_jobs_ready")

    for job in jobs:
        for key in ("expected_training_hash", "validation_hash", "test_hash"):
            if job.get(key) in (None, ""):
                _append(failures, f"missing_{key}")

    for topology_id in topologies:
        hashes = {
            str(job.get("test_hash", ""))
            for job in jobs
            if str(job.get("topology_id", "")) == topology_id and job.get("test_hash")
        }
        if len(hashes) > 1:
            _append(failures, f"test_hash_not_shared_for_topology_{topology_id}")

    return {
        "passed": not failures,
        "failures": failures,
        "job_count": len(jobs),
        "ready_count": len(ready_jobs),
        "topology_count": len(topologies),
        "data_seed_count": len(data_seeds),
        "sample_sizes": sample_sizes,
    }


def _read_jobs_csv(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-bank", type=Path, default=None)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--eval-manifest", type=Path, action="append", default=None)
    parser.add_argument("--expected-test-size", type=int, default=None)
    parser.add_argument("--plan-json", type=Path, default=None)
    parser.add_argument("--jobs-csv", type=Path, default=None)
    parser.add_argument("--expected-topology-count", type=int, default=None)
    parser.add_argument("--expected-data-seed-count", type=int, default=None)
    parser.add_argument("--expected-sample-sizes", default=None)
    parser.add_argument("--expected-job-count", type=int, default=None)
    parser.add_argument("--allow-non-ready", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.plan_json or args.jobs_csv:
        if args.plan_json:
            payload = json.loads(args.plan_json.read_text(encoding="utf-8"))
            jobs = list(payload.get("jobs", []))
        else:
            jobs = _read_jobs_csv(args.jobs_csv)
        expected_sample_sizes = None
        if args.expected_sample_sizes:
            expected_sample_sizes = [
                int(part.strip())
                for part in str(args.expected_sample_sizes).split(",")
                if part.strip()
            ]
        result = audit_plan_jobs(
            jobs,
            expected_topology_count=args.expected_topology_count,
            expected_data_seed_count=args.expected_data_seed_count,
            expected_sample_sizes=expected_sample_sizes,
            expected_job_count=args.expected_job_count,
            require_ready=not args.allow_non_ready,
        )
    else:
        if not args.train_bank or not args.split_manifest or not args.eval_manifest:
            raise SystemExit("--train-bank, --split-manifest, and --eval-manifest are required for bundle audit")
        result = audit_artifacts(
            train_bank_path=args.train_bank,
            split_manifest_path=args.split_manifest,
            eval_manifest_paths=args.eval_manifest,
            expected_test_size=args.expected_test_size,
        )
    if args.output:
        common.atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
