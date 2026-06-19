#!/usr/bin/env python3
"""Audit Step3 Phase-B materialized fixed-topology datasets."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_DIR = (
    PROJECT_ROOT / "surrogate_experiment_results" / "Step3" / "pairs20_ndd2" / "phase_b" / "datasets"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "surrogate_experiment_results" / "Step3" / "pairs20_ndd2" / "phase_b" / "audit"
)

SAMPLE_FIELDS = [
    "topology_id",
    "split",
    "train_seed",
    "sample_idx",
    "label_seed",
    "path",
    "topology_hash",
    "label_hash",
    "edge_count",
]

INDEX_FIELDS = [
    "topology_id",
    "source_graph_path",
    "output_dir",
    "train_seed_count",
    "training_size_budget",
    "train_sample_count",
    "validation_sample_count",
    "test_sample_count",
    "topology_hash",
    "topology_bank_hash",
    "feasible_set_hash",
    "status",
]

AUDIT_FIELDS = [
    "topology_id",
    "status",
    "failure_reasons",
    "validation_json_count",
    "test_json_count",
    "train_seed_dir_count",
    "train_seed_min",
    "train_seed_max",
    "train_json_total",
    "min_train_json_per_seed",
    "max_train_json_per_seed",
    "samples_csv_rows",
    "samples_validation_rows",
    "samples_test_rows",
    "samples_train_rows",
    "unique_sample_topology_hashes",
    "unique_sample_label_hashes",
    "manifest_num_sample_files",
    "manifest_topology_hash",
    "index_topology_hash",
    "manifest_topology_bank_hash",
    "index_topology_bank_hash",
    "manifest_feasible_set_hash",
    "index_feasible_set_hash",
]


class ExpectedLayout:
    def __init__(
        self,
        train_seed_count: int = 50,
        train_sample_count: int = 40,
        validation_sample_count: int = 10,
        test_sample_count: int = 1000,
    ) -> None:
        self.train_seed_count = int(train_seed_count)
        self.train_sample_count = int(train_sample_count)
        self.validation_sample_count = int(validation_sample_count)
        self.test_sample_count = int(test_sample_count)

    @property
    def samples_per_topology(self) -> int:
        return (
            self.train_seed_count * self.train_sample_count
            + self.validation_sample_count
            + self.test_sample_count
        )


def int_or_text_key(value: Any) -> tuple[int, int | str]:
    text = str(value)
    if text.startswith("G-"):
        text = text[2:]
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def read_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def to_int(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    return int(float(value))


def train_seed_from_dir(path: Path) -> int | None:
    name = path.parent.name if path.name == "train" else path.name
    if not name.startswith("train_seed="):
        return None
    try:
        return int(name.split("=", 1)[1])
    except ValueError:
        return None


def count_split_rows(sample_rows: list[dict[str, Any]], split: str) -> int:
    return sum(1 for row in sample_rows if str(row.get("split")) == split)


def sample_train_seed_values(sample_rows: list[dict[str, Any]]) -> set[int]:
    values: set[int] = set()
    for row in sample_rows:
        if str(row.get("split")) != "train":
            continue
        raw = row.get("train_seed")
        if raw in (None, ""):
            continue
        values.add(to_int(raw))
    return values


def row_failure_text(failures: list[str]) -> str:
    return "OK" if not failures else "; ".join(failures)


def audit_topology(
    base_dir: Path,
    index_row: dict[str, Any],
    expected: ExpectedLayout,
) -> dict[str, Any]:
    topology_id = str(index_row["topology_id"])
    topology_dir = base_dir / topology_id
    failures: list[str] = []

    validation_files = sorted((topology_dir / "validation").glob("G-*.json"))
    test_files = sorted((topology_dir / "test").glob("G-*.json"))
    train_dirs = sorted(topology_dir.glob("train_seed=*/train"), key=train_seed_from_dir)
    train_seed_values = [
        value for value in (train_seed_from_dir(path) for path in train_dirs) if value is not None
    ]
    train_counts = [len(list(path.glob("G-*.json"))) for path in train_dirs]
    train_total = sum(train_counts)

    if not topology_dir.exists():
        failures.append("topology_dir_missing")
    if len(validation_files) != expected.validation_sample_count:
        failures.append(
            f"validation_json_count={len(validation_files)} expected={expected.validation_sample_count}"
        )
    if len(test_files) != expected.test_sample_count:
        failures.append(f"test_json_count={len(test_files)} expected={expected.test_sample_count}")
    if len(train_dirs) != expected.train_seed_count:
        failures.append(f"train_seed_dir_count={len(train_dirs)} expected={expected.train_seed_count}")
    expected_seed_values = set(range(1, expected.train_seed_count + 1))
    if set(train_seed_values) != expected_seed_values:
        missing = sorted(expected_seed_values - set(train_seed_values))
        extra = sorted(set(train_seed_values) - expected_seed_values)
        failures.append(f"train_seed_values_mismatch missing={missing[:5]} extra={extra[:5]}")
    if train_counts and (
        min(train_counts) != expected.train_sample_count
        or max(train_counts) != expected.train_sample_count
    ):
        failures.append(
            f"train_json_per_seed_range={min(train_counts)}..{max(train_counts)} "
            f"expected={expected.train_sample_count}"
        )
    if not train_counts and expected.train_seed_count:
        failures.append("train_json_per_seed_missing")

    manifest_path = topology_dir / "dataset_manifest.json"
    sample_path = topology_dir / "samples.csv"
    manifest: dict[str, Any] = {}
    sample_rows: list[dict[str, Any]] = []
    if manifest_path.exists():
        manifest = read_json(manifest_path)
    else:
        failures.append("dataset_manifest_missing")
    if sample_path.exists():
        sample_rows = read_csv_rows(sample_path)
    else:
        failures.append("samples_csv_missing")

    samples_validation = count_split_rows(sample_rows, "validation")
    samples_test = count_split_rows(sample_rows, "test")
    samples_train = count_split_rows(sample_rows, "train")
    sample_topology_hashes = {str(row.get("topology_hash", "")) for row in sample_rows}
    sample_label_hashes = {str(row.get("label_hash", "")) for row in sample_rows}
    sample_train_seeds = sample_train_seed_values(sample_rows)

    if len(sample_rows) != expected.samples_per_topology:
        failures.append(f"samples_csv_rows={len(sample_rows)} expected={expected.samples_per_topology}")
    if samples_validation != expected.validation_sample_count:
        failures.append(
            f"samples_validation_rows={samples_validation} expected={expected.validation_sample_count}"
        )
    if samples_test != expected.test_sample_count:
        failures.append(f"samples_test_rows={samples_test} expected={expected.test_sample_count}")
    expected_train_rows = expected.train_seed_count * expected.train_sample_count
    if samples_train != expected_train_rows:
        failures.append(f"samples_train_rows={samples_train} expected={expected_train_rows}")
    if sample_train_seeds and sample_train_seeds != expected_seed_values:
        missing = sorted(expected_seed_values - sample_train_seeds)
        extra = sorted(sample_train_seeds - expected_seed_values)
        failures.append(f"samples_train_seed_values_mismatch missing={missing[:5]} extra={extra[:5]}")

    manifest_topology_hash = str(manifest.get("topology_hash", ""))
    index_topology_hash = str(index_row.get("topology_hash", ""))
    if manifest_topology_hash != index_topology_hash:
        failures.append(
            f"topology_hash mismatch index={index_topology_hash} manifest={manifest_topology_hash}"
        )
    if sample_rows and sample_topology_hashes != {index_topology_hash}:
        failures.append(
            f"sample_topology_hash_count={len(sample_topology_hashes)} expected={index_topology_hash}"
        )

    manifest_bank_hash = str(manifest.get("topology_bank_hash", ""))
    index_bank_hash = str(index_row.get("topology_bank_hash", ""))
    if manifest_bank_hash != index_bank_hash:
        failures.append(
            f"topology_bank_hash mismatch index={index_bank_hash} manifest={manifest_bank_hash}"
        )

    manifest_feasible_hash = str(manifest.get("feasible_set_hash", ""))
    index_feasible_hash = str(index_row.get("feasible_set_hash", ""))
    if manifest_feasible_hash != index_feasible_hash:
        failures.append(
            f"feasible_set_hash mismatch index={index_feasible_hash} manifest={manifest_feasible_hash}"
        )

    manifest_num_sample_files = to_int(manifest.get("num_sample_files"), default=-1)
    if manifest and manifest_num_sample_files != expected.samples_per_topology:
        failures.append(
            f"manifest_num_sample_files={manifest_num_sample_files} expected={expected.samples_per_topology}"
        )

    return {
        "topology_id": topology_id,
        "status": "pass" if not failures else "fail",
        "failure_reasons": row_failure_text(failures),
        "validation_json_count": len(validation_files),
        "test_json_count": len(test_files),
        "train_seed_dir_count": len(train_dirs),
        "train_seed_min": min(train_seed_values) if train_seed_values else "",
        "train_seed_max": max(train_seed_values) if train_seed_values else "",
        "train_json_total": train_total,
        "min_train_json_per_seed": min(train_counts) if train_counts else "",
        "max_train_json_per_seed": max(train_counts) if train_counts else "",
        "samples_csv_rows": len(sample_rows),
        "samples_validation_rows": samples_validation,
        "samples_test_rows": samples_test,
        "samples_train_rows": samples_train,
        "unique_sample_topology_hashes": len(sample_topology_hashes) if sample_rows else 0,
        "unique_sample_label_hashes": len(sample_label_hashes) if sample_rows else 0,
        "manifest_num_sample_files": manifest_num_sample_files if manifest else "",
        "manifest_topology_hash": manifest_topology_hash,
        "index_topology_hash": index_topology_hash,
        "manifest_topology_bank_hash": manifest_bank_hash,
        "index_topology_bank_hash": index_bank_hash,
        "manifest_feasible_set_hash": manifest_feasible_hash,
        "index_feasible_set_hash": index_feasible_hash,
    }


def audit_phase_b_dataset(
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    expected: ExpectedLayout | None = None,
) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    expected = expected or ExpectedLayout()
    index_path = dataset_dir / "phase_b_dataset_index.csv"
    manifest_path = dataset_dir / "phase_b_dataset_manifest.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing dataset index: {index_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing dataset manifest: {manifest_path}")

    index_rows = read_csv_rows(index_path)
    root_manifest = read_json(manifest_path)
    topology_rows = [
        audit_topology(dataset_dir, row, expected)
        for row in sorted(index_rows, key=lambda item: int_or_text_key(item["topology_id"]))
    ]
    failed = [row for row in topology_rows if row["status"] != "pass"]
    total_json_files = sum(
        int(row["validation_json_count"])
        + int(row["test_json_count"])
        + int(row["train_json_total"])
        for row in topology_rows
    )
    expected_total_json = len(index_rows) * expected.samples_per_topology
    root_status = str(root_manifest.get("status", ""))
    summary = {
        "generated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "dataset_dir": str(dataset_dir),
        "root_manifest_status": root_status,
        "num_topologies": len(index_rows),
        "num_failed_topologies": len(failed),
        "total_json_files": total_json_files,
        "expected_total_json_files": expected_total_json,
        "train_seed_count": expected.train_seed_count,
        "train_sample_count": expected.train_sample_count,
        "validation_sample_count": expected.validation_sample_count,
        "test_sample_count": expected.test_sample_count,
        "samples_per_topology": expected.samples_per_topology,
        "passed": (
            len(failed) == 0
            and root_status == "materialized"
            and total_json_files == expected_total_json
        ),
    }
    return {
        "passed": bool(summary["passed"]),
        "summary": summary,
        "topology_rows": topology_rows,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-seed-count", type=int, default=50)
    parser.add_argument("--train-sample-count", type=int, default=40)
    parser.add_argument("--validation-sample-count", type=int, default=10)
    parser.add_argument("--test-sample-count", type=int, default=1000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    expected = ExpectedLayout(
        train_seed_count=args.train_seed_count,
        train_sample_count=args.train_sample_count,
        validation_sample_count=args.validation_sample_count,
        test_sample_count=args.test_sample_count,
    )
    result = audit_phase_b_dataset(args.dataset_dir, expected=expected)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "phase_b_dataset_audit.csv", result["topology_rows"], AUDIT_FIELDS)
    write_json(args.output_dir / "phase_b_dataset_audit_summary.json", result["summary"])
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
