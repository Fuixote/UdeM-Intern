#!/usr/bin/env python3
"""Materialize Step3 Phase-B fixed-topology Step2c train/validation/test data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SELECTION_CSV = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "Step3"
    / "pairs20_ndd2"
    / "screening"
    / "phase_b_topologies.csv"
)
DEFAULT_PROCESSED_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "step3_pairs20_ndd2_step2c_poly_d8_mult_eps050_seed20260619"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "surrogate_experiment_results" / "Step3" / "pairs20_ndd2" / "phase_b" / "datasets"
)
DEFAULT_TOPOLOGY_DIR = (
    PROJECT_ROOT / "surrogate_experiment_results" / "Step3" / "pairs20_ndd2" / "data" / "topologies"
)
DEFAULT_TRAINING_SIZE_BUDGET = 50
DEFAULT_TRAIN_SEED_START = 1
DEFAULT_TRAIN_SEED_COUNT = 50
DEFAULT_TEST_SIZE = 1000
DEFAULT_EPSILON_BAR = 0.5

LABEL_SEED_NAMESPACES = {
    "train": 1,
    "validation": 2,
    "test": 3,
}
TOPOLOGY_BLOCK = 100_000_000
TRAIN_SEED_BLOCK = 10_000
NAMESPACE_BLOCK = 1_000_000_000_000_000

SAMPLE_MANIFEST_FIELDS = [
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

DATASET_INDEX_FIELDS = [
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


DECISION_ANALYSIS_SCRIPT_DIR = (
    PROJECT_ROOT / "surrogate_experiment_results" / "decision_analysis" / "scripts"
)
if str(DECISION_ANALYSIS_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(DECISION_ANALYSIS_SCRIPT_DIR))

from audit_fixed_topology_label_seed import (  # noqa: E402
    edge_count,
    edge_label_hash,
    relabel_payload_step2c,
    topology_hash,
)


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_topology_template_metadata(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "topology_bank_hash": "",
            "feasible_set_hash": "",
            "arc_order_hash": "",
            "num_exchange_candidates": "",
        }
    payload = read_json(path)
    return {
        "topology_bank_hash": payload.get("topology_hash", ""),
        "feasible_set_hash": payload.get("feasible_set_hash", ""),
        "arc_order_hash": payload.get("arc_order_hash", ""),
        "num_exchange_candidates": payload.get("num_exchange_candidates", ""),
    }


def phase_b_train_validation_counts(training_size_budget: int) -> tuple[int, int]:
    """Split the Phase-B training budget into 4:1 train:validation counts."""
    budget = int(training_size_budget)
    if budget < 5:
        raise ValueError("training_size_budget must be at least 5 for a 4:1 split")
    train_count = (4 * budget) // 5
    validation_count = budget - train_count
    if train_count <= 0 or validation_count <= 0:
        raise ValueError(f"Invalid train/validation split for budget={budget}")
    return train_count, validation_count


def topology_numeric_id(topology_id: str) -> int:
    text = str(topology_id)
    if text.startswith("G-"):
        text = text[2:]
    try:
        return int(text)
    except ValueError:
        digest = hashlib.sha256(str(topology_id).encode("utf-8")).hexdigest()
        return int(digest[:8], 16)


def phase_b_label_seed(
    topology_id: str,
    split: str,
    sample_idx: int,
    train_seed: int | None = None,
) -> int:
    """Return deterministic disjoint Step2c relabel seed for Phase-B samples."""
    if split not in LABEL_SEED_NAMESPACES:
        raise ValueError(f"Unknown split {split!r}; expected {sorted(LABEL_SEED_NAMESPACES)}")
    if split == "train" and train_seed is None:
        raise ValueError("train split requires train_seed")
    if split != "train" and train_seed is not None:
        raise ValueError(f"{split} split must not receive train_seed")
    sample_idx = int(sample_idx)
    if sample_idx < 0 or sample_idx >= TRAIN_SEED_BLOCK:
        raise ValueError(f"sample_idx must be in [0, {TRAIN_SEED_BLOCK}), got {sample_idx}")
    train_seed_value = 0 if train_seed is None else int(train_seed)
    max_train_seed = TOPOLOGY_BLOCK // TRAIN_SEED_BLOCK
    if train_seed_value < 0 or train_seed_value >= max_train_seed:
        raise ValueError(
            f"train_seed must be in [0, {max_train_seed}) for the default seed layout"
        )
    namespace = LABEL_SEED_NAMESPACES[split]
    return (
        namespace * NAMESPACE_BLOCK
        + topology_numeric_id(topology_id) * TOPOLOGY_BLOCK
        + train_seed_value * TRAIN_SEED_BLOCK
        + sample_idx
    )


def enrich_phase_b_metadata(
    payload: dict[str, Any],
    *,
    topology_id: str,
    split: str,
    train_seed: int | None,
    sample_idx: int,
    label_seed: int,
    training_size_budget: int,
    train_sample_count: int,
    validation_sample_count: int,
    test_sample_count: int,
) -> dict[str, Any]:
    payload.setdefault("metadata", {})
    payload["metadata"]["step3_phase_b"] = {
        "topology_id": str(topology_id),
        "split": str(split),
        "train_seed": None if train_seed is None else int(train_seed),
        "sample_idx": int(sample_idx),
        "label_seed": int(label_seed),
        "label_regime": "Step2c poly d8 multiplicative eps050",
        "training_size_budget": int(training_size_budget),
        "train_validation_ratio": "4:1",
        "train_sample_count": int(train_sample_count),
        "validation_sample_count": int(validation_sample_count),
        "test_sample_count": int(test_sample_count),
        "test_unseen_from_train_and_validation": True,
    }
    return payload


def materialize_sample(
    *,
    base_payload: dict[str, Any],
    base_topology_hash: str,
    output_path: Path,
    topology_id: str,
    split: str,
    train_seed: int | None,
    sample_idx: int,
    label_seed: int,
    training_size_budget: int,
    train_sample_count: int,
    validation_sample_count: int,
    test_sample_count: int,
    epsilon_bar: float,
) -> dict[str, Any]:
    relabeled = relabel_payload_step2c(
        base_payload,
        label_seed=int(label_seed),
        epsilon_bar=float(epsilon_bar),
    )
    if topology_hash(relabeled) != base_topology_hash:
        raise AssertionError(
            f"Topology changed for {topology_id} split={split} label_seed={label_seed}"
        )
    enrich_phase_b_metadata(
        relabeled,
        topology_id=topology_id,
        split=split,
        train_seed=train_seed,
        sample_idx=sample_idx,
        label_seed=label_seed,
        training_size_budget=training_size_budget,
        train_sample_count=train_sample_count,
        validation_sample_count=validation_sample_count,
        test_sample_count=test_sample_count,
    )
    write_json(output_path, relabeled)
    return {
        "topology_id": topology_id,
        "split": split,
        "train_seed": "" if train_seed is None else int(train_seed),
        "sample_idx": int(sample_idx),
        "label_seed": int(label_seed),
        "path": str(output_path),
        "topology_hash": base_topology_hash,
        "label_hash": edge_label_hash(relabeled),
        "edge_count": edge_count(relabeled),
    }


def materialize_split_samples(
    *,
    base_payload: dict[str, Any],
    base_topology_hash: str,
    output_dir: Path,
    topology_id: str,
    split: str,
    sample_count: int,
    train_seed: int | None,
    training_size_budget: int,
    train_sample_count: int,
    validation_sample_count: int,
    test_sample_count: int,
    epsilon_bar: float,
) -> list[dict[str, Any]]:
    rows = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for sample_idx in range(int(sample_count)):
        label_seed = phase_b_label_seed(
            topology_id,
            split,
            sample_idx=sample_idx,
            train_seed=train_seed,
        )
        rows.append(
            materialize_sample(
                base_payload=base_payload,
                base_topology_hash=base_topology_hash,
                output_path=output_dir / f"G-{sample_idx:06d}.json",
                topology_id=topology_id,
                split=split,
                train_seed=train_seed,
                sample_idx=sample_idx,
                label_seed=label_seed,
                training_size_budget=training_size_budget,
                train_sample_count=train_sample_count,
                validation_sample_count=validation_sample_count,
                test_sample_count=test_sample_count,
                epsilon_bar=epsilon_bar,
            )
        )
    return rows


def prepare_output_dir(output_dir: Path, force: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            raise FileExistsError(f"{output_dir} is not empty; pass --force to overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def materialize_topology_dataset(
    *,
    topology_id: str,
    source_graph_path: str | Path,
    output_dir: str | Path,
    topology_template_path: str | Path | None = None,
    train_seed_start: int = DEFAULT_TRAIN_SEED_START,
    train_seed_count: int = DEFAULT_TRAIN_SEED_COUNT,
    training_size_budget: int = DEFAULT_TRAINING_SIZE_BUDGET,
    test_size: int = DEFAULT_TEST_SIZE,
    epsilon_bar: float = DEFAULT_EPSILON_BAR,
    force: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    source_graph_path = Path(source_graph_path)
    prepare_output_dir(output_dir, force=force)

    train_sample_count, validation_sample_count = phase_b_train_validation_counts(
        training_size_budget
    )
    test_sample_count = int(test_size)
    if test_sample_count <= 0:
        raise ValueError("test_size must be positive")

    base_payload = read_json(source_graph_path)
    base_topology_hash = topology_hash(base_payload)
    template_metadata = load_topology_template_metadata(topology_template_path)

    sample_rows: list[dict[str, Any]] = []
    sample_rows.extend(
        materialize_split_samples(
            base_payload=base_payload,
            base_topology_hash=base_topology_hash,
            output_dir=output_dir / "validation",
            topology_id=topology_id,
            split="validation",
            sample_count=validation_sample_count,
            train_seed=None,
            training_size_budget=training_size_budget,
            train_sample_count=train_sample_count,
            validation_sample_count=validation_sample_count,
            test_sample_count=test_sample_count,
            epsilon_bar=epsilon_bar,
        )
    )
    sample_rows.extend(
        materialize_split_samples(
            base_payload=base_payload,
            base_topology_hash=base_topology_hash,
            output_dir=output_dir / "test",
            topology_id=topology_id,
            split="test",
            sample_count=test_sample_count,
            train_seed=None,
            training_size_budget=training_size_budget,
            train_sample_count=train_sample_count,
            validation_sample_count=validation_sample_count,
            test_sample_count=test_sample_count,
            epsilon_bar=epsilon_bar,
        )
    )

    train_seed_values = [
        int(train_seed_start) + offset for offset in range(int(train_seed_count))
    ]
    for train_seed in train_seed_values:
        sample_rows.extend(
            materialize_split_samples(
                base_payload=base_payload,
                base_topology_hash=base_topology_hash,
                output_dir=output_dir / f"train_seed={train_seed:06d}" / "train",
                topology_id=topology_id,
                split="train",
                sample_count=train_sample_count,
                train_seed=train_seed,
                training_size_budget=training_size_budget,
                train_sample_count=train_sample_count,
                validation_sample_count=validation_sample_count,
                test_sample_count=test_sample_count,
                epsilon_bar=epsilon_bar,
            )
        )

    write_csv(output_dir / "samples.csv", sample_rows, SAMPLE_MANIFEST_FIELDS)
    manifest = {
        "generated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "topology_id": str(topology_id),
        "source_graph_path": str(source_graph_path),
        "output_dir": str(output_dir),
        "topology_hash": base_topology_hash,
        "processed_topology_hash": base_topology_hash,
        **template_metadata,
        "edge_count": edge_count(base_payload),
        "label_regime": "Step2c poly d8 multiplicative eps050",
        "epsilon_bar": float(epsilon_bar),
        "train_validation_ratio": "4:1",
        "training_size_budget": int(training_size_budget),
        "train_sample_count": int(train_sample_count),
        "validation_sample_count": int(validation_sample_count),
        "test_sample_count": int(test_sample_count),
        "train_seed_start": int(train_seed_start),
        "train_seed_count": int(train_seed_count),
        "train_seed_stop_inclusive": train_seed_values[-1] if train_seed_values else None,
        "num_sample_files": len(sample_rows),
        "sample_manifest_path": str(output_dir / "samples.csv"),
    }
    write_json(output_dir / "dataset_manifest.json", manifest)
    return manifest


def read_csv_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def selected_topology_ids(selection_csv: Path, requested_ids: list[str] | None = None) -> list[str]:
    if requested_ids:
        return list(dict.fromkeys(str(value) for value in requested_ids))
    rows = read_csv_rows(selection_csv)
    if not rows:
        raise ValueError(f"No selected topology rows found in {selection_csv}")
    return [str(row["topology_id"]) for row in rows]


def source_graph_for_topology(processed_dir: Path, topology_id: str) -> Path:
    path = processed_dir / f"{topology_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing processed graph for {topology_id}: {path}")
    return path


def topology_template_for_topology(topology_dir: Path | None, topology_id: str) -> Path | None:
    if topology_dir is None:
        return None
    path = Path(topology_dir) / topology_id / "template.json"
    return path if path.exists() else None


def materialize_phase_b_datasets(
    *,
    selection_csv: str | Path = DEFAULT_SELECTION_CSV,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    topology_dir: str | Path | None = DEFAULT_TOPOLOGY_DIR,
    topology_ids: list[str] | None = None,
    max_topologies: int | None = None,
    train_seed_start: int = DEFAULT_TRAIN_SEED_START,
    train_seed_count: int = DEFAULT_TRAIN_SEED_COUNT,
    training_size_budget: int = DEFAULT_TRAINING_SIZE_BUDGET,
    test_size: int = DEFAULT_TEST_SIZE,
    epsilon_bar: float = DEFAULT_EPSILON_BAR,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    selection_csv = Path(selection_csv)
    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir)
    topology_dir = None if topology_dir is None else Path(topology_dir)
    topology_id_list = selected_topology_ids(selection_csv, requested_ids=topology_ids)
    if max_topologies is not None:
        topology_id_list = topology_id_list[: int(max_topologies)]

    train_sample_count, validation_sample_count = phase_b_train_validation_counts(
        training_size_budget
    )
    plan = {
        "generated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "selection_csv": str(selection_csv),
        "processed_dir": str(processed_dir),
        "topology_dir": "" if topology_dir is None else str(topology_dir),
        "output_dir": str(output_dir),
        "num_topologies": len(topology_id_list),
        "topology_ids": topology_id_list,
        "label_regime": "Step2c poly d8 multiplicative eps050",
        "epsilon_bar": float(epsilon_bar),
        "train_validation_ratio": "4:1",
        "training_size_budget": int(training_size_budget),
        "train_sample_count": int(train_sample_count),
        "validation_sample_count": int(validation_sample_count),
        "test_sample_count": int(test_size),
        "train_seed_start": int(train_seed_start),
        "train_seed_count": int(train_seed_count),
        "dry_run": bool(dry_run),
    }
    if dry_run:
        return plan

    output_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []
    for pos, topology_id in enumerate(topology_id_list, start=1):
        source_graph_path = source_graph_for_topology(processed_dir, topology_id)
        topology_template_path = topology_template_for_topology(topology_dir, topology_id)
        topology_output_dir = output_dir / topology_id
        print(
            f"[{pos}/{len(topology_id_list)}] materializing {topology_id} -> {topology_output_dir}",
            flush=True,
        )
        manifest = materialize_topology_dataset(
            topology_id=topology_id,
            source_graph_path=source_graph_path,
            output_dir=topology_output_dir,
            topology_template_path=topology_template_path,
            train_seed_start=train_seed_start,
            train_seed_count=train_seed_count,
            training_size_budget=training_size_budget,
            test_size=test_size,
            epsilon_bar=epsilon_bar,
            force=force,
        )
        index_rows.append(
            {
                "topology_id": topology_id,
                "source_graph_path": str(source_graph_path),
                "output_dir": str(topology_output_dir),
                "train_seed_count": int(train_seed_count),
                "training_size_budget": int(training_size_budget),
                "train_sample_count": int(manifest["train_sample_count"]),
                "validation_sample_count": int(manifest["validation_sample_count"]),
                "test_sample_count": int(manifest["test_sample_count"]),
                "topology_hash": manifest["topology_hash"],
                "topology_bank_hash": manifest["topology_bank_hash"],
                "feasible_set_hash": manifest["feasible_set_hash"],
                "status": "materialized",
            }
        )

    write_csv(output_dir / "phase_b_dataset_index.csv", index_rows, DATASET_INDEX_FIELDS)
    plan.update(
        {
            "dataset_index_path": str(output_dir / "phase_b_dataset_index.csv"),
            "status": "materialized",
        }
    )
    write_json(output_dir / "phase_b_dataset_manifest.json", plan)
    return plan


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-csv", type=Path, default=DEFAULT_SELECTION_CSV)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--topology-dir", type=Path, default=DEFAULT_TOPOLOGY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--topology-id", action="append", default=None)
    parser.add_argument("--max-topologies", type=int, default=None)
    parser.add_argument("--train-seed-start", type=int, default=DEFAULT_TRAIN_SEED_START)
    parser.add_argument("--train-seed-count", type=int, default=DEFAULT_TRAIN_SEED_COUNT)
    parser.add_argument("--training-size", type=int, default=DEFAULT_TRAINING_SIZE_BUDGET)
    parser.add_argument("--test-size", type=int, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--epsilon-bar", type=float, default=DEFAULT_EPSILON_BAR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = materialize_phase_b_datasets(
        selection_csv=args.selection_csv,
        processed_dir=args.processed_dir,
        topology_dir=args.topology_dir,
        output_dir=args.output_dir,
        topology_ids=args.topology_id,
        max_topologies=args.max_topologies,
        train_seed_start=args.train_seed_start,
        train_seed_count=args.train_seed_count,
        training_size_budget=args.training_size,
        test_size=args.test_size,
        epsilon_bar=args.epsilon_bar,
        force=args.force,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
