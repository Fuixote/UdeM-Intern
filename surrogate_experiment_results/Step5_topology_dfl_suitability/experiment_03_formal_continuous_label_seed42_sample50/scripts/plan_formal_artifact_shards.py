#!/usr/bin/env python3
"""Create deterministic balanced topology shards for formal artifact construction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOPOLOGIES = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "Step5_topology_dfl_suitability"
    / "experiment_01_weak_label_seed42_sample50"
    / "configs"
    / "topologies.locked.csv"
)
DEFAULT_OUTPUT_DIR = EXPERIMENT_ROOT / "results" / "formal1000" / "artifact_shards"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def topology_weight(row: dict[str, str]) -> int:
    arcs = int(row.get("num_arcs") or 0)
    candidates = int(row.get("num_feasible_candidates") or 0)
    conflict_edges = int(row.get("candidate_conflict_edges") or 0)
    return max(1, arcs + candidates + conflict_edges // 100)


def build_shard_plan(
    rows: list[dict[str, str]],
    shard_count: int,
) -> list[dict[str, Any]]:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if not rows:
        raise ValueError("topology manifest is empty")
    topology_ids = [str(row.get("topology_id", "")) for row in rows]
    if any(not topology_id for topology_id in topology_ids):
        raise ValueError("every row must have topology_id")
    if len(topology_ids) != len(set(topology_ids)):
        raise ValueError("topology_id values must be unique")

    bins: list[dict[str, Any]] = [
        {"shard_index": index, "estimated_weight": 0, "items": []}
        for index in range(shard_count)
    ]
    weighted = sorted(
        (
            (topology_weight(row), original_index, row)
            for original_index, row in enumerate(rows)
        ),
        key=lambda item: (-item[0], item[1]),
    )
    for weight, original_index, row in weighted:
        target = min(
            bins,
            key=lambda shard: (
                shard["estimated_weight"],
                len(shard["items"]),
                shard["shard_index"],
            ),
        )
        target["estimated_weight"] += weight
        target["items"].append((original_index, row))

    for shard in bins:
        shard["items"].sort(key=lambda item: item[0])
        shard["rows"] = [row for _, row in shard.pop("items")]
        shard["topology_ids"] = [row["topology_id"] for row in shard["rows"]]
        shard["topology_count"] = len(shard["rows"])
    return bins


def atomic_write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        temporary = handle.name
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def create_shards(
    topologies_csv: Path,
    output_dir: Path,
    *,
    shard_count: int,
    expected_count: int,
) -> dict[str, Any]:
    with topologies_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} topologies, observed {len(rows)}")
    shards = build_shard_plan(rows, shard_count)
    shard_summaries = []
    for shard in shards:
        path = output_dir / f"artifact_shard_{shard['shard_index']:02d}.csv"
        atomic_write_csv(path, shard["rows"], fields)
        shard_summaries.append(
            {
                "shard_index": shard["shard_index"],
                "topology_count": shard["topology_count"],
                "estimated_weight": shard["estimated_weight"],
                "topology_ids": shard["topology_ids"],
                "csv_path": str(path),
                "csv_sha256": sha256_file(path),
            }
        )

    flattened = [
        topology_id
        for shard in shard_summaries
        for topology_id in shard["topology_ids"]
    ]
    failures = []
    if len(flattened) != expected_count:
        failures.append("sharded_topology_count_mismatch")
    if len(set(flattened)) != expected_count:
        failures.append("sharded_topologies_not_unique")
    if set(flattened) != {row["topology_id"] for row in rows}:
        failures.append("sharded_topology_set_mismatch")
    summary = {
        "passed": not failures,
        "failures": failures,
        "topologies_csv": str(topologies_csv),
        "topologies_csv_sha256": sha256_file(topologies_csv),
        "expected_topology_count": expected_count,
        "observed_topology_count": len(rows),
        "shard_count": shard_count,
        "min_topologies_per_shard": min(shard["topology_count"] for shard in shard_summaries),
        "max_topologies_per_shard": max(shard["topology_count"] for shard in shard_summaries),
        "min_estimated_weight": min(shard["estimated_weight"] for shard in shard_summaries),
        "max_estimated_weight": max(shard["estimated_weight"] for shard in shard_summaries),
        "shards": shard_summaries,
    }
    atomic_write_json(output_dir / "artifact_shard_plan.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topologies-csv", type=Path, default=DEFAULT_TOPOLOGIES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--shards", type=int, default=16)
    parser.add_argument("--expected-count", type=int, default=1000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = create_shards(
        args.topologies_csv,
        args.output_dir,
        shard_count=args.shards,
        expected_count=args.expected_count,
    )
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "passed",
                    "failures",
                    "observed_topology_count",
                    "shard_count",
                    "min_topologies_per_shard",
                    "max_topologies_per_shard",
                    "min_estimated_weight",
                    "max_estimated_weight",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
