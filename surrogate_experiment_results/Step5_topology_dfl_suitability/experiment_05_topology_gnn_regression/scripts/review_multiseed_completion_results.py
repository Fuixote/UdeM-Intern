#!/usr/bin/env python3
"""Review the 1,880 missing seed-43/44 jobs and export repeat labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

import gnn_data_common as gnn_common


EXP4_SCRIPTS = gnn_common.STEP5_ROOT / "experiment_04_repeat_seed_stability_sample50" / "scripts"
if str(EXP4_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(EXP4_SCRIPTS))

import repeat_seed_common as repeat_common  # noqa: E402


reviewer = repeat_common.base_reviewer()
DEFAULT_TOPOLOGIES = gnn_common.EXPERIMENT_ROOT / "configs" / "multiseed_label_completion940.csv"
DEFAULT_OUTPUT_ROOT = gnn_common.EXPERIMENT_ROOT / "results" / "multiseed_completion1880"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    temporary.replace(path)


def review_completion(rows: list[dict[str, str]], output_root: Path) -> dict[str, Any]:
    job_rows: list[dict[str, Any]] = []
    repeat_labels: list[dict[str, Any]] = []
    failures: list[str] = []
    for topology_index, row in enumerate(rows):
        for seed_index, seed in enumerate(repeat_common.TRAIN_SEEDS):
            job_row, label_row, job_failures = reviewer.review_one_job(
                row,
                manifest_index=topology_index * len(repeat_common.TRAIN_SEEDS) + seed_index,
                output_root=output_root,
                regime=repeat_common.DEFAULT_REGIME,
                protocol=repeat_common.DEFAULT_PROTOCOL,
                data_seed=seed,
                sample_size=repeat_common.SAMPLE_SIZE,
                test_size=repeat_common.TEST_SIZE,
                theta_seed=repeat_common.THETA_SEED,
                gurobi_seed=repeat_common.GUROBI_SEED,
                max_epochs=repeat_common.MAX_EPOCHS,
                metric_stride=repeat_common.METRIC_STRIDE,
                early_stop_patience=repeat_common.EARLY_STOP_PATIENCE,
                early_stop_min_delta=repeat_common.EARLY_STOP_MIN_DELTA,
                threshold=0.1,
                require_early_stop=True,
            )
            job_rows.append(job_row)
            if job_failures or label_row is None:
                failures.append(f"{row['topology_id']}@{seed}:{','.join(job_failures) or 'label_missing'}")
                continue
            if str(label_row["test_hash"]) != str(row["test_hash"]):
                failures.append(f"{row['topology_id']}@{seed}:fixed_test_hash_mismatch")
            repeat_labels.append(
                {
                    "topology_id": row["topology_id"],
                    "train_seed": seed,
                    "normalized_improvement_pp": float(label_row["normalized_improvement_pp"]),
                    "test_hash": label_row["test_hash"],
                    "source": "experiment_05_multiseed_completion",
                }
            )
    expected = len(rows) * len(repeat_common.TRAIN_SEEDS)
    return {
        "integrity_passed": not failures and len(job_rows) == expected and len(repeat_labels) == expected,
        "topology_count": len(rows),
        "expected_job_count": expected,
        "reviewed_job_count": len(job_rows),
        "repeat_label_count": len(repeat_labels),
        "failures": failures,
        "job_rows": job_rows,
        "repeat_labels": repeat_labels,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topologies-csv", type=Path, default=DEFAULT_TOPOLOGIES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    result = review_completion(gnn_common.read_csv(args.topologies_csv), args.output_root)
    output_dir = args.output_dir or args.output_root / "results"
    write_csv(output_dir / "completion_job_metrics.csv", result.pop("job_rows"))
    write_csv(output_dir / "completion_repeat_labels.csv", result.pop("repeat_labels"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "completion_review_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["integrity_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
