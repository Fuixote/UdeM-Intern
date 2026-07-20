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


def parse_cap_hit(value: str) -> tuple[str, int]:
    topology_id, separator, raw_seed = str(value).rpartition("@")
    if not separator or not topology_id or not raw_seed:
        raise argparse.ArgumentTypeError("cap hit must use TOPOLOGY_ID@TRAIN_SEED")
    try:
        seed = int(raw_seed)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("cap-hit train seed must be an integer") from exc
    return topology_id, seed


def validate_accepted_spoplus_cap_hit(
    job_row: dict[str, Any],
    *,
    topology_id: str,
    seed: int,
    accepted_cap_hits: set[tuple[str, int]],
) -> tuple[bool, str | None]:
    if (topology_id, seed) not in accepted_cap_hits:
        return False, None
    if any(
        job_row.get(field) != "success"
        for field in ("status", "two_stage_status", "spoplus_status", "evaluation_status")
    ):
        return False, "accepted_spoplus_cap_hit_status_not_success"
    record_path = Path(job_row["job_dir"]) / "spoplus" / "metrics" / "early_stopping.json"
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"accepted_spoplus_cap_hit_record_invalid:{exc}"
    expected = {
        "enabled": True,
        "should_stop": False,
        "metric": "validation_spoplus_loss",
        "max_epochs": repeat_common.MAX_EPOCHS,
        "stopped_epoch": repeat_common.MAX_EPOCHS,
    }
    mismatches = [
        f"{field}={record.get(field)}!={expected_value}"
        for field, expected_value in expected.items()
        if record.get(field) != expected_value
    ]
    if mismatches:
        return False, "accepted_spoplus_cap_hit_mismatch:" + ",".join(mismatches)
    return True, None


def review_completion(
    rows: list[dict[str, str]],
    output_root: Path,
    *,
    accepted_spoplus_cap_hits: set[tuple[str, int]] | None = None,
) -> dict[str, Any]:
    accepted_spoplus_cap_hits = accepted_spoplus_cap_hits or set()
    accepted_cap_hits_used: list[dict[str, Any]] = []
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
                require_early_stop=False,
            )
            job_failures = list(job_failures)
            if job_row.get("two_stage_early_stop_triggered") is not True:
                job_failures.append("2stage_early_stop_not_triggered")
            if job_row.get("spoplus_early_stop_triggered") is not True:
                accepted, acceptance_failure = validate_accepted_spoplus_cap_hit(
                    job_row,
                    topology_id=row["topology_id"],
                    seed=seed,
                    accepted_cap_hits=accepted_spoplus_cap_hits,
                )
                if accepted:
                    job_row["spoplus_cap_hit_accepted"] = True
                    accepted_cap_hits_used.append(
                        {
                            "topology_id": row["topology_id"],
                            "train_seed": seed,
                            "stopped_epoch": repeat_common.MAX_EPOCHS,
                            "reason": "user_approved_decision_plateau_or_valid_cap_checkpoint",
                        }
                    )
                else:
                    job_failures.append(acceptance_failure or "spoplus_early_stop_not_triggered")
            else:
                job_row["spoplus_cap_hit_accepted"] = False
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
    used_keys = {(row["topology_id"], int(row["train_seed"])) for row in accepted_cap_hits_used}
    for topology_id, seed in sorted(accepted_spoplus_cap_hits - used_keys):
        failures.append(f"unused_accepted_spoplus_cap_hit:{topology_id}@{seed}")
    expected = len(rows) * len(repeat_common.TRAIN_SEEDS)
    return {
        "integrity_passed": not failures and len(job_rows) == expected and len(repeat_labels) == expected,
        "topology_count": len(rows),
        "expected_job_count": expected,
        "reviewed_job_count": len(job_rows),
        "repeat_label_count": len(repeat_labels),
        "accepted_spoplus_cap_hits": accepted_cap_hits_used,
        "failures": failures,
        "job_rows": job_rows,
        "repeat_labels": repeat_labels,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topologies-csv", type=Path, default=DEFAULT_TOPOLOGIES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--accept-spoplus-cap-hit",
        action="append",
        default=[],
        type=parse_cap_hit,
        metavar="TOPOLOGY_ID@TRAIN_SEED",
    )
    args = parser.parse_args()
    result = review_completion(
        gnn_common.read_csv(args.topologies_csv),
        args.output_root,
        accepted_spoplus_cap_hits=set(args.accept_spoplus_cap_hit),
    )
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
