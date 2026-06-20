#!/usr/bin/env python3
"""Run a tiny Phase-B' full-(X,y) screening pilot in scratch space."""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import audit_fixed_topology_xy  # noqa: E402
import build_fixed_eval_sets  # noqa: E402
import build_nested_train_bank  # noqa: E402
import fixed_topology_xy_common as common  # noqa: E402
import plan_full_xy_screening as planner  # noqa: E402
import run_one_job  # noqa: E402
import sample_fixed_topology_context as context_sampler  # noqa: E402


DEFAULT_OUTPUT_ROOT = Path("/tmp/step3_full_xy_screening_pilot")
JOB_FIELDS = [
    *planner.JOB_FIELDS,
    "dry_run_status",
    "audit_status",
    "execute_status",
]


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def validate_scratch_output_root(output_root: str | Path) -> Path:
    path = Path(output_root).expanduser().resolve()
    allowed = [Path("/tmp").resolve(), Path("/local1").resolve()]
    if not any(path == root or root in path.parents for root in allowed):
        raise ValueError("pilot output_root must be under /tmp or /local1 scratch")
    return path


def parse_prefix_sizes(raw: str | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(raw, str):
        return [int(part) for part in raw.split(",") if part]
    return [int(part) for part in raw]


def run_one_job_args(job: dict[str, Any], *, max_epochs: int, metric_stride: int, early_stop_patience: int) -> list[str]:
    return [
        "--train-bank",
        str(job["train_bank_path"]),
        "--eval-manifest",
        str(job["eval_manifest_path"]),
        "--topology-id",
        str(job["topology_id"]),
        "--regime",
        str(job["regime"]),
        "--protocol",
        str(job["protocol"]),
        "--train-seed",
        str(job["train_seed"]),
        "--train-size",
        str(job["train_size"]),
        "--output-dir",
        str(job["output_dir"]),
        "--max-epochs",
        str(int(max_epochs)),
        "--metric-stride",
        str(int(metric_stride)),
        "--early-stop-patience",
        str(int(early_stop_patience)),
    ]


def materialize_one_topology(
    *,
    topology_id: str,
    topology_template_path: Path,
    base_payload_path: Path,
    output_root: Path,
    regime: str,
    train_seed_start: int,
    train_seed_count: int,
    max_train_size: int,
    prefix_sizes: list[int],
    validation_size: int,
    test_size: int,
    protocol: str,
    experiment_version: str,
    master_label_seed: int,
    generator_config: dict[str, Any],
) -> tuple[dict[int, Path], dict[str, Any], dict[str, Any], dict[str, Any]]:
    template = json.loads(topology_template_path.read_text(encoding="utf-8"))
    base_payload = json.loads(base_payload_path.read_text(encoding="utf-8"))
    data_dir = output_root / "data" / str(regime) / topology_id
    train_banks: dict[int, Path] = {}
    for train_seed in range(int(train_seed_start), int(train_seed_start) + int(train_seed_count)):
        bank_path = data_dir / "train_banks" / f"train_seed={int(train_seed):06d}.npz"
        build_nested_train_bank.build_nested_train_bank(
            topology_template=template,
            base_payload=base_payload,
            output_path=bank_path,
            topology_id=topology_id,
            regime=regime,
            train_seed=int(train_seed),
            max_train_size=max_train_size,
            prefix_sizes=prefix_sizes,
            experiment_version=experiment_version,
            master_label_seed=master_label_seed,
            generator_config=generator_config,
            protocol=protocol,
        )
        train_banks[int(train_seed)] = bank_path
    eval_result = build_fixed_eval_sets.build_fixed_eval_sets_for_topology(
        topology_template=template,
        base_payload=base_payload,
        output_dir=data_dir / "eval",
        topology_id=topology_id,
        regime=regime,
        validation_size=validation_size,
        test_size=test_size,
        experiment_version=experiment_version,
        master_label_seed=master_label_seed,
        generator_config=generator_config,
        protocol=protocol,
    )
    return train_banks, eval_result, template, base_payload


def run_pilot(
    *,
    selected_topologies_csv: str | Path = planner.DEFAULT_SELECTED_TOPOLOGIES_CSV,
    topology_root: str | Path = planner.DEFAULT_TOPOLOGY_ROOT,
    base_payload_dir: str | Path = planner.DEFAULT_BASE_PAYLOAD_DIR,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    regime: str = planner.DEFAULT_REGIME,
    train_seed_start: int = 1,
    train_seed_count: int = 1,
    train_size: int = 2,
    max_train_size: int = 5,
    prefix_sizes: list[int] | tuple[int, ...] = (2, 3, 5),
    validation_size: int = 2,
    test_size: int = 2,
    max_topologies: int = 3,
    max_epochs: int = 2,
    metric_stride: int = 1,
    early_stop_patience: int = 2,
    protocol: str = "screen",
    context_generator_config: str | Path | None = None,
    execute_one: bool = False,
    experiment_version: str = "phase_b_prime_full_xy_screening_pilot_v1",
    master_label_seed: int = 20260619,
) -> dict[str, Any]:
    if context_generator_config is None:
        raise ValueError("context_generator_config is required")
    output_root = validate_scratch_output_root(output_root)
    prefix_sizes = parse_prefix_sizes(prefix_sizes)
    generator_config = context_sampler.load_generator_config(context_generator_config)
    selected_rows = planner.selected_topology_rows(selected_topologies_csv, max_topologies=max_topologies)
    jobs: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for row in selected_rows:
        topology_id = str(row["topology_id"])
        template_path = planner.resolve_topology_template_path(topology_root, topology_id)
        base_payload_path = planner.resolve_base_payload_path(base_payload_dir, topology_id)
        if not template_path.exists():
            raise FileNotFoundError(f"Missing topology template for {topology_id}: {template_path}")
        if not base_payload_path.exists():
            raise FileNotFoundError(f"Missing base payload for {topology_id}: {base_payload_path}")
        train_banks, eval_result, template, base_payload = materialize_one_topology(
            topology_id=topology_id,
            topology_template_path=template_path,
            base_payload_path=base_payload_path,
            output_root=output_root,
            regime=regime,
            train_seed_start=train_seed_start,
            train_seed_count=train_seed_count,
            max_train_size=max_train_size,
            prefix_sizes=prefix_sizes,
            validation_size=validation_size,
            test_size=test_size,
            protocol=protocol,
            experiment_version=experiment_version,
            master_label_seed=master_label_seed,
            generator_config=generator_config,
        )
        for train_seed, train_bank_path in train_banks.items():
            audit_result = audit_fixed_topology_xy.audit_fixed_topology_xy(
                train_bank_path=train_bank_path,
                eval_manifest_path=eval_result["eval_manifest_path"],
                topology_template=template,
                base_payload=base_payload,
                generator_config=generator_config,
                protocol=protocol,
            )
            audit_rows.append(
                {
                    "topology_id": topology_id,
                    "train_seed": int(train_seed),
                    "passed": bool(audit_result["passed"]),
                    "failures": audit_result["failures"],
                    "train_bank_path": str(train_bank_path),
                    "eval_manifest_path": str(eval_result["eval_manifest_path"]),
                }
            )
            output_dir = (
                output_root
                / "jobs"
                / str(regime)
                / topology_id
                / f"train_seed={int(train_seed):06d}"
                / f"train_size={int(train_size)}"
            )
            job = {
                "topology_id": topology_id,
                "regime": str(regime),
                "protocol": protocol,
                "train_seed": int(train_seed),
                "train_size": int(train_size),
                "topology_template_path": str(template_path),
                "base_payload_path": str(base_payload_path),
                "train_bank_path": str(train_bank_path),
                "eval_manifest_path": str(eval_result["eval_manifest_path"]),
                "validation_path": str(eval_result["validation_path"]),
                "test_path": str(eval_result["test_path"]),
                "output_dir": str(output_dir),
                "expected_train_prefix_hash": common.read_npz_dataset(train_bank_path)["manifest"]["prefix_hashes"].get(str(int(train_size))),
                "validation_hash": eval_result["validation_hash"],
                "test_hash": eval_result["test_hash"],
                "audit_status": "passed" if audit_result["passed"] else "failed",
                "dry_run_status": "not_run",
                "execute_status": "not_executed",
            }
            job["run_one_job_command"] = planner.build_run_one_job_command(job)
            jobs.append(job)
    for job in jobs:
        with contextlib.redirect_stdout(io.StringIO()):
            rc = run_one_job.main(
                [
                    *run_one_job_args(
                        job,
                        max_epochs=max_epochs,
                        metric_stride=metric_stride,
                        early_stop_patience=early_stop_patience,
                    ),
                    "--dry-run",
                ]
            )
        job["dry_run_status"] = "ready" if rc == 0 else f"failed:{rc}"
    executed_job_count = 0
    if execute_one and jobs:
        target = jobs[0]
        with contextlib.redirect_stdout(io.StringIO()):
            rc = run_one_job.main(
                [
                    *run_one_job_args(
                        target,
                        max_epochs=max_epochs,
                        metric_stride=metric_stride,
                        early_stop_patience=early_stop_patience,
                    ),
                    "--execute",
                ]
            )
        target["execute_status"] = "success" if rc == 0 else f"failed:{rc}"
        executed_job_count = 1
    audit_path = output_root / "audit_results.json"
    common.atomic_write_json(audit_path, audit_rows)
    write_csv(output_root / "pilot_jobs.csv", jobs, JOB_FIELDS)
    summary = {
        "status": "success" if all(row["passed"] for row in audit_rows) and all(job["dry_run_status"] == "ready" for job in jobs) else "failed",
        "protocol": protocol,
        "output_root": str(output_root),
        "topology_ids": [str(row["topology_id"]) for row in selected_rows],
        "topology_count": len(selected_rows),
        "train_seed_count": int(train_seed_count),
        "train_size": int(train_size),
        "max_train_size": int(max_train_size),
        "prefix_sizes": prefix_sizes,
        "validation_size": int(validation_size),
        "test_size": int(test_size),
        "max_epochs": int(max_epochs),
        "job_count": len(jobs),
        "execute_one": bool(execute_one),
        "executed_job_count": int(executed_job_count),
        "audit_passed": all(row["passed"] for row in audit_rows),
    }
    common.atomic_write_json(output_root / "pilot_summary.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-topologies-csv", type=Path, default=planner.DEFAULT_SELECTED_TOPOLOGIES_CSV)
    parser.add_argument("--topology-root", type=Path, default=planner.DEFAULT_TOPOLOGY_ROOT)
    parser.add_argument("--base-payload-dir", type=Path, default=planner.DEFAULT_BASE_PAYLOAD_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--regime", default=planner.DEFAULT_REGIME)
    parser.add_argument("--train-seed-start", type=int, default=1)
    parser.add_argument("--train-seed-count", type=int, default=1)
    parser.add_argument("--train-size", type=int, default=2)
    parser.add_argument("--max-train-size", type=int, default=5)
    parser.add_argument("--prefix-sizes", default="2,3,5")
    parser.add_argument("--validation-size", type=int, default=2)
    parser.add_argument("--test-size", type=int, default=2)
    parser.add_argument("--max-topologies", type=int, default=3)
    parser.add_argument("--max-epochs", type=int, default=2)
    parser.add_argument("--metric-stride", type=int, default=1)
    parser.add_argument("--early-stop-patience", type=int, default=2)
    parser.add_argument("--protocol", choices=("screen", "confirm"), default="screen")
    parser.add_argument("--context-generator-config", type=Path, required=True)
    parser.add_argument("--execute-one", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_pilot(
        selected_topologies_csv=args.selected_topologies_csv,
        topology_root=args.topology_root,
        base_payload_dir=args.base_payload_dir,
        output_root=args.output_root,
        regime=args.regime,
        train_seed_start=args.train_seed_start,
        train_seed_count=args.train_seed_count,
        train_size=args.train_size,
        max_train_size=args.max_train_size,
        prefix_sizes=parse_prefix_sizes(args.prefix_sizes),
        validation_size=args.validation_size,
        test_size=args.test_size,
        max_topologies=args.max_topologies,
        max_epochs=args.max_epochs,
        metric_stride=args.metric_stride,
        early_stop_patience=args.early_stop_patience,
        protocol=args.protocol,
        context_generator_config=args.context_generator_config,
        execute_one=args.execute_one,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
