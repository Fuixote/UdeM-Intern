#!/usr/bin/env python3
"""Plan one dry-run paired 2stage/SPO+ job per Step5 topology."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
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


RUN_ONE_JOB = STEP3_SCRIPTS / "run_one_job.py"
DEFAULT_TOPOLOGIES = EXPERIMENT_ROOT / "configs" / "topologies.locked.csv"
DEFAULT_MAX_EPOCHS = 1500
DEFAULT_METRIC_STRIDE = 1
DEFAULT_EARLY_STOP_PATIENCE = 20
DEFAULT_EARLY_STOP_MIN_DELTA = 0.0001
JOB_FIELDS = [
    "job_id",
    "topology_id",
    "manifest_index",
    "regime",
    "protocol",
    "data_seed",
    "trainer_train_seed_arg",
    "sample_size",
    "training_size",
    "validation_size",
    "trainer_train_size_arg",
    "test_size",
    "train_bank_path",
    "eval_manifest_path",
    "validation_path",
    "test_path",
    "output_dir",
    "expected_training_hash",
    "validation_hash",
    "test_hash",
    "theta_seed",
    "gurobi_seed",
    "max_epochs",
    "metric_stride",
    "early_stop_patience",
    "early_stop_min_delta",
    "weak_label",
    "weak_label_threshold",
    "runtime_class",
    "run_one_job_command",
    "status",
    "readiness_failures",
]


def atomic_write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=JOB_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        temp_name = handle.name
    os.replace(temp_name, output)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _training_hash(train_bank_path: Path, training_size: int) -> str | None:
    if not train_bank_path.is_file():
        return None
    dataset = common.read_npz_dataset(train_bank_path)
    return dataset["manifest"].get("prefix_hashes", {}).get(str(int(training_size)))


def _resolve_eval_path(
    eval_manifest_path: Path,
    raw_path: str | Path,
    *,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path

    manifest_relative = eval_manifest_path.parent / path
    if manifest_relative.is_file():
        return manifest_relative

    project_relative = project_root / path
    if project_relative.is_file():
        return project_relative

    # Preserve the manifest-relative convention for useful missing-file errors.
    return manifest_relative


def readiness_failures(
    *,
    topology_id: str,
    regime: str,
    protocol: str,
    data_seed: int,
    sample_size: int,
    training_size: int,
    validation_size: int,
    test_size: int,
    train_bank_path: Path,
    eval_manifest_path: Path,
    eval_manifest: dict[str, Any],
    training_hash: str | None,
) -> list[str]:
    failures: list[str] = []
    if not train_bank_path.is_file():
        failures.append("train_bank_missing")
    if not eval_manifest_path.is_file():
        failures.append("eval_manifest_missing")
    if not training_hash:
        failures.append("training_hash_missing")
    expected = {
        "topology_id": str(topology_id),
        "regime": str(regime),
        "protocol": str(protocol),
        "data_seed": int(data_seed),
        "sample_size": int(sample_size),
        "training_size": int(training_size),
        "validation_size": int(validation_size),
        "trainer_train_size_arg": int(training_size),
    }
    for field, value in expected.items():
        observed = eval_manifest.get(field)
        if field in {"data_seed", "sample_size", "training_size", "validation_size", "trainer_train_size_arg"}:
            try:
                observed = int(observed)
            except (TypeError, ValueError):
                observed = None
        else:
            observed = str(observed or "")
        if observed != value:
            failures.append(f"eval_{field}_mismatch")
    for field in ("validation_path", "validation_hash", "test_path", "test_hash"):
        if eval_manifest.get(field) in (None, ""):
            failures.append(f"{field}_missing")
    if eval_manifest.get("validation_path"):
        validation_path = _resolve_eval_path(eval_manifest_path, eval_manifest["validation_path"])
        if not validation_path.is_file():
            failures.append("validation_npz_missing")
    if eval_manifest.get("test_path"):
        test_path = _resolve_eval_path(eval_manifest_path, eval_manifest["test_path"])
        if not test_path.is_file():
            failures.append("test_npz_missing")
        else:
            try:
                observed_test_size = int(common.read_npz_dataset(test_path)["manifest"].get("sample_count", -1))
            except Exception:
                failures.append("test_npz_invalid")
            else:
                if observed_test_size != int(test_size):
                    failures.append("test_size_mismatch")
    protocol_record = eval_manifest.get("step5_protocol", {})
    if protocol_record.get("weak_label") is not True:
        failures.append("weak_label_protocol_missing")
    return failures


def build_run_one_job_command(job: dict[str, Any], *, python_bin: str | None = None) -> str:
    command = [
        python_bin or os.environ.get("KEP_PYTHON") or sys.executable,
        str(RUN_ONE_JOB),
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
        str(job["trainer_train_seed_arg"]),
        "--train-size",
        str(job["trainer_train_size_arg"]),
        "--sample-size",
        str(job["sample_size"]),
        "--theta-seed",
        str(job["theta_seed"]),
        "--gurobi-seed",
        str(job["gurobi_seed"]),
        "--max-epochs",
        str(job["max_epochs"]),
        "--metric-stride",
        str(job["metric_stride"]),
        "--early-stop-patience",
        str(job["early_stop_patience"]),
        "--early-stop-min-delta",
        str(job["early_stop_min_delta"]),
        "--output-dir",
        str(job["output_dir"]),
        "--dry-run",
    ]
    return shlex.join(command)


def build_plan(
    topology_rows: list[dict[str, str]],
    *,
    output_root: str | Path,
    regime: str = builder.DEFAULT_REGIME,
    protocol: str = "screen",
    data_seed: int = builder.DEFAULT_DATA_SEED,
    sample_size: int = builder.DEFAULT_SAMPLE_SIZE,
    test_size: int = builder.DEFAULT_TEST_SIZE,
    theta_seed: int = 42,
    gurobi_seed: int = 42,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    metric_stride: int = DEFAULT_METRIC_STRIDE,
    early_stop_patience: int = DEFAULT_EARLY_STOP_PATIENCE,
    early_stop_min_delta: float = DEFAULT_EARLY_STOP_MIN_DELTA,
    long_topologies: set[str] | None = None,
    python_bin: str | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    output_root = Path(output_root)
    if int(max_epochs) <= 0:
        raise ValueError("max_epochs must be positive")
    if int(metric_stride) <= 0:
        raise ValueError("metric_stride must be positive")
    if int(early_stop_patience) <= 0:
        raise ValueError("early_stop_patience must be positive")
    if float(early_stop_min_delta) < 0:
        raise ValueError("early_stop_min_delta must be non-negative")
    training_size, validation_size = builder.validate_sample_size(sample_size)
    long_topologies = long_topologies or set()
    jobs: list[dict[str, Any]] = []
    for manifest_index, topology_row in enumerate(topology_rows):
        topology_id = str(topology_row["topology_id"])
        paths = builder.artifact_paths(
            output_root,
            regime=regime,
            topology_id=topology_id,
            data_seed=data_seed,
            sample_size=sample_size,
        )
        eval_manifest = _read_json(paths["eval_manifest"])
        training_hash = _training_hash(paths["train_bank"], training_size)
        failures = readiness_failures(
            topology_id=topology_id,
            regime=regime,
            protocol=protocol,
            data_seed=data_seed,
            sample_size=sample_size,
            training_size=training_size,
            validation_size=validation_size,
            test_size=test_size,
            train_bank_path=paths["train_bank"],
            eval_manifest_path=paths["eval_manifest"],
            eval_manifest=eval_manifest,
            training_hash=training_hash,
        )
        if strict and failures:
            raise ValueError(f"artifact_not_ready {topology_id}:" + ",".join(failures))
        job = {
            "job_id": (
                f"{topology_id}|data_seed={int(data_seed):06d}|"
                f"sample_size={int(sample_size):03d}|training={training_size:03d}|"
                f"validation={validation_size:03d}"
            ),
            "topology_id": topology_id,
            "manifest_index": int(manifest_index),
            "regime": str(regime),
            "protocol": str(protocol),
            "data_seed": int(data_seed),
            "trainer_train_seed_arg": int(data_seed),
            "sample_size": int(sample_size),
            "training_size": training_size,
            "validation_size": validation_size,
            "trainer_train_size_arg": training_size,
            "test_size": int(test_size),
            "train_bank_path": str(paths["train_bank"]),
            "eval_manifest_path": str(paths["eval_manifest"]),
            "validation_path": str(eval_manifest.get("validation_path", "")),
            "test_path": str(eval_manifest.get("test_path", "")),
            "output_dir": str(
                output_root
                / "jobs"
                / str(regime)
                / topology_id
                / f"data_seed={int(data_seed):06d}"
                / f"sample_size={int(sample_size):03d}"
            ),
            "expected_training_hash": training_hash or "",
            "validation_hash": eval_manifest.get("validation_hash", ""),
            "test_hash": eval_manifest.get("test_hash", ""),
            "theta_seed": int(theta_seed),
            "gurobi_seed": int(gurobi_seed),
            "max_epochs": int(max_epochs),
            "metric_stride": int(metric_stride),
            "early_stop_patience": int(early_stop_patience),
            "early_stop_min_delta": float(early_stop_min_delta),
            "weak_label": True,
            "weak_label_threshold": 0.1,
            "runtime_class": "long" if topology_id in long_topologies else "normal",
            "status": "ready" if not failures else "planned",
            "readiness_failures": ";".join(failures),
        }
        job["run_one_job_command"] = build_run_one_job_command(job, python_bin=python_bin)
        command_tokens = shlex.split(job["run_one_job_command"])
        if "--dry-run" not in command_tokens or "--execute" in command_tokens:
            raise AssertionError("planner produced an executable command")
        jobs.append(job)

    duplicate_job_ids = sorted(
        job_id
        for job_id in {job["job_id"] for job in jobs}
        if sum(row["job_id"] == job_id for row in jobs) > 1
    )
    failures: list[str] = []
    if duplicate_job_ids:
        failures.append("duplicate_job_ids")
    if strict and any(job["status"] != "ready" for job in jobs):
        failures.append("not_all_jobs_ready")
    return {
        "passed": not failures,
        "failures": failures,
        "status": "ready" if all(job["status"] == "ready" for job in jobs) else "planned",
        "job_count": len(jobs),
        "ready_count": sum(job["status"] == "ready" for job in jobs),
        "topology_count": len(topology_rows),
        "data_seed": int(data_seed),
        "sample_size": int(sample_size),
        "training_size": training_size,
        "validation_size": validation_size,
        "test_size": int(test_size),
        "theta_seed": int(theta_seed),
        "gurobi_seed": int(gurobi_seed),
        "protocol": str(protocol),
        "regime": str(regime),
        "commands_are_dry_run_only": True,
        "jobs": jobs,
    }


def parse_csv_set(raw: str | None) -> set[str]:
    return {part.strip() for part in str(raw or "").split(",") if part.strip()}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topologies-csv", type=Path, default=DEFAULT_TOPOLOGIES)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--regime", default=builder.DEFAULT_REGIME)
    parser.add_argument("--protocol", choices=("screen", "confirm"), default="screen")
    parser.add_argument("--data-seed", type=int, default=builder.DEFAULT_DATA_SEED)
    parser.add_argument("--sample-size", type=int, default=builder.DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--test-size", type=int, default=builder.DEFAULT_TEST_SIZE)
    parser.add_argument("--theta-seed", type=int, default=42)
    parser.add_argument("--gurobi-seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--metric-stride", type=int, default=DEFAULT_METRIC_STRIDE)
    parser.add_argument("--early-stop-patience", type=int, default=DEFAULT_EARLY_STOP_PATIENCE)
    parser.add_argument("--early-stop-min-delta", type=float, default=DEFAULT_EARLY_STOP_MIN_DELTA)
    parser.add_argument("--long-topologies", default="")
    parser.add_argument("--python", default=None)
    parser.add_argument("--topology-id", action="append", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--allow-missing-artifacts", action="store_true")
    parser.add_argument("--plan-output", type=Path, default=None)
    parser.add_argument("--jobs-csv-output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = builder.selected_topology_rows(
        args.topologies_csv,
        topology_ids=args.topology_id,
        limit=args.limit,
    )
    plan = build_plan(
        rows,
        output_root=args.output_root,
        regime=args.regime,
        protocol=args.protocol,
        data_seed=args.data_seed,
        sample_size=args.sample_size,
        test_size=args.test_size,
        theta_seed=args.theta_seed,
        gurobi_seed=args.gurobi_seed,
        max_epochs=args.max_epochs,
        metric_stride=args.metric_stride,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        long_topologies=parse_csv_set(args.long_topologies),
        python_bin=args.python,
        strict=not args.allow_missing_artifacts,
    )
    plan_output = args.plan_output or Path(args.output_root) / "plans" / "weak_label_plan.json"
    jobs_output = args.jobs_csv_output or Path(args.output_root) / "plans" / "weak_label_jobs.csv"
    common.atomic_write_json(plan_output, plan)
    atomic_write_csv(jobs_output, plan["jobs"])
    print(
        json.dumps(
            {
                "passed": plan["passed"],
                "status": plan["status"],
                "job_count": plan["job_count"],
                "ready_count": plan["ready_count"],
                "topology_count": plan["topology_count"],
                "commands_are_dry_run_only": plan["commands_are_dry_run_only"],
                "plan_output": str(plan_output),
                "jobs_csv_output": str(jobs_output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if plan["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
