#!/usr/bin/env python3
"""Create an audited dry-run plan over two repeat train seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import repeat_seed_common as common


planner = common.base_planner()


def combined_plan(
    rows: list[dict[str, str]],
    artifact_root: Path,
    job_output_root: Path,
    *,
    expected_job_count: int = 120,
    strict: bool = True,
    python_bin: str | None = None,
) -> dict:
    jobs = []
    failures: list[str] = []
    per_seed = []
    for seed in common.TRAIN_SEEDS:
        plan = planner.build_plan(
            rows,
            output_root=artifact_root,
            job_output_root=job_output_root,
            regime=common.DEFAULT_REGIME,
            protocol=common.DEFAULT_PROTOCOL,
            data_seed=seed,
            sample_size=common.SAMPLE_SIZE,
            test_size=common.TEST_SIZE,
            theta_seed=common.THETA_SEED,
            gurobi_seed=common.GUROBI_SEED,
            max_epochs=common.MAX_EPOCHS,
            metric_stride=common.METRIC_STRIDE,
            early_stop_patience=common.EARLY_STOP_PATIENCE,
            early_stop_min_delta=common.EARLY_STOP_MIN_DELTA,
            python_bin=python_bin,
            strict=strict,
        )
        per_seed.append({"train_seed": seed, "passed": plan["passed"], "job_count": plan["job_count"], "ready_count": plan["ready_count"]})
        failures.extend(f"seed{seed}:{failure}" for failure in plan["failures"])
        jobs.extend(plan["jobs"])
    for index, job in enumerate(jobs):
        job["manifest_index"] = index
    job_ids = [job["job_id"] for job in jobs]
    if len(job_ids) != len(set(job_ids)):
        failures.append("duplicate_job_ids")
    if len(jobs) != int(expected_job_count):
        failures.append(f"job_count_mismatch:{len(jobs)}!={int(expected_job_count)}")
    if strict and any(job["status"] != "ready" for job in jobs):
        failures.append("not_all_jobs_ready")
    all_ready = all(job["status"] == "ready" for job in jobs)
    return {
        "passed": not failures,
        "status": "ready" if all_ready and not failures else "planned",
        "experiment": common.EXPERIMENT_VERSION,
        "topology_count": len(rows),
        "train_seeds": list(common.TRAIN_SEEDS),
        "job_count": len(jobs),
        "expected_job_count": int(expected_job_count),
        "ready_count": sum(job["status"] == "ready" for job in jobs),
        "fixed_test_bank": True,
        "reference_test_seed": common.REFERENCE_SEED,
        "sample_size": common.SAMPLE_SIZE,
        "training_size": common.TRAINING_SIZE,
        "validation_size": common.VALIDATION_SIZE,
        "test_size": common.TEST_SIZE,
        "max_epochs": common.MAX_EPOCHS,
        "early_stop_patience": common.EARLY_STOP_PATIENCE,
        "commands_are_dry_run_only": True,
        "artifact_root": str(artifact_root),
        "job_output_root": str(job_output_root),
        "per_seed": per_seed,
        "failures": failures,
        "jobs": jobs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topologies-csv", type=Path, default=common.DEFAULT_SELECTED_TOPOLOGIES)
    parser.add_argument("--artifact-root", type=Path, default=common.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--job-output-root", type=Path, default=common.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--plan-output", type=Path)
    parser.add_argument("--jobs-csv-output", type=Path)
    parser.add_argument("--python", default=None)
    parser.add_argument("--allow-missing-artifacts", action="store_true")
    parser.add_argument("--expected-job-count", type=int, default=120)
    args = parser.parse_args()
    rows = common.read_csv(args.topologies_csv)
    plan = combined_plan(
        rows,
        args.artifact_root,
        args.job_output_root,
        expected_job_count=args.expected_job_count,
        strict=not args.allow_missing_artifacts,
        python_bin=args.python,
    )
    plan_output = args.plan_output or args.job_output_root / "plans" / "repeat_seed120_plan.json"
    csv_output = args.jobs_csv_output or args.job_output_root / "plans" / "repeat_seed120_jobs.csv"
    planner.common.atomic_write_json(plan_output, plan)
    planner.atomic_write_csv(csv_output, plan["jobs"])
    print(json.dumps({
        "passed": plan["passed"],
        "status": plan["status"],
        "topology_count": plan["topology_count"],
        "job_count": plan["job_count"],
        "ready_count": plan["ready_count"],
        "commands_are_dry_run_only": plan["commands_are_dry_run_only"],
        "plan_output": str(plan_output),
        "jobs_csv_output": str(csv_output),
        "failures": plan["failures"],
    }, indent=2, sort_keys=True))
    return 0 if plan["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
