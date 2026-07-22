#!/usr/bin/env python3
"""Plan the preregistered seed-42 five-fold helpful incidence screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys
from typing import Any

import material_common as common
import train_incidence_classifier as three_class_gnn


TRAINER = common.EXPERIMENT_ROOT / "scripts" / "train_helpful_incidence_classifier.py"


def build_jobs(args: argparse.Namespace, ready: bool) -> list[dict[str, Any]]:
    jobs = []
    for fold in args.selected_folds:
        for seed in args.selected_seeds:
            job_id = f"helpful_incidence_fold{fold}_seed{seed}"
            output_dir = args.output_root / "folds" / f"fold{fold}" / f"seed{seed}"
            command = [
                args.python,
                str(TRAINER),
                "--graph-jsonl", str(args.graph_jsonl),
                "--labels", str(args.labels),
                "--folds", str(args.folds),
                "--label-audit", str(args.label_audit),
                "--fold-audit", str(args.fold_audit),
                "--output-dir", str(output_dir),
                "--fold", str(fold),
                "--seed", str(seed),
                "--hidden-dim", "64",
                "--layers", "3",
                "--dropout", "0.1",
                "--batch-size", "32",
                "--learning-rate", "0.001",
                "--weight-decay", "0.0001",
                "--max-epochs", str(args.max_epochs),
                "--early-stop-patience", str(args.early_stop_patience),
                "--early-stop-min-delta", "0.0001",
                "--regret-compute-cost-pp", "0.0",
                "--precision-constraints", "0.4,0.5",
                "--compute-costs-pp", "0,0.05,0.1,0.25,0.5",
                "--threads", str(args.threads),
                "--formal",
            ]
            jobs.append(
                {
                    "manifest_index": len(jobs),
                    "job_id": job_id,
                    "fold": fold,
                    "validation_fold": (fold + 1) % 5,
                    "seed": seed,
                    "status": "ready" if ready else "blocked",
                    "output_dir": str(output_dir),
                    "log_path": str(args.output_root / "logs" / "jobs" / f"{job_id}.log"),
                    "threads": args.threads,
                    "command_preview": shlex.join(command),
                }
            )
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-jsonl", type=Path, default=common.DEFAULT_INCIDENCE_GRAPHS)
    parser.add_argument("--labels", type=Path, default=common.DEFAULT_OUTPUT_ROOT / "labels" / "material_labels.csv")
    parser.add_argument("--folds", type=Path, default=common.DEFAULT_OUTPUT_ROOT / "splits" / "material_folds.csv")
    parser.add_argument("--label-audit", type=Path, default=common.DEFAULT_OUTPUT_ROOT / "labels" / "material_labels.audit.json")
    parser.add_argument("--fold-audit", type=Path, default=common.DEFAULT_OUTPUT_ROOT / "splits" / "material_folds.audit.json")
    parser.add_argument("--output-root", type=Path, default=common.DEFAULT_OUTPUT_ROOT / "incidence_gnn_helpful" / "formal_seed42")
    parser.add_argument("--plan-output", type=Path)
    parser.add_argument("--jobs-output", type=Path)
    parser.add_argument("--fold", action="append", type=int, dest="selected_folds")
    parser.add_argument("--seed", action="append", type=int, dest="selected_seeds")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--early-stop-patience", type=int, default=30)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    args.selected_folds = sorted(set(args.selected_folds or range(5)))
    args.selected_seeds = sorted(set(args.selected_seeds or (42,)))
    if any(fold not in range(5) for fold in args.selected_folds):
        raise ValueError("folds must lie in [0,4]")
    graph_rows = common.read_jsonl(args.graph_jsonl)
    label_rows = common.read_csv(args.labels)
    fold_rows = common.read_csv(args.folds)
    input_audit = three_class_gnn.audit_inputs(
        graph_rows,
        label_rows,
        fold_rows,
        graph_path=args.graph_jsonl,
        label_path=args.labels,
        fold_path=args.folds,
        label_audit_path=args.label_audit,
        fold_audit_path=args.fold_audit,
        label_field="primary_label",
    )
    jobs = build_jobs(args, input_audit["passed"])
    failures = list(input_audit["failures"])
    if len(jobs) != len(args.selected_folds) * len(args.selected_seeds):
        failures.append("job_count_mismatch")
    if len({job["job_id"] for job in jobs}) != len(jobs):
        failures.append("duplicate_job_ids")
    if any("--execute" in shlex.split(job["command_preview"]) for job in jobs):
        failures.append("execute_flag_in_preview_plan")
    plan = {
        "passed": not failures,
        "status": "ready" if not failures else "blocked",
        "experiment": "step5_exp07_helpful_incidence_seed42_screen",
        "task": "material_helpful_vs_non_helpful",
        "selected_folds": args.selected_folds,
        "selected_seeds": args.selected_seeds,
        "job_count": len(jobs),
        "commands_are_preview_only": True,
        "input_audit": input_audit,
        "protocol": {
            "outer_split": "five_disjoint_test_folds",
            "inner_split": "next_fold_is_validation_600_200_200",
            "checkpoint": "minimum_validation_weighted_bce",
            "calibration": "validation_only_temperature_scaling",
            "primary_threshold": "validation_only_regret_optimal_cost_0pp",
            "secondary_thresholds": ["precision_at_least_0.4", "precision_at_least_0.5"],
            "primary_seed_gate": 42,
            "followup_seeds_only_after_gate": [43, 44],
        },
        "failures": failures,
        "jobs": jobs,
    }
    plan_output = args.plan_output or args.output_root / "plans" / "helpful_incidence_plan.json"
    jobs_output = args.jobs_output or args.output_root / "plans" / "helpful_incidence_jobs.csv"
    common.atomic_write_json(plan_output, plan)
    common.atomic_write_csv(jobs_output, jobs)
    print(json.dumps({
        "passed": plan["passed"],
        "status": plan["status"],
        "job_count": len(jobs),
        "plan_output": str(plan_output),
        "jobs_output": str(jobs_output),
        "failures": failures,
    }, indent=2, sort_keys=True))
    return 0 if plan["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
