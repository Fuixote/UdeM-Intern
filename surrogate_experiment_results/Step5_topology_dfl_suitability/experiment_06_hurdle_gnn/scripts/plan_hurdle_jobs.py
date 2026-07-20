#!/usr/bin/env python3
"""Create dependency-aware classifier and regressor plans for Experiment 06."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shlex
import sys
import tempfile
from typing import Any

import hurdle_common as common


CLASSIFIER = common.EXPERIMENT_ROOT / "scripts" / "train_hurdle_classifier.py"
REGRESSOR = common.EXPERIMENT_ROOT / "scripts" / "train_hurdle_regressor.py"


def atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    temporary.replace(path)


def locked_common_args(args: argparse.Namespace) -> list[str]:
    return [
        "--graph-jsonl",
        str(args.graph_jsonl),
        "--folds",
        str(args.folds),
        "--hidden-dim",
        "64",
        "--dropout",
        "0.1",
        "--batch-size",
        "32",
        "--learning-rate",
        "0.001",
        "--weight-decay",
        "0.0001",
        "--max-epochs",
        str(args.max_epochs),
        "--early-stop-patience",
        str(args.early_stop_patience),
        "--early-stop-min-delta",
        "0.0001",
        "--threads",
        str(args.threads),
    ]


def build_jobs(args: argparse.Namespace, input_ready: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    classifier_jobs = []
    regressor_jobs = []
    common_args = locked_common_args(args)
    for fold in args.selected_folds:
        for seed in args.selected_seeds:
            classifier_id = f"classifier_fold{fold}_seed{seed}"
            classifier_output = args.output_root / "classifiers" / f"fold{fold}" / f"seed{seed}"
            classifier_command = [
                args.python,
                str(CLASSIFIER),
                *common_args,
                "--output-dir",
                str(classifier_output),
                "--fold",
                str(fold),
                "--seed",
                str(seed),
                "--threshold",
                "0.5",
            ]
            classifier_jobs.append(
                {
                    "manifest_index": len(classifier_jobs),
                    "job_id": classifier_id,
                    "stage": "classifier",
                    "fold": fold,
                    "validation_fold": (fold + 1) % 5,
                    "seed": seed,
                    "regression_subset": "",
                    "objective": "class_balanced_bce",
                    "dependency_job_id": "",
                    "status": "ready" if input_ready else "blocked",
                    "output_dir": str(classifier_output),
                    "log_path": str(args.output_root / "logs" / "jobs" / f"{classifier_id}.log"),
                    "threads": args.threads,
                    "command_preview": shlex.join(classifier_command),
                }
            )
            classifier_predictions = classifier_output / "test_predictions.csv"
            for subset in common.REGRESSION_SUBSETS:
                for objective in common.OBJECTIVES:
                    job_id = f"regressor_{subset}_{objective}_fold{fold}_seed{seed}"
                    output_dir = args.output_root / "regressors" / subset / objective / f"fold{fold}" / f"seed{seed}"
                    command = [
                        args.python,
                        str(REGRESSOR),
                        *common_args,
                        "--classifier-predictions",
                        str(classifier_predictions),
                        "--output-dir",
                        str(output_dir),
                        "--fold",
                        str(fold),
                        "--seed",
                        str(seed),
                        "--regression-subset",
                        subset,
                        "--objective",
                        objective,
                        "--ranking-weight",
                        "0.25",
                        "--classifier-threshold",
                        "0.5",
                    ]
                    regressor_jobs.append(
                        {
                            "manifest_index": len(regressor_jobs),
                            "job_id": job_id,
                            "stage": "regressor",
                            "fold": fold,
                            "validation_fold": (fold + 1) % 5,
                            "seed": seed,
                            "regression_subset": subset,
                            "objective": objective,
                            "dependency_job_id": classifier_id,
                            "dependency_path": str(classifier_predictions),
                            "status": "ready_after_classifier" if input_ready else "blocked",
                            "output_dir": str(output_dir),
                            "log_path": str(args.output_root / "logs" / "jobs" / f"{job_id}.log"),
                            "threads": args.threads,
                            "command_preview": shlex.join(command),
                        }
                    )
    return classifier_jobs, regressor_jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-jsonl", type=Path, default=common.DEFAULT_GRAPHS)
    parser.add_argument("--folds", type=Path, default=common.DEFAULT_FOLDS)
    parser.add_argument("--output-root", type=Path, default=common.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--plan-output", type=Path)
    parser.add_argument("--classifier-csv-output", type=Path)
    parser.add_argument("--regressor-csv-output", type=Path)
    parser.add_argument("--fold", action="append", type=int, dest="selected_folds")
    parser.add_argument("--seed", action="append", type=int, dest="selected_seeds")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--early-stop-patience", type=int, default=30)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--expected-classifier-jobs", type=int)
    parser.add_argument("--expected-regressor-jobs", type=int)
    args = parser.parse_args()
    args.selected_folds = sorted(set(args.selected_folds or range(5)))
    args.selected_seeds = sorted(set(args.selected_seeds or (42, 43, 44)))
    if any(fold not in range(5) for fold in args.selected_folds):
        raise ValueError("folds must be in [0,4]")
    if args.threads < 1 or args.max_epochs < 1 or args.early_stop_patience < 1:
        raise ValueError("threads, epochs, and patience must be positive")
    graph_rows = common.read_jsonl(args.graph_jsonl)
    fold_rows = common.read_csv(args.folds)
    input_audit = common.audit_inputs(graph_rows, fold_rows, args.graph_jsonl, args.folds)
    classifier_jobs, regressor_jobs = build_jobs(args, input_audit["passed"])
    failures = list(input_audit["failures"])
    expected_classifier = args.expected_classifier_jobs or len(args.selected_folds) * len(args.selected_seeds)
    expected_regressor = args.expected_regressor_jobs or expected_classifier * len(common.REGRESSION_SUBSETS) * len(common.OBJECTIVES)
    if len(classifier_jobs) != expected_classifier:
        failures.append(f"classifier_job_count_mismatch:{len(classifier_jobs)}!={expected_classifier}")
    if len(regressor_jobs) != expected_regressor:
        failures.append(f"regressor_job_count_mismatch:{len(regressor_jobs)}!={expected_regressor}")
    all_jobs = classifier_jobs + regressor_jobs
    if len({row["job_id"] for row in all_jobs}) != len(all_jobs):
        failures.append("duplicate_job_ids")
    if any("--execute" in shlex.split(row["command_preview"]) for row in all_jobs):
        failures.append("execute_flag_in_preview_plan")
    passed = not failures
    plan = {
        "passed": passed,
        "status": "ready" if passed else "blocked",
        "experiment": "step5_exp6_hurdle_gnn_v1",
        "target": "formal_label_mean_pp",
        "selected_folds": args.selected_folds,
        "selected_seeds": args.selected_seeds,
        "classifier_job_count": len(classifier_jobs),
        "regressor_job_count": len(regressor_jobs),
        "regression_subsets": list(common.REGRESSION_SUBSETS),
        "objectives": list(common.OBJECTIVES),
        "commands_are_preview_only": True,
        "input_audit": input_audit,
        "protocol": {
            "classifier_objective": "class_balanced_bce",
            "classifier_threshold": 0.5,
            "regression_subset_material_threshold_pp": 0.1,
            "weighted_huber_formula": "(1 + min(abs(y)/train_q90_abs,1))/train_mean_weight",
            "signed_log_formula": "sign(y)*log1p(abs(y))",
            "ranking_objective": "huber + 0.25*pairwise_logistic_ranking",
            "selection_metric_classifier": "validation_weighted_bce",
            "selection_metric_regressor": "validation_subset_mae_pp",
            "prediction_modes": ["raw_regressor", "hard_hurdle", "soft_hurdle", "oracle_nonzero_gate"],
            "max_epochs": args.max_epochs,
            "early_stop_patience": args.early_stop_patience,
            "early_stop_min_delta": 0.0001,
            "threads_per_job": args.threads,
        },
        "failures": failures,
        "classifier_jobs": classifier_jobs,
        "regressor_jobs": regressor_jobs,
    }
    plan_output = args.plan_output or args.output_root / "plans" / "hurdle_plan.json"
    classifier_output = args.classifier_csv_output or args.output_root / "plans" / "classifier_jobs.csv"
    regressor_output = args.regressor_csv_output or args.output_root / "plans" / "regressor_jobs.csv"
    common.atomic_write_json(plan_output, plan)
    atomic_write_csv(classifier_output, classifier_jobs)
    atomic_write_csv(regressor_output, regressor_jobs)
    print(
        json.dumps(
            {
                "passed": passed,
                "status": plan["status"],
                "classifier_job_count": len(classifier_jobs),
                "regressor_job_count": len(regressor_jobs),
                "plan_output": str(plan_output),
                "classifier_csv_output": str(classifier_output),
                "regressor_csv_output": str(regressor_output),
                "input_audit": input_audit,
                "failures": failures,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
