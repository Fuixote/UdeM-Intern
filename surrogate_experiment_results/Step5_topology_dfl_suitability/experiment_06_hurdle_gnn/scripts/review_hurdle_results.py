#!/usr/bin/env python3
"""Audit Experiment 06 classifier/regressor jobs and rank variants."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import hurdle_common as common


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier-jobs", type=Path, required=True)
    parser.add_argument("--regressor-jobs", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-classifier-jobs", type=int, required=True)
    parser.add_argument("--expected-regressor-jobs", type=int, required=True)
    args = parser.parse_args()
    classifier_jobs = common.read_csv(args.classifier_jobs)
    regressor_jobs = common.read_csv(args.regressor_jobs)
    failures: list[str] = []
    classifier_rows = []
    metric_rows = []
    run_rows = []
    if len(classifier_jobs) != args.expected_classifier_jobs:
        failures.append(f"classifier_job_count_mismatch:{len(classifier_jobs)}!={args.expected_classifier_jobs}")
    if len(regressor_jobs) != args.expected_regressor_jobs:
        failures.append(f"regressor_job_count_mismatch:{len(regressor_jobs)}!={args.expected_regressor_jobs}")
    for job in classifier_jobs:
        result_path = Path(job["output_dir"]) / "run_result.json"
        prediction_path = Path(job["output_dir"]) / "test_predictions.csv"
        if not result_path.is_file() or not prediction_path.is_file():
            failures.append(f"missing_classifier_result:{job['job_id']}")
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        predictions = common.read_csv(prediction_path)
        if result.get("status") != "success" or result.get("task") != "zero_nonzero_classifier":
            failures.append(f"invalid_classifier_result:{job['job_id']}")
        if len(predictions) != 200 or len({row["topology_id"] for row in predictions}) != 200:
            failures.append(f"classifier_prediction_mismatch:{job['job_id']}")
        test = result.get("test_metrics", {})
        classifier_rows.append(
            {
                "job_id": job["job_id"],
                "fold": int(job["fold"]),
                "seed": int(job["seed"]),
                "epochs_completed": int(result.get("epochs_completed", 0)),
                "best_epoch": int(result.get("best_epoch", -1)),
                "accuracy": test.get("accuracy"),
                "balanced_accuracy": test.get("balanced_accuracy"),
                "precision": test.get("precision"),
                "recall": test.get("recall"),
                "f1": test.get("f1"),
                "auroc": test.get("auroc"),
                "average_precision": test.get("average_precision"),
            }
        )
    expected_matrix = {
        (int(job["fold"]), int(job["seed"]), subset, objective)
        for job in classifier_jobs
        for subset in common.REGRESSION_SUBSETS
        for objective in common.OBJECTIVES
    }
    observed_matrix = {
        (int(job["fold"]), int(job["seed"]), job["regression_subset"], job["objective"])
        for job in regressor_jobs
    }
    if observed_matrix != expected_matrix:
        failures.append("regressor_matrix_mismatch")
    for job in regressor_jobs:
        result_path = Path(job["output_dir"]) / "run_result.json"
        prediction_path = Path(job["output_dir"]) / "test_predictions.csv"
        if not result_path.is_file() or not prediction_path.is_file():
            failures.append(f"missing_regressor_result:{job['job_id']}")
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        predictions = common.read_csv(prediction_path)
        if result.get("status") != "success" or result.get("task") != "subset_regressor":
            failures.append(f"invalid_regressor_result:{job['job_id']}")
        if result.get("regression_subset") != job["regression_subset"] or result.get("objective") != job["objective"]:
            failures.append(f"regressor_identity_mismatch:{job['job_id']}")
        if len(predictions) != 200 or len({row["topology_id"] for row in predictions}) != 200:
            failures.append(f"regressor_prediction_mismatch:{job['job_id']}")
        run_rows.append(
            {
                "job_id": job["job_id"],
                "fold": int(job["fold"]),
                "seed": int(job["seed"]),
                "regression_subset": job["regression_subset"],
                "objective": job["objective"],
                "epochs_completed": int(result.get("epochs_completed", 0)),
                "best_epoch": int(result.get("best_epoch", -1)),
                "best_validation_subset_mae_pp": float(result.get("best_validation_subset_mae_pp", math.nan)),
                "training_seconds": float(result.get("timing", {}).get("training_seconds", math.nan)),
            }
        )
        for metric in result.get("test_metrics", []):
            metric_rows.append(
                {
                    "job_id": job["job_id"],
                    "fold": int(job["fold"]),
                    "seed": int(job["seed"]),
                    "regression_subset": job["regression_subset"],
                    "objective": job["objective"],
                    **metric,
                }
            )
    for row in run_rows:
        if not math.isfinite(row["best_validation_subset_mae_pp"]):
            failures.append(f"nonfinite_validation_metric:{row['job_id']}")
    ranking_rows = [
        row
        for row in metric_rows
        if row["subset"] in {"all", "material_abs_gt_0.1pp", "nonzero"}
        and row["prediction_mode"] in {"hard_hurdle", "soft_hurdle", "oracle_nonzero_gate"}
    ]
    ranking_rows.sort(key=lambda row: (row["subset"], row["prediction_mode"], float(row["mae"])))
    best_by_target = {}
    for row in ranking_rows:
        key = f"{row['prediction_mode']}:{row['subset']}"
        best_by_target.setdefault(
            key,
            {
                "regression_subset": row["regression_subset"],
                "objective": row["objective"],
                "mae": row["mae"],
                "rmse": row["rmse"],
                "r2": row["r2"],
                "spearman": row["spearman"],
            },
        )
    review_dir = args.output_root / "review"
    common.atomic_write_csv(
        review_dir / "classifier_metrics.csv",
        classifier_rows,
        [
            "job_id",
            "fold",
            "seed",
            "epochs_completed",
            "best_epoch",
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "auroc",
            "average_precision",
        ],
    )
    common.atomic_write_csv(
        review_dir / "regressor_runs.csv",
        run_rows,
        [
            "job_id",
            "fold",
            "seed",
            "regression_subset",
            "objective",
            "epochs_completed",
            "best_epoch",
            "best_validation_subset_mae_pp",
            "training_seconds",
        ],
    )
    metric_fields = [
        "job_id",
        "fold",
        "seed",
        "regression_subset",
        "objective",
        "prediction_mode",
        "subset",
        "count",
        "mae",
        "rmse",
        "r2",
        "pearson",
        "spearman",
    ]
    common.atomic_write_csv(review_dir / "variant_metrics.csv", metric_rows, metric_fields)
    audit = {
        "passed": not failures,
        "status": "success" if not failures else "failed",
        "classifier_job_count": len(classifier_jobs),
        "successful_classifier_count": len(classifier_rows),
        "regressor_job_count": len(regressor_jobs),
        "successful_regressor_count": len(run_rows),
        "variant_metric_count": len(metric_rows),
        "regression_subsets": list(common.REGRESSION_SUBSETS),
        "objectives": list(common.OBJECTIVES),
        "best_by_prediction_mode_and_subset": best_by_target,
        "failures": failures,
    }
    common.atomic_write_json(review_dir / "hurdle_review.audit.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
