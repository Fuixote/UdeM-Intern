#!/usr/bin/env python3
"""Audit Experiment 06 classifier/regressor jobs and rank variants."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import hurdle_common as common


PREDICTION_COLUMNS = {
    "raw_regressor": "raw_regression_prediction_pp",
    "hard_hurdle": "hard_hurdle_prediction_pp",
    "soft_hurdle": "soft_hurdle_prediction_pp",
    "oracle_nonzero_gate": "oracle_nonzero_gate_prediction_pp",
}


def aggregate_classifier_predictions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["seed"])].append(row)
    output = []
    for seed, seed_rows in sorted(grouped.items()):
        target = np.asarray([int(row["target_is_nonzero"]) for row in seed_rows], dtype=int)
        probability = np.asarray([float(row["probability_nonzero"]) for row in seed_rows], dtype=float)
        metrics = common.binary_metrics(target, probability, threshold=0.5)
        confusion = metrics.pop("confusion")
        output.append(
            {
                "seed": seed,
                "fold_count": len({int(row["fold"]) for row in seed_rows}),
                **metrics,
                **confusion,
            }
        )
    return output


def aggregate_regressor_predictions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["seed"]), row["regression_subset"], row["objective"])].append(row)
    output = []
    for (seed, regression_subset, objective), variant_rows in sorted(grouped.items()):
        target = np.asarray([float(row["target_formal_label_mean_pp"]) for row in variant_rows], dtype=float)
        masks = {
            "all": np.ones(len(target), dtype=bool),
            "zero": target == 0.0,
            "nonzero": target != 0.0,
            "material_abs_gt_0.1pp": np.abs(target) > 0.1,
        }
        fold_count = len({int(row["fold"]) for row in variant_rows})
        for prediction_mode, column in PREDICTION_COLUMNS.items():
            prediction = np.asarray([float(row[column]) for row in variant_rows], dtype=float)
            for subset, mask in masks.items():
                output.append(
                    {
                        "seed": seed,
                        "fold_count": fold_count,
                        "regression_subset": regression_subset,
                        "objective": objective,
                        "prediction_mode": prediction_mode,
                        "subset": subset,
                        **common.regression_metrics(target[mask], prediction[mask]),
                    }
                )
    return output


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
    classifier_prediction_rows = []
    metric_rows = []
    run_rows = []
    regressor_prediction_rows = []
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
        classifier_prediction_rows.extend(predictions)
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
        regressor_prediction_rows.extend(predictions)
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
    expected_folds_by_seed: dict[int, set[int]] = defaultdict(set)
    for job in classifier_jobs:
        expected_folds_by_seed[int(job["seed"])].add(int(job["fold"]))
    for seed, folds in expected_folds_by_seed.items():
        classifier_seed_rows = [row for row in classifier_prediction_rows if int(row["seed"]) == seed]
        expected_count = 200 * len(folds)
        if len(classifier_seed_rows) != expected_count:
            failures.append(f"classifier_oof_count_mismatch:seed{seed}:{len(classifier_seed_rows)}!={expected_count}")
        if len({row["topology_id"] for row in classifier_seed_rows}) != expected_count:
            failures.append(f"classifier_oof_topology_mismatch:seed{seed}")
        for regression_subset in common.REGRESSION_SUBSETS:
            for objective in common.OBJECTIVES:
                variant_rows = [
                    row
                    for row in regressor_prediction_rows
                    if int(row["seed"]) == seed
                    and row["regression_subset"] == regression_subset
                    and row["objective"] == objective
                ]
                if len(variant_rows) != expected_count:
                    failures.append(
                        f"regressor_oof_count_mismatch:seed{seed}:{regression_subset}:{objective}:"
                        f"{len(variant_rows)}!={expected_count}"
                    )
                if len({row["topology_id"] for row in variant_rows}) != expected_count:
                    failures.append(f"regressor_oof_topology_mismatch:seed{seed}:{regression_subset}:{objective}")
    classifier_oof_rows = aggregate_classifier_predictions(classifier_prediction_rows)
    aggregate_metric_rows = aggregate_regressor_predictions(regressor_prediction_rows)
    ranking_rows = [
        row
        for row in aggregate_metric_rows
        if row["subset"] in {"all", "material_abs_gt_0.1pp", "nonzero"}
        and row["prediction_mode"] in {"hard_hurdle", "soft_hurdle", "oracle_nonzero_gate"}
    ]
    ranking_rows.sort(key=lambda row: (row["seed"], row["subset"], row["prediction_mode"], float(row["mae"])))
    best_by_target = {}
    seed_count = len({row["seed"] for row in aggregate_metric_rows})
    for row in ranking_rows:
        prefix = f"seed{row['seed']}:" if seed_count > 1 else ""
        key = f"{prefix}{row['prediction_mode']}:{row['subset']}"
        best_by_target.setdefault(
            key,
            {
                "seed": row["seed"],
                "fold_count": row["fold_count"],
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
        review_dir / "classifier_oof_metrics.csv",
        classifier_oof_rows,
        [
            "seed",
            "fold_count",
            "count",
            "positive_count",
            "negative_count",
            "threshold",
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "specificity",
            "f1",
            "auroc",
            "average_precision",
            "tp",
            "tn",
            "fp",
            "fn",
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
    aggregate_metric_fields = [
        "seed",
        "fold_count",
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
    common.atomic_write_csv(
        review_dir / "aggregate_variant_metrics.csv",
        aggregate_metric_rows,
        aggregate_metric_fields,
    )
    audit = {
        "passed": not failures,
        "status": "success" if not failures else "failed",
        "classifier_job_count": len(classifier_jobs),
        "successful_classifier_count": len(classifier_rows),
        "regressor_job_count": len(regressor_jobs),
        "successful_regressor_count": len(run_rows),
        "variant_metric_count": len(metric_rows),
        "classifier_oof_metric_count": len(classifier_oof_rows),
        "aggregate_variant_metric_count": len(aggregate_metric_rows),
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
