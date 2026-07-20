#!/usr/bin/env python3
"""Audit 15 formal GNN runs and build three-seed ensemble predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import tempfile
from typing import Any

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def atomic_write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    temporary.replace(path)


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    output = np.empty(len(values), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        output[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return output


def correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    residual = target - prediction
    denominator = float(np.sum((target - np.mean(target)) ** 2))
    return {
        "count": len(target),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": None if denominator == 0 else 1.0 - float(np.sum(residual**2)) / denominator,
        "pearson": correlation(target, prediction),
        "spearman": correlation(ranks(target), ranks(prediction)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--expected-job-count", type=int, default=15)
    args = parser.parse_args()
    failures: list[str] = []
    jobs = read_csv(args.jobs_csv)
    fold_rows = read_csv(args.folds)
    fold_by_id = {row["topology_id"]: int(row["fold"]) for row in fold_rows}
    if len(jobs) != args.expected_job_count:
        failures.append(f"job_count_mismatch:{len(jobs)}!={args.expected_job_count}")
    if len(fold_by_id) != 1000 or len(fold_rows) != 1000:
        failures.append(f"fold_count_or_uniqueness_mismatch:{len(fold_rows)}/{len(fold_by_id)}")
    all_prediction_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    for job in jobs:
        job_id = job["job_id"]
        fold = int(job["fold"])
        seed = int(job["seed"])
        output_dir = Path(job["output_dir"])
        result_path = output_dir / "run_result.json"
        predictions_path = output_dir / "test_predictions.csv"
        if not result_path.is_file() or not predictions_path.is_file():
            failures.append(f"missing_result:{job_id}")
            continue
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            predictions = read_csv(predictions_path)
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
            failures.append(f"unreadable_result:{job_id}:{type(exc).__name__}")
            continue
        if result.get("status") != "success" or result.get("formal") is not True:
            failures.append(f"unsuccessful_or_nonformal_result:{job_id}")
        if int(result.get("fold", -1)) != fold or int(result.get("seed", -1)) != seed:
            failures.append(f"result_identity_mismatch:{job_id}")
        if result.get("target") != "formal_label_mean_pp":
            failures.append(f"result_target_mismatch:{job_id}")
        topology_ids = [row.get("topology_id", "") for row in predictions]
        if len(predictions) != 200 or len(set(topology_ids)) != 200:
            failures.append(f"prediction_count_or_uniqueness_mismatch:{job_id}:{len(predictions)}/{len(set(topology_ids))}")
        for row in predictions:
            topology_id = row["topology_id"]
            if fold_by_id.get(topology_id) != fold:
                failures.append(f"prediction_fold_mismatch:{job_id}:{topology_id}")
            all_prediction_rows.append(
                {
                    "topology_id": topology_id,
                    "fold": fold,
                    "seed": seed,
                    "target_formal_label_mean_pp": float(row["target_formal_label_mean_pp"]),
                    "prediction_formal_label_mean_pp": float(row["prediction_formal_label_mean_pp"]),
                }
            )
        metric = result.get("test_metrics", {})
        timing = result.get("timing", {})
        run_rows.append(
            {
                "job_id": job_id,
                "fold": fold,
                "seed": seed,
                "epochs_completed": int(result.get("epochs_completed", 0)),
                "early_stop_triggered": bool(result.get("early_stop_triggered")),
                "best_epoch": int(result.get("best_epoch", -1)),
                "best_validation_mae_pp": float(result.get("best_validation_mae_pp", math.nan)),
                "test_mae_pp": float(metric.get("mae", math.nan)),
                "test_rmse_pp": float(metric.get("rmse", math.nan)),
                "test_r2": metric.get("r2"),
                "test_spearman": metric.get("spearman"),
                "training_seconds": float(timing.get("training_seconds", math.nan)),
            }
        )
    if len(all_prediction_rows) != 3000:
        failures.append(f"all_prediction_count_mismatch:{len(all_prediction_rows)}!=3000")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in all_prediction_rows:
        grouped.setdefault(row["topology_id"], []).append(row)
    ensemble_rows = []
    for topology_id in sorted(grouped, key=lambda value: int(value.split("-")[-1])):
        rows = grouped[topology_id]
        seeds = {int(row["seed"]) for row in rows}
        targets = [float(row["target_formal_label_mean_pp"]) for row in rows]
        predictions = [float(row["prediction_formal_label_mean_pp"]) for row in rows]
        if len(rows) != 3 or seeds != {42, 43, 44}:
            failures.append(f"ensemble_seed_coverage_mismatch:{topology_id}:{sorted(seeds)}")
            continue
        if max(targets) - min(targets) > 1e-9:
            failures.append(f"ensemble_target_mismatch:{topology_id}")
            continue
        ensemble_rows.append(
            {
                "topology_id": topology_id,
                "fold": int(rows[0]["fold"]),
                "target_formal_label_mean_pp": targets[0],
                "prediction_seed42_pp": next(float(row["prediction_formal_label_mean_pp"]) for row in rows if int(row["seed"]) == 42),
                "prediction_seed43_pp": next(float(row["prediction_formal_label_mean_pp"]) for row in rows if int(row["seed"]) == 43),
                "prediction_seed44_pp": next(float(row["prediction_formal_label_mean_pp"]) for row in rows if int(row["seed"]) == 44),
                "prediction_three_seed_mean_pp": statistics.fmean(predictions),
                "prediction_three_seed_std_pp": statistics.pstdev(predictions),
            }
        )
    if len(ensemble_rows) != 1000:
        failures.append(f"ensemble_count_mismatch:{len(ensemble_rows)}!=1000")
    metric_rows = []
    ranking: dict[str, Any] = {}
    if ensemble_rows:
        target = np.asarray([row["target_formal_label_mean_pp"] for row in ensemble_rows], dtype=float)
        prediction = np.asarray([row["prediction_three_seed_mean_pp"] for row in ensemble_rows], dtype=float)
        subsets = {
            "all": np.ones(len(target), dtype=bool),
            "nonzero": target != 0,
            "material_abs_gt_0.1pp": np.abs(target) > 0.1,
        }
        for subset, mask in subsets.items():
            metric_rows.append({"model": "gnn_three_seed_mean", "subset": subset, **metrics(target[mask], prediction[mask])})
        top_count = 50
        true_top = set(np.argsort(target)[-top_count:].tolist())
        predicted_top = set(np.argsort(prediction)[-top_count:].tolist())
        ranking = {"top50_overlap": len(true_top & predicted_top) / top_count}
    audit = {
        "passed": not failures,
        "status": "success" if not failures else "failed",
        "target": "formal_label_mean_pp",
        "job_count": len(jobs),
        "successful_run_count": len(run_rows),
        "all_prediction_count": len(all_prediction_rows),
        "ensemble_prediction_count": len(ensemble_rows),
        "seed_set": [42, 43, 44],
        "fold_set": list(range(5)),
        "metrics": metric_rows,
        "ranking": ranking,
        "run_summary": {
            "early_stop_count": sum(bool(row["early_stop_triggered"]) for row in run_rows),
            "epoch_cap_count": sum(not bool(row["early_stop_triggered"]) for row in run_rows),
            "epochs_completed": [row["epochs_completed"] for row in run_rows],
            "total_training_seconds": sum(row["training_seconds"] for row in run_rows if math.isfinite(row["training_seconds"])),
        },
        "failures": failures,
    }
    review_dir = args.output_root / "review"
    atomic_write_csv(
        review_dir / "all_test_predictions.csv",
        all_prediction_rows,
        ["topology_id", "fold", "seed", "target_formal_label_mean_pp", "prediction_formal_label_mean_pp"],
    )
    atomic_write_csv(
        review_dir / "ensemble_test_predictions.csv",
        ensemble_rows,
        [
            "topology_id",
            "fold",
            "target_formal_label_mean_pp",
            "prediction_seed42_pp",
            "prediction_seed43_pp",
            "prediction_seed44_pp",
            "prediction_three_seed_mean_pp",
            "prediction_three_seed_std_pp",
        ],
    )
    atomic_write_csv(
        review_dir / "run_metrics.csv",
        run_rows,
        [
            "job_id",
            "fold",
            "seed",
            "epochs_completed",
            "early_stop_triggered",
            "best_epoch",
            "best_validation_mae_pp",
            "test_mae_pp",
            "test_rmse_pp",
            "test_r2",
            "test_spearman",
            "training_seconds",
        ],
    )
    atomic_write_json(review_dir / "formal_gnn_review.audit.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
