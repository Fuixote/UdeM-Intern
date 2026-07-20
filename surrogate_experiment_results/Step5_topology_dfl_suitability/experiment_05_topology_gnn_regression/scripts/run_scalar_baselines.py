#!/usr/bin/env python3
"""Run leakage-controlled out-of-fold scalar baselines for Experiment 05."""

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

import gnn_data_common as common


def pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


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


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    residual = y - prediction
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "count": len(y),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "r2": None if denominator == 0 else 1.0 - float(np.sum(residual ** 2)) / denominator,
        "pearson": pearson(y, prediction),
        "spearman": pearson(ranks(y), ranks(prediction)),
    }


def ridge_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, alpha: float) -> np.ndarray:
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale == 0] = 1.0
    train_scaled = (train_x - mean) / scale
    test_scaled = (test_x - mean) / scale
    design = np.column_stack([np.ones(len(train_scaled)), train_scaled])
    test_design = np.column_stack([np.ones(len(test_scaled)), test_scaled])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.pinv(design.T @ design + penalty) @ design.T @ train_y
    return test_design @ coefficients


def run(rows: list[dict[str, str]], fold_rows: list[dict[str, str]], alpha: float = 10.0) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fold_by_id = {row["topology_id"]: int(row["fold"]) for row in fold_rows}
    if set(fold_by_id) != {row["topology_id"] for row in rows}:
        raise ValueError("fold topology set differs from formal summary")
    x = np.asarray([[float(row[field]) for field in common.SCALAR_FEATURES] for row in rows], dtype=float)
    y = np.asarray([float(row["normalized_improvement_pp"]) for row in rows], dtype=float)
    folds = np.asarray([fold_by_id[row["topology_id"]] for row in rows], dtype=int)
    predictions = {
        "zero": np.zeros(len(rows), dtype=float),
        "fold_mean": np.zeros(len(rows), dtype=float),
        "ridge": np.zeros(len(rows), dtype=float),
    }
    for fold in sorted(set(folds.tolist())):
        train_mask = folds != fold
        test_mask = folds == fold
        predictions["fold_mean"][test_mask] = float(np.mean(y[train_mask]))
        predictions["ridge"][test_mask] = ridge_predict(x[train_mask], y[train_mask], x[test_mask], alpha)
    prediction_rows = []
    for index, row in enumerate(rows):
        prediction_rows.append({
            "topology_id": row["topology_id"],
            "fold": int(folds[index]),
            "target_normalized_improvement_pp": float(y[index]),
            "zero_prediction": float(predictions["zero"][index]),
            "fold_mean_prediction": float(predictions["fold_mean"][index]),
            "ridge_prediction": float(predictions["ridge"][index]),
        })
    subsets = {
        "all": np.ones(len(y), dtype=bool),
        "nonzero": y != 0,
        "material": np.abs(y) > 0.1,
    }
    metric_rows = []
    for model, prediction in predictions.items():
        for subset, mask in subsets.items():
            metric_rows.append({"model": model, "subset": subset, **metrics(y[mask], prediction[mask])})
    top_count = 50
    true_top = set(np.argsort(y)[-top_count:].tolist())
    ranking = {}
    for model, prediction in predictions.items():
        predicted_top = set(np.argsort(prediction)[-top_count:].tolist())
        ranking[model] = {"top50_overlap": len(true_top & predicted_top) / top_count}
    audit = {
        "passed": len(rows) == 1000 and len(prediction_rows) == 1000,
        "sample_count": len(rows),
        "fold_count": len(set(folds.tolist())),
        "features": common.SCALAR_FEATURES,
        "target_is_not_an_input_feature": "normalized_improvement_pp" not in common.SCALAR_FEATURES,
        "ridge_alpha": alpha,
        "metrics": metric_rows,
        "ranking": ranking,
    }
    return prediction_rows, audit


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-summary", type=Path, default=common.DEFAULT_FORMAL_SUMMARY)
    parser.add_argument("--folds", type=Path, default=common.DEFAULT_OUTPUT_ROOT / "splits" / "folds.csv")
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--prediction-output", type=Path, default=common.DEFAULT_OUTPUT_ROOT / "baselines" / "scalar_oof_predictions.csv")
    parser.add_argument("--audit-output", type=Path, default=common.DEFAULT_OUTPUT_ROOT / "baselines" / "scalar_baselines.audit.json")
    args = parser.parse_args()
    predictions, result = run(common.read_csv(args.formal_summary), common.read_csv(args.folds), args.alpha)
    write_csv(args.prediction_output, predictions)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**result, "prediction_output": str(args.prediction_output), "audit_output": str(args.audit_output)}, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
