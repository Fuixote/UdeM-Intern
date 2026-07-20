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


def merge_feature_targets(
    feature_rows: list[dict[str, str]],
    target_rows: list[dict[str, str]],
    *,
    target_field: str,
    require_formal_targets: bool,
) -> list[dict[str, str]]:
    feature_by_id = {row["topology_id"]: row for row in feature_rows}
    target_by_id = {row["topology_id"]: row for row in target_rows}
    if len(feature_by_id) != len(feature_rows):
        raise ValueError("feature topology ids are not unique")
    if len(target_by_id) != len(target_rows):
        raise ValueError("target topology ids are not unique")
    if set(feature_by_id) != set(target_by_id):
        raise ValueError("feature and target topology sets differ")
    merged = []
    for feature_row in feature_rows:
        topology_id = feature_row["topology_id"]
        feature = dict(feature_row)
        target = target_by_id[topology_id]
        for hash_field in ("topology_hash", "feasible_set_hash"):
            if str(feature.get(hash_field, "")) != str(target.get(hash_field, "")):
                raise ValueError(f"{topology_id}:{hash_field}_mismatch")
        if require_formal_targets and not common.truthy(target.get("formal_label_ready")):
            raise ValueError(f"{topology_id}:formal_target_not_ready")
        try:
            feature[target_field] = str(float(target[target_field]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{topology_id}:target_unavailable:{target_field}") from exc
        feature["formal_label_ready"] = target.get("formal_label_ready", "")
        feature["label_uncertainty_std_pp"] = target.get("label_uncertainty_std_pp", "")
        merged.append(feature)
    return merged


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


def run(
    rows: list[dict[str, str]],
    fold_rows: list[dict[str, str]],
    alpha: float = 10.0,
    *,
    target_field: str = "normalized_improvement_pp",
    require_formal_targets: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fold_by_id = {row["topology_id"]: int(row["fold"]) for row in fold_rows}
    if len(fold_by_id) != len(fold_rows):
        raise ValueError("fold topology ids are not unique")
    if set(fold_by_id) != {row["topology_id"] for row in rows}:
        raise ValueError("fold topology set differs from formal summary")
    target_by_id = {row["topology_id"]: float(row[target_field]) for row in rows}
    x = np.asarray([[float(row[field]) for field in common.SCALAR_FEATURES] for row in rows], dtype=float)
    y = np.asarray([float(row[target_field]) for row in rows], dtype=float)
    folds = np.asarray([fold_by_id[row["topology_id"]] for row in rows], dtype=int)
    failures: list[str] = []
    if len(rows) != 1000:
        failures.append(f"sample_count_mismatch:{len(rows)}!=1000")
    if len(set(folds.tolist())) != 5:
        failures.append(f"fold_count_mismatch:{len(set(folds.tolist()))}!=5")
    if not np.all(np.isfinite(x)):
        failures.append("nonfinite_scalar_features")
    if not np.all(np.isfinite(y)):
        failures.append("nonfinite_targets")
    for fold_row in fold_rows:
        if fold_row.get("target_name") and fold_row["target_name"] != target_field:
            failures.append(f"fold_target_name_mismatch:{fold_row['topology_id']}")
        if fold_row.get("target_value") not in (None, ""):
            expected = target_by_id[fold_row["topology_id"]]
            if not math.isclose(float(fold_row["target_value"]), expected, rel_tol=0.0, abs_tol=1e-12):
                failures.append(f"fold_target_value_mismatch:{fold_row['topology_id']}")
    if require_formal_targets and any(not common.truthy(row.get("formal_label_ready")) for row in rows):
        failures.append("formal_targets_not_ready")
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
            "target_name": target_field,
            "target_value": float(y[index]),
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
        "passed": not failures and len(prediction_rows) == 1000,
        "sample_count": len(rows),
        "fold_count": len(set(folds.tolist())),
        "features": common.SCALAR_FEATURES,
        "target_field": target_field,
        "formal_targets_required": require_formal_targets,
        "target_is_not_an_input_feature": target_field not in common.SCALAR_FEATURES,
        "ridge_alpha": alpha,
        "metrics": metric_rows,
        "ranking": ranking,
        "failures": failures,
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
    parser.add_argument("--target-table", type=Path)
    parser.add_argument("--target-field", default="normalized_improvement_pp")
    parser.add_argument("--require-formal-targets", action="store_true")
    parser.add_argument("--folds", type=Path, default=common.DEFAULT_OUTPUT_ROOT / "splits" / "folds.csv")
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--prediction-output", type=Path, default=common.DEFAULT_OUTPUT_ROOT / "baselines" / "scalar_oof_predictions.csv")
    parser.add_argument("--audit-output", type=Path, default=common.DEFAULT_OUTPUT_ROOT / "baselines" / "scalar_baselines.audit.json")
    args = parser.parse_args()
    rows = common.read_csv(args.formal_summary)
    if args.target_table is not None:
        rows = merge_feature_targets(
            rows,
            common.read_csv(args.target_table),
            target_field=args.target_field,
            require_formal_targets=args.require_formal_targets,
        )
    elif args.require_formal_targets:
        raise ValueError("--require-formal-targets requires --target-table")
    predictions, result = run(
        rows,
        common.read_csv(args.folds),
        args.alpha,
        target_field=args.target_field,
        require_formal_targets=args.require_formal_targets,
    )
    write_csv(args.prediction_output, predictions)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**result, "prediction_output": str(args.prediction_output), "audit_output": str(args.audit_output)}, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
