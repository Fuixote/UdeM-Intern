#!/usr/bin/env python3
"""Run leakage-controlled five-fold scalar classification baselines."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import material_common as common


def merge_inputs(
    feature_rows: list[dict[str, str]],
    label_rows: list[dict[str, str]],
    fold_rows: list[dict[str, str]],
    *,
    label_field: str,
) -> list[dict[str, Any]]:
    feature_by_id = {row["topology_id"]: row for row in feature_rows}
    label_by_id = {row["topology_id"]: row for row in label_rows}
    fold_by_id = {row["topology_id"]: row for row in fold_rows}
    for name, rows, mapping in (
        ("feature", feature_rows, feature_by_id),
        ("label", label_rows, label_by_id),
        ("fold", fold_rows, fold_by_id),
    ):
        if len(mapping) != len(rows):
            raise ValueError(f"{name} topology ids are not unique")
    if not (set(feature_by_id) == set(label_by_id) == set(fold_by_id)):
        raise ValueError("feature, label, and fold topology sets differ")
    output: list[dict[str, Any]] = []
    for topology_id in sorted(feature_by_id, key=common.topology_sort_key):
        feature = feature_by_id[topology_id]
        label = label_by_id[topology_id]
        fold = fold_by_id[topology_id]
        for hash_field in ("topology_hash", "feasible_set_hash"):
            if feature[hash_field] != label[hash_field] or label[hash_field] != fold[hash_field]:
                raise ValueError(f"{topology_id}:{hash_field}_mismatch")
        target = label[label_field]
        if target not in common.CLASS_LABELS:
            raise ValueError(f"{topology_id}:unknown_target:{target}")
        values = [float(feature[field]) for field in common.SCALAR_FEATURES]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{topology_id}:nonfinite_feature")
        output.append(
            {
                "topology_id": topology_id,
                "topology_hash": label["topology_hash"],
                "feasible_set_hash": label["feasible_set_hash"],
                "fold": int(fold["fold"]),
                "label": target,
                "formal_label_mean_pp": float(label["formal_label_mean_pp"]),
                "features": values,
            }
        )
    return output


def align_probabilities(
    model_classes: Sequence[str],
    probabilities: np.ndarray,
) -> np.ndarray:
    aligned = np.zeros((len(probabilities), len(common.CLASS_LABELS)), dtype=float)
    class_index = {label: index for index, label in enumerate(model_classes)}
    for output_index, label in enumerate(common.CLASS_LABELS):
        if label in class_index:
            aligned[:, output_index] = probabilities[:, class_index[label]]
    row_sums = aligned.sum(axis=1)
    if not np.allclose(row_sums, 1.0):
        raise ValueError("aligned probabilities do not sum to one")
    return aligned


def run_oof(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    extra_trees_estimators: int,
    extra_trees_jobs: int,
) -> dict[str, np.ndarray]:
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x = np.asarray([row["features"] for row in rows], dtype=float)
    y = np.asarray([row["label"] for row in rows], dtype=object)
    folds = np.asarray([row["fold"] for row in rows], dtype=int)
    outputs = {
        model: np.zeros((len(rows), len(common.CLASS_LABELS)), dtype=float)
        for model in ("class_prior", "logistic", "extra_trees")
    }
    for fold in sorted(set(folds.tolist())):
        validation_fold = (fold + 1) % 5
        train_mask = (folds != fold) & (folds != validation_fold)
        test_mask = folds == fold
        train_y = y[train_mask]
        counts = Counter(str(value) for value in train_y)
        prior = np.asarray(
            [counts[label] / len(train_y) for label in common.CLASS_LABELS],
            dtype=float,
        )
        outputs["class_prior"][test_mask] = prior

        logistic = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=5000,
                random_state=seed + fold,
                solver="lbfgs",
            ),
        )
        logistic.fit(x[train_mask], train_y)
        outputs["logistic"][test_mask] = align_probabilities(
            logistic.named_steps["logisticregression"].classes_,
            logistic.predict_proba(x[test_mask]),
        )

        extra_trees = ExtraTreesClassifier(
            n_estimators=extra_trees_estimators,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=seed + fold,
            n_jobs=extra_trees_jobs,
        )
        extra_trees.fit(x[train_mask], train_y)
        outputs["extra_trees"][test_mask] = align_probabilities(
            extra_trees.classes_,
            extra_trees.predict_proba(x[test_mask]),
        )
    return outputs


def evaluate_model(
    target: np.ndarray,
    delta_pp: np.ndarray,
    probabilities: np.ndarray,
    *,
    helpful_threshold: float,
    compute_costs_pp: Sequence[float],
) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    class_to_index = common.CLASS_TO_INDEX
    target_index = np.asarray([class_to_index[str(label)] for label in target], dtype=int)
    target_one_hot = np.eye(len(common.CLASS_LABELS), dtype=float)[target_index]
    predicted_index = np.argmax(probabilities, axis=1)
    predicted = np.asarray([common.CLASS_LABELS[index] for index in predicted_index], dtype=object)
    precision, recall, f1, support = precision_recall_fscore_support(
        target,
        predicted,
        labels=list(common.CLASS_LABELS),
        zero_division=0,
    )
    per_class: list[dict[str, Any]] = []
    for index, label in enumerate(common.CLASS_LABELS):
        binary_target = target_one_hot[:, index]
        per_class.append(
            {
                "class": label,
                "support": int(support[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "auroc": float(roc_auc_score(binary_target, probabilities[:, index])),
                "auprc": float(average_precision_score(binary_target, probabilities[:, index])),
            }
        )
    helpful_index = class_to_index["material_helpful"]
    helpful_target = target_one_hot[:, helpful_index].astype(bool)
    multiclass_brier = float(np.mean(np.sum((target_one_hot - probabilities) ** 2, axis=1)))
    metrics = {
        "count": len(target),
        "accuracy": float(accuracy_score(target, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(target, predicted)),
        "macro_f1": float(f1_score(target, predicted, labels=list(common.CLASS_LABELS), average="macro")),
        "macro_auroc_ovr": float(np.mean([row["auroc"] for row in per_class])),
        "macro_auprc_ovr": float(np.mean([row["auprc"] for row in per_class])),
        "log_loss": float(
            -np.mean(np.log(np.clip(probabilities[np.arange(len(target)), target_index], 1e-15, 1.0)))
        ),
        "multiclass_brier": multiclass_brier,
        "helpful_brier": float(np.mean((helpful_target.astype(float) - probabilities[:, helpful_index]) ** 2)),
        "helpful_ece_10bin": common.expected_calibration_error(
            helpful_target.astype(float),
            probabilities[:, helpful_index],
            bins=10,
        ),
        "top_label_ece_10bin": common.top_label_calibration_error(
            target_index,
            probabilities,
            bins=10,
        ),
        "confusion_matrix_class_order": list(common.CLASS_LABELS),
        "confusion_matrix": confusion_matrix(
            target,
            predicted,
            labels=list(common.CLASS_LABELS),
        ).tolist(),
        "per_class": per_class,
        "argmax_helpful_policy": common.policy_metrics_from_selection(
            delta_pp,
            helpful_target,
            predicted == "material_helpful",
            selection_rule="predicted_class_is_material_helpful",
            compute_costs_pp=compute_costs_pp,
        ),
        "policy": common.policy_metrics(
            delta_pp,
            helpful_target,
            probabilities[:, helpful_index],
            threshold=helpful_threshold,
            compute_costs_pp=compute_costs_pp,
        ),
    }
    return metrics


def build_prediction_rows(
    rows: list[dict[str, Any]],
    outputs: dict[str, np.ndarray],
    *,
    label_field: str,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for model, probabilities in outputs.items():
        for row, probability in zip(rows, probabilities, strict=True):
            predicted_index = int(np.argmax(probability))
            predictions.append(
                {
                    "topology_id": row["topology_id"],
                    "fold": row["fold"],
                    "model": model,
                    "label_field": label_field,
                    "target_label": row["label"],
                    "formal_label_mean_pp": row["formal_label_mean_pp"],
                    "probability_material_harmful": float(probability[0]),
                    "probability_neutral_or_uncertain": float(probability[1]),
                    "probability_material_helpful": float(probability[2]),
                    "predicted_label": common.CLASS_LABELS[predicted_index],
                }
            )
    return predictions


def parse_costs(raw: str) -> list[float]:
    values = [float(value) for value in raw.split(",") if value.strip()]
    if not values or any(value < 0.0 for value in values):
        raise ValueError("compute costs must be a nonempty list of nonnegative pp values")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-summary", type=Path, default=common.DEFAULT_FORMAL_SUMMARY)
    parser.add_argument(
        "--labels",
        type=Path,
        default=common.DEFAULT_OUTPUT_ROOT / "labels" / "material_labels.csv",
    )
    parser.add_argument(
        "--folds",
        type=Path,
        default=common.DEFAULT_OUTPUT_ROOT / "splits" / "material_folds.csv",
    )
    parser.add_argument("--label-field", choices=("primary_label", "confidence_label"), default="primary_label")
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--extra-trees-estimators", type=int, default=500)
    parser.add_argument("--extra-trees-jobs", type=int, default=1)
    parser.add_argument("--helpful-threshold", type=float, default=0.5)
    parser.add_argument("--compute-costs-pp", default="0,0.05,0.1,0.25,0.5")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=common.DEFAULT_OUTPUT_ROOT / "baselines" / "primary",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    compute_costs = parse_costs(args.compute_costs_pp)
    summary_sha256 = common.sha256_file(args.formal_summary)
    failures: list[str] = []
    if summary_sha256 != common.LOCKED_SUMMARY_SHA256:
        failures.append(f"formal_summary_sha256_mismatch:{summary_sha256}")
    rows = merge_inputs(
        common.read_csv(args.formal_summary),
        common.read_csv(args.labels),
        common.read_csv(args.folds),
        label_field=args.label_field,
    )
    if len(rows) != 1000:
        failures.append(f"input_count_mismatch:{len(rows)}!=1000")
    fold_counts = Counter(row["fold"] for row in rows)
    if set(fold_counts) != set(range(5)) or set(fold_counts.values()) != {200}:
        failures.append(f"fold_size_mismatch:{dict(fold_counts)}")
    target = np.asarray([row["label"] for row in rows], dtype=object)
    delta_pp = np.asarray([row["formal_label_mean_pp"] for row in rows], dtype=float)
    outputs = run_oof(
        rows,
        seed=args.seed,
        extra_trees_estimators=args.extra_trees_estimators,
        extra_trees_jobs=args.extra_trees_jobs,
    )
    model_metrics = {
        model: evaluate_model(
            target,
            delta_pp,
            probability,
            helpful_threshold=args.helpful_threshold,
            compute_costs_pp=compute_costs,
        )
        for model, probability in outputs.items()
    }
    prediction_rows = build_prediction_rows(rows, outputs, label_field=args.label_field)
    for model in outputs:
        model_rows = [row for row in prediction_rows if row["model"] == model]
        if len(model_rows) != 1000 or len({row["topology_id"] for row in model_rows}) != 1000:
            failures.append(f"prediction_count_or_uniqueness_mismatch:{model}")
    audit = {
        "passed": not failures,
        "status": "success" if not failures else "failed",
        "sample_count": len(rows),
        "fold_counts": {str(fold): fold_counts[fold] for fold in sorted(fold_counts)},
        "nested_split": {
            "train_folds": 3,
            "validation_fold": "(test_fold + 1) mod 5",
            "test_folds": 1,
            "sizes": {"train": 600, "validation": 200, "test": 200},
            "validation_is_reserved_for_later_model_selection": True,
        },
        "label_field": args.label_field,
        "label_counts": common.label_counts(
            [{"value": label} for label in target],
            "value",
        ),
        "models": list(outputs),
        "scalar_features": list(common.SCALAR_FEATURES),
        "formal_summary_sha256": summary_sha256,
        "expected_formal_summary_sha256": common.LOCKED_SUMMARY_SHA256,
        "target_or_uncertainty_used_as_input_feature": False,
        "extra_trees_estimators": args.extra_trees_estimators,
        "extra_trees_jobs": args.extra_trees_jobs,
        "helpful_probability_threshold": args.helpful_threshold,
        "compute_costs_pp": compute_costs,
        "metrics": model_metrics,
        "failures": failures,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    common.atomic_write_csv(args.output_dir / "oof_predictions.csv", prediction_rows)
    summary_rows = []
    per_class_rows = []
    for model, metrics in model_metrics.items():
        policy = metrics["policy"]
        argmax_policy = metrics["argmax_helpful_policy"]
        summary_rows.append(
            {
                "model": model,
                "label_field": args.label_field,
                "count": metrics["count"],
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "macro_auroc_ovr": metrics["macro_auroc_ovr"],
                "macro_auprc_ovr": metrics["macro_auprc_ovr"],
                "log_loss": metrics["log_loss"],
                "multiclass_brier": metrics["multiclass_brier"],
                "helpful_ece_10bin": metrics["helpful_ece_10bin"],
                "top_label_ece_10bin": metrics["top_label_ece_10bin"],
                "helpful_precision_at_0.5": policy["helpful_precision"],
                "selection_rate_at_0.5": policy["selection_rate"],
                "policy_regret_pp_at_0.5": policy["policy_regret_pp"],
                "oracle_improvement_captured_at_0.5": policy["oracle_improvement_captured"],
                "mean_realized_improvement_pp_at_0.5": policy["mean_realized_improvement_pp"],
                "argmax_helpful_precision": argmax_policy["helpful_precision"],
                "argmax_selection_rate": argmax_policy["selection_rate"],
                "argmax_policy_regret_pp": argmax_policy["policy_regret_pp"],
                "argmax_oracle_improvement_captured": argmax_policy["oracle_improvement_captured"],
                "argmax_mean_realized_improvement_pp": argmax_policy["mean_realized_improvement_pp"],
            }
        )
        for row in metrics["per_class"]:
            per_class_rows.append({"model": model, "label_field": args.label_field, **row})
    common.atomic_write_csv(args.output_dir / "model_metrics.csv", summary_rows)
    common.atomic_write_csv(args.output_dir / "per_class_metrics.csv", per_class_rows)
    common.atomic_write_json(args.output_dir / "baseline_review.audit.json", audit)
    print(
        json.dumps(
            {
                **audit,
                "prediction_output": str(args.output_dir / "oof_predictions.csv"),
                "metric_output": str(args.output_dir / "model_metrics.csv"),
                "audit_output": str(args.output_dir / "baseline_review.audit.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if audit["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
