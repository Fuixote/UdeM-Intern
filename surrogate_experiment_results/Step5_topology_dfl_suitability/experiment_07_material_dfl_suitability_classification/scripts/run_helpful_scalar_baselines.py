#!/usr/bin/env python3
"""Run primary helpful-vs-rest scalar baselines with nested decision selection."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np

import helpful_policy_common as policy
import material_common as common
import run_scalar_classification_baselines as three_class


MODEL_NAMES = ("class_prior", "random_score", "logistic", "extra_trees")


def stable_random_score(seed: int, topology_id: str) -> float:
    digest = common.stable_key(seed, "random_score", topology_id)
    return int(digest[:16], 16) / float(16**16 - 1)


def fit_model_probabilities(
    model_name: str,
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    test_x: np.ndarray,
    validation_ids: list[str],
    test_ids: list[str],
    seed: int,
    fold: int,
    extra_trees_estimators: int,
    extra_trees_jobs: int,
) -> tuple[np.ndarray, np.ndarray]:
    if model_name == "class_prior":
        prior = float(np.mean(train_y))
        return (
            np.full(len(validation_x), prior, dtype=float),
            np.full(len(test_x), prior, dtype=float),
        )
    if model_name == "random_score":
        return (
            np.asarray(
                [stable_random_score(seed, topology_id) for topology_id in validation_ids],
                dtype=float,
            ),
            np.asarray(
                [stable_random_score(seed, topology_id) for topology_id in test_ids],
                dtype=float,
            ),
        )

    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if model_name == "logistic":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=5000,
                random_state=seed + fold,
                solver="lbfgs",
            ),
        )
    elif model_name == "extra_trees":
        model = ExtraTreesClassifier(
            n_estimators=extra_trees_estimators,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=seed + fold,
            n_jobs=extra_trees_jobs,
        )
    else:
        raise ValueError(f"unknown model:{model_name}")
    model.fit(train_x, train_y)
    classes = list(model.classes_)
    positive_index = classes.index(True)
    return (
        model.predict_proba(validation_x)[:, positive_index],
        model.predict_proba(test_x)[:, positive_index],
    )


def run_nested_oof(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    extra_trees_estimators: int,
    extra_trees_jobs: int,
    regret_compute_cost_pp: float,
    precision_constraints: tuple[float, ...],
    compute_costs_pp: tuple[float, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    x = np.asarray([row["features"] for row in rows], dtype=float)
    target = np.asarray(
        [row["label"] == "material_helpful" for row in rows],
        dtype=bool,
    )
    delta = np.asarray([row["formal_label_mean_pp"] for row in rows], dtype=float)
    folds = np.asarray([row["fold"] for row in rows], dtype=int)
    topology_ids = [str(row["topology_id"]) for row in rows]
    prediction_rows: list[dict[str, Any]] = []
    protocol_rows: list[dict[str, Any]] = []

    for test_fold in range(5):
        validation_fold = (test_fold + 1) % 5
        train_mask = (folds != test_fold) & (folds != validation_fold)
        validation_mask = folds == validation_fold
        test_mask = folds == test_fold
        train_y = target[train_mask]
        validation_y = target[validation_mask]
        test_y = target[test_mask]
        validation_delta = delta[validation_mask]
        validation_ids = [
            topology_ids[index]
            for index in np.flatnonzero(validation_mask)
        ]
        test_ids = [
            topology_ids[index]
            for index in np.flatnonzero(test_mask)
        ]
        for model_name in MODEL_NAMES:
            validation_raw, test_raw = fit_model_probabilities(
                model_name,
                train_x=x[train_mask],
                train_y=train_y,
                validation_x=x[validation_mask],
                test_x=x[test_mask],
                validation_ids=validation_ids,
                test_ids=test_ids,
                seed=seed,
                fold=test_fold,
                extra_trees_estimators=extra_trees_estimators,
                extra_trees_jobs=extra_trees_jobs,
            )
            validation_logits = policy.logit(validation_raw)
            test_logits = policy.logit(test_raw)
            temperature = policy.fit_temperature(validation_y, validation_logits)
            validation_probability = policy.apply_temperature(
                validation_logits,
                temperature["temperature"],
            )
            test_probability = policy.apply_temperature(
                test_logits,
                temperature["temperature"],
            )
            regret_rule = policy.select_regret_threshold(
                validation_delta,
                validation_y,
                validation_probability,
                compute_cost_pp=regret_compute_cost_pp,
                compute_costs_pp=compute_costs_pp,
            )
            precision_rules = {
                constraint: policy.select_precision_constrained_threshold(
                    validation_delta,
                    validation_y,
                    validation_probability,
                    minimum_precision=constraint,
                    compute_costs_pp=compute_costs_pp,
                )
                for constraint in precision_constraints
            }
            validation_regret_selected = (
                validation_probability >= regret_rule["threshold"]
            )
            test_regret_selected = test_probability >= regret_rule["threshold"]
            protocol_rows.append(
                {
                    "model": model_name,
                    "test_fold": test_fold,
                    "validation_fold": validation_fold,
                    "train_count": int(np.sum(train_mask)),
                    "validation_count": int(np.sum(validation_mask)),
                    "test_count": int(np.sum(test_mask)),
                    "train_helpful_count": int(np.sum(train_y)),
                    "validation_helpful_count": int(np.sum(validation_y)),
                    "test_helpful_count": int(np.sum(test_y)),
                    "temperature": temperature["temperature"],
                    "validation_nll_before": temperature["validation_nll_before"],
                    "validation_nll_after": temperature["validation_nll_after"],
                    "regret_threshold": regret_rule["threshold"],
                    "regret_validation_selected_count": int(
                        np.sum(validation_regret_selected)
                    ),
                    "precision_0_4_threshold": precision_rules[0.4]["threshold"],
                    "precision_0_5_threshold": precision_rules[0.5]["threshold"],
                }
            )
            for local_index, global_index in enumerate(np.flatnonzero(test_mask)):
                row = rows[int(global_index)]
                prediction_rows.append(
                    {
                        "topology_id": row["topology_id"],
                        "fold": test_fold,
                        "model": model_name,
                        "target_is_material_helpful": int(test_y[local_index]),
                        "secondary_three_class_label": row["label"],
                        "formal_label_mean_pp": float(delta[global_index]),
                        "raw_probability_helpful": float(test_raw[local_index]),
                        "calibrated_probability_helpful": float(
                            test_probability[local_index]
                        ),
                        "temperature": temperature["temperature"],
                        "regret_threshold": regret_rule["threshold"],
                        "selected_regret": int(test_regret_selected[local_index]),
                        "precision_0_4_threshold": precision_rules[0.4]["threshold"],
                        "selected_precision_0_4": int(
                            test_probability[local_index]
                            >= precision_rules[0.4]["threshold"]
                        ),
                        "precision_0_5_threshold": precision_rules[0.5]["threshold"],
                        "selected_precision_0_5": int(
                            test_probability[local_index]
                            >= precision_rules[0.5]["threshold"]
                        ),
                    }
                )
    return prediction_rows, protocol_rows


def review_model(
    prediction_rows: list[dict[str, Any]],
    *,
    compute_costs_pp: tuple[float, ...],
) -> dict[str, Any]:
    target = np.asarray(
        [bool(int(row["target_is_material_helpful"])) for row in prediction_rows],
        dtype=bool,
    )
    delta = np.asarray(
        [float(row["formal_label_mean_pp"]) for row in prediction_rows],
        dtype=float,
    )
    raw_probability = np.asarray(
        [float(row["raw_probability_helpful"]) for row in prediction_rows],
        dtype=float,
    )
    probability = np.asarray(
        [float(row["calibrated_probability_helpful"]) for row in prediction_rows],
        dtype=float,
    )
    selected_regret = np.asarray(
        [bool(int(row["selected_regret"])) for row in prediction_rows],
        dtype=bool,
    )
    selected_precision_0_4 = np.asarray(
        [bool(int(row["selected_precision_0_4"])) for row in prediction_rows],
        dtype=bool,
    )
    selected_precision_0_5 = np.asarray(
        [bool(int(row["selected_precision_0_5"])) for row in prediction_rows],
        dtype=bool,
    )
    fold_metrics = []
    for fold in range(5):
        mask = np.asarray([int(row["fold"]) == fold for row in prediction_rows])
        fold_metrics.append(
            {
                "fold": fold,
                "predictive": policy.binary_predictive_metrics(
                    target[mask],
                    probability[mask],
                    selected=selected_regret[mask],
                ),
                "policy": common.policy_metrics_from_selection(
                    delta[mask],
                    target[mask],
                    selected_regret[mask],
                    selection_rule="validation_regret_optimal",
                    compute_costs_pp=compute_costs_pp,
                ),
            }
        )
    return {
        "raw_predictive": policy.binary_predictive_metrics(target, raw_probability),
        "calibrated_predictive": policy.binary_predictive_metrics(
            target,
            probability,
            selected=selected_regret,
        ),
        "regret_optimal_policy": common.policy_metrics_from_selection(
            delta,
            target,
            selected_regret,
            selection_rule="foldwise_validation_regret_optimal",
            compute_costs_pp=compute_costs_pp,
        ),
        "precision_0_4_policy": common.policy_metrics_from_selection(
            delta,
            target,
            selected_precision_0_4,
            selection_rule="foldwise_validation_precision_at_least_0.4",
            compute_costs_pp=compute_costs_pp,
        ),
        "precision_0_5_policy": common.policy_metrics_from_selection(
            delta,
            target,
            selected_precision_0_5,
            selection_rule="foldwise_validation_precision_at_least_0.5",
            compute_costs_pp=compute_costs_pp,
        ),
        "ranking_capture": policy.ranking_capture_metrics(delta, probability),
        "outlier_sensitivity": policy.outlier_sensitivity(delta, selected_regret),
        "fold_metrics": fold_metrics,
    }


def policy_baselines(
    rows: list[dict[str, Any]],
    *,
    compute_costs_pp: tuple[float, ...],
) -> dict[str, Any]:
    delta = np.asarray([row["formal_label_mean_pp"] for row in rows], dtype=float)
    helpful = delta > 0.5
    return {
        "always_2stage": common.policy_metrics_from_selection(
            delta,
            helpful,
            np.zeros(len(delta), dtype=bool),
            selection_rule="always_2stage",
            compute_costs_pp=compute_costs_pp,
        ),
        "always_spoplus": common.policy_metrics_from_selection(
            delta,
            helpful,
            np.ones(len(delta), dtype=bool),
            selection_rule="always_spoplus",
            compute_costs_pp=compute_costs_pp,
        ),
        "oracle_positive_delta": common.policy_metrics_from_selection(
            delta,
            helpful,
            delta > 0.0,
            selection_rule="oracle_choose_spoplus_if_delta_positive",
            compute_costs_pp=compute_costs_pp,
        ),
    }


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
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--extra-trees-estimators", type=int, default=500)
    parser.add_argument("--extra-trees-jobs", type=int, default=1)
    parser.add_argument("--regret-compute-cost-pp", type=float, default=0.0)
    parser.add_argument("--compute-costs-pp", default="0,0.05,0.1,0.25,0.5")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=common.DEFAULT_OUTPUT_ROOT / "baselines" / "helpful_binary",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    compute_costs = tuple(three_class.parse_costs(args.compute_costs_pp))
    feature_rows = common.read_csv(args.formal_summary)
    label_rows = common.read_csv(args.labels)
    fold_rows = common.read_csv(args.folds)
    rows = three_class.merge_inputs(
        feature_rows,
        label_rows,
        fold_rows,
        label_field="primary_label",
    )
    failures: list[str] = []
    summary_sha256 = common.sha256_file(args.formal_summary)
    if summary_sha256 != common.LOCKED_SUMMARY_SHA256:
        failures.append(f"formal_summary_sha256_mismatch:{summary_sha256}")
    if len(rows) != 1000:
        failures.append(f"row_count_mismatch:{len(rows)}!=1000")
    prediction_rows, protocol_rows = run_nested_oof(
        rows,
        seed=args.seed,
        extra_trees_estimators=args.extra_trees_estimators,
        extra_trees_jobs=args.extra_trees_jobs,
        regret_compute_cost_pp=args.regret_compute_cost_pp,
        precision_constraints=(0.4, 0.5),
        compute_costs_pp=compute_costs,
    )
    reviews: dict[str, Any] = {}
    for model_name in MODEL_NAMES:
        model_rows = [
            row for row in prediction_rows
            if row["model"] == model_name
        ]
        if len(model_rows) != 1000 or len({row["topology_id"] for row in model_rows}) != 1000:
            failures.append(f"prediction_count_or_uniqueness_mismatch:{model_name}")
        reviews[model_name] = review_model(
            model_rows,
            compute_costs_pp=compute_costs,
        )
    audit = {
        "passed": not failures,
        "status": "success" if not failures else "failed",
        "primary_task": "material_helpful_vs_non_helpful",
        "secondary_task": "three_class_material_harmful_neutral_helpful",
        "confidence_labels_used_for_training": False,
        "sample_count": len(rows),
        "helpful_count": sum(row["label"] == "material_helpful" for row in rows),
        "non_helpful_count": sum(row["label"] != "material_helpful" for row in rows),
        "models": list(MODEL_NAMES),
        "nested_split": {
            "train": 600,
            "validation": 200,
            "test": 200,
            "validation_uses": [
                "temperature_scaling",
                "regret_threshold_selection",
                "precision_constraint_sensitivity_thresholds",
            ],
            "test_threshold_retuning_forbidden": True,
        },
        "temperature_scaling": "single_positive_temperature_fit_by_validation_binary_nll",
        "primary_threshold_rule": {
            "objective": "maximize_validation_net_benefit_equivalent_to_minimize_regret",
            "compute_cost_pp": args.regret_compute_cost_pp,
        },
        "secondary_precision_constraints": [0.4, 0.5],
        "policy_baselines": policy_baselines(
            rows,
            compute_costs_pp=compute_costs,
        ),
        "reviews": reviews,
        "formal_summary_sha256": summary_sha256,
        "label_sha256": common.sha256_file(args.labels),
        "fold_sha256": common.sha256_file(args.folds),
        "target_or_uncertainty_used_as_input_feature": False,
        "failures": failures,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    common.atomic_write_csv(args.output_dir / "oof_predictions.csv", prediction_rows)
    common.atomic_write_csv(args.output_dir / "fold_protocols.csv", protocol_rows)
    compact_rows = []
    for model_name, review in reviews.items():
        predictive = review["calibrated_predictive"]
        model_policy = review["regret_optimal_policy"]
        compact_rows.append(
            {
                "model": model_name,
                "count": predictive["count"],
                "helpful_prevalence": predictive["prevalence"],
                "auroc": predictive["auroc"],
                "auprc": predictive["auprc"],
                "brier": predictive["brier"],
                "nll": predictive["nll"],
                "ece_10bin": predictive["ece_10bin"],
                "selected_count": model_policy["selected_count"],
                "helpful_precision": model_policy["helpful_precision"],
                "helpful_recall": model_policy["helpful_recall"],
                "harmful_selected_count": model_policy["harmful_selected_count"],
                "mean_realized_improvement_pp": model_policy["mean_realized_improvement_pp"],
                "policy_regret_pp": model_policy["policy_regret_pp"],
                "oracle_improvement_captured": model_policy["oracle_improvement_captured"],
                "total_negative_uplift_incurred_pp": model_policy["total_negative_uplift_incurred_pp"],
            }
        )
    common.atomic_write_csv(args.output_dir / "model_metrics.csv", compact_rows)
    common.atomic_write_json(args.output_dir / "helpful_baseline_review.audit.json", audit)
    print(
        json.dumps(
            {
                "passed": audit["passed"],
                "status": audit["status"],
                "primary_task": audit["primary_task"],
                "helpful_count": audit["helpful_count"],
                "non_helpful_count": audit["non_helpful_count"],
                "compact_metrics": compact_rows,
                "output_dir": str(args.output_dir),
                "failures": failures,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if audit["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
