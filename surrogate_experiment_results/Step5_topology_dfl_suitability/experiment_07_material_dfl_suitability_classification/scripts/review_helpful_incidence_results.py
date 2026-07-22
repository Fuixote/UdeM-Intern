#!/usr/bin/env python3
"""Pool five-fold helpful GNN OOF predictions and apply the locked gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import helpful_policy_common as policy
import material_common as common
import run_helpful_scalar_baselines as scalar


def read_gnn_predictions(input_root: Path, seed: int) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for fold in range(5):
        run_dir = input_root / "folds" / f"fold{fold}" / f"seed{seed}"
        result_path = run_dir / "run_result.json"
        prediction_path = run_dir / "test_predictions.csv"
        if not result_path.is_file() or not prediction_path.is_file():
            failures.append(f"missing_fold_artifact:fold{fold}:seed{seed}")
            continue
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            fold_rows = common.read_csv(prediction_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            failures.append(f"unreadable_fold_artifact:fold{fold}:{exc}")
            continue
        if not (
            result.get("status") == "success"
            and result.get("formal") is True
            and result.get("task") == "material_helpful_vs_non_helpful"
            and int(result.get("fold", -1)) == fold
            and int(result.get("seed", -1)) == seed
        ):
            failures.append(f"invalid_run_identity:fold{fold}:seed{seed}")
        if len(fold_rows) != 200 or len({row.get("topology_id") for row in fold_rows}) != 200:
            failures.append(f"invalid_fold_prediction_count:fold{fold}:{len(fold_rows)}")
        if any(int(row.get("fold", -1)) != fold for row in fold_rows):
            failures.append(f"prediction_fold_mismatch:fold{fold}")
        for row in fold_rows:
            row["model"] = "incidence_gnn"
        rows.extend(fold_rows)
    if len(rows) != 1000 or len({row.get("topology_id") for row in rows}) != 1000:
        failures.append(f"oof_count_or_uniqueness_mismatch:{len(rows)}")
    return rows, failures


def arrays(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {
        "target": np.asarray(
            [bool(int(row["target_is_material_helpful"])) for row in rows], dtype=bool
        ),
        "delta": np.asarray([float(row["formal_label_mean_pp"]) for row in rows]),
        "probability": np.asarray(
            [float(row["calibrated_probability_helpful"]) for row in rows]
        ),
        "selected": np.asarray([bool(int(row["selected_regret"])) for row in rows]),
        "fold": np.asarray([int(row["fold"]) for row in rows], dtype=int),
    }


def align_authoritative_targets(
    gnn_rows: list[dict[str, Any]], reference_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    reference_by_id = {row["topology_id"]: row for row in reference_rows}
    maximum_delta_drift = 0.0
    for row in gnn_rows:
        reference = reference_by_id.get(row["topology_id"])
        if reference is None:
            raise ValueError(f"missing authoritative target:{row['topology_id']}")
        if int(row["fold"]) != int(reference["fold"]):
            raise ValueError(f"fold mismatch:{row['topology_id']}")
        if int(row["target_is_material_helpful"]) != int(
            reference["target_is_material_helpful"]
        ):
            raise ValueError(f"binary target mismatch:{row['topology_id']}")
        drift = abs(
            float(row["formal_label_mean_pp"])
            - float(reference["formal_label_mean_pp"])
        )
        maximum_delta_drift = max(maximum_delta_drift, drift)
        row["formal_label_mean_pp"] = reference["formal_label_mean_pp"]
    return {
        "source": "scalar_oof_locked_double_precision_target_by_topology_id",
        "maximum_remote_float32_delta_drift_pp": maximum_delta_drift,
        "probabilities_or_selections_changed": False,
    }


def paired_bootstrap(
    gnn_rows: list[dict[str, Any]],
    scalar_rows: list[dict[str, Any]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score

    scalar_by_id = {row["topology_id"]: row for row in scalar_rows}
    ordered_gnn = sorted(gnn_rows, key=lambda row: common.topology_sort_key(row["topology_id"]))
    ordered_scalar = [scalar_by_id[row["topology_id"]] for row in ordered_gnn]
    gnn = arrays(ordered_gnn)
    baseline = arrays(ordered_scalar)
    if not np.array_equal(gnn["target"], baseline["target"]):
        raise ValueError("paired targets differ")
    rng = np.random.default_rng(seed)
    auprc_difference = np.empty(iterations, dtype=float)
    regret_difference = np.empty(iterations, dtype=float)
    count = len(ordered_gnn)
    for iteration in range(iterations):
        index = rng.integers(0, count, size=count)
        target = gnn["target"][index]
        auprc_difference[iteration] = average_precision_score(
            target, gnn["probability"][index]
        ) - average_precision_score(target, baseline["probability"][index])
        delta = gnn["delta"][index]
        oracle = np.maximum(delta, 0.0)
        gnn_regret = np.mean(oracle - delta * gnn["selected"][index])
        baseline_regret = np.mean(oracle - delta * baseline["selected"][index])
        regret_difference[iteration] = baseline_regret - gnn_regret
    return {
        "iterations": iterations,
        "seed": seed,
        "auprc_difference_gnn_minus_scalar": {
            "mean": float(np.mean(auprc_difference)),
            "ci95": [float(value) for value in np.quantile(auprc_difference, [0.025, 0.975])],
            "probability_above_zero": float(np.mean(auprc_difference > 0.0)),
        },
        "policy_regret_improvement_scalar_minus_gnn_pp": {
            "mean": float(np.mean(regret_difference)),
            "ci95": [float(value) for value in np.quantile(regret_difference, [0.025, 0.975])],
            "probability_above_zero": float(np.mean(regret_difference > 0.0)),
        },
    }


def fold_direction_count(
    gnn_rows: list[dict[str, Any]], scalar_rows_by_model: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    output = []
    for fold in range(5):
        gnn_fold = [row for row in gnn_rows if int(row["fold"]) == fold]
        gnn_arrays = arrays(gnn_fold)
        gnn_ap = policy.binary_predictive_metrics(
            gnn_arrays["target"], gnn_arrays["probability"]
        )["auprc"]
        scalar_ap = {}
        for model, rows in scalar_rows_by_model.items():
            fold_rows = [row for row in rows if int(row["fold"]) == fold]
            fold_arrays = arrays(fold_rows)
            scalar_ap[model] = policy.binary_predictive_metrics(
                fold_arrays["target"], fold_arrays["probability"]
            )["auprc"]
        best_name = max(scalar_ap, key=scalar_ap.get)
        output.append({
            "fold": fold,
            "incidence_gnn_auprc": gnn_ap,
            "best_scalar_model": best_name,
            "best_scalar_auprc": scalar_ap[best_name],
            "gnn_higher": gnn_ap > scalar_ap[best_name],
        })
    return {
        "folds_gnn_higher_than_best_scalar": sum(row["gnn_higher"] for row in output),
        "folds": output,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=common.DEFAULT_OUTPUT_ROOT / "incidence_gnn_helpful" / "formal_seed42",
    )
    parser.add_argument(
        "--scalar-oof",
        type=Path,
        default=common.DEFAULT_OUTPUT_ROOT / "baselines" / "helpful_binary" / "oof_predictions.csv",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260722)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.input_root / "review"
    gnn_rows, failures = read_gnn_predictions(args.input_root, args.seed)
    if failures:
        audit = {
            "passed": False,
            "status": "blocked_incomplete_inputs",
            "failures": failures,
        }
        common.atomic_write_json(output_dir / "helpful_incidence_review.audit.json", audit)
        print(json.dumps(audit, indent=2, sort_keys=True))
        return 1
    scalar_rows_all = common.read_csv(args.scalar_oof)
    scalar_rows_by_model = {
        model: [row for row in scalar_rows_all if row["model"] == model]
        for model in ("logistic", "extra_trees")
    }
    for model, rows in scalar_rows_by_model.items():
        if len(rows) != 1000 or len({row["topology_id"] for row in rows}) != 1000:
            failures.append(f"scalar_oof_invalid:{model}:{len(rows)}")
    target_alignment = align_authoritative_targets(
        gnn_rows, scalar_rows_by_model["logistic"]
    )
    gnn_review = scalar.review_model(gnn_rows, compute_costs_pp=(0.0, 0.05, 0.1, 0.25, 0.5))
    scalar_reviews = {
        model: scalar.review_model(rows, compute_costs_pp=(0.0, 0.05, 0.1, 0.25, 0.5))
        for model, rows in scalar_rows_by_model.items()
    }
    fold_directions = fold_direction_count(gnn_rows, scalar_rows_by_model)
    bootstraps = {
        model: paired_bootstrap(
            gnn_rows,
            rows,
            iterations=args.bootstrap_iterations,
            seed=args.bootstrap_seed + index,
        )
        for index, (model, rows) in enumerate(scalar_rows_by_model.items())
    }
    prevalence = gnn_review["calibrated_predictive"]["prevalence"]
    gnn_auprc = gnn_review["calibrated_predictive"]["auprc"]
    best_scalar_auprc = max(
        review["calibrated_predictive"]["auprc"] for review in scalar_reviews.values()
    )
    gnn_policy = gnn_review["regret_optimal_policy"]
    gate_checks = {
        "predictive_auprc_above_prevalence": gnn_auprc > prevalence,
        "predictive_auprc_above_best_scalar": gnn_auprc > best_scalar_auprc,
        "at_least_four_folds_directionally_above_best_scalar": (
            fold_directions["folds_gnn_higher_than_best_scalar"] >= 4
        ),
        "paired_bootstrap_auprc_probability_above_zero_at_least_0_95": all(
            value["auprc_difference_gnn_minus_scalar"]["probability_above_zero"] >= 0.95
            for value in bootstraps.values()
        ),
        "policy_regret_below_both_scalar_policies": all(
            gnn_policy["policy_regret_pp"] < review["regret_optimal_policy"]["policy_regret_pp"]
            for review in scalar_reviews.values()
        ),
        "positive_uplift_capture_above_both_scalar_policies": all(
            gnn_policy["oracle_improvement_captured"]
            > review["regret_optimal_policy"]["oracle_improvement_captured"]
            for review in scalar_reviews.values()
        ),
        "harmful_selection_no_worse_than_worst_scalar": (
            gnn_policy["harmful_selected_count"]
            <= max(review["regret_optimal_policy"]["harmful_selected_count"] for review in scalar_reviews.values())
        ),
        "remove_top5_regret_below_both_scalar": all(
            gnn_review["outlier_sensitivity"]["remove_top_5"]["policy_regret_pp"]
            < review["outlier_sensitivity"]["remove_top_5"]["policy_regret_pp"]
            for review in scalar_reviews.values()
        ),
    }
    promotion_passed = all(gate_checks.values())
    audit = {
        "passed": not failures,
        "status": "success" if not failures else "failed",
        "formal": True,
        "task": "material_helpful_vs_non_helpful",
        "seed": args.seed,
        "oof_count": len(gnn_rows),
        "oof_unique_topology_count": len({row["topology_id"] for row in gnn_rows}),
        "test_coverage": "each_topology_exactly_once",
        "test_threshold_retuning_performed": False,
        "authoritative_target_alignment": target_alignment,
        "incidence_gnn_review": gnn_review,
        "scalar_reviews": scalar_reviews,
        "fold_directions": fold_directions,
        "paired_bootstraps": bootstraps,
        "promotion_gate": {
            "passed": promotion_passed,
            "interpretation": (
                "promote_to_seeds_43_44"
                if promotion_passed
                else "do_not_promote_without_protocol_change_or_new_evidence"
            ),
            "checks": gate_checks,
        },
        "failures": failures,
    }
    common.atomic_write_csv(output_dir / "incidence_gnn_oof_predictions.csv", gnn_rows)
    metric_rows = []
    for model, review in {"incidence_gnn": gnn_review, **scalar_reviews}.items():
        predictive = review["calibrated_predictive"]
        model_policy = review["regret_optimal_policy"]
        metric_rows.append({
            "model": model,
            "auroc": predictive["auroc"],
            "auprc": predictive["auprc"],
            "prevalence": predictive["prevalence"],
            "brier": predictive["brier"],
            "nll": predictive["nll"],
            "ece_10bin": predictive["ece_10bin"],
            "selected_count": model_policy["selected_count"],
            "helpful_precision": model_policy["helpful_precision"],
            "helpful_recall": model_policy["helpful_recall"],
            "harmful_selected_count": model_policy["harmful_selected_count"],
            "policy_regret_pp": model_policy["policy_regret_pp"],
            "oracle_improvement_captured": model_policy["oracle_improvement_captured"],
            "total_negative_uplift_incurred_pp": model_policy["total_negative_uplift_incurred_pp"],
        })
    common.atomic_write_csv(output_dir / "model_comparison.csv", metric_rows)
    common.atomic_write_json(output_dir / "helpful_incidence_review.audit.json", audit)
    print(json.dumps({
        "passed": audit["passed"],
        "status": audit["status"],
        "oof_count": audit["oof_count"],
        "promotion_gate": audit["promotion_gate"],
        "model_comparison": metric_rows,
        "output_dir": str(output_dir),
        "failures": failures,
    }, indent=2, sort_keys=True))
    return 0 if audit["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
