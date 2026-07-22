#!/usr/bin/env python3
"""Binary helpful-detection calibration, threshold, and policy helpers."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

import material_common as common


def sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    output = np.empty_like(logits)
    positive = logits >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    negative = ~positive
    exponential = np.exp(logits[negative])
    output[negative] = exponential / (1.0 + exponential)
    return output


def logit(probability: np.ndarray, epsilon: float = 1e-7) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), epsilon, 1.0 - epsilon)
    return np.log(clipped) - np.log1p(-clipped)


def binary_nll(target: np.ndarray, logits: np.ndarray) -> float:
    target = np.asarray(target, dtype=float)
    logits = np.asarray(logits, dtype=float)
    if len(target) != len(logits):
        raise ValueError("target and logits lengths differ")
    return float(np.mean(np.logaddexp(0.0, logits) - target * logits))


def fit_temperature(
    validation_target: np.ndarray,
    validation_logits: np.ndarray,
    *,
    lower_log_temperature: float = -5.0,
    upper_log_temperature: float = 5.0,
    iterations: int = 120,
) -> dict[str, float]:
    """Fit one positive temperature by deterministic golden-section search."""
    target = np.asarray(validation_target, dtype=float)
    logits = np.asarray(validation_logits, dtype=float)
    if len(target) != len(logits) or len(target) == 0:
        raise ValueError("temperature fitting requires aligned nonempty arrays")
    if set(np.unique(target).tolist()) != {0.0, 1.0}:
        raise ValueError("temperature fitting requires both binary classes")

    def objective(log_temperature: float) -> float:
        return binary_nll(target, logits / math.exp(log_temperature))

    left = float(lower_log_temperature)
    right = float(upper_log_temperature)
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    middle_left = right - ratio * (right - left)
    middle_right = left + ratio * (right - left)
    value_left = objective(middle_left)
    value_right = objective(middle_right)
    for _ in range(iterations):
        if value_left <= value_right:
            right = middle_right
            middle_right = middle_left
            value_right = value_left
            middle_left = right - ratio * (right - left)
            value_left = objective(middle_left)
        else:
            left = middle_left
            middle_left = middle_right
            value_left = value_right
            middle_right = left + ratio * (right - left)
            value_right = objective(middle_right)
    log_temperature = (left + right) / 2.0
    temperature = math.exp(log_temperature)
    return {
        "temperature": temperature,
        "validation_nll_before": binary_nll(target, logits),
        "validation_nll_after": binary_nll(target, logits / temperature),
    }


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    return sigmoid(np.asarray(logits, dtype=float) / temperature)


def candidate_thresholds(probability: np.ndarray) -> list[float]:
    probability = np.asarray(probability, dtype=float)
    if len(probability) == 0 or not np.all(np.isfinite(probability)):
        raise ValueError("threshold selection requires finite probabilities")
    unique = sorted(set(float(value) for value in probability))
    none_threshold = float(np.nextafter(max(unique), math.inf))
    return sorted(set([0.0, *unique, none_threshold]))


def selected_helpful_recall(target: np.ndarray, selected: np.ndarray) -> float:
    target = np.asarray(target, dtype=bool)
    selected = np.asarray(selected, dtype=bool)
    positive_count = int(np.sum(target))
    return 0.0 if positive_count == 0 else float(np.sum(target & selected) / positive_count)


def select_regret_threshold(
    delta_pp: np.ndarray,
    helpful_target: np.ndarray,
    probability: np.ndarray,
    *,
    compute_cost_pp: float = 0.0,
    compute_costs_pp: Sequence[float] = (0.0, 0.05, 0.1, 0.25, 0.5),
) -> dict[str, Any]:
    """Select threshold on validation by net benefit, with conservative ties."""
    best: tuple[tuple[float, float, int, float], dict[str, Any]] | None = None
    for threshold in candidate_thresholds(probability):
        selected = probability >= threshold
        policy = common.policy_metrics_from_selection(
            delta_pp,
            helpful_target,
            selected,
            selection_rule="validation_regret_optimal",
            compute_costs_pp=compute_costs_pp,
            probability_threshold=threshold,
        )
        net_benefit = (
            policy["mean_realized_improvement_pp"]
            - compute_cost_pp * policy["selection_rate"]
        )
        precision = -1.0 if policy["helpful_precision"] is None else policy["helpful_precision"]
        key = (net_benefit, precision, -policy["selected_count"], threshold)
        if best is None or key > best[0]:
            best = (
                key,
                {
                    "threshold": threshold,
                    "selection_objective": "maximize_validation_net_benefit",
                    "selection_compute_cost_pp": compute_cost_pp,
                    "tie_break": "higher_helpful_precision_then_fewer_selected_then_higher_threshold",
                    "validation_policy": policy,
                },
            )
    if best is None:
        raise AssertionError("no regret threshold candidate")
    return best[1]


def select_precision_constrained_threshold(
    delta_pp: np.ndarray,
    helpful_target: np.ndarray,
    probability: np.ndarray,
    *,
    minimum_precision: float,
    compute_costs_pp: Sequence[float] = (0.0, 0.05, 0.1, 0.25, 0.5),
) -> dict[str, Any]:
    if not 0.0 < minimum_precision <= 1.0:
        raise ValueError("minimum precision must lie in (0,1]")
    best: tuple[tuple[float, float, float, int, float], dict[str, Any]] | None = None
    for threshold in candidate_thresholds(probability):
        selected = probability >= threshold
        policy = common.policy_metrics_from_selection(
            delta_pp,
            helpful_target,
            selected,
            selection_rule=f"validation_precision_at_least_{minimum_precision}",
            compute_costs_pp=compute_costs_pp,
            probability_threshold=threshold,
        )
        if policy["helpful_precision"] is None or policy["helpful_precision"] < minimum_precision:
            continue
        recall = selected_helpful_recall(helpful_target, selected)
        key = (
            recall,
            policy["mean_realized_improvement_pp"],
            policy["helpful_precision"],
            -policy["selected_count"],
            threshold,
        )
        if best is None or key > best[0]:
            best = (
                key,
                {
                    "threshold": threshold,
                    "minimum_precision": minimum_precision,
                    "selection_objective": "maximize_validation_helpful_recall_subject_to_precision",
                    "tie_break": "higher_net_benefit_then_precision_then_fewer_selected_then_higher_threshold",
                    "validation_helpful_recall": recall,
                    "validation_policy": policy,
                },
            )
    if best is not None:
        return best[1]
    threshold = float(np.nextafter(float(np.max(probability)), math.inf))
    policy = common.policy_metrics_from_selection(
        delta_pp,
        helpful_target,
        np.zeros(len(probability), dtype=bool),
        selection_rule=f"validation_precision_at_least_{minimum_precision}_no_feasible_nonempty_policy",
        compute_costs_pp=compute_costs_pp,
        probability_threshold=threshold,
    )
    return {
        "threshold": threshold,
        "minimum_precision": minimum_precision,
        "selection_objective": "no_nonempty_validation_policy_satisfied_precision",
        "validation_helpful_recall": 0.0,
        "validation_policy": policy,
    }


def binary_predictive_metrics(
    target: np.ndarray,
    probability: np.ndarray,
    *,
    selected: np.ndarray | None = None,
) -> dict[str, Any]:
    target = np.asarray(target, dtype=bool)
    probability = np.asarray(probability, dtype=float)
    if len(target) != len(probability):
        raise ValueError("binary metric arrays have inconsistent lengths")
    positive_count = int(np.sum(target))
    negative_count = len(target) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("binary ranking metrics require both classes")

    ascending_order = np.argsort(probability, kind="mergesort")
    ranks = np.empty(len(probability), dtype=float)
    start = 0
    while start < len(probability):
        end = start + 1
        while (
            end < len(probability)
            and probability[ascending_order[end]] == probability[ascending_order[start]]
        ):
            end += 1
        ranks[ascending_order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum_positive = float(np.sum(ranks[target]))
    auroc = (
        rank_sum_positive - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)

    descending_order = np.argsort(-probability, kind="mergesort")
    sorted_target = target[descending_order]
    sorted_probability = probability[descending_order]
    cumulative_true = np.cumsum(sorted_target)
    cumulative_false = np.cumsum(~sorted_target)
    distinct_ends = np.flatnonzero(
        np.r_[sorted_probability[1:] != sorted_probability[:-1], True]
    )
    precision_curve = cumulative_true[distinct_ends] / (
        cumulative_true[distinct_ends] + cumulative_false[distinct_ends]
    )
    recall_curve = cumulative_true[distinct_ends] / positive_count
    auprc = float(np.sum(np.diff(np.r_[0.0, recall_curve]) * precision_curve))
    result: dict[str, Any] = {
        "count": len(target),
        "positive_count": positive_count,
        "prevalence": float(np.mean(target)),
        "auroc": float(auroc),
        "auprc": auprc,
        "nll": binary_nll(target.astype(float), logit(probability)),
        "brier": float(np.mean((target.astype(float) - probability) ** 2)),
        "ece_10bin": common.expected_calibration_error(
            target.astype(float),
            probability,
            bins=10,
        ),
    }
    if selected is not None:
        selected = np.asarray(selected, dtype=bool)
        true_positive = int(np.sum(target & selected))
        selected_count = int(np.sum(selected))
        precision = 0.0 if selected_count == 0 else true_positive / selected_count
        recall = true_positive / positive_count
        f1 = (
            0.0
            if precision + recall == 0.0
            else 2.0 * precision * recall / (precision + recall)
        )
        result.update(
            {
                "selected_count": selected_count,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )
    return result


def ranking_capture_metrics(
    delta_pp: np.ndarray,
    probability: np.ndarray,
    *,
    top_counts: Sequence[int] = (10, 25, 50, 100),
) -> list[dict[str, Any]]:
    delta_pp = np.asarray(delta_pp, dtype=float)
    probability = np.asarray(probability, dtype=float)
    oracle_positive = np.maximum(delta_pp, 0.0)
    oracle_total = float(np.sum(oracle_positive))
    order = np.argsort(-probability, kind="mergesort")
    output: list[dict[str, Any]] = []
    for count in top_counts:
        selected_indices = order[: min(count, len(order))]
        selected_delta = delta_pp[selected_indices]
        output.append(
            {
                "top_k": int(count),
                "selected_count": len(selected_indices),
                "helpful_count": int(np.sum(selected_delta > 0.5)),
                "harmful_count": int(np.sum(selected_delta < -0.5)),
                "mean_realized_improvement_pp": float(np.sum(selected_delta) / len(delta_pp)),
                "selected_mean_delta_pp": float(np.mean(selected_delta)),
                "selected_median_delta_pp": float(np.median(selected_delta)),
                "oracle_improvement_captured": (
                    None
                    if oracle_total == 0.0
                    else float(np.sum(np.maximum(selected_delta, 0.0)) / oracle_total)
                ),
                "total_negative_uplift_incurred_pp": float(
                    np.sum(np.minimum(selected_delta, 0.0))
                ),
            }
        )
    return output


def outlier_sensitivity(
    delta_pp: np.ndarray,
    selected: np.ndarray,
) -> dict[str, Any]:
    delta_pp = np.asarray(delta_pp, dtype=float)
    selected = np.asarray(selected, dtype=bool)
    output: dict[str, Any] = {}
    positive_order = np.argsort(-delta_pp, kind="mergesort")
    for remove_count in (0, 1, 5):
        keep = np.ones(len(delta_pp), dtype=bool)
        if remove_count:
            keep[positive_order[:remove_count]] = False
        policy = common.policy_metrics_from_selection(
            delta_pp[keep],
            delta_pp[keep] > 0.5,
            selected[keep],
            selection_rule=f"remove_top_{remove_count}_positive_delta",
            compute_costs_pp=(0.0,),
        )
        output[f"remove_top_{remove_count}"] = policy
    lower, upper = np.quantile(delta_pp, [0.01, 0.99])
    winsorized = np.clip(delta_pp, lower, upper)
    output["winsorized_1_99"] = common.policy_metrics_from_selection(
        winsorized,
        delta_pp > 0.5,
        selected,
        selection_rule="winsorized_1_99_delta",
        compute_costs_pp=(0.0,),
    )
    return output
