#!/usr/bin/env python3
"""Shared protocol, label, metric, and I/O helpers for Experiment 07."""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Iterable, Sequence

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
STEP5_ROOT = EXPERIMENT_ROOT.parent
EXP3_ROOT = STEP5_ROOT / "experiment_03_formal_continuous_label_seed42_sample50"
EXP5_ROOT = STEP5_ROOT / "experiment_05_topology_gnn_regression"
DEFAULT_TARGET_TABLE = (
    EXP5_ROOT
    / "results"
    / "multiseed_completion1880"
    / "results"
    / "multiseed_targets.csv"
)
DEFAULT_FORMAL_SUMMARY = (
    EXP3_ROOT
    / "results"
    / "formal1000"
    / "results"
    / "weak_label_topology_summary.csv"
)
DEFAULT_INCIDENCE_GRAPHS = (
    EXP5_ROOT
    / "results"
    / "multiseed_completion1880"
    / "results"
    / "formal_topology_incidence_graphs.jsonl"
)
DEFAULT_OUTPUT_ROOT = EXPERIMENT_ROOT / "results"
LOCKED_TARGET_SHA256 = "e1b2cd6cfb575ce7924d07a665fc7c0755dc1758aad7ceb8c8a18d1d17a29f8b"
LOCKED_SUMMARY_SHA256 = "0e7b8554c63da668a2ad1b2ddd84386ec91970dd2a27d7986e417806af68deba"
LOCKED_INCIDENCE_GRAPH_SHA256 = "8a41232110bf7f151c0192c6723fe5e0d7f4b9dd83aa70aba7fa142e503fd522"

SEED_FIELDS = (
    "seed42_normalized_improvement_pp",
    "seed43_normalized_improvement_pp",
    "seed44_normalized_improvement_pp",
)
CLASS_LABELS = (
    "material_harmful",
    "neutral_or_uncertain",
    "material_helpful",
)
CLASS_TO_INDEX = {label: index for index, label in enumerate(CLASS_LABELS)}
SCALAR_FEATURES = (
    "num_vertices",
    "num_pairs",
    "num_ndds",
    "num_arcs",
    "max_cycle",
    "max_chain",
    "num_2cycles",
    "num_3cycles",
    "num_cycles_total",
    "num_chains_total",
    "num_feasible_candidates",
    "candidate_conflict_edges",
    "candidate_conflict_density",
    "mean_conflict_degree",
    "max_conflict_degree",
    "num_conflict_components",
    "largest_conflict_component_fraction",
    "num_vertices_in_any_candidate",
    "fraction_vertices_in_any_candidate",
    "mean_candidates_per_vertex",
    "max_candidates_per_vertex",
)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(output)


def atomic_write_csv(
    path: str | Path,
    rows: list[dict[str, Any]],
    fields: Sequence[str] | None = None,
) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fields or rows[0].keys())
    with tempfile.NamedTemporaryFile(
        "w",
        newline="",
        encoding="utf-8",
        dir=output.parent,
        delete=False,
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    temporary.replace(output)


def atomic_write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=output.parent,
        delete=False,
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False))
            handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(output)


def stable_key(seed: int, *parts: str) -> str:
    payload = "|".join([str(seed), *parts])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def topology_sort_key(topology_id: str) -> int:
    return int(topology_id.split("-")[-1])


def primary_label(mean_delta_pp: float, material_threshold_pp: float = 0.5) -> str:
    if mean_delta_pp > material_threshold_pp:
        return "material_helpful"
    if mean_delta_pp < -material_threshold_pp:
        return "material_harmful"
    return "neutral_or_uncertain"


def percentile_linear(sorted_values: Sequence[float], quantile: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0,1]")
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def exact_three_seed_bootstrap_ci(
    values: Sequence[float],
    *,
    alpha: float = 0.05,
) -> tuple[float, float]:
    if len(values) != 3:
        raise ValueError("the exact bootstrap protocol requires exactly three seed values")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0,1)")
    bootstrap_means = sorted(
        sum(values[index] for index in draw) / 3.0
        for draw in itertools.product(range(3), repeat=3)
    )
    return (
        percentile_linear(bootstrap_means, alpha / 2.0),
        percentile_linear(bootstrap_means, 1.0 - alpha / 2.0),
    )


def derive_label_row(
    source: dict[str, str],
    *,
    material_threshold_pp: float = 0.5,
    high_variance_std_pp: float = 0.5,
    bootstrap_alpha: float = 0.05,
) -> dict[str, Any]:
    topology_id = source["topology_id"]
    seed_values = [float(source[field]) for field in SEED_FIELDS]
    recorded_mean = float(source["formal_label_mean_pp"])
    recomputed_mean = sum(seed_values) / len(seed_values)
    if not math.isclose(recorded_mean, recomputed_mean, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{topology_id}:formal_mean_mismatch")
    recorded_std = float(source["label_uncertainty_std_pp"])
    recomputed_std = float(np.std(np.asarray(seed_values, dtype=float), ddof=0))
    if not math.isclose(recorded_std, recomputed_std, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{topology_id}:formal_std_mismatch")

    label = primary_label(recorded_mean, material_threshold_pp)
    expected_sign = 1 if label == "material_helpful" else -1 if label == "material_harmful" else 0
    agreement_count = sum(
        (value > 0.0 if expected_sign > 0 else value < 0.0)
        for value in seed_values
    ) if expected_sign else 0
    sign_agreement_passed = expected_sign == 0 or agreement_count >= 2
    ci_low, ci_high = exact_three_seed_bootstrap_ci(seed_values, alpha=bootstrap_alpha)
    bootstrap_ci_excludes_zero = (
        expected_sign == 0
        or (expected_sign > 0 and ci_low > 0.0)
        or (expected_sign < 0 and ci_high < 0.0)
    )
    high_variance = recorded_std > high_variance_std_pp

    sign_agreement_label = label if sign_agreement_passed else "neutral_or_uncertain"
    bootstrap_ci_label = label if bootstrap_ci_excludes_zero else "neutral_or_uncertain"
    confidence_passed = (
        expected_sign != 0
        and sign_agreement_passed
        and bootstrap_ci_excludes_zero
        and not high_variance
    )
    confidence_label = label if confidence_passed else "neutral_or_uncertain"
    reasons: list[str] = []
    if expected_sign != 0:
        if not sign_agreement_passed:
            reasons.append("fewer_than_two_seeds_match_material_direction")
        if not bootstrap_ci_excludes_zero:
            reasons.append("bootstrap_ci_crosses_zero")
        if high_variance:
            reasons.append("high_seed_std")
    confidence_state = (
        "neutral_by_effect_size"
        if expected_sign == 0
        else "confident_material"
        if confidence_passed
        else "uncertain_material"
    )

    return {
        "topology_id": topology_id,
        "topology_hash": source["topology_hash"],
        "feasible_set_hash": source["feasible_set_hash"],
        "test_hash": source["test_hash"],
        SEED_FIELDS[0]: seed_values[0],
        SEED_FIELDS[1]: seed_values[1],
        SEED_FIELDS[2]: seed_values[2],
        "formal_label_mean_pp": recorded_mean,
        "label_uncertainty_std_pp": recorded_std,
        "primary_label": label,
        "primary_label_index": CLASS_TO_INDEX[label],
        "target_is_material_helpful": int(label == "material_helpful"),
        "target_is_material_harmful": int(label == "material_harmful"),
        "material_direction_seed_agreement_count": agreement_count,
        "sign_agreement_passed": sign_agreement_passed,
        "sign_agreement_label": sign_agreement_label,
        "bootstrap_ci_low_pp": ci_low,
        "bootstrap_ci_high_pp": ci_high,
        "bootstrap_ci_excludes_zero": bootstrap_ci_excludes_zero,
        "bootstrap_ci_label": bootstrap_ci_label,
        "high_variance": high_variance,
        "confidence_passed": confidence_passed,
        "confidence_label": confidence_label,
        "confidence_label_index": CLASS_TO_INDEX[confidence_label],
        "confidence_state": confidence_state,
        "uncertainty_reasons": ";".join(reasons),
        "material_threshold_pp": material_threshold_pp,
        "high_variance_std_threshold_pp": high_variance_std_pp,
        "bootstrap_alpha": bootstrap_alpha,
    }


def label_counts(rows: Iterable[dict[str, Any]], field: str) -> dict[str, int]:
    counts = Counter(str(row[field]) for row in rows)
    return {label: counts[label] for label in CLASS_LABELS}


def expected_calibration_error(
    target: np.ndarray,
    probability: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    if len(target) != len(probability):
        raise ValueError("target and probability lengths differ")
    total = len(target)
    if total == 0:
        raise ValueError("calibration requires at least one sample")
    error = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        mask = (
            (probability >= lower)
            & (probability < upper if bin_index < bins - 1 else probability <= upper)
        )
        count = int(np.sum(mask))
        if count:
            accuracy = float(np.mean(target[mask]))
            confidence = float(np.mean(probability[mask]))
            error += count / total * abs(accuracy - confidence)
    return error


def top_label_calibration_error(
    target_index: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    predicted = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    correct = (predicted == target_index).astype(float)
    return expected_calibration_error(correct, confidence, bins=bins)


def policy_metrics(
    delta_pp: np.ndarray,
    helpful_target: np.ndarray,
    helpful_probability: np.ndarray,
    *,
    threshold: float = 0.5,
    compute_costs_pp: Sequence[float] = (0.0, 0.05, 0.1, 0.25, 0.5),
) -> dict[str, Any]:
    if not (len(delta_pp) == len(helpful_target) == len(helpful_probability)):
        raise ValueError("policy arrays have inconsistent lengths")
    selected = helpful_probability >= threshold
    return policy_metrics_from_selection(
        delta_pp,
        helpful_target,
        selected,
        selection_rule=f"helpful_probability_at_least_{threshold}",
        compute_costs_pp=compute_costs_pp,
        probability_threshold=threshold,
    )


def policy_metrics_from_selection(
    delta_pp: np.ndarray,
    helpful_target: np.ndarray,
    selected: np.ndarray,
    *,
    selection_rule: str,
    compute_costs_pp: Sequence[float] = (0.0, 0.05, 0.1, 0.25, 0.5),
    probability_threshold: float | None = None,
) -> dict[str, Any]:
    if not (len(delta_pp) == len(helpful_target) == len(selected)):
        raise ValueError("policy arrays have inconsistent lengths")
    selected = np.asarray(selected, dtype=bool)
    selected_count = int(np.sum(selected))
    realized = selected.astype(float) * delta_pp
    oracle_gain = np.maximum(delta_pp, 0.0)
    oracle_total = float(np.sum(oracle_gain))
    captured_positive = float(np.sum(selected.astype(float) * oracle_gain))
    net_gain = float(np.sum(realized))
    helpful_precision = (
        None
        if selected_count == 0
        else float(np.mean(helpful_target[selected]))
    )
    helpful_count = int(np.sum(helpful_target))
    helpful_recall = (
        None
        if helpful_count == 0
        else float(np.sum(helpful_target & selected) / helpful_count)
    )
    selected_delta = delta_pp[selected]
    selection_rate = selected_count / len(selected)
    return {
        "selection_rule": selection_rule,
        "helpful_probability_threshold": probability_threshold,
        "selected_count": selected_count,
        "selection_rate": selection_rate,
        "helpful_precision": helpful_precision,
        "helpful_recall": helpful_recall,
        "harmful_selected_count": int(np.sum(selected & (delta_pp < -0.5))),
        "positive_delta_selected_count": int(np.sum(selected & (delta_pp > 0.0))),
        "selected_positive_delta_fraction": (
            None if selected_count == 0 else float(np.mean(selected_delta > 0.0))
        ),
        "selected_median_delta_pp": (
            None if selected_count == 0 else float(np.median(selected_delta))
        ),
        "worst_selected_delta_pp": (
            None if selected_count == 0 else float(np.min(selected_delta))
        ),
        "total_negative_uplift_incurred_pp": float(
            np.sum(np.minimum(selected_delta, 0.0))
        ),
        "total_positive_uplift_captured_pp": captured_positive,
        "mean_realized_improvement_pp": float(np.mean(realized)),
        "policy_regret_pp": float(np.mean(oracle_gain - realized)),
        "oracle_positive_improvement_sum_pp": oracle_total,
        "oracle_improvement_captured": (
            None if oracle_total == 0.0 else captured_positive / oracle_total
        ),
        "net_oracle_fraction": None if oracle_total == 0.0 else net_gain / oracle_total,
        "compute_adjusted_net_benefit": [
            {
                "extra_compute_cost_pp": float(cost),
                "net_benefit_pp_per_topology": float(np.mean(realized) - cost * selection_rate),
            }
            for cost in compute_costs_pp
        ],
    }
