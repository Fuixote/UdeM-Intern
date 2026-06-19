#!/usr/bin/env python3
"""Audit Step3 Phase-C review topology candidates using Phase-B artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
STEP3_ROOT = PROJECT_ROOT / "surrogate_experiment_results" / "Step3" / "pairs20_ndd2"
DEFAULT_PHASE_C_IDS = STEP3_ROOT / "phase_b" / "selection" / "phase_c_topology_ids.txt"
DEFAULT_PHASE_B_SUMMARY = (
    STEP3_ROOT / "phase_b" / "selection" / "phase_b_topology_training_summary.csv"
)
DEFAULT_STATUS_CSV = STEP3_ROOT / "phase_b" / "results" / "phase_b_training_status_e100.csv"
DEFAULT_DATASET_DIR = STEP3_ROOT / "phase_b" / "datasets"
DEFAULT_RUNS_DIR = STEP3_ROOT / "phase_b" / "runs_e100"
DEFAULT_OUTPUT_DIR = STEP3_ROOT / "phase_c" / "preflight"
DEFAULT_REVIEW_EXTRAS = ["G-47", "G-79", "G-72"]
COMPLETED_STATUSES = {"success", "skipped"}
TOLERANCE = 1e-9

TOPOLOGY_FIELDS = [
    "topology_id",
    "review_role",
    "audit_pattern",
    "audit_interpretation",
    "phase_b_outcome",
    "complexity_bin",
    "structural_type",
    "landscape_regime",
    "num_exchange_candidates",
    "num_cycles_total",
    "num_3cycles",
    "num_chains_total",
    "status_rows",
    "metric_jobs",
    "success_jobs",
    "skipped_jobs",
    "failed_jobs",
    "train_seed_count",
    "train_seed_min",
    "train_seed_max",
    "train_sample_rows",
    "train_label_hash_count",
    "train_label_hashes_unique",
    "expected_train_label_hash_count",
    "validation_label_hash_count",
    "test_label_hash_count",
    "unique_train_seed_label_digest_count",
    "unique_theta_2stage",
    "unique_theta_spoplus",
    "theta_2stage_1_std",
    "theta_2stage_2_std",
    "theta_spoplus_1_std",
    "theta_spoplus_2_std",
    "selected_epoch_values_2stage",
    "spoplus_selected_epoch_values",
    "fraction_2stage_selected_at_max_epoch",
    "fraction_spoplus_selected_at_max_epoch",
    "unique_improvement_gap_count",
    "mean_improvement_gap",
    "std_improvement_gap",
    "min_improvement_gap",
    "max_improvement_gap",
    "mean_normalized_improvement_gap",
    "std_normalized_improvement_gap",
    "better_count",
    "worse_count",
    "tied_count",
    "fraction_better",
    "fraction_worse",
    "fraction_tied",
    "mean_improvement_gap_ci95_low",
    "mean_improvement_gap_ci95_high",
    "mean_normalized_improvement_gap_ci95_low",
    "mean_normalized_improvement_gap_ci95_high",
    "fraction_better_wilson95_low",
    "fraction_better_wilson95_high",
    "fraction_worse_wilson95_low",
    "fraction_worse_wilson95_high",
    "unique_test_gap_signature_2stage",
    "unique_test_gap_signature_spoplus",
    "missing_run_artifact_count",
]

SEED_FIELDS = [
    "topology_id",
    "train_seed",
    "status",
    "seed_outcome",
    "train_label_hash_count",
    "train_label_hash_digest",
    "theta_2stage_1",
    "theta_2stage_2",
    "theta_spoplus_1",
    "theta_spoplus_2",
    "selected_epoch_2stage",
    "selected_epoch_spoplus",
    "max_epochs",
    "test_gap_2stage",
    "test_gap_spoplus",
    "improvement_gap",
    "normalized_improvement_gap",
    "test_gap_signature_2stage",
    "test_gap_signature_spoplus",
    "run_artifacts_present",
]


def int_or_text_key(value: Any) -> tuple[int, int | str]:
    text = str(value)
    if text.startswith("G-"):
        text = text[2:]
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def to_float(value: Any, default: float = 0.0) -> float:
    if value in (None, "", "None", "nan"):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(result) else result


def to_optional_float(value: Any) -> float | None:
    if value in (None, "", "None", "nan"):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(result) else result


def mean_or_zero(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def pstdev_or_zero(values: list[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) >= 2 else 0.0


def normal_ci95(values: list[float]) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    mean = mean_or_zero(values)
    if len(values) <= 1:
        return (mean, mean)
    half_width = 1.96 * pstdev_or_zero(values) / math.sqrt(len(values))
    return (mean - half_width, mean + half_width)


def wilson_ci95(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    z = 1.96
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def stable_digest(parts: list[str]) -> str:
    payload = "\n".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_read_json(path: Path) -> Any:
    if not Path(path).exists():
        return None
    return read_json(Path(path))


def review_topology_ids(phase_c_ids_path: Path, extras: list[str] | None = None) -> list[str]:
    ids: list[str] = []
    if Path(phase_c_ids_path).exists():
        ids.extend(
            line.strip()
            for line in Path(phase_c_ids_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    ids.extend(extras or [])
    seen: set[str] = set()
    ordered: list[str] = []
    for topology_id in ids:
        if topology_id not in seen:
            ordered.append(topology_id)
            seen.add(topology_id)
    return ordered


def read_phase_b_summary(path: Path) -> dict[str, dict[str, Any]]:
    if not Path(path).exists():
        return {}
    return {str(row["topology_id"]): row for row in read_csv_rows(path)}


def read_status_by_topology(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not Path(path).exists():
        return {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_csv_rows(path):
        grouped[str(row.get("topology_id", ""))].append(row)
    return grouped


def role_for_topology(topology_id: str, phase_c_ids: set[str], extras: set[str]) -> str:
    in_auto = topology_id in phase_c_ids
    in_extra = topology_id in extras
    if in_auto and in_extra:
        return "auto_phase_c+seed_sensitive_extra"
    if in_auto:
        return "auto_phase_c"
    if in_extra:
        return "seed_sensitive_extra"
    return "manual_review"


def samples_index(samples_csv: Path) -> dict[str, Any]:
    if not Path(samples_csv).exists():
        return {
            "rows": [],
            "train_rows": [],
            "validation_rows": [],
            "test_rows": [],
            "train_hashes_by_seed": {},
        }
    rows = read_csv_rows(samples_csv)
    train_rows = [row for row in rows if row.get("split") == "train"]
    validation_rows = [row for row in rows if row.get("split") == "validation"]
    test_rows = [row for row in rows if row.get("split") == "test"]
    by_seed: dict[int, list[str]] = defaultdict(list)
    for row in train_rows:
        seed_text = row.get("train_seed", "")
        if seed_text in ("", None):
            continue
        by_seed[int(seed_text)].append(str(row.get("label_hash", "")))
    return {
        "rows": rows,
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "test_rows": test_rows,
        "train_hashes_by_seed": by_seed,
    }


def theta_tuple(payload: Any) -> tuple[float, ...] | None:
    if not isinstance(payload, dict):
        return None
    theta = payload.get("theta")
    if not isinstance(theta, list):
        return None
    return tuple(round(float(value), 12) for value in theta)


def theta_components(theta_values: list[tuple[float, ...]], index: int) -> list[float]:
    return [theta[index] for theta in theta_values if len(theta) > index]


def read_test_summary_metrics(path: Path) -> dict[str, dict[str, Any]]:
    payload = safe_read_json(path)
    if not isinstance(payload, list):
        return {}
    metrics: dict[str, dict[str, Any]] = {}
    for row in payload:
        if isinstance(row, dict) and row.get("method"):
            metrics[str(row["method"])] = row
    return metrics


def read_gap_signatures(path: Path) -> dict[str, str]:
    if not Path(path).exists():
        return {}
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in read_csv_rows(path):
        method = str(row.get("method", ""))
        grouped[method].append(
            "|".join(
                [
                    str(row.get("selection_metric", "")),
                    str(row.get("graph", "")),
                    str(row.get("optimal_obj", "")),
                    str(row.get("achieved_obj", "")),
                    str(row.get("gap", "")),
                ]
            )
        )
    return {method: stable_digest(sorted(parts)) for method, parts in grouped.items()}


def seed_outcome(improvement: float | None, tolerance: float = TOLERANCE) -> str:
    if improvement is None:
        return "missing"
    if improvement > tolerance:
        return "better"
    if improvement < -tolerance:
        return "worse"
    return "tie"


def audit_seed(
    *,
    topology_id: str,
    train_seed: int,
    status_row: dict[str, Any],
    train_hashes: list[str],
    run_dir: Path,
) -> dict[str, Any]:
    weights_dir = run_dir / "model_weights"
    metrics_dir = run_dir / "metrics"
    two_stage_weight = safe_read_json(weights_dir / "2stage_best_by_validation_mse_loss.json")
    spoplus_weight = safe_read_json(weights_dir / "spoplus_best_by_validation_spoplus_loss.json")
    early = safe_read_json(metrics_dir / "early_stopping.json")
    summary_metrics = read_test_summary_metrics(metrics_dir / "test_summary.json")
    gap_signatures = read_gap_signatures(metrics_dir / "test_per_graph.csv")

    theta_2stage = theta_tuple(two_stage_weight)
    theta_spoplus = theta_tuple(spoplus_weight)
    improvement = to_optional_float(status_row.get("spoplus_improvement_gap"))
    normalized_improvement = to_optional_float(
        status_row.get("spoplus_improvement_normalized_gap")
    )
    gap_2stage = to_optional_float(status_row.get("test_mean_decision_gap_2stage"))
    gap_spoplus = to_optional_float(status_row.get("test_mean_decision_gap_spoplus"))
    if gap_2stage is None:
        gap_2stage = to_optional_float(summary_metrics.get("2stage", {}).get("test_mean_decision_gap"))
    if gap_spoplus is None:
        gap_spoplus = to_optional_float(summary_metrics.get("spoplus", {}).get("test_mean_decision_gap"))

    selected_epoch_2stage = (
        "" if not isinstance(two_stage_weight, dict) else two_stage_weight.get("selected_epoch", "")
    )
    selected_epoch_spoplus = (
        "" if not isinstance(spoplus_weight, dict) else spoplus_weight.get("selected_epoch", "")
    )
    max_epochs = "" if not isinstance(early, dict) else early.get("max_epochs", "")

    return {
        "topology_id": topology_id,
        "train_seed": train_seed,
        "status": status_row.get("status", ""),
        "seed_outcome": seed_outcome(improvement),
        "train_label_hash_count": len(set(train_hashes)),
        "train_label_hash_digest": stable_digest(sorted(train_hashes)) if train_hashes else "",
        "theta_2stage_1": "" if not theta_2stage or len(theta_2stage) < 1 else theta_2stage[0],
        "theta_2stage_2": "" if not theta_2stage or len(theta_2stage) < 2 else theta_2stage[1],
        "theta_spoplus_1": "" if not theta_spoplus or len(theta_spoplus) < 1 else theta_spoplus[0],
        "theta_spoplus_2": "" if not theta_spoplus or len(theta_spoplus) < 2 else theta_spoplus[1],
        "selected_epoch_2stage": selected_epoch_2stage,
        "selected_epoch_spoplus": selected_epoch_spoplus,
        "max_epochs": max_epochs,
        "test_gap_2stage": "" if gap_2stage is None else gap_2stage,
        "test_gap_spoplus": "" if gap_spoplus is None else gap_spoplus,
        "improvement_gap": "" if improvement is None else improvement,
        "normalized_improvement_gap": "" if normalized_improvement is None else normalized_improvement,
        "test_gap_signature_2stage": gap_signatures.get("2stage", ""),
        "test_gap_signature_spoplus": gap_signatures.get("spoplus", ""),
        "run_artifacts_present": bool(two_stage_weight and spoplus_weight and summary_metrics),
    }


def selected_epoch_values(seed_rows: list[dict[str, Any]], field: str) -> str:
    values = sorted(
        {
            int(float(row[field]))
            for row in seed_rows
            if row.get(field, "") not in ("", None)
        }
    )
    return "|".join(str(value) for value in values)


def fraction_selected_at_max_epoch(seed_rows: list[dict[str, Any]], field: str) -> float:
    usable = [
        row
        for row in seed_rows
        if row.get(field, "") not in ("", None) and row.get("max_epochs", "") not in ("", None)
    ]
    if not usable:
        return 0.0
    return sum(
        1
        for row in usable
        if int(float(row[field])) >= int(float(row["max_epochs"]))
    ) / len(usable)


def outcome_pattern(
    *,
    mean_improvement: float,
    unique_improvement_count: int,
    fraction_better: float,
    fraction_worse: float,
    fraction_tied: float,
) -> str:
    if unique_improvement_count == 1:
        if fraction_better == 1.0:
            return "stable_helpful_saturated"
        if fraction_worse == 1.0:
            return "stable_harmful_saturated"
        if fraction_tied == 1.0:
            return "exact_tie_saturated"
    if fraction_better > 0.0 and fraction_worse > 0.0:
        return "mixed_bidirectional"
    if fraction_better > 0.0 and fraction_tied > 0.0:
        return "variable_helpful"
    if fraction_worse > 0.0 and fraction_tied > 0.0:
        return "variable_harmful"
    if abs(mean_improvement) <= TOLERANCE:
        return "neutral"
    return "mixed"


def audit_interpretation(
    *,
    train_label_hashes_unique: bool,
    metric_jobs: int,
    unique_theta_2stage: int,
    unique_theta_spoplus: int,
    unique_improvement_count: int,
) -> str:
    if (
        train_label_hashes_unique
        and metric_jobs > 0
        and unique_theta_2stage == metric_jobs
        and unique_theta_spoplus == metric_jobs
        and unique_improvement_count == 1
    ):
        return "different_labels_and_theta_same_test_outcome"
    if train_label_hashes_unique and unique_improvement_count > 1:
        return "seed_sensitive_test_outcome"
    if not train_label_hashes_unique:
        return "duplicate_train_label_hashes_present"
    return "insufficient_or_partial_artifacts"


def summarize_topology(
    *,
    topology_id: str,
    review_role: str,
    phase_b_row: dict[str, Any],
    status_rows: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
    sample_info: dict[str, Any],
) -> dict[str, Any]:
    completed_rows = [
        row
        for row in status_rows
        if str(row.get("status", "")).strip().lower() in COMPLETED_STATUSES
    ]
    metric_rows = [
        row
        for row in completed_rows
        if to_optional_float(row.get("spoplus_improvement_gap")) is not None
    ]
    improvements = [to_float(row.get("spoplus_improvement_gap")) for row in metric_rows]
    normalized = [
        to_float(row.get("spoplus_improvement_normalized_gap"))
        for row in metric_rows
        if to_optional_float(row.get("spoplus_improvement_normalized_gap")) is not None
    ]
    better_count = sum(1 for value in improvements if value > TOLERANCE)
    worse_count = sum(1 for value in improvements if value < -TOLERANCE)
    tied_count = sum(1 for value in improvements if abs(value) <= TOLERANCE)
    metric_jobs = len(improvements)
    mean_ci = normal_ci95(improvements)
    norm_ci = normal_ci95(normalized)
    better_ci = wilson_ci95(better_count, metric_jobs)
    worse_ci = wilson_ci95(worse_count, metric_jobs)

    train_hashes = [str(row.get("label_hash", "")) for row in sample_info["train_rows"]]
    train_hashes = [value for value in train_hashes if value]
    train_seed_digests = {
        row.get("train_label_hash_digest")
        for row in seed_rows
        if row.get("train_label_hash_digest")
    }
    theta_2stage = [
        (to_float(row.get("theta_2stage_1")), to_float(row.get("theta_2stage_2")))
        for row in seed_rows
        if row.get("theta_2stage_1", "") not in ("", None)
        and row.get("theta_2stage_2", "") not in ("", None)
    ]
    theta_spoplus = [
        (to_float(row.get("theta_spoplus_1")), to_float(row.get("theta_spoplus_2")))
        for row in seed_rows
        if row.get("theta_spoplus_1", "") not in ("", None)
        and row.get("theta_spoplus_2", "") not in ("", None)
    ]
    unique_improvement_count = len({round(value, 12) for value in improvements})
    train_seeds = [
        int(row.get("train_seed"))
        for row in seed_rows
        if row.get("train_seed", "") not in ("", None)
    ]
    train_label_hashes_unique = len(set(train_hashes)) == len(train_hashes)
    expected_train_hash_count = sum(
        int(row.get("train_label_hash_count", 0))
        for row in seed_rows
        if row.get("train_label_hash_count", "") not in ("", None)
    )
    pattern = outcome_pattern(
        mean_improvement=mean_or_zero(improvements),
        unique_improvement_count=unique_improvement_count,
        fraction_better=better_count / metric_jobs if metric_jobs else 0.0,
        fraction_worse=worse_count / metric_jobs if metric_jobs else 0.0,
        fraction_tied=tied_count / metric_jobs if metric_jobs else 0.0,
    )
    interpretation = audit_interpretation(
        train_label_hashes_unique=train_label_hashes_unique,
        metric_jobs=metric_jobs,
        unique_theta_2stage=len(set(theta_2stage)),
        unique_theta_spoplus=len(set(theta_spoplus)),
        unique_improvement_count=unique_improvement_count,
    )

    return {
        "topology_id": topology_id,
        "review_role": review_role,
        "audit_pattern": pattern,
        "audit_interpretation": interpretation,
        "phase_b_outcome": phase_b_row.get("phase_b_outcome", ""),
        "complexity_bin": phase_b_row.get("complexity_bin", ""),
        "structural_type": phase_b_row.get("structural_type", ""),
        "landscape_regime": phase_b_row.get("landscape_regime", ""),
        "num_exchange_candidates": phase_b_row.get("num_exchange_candidates", ""),
        "num_cycles_total": phase_b_row.get("num_cycles_total", ""),
        "num_3cycles": phase_b_row.get("num_3cycles", ""),
        "num_chains_total": phase_b_row.get("num_chains_total", ""),
        "status_rows": len(status_rows),
        "metric_jobs": metric_jobs,
        "success_jobs": sum(1 for row in status_rows if row.get("status") == "success"),
        "skipped_jobs": sum(1 for row in status_rows if row.get("status") == "skipped"),
        "failed_jobs": sum(
            1
            for row in status_rows
            if str(row.get("status", "")).strip().lower() not in COMPLETED_STATUSES
        ),
        "train_seed_count": len(set(train_seeds)),
        "train_seed_min": min(train_seeds) if train_seeds else "",
        "train_seed_max": max(train_seeds) if train_seeds else "",
        "train_sample_rows": len(sample_info["train_rows"]),
        "train_label_hash_count": len(set(train_hashes)),
        "train_label_hashes_unique": train_label_hashes_unique,
        "expected_train_label_hash_count": expected_train_hash_count,
        "validation_label_hash_count": len(
            {row.get("label_hash") for row in sample_info["validation_rows"] if row.get("label_hash")}
        ),
        "test_label_hash_count": len(
            {row.get("label_hash") for row in sample_info["test_rows"] if row.get("label_hash")}
        ),
        "unique_train_seed_label_digest_count": len(train_seed_digests),
        "unique_theta_2stage": len(set(theta_2stage)),
        "unique_theta_spoplus": len(set(theta_spoplus)),
        "theta_2stage_1_std": pstdev_or_zero(theta_components(theta_2stage, 0)),
        "theta_2stage_2_std": pstdev_or_zero(theta_components(theta_2stage, 1)),
        "theta_spoplus_1_std": pstdev_or_zero(theta_components(theta_spoplus, 0)),
        "theta_spoplus_2_std": pstdev_or_zero(theta_components(theta_spoplus, 1)),
        "selected_epoch_values_2stage": selected_epoch_values(seed_rows, "selected_epoch_2stage"),
        "spoplus_selected_epoch_values": selected_epoch_values(seed_rows, "selected_epoch_spoplus"),
        "fraction_2stage_selected_at_max_epoch": fraction_selected_at_max_epoch(
            seed_rows, "selected_epoch_2stage"
        ),
        "fraction_spoplus_selected_at_max_epoch": fraction_selected_at_max_epoch(
            seed_rows, "selected_epoch_spoplus"
        ),
        "unique_improvement_gap_count": unique_improvement_count,
        "mean_improvement_gap": mean_or_zero(improvements),
        "std_improvement_gap": pstdev_or_zero(improvements),
        "min_improvement_gap": min(improvements) if improvements else 0.0,
        "max_improvement_gap": max(improvements) if improvements else 0.0,
        "mean_normalized_improvement_gap": mean_or_zero(normalized),
        "std_normalized_improvement_gap": pstdev_or_zero(normalized),
        "better_count": better_count,
        "worse_count": worse_count,
        "tied_count": tied_count,
        "fraction_better": better_count / metric_jobs if metric_jobs else 0.0,
        "fraction_worse": worse_count / metric_jobs if metric_jobs else 0.0,
        "fraction_tied": tied_count / metric_jobs if metric_jobs else 0.0,
        "mean_improvement_gap_ci95_low": mean_ci[0],
        "mean_improvement_gap_ci95_high": mean_ci[1],
        "mean_normalized_improvement_gap_ci95_low": norm_ci[0],
        "mean_normalized_improvement_gap_ci95_high": norm_ci[1],
        "fraction_better_wilson95_low": better_ci[0],
        "fraction_better_wilson95_high": better_ci[1],
        "fraction_worse_wilson95_low": worse_ci[0],
        "fraction_worse_wilson95_high": worse_ci[1],
        "unique_test_gap_signature_2stage": len(
            {row.get("test_gap_signature_2stage") for row in seed_rows if row.get("test_gap_signature_2stage")}
        ),
        "unique_test_gap_signature_spoplus": len(
            {row.get("test_gap_signature_spoplus") for row in seed_rows if row.get("test_gap_signature_spoplus")}
        ),
        "missing_run_artifact_count": sum(
            1 for row in seed_rows if not bool(row.get("run_artifacts_present"))
        ),
    }


def audit_review_topologies(
    topology_ids: list[str],
    *,
    dataset_dir: Path,
    runs_dir: Path,
    status_csv: Path,
    phase_b_summary_csv: Path,
    phase_c_ids_path: Path | None = None,
    extras: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if phase_c_ids_path is None:
        phase_c_ids = set(topology_ids)
    else:
        phase_c_ids = set(review_topology_ids(phase_c_ids_path, extras=[]))
    extras_set = set(extras or [])
    phase_b_summary = read_phase_b_summary(phase_b_summary_csv)
    status_by_topology = read_status_by_topology(status_csv)
    topology_rows: list[dict[str, Any]] = []
    all_seed_rows: list[dict[str, Any]] = []

    for topology_id in topology_ids:
        status_rows = sorted(
            status_by_topology.get(topology_id, []),
            key=lambda row: int(row.get("train_seed", 0) or 0),
        )
        sample_info = samples_index(Path(dataset_dir) / topology_id / "samples.csv")
        train_hashes_by_seed = sample_info["train_hashes_by_seed"]
        seed_rows: list[dict[str, Any]] = []
        for status_row in status_rows:
            status = str(status_row.get("status", "")).strip().lower()
            if status not in COMPLETED_STATUSES:
                continue
            train_seed = int(status_row.get("train_seed", 0) or 0)
            seed_row = audit_seed(
                topology_id=topology_id,
                train_seed=train_seed,
                status_row=status_row,
                train_hashes=train_hashes_by_seed.get(train_seed, []),
                run_dir=Path(runs_dir) / topology_id / f"train_seed={train_seed:06d}",
            )
            seed_rows.append(seed_row)
        review_role = role_for_topology(topology_id, phase_c_ids, extras_set)
        topology_rows.append(
            summarize_topology(
                topology_id=topology_id,
                review_role=review_role,
                phase_b_row=phase_b_summary.get(topology_id, {}),
                status_rows=status_rows,
                seed_rows=seed_rows,
                sample_info=sample_info,
            )
        )
        all_seed_rows.extend(seed_rows)
    return topology_rows, all_seed_rows


def ordered_fieldnames(rows: list[dict[str, Any]], preferred: list[str]) -> list[str]:
    present = set()
    for row in rows:
        present.update(row)
    fields = [field for field in preferred if field in present]
    fields.extend(sorted(present - set(fields)))
    return fields


def write_audit_outputs(
    output_dir: Path,
    topology_rows: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "phase_c_review_topology_audit.csv",
        topology_rows,
        ordered_fieldnames(topology_rows, TOPOLOGY_FIELDS),
    )
    write_csv(
        output_dir / "phase_c_review_seed_audit.csv",
        seed_rows,
        ordered_fieldnames(seed_rows, SEED_FIELDS),
    )
    (output_dir / "phase_c_review_topology_ids.txt").write_text(
        "\n".join(str(row["topology_id"]) for row in topology_rows)
        + ("\n" if topology_rows else ""),
        encoding="utf-8",
    )
    counts = Counter(str(row.get("audit_pattern", "")) for row in topology_rows)
    interpretations = Counter(str(row.get("audit_interpretation", "")) for row in topology_rows)
    summary = {
        "generated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "num_review_topologies": len(topology_rows),
        "num_seed_rows": len(seed_rows),
        "audit_pattern_counts": {key: counts[key] for key in sorted(counts)},
        "audit_interpretation_counts": {
            key: interpretations[key] for key in sorted(interpretations)
        },
        "topology_ids": [row["topology_id"] for row in topology_rows],
    }
    (output_dir / "phase_c_review_audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def prepare_output_dir(output_dir: Path, force: bool) -> None:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            raise FileExistsError(f"{output_dir} is not empty; pass --force to overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-c-ids", type=Path, default=DEFAULT_PHASE_C_IDS)
    parser.add_argument("--phase-b-summary", type=Path, default=DEFAULT_PHASE_B_SUMMARY)
    parser.add_argument("--status-csv", type=Path, default=DEFAULT_STATUS_CSV)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--extra-topology",
        action="append",
        default=None,
        help="Additional review topology id. Defaults to G-47, G-79, G-72.",
    )
    parser.add_argument("--no-default-extras", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    extras = [] if args.no_default_extras else list(DEFAULT_REVIEW_EXTRAS)
    extras.extend(args.extra_topology or [])
    topology_ids = review_topology_ids(args.phase_c_ids, extras=extras)
    if not topology_ids:
        raise ValueError("No review topology ids found")
    prepare_output_dir(args.output_dir, force=args.force)
    topology_rows, seed_rows = audit_review_topologies(
        topology_ids,
        dataset_dir=args.dataset_dir,
        runs_dir=args.runs_dir,
        status_csv=args.status_csv,
        phase_b_summary_csv=args.phase_b_summary,
        phase_c_ids_path=args.phase_c_ids,
        extras=extras,
    )
    write_audit_outputs(args.output_dir, topology_rows, seed_rows)
    print(
        json.dumps(
            {
                "num_review_topologies": len(topology_rows),
                "num_seed_rows": len(seed_rows),
                "audit_pattern_counts": Counter(
                    str(row.get("audit_pattern", "")) for row in topology_rows
                ),
                "output_dir": str(args.output_dir),
                "topology_ids": [row["topology_id"] for row in topology_rows],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
