#!/usr/bin/env python3
"""Aggregate Step3 Phase-B training results and select Phase-C candidates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STATUS_CSV = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "Step3"
    / "pairs20_ndd2"
    / "phase_b"
    / "results"
    / "phase_b_training_status_e100.csv"
)
DEFAULT_PHASE_B_TOPOLOGIES = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "Step3"
    / "pairs20_ndd2"
    / "screening"
    / "phase_b_topologies.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "Step3"
    / "pairs20_ndd2"
    / "phase_b"
    / "selection"
)

COMPLETED_STATUSES = {"success", "skipped"}
DEFAULT_SELECTION_QUOTAS = {
    "helpful": 4,
    "harmful": 2,
    "neutral": 2,
    "control": 2,
}
OUTCOME_ORDER = ["helpful", "harmful", "neutral", "control", "mixed", "incomplete"]

DEFAULT_THRESHOLDS = {
    "tolerance": 1e-9,
    "helpful_mean_gap": 0.5,
    "helpful_fraction_better": 0.5,
    "helpful_max_fraction_worse": 0.1,
    "harmful_mean_gap": -0.5,
    "harmful_fraction_worse": 0.5,
    "harmful_max_fraction_better": 0.1,
    "neutral_abs_mean_gap": 0.1,
    "neutral_max_fraction_better": 0.1,
    "neutral_max_fraction_worse": 0.1,
    "neutral_mean_abs_gap": 0.1,
    "neutral_max_abs_gap": 0.5,
}

SUMMARY_BASE_FIELDS = [
    "topology_id",
    "phase_b_outcome",
    "total_jobs",
    "completed_jobs",
    "success_jobs",
    "skipped_jobs",
    "failed_jobs",
    "metric_jobs",
    "train_seed_min",
    "train_seed_max",
    "train_sample_count",
    "validation_sample_count",
    "test_sample_count",
    "mean_spoplus_improvement_gap",
    "median_spoplus_improvement_gap",
    "std_spoplus_improvement_gap",
    "min_spoplus_improvement_gap",
    "max_spoplus_improvement_gap",
    "max_abs_spoplus_improvement_gap",
    "mean_abs_spoplus_improvement_gap",
    "p10_spoplus_improvement_gap",
    "p90_spoplus_improvement_gap",
    "better_count",
    "worse_count",
    "tied_count",
    "fraction_spoplus_better",
    "fraction_spoplus_worse",
    "fraction_tied",
    "mean_test_gap_2stage",
    "mean_test_gap_spoplus",
    "mean_test_normalized_gap_2stage",
    "mean_test_normalized_gap_spoplus",
    "mean_spoplus_improvement_normalized_gap",
    "mean_success_elapsed_seconds",
    "median_success_elapsed_seconds",
]

DESCRIPTOR_FIELDS = [
    "phase_b_selection_rank",
    "complexity_bin",
    "structural_type",
    "landscape_regime",
    "screening_score",
    "phase_b_selection_reason",
    "num_exchange_candidates",
    "num_cycles_total",
    "num_3cycles",
    "num_chains_total",
    "candidate_conflict_density",
    "mean_candidates_per_vertex",
    "num_distinct_oracle_solutions",
    "oracle_solution_entropy",
    "dominant_oracle_solution_fraction",
    "fraction_linear_proxy_differs_from_oracle",
    "mean_linear_proxy_normalized_gap_to_oracle",
    "median_top1_top2_margin",
    "mean_pairwise_oracle_jaccard",
]

PHASE_C_EXTRA_FIELDS = ["phase_c_rank", "phase_c_selection_reason"]


def int_or_text_key(value: Any) -> tuple[int, int | str]:
    text = str(value)
    if text.startswith("G-"):
        text = text[2:]
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def to_float(row: dict[str, Any], field: str, default: float = 0.0) -> float:
    value = row.get(field, default)
    if value in (None, "", "None", "nan"):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(result) else result


def to_optional_float(row: dict[str, Any], field: str) -> float | None:
    value = row.get(field)
    if value in (None, "", "None", "nan"):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(result) else result


def to_int(row: dict[str, Any], field: str, default: int = 0) -> int:
    return int(round(to_float(row, field, float(default))))


def mean_or_zero(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def median_or_zero(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def pstdev_or_zero(values: list[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) >= 2 else 0.0


def percentile_or_zero(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered[low])
    weight = rank - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def common_value(rows: list[dict[str, Any]], field: str) -> str:
    counts: Counter[str] = Counter(
        str(row.get(field, "")) for row in rows if row.get(field, "") not in (None, "")
    )
    if not counts:
        return ""
    value, _ = counts.most_common(1)[0]
    return value


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_phase_b_descriptors(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_csv_rows(path)
    return {str(row["topology_id"]): row for row in rows}


def summarize_training_results(
    status_rows: list[dict[str, Any]],
    tolerance: float = DEFAULT_THRESHOLDS["tolerance"],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in status_rows:
        topology_id = str(row.get("topology_id", "")).strip()
        if topology_id:
            grouped[topology_id].append(row)

    summaries: list[dict[str, Any]] = []
    for topology_id in sorted(grouped, key=int_or_text_key):
        rows = grouped[topology_id]
        completed_rows = [
            row for row in rows if str(row.get("status", "")).strip().lower() in COMPLETED_STATUSES
        ]
        success_rows = [row for row in rows if str(row.get("status", "")).strip().lower() == "success"]
        skipped_rows = [row for row in rows if str(row.get("status", "")).strip().lower() == "skipped"]
        failed_rows = [row for row in rows if row not in completed_rows]
        metric_rows = [
            row for row in completed_rows if to_optional_float(row, "spoplus_improvement_gap") is not None
        ]

        improvements = [
            float(to_optional_float(row, "spoplus_improvement_gap")) for row in metric_rows
        ]
        normalized_improvements = [
            float(to_optional_float(row, "spoplus_improvement_normalized_gap"))
            for row in metric_rows
            if to_optional_float(row, "spoplus_improvement_normalized_gap") is not None
        ]
        better_count = sum(1 for value in improvements if value > tolerance)
        worse_count = sum(1 for value in improvements if value < -tolerance)
        tied_count = sum(1 for value in improvements if abs(value) <= tolerance)
        metric_count = len(improvements)
        elapsed_success = [
            value
            for row in success_rows
            if (value := to_optional_float(row, "elapsed_seconds")) is not None and value > 0.0
        ]
        train_seeds = [
            to_int(row, "train_seed")
            for row in completed_rows
            if row.get("train_seed", "") not in (None, "")
        ]

        summary = {
            "topology_id": topology_id,
            "total_jobs": len(rows),
            "completed_jobs": len(completed_rows),
            "success_jobs": len(success_rows),
            "skipped_jobs": len(skipped_rows),
            "failed_jobs": len(failed_rows),
            "metric_jobs": metric_count,
            "train_seed_min": min(train_seeds) if train_seeds else "",
            "train_seed_max": max(train_seeds) if train_seeds else "",
            "train_sample_count": common_value(completed_rows, "train_sample_count"),
            "validation_sample_count": common_value(completed_rows, "validation_sample_count"),
            "test_sample_count": common_value(completed_rows, "test_sample_count"),
            "mean_spoplus_improvement_gap": mean_or_zero(improvements),
            "median_spoplus_improvement_gap": median_or_zero(improvements),
            "std_spoplus_improvement_gap": pstdev_or_zero(improvements),
            "min_spoplus_improvement_gap": min(improvements) if improvements else 0.0,
            "max_spoplus_improvement_gap": max(improvements) if improvements else 0.0,
            "max_abs_spoplus_improvement_gap": max((abs(value) for value in improvements), default=0.0),
            "mean_abs_spoplus_improvement_gap": mean_or_zero([abs(value) for value in improvements]),
            "p10_spoplus_improvement_gap": percentile_or_zero(improvements, 0.10),
            "p90_spoplus_improvement_gap": percentile_or_zero(improvements, 0.90),
            "better_count": better_count,
            "worse_count": worse_count,
            "tied_count": tied_count,
            "fraction_spoplus_better": better_count / metric_count if metric_count else 0.0,
            "fraction_spoplus_worse": worse_count / metric_count if metric_count else 0.0,
            "fraction_tied": tied_count / metric_count if metric_count else 0.0,
            "mean_test_gap_2stage": mean_or_zero(
                [
                    value
                    for row in metric_rows
                    if (value := to_optional_float(row, "test_mean_decision_gap_2stage")) is not None
                ]
            ),
            "mean_test_gap_spoplus": mean_or_zero(
                [
                    value
                    for row in metric_rows
                    if (value := to_optional_float(row, "test_mean_decision_gap_spoplus")) is not None
                ]
            ),
            "mean_test_normalized_gap_2stage": mean_or_zero(
                [
                    value
                    for row in metric_rows
                    if (value := to_optional_float(row, "test_mean_normalized_gap_2stage")) is not None
                ]
            ),
            "mean_test_normalized_gap_spoplus": mean_or_zero(
                [
                    value
                    for row in metric_rows
                    if (value := to_optional_float(row, "test_mean_normalized_gap_spoplus")) is not None
                ]
            ),
            "mean_spoplus_improvement_normalized_gap": mean_or_zero(normalized_improvements),
            "mean_success_elapsed_seconds": mean_or_zero(elapsed_success),
            "median_success_elapsed_seconds": median_or_zero(elapsed_success),
        }
        summaries.append(summary)
    return summaries


def classify_phase_b_outcome(
    row: dict[str, Any],
    thresholds: dict[str, float] | None = None,
) -> str:
    thresholds = thresholds or DEFAULT_THRESHOLDS
    metric_jobs = to_int(row, "metric_jobs")
    if metric_jobs <= 0:
        return "incomplete"

    mean_gap = to_float(row, "mean_spoplus_improvement_gap")
    frac_better = to_float(row, "fraction_spoplus_better")
    frac_worse = to_float(row, "fraction_spoplus_worse")
    mean_abs = to_float(row, "mean_abs_spoplus_improvement_gap")
    max_abs = to_float(row, "max_abs_spoplus_improvement_gap")

    if (
        mean_gap >= thresholds["helpful_mean_gap"]
        and frac_better >= thresholds["helpful_fraction_better"]
        and frac_worse <= thresholds["helpful_max_fraction_worse"]
    ):
        return "helpful"
    if (
        mean_gap <= thresholds["harmful_mean_gap"]
        and frac_worse >= thresholds["harmful_fraction_worse"]
        and frac_better <= thresholds["harmful_max_fraction_better"]
    ):
        return "harmful"

    neutral_like = (
        abs(mean_gap) <= thresholds["neutral_abs_mean_gap"]
        and frac_better <= thresholds["neutral_max_fraction_better"]
        and frac_worse <= thresholds["neutral_max_fraction_worse"]
        and mean_abs <= thresholds["neutral_mean_abs_gap"]
        and max_abs <= thresholds["neutral_max_abs_gap"]
    )
    if neutral_like:
        landscape = str(row.get("landscape_regime", ""))
        complexity = str(row.get("complexity_bin", ""))
        if landscape in {"easy_control", "proxy_aligned"} or complexity == "sparse_simple":
            return "control"
        return "neutral"
    return "mixed"


def merge_phase_b_descriptors(
    summaries: list[dict[str, Any]],
    descriptors: dict[str, dict[str, Any]],
    thresholds: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    merged_rows: list[dict[str, Any]] = []
    for summary in summaries:
        topology_id = str(summary["topology_id"])
        merged = dict(summary)
        descriptor = descriptors.get(topology_id, {})
        for key, value in descriptor.items():
            if key == "topology_id":
                continue
            if key == "selection_rank":
                merged["phase_b_selection_rank"] = value
            elif key == "selection_reason":
                merged["phase_b_selection_reason"] = value
            else:
                merged[key] = value
        merged["phase_b_outcome"] = classify_phase_b_outcome(merged, thresholds=thresholds)
        merged_rows.append(merged)
    return merged_rows


def count_by(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts = Counter(str(row.get(field, "")) for row in rows)
    return {key: counts[key] for key in sorted(counts)}


def outcome_sort_key(row: dict[str, Any], outcome: str) -> tuple[Any, ...]:
    topology_key = int_or_text_key(row.get("topology_id", ""))
    mean_gap = to_float(row, "mean_spoplus_improvement_gap")
    frac_better = to_float(row, "fraction_spoplus_better")
    frac_worse = to_float(row, "fraction_spoplus_worse")
    max_abs = to_float(row, "max_abs_spoplus_improvement_gap")
    mean_abs = to_float(row, "mean_abs_spoplus_improvement_gap")
    if outcome == "helpful":
        return (-mean_gap, -frac_better, frac_worse, topology_key)
    if outcome == "harmful":
        return (mean_gap, -frac_worse, frac_better, topology_key)
    if outcome in {"neutral", "control"}:
        return (abs(mean_gap), mean_abs, max_abs, topology_key)
    return (-max_abs, -abs(mean_gap), topology_key)


def select_phase_c_candidates(
    rows: list[dict[str, Any]],
    quotas: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    quotas = quotas or DEFAULT_SELECTION_QUOTAS
    selected: list[dict[str, Any]] = []
    used: set[str] = set()

    ordered_outcomes = [outcome for outcome in OUTCOME_ORDER if outcome in quotas]
    ordered_outcomes.extend(outcome for outcome in quotas if outcome not in ordered_outcomes)
    for outcome in ordered_outcomes:
        quota = int(quotas.get(outcome, 0))
        if quota <= 0:
            continue
        pool = [
            row
            for row in rows
            if str(row.get("phase_b_outcome")) == outcome and str(row.get("topology_id")) not in used
        ]
        for row in sorted(pool, key=lambda item: outcome_sort_key(item, outcome))[:quota]:
            candidate = dict(row)
            candidate["phase_c_selection_reason"] = f"quota_{outcome}"
            selected.append(candidate)
            used.add(str(row["topology_id"]))

    for rank, row in enumerate(selected, start=1):
        row["phase_c_rank"] = rank
    return selected


def ordered_fieldnames(rows: list[dict[str, Any]], preferred: list[str]) -> list[str]:
    present = set()
    for row in rows:
        present.update(row)
    fields = [field for field in preferred if field in present]
    fields.extend(sorted(present - set(fields)))
    return fields


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def outcome_topology_ids(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("phase_b_outcome", ""))].append(str(row.get("topology_id", "")))
    return {
        outcome: sorted(ids, key=int_or_text_key)
        for outcome, ids in sorted(grouped.items(), key=lambda item: item[0])
    }


def write_aggregation_outputs(
    output_dir: Path,
    summaries: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    thresholds: dict[str, float] | None = None,
    quotas: dict[str, int] | None = None,
    status_csv: Path | None = None,
    descriptor_csv: Path | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_fields = ordered_fieldnames(summaries, SUMMARY_BASE_FIELDS + DESCRIPTOR_FIELDS)
    candidate_fields = ordered_fieldnames(
        selected,
        PHASE_C_EXTRA_FIELDS + SUMMARY_BASE_FIELDS + DESCRIPTOR_FIELDS,
    )
    write_csv(output_dir / "phase_b_topology_training_summary.csv", summaries, summary_fields)
    write_csv(output_dir / "phase_c_candidate_topologies.csv", selected, candidate_fields)
    (output_dir / "phase_c_topology_ids.txt").write_text(
        "\n".join(str(row["topology_id"]) for row in selected) + ("\n" if selected else ""),
        encoding="utf-8",
    )

    outcome_counts = count_by(summaries, "phase_b_outcome")
    selected_counts = count_by(selected, "phase_b_outcome")
    topology_ids = outcome_topology_ids(summaries)
    selected_ids = outcome_topology_ids(selected)

    counts_payload = {
        "generated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "outcome_counts": outcome_counts,
        "selected_outcome_counts": selected_counts,
        "topology_ids_by_outcome": topology_ids,
        "selected_topology_ids_by_outcome": selected_ids,
    }
    (output_dir / "phase_b_outcome_counts.json").write_text(
        json.dumps(counts_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result_summary = {
        "generated_at": counts_payload["generated_at"],
        "status_csv": "" if status_csv is None else str(status_csv),
        "phase_b_topologies_csv": "" if descriptor_csv is None else str(descriptor_csv),
        "num_topologies": len(summaries),
        "num_phase_c_candidates": len(selected),
        "outcome_counts": outcome_counts,
        "selected_outcome_counts": selected_counts,
        "selection_quotas": quotas or {},
        "classification_thresholds": thresholds or DEFAULT_THRESHOLDS,
        "mean_spoplus_improvement_gap": mean_or_zero(
            [to_float(row, "mean_spoplus_improvement_gap") for row in summaries]
        ),
        "median_spoplus_improvement_gap": median_or_zero(
            [to_float(row, "mean_spoplus_improvement_gap") for row in summaries]
        ),
        "fraction_helpful_topologies": outcome_counts.get("helpful", 0) / len(summaries)
        if summaries
        else 0.0,
        "fraction_harmful_topologies": outcome_counts.get("harmful", 0) / len(summaries)
        if summaries
        else 0.0,
    }
    (output_dir / "phase_b_result_summary.json").write_text(
        json.dumps(result_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def prepare_output_dir(output_dir: Path, force: bool) -> None:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            raise FileExistsError(f"{output_dir} is not empty; pass --force to overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def parse_quota_overrides(values: list[str] | None) -> dict[str, int]:
    quotas = dict(DEFAULT_SELECTION_QUOTAS)
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Quota override must be outcome=count, got {value!r}")
        key, raw_count = value.split("=", 1)
        key = key.strip()
        if key not in OUTCOME_ORDER:
            raise ValueError(f"Unknown outcome {key!r}; expected one of {OUTCOME_ORDER}")
        quotas[key] = int(raw_count)
        if quotas[key] < 0:
            raise ValueError(f"Quota for {key} must be non-negative")
    return quotas


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-csv", type=Path, default=DEFAULT_STATUS_CSV)
    parser.add_argument("--phase-b-topologies", type=Path, default=DEFAULT_PHASE_B_TOPOLOGIES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--quota",
        action="append",
        default=None,
        help="Override a default Phase-C quota with outcome=count, e.g. --quota helpful=5",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    args.quotas = parse_quota_overrides(args.quota)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    status_rows = read_csv_rows(args.status_csv)
    if not status_rows:
        raise ValueError(f"No status rows found in {args.status_csv}")
    descriptors = read_phase_b_descriptors(args.phase_b_topologies)
    prepare_output_dir(args.output_dir, force=args.force)
    summaries = merge_phase_b_descriptors(
        summarize_training_results(status_rows),
        descriptors,
        thresholds=DEFAULT_THRESHOLDS,
    )
    selected = select_phase_c_candidates(summaries, quotas=args.quotas)
    write_aggregation_outputs(
        args.output_dir,
        summaries,
        selected,
        thresholds=DEFAULT_THRESHOLDS,
        quotas=args.quotas,
        status_csv=args.status_csv,
        descriptor_csv=args.phase_b_topologies,
    )
    print(
        json.dumps(
            {
                "num_status_rows": len(status_rows),
                "num_topologies": len(summaries),
                "outcome_counts": count_by(summaries, "phase_b_outcome"),
                "num_phase_c_candidates": len(selected),
                "phase_c_topology_ids": [row["topology_id"] for row in selected],
                "output_dir": str(args.output_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
