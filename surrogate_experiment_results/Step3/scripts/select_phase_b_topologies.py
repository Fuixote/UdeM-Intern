#!/usr/bin/env python3
"""Select a stratified Step3 Phase-B topology set from Phase-A landscape summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "Step3"
    / "pairs20_ndd2"
    / "landscape"
    / "topology_landscape_summary.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "surrogate_experiment_results" / "Step3" / "pairs20_ndd2" / "screening"
)

DEFAULT_QUOTAS = {
    "sparse_simple": 25,
    "low_medium": 35,
    "medium_rich": 45,
    "rich": 37,
    "extreme": 18,
}

COMPLEXITY_ORDER = ["sparse_simple", "low_medium", "medium_rich", "rich", "extreme"]
STRUCTURAL_ORDER = ["cycle_chain", "chain_only", "cycle_only", "empty_candidate_structure"]
LANDSCAPE_ORDER = ["proxy_hard", "high_variance", "neutral", "proxy_aligned", "easy_control"]

OUTPUT_FIELDS = [
    "selection_rank",
    "topology_id",
    "complexity_bin",
    "structural_type",
    "landscape_regime",
    "screening_score",
    "selection_reason",
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
        if isinstance(value, float) and math.isnan(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(row: dict[str, Any], field: str, default: int = 0) -> int:
    return int(round(to_float(row, field, float(default))))


def complexity_bin(row: dict[str, Any]) -> str:
    num_candidates = to_int(row, "num_exchange_candidates")
    if num_candidates <= 8:
        return "sparse_simple"
    if num_candidates <= 20:
        return "low_medium"
    if num_candidates <= 40:
        return "medium_rich"
    if num_candidates <= 80:
        return "rich"
    return "extreme"


def structural_type(row: dict[str, Any]) -> str:
    cycles = to_int(row, "num_cycles_total")
    chains = to_int(row, "num_chains_total")
    if cycles > 0 and chains > 0:
        return "cycle_chain"
    if chains > 0:
        return "chain_only"
    if cycles > 0:
        return "cycle_only"
    return "empty_candidate_structure"


def landscape_regime(row: dict[str, Any]) -> str:
    distinct = to_int(row, "num_distinct_oracle_solutions")
    entropy = to_float(row, "oracle_solution_entropy")
    proxy_diff = to_float(row, "fraction_linear_proxy_differs_from_oracle")
    normalized_gap = to_float(row, "mean_linear_proxy_normalized_gap_to_oracle")

    if distinct <= 1 and entropy <= 0.05 and proxy_diff <= 0.05 and normalized_gap <= 0.005:
        return "easy_control"
    if proxy_diff >= 0.50 or normalized_gap >= 0.05:
        return "proxy_hard"
    if entropy >= 0.80 or distinct >= 4:
        return "high_variance"
    if proxy_diff <= 0.10 and normalized_gap <= 0.01:
        return "proxy_aligned"
    return "neutral"


def bounded(value: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return max(0.0, min(1.0, value / denominator))


def screening_score(row: dict[str, Any]) -> float:
    """Score used only for ordering candidates inside strata."""
    num_candidates = to_float(row, "num_exchange_candidates")
    cycles = to_float(row, "num_cycles_total")
    three_cycles = to_float(row, "num_3cycles")
    conflict_density = to_float(row, "candidate_conflict_density")
    entropy = to_float(row, "oracle_solution_entropy")
    proxy_diff = to_float(row, "fraction_linear_proxy_differs_from_oracle")
    normalized_gap = to_float(row, "mean_linear_proxy_normalized_gap_to_oracle")
    margin = to_float(row, "median_top1_top2_margin")
    candidate_pressure = to_float(row, "mean_candidates_per_vertex")

    near_tie_score = 1.0 / (1.0 + max(0.0, margin))
    score = (
        0.15 * bounded(num_candidates, 80.0)
        + 0.05 * min(1.0, cycles)
        + 0.03 * min(1.0, three_cycles)
        + 0.07 * bounded(candidate_pressure, 8.0)
        + 0.08 * max(0.0, min(1.0, conflict_density))
        + 0.20 * bounded(entropy, 1.2)
        + 0.22 * max(0.0, min(1.0, proxy_diff))
        + 0.15 * bounded(normalized_gap, 0.12)
        + 0.05 * near_tie_score
    )
    return round(score, 8)


def derive_selection_fields(row: dict[str, Any]) -> dict[str, Any]:
    output = dict(row)
    output["complexity_bin"] = complexity_bin(row)
    output["structural_type"] = structural_type(row)
    output["landscape_regime"] = landscape_regime(row)
    output["screening_score"] = screening_score(row)
    return output


def selection_sort_key(row: dict[str, Any]) -> tuple[float, tuple[int, int | str]]:
    return (-float(row["screening_score"]), int_or_text_key(row["topology_id"]))


def group_priority_key(group_key: tuple[str, str]) -> tuple[int, int, str, str]:
    structural, landscape = group_key
    structural_rank = STRUCTURAL_ORDER.index(structural) if structural in STRUCTURAL_ORDER else len(STRUCTURAL_ORDER)
    landscape_rank = LANDSCAPE_ORDER.index(landscape) if landscape in LANDSCAPE_ORDER else len(LANDSCAPE_ORDER)
    return (landscape_rank, structural_rank, structural, landscape)


def balanced_select_from_pool(
    pool: list[dict[str, Any]],
    quota: int,
    reason_prefix: str,
) -> list[dict[str, Any]]:
    if quota <= 0 or not pool:
        return []
    if len(pool) <= quota:
        selected_all = sorted(pool, key=selection_sort_key)
        for item in selected_all:
            item["selection_reason"] = f"{reason_prefix}:all_available"
        return selected_all

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in pool:
        key = (str(item["structural_type"]), str(item["landscape_regime"]))
        groups.setdefault(key, []).append(item)
    for key in groups:
        groups[key] = sorted(groups[key], key=selection_sort_key)

    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    group_keys = sorted(groups, key=group_priority_key)
    while len(selected) < quota and group_keys:
        progressed = False
        for key in group_keys:
            candidates = groups[key]
            while candidates and candidates[0]["topology_id"] in used:
                candidates.pop(0)
            if not candidates:
                continue
            item = candidates.pop(0)
            used.add(str(item["topology_id"]))
            item["selection_reason"] = f"{reason_prefix}:{key[0]}:{key[1]}"
            selected.append(item)
            progressed = True
            if len(selected) >= quota:
                break
        group_keys = [key for key in group_keys if groups[key]]
        if not progressed:
            break

    if len(selected) < quota:
        remaining = [
            item
            for item in sorted(pool, key=selection_sort_key)
            if item["topology_id"] not in used
        ]
        for item in remaining[: quota - len(selected)]:
            item["selection_reason"] = f"{reason_prefix}:quota_fill"
            selected.append(item)
            used.add(str(item["topology_id"]))
    return selected


def select_phase_b_topologies(
    rows: list[dict[str, Any]],
    quotas: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    quotas = quotas or DEFAULT_QUOTAS
    enriched = [derive_selection_fields(row) for row in rows]
    selected: list[dict[str, Any]] = []
    used: set[str] = set()

    for bin_name in COMPLEXITY_ORDER:
        quota = int(quotas.get(bin_name, 0))
        pool = [
            item
            for item in enriched
            if item["complexity_bin"] == bin_name and item["topology_id"] not in used
        ]
        bin_selected = balanced_select_from_pool(pool, quota, reason_prefix=f"quota_{bin_name}")
        for item in bin_selected:
            if item["topology_id"] not in used:
                selected.append(item)
                used.add(str(item["topology_id"]))

    target = sum(int(value) for value in quotas.values())
    if len(selected) < target:
        remaining = [
            item
            for item in sorted(enriched, key=selection_sort_key)
            if item["topology_id"] not in used
        ]
        for item in remaining[: target - len(selected)]:
            item["selection_reason"] = "global_quota_fill"
            selected.append(item)
            used.add(str(item["topology_id"]))

    for rank, item in enumerate(selected, start=1):
        item["selection_rank"] = rank
    return selected


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def count_by(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts = Counter(str(row.get(field, "")) for row in rows)
    return {key: counts[key] for key in sorted(counts)}


def strata_rows(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str, str]] = Counter(
        (
            str(row.get("complexity_bin")),
            str(row.get("structural_type")),
            str(row.get("landscape_regime")),
        )
        for row in selected
    )
    return [
        {
            "complexity_bin": complexity,
            "structural_type": structural,
            "landscape_regime": landscape,
            "count": count,
        }
        for (complexity, structural, landscape), count in sorted(counts.items())
    ]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_selection_outputs(
    output_dir: Path,
    selected: list[dict[str, Any]],
    quotas: dict[str, int] | None = None,
    input_path: Path | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "phase_b_topologies.csv", selected, OUTPUT_FIELDS)
    (output_dir / "phase_b_topology_ids.txt").write_text(
        "\n".join(str(row["topology_id"]) for row in selected) + "\n",
        encoding="utf-8",
    )
    strata = strata_rows(selected)
    write_csv(
        output_dir / "phase_b_strata_summary.csv",
        strata,
        ["complexity_bin", "structural_type", "landscape_regime", "count"],
    )
    summary = {
        "generated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "input_path": "" if input_path is None else str(input_path),
        "num_selected": len(selected),
        "target_quotas": quotas or {},
        "complexity_counts": count_by(selected, "complexity_bin"),
        "structural_counts": count_by(selected, "structural_type"),
        "landscape_counts": count_by(selected, "landscape_regime"),
        "strata_count": len(strata),
    }
    (output_dir / "phase_b_selection_summary.json").write_text(
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


def parse_quota_overrides(values: list[str] | None) -> dict[str, int]:
    quotas = dict(DEFAULT_QUOTAS)
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Quota override must be bin=count, got {value!r}")
        key, raw_count = value.split("=", 1)
        key = key.strip()
        if key not in DEFAULT_QUOTAS:
            raise ValueError(f"Unknown complexity bin {key!r}; expected one of {sorted(DEFAULT_QUOTAS)}")
        quotas[key] = int(raw_count)
        if quotas[key] < 0:
            raise ValueError(f"Quota for {key} must be non-negative")
    return quotas


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--quota",
        action="append",
        default=None,
        help="Override a default quota with bin=count, e.g. --quota rich=40",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    args.quotas = parse_quota_overrides(args.quota)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = read_csv_rows(args.input)
    if not rows:
        raise ValueError(f"No rows found in {args.input}")
    prepare_output_dir(args.output_dir, force=args.force)
    selected = select_phase_b_topologies(rows, quotas=args.quotas)
    write_selection_outputs(args.output_dir, selected, quotas=args.quotas, input_path=args.input)
    print(
        json.dumps(
            {
                "num_input_topologies": len(rows),
                "num_selected": len(selected),
                "complexity_counts": count_by(selected, "complexity_bin"),
                "landscape_counts": count_by(selected, "landscape_regime"),
                "output_dir": str(args.output_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
