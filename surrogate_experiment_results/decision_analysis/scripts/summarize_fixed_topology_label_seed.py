#!/usr/bin/env python3
"""Summarize fixed-topology Step2c label-seed replay rows."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DIR = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "decision_analysis"
    / "results"
    / "fixed_topology_label_seed"
)
DEFAULT_INPUT = DEFAULT_DIR / "step2c_fixed_topology_label_seed_1000_rows.csv"
DEFAULT_SUMMARY = DEFAULT_DIR / "step2c_fixed_topology_label_seed_1000_summary.csv"
DEFAULT_READOUT = DEFAULT_DIR / "step2c_fixed_topology_label_seed_1000_readout.md"

SUMMARY_FIELDS = [
    "base_graph_id",
    "label_seed_count",
    "unique_topology_hashes",
    "unique_label_hashes",
    "case_c_preserved_rate",
    "case_c_preserved_se",
    "case_c_preserved_ci95_halfwidth",
    "spoplus_better_rate",
    "correction_persistence_rate",
    "correction_persistence_se",
    "correction_persistence_ci95_halfwidth",
    "rank2_promotion_persistence_rate",
    "rank2_promotion_persistence_se",
    "rank2_promotion_persistence_ci95_halfwidth",
    "mean_2stage_rank1_gap",
    "mean_2stage_rank2_gap",
    "mean_spoplus_rank1_gap",
    "mean_spoplus_rank2_gap",
    "mean_rank1_gap_reduction",
    "median_rank1_gap_reduction",
    "oracle_solution_unique_count",
    "mean_oracle_jaccard_to_first_seed",
]


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def finite_mean(values: list[float]) -> float:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(clean) / len(clean)) if clean else float("nan")


def finite_median(values: list[float]) -> float:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        return float("nan")
    midpoint = len(clean) // 2
    if len(clean) % 2:
        return clean[midpoint]
    return 0.5 * (clean[midpoint - 1] + clean[midpoint])


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def rate_se(rate: float, count: int) -> float:
    if count <= 0 or not math.isfinite(rate):
        return float("nan")
    return math.sqrt(max(rate * (1.0 - rate), 0.0) / count)


def ci95_halfwidth(rate: float, count: int) -> float:
    se = rate_se(rate, count)
    return 1.96 * se if math.isfinite(se) else float("nan")


def edge_set(signature: str) -> set[str]:
    return {token for token in str(signature).split("|") if token}


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def rows_by_graph_seed(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["base_graph_id"], row["label_seed"])].append(row)
    return grouped


def rank_lookup(seed_rows: list[dict[str, str]]) -> dict[tuple[str, int], dict[str, str]]:
    lookup: dict[tuple[str, int], dict[str, str]] = {}
    for row in seed_rows:
        key = (row["method_label"], int(finite_float(row["solution_rank"], default=-1)))
        lookup[key] = row
    return lookup


def gap(
    lookup: dict[tuple[str, int], dict[str, str]],
    method_label: str,
    rank: int,
) -> float:
    row = lookup.get((method_label, rank))
    if row is None:
        return float("nan")
    return finite_float(row.get("normalized_gap_to_oracle"))


def signature(
    lookup: dict[tuple[str, int], dict[str, str]],
    method_label: str,
    rank: int,
) -> str:
    row = lookup.get((method_label, rank))
    if row is None:
        return ""
    return str(row.get("solution_arc_key_signature", ""))


def case_c_preserved(
    two_stage_rank1_gap: float,
    spoplus_rank1_gap: float,
    two_stage_min_gap: float,
    spoplus_max_gap: float,
    min_gap_reduction: float,
) -> bool:
    return (
        math.isfinite(two_stage_rank1_gap)
        and math.isfinite(spoplus_rank1_gap)
        and two_stage_rank1_gap >= two_stage_min_gap
        and spoplus_rank1_gap <= spoplus_max_gap
        and (two_stage_rank1_gap - spoplus_rank1_gap) >= min_gap_reduction
    )


def correction_persists(
    two_stage_rank1_gap: float,
    two_stage_rank2_gap: float,
    spoplus_rank1_gap: float,
    bad_gap_threshold: float,
    good_gap_threshold: float,
    min_gap_reduction: float,
) -> bool:
    return (
        math.isfinite(two_stage_rank1_gap)
        and math.isfinite(two_stage_rank2_gap)
        and math.isfinite(spoplus_rank1_gap)
        and two_stage_rank1_gap >= bad_gap_threshold
        and two_stage_rank2_gap >= bad_gap_threshold
        and spoplus_rank1_gap <= good_gap_threshold
        and (two_stage_rank1_gap - spoplus_rank1_gap) >= min_gap_reduction
    )


def rank2_promotion_persists(
    two_stage_rank1_gap: float,
    two_stage_rank2_gap: float,
    spoplus_rank1_gap: float,
    two_stage_rank2_signature: str,
    spoplus_rank1_signature: str,
    bad_gap_threshold: float,
    good_gap_threshold: float,
) -> bool:
    return (
        math.isfinite(two_stage_rank1_gap)
        and math.isfinite(two_stage_rank2_gap)
        and math.isfinite(spoplus_rank1_gap)
        and two_stage_rank1_gap >= bad_gap_threshold
        and two_stage_rank2_gap <= good_gap_threshold
        and spoplus_rank1_gap <= good_gap_threshold
        and bool(two_stage_rank2_signature)
        and two_stage_rank2_signature == spoplus_rank1_signature
    )


def summarize(
    rows: list[dict[str, str]],
    *,
    two_stage_min_gap: float = 0.05,
    spoplus_max_gap: float = 0.02,
    min_gap_reduction: float = 0.05,
) -> list[dict[str, Any]]:
    grouped = rows_by_graph_seed(rows)
    by_graph: dict[str, list[list[dict[str, str]]]] = defaultdict(list)
    for (graph_id, _), seed_rows in grouped.items():
        by_graph[graph_id].append(seed_rows)

    summaries: list[dict[str, Any]] = []
    for graph_id in sorted(by_graph):
        seed_groups = sorted(
            by_graph[graph_id],
            key=lambda group: int(finite_float(group[0]["label_seed"], default=-1)),
        )
        two_stage_rank1_gaps: list[float] = []
        two_stage_rank2_gaps: list[float] = []
        spoplus_rank1_gaps: list[float] = []
        spoplus_rank2_gaps: list[float] = []
        rank1_reductions: list[float] = []
        case_flags: list[bool] = []
        better_flags: list[bool] = []
        correction_flags: list[bool] = []
        promotion_flags: list[bool] = []
        topology_hashes: set[str] = set()
        label_hashes: set[str] = set()
        oracle_signatures: list[str] = []

        for seed_rows in seed_groups:
            lookup = rank_lookup(seed_rows)
            ts1 = gap(lookup, "2stage_val_mse", 1)
            ts2 = gap(lookup, "2stage_val_mse", 2)
            sp1 = gap(lookup, "spoplus_val_spoplus_loss", 1)
            sp2 = gap(lookup, "spoplus_val_spoplus_loss", 2)

            two_stage_rank1_gaps.append(ts1)
            two_stage_rank2_gaps.append(ts2)
            spoplus_rank1_gaps.append(sp1)
            spoplus_rank2_gaps.append(sp2)
            rank1_reductions.append(ts1 - sp1)
            better_flags.append(math.isfinite(ts1) and math.isfinite(sp1) and sp1 < ts1)

            case_from_row = bool_value(seed_rows[0].get("case_c_signature_for_label_seed", ""))
            case_flags.append(
                case_from_row
                or case_c_preserved(
                    ts1,
                    sp1,
                    two_stage_min_gap,
                    spoplus_max_gap,
                    min_gap_reduction,
                )
            )
            correction_flags.append(
                correction_persists(
                    ts1,
                    ts2,
                    sp1,
                    two_stage_min_gap,
                    spoplus_max_gap,
                    min_gap_reduction,
                )
            )
            promotion_flags.append(
                rank2_promotion_persists(
                    ts1,
                    ts2,
                    sp1,
                    signature(lookup, "2stage_val_mse", 2),
                    signature(lookup, "spoplus_val_spoplus_loss", 1),
                    two_stage_min_gap,
                    spoplus_max_gap,
                )
            )

            topology_hashes.update(row.get("topology_hash", "") for row in seed_rows)
            label_hashes.update(row.get("label_hash", "") for row in seed_rows)
            oracle_signatures.append(seed_rows[0].get("oracle_arc_key_signature", ""))

        seed_count = len(seed_groups)
        case_rate = finite_mean([float(flag) for flag in case_flags])
        correction_rate = finite_mean([float(flag) for flag in correction_flags])
        promotion_rate = finite_mean([float(flag) for flag in promotion_flags])
        first_oracle = edge_set(oracle_signatures[0]) if oracle_signatures else set()
        oracle_jaccards = [jaccard(edge_set(sig), first_oracle) for sig in oracle_signatures]

        summaries.append(
            {
                "base_graph_id": graph_id,
                "label_seed_count": seed_count,
                "unique_topology_hashes": len({value for value in topology_hashes if value}),
                "unique_label_hashes": len({value for value in label_hashes if value}),
                "case_c_preserved_rate": case_rate,
                "case_c_preserved_se": rate_se(case_rate, seed_count),
                "case_c_preserved_ci95_halfwidth": ci95_halfwidth(case_rate, seed_count),
                "spoplus_better_rate": finite_mean([float(flag) for flag in better_flags]),
                "correction_persistence_rate": correction_rate,
                "correction_persistence_se": rate_se(correction_rate, seed_count),
                "correction_persistence_ci95_halfwidth": ci95_halfwidth(correction_rate, seed_count),
                "rank2_promotion_persistence_rate": promotion_rate,
                "rank2_promotion_persistence_se": rate_se(promotion_rate, seed_count),
                "rank2_promotion_persistence_ci95_halfwidth": ci95_halfwidth(
                    promotion_rate,
                    seed_count,
                ),
                "mean_2stage_rank1_gap": finite_mean(two_stage_rank1_gaps),
                "mean_2stage_rank2_gap": finite_mean(two_stage_rank2_gaps),
                "mean_spoplus_rank1_gap": finite_mean(spoplus_rank1_gaps),
                "mean_spoplus_rank2_gap": finite_mean(spoplus_rank2_gaps),
                "mean_rank1_gap_reduction": finite_mean(rank1_reductions),
                "median_rank1_gap_reduction": finite_median(rank1_reductions),
                "oracle_solution_unique_count": len(set(oracle_signatures)),
                "mean_oracle_jaccard_to_first_seed": finite_mean(oracle_jaccards),
            }
        )

    return summaries


def write_readout(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Step2c Fixed-Topology Label-Seed Robustness",
        "",
        "Scope: topology and trained Step2c model weights are fixed; only Step2c label_seed varies.",
        "",
    ]
    for row in summary_rows:
        graph_id = row["base_graph_id"]
        lines.extend(
            [
                f"## {graph_id}",
                f"- label seeds: {row['label_seed_count']}",
                f"- unique topology hashes: {row['unique_topology_hashes']}",
                f"- unique label hashes: {row['unique_label_hashes']}",
                f"- Case C preserved rate: {float(row['case_c_preserved_rate']):.3f}"
                f" +/- {float(row['case_c_preserved_ci95_halfwidth']):.3f}",
                f"- SPO+ better rate: {float(row['spoplus_better_rate']):.3f}",
                f"- correction persistence rate: {float(row['correction_persistence_rate']):.3f}"
                f" +/- {float(row['correction_persistence_ci95_halfwidth']):.3f}",
                f"- rank-2 promotion persistence rate: "
                f"{float(row['rank2_promotion_persistence_rate']):.3f}"
                f" +/- {float(row['rank2_promotion_persistence_ci95_halfwidth']):.3f}",
                f"- mean rank-1 gap reduction: {float(row['mean_rank1_gap_reduction']):.4f}",
                f"- oracle unique solutions: {row['oracle_solution_unique_count']}",
                f"- mean oracle Jaccard to first seed: "
                f"{float(row['mean_oracle_jaccard_to_first_seed']):.3f}",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Summarize fixed-topology Step2c label-seed audit rows."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--readout-output", type=Path, default=DEFAULT_READOUT)
    parser.add_argument("--two-stage-min-gap", type=float, default=0.05)
    parser.add_argument("--spoplus-max-gap", type=float, default=0.02)
    parser.add_argument("--min-gap-reduction", type=float, default=0.05)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    rows = read_csv_rows(args.input)
    summary_rows = summarize(
        rows,
        two_stage_min_gap=args.two_stage_min_gap,
        spoplus_max_gap=args.spoplus_max_gap,
        min_gap_reduction=args.min_gap_reduction,
    )
    write_csv(args.summary_output, summary_rows, SUMMARY_FIELDS)
    write_readout(args.readout_output, summary_rows)
    print(f"Saved {len(summary_rows)} summary rows to {args.summary_output}")
    print(f"Saved readout to {args.readout_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
