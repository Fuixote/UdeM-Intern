#!/usr/bin/env python3
"""Summarize fixed graph-instance x subset_seed model robustness audits.

Input is the top-K output from ``compute_second_best_solutions.py``. The script
pivots rank-1/rank-2/top-K solutions for 2stage and SPO+, then writes:

  1. one row per graph x subset_seed with Case C and mechanism flags;
  2. one summary row per graph for all seeds and discovery-seed-excluded seeds.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "decision_analysis"
    / "results"
    / "fixed_graph_model_seed"
)
DEFAULT_INPUT = DEFAULT_OUTPUT_DIR / "step2c_g392_g1560_all50_top5_second_best.csv"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "step2c_g392_g1560_all50_seed_audit.csv"
DEFAULT_SUMMARY_OUTPUT = (
    DEFAULT_OUTPUT_DIR / "step2c_g392_g1560_all50_seed_audit_summary.csv"
)

TWO_STAGE_LABEL = "2stage_val_mse"
SPOPLUS_LABEL = "spoplus_val_spoplus_loss"

BINARY_METRICS = [
    "spo_better",
    "meaningful_spo_benefit",
    "strict_case_c",
    "strong_case_c",
    "correction_preserved",
    "spo_rank1_equals_2stage_rank2",
    "rank2_promotion_preserved",
    "spo_rank1_in_2stage_topk",
    "topk_promotion_preserved",
]

CONTINUOUS_METRICS = [
    "delta_pp",
    "two_stage_rank1_gap_pct",
    "spoplus_rank1_gap_pct",
    "two_stage_rank2_gap_pct",
    "spoplus_rank2_gap_pct",
]

AUDIT_FIELDS = [
    "regime",
    "graph_id",
    "subset_seed",
    "discovery_seed",
    "is_discovery_seed",
    "top_k",
    "two_stage_rank1_gap_pct",
    "two_stage_rank2_gap_pct",
    "spoplus_rank1_gap_pct",
    "spoplus_rank2_gap_pct",
    "delta_pp",
    "spo_better",
    "meaningful_spo_benefit",
    "strict_case_c",
    "strong_case_c",
    "correction_preserved",
    "spo_rank1_equals_2stage_rank2",
    "rank2_promotion_preserved",
    "spo_rank1_in_2stage_topk",
    "spoplus_matches_2stage_rank",
    "topk_promotion_preserved",
    "two_stage_rank1_signature",
    "two_stage_rank2_signature",
    "spoplus_rank1_signature",
]

SUMMARY_FIELDS = [
    "regime",
    "graph_id",
    "seed_filter",
    "seed_count",
    "discovery_seed",
    "excluded_seed",
    "subset_seed_min",
    "subset_seed_max",
]

for metric in BINARY_METRICS:
    SUMMARY_FIELDS.extend(
        [
            f"{metric}_count",
            f"{metric}_rate",
            f"{metric}_wilson95_low",
            f"{metric}_wilson95_high",
        ]
    )

for metric in CONTINUOUS_METRICS:
    SUMMARY_FIELDS.extend(
        [
            f"mean_{metric}",
            f"median_{metric}",
            f"q25_{metric}",
            f"q75_{metric}",
        ]
    )


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_float(value: object) -> float:
    if value is None or str(value).strip() == "":
        return float("nan")
    return float(value)


def finite_values(values: list[float]) -> list[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


def finite_mean(values: list[float]) -> float:
    clean = finite_values(values)
    if not clean:
        return float("nan")
    return float(sum(clean) / len(clean))


def finite_median(values: list[float]) -> float:
    return finite_quantile(values, 0.5)


def finite_quantile(values: list[float], q: float) -> float:
    clean = sorted(finite_values(values))
    if not clean:
        return float("nan")
    if len(clean) == 1:
        return float(clean[0])
    position = (len(clean) - 1) * float(q)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(clean[lower])
    weight = position - lower
    return float(clean[lower] * (1.0 - weight) + clean[upper] * weight)


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float | str, float | str]:
    if total <= 0:
        return "", ""
    n = float(total)
    p_hat = float(successes) / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt((p_hat * (1.0 - p_hat) / n) + (z2 / (4.0 * n * n)))
    return float(max(0.0, center - margin)), float(min(1.0, center + margin))


def parse_discovery_seeds(values: list[str] | None) -> dict[str, int]:
    seeds: dict[str, int] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Discovery seed must be GRAPH=SEED, got {value!r}")
        graph_id, seed_text = value.split("=", 1)
        seeds[graph_id] = int(seed_text)
    return seeds


def solution_group_key(row: dict[str, str]) -> tuple[str, str, int]:
    return (
        str(row["regime"]),
        str(row["graph_id"]),
        int(float(row["subset_seed"])),
    )


def solution_rank(row: dict[str, str]) -> int:
    return int(float(row["solution_rank"]))


def normalized_gap_pct(row: dict[str, str] | None) -> float:
    if row is None:
        return float("nan")
    return 100.0 * parse_float(row.get("normalized_gap_to_oracle"))


def solution_signature(row: dict[str, str] | None) -> str:
    if row is None:
        return ""
    return str(row.get("solution_edge_signature", ""))


def find_matching_2stage_rank(
    spoplus_rank1_signature: str,
    two_stage_by_rank: dict[int, dict[str, str]],
    top_k: int,
) -> int | str:
    if not spoplus_rank1_signature:
        return ""
    for rank in range(2, int(top_k) + 1):
        if solution_signature(two_stage_by_rank.get(rank)) == spoplus_rank1_signature:
            return rank
    return ""


def build_seed_audit_rows(
    solution_rows: list[dict[str, str]],
    *,
    discovery_seeds: dict[str, int] | None = None,
    top_k: int = 5,
    correction_graphs: set[str] | None = None,
    promotion_graphs: set[str] | None = None,
    spo_better_epsilon_pp: float = 0.1,
    meaningful_delta_pp: float = 5.0,
    casec_two_stage_min_pct: float = 10.0,
    casec_spoplus_max_pct: float = 5.0,
    casec_delta_min_pp: float = 5.0,
    strong_two_stage_min_pct: float = 20.0,
    strong_spoplus_max_pct: float = 5.0,
    strong_delta_min_pp: float = 10.0,
    correction_two_stage_rank2_min_pct: float = 10.0,
    promotion_two_stage_rank2_max_pct: float = 5.0,
    promotion_spoplus_rank1_max_pct: float = 5.0,
) -> list[dict[str, Any]]:
    discovery_seeds = discovery_seeds or {}
    correction_graphs = correction_graphs or {"G-392.json"}
    promotion_graphs = promotion_graphs or {"G-1560.json"}

    grouped: dict[tuple[str, str, int], dict[str, dict[int, dict[str, str]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in solution_rows:
        label = str(row.get("method_label", ""))
        if label not in {TWO_STAGE_LABEL, SPOPLUS_LABEL}:
            continue
        grouped[solution_group_key(row)][label][solution_rank(row)] = row

    audit_rows: list[dict[str, Any]] = []
    for regime, graph_id, subset_seed in sorted(grouped, key=lambda key: (key[0], key[1], key[2])):
        methods = grouped[(regime, graph_id, subset_seed)]
        two_stage_by_rank = methods.get(TWO_STAGE_LABEL, {})
        spoplus_by_rank = methods.get(SPOPLUS_LABEL, {})
        two_stage_rank1 = two_stage_by_rank.get(1)
        spoplus_rank1 = spoplus_by_rank.get(1)
        if two_stage_rank1 is None or spoplus_rank1 is None:
            continue

        two_stage_rank2 = two_stage_by_rank.get(2)
        spoplus_rank2 = spoplus_by_rank.get(2)
        two_stage_rank1_gap = normalized_gap_pct(two_stage_rank1)
        two_stage_rank2_gap = normalized_gap_pct(two_stage_rank2)
        spoplus_rank1_gap = normalized_gap_pct(spoplus_rank1)
        spoplus_rank2_gap = normalized_gap_pct(spoplus_rank2)
        delta_pp = two_stage_rank1_gap - spoplus_rank1_gap

        spoplus_rank1_signature = solution_signature(spoplus_rank1)
        two_stage_rank2_signature = solution_signature(two_stage_rank2)
        matching_rank = find_matching_2stage_rank(
            spoplus_rank1_signature,
            two_stage_by_rank,
            top_k=top_k,
        )
        equals_rank2 = matching_rank == 2
        in_topk = matching_rank != ""

        strict_case_c = (
            two_stage_rank1_gap >= casec_two_stage_min_pct
            and spoplus_rank1_gap <= casec_spoplus_max_pct
            and delta_pp >= casec_delta_min_pp
        )
        strong_case_c = (
            two_stage_rank1_gap >= strong_two_stage_min_pct
            and spoplus_rank1_gap <= strong_spoplus_max_pct
            and delta_pp >= strong_delta_min_pp
        )
        correction_preserved = (
            graph_id in correction_graphs
            and strict_case_c
            and math.isfinite(two_stage_rank2_gap)
            and two_stage_rank2_gap > correction_two_stage_rank2_min_pct
        )
        rank2_promotion_preserved = (
            graph_id in promotion_graphs
            and strict_case_c
            and math.isfinite(two_stage_rank2_gap)
            and two_stage_rank2_gap <= promotion_two_stage_rank2_max_pct
            and equals_rank2
        )
        topk_promotion_preserved = (
            graph_id in promotion_graphs
            and in_topk
            and spoplus_rank1_gap <= promotion_spoplus_rank1_max_pct
        )

        discovery_seed = discovery_seeds.get(graph_id, "")
        audit_rows.append(
            {
                "regime": regime,
                "graph_id": graph_id,
                "subset_seed": subset_seed,
                "discovery_seed": discovery_seed,
                "is_discovery_seed": discovery_seed != "" and subset_seed == int(discovery_seed),
                "top_k": int(top_k),
                "two_stage_rank1_gap_pct": two_stage_rank1_gap,
                "two_stage_rank2_gap_pct": two_stage_rank2_gap,
                "spoplus_rank1_gap_pct": spoplus_rank1_gap,
                "spoplus_rank2_gap_pct": spoplus_rank2_gap,
                "delta_pp": delta_pp,
                "spo_better": delta_pp > spo_better_epsilon_pp,
                "meaningful_spo_benefit": delta_pp >= meaningful_delta_pp,
                "strict_case_c": strict_case_c,
                "strong_case_c": strong_case_c,
                "correction_preserved": correction_preserved,
                "spo_rank1_equals_2stage_rank2": equals_rank2,
                "rank2_promotion_preserved": rank2_promotion_preserved,
                "spo_rank1_in_2stage_topk": in_topk,
                "spoplus_matches_2stage_rank": matching_rank,
                "topk_promotion_preserved": topk_promotion_preserved,
                "two_stage_rank1_signature": solution_signature(two_stage_rank1),
                "two_stage_rank2_signature": two_stage_rank2_signature,
                "spoplus_rank1_signature": spoplus_rank1_signature,
            }
        )

    return audit_rows


def summarize_one_group(
    rows: list[dict[str, Any]],
    *,
    seed_filter: str,
    excluded_seed: int | str,
) -> dict[str, Any]:
    first = rows[0] if rows else {}
    seeds = [int(row["subset_seed"]) for row in rows]
    summary: dict[str, Any] = {
        "regime": first.get("regime", ""),
        "graph_id": first.get("graph_id", ""),
        "seed_filter": seed_filter,
        "seed_count": len(rows),
        "discovery_seed": first.get("discovery_seed", ""),
        "excluded_seed": excluded_seed,
        "subset_seed_min": min(seeds) if seeds else "",
        "subset_seed_max": max(seeds) if seeds else "",
    }

    for metric in BINARY_METRICS:
        count = sum(1 for row in rows if parse_bool(row.get(metric)))
        low, high = wilson_interval(count, len(rows))
        summary.update(
            {
                f"{metric}_count": count,
                f"{metric}_rate": count / len(rows) if rows else float("nan"),
                f"{metric}_wilson95_low": low,
                f"{metric}_wilson95_high": high,
            }
        )

    for metric in CONTINUOUS_METRICS:
        values = [parse_float(row.get(metric)) for row in rows]
        summary.update(
            {
                f"mean_{metric}": finite_mean(values),
                f"median_{metric}": finite_median(values),
                f"q25_{metric}": finite_quantile(values, 0.25),
                f"q75_{metric}": finite_quantile(values, 0.75),
            }
        )

    return summary


def summarize_audit_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["regime"]), str(row["graph_id"]))].append(row)

    summary_rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        group = sorted(grouped[key], key=lambda row: int(row["subset_seed"]))
        discovery_seed = group[0].get("discovery_seed", "")
        summary_rows.append(
            summarize_one_group(group, seed_filter="all", excluded_seed="")
        )
        if discovery_seed != "":
            filtered = [row for row in group if not parse_bool(row.get("is_discovery_seed"))]
            summary_rows.append(
                summarize_one_group(
                    filtered,
                    seed_filter="exclude_discovery_seed",
                    excluded_seed=int(discovery_seed),
                )
            )

    return summary_rows


def print_summary(summary_rows: list[dict[str, Any]]) -> None:
    for row in summary_rows:
        print(
            "{graph} {seed_filter}: n={n} strict_case_c={strict:.3g} "
            "spo_better={spo:.3g} correction={corr:.3g} rank2_promotion={r2:.3g} "
            "topk_promotion={topk:.3g} median_delta_pp={delta:.3g}".format(
                graph=row["graph_id"],
                seed_filter=row["seed_filter"],
                n=row["seed_count"],
                strict=parse_float(row["strict_case_c_rate"]),
                spo=parse_float(row["spo_better_rate"]),
                corr=parse_float(row["correction_preserved_rate"]),
                r2=parse_float(row["rank2_promotion_preserved_rate"]),
                topk=parse_float(row["topk_promotion_preserved_rate"]),
                delta=parse_float(row["median_delta_pp"]),
            )
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Summarize fixed graph-instance x all subset_seed model robustness "
            "from top-K second-best replay rows."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--discovery-seed",
        nargs="*",
        default=["G-392.json=1", "G-1560.json=30"],
        help="Discovery seed mapping as GRAPH=SEED.",
    )
    parser.add_argument("--correction-graphs", nargs="*", default=["G-392.json"])
    parser.add_argument("--promotion-graphs", nargs="*", default=["G-1560.json"])
    parser.add_argument("--spo-better-epsilon-pp", type=float, default=0.1)
    parser.add_argument("--meaningful-delta-pp", type=float, default=5.0)
    parser.add_argument("--casec-two-stage-min-pct", type=float, default=10.0)
    parser.add_argument("--casec-spoplus-max-pct", type=float, default=5.0)
    parser.add_argument("--casec-delta-min-pp", type=float, default=5.0)
    parser.add_argument("--strong-two-stage-min-pct", type=float, default=20.0)
    parser.add_argument("--strong-spoplus-max-pct", type=float, default=5.0)
    parser.add_argument("--strong-delta-min-pp", type=float, default=10.0)
    parser.add_argument("--correction-two-stage-rank2-min-pct", type=float, default=10.0)
    parser.add_argument("--promotion-two-stage-rank2-max-pct", type=float, default=5.0)
    parser.add_argument("--promotion-spoplus-rank1-max-pct", type=float, default=5.0)
    args = parser.parse_args(argv)
    if args.top_k < 2:
        parser.error("--top-k must be >= 2")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    rows = read_csv_rows(args.input)
    audit_rows = build_seed_audit_rows(
        rows,
        discovery_seeds=parse_discovery_seeds(args.discovery_seed),
        top_k=args.top_k,
        correction_graphs=set(args.correction_graphs),
        promotion_graphs=set(args.promotion_graphs),
        spo_better_epsilon_pp=args.spo_better_epsilon_pp,
        meaningful_delta_pp=args.meaningful_delta_pp,
        casec_two_stage_min_pct=args.casec_two_stage_min_pct,
        casec_spoplus_max_pct=args.casec_spoplus_max_pct,
        casec_delta_min_pp=args.casec_delta_min_pp,
        strong_two_stage_min_pct=args.strong_two_stage_min_pct,
        strong_spoplus_max_pct=args.strong_spoplus_max_pct,
        strong_delta_min_pp=args.strong_delta_min_pp,
        correction_two_stage_rank2_min_pct=args.correction_two_stage_rank2_min_pct,
        promotion_two_stage_rank2_max_pct=args.promotion_two_stage_rank2_max_pct,
        promotion_spoplus_rank1_max_pct=args.promotion_spoplus_rank1_max_pct,
    )
    summary_rows = summarize_audit_rows(audit_rows)
    write_csv(args.output, audit_rows, AUDIT_FIELDS)
    write_csv(args.summary_output, summary_rows, SUMMARY_FIELDS)
    print(f"Saved {len(audit_rows)} per-seed rows to {args.output}")
    print(f"Saved {len(summary_rows)} summary rows to {args.summary_output}")
    print_summary(summary_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
