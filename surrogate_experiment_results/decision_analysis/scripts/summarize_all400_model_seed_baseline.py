#!/usr/bin/env python3
"""Summarize Step2c all-400 heldout graph x subset_seed baseline.

Input is the top-K output from ``compute_second_best_solutions.py``. Outputs:

  1. per graph x subset_seed audit rows;
  2. one graph-level summary row per heldout graph;
  3. target-graph percentiles for G-392/G-1560 or user-specified targets;
  4. metric distribution summaries across the heldout graph population.
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
    / "all400_model_seed_baseline"
)
DEFAULT_INPUT = DEFAULT_OUTPUT_DIR / "step2c_all400_all50_top5_second_best.csv"
DEFAULT_SEED_AUDIT = DEFAULT_OUTPUT_DIR / "step2c_all400_all50_seed_audit.csv"
DEFAULT_GRAPH_SUMMARY = DEFAULT_OUTPUT_DIR / "step2c_all400_all50_graph_summary.csv"
DEFAULT_PERCENTILES = DEFAULT_OUTPUT_DIR / "step2c_g392_g1560_all400_percentiles.csv"
DEFAULT_DISTRIBUTION = DEFAULT_OUTPUT_DIR / "step2c_all400_metric_distribution.csv"

TWO_STAGE_LABEL = "2stage_val_mse"
SPOPLUS_LABEL = "spoplus_val_spoplus_loss"

BINARY_METRICS = [
    "spo_better",
    "meaningful_spo_benefit",
    "strict_case_c",
    "strong_case_c",
    "correction",
    "exact_rank2_promotion",
    "topk_promotion",
]

CONTINUOUS_METRICS = [
    "delta_pp",
    "two_stage_rank1_gap_pct",
    "spoplus_rank1_gap_pct",
    "two_stage_rank2_gap_pct",
    "spoplus_rank2_gap_pct",
]

DEFAULT_PERCENTILE_METRICS = [
    "strict_case_c_rate",
    "strong_case_c_rate",
    "spo_better_rate",
    "meaningful_spo_benefit_rate",
    "correction_rate",
    "exact_rank2_promotion_rate",
    "topk_promotion_rate",
    "mean_delta_pp",
    "median_delta_pp",
    "median_two_stage_rank1_gap_pct",
    "median_spoplus_rank1_gap_pct",
    "median_two_stage_rank2_gap_pct",
]

SEED_AUDIT_FIELDS = [
    "regime",
    "graph_id",
    "subset_seed",
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
    "correction",
    "exact_rank2_promotion",
    "topk_promotion",
    "spoplus_matches_2stage_rank",
    "two_stage_rank1_signature",
    "two_stage_rank2_signature",
    "spoplus_rank1_signature",
]

GRAPH_SUMMARY_FIELDS = [
    "regime",
    "graph_id",
    "seed_count",
    "subset_seed_min",
    "subset_seed_max",
]

for metric in BINARY_METRICS:
    GRAPH_SUMMARY_FIELDS.extend(
        [
            f"{metric}_count",
            f"{metric}_rate",
            f"{metric}_wilson95_low",
            f"{metric}_wilson95_high",
        ]
    )

for metric in CONTINUOUS_METRICS:
    GRAPH_SUMMARY_FIELDS.extend(
        [
            f"mean_{metric}",
            f"median_{metric}",
            f"q25_{metric}",
            f"q75_{metric}",
        ]
    )

PERCENTILE_FIELDS = [
    "graph_id",
    "metric",
    "value",
    "percentile_midrank",
    "percentile_leq",
    "rank_desc",
    "tie_count",
    "graph_count",
]

DISTRIBUTION_FIELDS = [
    "metric",
    "graph_count",
    "min",
    "median",
    "p90",
    "p95",
    "max",
]


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


def parse_int(value: object) -> int:
    return int(float(str(value)))


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def finite_values(values: list[float]) -> list[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


def finite_mean(values: list[float]) -> float:
    clean = finite_values(values)
    if not clean:
        return float("nan")
    return float(sum(clean) / len(clean))


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


def finite_median(values: list[float]) -> float:
    return finite_quantile(values, 0.5)


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


def solution_group_key(row: dict[str, str]) -> tuple[str, str, int]:
    return str(row["regime"]), str(row["graph_id"]), parse_int(row["subset_seed"])


def solution_rank(row: dict[str, str]) -> int:
    return parse_int(row["solution_rank"])


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
    top_k: int = 5,
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
    grouped: dict[tuple[str, str, int], dict[str, dict[int, dict[str, str]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in solution_rows:
        label = str(row.get("method_label", ""))
        if label not in {TWO_STAGE_LABEL, SPOPLUS_LABEL}:
            continue
        grouped[solution_group_key(row)][label][solution_rank(row)] = row

    output_rows: list[dict[str, Any]] = []
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
        matching_rank = find_matching_2stage_rank(
            spoplus_rank1_signature,
            two_stage_by_rank,
            top_k=top_k,
        )

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
        correction = (
            strict_case_c
            and math.isfinite(two_stage_rank2_gap)
            and two_stage_rank2_gap > correction_two_stage_rank2_min_pct
        )
        exact_rank2_promotion = (
            strict_case_c
            and math.isfinite(two_stage_rank2_gap)
            and two_stage_rank2_gap <= promotion_two_stage_rank2_max_pct
            and matching_rank == 2
        )
        topk_promotion = (
            matching_rank != ""
            and spoplus_rank1_gap <= promotion_spoplus_rank1_max_pct
        )

        output_rows.append(
            {
                "regime": regime,
                "graph_id": graph_id,
                "subset_seed": subset_seed,
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
                "correction": correction,
                "exact_rank2_promotion": exact_rank2_promotion,
                "topk_promotion": topk_promotion,
                "spoplus_matches_2stage_rank": matching_rank,
                "two_stage_rank1_signature": solution_signature(two_stage_rank1),
                "two_stage_rank2_signature": solution_signature(two_stage_rank2),
                "spoplus_rank1_signature": spoplus_rank1_signature,
            }
        )
    return output_rows


def summarize_one_graph(rows: list[dict[str, Any]]) -> dict[str, Any]:
    seeds = [int(row["subset_seed"]) for row in rows]
    first = rows[0] if rows else {}
    summary: dict[str, Any] = {
        "regime": first.get("regime", ""),
        "graph_id": first.get("graph_id", ""),
        "seed_count": len(rows),
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


def summarize_by_graph(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        grouped[(str(row["regime"]), str(row["graph_id"]))].append(row)

    summaries = []
    for key in sorted(grouped):
        summaries.append(
            summarize_one_graph(
                sorted(grouped[key], key=lambda row: int(row["subset_seed"]))
            )
        )
    return summaries


def rank_desc(values: list[float], value: float) -> int:
    return 1 + sum(1 for other in values if math.isfinite(other) and other > value)


def percentile_midrank(values: list[float], value: float) -> float:
    finite = finite_values(values)
    if not finite or not math.isfinite(value):
        return float("nan")
    less = sum(1 for other in finite if other < value)
    equal = sum(1 for other in finite if other == value)
    return 100.0 * (less + 0.5 * equal) / len(finite)


def percentile_leq(values: list[float], value: float) -> float:
    finite = finite_values(values)
    if not finite or not math.isfinite(value):
        return float("nan")
    return 100.0 * sum(1 for other in finite if other <= value) / len(finite)


def build_target_percentile_rows(
    graph_rows: list[dict[str, Any]],
    *,
    target_graphs: list[str],
    metrics: list[str],
) -> list[dict[str, Any]]:
    by_graph = {str(row["graph_id"]): row for row in graph_rows}
    output_rows: list[dict[str, Any]] = []
    for graph_id in target_graphs:
        if graph_id not in by_graph:
            continue
        target = by_graph[graph_id]
        for metric in metrics:
            values = [parse_float(row.get(metric)) for row in graph_rows]
            value = parse_float(target.get(metric))
            output_rows.append(
                {
                    "graph_id": graph_id,
                    "metric": metric,
                    "value": value,
                    "percentile_midrank": percentile_midrank(values, value),
                    "percentile_leq": percentile_leq(values, value),
                    "rank_desc": rank_desc(finite_values(values), value),
                    "tie_count": sum(1 for other in finite_values(values) if other == value),
                    "graph_count": len(finite_values(values)),
                }
            )
    return output_rows


def build_distribution_rows(
    graph_rows: list[dict[str, Any]],
    *,
    metrics: list[str],
) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    for metric in metrics:
        values = [parse_float(row.get(metric)) for row in graph_rows]
        clean = finite_values(values)
        output_rows.append(
            {
                "metric": metric,
                "graph_count": len(clean),
                "min": min(clean) if clean else float("nan"),
                "median": finite_quantile(clean, 0.5),
                "p90": finite_quantile(clean, 0.9),
                "p95": finite_quantile(clean, 0.95),
                "max": max(clean) if clean else float("nan"),
            }
        )
    return output_rows


def print_target_rows(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        print(
            "{graph} {metric}: value={value:.4g} midrank={mid:.1f} "
            "leq={leq:.1f} rank_desc={rank} ties={ties}/{n}".format(
                graph=row["graph_id"],
                metric=row["metric"],
                value=parse_float(row["value"]),
                mid=parse_float(row["percentile_midrank"]),
                leq=parse_float(row["percentile_leq"]),
                rank=row["rank_desc"],
                ties=row["tie_count"],
                n=row["graph_count"],
            )
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Summarize all-400 heldout graph x model-seed Step2c baseline."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--seed-audit-output", type=Path, default=DEFAULT_SEED_AUDIT)
    parser.add_argument("--graph-summary-output", type=Path, default=DEFAULT_GRAPH_SUMMARY)
    parser.add_argument("--percentile-output", type=Path, default=DEFAULT_PERCENTILES)
    parser.add_argument("--distribution-output", type=Path, default=DEFAULT_DISTRIBUTION)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--target-graphs",
        nargs="*",
        default=["G-392.json", "G-1560.json"],
    )
    parser.add_argument(
        "--percentile-metrics",
        nargs="*",
        default=list(DEFAULT_PERCENTILE_METRICS),
    )
    args = parser.parse_args(argv)
    if args.top_k < 2:
        parser.error("--top-k must be >= 2")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    solution_rows = read_csv_rows(args.input)
    seed_rows = build_seed_audit_rows(solution_rows, top_k=args.top_k)
    graph_rows = summarize_by_graph(seed_rows)
    percentile_rows = build_target_percentile_rows(
        graph_rows,
        target_graphs=args.target_graphs,
        metrics=args.percentile_metrics,
    )
    distribution_rows = build_distribution_rows(
        graph_rows,
        metrics=args.percentile_metrics,
    )
    write_csv(args.seed_audit_output, seed_rows, SEED_AUDIT_FIELDS)
    write_csv(args.graph_summary_output, graph_rows, GRAPH_SUMMARY_FIELDS)
    write_csv(args.percentile_output, percentile_rows, PERCENTILE_FIELDS)
    write_csv(args.distribution_output, distribution_rows, DISTRIBUTION_FIELDS)
    print(f"Saved {len(seed_rows)} seed audit rows to {args.seed_audit_output}")
    print(f"Saved {len(graph_rows)} graph summary rows to {args.graph_summary_output}")
    print(f"Saved {len(percentile_rows)} target percentile rows to {args.percentile_output}")
    print(f"Saved {len(distribution_rows)} distribution rows to {args.distribution_output}")
    print_target_rows(percentile_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
