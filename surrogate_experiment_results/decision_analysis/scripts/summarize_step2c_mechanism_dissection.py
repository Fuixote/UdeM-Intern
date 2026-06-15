#!/usr/bin/env python3
"""Summarize Step2c selected-graph mechanism dissection diagnostics."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "decision_analysis"
    / "results"
    / "step2c_mechanism_dissection"
)
DEFAULT_GRAPH_SUMMARY = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "decision_analysis"
    / "results"
    / "all400_model_seed_baseline"
    / "step2c_all400_all50_graph_summary.csv"
)
DEFAULT_PREDICTED_TOPM = DEFAULT_RESULTS_DIR / "step2c_selected_graphs_all50_top20_predicted.csv"
DEFAULT_ORACLE_LANDSCAPE = (
    DEFAULT_RESULTS_DIR / "step2c_selected_graphs_true_top50_oracle_landscape.csv"
)
DEFAULT_ATLAS_OUTPUT = DEFAULT_RESULTS_DIR / "step2c_selected_graphs_mechanism_atlas.csv"
DEFAULT_BASIN_OUTPUT = DEFAULT_RESULTS_DIR / "step2c_selected_graphs_candidate_basin_diagnostic.csv"

DEFAULT_FAMILY_BY_GRAPH = {
    "G-392.json": "clean_correction",
    "G-1285.json": "clean_exact_rank2_promotion",
    "G-1560.json": "large_delta_topk_promotion",
    "G-1169.json": "unexplained_stable_spoplus_success",
    "G-1449.json": "unexplained_stable_spoplus_success",
    "G-1657.json": "correction_topk_boundary",
    "G-191.json": "correction_topk_boundary",
    "G-142.json": "negative_control_both_poor",
    "G-946.json": "negative_control_both_poor",
    "G-14.json": "negative_control_spoplus_worse",
    "G-163.json": "negative_control_spoplus_worse",
    "G-552.json": "appendix_pure_correction_replication",
    "G-1110.json": "appendix_pure_correction_replication",
    "G-178.json": "appendix_pure_correction_replication",
    "G-1206.json": "appendix_negative_control_both_poor",
    "G-1308.json": "appendix_negative_control_spoplus_worse",
}

MECHANISM_ATLAS_FIELDS = [
    "graph_id",
    "assigned_family",
    "strict_case_c_rate",
    "strong_case_c_rate",
    "correction_rate",
    "exact_rank2_promotion_rate",
    "topk_promotion_rate",
    "median_delta_pp",
    "median_two_stage_rank1_gap_pct",
    "median_spoplus_rank1_gap_pct",
    "median_two_stage_rank2_gap_pct",
    "rate_2stage_top5_contains_near_oracle",
    "rate_2stage_top20_contains_near_oracle",
    "rate_spoplus_rank1_in_2stage_top5",
    "rate_spoplus_rank1_in_2stage_top20",
    "rate_spoplus_rank1_in_true_top50",
    "median_rank_of_best_near_oracle_under_2stage",
    "median_rank_of_spoplus_match_under_2stage",
    "median_true_rank_of_spoplus_rank1",
    "median_jaccard_2stage_rank1_to_oracle",
    "median_jaccard_spoplus_rank1_to_oracle",
    "median_jaccard_spoplus_rank1_to_nearest_2stage_top20",
]

CANDIDATE_BASIN_FIELDS = [
    "graph_id",
    "seed_count",
    "rate_2stage_top5_contains_near_oracle",
    "rate_2stage_top20_contains_near_oracle",
    "rate_spoplus_rank1_in_2stage_top5",
    "rate_spoplus_rank1_in_2stage_top20",
    "rate_spoplus_rank1_in_true_top50",
    "rate_spoplus_rank1_near_oracle",
    "median_rank_of_best_near_oracle_under_2stage",
    "median_rank_of_spoplus_match_under_2stage",
    "median_true_rank_of_spoplus_rank1",
    "median_jaccard_2stage_rank1_to_oracle",
    "median_jaccard_spoplus_rank1_to_oracle",
    "median_jaccard_spoplus_rank1_to_nearest_2stage_top20",
]


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_float(value: Any, default: float = float("nan")) -> float:
    if value is None or value == "":
        return default
    return float(value)


def parse_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    return int(float(value))


def finite_median(values: list[float]) -> float:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        return float("nan")
    mid = len(clean) // 2
    if len(clean) % 2:
        return float(clean[mid])
    return float(0.5 * (clean[mid - 1] + clean[mid]))


def binary_rate(values: list[bool]) -> float:
    if not values:
        return float("nan")
    return float(sum(1 for value in values if value) / len(values))


def signature_set(signature: Any) -> set[str]:
    text = str(signature or "")
    if not text:
        return set()
    return {part for part in text.split("|") if part != ""}


def signature_jaccard(left: Any, right: Any) -> float:
    left_set = signature_set(left)
    right_set = signature_set(right)
    union = left_set | right_set
    if not union:
        return 1.0
    return float(len(left_set & right_set) / len(union))


def group_predicted_rows(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, int], dict[str, dict[int, dict[str, Any]]]]:
    grouped: dict[tuple[str, int], dict[str, dict[int, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in rows:
        graph_id = str(row["graph_id"])
        seed = parse_int(row["subset_seed"])
        method = str(row["method_label"])
        rank = parse_int(row["solution_rank"])
        grouped[(graph_id, seed)][method][rank] = row
    return grouped


def oracle_rank_by_graph(
    oracle_rows: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    ranks: dict[str, dict[str, int]] = defaultdict(dict)
    for row in oracle_rows:
        graph_id = str(row["graph_id"])
        signature = str(row.get("solution_edge_signature", ""))
        rank = parse_int(row.get("solution_rank"))
        if signature and signature not in ranks[graph_id]:
            ranks[graph_id][signature] = rank
    return ranks


def build_candidate_basin_diagnostics(
    predicted_rows: list[dict[str, Any]],
    oracle_rows: list[dict[str, Any]],
    top_k_values: tuple[int, int] = (5, 20),
    near_oracle_gap: float = 0.05,
) -> list[dict[str, Any]]:
    top5, top20 = top_k_values
    grouped = group_predicted_rows(predicted_rows)
    true_rank = oracle_rank_by_graph(oracle_rows)

    per_graph: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (graph_id, seed), methods in grouped.items():
        two_stage = methods.get("2stage_val_mse", {})
        spoplus = methods.get("spoplus_val_spoplus_loss", {})
        if 1 not in two_stage or 1 not in spoplus:
            continue

        spo1 = spoplus[1]
        two1 = two_stage[1]
        spo_signature = str(spo1.get("solution_edge_signature", ""))

        near_ranks = [
            rank
            for rank, row in two_stage.items()
            if rank <= top20
            and parse_float(row.get("normalized_gap_to_oracle")) <= near_oracle_gap + 1e-12
        ]
        top5_rows = [row for rank, row in two_stage.items() if rank <= top5]
        top20_rows = [row for rank, row in two_stage.items() if rank <= top20]
        top5_signatures = {str(row.get("solution_edge_signature", "")) for row in top5_rows}
        top20_signatures = {str(row.get("solution_edge_signature", "")) for row in top20_rows}
        match_rank = next(
            (
                rank
                for rank, row in sorted(two_stage.items())
                if rank <= top20 and str(row.get("solution_edge_signature", "")) == spo_signature
            ),
            float("nan"),
        )
        nearest_jaccard = (
            max(
                signature_jaccard(spo_signature, row.get("solution_edge_signature", ""))
                for row in top20_rows
            )
            if top20_rows
            else float("nan")
        )

        per_graph[graph_id].append(
            {
                "subset_seed": seed,
                "top5_contains_near": any(rank <= top5 for rank in near_ranks),
                "top20_contains_near": bool(near_ranks),
                "spo_in_top5": spo_signature in top5_signatures,
                "spo_in_top20": spo_signature in top20_signatures,
                "spo_in_true_top50": spo_signature in true_rank.get(graph_id, {}),
                "spo_near_oracle": parse_float(spo1.get("normalized_gap_to_oracle"))
                <= near_oracle_gap + 1e-12,
                "best_near_rank": min(near_ranks) if near_ranks else float("nan"),
                "match_rank": match_rank,
                "true_rank": true_rank.get(graph_id, {}).get(spo_signature, float("nan")),
                "two1_jaccard_oracle": parse_float(two1.get("edge_jaccard_with_oracle")),
                "spo1_jaccard_oracle": parse_float(spo1.get("edge_jaccard_with_oracle")),
                "nearest_jaccard": nearest_jaccard,
            }
        )

    output_rows: list[dict[str, Any]] = []
    for graph_id in sorted(per_graph):
        group = per_graph[graph_id]
        output_rows.append(
            {
                "graph_id": graph_id,
                "seed_count": len(group),
                "rate_2stage_top5_contains_near_oracle": binary_rate(
                    [bool(row["top5_contains_near"]) for row in group]
                ),
                "rate_2stage_top20_contains_near_oracle": binary_rate(
                    [bool(row["top20_contains_near"]) for row in group]
                ),
                "rate_spoplus_rank1_in_2stage_top5": binary_rate(
                    [bool(row["spo_in_top5"]) for row in group]
                ),
                "rate_spoplus_rank1_in_2stage_top20": binary_rate(
                    [bool(row["spo_in_top20"]) for row in group]
                ),
                "rate_spoplus_rank1_in_true_top50": binary_rate(
                    [bool(row["spo_in_true_top50"]) for row in group]
                ),
                "rate_spoplus_rank1_near_oracle": binary_rate(
                    [bool(row["spo_near_oracle"]) for row in group]
                ),
                "median_rank_of_best_near_oracle_under_2stage": finite_median(
                    [float(row["best_near_rank"]) for row in group]
                ),
                "median_rank_of_spoplus_match_under_2stage": finite_median(
                    [float(row["match_rank"]) for row in group]
                ),
                "median_true_rank_of_spoplus_rank1": finite_median(
                    [float(row["true_rank"]) for row in group]
                ),
                "median_jaccard_2stage_rank1_to_oracle": finite_median(
                    [float(row["two1_jaccard_oracle"]) for row in group]
                ),
                "median_jaccard_spoplus_rank1_to_oracle": finite_median(
                    [float(row["spo1_jaccard_oracle"]) for row in group]
                ),
                "median_jaccard_spoplus_rank1_to_nearest_2stage_top20": finite_median(
                    [float(row["nearest_jaccard"]) for row in group]
                ),
            }
        )

    return output_rows


def assigned_family_for_graph(graph_id: str, row: dict[str, Any]) -> str:
    if graph_id in DEFAULT_FAMILY_BY_GRAPH:
        return DEFAULT_FAMILY_BY_GRAPH[graph_id]
    if parse_float(row.get("correction_rate"), 0.0) >= 0.95:
        return "candidate_correction"
    if parse_float(row.get("exact_rank2_promotion_rate"), 0.0) >= 0.50:
        return "candidate_exact_rank2_promotion"
    if parse_float(row.get("topk_promotion_rate"), 0.0) >= 0.95:
        return "candidate_topk_promotion"
    if parse_float(row.get("median_delta_pp"), 0.0) < -5.0:
        return "candidate_spoplus_worse"
    return "unassigned"


def build_mechanism_atlas(
    graph_summary_rows: list[dict[str, Any]],
    basin_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    basin_by_graph = {str(row["graph_id"]): row for row in basin_rows}
    output_rows: list[dict[str, Any]] = []

    for row in sorted(graph_summary_rows, key=lambda item: str(item["graph_id"])):
        graph_id = str(row["graph_id"])
        basin = basin_by_graph.get(graph_id, {})
        merged = {
            "graph_id": graph_id,
            "assigned_family": assigned_family_for_graph(graph_id, row),
        }
        for key in MECHANISM_ATLAS_FIELDS:
            if key in {"graph_id", "assigned_family"}:
                continue
            if key in basin:
                merged[key] = basin[key]
            else:
                merged[key] = parse_float(row.get(key), float("nan"))
        output_rows.append(merged)

    return output_rows


def filter_graphs(rows: list[dict[str, Any]], graphs: set[str] | None) -> list[dict[str, Any]]:
    if not graphs:
        return rows
    return [row for row in rows if str(row.get("graph_id", "")) in graphs]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build Step2c mechanism-dissection atlas and candidate-basin diagnostics."
    )
    parser.add_argument("--predicted-topm-input", type=Path, default=DEFAULT_PREDICTED_TOPM)
    parser.add_argument("--oracle-landscape-input", type=Path, default=DEFAULT_ORACLE_LANDSCAPE)
    parser.add_argument("--graph-summary-input", type=Path, default=DEFAULT_GRAPH_SUMMARY)
    parser.add_argument("--atlas-output", type=Path, default=DEFAULT_ATLAS_OUTPUT)
    parser.add_argument("--basin-output", type=Path, default=DEFAULT_BASIN_OUTPUT)
    parser.add_argument(
        "--graphs",
        nargs="+",
        default=list(DEFAULT_FAMILY_BY_GRAPH.keys()),
        help="Graph filenames to include.",
    )
    parser.add_argument("--near-oracle-gap", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--top-m", type=int, default=20)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    graphs = set(args.graphs) if args.graphs else None

    predicted_rows = filter_graphs(read_csv_rows(args.predicted_topm_input), graphs)
    oracle_rows = filter_graphs(read_csv_rows(args.oracle_landscape_input), graphs)
    graph_summary_rows = filter_graphs(read_csv_rows(args.graph_summary_input), graphs)

    basin_rows = build_candidate_basin_diagnostics(
        predicted_rows,
        oracle_rows,
        top_k_values=(args.top_k, args.top_m),
        near_oracle_gap=args.near_oracle_gap,
    )
    atlas_rows = build_mechanism_atlas(graph_summary_rows, basin_rows)

    write_csv(args.basin_output, basin_rows, CANDIDATE_BASIN_FIELDS)
    write_csv(args.atlas_output, atlas_rows, MECHANISM_ATLAS_FIELDS)

    print(f"Saved {len(basin_rows)} candidate-basin rows to {args.basin_output}")
    print(f"Saved {len(atlas_rows)} mechanism-atlas rows to {args.atlas_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
