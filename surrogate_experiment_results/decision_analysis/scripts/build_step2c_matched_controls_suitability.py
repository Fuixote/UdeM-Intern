#!/usr/bin/env python3
"""Build Phase 3 matched-control DFL suitability tables for Step2c."""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_step2c_graph_level_suitability import (  # noqa: E402
    finite_mean,
    finite_std,
    parse_float,
    percentile_rank,
    read_csv,
    write_csv,
)


AUDIT_DIR = PROJECT_ROOT / "surrogate_experiment_results" / "Step2c Graph-Level DFL Suitability Audit"
DEFAULT_RESULTS_DIR = AUDIT_DIR / "results"
DEFAULT_PRESENTATION_DIR = AUDIT_DIR / "presentation"
DEFAULT_INPUT = DEFAULT_RESULTS_DIR / "step2c_all400_graph_boundary_outcome_table.csv"
DEFAULT_MATCH_OUTPUT = DEFAULT_RESULTS_DIR / "step2c_phase3_matched_controls.csv"
DEFAULT_SUMMARY_OUTPUT = DEFAULT_RESULTS_DIR / "step2c_phase3_target_vs_matched_summary.csv"
DEFAULT_STORY_OUTPUT = DEFAULT_PRESENTATION_DIR / "step2c_phase3_matched_controls_story.md"

README_MATCH_FEATURES = (
    "num_vertices",
    "num_arcs",
    "density",
    "num_2cycles",
    "num_3cycles",
    "largest_scc_fraction",
)
MATCH_FEATURES = README_MATCH_FEATURES

TARGET_CASE_GROUPS = {
    "helpful_success": (
        "G-392.json",
        "G-1285.json",
        "G-1560.json",
        "G-1169.json",
        "G-1449.json",
    ),
    "both_poor_control": (
        "G-142.json",
        "G-946.json",
    ),
    "harmful_reranking_control": (
        "G-14.json",
        "G-163.json",
    ),
}
TARGET_GRAPHS = tuple(graph for graphs in TARGET_CASE_GROUPS.values() for graph in graphs)

COMPARISON_FEATURES = (
    "median_delta_pp",
    "strict_case_c_rate",
    "helpful_graph",
    "harmful_graph",
    "feasible_set_richness_score",
    "num_exchange_candidates",
    "conflict_graph_density",
    "ranking_ambiguity_score",
    "mean_2stage_top5_within_1pct_count",
    "median_2stage_top5_diversity_from_rank1",
)

MATCH_FIELDS = [
    "target_graph_id",
    "target_case_group",
    "match_rank",
    "control_graph_id",
    "match_distance",
    *[f"target_{feature}" for feature in MATCH_FEATURES],
    *[f"control_{feature}" for feature in MATCH_FEATURES],
    *[f"absdiff_{feature}" for feature in MATCH_FEATURES],
    *[f"target_{feature}" for feature in COMPARISON_FEATURES],
    *[f"control_{feature}" for feature in COMPARISON_FEATURES],
]

SUMMARY_FIELDS = [
    "target_graph_id",
    "target_case_group",
    "n_controls",
    "mean_match_distance",
    "closest_control_graph_id",
    "target_median_delta_pp",
    "matched_median_delta_pp_median",
    "matched_median_delta_pp_q25",
    "matched_median_delta_pp_q75",
    "target_delta_percentile_within_matched",
    "target_helpful_graph",
    "matched_helpful_rate",
    "target_harmful_graph",
    "matched_harmful_rate",
    "target_ranking_ambiguity_score",
    "matched_ranking_ambiguity_median",
    "target_ranking_ambiguity_percentile_within_matched",
    "target_feasible_richness",
    "matched_feasible_richness_median",
    "target_feasible_richness_percentile_within_matched",
    "target_num_exchange_candidates",
    "matched_num_exchange_candidates_median",
    "target_conflict_graph_density",
    "matched_conflict_graph_density_median",
]


def graph_sort_key(graph_id: str) -> tuple[int, int | str]:
    stem = graph_id.removesuffix(".json")
    if stem.startswith("G-"):
        try:
            return (0, int(stem.split("-", 1)[1]))
        except ValueError:
            pass
    return (1, stem)


def case_group_for_graph(graph_id: str) -> str:
    for group, graph_ids in TARGET_CASE_GROUPS.items():
        if graph_id in graph_ids:
            return group
    return "matched_control"


def target_catalog_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for case_group, graph_ids in TARGET_CASE_GROUPS.items():
        for graph_id in graph_ids:
            rows.append({"graph_id": graph_id, "case_group": case_group})
    return rows


def finite_median(values: list[Any]) -> float:
    clean = sorted(value for value in (parse_float(item) for item in values) if math.isfinite(value))
    if not clean:
        return float("nan")
    mid = len(clean) // 2
    if len(clean) % 2:
        return float(clean[mid])
    return float((clean[mid - 1] + clean[mid]) / 2.0)


def finite_quantile(values: list[Any], quantile: float) -> float:
    clean = sorted(value for value in (parse_float(item) for item in values) if math.isfinite(value))
    if not clean:
        return float("nan")
    index = max(0, min(len(clean) - 1, round((len(clean) - 1) * quantile)))
    return float(clean[index])


def zscore_lookup(rows: list[dict[str, Any]], features: tuple[str, ...]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {str(row["graph_id"]): {} for row in rows}
    for feature in features:
        values = [parse_float(row.get(feature)) for row in rows]
        mean = finite_mean(values)
        std = finite_std(values)
        for row, value in zip(rows, values):
            graph_id = str(row["graph_id"])
            output[graph_id][feature] = 0.0 if std == 0 or not math.isfinite(value) else (value - mean) / std
    return output


def match_distance(
    target_graph_id: str,
    control_graph_id: str,
    zscores: dict[str, dict[str, float]],
    features: tuple[str, ...] = MATCH_FEATURES,
) -> float:
    diffs = [
        zscores[target_graph_id].get(feature, 0.0) - zscores[control_graph_id].get(feature, 0.0)
        for feature in features
    ]
    return float(math.sqrt(sum(diff * diff for diff in diffs)))


def build_matched_control_rows(
    rows: list[dict[str, Any]],
    *,
    target_graphs: tuple[str, ...] = TARGET_GRAPHS,
    n_controls: int = 20,
    exclude_graphs: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    by_graph = {str(row["graph_id"]): row for row in rows}
    zscores = zscore_lookup(rows, MATCH_FEATURES)
    excluded = set(target_graphs if exclude_graphs is None else exclude_graphs)
    output: list[dict[str, Any]] = []

    for target_graph_id in target_graphs:
        if target_graph_id not in by_graph:
            continue
        candidates: list[tuple[float, str]] = []
        for control_graph_id in by_graph:
            if control_graph_id in excluded:
                continue
            distance = match_distance(target_graph_id, control_graph_id, zscores)
            candidates.append((distance, control_graph_id))
        candidates.sort(key=lambda item: (item[0], graph_sort_key(item[1])))

        target = by_graph[target_graph_id]
        for rank, (distance, control_graph_id) in enumerate(candidates[:n_controls], start=1):
            control = by_graph[control_graph_id]
            row: dict[str, Any] = {
                "target_graph_id": target_graph_id,
                "target_case_group": case_group_for_graph(target_graph_id),
                "match_rank": rank,
                "control_graph_id": control_graph_id,
                "match_distance": distance,
            }
            for feature in MATCH_FEATURES:
                target_value = parse_float(target.get(feature))
                control_value = parse_float(control.get(feature))
                row[f"target_{feature}"] = target_value
                row[f"control_{feature}"] = control_value
                row[f"absdiff_{feature}"] = abs(target_value - control_value)
            for feature in COMPARISON_FEATURES:
                row[f"target_{feature}"] = parse_float(target.get(feature))
                row[f"control_{feature}"] = parse_float(control.get(feature))
            output.append(row)
    return output


def build_target_vs_matched_summary_rows(
    rows: list[dict[str, Any]],
    match_rows: list[dict[str, Any]],
    *,
    target_order: tuple[str, ...] = TARGET_GRAPHS,
) -> list[dict[str, Any]]:
    by_graph = {str(row["graph_id"]): row for row in rows}
    controls_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in match_rows:
        controls_by_target[str(row["target_graph_id"])].append(row)

    output: list[dict[str, Any]] = []
    ordered_targets = [graph_id for graph_id in target_order if graph_id in controls_by_target]
    ordered_targets.extend(
        graph_id for graph_id in sorted(controls_by_target, key=graph_sort_key) if graph_id not in set(ordered_targets)
    )
    for target_graph_id in ordered_targets:
        target = by_graph[target_graph_id]
        matches = sorted(controls_by_target[target_graph_id], key=lambda row: int(row["match_rank"]))
        control_graph_ids = [str(row["control_graph_id"]) for row in matches]
        controls = [by_graph[graph_id] for graph_id in control_graph_ids if graph_id in by_graph]

        control_delta = [parse_float(row.get("median_delta_pp")) for row in controls]
        control_ambiguity = [parse_float(row.get("ranking_ambiguity_score")) for row in controls]
        control_richness = [parse_float(row.get("feasible_set_richness_score")) for row in controls]
        control_exchange_count = [parse_float(row.get("num_exchange_candidates")) for row in controls]
        control_conflict_density = [parse_float(row.get("conflict_graph_density")) for row in controls]

        target_delta = parse_float(target.get("median_delta_pp"))
        target_ambiguity = parse_float(target.get("ranking_ambiguity_score"))
        target_richness = parse_float(target.get("feasible_set_richness_score"))

        output.append(
            {
                "target_graph_id": target_graph_id,
                "target_case_group": case_group_for_graph(target_graph_id),
                "n_controls": len(controls),
                "mean_match_distance": finite_mean([row.get("match_distance") for row in matches]),
                "closest_control_graph_id": control_graph_ids[0] if control_graph_ids else "",
                "target_median_delta_pp": target_delta,
                "matched_median_delta_pp_median": finite_median(control_delta),
                "matched_median_delta_pp_q25": finite_quantile(control_delta, 0.25),
                "matched_median_delta_pp_q75": finite_quantile(control_delta, 0.75),
                "target_delta_percentile_within_matched": percentile_rank(control_delta, target_delta),
                "target_helpful_graph": int(parse_float(target.get("helpful_graph"), 0.0)),
                "matched_helpful_rate": finite_mean([parse_float(row.get("helpful_graph")) for row in controls]),
                "target_harmful_graph": int(parse_float(target.get("harmful_graph"), 0.0)),
                "matched_harmful_rate": finite_mean([parse_float(row.get("harmful_graph")) for row in controls]),
                "target_ranking_ambiguity_score": target_ambiguity,
                "matched_ranking_ambiguity_median": finite_median(control_ambiguity),
                "target_ranking_ambiguity_percentile_within_matched": percentile_rank(
                    control_ambiguity,
                    target_ambiguity,
                ),
                "target_feasible_richness": target_richness,
                "matched_feasible_richness_median": finite_median(control_richness),
                "target_feasible_richness_percentile_within_matched": percentile_rank(
                    control_richness,
                    target_richness,
                ),
                "target_num_exchange_candidates": parse_float(target.get("num_exchange_candidates")),
                "matched_num_exchange_candidates_median": finite_median(control_exchange_count),
                "target_conflict_graph_density": parse_float(target.get("conflict_graph_density")),
                "matched_conflict_graph_density_median": finite_median(control_conflict_density),
            }
        )
    return output


def write_phase3_story(
    path: str | Path,
    *,
    summary_rows: list[dict[str, Any]],
    n_controls: int,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Step2c Graph-Level DFL Suitability: Phase 3 Matched Controls",
        "",
        "## Scope",
        "",
        "Phase 3 matches each README target graph to nearest heldout controls using only the six raw-topology variables specified in the protocol.",
        f"Each target uses up to {n_controls} controls, excluding the README target set from the control pool.",
        "",
        "## Match Variables",
        "",
        "```text",
        *MATCH_FEATURES,
        "```",
        "",
        "## Target Vs Matched Controls",
        "",
        "| Target | Group | Delta | Matched delta median | Delta pct | Ambiguity pct | Richness pct | Closest control |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['target_graph_id']} | {row['target_case_group']} | "
            f"{parse_float(row.get('target_median_delta_pp')):.2f} | "
            f"{parse_float(row.get('matched_median_delta_pp_median')):.2f} | "
            f"{parse_float(row.get('target_delta_percentile_within_matched')):.2f} | "
            f"{parse_float(row.get('target_ranking_ambiguity_percentile_within_matched')):.2f} | "
            f"{parse_float(row.get('target_feasible_richness_percentile_within_matched')):.2f} | "
            f"{row.get('closest_control_graph_id', '')} |"
        )
    lines.extend(
        [
            "",
            "## Report-Safe Interpretation",
            "",
            "This is a matched-control association audit. Matching is intentionally restricted to simple raw topology, then outcomes and higher-level feasible-set / prediction-boundary diagnostics are compared after matching.",
            "A target that remains extreme relative to matched controls supports graph-instance specificity beyond the coarse topology variables used for matching. It still does not prove topology causality.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Phase 3 matched-control DFL suitability tables for Step2c."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--n-controls", type=int, default=20)
    parser.add_argument("--match-output", type=Path, default=DEFAULT_MATCH_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--story-output", type=Path, default=DEFAULT_STORY_OUTPUT)
    return parser.parse_args(argv)


def build_phase3_outputs(args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    rows = read_csv(args.input)
    match_rows = build_matched_control_rows(
        rows,
        target_graphs=TARGET_GRAPHS,
        n_controls=args.n_controls,
        exclude_graphs=TARGET_GRAPHS,
    )
    summary_rows = build_target_vs_matched_summary_rows(rows, match_rows)
    write_csv(args.match_output, match_rows, MATCH_FIELDS)
    write_csv(args.summary_output, summary_rows, SUMMARY_FIELDS)
    write_phase3_story(args.story_output, summary_rows=summary_rows, n_controls=args.n_controls)
    return {
        "match_rows": match_rows,
        "summary_rows": summary_rows,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    outputs = build_phase3_outputs(args)
    print(f"Saved {len(outputs['match_rows'])} matched-control rows to {args.match_output}")
    print(f"Saved {len(outputs['summary_rows'])} target summary rows to {args.summary_output}")
    print(f"Saved Phase 3 readout to {args.story_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
