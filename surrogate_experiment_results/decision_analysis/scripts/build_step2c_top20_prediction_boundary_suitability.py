#!/usr/bin/env python3
"""Build Phase 4 top20 prediction-boundary DFL suitability features for Step2c."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_step2c_graph_level_suitability import (  # noqa: E402
    ASSOCIATION_FIELDS,
    SELECTED_GRAPHS,
    auroc,
    finite_mean,
    finite_std,
    parse_float,
    percentile_rank,
    read_csv,
    spearman,
    write_csv,
)
from build_step2c_prediction_boundary_suitability import (  # noqa: E402
    PHASE2_FEATURE_FAMILIES,
    graph_sort_key,
)


AUDIT_DIR = PROJECT_ROOT / "surrogate_experiment_results" / "Step2c Graph-Level DFL Suitability Audit"
DEFAULT_RESULTS_DIR = AUDIT_DIR / "results"
DEFAULT_PRESENTATION_DIR = AUDIT_DIR / "presentation"
DEFAULT_TOP20_INPUT = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "decision_analysis"
    / "results"
    / "all400_model_seed_baseline"
    / "step2c_all400_all50_top20_2stage.csv"
)
DEFAULT_PHASE2_JOIN_INPUT = DEFAULT_RESULTS_DIR / "step2c_all400_graph_boundary_outcome_table.csv"
DEFAULT_BOUNDARY_OUTPUT = DEFAULT_RESULTS_DIR / "step2c_all400_top20_prediction_boundary_features.csv"
DEFAULT_JOIN_OUTPUT = DEFAULT_RESULTS_DIR / "step2c_all400_graph_top20_boundary_outcome_table.csv"
DEFAULT_ASSOC_OUTPUT = DEFAULT_RESULTS_DIR / "step2c_phase4_top20_feature_family_association.csv"
DEFAULT_OVERLAY_OUTPUT = DEFAULT_RESULTS_DIR / "step2c_phase4_top20_selected_case_overlay.csv"
DEFAULT_MATCHED_CONTROLS_INPUT = DEFAULT_RESULTS_DIR / "step2c_phase3_matched_controls.csv"
DEFAULT_TARGET_MATCHED_OUTPUT = DEFAULT_RESULTS_DIR / "step2c_phase4_top20_target_vs_matched_summary.csv"
DEFAULT_STORY_OUTPUT = DEFAULT_PRESENTATION_DIR / "step2c_phase4_top20_boundary_story.md"

TOP20_PREDICTION_BOUNDARY_FEATURES = (
    "median_2stage_top1_top20_pred_margin",
    "median_2stage_top1_top20_pred_margin_pct",
    "mean_2stage_top20_within_1pct_count",
    "mean_2stage_top20_within_5pct_count",
    "median_2stage_top20_mean_jaccard_to_rank1",
    "median_2stage_top20_diversity_from_rank1",
    "median_2stage_top20_mean_pairwise_jaccard",
    "median_2stage_top20_pairwise_diversity",
    "rank1_unique_signature_count",
    "rank1_unique_signature_rate",
    "rank1_modal_signature_rate",
    "top20_unique_signature_count_median",
    "ranking_ambiguity_top20_score",
)
TOP20_AMBIGUITY_COMPONENTS = (
    "median_2stage_top1_top20_pred_margin_pct",
    "mean_2stage_top20_within_1pct_count",
    "median_2stage_top20_diversity_from_rank1",
    "median_2stage_top20_pairwise_diversity",
)

PHASE4_FEATURE_FAMILIES = {
    **PHASE2_FEATURE_FAMILIES,
    "top20_prediction_boundary": TOP20_PREDICTION_BOUNDARY_FEATURES,
}
PHASE4_FEATURE_KEYS = tuple(
    feature for features in PHASE4_FEATURE_FAMILIES.values() for feature in features
)

BOUNDARY_FIELDS = ["graph_id", "seed_count", *TOP20_PREDICTION_BOUNDARY_FEATURES]
TARGET_MATCHED_FEATURES = (
    "ranking_ambiguity_top20_score",
    "mean_2stage_top20_within_1pct_count",
    "median_2stage_top20_pairwise_diversity",
    "median_2stage_top1_top20_pred_margin_pct",
)
TARGET_MATCHED_FIELDS = [
    "target_graph_id",
    "target_case_group",
    "n_controls_with_top20",
    *[
        name
        for feature in TARGET_MATCHED_FEATURES
        for name in (
            f"target_{feature}",
            f"matched_{feature}_median",
            f"target_{feature}_percentile_within_matched",
        )
    ],
]


def finite_median(values: list[Any]) -> float:
    clean = sorted(value for value in (parse_float(item) for item in values) if math.isfinite(value))
    if not clean:
        return float("nan")
    mid = len(clean) // 2
    if len(clean) % 2:
        return float(clean[mid])
    return float((clean[mid - 1] + clean[mid]) / 2.0)


def normalize_margin(margin: float, rank1_obj: float) -> float:
    return float(margin / (abs(rank1_obj) + 1e-9))


def signature_set(signature: str) -> set[str]:
    if not signature:
        return set()
    return {part for part in str(signature).split("|") if part != ""}


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 1.0
    return float(len(left & right) / len(union))


def mean_pairwise_jaccard(signatures: list[str]) -> float:
    sets = [signature_set(signature) for signature in signatures if signature]
    if len(sets) < 2:
        return float("nan")
    return finite_mean([jaccard(left, right) for left, right in combinations(sets, 2)])


def build_top20_boundary_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_graph_seed: dict[tuple[str, int], dict[int, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        if row.get("method_label") != "2stage_val_mse":
            continue
        rank = int(float(row.get("solution_rank", 0)))
        if rank < 1 or rank > 20:
            continue
        graph_id = row["graph_id"]
        seed = int(float(row.get("subset_seed", 0)))
        by_graph_seed[(graph_id, seed)][rank] = row

    per_graph: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rank1_signatures: dict[str, list[str]] = defaultdict(list)
    for (graph_id, seed), ranks in sorted(by_graph_seed.items()):
        rank1 = ranks.get(1)
        if not rank1:
            continue
        rank1_obj = parse_float(rank1.get("predicted_obj"), 0.0)
        top20_margin = parse_float(ranks.get(20, {}).get("predicted_margin_from_best"), float("nan"))
        margin_pcts: list[float] = []
        jaccards_to_rank1: list[float] = []
        signatures: list[str] = []
        for rank in range(1, 21):
            candidate = ranks.get(rank)
            if not candidate:
                continue
            margin = parse_float(candidate.get("predicted_margin_from_best"), 0.0)
            margin_pcts.append(normalize_margin(margin, rank1_obj))
            signatures.append(candidate.get("solution_edge_signature", ""))
            if rank > 1:
                jaccards_to_rank1.append(parse_float(candidate.get("edge_jaccard_with_rank1"), float("nan")))

        pairwise_jaccard = mean_pairwise_jaccard(signatures)
        per_graph[graph_id].append(
            {
                "subset_seed": seed,
                "top1_top20_margin": top20_margin,
                "top1_top20_margin_pct": normalize_margin(top20_margin, rank1_obj),
                "top20_within_1pct_count": sum(value <= 0.01 + 1e-12 for value in margin_pcts),
                "top20_within_5pct_count": sum(value <= 0.05 + 1e-12 for value in margin_pcts),
                "top20_mean_jaccard_to_rank1": finite_mean(jaccards_to_rank1),
                "top20_diversity_from_rank1": 1.0 - finite_mean(jaccards_to_rank1),
                "top20_mean_pairwise_jaccard": pairwise_jaccard,
                "top20_pairwise_diversity": 1.0 - pairwise_jaccard,
                "top20_unique_signature_count": len(set(signature for signature in signatures if signature)),
            }
        )
        rank1_signatures[graph_id].append(rank1.get("solution_edge_signature", ""))

    output: list[dict[str, Any]] = []
    for graph_id in sorted(per_graph, key=graph_sort_key):
        seed_rows = per_graph[graph_id]
        signatures = [signature for signature in rank1_signatures[graph_id] if signature]
        signature_counts = Counter(signatures)
        seed_count = len(seed_rows)
        unique_count = len(signature_counts)
        modal_count = max(signature_counts.values(), default=0)
        output.append(
            {
                "graph_id": graph_id,
                "seed_count": seed_count,
                "median_2stage_top1_top20_pred_margin": finite_median(
                    [row["top1_top20_margin"] for row in seed_rows]
                ),
                "median_2stage_top1_top20_pred_margin_pct": finite_median(
                    [row["top1_top20_margin_pct"] for row in seed_rows]
                ),
                "mean_2stage_top20_within_1pct_count": finite_mean(
                    [row["top20_within_1pct_count"] for row in seed_rows]
                ),
                "mean_2stage_top20_within_5pct_count": finite_mean(
                    [row["top20_within_5pct_count"] for row in seed_rows]
                ),
                "median_2stage_top20_mean_jaccard_to_rank1": finite_median(
                    [row["top20_mean_jaccard_to_rank1"] for row in seed_rows]
                ),
                "median_2stage_top20_diversity_from_rank1": finite_median(
                    [row["top20_diversity_from_rank1"] for row in seed_rows]
                ),
                "median_2stage_top20_mean_pairwise_jaccard": finite_median(
                    [row["top20_mean_pairwise_jaccard"] for row in seed_rows]
                ),
                "median_2stage_top20_pairwise_diversity": finite_median(
                    [row["top20_pairwise_diversity"] for row in seed_rows]
                ),
                "rank1_unique_signature_count": unique_count,
                "rank1_unique_signature_rate": 0.0 if seed_count == 0 else unique_count / seed_count,
                "rank1_modal_signature_rate": 0.0 if seed_count == 0 else modal_count / seed_count,
                "top20_unique_signature_count_median": finite_median(
                    [row["top20_unique_signature_count"] for row in seed_rows]
                ),
                "ranking_ambiguity_top20_score": 0.0,
            }
        )
    add_top20_ambiguity_score(output)
    return output


def read_top20_boundary_rows(path: str | Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("method_label") == "2stage_val_mse":
                rows.append(row)
    return rows


def zscores(
    rows: list[dict[str, Any]],
    key: str,
    *,
    eligible_graphs: set[str] | None = None,
) -> dict[str, float]:
    values = [parse_float(row.get(key)) for row in rows]
    clean = [value for value in values if math.isfinite(value)]
    mean = finite_mean(clean)
    std = finite_std(clean)
    output: dict[str, float] = {}
    for row, value in zip(rows, values):
        graph_id = str(row["graph_id"])
        if eligible_graphs is not None and graph_id not in eligible_graphs:
            output[graph_id] = float("nan")
        else:
            output[graph_id] = 0.0 if std == 0 or not math.isfinite(value) else (value - mean) / std
    return output


def add_top20_ambiguity_score(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    eligible_graphs = {
        str(row["graph_id"])
        for row in rows
        if any(math.isfinite(parse_float(row.get(feature))) for feature in TOP20_AMBIGUITY_COMPONENTS)
    }
    z_margin = zscores(
        rows,
        "median_2stage_top1_top20_pred_margin_pct",
        eligible_graphs=eligible_graphs,
    )
    z_within = zscores(
        rows,
        "mean_2stage_top20_within_1pct_count",
        eligible_graphs=eligible_graphs,
    )
    z_rank1_diversity = zscores(
        rows,
        "median_2stage_top20_diversity_from_rank1",
        eligible_graphs=eligible_graphs,
    )
    z_pairwise_diversity = zscores(
        rows,
        "median_2stage_top20_pairwise_diversity",
        eligible_graphs=eligible_graphs,
    )
    for row in rows:
        graph_id = str(row["graph_id"])
        if graph_id not in eligible_graphs:
            row["ranking_ambiguity_top20_score"] = float("nan")
        else:
            row["ranking_ambiguity_top20_score"] = (
                -z_margin[graph_id]
                + z_within[graph_id]
                + z_rank1_diversity[graph_id]
                + z_pairwise_diversity[graph_id]
            )


def join_phase2_with_top20(
    phase2_rows: list[dict[str, Any]],
    top20_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    top20_by_graph = {row["graph_id"]: dict(row) for row in top20_rows}
    joined: list[dict[str, Any]] = []
    for row in phase2_rows:
        graph_id = row["graph_id"]
        output = dict(row)
        output.update(top20_by_graph.get(graph_id, {}))
        joined.append(output)
    add_top20_ambiguity_score(joined)
    return joined


def feature_family(feature: str) -> str:
    for family, features in PHASE4_FEATURE_FAMILIES.items():
        if feature in features:
            return family
    return "other"


def build_phase4_association_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for feature in PHASE4_FEATURE_KEYS:
        values: list[float] = []
        deltas: list[float] = []
        helpful: list[int] = []
        harmful: list[int] = []
        for row in rows:
            value = parse_float(row.get(feature))
            delta = parse_float(row.get("median_delta_pp"))
            if not math.isfinite(value) or not math.isfinite(delta):
                continue
            values.append(value)
            deltas.append(delta)
            helpful.append(int(parse_float(row.get("helpful_graph"), 0.0)))
            harmful.append(int(parse_float(row.get("harmful_graph"), 0.0)))
        output.append(
            {
                "feature_family": feature_family(feature),
                "feature": feature,
                "n_graphs": len(values),
                "spearman_median_delta_pp": spearman(values, deltas),
                "auroc_helpful": auroc(values, helpful),
                "auroc_harmful": auroc(values, harmful),
            }
        )
    return output


def build_phase4_selected_overlay_rows(
    rows: list[dict[str, Any]],
    selected_graphs: tuple[str, ...] = SELECTED_GRAPHS,
) -> list[dict[str, Any]]:
    features = (
        "ranking_ambiguity_score",
        "ranking_ambiguity_top20_score",
        "median_2stage_top1_top20_pred_margin_pct",
        "mean_2stage_top20_within_1pct_count",
        "median_2stage_top20_pairwise_diversity",
        "rank1_modal_signature_rate",
    )
    by_graph = {str(row["graph_id"]): row for row in rows}
    feature_values = {feature: [parse_float(row.get(feature)) for row in rows] for feature in features}
    output: list[dict[str, Any]] = []
    for graph_id in selected_graphs:
        if graph_id not in by_graph:
            continue
        source = by_graph[graph_id]
        row: dict[str, Any] = {
            "graph_id": graph_id,
            "median_delta_pp": source.get("median_delta_pp", ""),
            "strict_case_c_rate": source.get("strict_case_c_rate", ""),
            "helpful_graph": source.get("helpful_graph", ""),
            "harmful_graph": source.get("harmful_graph", ""),
        }
        for feature in features:
            value = parse_float(source.get(feature))
            row[feature] = value
            row[f"{feature}_percentile"] = percentile_rank(feature_values[feature], value)
        output.append(row)
    return output


def build_top20_target_vs_matched_summary_rows(
    rows: list[dict[str, Any]],
    match_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_graph = {str(row["graph_id"]): row for row in rows}
    controls_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in match_rows:
        controls_by_target[str(row["target_graph_id"])].append(row)

    output: list[dict[str, Any]] = []
    for target_graph_id in sorted(controls_by_target, key=graph_sort_key):
        target = by_graph.get(target_graph_id)
        if not target:
            continue
        matches = controls_by_target[target_graph_id]
        controls = [
            by_graph[str(row["control_graph_id"])]
            for row in matches
            if str(row["control_graph_id"]) in by_graph
            and math.isfinite(
                parse_float(by_graph[str(row["control_graph_id"])].get("ranking_ambiguity_top20_score"))
            )
        ]
        row: dict[str, Any] = {
            "target_graph_id": target_graph_id,
            "target_case_group": matches[0].get("target_case_group", "") if matches else "",
            "n_controls_with_top20": len(controls),
        }
        for feature in TARGET_MATCHED_FEATURES:
            target_value = parse_float(target.get(feature))
            control_values = [parse_float(control.get(feature)) for control in controls]
            row[f"target_{feature}"] = target_value
            row[f"matched_{feature}_median"] = finite_median(control_values)
            row[f"target_{feature}_percentile_within_matched"] = percentile_rank(
                control_values,
                target_value,
            )
        output.append(row)
    return output


def fieldnames_from_rows(rows: list[dict[str, Any]]) -> list[str]:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    return keys


def best_by_family(
    association_rows: list[dict[str, Any]],
    *,
    metric: str,
    absolute: bool = False,
) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in association_rows:
        family = row["feature_family"]
        score = parse_float(row.get(metric))
        if absolute:
            score = abs(score)
        current = best.get(family)
        current_score = parse_float(current.get(metric)) if current else float("-inf")
        if current and absolute:
            current_score = abs(current_score)
        if current is None or score > current_score:
            best[family] = row
    return best


def write_phase4_story(
    path: str | Path,
    *,
    joined_rows: list[dict[str, Any]],
    association_rows: list[dict[str, Any]],
    overlay_rows: list[dict[str, Any]],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_helpful = best_by_family(association_rows, metric="auroc_helpful")
    top20 = best_helpful.get("top20_prediction_boundary", {})
    top20_covered = sum(
        math.isfinite(parse_float(row.get("ranking_ambiguity_top20_score"))) for row in joined_rows
    )

    lines = [
        "# Step2c Graph-Level DFL Suitability: Phase 4 Top20 Boundary Readout",
        "",
        "## Scope",
        "",
        "Phase 4 extends Phase 2 from 2stage top5 to 2stage top20 candidate-boundary diagnostics. "
        "The raw top20 candidate artifact is generated outside this audit directory and is not committed to git.",
        "",
        "## Population",
        "",
        f"- joined graphs: {len(joined_rows)}",
        f"- top20-covered graphs: {top20_covered}",
        "",
        "## Top20 Boundary Signal",
        "",
        "| Feature | Helpful AUROC | Harmful AUROC | Spearman with median Delta |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| {top20.get('feature', 'NA')} | "
            f"{parse_float(top20.get('auroc_helpful')):.3f} | "
            f"{parse_float(top20.get('auroc_harmful')):.3f} | "
            f"{parse_float(top20.get('spearman_median_delta_pp')):.3f} |"
        ),
        "",
        "## Selected Case Overlay",
        "",
        "| Graph | median Delta pp | top5 ambiguity pct | top20 ambiguity pct | top20 within-1pct pct | top20 diversity pct |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in overlay_rows:
        lines.append(
            f"| {row['graph_id']} | {parse_float(row.get('median_delta_pp')):.2f} | "
            f"{parse_float(row.get('ranking_ambiguity_score_percentile')):.2f} | "
            f"{parse_float(row.get('ranking_ambiguity_top20_score_percentile')):.2f} | "
            f"{parse_float(row.get('mean_2stage_top20_within_1pct_count_percentile')):.2f} | "
            f"{parse_float(row.get('median_2stage_top20_pairwise_diversity_percentile')):.2f} |"
        )

    lines.extend(
        [
            "",
            "## Report-Safe Interpretation",
            "",
            "This is a post-training boundary diagnostic, not a topology-only rule. "
            "Use it to test whether the broader 2stage candidate landscape explains cases that top5 diagnostics under-rank.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Phase 4 top20 prediction-boundary DFL suitability tables for Step2c."
    )
    parser.add_argument("--top20-input", type=Path, default=DEFAULT_TOP20_INPUT)
    parser.add_argument("--phase2-join-input", type=Path, default=DEFAULT_PHASE2_JOIN_INPUT)
    parser.add_argument("--boundary-output", type=Path, default=DEFAULT_BOUNDARY_OUTPUT)
    parser.add_argument("--join-output", type=Path, default=DEFAULT_JOIN_OUTPUT)
    parser.add_argument("--association-output", type=Path, default=DEFAULT_ASSOC_OUTPUT)
    parser.add_argument("--overlay-output", type=Path, default=DEFAULT_OVERLAY_OUTPUT)
    parser.add_argument("--matched-controls-input", type=Path, default=DEFAULT_MATCHED_CONTROLS_INPUT)
    parser.add_argument("--target-matched-output", type=Path, default=DEFAULT_TARGET_MATCHED_OUTPUT)
    parser.add_argument("--story-output", type=Path, default=DEFAULT_STORY_OUTPUT)
    return parser.parse_args(argv)


def build_phase4_outputs(args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    top20_rows = read_top20_boundary_rows(args.top20_input)
    boundary_rows = build_top20_boundary_rows(top20_rows)
    phase2_rows = read_csv(args.phase2_join_input)
    joined_rows = join_phase2_with_top20(phase2_rows, boundary_rows)
    association_rows = build_phase4_association_rows(joined_rows)
    overlay_rows = build_phase4_selected_overlay_rows(joined_rows)
    match_rows = read_csv(args.matched_controls_input) if args.matched_controls_input.exists() else []
    target_matched_rows = build_top20_target_vs_matched_summary_rows(joined_rows, match_rows)

    write_csv(args.boundary_output, boundary_rows, BOUNDARY_FIELDS)
    write_csv(args.join_output, joined_rows, fieldnames_from_rows(joined_rows))
    write_csv(args.association_output, association_rows, ASSOCIATION_FIELDS)
    write_csv(args.overlay_output, overlay_rows, fieldnames_from_rows(overlay_rows))
    write_csv(args.target_matched_output, target_matched_rows, TARGET_MATCHED_FIELDS)
    write_phase4_story(
        args.story_output,
        joined_rows=joined_rows,
        association_rows=association_rows,
        overlay_rows=overlay_rows,
    )
    return {
        "boundary_rows": boundary_rows,
        "joined_rows": joined_rows,
        "association_rows": association_rows,
        "overlay_rows": overlay_rows,
        "target_matched_rows": target_matched_rows,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    outputs = build_phase4_outputs(args)
    print(f"Saved {len(outputs['boundary_rows'])} top20 boundary rows to {args.boundary_output}")
    print(f"Saved {len(outputs['joined_rows'])} graph-top20 rows to {args.join_output}")
    print(f"Saved {len(outputs['association_rows'])} Phase 4 association rows to {args.association_output}")
    print(f"Saved {len(outputs['overlay_rows'])} selected case rows to {args.overlay_output}")
    print(f"Saved {len(outputs['target_matched_rows'])} target-matched rows to {args.target_matched_output}")
    print(f"Saved Phase 4 readout to {args.story_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
