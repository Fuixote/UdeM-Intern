#!/usr/bin/env python3
"""Build Phase 2 prediction-boundary DFL suitability features for Step2c."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_step2c_graph_level_suitability import (  # noqa: E402
    ASSOCIATION_FIELDS,
    DEFAULT_RESULTS_DIR as PHASE1_RESULTS_DIR,
    FEATURE_FAMILIES as PHASE1_FEATURE_FAMILIES,
    FEATURE_KEYS as PHASE1_FEATURE_KEYS,
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


AUDIT_DIR = PROJECT_ROOT / "surrogate_experiment_results" / "Step2c Graph-Level DFL Suitability Audit"
DEFAULT_RESULTS_DIR = AUDIT_DIR / "results"
DEFAULT_PRESENTATION_DIR = AUDIT_DIR / "presentation"
DEFAULT_TOP5_INPUT = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "decision_analysis"
    / "results"
    / "all400_model_seed_baseline"
    / "step2c_all400_all50_top5_second_best.csv"
)
DEFAULT_PHASE1_JOIN_INPUT = DEFAULT_RESULTS_DIR / "step2c_all400_graph_feature_outcome_table.csv"
DEFAULT_BOUNDARY_OUTPUT = DEFAULT_RESULTS_DIR / "step2c_all400_prediction_boundary_features.csv"
DEFAULT_JOIN_OUTPUT = DEFAULT_RESULTS_DIR / "step2c_all400_graph_boundary_outcome_table.csv"
DEFAULT_ASSOC_OUTPUT = DEFAULT_RESULTS_DIR / "step2c_phase2_feature_family_association.csv"
DEFAULT_OVERLAY_OUTPUT = DEFAULT_RESULTS_DIR / "step2c_phase2_selected_case_overlay.csv"
DEFAULT_STORY_OUTPUT = DEFAULT_PRESENTATION_DIR / "step2c_phase2_dfl_suitability_story.md"

PREDICTION_BOUNDARY_FEATURES = (
    "median_2stage_top1_top2_pred_margin",
    "median_2stage_top1_top5_pred_margin",
    "median_2stage_top1_top2_pred_margin_pct",
    "median_2stage_top1_top5_pred_margin_pct",
    "mean_2stage_top5_within_1pct_count",
    "mean_2stage_top5_within_5pct_count",
    "median_2stage_top5_mean_jaccard_to_rank1",
    "median_2stage_top5_diversity_from_rank1",
    "rank1_unique_signature_count",
    "rank1_unique_signature_rate",
    "rank1_modal_signature_rate",
    "ranking_ambiguity_score",
)

PHASE2_FEATURE_FAMILIES = {
    **PHASE1_FEATURE_FAMILIES,
    "prediction_boundary": PREDICTION_BOUNDARY_FEATURES,
}
PHASE2_FEATURE_KEYS = tuple(
    feature for features in PHASE2_FEATURE_FAMILIES.values() for feature in features
)

BOUNDARY_FIELDS = ["graph_id", "seed_count", *PREDICTION_BOUNDARY_FEATURES]
JOIN_FIELDS = None


def graph_sort_key(graph_id: str) -> tuple[int, int | str]:
    stem = graph_id.removesuffix(".json")
    if stem.startswith("G-"):
        try:
            return (0, int(stem.split("-", 1)[1]))
        except ValueError:
            pass
    return (1, stem)


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


def build_prediction_boundary_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_graph_seed: dict[tuple[str, int], dict[int, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        if row.get("method_label") != "2stage_val_mse":
            continue
        rank = int(float(row.get("solution_rank", 0)))
        if rank < 1 or rank > 5:
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
        top2_margin = parse_float(ranks.get(2, {}).get("predicted_margin_from_best"), float("nan"))
        top5_margin = parse_float(ranks.get(5, {}).get("predicted_margin_from_best"), float("nan"))
        margin_pcts: list[float] = []
        jaccards: list[float] = []
        for rank in range(1, 6):
            candidate = ranks.get(rank)
            if not candidate:
                continue
            margin = parse_float(candidate.get("predicted_margin_from_best"), 0.0)
            margin_pcts.append(normalize_margin(margin, rank1_obj))
            if rank > 1:
                jaccards.append(parse_float(candidate.get("edge_jaccard_with_rank1"), float("nan")))

        per_graph[graph_id].append(
            {
                "subset_seed": seed,
                "top1_top2_margin": top2_margin,
                "top1_top5_margin": top5_margin,
                "top1_top2_margin_pct": normalize_margin(top2_margin, rank1_obj),
                "top1_top5_margin_pct": normalize_margin(top5_margin, rank1_obj),
                "top5_within_1pct_count": sum(value <= 0.01 + 1e-12 for value in margin_pcts),
                "top5_within_5pct_count": sum(value <= 0.05 + 1e-12 for value in margin_pcts),
                "top5_mean_jaccard_to_rank1": finite_mean(jaccards),
                "top5_diversity_from_rank1": 1.0 - finite_mean(jaccards),
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
                "median_2stage_top1_top2_pred_margin": finite_median(
                    [row["top1_top2_margin"] for row in seed_rows]
                ),
                "median_2stage_top1_top5_pred_margin": finite_median(
                    [row["top1_top5_margin"] for row in seed_rows]
                ),
                "median_2stage_top1_top2_pred_margin_pct": finite_median(
                    [row["top1_top2_margin_pct"] for row in seed_rows]
                ),
                "median_2stage_top1_top5_pred_margin_pct": finite_median(
                    [row["top1_top5_margin_pct"] for row in seed_rows]
                ),
                "mean_2stage_top5_within_1pct_count": finite_mean(
                    [row["top5_within_1pct_count"] for row in seed_rows]
                ),
                "mean_2stage_top5_within_5pct_count": finite_mean(
                    [row["top5_within_5pct_count"] for row in seed_rows]
                ),
                "median_2stage_top5_mean_jaccard_to_rank1": finite_median(
                    [row["top5_mean_jaccard_to_rank1"] for row in seed_rows]
                ),
                "median_2stage_top5_diversity_from_rank1": finite_median(
                    [row["top5_diversity_from_rank1"] for row in seed_rows]
                ),
                "rank1_unique_signature_count": unique_count,
                "rank1_unique_signature_rate": 0.0 if seed_count == 0 else unique_count / seed_count,
                "rank1_modal_signature_rate": 0.0 if seed_count == 0 else modal_count / seed_count,
                "ranking_ambiguity_score": 0.0,
            }
        )
    add_ranking_ambiguity_score(output)
    return output


def read_top5_boundary_rows(path: str | Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("method_label") == "2stage_val_mse":
                rows.append(row)
    return rows


def zscores(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = [parse_float(row.get(key)) for row in rows]
    clean = [value for value in values if math.isfinite(value)]
    mean = finite_mean(clean)
    std = finite_std(clean)
    output: dict[str, float] = {}
    for row, value in zip(rows, values):
        graph_id = str(row["graph_id"])
        output[graph_id] = 0.0 if std == 0 or not math.isfinite(value) else (value - mean) / std
    return output


def add_ranking_ambiguity_score(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    z_top2 = zscores(rows, "median_2stage_top1_top2_pred_margin_pct")
    z_top5 = zscores(rows, "median_2stage_top1_top5_pred_margin_pct")
    z_within = zscores(rows, "mean_2stage_top5_within_1pct_count")
    z_diversity = zscores(rows, "median_2stage_top5_diversity_from_rank1")
    for row in rows:
        graph_id = str(row["graph_id"])
        row["ranking_ambiguity_score"] = (
            -z_top2[graph_id] - z_top5[graph_id] + z_within[graph_id] + z_diversity[graph_id]
        )


def join_phase1_with_boundary(
    phase1_rows: list[dict[str, Any]],
    boundary_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    boundary_by_graph = {row["graph_id"]: dict(row) for row in boundary_rows}
    joined: list[dict[str, Any]] = []
    for row in phase1_rows:
        graph_id = row["graph_id"]
        output = dict(row)
        output.update(boundary_by_graph.get(graph_id, {}))
        joined.append(output)
    add_ranking_ambiguity_score(joined)
    return joined


def feature_family(feature: str) -> str:
    for family, features in PHASE2_FEATURE_FAMILIES.items():
        if feature in features:
            return family
    return "other"


def build_phase2_association_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for feature in PHASE2_FEATURE_KEYS:
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


def build_phase2_selected_overlay_rows(
    rows: list[dict[str, Any]],
    selected_graphs: tuple[str, ...] = SELECTED_GRAPHS,
) -> list[dict[str, Any]]:
    features = (
        "ranking_ambiguity_score",
        "median_2stage_top1_top2_pred_margin_pct",
        "median_2stage_top1_top5_pred_margin_pct",
        "mean_2stage_top5_within_1pct_count",
        "median_2stage_top5_diversity_from_rank1",
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


def write_phase2_story(
    path: str | Path,
    *,
    joined_rows: list[dict[str, Any]],
    association_rows: list[dict[str, Any]],
    overlay_rows: list[dict[str, Any]],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_spearman = best_by_family(association_rows, metric="spearman_median_delta_pp", absolute=True)
    best_helpful = best_by_family(association_rows, metric="auroc_helpful")
    best_harmful = best_by_family(association_rows, metric="auroc_harmful")

    lines = [
        "# Step2c Graph-Level DFL Suitability: Phase 2 Readout",
        "",
        "## Scope",
        "",
        "Phase 2 adds prediction-boundary features from existing all-400 2stage top5 candidate lists. "
        "No all-400 top20 rerun is used in this phase.",
        "",
        "## Population",
        "",
        f"- graphs: {len(joined_rows)}",
        "",
        "## Best Spearman Association By Feature Family",
        "",
        "| Family | Feature | Spearman with median Delta | AUROC helpful | AUROC harmful |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for family in [
        "raw_topology",
        "cycle_chain",
        "exchange_geometry",
        "conflict_geometry",
        "prediction_boundary",
    ]:
        row = best_spearman.get(family, {})
        lines.append(
            f"| {family} | {row.get('feature', 'NA')} | "
            f"{parse_float(row.get('spearman_median_delta_pp')):.3f} | "
            f"{parse_float(row.get('auroc_helpful')):.3f} | "
            f"{parse_float(row.get('auroc_harmful')):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Best Helpful / Harmful AUROC By Feature Family",
            "",
            "| Family | Best helpful feature | Helpful AUROC | Best harmful feature | Harmful AUROC |",
            "| --- | --- | ---: | --- | ---: |",
        ]
    )
    for family in [
        "raw_topology",
        "cycle_chain",
        "exchange_geometry",
        "conflict_geometry",
        "prediction_boundary",
    ]:
        helpful = best_helpful.get(family, {})
        harmful = best_harmful.get(family, {})
        lines.append(
            f"| {family} | {helpful.get('feature', 'NA')} | "
            f"{parse_float(helpful.get('auroc_helpful')):.3f} | "
            f"{harmful.get('feature', 'NA')} | {parse_float(harmful.get('auroc_harmful')):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Selected Case Overlay",
            "",
            "| Graph | median Delta pp | ambiguity pct | top1-top2 margin pct percentile | within-1pct count pct | diversity pct | modal rank1 rate |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in overlay_rows:
        lines.append(
            f"| {row['graph_id']} | {parse_float(row.get('median_delta_pp')):.2f} | "
            f"{parse_float(row.get('ranking_ambiguity_score_percentile')):.2f} | "
            f"{parse_float(row.get('median_2stage_top1_top2_pred_margin_pct_percentile')):.2f} | "
            f"{parse_float(row.get('mean_2stage_top5_within_1pct_count_percentile')):.2f} | "
            f"{parse_float(row.get('median_2stage_top5_diversity_from_rank1_percentile')):.2f} | "
            f"{parse_float(row.get('rank1_modal_signature_rate')):.2f} |"
        )

    lines.extend(
        [
            "",
            "## Report-Safe Interpretation",
            "",
            "This is still an association audit. Prediction-boundary features are available after training a standard 2stage model, not before any modeling. "
            "Use these results to test whether learned candidate margins add signal beyond raw graph and feasible-set geometry.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Phase 2 prediction-boundary DFL suitability tables for Step2c."
    )
    parser.add_argument("--top5-input", type=Path, default=DEFAULT_TOP5_INPUT)
    parser.add_argument("--phase1-join-input", type=Path, default=DEFAULT_PHASE1_JOIN_INPUT)
    parser.add_argument("--boundary-output", type=Path, default=DEFAULT_BOUNDARY_OUTPUT)
    parser.add_argument("--join-output", type=Path, default=DEFAULT_JOIN_OUTPUT)
    parser.add_argument("--association-output", type=Path, default=DEFAULT_ASSOC_OUTPUT)
    parser.add_argument("--overlay-output", type=Path, default=DEFAULT_OVERLAY_OUTPUT)
    parser.add_argument("--story-output", type=Path, default=DEFAULT_STORY_OUTPUT)
    return parser.parse_args(argv)


def build_phase2_outputs(args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    top5_rows = read_top5_boundary_rows(args.top5_input)
    boundary_rows = build_prediction_boundary_rows(top5_rows)
    phase1_rows = read_csv(args.phase1_join_input)
    joined_rows = join_phase1_with_boundary(phase1_rows, boundary_rows)
    association_rows = build_phase2_association_rows(joined_rows)
    overlay_rows = build_phase2_selected_overlay_rows(joined_rows)

    write_csv(args.boundary_output, boundary_rows, BOUNDARY_FIELDS)
    write_csv(args.join_output, joined_rows, fieldnames_from_rows(joined_rows))
    write_csv(args.association_output, association_rows, ASSOCIATION_FIELDS)
    write_csv(args.overlay_output, overlay_rows, fieldnames_from_rows(overlay_rows))
    write_phase2_story(
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
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    outputs = build_phase2_outputs(args)
    print(f"Saved {len(outputs['boundary_rows'])} prediction-boundary rows to {args.boundary_output}")
    print(f"Saved {len(outputs['joined_rows'])} graph-boundary rows to {args.join_output}")
    print(f"Saved {len(outputs['association_rows'])} Phase 2 association rows to {args.association_output}")
    print(f"Saved {len(outputs['overlay_rows'])} selected case rows to {args.overlay_output}")
    print(f"Saved Phase 2 readout to {args.story_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
