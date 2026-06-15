#!/usr/bin/env python3
"""Build paper-facing presentation artifacts for the Step2c mechanism audit."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUDIT_DIR = PROJECT_ROOT / "surrogate_experiment_results" / "Step2c Mechanism Dissection Audit"
RESULTS_DIR = AUDIT_DIR / "results"
DEFAULT_PRESENTATION_DIR = AUDIT_DIR / "presentation"

DEFAULT_ATLAS_INPUT = RESULTS_DIR / "step2c_selected_graphs_mechanism_atlas.csv"
DEFAULT_RANK_SUMMARY_INPUT = RESULTS_DIR / "step2c_rank_reversal_summary.csv"
DEFAULT_EDGE_SUMMARY_INPUT = RESULTS_DIR / "step2c_critical_edge_summary.csv"
DEFAULT_CRITICAL_EDGE_INPUT = RESULTS_DIR / "step2c_critical_edge_table.csv"

DEFAULT_GRAPH_ORDER = [
    "G-392.json",
    "G-1285.json",
    "G-1560.json",
    "G-1169.json",
    "G-1449.json",
    "G-142.json",
    "G-946.json",
    "G-14.json",
    "G-163.json",
]
DEFAULT_CASE_PANEL_GRAPHS = ["G-392", "G-1285", "G-1560", "G-14"]

MECHANISM_LABELS = {
    "G-392.json": "Deep-candidate correction",
    "G-1285.json": "Clean exact rank-2 promotion",
    "G-1560.json": "Large-effect top-K promotion",
    "G-1169.json": "Broad top20 promotion",
    "G-1449.json": "Deep top20 promotion outside top5",
    "G-142.json": "Both-poor negative control",
    "G-946.json": "Both-poor negative control",
    "G-14.json": "Harmful SPO+ reranking",
    "G-163.json": "Harmful SPO+ reranking",
}

INTERPRETATIONS = {
    "G-392": "2stage has the near-oracle candidate only deep in top20; SPO+ makes it rank1.",
    "G-1285": "2stage rank2 is the oracle solution; SPO+ promotes it exactly to rank1.",
    "G-1560": "Largest-effect promotion case; SPO+ promotes a near-oracle top-K candidate.",
    "G-1169": "SPO+ selects a near-oracle candidate from deeper in the 2stage top20 list.",
    "G-1449": "A top5-unexplained case becomes deep top20 promotion after larger-K audit.",
    "G-142": "No near-oracle candidate is present in 2stage top20; both methods keep the same bad rank1.",
    "G-946": "No near-oracle candidate is present in 2stage top20; both methods keep the same bad rank1.",
    "G-14": "SPO+ promotes a lower-ranked candidate, but that candidate is worse under true labels.",
    "G-163": "SPO+ reranks away from a better 2stage decision toward a worse candidate.",
}

PAPER_FIELDS = [
    "graph_id",
    "mechanism_label",
    "story_role",
    "b_found_rate",
    "spoplus_equals_b_rate",
    "c_rank_under_2stage",
    "c_true_rank",
    "two_stage_rank1_gap_pct",
    "spoplus_rank1_gap_pct",
    "delta_gap_pp",
    "true_delta_a_to_c",
    "two_stage_pred_delta_a_to_c",
    "spoplus_pred_delta_a_to_c",
    "rank_reversal_pattern",
    "interpretation",
]

EDGE_FIELDS = [
    "graph_id",
    "comparison",
    "edge_rank",
    "edge_id",
    "src",
    "dst",
    "edge_role",
    "edge_frequency",
    "mean_signed_true_delta",
    "mean_signed_pred_2stage_delta",
    "mean_signed_pred_spoplus_delta",
    "mean_spoplus_minus_2stage_pred",
    "mean_abs_error_2stage",
    "mean_abs_error_spoplus",
    "edge_in_oracle_rate",
    "interpretation",
]

CASE_PANEL_FIELDS = [
    "panel",
    "graph_id",
    "mechanism_label",
    "outcome",
    "c_rank_under_2stage",
    "c_true_rank",
    "true_delta_a_to_c",
    "two_stage_pred_delta_a_to_c",
    "spoplus_pred_delta_a_to_c",
]

FIGURE_DATA_FIELDS = ["graph_id", "series", "value"]


def read_csv(path: str | Path) -> list[dict[str, str]]:
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
    try:
        return float(value)
    except ValueError:
        return default


def fmt(value: Any, digits: int = 2, *, na: str = "NA") -> str:
    parsed = parse_float(value)
    if not math.isfinite(parsed):
        return na
    return f"{parsed:.{digits}f}"


def fmt_rate(value: Any) -> str:
    return fmt(value, 2)


def fmt_rank(value: Any) -> str:
    parsed = parse_float(value)
    if not math.isfinite(parsed):
        return "not in top50"
    if abs(parsed - round(parsed)) < 1e-9:
        return str(int(round(parsed)))
    return fmt(parsed, 1)


def graph_stem(graph_id: str) -> str:
    return graph_id[:-5] if graph_id.endswith(".json") else graph_id


def normalize_graph_id(graph_id: str) -> str:
    return graph_id if graph_id.endswith(".json") else f"{graph_id}.json"


def story_role(graph_id: str) -> str:
    graph = graph_stem(graph_id)
    if graph in {"G-392", "G-1285", "G-1560"}:
        return "primary_success_case"
    if graph in {"G-1169", "G-1449"}:
        return "deep_promotion_case"
    if graph in {"G-142", "G-946"}:
        return "both_poor_control"
    return "harmful_reranking_control"


def edge_summary_lookup(
    edge_summary_rows: list[dict[str, str]],
) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    for row in edge_summary_rows:
        if row.get("comparison") == "two_stage_rank1_vs_spoplus_rank1":
            lookup[row["graph_id"]] = row
    return lookup


def reversal_pattern(true_delta: float, pred2_delta: float, predspo_delta: float) -> str:
    if abs(true_delta) < 1e-9 and abs(pred2_delta) < 1e-9 and abs(predspo_delta) < 1e-9:
        return "same_rank1_or_no_reversal"
    if true_delta > 0 and pred2_delta < 0 and predspo_delta > 0:
        return "helpful_reversal"
    if true_delta < 0 and predspo_delta > 0:
        return "harmful_reversal"
    return "mixed_or_partial_reversal"


def build_paper_mechanism_rows(
    *,
    atlas_rows: list[dict[str, str]],
    rank_summary_rows: list[dict[str, str]],
    edge_summary_rows: list[dict[str, str]],
    graph_order: list[str] | tuple[str, ...] = tuple(DEFAULT_GRAPH_ORDER),
    mechanism_labels: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    labels = mechanism_labels or MECHANISM_LABELS
    atlas_by_graph = {row["graph_id"]: row for row in atlas_rows}
    rank_by_graph = {row["graph_id"]: row for row in rank_summary_rows}
    edge_by_graph = edge_summary_lookup(edge_summary_rows)
    output: list[dict[str, str]] = []

    for graph_id in graph_order:
        graph_id = normalize_graph_id(graph_id)
        graph = graph_stem(graph_id)
        atlas = atlas_by_graph.get(graph_id, {})
        rank = rank_by_graph.get(graph_id, {})
        edge = edge_by_graph.get(graph_id, {})

        true_delta = parse_float(edge.get("median_signed_true_value_delta"), 0.0)
        pred2_delta = parse_float(edge.get("median_signed_pred_2stage_delta"), 0.0)
        predspo_delta = parse_float(edge.get("median_signed_pred_spoplus_delta"), 0.0)

        output.append(
            {
                "graph_id": graph,
                "mechanism_label": labels.get(graph_id, atlas.get("assigned_family", "")),
                "story_role": story_role(graph_id),
                "b_found_rate": fmt_rate(rank.get("best_near_found_rate", "")),
                "spoplus_equals_b_rate": fmt_rate(rank.get("spoplus_equals_best_near_rate", "")),
                "c_rank_under_2stage": fmt_rank(rank.get("median_spoplus_rank_under_2stage", "")),
                "c_true_rank": fmt_rank(rank.get("median_spoplus_true_rank", "")),
                "two_stage_rank1_gap_pct": fmt(atlas.get("median_two_stage_rank1_gap_pct", "")),
                "spoplus_rank1_gap_pct": fmt(atlas.get("median_spoplus_rank1_gap_pct", "")),
                "delta_gap_pp": fmt(atlas.get("median_delta_pp", "")),
                "true_delta_a_to_c": fmt(true_delta),
                "two_stage_pred_delta_a_to_c": fmt(pred2_delta),
                "spoplus_pred_delta_a_to_c": fmt(predspo_delta),
                "rank_reversal_pattern": reversal_pattern(
                    true_delta,
                    pred2_delta,
                    predspo_delta,
                ),
                "interpretation": INTERPRETATIONS.get(graph, ""),
            }
        )
    return output


def bool_value(value: Any) -> bool:
    return str(value) == "True"


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def edge_role(rows: list[dict[str, str]]) -> str:
    votes: Counter[str] = Counter()
    for row in rows:
        left = bool_value(row.get("edge_in_left"))
        right = bool_value(row.get("edge_in_right"))
        if left and not right:
            votes["removed_from_2stage_rank1"] += 1
        elif right and not left:
            votes["added_by_spoplus_rank1"] += 1
        else:
            votes["mixed_membership"] += 1
    return votes.most_common(1)[0][0] if votes else "mixed_membership"


def build_top_critical_edge_rows(
    edge_rows: list[dict[str, str]],
    *,
    graph_order: list[str] | tuple[str, ...] = tuple(DEFAULT_GRAPH_ORDER),
    comparison: str = "two_stage_rank1_vs_spoplus_rank1",
    top_n: int = 5,
) -> list[dict[str, str]]:
    graph_ids = [normalize_graph_id(graph_id) for graph_id in graph_order]
    rows_by_graph_edge: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    seed_counts: dict[str, set[str]] = defaultdict(set)
    for row in edge_rows:
        if row.get("comparison") != comparison:
            continue
        graph_id = row["graph_id"]
        if graph_id not in graph_ids:
            continue
        rows_by_graph_edge[(graph_id, row["edge_id"])].append(row)
        seed_counts[graph_id].add(row["subset_seed"])

    grouped_by_graph: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (graph_id, edge_id), group in rows_by_graph_edge.items():
        denom = max(1, len(seed_counts[graph_id]))
        signed_true = [parse_float(row["signed_true_value_delta_right_minus_left"]) for row in group]
        pred2 = [parse_float(row["signed_pred_2stage_delta_right_minus_left"]) for row in group]
        predspo = [parse_float(row["signed_pred_spoplus_delta_right_minus_left"]) for row in group]
        pred_shift = [parse_float(row["delta_prediction_spoplus_minus_2stage"]) for row in group]
        err2 = [abs(parse_float(row["error_2stage"])) for row in group]
        errspo = [abs(parse_float(row["error_spoplus"])) for row in group]
        oracle_rate = sum(bool_value(row.get("edge_in_oracle")) for row in group) / len(group)
        frequency = len({row["subset_seed"] for row in group}) / denom
        first = group[0]
        mean_true = mean(signed_true)
        grouped_by_graph[graph_id].append(
            {
                "graph_id": graph_stem(graph_id),
                "comparison": comparison,
                "edge_id": edge_id,
                "src": first.get("src", ""),
                "dst": first.get("dst", ""),
                "edge_role": edge_role(group),
                "edge_frequency_raw": frequency,
                "mean_signed_true_delta_raw": mean_true,
                "mean_signed_pred_2stage_delta_raw": mean(pred2),
                "mean_signed_pred_spoplus_delta_raw": mean(predspo),
                "mean_spoplus_minus_2stage_pred_raw": mean(pred_shift),
                "mean_abs_error_2stage_raw": mean(err2),
                "mean_abs_error_spoplus_raw": mean(errspo),
                "edge_in_oracle_rate_raw": oracle_rate,
                "sort_score": abs(mean_true) * frequency,
            }
        )

    output: list[dict[str, str]] = []
    for graph_id in graph_ids:
        candidates = sorted(
            grouped_by_graph.get(graph_id, []),
            key=lambda row: (-row["sort_score"], int(row["edge_id"])),
        )[:top_n]
        for idx, row in enumerate(candidates, start=1):
            output.append(
                {
                    "graph_id": row["graph_id"],
                    "comparison": row["comparison"],
                    "edge_rank": str(idx),
                    "edge_id": str(row["edge_id"]),
                    "src": str(row["src"]),
                    "dst": str(row["dst"]),
                    "edge_role": row["edge_role"],
                    "edge_frequency": fmt(row["edge_frequency_raw"]),
                    "mean_signed_true_delta": fmt(row["mean_signed_true_delta_raw"]),
                    "mean_signed_pred_2stage_delta": fmt(
                        row["mean_signed_pred_2stage_delta_raw"]
                    ),
                    "mean_signed_pred_spoplus_delta": fmt(
                        row["mean_signed_pred_spoplus_delta_raw"]
                    ),
                    "mean_spoplus_minus_2stage_pred": fmt(
                        row["mean_spoplus_minus_2stage_pred_raw"]
                    ),
                    "mean_abs_error_2stage": fmt(row["mean_abs_error_2stage_raw"]),
                    "mean_abs_error_spoplus": fmt(row["mean_abs_error_spoplus_raw"]),
                    "edge_in_oracle_rate": fmt(row["edge_in_oracle_rate_raw"]),
                    "interpretation": edge_interpretation(row["edge_role"]),
                }
            )
    return output


def edge_interpretation(role: str) -> str:
    if role == "added_by_spoplus_rank1":
        return "edge added by promoted SPO+ solution"
    if role == "removed_from_2stage_rank1":
        return "edge removed from 2stage rank1 solution"
    return "edge has mixed membership across seeds"


def build_case_panel_rows(
    paper_rows: list[dict[str, str]],
    case_graphs: list[str] | tuple[str, ...] = tuple(DEFAULT_CASE_PANEL_GRAPHS),
) -> list[dict[str, str]]:
    by_graph = {row["graph_id"]: row for row in paper_rows}
    output: list[dict[str, str]] = []
    for idx, graph in enumerate(case_graphs):
        graph = graph_stem(graph)
        row = by_graph[graph]
        true_delta = parse_float(row["true_delta_a_to_c"], 0.0)
        if true_delta > 0:
            outcome = "SPO+ helps"
        elif true_delta < 0:
            outcome = "SPO+ hurts"
        else:
            outcome = "no decision change"
        output.append(
            {
                "panel": chr(ord("A") + idx),
                "graph_id": graph,
                "mechanism_label": row["mechanism_label"],
                "outcome": outcome,
                "c_rank_under_2stage": row.get("c_rank_under_2stage", ""),
                "c_true_rank": row.get("c_true_rank", ""),
                "true_delta_a_to_c": row["true_delta_a_to_c"],
                "two_stage_pred_delta_a_to_c": row.get("two_stage_pred_delta_a_to_c", ""),
                "spoplus_pred_delta_a_to_c": row.get("spoplus_pred_delta_a_to_c", ""),
            }
        )
    return output


def build_figure_data_rows(paper_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for row in paper_rows:
        for field, label in [
            ("true_delta_a_to_c", "True value delta"),
            ("two_stage_pred_delta_a_to_c", "2stage predicted delta"),
            ("spoplus_pred_delta_a_to_c", "SPO+ predicted delta"),
        ]:
            output.append(
                {
                    "graph_id": row["graph_id"],
                    "series": label,
                    "value": row[field],
                }
            )
    return output


def latex_escape(text: Any) -> str:
    value = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value


def write_latex_mechanism_table(path: str | Path, rows: list[dict[str, str]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        ("graph_id", "Graph"),
        ("mechanism_label", "Mechanism"),
        ("b_found_rate", "B found"),
        ("spoplus_equals_b_rate", "C=B"),
        ("c_rank_under_2stage", "2stage rank(C)"),
        ("c_true_rank", "true rank(C)"),
        ("true_delta_a_to_c", "$\\Delta_{true}$"),
        ("two_stage_pred_delta_a_to_c", "$\\Delta_{2stage}$"),
        ("spoplus_pred_delta_a_to_c", "$\\Delta_{SPO+}$"),
    ]
    with output.open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{llrrrrrrr}\n")
        handle.write("\\toprule\n")
        handle.write(" & ".join(label for _, label in columns) + " \\\\\n")
        handle.write("\\midrule\n")
        for row in rows:
            handle.write(
                " & ".join(latex_escape(row[key]) for key, _ in columns) + " \\\\\n"
            )
        handle.write("\\bottomrule\n")
        handle.write("\\end{tabular}\n")


def write_story_markdown(
    path: str | Path,
    paper_rows: list[dict[str, str]],
    case_rows: list[dict[str, str]],
    edge_rows: list[dict[str, str]],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Step2c Rank-Reversal Presentation Notes",
        "",
        "## Main Claim",
        "",
        "SPO+ helps in the selected success cases by reranking decision-critical feasible solutions. "
        "The 2stage model often has a near-oracle candidate in its top20 list, but ranks it below a worse rank1 decision; SPO+ reverses that ordering. "
        "The negative controls show the boundary: reranking fails when no near-oracle candidate is present, or when SPO+ promotes the wrong lower-ranked candidate.",
        "",
        "## Mechanism Table",
        "",
        "| Graph | Mechanism | Pattern | True delta | 2stage pred delta | SPO+ pred delta | Interpretation |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in paper_rows:
        lines.append(
            f"| {row['graph_id']} | {row['mechanism_label']} | {row['rank_reversal_pattern']} | "
            f"{row['true_delta_a_to_c']} | {row['two_stage_pred_delta_a_to_c']} | "
            f"{row['spoplus_pred_delta_a_to_c']} | {row['interpretation']} |"
        )
    lines.extend(["", "## Main Figure Panels", ""])
    for row in case_rows:
        lines.append(
            f"- Panel {row['panel']}: {row['graph_id']} ({row['mechanism_label']}), "
            f"{row['outcome']}; C is 2stage rank {row['c_rank_under_2stage']} and true rank {row['c_true_rank']}."
        )
    lines.extend(["", "## Top Critical Edges", ""])
    graphs_with_edges = []
    for row in paper_rows:
        graph = row["graph_id"]
        if any(edge_row["graph_id"] == graph for edge_row in edge_rows):
            graphs_with_edges.append(graph)
    for graph in graphs_with_edges:
        graph_edges = [row for row in edge_rows if row["graph_id"] == graph][:5]
        if not graph_edges:
            continue
        lines.append(f"### {graph}")
        for row in graph_edges:
            lines.append(
                f"- edge {row['edge_id']} ({row['src']}->{row['dst']}), {row['edge_role']}, "
                f"freq={row['edge_frequency']}, true_delta={row['mean_signed_true_delta']}, "
                f"2stage_pred_delta={row['mean_signed_pred_2stage_delta']}, "
                f"SPO+_pred_delta={row['mean_signed_pred_spoplus_delta']}."
            )
        lines.append("")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def import_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_figure(fig: Any, output_stem: Path) -> list[Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [output_stem.with_suffix(".png"), output_stem.with_suffix(".pdf")]
    fig.savefig(paths[0], dpi=220, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    return paths


def plot_rank_reversal_triplets(paper_rows: list[dict[str, str]], output_stem: Path) -> list[Path]:
    plt = import_matplotlib()
    labels = [row["graph_id"] for row in paper_rows]
    true_values = [parse_float(row["true_delta_a_to_c"]) for row in paper_rows]
    pred2_values = [parse_float(row["two_stage_pred_delta_a_to_c"]) for row in paper_rows]
    predspo_values = [parse_float(row["spoplus_pred_delta_a_to_c"]) for row in paper_rows]
    x = list(range(len(labels)))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12.8, 5.2))
    ax.axhline(0, color="#2d2d2d", linewidth=0.9)
    ax.bar([i - width for i in x], true_values, width, label="True value delta", color="#3d6fb6")
    ax.bar(x, pred2_values, width, label="2stage predicted delta", color="#9a9a9a")
    ax.bar([i + width for i in x], predspo_values, width, label="SPO+ predicted delta", color="#d36b36")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Delta from 2stage rank1 to SPO+ rank1")
    ax.set_title("Decision-critical rank reversal across selected Step2c graph instances")
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    paths = save_figure(fig, output_stem)
    plt.close(fig)
    return paths


def plot_case_panels(case_rows: list[dict[str, str]], output_stem: Path) -> list[Path]:
    plt = import_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6), sharey=False)
    axes_flat = list(axes.ravel())
    colors = ["#3d6fb6", "#9a9a9a", "#d36b36"]
    series = [
        ("true_delta_a_to_c", "True"),
        ("two_stage_pred_delta_a_to_c", "2stage pred"),
        ("spoplus_pred_delta_a_to_c", "SPO+ pred"),
    ]
    for ax, row in zip(axes_flat, case_rows):
        values = [parse_float(row[key]) for key, _ in series]
        ax.axhline(0, color="#2d2d2d", linewidth=0.8)
        ax.bar([label for _, label in series], values, color=colors, width=0.62)
        ax.set_title(f"{row['panel']}. {row['graph_id']}: {row['mechanism_label']}", fontsize=10)
        ax.text(
            0.02,
            0.96,
            f"C = 2stage rank {row['c_rank_under_2stage']}; true rank {row['c_true_rank']}\n{row['outcome']}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#d0d0d0"},
        )
        ax.tick_params(axis="x", labelrotation=20)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes_flat[::2]:
        ax.set_ylabel("Delta from A to C")
    fig.suptitle("Representative rank-reversal panels", y=1.01)
    fig.tight_layout()
    paths = save_figure(fig, output_stem)
    plt.close(fig)
    return paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build presentation tables and figures from Step2c mechanism audit CSVs."
    )
    parser.add_argument("--atlas-input", type=Path, default=DEFAULT_ATLAS_INPUT)
    parser.add_argument("--rank-summary-input", type=Path, default=DEFAULT_RANK_SUMMARY_INPUT)
    parser.add_argument("--edge-summary-input", type=Path, default=DEFAULT_EDGE_SUMMARY_INPUT)
    parser.add_argument("--critical-edge-input", type=Path, default=DEFAULT_CRITICAL_EDGE_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PRESENTATION_DIR)
    parser.add_argument("--top-critical-edges", type=int, default=5)
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args(argv)


def build_artifacts(args: argparse.Namespace) -> dict[str, Path | list[Path]]:
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    atlas_rows = read_csv(args.atlas_input)
    rank_summary_rows = read_csv(args.rank_summary_input)
    edge_summary_rows = read_csv(args.edge_summary_input)
    critical_edge_rows = read_csv(args.critical_edge_input)

    paper_rows = build_paper_mechanism_rows(
        atlas_rows=atlas_rows,
        rank_summary_rows=rank_summary_rows,
        edge_summary_rows=edge_summary_rows,
    )
    edge_rows = build_top_critical_edge_rows(
        critical_edge_rows,
        top_n=args.top_critical_edges,
    )
    case_rows = build_case_panel_rows(paper_rows)
    figure_data_rows = build_figure_data_rows(paper_rows)

    outputs: dict[str, Path | list[Path]] = {}
    outputs["paper_table_csv"] = output_dir / "step2c_paper_mechanism_table.csv"
    outputs["figure_data_csv"] = output_dir / "step2c_rank_reversal_figure_data.csv"
    outputs["case_panel_csv"] = output_dir / "step2c_case_panel_table.csv"
    outputs["top_edges_csv"] = output_dir / "step2c_top_critical_edges_by_case.csv"
    outputs["paper_table_tex"] = output_dir / "step2c_paper_mechanism_table.tex"
    outputs["story_markdown"] = output_dir / "step2c_rank_reversal_story.md"

    write_csv(outputs["paper_table_csv"], paper_rows, PAPER_FIELDS)
    write_csv(outputs["figure_data_csv"], figure_data_rows, FIGURE_DATA_FIELDS)
    write_csv(outputs["case_panel_csv"], case_rows, CASE_PANEL_FIELDS)
    write_csv(outputs["top_edges_csv"], edge_rows, EDGE_FIELDS)
    write_latex_mechanism_table(outputs["paper_table_tex"], paper_rows)
    write_story_markdown(outputs["story_markdown"], paper_rows, case_rows, edge_rows)

    if not args.skip_plots:
        outputs["rank_reversal_triplets"] = plot_rank_reversal_triplets(
            paper_rows,
            figures_dir / "step2c_rank_reversal_triplets",
        )
        outputs["case_panels"] = plot_case_panels(
            case_rows,
            figures_dir / "step2c_rank_reversal_case_panels",
        )
    return outputs


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    outputs = build_artifacts(args)
    for name, path in outputs.items():
        if isinstance(path, list):
            print(f"{name}: {', '.join(str(item) for item in path)}")
        else:
            print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
