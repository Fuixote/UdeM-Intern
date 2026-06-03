#!/usr/bin/env python3
"""Extract interpretable graph-level case-study edge tables."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = (
    PROJECT_ROOT / "surrogate_experiment_results" / "decision_analysis" / "results"
)
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR / "case_studies"

TWO_STAGE_LABEL = "2stage_val_mse"
SPOPLUS_LABEL = "spoplus_val_spoplus_loss"

CASE_A = "case_a_bad_prediction_irrelevant"
CASE_B = "case_b_different_solution_near_optimal"
CASE_C = "case_c_spoplus_fixes_2stage"
CASE_LABELS = (CASE_A, CASE_B, CASE_C)

CASE_EDGE_FIELDS = [
    "edge_id",
    "src_dst",
    "w_true",
    "w_hat_2stage",
    "w_hat_spoplus",
    "abs_err_2stage",
    "abs_err_spoplus",
    "in_opt",
    "in_2stage",
    "in_spoplus",
    "in_2stage_symdiff",
    "in_spoplus_symdiff",
    "utility",
    "recipient_cPRA",
]

INDEX_FIELDS = [
    "case_id",
    "case_label",
    "regime",
    "case_type",
    "subset_seed",
    "graph_id",
    "case_table_file",
    "selection_reason",
    "selection_score",
    "2stage_normalized_gap",
    "spoplus_normalized_gap",
    "normalized_gap_reduction",
    "2stage_mse_all_edges",
    "spoplus_mse_all_edges",
    "2stage_edge_jaccard_with_opt",
    "spoplus_edge_jaccard_with_opt",
    "2stage_same_solution_as_opt",
    "spoplus_same_solution_as_opt",
    "2stage_top10_error_edges_in_opt_rate",
    "2stage_top10_error_edges_in_pred_rate",
    "2stage_top10_error_edges_in_symdiff_rate",
    "spoplus_top10_error_edges_in_opt_rate",
    "spoplus_top10_error_edges_in_pred_rate",
    "spoplus_top10_error_edges_in_symdiff_rate",
]


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: list[dict[str, object]], fieldnames) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def parse_float(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    if value == "":
        return float("nan")
    return float(value)


def graph_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row["regime"], str(row["subset_seed"]), row["graph_id"])


def method_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (*graph_key(row), row["method_label"])


def quantile(values: list[float], q: float) -> float:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return float("nan")
    if len(clean) == 1:
        return clean[0]
    position = (len(clean) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return clean[int(position)]
    fraction = position - lower
    return clean[lower] * (1.0 - fraction) + clean[upper] * fraction


def safe_slug(value: str) -> str:
    stem = Path(value).stem
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", stem)


def top10_any_decision_rate(summary_row: dict[str, str]) -> float:
    return max(
        parse_float(summary_row, "top10_error_edges_in_opt_rate"),
        parse_float(summary_row, "top10_error_edges_in_pred_rate"),
        parse_float(summary_row, "top10_error_edges_in_symdiff_rate"),
    )


def build_graph_records(
    per_graph_rows: list[dict[str, str]],
    edge_summary_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    per_graph_by_key = {method_key(row): row for row in per_graph_rows}
    summary_by_key = {method_key(row): row for row in edge_summary_rows}
    graph_keys = sorted(
        {
            graph_key(row)
            for row in per_graph_rows
            if row.get("method_label") in {TWO_STAGE_LABEL, SPOPLUS_LABEL}
        }
    )
    records: list[dict[str, object]] = []
    for key in graph_keys:
        two_graph = per_graph_by_key.get((*key, TWO_STAGE_LABEL))
        spo_graph = per_graph_by_key.get((*key, SPOPLUS_LABEL))
        two_summary = summary_by_key.get((*key, TWO_STAGE_LABEL))
        spo_summary = summary_by_key.get((*key, SPOPLUS_LABEL))
        if not all([two_graph, spo_graph, two_summary, spo_summary]):
            continue
        records.append(
            {
                "key": key,
                "regime": key[0],
                "subset_seed": key[1],
                "graph_id": key[2],
                "case_type": two_graph.get("case_type", ""),
                "two_graph": two_graph,
                "spo_graph": spo_graph,
                "two_summary": two_summary,
                "spo_summary": spo_summary,
            }
        )
    return records


def two_gap(record: dict[str, object]) -> float:
    return parse_float(record["two_graph"], "normalized_gap")


def spo_gap(record: dict[str, object]) -> float:
    return parse_float(record["spo_graph"], "normalized_gap")


def gap_reduction(record: dict[str, object]) -> float:
    return two_gap(record) - spo_gap(record)


def two_mse(record: dict[str, object]) -> float:
    return parse_float(record["two_summary"], "mse_all_edges")


def spo_mse(record: dict[str, object]) -> float:
    return parse_float(record["spo_summary"], "mse_all_edges")


def two_jaccard(record: dict[str, object]) -> float:
    return parse_float(record["two_graph"], "edge_jaccard_with_opt")


def spo_jaccard(record: dict[str, object]) -> float:
    return parse_float(record["spo_graph"], "edge_jaccard_with_opt")


def two_symdiff_rate(record: dict[str, object]) -> float:
    return parse_float(record["two_summary"], "top10_error_edges_in_symdiff_rate")


def spo_symdiff_rate(record: dict[str, object]) -> float:
    return parse_float(record["spo_summary"], "top10_error_edges_in_symdiff_rate")


def case_a_candidates(records: list[dict[str, object]]) -> list[dict[str, object]]:
    gap_cutoff = quantile([two_gap(record) for record in records], 0.25)
    mse_cutoff = quantile([two_mse(record) for record in records], 0.75)
    primary = [
        record
        for record in records
        if two_gap(record) <= gap_cutoff and two_mse(record) >= mse_cutoff
    ]
    sort_key = lambda record: (
            top10_any_decision_rate(record["two_summary"]),
            two_gap(record),
            -two_mse(record),
            int(record["subset_seed"]),
            record["graph_id"],
    )
    ranked_primary = sorted(primary, key=sort_key)
    primary_keys = {record["key"] for record in ranked_primary}
    ranked_fallback = sorted(
        [record for record in records if record["key"] not in primary_keys],
        key=sort_key,
    )
    ranked = ranked_primary + ranked_fallback
    return [
        {
            "record": record,
            "selection_score": (
                two_mse(record)
                / max(two_gap(record), 1.0e-12)
                / max(top10_any_decision_rate(record["two_summary"]), 0.05)
            ),
            "selection_reason": (
                "high 2stage all-edge MSE, low 2stage normalized gap, "
                "and low top-error decision membership"
            ),
        }
        for record in ranked
    ]


def case_b_candidates(records: list[dict[str, object]]) -> list[dict[str, object]]:
    gap_cutoff = quantile([two_gap(record) for record in records], 0.50)
    primary = [
        record
        for record in records
        if not parse_bool(record["two_graph"].get("same_solution_as_opt", "False"))
        and 0.35 <= two_jaccard(record) <= 0.75
        and two_gap(record) <= gap_cutoff
    ]
    fallback = [
        record
        for record in records
        if not parse_bool(record["two_graph"].get("same_solution_as_opt", "False"))
    ]
    sort_key = lambda record: (
            two_gap(record),
            abs(two_jaccard(record) - 0.55),
            int(record["subset_seed"]),
            record["graph_id"],
    )
    ranked_primary = sorted(primary, key=sort_key)
    primary_keys = {record["key"] for record in ranked_primary}
    ranked_fallback = sorted(
        [record for record in fallback if record["key"] not in primary_keys],
        key=sort_key,
    )
    ranked = ranked_primary + ranked_fallback
    return [
        {
            "record": record,
            "selection_score": 1.0 / max(two_gap(record), 1.0e-12),
            "selection_reason": (
                "2stage chooses a different medium-overlap solution but has low "
                "normalized gap"
            ),
        }
        for record in ranked
    ]


def case_c_candidates(records: list[dict[str, object]]) -> list[dict[str, object]]:
    primary = [
        record
        for record in records
        if gap_reduction(record) > 0.0
        and two_symdiff_rate(record) >= spo_symdiff_rate(record)
    ]
    fallback = [record for record in records if gap_reduction(record) > 0.0]
    sort_key = lambda record: (
            -gap_reduction(record),
            -(two_symdiff_rate(record) - spo_symdiff_rate(record)),
            spo_gap(record),
            -two_gap(record),
            int(record["subset_seed"]),
            record["graph_id"],
    )
    ranked_primary = sorted(primary, key=sort_key)
    primary_keys = {record["key"] for record in ranked_primary}
    ranked_fallback = sorted(
        [record for record in fallback if record["key"] not in primary_keys],
        key=sort_key,
    )
    ranked = ranked_primary + ranked_fallback
    return [
        {
            "record": record,
            "selection_score": gap_reduction(record)
            * (1.0 + two_symdiff_rate(record) - spo_symdiff_rate(record)),
            "selection_reason": (
                "SPO+ sharply lowers normalized gap while reducing high-error "
                "symdiff involvement"
            ),
        }
        for record in ranked
    ]


def make_case_index_row(
    case_label: str,
    rank: int,
    candidate: dict[str, object],
) -> dict[str, object]:
    record = candidate["record"]
    two_graph = record["two_graph"]
    spo_graph = record["spo_graph"]
    two_summary = record["two_summary"]
    spo_summary = record["spo_summary"]
    case_id = f"{case_label}_{rank:03d}"
    table_file = (
        f"{case_id}_seed={record['subset_seed']}_graph={safe_slug(record['graph_id'])}.csv"
    )
    return {
        "case_id": case_id,
        "case_label": case_label,
        "regime": record["regime"],
        "case_type": record["case_type"],
        "subset_seed": record["subset_seed"],
        "graph_id": record["graph_id"],
        "case_table_file": table_file,
        "selection_reason": candidate["selection_reason"],
        "selection_score": candidate["selection_score"],
        "2stage_normalized_gap": two_gap(record),
        "spoplus_normalized_gap": spo_gap(record),
        "normalized_gap_reduction": gap_reduction(record),
        "2stage_mse_all_edges": two_mse(record),
        "spoplus_mse_all_edges": spo_mse(record),
        "2stage_edge_jaccard_with_opt": two_jaccard(record),
        "spoplus_edge_jaccard_with_opt": spo_jaccard(record),
        "2stage_same_solution_as_opt": two_graph.get("same_solution_as_opt", ""),
        "spoplus_same_solution_as_opt": spo_graph.get("same_solution_as_opt", ""),
        "2stage_top10_error_edges_in_opt_rate": two_summary[
            "top10_error_edges_in_opt_rate"
        ],
        "2stage_top10_error_edges_in_pred_rate": two_summary[
            "top10_error_edges_in_pred_rate"
        ],
        "2stage_top10_error_edges_in_symdiff_rate": two_summary[
            "top10_error_edges_in_symdiff_rate"
        ],
        "spoplus_top10_error_edges_in_opt_rate": spo_summary[
            "top10_error_edges_in_opt_rate"
        ],
        "spoplus_top10_error_edges_in_pred_rate": spo_summary[
            "top10_error_edges_in_pred_rate"
        ],
        "spoplus_top10_error_edges_in_symdiff_rate": spo_summary[
            "top10_error_edges_in_symdiff_rate"
        ],
    }


def select_case_studies(
    per_graph_rows: list[dict[str, str]],
    edge_summary_rows: list[dict[str, str]],
    per_case_count: int = 3,
) -> list[dict[str, object]]:
    records = build_graph_records(per_graph_rows, edge_summary_rows)
    if not records:
        raise ValueError("No paired 2stage/SPO+ graph records found")

    selected: list[dict[str, object]] = []
    used_keys: set[tuple[str, str, str]] = set()
    chooser_by_label = [
        (CASE_A, case_a_candidates),
        (CASE_B, case_b_candidates),
        (CASE_C, case_c_candidates),
    ]
    for case_label, chooser in chooser_by_label:
        available = [record for record in records if record["key"] not in used_keys]
        candidates = chooser(available)
        selected_for_label: list[dict[str, object]] = []
        seen_graph_ids: set[str] = set()
        for candidate in candidates:
            graph_id = str(candidate["record"]["graph_id"])
            if graph_id in seen_graph_ids:
                continue
            selected_for_label.append(candidate)
            seen_graph_ids.add(graph_id)
            if len(selected_for_label) == per_case_count:
                break
        if len(selected_for_label) < per_case_count:
            raise ValueError(
                f"Requested {per_case_count} distinct-graph {case_label} rows but "
                f"found {len(selected_for_label)} candidates"
            )
        for rank, candidate in enumerate(selected_for_label, start=1):
            selected.append(make_case_index_row(case_label, rank, candidate))
            used_keys.add(candidate["record"]["key"])
    return selected


def edge_group_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row["regime"], str(row["subset_seed"]), row["graph_id"])


def edge_int(row: dict[str, str], field: str) -> int:
    return int(float(row[field]))


def sort_edge_rows(case_label: str, edge_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if case_label == CASE_A:
        return sorted(edge_rows, key=lambda row: (edge_int(row, "rank_err_2stage"), edge_int(row, "edge_id")))
    if case_label == CASE_B:
        return sorted(
            edge_rows,
            key=lambda row: (
                not (
                    parse_bool(row["in_opt"])
                    or parse_bool(row["in_2stage"])
                    or parse_bool(row["in_spoplus"])
                    or parse_bool(row["in_2stage_symdiff"])
                    or parse_bool(row["in_spoplus_symdiff"])
                ),
                -parse_float(row, "w_true"),
                edge_int(row, "edge_id"),
            ),
        )
    if case_label == CASE_C:
        return sorted(
            edge_rows,
            key=lambda row: (
                not parse_bool(row["in_2stage_symdiff"]),
                edge_int(row, "rank_err_2stage"),
                parse_bool(row["in_spoplus_symdiff"]),
                edge_int(row, "edge_id"),
            ),
        )
    return sorted(edge_rows, key=lambda row: edge_int(row, "edge_id"))


def format_edge_row(row: dict[str, str]) -> dict[str, object]:
    return {
        "edge_id": row["edge_id"],
        "src_dst": f"{row['src']} -> {row['dst']}",
        "w_true": row["w_true"],
        "w_hat_2stage": row["w_hat_2stage"],
        "w_hat_spoplus": row["w_hat_spoplus"],
        "abs_err_2stage": row["abs_err_2stage"],
        "abs_err_spoplus": row["abs_err_spoplus"],
        "in_opt": row["in_opt"],
        "in_2stage": row["in_2stage"],
        "in_spoplus": row["in_spoplus"],
        "in_2stage_symdiff": row["in_2stage_symdiff"],
        "in_spoplus_symdiff": row["in_spoplus_symdiff"],
        "utility": row["utility"],
        "recipient_cPRA": row["recipient_cPRA"],
    }


def write_case_study_outputs(
    selected_cases: list[dict[str, object]],
    edge_rows: list[dict[str, str]],
    output_dir: str | Path,
) -> list[dict[str, object]]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    edges_by_key: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in edge_rows:
        edges_by_key.setdefault(edge_group_key(row), []).append(row)

    index_rows: list[dict[str, object]] = []
    for case in selected_cases:
        key = (case["regime"], str(case["subset_seed"]), case["graph_id"])
        case_edges = edges_by_key.get(key)
        if not case_edges:
            raise ValueError(f"No edge rows found for case key={key}")
        table_file = case.get("case_table_file") or (
            f"{case['case_id']}_seed={case['subset_seed']}_graph={safe_slug(case['graph_id'])}.csv"
        )
        table_rows = [
            format_edge_row(row)
            for row in sort_edge_rows(str(case["case_label"]), case_edges)
        ]
        write_csv(output_path / table_file, table_rows, CASE_EDGE_FIELDS)
        index_row = dict(case)
        index_row["case_table_file"] = table_file
        index_rows.append(index_row)

    write_csv(output_path / "case_study_index.csv", index_rows, INDEX_FIELDS)
    return index_rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Create A/B/C case-study edge tables from decision-analysis CSVs."
    )
    parser.add_argument(
        "--per-graph",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "per_graph_decision_comparison.csv",
    )
    parser.add_argument(
        "--edge-summary",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "graph_level_edge_criticality_summary.csv",
    )
    parser.add_argument(
        "--edge-rows",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "edge_error_criticality.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--per-case-count", type=int, default=3)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    per_graph_rows = read_csv_rows(args.per_graph)
    edge_summary_rows = read_csv_rows(args.edge_summary)
    selected = select_case_studies(
        per_graph_rows,
        edge_summary_rows,
        per_case_count=args.per_case_count,
    )
    edge_rows = read_csv_rows(args.edge_rows)
    index_rows = write_case_study_outputs(selected, edge_rows, args.output_dir)
    print(f"Saved {len(index_rows)} case-study tables to {args.output_dir}")
    print(f"Saved case-study index to {args.output_dir / 'case_study_index.csv'}")
    for row in index_rows:
        print(
            f"{row['case_id']}: seed={row['subset_seed']} graph={row['graph_id']} "
            f"2stage_gap={float(row['2stage_normalized_gap']):.6g} "
            f"spoplus_gap={float(row['spoplus_normalized_gap']):.6g} "
            f"table={row['case_table_file']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
