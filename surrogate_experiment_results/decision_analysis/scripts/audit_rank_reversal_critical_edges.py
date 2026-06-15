#!/usr/bin/env python3
"""Audit rank reversals and critical edges for selected Step2c mechanisms."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compare_decisions_per_graph import (  # noqa: E402
    ensure_step1c_imports,
    filter_split_entries_by_graphs,
    load_or_make_split_entries,
    resolve_graph_path,
)
from analyze_edge_error_criticality import edge_index_arrays  # noqa: E402


AUDIT_DIR = PROJECT_ROOT / "surrogate_experiment_results" / "Step2c Mechanism Dissection Audit"
DEFAULT_RESULTS_DIR = AUDIT_DIR / "results"
DEFAULT_REGIME = "step2c_poly_d8_mult_eps050"
DEFAULT_DATASET_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "step2c_poly_d8_mult_eps050_main2000_seed20260523"
)
DEFAULT_SPLIT_PATH = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "Step2_resampling"
    / "splits"
    / DEFAULT_REGIME
    / "master_split_seed=42.json"
)
DEFAULT_PREDICTED_TOPM = DEFAULT_RESULTS_DIR / "step2c_selected_graphs_all50_top20_predicted.csv"
DEFAULT_ORACLE_LANDSCAPE = (
    DEFAULT_RESULTS_DIR / "step2c_selected_graphs_true_top50_oracle_landscape.csv"
)
DEFAULT_RANK_OUTPUT = DEFAULT_RESULTS_DIR / "step2c_rank_reversal_table.csv"
DEFAULT_EDGE_OUTPUT = DEFAULT_RESULTS_DIR / "step2c_critical_edge_table.csv"
DEFAULT_RANK_SUMMARY_OUTPUT = DEFAULT_RESULTS_DIR / "step2c_rank_reversal_summary.csv"
DEFAULT_EDGE_SUMMARY_OUTPUT = DEFAULT_RESULTS_DIR / "step2c_critical_edge_summary.csv"
DEFAULT_GRAPHS = (
    "G-392.json",
    "G-1285.json",
    "G-1560.json",
    "G-1169.json",
    "G-1449.json",
    "G-142.json",
    "G-946.json",
    "G-14.json",
    "G-163.json",
)

RANK_REVERSAL_FIELDS = [
    "regime",
    "graph_id",
    "subset_seed",
    "solution_role",
    "found",
    "rank_under_2stage",
    "rank_under_spoplus",
    "rank_under_true_oracle_top50",
    "true_gap_pct",
    "true_obj",
    "predicted_score_under_2stage",
    "predicted_score_under_spoplus",
    "two_stage_margin_vs_rank1",
    "spoplus_margin_vs_rank1",
    "jaccard_to_oracle",
    "solution_edge_signature",
]

CRITICAL_EDGE_FIELDS = [
    "regime",
    "graph_id",
    "subset_seed",
    "comparison",
    "left_role",
    "right_role",
    "edge_id",
    "src",
    "dst",
    "edge_in_left",
    "edge_in_right",
    "edge_in_2stage_rank1",
    "edge_in_best_2stage_near_oracle",
    "edge_in_spoplus_rank1",
    "edge_in_oracle",
    "edge_in_oracle_vs_left_symdiff",
    "edge_in_oracle_vs_right_symdiff",
    "true_weight",
    "pred_2stage",
    "pred_spoplus",
    "error_2stage",
    "error_spoplus",
    "delta_prediction_spoplus_minus_2stage",
    "signed_true_value_delta_right_minus_left",
    "signed_pred_2stage_delta_right_minus_left",
    "signed_pred_spoplus_delta_right_minus_left",
]

RANK_SUMMARY_FIELDS = [
    "graph_id",
    "seed_count",
    "best_near_found_rate",
    "spoplus_equals_best_near_rate",
    "median_spoplus_rank_under_2stage",
    "median_best_near_rank_under_2stage",
    "median_spoplus_true_rank",
    "median_best_near_true_rank",
    "median_two_stage_margin_to_spoplus",
    "median_spoplus_margin_to_two_stage",
    "median_spoplus_true_gap_pct",
    "median_two_stage_true_gap_pct",
]

EDGE_SUMMARY_FIELDS = [
    "graph_id",
    "comparison",
    "seed_count",
    "median_signed_true_value_delta",
    "median_signed_pred_2stage_delta",
    "median_signed_pred_spoplus_delta",
    "mean_signed_true_value_delta",
    "mean_signed_pred_2stage_delta",
    "mean_signed_pred_spoplus_delta",
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


def finite_values(values: list[Any]) -> list[float]:
    clean: list[float] = []
    for value in values:
        parsed = parse_float(value)
        if math.isfinite(parsed):
            clean.append(parsed)
    return clean


def finite_median(values: list[Any]) -> float:
    clean = sorted(finite_values(values))
    if not clean:
        return float("nan")
    mid = len(clean) // 2
    if len(clean) % 2:
        return float(clean[mid])
    return float(0.5 * (clean[mid - 1] + clean[mid]))


def finite_mean(values: list[Any]) -> float:
    clean = finite_values(values)
    if not clean:
        return float("nan")
    return float(sum(clean) / len(clean))


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value) == "True"


def signature_indices(signature: Any) -> set[int]:
    text = str(signature or "")
    if not text:
        return set()
    return {int(part) for part in text.split("|") if part != ""}


def mask_from_signature(signature: Any, num_edges: int) -> np.ndarray:
    mask = np.zeros(int(num_edges), dtype=bool)
    for edge_id in signature_indices(signature):
        if 0 <= edge_id < len(mask):
            mask[edge_id] = True
    return mask


def signature_from_mask(mask: np.ndarray) -> str:
    return "|".join(str(int(idx)) for idx in np.flatnonzero(np.asarray(mask, dtype=bool)))


def mask_jaccard(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    union = left | right
    if not np.any(union):
        return 1.0
    return float(np.sum(left & right) / np.sum(union))


def normalized_gap_pct(oracle_obj: float, true_obj: float) -> float:
    denom = abs(float(oracle_obj)) + 1e-9
    return float(100.0 * (float(oracle_obj) - float(true_obj)) / denom)


def group_predicted_rows(
    rows: list[dict[str, str]],
    graphs: set[str],
) -> dict[tuple[str, int], dict[str, dict[int, dict[str, str]]]]:
    grouped: dict[tuple[str, int], dict[str, dict[int, dict[str, str]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in rows:
        graph_id = row["graph_id"]
        if graph_id not in graphs:
            continue
        seed = parse_int(row["subset_seed"])
        method = row["method_label"]
        rank = parse_int(row["solution_rank"])
        grouped[(graph_id, seed)][method][rank] = row
    return grouped


def oracle_rank_by_graph(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    ranks: dict[str, dict[str, int]] = defaultdict(dict)
    for row in rows:
        graph_id = row["graph_id"]
        signature = row.get("solution_edge_signature", "")
        if signature and signature not in ranks[graph_id]:
            ranks[graph_id][signature] = parse_int(row["solution_rank"])
    return ranks


def rank_for_signature(rows_by_rank: dict[int, dict[str, str]], signature: str) -> int | str:
    for rank, row in sorted(rows_by_rank.items()):
        if row.get("solution_edge_signature", "") == signature:
            return int(rank)
    return ""


def find_best_near_oracle_row(
    two_stage_rows: dict[int, dict[str, str]],
    near_oracle_gap: float,
    top_m: int,
) -> dict[str, str] | None:
    candidates = [
        row
        for rank, row in two_stage_rows.items()
        if rank <= top_m
        and parse_float(row.get("normalized_gap_to_oracle")) <= near_oracle_gap + 1e-12
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: (
            parse_float(row.get("normalized_gap_to_oracle")),
            parse_int(row.get("solution_rank")),
        ),
    )


def role_row_missing(
    *,
    graph_id: str,
    subset_seed: int,
    role: str,
) -> dict[str, Any]:
    return {
        "regime": DEFAULT_REGIME,
        "graph_id": graph_id,
        "subset_seed": int(subset_seed),
        "solution_role": role,
        "found": False,
        "rank_under_2stage": "",
        "rank_under_spoplus": "",
        "rank_under_true_oracle_top50": "",
        "true_gap_pct": "",
        "true_obj": "",
        "predicted_score_under_2stage": "",
        "predicted_score_under_spoplus": "",
        "two_stage_margin_vs_rank1": "",
        "spoplus_margin_vs_rank1": "",
        "jaccard_to_oracle": "",
        "solution_edge_signature": "",
    }


def build_rank_reversal_rows_for_seed(
    *,
    graph_id: str,
    subset_seed: int,
    two_stage_rows: dict[int, dict[str, str]],
    spoplus_rows: dict[int, dict[str, str]],
    true_rank_by_signature: dict[str, int],
    w_true: np.ndarray,
    w_hat_2stage: np.ndarray,
    w_hat_spoplus: np.ndarray,
    y_oracle: np.ndarray,
    num_edges: int,
    near_oracle_gap: float,
    regime: str = DEFAULT_REGIME,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    roles: dict[str, dict[str, str]] = {
        "two_stage_rank1": two_stage_rows.get(1, {}),
        "best_2stage_near_oracle_top20": find_best_near_oracle_row(
            two_stage_rows,
            near_oracle_gap=near_oracle_gap,
            top_m=20,
        )
        or {},
        "spoplus_rank1": spoplus_rows.get(1, {}),
    }

    if not roles["two_stage_rank1"] or not roles["spoplus_rank1"]:
        return [], roles

    y_two_rank1 = mask_from_signature(
        roles["two_stage_rank1"]["solution_edge_signature"], num_edges
    )
    y_spo_rank1 = mask_from_signature(
        roles["spoplus_rank1"]["solution_edge_signature"], num_edges
    )
    two_stage_rank1_score = float(np.dot(w_hat_2stage, y_two_rank1.astype(float)))
    spoplus_rank1_score = float(np.dot(w_hat_spoplus, y_spo_rank1.astype(float)))
    oracle_obj = float(np.dot(w_true, np.asarray(y_oracle, dtype=float)))

    output_rows: list[dict[str, Any]] = []
    for role in [
        "two_stage_rank1",
        "best_2stage_near_oracle_top20",
        "spoplus_rank1",
    ]:
        row = roles.get(role, {})
        if not row:
            output_rows.append(
                role_row_missing(graph_id=graph_id, subset_seed=subset_seed, role=role)
            )
            output_rows[-1]["regime"] = regime
            continue

        signature = row["solution_edge_signature"]
        y = mask_from_signature(signature, num_edges)
        true_obj = float(np.dot(w_true, y.astype(float)))
        score_2stage = float(np.dot(w_hat_2stage, y.astype(float)))
        score_spoplus = float(np.dot(w_hat_spoplus, y.astype(float)))

        output_rows.append(
            {
                "regime": regime,
                "graph_id": graph_id,
                "subset_seed": int(subset_seed),
                "solution_role": role,
                "found": True,
                "rank_under_2stage": rank_for_signature(two_stage_rows, signature),
                "rank_under_spoplus": rank_for_signature(spoplus_rows, signature),
                "rank_under_true_oracle_top50": true_rank_by_signature.get(signature, ""),
                "true_gap_pct": normalized_gap_pct(oracle_obj, true_obj),
                "true_obj": true_obj,
                "predicted_score_under_2stage": score_2stage,
                "predicted_score_under_spoplus": score_spoplus,
                "two_stage_margin_vs_rank1": two_stage_rank1_score - score_2stage,
                "spoplus_margin_vs_rank1": spoplus_rank1_score - score_spoplus,
                "jaccard_to_oracle": mask_jaccard(y, y_oracle),
                "solution_edge_signature": signature,
            }
        )

    return output_rows, roles


def build_critical_edge_rows_for_comparison(
    *,
    graph_id: str,
    subset_seed: int,
    comparison: str,
    left_role: str,
    right_role: str,
    roles: dict[str, dict[str, str]],
    w_true: np.ndarray,
    w_hat_2stage: np.ndarray,
    w_hat_spoplus: np.ndarray,
    y_oracle: np.ndarray,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    regime: str = DEFAULT_REGIME,
) -> list[dict[str, Any]]:
    if not roles.get(left_role) or not roles.get(right_role):
        return []

    num_edges = len(w_true)
    left = mask_from_signature(roles[left_role]["solution_edge_signature"], num_edges)
    right = mask_from_signature(roles[right_role]["solution_edge_signature"], num_edges)
    two_stage = mask_from_signature(
        roles.get("two_stage_rank1", {}).get("solution_edge_signature", ""), num_edges
    )
    best_near = mask_from_signature(
        roles.get("best_2stage_near_oracle_top20", {}).get("solution_edge_signature", ""),
        num_edges,
    )
    spoplus = mask_from_signature(
        roles.get("spoplus_rank1", {}).get("solution_edge_signature", ""), num_edges
    )
    oracle = np.asarray(y_oracle, dtype=bool)

    symdiff_edges = np.flatnonzero(left ^ right)
    output_rows: list[dict[str, Any]] = []
    for edge_id in symdiff_edges:
        left_value = 1.0 if left[edge_id] else 0.0
        right_value = 1.0 if right[edge_id] else 0.0
        delta = right_value - left_value
        output_rows.append(
            {
                "regime": regime,
                "graph_id": graph_id,
                "subset_seed": int(subset_seed),
                "comparison": comparison,
                "left_role": left_role,
                "right_role": right_role,
                "edge_id": int(edge_id),
                "src": int(edge_src[edge_id]),
                "dst": int(edge_dst[edge_id]),
                "edge_in_left": bool(left[edge_id]),
                "edge_in_right": bool(right[edge_id]),
                "edge_in_2stage_rank1": bool(two_stage[edge_id]),
                "edge_in_best_2stage_near_oracle": bool(best_near[edge_id]),
                "edge_in_spoplus_rank1": bool(spoplus[edge_id]),
                "edge_in_oracle": bool(oracle[edge_id]),
                "edge_in_oracle_vs_left_symdiff": bool(oracle[edge_id] != left[edge_id]),
                "edge_in_oracle_vs_right_symdiff": bool(oracle[edge_id] != right[edge_id]),
                "true_weight": float(w_true[edge_id]),
                "pred_2stage": float(w_hat_2stage[edge_id]),
                "pred_spoplus": float(w_hat_spoplus[edge_id]),
                "error_2stage": float(w_hat_2stage[edge_id] - w_true[edge_id]),
                "error_spoplus": float(w_hat_spoplus[edge_id] - w_true[edge_id]),
                "delta_prediction_spoplus_minus_2stage": float(
                    w_hat_spoplus[edge_id] - w_hat_2stage[edge_id]
                ),
                "signed_true_value_delta_right_minus_left": float(delta * w_true[edge_id]),
                "signed_pred_2stage_delta_right_minus_left": float(
                    delta * w_hat_2stage[edge_id]
                ),
                "signed_pred_spoplus_delta_right_minus_left": float(
                    delta * w_hat_spoplus[edge_id]
                ),
            }
        )
    return output_rows


def theta_for_method(rows_by_rank: dict[int, dict[str, str]]) -> np.ndarray:
    row = rows_by_rank.get(1) or next(iter(rows_by_rank.values()))
    return np.asarray([parse_float(row["theta_1"]), parse_float(row["theta_2"])], dtype=float)


def build_record_lookup(args, graphs: set[str]) -> dict[str, dict[str, Any]]:
    common, _, _, _, _ = ensure_step1c_imports()

    import gurobipy as gp

    test_entries = load_or_make_split_entries(
        split_path=args.split_path,
        dataset_dir=args.dataset_dir,
        split_seed=args.split_seed,
        train_pool_size=args.train_pool_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
    )
    test_entries = filter_split_entries_by_graphs(test_entries, graphs)
    graph_paths = [resolve_graph_path(entry, args.dataset_dir) for entry in test_entries]

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Seed", args.gurobi_seed)
    env.start()

    records: list[dict[str, Any]] = []
    try:
        records = common.load_graph_records(
            graph_paths,
            env,
            max_cycle=args.max_cycle,
            max_chain=args.max_chain,
        )
        # Keep records alive for the current process; caller disposes through process exit.
        return {record["filename"]: record for record in records}
    except Exception:
        common.dispose_graph_records(records)
        env.dispose()
        raise


def audit_rank_reversal_and_edges(args) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    graphs = set(args.graphs)
    predicted_rows = read_csv_rows(args.predicted_topm_input)
    oracle_rows = read_csv_rows(args.oracle_landscape_input)
    grouped = group_predicted_rows(predicted_rows, graphs)
    true_ranks = oracle_rank_by_graph(oracle_rows)
    records = build_record_lookup(args, graphs)

    rank_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []

    for key in sorted(grouped):
        graph_id, subset_seed = key
        methods = grouped[key]
        two_stage_rows = methods.get("2stage_val_mse", {})
        spoplus_rows = methods.get("spoplus_val_spoplus_loss", {})
        if not two_stage_rows or not spoplus_rows:
            continue
        record = records[graph_id]
        x = np.asarray(record["X"], dtype=float)
        w_true = np.asarray(record["w_true"], dtype=float)
        y_oracle = np.asarray(record["y_optimal"], dtype=float) > 0.5
        w_hat_2stage = x @ theta_for_method(two_stage_rows)
        w_hat_spoplus = x @ theta_for_method(spoplus_rows)
        edge_src, edge_dst = edge_index_arrays(record["graph"])

        rows, roles = build_rank_reversal_rows_for_seed(
            graph_id=graph_id,
            subset_seed=subset_seed,
            two_stage_rows=two_stage_rows,
            spoplus_rows=spoplus_rows,
            true_rank_by_signature=true_ranks.get(graph_id, {}),
            w_true=w_true,
            w_hat_2stage=w_hat_2stage,
            w_hat_spoplus=w_hat_spoplus,
            y_oracle=y_oracle,
            num_edges=len(w_true),
            near_oracle_gap=args.near_oracle_gap,
            regime=args.regime,
        )
        rank_rows.extend(rows)

        edge_rows.extend(
            build_critical_edge_rows_for_comparison(
                graph_id=graph_id,
                subset_seed=subset_seed,
                comparison="two_stage_rank1_vs_spoplus_rank1",
                left_role="two_stage_rank1",
                right_role="spoplus_rank1",
                roles=roles,
                w_true=w_true,
                w_hat_2stage=w_hat_2stage,
                w_hat_spoplus=w_hat_spoplus,
                y_oracle=y_oracle,
                edge_src=edge_src,
                edge_dst=edge_dst,
                regime=args.regime,
            )
        )
        edge_rows.extend(
            build_critical_edge_rows_for_comparison(
                graph_id=graph_id,
                subset_seed=subset_seed,
                comparison="two_stage_rank1_vs_best_2stage_near_oracle",
                left_role="two_stage_rank1",
                right_role="best_2stage_near_oracle_top20",
                roles=roles,
                w_true=w_true,
                w_hat_2stage=w_hat_2stage,
                w_hat_spoplus=w_hat_spoplus,
                y_oracle=y_oracle,
                edge_src=edge_src,
                edge_dst=edge_dst,
                regime=args.regime,
            )
        )

    return rank_rows, edge_rows


def summarize_rank_reversal_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[int, dict[str, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        grouped[str(row["graph_id"])][parse_int(row["subset_seed"])][
            str(row["solution_role"])
        ] = row

    output_rows: list[dict[str, Any]] = []
    for graph_id in sorted(grouped):
        seed_groups = grouped[graph_id]
        seeds = sorted(seed_groups)
        best_rows = [
            seed_groups[seed].get("best_2stage_near_oracle_top20", {}) for seed in seeds
        ]
        spo_rows = [seed_groups[seed].get("spoplus_rank1", {}) for seed in seeds]
        two_rows = [seed_groups[seed].get("two_stage_rank1", {}) for seed in seeds]

        best_found = [bool_value(row.get("found")) for row in best_rows]
        spo_equals_best: list[bool] = []
        for best, spo in zip(best_rows, spo_rows):
            spo_equals_best.append(
                bool_value(best.get("found"))
                and best.get("solution_edge_signature", "") != ""
                and best.get("solution_edge_signature", "") == spo.get("solution_edge_signature", "")
            )

        output_rows.append(
            {
                "graph_id": graph_id,
                "seed_count": len(seeds),
                "best_near_found_rate": (
                    sum(best_found) / len(best_found) if best_found else float("nan")
                ),
                "spoplus_equals_best_near_rate": (
                    sum(spo_equals_best) / len(spo_equals_best)
                    if spo_equals_best
                    else float("nan")
                ),
                "median_spoplus_rank_under_2stage": finite_median(
                    [row.get("rank_under_2stage", "") for row in spo_rows]
                ),
                "median_best_near_rank_under_2stage": finite_median(
                    [row.get("rank_under_2stage", "") for row in best_rows]
                ),
                "median_spoplus_true_rank": finite_median(
                    [row.get("rank_under_true_oracle_top50", "") for row in spo_rows]
                ),
                "median_best_near_true_rank": finite_median(
                    [row.get("rank_under_true_oracle_top50", "") for row in best_rows]
                ),
                "median_two_stage_margin_to_spoplus": finite_median(
                    [row.get("two_stage_margin_vs_rank1", "") for row in spo_rows]
                ),
                "median_spoplus_margin_to_two_stage": finite_median(
                    [row.get("spoplus_margin_vs_rank1", "") for row in two_rows]
                ),
                "median_spoplus_true_gap_pct": finite_median(
                    [row.get("true_gap_pct", "") for row in spo_rows]
                ),
                "median_two_stage_true_gap_pct": finite_median(
                    [row.get("true_gap_pct", "") for row in two_rows]
                ),
            }
        )
    return output_rows


def summarize_critical_edge_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    per_seed: dict[tuple[str, int, str], dict[str, float]] = defaultdict(
        lambda: {"true": 0.0, "pred2": 0.0, "predspo": 0.0}
    )
    for row in rows:
        key = (str(row["graph_id"]), parse_int(row["subset_seed"]), str(row["comparison"]))
        per_seed[key]["true"] += parse_float(row["signed_true_value_delta_right_minus_left"], 0.0)
        per_seed[key]["pred2"] += parse_float(row["signed_pred_2stage_delta_right_minus_left"], 0.0)
        per_seed[key]["predspo"] += parse_float(row["signed_pred_spoplus_delta_right_minus_left"], 0.0)

    grouped: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    for graph_id, seed, comparison in sorted(per_seed):
        grouped[(graph_id, comparison)].append(per_seed[(graph_id, seed, comparison)])

    output_rows: list[dict[str, Any]] = []
    for graph_id, comparison in sorted(grouped):
        group = grouped[(graph_id, comparison)]
        true_values = [row["true"] for row in group]
        pred2_values = [row["pred2"] for row in group]
        predspo_values = [row["predspo"] for row in group]
        output_rows.append(
            {
                "graph_id": graph_id,
                "comparison": comparison,
                "seed_count": len(group),
                "median_signed_true_value_delta": finite_median(true_values),
                "median_signed_pred_2stage_delta": finite_median(pred2_values),
                "median_signed_pred_spoplus_delta": finite_median(predspo_values),
                "mean_signed_true_value_delta": finite_mean(true_values),
                "mean_signed_pred_2stage_delta": finite_mean(pred2_values),
                "mean_signed_pred_spoplus_delta": finite_mean(predspo_values),
            }
        )
    return output_rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build rank-reversal and critical-edge tables for Step2c selected graphs."
    )
    parser.add_argument("--regime", default=DEFAULT_REGIME)
    parser.add_argument("--predicted-topm-input", type=Path, default=DEFAULT_PREDICTED_TOPM)
    parser.add_argument("--oracle-landscape-input", type=Path, default=DEFAULT_ORACLE_LANDSCAPE)
    parser.add_argument("--rank-output", type=Path, default=DEFAULT_RANK_OUTPUT)
    parser.add_argument("--edge-output", type=Path, default=DEFAULT_EDGE_OUTPUT)
    parser.add_argument("--rank-summary-output", type=Path, default=DEFAULT_RANK_SUMMARY_OUTPUT)
    parser.add_argument("--edge-summary-output", type=Path, default=DEFAULT_EDGE_SUMMARY_OUTPUT)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--graphs", nargs="+", default=list(DEFAULT_GRAPHS))
    parser.add_argument("--near-oracle-gap", type=float, default=0.05)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--train-pool-size", type=int, default=1200)
    parser.add_argument("--validation-size", type=int, default=400)
    parser.add_argument("--test-size", type=int, default=400)
    parser.add_argument("--gurobi-seed", type=int, default=42)
    parser.add_argument("--max-cycle", type=int, default=3)
    parser.add_argument("--max-chain", type=int, default=4)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    rank_rows, edge_rows = audit_rank_reversal_and_edges(args)
    write_csv(args.rank_output, rank_rows, RANK_REVERSAL_FIELDS)
    write_csv(args.edge_output, edge_rows, CRITICAL_EDGE_FIELDS)
    rank_summary_rows = summarize_rank_reversal_rows(rank_rows)
    edge_summary_rows = summarize_critical_edge_rows(edge_rows)
    write_csv(args.rank_summary_output, rank_summary_rows, RANK_SUMMARY_FIELDS)
    write_csv(args.edge_summary_output, edge_summary_rows, EDGE_SUMMARY_FIELDS)
    print(f"Saved {len(rank_rows)} rank-reversal rows to {args.rank_output}")
    print(f"Saved {len(edge_rows)} critical-edge rows to {args.edge_output}")
    print(f"Saved {len(rank_summary_rows)} rank-reversal summary rows to {args.rank_summary_output}")
    print(f"Saved {len(edge_summary_rows)} critical-edge summary rows to {args.edge_summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
