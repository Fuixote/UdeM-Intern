#!/usr/bin/env python3
"""Compute topology-first decision overlays for Step4.

The overlay maps oracle / 2stage / SPO+ selected edge sets back to structural
candidate IDs from Step4 Topology Structural Atlas, then aggregates candidate
selection frequencies by topology and sample size.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
OVERLAY_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = OVERLAY_DIR.parents[1]
STRUCTURAL_SCRIPTS = PROJECT_ROOT / "surrogate_experiment_results" / "Step4 Topology Structural Atlas" / "scripts"
STEP3_SCRIPTS = PROJECT_ROOT / "surrogate_experiment_results" / "Step3" / "scripts"
STEP1C_DIR = PROJECT_ROOT / "surrogate_experiment_results" / "Step1c"
for path in (STRUCTURAL_SCRIPTS, STEP3_SCRIPTS, STEP1C_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import step4_topology_common as common  # noqa: E402


DEFAULT_JOB_INDEX = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "Step2c K18-E1 Mechanism Visualization and Rank-Reversal Audit"
    / "results"
    / "k18_e1_sentinel_job_index.csv"
)
DEFAULT_JOB_METRICS = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "Step3"
    / "K18_analysis"
    / "experiment_01_budget4to1"
    / "results"
    / "formal_270_full_epoch_20260626"
    / "post_run_review"
    / "formal_post_run_job_metrics.csv"
)
DEFAULT_CANDIDATES = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "Step4 Topology Structural Atlas"
    / "results"
    / "feasible_candidates.csv"
)
DEFAULT_ARCS = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "Step4 Topology Structural Atlas"
    / "results"
    / "compatibility_arcs.csv"
)
DEFAULT_DECISION_OUTPUT = OVERLAY_DIR / "results" / "decision_solution_rows.csv"
DEFAULT_SUMMARY_OUTPUT = OVERLAY_DIR / "results" / "candidate_overlay_summary.csv"

DECISION_FIELDS = [
    "topology_id",
    "data_seed",
    "sample_size",
    "training_size",
    "validation_size",
    "test_sample_index",
    "graph",
    "solution_source",
    "method",
    "selection_metric",
    "selected_epoch",
    "theta_1",
    "theta_2",
    "edge_signature",
    "selected_candidate_ids",
    "candidate_count",
    "true_obj",
    "oracle_obj",
    "gap_to_oracle",
    "normalized_gap_to_oracle",
    "predicted_obj",
    "model_path",
    "eval_set_path",
]

SUMMARY_FIELDS = [
    "topology_id",
    "sample_size",
    "candidate_id",
    "candidate_type",
    "length",
    "oracle_denominator",
    "two_stage_denominator",
    "spoplus_denominator",
    "oracle_selection_count",
    "two_stage_selection_count",
    "spoplus_selection_count",
    "oracle_selection_rate",
    "two_stage_selection_rate",
    "spoplus_selection_rate",
    "spoplus_minus_2stage_rate",
]


def candidate_id_set(text: Any) -> set[str]:
    if text in ("", None):
        return set()
    return {part for part in str(text).split("|") if part}


def finite_rate(count: int, denom: int) -> float:
    return float(count / denom) if denom else float("nan")


def summarize_candidate_overlay(
    decision_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates_by_topology = common.candidate_rows_by_topology(candidate_rows)
    group_rows: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in decision_rows:
        key = (str(row["topology_id"]), common.parse_int(row["sample_size"]))
        group_rows.setdefault(key, []).append(row)

    output_rows: list[dict[str, Any]] = []
    for key in sorted(group_rows):
        topology_id, sample_size = key
        rows = group_rows[key]
        source_rows = {
            "oracle": [row for row in rows if str(row["solution_source"]) == "oracle"],
            "2stage": [row for row in rows if str(row["solution_source"]) == "2stage"],
            "spoplus": [row for row in rows if str(row["solution_source"]) == "spoplus"],
        }
        denominators = {source: len(items) for source, items in source_rows.items()}
        for candidate in candidates_by_topology.get(topology_id, []):
            candidate_id = str(candidate["candidate_id"])
            counts = {
                source: sum(
                    1
                    for row in items
                    if candidate_id in candidate_id_set(row.get("selected_candidate_ids", ""))
                )
                for source, items in source_rows.items()
            }
            two_rate = finite_rate(counts["2stage"], denominators["2stage"])
            spo_rate = finite_rate(counts["spoplus"], denominators["spoplus"])
            output_rows.append(
                {
                    "topology_id": topology_id,
                    "sample_size": sample_size,
                    "candidate_id": candidate_id,
                    "candidate_type": candidate.get("candidate_type", ""),
                    "length": common.parse_int(candidate.get("length")),
                    "oracle_denominator": denominators["oracle"],
                    "two_stage_denominator": denominators["2stage"],
                    "spoplus_denominator": denominators["spoplus"],
                    "oracle_selection_count": counts["oracle"],
                    "two_stage_selection_count": counts["2stage"],
                    "spoplus_selection_count": counts["spoplus"],
                    "oracle_selection_rate": finite_rate(counts["oracle"], denominators["oracle"]),
                    "two_stage_selection_rate": two_rate,
                    "spoplus_selection_rate": spo_rate,
                    "spoplus_minus_2stage_rate": spo_rate - two_rate
                    if np.isfinite(two_rate) and np.isfinite(spo_rate)
                    else float("nan"),
                }
            )
    return output_rows


def edge_signature(y: np.ndarray) -> str:
    selected = np.flatnonzero(np.asarray(y, dtype=float) > 0.5)
    return "|".join(str(int(idx)) for idx in selected)


def translated_edge_signature(y: np.ndarray, solver_to_template_edge: dict[int, int]) -> str:
    selected_solver_edges = [int(idx) for idx in np.flatnonzero(np.asarray(y, dtype=float) > 0.5)]
    selected_template_edges = []
    for edge_idx in selected_solver_edges:
        if edge_idx not in solver_to_template_edge:
            raise KeyError(f"solver edge index {edge_idx} has no template edge mapping")
        selected_template_edges.append(int(solver_to_template_edge[edge_idx]))
    return "|".join(str(idx) for idx in sorted(selected_template_edges))


def arcs_by_topology(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["topology_id"]), []).append(row)
    return grouped


def solver_to_template_edge_map(graph_path: str | Path, arc_rows: list[dict[str, Any]]) -> dict[int, int]:
    payload = common.read_json(graph_path)
    template_edge_by_arc = {
        (str(row["source"]), str(row["target"])): common.parse_int(row["edge_idx"])
        for row in arc_rows
    }
    mapping: dict[int, int] = {}
    edge_count = 0
    for source, node in payload["data"].items():
        for match in node.get("matches", []) or []:
            key = (str(source), str(match["recipient"]))
            if key not in template_edge_by_arc:
                raise KeyError(f"Arc {key} from {graph_path} is missing from template arcs")
            mapping[edge_count] = template_edge_by_arc[key]
            edge_count += 1
    return mapping


def normalized_gap(gap: float, oracle_obj: float, epsilon: float = 1e-9) -> float:
    return float(gap) / (abs(float(oracle_obj)) + float(epsilon))


def load_model(path: str | Path) -> dict[str, Any]:
    from evaluate_models import load_model_weight

    return load_model_weight(path)


def model_label(model: dict[str, Any]) -> str:
    if model["method"] == "2stage":
        return "2stage"
    if model["method"] == "spoplus":
        return "spoplus"
    return str(model["method"])


def materialize_sample(dataset: dict[str, Any], sample_index: int, output_dir: Path) -> Path:
    import fixed_topology_xy_common as fixed_common

    payloads = dataset["payloads"]
    if sample_index < 0 or sample_index >= len(payloads):
        raise IndexError(f"sample_index={sample_index} outside 0..{len(payloads) - 1}")
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_path = output_dir / f"G-{sample_index:06d}.json"
    fixed_common.atomic_write_json(graph_path, payloads[sample_index])
    return graph_path


def sample_indices_for_dataset(args: argparse.Namespace, sample_count: int) -> list[int]:
    if args.context_indices:
        return [int(value) for value in args.context_indices]
    limit = min(int(args.context_limit), int(sample_count))
    return list(range(limit))


def filter_job_rows(args: argparse.Namespace, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = rows
    if args.topology_ids:
        wanted = set(args.topology_ids)
        output = [row for row in output if str(row["topology_id"]) in wanted]
    if args.data_seeds:
        wanted = {int(value) for value in args.data_seeds}
        output = [row for row in output if common.parse_int(row["data_seed"]) in wanted]
    if args.sample_sizes:
        wanted = {int(value) for value in args.sample_sizes}
        output = [row for row in output if common.parse_int(row["sample_size"]) in wanted]
    if args.job_limit is not None:
        output = output[: int(args.job_limit)]
    return output


def fallback_weight_path(job_dir: Path, method: str) -> str:
    if method == "2stage":
        return str(job_dir / "2stage" / "model_weights" / "2stage_best_by_validation_mse_loss.npz")
    if method == "spoplus":
        return str(job_dir / "spoplus" / "model_weights" / "spoplus_best_by_validation_spoplus_loss.npz")
    raise ValueError(method)


def manifest_fields(job_dir: Path) -> dict[str, str]:
    manifest_path = job_dir / "evaluation" / "evaluation_input_manifest.json"
    if not manifest_path.exists():
        return {"eval_set_path": "", "two_stage_weight_path": "", "spoplus_weight_path": ""}
    payload = common.read_json(manifest_path)
    weights = [str(path) for path in payload.get("weights", [])]
    return {
        "eval_set_path": str(payload.get("eval_set_path", "")),
        "two_stage_weight_path": next(
            (path for path in weights if "2stage_best_by_validation_mse_loss" in path),
            "",
        ),
        "spoplus_weight_path": next(
            (path for path in weights if "spoplus_best_by_validation_spoplus_loss" in path),
            "",
        ),
    }


def job_rows_from_metrics(metrics_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in common.read_csv_rows(metrics_path):
        job_dir = Path(row["job_dir"])
        manifest = manifest_fields(job_dir)
        rows.append(
            {
                **row,
                "eval_set_path": manifest["eval_set_path"],
                "two_stage_weight_path": manifest["two_stage_weight_path"]
                or fallback_weight_path(job_dir, "2stage"),
                "spoplus_weight_path": manifest["spoplus_weight_path"]
                or fallback_weight_path(job_dir, "spoplus"),
            }
        )
    return rows


def load_job_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if Path(args.job_index).exists():
        return common.read_csv_rows(args.job_index)
    return job_rows_from_metrics(args.job_metrics)


def decision_row(
    *,
    job: dict[str, Any],
    sample_index: int,
    source: str,
    y_selected: np.ndarray,
    true_obj: float,
    oracle_obj: float,
    predicted_obj: float | str,
    candidates: list[dict[str, Any]],
    model: dict[str, Any] | None,
    solver_to_template_edge: dict[int, int],
) -> dict[str, Any]:
    signature = translated_edge_signature(y_selected, solver_to_template_edge)
    selected_candidate_ids = common.candidate_id_set_text(signature, candidates)
    gap = float(oracle_obj - true_obj)
    return {
        "topology_id": job["topology_id"],
        "data_seed": common.parse_int(job["data_seed"]),
        "sample_size": common.parse_int(job["sample_size"]),
        "training_size": common.parse_int(job["training_size"]),
        "validation_size": common.parse_int(job["validation_size"]),
        "test_sample_index": int(sample_index),
        "graph": f"G-{int(sample_index):06d}.json",
        "solution_source": source,
        "method": "" if model is None else model["method"],
        "selection_metric": "" if model is None else model["selection_metric"],
        "selected_epoch": "" if model is None else int(model["selected_epoch"]),
        "theta_1": "" if model is None else float(model["theta"][0]),
        "theta_2": "" if model is None else float(model["theta"][1]),
        "edge_signature": signature,
        "selected_candidate_ids": selected_candidate_ids,
        "candidate_count": len(common.pipe_text_set(selected_candidate_ids)),
        "true_obj": true_obj,
        "oracle_obj": oracle_obj,
        "gap_to_oracle": gap,
        "normalized_gap_to_oracle": normalized_gap(gap, oracle_obj),
        "predicted_obj": predicted_obj,
        "model_path": "" if model is None else model["path"],
        "eval_set_path": job["eval_set_path"],
    }


def compute_decision_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    candidate_rows = common.read_csv_rows(args.candidates)
    arc_rows = common.read_csv_rows(args.arcs)
    candidates_by_topology = common.candidate_rows_by_topology(candidate_rows)
    arcs_lookup = arcs_by_topology(arc_rows)
    job_rows = filter_job_rows(args, load_job_rows(args))

    if args.dry_run:
        print(f"Dry run jobs: {len(job_rows)}")
        print(f"Context limit: {args.context_limit}")
        return []

    import fixed_topology_xy_common as fixed_common
    import gurobipy as gp
    import step1c_common

    step1a = step1c_common.load_step1a_module()

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Seed", args.gurobi_seed)
    env.setParam("Threads", args.gurobi_threads)
    env.start()
    dataset_cache: dict[str, dict[str, Any]] = {}
    output_rows: list[dict[str, Any]] = []
    try:
        for job_idx, job in enumerate(job_rows, start=1):
            eval_set_path = str(job["eval_set_path"])
            if not eval_set_path:
                raise ValueError(
                    f"Missing eval_set_path for {job['topology_id']} seed={job['data_seed']} "
                    f"sample_size={job['sample_size']}; rebuild job index on garnet."
                )
            if eval_set_path not in dataset_cache:
                dataset_cache[eval_set_path] = fixed_common.read_npz_dataset(eval_set_path)
            dataset = dataset_cache[eval_set_path]
            sample_indices = sample_indices_for_dataset(args, len(dataset["payloads"]))
            models = [
                load_model(job["two_stage_weight_path"]),
                load_model(job["spoplus_weight_path"]),
            ]
            print(
                f"[{job_idx}/{len(job_rows)}] {job['topology_id']} "
                f"seed={job['data_seed']} sample_size={job['sample_size']} contexts={len(sample_indices)}",
                flush=True,
            )
            topology_id = str(job["topology_id"])
            candidates = candidates_by_topology[topology_id]
            topology_arcs = arcs_lookup[topology_id]
            for sample_index in sample_indices:
                graph_path = materialize_sample(
                    dataset,
                    sample_index,
                    args.scratch_dir
                    / str(job["topology_id"])
                    / f"data_seed={common.parse_int(job['data_seed']):06d}"
                    / f"sample_size={common.parse_int(job['sample_size']):03d}",
                )
                records: list[dict[str, Any]] = []
                try:
                    records = step1c_common.load_graph_records(
                        [graph_path],
                        env,
                        max_cycle=args.max_cycle,
                        max_chain=args.max_chain,
                    )
                    record = records[0]
                    edge_map = solver_to_template_edge_map(graph_path, topology_arcs)
                    w_true = np.asarray(record["w_true"], dtype=float)
                    y_oracle = np.asarray(record["y_optimal"], dtype=float)
                    oracle_obj = float(np.dot(w_true, y_oracle))
                    output_rows.append(
                        decision_row(
                            job=job,
                            sample_index=sample_index,
                            source="oracle",
                            y_selected=y_oracle,
                            true_obj=oracle_obj,
                            oracle_obj=oracle_obj,
                            predicted_obj="",
                            candidates=candidates,
                            model=None,
                            solver_to_template_edge=edge_map,
                        )
                    )
                    for model in models:
                        theta = np.asarray(model["theta"], dtype=float)
                        w_hat = np.asarray(record["X"], dtype=float) @ theta
                        y_selected = np.asarray(step1a.solve_once(w_hat, record["graph"], env), dtype=float)
                        true_obj = float(np.dot(w_true, y_selected))
                        output_rows.append(
                            decision_row(
                                job=job,
                                sample_index=sample_index,
                                source=model_label(model),
                                y_selected=y_selected,
                                true_obj=true_obj,
                                oracle_obj=oracle_obj,
                                predicted_obj=float(np.dot(w_hat, y_selected)),
                                candidates=candidates,
                                model=model,
                                solver_to_template_edge=edge_map,
                            )
                        )
                finally:
                    step1c_common.dispose_graph_records(records)
    finally:
        env.dispose()
    return output_rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-index", type=Path, default=DEFAULT_JOB_INDEX)
    parser.add_argument("--job-metrics", type=Path, default=DEFAULT_JOB_METRICS)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--arcs", type=Path, default=DEFAULT_ARCS)
    parser.add_argument("--decision-output", type=Path, default=DEFAULT_DECISION_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--topology-ids", nargs="+")
    parser.add_argument("--data-seeds", type=int, nargs="+")
    parser.add_argument("--sample-sizes", type=int, nargs="+")
    parser.add_argument("--job-limit", type=int)
    parser.add_argument("--context-limit", type=int, default=25)
    parser.add_argument("--context-indices", type=int, nargs="+")
    parser.add_argument("--max-cycle", type=int, default=3)
    parser.add_argument("--max-chain", type=int, default=4)
    parser.add_argument("--gurobi-seed", type=int, default=42)
    parser.add_argument("--gurobi-threads", type=int, default=1)
    parser.add_argument("--scratch-dir", type=Path, default=OVERLAY_DIR / "_scratch" / "materialized_contexts")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    candidate_rows = common.read_csv_rows(args.candidates)
    decision_rows = compute_decision_rows(args)
    if args.dry_run:
        return 0
    summary_rows = summarize_candidate_overlay(decision_rows, candidate_rows)
    common.write_csv(args.decision_output, decision_rows, DECISION_FIELDS)
    common.write_csv(args.summary_output, summary_rows, SUMMARY_FIELDS)
    print(f"Saved decision rows: {len(decision_rows)} to {args.decision_output}")
    print(f"Saved candidate overlay rows: {len(summary_rows)} to {args.summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
