#!/usr/bin/env python3
"""Audit whether Case C persists when labels vary on a fixed KEP topology."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
STEP2C_SCRIPT = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "Step2"
    / "Step2c_polynomial_degree_multiplicative_noise"
    / "data-processing.py"
)
DEFAULT_DATASET_DIR = (
    PROJECT_ROOT / "dataset" / "processed" / "step2b_poly_d8_main2000_seed20260523"
)
DEFAULT_CASE_INDEX = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "decision_analysis"
    / "results"
    / "case_studies"
    / "case_study_index.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "decision_analysis"
    / "results"
    / "fixed_topology_label_seed"
)

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compare_decisions_per_graph import (  # noqa: E402
    DEFAULT_RUN_ROOT,
    ensure_step1c_imports,
    load_models,
    method_label,
    resolve_run_dir,
)
from compute_second_best_solutions import (  # noqa: E402
    CSV_FIELDS as SECOND_BEST_FIELDS,
    DEFAULT_METHOD_LABELS,
    rows_for_model_record,
)


FIXED_TOPOLOGY_FIELDS = [
    "base_graph_id",
    "base_subset_seed",
    "label_seed",
    "epsilon_bar",
    "topology_hash",
    "label_hash",
    "topology_edge_count",
    "relabeled_graph_path",
    "case_c_signature_for_label_seed",
    "solution_arc_key_signature",
    "oracle_arc_key_signature",
]
CSV_FIELDS = FIXED_TOPOLOGY_FIELDS + SECOND_BEST_FIELDS


def load_step2c_module():
    spec = importlib.util.spec_from_file_location("step2c_data_processing", STEP2C_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def iter_edge_items(payload: dict[str, Any]):
    for source_id, node in payload.get("data", {}).items():
        for edge_idx, match in enumerate(node.get("matches", []) or []):
            yield str(source_id), int(edge_idx), match


def edge_count(payload: dict[str, Any]) -> int:
    return sum(1 for _ in iter_edge_items(payload))


def topology_edges(payload: dict[str, Any]) -> list[str]:
    edges = []
    for source_id, _, match in iter_edge_items(payload):
        edges.append(f"{source_id}->{match['recipient']}")
    return sorted(edges)


def topology_hash(payload: dict[str, Any]) -> str:
    material = "\n".join(topology_edges(payload)).encode("utf-8")
    return hashlib.sha256(material).hexdigest()[:16]


def edge_labels(payload: dict[str, Any]) -> list[float]:
    return [
        float(match.get("ground_truth_label", 0.0))
        for _, _, match in iter_edge_items(payload)
    ]


def edge_label_hash(payload: dict[str, Any]) -> str:
    material = "\n".join(f"{value:.10g}" for value in edge_labels(payload)).encode("utf-8")
    return hashlib.sha256(material).hexdigest()[:16]


def normalize_arc_signature(keys: list[str] | set[str]) -> str:
    return "|".join(sorted(set(keys), key=sort_arc_key))


def sort_vertex_id(value: str) -> tuple[int, Any]:
    text = str(value)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def sort_arc_key(key: str) -> tuple[tuple[int, Any], tuple[int, Any]]:
    src, dst = key.split("->", 1)
    return sort_vertex_id(src), sort_vertex_id(dst)


def edge_arc_keys_from_payload(payload: dict[str, Any]) -> list[str]:
    return [
        f"{source_id}->{match['recipient']}"
        for source_id, _, match in iter_edge_items(payload)
    ]


def arc_signature_from_y(y: Any, edge_keys: list[str]) -> str:
    selected = np.asarray(y, dtype=float) > 0.5
    if len(selected) != len(edge_keys):
        raise ValueError(f"Shape mismatch: selected={len(selected)}, edge_keys={len(edge_keys)}")
    return normalize_arc_signature(
        [edge_keys[idx] for idx, keep in enumerate(selected) if keep]
    )


def arc_signature_from_edge_signature(edge_signature: str, edge_keys: list[str]) -> str:
    if not str(edge_signature).strip():
        return ""
    selected_keys: list[str] = []
    for token in str(edge_signature).split("|"):
        if token == "":
            continue
        idx = int(token)
        if idx < 0 or idx >= len(edge_keys):
            raise ValueError(f"Edge index {idx} out of range for {len(edge_keys)} edge keys")
        selected_keys.append(edge_keys[idx])
    return normalize_arc_signature(selected_keys)


def label_config_from_payload(
    payload: dict[str, Any],
    label_seed: int,
    epsilon_bar: float,
) -> dict[str, Any]:
    dp = load_step2c_module()
    metadata = payload.get("metadata", {})
    return {
        "label_mode": dp.LABEL_MODE_STEP2C_POLYNOMIAL_DEGREE_MULTIPLICATIVE_NOISE,
        "clean_linear_utility_weight": float(
            metadata.get("clean_linear_utility_weight", 10.0)
        ),
        "clean_linear_cpra_weight": float(metadata.get("clean_linear_cpra_weight", 5.0)),
        "clean_linear_noise_sigma": float(metadata.get("clean_linear_noise_sigma", 0.08)),
        "step2c_degree": int(metadata.get("step2b_degree", metadata.get("step2c_degree", 8))),
        "step2c_kappa": float(metadata.get("step2b_kappa", metadata.get("step2c_kappa", 3.0))),
        "step2c_delta": float(
            metadata.get("step2b_delta", metadata.get("step2c_delta", 1e-12))
        ),
        "step2c_epsilon_bar": float(epsilon_bar),
        "label_seed": int(label_seed),
    }


def graph_label_context(
    payload: dict[str, Any],
    label_config: dict[str, Any],
) -> dict[str, Any]:
    dp = load_step2c_module()
    latent_values = []
    for _, _, match in iter_edge_items(payload):
        latent_values.append(
            dp.clean_linear_utility_cpra_label(
                match.get("utility", 0.0),
                match.get("recipient_cpra", 0.0),
                label_config,
            )
        )
    clean_mean = sum(latent_values) / len(latent_values) if latent_values else 0.0
    scores = [
        dp.step2c_polynomial_score(value, clean_mean, label_config)[1]
        for value in latent_values
    ]
    score_mean = sum(scores) / len(scores) if scores else 1.0
    return {
        "clean_linear_mean": clean_mean,
        "polynomial_score_mean": score_mean,
        "clean_linear_edge_count": len(latent_values),
    }


def stable_source_key(
    payload: dict[str, Any],
    source_id: str,
    edge_idx: int,
    match: dict[str, Any],
) -> str:
    source_file = payload.get("metadata", {}).get("source_file", "processed_graph")
    return f"{source_file}|{source_id}|{edge_idx}|{match.get('recipient')}"


def relabel_payload_step2c(
    payload: dict[str, Any],
    label_seed: int,
    epsilon_bar: float = 0.5,
) -> dict[str, Any]:
    dp = load_step2c_module()
    output = copy.deepcopy(payload)
    config = label_config_from_payload(
        output,
        label_seed=label_seed,
        epsilon_bar=epsilon_bar,
    )
    context = graph_label_context(output, config)
    for source_id, edge_idx, match in iter_edge_items(output):
        fields = dp.compute_ground_truth_label_fields(
            config,
            expected_transplant_count=match.get("expected_transplant_count", 0.0),
            qaly=match.get("qaly", 0.0),
            priority_multiplier=match.get("priority_multiplier", 1.0),
            source_key=stable_source_key(output, source_id, edge_idx, match),
            utility=match.get("utility", 0.0),
            cpra=match.get("recipient_cpra", 0.0),
            graph_label_context=context,
        )
        match.update(fields)
    output.setdefault("metadata", {})
    output["metadata"].update(
        {
            "ground_truth_label_mode": "fixed_topology_step2c_multiplicative_relabel",
            "fixed_topology_relabel_seed": int(label_seed),
            "fixed_topology_epsilon_bar": float(epsilon_bar),
            "fixed_topology_source_topology_hash": topology_hash(payload),
        }
    )
    return output


def case_c_signature(
    rows: list[dict[str, Any]],
    two_stage_min_gap: float = 0.05,
    spoplus_max_gap: float = 0.02,
    min_gap_reduction: float = 0.05,
) -> bool:
    rank1 = {
        row["method_label"]: float(row["normalized_gap_to_oracle"])
        for row in rows
        if int(float(row["solution_rank"])) == 1
    }
    two_stage_gap = rank1.get("2stage_val_mse")
    spoplus_gap = rank1.get("spoplus_val_spoplus_loss")
    if two_stage_gap is None or spoplus_gap is None:
        return False
    return (
        two_stage_gap >= two_stage_min_gap
        and spoplus_gap <= spoplus_max_gap
        and (two_stage_gap - spoplus_gap) >= min_gap_reduction
    )


def case_seed_by_graph(case_index_path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv_rows(case_index_path)
    return {row["graph_id"]: row for row in rows}


def relabeled_graph_path(output_dir: Path, graph_id: str, label_seed: int) -> Path:
    stem = Path(graph_id).stem
    return output_dir / "graphs" / f"{stem}__fixed_topology_step2c_seed{int(label_seed)}.json"


def write_relabeled_graphs(args) -> list[dict[str, Any]]:
    output_rows = []
    for graph_id in args.graphs:
        base_path = args.dataset_dir / graph_id
        payload = read_json(base_path)
        base_hash = topology_hash(payload)
        for label_seed in args.label_seeds:
            relabeled = relabel_payload_step2c(
                payload,
                label_seed=int(label_seed),
                epsilon_bar=float(args.epsilon_bar),
            )
            if topology_hash(relabeled) != base_hash:
                raise AssertionError(
                    f"Topology changed for {graph_id} label_seed={label_seed}"
                )
            path = relabeled_graph_path(args.output_dir, graph_id, int(label_seed))
            write_json(path, relabeled)
            output_rows.append(
                {
                    "base_graph_id": graph_id,
                    "label_seed": int(label_seed),
                    "epsilon_bar": float(args.epsilon_bar),
                    "topology_hash": topology_hash(relabeled),
                    "label_hash": edge_label_hash(relabeled),
                    "topology_edge_count": edge_count(relabeled),
                    "relabeled_graph_path": path,
                    "edge_keys": edge_arc_keys_from_payload(relabeled),
                }
            )
    return output_rows


def replay_relabel_rows(args) -> list[dict[str, Any]]:
    common, load_model_weight, _, _, _ = ensure_step1c_imports()

    import gurobipy as gp

    case_by_graph = case_seed_by_graph(args.case_index)
    requested = set(args.graphs)
    missing = sorted(requested - set(case_by_graph))
    if missing:
        raise ValueError(f"Requested graphs are missing from case index: {missing}")

    relabeled_specs = write_relabeled_graphs(args)
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Seed", int(args.gurobi_seed))
    env.start()

    all_rows: list[dict[str, Any]] = []
    try:
        for spec_idx, spec in enumerate(relabeled_specs, start=1):
            graph_id = str(spec["base_graph_id"])
            case = case_by_graph[graph_id]
            subset_seed = int(case["subset_seed"])
            run_dir = resolve_run_dir(args.run_root, args.regime, subset_seed)
            models = load_models(run_dir, load_model_weight)
            print(
                "Replaying "
                f"{spec_idx}/{len(relabeled_specs)} graph={graph_id} "
                f"label_seed={spec['label_seed']} subset_seed={subset_seed}",
                flush=True,
            )
            records = common.load_graph_records(
                [Path(spec["relabeled_graph_path"])],
                env,
                max_cycle=args.max_cycle,
                max_chain=args.max_chain,
            )
            try:
                record = records[0]
                label_seed_rows: list[dict[str, Any]] = []
                for model in models:
                    label = method_label(model)
                    if label not in set(args.method_labels):
                        continue
                    rows = rows_for_model_record(
                        args=args,
                        case=case,
                        model=model,
                        label=label,
                        record=record,
                    )
                    label_seed_rows.extend(rows)

                signature = case_c_signature(
                    label_seed_rows,
                    two_stage_min_gap=args.two_stage_min_gap,
                    spoplus_max_gap=args.spoplus_max_gap,
                    min_gap_reduction=args.min_gap_reduction,
                )
                oracle_signature = arc_signature_from_y(
                    record["y_optimal"],
                    spec["edge_keys"],
                )
                for row in label_seed_rows:
                    row.update(
                        {
                            "base_graph_id": graph_id,
                            "base_subset_seed": subset_seed,
                            "label_seed": int(spec["label_seed"]),
                            "epsilon_bar": float(spec["epsilon_bar"]),
                            "topology_hash": spec["topology_hash"],
                            "label_hash": spec["label_hash"],
                            "topology_edge_count": int(spec["topology_edge_count"]),
                            "relabeled_graph_path": str(spec["relabeled_graph_path"]),
                            "case_c_signature_for_label_seed": signature,
                            "solution_arc_key_signature": arc_signature_from_edge_signature(
                                str(row.get("solution_edge_signature", "")),
                                spec["edge_keys"],
                            ),
                            "oracle_arc_key_signature": oracle_signature,
                        }
                    )
                all_rows.extend(label_seed_rows)
            finally:
                common.dispose_graph_records(records)
    finally:
        env.dispose()
    return all_rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Replay fixed-topology Step2c label-seed perturbations for selected "
            "Case C graphs."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--case-index", type=Path, default=DEFAULT_CASE_INDEX)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--regime", default="step2b_poly_d8")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "fixed_topology_label_seed_rows.csv",
    )
    parser.add_argument("--graphs", nargs="+", default=["G-392.json", "G-1560.json"])
    parser.add_argument("--label-seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--label-seed-start", type=int)
    parser.add_argument("--label-seed-stop", type=int)
    parser.add_argument("--epsilon-bar", type=float, default=0.5)
    parser.add_argument("--max-cycle", type=int, default=3)
    parser.add_argument("--max-chain", type=int, default=4)
    parser.add_argument("--max-solutions", type=int, default=2)
    parser.add_argument("--max-cut-attempts", type=int, default=20)
    parser.add_argument("--method-labels", nargs="+", default=list(DEFAULT_METHOD_LABELS))
    parser.add_argument("--gurobi-seed", type=int, default=42)
    parser.add_argument("--two-stage-min-gap", type=float, default=0.05)
    parser.add_argument("--spoplus-max-gap", type=float, default=0.02)
    parser.add_argument("--min-gap-reduction", type=float, default=0.05)
    parser.add_argument("--no-reset-before-solve", action="store_true")
    args = parser.parse_args(argv)
    if args.label_seed_start is not None or args.label_seed_stop is not None:
        if args.label_seed_start is None or args.label_seed_stop is None:
            parser.error("--label-seed-start and --label-seed-stop must be provided together")
        if args.label_seed_stop < args.label_seed_start:
            parser.error("--label-seed-stop must be >= --label-seed-start")
        args.label_seeds = list(range(args.label_seed_start, args.label_seed_stop + 1))
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    rows = replay_relabel_rows(args)
    write_csv(args.output, rows, CSV_FIELDS)
    print(f"Saved {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
