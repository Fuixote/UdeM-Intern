#!/usr/bin/env python3
"""Compute rank-1/rank-2 KEP solutions on arc-density graph variants."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
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


DEFAULT_RESULTS_DIR = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "decision_analysis"
    / "results"
    / "density_sensitivity"
)
DEFAULT_MANIFEST = DEFAULT_RESULTS_DIR / "arc_density_graph_manifest.csv"
DEFAULT_OUTPUT = DEFAULT_RESULTS_DIR / "arc_density_second_best_gap.csv"


DENSITY_METADATA_FIELDS = [
    "case_id",
    "case_label",
    "base_graph_id",
    "variant_id",
    "variant_graph_path",
    "density_variant",
    "arc_delta_type",
    "original_num_arcs",
    "variant_num_arcs",
    "arc_delta",
    "added_arc_count",
    "removed_arc_count",
    "added_arc_keys",
    "removed_arc_keys",
    "perturb_seed",
    "generation_policy",
    "label_policy",
    "added_arc_source_policy",
    "added_arc_label_policy",
    "removed_arc_policy",
    "new_arc_label_mean",
    "existing_arc_label_mean",
]

ARC_SIGNATURE_FIELDS = [
    "solution_selected_edge_count",
    "solution_arc_key_signature",
    "oracle_arc_key_signature",
    "rank1_arc_key_signature",
    "original_oracle_arc_key_signature",
]

CSV_FIELDS = DENSITY_METADATA_FIELDS + SECOND_BEST_FIELDS + ARC_SIGNATURE_FIELDS


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


def sort_vertex_id(value: str) -> tuple[int, Any]:
    text = str(value)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def sort_arc_key(key: str) -> tuple[tuple[int, Any], tuple[int, Any]]:
    src, dst = key.split("->", 1)
    return sort_vertex_id(src), sort_vertex_id(dst)


def normalize_signature(keys: list[str] | set[str]) -> str:
    return "|".join(sorted(set(keys), key=sort_arc_key))


def to_numpy_edge_index(edge_index: Any) -> np.ndarray:
    if hasattr(edge_index, "detach"):
        edge_index = edge_index.detach().cpu().numpy()
    return np.asarray(edge_index, dtype=int)


def edge_arc_keys_from_json(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    keys: list[str] = []
    for src_id, node in payload.get("data", {}).items():
        for match in node.get("matches", []) or []:
            keys.append(f"{src_id}->{match['recipient']}")
    return keys


def edge_arc_keys(record: dict[str, Any]) -> list[str]:
    graph = record.get("graph", {})
    id_map_rev = graph.get("id_map_rev")
    if id_map_rev is not None and "edge_index" in graph:
        edge_index = to_numpy_edge_index(graph["edge_index"])
        if edge_index.shape[0] != 2:
            raise ValueError(f"Expected edge_index shape [2, E], got {edge_index.shape}")
        keys: list[str] = []
        for edge_idx in range(edge_index.shape[1]):
            src = str(id_map_rev[int(edge_index[0, edge_idx])])
            dst = str(id_map_rev[int(edge_index[1, edge_idx])])
            keys.append(f"{src}->{dst}")
        return keys

    if "path" not in record:
        raise ValueError("Record has neither graph.id_map_rev nor path for stable arc keys")
    return edge_arc_keys_from_json(record["path"])


def arc_signature_from_y(y: Any, edge_keys: list[str]) -> str:
    selected = np.asarray(y, dtype=float) > 0.5
    if len(selected) != len(edge_keys):
        raise ValueError(f"Shape mismatch: selected={len(selected)}, edge_keys={len(edge_keys)}")
    return normalize_signature([edge_keys[idx] for idx, keep in enumerate(selected) if keep])


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
    return normalize_signature(selected_keys)


def density_metadata(manifest_row: dict[str, str]) -> dict[str, Any]:
    return {field: manifest_row.get(field, "") for field in DENSITY_METADATA_FIELDS}


def augment_solution_rows(
    *,
    rows: list[dict[str, Any]],
    record: dict[str, Any],
    manifest_row: dict[str, str],
) -> list[dict[str, Any]]:
    edge_keys = edge_arc_keys(record)
    oracle_signature = arc_signature_from_y(record["y_optimal"], edge_keys)
    metadata = density_metadata(manifest_row)

    output: list[dict[str, Any]] = []
    for row in rows:
        augmented = dict(row)
        augmented.update(metadata)
        augmented["solution_selected_edge_count"] = int(row.get("edge_count", 0))
        augmented["solution_arc_key_signature"] = arc_signature_from_edge_signature(
            row.get("solution_edge_signature", ""),
            edge_keys,
        )
        augmented["oracle_arc_key_signature"] = oracle_signature
        output.append(augmented)
    return output


def row_int(row: dict[str, Any], key: str) -> int:
    return int(float(row[key]))


def finalize_cross_variant_signatures(rows: list[dict[str, Any]]) -> None:
    original_oracle_by_graph: dict[str, str] = {}
    rank1_by_variant_method: dict[tuple[str, str, str], str] = {}

    for row in rows:
        base_graph_id = str(row.get("base_graph_id", ""))
        density_variant = str(row.get("density_variant", ""))
        method = str(row.get("method_label", ""))
        if density_variant == "original" and base_graph_id not in original_oracle_by_graph:
            original_oracle_by_graph[base_graph_id] = str(row.get("oracle_arc_key_signature", ""))
        if row_int(row, "solution_rank") == 1:
            rank1_by_variant_method[(base_graph_id, density_variant, method)] = str(
                row.get("solution_arc_key_signature", "")
            )

    for row in rows:
        key = (
            str(row.get("base_graph_id", "")),
            str(row.get("density_variant", "")),
            str(row.get("method_label", "")),
        )
        row["rank1_arc_key_signature"] = rank1_by_variant_method.get(key, "")
        row["original_oracle_arc_key_signature"] = original_oracle_by_graph.get(
            str(row.get("base_graph_id", "")),
            "",
        )


def selected_manifest_rows(args) -> list[dict[str, str]]:
    rows = read_csv_rows(args.manifest)
    if args.graphs:
        wanted_graphs = set(args.graphs)
        rows = [row for row in rows if row.get("base_graph_id") in wanted_graphs]
    if args.variants:
        wanted_variants = set(args.variants)
        rows = [row for row in rows if row.get("density_variant") in wanted_variants]
    if args.variant_limit is not None:
        rows = rows[: args.variant_limit]
    if not rows:
        raise ValueError(f"No manifest rows selected from {args.manifest}")
    return rows


def resolve_variant_graph_path(path: str | Path) -> Path:
    graph_path = Path(path)
    if graph_path.is_absolute():
        return graph_path
    return PROJECT_ROOT / graph_path


def compute_arc_density_rows(args) -> list[dict[str, Any]]:
    common, load_model_weight, _, _, _ = ensure_step1c_imports()

    import gurobipy as gp

    manifest_rows = selected_manifest_rows(args)
    manifest_by_seed: dict[int, list[dict[str, str]]] = {}
    for row in manifest_rows:
        manifest_by_seed.setdefault(int(row["subset_seed"]), []).append(row)

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Seed", args.gurobi_seed)
    env.start()

    output_rows: list[dict[str, Any]] = []
    try:
        for seed_idx, (subset_seed, seed_rows) in enumerate(sorted(manifest_by_seed.items()), start=1):
            run_dir = resolve_run_dir(args.run_root, args.regime, subset_seed)
            models = load_models(run_dir, load_model_weight)
            graph_paths = [resolve_variant_graph_path(row["variant_graph_path"]) for row in seed_rows]
            case = {
                "case_type": "",
                "subset_seed": str(subset_seed),
            }
            print(
                f"[{seed_idx}/{len(manifest_by_seed)}] seed={subset_seed} variants={len(graph_paths)}",
                flush=True,
            )

            records = []
            try:
                records = common.load_graph_records(
                    graph_paths,
                    env,
                    max_cycle=args.max_cycle,
                    max_chain=args.max_chain,
                )
                for model in models:
                    label = method_label(model)
                    if label not in set(args.method_labels):
                        continue
                    print(f"  method={label}", flush=True)
                    for record, manifest_row in zip(records, seed_rows):
                        case["case_type"] = manifest_row.get("case_label", "")
                        rows = rows_for_model_record(
                            args=args,
                            case=case,
                            model=model,
                            label=label,
                            record=record,
                        )
                        output_rows.extend(
                            augment_solution_rows(
                                rows=rows,
                                record=record,
                                manifest_row=manifest_row,
                            )
                        )
            finally:
                common.dispose_graph_records(records)
    finally:
        env.dispose()

    finalize_cross_variant_signatures(output_rows)
    return output_rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compute second-best solution rows for arc-density graph variants."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--regime", default="step2b_poly_d8")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-cycle", type=int, default=3)
    parser.add_argument("--max-chain", type=int, default=4)
    parser.add_argument("--max-solutions", type=int, default=2)
    parser.add_argument("--max-cut-attempts", type=int, default=25)
    parser.add_argument("--no-reset-before-solve", action="store_true")
    parser.add_argument("--gurobi-seed", type=int, default=42)
    parser.add_argument("--method-labels", nargs="+", default=list(DEFAULT_METHOD_LABELS))
    parser.add_argument("--graphs", nargs="+")
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--variant-limit", type=int)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    rows = compute_arc_density_rows(args)
    write_csv(args.output, rows, CSV_FIELDS)
    print(f"Saved {len(rows)} density sensitivity rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
