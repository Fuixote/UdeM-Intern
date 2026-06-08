#!/usr/bin/env python3
"""Generate controlled arc-density variants for selected Step2b KEP graphs."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CASE_INDEX = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "decision_analysis"
    / "results"
    / "case_studies"
    / "case_study_index.csv"
)
DEFAULT_DATASET_DIR = PROJECT_ROOT / "dataset" / "processed" / "step2b_poly_d8_main2000_seed20260523"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "decision_analysis"
    / "results"
    / "density_sensitivity"
)
DEFAULT_GRAPHS = ("G-696.json", "G-392.json", "G-1560.json")
DEFAULT_VARIANTS = ("original", "add25pct", "add25arcs", "remove25arcs", "remove25pct")


MANIFEST_FIELDS = [
    "case_id",
    "case_label",
    "subset_seed",
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

GENERATION_POLICY = "density_sensitivity_structural_perturbation"
LABEL_POLICY = "frozen_existing_edges_original_scale_new_edges"
ADDED_ARC_SOURCE_POLICY = "sample_missing_vertex_arcs"
ADDED_ARC_LABEL_POLICY = "synthetic_step2b_original_scale"
REMOVED_ARC_POLICY = "sample_existing_vertex_arcs"


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_manifest(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in MANIFEST_FIELDS})


def manifest_graph_path(path: str | Path) -> str:
    graph_path = Path(path)
    try:
        return graph_path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(graph_path)


def parse_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "Unknown":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def sort_vertex_id(value: str) -> tuple[int, Any]:
    text = str(value)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def arc_key(src_vertex_id: str, dst_vertex_id: str) -> str:
    return f"{src_vertex_id}->{dst_vertex_id}"


def split_arc_key(key: str) -> tuple[str, str]:
    src, dst = key.split("->", 1)
    return src, dst


def join_keys(keys: set[str] | list[str]) -> str:
    return "|".join(sorted(keys, key=lambda item: (sort_vertex_id(item.split('->', 1)[0]), sort_vertex_id(item.split('->', 1)[1]))))


def iter_arc_items(payload: dict[str, Any]):
    for src_id, node in payload.get("data", {}).items():
        for match in node.get("matches", []) or []:
            yield str(src_id), match


def count_arcs(payload: dict[str, Any]) -> int:
    return sum(1 for _ in iter_arc_items(payload))


def existing_arc_keys(payload: dict[str, Any]) -> set[str]:
    return {arc_key(src_id, str(match["recipient"])) for src_id, match in iter_arc_items(payload)}


def pair_vertex_ids(payload: dict[str, Any]) -> list[str]:
    return sorted(
        [str(node_id) for node_id, node in payload.get("data", {}).items() if node.get("type") == "Pair"],
        key=sort_vertex_id,
    )


def source_vertex_ids(payload: dict[str, Any]) -> list[str]:
    return sorted([str(node_id) for node_id in payload.get("data", {})], key=sort_vertex_id)


def variant_rng(seed: int, graph_id: str, density_variant: str) -> random.Random:
    material = f"{seed}|{graph_id}|{density_variant}".encode("utf-8")
    digest = hashlib.sha256(material).hexdigest()
    return random.Random(int(digest[:16], 16))


def sample_keys(keys: list[str], count: int, rng: random.Random, what: str) -> set[str]:
    if count < 0:
        raise ValueError(f"{what} count must be non-negative")
    if count > len(keys):
        raise ValueError(f"Cannot sample {count} {what}; only {len(keys)} available")
    if count == 0:
        return set()
    return set(rng.sample(keys, count))


def all_missing_arc_keys(payload: dict[str, Any]) -> list[str]:
    existing = existing_arc_keys(payload)
    missing: list[str] = []
    for src_id in source_vertex_ids(payload):
        for dst_id in pair_vertex_ids(payload):
            if src_id == dst_id:
                continue
            key = arc_key(src_id, dst_id)
            if key not in existing:
                missing.append(key)
    return sorted(missing, key=lambda item: (sort_vertex_id(item.split("->", 1)[0]), sort_vertex_id(item.split("->", 1)[1])))


def remove_arcs(payload: dict[str, Any], removed_keys: set[str]) -> None:
    for src_id, node in payload.get("data", {}).items():
        matches = node.get("matches", []) or []
        node["matches"] = [
            match for match in matches if arc_key(str(src_id), str(match["recipient"])) not in removed_keys
        ]


def graph_label_config(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata", {})
    return {
        "clean_linear_utility_weight": parse_float(metadata.get("clean_linear_utility_weight"), 10.0),
        "clean_linear_cpra_weight": parse_float(metadata.get("clean_linear_cpra_weight"), 5.0),
        "step2b_degree": int(parse_float(metadata.get("step2b_degree"), 8)),
        "step2b_kappa": parse_float(metadata.get("step2b_kappa"), 3.0),
        "step2b_delta": parse_float(metadata.get("step2b_delta"), 1e-12),
    }


def clean_linear_label(utility: float, cpra: float, config: dict[str, Any]) -> float:
    utility_norm = clamp(parse_float(utility) / 100.0, 0.0, 1.0)
    cpra_value = clamp(parse_float(cpra), 0.0, 1.0)
    value = config["clean_linear_utility_weight"] * utility_norm + config["clean_linear_cpra_weight"] * cpra_value
    return round(max(0.0, value), 4)


def original_graph_label_context(payload: dict[str, Any]) -> dict[str, Any]:
    for _, match in iter_arc_items(payload):
        if "step2b_graph_clean_linear_mean" in match and "step2b_graph_polynomial_score_mean" in match:
            return {
                "clean_linear_mean": parse_float(match.get("step2b_graph_clean_linear_mean")),
                "polynomial_score_mean": parse_float(match.get("step2b_graph_polynomial_score_mean"), 1.0),
                "clean_linear_edge_count": int(parse_float(match.get("step2b_graph_clean_linear_edge_count"), count_arcs(payload))),
            }

    config = graph_label_config(payload)
    latent_values = [
        clean_linear_label(match.get("utility", 0.0), match.get("recipient_cpra", 0.0), config)
        for _, match in iter_arc_items(payload)
    ]
    clean_mean = sum(latent_values) / len(latent_values) if latent_values else 0.0
    scores = [
        ((value / (clean_mean + config["step2b_delta"])) + config["step2b_kappa"]) ** config["step2b_degree"]
        - config["step2b_kappa"] ** config["step2b_degree"]
        for value in latent_values
    ]
    score_mean = sum(scores) / len(scores) if scores else 1.0
    return {
        "clean_linear_mean": clean_mean,
        "polynomial_score_mean": score_mean,
        "clean_linear_edge_count": len(latent_values),
    }


def step2b_label_fields(
    utility: float,
    cpra: float,
    config: dict[str, Any],
    label_context: dict[str, Any],
) -> dict[str, Any]:
    latent = clean_linear_label(utility, cpra, config)
    graph_mean = parse_float(label_context.get("clean_linear_mean"), latent)
    score_mean = parse_float(label_context.get("polynomial_score_mean"), 1.0)
    edge_count = int(parse_float(label_context.get("clean_linear_edge_count"), 0))
    delta = config["step2b_delta"]
    kappa = config["step2b_kappa"]
    degree = int(config["step2b_degree"])
    q_value = latent / (graph_mean + delta) if graph_mean + delta != 0.0 else 0.0
    polynomial_score = (q_value + kappa) ** degree - kappa**degree
    label = graph_mean * polynomial_score / (score_mean + delta) if score_mean + delta != 0.0 else 0.0
    label = round(max(0.0, label), 4)
    return {
        "ground_truth_label": label,
        "latent_clean_linear_label": round(latent, 4),
        "step2b_polynomial_label": label,
        "step2b_q_value": round(q_value, 6),
        "step2b_polynomial_score": round(polynomial_score, 6),
        "step2b_graph_clean_linear_mean": round(graph_mean, 6),
        "step2b_graph_polynomial_score_mean": round(score_mean, 6),
        "step2b_graph_clean_linear_edge_count": edge_count,
        "step2b_degree": degree,
        "step2b_kappa": round(kappa, 6),
        "step2b_delta": delta,
    }


def source_donor(node: dict[str, Any]) -> dict[str, Any]:
    if node.get("type") == "Pair":
        donors = node.get("donors", []) or []
        if not donors:
            raise ValueError(f"Pair node {node.get('id')} has no donors")
        return donors[0]
    return node.get("donor", {})


def utilities_by_source(payload: dict[str, Any]) -> dict[str, list[float]]:
    output: dict[str, list[float]] = {}
    for src_id, match in iter_arc_items(payload):
        output.setdefault(src_id, []).append(parse_float(match.get("utility"), 0.0))
    return output


def graph_context_from_keys(payload: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    outgoing: dict[str, set[str]] = {src_id: set() for src_id in source_vertex_ids(payload)}
    incoming: dict[str, set[str]] = {dst_id: set() for dst_id in pair_vertex_ids(payload)}
    for key in keys:
        src, dst = split_arc_key(key)
        outgoing.setdefault(src, set()).add(dst)
        incoming.setdefault(dst, set()).add(src)
    max_out = max((len(targets) for targets in outgoing.values()), default=0)
    return {
        "donor_out_degree": {src: len(targets) for src, targets in outgoing.items()},
        "recipient_in_degree": {dst: len(sources) for dst, sources in incoming.items()},
        "max_log_out_degree": math.log1p(max_out) if max_out > 0 else 1.0,
        "outgoing": outgoing,
    }


def synthetic_match(
    *,
    original_payload: dict[str, Any],
    final_keys: set[str],
    src_id: str,
    dst_id: str,
    rng: random.Random,
) -> dict[str, Any]:
    nodes = original_payload["data"]
    src_node = nodes[src_id]
    dst_node = nodes[dst_id]
    donor = source_donor(src_node)
    patient = dst_node["patient"]
    source_utilities = utilities_by_source(original_payload)
    all_utilities = [utility for values in source_utilities.values() for utility in values] or [50.0]
    utility_pool = source_utilities.get(src_id) or all_utilities
    utility = int(round(rng.choice(utility_pool)))
    cpra = parse_float(patient.get("cPRA"), 0.0)
    donor_age = donor.get("dage", donor.get("age", "Unknown"))
    donor_bt = donor.get("bloodtype", "Unknown")
    recipient_age = patient.get("age", "Unknown")
    recipient_bt = patient.get("bloodtype", "Unknown")
    utility_norm = clamp(utility / 100.0, 0.0, 1.0)
    survival_time = round(5.0 + 20.0 * math.sqrt(utility_norm), 4)
    qaly = round(survival_time * (0.75 + 0.25 * math.sqrt(utility_norm)), 4)
    arc_failure_prob = round(clamp(0.05 + 0.2 * (1.0 - utility_norm) + 0.05 * cpra, 0.01, 0.99), 4)
    success_prob = round(1.0 - arc_failure_prob, 4)
    source_vertex_failure_prob = parse_float(src_node.get("vertex_failure_prob"), 0.0)
    target_vertex_failure_prob = parse_float(dst_node.get("vertex_failure_prob"), 0.0)
    expected_transplant_count = round(
        success_prob
        * (1.0 - source_vertex_failure_prob)
        * (1.0 - target_vertex_failure_prob),
        4,
    )
    context = graph_context_from_keys(original_payload, final_keys)
    recipient_in_degree = context["recipient_in_degree"].get(dst_id, 0)
    donor_out_degree = context["donor_out_degree"].get(src_id, 0)
    target_scarcity = round(1.0 / math.sqrt(recipient_in_degree + 1.0), 4)
    donor_flexibility = round(math.log1p(donor_out_degree) / context["max_log_out_degree"], 4)
    has_reciprocal_edge = arc_key(dst_id, src_id) in final_keys
    priority_multiplier = round(
        clamp(
            1.0
            + 0.4 * cpra * target_scarcity
            + 0.12 * (1.0 if has_reciprocal_edge else 0.0)
            - 0.2 * donor_flexibility * (1.0 - target_scarcity),
            0.65,
            1.85,
        ),
        4,
    )
    config = graph_label_config(original_payload)
    label_fields = step2b_label_fields(
        utility,
        cpra,
        config,
        original_graph_label_context(original_payload),
    )
    match = {
        "recipient": dst_id,
        "utility": utility,
        "graft_survival_time": survival_time,
        "qaly": qaly,
        "medical_success_score": round(clamp(0.7 * utility_norm + 0.3 * (1.0 - cpra), 0.0, 1.0), 4),
        "success_prob": success_prob,
        "arc_failure_prob": arc_failure_prob,
        "source_vertex_failure_prob": source_vertex_failure_prob,
        "target_vertex_failure_prob": target_vertex_failure_prob,
        "expected_transplant_count": expected_transplant_count,
        "recipient_in_degree": recipient_in_degree,
        "donor_out_degree": donor_out_degree,
        "target_scarcity": target_scarcity,
        "donor_flexibility": donor_flexibility,
        "has_reciprocal_edge": has_reciprocal_edge,
        "priority_multiplier": priority_multiplier,
        "donor_age": donor_age,
        "donor_bt": donor_bt,
        "recipient_age": recipient_age,
        "recipient_cpra": cpra,
        "recipient_bt": recipient_bt,
        "label_policy": ADDED_ARC_LABEL_POLICY,
    }
    if "original_node_id" in donor:
        match["winning_donor_id"] = str(donor["original_node_id"])
    match.update(label_fields)
    return match


def add_synthetic_arcs(
    variant_payload: dict[str, Any],
    original_payload: dict[str, Any],
    added_keys: set[str],
    rng: random.Random,
) -> None:
    final_keys = existing_arc_keys(variant_payload) | added_keys
    for key in sorted(added_keys, key=lambda item: (sort_vertex_id(item.split("->", 1)[0]), sort_vertex_id(item.split("->", 1)[1]))):
        src_id, dst_id = split_arc_key(key)
        variant_payload["data"][src_id].setdefault("matches", []).append(
            synthetic_match(
                original_payload=original_payload,
                final_keys=final_keys,
                src_id=src_id,
                dst_id=dst_id,
                rng=rng,
            )
        )


def update_variant_metadata(
    payload: dict[str, Any],
    *,
    base_graph_id: str,
    density_variant: str,
    original_num_arcs: int,
    variant_num_arcs: int,
    perturb_seed: int,
    added_keys: set[str],
    removed_keys: set[str],
) -> None:
    metadata = payload.setdefault("metadata", {})
    metadata["density_sensitivity"] = {
        "base_graph_id": base_graph_id,
        "density_variant": density_variant,
        "original_num_arcs": original_num_arcs,
        "variant_num_arcs": variant_num_arcs,
        "arc_delta": variant_num_arcs - original_num_arcs,
        "added_arc_count": len(added_keys),
        "removed_arc_count": len(removed_keys),
        "added_arc_keys": sorted(added_keys),
        "removed_arc_keys": sorted(removed_keys),
        "perturb_seed": perturb_seed,
        "generation_policy": GENERATION_POLICY,
        "label_policy": LABEL_POLICY,
        "added_arc_source_policy": ADDED_ARC_SOURCE_POLICY,
        "added_arc_label_policy": ADDED_ARC_LABEL_POLICY,
        "removed_arc_policy": REMOVED_ARC_POLICY,
    }


def mean_label_for_keys(payload: dict[str, Any], keys: set[str] | None = None) -> float | str:
    values: list[float] = []
    for src_id, match in iter_arc_items(payload):
        if keys is not None and arc_key(src_id, str(match["recipient"])) not in keys:
            continue
        values.append(parse_float(match.get("ground_truth_label"), float("nan")))
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return ""
    return float(sum(values) / len(values))


def variant_counts(
    original_num_arcs: int,
    density_variant: str,
    fixed_add_count: int,
    fixed_remove_count: int,
) -> tuple[int, int]:
    if density_variant == "original":
        return 0, 0
    if density_variant == "add25pct":
        return int(round(0.25 * original_num_arcs)), 0
    if density_variant == "add25arcs":
        return int(fixed_add_count), 0
    if density_variant == "remove25arcs":
        return 0, int(fixed_remove_count)
    if density_variant == "remove25pct":
        return 0, int(round(0.25 * original_num_arcs))
    raise ValueError(f"Unsupported density variant: {density_variant}")


def variant_arc_delta_type(density_variant: str) -> str:
    return {
        "original": "none",
        "add25pct": "add_fraction",
        "add25arcs": "add_fixed",
        "remove25arcs": "remove_fixed",
        "remove25pct": "remove_fraction",
    }[density_variant]


def generate_variants_for_graph(
    *,
    graph_path: str | Path,
    output_dir: str | Path,
    case: dict[str, str],
    variants: list[str] | tuple[str, ...],
    perturb_seed: int,
    fixed_add_count: int = 25,
    fixed_remove_count: int = 25,
) -> list[dict[str, Any]]:
    graph_path = Path(graph_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    original_payload = read_json(graph_path)
    original_num_arcs = count_arcs(original_payload)
    if fixed_remove_count >= original_num_arcs and "remove25arcs" in variants:
        raise ValueError(
            f"Cannot remove {fixed_remove_count} arcs from {graph_path.name}; "
            f"graph has only {original_num_arcs} arcs"
        )
    if fixed_add_count < 0:
        raise ValueError("fixed_add_count must be non-negative")

    rows: list[dict[str, Any]] = []
    for density_variant in variants:
        add_count, remove_count = variant_counts(
            original_num_arcs,
            density_variant,
            fixed_add_count,
            fixed_remove_count,
        )
        rng = variant_rng(perturb_seed, graph_path.name, density_variant)
        variant_payload = copy.deepcopy(original_payload)

        removed_keys = sample_keys(
            sorted(existing_arc_keys(original_payload), key=lambda item: (sort_vertex_id(item.split("->", 1)[0]), sort_vertex_id(item.split("->", 1)[1]))),
            remove_count,
            rng,
            "existing arcs",
        )
        if removed_keys:
            remove_arcs(variant_payload, removed_keys)

        added_keys = sample_keys(
            all_missing_arc_keys(variant_payload),
            add_count,
            rng,
            "missing arcs",
        )
        if added_keys:
            add_synthetic_arcs(variant_payload, original_payload, added_keys, rng)

        variant_num_arcs = count_arcs(variant_payload)
        update_variant_metadata(
            variant_payload,
            base_graph_id=graph_path.name,
            density_variant=density_variant,
            original_num_arcs=original_num_arcs,
            variant_num_arcs=variant_num_arcs,
            perturb_seed=perturb_seed,
            added_keys=added_keys,
            removed_keys=removed_keys,
        )

        variant_id = f"{graph_path.stem}__{density_variant}__seed{perturb_seed}"
        variant_path = output_dir / f"{variant_id}.json"
        write_json(variant_path, variant_payload)

        rows.append(
            {
                "case_id": case.get("case_id", ""),
                "case_label": case.get("case_label", ""),
                "subset_seed": case.get("subset_seed", ""),
                "base_graph_id": case.get("graph_id", graph_path.name),
                "variant_id": variant_id,
                "variant_graph_path": manifest_graph_path(variant_path),
                "density_variant": density_variant,
                "arc_delta_type": variant_arc_delta_type(density_variant),
                "original_num_arcs": original_num_arcs,
                "variant_num_arcs": variant_num_arcs,
                "arc_delta": variant_num_arcs - original_num_arcs,
                "added_arc_count": len(added_keys),
                "removed_arc_count": len(removed_keys),
                "added_arc_keys": join_keys(added_keys),
                "removed_arc_keys": join_keys(removed_keys),
                "perturb_seed": perturb_seed,
                "generation_policy": GENERATION_POLICY,
                "label_policy": LABEL_POLICY,
                "added_arc_source_policy": ADDED_ARC_SOURCE_POLICY,
                "added_arc_label_policy": ADDED_ARC_LABEL_POLICY,
                "removed_arc_policy": REMOVED_ARC_POLICY,
                "new_arc_label_mean": mean_label_for_keys(variant_payload, added_keys),
                "existing_arc_label_mean": mean_label_for_keys(original_payload),
            }
        )

    return rows


def cases_by_graph(case_index: str | Path, graphs: list[str] | tuple[str, ...]) -> dict[str, dict[str, str]]:
    graph_set = set(graphs)
    output: dict[str, dict[str, str]] = {}
    for row in read_csv_rows(case_index):
        graph_id = row.get("graph_id", "")
        if graph_id in graph_set and graph_id not in output:
            output[graph_id] = row
    missing = [graph for graph in graphs if graph not in output]
    if missing:
        raise ValueError(f"Missing case-index rows for graphs: {', '.join(missing)}")
    return output


def generate_all_variants(args) -> list[dict[str, Any]]:
    graphs_dir = Path(args.output_dir) / "graphs"
    case_map = cases_by_graph(args.case_index, args.graphs)
    rows: list[dict[str, Any]] = []
    for graph in args.graphs:
        graph_path = Path(args.dataset_dir) / graph
        if not graph_path.exists():
            raise FileNotFoundError(f"Missing graph: {graph_path}")
        for perturb_seed in args.perturb_seeds:
            rows.extend(
                generate_variants_for_graph(
                    graph_path=graph_path,
                    output_dir=graphs_dir,
                    case=case_map[graph],
                    variants=args.variants,
                    perturb_seed=perturb_seed,
                    fixed_add_count=args.fixed_add_count,
                    fixed_remove_count=args.fixed_remove_count,
                )
            )
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate arc-density graph variants for decision-analysis case graphs."
    )
    parser.add_argument("--case-index", type=Path, default=DEFAULT_CASE_INDEX)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--graphs", nargs="+", default=list(DEFAULT_GRAPHS))
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--perturb-seed", type=int, default=42)
    parser.add_argument(
        "--perturb-seeds",
        nargs="+",
        type=int,
        help=(
            "Optional list of perturbation seeds. If omitted, the single "
            "--perturb-seed value is used for backward compatibility."
        ),
    )
    parser.add_argument("--fixed-add-count", type=int, default=25)
    parser.add_argument("--fixed-remove-count", type=int, default=25)
    args = parser.parse_args(argv)
    if args.perturb_seeds is None:
        args.perturb_seeds = [args.perturb_seed]
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    rows = generate_all_variants(args)
    manifest_output = args.manifest_output or Path(args.output_dir) / "arc_density_graph_manifest.csv"
    write_manifest(manifest_output, rows)
    print(f"Wrote {len(rows)} variant rows to {manifest_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
