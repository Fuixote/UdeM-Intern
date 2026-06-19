#!/usr/bin/env python3
"""Oracle-only Step3 label-landscape probe for fixed topology banks.

This script does not train a prediction model. It fixes each topology and edge
feature table, resamples Step2c multiplicative label noise over label seeds,
and solves the resulting oracle decision plus a clean-linear proxy decision.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TOPOLOGY_DIR = (
    PROJECT_ROOT / "surrogate_experiment_results" / "Step3" / "pairs20_ndd2" / "data" / "topologies"
)
DEFAULT_PROCESSED_DIR = (
    PROJECT_ROOT / "dataset" / "processed" / "step3_pairs20_ndd2_step2c_poly_d8_mult_eps050_seed20260619"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "surrogate_experiment_results" / "Step3" / "pairs20_ndd2" / "landscape"

DEFAULT_LABEL_SEED_START = 2026061900
DEFAULT_NUM_LABEL_SEEDS = 100
DEFAULT_TOP_K = 5
DEFAULT_STEP2C_DEGREE = 8
DEFAULT_STEP2C_KAPPA = 3.0
DEFAULT_STEP2C_DELTA = 1e-12
DEFAULT_STEP2C_EPSILON_BAR = 0.5
DEFAULT_CLEAN_LINEAR_UTILITY_WEIGHT = 10.0
DEFAULT_CLEAN_LINEAR_CPRA_WEIGHT = 5.0
DEFAULT_GUROBI_SEED = 20260619

SAMPLE_FIELDS = [
    "topology_id",
    "label_seed",
    "num_exchange_candidates",
    "oracle_solution_signature",
    "oracle_solution_size",
    "oracle_objective",
    "num_oracle_top_k_solutions",
    "oracle_top1_top2_margin",
    "oracle_top1_top5_margin",
    "linear_proxy_solution_signature",
    "linear_proxy_solution_size",
    "linear_proxy_objective_under_true",
    "linear_proxy_gap_to_oracle",
    "linear_proxy_normalized_gap_to_oracle",
    "linear_proxy_differs_from_oracle",
    "linear_proxy_jaccard_with_oracle",
]

DESCRIPTOR_FIELDS = [
    "num_vertices",
    "num_pairs",
    "num_ndds",
    "num_arcs",
    "num_exchange_candidates",
    "num_2cycles",
    "num_3cycles",
    "num_cycles_total",
    "num_chains_total",
    "num_chains_len1",
    "num_chains_len2",
    "num_chains_len3",
    "num_chains_len4",
    "candidate_conflict_density",
    "mean_conflict_degree",
    "max_conflict_degree",
    "largest_conflict_component_fraction",
    "fraction_vertices_in_any_candidate",
    "mean_candidates_per_vertex",
    "max_candidates_per_vertex",
]

SUMMARY_FIELDS = [
    "topology_id",
    *DESCRIPTOR_FIELDS,
    "num_label_seeds",
    "num_distinct_oracle_solutions",
    "oracle_solution_entropy",
    "dominant_oracle_solution_fraction",
    "mean_pairwise_oracle_jaccard",
    "mean_oracle_objective",
    "median_oracle_objective",
    "mean_top1_top2_margin",
    "median_top1_top2_margin",
    "mean_top1_top5_margin",
    "median_top1_top5_margin",
    "fraction_linear_proxy_differs_from_oracle",
    "mean_linear_proxy_gap_to_oracle",
    "median_linear_proxy_gap_to_oracle",
    "mean_linear_proxy_normalized_gap_to_oracle",
    "median_linear_proxy_normalized_gap_to_oracle",
    "mean_linear_proxy_jaccard_with_oracle",
]


def int_or_text_key(value: Any) -> tuple[int, int | str]:
    text = str(value)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def topology_sort_key(topology_id: str) -> tuple[int, int | str]:
    return int_or_text_key(str(topology_id).replace("G-", "", 1))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def numeric_or_default(value: Any, default: float = 0.0) -> float:
    if value is None or value == "Unknown":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def deterministic_uniform(source_key: str, low: float, high: float) -> float:
    digest = hashlib.sha256(source_key.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
    rng = random.Random(seed)
    return rng.uniform(low, high)


def clean_linear_label(
    utility: Any,
    cpra: Any,
    utility_weight: float = DEFAULT_CLEAN_LINEAR_UTILITY_WEIGHT,
    cpra_weight: float = DEFAULT_CLEAN_LINEAR_CPRA_WEIGHT,
) -> float:
    utility_norm = clamp(numeric_or_default(utility, 0.0) / 100.0, 0.0, 1.0)
    cpra_value = clamp(numeric_or_default(cpra, 0.0), 0.0, 1.0)
    value = utility_weight * utility_norm + cpra_weight * cpra_value
    return round(max(0.0, value), 4)


def step2c_polynomial_score(
    clean_linear_value: float,
    clean_linear_mean: float,
    degree: int = DEFAULT_STEP2C_DEGREE,
    kappa: float = DEFAULT_STEP2C_KAPPA,
    delta: float = DEFAULT_STEP2C_DELTA,
) -> tuple[float, float]:
    denominator = clean_linear_mean + delta
    q_value = clean_linear_value / denominator if denominator != 0.0 else 0.0
    polynomial_score = (q_value + kappa) ** degree - kappa**degree
    return q_value, polynomial_score


def edge_source_key(row: dict[str, Any], topology_id: str) -> str:
    if row.get("label_source_key"):
        return str(row["label_source_key"])
    return "|".join(
        [
            str(topology_id),
            f"edge_idx={row.get('edge_idx')}",
            str(row.get("source")),
            str(row.get("target")),
            str(row.get("utility")),
        ]
    )


def compute_step2c_labels(
    edge_rows: list[dict[str, Any]],
    label_seed: int,
    topology_id: str,
    degree: int = DEFAULT_STEP2C_DEGREE,
    kappa: float = DEFAULT_STEP2C_KAPPA,
    delta: float = DEFAULT_STEP2C_DELTA,
    epsilon_bar: float = DEFAULT_STEP2C_EPSILON_BAR,
    utility_weight: float = DEFAULT_CLEAN_LINEAR_UTILITY_WEIGHT,
    cpra_weight: float = DEFAULT_CLEAN_LINEAR_CPRA_WEIGHT,
) -> list[float]:
    clean_values = [
        clean_linear_label(
            row.get("utility"),
            row.get("recipient_cpra"),
            utility_weight=utility_weight,
            cpra_weight=cpra_weight,
        )
        for row in edge_rows
    ]
    clean_mean = statistics.mean(clean_values) if clean_values else 0.0
    polynomial_scores = [
        step2c_polynomial_score(value, clean_mean, degree=degree, kappa=kappa, delta=delta)[1]
        for value in clean_values
    ]
    polynomial_score_mean = statistics.mean(polynomial_scores) if polynomial_scores else 0.0

    labels: list[float] = []
    for row, clean_value, polynomial_score in zip(edge_rows, clean_values, polynomial_scores):
        denominator = polynomial_score_mean + delta
        rescaled = clean_mean * polynomial_score / denominator if denominator != 0.0 else 0.0
        polynomial_label = round(max(0.0, rescaled), 4)
        multiplier = deterministic_uniform(
            f"{edge_source_key(row, topology_id)}|step2c_multiplicative_noise|label_seed={label_seed}",
            1.0 - epsilon_bar,
            1.0 + epsilon_bar,
        )
        labels.append(round(max(0.0, polynomial_label * multiplier), 4))
    return labels


def signature_from_edges(edges: set[int] | list[int]) -> str:
    ordered = sorted(int(edge) for edge in edges)
    if not ordered:
        return "EMPTY"
    return "|".join(str(edge) for edge in ordered)


def edge_set_from_signature(signature: str | None) -> set[int]:
    if not signature or signature == "EMPTY":
        return set()
    return {int(part) for part in str(signature).split("|") if part != ""}


def edge_jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def bool_mean(values: list[Any]) -> float | None:
    if not values:
        return None
    return sum(1 for value in values if bool(value)) / len(values)


class CandidatePackingSolver:
    """Solve vertex-disjoint exchange-candidate packing for a fixed topology."""

    def __init__(self, template: dict[str, Any], gurobi_seed: int = DEFAULT_GUROBI_SEED, threads: int = 1):
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError as exc:
            raise RuntimeError("gurobipy is required for the oracle-only landscape probe") from exc

        self.gp = gp
        self.GRB = GRB
        self.template = template
        self.candidates = list(template.get("feasible_candidates", []))
        self.candidate_edge_sets = [
            {int(edge_idx) for edge_idx in candidate.get("edges", [])}
            for candidate in self.candidates
        ]
        self.model = gp.Model(f"step3_landscape_{template.get('topology_id', 'topology')}")
        self.model.Params.OutputFlag = 0
        self.model.Params.Seed = int(gurobi_seed)
        self.model.Params.Threads = max(1, int(threads))
        self.variables = [
            self.model.addVar(vtype=GRB.BINARY, name=f"cand_{idx}")
            for idx in range(len(self.candidates))
        ]

        vertex_to_candidates: dict[str, list[int]] = {}
        for idx, candidate in enumerate(self.candidates):
            for node in candidate.get("nodes", []):
                vertex_to_candidates.setdefault(str(node), []).append(idx)
        for vertex, candidate_indices in vertex_to_candidates.items():
            self.model.addConstr(
                gp.quicksum(self.variables[idx] for idx in candidate_indices) <= 1,
                name=f"vertex_disjoint_{vertex}",
            )
        self.model.ModelSense = GRB.MAXIMIZE
        self.model.update()

    def candidate_weights(self, edge_weights: list[float]) -> list[float]:
        return [
            sum(float(edge_weights[edge_idx]) for edge_idx in edge_set)
            for edge_set in self.candidate_edge_sets
        ]

    def solve_top_k(self, edge_weights: list[float], top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        top_k = max(1, int(top_k))
        candidate_weights = self.candidate_weights(edge_weights)
        for variable, weight in zip(self.variables, candidate_weights):
            variable.Obj = float(weight)

        self.model.Params.PoolSolutions = top_k
        self.model.Params.PoolSearchMode = 2 if top_k > 1 else 0
        self.model.optimize()
        if self.model.Status not in {self.GRB.OPTIMAL, self.GRB.SUBOPTIMAL} or self.model.SolCount == 0:
            return []

        solutions: list[dict[str, Any]] = []
        seen_signatures: set[str] = set()
        for solution_number in range(min(self.model.SolCount, top_k)):
            self.model.Params.SolutionNumber = solution_number
            selected_candidates = [
                idx for idx, variable in enumerate(self.variables) if variable.Xn > 0.5
            ]
            selected_edges: set[int] = set()
            for candidate_idx in selected_candidates:
                selected_edges.update(self.candidate_edge_sets[candidate_idx])
            signature = signature_from_edges(selected_edges)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            objective = sum(float(edge_weights[edge_idx]) for edge_idx in selected_edges)
            solutions.append(
                {
                    "selected_candidates": selected_candidates,
                    "selected_edges": sorted(selected_edges),
                    "edge_set": selected_edges,
                    "edge_signature": signature,
                    "solution_size": len(selected_edges),
                    "objective": objective,
                    "pool_objective": float(self.model.PoolObjVal),
                }
            )
        return solutions

    def dispose(self) -> None:
        self.model.dispose()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def graph_data(graph_json: dict[str, Any]) -> dict[str, Any]:
    data = graph_json.get("data", {})
    if not isinstance(data, dict):
        raise ValueError("processed graph data must be an object")
    return data


def build_label_source_key(raw_batch_name: str | None, original_file: str | None, source: str, target: str, utility: Any) -> str | None:
    if not raw_batch_name or not original_file:
        return None
    return "|".join([str(raw_batch_name), str(original_file), str(source), str(target), str(utility)])


def edge_feature_rows_from_processed_graph(
    template: dict[str, Any],
    processed_graph: dict[str, Any],
    raw_batch_name: str | None = None,
) -> list[dict[str, Any]]:
    data = graph_data(processed_graph)
    metadata = processed_graph.get("metadata", {})
    original_file = metadata.get("original_file")
    rows: list[dict[str, Any]] = []

    by_source_target: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for source, payload in data.items():
        for position, match in enumerate(payload.get("matches", []) or []):
            target = str(match.get("recipient"))
            record = {"position": position, "match": match}
            by_source_target.setdefault((str(source), target), []).append(record)

    for arc in sorted(template.get("arcs", []), key=lambda item: int(item["edge_idx"])):
        source = str(arc["source"])
        target = str(arc["target"])
        matches = data.get(source, {}).get("matches", []) or []
        match = None
        source_position = arc.get("source_match_position")
        if source_position is not None and int(source_position) < len(matches):
            candidate = matches[int(source_position)]
            if str(candidate.get("recipient")) == target:
                match = candidate
        if match is None:
            keyed_matches = by_source_target.get((source, target), [])
            if len(keyed_matches) == 1:
                match = keyed_matches[0]["match"]
            elif keyed_matches:
                match = keyed_matches[0]["match"]
        if match is None:
            raise ValueError(
                f"{template.get('topology_id', 'topology')}: missing processed match for arc {source}->{target}"
            )
        utility = match.get("utility", 0.0)
        rows.append(
            {
                "edge_idx": int(arc["edge_idx"]),
                "source": source,
                "target": target,
                "utility": utility,
                "recipient_cpra": match.get("recipient_cpra"),
                "ground_truth_label": match.get("ground_truth_label"),
                "latent_clean_linear_label": match.get("latent_clean_linear_label"),
                "label_source_key": build_label_source_key(raw_batch_name, original_file, source, target, utility),
            }
        )
    return rows


def read_topology_descriptors(topology_dir: Path) -> dict[str, dict[str, Any]]:
    bank_path = topology_dir / "topology_bank.csv"
    if not bank_path.exists():
        return {}
    with bank_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {
            row["topology_id"]: {
                field: coerce_csv_value(row.get(field, ""))
                for field in DESCRIPTOR_FIELDS
                if field in row
            }
            for row in reader
        }


def coerce_csv_value(value: str) -> Any:
    if value == "":
        return ""
    try:
        if all(marker not in value for marker in [".", "e", "E"]):
            return int(value)
        return float(value)
    except (TypeError, ValueError):
        return value


def topology_template_paths(
    topology_dir: Path,
    topology_ids: set[str] | None = None,
    max_topologies: int | None = None,
) -> list[Path]:
    paths = sorted(
        topology_dir.glob("G-*/template.json"),
        key=lambda path: topology_sort_key(path.parent.name),
    )
    if topology_ids is not None:
        paths = [path for path in paths if path.parent.name in topology_ids]
    if max_topologies is not None:
        paths = paths[:max_topologies]
    return paths


def true_objective(edge_weights: list[float], edge_set: set[int]) -> float:
    return sum(float(edge_weights[edge_idx]) for edge_idx in edge_set)


def margin(top_solutions: list[dict[str, Any]], rank: int) -> float | None:
    if len(top_solutions) <= rank:
        return None
    return float(top_solutions[0]["objective"]) - float(top_solutions[rank]["objective"])


def probe_one_topology(
    template_path: Path,
    processed_dir: Path,
    label_seeds: list[int],
    topology_descriptor: dict[str, Any],
    raw_batch_name: str | None,
    top_k: int,
    step2c_degree: int,
    step2c_kappa: float,
    step2c_delta: float,
    step2c_epsilon_bar: float,
    clean_linear_utility_weight: float,
    clean_linear_cpra_weight: float,
    gurobi_seed: int,
    threads: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    template = load_json(template_path)
    topology_id = str(template.get("topology_id") or template_path.parent.name)
    processed_path = processed_dir / f"{topology_id}.json"
    if not processed_path.exists():
        raise FileNotFoundError(f"Missing processed graph for {topology_id}: {processed_path}")
    processed_graph = load_json(processed_path)
    edge_rows = edge_feature_rows_from_processed_graph(template, processed_graph, raw_batch_name=raw_batch_name)

    clean_weights = [
        clean_linear_label(
            row.get("utility"),
            row.get("recipient_cpra"),
            utility_weight=clean_linear_utility_weight,
            cpra_weight=clean_linear_cpra_weight,
        )
        for row in edge_rows
    ]

    solver = CandidatePackingSolver(template, gurobi_seed=gurobi_seed, threads=threads)
    try:
        proxy_solutions = solver.solve_top_k(clean_weights, top_k=1)
        proxy_solution = proxy_solutions[0] if proxy_solutions else {
            "edge_signature": "EMPTY",
            "edge_set": set(),
            "solution_size": 0,
        }

        sample_rows: list[dict[str, Any]] = []
        for label_seed in label_seeds:
            true_weights = compute_step2c_labels(
                edge_rows,
                label_seed=label_seed,
                topology_id=topology_id,
                degree=step2c_degree,
                kappa=step2c_kappa,
                delta=step2c_delta,
                epsilon_bar=step2c_epsilon_bar,
                utility_weight=clean_linear_utility_weight,
                cpra_weight=clean_linear_cpra_weight,
            )
            oracle_solutions = solver.solve_top_k(true_weights, top_k=top_k)
            if not oracle_solutions:
                raise RuntimeError(f"{topology_id}: oracle solver returned no solution for label_seed={label_seed}")
            oracle_solution = oracle_solutions[0]
            oracle_objective = float(oracle_solution["objective"])
            proxy_edge_set = set(proxy_solution["edge_set"])
            oracle_edge_set = set(oracle_solution["edge_set"])
            proxy_objective_under_true = true_objective(true_weights, proxy_edge_set)
            proxy_gap = oracle_objective - proxy_objective_under_true
            if -1e-8 < proxy_gap < 0.0:
                proxy_gap = 0.0
            normalized_gap = proxy_gap / abs(oracle_objective) if abs(oracle_objective) > 1e-12 else None

            sample_rows.append(
                {
                    "topology_id": topology_id,
                    "label_seed": label_seed,
                    "num_exchange_candidates": len(template.get("feasible_candidates", [])),
                    "oracle_solution_signature": oracle_solution["edge_signature"],
                    "oracle_solution_size": oracle_solution["solution_size"],
                    "oracle_objective": oracle_objective,
                    "num_oracle_top_k_solutions": len(oracle_solutions),
                    "oracle_top1_top2_margin": margin(oracle_solutions, 1),
                    "oracle_top1_top5_margin": margin(oracle_solutions, 4),
                    "linear_proxy_solution_signature": proxy_solution["edge_signature"],
                    "linear_proxy_solution_size": proxy_solution["solution_size"],
                    "linear_proxy_objective_under_true": proxy_objective_under_true,
                    "linear_proxy_gap_to_oracle": proxy_gap,
                    "linear_proxy_normalized_gap_to_oracle": normalized_gap,
                    "linear_proxy_differs_from_oracle": proxy_solution["edge_signature"] != oracle_solution["edge_signature"],
                    "linear_proxy_jaccard_with_oracle": edge_jaccard(proxy_edge_set, oracle_edge_set),
                }
            )
    finally:
        solver.dispose()

    return sample_rows, summarize_topology_samples(topology_id, sample_rows, topology_descriptor=topology_descriptor)


def summarize_topology_samples(
    topology_id: str,
    sample_rows: list[dict[str, Any]],
    topology_descriptor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    topology_descriptor = topology_descriptor or {}
    signatures = [str(row["oracle_solution_signature"]) for row in sample_rows]
    signature_counts = Counter(signatures)
    num_samples = len(sample_rows)
    entropy = 0.0
    for count in signature_counts.values():
        probability = count / num_samples if num_samples else 0.0
        if probability > 0.0:
            entropy -= probability * math.log(probability)

    edge_sets = [edge_set_from_signature(signature) for signature in signatures]
    pairwise_jaccards: list[float] = []
    for left_idx in range(len(edge_sets)):
        for right_idx in range(left_idx + 1, len(edge_sets)):
            pairwise_jaccards.append(edge_jaccard(edge_sets[left_idx], edge_sets[right_idx]))

    objectives = [float(row["oracle_objective"]) for row in sample_rows]
    top1_top2 = [
        float(row["oracle_top1_top2_margin"])
        for row in sample_rows
        if row.get("oracle_top1_top2_margin") is not None
    ]
    top1_top5 = [
        float(row["oracle_top1_top5_margin"])
        for row in sample_rows
        if row.get("oracle_top1_top5_margin") is not None
    ]
    proxy_gaps = [float(row["linear_proxy_gap_to_oracle"]) for row in sample_rows]
    normalized_proxy_gaps = [
        float(row["linear_proxy_normalized_gap_to_oracle"])
        for row in sample_rows
        if row.get("linear_proxy_normalized_gap_to_oracle") is not None
    ]
    proxy_jaccards = [float(row["linear_proxy_jaccard_with_oracle"]) for row in sample_rows]

    summary = {
        "topology_id": topology_id,
        "num_label_seeds": num_samples,
        "num_distinct_oracle_solutions": len(signature_counts),
        "oracle_solution_entropy": entropy,
        "dominant_oracle_solution_fraction": (
            max(signature_counts.values()) / num_samples if num_samples and signature_counts else None
        ),
        "mean_pairwise_oracle_jaccard": mean(pairwise_jaccards),
        "mean_oracle_objective": mean(objectives),
        "median_oracle_objective": median(objectives),
        "mean_top1_top2_margin": mean(top1_top2),
        "median_top1_top2_margin": median(top1_top2),
        "mean_top1_top5_margin": mean(top1_top5),
        "median_top1_top5_margin": median(top1_top5),
        "fraction_linear_proxy_differs_from_oracle": bool_mean(
            [row["linear_proxy_differs_from_oracle"] for row in sample_rows]
        ),
        "mean_linear_proxy_gap_to_oracle": mean(proxy_gaps),
        "median_linear_proxy_gap_to_oracle": median(proxy_gaps),
        "mean_linear_proxy_normalized_gap_to_oracle": mean(normalized_proxy_gaps),
        "median_linear_proxy_normalized_gap_to_oracle": median(normalized_proxy_gaps),
        "mean_linear_proxy_jaccard_with_oracle": mean(proxy_jaccards),
    }
    for field in DESCRIPTOR_FIELDS:
        if field in topology_descriptor:
            summary[field] = topology_descriptor[field]
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def prepare_output_dir(output_dir: Path, force: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            raise FileExistsError(f"{output_dir} already exists and is not empty; pass --force to overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def infer_raw_batch_name(topology_dir: Path) -> str | None:
    output_root = topology_dir.parent.parent
    manifest_path = output_root / "run_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = load_json(manifest_path)
    except Exception:
        return None
    raw_batch_dir = manifest.get("config", {}).get("raw_batch_dir")
    if raw_batch_dir:
        return Path(raw_batch_dir).name
    process_cmd = manifest.get("commands", {}).get("process", [])
    if isinstance(process_cmd, list) and len(process_cmd) >= 3:
        return Path(process_cmd[2]).name
    return None


def aggregate_probe_summary(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    def numeric_field(field: str) -> list[float]:
        values = []
        for row in summary_rows:
            value = row.get(field)
            if value is not None and value != "":
                values.append(float(value))
        return values

    return {
        "num_topologies": len(summary_rows),
        "num_samples": sum(int(row.get("num_label_seeds", 0)) for row in summary_rows),
        "mean_distinct_oracle_solutions": mean(numeric_field("num_distinct_oracle_solutions")),
        "median_distinct_oracle_solutions": median(numeric_field("num_distinct_oracle_solutions")),
        "mean_oracle_solution_entropy": mean(numeric_field("oracle_solution_entropy")),
        "mean_dominant_oracle_solution_fraction": mean(numeric_field("dominant_oracle_solution_fraction")),
        "mean_fraction_linear_proxy_differs_from_oracle": mean(
            numeric_field("fraction_linear_proxy_differs_from_oracle")
        ),
        "mean_linear_proxy_gap_to_oracle": mean(numeric_field("mean_linear_proxy_gap_to_oracle")),
        "mean_linear_proxy_normalized_gap_to_oracle": mean(
            numeric_field("mean_linear_proxy_normalized_gap_to_oracle")
        ),
    }


def parse_topology_ids(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology-dir", type=Path, default=DEFAULT_TOPOLOGY_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label-seed-start", type=int, default=DEFAULT_LABEL_SEED_START)
    parser.add_argument("--num-label-seeds", type=int, default=DEFAULT_NUM_LABEL_SEEDS)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--max-topologies", type=int, default=None)
    parser.add_argument("--topology-ids", type=str, default=None, help="Comma-separated topology ids, e.g. G-14,G-982")
    parser.add_argument("--raw-batch-name", type=str, default=None)
    parser.add_argument("--gurobi-seed", type=int, default=DEFAULT_GUROBI_SEED)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--force", action="store_true")

    parser.add_argument("--step2c-degree", type=int, default=DEFAULT_STEP2C_DEGREE)
    parser.add_argument("--step2c-kappa", type=float, default=DEFAULT_STEP2C_KAPPA)
    parser.add_argument("--step2c-delta", type=float, default=DEFAULT_STEP2C_DELTA)
    parser.add_argument("--step2c-epsilon-bar", type=float, default=DEFAULT_STEP2C_EPSILON_BAR)
    parser.add_argument("--clean-linear-utility-weight", type=float, default=DEFAULT_CLEAN_LINEAR_UTILITY_WEIGHT)
    parser.add_argument("--clean-linear-cpra-weight", type=float, default=DEFAULT_CLEAN_LINEAR_CPRA_WEIGHT)

    args = parser.parse_args(argv)
    if args.num_label_seeds <= 0:
        raise ValueError("--num-label-seeds must be positive")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.max_topologies is not None and args.max_topologies <= 0:
        raise ValueError("--max-topologies must be positive when provided")
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    topology_ids = parse_topology_ids(args.topology_ids)
    template_paths = topology_template_paths(
        args.topology_dir,
        topology_ids=topology_ids,
        max_topologies=args.max_topologies,
    )
    if not template_paths:
        raise ValueError(f"No topology templates found in {args.topology_dir}")

    prepare_output_dir(args.output_dir, force=args.force)
    raw_batch_name = args.raw_batch_name or infer_raw_batch_name(args.topology_dir)
    topology_descriptors = read_topology_descriptors(args.topology_dir)
    label_seeds = list(range(args.label_seed_start, args.label_seed_start + args.num_label_seeds))

    config = {
        "topology_dir": str(args.topology_dir),
        "processed_dir": str(args.processed_dir),
        "output_dir": str(args.output_dir),
        "num_topologies": len(template_paths),
        "label_seed_start": args.label_seed_start,
        "num_label_seeds": args.num_label_seeds,
        "top_k": args.top_k,
        "raw_batch_name": raw_batch_name,
        "step2c_degree": args.step2c_degree,
        "step2c_kappa": args.step2c_kappa,
        "step2c_delta": args.step2c_delta,
        "step2c_epsilon_bar": args.step2c_epsilon_bar,
        "clean_linear_utility_weight": args.clean_linear_utility_weight,
        "clean_linear_cpra_weight": args.clean_linear_cpra_weight,
        "gurobi_seed": args.gurobi_seed,
        "threads": args.threads,
        "probe_boundary": (
            "fixed topology and fixed edge utility/cpra; only Step2c multiplicative "
            "label noise is resampled over label seeds; no model training"
        ),
    }
    (args.output_dir / "probe_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    sample_rows_all: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for index, template_path in enumerate(template_paths, start=1):
        topology_id = template_path.parent.name
        sample_rows, summary_row = probe_one_topology(
            template_path,
            processed_dir=args.processed_dir,
            label_seeds=label_seeds,
            topology_descriptor=topology_descriptors.get(topology_id, {}),
            raw_batch_name=raw_batch_name,
            top_k=args.top_k,
            step2c_degree=args.step2c_degree,
            step2c_kappa=args.step2c_kappa,
            step2c_delta=args.step2c_delta,
            step2c_epsilon_bar=args.step2c_epsilon_bar,
            clean_linear_utility_weight=args.clean_linear_utility_weight,
            clean_linear_cpra_weight=args.clean_linear_cpra_weight,
            gurobi_seed=args.gurobi_seed,
            threads=args.threads,
        )
        sample_rows_all.extend(sample_rows)
        summary_rows.append(summary_row)
        if index == 1 or index % args.progress_every == 0 or index == len(template_paths):
            print(f"probed {index}/{len(template_paths)} topologies", flush=True)

    sample_path = args.output_dir / "label_seed_samples.csv"
    summary_path = args.output_dir / "topology_landscape_summary.csv"
    write_csv(sample_path, sample_rows_all, SAMPLE_FIELDS)
    write_csv(summary_path, summary_rows, SUMMARY_FIELDS)
    aggregate = aggregate_probe_summary(summary_rows)
    (args.output_dir / "probe_summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(aggregate, indent=2, sort_keys=True))
    print(f"wrote {sample_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
