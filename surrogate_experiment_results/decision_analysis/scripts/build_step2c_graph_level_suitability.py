#!/usr/bin/env python3
"""Build Phase 1 graph-level DFL suitability features for Step2c."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUDIT_DIR = PROJECT_ROOT / "surrogate_experiment_results" / "Step2c Graph-Level DFL Suitability Audit"
DEFAULT_REGIME = "step2c_poly_d8_mult_eps050"
DEFAULT_DATASET_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "step2c_poly_d8_mult_eps050_main2000_seed20260523"
)
DEFAULT_OUTCOME_INPUT = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "decision_analysis"
    / "results"
    / "all400_model_seed_baseline"
    / "step2c_all400_all50_graph_summary.csv"
)
DEFAULT_RESULTS_DIR = AUDIT_DIR / "results"
DEFAULT_PRESENTATION_DIR = AUDIT_DIR / "presentation"
DEFAULT_FEATURE_OUTPUT = DEFAULT_RESULTS_DIR / "step2c_all400_graph_features.csv"
DEFAULT_JOIN_OUTPUT = DEFAULT_RESULTS_DIR / "step2c_all400_graph_feature_outcome_table.csv"
DEFAULT_ASSOC_OUTPUT = DEFAULT_RESULTS_DIR / "step2c_feature_family_association.csv"
DEFAULT_OVERLAY_OUTPUT = DEFAULT_RESULTS_DIR / "step2c_selected_case_feature_overlay.csv"
DEFAULT_STORY_OUTPUT = DEFAULT_PRESENTATION_DIR / "step2c_dfl_suitability_story.md"

SELECTED_GRAPHS = (
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

FEATURE_FAMILIES = {
    "raw_topology": (
        "num_vertices",
        "num_arcs",
        "density",
        "mean_in_degree",
        "mean_out_degree",
        "max_in_degree",
        "max_out_degree",
        "in_degree_gini",
        "out_degree_gini",
        "reciprocal_arc_count",
        "reciprocity_rate",
        "num_weak_components",
        "num_strong_components",
        "largest_scc_fraction",
        "scc_entropy",
        "num_sources",
        "num_sinks",
    ),
    "cycle_chain": (
        "num_2cycles",
        "num_3cycles",
        "num_cycles_total",
        "num_chain_len1",
        "num_chain_len2",
        "num_chain_len3",
        "num_chain_len4",
        "num_chains_total",
        "cycle_to_chain_ratio",
    ),
    "exchange_geometry": (
        "num_exchange_candidates",
        "exchange_size_mean",
        "exchange_size_std",
        "exchange_size_entropy",
        "vertex_exchange_participation_mean",
        "vertex_exchange_participation_gini",
        "max_vertex_exchange_participation",
        "fraction_vertices_in_any_exchange",
        "fraction_vertices_in_many_exchanges",
        "feasible_set_richness_score",
    ),
    "conflict_geometry": (
        "conflict_graph_num_nodes",
        "conflict_graph_num_edges",
        "conflict_graph_density",
        "conflict_degree_mean",
        "conflict_degree_gini",
        "conflict_components",
        "largest_conflict_component_fraction",
    ),
}

FEATURE_KEYS = tuple(feature for features in FEATURE_FAMILIES.values() for feature in features)

GRAPH_FEATURE_FIELDS = [
    "regime",
    "graph_id",
    "max_cycle",
    "max_chain",
    *FEATURE_KEYS,
]

OUTCOME_FIELDS = [
    "median_delta_pp",
    "strict_case_c_rate",
    "meaningful_spo_benefit_rate",
    "median_two_stage_rank1_gap_pct",
    "median_spoplus_rank1_gap_pct",
    "topk_promotion_rate",
    "exact_rank2_promotion_rate",
    "correction_rate",
]

JOIN_FIELDS = [
    *GRAPH_FEATURE_FIELDS,
    *OUTCOME_FIELDS,
    "helpful_graph",
    "extreme_helpful_graph",
    "harmful_graph",
    "neutral_graph",
]

ASSOCIATION_FIELDS = [
    "feature_family",
    "feature",
    "n_graphs",
    "spearman_median_delta_pp",
    "auroc_helpful",
    "auroc_harmful",
]


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_float(value: Any, default: float = float("nan")) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def finite_values(values: list[Any]) -> list[float]:
    return [value for value in (parse_float(item) for item in values) if math.isfinite(value)]


def finite_mean(values: list[Any]) -> float:
    clean = finite_values(values)
    return float(sum(clean) / len(clean)) if clean else float("nan")


def finite_std(values: list[Any]) -> float:
    clean = finite_values(values)
    if len(clean) <= 1:
        return 0.0
    mean = finite_mean(clean)
    return float(math.sqrt(sum((value - mean) ** 2 for value in clean) / len(clean)))


def gini(values: list[Any]) -> float:
    clean = sorted(finite_values(values))
    if not clean:
        return 0.0
    total = sum(clean)
    if abs(total) < 1e-12:
        return 0.0
    n = len(clean)
    weighted = sum((idx + 1) * value for idx, value in enumerate(clean))
    return float((2.0 * weighted) / (n * total) - (n + 1.0) / n)


def entropy_from_counts(counts: list[int | float]) -> float:
    clean = [float(value) for value in counts if float(value) > 0]
    total = sum(clean)
    if total <= 0:
        return 0.0
    return float(-sum((value / total) * math.log(value / total) for value in clean))


def graph_edges(graph_json: dict[str, Any]) -> list[tuple[str, str]]:
    edges: list[tuple[str, str]] = []
    for source, payload in graph_json.get("data", {}).items():
        for match in payload.get("matches", []) or []:
            recipient = match.get("recipient")
            if recipient is not None:
                edges.append((str(source), str(recipient)))
    return edges


def graph_vertices(graph_json: dict[str, Any], edges: list[tuple[str, str]]) -> set[str]:
    vertices = {str(vertex) for vertex in graph_json.get("data", {}).keys()}
    for source, target in edges:
        vertices.add(source)
        vertices.add(target)
    metadata_vertices = graph_json.get("metadata", {}).get("total_vertices")
    if metadata_vertices is not None:
        vertices.update(str(idx) for idx in range(int(metadata_vertices)))
    return vertices


def vertex_types(graph_json: dict[str, Any]) -> dict[str, str]:
    return {
        str(vertex): str(payload.get("type", "Pair"))
        for vertex, payload in graph_json.get("data", {}).items()
    }


def adjacency_from_edges(edges: list[tuple[str, str]]) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for source, target in edges:
        adjacency[source].add(target)
    return adjacency


def weak_components(vertices: set[str], edges: list[tuple[str, str]]) -> list[set[str]]:
    undirected: dict[str, set[str]] = {vertex: set() for vertex in vertices}
    for source, target in edges:
        undirected.setdefault(source, set()).add(target)
        undirected.setdefault(target, set()).add(source)
    seen: set[str] = set()
    components: list[set[str]] = []
    for vertex in sorted(vertices):
        if vertex in seen:
            continue
        stack = [vertex]
        component: set[str] = set()
        seen.add(vertex)
        while stack:
            current = stack.pop()
            component.add(current)
            for neighbor in undirected.get(current, set()):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components


def strongly_connected_components(vertices: set[str], edges: list[tuple[str, str]]) -> list[list[str]]:
    adjacency = adjacency_from_edges(edges)
    index = 0
    indices: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[list[str]] = []

    def strongconnect(vertex: str) -> None:
        nonlocal index
        indices[vertex] = index
        lowlink[vertex] = index
        index += 1
        stack.append(vertex)
        on_stack.add(vertex)

        for target in adjacency.get(vertex, set()):
            if target not in indices:
                strongconnect(target)
                lowlink[vertex] = min(lowlink[vertex], lowlink[target])
            elif target in on_stack:
                lowlink[vertex] = min(lowlink[vertex], indices[target])

        if lowlink[vertex] == indices[vertex]:
            component: list[str] = []
            while True:
                item = stack.pop()
                on_stack.remove(item)
                component.append(item)
                if item == vertex:
                    break
            components.append(component)

    for vertex in sorted(vertices):
        if vertex not in indices:
            strongconnect(vertex)
    return components


def count_directed_3cycles(edges: list[tuple[str, str]], pair_vertices: set[str]) -> int:
    edge_set = set(edges)
    count = 0
    vertices = sorted(pair_vertices)
    for i, a in enumerate(vertices):
        for b in vertices[i + 1 :]:
            for c in vertices:
                if c <= b or c == a:
                    continue
                triples = ((a, b, c), (a, c, b))
                for x, y, z in triples:
                    if (x, y) in edge_set and (y, z) in edge_set and (z, x) in edge_set:
                        count += 1
    return count


def enumerate_cycle_candidates(
    edges: list[tuple[str, str]],
    pair_vertices: set[str],
    max_cycle: int,
) -> list[tuple[str, ...]]:
    edge_set = set(edges)
    cycles: set[tuple[str, ...]] = set()
    if max_cycle >= 2:
        for source, target in edge_set:
            if source < target and source in pair_vertices and target in pair_vertices:
                if (target, source) in edge_set:
                    cycles.add(tuple(sorted((source, target), key=int_or_text_key)))
    if max_cycle >= 3:
        vertices = sorted(pair_vertices, key=int_or_text_key)
        for a in vertices:
            for b in vertices:
                if b == a or (a, b) not in edge_set:
                    continue
                for c in vertices:
                    if c in {a, b}:
                        continue
                    if (b, c) in edge_set and (c, a) in edge_set:
                        cycles.add(canonical_cycle((a, b, c)))
    return sorted(cycles, key=lambda item: tuple(int_or_text_key(part) for part in item))


def int_or_text_key(value: str) -> tuple[int, Any]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def canonical_cycle(cycle: tuple[str, ...]) -> tuple[str, ...]:
    rotations = [cycle[idx:] + cycle[:idx] for idx in range(len(cycle))]
    return min(rotations, key=lambda item: tuple(int_or_text_key(part) for part in item))


def enumerate_chain_candidates(
    edges: list[tuple[str, str]],
    types: dict[str, str],
    max_chain: int,
) -> list[tuple[str, ...]]:
    adjacency = adjacency_from_edges(edges)
    ndds = sorted([vertex for vertex, kind in types.items() if kind == "NDD"], key=int_or_text_key)
    chains: list[tuple[str, ...]] = []
    for ndd in ndds:
        queue: deque[tuple[str, ...]] = deque([(ndd,)])
        while queue:
            path = queue.popleft()
            if len(path) - 1 >= max_chain:
                continue
            for target in sorted(adjacency.get(path[-1], set()), key=int_or_text_key):
                if target in path:
                    continue
                next_path = path + (target,)
                chains.append(next_path)
                queue.append(next_path)
    return chains


def conflict_graph_features(exchanges: list[tuple[str, ...]]) -> dict[str, Any]:
    n = len(exchanges)
    exchange_sets = [set(exchange) for exchange in exchanges]
    conflict_degrees = [0 for _ in exchanges]
    conflict_edges = 0
    adjacency: list[set[int]] = [set() for _ in exchanges]
    for i in range(n):
        for j in range(i + 1, n):
            if exchange_sets[i] & exchange_sets[j]:
                conflict_edges += 1
                conflict_degrees[i] += 1
                conflict_degrees[j] += 1
                adjacency[i].add(j)
                adjacency[j].add(i)

    seen: set[int] = set()
    component_sizes: list[int] = []
    for idx in range(n):
        if idx in seen:
            continue
        stack = [idx]
        seen.add(idx)
        size = 0
        while stack:
            current = stack.pop()
            size += 1
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        component_sizes.append(size)

    possible = n * (n - 1) / 2
    return {
        "conflict_graph_num_nodes": n,
        "conflict_graph_num_edges": conflict_edges,
        "conflict_graph_density": 0.0 if possible == 0 else float(conflict_edges / possible),
        "conflict_degree_mean": finite_mean(conflict_degrees),
        "conflict_degree_gini": gini(conflict_degrees),
        "conflict_components": len(component_sizes),
        "largest_conflict_component_fraction": (
            0.0 if n == 0 else float(max(component_sizes, default=0) / n)
        ),
    }


def compute_graph_feature_row(
    *,
    graph_id: str,
    graph_json: dict[str, Any],
    regime: str = DEFAULT_REGIME,
    max_cycle: int = 3,
    max_chain: int = 4,
) -> dict[str, Any]:
    edges = graph_edges(graph_json)
    vertices = graph_vertices(graph_json, edges)
    types = vertex_types(graph_json)
    pair_vertices = {vertex for vertex in vertices if types.get(vertex, "Pair") != "NDD"}
    edge_set = set(edges)
    adjacency = adjacency_from_edges(edges)

    in_degree = {vertex: 0 for vertex in vertices}
    out_degree = {vertex: 0 for vertex in vertices}
    for source, target in edges:
        out_degree[source] = out_degree.get(source, 0) + 1
        in_degree[target] = in_degree.get(target, 0) + 1
        in_degree.setdefault(source, in_degree.get(source, 0))
        out_degree.setdefault(target, out_degree.get(target, 0))

    num_vertices = len(vertices)
    num_arcs = len(edges)
    density = 0.0 if num_vertices <= 1 else float(num_arcs / (num_vertices * (num_vertices - 1)))
    reciprocal_arc_count = sum(1 for source, target in edges if (target, source) in edge_set)
    weak = weak_components(vertices, edges)
    strong = strongly_connected_components(vertices, edges)
    largest_scc = max((len(component) for component in strong), default=0)
    component_sizes = [len(component) for component in strong]

    cycles = enumerate_cycle_candidates(edges, pair_vertices, max_cycle=max_cycle)
    chains = enumerate_chain_candidates(edges, types, max_chain=max_chain)
    chain_length_counts = Counter(len(chain) - 1 for chain in chains)
    exchanges = [*cycles, *chains]
    exchange_sizes = [len(exchange) for exchange in exchanges]
    participation = Counter(vertex for exchange in exchanges for vertex in set(exchange))
    participation_values = [participation.get(vertex, 0) for vertex in vertices]

    row = {
        "regime": regime,
        "graph_id": graph_id,
        "max_cycle": max_cycle,
        "max_chain": max_chain,
        "num_vertices": num_vertices,
        "num_arcs": num_arcs,
        "density": density,
        "mean_in_degree": finite_mean(list(in_degree.values())),
        "mean_out_degree": finite_mean(list(out_degree.values())),
        "max_in_degree": max(in_degree.values(), default=0),
        "max_out_degree": max(out_degree.values(), default=0),
        "in_degree_gini": gini(list(in_degree.values())),
        "out_degree_gini": gini(list(out_degree.values())),
        "reciprocal_arc_count": reciprocal_arc_count,
        "reciprocity_rate": 0.0 if num_arcs == 0 else float(reciprocal_arc_count / num_arcs),
        "num_weak_components": len(weak),
        "num_strong_components": len(strong),
        "largest_scc_fraction": 0.0 if num_vertices == 0 else float(largest_scc / num_vertices),
        "scc_entropy": entropy_from_counts(component_sizes),
        "num_sources": sum(1 for vertex in vertices if in_degree.get(vertex, 0) == 0),
        "num_sinks": sum(1 for vertex in vertices if out_degree.get(vertex, 0) == 0),
        "num_2cycles": sum(1 for cycle in cycles if len(cycle) == 2),
        "num_3cycles": sum(1 for cycle in cycles if len(cycle) == 3),
        "num_cycles_total": len(cycles),
        "num_chain_len1": chain_length_counts.get(1, 0),
        "num_chain_len2": chain_length_counts.get(2, 0),
        "num_chain_len3": chain_length_counts.get(3, 0),
        "num_chain_len4": chain_length_counts.get(4, 0),
        "num_chains_total": len(chains),
        "cycle_to_chain_ratio": float(len(cycles) / max(1, len(chains))),
        "num_exchange_candidates": len(exchanges),
        "exchange_size_mean": finite_mean(exchange_sizes),
        "exchange_size_std": finite_std(exchange_sizes),
        "exchange_size_entropy": entropy_from_counts(list(Counter(exchange_sizes).values())),
        "vertex_exchange_participation_mean": finite_mean(participation_values),
        "vertex_exchange_participation_gini": gini(participation_values),
        "max_vertex_exchange_participation": max(participation_values, default=0),
        "fraction_vertices_in_any_exchange": (
            0.0 if num_vertices == 0 else sum(value > 0 for value in participation_values) / num_vertices
        ),
        "fraction_vertices_in_many_exchanges": (
            0.0 if num_vertices == 0 else sum(value >= 5 for value in participation_values) / num_vertices
        ),
    }
    row.update(conflict_graph_features(exchanges))
    row["feasible_set_richness_score"] = 0.0
    return row


def zscore_map(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = [parse_float(row.get(key)) for row in rows]
    clean = [value for value in values if math.isfinite(value)]
    mean = finite_mean(clean)
    std = finite_std(clean)
    output: dict[str, float] = {}
    for row, value in zip(rows, values):
        graph_id = str(row["graph_id"])
        output[graph_id] = 0.0 if not math.isfinite(value) or std == 0 else (value - mean) / std
    return output


def add_richness_score(rows: list[dict[str, Any]]) -> None:
    score_keys = [
        "num_exchange_candidates",
        "num_cycles_total",
        "exchange_size_entropy",
        "vertex_exchange_participation_mean",
    ]
    zscores = {key: zscore_map(rows, key) for key in score_keys}
    for row in rows:
        graph_id = str(row["graph_id"])
        row["feasible_set_richness_score"] = sum(zscores[key][graph_id] for key in score_keys)


def load_graph_features_from_outcomes(
    *,
    dataset_dir: Path,
    outcome_rows: list[dict[str, str]],
    regime: str,
    max_cycle: int,
    max_chain: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for outcome in sorted(outcome_rows, key=lambda row: int(row["graph_id"].split("-")[1].split(".")[0])):
        graph_id = outcome["graph_id"]
        path = dataset_dir / graph_id
        graph_json = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            compute_graph_feature_row(
                graph_id=graph_id,
                graph_json=graph_json,
                regime=regime,
                max_cycle=max_cycle,
                max_chain=max_chain,
            )
        )
    add_richness_score(rows)
    return rows


def percentile_threshold(values: list[float], percentile: float) -> float:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return float("nan")
    idx = max(0, min(len(clean) - 1, math.ceil(percentile * len(clean)) - 1))
    return clean[idx]


def join_features_with_outcomes(
    feature_rows: list[dict[str, Any]],
    outcome_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    features = [dict(row) for row in feature_rows]
    add_richness_score(features)
    outcomes = {row["graph_id"]: row for row in outcome_rows}
    delta_values = [parse_float(row.get("median_delta_pp")) for row in outcome_rows]
    top5_delta = percentile_threshold(delta_values, 0.95)

    joined: list[dict[str, Any]] = []
    for feature in features:
        graph_id = str(feature["graph_id"])
        outcome = outcomes.get(graph_id, {})
        row: dict[str, Any] = dict(feature)
        for key in OUTCOME_FIELDS:
            row[key] = parse_float(outcome.get(key), 0.0)

        median_delta = parse_float(row["median_delta_pp"], 0.0)
        strict_rate = parse_float(row["strict_case_c_rate"], 0.0)
        row["helpful_graph"] = int(median_delta >= 10.0 or strict_rate >= 0.5)
        row["extreme_helpful_graph"] = int(strict_rate >= 1.0 or median_delta >= top5_delta)
        row["harmful_graph"] = int(median_delta <= -10.0)
        row["neutral_graph"] = int(abs(median_delta) <= 0.1 and strict_rate == 0.0)
        joined.append(row)
    return joined


def rankdata(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0 for _ in values]
    idx = 0
    while idx < len(indexed):
        end = idx + 1
        while end < len(indexed) and indexed[end][1] == indexed[idx][1]:
            end += 1
        rank = (idx + 1 + end) / 2.0
        for original_idx, _ in indexed[idx:end]:
            ranks[original_idx] = rank
        idx = end
    return ranks


def pearson(x_values: list[float], y_values: list[float]) -> float:
    if len(x_values) < 2:
        return float("nan")
    x_mean = finite_mean(x_values)
    y_mean = finite_mean(y_values)
    x_centered = [value - x_mean for value in x_values]
    y_centered = [value - y_mean for value in y_values]
    x_ss = sum(value * value for value in x_centered)
    y_ss = sum(value * value for value in y_centered)
    if x_ss <= 0 or y_ss <= 0:
        return float("nan")
    return float(sum(x * y for x, y in zip(x_centered, y_centered)) / math.sqrt(x_ss * y_ss))


def spearman(x_values: list[float], y_values: list[float]) -> float:
    return pearson(rankdata(x_values), rankdata(y_values))


def auroc(scores: list[float], labels: list[int]) -> float:
    positives = [score for score, label in zip(scores, labels) if label == 1]
    negatives = [score for score, label in zip(scores, labels) if label == 0]
    if not positives or not negatives:
        return float("nan")
    wins = 0.0
    total = 0
    for positive in positives:
        for negative in negatives:
            total += 1
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return float(wins / total)


def feature_family(feature: str) -> str:
    for family, features in FEATURE_FAMILIES.items():
        if feature in features:
            return family
    return "other"


def build_feature_association_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for feature in FEATURE_KEYS:
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


def percentile_rank(values: list[float], value: float) -> float:
    clean = [item for item in values if math.isfinite(item)]
    if not clean or not math.isfinite(value):
        return float("nan")
    return float(sum(item <= value for item in clean) / len(clean))


def build_selected_case_overlay_rows(
    rows: list[dict[str, Any]],
    *,
    selected_graphs: list[str] | tuple[str, ...] = SELECTED_GRAPHS,
    feature_keys: list[str] | tuple[str, ...] = (
        "density",
        "num_cycles_total",
        "num_chains_total",
        "num_exchange_candidates",
        "conflict_graph_density",
        "feasible_set_richness_score",
    ),
) -> list[dict[str, Any]]:
    by_graph = {str(row["graph_id"]): row for row in rows}
    feature_values = {
        feature: [parse_float(row.get(feature)) for row in rows] for feature in feature_keys
    }
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
        for feature in feature_keys:
            value = parse_float(source.get(feature))
            row[feature] = value
            row[f"{feature}_percentile"] = percentile_rank(feature_values[feature], value)
        output.append(row)
    return output


def build_story_markdown(
    *,
    path: str | Path,
    feature_rows: list[dict[str, Any]],
    joined_rows: list[dict[str, Any]],
    association_rows: list[dict[str, Any]],
    overlay_rows: list[dict[str, Any]],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    graph_count = len(joined_rows)
    helpful_count = sum(int(row.get("helpful_graph", 0)) for row in joined_rows)
    harmful_count = sum(int(row.get("harmful_graph", 0)) for row in joined_rows)
    neutral_count = sum(int(row.get("neutral_graph", 0)) for row in joined_rows)

    best_by_family: dict[str, dict[str, Any]] = {}
    best_helpful_by_family: dict[str, dict[str, Any]] = {}
    best_harmful_by_family: dict[str, dict[str, Any]] = {}
    for row in association_rows:
        family = row["feature_family"]
        current = best_by_family.get(family)
        score = abs(parse_float(row.get("spearman_median_delta_pp")))
        if current is None or score > abs(parse_float(current.get("spearman_median_delta_pp"))):
            best_by_family[family] = row
        helpful_current = best_helpful_by_family.get(family)
        helpful_score = parse_float(row.get("auroc_helpful"))
        if helpful_current is None or helpful_score > parse_float(
            helpful_current.get("auroc_helpful")
        ):
            best_helpful_by_family[family] = row
        harmful_current = best_harmful_by_family.get(family)
        harmful_score = parse_float(row.get("auroc_harmful"))
        if harmful_current is None or harmful_score > parse_float(
            harmful_current.get("auroc_harmful")
        ):
            best_harmful_by_family[family] = row

    lines = [
        "# Step2c Graph-Level DFL Suitability: Phase 1 Readout",
        "",
        "## Scope",
        "",
        "Phase 1 uses prospectively available graph and feasible-set descriptors only. "
        "It joins those descriptors to the existing all-400 Step2c model-seed outcome table.",
        "",
        "## Population",
        "",
        f"- graphs: {graph_count}",
        f"- helpful_graph: {helpful_count}",
        f"- harmful_graph: {harmful_count}",
        f"- neutral_graph: {neutral_count}",
        "",
        "## Best Spearman Association By Feature Family",
        "",
        "| Family | Feature | Spearman with median Delta | AUROC helpful | AUROC harmful |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for family in ["raw_topology", "cycle_chain", "exchange_geometry", "conflict_geometry"]:
        row = best_by_family.get(family, {})
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
    for family in ["raw_topology", "cycle_chain", "exchange_geometry", "conflict_geometry"]:
        helpful_row = best_helpful_by_family.get(family, {})
        harmful_row = best_harmful_by_family.get(family, {})
        lines.append(
            f"| {family} | {helpful_row.get('feature', 'NA')} | "
            f"{parse_float(helpful_row.get('auroc_helpful')):.3f} | "
            f"{harmful_row.get('feature', 'NA')} | "
            f"{parse_float(harmful_row.get('auroc_harmful')):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Selected Case Overlay",
            "",
            "| Graph | median Delta pp | density pct | exchanges pct | conflict density pct | richness pct |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in overlay_rows:
        lines.append(
            f"| {row['graph_id']} | {parse_float(row.get('median_delta_pp')):.2f} | "
            f"{parse_float(row.get('density_percentile')):.2f} | "
            f"{parse_float(row.get('num_exchange_candidates_percentile')):.2f} | "
            f"{parse_float(row.get('conflict_graph_density_percentile')):.2f} | "
            f"{parse_float(row.get('feasible_set_richness_score_percentile')):.2f} |"
        )

    lines.extend(
        [
            "",
            "## Report-Safe Interpretation",
            "",
            "This table is an association audit. A strong topology-only rule is not assumed. "
            "The result should be read as evidence about whether deployable graph and feasible-set descriptors contain signal about DFL suitability.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def overlay_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    return keys


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Phase 1 graph-level DFL suitability tables for Step2c."
    )
    parser.add_argument("--regime", default=DEFAULT_REGIME)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--outcome-input", type=Path, default=DEFAULT_OUTCOME_INPUT)
    parser.add_argument("--feature-output", type=Path, default=DEFAULT_FEATURE_OUTPUT)
    parser.add_argument("--join-output", type=Path, default=DEFAULT_JOIN_OUTPUT)
    parser.add_argument("--association-output", type=Path, default=DEFAULT_ASSOC_OUTPUT)
    parser.add_argument("--overlay-output", type=Path, default=DEFAULT_OVERLAY_OUTPUT)
    parser.add_argument("--story-output", type=Path, default=DEFAULT_STORY_OUTPUT)
    parser.add_argument("--max-cycle", type=int, default=3)
    parser.add_argument("--max-chain", type=int, default=4)
    return parser.parse_args(argv)


def build_phase1_outputs(args: argparse.Namespace) -> dict[str, Any]:
    outcome_rows = read_csv(args.outcome_input)
    feature_rows = load_graph_features_from_outcomes(
        dataset_dir=args.dataset_dir,
        outcome_rows=outcome_rows,
        regime=args.regime,
        max_cycle=args.max_cycle,
        max_chain=args.max_chain,
    )
    joined_rows = join_features_with_outcomes(feature_rows, outcome_rows)
    association_rows = build_feature_association_rows(joined_rows)
    overlay_rows = build_selected_case_overlay_rows(joined_rows)

    write_csv(args.feature_output, feature_rows, GRAPH_FEATURE_FIELDS)
    write_csv(args.join_output, joined_rows, JOIN_FIELDS)
    write_csv(args.association_output, association_rows, ASSOCIATION_FIELDS)
    write_csv(args.overlay_output, overlay_rows, overlay_fieldnames(overlay_rows))
    build_story_markdown(
        path=args.story_output,
        feature_rows=feature_rows,
        joined_rows=joined_rows,
        association_rows=association_rows,
        overlay_rows=overlay_rows,
    )

    return {
        "feature_rows": feature_rows,
        "joined_rows": joined_rows,
        "association_rows": association_rows,
        "overlay_rows": overlay_rows,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    outputs = build_phase1_outputs(args)
    print(f"Saved {len(outputs['feature_rows'])} graph feature rows to {args.feature_output}")
    print(f"Saved {len(outputs['joined_rows'])} feature-outcome rows to {args.join_output}")
    print(f"Saved {len(outputs['association_rows'])} association rows to {args.association_output}")
    print(f"Saved {len(outputs['overlay_rows'])} selected case rows to {args.overlay_output}")
    print(f"Saved Phase 1 readout to {args.story_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
