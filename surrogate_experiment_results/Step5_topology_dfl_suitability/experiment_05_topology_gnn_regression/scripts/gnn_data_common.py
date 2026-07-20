#!/usr/bin/env python3
"""Shared topology-only graph and target helpers for Experiment 05."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
STEP5_ROOT = EXPERIMENT_ROOT.parent
EXP3_ROOT = STEP5_ROOT / "experiment_03_formal_continuous_label_seed42_sample50"
DEFAULT_FORMAL_SUMMARY = EXP3_ROOT / "results" / "formal1000" / "results" / "weak_label_topology_summary.csv"
DEFAULT_OUTPUT_ROOT = EXPERIMENT_ROOT / "results" / "scaffold"
NODE_FEATURE_NAMES = [
    "is_vertex",
    "is_candidate",
    "is_pair",
    "is_ndd",
    "is_cycle",
    "is_chain",
    "candidate_length",
    "compat_in_degree",
    "compat_out_degree",
    "candidate_membership_count",
]
RELATION_TYPES = {
    "compatibility": 0,
    "vertex_to_candidate": 1,
    "candidate_to_vertex": 2,
}
SCALAR_FEATURES = [
    "num_vertices",
    "num_pairs",
    "num_ndds",
    "num_arcs",
    "max_cycle",
    "max_chain",
    "num_2cycles",
    "num_3cycles",
    "num_cycles_total",
    "num_chains_total",
    "num_feasible_candidates",
    "candidate_conflict_edges",
    "candidate_conflict_density",
    "mean_conflict_degree",
    "max_conflict_degree",
    "num_conflict_components",
    "largest_conflict_component_fraction",
    "num_vertices_in_any_candidate",
    "fraction_vertices_in_any_candidate",
    "mean_candidates_per_vertex",
    "max_candidates_per_vertex",
]


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def resolve_project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def stable_key(seed: int, *parts: str) -> str:
    return hashlib.sha256("|".join([str(seed), *parts]).encode("utf-8")).hexdigest()


def truthy(value: Any) -> bool:
    return value is True or str(value).lower() in {"true", "1", "yes"}


def build_graph_record(
    summary_row: dict[str, str],
    template: dict[str, Any],
    *,
    formal_target_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    topology_id = str(summary_row["topology_id"])
    if str(template.get("topology_id")) != topology_id:
        raise ValueError(f"template topology mismatch for {topology_id}")
    vertices = list(template.get("vertices", []))
    candidates = list(template.get("feasible_candidates", []))
    vertex_index = {str(vertex["id"]): index for index, vertex in enumerate(vertices)}
    if len(vertex_index) != len(vertices):
        raise ValueError(f"duplicate vertex ids in {topology_id}")
    in_degree = {vertex_id: 0 for vertex_id in vertex_index}
    out_degree = {vertex_id: 0 for vertex_id in vertex_index}
    membership = {vertex_id: 0 for vertex_id in vertex_index}
    for arc in template.get("arcs", []):
        source, target = str(arc["source"]), str(arc["target"])
        out_degree[source] += 1
        in_degree[target] += 1
    candidate_vertex_sets: list[list[str]] = []
    for candidate in candidates:
        seen: set[str] = set()
        members = []
        for node in candidate.get("nodes", []):
            node_id = str(node)
            if node_id in vertex_index and node_id not in seen:
                members.append(node_id)
                seen.add(node_id)
                membership[node_id] += 1
        candidate_vertex_sets.append(members)

    node_ids: list[str] = []
    node_features: list[list[float]] = []
    for vertex in vertices:
        vertex_id = str(vertex["id"])
        is_pair = str(vertex.get("type", "")).lower() == "pair"
        node_ids.append(f"v:{vertex_id}")
        node_features.append([
            1.0, 0.0, float(is_pair), float(not is_pair), 0.0, 0.0, 0.0,
            float(in_degree[vertex_id]), float(out_degree[vertex_id]), float(membership[vertex_id]),
        ])
    for candidate_index, candidate in enumerate(candidates):
        candidate_type = str(candidate.get("type", "")).lower()
        node_ids.append(f"c:{candidate_index}")
        node_features.append([
            0.0, 1.0, 0.0, 0.0,
            float(candidate_type == "cycle"), float(candidate_type == "chain"),
            float(candidate.get("length", len(candidate_vertex_sets[candidate_index]))),
            0.0, 0.0, 0.0,
        ])

    edge_source: list[int] = []
    edge_target: list[int] = []
    edge_type: list[int] = []
    for arc in template.get("arcs", []):
        edge_source.append(vertex_index[str(arc["source"])])
        edge_target.append(vertex_index[str(arc["target"])])
        edge_type.append(RELATION_TYPES["compatibility"])
    candidate_offset = len(vertices)
    for candidate_index, members in enumerate(candidate_vertex_sets):
        candidate_node = candidate_offset + candidate_index
        for vertex_id in members:
            vertex_node = vertex_index[vertex_id]
            edge_source.extend([vertex_node, candidate_node])
            edge_target.extend([candidate_node, vertex_node])
            edge_type.extend([RELATION_TYPES["vertex_to_candidate"], RELATION_TYPES["candidate_to_vertex"]])

    structural = {field: float(summary_row[field]) for field in SCALAR_FEATURES}
    if formal_target_row is None:
        target = {
            "name": "normalized_improvement_pp",
            "value": float(summary_row["normalized_improvement_pp"]),
            "source": "experiment_03_seed42_provisional_pending_full_multiseed_labels",
            "formal": False,
        }
        label_uncertainty = None
    elif truthy(formal_target_row.get("formal_label_ready")):
        target = {
            "name": "formal_label_mean_pp",
            "value": float(formal_target_row["formal_label_mean_pp"]),
            "source": "mean_over_train_seeds_42_43_44",
            "formal": True,
        }
        label_uncertainty = {
            "name": "label_uncertainty_std_pp",
            "value": float(formal_target_row["label_uncertainty_std_pp"]),
            "ddof": int(formal_target_row["uncertainty_ddof"]),
            "source": "population_std_over_train_seeds_42_43_44",
        }
    else:
        target = {
            "name": "formal_label_mean_pp",
            "value": None,
            "source": "incomplete_three_seed_label",
            "formal": False,
        }
        label_uncertainty = None

    record = {
        "topology_id": topology_id,
        "topology_hash": summary_row["topology_hash"],
        "feasible_set_hash": summary_row["feasible_set_hash"],
        "template_path": summary_row["template_path"],
        "node_ids": node_ids,
        "node_feature_names": NODE_FEATURE_NAMES,
        "node_features": node_features,
        "edge_source": edge_source,
        "edge_target": edge_target,
        "edge_type": edge_type,
        "edge_type_names": RELATION_TYPES,
        "scalar_topology_features": structural,
        "target": target,
        "label_uncertainty": label_uncertainty,
    }
    if len(node_ids) != len(node_features):
        raise AssertionError("node id/feature length mismatch")
    if not (len(edge_source) == len(edge_target) == len(edge_type)):
        raise AssertionError("edge arrays have inconsistent lengths")
    return record


def validate_no_target_leakage(record: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    feature_names = set(record.get("node_feature_names", []))
    scalar_names = set(record.get("scalar_topology_features", {}))
    forbidden = {"normalized_improvement_pp", "data_seed", "train_seed", "test_gap_2stage", "test_gap_spoplus", "delta"}
    leaked = sorted((feature_names | scalar_names) & forbidden)
    if leaked:
        failures.append(f"forbidden_input_features:{','.join(leaked)}")
    if record.get("target", {}).get("name") not in {
        "normalized_improvement_pp",
        "formal_label_mean_pp",
    }:
        failures.append("target_name_mismatch")
    return failures
