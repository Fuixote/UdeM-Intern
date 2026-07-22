#!/usr/bin/env python3
"""Derive target-free candidate-conflict graphs from the locked incidence graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import material_common as common


NODE_FEATURE_NAMES = [
    "is_cycle",
    "is_chain",
    "candidate_length",
    "num_pair_members",
    "num_ndd_members",
    "mean_member_compat_in_degree",
    "mean_member_compat_out_degree",
    "max_member_candidate_membership_count",
]


def build_record(graph: dict[str, Any]) -> dict[str, Any]:
    topology_id = str(graph["topology_id"])
    features = graph["node_features"]
    candidate_nodes = [
        index
        for index, feature in enumerate(features)
        if float(feature[1]) == 1.0
    ]
    candidate_set = set(candidate_nodes)
    candidate_code = {
        original_index: new_index
        for new_index, original_index in enumerate(candidate_nodes)
    }
    members: dict[int, set[int]] = {index: set() for index in candidate_nodes}
    for source, target, edge_type in zip(
        graph["edge_source"],
        graph["edge_target"],
        graph["edge_type"],
        strict=True,
    ):
        if int(edge_type) == 1:
            if int(target) not in candidate_set:
                raise ValueError(f"{topology_id}:vertex_to_candidate_target_is_not_candidate")
            members[int(target)].add(int(source))
    node_features: list[list[float]] = []
    node_ids: list[str] = []
    for candidate_index in candidate_nodes:
        candidate_feature = features[candidate_index]
        candidate_members = sorted(members[candidate_index])
        if not candidate_members:
            raise ValueError(f"{topology_id}:candidate_without_members:{candidate_index}")
        member_features = [features[index] for index in candidate_members]
        node_ids.append(str(graph["node_ids"][candidate_index]))
        node_features.append(
            [
                float(candidate_feature[4]),
                float(candidate_feature[5]),
                float(candidate_feature[6]),
                float(sum(float(feature[2]) for feature in member_features)),
                float(sum(float(feature[3]) for feature in member_features)),
                float(sum(float(feature[7]) for feature in member_features) / len(member_features)),
                float(sum(float(feature[8]) for feature in member_features) / len(member_features)),
                float(max(float(feature[9]) for feature in member_features)),
            ]
        )
    edge_source: list[int] = []
    edge_target: list[int] = []
    undirected_count = 0
    for left_position, left in enumerate(candidate_nodes):
        for right in candidate_nodes[left_position + 1 :]:
            if members[left] & members[right]:
                left_code = candidate_code[left]
                right_code = candidate_code[right]
                edge_source.extend([left_code, right_code])
                edge_target.extend([right_code, left_code])
                undirected_count += 1
    expected_count = int(round(float(graph["scalar_topology_features"]["candidate_conflict_edges"])))
    if undirected_count != expected_count:
        raise ValueError(
            f"{topology_id}:candidate_conflict_edge_count_mismatch:"
            f"{undirected_count}!={expected_count}"
        )
    return {
        "topology_id": topology_id,
        "topology_hash": graph["topology_hash"],
        "feasible_set_hash": graph["feasible_set_hash"],
        "node_ids": node_ids,
        "node_feature_names": NODE_FEATURE_NAMES,
        "node_features": node_features,
        "edge_source": edge_source,
        "edge_target": edge_target,
        "edge_semantics": "bidirectional_candidate_conflict_if_candidates_share_a_vertex",
        "candidate_count": len(candidate_nodes),
        "undirected_conflict_edge_count": undirected_count,
        "target_free": True,
    }


def build_all(graph_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    failures: list[str] = []
    output: list[dict[str, Any]] = []
    for graph in graph_rows:
        try:
            output.append(build_record(graph))
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(str(exc))
    output.sort(key=lambda row: common.topology_sort_key(str(row["topology_id"])))
    return output, failures


def audit(
    rows: list[dict[str, Any]],
    failures: list[str],
    *,
    source_path: Path,
    source_sha256: str,
    output_path: Path,
) -> dict[str, Any]:
    topology_ids = [str(row["topology_id"]) for row in rows]
    topology_hashes = [str(row["topology_hash"]) for row in rows]
    feasible_hashes = [str(row["feasible_set_hash"]) for row in rows]
    if source_sha256 != common.LOCKED_INCIDENCE_GRAPH_SHA256:
        failures.append(f"source_sha256_mismatch:{source_sha256}")
    if len(rows) != 1000:
        failures.append(f"record_count_mismatch:{len(rows)}!=1000")
    if len(topology_ids) != len(set(topology_ids)):
        failures.append("topology_ids_not_unique")
    if len(topology_hashes) != len(set(topology_hashes)):
        failures.append("topology_hashes_not_unique")
    if len(feasible_hashes) != len(set(feasible_hashes)):
        failures.append("feasible_set_hashes_not_unique")
    if any(not row.get("target_free") or "target" in row for row in rows):
        failures.append("target_metadata_present")
    if any(len(row["edge_source"]) != len(row["edge_target"]) for row in rows):
        failures.append("edge_array_length_mismatch")
    return {
        "passed": not failures,
        "status": "success" if not failures else "failed",
        "record_count": len(rows),
        "node_feature_names": NODE_FEATURE_NAMES,
        "source_path": str(source_path),
        "source_sha256": source_sha256,
        "expected_source_sha256": common.LOCKED_INCIDENCE_GRAPH_SHA256,
        "output_path": str(output_path),
        "output_sha256": common.sha256_file(output_path) if output_path.exists() else None,
        "total_candidate_nodes": sum(int(row["candidate_count"]) for row in rows),
        "total_directed_conflict_edges": sum(len(row["edge_source"]) for row in rows),
        "target_or_uncertainty_present": False,
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--incidence-graphs",
        type=Path,
        default=common.DEFAULT_INCIDENCE_GRAPHS,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=common.DEFAULT_OUTPUT_ROOT
        / "candidate_conflict"
        / "candidate_conflict_graphs.jsonl",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=common.DEFAULT_OUTPUT_ROOT
        / "candidate_conflict"
        / "candidate_conflict_graphs.audit.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_sha256 = common.sha256_file(args.incidence_graphs)
    rows, failures = build_all(common.read_jsonl(args.incidence_graphs))
    if rows:
        common.atomic_write_jsonl(args.output, rows)
    result = audit(
        rows,
        failures,
        source_path=args.incidence_graphs,
        source_sha256=source_sha256,
        output_path=args.output,
    )
    common.atomic_write_json(args.audit_output, result)
    print(
        json.dumps(
            {
                **result,
                "audit_output": str(args.audit_output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
