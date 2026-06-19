#!/usr/bin/env python3
"""Build immutable Step3 topology templates from processed KEP graphs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "step3_pairs7_step2c_poly_d8_mult_eps050_seed20260619"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "surrogate_experiment_results" / "Step3" / "pairs7" / "data" / "topologies"


BANK_FIELDS = [
    "topology_id",
    "source_path",
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
    "num_exchange_candidates",
    "num_chains_len1",
    "num_chains_len2",
    "num_chains_len3",
    "num_chains_len4",
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
    "topology_hash",
    "arc_order_hash",
    "feasible_set_hash",
    "template_path",
]

REJECT_FIELDS = [
    "topology_id",
    "source_path",
    "reason",
    "num_pairs",
    "num_ndds",
    "num_arcs",
    "num_feasible_candidates",
]


def int_or_text_key(value: Any) -> tuple[int, int | str]:
    text = str(value)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def graph_data(graph_json: dict[str, Any]) -> dict[str, Any]:
    data = graph_json.get("data", {})
    if not isinstance(data, dict):
        raise ValueError("processed graph data must be an object")
    return data


def canonical_vertices(graph_json: dict[str, Any]) -> list[dict[str, str]]:
    data = graph_data(graph_json)
    vertices: set[str] = {str(vertex) for vertex in data.keys()}
    for source, payload in data.items():
        for match in payload.get("matches", []) or []:
            recipient = match.get("recipient")
            if recipient is not None:
                vertices.add(str(recipient))

    output = []
    for vertex in sorted(vertices, key=int_or_text_key):
        payload = data.get(vertex, {})
        output.append({"id": vertex, "type": str(payload.get("type", "Pair"))})
    return output


def canonical_arcs(graph_json: dict[str, Any]) -> list[dict[str, Any]]:
    data = graph_data(graph_json)
    arcs: list[dict[str, Any]] = []
    edge_idx = 0
    for source in sorted(data.keys(), key=int_or_text_key):
        payload = data[source]
        matches = payload.get("matches", []) or []
        sortable_matches = sorted(
            enumerate(matches),
            key=lambda item: (int_or_text_key(str(item[1].get("recipient", ""))), item[0]),
        )
        for original_position, match in sortable_matches:
            recipient = match.get("recipient")
            if recipient is None:
                continue
            arcs.append(
                {
                    "edge_idx": edge_idx,
                    "source": str(source),
                    "target": str(recipient),
                    "source_type": str(payload.get("type", "Pair")),
                    "target_type": str(data.get(str(recipient), {}).get("type", "Pair")),
                    "source_match_position": int(original_position),
                }
            )
            edge_idx += 1
    return arcs


def arc_lookup(arcs: list[dict[str, Any]]) -> dict[tuple[str, str], int]:
    return {(arc["source"], arc["target"]): int(arc["edge_idx"]) for arc in arcs}


def enumerate_cycle_candidates(
    vertices: list[dict[str, str]],
    arcs: list[dict[str, Any]],
    max_cycle: int,
) -> list[dict[str, Any]]:
    pair_vertices = [vertex["id"] for vertex in vertices if vertex["type"] == "Pair"]
    pair_set = set(pair_vertices)
    edge_to_idx = arc_lookup(arcs)
    adjacency: dict[str, list[str]] = {vertex: [] for vertex in pair_vertices}
    for source, target in edge_to_idx:
        if source in pair_set and target in pair_set:
            adjacency.setdefault(source, []).append(target)
    for source in adjacency:
        adjacency[source] = sorted(set(adjacency[source]), key=int_or_text_key)

    cycles: list[dict[str, Any]] = []

    def dfs(start: str, current: str, path: list[str]) -> None:
        if len(path) > max_cycle:
            return
        for target in adjacency.get(current, []):
            if target == start:
                if len(path) >= 2 and start == min(path, key=int_or_text_key):
                    edge_path = []
                    closed = path + [start]
                    for left, right in zip(closed, closed[1:]):
                        edge_path.append(edge_to_idx[(left, right)])
                    cycles.append(
                        {
                            "type": "cycle",
                            "nodes": path.copy(),
                            "edges": edge_path,
                            "length": len(path),
                            "signature": "cycle:" + "->".join(closed),
                        }
                    )
            elif target not in path and int_or_text_key(target) > int_or_text_key(start):
                dfs(start, target, path + [target])

    if max_cycle >= 2:
        for start in sorted(pair_vertices, key=int_or_text_key):
            dfs(start, start, [start])

    unique = {candidate["signature"]: candidate for candidate in cycles}
    return [unique[key] for key in sorted(unique.keys())]


def enumerate_chain_candidates(
    vertices: list[dict[str, str]],
    arcs: list[dict[str, Any]],
    max_chain: int,
) -> list[dict[str, Any]]:
    edge_to_idx = arc_lookup(arcs)
    adjacency: dict[str, list[str]] = {}
    for source, target in edge_to_idx:
        adjacency.setdefault(source, []).append(target)
    for source in adjacency:
        adjacency[source] = sorted(set(adjacency[source]), key=int_or_text_key)

    ndds = sorted((vertex["id"] for vertex in vertices if vertex["type"] != "Pair"), key=int_or_text_key)
    chains: list[dict[str, Any]] = []

    def dfs(current: str, path: list[str], edge_path: list[int]) -> None:
        if len(edge_path) >= max_chain:
            return
        for target in adjacency.get(current, []):
            if target in path:
                continue
            new_path = path + [target]
            new_edges = edge_path + [edge_to_idx[(current, target)]]
            chains.append(
                {
                    "type": "chain",
                    "nodes": new_path,
                    "edges": new_edges,
                    "length": len(new_edges),
                    "signature": "chain:" + "->".join(new_path),
                }
            )
            dfs(target, new_path, new_edges)

    if max_chain >= 1:
        for ndd in ndds:
            dfs(ndd, [ndd], [])

    unique = {candidate["signature"]: candidate for candidate in chains}
    return [unique[key] for key in sorted(unique.keys())]


def feasible_candidates(
    vertices: list[dict[str, str]],
    arcs: list[dict[str, Any]],
    max_cycle: int,
    max_chain: int,
) -> list[dict[str, Any]]:
    cycles = enumerate_cycle_candidates(vertices, arcs, max_cycle=max_cycle)
    chains = enumerate_chain_candidates(vertices, arcs, max_chain=max_chain)
    return cycles + chains


def candidate_structure_descriptors(
    vertices: list[dict[str, str]],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    candidate_node_sets = [set(candidate["nodes"]) for candidate in candidates]
    num_candidates = len(candidate_node_sets)

    degrees = [0 for _ in candidate_node_sets]
    conflict_adjacency: list[set[int]] = [set() for _ in candidate_node_sets]
    conflict_edges = 0
    for left in range(num_candidates):
        for right in range(left + 1, num_candidates):
            if candidate_node_sets[left] & candidate_node_sets[right]:
                conflict_edges += 1
                degrees[left] += 1
                degrees[right] += 1
                conflict_adjacency[left].add(right)
                conflict_adjacency[right].add(left)

    possible_conflict_edges = num_candidates * (num_candidates - 1) / 2
    conflict_density = (
        conflict_edges / possible_conflict_edges if possible_conflict_edges else 0.0
    )
    mean_conflict_degree = sum(degrees) / num_candidates if num_candidates else 0.0
    max_conflict_degree = max(degrees) if degrees else 0

    seen: set[int] = set()
    component_sizes: list[int] = []
    for start in range(num_candidates):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        size = 0
        while stack:
            current = stack.pop()
            size += 1
            for neighbor in conflict_adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        component_sizes.append(size)

    largest_component_fraction = (
        max(component_sizes) / num_candidates if component_sizes else 0.0
    )

    vertex_ids = [vertex["id"] for vertex in vertices]
    candidates_per_vertex = {vertex_id: 0 for vertex_id in vertex_ids}
    for node_set in candidate_node_sets:
        for node in node_set:
            candidates_per_vertex.setdefault(node, 0)
            candidates_per_vertex[node] += 1

    covered_vertices = [
        vertex_id for vertex_id in vertex_ids if candidates_per_vertex.get(vertex_id, 0) > 0
    ]
    num_vertices = len(vertex_ids)
    total_candidate_vertex_incidence = sum(candidates_per_vertex.get(vertex_id, 0) for vertex_id in vertex_ids)

    return {
        "num_exchange_candidates": num_candidates,
        "num_chains_len1": sum(
            1 for candidate in candidates if candidate["type"] == "chain" and candidate["length"] == 1
        ),
        "num_chains_len2": sum(
            1 for candidate in candidates if candidate["type"] == "chain" and candidate["length"] == 2
        ),
        "num_chains_len3": sum(
            1 for candidate in candidates if candidate["type"] == "chain" and candidate["length"] == 3
        ),
        "num_chains_len4": sum(
            1 for candidate in candidates if candidate["type"] == "chain" and candidate["length"] == 4
        ),
        "candidate_conflict_edges": conflict_edges,
        "candidate_conflict_density": conflict_density,
        "mean_conflict_degree": mean_conflict_degree,
        "max_conflict_degree": max_conflict_degree,
        "num_conflict_components": len(component_sizes),
        "largest_conflict_component_fraction": largest_component_fraction,
        "num_vertices_in_any_candidate": len(covered_vertices),
        "fraction_vertices_in_any_candidate": (
            len(covered_vertices) / num_vertices if num_vertices else 0.0
        ),
        "mean_candidates_per_vertex": (
            total_candidate_vertex_incidence / num_vertices if num_vertices else 0.0
        ),
        "max_candidates_per_vertex": (
            max((candidates_per_vertex.get(vertex_id, 0) for vertex_id in vertex_ids), default=0)
        ),
    }


def build_topology_template(
    topology_id: str,
    graph_json: dict[str, Any],
    max_cycle: int = 3,
    max_chain: int = 4,
    source_path: str | None = None,
) -> dict[str, Any]:
    vertices = canonical_vertices(graph_json)
    arcs = canonical_arcs(graph_json)
    candidates = feasible_candidates(vertices, arcs, max_cycle=max_cycle, max_chain=max_chain)

    topology_payload = {"vertices": vertices, "arcs": arcs}
    arc_order_payload = [{"edge_idx": arc["edge_idx"], "source": arc["source"], "target": arc["target"]} for arc in arcs]
    feasible_payload = {
        "max_cycle": int(max_cycle),
        "max_chain": int(max_chain),
        "candidates": [
            {
                "type": candidate["type"],
                "nodes": candidate["nodes"],
                "edges": candidate["edges"],
                "signature": candidate["signature"],
            }
            for candidate in candidates
        ],
    }

    num_pairs = sum(1 for vertex in vertices if vertex["type"] == "Pair")
    num_ndds = len(vertices) - num_pairs
    num_2cycles = sum(1 for candidate in candidates if candidate["type"] == "cycle" and candidate["length"] == 2)
    num_3cycles = sum(1 for candidate in candidates if candidate["type"] == "cycle" and candidate["length"] == 3)
    num_chains = sum(1 for candidate in candidates if candidate["type"] == "chain")
    num_cycles = sum(1 for candidate in candidates if candidate["type"] == "cycle")
    descriptors = candidate_structure_descriptors(vertices, candidates)

    return {
        "topology_id": topology_id,
        "source_path": source_path,
        "max_cycle": int(max_cycle),
        "max_chain": int(max_chain),
        "vertices": vertices,
        "arcs": arcs,
        "feasible_candidates": candidates,
        "num_vertices": len(vertices),
        "num_pairs": num_pairs,
        "num_ndds": num_ndds,
        "num_arcs": len(arcs),
        "num_2cycles": num_2cycles,
        "num_3cycles": num_3cycles,
        "num_cycles_total": num_cycles,
        "num_chains_total": num_chains,
        "num_feasible_candidates": len(candidates),
        **descriptors,
        "topology_hash": stable_hash(topology_payload),
        "arc_order_hash": stable_hash(arc_order_payload),
        "feasible_set_hash": stable_hash(feasible_payload),
    }


def template_bank_row(template: dict[str, Any], template_path: Path) -> dict[str, Any]:
    return {
        "topology_id": template["topology_id"],
        "source_path": template.get("source_path") or "",
        "num_vertices": template["num_vertices"],
        "num_pairs": template["num_pairs"],
        "num_ndds": template["num_ndds"],
        "num_arcs": template["num_arcs"],
        "max_cycle": template["max_cycle"],
        "max_chain": template["max_chain"],
        "num_2cycles": template["num_2cycles"],
        "num_3cycles": template["num_3cycles"],
        "num_cycles_total": template["num_cycles_total"],
        "num_chains_total": template["num_chains_total"],
        "num_feasible_candidates": template["num_feasible_candidates"],
        "num_exchange_candidates": template["num_exchange_candidates"],
        "num_chains_len1": template["num_chains_len1"],
        "num_chains_len2": template["num_chains_len2"],
        "num_chains_len3": template["num_chains_len3"],
        "num_chains_len4": template["num_chains_len4"],
        "candidate_conflict_edges": template["candidate_conflict_edges"],
        "candidate_conflict_density": template["candidate_conflict_density"],
        "mean_conflict_degree": template["mean_conflict_degree"],
        "max_conflict_degree": template["max_conflict_degree"],
        "num_conflict_components": template["num_conflict_components"],
        "largest_conflict_component_fraction": template["largest_conflict_component_fraction"],
        "num_vertices_in_any_candidate": template["num_vertices_in_any_candidate"],
        "fraction_vertices_in_any_candidate": template["fraction_vertices_in_any_candidate"],
        "mean_candidates_per_vertex": template["mean_candidates_per_vertex"],
        "max_candidates_per_vertex": template["max_candidates_per_vertex"],
        "topology_hash": template["topology_hash"],
        "arc_order_hash": template["arc_order_hash"],
        "feasible_set_hash": template["feasible_set_hash"],
        "template_path": str(template_path),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def topology_id_from_path(path: Path) -> str:
    return path.stem


def reject_row(path: Path, reason: str, template: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "topology_id": topology_id_from_path(path),
        "source_path": str(path),
        "reason": reason,
        "num_pairs": "" if template is None else template["num_pairs"],
        "num_ndds": "" if template is None else template["num_ndds"],
        "num_arcs": "" if template is None else template["num_arcs"],
        "num_feasible_candidates": "" if template is None else template["num_feasible_candidates"],
    }


def prepare_output_dir(output_dir: Path, force: bool) -> None:
    if output_dir.exists() and force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def build_topology_bank(
    source_dir: Path,
    output_dir: Path,
    max_cycle: int = 3,
    max_chain: int = 4,
    expected_pairs: int = 7,
    min_candidates: int = 1,
    force: bool = False,
) -> dict[str, Any]:
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    prepare_output_dir(output_dir, force=force)

    accepted_rows: list[dict[str, Any]] = []
    hash_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []

    graph_paths = sorted(source_dir.glob("G-*.json"), key=lambda path: int_or_text_key(path.stem.replace("G-", "")))
    if not graph_paths:
        raise ValueError(f"No G-*.json files found in {source_dir}")

    for path in graph_paths:
        topology_id = topology_id_from_path(path)
        try:
            graph_json = json.loads(path.read_text(encoding="utf-8"))
            template = build_topology_template(
                topology_id,
                graph_json,
                max_cycle=max_cycle,
                max_chain=max_chain,
                source_path=str(path),
            )
        except Exception as exc:
            rejected_rows.append(reject_row(path, f"template_error: {exc}"))
            continue

        if template["num_pairs"] != expected_pairs:
            rejected_rows.append(
                reject_row(path, f"expected {expected_pairs} Pair vertices", template=template)
            )
            continue
        if template["num_feasible_candidates"] < min_candidates:
            rejected_rows.append(
                reject_row(path, f"fewer than {min_candidates} feasible candidates", template=template)
            )
            continue

        template_dir = output_dir / topology_id
        template_dir.mkdir(parents=True, exist_ok=True)
        template_path = template_dir / "template.json"
        template_path.write_text(
            json.dumps(template, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        row = template_bank_row(template, template_path)
        accepted_rows.append(row)
        hash_rows.append(
            {
                "topology_id": topology_id,
                "topology_hash": template["topology_hash"],
                "arc_order_hash": template["arc_order_hash"],
                "feasible_set_hash": template["feasible_set_hash"],
            }
        )

    write_csv(output_dir / "topology_bank.csv", accepted_rows, BANK_FIELDS)
    write_csv(
        output_dir / "topology_hashes.csv",
        hash_rows,
        ["topology_id", "topology_hash", "arc_order_hash", "feasible_set_hash"],
    )
    write_csv(output_dir / "rejected_topologies.csv", rejected_rows, REJECT_FIELDS)

    summary = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "max_cycle": int(max_cycle),
        "max_chain": int(max_chain),
        "expected_pairs": int(expected_pairs),
        "min_candidates": int(min_candidates),
        "input_graphs": len(graph_paths),
        "accepted": len(accepted_rows),
        "rejected": len(rejected_rows),
    }
    (output_dir / "build_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step3 fixed-topology templates and hashes.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-cycle", type=int, default=3)
    parser.add_argument("--max-chain", type=int, default=4)
    parser.add_argument("--expected-pairs", type=int, default=7)
    parser.add_argument("--min-candidates", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_topology_bank(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        max_cycle=args.max_cycle,
        max_chain=args.max_chain,
        expected_pairs=args.expected_pairs,
        min_candidates=args.min_candidates,
        force=args.force,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
