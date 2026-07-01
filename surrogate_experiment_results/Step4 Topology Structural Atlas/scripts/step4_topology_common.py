#!/usr/bin/env python3
"""Shared topology-first helpers for Step4 mechanism audits."""

from __future__ import annotations

import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
STRUCTURAL_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = STRUCTURAL_DIR.parents[1]
STEP4_ROOT = PROJECT_ROOT / "surrogate_experiment_results"
K18_E1_ROOT = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "Step3"
    / "K18_analysis"
    / "experiment_01_budget4to1"
)
DEFAULT_K18_TOPOLOGIES = K18_E1_ROOT / "configs" / "k18_topologies.csv"
DEFAULT_TEMPLATE_ROOT = PROJECT_ROOT / "surrogate_experiment_results" / "Step3" / "pairs20_ndd2" / "data" / "topologies"
DEFAULT_SENTINEL_TOPOLOGIES = ("G-269", "G-398", "G-784", "G-970", "G-364", "G-836", "G-79", "G-670")


TOPOLOGY_SUMMARY_FIELDS = [
    "selection_rank",
    "topology_id",
    "selection_bucket",
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
    "num_chains_len1",
    "num_chains_len2",
    "num_chains_len3",
    "num_chains_len4",
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
    "topology_hash",
    "arc_order_hash",
    "feasible_set_hash",
    "template_path",
]

ARC_FIELDS = [
    "topology_id",
    "edge_idx",
    "source",
    "target",
    "source_type",
    "target_type",
    "source_match_position",
]

CANDIDATE_FIELDS = [
    "topology_id",
    "candidate_id",
    "candidate_index",
    "candidate_type",
    "length",
    "node_set",
    "edge_set",
    "signature",
]

CONFLICT_FIELDS = [
    "topology_id",
    "left_candidate_id",
    "right_candidate_id",
    "shared_vertices",
    "left_type",
    "right_type",
    "left_length",
    "right_length",
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
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_int(value: Any, default: int = 0) -> int:
    if value in ("", None):
        return default
    return int(float(value))


def parse_float(value: Any, default: float = float("nan")) -> float:
    if value in ("", None):
        return default
    return float(value)


def topology_rows_by_id(path: str | Path = DEFAULT_K18_TOPOLOGIES) -> dict[str, dict[str, str]]:
    return {row["topology_id"]: row for row in read_csv_rows(path)}


def selected_topology_ids(value: list[str] | None, *, use_k18_all: bool = False) -> list[str]:
    if value:
        return [str(item) for item in value]
    if use_k18_all:
        rows = read_csv_rows(DEFAULT_K18_TOPOLOGIES)
        return [row["topology_id"] for row in sorted(rows, key=lambda row: parse_int(row["selection_rank"]))]
    return list(DEFAULT_SENTINEL_TOPOLOGIES)


def template_path(template_root: str | Path, topology_id: str) -> Path:
    return Path(template_root) / str(topology_id) / "template.json"


def load_template(template_root: str | Path, topology_id: str) -> dict[str, Any]:
    path = template_path(template_root, topology_id)
    payload = read_json(path)
    payload["_template_path"] = str(path)
    return payload


def pipe_join(values: list[Any]) -> str:
    return "|".join(str(value) for value in values)


def pipe_int_set(text: Any) -> set[int]:
    if text in ("", None):
        return set()
    return {int(part) for part in str(text).split("|") if part != ""}


def pipe_text_set(text: Any) -> set[str]:
    if text in ("", None):
        return set()
    return {str(part) for part in str(text).split("|") if part != ""}


def topology_summary_row(
    template: dict[str, Any],
    *,
    k18_row: dict[str, str] | None = None,
) -> dict[str, Any]:
    k18_row = k18_row or {}
    output = {
        "selection_rank": k18_row.get("selection_rank", ""),
        "topology_id": template["topology_id"],
        "selection_bucket": k18_row.get("selection_bucket", ""),
        "template_path": template.get("_template_path", ""),
    }
    for field in TOPOLOGY_SUMMARY_FIELDS:
        if field in output:
            continue
        output[field] = template.get(field, "")
    return output


def arc_rows_from_template(template: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arc in sorted(template.get("arcs", []), key=lambda item: parse_int(item["edge_idx"])):
        rows.append(
            {
                "topology_id": template["topology_id"],
                "edge_idx": parse_int(arc["edge_idx"]),
                "source": str(arc["source"]),
                "target": str(arc["target"]),
                "source_type": arc.get("source_type", ""),
                "target_type": arc.get("target_type", ""),
                "source_match_position": arc.get("source_match_position", ""),
            }
        )
    return rows


def candidate_rows_from_template(template: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    topology_id = str(template["topology_id"])
    for idx, candidate in enumerate(template.get("feasible_candidates", [])):
        rows.append(
            {
                "topology_id": topology_id,
                "candidate_id": f"{topology_id}:c{idx:04d}",
                "candidate_index": idx,
                "candidate_type": str(candidate["type"]),
                "length": parse_int(candidate["length"]),
                "node_set": pipe_join(list(candidate["nodes"])),
                "edge_set": pipe_join([int(edge) for edge in candidate["edges"]]),
                "signature": str(candidate["signature"]),
            }
        )
    return rows


def conflict_rows_from_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[str(candidate["topology_id"])].append(candidate)
    for topology_id in sorted(grouped):
        group = grouped[topology_id]
        for left_idx, left in enumerate(group):
            left_nodes = pipe_text_set(left["node_set"])
            for right in group[left_idx + 1 :]:
                shared = sorted(left_nodes & pipe_text_set(right["node_set"]))
                if not shared:
                    continue
                rows.append(
                    {
                        "topology_id": topology_id,
                        "left_candidate_id": left["candidate_id"],
                        "right_candidate_id": right["candidate_id"],
                        "shared_vertices": pipe_join(shared),
                        "left_type": left.get("candidate_type", ""),
                        "right_type": right.get("candidate_type", ""),
                        "left_length": parse_int(left.get("length")),
                        "right_length": parse_int(right.get("length")),
                    }
                )
    return rows


def candidate_rows_by_topology(candidates: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[str(candidate["topology_id"])].append(candidate)
    return grouped


def selected_candidate_ids_for_edge_signature(
    edge_signature: Any,
    candidates: list[dict[str, Any]],
) -> list[str]:
    selected_edges = pipe_int_set(edge_signature)
    if not selected_edges:
        return []
    matching: list[tuple[str, set[int], int]] = []
    for candidate in candidates:
        edge_set = pipe_int_set(candidate["edge_set"])
        if edge_set and edge_set <= selected_edges:
            matching.append((str(candidate["candidate_id"]), edge_set, parse_int(candidate.get("length"))))
    maximal: list[tuple[str, set[int], int]] = []
    for candidate_id, edge_set, length in matching:
        if any(edge_set < other_edges for _, other_edges, _ in matching):
            continue
        maximal.append((candidate_id, edge_set, length))
    return [
        candidate_id
        for candidate_id, _, _ in sorted(
            maximal,
            key=lambda item: (min(item[1]) if item[1] else 10**9, -item[2], item[0]),
        )
    ]


def candidate_id_set_text(edge_signature: Any, candidates: list[dict[str, Any]]) -> str:
    return pipe_join(selected_candidate_ids_for_edge_signature(edge_signature, candidates))


def circle_layout(ids: list[str], *, cx: float, cy: float, radius: float) -> dict[str, tuple[float, float]]:
    if not ids:
        return {}
    coords: dict[str, tuple[float, float]] = {}
    for idx, item in enumerate(ids):
        angle = 2.0 * math.pi * idx / len(ids) - math.pi / 2.0
        coords[item] = (cx + radius * math.cos(angle), cy + radius * math.sin(angle))
    return coords


def svg_text(x: float, y: float, text: Any, *, size: int = 12, weight: str = "400") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="#1f2933">'
        f"{html.escape(str(text))}</text>"
    )


def compatibility_graph_svg(
    topology_id: str,
    vertices: list[dict[str, Any]],
    arcs: list[dict[str, Any]],
) -> str:
    width, height = 900, 760
    vertex_ids = [str(vertex["id"]) for vertex in vertices]
    coords = circle_layout(vertex_ids, cx=450, cy=390, radius=275)
    type_by_id = {str(vertex["id"]): str(vertex.get("type", "Pair")) for vertex in vertices}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#6b7280"/></marker></defs>',
        svg_text(24, 36, f"{topology_id} compatibility graph", size=20, weight="700"),
        svg_text(24, 58, f"{len(vertex_ids)} vertices, {len(arcs)} arcs", size=12),
    ]
    for arc in arcs:
        source = str(arc["source"])
        target = str(arc["target"])
        if source not in coords or target not in coords:
            continue
        x1, y1 = coords[source]
        x2, y2 = coords[target]
        parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            'stroke="#9aa5b1" stroke-width="1.0" stroke-opacity="0.55" marker-end="url(#arrow)"/>'
        )
    for vertex_id in vertex_ids:
        x, y = coords[vertex_id]
        is_ndd = type_by_id.get(vertex_id) != "Pair"
        fill = "#f59f00" if is_ndd else "#0b7285"
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="15" fill="{fill}" stroke="#102a43" stroke-width="1"/>')
        parts.append(svg_text(x - 7, y + 4, vertex_id, size=10, weight="700"))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def candidate_conflict_svg(
    topology_id: str,
    candidates: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
) -> str:
    width, height = 920, 780
    candidate_ids = [str(row["candidate_id"]) for row in candidates]
    coords = circle_layout(candidate_ids, cx=460, cy=400, radius=285)
    candidate_by_id = {str(row["candidate_id"]): row for row in candidates}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(24, 36, f"{topology_id} candidate conflict graph", size=20, weight="700"),
        svg_text(24, 58, f"{len(candidate_ids)} candidates, {len(conflicts)} conflicts", size=12),
    ]
    for conflict in conflicts:
        left = str(conflict["left_candidate_id"])
        right = str(conflict["right_candidate_id"])
        if left not in coords or right not in coords:
            continue
        x1, y1 = coords[left]
        x2, y2 = coords[right]
        parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            'stroke="#d0d7de" stroke-width="0.8" stroke-opacity="0.35"/>'
        )
    for candidate_id in candidate_ids:
        row = candidate_by_id[candidate_id]
        x, y = coords[candidate_id]
        fill = "#2f9e44" if row.get("candidate_type") == "chain" else "#7048e8"
        radius = 4.0 + min(8.0, 1.8 * parse_int(row.get("length")))
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{fill}" fill-opacity="0.85"/>')
    parts.append(svg_text(720, 40, "green=chain", size=11))
    parts.append(svg_text(720, 58, "purple=cycle", size=11))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"
