#!/usr/bin/env python3
"""Shared helpers for Step3 fixed-topology full-(X,y) resampling."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_topology_bank  # noqa: E402


TRAIN_NAMESPACES = {"screen_train", "confirm_train"}
EVAL_NAMESPACES = {
    "screen_validation",
    "screen_test",
    "confirm_validation",
    "confirm_test",
}
ALL_NAMESPACES = TRAIN_NAMESPACES | EVAL_NAMESPACES
EVAL_TRAIN_SEED_SENTINEL = "EVAL"


def stable_json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hexdigest(payload: Any) -> str:
    return hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()


def stable_int_seed(*parts: Any) -> int:
    material = stable_json_dumps([str(part) for part in parts]).encode("utf-8")
    digest = hashlib.sha256(material).digest()
    return int.from_bytes(digest[:8], "big") % (2**63)


def normalize_train_seed(split_namespace: str, train_seed: int | None) -> int | None:
    if split_namespace not in ALL_NAMESPACES:
        raise ValueError(f"Unknown split namespace {split_namespace!r}")
    if split_namespace in TRAIN_NAMESPACES:
        if train_seed is None:
            raise ValueError(f"{split_namespace} requires train_seed")
        return int(train_seed)
    if train_seed is not None:
        raise ValueError(f"{split_namespace} must not vary with train_seed")
    return None


def seed_key_train_seed(split_namespace: str, train_seed: int | None) -> int | str:
    normalized = normalize_train_seed(split_namespace, train_seed)
    return EVAL_TRAIN_SEED_SENTINEL if normalized is None else int(normalized)


def derive_sample_seeds(
    *,
    experiment_version: str,
    regime: str,
    topology_id: str,
    split_namespace: str,
    train_seed: int | None,
    sample_index: int,
    master_label_seed: int,
) -> dict[str, int]:
    seed_train_part = seed_key_train_seed(split_namespace, train_seed)
    context_seed = stable_int_seed(
        "context",
        experiment_version,
        regime,
        topology_id,
        split_namespace,
        seed_train_part,
        int(sample_index),
    )
    label_noise_seed = stable_int_seed(
        "label",
        int(master_label_seed),
        experiment_version,
        regime,
        topology_id,
        split_namespace,
        seed_train_part,
        int(sample_index),
    )
    return {
        "context_seed": int(context_seed),
        "label_noise_seed": int(label_noise_seed),
    }


def generator_config_hash(config: dict[str, Any]) -> str:
    return stable_hexdigest(config)


def sorted_template_arcs(template: dict[str, Any]) -> list[dict[str, Any]]:
    arcs = list(template.get("arcs", []))
    return sorted(arcs, key=lambda arc: int(arc.get("edge_idx", len(arcs))))


def data_node(payload: dict[str, Any], node_id: str) -> dict[str, Any]:
    data = payload.get("data", {})
    if str(node_id) in data:
        return data[str(node_id)]
    if node_id in data:
        return data[node_id]
    raise KeyError(f"Missing node {node_id!r}")


def match_for_arc(payload: dict[str, Any], arc: dict[str, Any]) -> dict[str, Any]:
    source = str(arc["source"])
    target = str(arc["target"])
    node = data_node(payload, source)
    matches = node.get("matches", []) or []
    if "source_match_position" in arc:
        pos = int(arc["source_match_position"])
        if 0 <= pos < len(matches) and str(matches[pos].get("recipient")) == target:
            return matches[pos]
    candidates = [match for match in matches if str(match.get("recipient")) == target]
    if len(candidates) != 1:
        raise ValueError(f"Expected one match for arc {source}->{target}, found {len(candidates)}")
    return candidates[0]


def feature_matrix_from_payload(payload: dict[str, Any], template: dict[str, Any]) -> np.ndarray:
    rows: list[list[float]] = []
    for arc in sorted_template_arcs(template):
        match = match_for_arc(payload, arc)
        target_node = data_node(payload, str(arc["target"]))
        cpra = target_node.get("patient", {}).get("cPRA")
        rows.append([float(match.get("utility", 0.0)), float(cpra)])
    return np.asarray(rows, dtype=float)


def step2c_context_matrix_from_payload(payload: dict[str, Any], template: dict[str, Any]) -> np.ndarray:
    rows: list[list[float]] = []
    for arc in sorted_template_arcs(template):
        match = match_for_arc(payload, arc)
        rows.append([float(match.get("utility", 0.0)), float(match.get("recipient_cpra", 0.0))])
    return np.asarray(rows, dtype=float)


def labels_from_payload(payload: dict[str, Any], template: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            float(match_for_arc(payload, arc).get("ground_truth_label", 0.0))
            for arc in sorted_template_arcs(template)
        ],
        dtype=float,
    )


def matrix_hash(matrix: np.ndarray) -> str:
    arr = np.asarray(matrix, dtype=float)
    material = [[format(float(value), ".17g") for value in row] for row in arr.tolist()]
    return stable_hexdigest(material)


def vector_hash(vector: np.ndarray) -> str:
    arr = np.asarray(vector, dtype=float).reshape(-1)
    return stable_hexdigest([format(float(value), ".17g") for value in arr.tolist()])


def x_hash(payload: dict[str, Any], template: dict[str, Any]) -> str:
    return matrix_hash(feature_matrix_from_payload(payload, template))


def label_hash(payload: dict[str, Any], template: dict[str, Any]) -> str:
    return vector_hash(labels_from_payload(payload, template))


def template_hashes(template: dict[str, Any]) -> dict[str, str]:
    return {
        "topology_hash": str(template.get("topology_hash", "")),
        "arc_order_hash": str(template.get("arc_order_hash", "")),
        "feasible_set_hash": str(template.get("feasible_set_hash", "")),
    }


def payload_structural_hashes(payload: dict[str, Any], template: dict[str, Any]) -> dict[str, str]:
    rebuilt = build_topology_bank.build_topology_template(
        str(template.get("topology_id", "G")),
        payload,
        max_cycle=int(template.get("max_cycle", 3)),
        max_chain=int(template.get("max_chain", 4)),
    )
    return template_hashes(rebuilt)


def assert_structure_matches_template(payload: dict[str, Any], template: dict[str, Any]) -> None:
    expected = template_hashes(template)
    observed = payload_structural_hashes(payload, template)
    for key, expected_value in expected.items():
        if expected_value and observed.get(key) != expected_value:
            raise AssertionError(f"{key} changed: expected={expected_value} observed={observed.get(key)}")


def assert_context_label_consistency(payload: dict[str, Any], template: dict[str, Any]) -> None:
    model_x = feature_matrix_from_payload(payload, template)
    step2c_x = step2c_context_matrix_from_payload(payload, template)
    if not np.allclose(model_x, step2c_x, atol=0.0, rtol=0.0):
        raise AssertionError("Model-visible X differs from Step2c-visible X")

    incoming: dict[str, set[float]] = {}
    for arc in sorted_template_arcs(template):
        match = match_for_arc(payload, arc)
        target = str(arc["target"])
        target_cpra = float(data_node(payload, target).get("patient", {}).get("cPRA"))
        arc_cpra = float(match.get("recipient_cpra", 0.0))
        if arc_cpra != target_cpra:
            raise AssertionError(
                f"recipient cPRA mismatch for target {target}: node={target_cpra} arc={arc_cpra}"
            )
        incoming.setdefault(target, set()).add(arc_cpra)
    for target, values in incoming.items():
        if len(values) != 1:
            raise AssertionError(f"Incoming arcs for recipient {target} have multiple cPRA values")


def sample_manifest_hashes(sample_rows: list[dict[str, Any]]) -> str:
    return stable_hexdigest(
        [
            {
                "sample_id": row["sample_id"],
                "x_hash": row["x_hash"],
                "label_hash": row["label_hash"],
            }
            for row in sample_rows
        ]
    )


def prefix_hash(sample_rows: list[dict[str, Any]], size: int) -> str:
    return sample_manifest_hashes(sample_rows[: int(size)])


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=output.parent,
        delete=False,
        prefix=f".{output.name}.",
        suffix=".tmp",
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_name = handle.name
    os.replace(temp_name, output)


def atomic_savez(path: str | Path, **arrays: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb",
        dir=output.parent,
        delete=False,
        prefix=f".{output.name}.",
        suffix=".npz",
    ) as handle:
        temp_name = handle.name
    try:
        np.savez_compressed(temp_name, **arrays)
        saved_name = temp_name if temp_name.endswith(".npz") else temp_name + ".npz"
        os.replace(saved_name, output)
    finally:
        for candidate in (temp_name, temp_name + ".npz"):
            if os.path.exists(candidate):
                os.unlink(candidate)


def write_npz_dataset(
    path: str | Path,
    *,
    samples: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    X = np.asarray([sample["X"] for sample in samples], dtype=float)
    y = np.asarray([sample["y"] for sample in samples], dtype=float)
    payloads = np.asarray(
        [stable_json_dumps(sample["payload"]) for sample in samples],
        dtype=object,
    )
    sample_manifests = np.asarray(
        [stable_json_dumps(sample["manifest"]) for sample in samples],
        dtype=object,
    )
    atomic_savez(
        path,
        X=X,
        y=y,
        payloads=payloads,
        sample_manifests=sample_manifests,
        manifest_json=np.asarray(stable_json_dumps(manifest), dtype=object),
    )


def read_npz_dataset(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        manifest_raw = data["manifest_json"].item()
        return {
            "X": np.asarray(data["X"], dtype=float),
            "y": np.asarray(data["y"], dtype=float),
            "payloads": [json.loads(str(value)) for value in data["payloads"].tolist()],
            "sample_manifests": [
                json.loads(str(value)) for value in data["sample_manifests"].tolist()
            ],
            "manifest": json.loads(str(manifest_raw)),
        }


def materialize_npz_payloads_to_dir(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    dataset = read_npz_dataset(dataset_path)
    payloads = dataset["payloads"]
    if limit is not None:
        payloads = payloads[: int(limit)]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    for index, payload in enumerate(payloads):
        path = output / f"G-{index:06d}.json"
        atomic_write_json(path, payload)
        entries.append({"index": index, "graph_id": index, "path": str(path)})
    return entries
