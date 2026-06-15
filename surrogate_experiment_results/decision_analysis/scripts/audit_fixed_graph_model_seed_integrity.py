#!/usr/bin/env python3
"""Integrity checks for fixed graph-instance x subset_seed audits.

This script does not solve KEP models. It verifies that the all-seed replay rows
come from different training artifacts and different prediction vectors, then
separates model-output variation from decision-signature stability.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULT_DIR = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "decision_analysis"
    / "results"
    / "fixed_graph_model_seed"
)
DEFAULT_INPUT = DEFAULT_RESULT_DIR / "step2c_g392_g1560_all50_top5_second_best.csv"
DEFAULT_OUTPUT = DEFAULT_RESULT_DIR / "step2c_g392_g1560_all50_model_seed_integrity.csv"
DEFAULT_SUMMARY_OUTPUT = (
    DEFAULT_RESULT_DIR / "step2c_g392_g1560_all50_model_seed_integrity_summary.csv"
)
DEFAULT_DATASET_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "step2c_poly_d8_mult_eps050_main2000_seed20260523"
)


BASE_FIELDS = [
    "regime",
    "graph_id",
    "subset_seed",
    "method_label",
    "checkpoint_path",
    "checkpoint_sha_or_mtime_size_hash",
    "checkpoint_sha256",
    "checkpoint_size_bytes",
    "checkpoint_mtime_ns",
    "train_subset_path",
    "train_subset_hash",
    "train_subset_size",
    "prediction_vector_hash",
    "prediction_l2_from_discovery_seed",
    "prediction_corr_with_discovery_seed",
    "rank1_signature",
    "rank2_signature",
    "rank1_gap_pct",
    "rank2_gap_pct",
]

SUMMARY_BASE_FIELDS = [
    "regime",
    "graph_id",
    "method_label",
    "row_count",
    "unique_checkpoint_hashes",
    "unique_train_subset_hashes",
    "unique_prediction_hashes",
    "unique_rank1_signatures",
    "unique_rank2_signatures",
    "mean_prediction_l2_from_discovery_seed",
    "min_prediction_l2_from_discovery_seed",
    "max_prediction_l2_from_discovery_seed",
    "mean_prediction_corr_with_discovery_seed",
    "min_prediction_corr_with_discovery_seed",
    "max_prediction_corr_with_discovery_seed",
]


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


def parse_float(value: object) -> float:
    if value is None or str(value).strip() == "":
        return float("nan")
    return float(value)


def parse_int(value: object) -> int:
    return int(float(str(value)))


def finite_values(values: list[float]) -> list[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


def finite_mean(values: list[float]) -> float:
    clean = finite_values(values)
    if not clean:
        return float("nan")
    return float(sum(clean) / len(clean))


def finite_min(values: list[float]) -> float:
    clean = finite_values(values)
    return float(min(clean)) if clean else float("nan")


def finite_max(values: list[float]) -> float:
    clean = finite_values(values)
    return float(max(clean)) if clean else float("nan")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_payload_hash(payload: Any) -> str:
    material = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(material)


def normalized_train_subset_payload(payload: Any) -> list[str]:
    if not isinstance(payload, list):
        return [str(payload)]
    graph_ids: list[str] = []
    for item in payload:
        if isinstance(item, dict):
            graph_ids.append(Path(str(item.get("path", item.get("graph_id", "")))).name)
        else:
            graph_ids.append(Path(str(item)).name)
    return sorted(graph_ids)


def train_subset_hash(path: str | Path) -> tuple[str, int]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    normalized = normalized_train_subset_payload(payload)
    return json_payload_hash(normalized), len(normalized)


def prediction_vector_hash(values: Any) -> str:
    arr = np.asarray(values, dtype=np.float64)
    material = arr.shape.__repr__().encode("utf-8") + b"\0" + np.ascontiguousarray(arr).tobytes()
    return sha256_bytes(material)


def top_signature_hash(signatures: list[str]) -> str:
    return sha256_bytes("\n".join(signatures).encode("utf-8"))


def vector_corr(left: Any, right: Any) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"Prediction vector shape mismatch: {a.shape} != {b.shape}")
    if len(a) == 0:
        return float("nan")
    if np.allclose(a, b):
        return 1.0
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def parse_discovery_seeds(values: list[str] | None) -> dict[str, int]:
    seeds: dict[str, int] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Discovery seed must be GRAPH=SEED, got {value!r}")
        graph_id, seed_text = value.split("=", 1)
        seeds[graph_id] = int(seed_text)
    return seeds


def resolve_existing_path(path_text: str, project_root: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return project_root / path


def feature_matrix_fields(top_k: int) -> tuple[str, str]:
    signature_field = f"top{int(top_k)}_signature_hash"
    unique_field = f"unique_top{int(top_k)}_signature_hashes"
    return signature_field, unique_field


def output_fields(top_k: int) -> list[str]:
    signature_field, _ = feature_matrix_fields(top_k)
    return BASE_FIELDS + [signature_field]


def summary_fields(top_k: int) -> list[str]:
    _, unique_field = feature_matrix_fields(top_k)
    return SUMMARY_BASE_FIELDS + [unique_field]


def solution_group_key(row: dict[str, str]) -> tuple[str, str, int, str]:
    return (
        str(row["regime"]),
        str(row["graph_id"]),
        parse_int(row["subset_seed"]),
        str(row["method_label"]),
    )


def rank_row_by_solution_rank(rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    return {parse_int(row["solution_rank"]): row for row in rows}


def rank_gap_pct(row: dict[str, str] | None) -> float:
    if row is None:
        return float("nan")
    return 100.0 * parse_float(row.get("normalized_gap_to_oracle"))


def rank_signature(row: dict[str, str] | None) -> str:
    if row is None:
        return ""
    return str(row.get("solution_edge_signature", ""))


def checkpoint_hash_fields(path: Path) -> dict[str, Any]:
    stat = path.stat()
    sha = sha256_file(path)
    fallback = sha256_bytes(f"{path}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8"))
    return {
        "checkpoint_sha_or_mtime_size_hash": sha or fallback,
        "checkpoint_sha256": sha,
        "checkpoint_size_bytes": int(stat.st_size),
        "checkpoint_mtime_ns": int(stat.st_mtime_ns),
    }


def load_feature_matrices(
    *,
    dataset_dir: Path,
    graph_ids: list[str],
    max_cycle: int,
    max_chain: int,
    project_root: Path,
) -> dict[str, np.ndarray]:
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    import linear_probe_landscape as landscape

    probe = {
        "feature_names": ["utility", "recipient_cPRA"],
    }
    matrices: dict[str, np.ndarray] = {}
    for graph_id in sorted(set(graph_ids)):
        graph = landscape.load_graph(
            dataset_dir / graph_id,
            max_cycle=max_cycle,
            max_chain=max_chain,
        )
        matrices[graph_id] = np.asarray(landscape.feature_matrix(graph, probe), dtype=float)
    return matrices


def build_integrity_rows(
    solution_rows: list[dict[str, str]],
    *,
    feature_matrices: dict[str, Any],
    project_root: Path = PROJECT_ROOT,
    discovery_seeds: dict[str, int] | None = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    discovery_seeds = discovery_seeds or {}
    signature_field, _ = feature_matrix_fields(top_k)

    grouped_raw: dict[tuple[str, str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in solution_rows:
        grouped_raw[solution_group_key(row)].append(row)

    staged_rows: list[dict[str, Any]] = []
    predictions_by_key: dict[tuple[str, str, int, str], np.ndarray] = {}
    for key in sorted(grouped_raw, key=lambda item: (item[0], item[1], item[3], item[2])):
        regime, graph_id, subset_seed, method_label = key
        by_rank = rank_row_by_solution_rank(grouped_raw[key])
        rank1 = by_rank.get(1)
        if rank1 is None:
            continue

        theta = np.asarray(
            [parse_float(rank1["theta_1"]), parse_float(rank1["theta_2"])],
            dtype=float,
        )
        features = np.asarray(feature_matrices[graph_id], dtype=float)
        prediction = features @ theta
        predictions_by_key[key] = prediction

        checkpoint_path = resolve_existing_path(str(rank1["model_path"]), project_root)
        train_subset_path = checkpoint_path.parents[1] / "train_subset.json"
        subset_hash, subset_size = train_subset_hash(train_subset_path)
        signatures = [rank_signature(by_rank.get(rank)) for rank in range(1, int(top_k) + 1)]

        staged_rows.append(
            {
                "regime": regime,
                "graph_id": graph_id,
                "subset_seed": subset_seed,
                "method_label": method_label,
                "checkpoint_path": str(checkpoint_path),
                **checkpoint_hash_fields(checkpoint_path),
                "train_subset_path": str(train_subset_path),
                "train_subset_hash": subset_hash,
                "train_subset_size": subset_size,
                "prediction_vector_hash": prediction_vector_hash(prediction),
                "rank1_signature": rank_signature(by_rank.get(1)),
                "rank2_signature": rank_signature(by_rank.get(2)),
                "rank1_gap_pct": rank_gap_pct(by_rank.get(1)),
                "rank2_gap_pct": rank_gap_pct(by_rank.get(2)),
                signature_field: top_signature_hash(signatures),
            }
        )

    baseline_predictions: dict[tuple[str, str, str], np.ndarray] = {}
    for key, prediction in predictions_by_key.items():
        regime, graph_id, subset_seed, method_label = key
        discovery_seed = discovery_seeds.get(graph_id)
        if discovery_seed is not None and subset_seed == int(discovery_seed):
            baseline_predictions[(regime, graph_id, method_label)] = prediction

    if discovery_seeds:
        for key, prediction in predictions_by_key.items():
            regime, graph_id, subset_seed, method_label = key
            baseline_key = (regime, graph_id, method_label)
            if baseline_key not in baseline_predictions:
                # Fallback to seed 0 if the declared discovery seed is absent.
                zero_key = (regime, graph_id, 0, method_label)
                if zero_key in predictions_by_key:
                    baseline_predictions[baseline_key] = predictions_by_key[zero_key]

    output_rows: list[dict[str, Any]] = []
    for row in staged_rows:
        key = (
            str(row["regime"]),
            str(row["graph_id"]),
            int(row["subset_seed"]),
            str(row["method_label"]),
        )
        baseline = baseline_predictions.get((key[0], key[1], key[3]))
        prediction = predictions_by_key[key]
        if baseline is None:
            l2 = float("nan")
            corr = float("nan")
        else:
            l2 = float(np.linalg.norm(prediction - baseline))
            corr = vector_corr(prediction, baseline)
        row = dict(row)
        row["prediction_l2_from_discovery_seed"] = l2
        row["prediction_corr_with_discovery_seed"] = corr
        output_rows.append(row)

    return output_rows


def unique_count(rows: list[dict[str, Any]], field: str) -> int:
    return len({str(row.get(field, "")) for row in rows if str(row.get(field, "")) != ""})


def unique_checkpoint_hash_count(rows: list[dict[str, Any]]) -> int:
    values = []
    for row in rows:
        value = row.get("checkpoint_sha_or_mtime_size_hash") or row.get("checkpoint_sha256")
        if str(value or ""):
            values.append(str(value))
    return len(set(values))


def summarize_integrity_rows(rows: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
    _, unique_top_field = feature_matrix_fields(top_k)
    signature_field, _ = feature_matrix_fields(top_k)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["regime"]), str(row["graph_id"]), str(row["method_label"]))].append(row)

    summary_rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        regime, graph_id, method_label = key
        group = grouped[key]
        l2_values = [parse_float(row.get("prediction_l2_from_discovery_seed")) for row in group]
        corr_values = [
            parse_float(row.get("prediction_corr_with_discovery_seed")) for row in group
        ]
        summary_rows.append(
            {
                "regime": regime,
                "graph_id": graph_id,
                "method_label": method_label,
                "row_count": len(group),
                "unique_checkpoint_hashes": unique_checkpoint_hash_count(group),
                "unique_train_subset_hashes": unique_count(group, "train_subset_hash"),
                "unique_prediction_hashes": unique_count(group, "prediction_vector_hash"),
                "unique_rank1_signatures": unique_count(group, "rank1_signature"),
                "unique_rank2_signatures": unique_count(group, "rank2_signature"),
                unique_top_field: unique_count(group, signature_field),
                "mean_prediction_l2_from_discovery_seed": finite_mean(l2_values),
                "min_prediction_l2_from_discovery_seed": finite_min(l2_values),
                "max_prediction_l2_from_discovery_seed": finite_max(l2_values),
                "mean_prediction_corr_with_discovery_seed": finite_mean(corr_values),
                "min_prediction_corr_with_discovery_seed": finite_min(corr_values),
                "max_prediction_corr_with_discovery_seed": finite_max(corr_values),
            }
        )
    return summary_rows


def print_summary(rows: list[dict[str, Any]], top_k: int) -> None:
    _, unique_top_field = feature_matrix_fields(top_k)
    for row in rows:
        print(
            "{graph} {method}: checkpoints={ckpt} train_subsets={train} "
            "predictions={pred} rank1={rank1} rank2={rank2} topK={topk} "
            "l2_range=[{l2_min:.4g}, {l2_max:.4g}]".format(
                graph=row["graph_id"],
                method=row["method_label"],
                ckpt=row["unique_checkpoint_hashes"],
                train=row["unique_train_subset_hashes"],
                pred=row["unique_prediction_hashes"],
                rank1=row["unique_rank1_signatures"],
                rank2=row["unique_rank2_signatures"],
                topk=row[unique_top_field],
                l2_min=parse_float(row["min_prediction_l2_from_discovery_seed"]),
                l2_max=parse_float(row["max_prediction_l2_from_discovery_seed"]),
            )
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether fixed-graph all-subset_seed replay results used "
            "distinct checkpoints, train subsets, and prediction vectors."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-cycle", type=int, default=3)
    parser.add_argument("--max-chain", type=int, default=4)
    parser.add_argument(
        "--discovery-seed",
        nargs="*",
        default=["G-392.json=1", "G-1560.json=30"],
        help="Discovery seed mapping as GRAPH=SEED.",
    )
    args = parser.parse_args(argv)
    if args.top_k < 2:
        parser.error("--top-k must be >= 2")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    solution_rows = read_csv_rows(args.input)
    graph_ids = sorted({str(row["graph_id"]) for row in solution_rows})
    features = load_feature_matrices(
        dataset_dir=args.dataset_dir,
        graph_ids=graph_ids,
        max_cycle=args.max_cycle,
        max_chain=args.max_chain,
        project_root=args.project_root,
    )
    integrity_rows = build_integrity_rows(
        solution_rows,
        feature_matrices=features,
        project_root=args.project_root,
        discovery_seeds=parse_discovery_seeds(args.discovery_seed),
        top_k=args.top_k,
    )
    summary_rows = summarize_integrity_rows(integrity_rows, top_k=args.top_k)
    write_csv(args.output, integrity_rows, output_fields(args.top_k))
    write_csv(args.summary_output, summary_rows, summary_fields(args.top_k))
    print(f"Saved {len(integrity_rows)} integrity rows to {args.output}")
    print(f"Saved {len(summary_rows)} summary rows to {args.summary_output}")
    print_summary(summary_rows, top_k=args.top_k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
