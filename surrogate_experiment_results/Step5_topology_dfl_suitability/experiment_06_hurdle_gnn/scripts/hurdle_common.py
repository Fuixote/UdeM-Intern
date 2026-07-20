#!/usr/bin/env python3
"""Shared locked data/model/metric helpers for Experiment 06."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import random
import tempfile
from typing import Any

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
STEP5_ROOT = EXPERIMENT_ROOT.parent
EXP5_ROOT = STEP5_ROOT / "experiment_05_topology_gnn_regression"
DEFAULT_GRAPHS = (
    EXP5_ROOT
    / "results"
    / "multiseed_completion1880"
    / "results"
    / "formal_topology_incidence_graphs.jsonl"
)
DEFAULT_FOLDS = EXP5_ROOT / "results" / "formal_three_seed" / "splits" / "folds.csv"
DEFAULT_OUTPUT_ROOT = EXPERIMENT_ROOT / "results" / "smoke_fold0_seed42"
LOCKED_GRAPH_SHA256 = "8a41232110bf7f151c0192c6723fe5e0d7f4b9dd83aa70aba7fa142e503fd522"
LOCKED_FOLD_SHA256 = "b66a9c2e529e1c23fd96762340c07ddc9d3aa4f8c0a25aa16cc77f3dac7c3b39"
REGRESSION_SUBSETS = ("nonzero", "material")
OBJECTIVES = ("huber", "weighted_huber", "mse", "signed_log_mse", "huber_rank")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def atomic_write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    temporary.replace(path)


def set_seed(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_topology_ids(
    fold_rows: list[dict[str, str]],
    *,
    test_fold: int,
    fold_count: int = 5,
) -> dict[str, set[str]]:
    if test_fold not in range(fold_count):
        raise ValueError(f"test fold must be in [0,{fold_count - 1}]")
    validation_fold = (test_fold + 1) % fold_count
    output = {"train": set(), "validation": set(), "test": set()}
    seen: set[str] = set()
    for row in fold_rows:
        topology_id = row["topology_id"]
        fold = int(row["fold"])
        if topology_id in seen:
            raise ValueError(f"duplicate fold topology id:{topology_id}")
        if fold not in range(fold_count):
            raise ValueError(f"invalid fold:{topology_id}:{fold}")
        seen.add(topology_id)
        if fold == test_fold:
            output["test"].add(topology_id)
        elif fold == validation_fold:
            output["validation"].add(topology_id)
        else:
            output["train"].add(topology_id)
    if {name: len(ids) for name, ids in output.items()} != {"train": 600, "validation": 200, "test": 200}:
        raise ValueError("expected a 600/200/200 nested split")
    if any(output[a] & output[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("split overlap")
    return output


def audit_inputs(
    graph_rows: list[dict[str, Any]],
    fold_rows: list[dict[str, str]],
    graph_path: Path,
    fold_path: Path,
) -> dict[str, Any]:
    failures: list[str] = []
    graph_ids = [row.get("topology_id") for row in graph_rows]
    fold_ids = [row.get("topology_id") for row in fold_rows]
    graph_sha256 = sha256_file(graph_path)
    fold_sha256 = sha256_file(fold_path)
    if graph_sha256 != LOCKED_GRAPH_SHA256:
        failures.append(f"graph_sha256_mismatch:{graph_sha256}")
    if fold_sha256 != LOCKED_FOLD_SHA256:
        failures.append(f"fold_sha256_mismatch:{fold_sha256}")
    if len(graph_rows) != 1000 or len(set(graph_ids)) != 1000:
        failures.append(f"graph_count_or_uniqueness_mismatch:{len(graph_rows)}/{len(set(graph_ids))}")
    if len(fold_rows) != 1000 or len(set(fold_ids)) != 1000:
        failures.append(f"fold_count_or_uniqueness_mismatch:{len(fold_rows)}/{len(set(fold_ids))}")
    if set(graph_ids) != set(fold_ids):
        failures.append("graph_fold_topology_sets_differ")
    for row in graph_rows:
        target = row.get("target", {})
        if target.get("formal") is not True or target.get("name") != "formal_label_mean_pp":
            failures.append(f"nonformal_target:{row.get('topology_id')}")
        try:
            value = float(target.get("value"))
        except (TypeError, ValueError):
            failures.append(f"invalid_target:{row.get('topology_id')}")
        else:
            if not math.isfinite(value):
                failures.append(f"nonfinite_target:{row.get('topology_id')}")
    return {
        "passed": not failures,
        "graph_count": len(graph_rows),
        "fold_assignment_count": len(fold_rows),
        "graph_sha256": graph_sha256,
        "fold_sha256": fold_sha256,
        "target": "formal_label_mean_pp",
        "failures": failures,
    }


def ordered_topology_ids(graph_rows: list[dict[str, Any]]) -> list[str]:
    return sorted((row["topology_id"] for row in graph_rows), key=lambda value: int(value.split("-")[-1]))


def build_datasets(
    graph_rows: list[dict[str, Any]],
    splits: dict[str, set[str]],
    *,
    torch: Any,
    Data: Any,
) -> tuple[dict[str, list[Any]], list[str]]:
    graph_by_id = {row["topology_id"]: row for row in graph_rows}
    ordered_ids = ordered_topology_ids(graph_rows)
    topology_code = {topology_id: index for index, topology_id in enumerate(ordered_ids)}
    datasets: dict[str, list[Any]] = {name: [] for name in splits}
    for split_name, topology_ids in splits.items():
        for topology_id in ordered_ids:
            if topology_id not in topology_ids:
                continue
            row = graph_by_id[topology_id]
            features = row["node_features"]
            y = float(row["target"]["value"])
            datasets[split_name].append(
                Data(
                    x=torch.tensor(features, dtype=torch.float32),
                    edge_index=torch.tensor([row["edge_source"], row["edge_target"]], dtype=torch.long),
                    edge_type=torch.tensor(row["edge_type"], dtype=torch.long),
                    node_type=torch.tensor([0 if feature[0] == 1.0 else 1 for feature in features], dtype=torch.long),
                    y=torch.tensor([y], dtype=torch.float32),
                    is_nonzero=torch.tensor([float(y != 0.0)], dtype=torch.float32),
                    is_material=torch.tensor([float(abs(y) > 0.1)], dtype=torch.float32),
                    topology_code=torch.tensor([topology_code[topology_id]], dtype=torch.long),
                )
            )
    return datasets, ordered_ids


def make_model(
    *,
    torch: Any,
    RGCNConv: Any,
    global_mean_pool: Any,
    global_max_pool: Any,
    node_input_dim: int = 10,
    hidden_dim: int = 64,
    layers: int = 3,
    relation_count: int = 3,
    dropout: float = 0.1,
) -> Any:
    class IncidenceGNN(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            dimensions = [node_input_dim] + [hidden_dim] * layers
            self.convolutions = torch.nn.ModuleList(
                RGCNConv(dimensions[index], dimensions[index + 1], relation_count)
                for index in range(layers)
            )
            self.head = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim * 4, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(64, 1),
            )

        def forward(self, batch: Any) -> Any:
            hidden = batch.x
            for convolution in self.convolutions:
                hidden = torch.nn.functional.relu(convolution(hidden, batch.edge_index, batch.edge_type))
                hidden = torch.nn.functional.dropout(hidden, p=dropout, training=self.training)
            pooled = []
            for node_type in (0, 1):
                mask = batch.node_type == node_type
                pooled.append(global_mean_pool(hidden[mask], batch.batch[mask], size=batch.num_graphs))
                pooled.append(global_max_pool(hidden[mask], batch.batch[mask], size=batch.num_graphs))
            return self.head(torch.cat(pooled, dim=1)).view(-1)

    return IncidenceGNN()


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    output = np.empty(len(values), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        output[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return output


def correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    residual = target - prediction
    denominator = float(np.sum((target - np.mean(target)) ** 2))
    return {
        "count": len(target),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": None if denominator == 0 else 1.0 - float(np.sum(residual**2)) / denominator,
        "pearson": correlation(target, prediction),
        "spearman": correlation(ranks(target), ranks(prediction)),
    }


def binary_metrics(target: np.ndarray, probability: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    target = target.astype(int)
    prediction = (probability >= threshold).astype(int)
    tp = int(np.sum((target == 1) & (prediction == 1)))
    tn = int(np.sum((target == 0) & (prediction == 0)))
    fp = int(np.sum((target == 0) & (prediction == 1)))
    fn = int(np.sum((target == 1) & (prediction == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    order = np.argsort(-probability, kind="mergesort")
    ranked_target = target[order]
    positives = int(np.sum(target == 1))
    negatives = int(np.sum(target == 0))
    tpr = np.concatenate([[0.0], np.cumsum(ranked_target) / positives, [1.0]]) if positives else np.array([0.0, 1.0])
    fpr = np.concatenate([[0.0], np.cumsum(1 - ranked_target) / negatives, [1.0]]) if negatives else np.array([0.0, 1.0])
    auroc = float(np.trapezoid(tpr, fpr)) if positives and negatives else None
    cumulative_tp = np.cumsum(ranked_target)
    precision_curve = cumulative_tp / np.arange(1, len(target) + 1)
    average_precision = float(np.sum(precision_curve * ranked_target) / positives) if positives else None
    return {
        "count": len(target),
        "positive_count": positives,
        "negative_count": negatives,
        "threshold": threshold,
        "accuracy": (tp + tn) / len(target),
        "balanced_accuracy": (recall + specificity) / 2.0,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "auroc": auroc,
        "average_precision": average_precision,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def signed_log(value: Any, torch: Any) -> Any:
    return torch.sign(value) * torch.log1p(torch.abs(value))


def inverse_signed_log(value: Any, torch: Any) -> Any:
    return torch.sign(value) * torch.expm1(torch.abs(value))


def subset_mask(values: Any, subset: str) -> Any:
    if subset == "nonzero":
        return values != 0
    if subset == "material":
        return torch_abs(values) > 0.1
    raise ValueError(f"unknown regression subset:{subset}")


def torch_abs(values: Any) -> Any:
    return values.abs()
