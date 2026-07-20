#!/usr/bin/env python3
"""Train one formal Experiment 05 relation-aware GNN fold/seed run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import tempfile
import time
from typing import Any

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRAPH_JSONL = (
    EXPERIMENT_ROOT
    / "results"
    / "multiseed_completion1880"
    / "results"
    / "formal_topology_incidence_graphs.jsonl"
)
DEFAULT_FOLDS = EXPERIMENT_ROOT / "results" / "formal_three_seed" / "splits" / "folds.csv"
DEFAULT_OUTPUT_ROOT = EXPERIMENT_ROOT / "results" / "formal_three_seed" / "gnn"


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
            raise ValueError(f"duplicate fold topology id: {topology_id}")
        if fold not in range(fold_count):
            raise ValueError(f"invalid fold id for {topology_id}: {fold}")
        seen.add(topology_id)
        if fold == test_fold:
            output["test"].add(topology_id)
        elif fold == validation_fold:
            output["validation"].add(topology_id)
        else:
            output["train"].add(topology_id)
    if set().union(*output.values()) != seen:
        raise AssertionError("split union mismatch")
    if any(output[left] & output[right] for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise AssertionError("split overlap")
    return output


def audit_inputs(
    graph_rows: list[dict[str, Any]],
    fold_rows: list[dict[str, str]],
    splits: dict[str, set[str]],
) -> dict[str, Any]:
    failures: list[str] = []
    graph_ids = [row.get("topology_id") for row in graph_rows]
    fold_ids = [row.get("topology_id") for row in fold_rows]
    if len(graph_rows) != 1000:
        failures.append(f"graph_count_mismatch:{len(graph_rows)}!=1000")
    if len(fold_rows) != 1000:
        failures.append(f"fold_count_mismatch:{len(fold_rows)}!=1000")
    if len(graph_ids) != len(set(graph_ids)):
        failures.append("graph_topology_ids_not_unique")
    if len(fold_ids) != len(set(fold_ids)):
        failures.append("fold_topology_ids_not_unique")
    if set(graph_ids) != set(fold_ids):
        failures.append("graph_fold_topology_sets_differ")
    for row in graph_rows:
        target = row.get("target", {})
        if target.get("name") != "formal_label_mean_pp" or target.get("formal") is not True:
            failures.append(f"nonformal_target:{row.get('topology_id')}")
        try:
            value = float(target.get("value"))
        except (TypeError, ValueError):
            failures.append(f"target_unavailable:{row.get('topology_id')}")
        else:
            if not math.isfinite(value):
                failures.append(f"target_nonfinite:{row.get('topology_id')}")
        if len(row.get("node_features", [])) != len(row.get("node_ids", [])):
            failures.append(f"node_feature_count_mismatch:{row.get('topology_id')}")
        if not (
            len(row.get("edge_source", []))
            == len(row.get("edge_target", []))
            == len(row.get("edge_type", []))
        ):
            failures.append(f"edge_count_mismatch:{row.get('topology_id')}")
    expected_sizes = {"train": 600, "validation": 200, "test": 200}
    observed_sizes = {name: len(values) for name, values in splits.items()}
    if observed_sizes != expected_sizes:
        failures.append(f"split_sizes_mismatch:{observed_sizes}!={expected_sizes}")
    return {
        "passed": not failures,
        "graph_count": len(graph_rows),
        "fold_assignment_count": len(fold_rows),
        "split_sizes": observed_sizes,
        "target": "formal_label_mean_pp",
        "failures": failures,
    }


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    temporary.replace(path)


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


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    residual = target - prediction
    denominator = float(np.sum((target - np.mean(target)) ** 2))
    pearson = None
    spearman = None
    if len(target) >= 2 and np.std(target) > 0 and np.std(prediction) > 0:
        pearson = float(np.corrcoef(target, prediction)[0, 1])
        spearman = float(np.corrcoef(ranks(target), ranks(prediction))[0, 1])
    return {
        "count": len(target),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": None if denominator == 0 else 1.0 - float(np.sum(residual**2)) / denominator,
        "pearson": pearson,
        "spearman": spearman,
    }


def run_training(args: argparse.Namespace, graph_rows: list[dict[str, Any]], splits: dict[str, set[str]]) -> dict[str, Any]:
    try:
        import torch
        from torch import nn
        import torch.nn.functional as functional
        from torch_geometric.data import Data
        from torch_geometric.loader import DataLoader
        from torch_geometric.nn import RGCNConv, global_max_pool, global_mean_pool
    except ImportError as exc:
        raise RuntimeError("formal GNN training requires torch and torch_geometric") from exc

    class FormalIncidenceGNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            dimensions = [args.node_input_dim] + [args.hidden_dim] * args.layers
            self.convolutions = nn.ModuleList(
                RGCNConv(dimensions[index], dimensions[index + 1], args.relation_count)
                for index in range(args.layers)
            )
            self.head = nn.Sequential(
                nn.Linear(args.hidden_dim * 4, 128),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(64, 1),
            )

        def forward(self, batch: Any) -> Any:
            hidden = batch.x
            for convolution in self.convolutions:
                hidden = functional.relu(convolution(hidden, batch.edge_index, batch.edge_type))
                hidden = functional.dropout(hidden, p=args.dropout, training=self.training)
            pooled = []
            for node_type in (0, 1):
                mask = batch.node_type == node_type
                pooled.append(global_mean_pool(hidden[mask], batch.batch[mask], size=batch.num_graphs))
                pooled.append(global_max_pool(hidden[mask], batch.batch[mask], size=batch.num_graphs))
            return self.head(torch.cat(pooled, dim=1)).view(-1)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cpu")

    graph_by_id = {row["topology_id"]: row for row in graph_rows}
    ordered_ids = sorted(graph_by_id, key=lambda value: int(value.split("-")[-1]))
    topology_index = {topology_id: index for index, topology_id in enumerate(ordered_ids)}
    datasets: dict[str, list[Any]] = {name: [] for name in splits}
    for split_name, topology_ids in splits.items():
        for topology_id in ordered_ids:
            if topology_id not in topology_ids:
                continue
            row = graph_by_id[topology_id]
            datasets[split_name].append(
                Data(
                    x=torch.tensor(row["node_features"], dtype=torch.float32),
                    edge_index=torch.tensor([row["edge_source"], row["edge_target"]], dtype=torch.long),
                    edge_type=torch.tensor(row["edge_type"], dtype=torch.long),
                    node_type=torch.tensor(
                        [0 if features[0] == 1.0 else 1 for features in row["node_features"]],
                        dtype=torch.long,
                    ),
                    y=torch.tensor([float(row["target"]["value"])], dtype=torch.float32),
                    topology_index=torch.tensor([topology_index[topology_id]], dtype=torch.long),
                )
            )

    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, generator=generator),
        "validation": DataLoader(datasets["validation"], batch_size=args.batch_size, shuffle=False),
        "test": DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False),
    }
    train_targets = np.asarray([float(item.y.item()) for item in datasets["train"]], dtype=float)
    target_mean = float(np.mean(train_targets))
    target_scale = float(np.std(train_targets)) or 1.0

    model = FormalIncidenceGNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.HuberLoss(delta=1.0)

    def predict(loader: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        model.eval()
        targets: list[float] = []
        predictions: list[float] = []
        indices: list[int] = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                scaled = model(batch)
                raw = scaled * target_scale + target_mean
                targets.extend(batch.y.detach().cpu().numpy().tolist())
                predictions.extend(raw.detach().cpu().numpy().tolist())
                indices.extend(batch.topology_index.detach().cpu().numpy().tolist())
        return np.asarray(targets), np.asarray(predictions), np.asarray(indices, dtype=int)

    best_epoch = -1
    best_validation_mae = float("inf")
    best_state: dict[str, Any] | None = None
    bad_epochs = 0
    curve: list[dict[str, Any]] = []
    training_started = time.perf_counter()
    for epoch in range(1, args.max_epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for batch in loaders["train"]:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction_scaled = model(batch)
            target_scaled = (batch.y.view(-1) - target_mean) / target_scale
            loss = criterion(prediction_scaled, target_scaled)
            loss.backward()
            optimizer.step()
            count = int(batch.num_graphs)
            loss_sum += float(loss.item()) * count
            sample_count += count
        validation_target, validation_prediction, _ = predict(loaders["validation"])
        validation_mae = float(np.mean(np.abs(validation_target - validation_prediction)))
        improved = validation_mae < best_validation_mae - args.early_stop_min_delta
        if improved:
            best_epoch = epoch
            best_validation_mae = validation_mae
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        curve.append(
            {
                "epoch": epoch,
                "train_huber_loss_standardized": loss_sum / sample_count,
                "validation_mae_pp": validation_mae,
                "best_validation_mae_pp": best_validation_mae,
                "bad_epochs": bad_epochs,
                "epoch_seconds": time.perf_counter() - epoch_started,
            }
        )
        print(
            f"[formal-gnn] fold={args.fold} seed={args.seed} epoch={epoch} "
            f"train_loss={curve[-1]['train_huber_loss_standardized']:.6f} "
            f"validation_mae_pp={validation_mae:.6f} seconds={curve[-1]['epoch_seconds']:.3f}",
            flush=True,
        )
        if bad_epochs >= args.early_stop_patience:
            break

    if best_state is None:
        raise RuntimeError("no best model checkpoint was selected")
    model.load_state_dict(best_state)
    test_target, test_prediction, test_indices = predict(loaders["test"])
    test_metrics = regression_metrics(test_target, test_prediction)
    epoch_seconds = [float(row["epoch_seconds"]) for row in curve]
    steady_seconds = epoch_seconds[1:] if len(epoch_seconds) > 1 else epoch_seconds
    mean_steady_epoch_seconds = statistics.fmean(steady_seconds)
    result = {
        "status": "success",
        "formal": True,
        "target": "formal_label_mean_pp",
        "fold": args.fold,
        "validation_fold": (args.fold + 1) % 5,
        "seed": args.seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "split_sizes": {name: len(values) for name, values in datasets.items()},
        "epochs_completed": len(curve),
        "early_stop_triggered": len(curve) < args.max_epochs,
        "best_epoch": best_epoch,
        "best_validation_mae_pp": best_validation_mae,
        "test_metrics": test_metrics,
        "target_standardization": {"training_mean_pp": target_mean, "training_std_pp": target_scale},
        "timing": {
            "training_seconds": time.perf_counter() - training_started,
            "first_epoch_seconds": epoch_seconds[0],
            "mean_steady_epoch_seconds": mean_steady_epoch_seconds,
            "median_steady_epoch_seconds": statistics.median(steady_seconds),
            "projected_seconds_per_500_epoch_run": mean_steady_epoch_seconds * 500,
            "projected_seconds_15_runs_sequential": mean_steady_epoch_seconds * 500 * 15,
            "projected_seconds_15_runs_at_3_workers": mean_steady_epoch_seconds * 500 * 15 / 3,
        },
        "hyperparameters": {
            "node_input_dim": args.node_input_dim,
            "hidden_dim": args.hidden_dim,
            "layers": args.layers,
            "relation_count": args.relation_count,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_epochs": args.max_epochs,
            "early_stop_patience": args.early_stop_patience,
            "early_stop_min_delta": args.early_stop_min_delta,
            "threads": args.threads,
        },
    }
    prediction_rows = []
    for target, prediction, index in zip(test_target, test_prediction, test_indices, strict=True):
        prediction_rows.append(
            {
                "topology_id": ordered_ids[int(index)],
                "fold": args.fold,
                "seed": args.seed,
                "target_formal_label_mean_pp": float(target),
                "prediction_formal_label_mean_pp": float(prediction),
            }
        )
    prediction_rows.sort(key=lambda row: int(row["topology_id"].split("-")[-1]))
    atomic_write_csv(args.output_dir / "training_curve.csv", curve)
    atomic_write_csv(args.output_dir / "test_predictions.csv", prediction_rows)
    torch.save(
        {
            "model_state_dict": best_state,
            "target_mean": target_mean,
            "target_scale": target_scale,
            "result": result,
        },
        args.output_dir / "best_model.pt",
    )
    atomic_write_json(args.output_dir / "run_result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-jsonl", type=Path, default=DEFAULT_GRAPH_JSONL)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--node-input-dim", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--relation-count", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--early-stop-patience", type=int, default=30)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0001)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_started = time.perf_counter()
    graph_rows = read_jsonl(args.graph_jsonl)
    fold_rows = read_csv(args.folds)
    splits = split_topology_ids(fold_rows, test_fold=args.fold)
    input_audit = audit_inputs(graph_rows, fold_rows, splits)
    plan = {
        "execute": bool(args.execute),
        "graph_jsonl": str(args.graph_jsonl),
        "graph_sha256": sha256_file(args.graph_jsonl),
        "folds": str(args.folds),
        "folds_sha256": sha256_file(args.folds),
        "fold": args.fold,
        "validation_fold": (args.fold + 1) % 5,
        "seed": args.seed,
        "max_epochs": args.max_epochs,
        "output_dir": str(args.output_dir),
        "input_load_seconds": time.perf_counter() - input_started,
        "input_audit": input_audit,
    }
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if not input_audit["passed"]:
        return 1
    if not args.execute:
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_dir / "run_plan.json", plan)
    result = run_training(args, graph_rows, splits)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
