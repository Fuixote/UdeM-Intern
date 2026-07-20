#!/usr/bin/env python3
"""Benchmark the formal Experiment 05 RGCN on CPU or CUDA."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_records(graph_path: Path, fold_path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records = [json.loads(line) for line in graph_path.read_text(encoding="utf-8").splitlines() if line]
    with fold_path.open(newline="", encoding="utf-8") as handle:
        folds = {row["topology_id"]: int(row["fold"]) for row in csv.DictReader(handle)}
    if len(records) != 1000 or len(folds) != 1000:
        raise ValueError(f"expected 1000 graphs/folds, observed {len(records)}/{len(folds)}")
    if {row["topology_id"] for row in records} != set(folds):
        raise ValueError("graph and fold topology sets differ")
    for row in records:
        target = row.get("target", {})
        if target.get("formal") is not True or target.get("name") != "formal_label_mean_pp":
            raise ValueError(f"{row['topology_id']}:non-formal target")
    return records, folds


def set_seed(seed: int, torch: Any) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset(
    records: list[dict[str, Any]],
    folds: dict[str, int],
    *,
    test_fold: int,
    validation_fold: int,
    torch: Any,
    Data: Any,
) -> tuple[dict[str, list[Any]], dict[str, float]]:
    split_records = {
        "train": [row for row in records if folds[row["topology_id"]] not in {test_fold, validation_fold}],
        "validation": [row for row in records if folds[row["topology_id"]] == validation_fold],
        "test": [row for row in records if folds[row["topology_id"]] == test_fold],
    }
    if {name: len(rows) for name, rows in split_records.items()} != {
        "train": 600,
        "validation": 200,
        "test": 200,
    }:
        raise ValueError("expected deterministic 600/200/200 split")

    train_features = torch.cat(
        [torch.tensor(row["node_features"], dtype=torch.float32)[:, 6:10] for row in split_records["train"]],
        dim=0,
    )
    continuous_mean = train_features.mean(dim=0)
    continuous_std = train_features.std(dim=0, unbiased=False)
    continuous_std[continuous_std == 0] = 1.0
    train_targets = torch.tensor(
        [float(row["target"]["value"]) for row in split_records["train"]],
        dtype=torch.float32,
    )
    target_mean = train_targets.mean()
    target_std = train_targets.std(unbiased=False)
    if float(target_std) == 0.0:
        target_std = torch.tensor(1.0)

    datasets: dict[str, list[Any]] = {}
    for split_name, rows in split_records.items():
        data_rows = []
        for row in rows:
            x = torch.tensor(row["node_features"], dtype=torch.float32)
            x[:, 6:10] = (x[:, 6:10] - continuous_mean) / continuous_std
            edge_index = torch.tensor(
                [row["edge_source"], row["edge_target"]],
                dtype=torch.long,
            )
            y_pp = float(row["target"]["value"])
            data_rows.append(
                Data(
                    x=x,
                    edge_index=edge_index,
                    edge_type=torch.tensor(row["edge_type"], dtype=torch.long),
                    node_type=torch.argmax(x[:, :2], dim=1),
                    y=torch.tensor([(y_pp - float(target_mean)) / float(target_std)], dtype=torch.float32),
                    y_pp=torch.tensor([y_pp], dtype=torch.float32),
                    topology_index=torch.tensor([int(row["topology_id"].split("-")[-1])], dtype=torch.long),
                )
            )
        datasets[split_name] = data_rows
    scaling = {
        "target_mean": float(target_mean),
        "target_std": float(target_std),
        "continuous_mean": [float(value) for value in continuous_mean],
        "continuous_std": [float(value) for value in continuous_std],
    }
    return datasets, scaling


def make_model(torch: Any, RGCNConv: Any, global_mean_pool: Any, global_max_pool: Any) -> Any:
    class FormalRGCN(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            dimensions = [10, 64, 64, 64]
            self.layers = torch.nn.ModuleList(
                [RGCNConv(dimensions[index], dimensions[index + 1], num_relations=3) for index in range(3)]
            )
            self.dropout = torch.nn.Dropout(0.1)
            self.head = torch.nn.Sequential(
                torch.nn.Linear(64 * 4, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 1),
            )

        def forward(self, batch: Any) -> Any:
            hidden = batch.x
            for layer in self.layers:
                hidden = self.dropout(torch.relu(layer(hidden, batch.edge_index, batch.edge_type)))
            graph_count = int(batch.num_graphs)
            pooled = []
            for node_type in (0, 1):
                mask = batch.node_type == node_type
                pooled.append(global_mean_pool(hidden[mask], batch.batch[mask], size=graph_count))
                pooled.append(global_max_pool(hidden[mask], batch.batch[mask], size=graph_count))
            return self.head(torch.cat(pooled, dim=1)).reshape(-1)

    return FormalRGCN()


def synchronize(device: str, torch: Any) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--test-fold", type=int, default=0)
    parser.add_argument("--validation-fold", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--measure-epochs", type=int, default=30)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    import torch.nn.functional as functional
    import torch_geometric
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import RGCNConv, global_max_pool, global_mean_pool

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but torch.cuda.is_available() is false")
    torch.set_num_threads(args.torch_threads)
    set_seed(args.seed, torch)
    load_started = time.perf_counter()
    records, folds = load_records(args.graphs, args.folds)
    datasets, scaling = build_dataset(
        records,
        folds,
        test_fold=args.test_fold,
        validation_fold=args.validation_fold,
        torch=torch,
        Data=Data,
    )
    load_seconds = time.perf_counter() - load_started
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, generator=generator)
    validation_loader = DataLoader(datasets["validation"], batch_size=args.batch_size, shuffle=False)
    device = torch.device(args.device)
    model = make_model(torch, RGCNConv, global_mean_pool, global_max_pool).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    target_mean = scaling["target_mean"]
    target_std = scaling["target_std"]
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def run_epoch() -> tuple[float, float]:
        model.train()
        losses = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)
            loss = functional.huber_loss(prediction, batch.y.reshape(-1))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        absolute_errors = []
        with torch.no_grad():
            for batch in validation_loader:
                batch = batch.to(device)
                prediction_pp = model(batch) * target_std + target_mean
                absolute_errors.extend(torch.abs(prediction_pp - batch.y_pp.reshape(-1)).cpu().tolist())
        return statistics.fmean(losses), statistics.fmean(absolute_errors)

    for _ in range(args.warmup_epochs):
        run_epoch()
    synchronize(args.device, torch)
    epoch_seconds = []
    last_loss = math.nan
    last_validation_mae = math.nan
    measured_started = time.perf_counter()
    for _ in range(args.measure_epochs):
        synchronize(args.device, torch)
        epoch_started = time.perf_counter()
        last_loss, last_validation_mae = run_epoch()
        synchronize(args.device, torch)
        epoch_seconds.append(time.perf_counter() - epoch_started)
    measured_seconds = time.perf_counter() - measured_started

    payload = {
        "passed": True,
        "device": args.device,
        "device_name": torch.cuda.get_device_name(0) if args.device == "cuda" else "CPU",
        "torch_version": torch.__version__,
        "torch_geometric_version": torch_geometric.__version__,
        "cuda_version": torch.version.cuda,
        "graph_sha256": file_sha256(args.graphs),
        "fold_sha256": file_sha256(args.folds),
        "test_fold": args.test_fold,
        "validation_fold": args.validation_fold,
        "train_graphs": len(datasets["train"]),
        "validation_graphs": len(datasets["validation"]),
        "test_graphs": len(datasets["test"]),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "warmup_epochs": args.warmup_epochs,
        "measure_epochs": args.measure_epochs,
        "torch_threads": args.torch_threads,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "load_seconds": load_seconds,
        "measured_seconds": measured_seconds,
        "epoch_seconds_mean": statistics.fmean(epoch_seconds),
        "epoch_seconds_median": statistics.median(epoch_seconds),
        "epoch_seconds_min": min(epoch_seconds),
        "epoch_seconds_max": max(epoch_seconds),
        "last_train_huber_loss_scaled": last_loss,
        "last_validation_mae_pp": last_validation_mae,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else 0,
        "scaling": scaling,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
