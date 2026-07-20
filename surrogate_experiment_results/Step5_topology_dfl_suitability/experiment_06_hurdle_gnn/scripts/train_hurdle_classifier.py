#!/usr/bin/env python3
"""Train one zero/nonzero topology classifier for Experiment 06."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any

import numpy as np

import hurdle_common as common


def run_training(args: argparse.Namespace, graph_rows: list[dict[str, Any]], splits: dict[str, set[str]]) -> dict[str, Any]:
    import torch
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import RGCNConv, global_max_pool, global_mean_pool

    common.set_seed(args.seed, torch)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cpu")
    datasets, ordered_ids = common.build_datasets(graph_rows, splits, torch=torch, Data=Data)
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, generator=generator),
        "validation": DataLoader(datasets["validation"], batch_size=args.batch_size, shuffle=False),
        "test": DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False),
    }
    train_labels = np.asarray([float(item.is_nonzero.item()) for item in datasets["train"]])
    positive_count = int(np.sum(train_labels == 1))
    negative_count = int(np.sum(train_labels == 0))
    pos_weight = negative_count / positive_count
    model = common.make_model(
        torch=torch,
        RGCNConv=RGCNConv,
        global_mean_pool=global_mean_pool,
        global_max_pool=global_max_pool,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

    def predict(loader: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        model.eval()
        labels: list[float] = []
        probabilities: list[float] = []
        raw_targets: list[float] = []
        codes: list[int] = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                probability = torch.sigmoid(model(batch))
                labels.extend(batch.is_nonzero.view(-1).cpu().numpy().tolist())
                probabilities.extend(probability.cpu().numpy().tolist())
                raw_targets.extend(batch.y.view(-1).cpu().numpy().tolist())
                codes.extend(batch.topology_code.view(-1).cpu().numpy().tolist())
        return np.asarray(labels), np.asarray(probabilities), np.asarray(raw_targets), np.asarray(codes, dtype=int)

    best_epoch = -1
    best_validation_bce = float("inf")
    best_state: dict[str, Any] | None = None
    bad_epochs = 0
    curve = []
    training_started = time.perf_counter()
    for epoch in range(1, args.max_epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for batch in loaders["train"]:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), batch.is_nonzero.view(-1))
            loss.backward()
            optimizer.step()
            count = int(batch.num_graphs)
            loss_sum += float(loss.item()) * count
            sample_count += count
        model.eval()
        validation_loss_sum = 0.0
        validation_count = 0
        with torch.no_grad():
            for batch in loaders["validation"]:
                batch = batch.to(device)
                loss = criterion(model(batch), batch.is_nonzero.view(-1))
                count = int(batch.num_graphs)
                validation_loss_sum += float(loss.item()) * count
                validation_count += count
        validation_bce = validation_loss_sum / validation_count
        improved = validation_bce < best_validation_bce - args.early_stop_min_delta
        if improved:
            best_epoch = epoch
            best_validation_bce = validation_bce
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        curve.append(
            {
                "epoch": epoch,
                "train_weighted_bce": loss_sum / sample_count,
                "validation_weighted_bce": validation_bce,
                "best_validation_weighted_bce": best_validation_bce,
                "bad_epochs": bad_epochs,
                "epoch_seconds": time.perf_counter() - epoch_started,
            }
        )
        print(
            f"[hurdle-classifier] fold={args.fold} seed={args.seed} epoch={epoch} "
            f"train_bce={curve[-1]['train_weighted_bce']:.6f} validation_bce={validation_bce:.6f}",
            flush=True,
        )
        if bad_epochs >= args.early_stop_patience:
            break
    if best_state is None:
        raise RuntimeError("classifier selected no checkpoint")
    model.load_state_dict(best_state)
    validation_target, validation_probability, _, _ = predict(loaders["validation"])
    test_target, test_probability, test_raw_target, test_codes = predict(loaders["test"])
    validation_metrics = common.binary_metrics(validation_target, validation_probability, args.threshold)
    test_metrics = common.binary_metrics(test_target, test_probability, args.threshold)
    result = {
        "status": "success",
        "formal": True,
        "task": "zero_nonzero_classifier",
        "target": "is_nonzero_formal_label_mean_pp",
        "fold": args.fold,
        "validation_fold": (args.fold + 1) % 5,
        "seed": args.seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "split_sizes": {name: len(values) for name, values in datasets.items()},
        "train_class_counts": {"zero": negative_count, "nonzero": positive_count},
        "positive_class_weight": pos_weight,
        "epochs_completed": len(curve),
        "early_stop_triggered": len(curve) < args.max_epochs,
        "best_epoch": best_epoch,
        "best_validation_weighted_bce": best_validation_bce,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "timing": {"training_seconds": time.perf_counter() - training_started},
        "hyperparameters": {
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_epochs": args.max_epochs,
            "early_stop_patience": args.early_stop_patience,
            "early_stop_min_delta": args.early_stop_min_delta,
            "threshold": args.threshold,
            "threads": args.threads,
        },
    }
    predictions = []
    for target, probability, raw_target, code in zip(test_target, test_probability, test_raw_target, test_codes, strict=True):
        predictions.append(
            {
                "topology_id": ordered_ids[int(code)],
                "fold": args.fold,
                "seed": args.seed,
                "target_formal_label_mean_pp": float(raw_target),
                "target_is_nonzero": int(target),
                "probability_nonzero": float(probability),
                "predicted_is_nonzero": int(probability >= args.threshold),
            }
        )
    predictions.sort(key=lambda row: int(row["topology_id"].split("-")[-1]))
    common.atomic_write_csv(
        args.output_dir / "test_predictions.csv",
        predictions,
        [
            "topology_id",
            "fold",
            "seed",
            "target_formal_label_mean_pp",
            "target_is_nonzero",
            "probability_nonzero",
            "predicted_is_nonzero",
        ],
    )
    common.atomic_write_csv(
        args.output_dir / "training_curve.csv",
        curve,
        [
            "epoch",
            "train_weighted_bce",
            "validation_weighted_bce",
            "best_validation_weighted_bce",
            "bad_epochs",
            "epoch_seconds",
        ],
    )
    torch.save({"model_state_dict": best_state, "result": result}, args.output_dir / "best_model.pt")
    common.atomic_write_json(args.output_dir / "run_result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-jsonl", type=Path, default=common.DEFAULT_GRAPHS)
    parser.add_argument("--folds", type=Path, default=common.DEFAULT_FOLDS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--early-stop-patience", type=int, default=30)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0001)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    graph_rows = common.read_jsonl(args.graph_jsonl)
    fold_rows = common.read_csv(args.folds)
    audit = common.audit_inputs(graph_rows, fold_rows, args.graph_jsonl, args.folds)
    splits = common.split_topology_ids(fold_rows, test_fold=args.fold)
    plan = {
        "passed": audit["passed"],
        "execute": args.execute,
        "task": "zero_nonzero_classifier",
        "fold": args.fold,
        "validation_fold": (args.fold + 1) % 5,
        "seed": args.seed,
        "output_dir": str(args.output_dir),
        "input_audit": audit,
    }
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if not audit["passed"]:
        return 1
    if not args.execute:
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    common.atomic_write_json(args.output_dir / "run_plan.json", plan)
    result = run_training(args, graph_rows, splits)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
