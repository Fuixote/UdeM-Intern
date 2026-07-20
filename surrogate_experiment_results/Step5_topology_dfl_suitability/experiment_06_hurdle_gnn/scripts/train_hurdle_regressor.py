#!/usr/bin/env python3
"""Train one nonzero/material stage-two topology regressor."""

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


def keep_item(item: Any, subset: str) -> bool:
    value = float(item.y.item())
    if subset == "nonzero":
        return value != 0.0
    if subset == "material":
        return abs(value) > 0.1
    raise ValueError(f"unknown subset:{subset}")


def pairwise_ranking_loss(prediction: Any, target: Any, functional: Any) -> Any:
    if len(prediction) < 2:
        return prediction.sum() * 0.0
    row, column = target.new_ones((len(target), len(target))).triu(diagonal=1).nonzero(as_tuple=True)
    target_difference = target[row] - target[column]
    usable = target_difference.abs() > 1e-6
    if not bool(usable.any()):
        return prediction.sum() * 0.0
    prediction_difference = prediction[row[usable]] - prediction[column[usable]]
    direction = target_difference[usable].sign()
    return functional.softplus(-direction * prediction_difference).mean()


def run_training(args: argparse.Namespace, graph_rows: list[dict[str, Any]], splits: dict[str, set[str]]) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import RGCNConv, global_max_pool, global_mean_pool

    common.set_seed(args.seed, torch)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cpu")
    datasets, ordered_ids = common.build_datasets(graph_rows, splits, torch=torch, Data=Data)
    selected = {
        "train": [item for item in datasets["train"] if keep_item(item, args.regression_subset)],
        "validation": [item for item in datasets["validation"] if keep_item(item, args.regression_subset)],
        "test": datasets["test"],
    }
    if len(selected["train"]) < 2 or len(selected["validation"]) < 1:
        raise ValueError(f"insufficient {args.regression_subset} examples")
    raw_train_target = torch.tensor([float(item.y.item()) for item in selected["train"]], dtype=torch.float32)
    transformed_train_target = (
        common.signed_log(raw_train_target, torch) if args.objective == "signed_log_mse" else raw_train_target
    )
    target_mean = transformed_train_target.mean()
    target_scale = transformed_train_target.std(unbiased=False)
    if float(target_scale) == 0.0:
        target_scale = torch.tensor(1.0)
    magnitude_q90 = float(torch.quantile(raw_train_target.abs(), 0.9)) or 1.0
    raw_weights = 1.0 + torch.clamp(raw_train_target.abs() / magnitude_q90, max=1.0)
    weight_normalizer = float(raw_weights.mean())
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        "train": DataLoader(selected["train"], batch_size=args.batch_size, shuffle=True, generator=generator),
        "validation": DataLoader(selected["validation"], batch_size=args.batch_size, shuffle=False),
        "test": DataLoader(selected["test"], batch_size=args.batch_size, shuffle=False),
    }
    model = common.make_model(
        torch=torch,
        RGCNConv=RGCNConv,
        global_mean_pool=global_mean_pool,
        global_max_pool=global_max_pool,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    def transform(raw: Any) -> Any:
        value = common.signed_log(raw, torch) if args.objective == "signed_log_mse" else raw
        return (value - target_mean) / target_scale

    def inverse(prediction: Any) -> Any:
        transformed = prediction * target_scale + target_mean
        return common.inverse_signed_log(transformed, torch) if args.objective == "signed_log_mse" else transformed

    def objective_loss(prediction: Any, raw_target: Any) -> tuple[Any, Any, Any]:
        target = transform(raw_target)
        if args.objective == "mse" or args.objective == "signed_log_mse":
            pointwise = functional.mse_loss(prediction, target)
            ranking = prediction.sum() * 0.0
        elif args.objective == "weighted_huber":
            loss_values = functional.huber_loss(prediction, target, reduction="none")
            weights = (1.0 + torch.clamp(raw_target.abs() / magnitude_q90, max=1.0)) / weight_normalizer
            pointwise = (loss_values * weights).mean()
            ranking = prediction.sum() * 0.0
        elif args.objective == "huber_rank":
            pointwise = functional.huber_loss(prediction, target)
            ranking = pairwise_ranking_loss(prediction, target, functional)
        elif args.objective == "huber":
            pointwise = functional.huber_loss(prediction, target)
            ranking = prediction.sum() * 0.0
        else:
            raise ValueError(f"unknown objective:{args.objective}")
        total = pointwise + args.ranking_weight * ranking
        return total, pointwise, ranking

    def predict(loader: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        model.eval()
        targets: list[float] = []
        predictions: list[float] = []
        codes: list[int] = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                prediction = inverse(model(batch))
                targets.extend(batch.y.view(-1).cpu().numpy().tolist())
                predictions.extend(prediction.cpu().numpy().tolist())
                codes.extend(batch.topology_code.view(-1).cpu().numpy().tolist())
        return np.asarray(targets), np.asarray(predictions), np.asarray(codes, dtype=int)

    best_epoch = -1
    best_validation_mae = float("inf")
    best_state: dict[str, Any] | None = None
    bad_epochs = 0
    curve = []
    training_started = time.perf_counter()
    for epoch in range(1, args.max_epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        total_sum = 0.0
        point_sum = 0.0
        rank_sum = 0.0
        batch_count = 0
        for batch in loaders["train"]:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            total, pointwise, ranking = objective_loss(model(batch), batch.y.view(-1))
            total.backward()
            optimizer.step()
            total_sum += float(total.item())
            point_sum += float(pointwise.item())
            rank_sum += float(ranking.item())
            batch_count += 1
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
                "train_total_loss": total_sum / batch_count,
                "train_pointwise_loss": point_sum / batch_count,
                "train_ranking_loss": rank_sum / batch_count,
                "validation_subset_mae_pp": validation_mae,
                "best_validation_subset_mae_pp": best_validation_mae,
                "bad_epochs": bad_epochs,
                "epoch_seconds": time.perf_counter() - epoch_started,
            }
        )
        print(
            f"[hurdle-regressor] fold={args.fold} seed={args.seed} subset={args.regression_subset} "
            f"objective={args.objective} epoch={epoch} validation_mae_pp={validation_mae:.6f}",
            flush=True,
        )
        if bad_epochs >= args.early_stop_patience:
            break
    if best_state is None:
        raise RuntimeError("regressor selected no checkpoint")
    model.load_state_dict(best_state)
    test_target, regression_prediction, test_codes = predict(loaders["test"])
    classifier_rows = common.read_csv(args.classifier_predictions)
    classifier_by_id = {row["topology_id"]: row for row in classifier_rows}
    test_ids = [ordered_ids[int(code)] for code in test_codes]
    if len(classifier_rows) != 200 or len(classifier_by_id) != 200 or set(classifier_by_id) != set(test_ids):
        raise ValueError("classifier/test topology mismatch")
    for row in classifier_rows:
        if int(row["fold"]) != args.fold or int(row["seed"]) != args.seed:
            raise ValueError("classifier fold/seed mismatch")
    probability = np.asarray([float(classifier_by_id[topology_id]["probability_nonzero"]) for topology_id in test_ids])
    hard_prediction = np.where(probability >= args.classifier_threshold, regression_prediction, 0.0)
    soft_prediction = probability * regression_prediction
    oracle_prediction = np.where(test_target != 0.0, regression_prediction, 0.0)
    metrics = []
    subsets = {
        "all": np.ones(len(test_target), dtype=bool),
        "zero": test_target == 0.0,
        "nonzero": test_target != 0.0,
        "material_abs_gt_0.1pp": np.abs(test_target) > 0.1,
    }
    prediction_modes = {
        "raw_regressor": regression_prediction,
        "hard_hurdle": hard_prediction,
        "soft_hurdle": soft_prediction,
        "oracle_nonzero_gate": oracle_prediction,
    }
    for mode, predictions in prediction_modes.items():
        for subset, mask in subsets.items():
            metrics.append({"prediction_mode": mode, "subset": subset, **common.regression_metrics(test_target[mask], predictions[mask])})
    result = {
        "status": "success",
        "formal": True,
        "task": "subset_regressor",
        "target": "formal_label_mean_pp",
        "fold": args.fold,
        "validation_fold": (args.fold + 1) % 5,
        "seed": args.seed,
        "regression_subset": args.regression_subset,
        "objective": args.objective,
        "classifier_predictions": str(args.classifier_predictions),
        "classifier_predictions_sha256": common.sha256_file(args.classifier_predictions),
        "device": str(device),
        "torch_version": torch.__version__,
        "full_split_sizes": {name: len(values) for name, values in datasets.items()},
        "regression_split_sizes": {name: len(values) for name, values in selected.items()},
        "epochs_completed": len(curve),
        "early_stop_triggered": len(curve) < args.max_epochs,
        "best_epoch": best_epoch,
        "best_validation_subset_mae_pp": best_validation_mae,
        "target_transform": "signed_log1p" if args.objective == "signed_log_mse" else "identity",
        "target_scaling": {"training_mean": float(target_mean), "training_std": float(target_scale)},
        "magnitude_weighting": {
            "enabled": args.objective == "weighted_huber",
            "formula": "(1 + min(abs(y)/train_q90_abs,1))/train_mean_weight",
            "train_q90_abs_pp": magnitude_q90,
            "train_mean_weight": weight_normalizer,
        },
        "ranking_weight": args.ranking_weight if args.objective == "huber_rank" else 0.0,
        "classifier_threshold": args.classifier_threshold,
        "test_metrics": metrics,
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
            "threads": args.threads,
        },
    }
    predictions = []
    for index, topology_id in enumerate(test_ids):
        predictions.append(
            {
                "topology_id": topology_id,
                "fold": args.fold,
                "seed": args.seed,
                "regression_subset": args.regression_subset,
                "objective": args.objective,
                "target_formal_label_mean_pp": float(test_target[index]),
                "probability_nonzero": float(probability[index]),
                "raw_regression_prediction_pp": float(regression_prediction[index]),
                "hard_hurdle_prediction_pp": float(hard_prediction[index]),
                "soft_hurdle_prediction_pp": float(soft_prediction[index]),
                "oracle_nonzero_gate_prediction_pp": float(oracle_prediction[index]),
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
            "regression_subset",
            "objective",
            "target_formal_label_mean_pp",
            "probability_nonzero",
            "raw_regression_prediction_pp",
            "hard_hurdle_prediction_pp",
            "soft_hurdle_prediction_pp",
            "oracle_nonzero_gate_prediction_pp",
        ],
    )
    common.atomic_write_csv(
        args.output_dir / "training_curve.csv",
        curve,
        [
            "epoch",
            "train_total_loss",
            "train_pointwise_loss",
            "train_ranking_loss",
            "validation_subset_mae_pp",
            "best_validation_subset_mae_pp",
            "bad_epochs",
            "epoch_seconds",
        ],
    )
    torch.save(
        {
            "model_state_dict": best_state,
            "target_mean": float(target_mean),
            "target_scale": float(target_scale),
            "result": result,
        },
        args.output_dir / "best_model.pt",
    )
    common.atomic_write_json(args.output_dir / "run_result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-jsonl", type=Path, default=common.DEFAULT_GRAPHS)
    parser.add_argument("--folds", type=Path, default=common.DEFAULT_FOLDS)
    parser.add_argument("--classifier-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--regression-subset", choices=common.REGRESSION_SUBSETS, required=True)
    parser.add_argument("--objective", choices=common.OBJECTIVES, required=True)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--early-stop-patience", type=int, default=30)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0001)
    parser.add_argument("--ranking-weight", type=float, default=0.25)
    parser.add_argument("--classifier-threshold", type=float, default=0.5)
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
        "passed": audit["passed"] and args.classifier_predictions.is_file(),
        "execute": args.execute,
        "task": "subset_regressor",
        "fold": args.fold,
        "validation_fold": (args.fold + 1) % 5,
        "seed": args.seed,
        "regression_subset": args.regression_subset,
        "objective": args.objective,
        "classifier_predictions": str(args.classifier_predictions),
        "classifier_predictions_available": args.classifier_predictions.is_file(),
        "output_dir": str(args.output_dir),
        "input_audit": audit,
    }
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if not plan["passed"]:
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
