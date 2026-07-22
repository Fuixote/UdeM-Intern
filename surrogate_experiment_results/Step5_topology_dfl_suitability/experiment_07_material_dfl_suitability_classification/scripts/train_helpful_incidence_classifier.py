#!/usr/bin/env python3
"""Train one primary helpful-vs-rest incidence GNN with nested decisions."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np

import helpful_policy_common as policy
import material_common as common
import train_incidence_classifier as three_class_gnn


DEFAULT_LABELS = common.DEFAULT_OUTPUT_ROOT / "labels" / "material_labels.csv"
DEFAULT_FOLDS = common.DEFAULT_OUTPUT_ROOT / "splits" / "material_folds.csv"
DEFAULT_LABEL_AUDIT = common.DEFAULT_OUTPUT_ROOT / "labels" / "material_labels.audit.json"
DEFAULT_FOLD_AUDIT = common.DEFAULT_OUTPUT_ROOT / "splits" / "material_folds.audit.json"


def build_datasets(
    graph_rows: list[dict[str, Any]],
    label_rows: list[dict[str, str]],
    splits: dict[str, set[str]],
    *,
    torch: Any,
    Data: Any,
) -> tuple[dict[str, list[Any]], list[str]]:
    graph_by_id = {row["topology_id"]: row for row in graph_rows}
    label_by_id = {row["topology_id"]: row for row in label_rows}
    ordered_ids = sorted(graph_by_id, key=common.topology_sort_key)
    topology_code = {
        topology_id: index for index, topology_id in enumerate(ordered_ids)
    }
    datasets: dict[str, list[Any]] = {name: [] for name in splits}
    for split_name, topology_ids in splits.items():
        for topology_id in ordered_ids:
            if topology_id not in topology_ids:
                continue
            graph = graph_by_id[topology_id]
            label = label_by_id[topology_id]
            features = graph["node_features"]
            datasets[split_name].append(
                Data(
                    x=torch.tensor(features, dtype=torch.float32),
                    edge_index=torch.tensor(
                        [graph["edge_source"], graph["edge_target"]],
                        dtype=torch.long,
                    ),
                    edge_type=torch.tensor(graph["edge_type"], dtype=torch.long),
                    node_type=torch.tensor(
                        [0 if feature[0] == 1.0 else 1 for feature in features],
                        dtype=torch.long,
                    ),
                    y_helpful=torch.tensor(
                        [1.0 if label["primary_label"] == "material_helpful" else 0.0],
                        dtype=torch.float32,
                    ),
                    raw_delta_pp=torch.tensor(
                        [float(label["formal_label_mean_pp"])],
                        dtype=torch.float32,
                    ),
                    topology_code=torch.tensor(
                        [topology_code[topology_id]], dtype=torch.long
                    ),
                )
            )
    return datasets, ordered_ids


def make_model(
    *,
    torch: Any,
    RGCNConv: Any,
    global_mean_pool: Any,
    global_max_pool: Any,
    hidden_dim: int,
    layers: int,
    dropout: float,
) -> Any:
    class HelpfulIncidenceGNN(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            dimensions = [10] + [hidden_dim] * layers
            self.convolutions = torch.nn.ModuleList(
                RGCNConv(dimensions[index], dimensions[index + 1], 3)
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
                hidden = torch.nn.functional.relu(
                    convolution(hidden, batch.edge_index, batch.edge_type)
                )
                hidden = torch.nn.functional.dropout(
                    hidden, p=dropout, training=self.training
                )
            pooled = []
            for node_type in (0, 1):
                mask = batch.node_type == node_type
                pooled.append(
                    global_mean_pool(
                        hidden[mask], batch.batch[mask], size=batch.num_graphs
                    )
                )
                pooled.append(
                    global_max_pool(
                        hidden[mask], batch.batch[mask], size=batch.num_graphs
                    )
                )
            return self.head(torch.cat(pooled, dim=1)).view(-1)

    return HelpfulIncidenceGNN()


def evaluate_predictions(
    target: np.ndarray,
    delta: np.ndarray,
    raw_probability: np.ndarray,
    calibrated_probability: np.ndarray,
    regret_selected: np.ndarray,
    precision_selected: dict[float, np.ndarray],
    *,
    compute_costs_pp: tuple[float, ...],
) -> dict[str, Any]:
    result = {
        "predictive_raw": policy.binary_predictive_metrics(target, raw_probability),
        "predictive_calibrated": policy.binary_predictive_metrics(
            target, calibrated_probability, selected=regret_selected
        ),
        "regret_policy": common.policy_metrics_from_selection(
            delta,
            target,
            regret_selected,
            selection_rule="frozen_validation_regret_threshold",
            compute_costs_pp=compute_costs_pp,
        ),
        "precision_constrained_policies": {},
        "top_k": policy.ranking_capture_metrics(delta, calibrated_probability),
        "outlier_sensitivity": policy.outlier_sensitivity(delta, regret_selected),
    }
    for constraint, selected in precision_selected.items():
        result["precision_constrained_policies"][str(constraint)] = (
            common.policy_metrics_from_selection(
                delta,
                target,
                selected,
                selection_rule=f"frozen_validation_precision_{constraint}",
                compute_costs_pp=compute_costs_pp,
            )
        )
    return result


def run_training(
    args: argparse.Namespace,
    graph_rows: list[dict[str, Any]],
    label_rows: list[dict[str, str]],
    splits: dict[str, set[str]],
) -> dict[str, Any]:
    import torch
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import RGCNConv, global_max_pool, global_mean_pool

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cpu")
    datasets, ordered_ids = build_datasets(
        graph_rows, label_rows, splits, torch=torch, Data=Data
    )
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
        ),
        "validation": DataLoader(
            datasets["validation"], batch_size=args.batch_size, shuffle=False
        ),
        "test": DataLoader(
            datasets["test"], batch_size=args.batch_size, shuffle=False
        ),
    }
    train_target = np.asarray(
        [float(item.y_helpful.item()) for item in datasets["train"]], dtype=float
    )
    train_counts = Counter(int(value) for value in train_target)
    positive_weight = train_counts[0] / train_counts[1]
    model = make_model(
        torch=torch,
        RGCNConv=RGCNConv,
        global_mean_pool=global_mean_pool,
        global_max_pool=global_max_pool,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, dtype=torch.float32, device=device)
    )

    def predict(loader: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        model.eval()
        targets: list[float] = []
        logits: list[float] = []
        deltas: list[float] = []
        codes: list[int] = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                logits.extend(model(batch).cpu().numpy().tolist())
                targets.extend(batch.y_helpful.view(-1).cpu().numpy().tolist())
                deltas.extend(batch.raw_delta_pp.view(-1).cpu().numpy().tolist())
                codes.extend(batch.topology_code.view(-1).cpu().numpy().tolist())
        return (
            np.asarray(targets, dtype=bool),
            np.asarray(logits, dtype=float),
            np.asarray(deltas, dtype=float),
            np.asarray(codes, dtype=int),
        )

    best_epoch = -1
    best_validation_loss = float("inf")
    best_state: dict[str, Any] | None = None
    bad_epochs = 0
    curve: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, args.max_epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch in loaders["train"]:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            batch_logits = model(batch)
            loss = criterion(batch_logits, batch.y_helpful.view(-1))
            loss.backward()
            optimizer.step()
            count = int(batch.num_graphs)
            train_loss_sum += float(loss.item()) * count
            train_count += count
        model.eval()
        validation_loss_sum = 0.0
        validation_count = 0
        with torch.no_grad():
            for batch in loaders["validation"]:
                batch = batch.to(device)
                loss = criterion(model(batch), batch.y_helpful.view(-1))
                count = int(batch.num_graphs)
                validation_loss_sum += float(loss.item()) * count
                validation_count += count
        validation_loss = validation_loss_sum / validation_count
        if validation_loss < best_validation_loss - args.early_stop_min_delta:
            best_epoch = epoch
            best_validation_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
        curve.append(
            {
                "epoch": epoch,
                "train_weighted_bce": train_loss_sum / train_count,
                "validation_weighted_bce": validation_loss,
                "best_validation_weighted_bce": best_validation_loss,
                "bad_epochs": bad_epochs,
                "epoch_seconds": time.perf_counter() - epoch_started,
            }
        )
        print(
            f"[helpful-incidence] fold={args.fold} seed={args.seed} epoch={epoch} "
            f"train_bce={curve[-1]['train_weighted_bce']:.6f} "
            f"validation_bce={validation_loss:.6f}",
            flush=True,
        )
        if bad_epochs >= args.early_stop_patience:
            break
    if best_state is None:
        raise RuntimeError("no helpful classifier checkpoint was selected")
    model.load_state_dict(best_state)
    validation_target, validation_logits, validation_delta, _ = predict(
        loaders["validation"]
    )
    test_target, test_logits, test_delta, test_codes = predict(loaders["test"])
    calibration = policy.fit_temperature(validation_target, validation_logits)
    validation_probability = policy.apply_temperature(
        validation_logits, calibration["temperature"]
    )
    test_probability = policy.apply_temperature(
        test_logits, calibration["temperature"]
    )
    regret_rule = policy.select_regret_threshold(
        validation_delta,
        validation_target,
        validation_probability,
        compute_cost_pp=args.regret_compute_cost_pp,
        compute_costs_pp=args.compute_costs_pp,
    )
    precision_rules = {
        constraint: policy.select_precision_constrained_threshold(
            validation_delta,
            validation_target,
            validation_probability,
            minimum_precision=constraint,
            compute_costs_pp=args.compute_costs_pp,
        )
        for constraint in args.precision_constraints
    }
    validation_regret_selected = validation_probability >= regret_rule["threshold"]
    test_regret_selected = test_probability >= regret_rule["threshold"]
    validation_precision_selected = {
        constraint: validation_probability >= rule["threshold"]
        for constraint, rule in precision_rules.items()
    }
    test_precision_selected = {
        constraint: test_probability >= rule["threshold"]
        for constraint, rule in precision_rules.items()
    }
    result = {
        "status": "success",
        "formal": args.formal,
        "task": "material_helpful_vs_non_helpful",
        "model": "relation_aware_incidence_gnn",
        "fold": args.fold,
        "validation_fold": (args.fold + 1) % 5,
        "seed": args.seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "split_sizes": {name: len(values) for name, values in datasets.items()},
        "train_class_counts": {
            "non_helpful": train_counts[0],
            "material_helpful": train_counts[1],
        },
        "positive_weight": positive_weight,
        "epochs_completed": len(curve),
        "early_stop_triggered": len(curve) < args.max_epochs,
        "best_epoch": best_epoch,
        "best_validation_weighted_bce": best_validation_loss,
        "temperature_calibration": calibration,
        "decision_protocol": {
            "regret": regret_rule,
            "precision_constraints": {
                str(key): value for key, value in precision_rules.items()
            },
            "test_fold_never_used_for_checkpoint_calibration_or_threshold": True,
        },
        "validation_metrics": evaluate_predictions(
            validation_target,
            validation_delta,
            policy.sigmoid(validation_logits),
            validation_probability,
            validation_regret_selected,
            validation_precision_selected,
            compute_costs_pp=args.compute_costs_pp,
        ),
        "test_metrics": evaluate_predictions(
            test_target,
            test_delta,
            policy.sigmoid(test_logits),
            test_probability,
            test_regret_selected,
            test_precision_selected,
            compute_costs_pp=args.compute_costs_pp,
        ),
        "timing": {"training_seconds": time.perf_counter() - started},
        "hyperparameters": {
            "hidden_dim": args.hidden_dim,
            "layers": args.layers,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_epochs": args.max_epochs,
            "early_stop_patience": args.early_stop_patience,
            "early_stop_min_delta": args.early_stop_min_delta,
            "regret_compute_cost_pp": args.regret_compute_cost_pp,
            "precision_constraints": list(args.precision_constraints),
            "compute_costs_pp": list(args.compute_costs_pp),
            "threads": args.threads,
        },
    }
    prediction_rows: list[dict[str, Any]] = []
    for target, logit_value, probability_value, delta, code in zip(
        test_target,
        test_logits,
        test_probability,
        test_delta,
        test_codes,
        strict=True,
    ):
        row = {
            "topology_id": ordered_ids[int(code)],
            "fold": args.fold,
            "seed": args.seed,
            "target_is_material_helpful": int(target),
            "formal_label_mean_pp": float(delta),
            "raw_logit_helpful": float(logit_value),
            "raw_probability_helpful": float(policy.sigmoid(np.asarray([logit_value]))[0]),
            "calibrated_probability_helpful": float(probability_value),
            "temperature": calibration["temperature"],
            "regret_threshold": regret_rule["threshold"],
            "selected_regret": int(probability_value >= regret_rule["threshold"]),
        }
        for constraint, rule in precision_rules.items():
            key = str(constraint).replace(".", "_")
            row[f"precision_{key}_threshold"] = rule["threshold"]
            row[f"selected_precision_{key}"] = int(
                probability_value >= rule["threshold"]
            )
        prediction_rows.append(row)
    prediction_rows.sort(
        key=lambda row: common.topology_sort_key(str(row["topology_id"]))
    )
    common.atomic_write_csv(args.output_dir / "test_predictions.csv", prediction_rows)
    common.atomic_write_csv(args.output_dir / "training_curve.csv", curve)
    common.atomic_write_json(args.output_dir / "run_result.json", result)
    torch.save(
        {"model_state_dict": best_state, "result": result},
        args.output_dir / "best_model.pt",
    )
    return result


def parse_float_tuple(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one number")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-jsonl", type=Path, default=common.DEFAULT_INCIDENCE_GRAPHS)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--label-audit", type=Path, default=DEFAULT_LABEL_AUDIT)
    parser.add_argument("--fold-audit", type=Path, default=DEFAULT_FOLD_AUDIT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=common.DEFAULT_OUTPUT_ROOT / "incidence_gnn_helpful" / "smoke_fold0_seed42",
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--early-stop-patience", type=int, default=30)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0001)
    parser.add_argument("--regret-compute-cost-pp", type=float, default=0.0)
    parser.add_argument(
        "--precision-constraints", type=parse_float_tuple, default=(0.4, 0.5)
    )
    parser.add_argument(
        "--compute-costs-pp",
        type=parse_float_tuple,
        default=(0.0, 0.05, 0.1, 0.25, 0.5),
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    graph_rows = common.read_jsonl(args.graph_jsonl)
    label_rows = common.read_csv(args.labels)
    fold_rows = common.read_csv(args.folds)
    audit = three_class_gnn.audit_inputs(
        graph_rows,
        label_rows,
        fold_rows,
        graph_path=args.graph_jsonl,
        label_path=args.labels,
        fold_path=args.folds,
        label_audit_path=args.label_audit,
        fold_audit_path=args.fold_audit,
        label_field="primary_label",
    )
    splits = three_class_gnn.split_topology_ids(fold_rows, test_fold=args.fold)
    plan = {
        "passed": audit["passed"],
        "execute": args.execute,
        "formal": args.formal,
        "task": "material_helpful_vs_non_helpful",
        "model": "relation_aware_incidence_gnn",
        "fold": args.fold,
        "validation_fold": (args.fold + 1) % 5,
        "seed": args.seed,
        "split_sizes": {name: len(values) for name, values in splits.items()},
        "output_dir": str(args.output_dir),
        "input_audit": audit,
        "protocol": {
            "checkpoint": "minimum_validation_weighted_bce",
            "calibration": "temperature_scaling_on_validation_only",
            "primary_threshold": "validation_regret_optimal",
            "secondary_thresholds": [
                f"validation_precision_at_least_{value}"
                for value in args.precision_constraints
            ],
        },
    }
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if not audit["passed"]:
        return 1
    if not args.execute:
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    common.atomic_write_json(args.output_dir / "run_plan.json", plan)
    result = run_training(args, graph_rows, label_rows, splits)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
