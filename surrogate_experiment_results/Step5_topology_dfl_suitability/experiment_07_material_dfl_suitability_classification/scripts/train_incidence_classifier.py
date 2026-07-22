#!/usr/bin/env python3
"""Train one three-class incidence GNN for Experiment 07."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Sequence

import numpy as np

import material_common as common
import run_scalar_classification_baselines as baseline_metrics


DEFAULT_LABELS = common.DEFAULT_OUTPUT_ROOT / "labels" / "material_labels.csv"
DEFAULT_FOLDS = common.DEFAULT_OUTPUT_ROOT / "splits" / "material_folds.csv"
DEFAULT_LABEL_AUDIT = common.DEFAULT_OUTPUT_ROOT / "labels" / "material_labels.audit.json"
DEFAULT_FOLD_AUDIT = common.DEFAULT_OUTPUT_ROOT / "splits" / "material_folds.audit.json"


def split_topology_ids(
    fold_rows: list[dict[str, str]],
    *,
    test_fold: int,
    fold_count: int = 5,
) -> dict[str, set[str]]:
    if test_fold not in range(fold_count):
        raise ValueError(f"test fold must lie in [0,{fold_count - 1}]")
    validation_fold = (test_fold + 1) % fold_count
    output = {"train": set(), "validation": set(), "test": set()}
    seen: set[str] = set()
    for row in fold_rows:
        topology_id = row["topology_id"]
        fold = int(row["fold"])
        if topology_id in seen:
            raise ValueError(f"duplicate fold topology:{topology_id}")
        if fold not in range(fold_count):
            raise ValueError(f"invalid fold:{topology_id}:{fold}")
        seen.add(topology_id)
        if fold == test_fold:
            output["test"].add(topology_id)
        elif fold == validation_fold:
            output["validation"].add(topology_id)
        else:
            output["train"].add(topology_id)
    sizes = {name: len(values) for name, values in output.items()}
    if sizes != {"train": 600, "validation": 200, "test": 200}:
        raise ValueError(f"unexpected nested split sizes:{sizes}")
    return output


def audit_inputs(
    graph_rows: list[dict[str, Any]],
    label_rows: list[dict[str, str]],
    fold_rows: list[dict[str, str]],
    *,
    graph_path: Path,
    label_path: Path,
    fold_path: Path,
    label_audit_path: Path,
    fold_audit_path: Path,
    label_field: str,
) -> dict[str, Any]:
    failures: list[str] = []
    graph_by_id = {str(row.get("topology_id")): row for row in graph_rows}
    label_by_id = {row.get("topology_id", ""): row for row in label_rows}
    fold_by_id = {row.get("topology_id", ""): row for row in fold_rows}
    if len(graph_rows) != 1000 or len(graph_by_id) != 1000:
        failures.append(f"graph_count_or_uniqueness_mismatch:{len(graph_rows)}/{len(graph_by_id)}")
    if len(label_rows) != 1000 or len(label_by_id) != 1000:
        failures.append(f"label_count_or_uniqueness_mismatch:{len(label_rows)}/{len(label_by_id)}")
    if len(fold_rows) != 1000 or len(fold_by_id) != 1000:
        failures.append(f"fold_count_or_uniqueness_mismatch:{len(fold_rows)}/{len(fold_by_id)}")
    if not (set(graph_by_id) == set(label_by_id) == set(fold_by_id)):
        failures.append("graph_label_fold_topology_sets_differ")
    graph_sha256 = common.sha256_file(graph_path)
    if graph_sha256 != common.LOCKED_INCIDENCE_GRAPH_SHA256:
        failures.append(f"graph_sha256_mismatch:{graph_sha256}")
    try:
        label_audit = json.loads(label_audit_path.read_text(encoding="utf-8"))
        fold_audit = json.loads(fold_audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"audit_read_failed:{exc}")
        label_audit, fold_audit = {}, {}
    if label_audit.get("passed") is not True:
        failures.append("label_audit_not_passed")
    if fold_audit.get("passed") is not True:
        failures.append("fold_audit_not_passed")
    forbidden = {
        "formal_label_mean_pp",
        "normalized_improvement_pp",
        "label_uncertainty_std_pp",
        "data_seed",
        "train_seed",
        "test_gap_2stage",
        "test_gap_spoplus",
        "delta",
    }
    for topology_id in sorted(set(graph_by_id) & set(label_by_id) & set(fold_by_id)):
        graph = graph_by_id[topology_id]
        label = label_by_id[topology_id]
        fold = fold_by_id[topology_id]
        for hash_field in ("topology_hash", "feasible_set_hash"):
            if str(graph.get(hash_field, "")) != str(label.get(hash_field, "")):
                failures.append(f"{topology_id}:{hash_field}_graph_label_mismatch")
            if str(label.get(hash_field, "")) != str(fold.get(hash_field, "")):
                failures.append(f"{topology_id}:{hash_field}_label_fold_mismatch")
        if label.get(label_field) not in common.CLASS_LABELS:
            failures.append(f"{topology_id}:invalid_label:{label.get(label_field)}")
        feature_names = set(graph.get("node_feature_names", []))
        scalar_names = set(graph.get("scalar_topology_features", {}))
        leaked = sorted((feature_names | scalar_names) & forbidden)
        if leaked:
            failures.append(f"{topology_id}:forbidden_input_features:{','.join(leaked)}")
    label_counts = Counter(row.get(label_field, "") for row in label_rows)
    return {
        "passed": not failures,
        "graph_count": len(graph_rows),
        "label_count": len(label_rows),
        "fold_count": len(fold_rows),
        "label_field": label_field,
        "label_counts": {
            label: label_counts[label]
            for label in common.CLASS_LABELS
        },
        "graph_path": str(graph_path),
        "graph_sha256": graph_sha256,
        "expected_graph_sha256": common.LOCKED_INCIDENCE_GRAPH_SHA256,
        "label_path": str(label_path),
        "label_sha256": common.sha256_file(label_path),
        "fold_path": str(fold_path),
        "fold_sha256": common.sha256_file(fold_path),
        "label_audit_path": str(label_audit_path),
        "fold_audit_path": str(fold_audit_path),
        "graph_embeds_target_metadata_but_trainer_does_not_read_it": True,
        "target_or_uncertainty_used_as_input_feature": False,
        "failures": failures[:100],
    }


def build_datasets(
    graph_rows: list[dict[str, Any]],
    label_rows: list[dict[str, str]],
    splits: dict[str, set[str]],
    *,
    label_field: str,
    torch: Any,
    Data: Any,
) -> tuple[dict[str, list[Any]], list[str]]:
    graph_by_id = {row["topology_id"]: row for row in graph_rows}
    label_by_id = {row["topology_id"]: row for row in label_rows}
    ordered_ids = sorted(graph_by_id, key=common.topology_sort_key)
    topology_code = {topology_id: index for index, topology_id in enumerate(ordered_ids)}
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
                    y_class=torch.tensor(
                        [common.CLASS_TO_INDEX[label[label_field]]],
                        dtype=torch.long,
                    ),
                    raw_delta_pp=torch.tensor(
                        [float(label["formal_label_mean_pp"])],
                        dtype=torch.float32,
                    ),
                    topology_code=torch.tensor(
                        [topology_code[topology_id]],
                        dtype=torch.long,
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
    class MaterialIncidenceGNN(torch.nn.Module):
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
                torch.nn.Linear(64, len(common.CLASS_LABELS)),
            )

        def forward(self, batch: Any) -> Any:
            hidden = batch.x
            for convolution in self.convolutions:
                hidden = torch.nn.functional.relu(
                    convolution(hidden, batch.edge_index, batch.edge_type)
                )
                hidden = torch.nn.functional.dropout(
                    hidden,
                    p=dropout,
                    training=self.training,
                )
            pooled = []
            for node_type in (0, 1):
                mask = batch.node_type == node_type
                pooled.append(
                    global_mean_pool(hidden[mask], batch.batch[mask], size=batch.num_graphs)
                )
                pooled.append(
                    global_max_pool(hidden[mask], batch.batch[mask], size=batch.num_graphs)
                )
            return self.head(torch.cat(pooled, dim=1))

    return MaterialIncidenceGNN()


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
        graph_rows,
        label_rows,
        splits,
        label_field=args.label_field,
        torch=torch,
        Data=Data,
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
            datasets["validation"],
            batch_size=args.batch_size,
            shuffle=False,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=args.batch_size,
            shuffle=False,
        ),
    }
    train_target = np.asarray(
        [int(item.y_class.item()) for item in datasets["train"]],
        dtype=int,
    )
    train_counts = Counter(train_target.tolist())
    class_weights = np.asarray(
        [
            len(train_target) / (len(common.CLASS_LABELS) * train_counts[index])
            for index in range(len(common.CLASS_LABELS))
        ],
        dtype=float,
    )
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
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )

    def predict(loader: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        model.eval()
        targets: list[int] = []
        probabilities: list[list[float]] = []
        deltas: list[float] = []
        codes: list[int] = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                probability = torch.softmax(model(batch), dim=1)
                targets.extend(batch.y_class.view(-1).cpu().numpy().tolist())
                probabilities.extend(probability.cpu().numpy().tolist())
                deltas.extend(batch.raw_delta_pp.view(-1).cpu().numpy().tolist())
                codes.extend(batch.topology_code.view(-1).cpu().numpy().tolist())
        return (
            np.asarray(targets, dtype=int),
            np.asarray(probabilities, dtype=float),
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
            loss = criterion(model(batch), batch.y_class.view(-1))
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
                loss = criterion(model(batch), batch.y_class.view(-1))
                count = int(batch.num_graphs)
                validation_loss_sum += float(loss.item()) * count
                validation_count += count
        validation_loss = validation_loss_sum / validation_count
        improved = validation_loss < best_validation_loss - args.early_stop_min_delta
        if improved:
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
                "train_weighted_cross_entropy": train_loss_sum / train_count,
                "validation_weighted_cross_entropy": validation_loss,
                "best_validation_weighted_cross_entropy": best_validation_loss,
                "bad_epochs": bad_epochs,
                "epoch_seconds": time.perf_counter() - epoch_started,
            }
        )
        print(
            f"[material-incidence] fold={args.fold} seed={args.seed} epoch={epoch} "
            f"train_ce={curve[-1]['train_weighted_cross_entropy']:.6f} "
            f"validation_ce={validation_loss:.6f}",
            flush=True,
        )
        if bad_epochs >= args.early_stop_patience:
            break
    if best_state is None:
        raise RuntimeError("no classifier checkpoint was selected")
    model.load_state_dict(best_state)
    validation_target_index, validation_probability, validation_delta, _ = predict(
        loaders["validation"]
    )
    test_target_index, test_probability, test_delta, test_codes = predict(loaders["test"])
    validation_target = np.asarray(
        [common.CLASS_LABELS[index] for index in validation_target_index],
        dtype=object,
    )
    test_target = np.asarray(
        [common.CLASS_LABELS[index] for index in test_target_index],
        dtype=object,
    )
    validation_metrics = baseline_metrics.evaluate_model(
        validation_target,
        validation_delta,
        validation_probability,
        helpful_threshold=args.helpful_threshold,
        compute_costs_pp=args.compute_costs_pp,
    )
    test_metrics = baseline_metrics.evaluate_model(
        test_target,
        test_delta,
        test_probability,
        helpful_threshold=args.helpful_threshold,
        compute_costs_pp=args.compute_costs_pp,
    )
    result = {
        "status": "success",
        "formal": False,
        "task": "material_dfl_suitability_three_class",
        "model": "relation_aware_incidence_gnn",
        "label_field": args.label_field,
        "fold": args.fold,
        "validation_fold": (args.fold + 1) % 5,
        "seed": args.seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "split_sizes": {name: len(values) for name, values in datasets.items()},
        "train_class_counts": {
            common.CLASS_LABELS[index]: train_counts[index]
            for index in range(len(common.CLASS_LABELS))
        },
        "class_weights": {
            common.CLASS_LABELS[index]: float(class_weights[index])
            for index in range(len(common.CLASS_LABELS))
        },
        "epochs_completed": len(curve),
        "early_stop_triggered": len(curve) < args.max_epochs,
        "best_epoch": best_epoch,
        "best_validation_weighted_cross_entropy": best_validation_loss,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
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
            "helpful_threshold": args.helpful_threshold,
            "compute_costs_pp": args.compute_costs_pp,
            "threads": args.threads,
        },
    }
    prediction_rows: list[dict[str, Any]] = []
    for target_index, probability, delta, code in zip(
        test_target_index,
        test_probability,
        test_delta,
        test_codes,
        strict=True,
    ):
        prediction_rows.append(
            {
                "topology_id": ordered_ids[int(code)],
                "fold": args.fold,
                "seed": args.seed,
                "label_field": args.label_field,
                "target_label": common.CLASS_LABELS[int(target_index)],
                "formal_label_mean_pp": float(delta),
                "probability_material_harmful": float(probability[0]),
                "probability_neutral_or_uncertain": float(probability[1]),
                "probability_material_helpful": float(probability[2]),
                "predicted_label": common.CLASS_LABELS[int(np.argmax(probability))],
            }
        )
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-jsonl", type=Path, default=common.DEFAULT_INCIDENCE_GRAPHS)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--label-audit", type=Path, default=DEFAULT_LABEL_AUDIT)
    parser.add_argument("--fold-audit", type=Path, default=DEFAULT_FOLD_AUDIT)
    parser.add_argument("--label-field", choices=("primary_label", "confidence_label"), default="primary_label")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=common.DEFAULT_OUTPUT_ROOT / "incidence_gnn" / "smoke_fold0_seed42",
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
    parser.add_argument("--helpful-threshold", type=float, default=0.5)
    parser.add_argument("--compute-costs-pp", default="0,0.05,0.1,0.25,0.5")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    args.compute_costs_pp = baseline_metrics.parse_costs(args.compute_costs_pp)
    return args


def main() -> int:
    args = parse_args()
    graph_rows = common.read_jsonl(args.graph_jsonl)
    label_rows = common.read_csv(args.labels)
    fold_rows = common.read_csv(args.folds)
    audit = audit_inputs(
        graph_rows,
        label_rows,
        fold_rows,
        graph_path=args.graph_jsonl,
        label_path=args.labels,
        fold_path=args.folds,
        label_audit_path=args.label_audit,
        fold_audit_path=args.fold_audit,
        label_field=args.label_field,
    )
    splits = split_topology_ids(fold_rows, test_fold=args.fold)
    plan = {
        "passed": audit["passed"],
        "execute": args.execute,
        "task": "material_dfl_suitability_three_class",
        "model": "relation_aware_incidence_gnn",
        "fold": args.fold,
        "validation_fold": (args.fold + 1) % 5,
        "seed": args.seed,
        "split_sizes": {name: len(values) for name, values in splits.items()},
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
    result = run_training(args, graph_rows, label_rows, splits)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
