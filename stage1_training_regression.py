import argparse
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch_geometric.loader import DataLoader

from experiment_config import PROCESSED_DATA_DIR, RESULTS_ROOT, make_results_dir, resolve_path
from model.graph_utils import (
    failure_context_edge_features,
    load_graph_dataset,
    lr_small_edge_features,
    parse_json_to_pyg_data,
    resolve_graph_data_dir,
)
from model.model_structure import (
    EDGE_RAW_DIM,
    FAILURE_CONTEXT_DIM,
    LR_SMALL_FEATURE_DIM,
    NODE_FEATURE_DIM,
    build_tabular_regression_model,
    normalize_tabular_model_family,
    tabular_model_class_name,
    tabular_model_label,
    tabular_model_result_token,
    tabular_model_summary_label,
)
from split_binding import limit_training_dataset, save_split_files, split_dataset_counts

SEED = 42


def enable_strict_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)


enable_strict_reproducibility(SEED)


def load_real_dataset(directory):
    return load_graph_dataset(directory, parse_json_to_pyg_data, log_prefix="🔍 Loading")


def dataset_num_edges(dataset):
    return sum(data.num_edges for data in dataset)


def normalize_feature_mode(feature_mode):
    mode = (feature_mode or "full").strip().lower()
    if mode not in {"full", "utility_cpra", "failure_context", "lr_small"}:
        raise ValueError(f"Unsupported tabular feature mode: {feature_mode}")
    return mode


def tabular_input_dim(feature_mode):
    mode = normalize_feature_mode(feature_mode)
    if mode == "utility_cpra":
        return 2
    if mode == "lr_small":
        return LR_SMALL_FEATURE_DIM
    if mode == "failure_context":
        return FAILURE_CONTEXT_DIM
    return NODE_FEATURE_DIM * 2 + EDGE_RAW_DIM


def tabular_edge_features(batch, feature_mode="full"):
    mode = normalize_feature_mode(feature_mode)
    if mode == "utility_cpra":
        _, dst = batch.edge_index
        utility = batch.edge_attr[:, :1]
        recipient_cpra = batch.x[dst, 1:2]
        return torch.cat([utility, recipient_cpra], dim=-1)
    if mode == "lr_small":
        return lr_small_edge_features(batch)
    if mode == "failure_context":
        return failure_context_edge_features(batch)

    src, dst = batch.edge_index
    return torch.cat([batch.x[src], batch.x[dst], batch.edge_attr], dim=-1)


def train_baseline(model_family="mlp", data_dir=None, results_root=None, feature_mode="full", epochs=50, train_size=None):
    model_family = normalize_tabular_model_family(model_family)
    feature_mode = normalize_feature_mode(feature_mode)

    node_dim = NODE_FEATURE_DIM
    edge_raw_dim_value = EDGE_RAW_DIM
    input_dim = tabular_input_dim(feature_mode)
    hidden_dim = 256
    batch_size = 8
    learning_rate = 1e-3
    num_epochs = int(epochs)
    if num_epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = str(resolve_path(data_dir or PROCESSED_DATA_DIR))
    strict_reproducibility = True
    data_dir, _ = resolve_graph_data_dir(data_dir, log_prefix="🔍 Loading")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    result_prefix = f"2stg_{tabular_model_result_token(model_family)}_"
    results_dir = str(make_results_dir(result_prefix, timestamp=timestamp, results_root=results_root or RESULTS_ROOT))
    save_path = os.path.join(results_dir, "best_stage1_model_real.pth")

    print(f"📁 Results will be saved at: {results_dir}")

    full_dataset = load_real_dataset(data_dir)
    if not full_dataset:
        print("❌ Error: Processed JSON data not found!")
        return

    random.shuffle(full_dataset)
    total_len = len(full_dataset)
    train_count, val_count, test_count = split_dataset_counts(total_len)
    train_end = train_count
    val_end = train_count + val_count

    train_dataset = full_dataset[:train_end]
    val_dataset = full_dataset[train_end:val_end]
    test_dataset = full_dataset[val_end:]
    original_train_count = len(train_dataset)
    train_dataset = limit_training_dataset(train_dataset, train_size)
    effective_train_size = len(train_dataset)

    split_paths = save_split_files(results_dir, train_dataset, val_dataset, test_dataset)
    print(f"📄 Train split file list saved to: {split_paths['train']}")
    print(f"📄 Validation split file list saved to: {split_paths['val']}")
    print(f"📄 Test split file list saved to: {split_paths['test']}")

    print(
        f"📊 Split result: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)} "
        f"(train pool before --train_size: {original_train_count})"
    )
    if len(val_dataset) == 0:
        print("⚠️ Validation split is empty; training loss will be reused as the model-selection metric.")
    if len(test_dataset) == 0:
        print("⚠️ Test split is empty; final test evaluation and diagnostic plots will be skipped.")

    train_loader_generator = torch.Generator()
    train_loader_generator.manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=train_loader_generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    train_num_edges = dataset_num_edges(train_dataset)
    val_num_edges = dataset_num_edges(val_dataset)
    test_num_edges = dataset_num_edges(test_dataset)

    if train_num_edges == 0:
        print("❌ Error: Training split contains no edges.")
        return

    model = build_tabular_regression_model(
        model_family=model_family,
        node_dim=node_dim,
        edge_dim=edge_raw_dim_value,
        hidden_dim=hidden_dim,
        input_dim=input_dim,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"\n🚀 Starting {tabular_model_label(model_family)} training (Device: {device})...")
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            edge_features = tabular_edge_features(batch, feature_mode=feature_mode)
            w_pred = model(edge_features)
            loss = criterion(w_pred, batch.y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * batch.num_edges

        avg_train_loss = total_train_loss / train_num_edges

        if val_num_edges > 0:
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    edge_features = tabular_edge_features(batch, feature_mode=feature_mode)
                    w_pred = model(edge_features)
                    loss = criterion(w_pred, batch.y)
                    total_val_loss += loss.item() * batch.num_edges
            avg_val_loss = total_val_loss / val_num_edges
        else:
            avg_val_loss = avg_train_loss

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{num_epochs} | Train MSE: {avg_train_loss:.4f} | Val MSE: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "config": {
                    "MODEL_FAMILY": model_family,
                    "FEATURE_MODE": feature_mode,
                    "INPUT_DIM": input_dim,
                    "NODE_DIM": node_dim,
                    "EDGE_RAW_DIM": edge_raw_dim_value,
                    "LEARNING_RATE": learning_rate,
                    "BATCH_SIZE": batch_size,
                    "TRAIN_SIZE_REQUESTED": train_size,
                    "TRAIN_SIZE_EFFECTIVE": effective_train_size,
                    "TRAIN_POOL_SIZE": original_train_count,
                    "SEED": SEED,
                    "STRICT_REPRODUCIBILITY": strict_reproducibility,
                    "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
                },
            }
            if model_family == "mlp":
                checkpoint["config"]["HIDDEN_DIM"] = hidden_dim
            torch.save(checkpoint, save_path)

    print(f"\n✅ Training stage complete! Best Val MSE: {best_val_loss:.4f}")

    print("\n🔍 Evaluating test set using the best model (Final Test)...")
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    avg_test_loss = None
    y_true = None
    y_pred = None
    if test_num_edges > 0:
        model.eval()
        total_test_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                edge_features = tabular_edge_features(batch, feature_mode=feature_mode)
                w_pred = model(edge_features)
                loss = criterion(w_pred, batch.y)
                total_test_loss += loss.item() * batch.num_edges
                all_preds.append(w_pred.cpu())
                all_targets.append(batch.y.cpu())

        avg_test_loss = total_test_loss / test_num_edges
        print(f"🏁 Final Test Result: Test MSE = {avg_test_loss:.4f}")

        y_true = torch.cat(all_targets).numpy()
        y_pred = torch.cat(all_preds).numpy()
    else:
        print("ℹ️ Final test evaluation skipped because the test split is empty.")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), history["train_loss"], label="Train MSE")
    plt.plot(range(1, num_epochs + 1), history["val_loss"], label="Val MSE")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    title = "Training & Validation Loss Curve"
    if avg_test_loss is not None:
        title += f"\n(Final Test MSE: {avg_test_loss:.4f})"
    plt.title(title)
    plt.legend()
    plt.grid(True)
    curve_path = os.path.join(results_dir, "loss_curve.png")
    plt.savefig(curve_path)
    print(f"📊 Loss curve saved to: {curve_path}")

    if avg_test_loss is not None:
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true, y_pred, alpha=0.4, color="blue", label="Predictions")
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Ideal (y=x)")
        plt.xlabel("True Labels (Ground Truth)")
        plt.ylabel("Predicted Labels (Model Output)")
        plt.title(f"Diagnostic: Prediction Scatter Plot\n(Test MSE: {avg_test_loss:.4f})")
        plt.legend()
        plt.grid(True)
        scatter_path = os.path.join(results_dir, "prediction_scatter.png")
        plt.savefig(scatter_path)
        print(f"🎯 Diagnostic scatter plot saved to: {scatter_path}")

    summary_path = os.path.join(results_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(f"Training Time: {timestamp}\n")
        handle.write(
            f"Model Type: {tabular_model_summary_label(model_family)} ({tabular_model_class_name(model_family)})\n"
        )
        handle.write(f"Dataset Path: {data_dir}\n")
        handle.write("--- Hyperparameters ---\n")
        handle.write(f"MODEL_FAMILY: {model_family}\n")
        handle.write(f"FEATURE_MODE: {feature_mode}\n")
        handle.write(f"INPUT_DIM: {input_dim}\n")
        handle.write(f"NODE_DIM: {node_dim}\n")
        handle.write(f"EDGE_RAW_DIM: {edge_raw_dim_value}\n")
        if model_family == "mlp":
            handle.write(f"HIDDEN_DIM: {hidden_dim}\n")
        handle.write(f"BATCH_SIZE: {batch_size}\n")
        handle.write(f"TRAIN_SIZE_REQUESTED: {train_size}\n")
        handle.write(f"TRAIN_SIZE_EFFECTIVE: {effective_train_size}\n")
        handle.write(f"TRAIN_POOL_SIZE: {original_train_count}\n")
        handle.write(f"LEARNING_RATE: {learning_rate}\n")
        handle.write(f"NUM_EPOCHS: {num_epochs}\n")
        handle.write(f"SEED: {SEED}\n")
        handle.write(f"STRICT_REPRODUCIBILITY: {strict_reproducibility}\n")
        handle.write(f"CUBLAS_WORKSPACE_CONFIG: {os.environ.get('CUBLAS_WORKSPACE_CONFIG', '')}\n")
        handle.write("--- Results ---\n")
        handle.write(f"Best Val MSE: {best_val_loss:.4f}\n")
        if avg_test_loss is None:
            handle.write("Final Test MSE: skipped (empty test split)\n")
        else:
            handle.write(f"Final Test MSE: {avg_test_loss:.4f}\n")
    print(f"📝 Training summary saved at: {summary_path}")


def main(default_model_family="mlp", default_feature_mode="full", default_epochs=50, description="Stage-1 regression training"):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data_dir", type=str, default=str(PROCESSED_DATA_DIR), help="Directory containing processed G-*.json graphs")
    parser.add_argument("--results_root", type=str, default=str(RESULTS_ROOT), help="Root directory where timestamped training outputs will be created")
    parser.add_argument("--model_family", type=str, default=default_model_family, choices=["mlp", "lr"], help="Tabular regression baseline to train")
    parser.add_argument(
        "--feature_mode",
        type=str,
        default=default_feature_mode,
        choices=["full", "utility_cpra", "failure_context", "lr_small"],
        help=(
            "Tabular feature set: full features, failure-context features, "
            "lr_small 3D proxy features, or the 2D utility+recipient-cPRA sanity-check setup"
        ),
    )
    parser.add_argument("--epochs", type=int, default=default_epochs, help="Number of training epochs")
    parser.add_argument(
        "--train_size",
        type=int,
        default=None,
        help="Number of graphs to use from the training split; default uses the full training split",
    )
    args = parser.parse_args()
    train_baseline(
        model_family=args.model_family,
        data_dir=args.data_dir,
        results_root=args.results_root,
        feature_mode=args.feature_mode,
        epochs=args.epochs,
        train_size=args.train_size,
    )
