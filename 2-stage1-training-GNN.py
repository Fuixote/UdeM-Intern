import argparse
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np
import random
from datetime import datetime
import matplotlib.pyplot as plt

from experiment_config import PROCESSED_DATA_DIR, RESULTS_ROOT, make_results_dir, resolve_path
from split_binding import save_split_files, split_dataset_counts
# Set Matplotlib to Agg backend for running on headless servers
import matplotlib
matplotlib.use('Agg')

# Import model definition
from model.graph_utils import load_graph_dataset, parse_json_to_pyg_data, resolve_graph_data_dir
from model.model_structure import EDGE_RAW_DIM, NODE_FEATURE_DIM, KidneyEdgePredictor

# ==========================================
# 0. Global Settings (Random SEED, etc.)
# ==========================================
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

# ==========================================
# 2. Load Dataset
# ==========================================
def load_real_dataset(directory):
    return load_graph_dataset(directory, parse_json_to_pyg_data, log_prefix="🔍 Loading")


def dataset_num_edges(dataset):
    return sum(data.num_edges for data in dataset)

# ==========================================
# 3. Training Loop
# ==========================================
def train_baseline(data_dir=None, results_root=None):
    # --- Hyperparameters ---
    NODE_DIM = NODE_FEATURE_DIM
    EDGE_RAW_DIM_VALUE = EDGE_RAW_DIM
    HIDDEN_DIM = 64
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 50
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DATA_DIR = str(resolve_path(data_dir or PROCESSED_DATA_DIR))
    SEED = 42
    STRICT_REPRODUCIBILITY = True
    DATA_DIR, _ = resolve_graph_data_dir(DATA_DIR, log_prefix="🔍 Loading")

    # --- Archiving Settings ---
    TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    RESULTS_DIR = str(make_results_dir("2stg_Gnn_", timestamp=TIMESTAMP, results_root=results_root or RESULTS_ROOT))
    SAVE_PATH = os.path.join(RESULTS_DIR, 'best_stage1_model_real.pth')
    
    print(f"📁 Results will be saved at: {RESULTS_DIR}")

    # --- Data Preparation ---
    full_dataset = load_real_dataset(DATA_DIR)
    if not full_dataset:
        print("❌ Error: Processed JSON data not found!")
        return

    # Randomly shuffle dataset
    random.shuffle(full_dataset)

    # Split dataset (60/20/20)
    total_len = len(full_dataset)
    train_count, val_count, test_count = split_dataset_counts(total_len)
    train_end = train_count
    val_end = train_count + val_count

    train_dataset = full_dataset[:train_end]
    val_dataset = full_dataset[train_end:val_end]
    test_dataset = full_dataset[val_end:]
    
    split_paths = save_split_files(RESULTS_DIR, train_dataset, val_dataset, test_dataset)
    print(f"📄 Train split file list saved to: {split_paths['train']}")
    print(f"📄 Validation split file list saved to: {split_paths['val']}")
    print(f"📄 Test split file list saved to: {split_paths['test']}")
    
    print(f"📊 Split result: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    if len(val_dataset) == 0:
        print("⚠️ Validation split is empty; training loss will be reused as the model-selection metric.")
    if len(test_dataset) == 0:
        print("⚠️ Test split is empty; final test evaluation and diagnostic plots will be skipped.")

    train_loader_generator = torch.Generator()
    train_loader_generator.manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=train_loader_generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    train_num_edges = dataset_num_edges(train_dataset)
    val_num_edges = dataset_num_edges(val_dataset)
    test_num_edges = dataset_num_edges(test_dataset)

    if train_num_edges == 0:
        print("❌ Error: Training split contains no edges.")
        return

    # --- Initialization ---
    model = KidneyEdgePredictor(NODE_DIM, EDGE_RAW_DIM_VALUE, HIDDEN_DIM).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\n🚀 Starting training (Device: {DEVICE})...")
    best_val_loss = float('inf')
    
    # History for plotting
    history = {
        'train_loss': [],
        'val_loss': []
    }

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            w_pred = model(batch.x, batch.edge_index, batch.edge_attr)
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
                    batch = batch.to(DEVICE)
                    w_pred = model(batch.x, batch.edge_index, batch.edge_attr)
                    loss = criterion(w_pred, batch.y)
                    total_val_loss += loss.item() * batch.num_edges
            avg_val_loss = total_val_loss / val_num_edges
        else:
            avg_val_loss = avg_train_loss
        
        # Log history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{NUM_EPOCHS} | Train MSE: {avg_train_loss:.4f} | Val MSE: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            # Build professional Checkpoint with full info
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'config': {
                    'NODE_DIM': NODE_DIM,
                    'EDGE_RAW_DIM': EDGE_RAW_DIM_VALUE,
                    'HIDDEN_DIM': HIDDEN_DIM,
                    'LEARNING_RATE': LEARNING_RATE,
                    'BATCH_SIZE': BATCH_SIZE,
                    'SEED': SEED,
                    'STRICT_REPRODUCIBILITY': STRICT_REPRODUCIBILITY,
                    'CUBLAS_WORKSPACE_CONFIG': os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
                }
            }
            torch.save(checkpoint, SAVE_PATH)
            print(f"  --> [Model Saved] Epoch {epoch} - New Best Val MSE: {best_val_loss:.4f}")

    print(f"\n✅ Training stage complete! Best Val MSE: {best_val_loss:.4f}")

    # ==========================================
    # 4. Final Test Evaluation
    # ==========================================
    print("\n🔍 Evaluating test set using the best model (Final Test)...")
    
    # Load Checkpoint and restore state
    checkpoint = torch.load(SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

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
                batch = batch.to(DEVICE)
                w_pred = model(batch.x, batch.edge_index, batch.edge_attr)
                loss = criterion(w_pred, batch.y)
                total_test_loss += loss.item() * batch.num_edges

                # Collect results for diagnostic plotting
                all_preds.append(w_pred.cpu())
                all_targets.append(batch.y.cpu())

        avg_test_loss = total_test_loss / test_num_edges
        print(f"🏁 Final Test Result: Test MSE = {avg_test_loss:.4f}")

        # Merge results from all batches
        y_true = torch.cat(all_targets).numpy()
        y_pred = torch.cat(all_preds).numpy()
    else:
        print("ℹ️ Final test evaluation skipped because the test split is empty.")

    # ==========================================
    # 5. Visualize Training Curve
    # ==========================================
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, NUM_EPOCHS + 1), history['train_loss'], label='Train MSE')
    plt.plot(range(1, NUM_EPOCHS + 1), history['val_loss'], label='Val MSE')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    title = 'Training & Validation Loss Curve'
    if avg_test_loss is not None:
        title += f'\n(Final Test MSE: {avg_test_loss:.4f})'
    plt.title(title)
    plt.legend()
    plt.grid(True)

    curve_path = os.path.join(RESULTS_DIR, "loss_curve.png")
    plt.savefig(curve_path)
    print(f"📊 Loss curve saved to: {curve_path}")

    # ==========================================
    # 6. Diagnostic Test: Scatter Plot (Predictions vs Ground Truth)
    # ==========================================
    if avg_test_loss is not None:
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true, y_pred, alpha=0.4, color='blue', label='Predictions')

        # Draw ideal y=x line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal (y=x)')

        plt.xlabel("True Labels (Ground Truth)")
        plt.ylabel("Predicted Labels (Model Output)")
        plt.title(f"Diagnostic: Prediction Scatter Plot\n(Test MSE: {avg_test_loss:.4f})")
        plt.legend()
        plt.grid(True)

        scatter_path = os.path.join(RESULTS_DIR, "prediction_scatter.png")
        plt.savefig(scatter_path)
        print(f"🎯 Diagnostic scatter plot saved to: {scatter_path}")

    # 7. Save Summary Report
    summary_path = os.path.join(RESULTS_DIR, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Training Time: {TIMESTAMP}\n")
        f.write(f"Model Type: GNN (KidneyEdgePredictor)\n")
        f.write(f"Dataset Path: {DATA_DIR}\n")
        f.write(f"--- Hyperparameters ---\n")
        f.write(f"NODE_DIM: {NODE_DIM}\n")
        f.write(f"EDGE_RAW_DIM: {EDGE_RAW_DIM_VALUE}\n")
        f.write(f"HIDDEN_DIM: {HIDDEN_DIM}\n")
        f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
        f.write(f"LEARNING_RATE: {LEARNING_RATE}\n")
        f.write(f"NUM_EPOCHS: {NUM_EPOCHS}\n")
        f.write(f"SEED: {SEED}\n")
        f.write(f"STRICT_REPRODUCIBILITY: {STRICT_REPRODUCIBILITY}\n")
        f.write(f"CUBLAS_WORKSPACE_CONFIG: {os.environ.get('CUBLAS_WORKSPACE_CONFIG', '')}\n")
        f.write(f"--- Results ---\n")
        f.write(f"Best Val MSE: {best_val_loss:.4f}\n")
        if avg_test_loss is None:
            f.write("Final Test MSE: skipped (empty test split)\n")
        else:
            f.write(f"Final Test MSE: {avg_test_loss:.4f}\n")
    print(f"📝 Training summary saved at: {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage-1 GNN training")
    parser.add_argument("--data_dir", type=str, default=str(PROCESSED_DATA_DIR),
                        help="Directory containing processed G-*.json graphs")
    parser.add_argument("--results_root", type=str, default=str(RESULTS_ROOT),
                        help="Root directory where timestamped training outputs will be created")
    args = parser.parse_args()
    train_baseline(data_dir=args.data_dir, results_root=args.results_root)
