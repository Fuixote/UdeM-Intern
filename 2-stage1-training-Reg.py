import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np
import os
import random
from datetime import datetime
import matplotlib.pyplot as plt

# Set Matplotlib to Agg backend for running on headless servers
import matplotlib
matplotlib.use('Agg')

from model.graph_utils import load_graph_dataset, parse_json_to_pyg_data
from model.model_structure import EDGE_RAW_DIM, NODE_FEATURE_DIM, MLPBaseline

# ==========================================
# 0. Global Settings (Random SEED, etc.)
# ==========================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ==========================================
# 2. Load Dataset
# ==========================================
def load_real_dataset(directory):
    return load_graph_dataset(directory, parse_json_to_pyg_data, log_prefix="🔍 Loading")

# ==========================================
# 3. Training Loop
# ==========================================
def train_baseline():
    # --- Hyperparameters ---
    NODE_DIM = NODE_FEATURE_DIM
    EDGE_RAW_DIM_LOCAL = EDGE_RAW_DIM
    HIDDEN_DIM = 256
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 50
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DATA_DIR = "/home/weikang/projects/UdeM-Intern/Exps/dataset/processed"
    SEED = 42

    # --- Archiving Settings ---
    TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    RESULTS_DIR = os.path.join("results", f"2stg_Reg_{TIMESTAMP}")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    SAVE_PATH = os.path.join(RESULTS_DIR, 'best_stage1_model_real.pth')
    
    print(f"📁 Results will be saved at: {RESULTS_DIR}")

    # --- Data Preparation ---
    full_dataset = load_real_dataset(DATA_DIR)
    if not full_dataset:
        print("❌ Error: Processed JSON data not found!")
        return

    random.shuffle(full_dataset)
    total_len = len(full_dataset)
    train_end = int(0.6 * total_len)
    val_end = int(0.8 * total_len)

    train_dataset = full_dataset[:train_end]
    val_dataset = full_dataset[train_end:val_end]
    test_dataset = full_dataset[val_end:]
    
    # Save the list of test files to the results directory
    test_files_path = os.path.join(RESULTS_DIR, "test_files.txt")
    with open(test_files_path, 'w') as f:
        for data in test_dataset:
            f.write(data.filename + "\n")
    print(f"📄 Test set file list saved to: {test_files_path}")
    
    print(f"📊 Split result: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Initialization ---
    #[NEW] 使用我们刚刚定义的纯 MLP 模型
    model = MLPBaseline(NODE_DIM, EDGE_RAW_DIM_LOCAL, HIDDEN_DIM).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\n🚀 Starting training (Device: {DEVICE})...")
    best_val_loss = float('inf')
    
    history = {'train_loss': [], 'val_loss':[]}

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            
            # ==========================================
            # [NEW] 数据拼接逻辑：将图转为表格数据
            # 提取边的起点(src)和终点(dst)索引
            # ==========================================
            src, dst = batch.edge_index
            
            # 将 源节点特征 + 目标节点特征 + 边的固有特征 拼接在一起
            # 形状变化: [num_edges, 13] +[num_edges, 13] + [num_edges, 1] => [num_edges, 27]
            edge_features = torch.cat([batch.x[src], batch.x[dst], batch.edge_attr], dim=-1)
            
            # 直接喂给普通 MLP
            w_pred = model(edge_features)
            
            loss = criterion(w_pred, batch.y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * batch.num_edges
            
        avg_train_loss = total_train_loss / sum(data.num_edges for data in train_dataset)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                
                # [NEW] 同样的拼接逻辑
                src, dst = batch.edge_index
                edge_features = torch.cat([batch.x[src], batch.x[dst], batch.edge_attr], dim=-1)
                
                w_pred = model(edge_features)
                loss = criterion(w_pred, batch.y)
                total_val_loss += loss.item() * batch.num_edges
        
        avg_val_loss = total_val_loss / sum(data.num_edges for data in val_dataset)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{NUM_EPOCHS} | Train MSE: {avg_train_loss:.4f} | Val MSE: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'config': {
                    'NODE_DIM': NODE_DIM,
                    'EDGE_RAW_DIM': EDGE_RAW_DIM_LOCAL,
                    'HIDDEN_DIM': HIDDEN_DIM,
                    'LEARNING_RATE': LEARNING_RATE,
                    'BATCH_SIZE': BATCH_SIZE,
                    'SEED': SEED
                }
            }
            torch.save(checkpoint, SAVE_PATH)

    print(f"\n✅ Training stage complete! Best Val MSE: {best_val_loss:.4f}")

    # ==========================================
    # 4. Final Test Evaluation
    # ==========================================
    print("\n🔍 Evaluating test set using the best model (Final Test)...")
    checkpoint = torch.load(SAVE_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    total_test_loss = 0.0
    all_preds = []
    all_targets =[]
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            
            # [NEW] 同样的拼接逻辑
            src, dst = batch.edge_index
            edge_features = torch.cat([batch.x[src], batch.x[dst], batch.edge_attr], dim=-1)
            
            w_pred = model(edge_features)
            loss = criterion(w_pred, batch.y)
            total_test_loss += loss.item() * batch.num_edges
            
            all_preds.append(w_pred.cpu())
            all_targets.append(batch.y.cpu())
            
    avg_test_loss = total_test_loss / sum(data.num_edges for data in test_dataset)
    print(f"🏁 Final Test Result: Test MSE = {avg_test_loss:.4f}")

    y_true = torch.cat(all_targets).numpy()
    y_pred = torch.cat(all_preds).numpy()

    # (后面的画图代码没有任何改变，省略不影响运行)
    # 5. Visualize Training Curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, NUM_EPOCHS + 1), history['train_loss'], label='Train MSE')
    plt.plot(range(1, NUM_EPOCHS + 1), history['val_loss'], label='Val MSE')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title(f'Training & Validation Loss Curve\n(Final Test MSE: {avg_test_loss:.4f})')
    plt.legend()
    plt.grid(True)
    curve_path = os.path.join(RESULTS_DIR, "loss_curve.png")
    plt.savefig(curve_path)
    print(f"📊 Loss curve saved to: {curve_path}")

    # 6. Diagnostic Test: Scatter Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.4, color='blue', label='Predictions')
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
        f.write(f"Model Type: Regression (MLPBaseline)\n")
        f.write(f"Dataset Path: {DATA_DIR}\n")
        f.write(f"--- Hyperparameters ---\n")
        f.write(f"NODE_DIM: {NODE_DIM}\n")
        f.write(f"EDGE_RAW_DIM: {EDGE_RAW_DIM}\n")
        f.write(f"HIDDEN_DIM: {HIDDEN_DIM}\n")
        f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
        f.write(f"LEARNING_RATE: {LEARNING_RATE}\n")
        f.write(f"NUM_EPOCHS: {NUM_EPOCHS}\n")
        f.write(f"SEED: {SEED}\n")
        f.write(f"--- Results ---\n")
        f.write(f"Best Val MSE: {best_val_loss:.4f}\n")
        f.write(f"Final Test MSE: {avg_test_loss:.4f}\n")
    print(f"📝 Training summary saved at: {summary_path}")

if __name__ == "__main__":
    train_baseline()
