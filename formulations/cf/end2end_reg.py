import argparse
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import sys
from pathlib import Path
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import random
from datetime import datetime
import threading
import queue
import json
import re

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment_config import (
    PROCESSED_DATA_DIR,
    RESULTS_ROOT,
    SOLUTIONS_ROOT,
    make_results_dir,
    resolve_path,
    solution_dir_for_result_dir,
)
from model.graph_utils import load_graph_dataset, parse_json_to_dfl_data
from model.model_structure import DEFAULT_Y_SCALE, EDGE_RAW_DIM, NODE_FEATURE_DIM, MLPBaseline

# ==========================================
# Global Settings
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
# Gurobi 环境池（避免频繁创建/销毁）
# ==========================================
class GurobiEnvPool:
    """线程安全的 Gurobi 环境池"""
    def __init__(self, pool_size=4):
        self.pool_size = pool_size
        self.env_queue = queue.Queue()
        self._lock = threading.Lock()
        self._initialized = False

    def initialize(self):
        with self._lock:
            if not self._initialized:
                for _ in range(self.pool_size):
                    env = gp.Env(empty=True)
                    env.setParam('OutputFlag', 0)
                    env.setParam('Seed', SEED)
                    env.start()
                    self.env_queue.put(env)
                self._initialized = True
                print(f"Gurobi 环境池已初始化，大小: {self.pool_size}")

    def get_env(self):
        if not self._initialized:
            self.initialize()
        try:
            return self.env_queue.get(timeout=5.0)
        except queue.Empty:
            env = gp.Env(empty=True)
            env.setParam('OutputFlag', 0)
            env.setParam('Seed', SEED)
            env.start()
            return env

    def return_env(self, env):
        if env is not None:
            self.env_queue.put(env)

    def cleanup(self):
        with self._lock:
            while not self.env_queue.empty():
                try:
                    env = self.env_queue.get_nowait()
                    env.dispose()
                except queue.Empty:
                    break
            self._initialized = False
            print("Gurobi 环境池已清理")

# 全局环境池实例
gurobi_pool = GurobiEnvPool(pool_size=8)

# ==========================================
# 1. 核心求解器：带环境池的黑盒
# ==========================================
def blackbox_kep_solver(w_preds_np, candidates, num_edges, num_nodes):
    if not candidates:
        return np.zeros(num_edges)

    env = gurobi_pool.get_env()
    m = None
    try:
        m = gp.Model("KEP_Blackbox", env=env)
        m.setParam('Threads', 1)
        m.setParam('Seed', SEED)
        m.setParam('TimeLimit', 10)

        y_var = m.addVars(len(candidates), vtype=GRB.BINARY, name="y")
        cand_weights = [sum(w_preds_np[e_idx] for e_idx in c['edges']) for c in candidates]

        m.setObjective(
            gp.quicksum(cand_weights[i] * y_var[i] for i in range(len(candidates))),
            GRB.MAXIMIZE
        )

        for n_idx in range(num_nodes):
            involved = [i for i, c in enumerate(candidates) if n_idx in c['nodes']]
            if involved:
                m.addConstr(gp.quicksum(y_var[i] for i in involved) <= 1)

        m.optimize()

        edge_selections = np.zeros(num_edges)
        if m.status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and m.SolCount > 0:
            for i, c in enumerate(candidates):
                if y_var[i].X > 0.5:
                    for e_idx in c['edges']:
                        edge_selections[e_idx] = 1.0
        elif m.status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            print(f"警告: Gurobi 求解失败，状态码: {m.status}")

        return edge_selections
    finally:
        if m is not None:
            try:
                m.dispose()
            except Exception:
                pass
        gurobi_pool.return_env(env)


def get_expected_y(w_preds_tensor, candidates, num_nodes, num_edges, epsilon=0.1, M=8):
    """
    计算扰动优化器的期望决策分布（纯前向，不参与反向传播）。

    对应论文 NeurIPS 2020 "Learning with Differentiable Perturbed Optimizers" 中的
    E_Z[ y*(θ + εZ) ]：对预测权重加 M 次随机噪声，分别求解，取平均得到软决策。

    梯度不经过此函数传播；梯度通过 Fenchel-Young 代理损失直接作用于 w_preds。
    """
    w_np = w_preds_tensor.detach().cpu().numpy().flatten()

    if epsilon > 0:
        noise_list = [np.random.normal(0, epsilon, size=w_np.shape).astype(np.float32) for _ in range(M)]
    else:
        noise_list = [np.zeros_like(w_np, dtype=np.float32) for _ in range(M)]

    def solve_single(noise):
        return blackbox_kep_solver(w_np + noise, candidates, num_edges, num_nodes)

    results = [solve_single(noise) for noise in noise_list]
    y_soft = np.mean(results, axis=0)
    return torch.tensor(y_soft, device=w_preds_tensor.device, dtype=torch.float32).view(-1, 1)


def sample_perturbed_solutions(w_preds_tensor, candidates, num_nodes, num_edges, epsilon=0.1, M=8):
    """
    返回严格 FY 标量 surrogate 所需的 Monte Carlo 样本：
    - Y_samples[m] = y*(w + ε z^(m))
    - Z_samples[m] = z^(m)

    两者都作为常数样本参与后续标量目标构造，不经过求解器反向传播。
    """
    w_np = w_preds_tensor.detach().cpu().numpy().flatten()

    if epsilon > 0:
        noise_list = [np.random.normal(0, epsilon, size=w_np.shape).astype(np.float32) for _ in range(M)]
    else:
        noise_list = [np.zeros_like(w_np, dtype=np.float32) for _ in range(M)]

    def solve_single(noise):
        return blackbox_kep_solver(w_np + noise, candidates, num_edges, num_nodes)

    results = [solve_single(noise) for noise in noise_list]
    y_samples = torch.tensor(np.stack(results), device=w_preds_tensor.device, dtype=torch.float32)
    z_samples = torch.tensor(np.stack(noise_list), device=w_preds_tensor.device, dtype=torch.float32)
    return y_samples.detach(), z_samples.detach()

def save_solutions(model, dataset, sol_dir, device, model_tag="Reg-CF-FY"):
    """
    在给定数据集上用模型预测权重求解 KEP，输出与 formulations/cf/stage2_solver.py
    完全相同格式的 *_sol.json 文件，供 4-evaluation.py 横向对比。
    """
    os.makedirs(sol_dir, exist_ok=True)
    model.eval()
    saved = 0

    with torch.no_grad():
        for data in dataset:
            data = data.to(device)
            candidates  = data.candidates
            id_map_rev  = data.id_map_rev
            num_nodes   = data.num_nodes_custom[0].item()
            num_edges   = data.num_edges
            f_name      = data.filename

            if not candidates:
                continue

            # 模型预测权重（MLP 前向）
            src, dst      = data.edge_index
            edge_features = torch.cat([data.x[src], data.x[dst], data.edge_attr], dim=-1)
            w_preds       = model(edge_features).cpu().numpy().flatten()

            # Gurobi 求解（无扰动，确定性）
            env = gurobi_pool.get_env()
            m = None
            try:
                m = gp.Model("KEP_Sol", env=env)
                m.setParam('OutputFlag', 0)
                m.setParam('Threads', 1)
                m.setParam('Seed', SEED)
                y_var = m.addVars(len(candidates), vtype=GRB.BINARY, name="y")
                cand_weights = [sum(w_preds[e_idx] for e_idx in c['edges']) for c in candidates]
                m.setObjective(
                    gp.quicksum(cand_weights[i] * y_var[i] for i in range(len(candidates))),
                    GRB.MAXIMIZE
                )
                for n_idx in range(num_nodes):
                    involved = [i for i, c in enumerate(candidates) if n_idx in c['nodes']]
                    if involved:
                        m.addConstr(gp.quicksum(y_var[i] for i in involved) <= 1)
                m.optimize()

                if m.status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and m.SolCount > 0:
                    matches = []
                    for i, c in enumerate(candidates):
                        if y_var[i].X > 0.5:
                            node_ids_str = [id_map_rev[n] for n in c['nodes']]
                            matches.append({
                                'type': c['type'],   # 'cycle' or 'chain'，由 find_all_cycles_and_chains 标注
                                'node_ids': node_ids_str,
                                'predicted_w': float(cand_weights[i]),
                                'edge_weights': [float(w_preds[e_idx]) for e_idx in c['edges']]
                            })

                    sol = {
                        'graph': f_name,
                        'model_used': model_tag,
                        'formulation': 'cf',
                        'total_predicted_w': float(m.ObjVal),
                        'num_matches': len(matches),
                        'matches': matches
                    }
                    out_path = os.path.join(sol_dir, f_name.replace('.json', '_sol.json'))
                    with open(out_path, 'w') as f:
                        json.dump(sol, f, indent=4)
                    saved += 1
            finally:
                if m is not None:
                    try:
                        m.dispose()
                    except Exception:
                        pass
                gurobi_pool.return_env(env)

    print(f"Solutions saved: {saved} files → {sol_dir}")

# w = ground_truth_label = success_prob × QALY × (1+ε)，上界 ≈ 25
# Y_SCALE 用于归一化：batch.y = w / Y_SCALE ∈ [0, 1]
# Stage-1 脚本使用原始 w（∈[0,25]），End2end 脚本使用归一化后的值
Y_SCALE = DEFAULT_Y_SCALE


def load_real_dataset_dfl(directory, max_cycle=3, max_chain=4):
    parser = lambda path: parse_json_to_dfl_data(path, max_cycle=max_cycle, max_chain=max_chain, label_scale=Y_SCALE)
    return load_graph_dataset(directory, parser, log_prefix="Loading")


def build_reg_model_from_config(config):
    return MLPBaseline(
        node_dim=config.get("NODE_DIM", NODE_FEATURE_DIM),
        edge_dim=config.get("EDGE_RAW_DIM", EDGE_RAW_DIM),
        hidden_dim=config.get("HIDDEN_DIM", 256),
    )


def load_pretrained_reg_model(pretrain_path, device):
    ckpt = torch.load(pretrain_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    model = build_reg_model_from_config(config).to(device)
    model.load_state_dict(state_dict)
    return model, config


def pretrain_timestamp_from_path(pretrain_path):
    parent_name = os.path.basename(os.path.dirname(pretrain_path))
    match = re.search(r"(\d{4}-\d{2}-\d{2}_\d{6})", parent_name)
    if match:
        return match.group(1)
    raise ValueError(
        f"Unable to extract timestamp from pretrain checkpoint parent directory: {parent_name}"
    )

# ==========================================
# 3. 训练主流程
# ==========================================
def train_dfl(pretrain_path=None, data_dir=None, results_root=None, solutions_root=None):
    DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LR            = 5e-4
    EPOCHS        = 3
    EPSILON_INIT  = 0.5
    M_SAMPLES     = 8       # 扰动采样次数，越高梯度估计越稳定但越慢；forward 需 M 次求解
    DATA_DIR      = str(resolve_path(data_dir or PROCESSED_DATA_DIR))
    STRICT_REPRODUCIBILITY = True
    if pretrain_path is None:
        raise ValueError("--pretrain_PATH is required for dfl-reg")
    PRETRAIN_PATH = str(resolve_path(pretrain_path))
    PRETRAIN_TIMESTAMP = pretrain_timestamp_from_path(PRETRAIN_PATH)
    TIMESTAMP   = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    RESULTS_DIR = str(
        make_results_dir(
            "dfl_Reg_cf_",
            timestamp=f"{TIMESTAMP}_from_{PRETRAIN_TIMESTAMP}",
            results_root=results_root or RESULTS_ROOT,
        )
    )
    SAVE_PATH   = os.path.join(RESULTS_DIR, 'best_dfl_reg_model.pth')
    print(f"Results will be saved at: {RESULTS_DIR}")

    try:
        # ---- 数据加载 ----
        full_dataset = load_real_dataset_dfl(DATA_DIR)
        if not full_dataset:
            print("Error: Processed JSON data not found!")
            return

        random.shuffle(full_dataset)
        total_len = len(full_dataset)
        train_end = int(0.6 * total_len)
        val_end   = int(0.8 * total_len)

        train_dataset = full_dataset[:train_end]
        val_dataset   = full_dataset[train_end:val_end]
        test_dataset  = full_dataset[val_end:]
        print(f"Split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

        # batch_size 必须为 1：求解器以单图为单位调用
        train_loader_generator = torch.Generator()
        train_loader_generator.manual_seed(SEED)
        train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=0,
            generator=train_loader_generator,
        )
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

        # ---- 初始化 ----
        if PRETRAIN_PATH is not None:
            model, pretrain_config = load_pretrained_reg_model(PRETRAIN_PATH, DEVICE)
            # Stage-1 MLP 输出范围 ~[0, Y_SCALE]，缩放到合理的 logit 范围
            with torch.no_grad():
                model.net[-1].weight.data /= Y_SCALE
                model.net[-1].bias.data   /= Y_SCALE
            print(f"Loaded pre-trained MLP weights from: {PRETRAIN_PATH}")
            print(
                "Pre-train config: "
                f"NODE_DIM={pretrain_config.get('NODE_DIM', NODE_FEATURE_DIM)}, "
                f"EDGE_RAW_DIM={pretrain_config.get('EDGE_RAW_DIM', EDGE_RAW_DIM)}, "
                f"HIDDEN_DIM={pretrain_config.get('HIDDEN_DIM', 256)}"
            )
        else:
            # MLP 仅拼接 src/dst 节点特征和边特征，不感知全图拓扑
            model = MLPBaseline().to(DEVICE)
            print("Training MLP from scratch with Fenchel-Young loss.")

        optimizer     = optim.Adam(model.parameters(), lr=LR)
        best_val_loss = float('inf')

        for epoch in range(1, EPOCHS + 1):
            # epsilon 衰减：训练前期探索性强，后期收敛
            current_epsilon = EPSILON_INIT * (0.95 ** (epoch - 1))
            model.train()
            total_train_regret = 0.0
            total_train_graphs = 0

            for step, batch in enumerate(train_loader):
                batch = batch.to(DEVICE)
                optimizer.zero_grad()

                # MLP 前向：拼接 src/dst 节点特征 + 边特征，直接输出 logits
                src, dst      = batch.edge_index
                edge_features = torch.cat([batch.x[src], batch.x[dst], batch.edge_attr], dim=-1)
                w_preds       = model(edge_features).view(-1, 1)

                # 1. Monte Carlo 采样：y^(m) = y*(w + ε z^(m))
                y_samples, z_samples = sample_perturbed_solutions(
                    w_preds, batch.candidates[0],
                    batch.num_nodes_custom[0].item(),
                    batch.num_edges,
                    epsilon=current_epsilon, M=M_SAMPLES
                )
                y_pred_soft = y_samples.mean(dim=0, keepdim=False).view(-1, 1)

                # 2. 真实权重下最优决策（无扰动，不参与反向传播）
                with torch.no_grad():
                    y_optimal = get_expected_y(
                        batch.y.view(-1, 1), batch.candidates[0],
                        batch.num_nodes_custom[0].item(),
                        batch.num_edges,
                        epsilon=0.0, M=1
                    )

                # 3. 更严格的 FY 标量 surrogate（差一个与参数无关的常数项）：
                #    F_ε(w) ≈ (1/M) Σ_m <w + ε z^(m), y^(m)>
                #    L_FY(w; y*) = F_ε(w) - <w, y*>
                # 其中 y^(m), z^(m), y* 都视为 stop-gradient 常数。
                w_flat = w_preds.view(-1)
                f_hat = torch.mean(torch.sum((w_flat.unsqueeze(0) + z_samples) * y_samples, dim=1))
                target_term = torch.sum(w_flat * y_optimal.detach().view(-1))
                fy_loss = f_hat - target_term

                fy_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                # 4. Regret 仅用于日志记录（不参与反向传播）
                # 单位：期望收益/图 = sum_edges(ground_truth_label × decision_diff)
                # ground_truth_label = success_prob × QALY × (1+ε)，最大约 25
                true_w = batch.y.view(-1, 1)
                train_regret = torch.sum(true_w * Y_SCALE * (y_optimal - y_pred_soft)).item()
                total_train_regret += train_regret
                total_train_graphs += 1

                if (step + 1) % 50 == 0:
                    print(f"  [Epoch {epoch:03d} | Step {step+1}/{len(train_loader)}] "
                          f"Avg Regret so far: {total_train_regret / total_train_graphs:.4f} | "
                          f"ε={current_epsilon:.3f}", flush=True)

            avg_train_regret = total_train_regret / total_train_graphs

            # ---- 验证：用 epsilon=0 的确定性决策计算 regret ----
            model.eval()
            total_val_regret = 0.0
            total_val_optimal = 0.0
            total_val_graphs = 0

            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(DEVICE)
                    src, dst      = batch.edge_index
                    edge_features = torch.cat([batch.x[src], batch.x[dst], batch.edge_attr], dim=-1)
                    w_preds       = model(edge_features).view(-1, 1)

                    # 验证时用确定性求解（epsilon=0），衡量实际部署质量
                    y_pred = get_expected_y(
                        w_preds, batch.candidates[0],
                        batch.num_nodes_custom[0].item(),
                        batch.num_edges,
                        epsilon=0.0, M=1
                    )
                    y_optimal = get_expected_y(
                        batch.y.view(-1, 1), batch.candidates[0],
                        batch.num_nodes_custom[0].item(),
                        batch.num_edges,
                        epsilon=0.0, M=1
                    )

                    true_w = batch.y.view(-1, 1)
                    # regret = sum_edges(ground_truth_label × decision_diff)
                    # ground_truth_label = success_prob × QALY × (1+ε)，单位为期望收益
                    regret = torch.sum(true_w * Y_SCALE * (y_optimal - y_pred)).item()
                    # optimal_obj = 用真实权重求解时能获得的最大期望收益（per graph）
                    optimal_obj = torch.sum(true_w * Y_SCALE * y_optimal).item()
                    total_val_regret  += regret
                    total_val_optimal += optimal_obj
                    total_val_graphs  += 1

            avg_val_regret  = total_val_regret  / total_val_graphs if total_val_graphs > 0 else float('inf')
            avg_val_optimal = total_val_optimal / total_val_graphs if total_val_graphs > 0 else float('inf')
            # 相对 regret：损失的 QALY 占最优解的百分比
            rel_regret_pct = 100.0 * avg_val_regret / avg_val_optimal if avg_val_optimal > 0 else float('inf')

            print(f"Epoch {epoch:03d}/{EPOCHS} [regret(M={M_SAMPLES})] | "
                  f"Train Regret: {avg_train_regret:.4f} | "
                  f"Val Regret: {avg_val_regret:.4f} "
                  f"({rel_regret_pct:.2f}% of optimal {avg_val_optimal:.3f}) | "
                  f"Epsilon: {current_epsilon:.3f}")

            if avg_val_regret < best_val_loss:
                best_val_loss = avg_val_regret
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_regret': best_val_loss,
                    'config': {
                        'LR': LR, 'EPOCHS': EPOCHS,
                        'EPSILON_INIT': EPSILON_INIT,
                        'M_SAMPLES': M_SAMPLES, 'SEED': SEED,
                        'PRETRAIN_PATH': PRETRAIN_PATH,
                        'PRETRAIN_TIMESTAMP': PRETRAIN_TIMESTAMP,
                        'STRICT_REPRODUCIBILITY': STRICT_REPRODUCIBILITY,
                        'CUBLAS_WORKSPACE_CONFIG': os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
                    }
                }, SAVE_PATH)
                print(f"  --> [Saved] Epoch {epoch} | New Best Val Regret: {best_val_loss:.4f}")

        print(f"\nTraining complete! Best Val Regret: {best_val_loss:.4f}")
        print(f"Model saved to: {SAVE_PATH}")

        # 保存测试集文件列表，供 4-evaluation.py 自动过滤
        test_files_path = os.path.join(RESULTS_DIR, "test_files.txt")
        with open(test_files_path, 'w') as f:
            for data in test_dataset:
                f.write(data.filename + '\n')
        print(f"Test file list saved to: {test_files_path}")

        # ---- 测试集评估：加载最优模型，在测试集上选边并计算总 w ----
        print("\n" + "="*60)
        print("TEST SET EVALUATION")
        print("="*60)
        ckpt = torch.load(SAVE_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
        total_test_achieved = 0.0   # 模型决策获得的总 w
        total_test_optimal  = 0.0   # 理论最优总 w（oracle）
        total_test_graphs   = 0

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(DEVICE)
                src, dst      = batch.edge_index
                edge_features = torch.cat([batch.x[src], batch.x[dst], batch.edge_attr], dim=-1)
                w_preds       = model(edge_features).view(-1, 1)

                # 用模型预测权重选边
                y_pred = get_expected_y(
                    w_preds, batch.candidates[0],
                    batch.num_nodes_custom[0].item(),
                    batch.num_edges, epsilon=0.0, M=1
                )
                # 用真实 w 选边（Oracle 上界）
                y_optimal = get_expected_y(
                    batch.y.view(-1, 1), batch.candidates[0],
                    batch.num_nodes_custom[0].item(),
                    batch.num_edges, epsilon=0.0, M=1
                )

                true_w = batch.y.view(-1, 1)
                # 模型方案实际获得的期望收益
                achieved = torch.sum(true_w * Y_SCALE * y_pred).item()
                # Oracle 最优方案的期望收益
                optimal  = torch.sum(true_w * Y_SCALE * y_optimal).item()
                total_test_achieved += achieved
                total_test_optimal  += optimal
                total_test_graphs   += 1

        avg_achieved = total_test_achieved / total_test_graphs
        avg_optimal  = total_test_optimal  / total_test_graphs
        avg_regret   = avg_optimal - avg_achieved
        rel_pct      = 100.0 * avg_regret / avg_optimal if avg_optimal > 0 else float('inf')

        print(f"Test graphs      : {total_test_graphs}")
        print(f"Avg Optimal w    : {avg_optimal:.4f}  (oracle upper bound)")
        print(f"Avg Achieved w   : {avg_achieved:.4f}  (model decisions)")
        print(f"Avg Regret       : {avg_regret:.4f}  ({rel_pct:.2f}% of optimal)")

        # 保存测试结果
        test_result_path = os.path.join(RESULTS_DIR, "test_result.txt")
        with open(test_result_path, 'w') as f:
            f.write(f"Model: MLP (MLPBaseline) + FY Loss\n")
            f.write(f"Pretrain Path     : {PRETRAIN_PATH}\n")
            f.write(f"Pretrain Timestamp: {PRETRAIN_TIMESTAMP}\n")
            f.write(f"Strict Repro      : {STRICT_REPRODUCIBILITY}\n")
            f.write(f"CUBLAS Workspace  : {os.environ.get('CUBLAS_WORKSPACE_CONFIG', '')}\n")
            f.write(f"Test graphs      : {total_test_graphs}\n")
            f.write(f"Avg Optimal w    : {avg_optimal:.4f}\n")
            f.write(f"Avg Achieved w   : {avg_achieved:.4f}\n")
            f.write(f"Avg Regret       : {avg_regret:.4f}\n")
            f.write(f"Relative Regret  : {rel_pct:.2f}%\n")
        print(f"Test results saved to: {test_result_path}")

        # ---- 输出 solution 文件，供 4-evaluation.py 横向对比 ----
        sol_dir = str(solution_dir_for_result_dir(RESULTS_DIR, solutions_root=solutions_root or SOLUTIONS_ROOT))
        print(f"\nSaving solutions to: {sol_dir}")
        save_solutions(model, full_dataset, sol_dir, DEVICE, model_tag="Reg-CF-FY")

    except Exception as e:
        import traceback
        print(f"训练过程中发生错误: {e}")
        traceback.print_exc()

    finally:
        print("清理 Gurobi 资源...")
        try:
            gurobi_pool.cleanup()
        except Exception as cleanup_error:
            print(f"清理资源时发生错误: {cleanup_error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end Reg (MLP) training with Fenchel-Young loss")
    parser.add_argument("--pretrain_PATH", type=str, required=True,
                        help="Path to a pre-trained 2-stage model checkpoint (.pth) to initialize MLP weights")
    parser.add_argument("--data_dir", type=str, default=str(PROCESSED_DATA_DIR),
                        help="Directory containing processed G-*.json graphs")
    parser.add_argument("--results_root", type=str, default=str(RESULTS_ROOT),
                        help="Root directory where timestamped training outputs will be created")
    parser.add_argument("--solutions_root", type=str, default=str(SOLUTIONS_ROOT),
                        help="Root directory where timestamped solution outputs will be created")
    args = parser.parse_args()
    
    train_dfl(
        pretrain_path=args.pretrain_PATH,
        data_dir=args.data_dir,
        results_root=args.results_root,
        solutions_root=args.solutions_root,
    )
