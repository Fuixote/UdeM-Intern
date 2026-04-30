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
from formulations.common.backend_utils import infer_ndd_mask
from formulations.pief.backend import solve_pief
from model.graph_utils import failure_context_edge_features, load_graph_dataset, lr_small_edge_features, parse_json_to_dfl_data
from model.model_structure import (
    DEFAULT_Y_SCALE,
    EDGE_RAW_DIM,
    FAILURE_CONTEXT_DIM,
    LR_SMALL_FEATURE_DIM,
    NODE_FEATURE_DIM,
    build_tabular_regression_model,
    normalize_tabular_model_family,
    tabular_model_class_name,
    tabular_model_label,
    tabular_model_result_token,
)
from split_binding import bind_dataset_to_split_files, deterministic_split_dataset, limit_training_dataset, save_split_files

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


def normalize_feature_mode(feature_mode):
    mode = (feature_mode or "full").strip().lower()
    if mode not in {"full", "utility_cpra", "failure_context", "lr_small"}:
        raise ValueError(f"Unsupported tabular feature mode: {feature_mode}")
    return mode


def default_feature_mode_for_family(model_family, feature_mode=None):
    if feature_mode is not None:
        return normalize_feature_mode(feature_mode)
    return "utility_cpra" if normalize_tabular_model_family(model_family) == "lr" else "full"


def infer_feature_mode_from_input_dim(input_dim):
    input_dim = int(input_dim)
    if input_dim == 2:
        return "utility_cpra"
    if input_dim == LR_SMALL_FEATURE_DIM:
        return "lr_small"
    if input_dim == FAILURE_CONTEXT_DIM:
        return "failure_context"
    return "full"


def tabular_input_dim(feature_mode):
    mode = normalize_feature_mode(feature_mode)
    if mode == "utility_cpra":
        return 2
    if mode == "lr_small":
        return LR_SMALL_FEATURE_DIM
    if mode == "failure_context":
        return FAILURE_CONTEXT_DIM
    return NODE_FEATURE_DIM * 2 + EDGE_RAW_DIM


def tabular_edge_features(data, feature_mode="full"):
    src, dst = data.edge_index
    mode = normalize_feature_mode(feature_mode)
    if mode == "utility_cpra":
        utility = data.edge_attr[:, :1]
        recipient_cpra = data.x[dst, 1:2]
        return torch.cat([utility, recipient_cpra], dim=-1)
    if mode == "lr_small":
        return lr_small_edge_features(data)
    if mode == "failure_context":
        return failure_context_edge_features(data)
    return torch.cat([data.x[src], data.x[dst], data.edge_attr], dim=-1)

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
def blackbox_kep_solver(w_preds_np, edge_index, node_is_ndd, num_edges, num_nodes, max_cycle=3, max_chain=4):
    env = gurobi_pool.get_env()
    try:
        result = solve_pief(
            weights=w_preds_np,
            edge_index=edge_index,
            is_ndd_mask=node_is_ndd,
            num_nodes=num_nodes,
            max_cycle=max_cycle,
            max_chain=max_chain,
            env=env,
        )
        if result["status"] not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            print(f"警告: Gurobi 求解失败，状态码: {result['status']}")
        return result["edge_selection"]
    finally:
        gurobi_pool.return_env(env)


def get_expected_y(w_preds_tensor, edge_index, node_is_ndd, num_nodes, num_edges, epsilon=0.1, M=8):
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
        return blackbox_kep_solver(w_np + noise, edge_index, node_is_ndd, num_edges, num_nodes)

    results = [solve_single(noise) for noise in noise_list]
    y_soft = np.mean(results, axis=0)
    return torch.tensor(y_soft, device=w_preds_tensor.device, dtype=torch.float32).view(-1, 1)


def sample_perturbed_solutions(w_preds_tensor, edge_index, node_is_ndd, num_nodes, num_edges, epsilon=0.1, M=8):
    """
    返回 perturbed optimizer 的 Monte Carlo 样本：
    - Y_samples[m] = y*(w + ε z^(m))

    这些样本作为常数参与 FY doubly stochastic 梯度估计，
    不经过求解器反向传播。
    """
    w_np = w_preds_tensor.detach().cpu().numpy().flatten()

    if epsilon > 0:
        noise_list = [np.random.normal(0, epsilon, size=w_np.shape).astype(np.float32) for _ in range(M)]
    else:
        noise_list = [np.zeros_like(w_np, dtype=np.float32) for _ in range(M)]

    def solve_single(noise):
        return blackbox_kep_solver(w_np + noise, edge_index, node_is_ndd, num_edges, num_nodes)

    results = [solve_single(noise) for noise in noise_list]
    y_samples = torch.tensor(np.stack(results), device=w_preds_tensor.device, dtype=torch.float32)
    return y_samples.detach()

def save_solutions(model, dataset, sol_dir, device, model_tag="Reg-PIEF-FY", feature_mode="full"):
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
            id_map_rev  = data.id_map_rev
            num_nodes   = data.num_nodes_custom[0].item()
            f_name      = data.filename
            node_is_ndd = infer_ndd_mask(data.x)

            # 模型预测权重（MLP 前向）
            edge_features = tabular_edge_features(data, feature_mode=feature_mode)
            w_preds       = model(edge_features).cpu().numpy().flatten()

            env = gurobi_pool.get_env()
            try:
                result = solve_pief(
                    weights=w_preds,
                    edge_index=data.edge_index,
                    is_ndd_mask=node_is_ndd,
                    num_nodes=num_nodes,
                    env=env,
                    id_map_rev=id_map_rev,
                )
                if result["matches"]:
                    payload = {
                        'graph': f_name,
                        'model_used': model_tag,
                        'formulation': 'pief',
                        'total_predicted_w': float(result["objective"]),
                        'num_matches': len(result["formatted_matches"]),
                        'matches': result["formatted_matches"],
                    }
                    out_path = os.path.join(sol_dir, f_name.replace('.json', '_sol.json'))
                    with open(out_path, 'w', encoding='utf-8') as handle:
                        json.dump(payload, handle, indent=4)
                    saved += 1
            finally:
                gurobi_pool.return_env(env)

    print(f"Solutions saved: {saved} files → {sol_dir}")

# w = ground_truth_label = success_prob × QALY × (1+ε)，上界 ≈ 25
# Y_SCALE 用于归一化：batch.y = w / Y_SCALE ∈ [0, 1]
# Stage-1 脚本使用原始 w（∈[0,25]），End2end 脚本使用归一化后的值
Y_SCALE = DEFAULT_Y_SCALE


def load_real_dataset_dfl(directory, max_cycle=3, max_chain=4):
    parser = lambda path: parse_json_to_dfl_data(path, max_cycle=max_cycle, max_chain=max_chain, label_scale=Y_SCALE)
    return load_graph_dataset(directory, parser, log_prefix="Loading")


def build_reg_model_from_config(config, model_family="mlp"):
    resolved_family = normalize_tabular_model_family(config.get("MODEL_FAMILY", model_family))
    resolved_feature_mode = default_feature_mode_for_family(resolved_family, config.get("FEATURE_MODE"))
    input_dim = int(config.get("INPUT_DIM", tabular_input_dim(resolved_feature_mode)))
    return build_tabular_regression_model(
        model_family=resolved_family,
        node_dim=config.get("NODE_DIM", NODE_FEATURE_DIM),
        edge_dim=config.get("EDGE_RAW_DIM", EDGE_RAW_DIM),
        hidden_dim=config.get("HIDDEN_DIM", 256),
        input_dim=input_dim,
    )


def load_pretrained_reg_model(pretrain_path, device, model_family="mlp"):
    ckpt = torch.load(pretrain_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    config = dict(ckpt.get("config", {}) if isinstance(ckpt, dict) else {})
    resolved_family = normalize_tabular_model_family(config.get("MODEL_FAMILY", model_family))
    input_dim = config.get("INPUT_DIM")
    if input_dim is None:
        first_weight = state_dict.get("net.0.weight")
        if first_weight is not None and getattr(first_weight, "ndim", 0) == 2:
            input_dim = int(first_weight.shape[1])
            config["INPUT_DIM"] = input_dim
    if "FEATURE_MODE" not in config:
        if input_dim is not None:
            config["FEATURE_MODE"] = infer_feature_mode_from_input_dim(input_dim)
        else:
            config["FEATURE_MODE"] = default_feature_mode_for_family(resolved_family)
    config["MODEL_FAMILY"] = resolved_family
    model = build_reg_model_from_config(config, model_family=model_family).to(device)
    model.load_state_dict(state_dict)
    return model, config


def result_prefix(model_family):
    return f"dfl_{tabular_model_result_token(model_family)}_pief_"


def solution_model_tag(model_family):
    return "LR-PIEF-FY" if model_family == "lr" else "Reg-PIEF-FY"


def checkpoint_model_label(model_family):
    return f"{tabular_model_label(model_family)} ({tabular_model_class_name(model_family)})"


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
def train_dfl(pretrain_path=None, data_dir=None, results_root=None, solutions_root=None,
              model_family="mlp", feature_mode=None, train_size=None, epsilon_init=None):
    model_family = normalize_tabular_model_family(model_family)
    feature_mode = default_feature_mode_for_family(model_family, feature_mode)
    input_dim = tabular_input_dim(feature_mode)
    DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LR            = 5e-4
    EPOCHS        = 10
    EPSILON_INIT  = float(epsilon_init) if epsilon_init is not None else 2.0
    M_SAMPLES     = 8       # 扰动采样次数，越高梯度估计越稳定但越慢；forward 需 M 次求解
    DATA_DIR      = str(resolve_path(data_dir or PROCESSED_DATA_DIR))
    STRICT_REPRODUCIBILITY = True
    PRETRAIN_PATH = str(resolve_path(pretrain_path)) if pretrain_path is not None else None
    PRETRAIN_TIMESTAMP = pretrain_timestamp_from_path(PRETRAIN_PATH) if PRETRAIN_PATH is not None else "scratch"
    TIMESTAMP   = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    result_suffix = f"{TIMESTAMP}_from_{PRETRAIN_TIMESTAMP}" if PRETRAIN_PATH is not None else f"{TIMESTAMP}_scratch"
    RESULTS_DIR = str(
        make_results_dir(
            result_prefix(model_family),
            timestamp=result_suffix,
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

        if PRETRAIN_PATH is not None:
            split_datasets, bound_split_paths = bind_dataset_to_split_files(full_dataset, PRETRAIN_PATH)
            split_binding_mode = "warm_start_bound"
        else:
            split_datasets = deterministic_split_dataset(full_dataset, seed=SEED)
            bound_split_paths = {}
            split_binding_mode = "scratch_deterministic"
        train_dataset = split_datasets["train"]
        val_dataset = split_datasets["val"]
        test_dataset = split_datasets["test"]
        original_train_count = len(train_dataset)
        train_dataset = limit_training_dataset(train_dataset, train_size)
        effective_train_size = len(train_dataset)
        print(
            f"Split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)} "
            f"(train pool before --train_size: {original_train_count})"
        )
        if PRETRAIN_PATH is not None:
            print(f"Bound train split from: {bound_split_paths['train']}")
            print(f"Bound validation split from: {bound_split_paths['val']}")
            print(f"Bound test split from: {bound_split_paths['test']}")
        else:
            print(f"Using deterministic scratch split with seed {SEED}.")

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
            model, pretrain_config = load_pretrained_reg_model(PRETRAIN_PATH, DEVICE, model_family=model_family)
            with torch.no_grad():
                model.net[-1].weight.data /= Y_SCALE
                if model.net[-1].bias is not None:
                    model.net[-1].bias.data /= Y_SCALE
            config_family = pretrain_config.get("MODEL_FAMILY", model_family)
            config_feature_mode = pretrain_config.get("FEATURE_MODE", feature_mode)
            if config_feature_mode != feature_mode:
                raise ValueError(
                    f"Pretrain checkpoint feature mode mismatch: expected {feature_mode}, got {config_feature_mode}. "
                    "Use a matching 2-stage LR checkpoint or pass the checkpoint's --feature_mode for legacy checkpoints."
                )
            print(f"Loaded pre-trained {tabular_model_label(config_family)} weights from: {PRETRAIN_PATH}")
            config_msg = (
                "Pre-train config: "
                f"MODEL_FAMILY={config_family}, "
                f"FEATURE_MODE={config_feature_mode}, "
                f"NODE_DIM={pretrain_config.get('NODE_DIM', NODE_FEATURE_DIM)}, "
                f"EDGE_RAW_DIM={pretrain_config.get('EDGE_RAW_DIM', EDGE_RAW_DIM)}"
            )
            if config_family == "mlp":
                config_msg += f", HIDDEN_DIM={pretrain_config.get('HIDDEN_DIM', 256)}"
            print(config_msg)
        else:
            model = build_tabular_regression_model(model_family=model_family, input_dim=input_dim).to(DEVICE)
            print(
                f"Training {tabular_model_label(model_family)} from scratch with Fenchel-Young loss "
                f"(feature_mode={feature_mode})."
            )

        optimizer     = optim.Adam(model.parameters(), lr=LR)
        best_val_loss = float('inf')

        for epoch in range(1, EPOCHS + 1):
            # Keep the perturbed optimizer noise fixed across epochs.
            current_epsilon = EPSILON_INIT
            model.train()
            total_train_regret = 0.0
            total_train_graphs = 0

            for step, batch in enumerate(train_loader):
                batch = batch.to(DEVICE)
                optimizer.zero_grad()
                node_is_ndd = infer_ndd_mask(batch.x)

                # MLP 前向：拼接 src/dst 节点特征 + 边特征，直接输出 logits
                edge_features = tabular_edge_features(batch, feature_mode=feature_mode)
                w_preds       = model(edge_features).view(-1, 1)

                # 1. Monte Carlo 采样：y^(m) = y*(w + ε z^(m))
                y_samples = sample_perturbed_solutions(
                    w_preds, batch.edge_index, node_is_ndd,
                    batch.num_nodes_custom[0].item(),
                    batch.num_edges,
                    epsilon=current_epsilon, M=M_SAMPLES
                )
                y_pred_soft = y_samples.mean(dim=0, keepdim=False).view(-1, 1)

                # 2. 真实权重下最优决策（无扰动，不参与反向传播）
                with torch.no_grad():
                    y_optimal = get_expected_y(
                        batch.y.view(-1, 1), batch.edge_index, node_is_ndd,
                        batch.num_nodes_custom[0].item(),
                        batch.num_edges,
                        epsilon=0.0, M=1
                    )

                # 3. FY 的 doubly stochastic 梯度估计（Berthet et al., Eq. 6）：
                #    ∇_w L_FY ≈ ȳ_ε(w) - y*
                # 这里用一个线性 proxy 让 autograd 返回上述梯度；
                # 它不是 FY loss 的闭式本体，而是与 FY 梯度一致的实现。
                w_flat = w_preds.view(-1)
                fy_direction = (y_pred_soft.detach() - y_optimal.detach()).view(-1)
                fy_loss = torch.dot(w_flat, fy_direction)

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
                    node_is_ndd = infer_ndd_mask(batch.x)
                    src, dst      = batch.edge_index
                    edge_features = tabular_edge_features(batch, feature_mode=feature_mode)
                    w_preds       = model(edge_features).view(-1, 1)

                    # 验证时用确定性求解（epsilon=0），衡量实际部署质量
                    y_pred = get_expected_y(
                        w_preds, batch.edge_index, node_is_ndd,
                        batch.num_nodes_custom[0].item(),
                        batch.num_edges,
                        epsilon=0.0, M=1
                    )
                    y_optimal = get_expected_y(
                        batch.y.view(-1, 1), batch.edge_index, node_is_ndd,
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
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_regret': best_val_loss,
                    'config': {
                        'MODEL_FAMILY': model_family,
                        'FEATURE_MODE': feature_mode,
                        'INPUT_DIM': input_dim,
                        'NODE_DIM': NODE_FEATURE_DIM,
                        'EDGE_RAW_DIM': EDGE_RAW_DIM,
                        'LR': LR, 'EPOCHS': EPOCHS,
                        'EPSILON_INIT': EPSILON_INIT,
                        'M_SAMPLES': M_SAMPLES, 'SEED': SEED,
                        'PRETRAIN_PATH': PRETRAIN_PATH,
                        'PRETRAIN_TIMESTAMP': PRETRAIN_TIMESTAMP,
                        'TRAIN_SIZE_REQUESTED': train_size,
                        'TRAIN_SIZE_EFFECTIVE': effective_train_size,
                        'TRAIN_POOL_SIZE': original_train_count,
                        'SPLIT_BINDING_MODE': split_binding_mode,
                        'BOUND_SPLIT_FILES': dict(bound_split_paths),
                        'STRICT_REPRODUCIBILITY': STRICT_REPRODUCIBILITY,
                        'CUBLAS_WORKSPACE_CONFIG': os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
                    }
                }
                if model_family == "mlp":
                    checkpoint["config"]["HIDDEN_DIM"] = 256
                torch.save(checkpoint, SAVE_PATH)
                print(f"  --> [Saved] Epoch {epoch} | New Best Val Regret: {best_val_loss:.4f}")

        print(f"\nTraining complete! Best Val Regret: {best_val_loss:.4f}")
        print(f"Model saved to: {SAVE_PATH}")

        split_paths = save_split_files(RESULTS_DIR, train_dataset, val_dataset, test_dataset)
        print(f"Train split file list saved to: {split_paths['train']}")
        print(f"Validation split file list saved to: {split_paths['val']}")
        print(f"Test split file list saved to: {split_paths['test']}")

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
                node_is_ndd = infer_ndd_mask(batch.x)
                edge_features = tabular_edge_features(batch, feature_mode=feature_mode)
                w_preds       = model(edge_features).view(-1, 1)

                # 用模型预测权重选边
                y_pred = get_expected_y(
                    w_preds, batch.edge_index, node_is_ndd,
                    batch.num_nodes_custom[0].item(),
                    batch.num_edges, epsilon=0.0, M=1
                )
                # 用真实 w 选边（Oracle 上界）
                y_optimal = get_expected_y(
                    batch.y.view(-1, 1), batch.edge_index, node_is_ndd,
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
            f.write(f"Model: {checkpoint_model_label(model_family)} + FY Loss\n")
            f.write(f"Pretrain Path     : {PRETRAIN_PATH}\n")
            f.write(f"Pretrain Timestamp: {PRETRAIN_TIMESTAMP}\n")
            f.write(f"Train Size Requested: {train_size}\n")
            f.write(f"Train Size Effective: {effective_train_size}\n")
            f.write(f"Train Pool Size   : {original_train_count}\n")
            f.write(f"Feature Mode      : {feature_mode}\n")
            f.write(f"Epsilon Init      : {EPSILON_INIT}\n")
            f.write(f"M Samples         : {M_SAMPLES}\n")
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
        save_solutions(model, full_dataset, sol_dir, DEVICE, model_tag=solution_model_tag(model_family), feature_mode=feature_mode)

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
    parser = argparse.ArgumentParser(description="End-to-end tabular training with Fenchel-Young loss")
    parser.add_argument("--pretrain_PATH", type=str, default=None,
                        help="Optional path to a pre-trained 2-stage model checkpoint (.pth); if omitted, train from scratch")
    parser.add_argument("--data_dir", type=str, default=str(PROCESSED_DATA_DIR),
                        help="Directory containing processed G-*.json graphs")
    parser.add_argument("--results_root", type=str, default=str(RESULTS_ROOT),
                        help="Root directory where timestamped training outputs will be created")
    parser.add_argument("--solutions_root", type=str, default=str(SOLUTIONS_ROOT),
                        help="Root directory where timestamped solution outputs will be created")
    parser.add_argument("--model_family", type=str, default="mlp", choices=["mlp", "lr"],
                        help="Tabular prediction model used in the end-to-end pipeline")
    parser.add_argument("--feature_mode", type=str, default=None, choices=["full", "utility_cpra", "failure_context", "lr_small"],
                        help="Tabular feature set. Defaults to utility_cpra for lr, full features for reg.")
    parser.add_argument(
        "--train_size",
        type=int,
        default=None,
        help="Number of graphs to use from the training split; default uses the full training split",
    )
    parser.add_argument("--epsilon", type=float, default=None,
                        help="FY perturbation epsilon. Defaults to 2.0 for tabular end-to-end training.")
    args = parser.parse_args()
    
    train_dfl(
        pretrain_path=args.pretrain_PATH,
        data_dir=args.data_dir,
        results_root=args.results_root,
        solutions_root=args.solutions_root,
        model_family=args.model_family,
        feature_mode=args.feature_mode,
        train_size=args.train_size,
        epsilon_init=args.epsilon,
    )
