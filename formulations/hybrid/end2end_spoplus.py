import argparse
import json
import os
import queue
import random
import re
import sys
import threading
from datetime import datetime
from pathlib import Path

import gurobipy as gp
import numpy as np
import torch
import torch.optim as optim
from gurobipy import GRB
from torch_geometric.loader import DataLoader

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment_config import (  # noqa: E402
    PROCESSED_DATA_DIR,
    RESULTS_ROOT,
    SOLUTIONS_ROOT,
    make_results_dir,
    resolve_path,
    solution_dir_for_result_dir,
)
from formulations.common.backend_utils import infer_ndd_mask  # noqa: E402
from formulations.hybrid.backend import solve_cf_cycle_pief_chain  # noqa: E402
from model.graph_utils import load_graph_dataset, parse_json_to_dfl_data  # noqa: E402
from model.model_structure import (  # noqa: E402
    DEFAULT_Y_SCALE,
    EDGE_RAW_DIM,
    NODE_FEATURE_DIM,
    KidneyEdgePredictor,
    MLPBaseline,
)
from split_binding import (  # noqa: E402
    bind_dataset_to_split_files,
    deterministic_split_dataset,
    save_split_files,
)

SEED = 42
Y_SCALE = DEFAULT_Y_SCALE


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


class GurobiEnvPool:
    def __init__(self, pool_size=8):
        self.pool_size = pool_size
        self.env_queue = queue.Queue()
        self._lock = threading.Lock()
        self._initialized = False

    def initialize(self):
        with self._lock:
            if self._initialized:
                return
            for _ in range(self.pool_size):
                env = gp.Env(empty=True)
                env.setParam("OutputFlag", 0)
                env.setParam("Seed", SEED)
                env.start()
                self.env_queue.put(env)
            self._initialized = True
            print(f"Gurobi env pool initialized: size={self.pool_size}")

    def get_env(self):
        if not self._initialized:
            self.initialize()
        try:
            return self.env_queue.get(timeout=5.0)
        except queue.Empty:
            env = gp.Env(empty=True)
            env.setParam("OutputFlag", 0)
            env.setParam("Seed", SEED)
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
            print("Gurobi env pool cleaned up")


gurobi_pool = GurobiEnvPool(pool_size=8)


def blackbox_kep_solver(weights_np, cycle_candidates, edge_index, node_is_ndd, num_edges, num_nodes, max_chain=4):
    del num_edges
    env = gurobi_pool.get_env()
    try:
        result = solve_cf_cycle_pief_chain(
            weights=weights_np,
            edge_index=edge_index,
            is_ndd_mask=node_is_ndd,
            num_nodes=num_nodes,
            cycle_candidates=cycle_candidates,
            max_chain=max_chain,
            env=env,
        )
        if result["status"] not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            print(f"Warning: Gurobi solve returned status={result['status']}")
        return result["edge_selection"]
    finally:
        gurobi_pool.return_env(env)


def cycle_candidates_only(candidates_container):
    if candidates_container and isinstance(candidates_container[0], list):
        candidates = candidates_container[0]
    else:
        candidates = candidates_container
    return [candidate for candidate in candidates if candidate["type"] == "cycle"]


def forward_edge_weights(model, model_type, data):
    if model_type == "gnn":
        return model(data.x, data.edge_index, data.edge_attr).view(-1, 1)
    src, dst = data.edge_index
    edge_features = torch.cat([data.x[src], data.x[dst], data.edge_attr], dim=-1)
    return model(edge_features).view(-1, 1)


def solve_decision(weight_tensor, cycle_candidates, edge_index, node_is_ndd, num_nodes, num_edges):
    weights_np = weight_tensor.detach().cpu().numpy().flatten()
    selection = blackbox_kep_solver(weights_np, cycle_candidates, edge_index, node_is_ndd, num_edges, num_nodes)
    return torch.tensor(selection, device=weight_tensor.device, dtype=torch.float32).view(-1, 1)


def solve_loss_augmented_decision(pred_weight_tensor, true_weight_tensor, cycle_candidates, edge_index, node_is_ndd, num_nodes, num_edges):
    # Maximization-form SPO+: solve argmax_y (2 w_hat - w_true)^T y.
    adjusted_weights = (2.0 * pred_weight_tensor.detach() - true_weight_tensor.detach()).cpu().numpy().flatten()
    selection = blackbox_kep_solver(adjusted_weights, cycle_candidates, edge_index, node_is_ndd, num_edges, num_nodes)
    return torch.tensor(selection, device=pred_weight_tensor.device, dtype=torch.float32).view(-1, 1)


def evaluate_regret(model, model_type, dataset, device):
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    total_achieved = 0.0
    total_optimal = 0.0
    total_graphs = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            node_is_ndd = infer_ndd_mask(batch.x)
            cycle_candidates = cycle_candidates_only(batch.candidates)
            pred_w = forward_edge_weights(model, model_type, batch)
            y_pred = solve_decision(
                pred_w,
                cycle_candidates,
                batch.edge_index,
                node_is_ndd,
                batch.num_nodes_custom[0].item(),
                batch.num_edges,
            )
            y_optimal = solve_decision(
                batch.y.view(-1, 1),
                cycle_candidates,
                batch.edge_index,
                node_is_ndd,
                batch.num_nodes_custom[0].item(),
                batch.num_edges,
            )

            true_w = batch.y.view(-1, 1)
            achieved = torch.sum(true_w * Y_SCALE * y_pred).item()
            optimal = torch.sum(true_w * Y_SCALE * y_optimal).item()
            total_achieved += achieved
            total_optimal += optimal
            total_graphs += 1

    avg_achieved = total_achieved / total_graphs if total_graphs > 0 else 0.0
    avg_optimal = total_optimal / total_graphs if total_graphs > 0 else 0.0
    avg_regret = avg_optimal - avg_achieved
    rel_regret_pct = 100.0 * avg_regret / avg_optimal if avg_optimal > 0 else float("inf")
    return {
        "graphs": total_graphs,
        "avg_achieved": avg_achieved,
        "avg_optimal": avg_optimal,
        "avg_regret": avg_regret,
        "rel_regret_pct": rel_regret_pct,
    }


def load_real_dataset(directory, max_cycle=3, max_chain=4):
    parser = lambda path: parse_json_to_dfl_data(  # noqa: E731
        path, max_cycle=max_cycle, max_chain=max_chain, label_scale=Y_SCALE
    )
    return load_graph_dataset(directory, parser, log_prefix="Loading")


def build_model(model_type, config=None):
    config = config or {}
    if model_type == "gnn":
        return KidneyEdgePredictor(
            node_feature_dim=config.get("NODE_DIM", NODE_FEATURE_DIM),
            edge_raw_dim=config.get("EDGE_RAW_DIM", EDGE_RAW_DIM),
            hidden_dim=config.get("HIDDEN_DIM", 64),
        )
    return MLPBaseline(
        node_dim=config.get("NODE_DIM", NODE_FEATURE_DIM),
        edge_dim=config.get("EDGE_RAW_DIM", EDGE_RAW_DIM),
        hidden_dim=config.get("HIDDEN_DIM", 256),
    )


def load_pretrained_model(model_type, pretrain_path, device):
    checkpoint = torch.load(pretrain_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        config = checkpoint.get("config", {})
    else:
        state_dict = checkpoint
        config = {}
    model = build_model(model_type, config).to(device)
    model.load_state_dict(state_dict)
    return model, config


def rescale_pretrained_output_head(model, model_type):
    with torch.no_grad():
        if model_type == "gnn":
            model.mlp[-1].weight.data /= Y_SCALE
            model.mlp[-1].bias.data /= Y_SCALE
        else:
            model.net[-1].weight.data /= Y_SCALE
            model.net[-1].bias.data /= Y_SCALE


def result_prefix(model_type):
    return "spoplus_Gnn_hybrid_" if model_type == "gnn" else "spoplus_Reg_hybrid_"


def model_label(model_type):
    return "GNN" if model_type == "gnn" else "MLP"


def model_tag(model_type):
    return "GNN-Hybrid-SPOPlus" if model_type == "gnn" else "Reg-Hybrid-SPOPlus"


def pretrain_timestamp_from_path(pretrain_path):
    parent_name = os.path.basename(os.path.dirname(pretrain_path))
    match = re.search(r"(\d{4}-\d{2}-\d{2}_\d{6})", parent_name)
    if match:
        return match.group(1)
    raise ValueError(
        f"Unable to extract timestamp from pretrain checkpoint parent directory: {parent_name}"
    )


def save_model_solutions(model, model_type, dataset, sol_dir, device):
    os.makedirs(sol_dir, exist_ok=True)
    model.eval()
    saved = 0

    with torch.no_grad():
        for data in dataset:
            data = data.to(device)
            candidates = cycle_candidates_only(data.candidates)
            id_map_rev = data.id_map_rev
            num_nodes = data.num_nodes_custom[0].item()
            file_name = data.filename
            node_is_ndd = infer_ndd_mask(data.x)
            pred_w = forward_edge_weights(model, model_type, data).cpu().numpy().flatten()

            env = gurobi_pool.get_env()
            try:
                result = solve_cf_cycle_pief_chain(
                    weights=pred_w,
                    edge_index=data.edge_index,
                    is_ndd_mask=node_is_ndd,
                    num_nodes=num_nodes,
                    cycle_candidates=candidates,
                    env=env,
                    id_map_rev=id_map_rev,
                )
                if result["matches"]:
                    payload = {
                        "graph": file_name,
                        "model_used": model_tag(model_type),
                        "formulation": "hybrid",
                        "total_predicted_w": float(result["objective"]),
                        "num_matches": len(result["formatted_matches"]),
                        "matches": result["formatted_matches"],
                    }
                    out_path = os.path.join(sol_dir, file_name.replace(".json", "_sol.json"))
                    with open(out_path, "w", encoding="utf-8") as handle:
                        json.dump(payload, handle, indent=4)
                    saved += 1
            finally:
                gurobi_pool.return_env(env)

    print(f"Solutions saved: {saved} files -> {sol_dir}")


def write_epoch0_eval(path, pretrain_path, pretrain_timestamp, epoch0_val, epoch0_test):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("Epoch 0 warm-start evaluation (before SPO+ updates)\n")
        handle.write(f"Pretrain Path     : {pretrain_path}\n")
        handle.write(f"Pretrain Timestamp: {pretrain_timestamp}\n")
        handle.write(
            f"Validation        : graphs={epoch0_val['graphs']}, "
            f"avg_optimal={epoch0_val['avg_optimal']:.4f}, "
            f"avg_achieved={epoch0_val['avg_achieved']:.4f}, "
            f"avg_regret={epoch0_val['avg_regret']:.4f}, "
            f"relative_regret={epoch0_val['rel_regret_pct']:.2f}%\n"
        )
        handle.write(
            f"Test              : graphs={epoch0_test['graphs']}, "
            f"avg_optimal={epoch0_test['avg_optimal']:.4f}, "
            f"avg_achieved={epoch0_test['avg_achieved']:.4f}, "
            f"avg_regret={epoch0_test['avg_regret']:.4f}, "
            f"relative_regret={epoch0_test['rel_regret_pct']:.2f}%\n"
        )


def write_summary(path, model_type, data_dir, lr, epochs, pretrain_path, pretrain_timestamp, best_val_regret, test_metrics):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"Training Time: {os.path.basename(os.path.dirname(path))}\n")
        handle.write(f"Model Type: {model_label(model_type)} + SPO+\n")
        handle.write(f"Dataset Path: {data_dir}\n")
        handle.write("--- Hyperparameters ---\n")
        handle.write(f"LEARNING_RATE: {lr}\n")
        handle.write(f"NUM_EPOCHS: {epochs}\n")
        handle.write(f"SEED: {SEED}\n")
        handle.write(f"PRETRAIN_PATH: {pretrain_path}\n")
        handle.write(f"PRETRAIN_TIMESTAMP: {pretrain_timestamp}\n")
        handle.write(f"Y_SCALE: {Y_SCALE}\n")
        handle.write(f"STRICT_REPRODUCIBILITY: True\n")
        handle.write(f"CUBLAS_WORKSPACE_CONFIG: {os.environ.get('CUBLAS_WORKSPACE_CONFIG', '')}\n")
        handle.write("--- Results ---\n")
        handle.write(f"Best Val Regret: {best_val_regret:.4f}\n")
        handle.write(f"Test Avg Optimal w: {test_metrics['avg_optimal']:.4f}\n")
        handle.write(f"Test Avg Achieved w: {test_metrics['avg_achieved']:.4f}\n")
        handle.write(f"Test Avg Regret: {test_metrics['avg_regret']:.4f}\n")
        handle.write(f"Test Relative Regret: {test_metrics['rel_regret_pct']:.2f}%\n")


def train_spoplus(model_type, pretrain_path=None, data_dir=None, results_root=None, solutions_root=None, lr=1e-4, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = str(resolve_path(data_dir or PROCESSED_DATA_DIR))
    pretrain_path = str(resolve_path(pretrain_path)) if pretrain_path is not None else None
    pretrain_timestamp = pretrain_timestamp_from_path(pretrain_path) if pretrain_path is not None else "scratch"

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    result_suffix = f"{timestamp}_from_{pretrain_timestamp}" if pretrain_path is not None else f"{timestamp}_scratch"
    results_dir = str(
        make_results_dir(
            result_prefix(model_type),
            timestamp=result_suffix,
            results_root=results_root or RESULTS_ROOT,
        )
    )
    save_path = os.path.join(results_dir, "best_spoplus_model.pth")
    print(f"Results will be saved at: {results_dir}")

    try:
        full_dataset = load_real_dataset(data_dir)
        if not full_dataset:
            print("Error: processed JSON data not found")
            return

        if pretrain_path is not None:
            split_datasets, bound_split_paths = bind_dataset_to_split_files(full_dataset, pretrain_path)
            split_binding_mode = "warm_start_bound"
        else:
            split_datasets = deterministic_split_dataset(full_dataset, seed=SEED)
            bound_split_paths = {}
            split_binding_mode = "scratch_deterministic"

        train_dataset = split_datasets["train"]
        val_dataset = split_datasets["val"]
        test_dataset = split_datasets["test"]
        print(f"Split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
        if pretrain_path is not None:
            print(f"Bound train split from: {bound_split_paths['train']}")
            print(f"Bound validation split from: {bound_split_paths['val']}")
            print(f"Bound test split from: {bound_split_paths['test']}")
        else:
            print(f"Using deterministic scratch split with seed {SEED}.")

        train_loader_generator = torch.Generator()
        train_loader_generator.manual_seed(SEED)
        train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=0,
            generator=train_loader_generator,
        )

        if pretrain_path is not None:
            model, pretrain_config = load_pretrained_model(model_type, pretrain_path, device)
            rescale_pretrained_output_head(model, model_type)
            print(f"Loaded pre-trained {model_label(model_type)} weights from: {pretrain_path}")
            print(
                "Pre-train config: "
                f"NODE_DIM={pretrain_config.get('NODE_DIM', NODE_FEATURE_DIM)}, "
                f"EDGE_RAW_DIM={pretrain_config.get('EDGE_RAW_DIM', EDGE_RAW_DIM)}, "
                f"HIDDEN_DIM={pretrain_config.get('HIDDEN_DIM', 64 if model_type == 'gnn' else 256)}"
            )
        else:
            model = build_model(model_type).to(device)
            print(f"Training {model_label(model_type)} from scratch with SPO+ loss.")

        optimizer = optim.Adam(model.parameters(), lr=lr)

        epoch0_val = evaluate_regret(model, model_type, val_dataset, device)
        epoch0_test = evaluate_regret(model, model_type, test_dataset, device)
        epoch0_eval_path = os.path.join(results_dir, "epoch0_warmstart_eval.txt")
        write_epoch0_eval(epoch0_eval_path, pretrain_path, pretrain_timestamp, epoch0_val, epoch0_test)

        best_val_regret = epoch0_val["avg_regret"]
        torch.save(
            {
                "epoch": 0,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_regret": best_val_regret,
                "config": {
                    "MODEL_TYPE": model_type,
                    "LR": lr,
                    "EPOCHS": epochs,
                    "SEED": SEED,
                    "Y_SCALE": Y_SCALE,
                    "PRETRAIN_PATH": pretrain_path,
                    "PRETRAIN_TIMESTAMP": pretrain_timestamp,
                    "SPLIT_BINDING_MODE": split_binding_mode,
                    "BOUND_SPLIT_FILES": dict(bound_split_paths),
                    "STRICT_REPRODUCIBILITY": True,
                    "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
                    "NODE_DIM": NODE_FEATURE_DIM,
                    "EDGE_RAW_DIM": EDGE_RAW_DIM,
                    "HIDDEN_DIM": 64 if model_type == "gnn" else 256,
                },
            },
            save_path,
        )
        print(
            f"Epoch 0 baseline saved as initial best checkpoint | "
            f"Val Regret: {best_val_regret:.4f} ({epoch0_val['rel_regret_pct']:.2f}% of optimal)"
        )

        for epoch in range(1, epochs + 1):
            model.train()
            total_train_surrogate = 0.0
            total_train_regret = 0.0
            total_train_graphs = 0

            for step, batch in enumerate(train_loader):
                batch = batch.to(device)
                optimizer.zero_grad()
                node_is_ndd = infer_ndd_mask(batch.x)
                candidates = cycle_candidates_only(batch.candidates)

                pred_w = forward_edge_weights(model, model_type, batch)
                true_w = batch.y.view(-1, 1)

                with torch.no_grad():
                    y_optimal = solve_decision(
                        true_w,
                        candidates,
                        batch.edge_index,
                        node_is_ndd,
                        batch.num_nodes_custom[0].item(),
                        batch.num_edges,
                    )
                    y_loss_aug = solve_loss_augmented_decision(
                        pred_w,
                        true_w,
                        candidates,
                        batch.edge_index,
                        node_is_ndd,
                        batch.num_nodes_custom[0].item(),
                        batch.num_edges,
                    )
                    y_pred = solve_decision(
                        pred_w,
                        candidates,
                        batch.edge_index,
                        node_is_ndd,
                        batch.num_nodes_custom[0].item(),
                        batch.num_edges,
                    )

                decision_gap = (y_loss_aug - y_optimal).detach().view(-1)
                spo_loss = 2.0 * torch.sum(pred_w.view(-1) * decision_gap)
                spo_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_regret = torch.sum(true_w * Y_SCALE * (y_optimal - y_pred)).item()
                total_train_surrogate += spo_loss.item()
                total_train_regret += train_regret
                total_train_graphs += 1

                if (step + 1) % 25 == 0:
                    print(
                        f"  [Epoch {epoch:03d} | Step {step + 1}/{len(train_loader)}] "
                        f"Avg SPO+ surrogate: {total_train_surrogate / total_train_graphs:.4f} | "
                        f"Avg regret: {total_train_regret / total_train_graphs:.4f}",
                        flush=True,
                    )

            avg_train_surrogate = total_train_surrogate / total_train_graphs if total_train_graphs > 0 else 0.0
            avg_train_regret = total_train_regret / total_train_graphs if total_train_graphs > 0 else 0.0
            val_metrics = evaluate_regret(model, model_type, val_dataset, device)
            avg_val_regret = val_metrics["avg_regret"]

            print(
                f"Epoch {epoch:03d}/{epochs} [SPO+] | "
                f"Train Surrogate: {avg_train_surrogate:.4f} | "
                f"Train Regret: {avg_train_regret:.4f} | "
                f"Val Regret: {avg_val_regret:.4f} "
                f"({val_metrics['rel_regret_pct']:.2f}% of optimal {val_metrics['avg_optimal']:.3f})"
            )

            if avg_val_regret < best_val_regret:
                best_val_regret = avg_val_regret
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_regret": best_val_regret,
                        "config": {
                            "MODEL_TYPE": model_type,
                            "LR": lr,
                            "EPOCHS": epochs,
                            "SEED": SEED,
                            "Y_SCALE": Y_SCALE,
                            "PRETRAIN_PATH": pretrain_path,
                            "PRETRAIN_TIMESTAMP": pretrain_timestamp,
                            "SPLIT_BINDING_MODE": split_binding_mode,
                            "BOUND_SPLIT_FILES": dict(bound_split_paths),
                            "STRICT_REPRODUCIBILITY": True,
                            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
                            "NODE_DIM": NODE_FEATURE_DIM,
                            "EDGE_RAW_DIM": EDGE_RAW_DIM,
                            "HIDDEN_DIM": 64 if model_type == "gnn" else 256,
                        },
                    },
                    save_path,
                )
                print(f"  --> [Saved] Epoch {epoch} | New Best Val Regret: {best_val_regret:.4f}")

        print(f"\nTraining complete! Best Val Regret: {best_val_regret:.4f}")
        print(f"Model saved to: {save_path}")

        split_paths = save_split_files(results_dir, train_dataset, val_dataset, test_dataset)
        print(f"Train split file list saved to: {split_paths['train']}")
        print(f"Validation split file list saved to: {split_paths['val']}")
        print(f"Test split file list saved to: {split_paths['test']}")

        checkpoint = torch.load(save_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        test_metrics = evaluate_regret(model, model_type, test_dataset, device)
        test_result_path = os.path.join(results_dir, "test_result.txt")
        with open(test_result_path, "w", encoding="utf-8") as handle:
            handle.write(f"Model: {model_label(model_type)} + SPO+\n")
            handle.write(f"Pretrain Path     : {pretrain_path}\n")
            handle.write(f"Pretrain Timestamp: {pretrain_timestamp}\n")
            handle.write(f"Strict Repro      : True\n")
            handle.write(f"CUBLAS Workspace  : {os.environ.get('CUBLAS_WORKSPACE_CONFIG', '')}\n")
            handle.write(f"Test graphs      : {test_metrics['graphs']}\n")
            handle.write(f"Avg Optimal w    : {test_metrics['avg_optimal']:.4f}\n")
            handle.write(f"Avg Achieved w   : {test_metrics['avg_achieved']:.4f}\n")
            handle.write(f"Avg Regret       : {test_metrics['avg_regret']:.4f}\n")
            handle.write(f"Relative Regret  : {test_metrics['rel_regret_pct']:.2f}%\n")
        print(f"Test results saved to: {test_result_path}")

        summary_path = os.path.join(results_dir, "summary.txt")
        write_summary(
            summary_path,
            model_type,
            data_dir,
            lr,
            epochs,
            pretrain_path,
            pretrain_timestamp,
            best_val_regret,
            test_metrics,
        )
        print(f"Summary saved to: {summary_path}")

        sol_dir = str(solution_dir_for_result_dir(results_dir, solutions_root=solutions_root or SOLUTIONS_ROOT))
        print(f"\nSaving solutions to: {sol_dir}")
        save_model_solutions(model, model_type, full_dataset, sol_dir, device)

    except Exception as exc:
        import traceback

        print(f"Training failed: {exc}")
        traceback.print_exc()
    finally:
        print("Cleaning up Gurobi resources...")
        try:
            gurobi_pool.cleanup()
        except Exception as cleanup_error:
            print(f"Cleanup failed: {cleanup_error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid end-to-end SPO+ training for kidney exchange")
    parser.add_argument("--model_type", choices=["gnn", "reg"], default="gnn")
    parser.add_argument(
        "--pretrain_PATH",
        type=str,
        default=None,
        help="Optional 2-stage checkpoint (.pth). If omitted, train from scratch.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(PROCESSED_DATA_DIR),
        help="Directory containing processed G-*.json graphs",
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default=str(RESULTS_ROOT),
        help="Root directory where timestamped training outputs will be created",
    )
    parser.add_argument(
        "--solutions_root",
        type=str,
        default=str(SOLUTIONS_ROOT),
        help="Root directory where timestamped solution outputs will be created",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    train_spoplus(
        model_type=args.model_type,
        pretrain_path=args.pretrain_PATH,
        data_dir=args.data_dir,
        results_root=args.results_root,
        solutions_root=args.solutions_root,
        lr=args.lr,
        epochs=args.epochs,
    )
