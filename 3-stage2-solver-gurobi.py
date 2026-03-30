import argparse
import shutil
import sys
import torch
import gurobipy as gp
from gurobipy import GRB
import json
import os
import glob
from experiment_config import PROCESSED_DATA_DIR, RESULTS_ROOT, SOLUTIONS_ROOT, resolve_path, solution_dir_for_experiment
from model.graph_utils import find_all_cycles_and_chains, parse_json_to_graph_info, resolve_graph_data_dir
from model.model_structure import EDGE_RAW_DIM, NODE_FEATURE_DIM, KidneyEdgePredictor, MLPBaseline

# ==========================================
# 3. Solver logic
# ==========================================

def infer_model_type(summary_content, state_dict):
    if "GNN" in summary_content:
        return "GNN"
    if "Regression" in summary_content or "MLP" in summary_content:
        return "Regression"
    if any(key.startswith("conv1.") or key.startswith("edge_encoder.") for key in state_dict):
        return "GNN"
    return "Regression"


def build_model_from_checkpoint(model_type, config):
    if model_type == "GNN":
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


def load_prediction_model(model_path):
    model_dir = os.path.dirname(model_path)
    summary_path = os.path.join(model_dir, "summary.txt")
    summary_content = ""
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            summary_content = f.read()

    checkpoint = torch.load(model_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        config = checkpoint.get('config', {})
    else:
        state_dict = checkpoint
        config = {}

    model_type = infer_model_type(summary_content, state_dict)
    model = build_model_from_checkpoint(model_type, config)
    model.load_state_dict(state_dict)
    model.expected_node_dim = config.get("NODE_DIM", NODE_FEATURE_DIM)
    model.expected_edge_raw_dim = config.get("EDGE_RAW_DIM", EDGE_RAW_DIM)
    model.eval()
    return model_type, model

def solve_kep(json_path, model, model_type, output_dir, max_chain=4):
    f_name = os.path.basename(json_path)
    graph_data = parse_json_to_graph_info(json_path)
    
    # Prediction or Ground Truth selection
    x = graph_data['x']
    edge_index = graph_data['edge_index']
    edge_attr = graph_data['edge_attr']
    
    if model is not None:
        expected_node_dim = getattr(model, "expected_node_dim", x.size(-1))
        expected_edge_raw_dim = getattr(model, "expected_edge_raw_dim", edge_attr.size(-1))

        if x.size(-1) != expected_node_dim:
            raise ValueError(
                f"Node feature dimension mismatch for {json_path}: "
                f"graph has {x.size(-1)}, checkpoint expects {expected_node_dim}"
            )
        if edge_attr.size(-1) < expected_edge_raw_dim:
            raise ValueError(
                f"Edge feature dimension mismatch for {json_path}: "
                f"graph has {edge_attr.size(-1)}, checkpoint expects at least {expected_edge_raw_dim}"
            )

        edge_attr_for_model = edge_attr[:, :expected_edge_raw_dim]
        with torch.no_grad():
            if model_type == "GNN":
                w_preds = model(x, edge_index, edge_attr_for_model).numpy()
            else:
                src, dst = edge_index
                edge_features = torch.cat([x[src], x[dst], edge_attr_for_model], dim=-1)
                w_preds = model(edge_features).numpy()
    else:
        # Ground Truth Mode (Oracle)
        w_preds = graph_data['gt_labels']

    cycles, chains = find_all_cycles_and_chains(graph_data['adj'], graph_data['nodes_data'], graph_data['id_map_rev'], max_chain=max_chain)
    candidates = cycles + chains
    if not candidates: return

    m = gp.Model("KEP")
    m.Params.OutputFlag = 0
    y = m.addVars(len(candidates), vtype=GRB.BINARY, name="y")
    cand_weights = [sum(w_preds[e_idx] for e_idx in c['edges']) for c in candidates]
    m.setObjective(gp.quicksum(cand_weights[i] * y[i] for i in range(len(candidates))), GRB.MAXIMIZE)

    for n_idx in range(graph_data['num_nodes']):
        involved = [i for i, c in enumerate(candidates) if n_idx in c['nodes']]
        if involved: m.addConstr(gp.quicksum(y[i] for i in involved) <= 1)

    m.optimize()

    if m.status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and m.SolCount > 0:
        sel = [i for i in range(len(candidates)) if y[i].X > 0.5]
        matches = []
        for idx in sel:
            c = candidates[idx]
            matches.append({
                'type': 'cycle' if idx < len(cycles) else 'chain',
                'node_ids': [graph_data['id_map_rev'][nid] for nid in c['nodes']],
                'predicted_w': float(cand_weights[idx]),
                'edge_weights': [float(w_preds[e_idx]) for e_idx in c['edges']]
            })
        
        res = {
            'graph': f_name, 'model_used': model_type,
            'total_predicted_w': float(m.ObjVal),
            'num_matches': len(matches), 'matches': matches
        }
        with open(os.path.join(output_dir, f_name.replace(".json", "_sol.json")), 'w') as out_f:
            json.dump(res, out_f, indent=4)
        print(f"✅ Saved solution for {f_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to best_stage1_model_real.pth. In gt_mode: optional, used to copy test_files.txt for fair comparison")
    parser.add_argument("--max_chain", type=int, default=4,
                        help="Maximum number of transplant edges in a chain (excluding the initiating NDD node)")
    parser.add_argument("--data_dir", type=str, default=str(PROCESSED_DATA_DIR))
    parser.add_argument("--results_root", type=str, default=str(RESULTS_ROOT))
    parser.add_argument("--solutions_root", type=str, default=str(SOLUTIONS_ROOT))
    parser.add_argument("--gt_mode", action="store_true", help="Oracle mode: use Ground Truth labels as weights")
    args = parser.parse_args()
    data_dir = str(resolve_path(args.data_dir))
    results_root = str(resolve_path(args.results_root))
    solutions_root = str(resolve_path(args.solutions_root))
    if args.model_path is not None:
        args.model_path = str(resolve_path(args.model_path))

    if args.gt_mode:
        model_type, model = "GroundTruth", None
        sol_out = str(solution_dir_for_experiment("ground_truth", solutions_root=solutions_root))
        print("💡 Running in ORACLE mode (using Ground Truth labels)")

        # 创建 results/ground_truth/ 并复制 test_files.txt，使 4-evaluation 与预测模型使用相同测试集
        results_gt_dir = os.path.join(results_root, "ground_truth")
        if args.model_path:
            model_dir = os.path.dirname(args.model_path)
            src_test_files = os.path.join(model_dir, "test_files.txt")
            if os.path.exists(src_test_files):
                os.makedirs(results_gt_dir, exist_ok=True)
                dst_test_files = os.path.join(results_gt_dir, "test_files.txt")
                shutil.copy2(src_test_files, dst_test_files)
                print(f"📄 Copied test_files.txt from {model_dir} → {results_gt_dir} (for fair comparison)")
            else:
                print(f"⚠️ {src_test_files} not found, 4-evaluation will use all solutions")
        else:
            print("💡 Tip: Use --model_path <stage1_model.pth> to copy test_files.txt for fair comparison")
    else:
        if not args.model_path:
            print("❌ Error: --model_path is required unless --gt_mode is used.")
            sys.exit(1)
        model_type, model = load_prediction_model(args.model_path)
        model_dir = os.path.dirname(args.model_path)
        exp_id = os.path.basename(model_dir)
        sol_out = str(solution_dir_for_experiment(exp_id, solutions_root=solutions_root))
        print(f"🚀 Running in PREDICTION mode: {model_type}")

    os.makedirs(sol_out, exist_ok=True)
    print(f"📁 Solutions will be saved to: {sol_out}")

    data_dir, files = resolve_graph_data_dir(data_dir, log_prefix="🔍 Solving")
    for f in files:
        solve_kep(f, model, model_type, sol_out, args.max_chain)
