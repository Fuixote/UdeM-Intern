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
from model.graph_utils import find_all_cycles_and_chains, parse_json_to_graph_info
from model.model_structure import KidneyEdgePredictor, MLPBaseline

# ==========================================
# 3. Solver logic
# ==========================================

def solve_kep(json_path, model, model_type, output_dir, max_chain=5):
    f_name = os.path.basename(json_path)
    graph_data = parse_json_to_graph_info(json_path)
    
    # Prediction or Ground Truth selection
    x = graph_data['x']
    edge_index = graph_data['edge_index']
    edge_attr = graph_data['edge_attr']
    
    if model is not None:
        with torch.no_grad():
            if model_type == "GNN":
                w_preds = model(x, edge_index, edge_attr).numpy()
            else:
                src, dst = edge_index
                edge_features = torch.cat([x[src], x[dst], edge_attr], dim=-1)
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
    parser.add_argument("--max_chain", type=int, default=5,
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
        model_dir = os.path.dirname(args.model_path)
        summary_path = os.path.join(model_dir, "summary.txt")
        with open(summary_path, 'r') as f:
            summary_content = f.read()
        
        if "GNN" in summary_content:
            model_type, model = "GNN", KidneyEdgePredictor()
        else:
            model_type, model = "Regression", MLPBaseline()
        
        ckpt = torch.load(args.model_path, map_location='cpu')
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
        model.eval()
        exp_id = os.path.basename(model_dir)
        sol_out = str(solution_dir_for_experiment(exp_id, solutions_root=solutions_root))
        print(f"🚀 Running in PREDICTION mode: {model_type}")

    os.makedirs(sol_out, exist_ok=True)
    print(f"📁 Solutions will be saved to: {sol_out}")
    
    files = sorted(glob.glob(os.path.join(data_dir, "G-*.json")))
    for f in files:
        solve_kep(f, model, model_type, sol_out, args.max_chain)
