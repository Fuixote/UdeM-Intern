import argparse
import shutil
import sys
import torch
import torch.nn as nn
import gurobipy as gp
from gurobipy import GRB
import json
import os
import glob
from datetime import datetime
import time

# ==========================================
# 0. Model Architectures
# ==========================================

class MLPBaseline(nn.Module):
    def __init__(self, node_dim=13, edge_dim=1, hidden_dim=256):
        super(MLPBaseline, self).__init__()
        input_dim = node_dim * 2 + edge_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
    def forward(self, edge_features):
        return self.net(edge_features).squeeze(-1)

# GNN parts
from torch_geometric.utils import scatter

class DirectedEdgeConv(nn.Module):
    def __init__(self, hidden_dim):
        super(DirectedEdgeConv, self).__init__()
        self.W_self = nn.Linear(hidden_dim, hidden_dim)
        self.W_in = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_out = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.act = nn.LeakyReLU(0.2)
    def forward(self, edge_attr, edge_index, num_nodes):
        src, dst = edge_index
        h_self = self.W_self(edge_attr)
        node_in_feats = scatter(edge_attr, dst, dim=0, dim_size=num_nodes, reduce='mean')
        agg_in = node_in_feats[src]
        node_out_feats = scatter(edge_attr, src, dim=0, dim_size=num_nodes, reduce='mean')
        agg_out = node_out_feats[dst]
        h_neigh = self.W_in(agg_in) + self.W_out(agg_out)
        return self.act(h_self + h_neigh)

class KidneyEdgePredictor(nn.Module):
    def __init__(self, node_feature_dim=13, edge_raw_dim=1, hidden_dim=64):
        super(KidneyEdgePredictor, self).__init__()
        concat_dim = node_feature_dim * 2 + edge_raw_dim
        self.edge_encoder = nn.Sequential(nn.Linear(concat_dim, hidden_dim), nn.LeakyReLU(0.2))
        self.conv1 = DirectedEdgeConv(hidden_dim)
        self.conv2 = DirectedEdgeConv(hidden_dim)
        self.conv3 = DirectedEdgeConv(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 32), nn.LeakyReLU(0.2),
            nn.Linear(32, 1)
        )
    def forward(self, x, edge_index, raw_edge_attr):
        num_nodes = x.size(0)
        src, dst = edge_index
        edge_attr = torch.cat([x[src], x[dst], raw_edge_attr], dim=-1)
        h_e = self.edge_encoder(edge_attr)
        h_e = self.conv1(h_e, edge_index, num_nodes)
        h_e = self.conv2(h_e, edge_index, num_nodes)
        h_e = self.conv3(h_e, edge_index, num_nodes)
        weight_pred = self.mlp(h_e)
        return weight_pred.squeeze(-1)

# ==========================================
# 1. Feature Extraction & Data Loading
# ==========================================

BT_MAP = {"O": 0, "A": 1, "B": 2, "AB": 3}
def get_one_hot_bt(bt_str):
    vec = [0.0, 0.0, 0.0, 0.0]
    if bt_str in BT_MAP: vec[BT_MAP[bt_str]] = 1.0
    return vec

def parse_json_to_graph_info(json_path):
    with open(json_path, 'r') as f:
        content = json.load(f)
    nodes_data = content['data']
    node_ids = sorted(nodes_data.keys(), key=lambda x: int(x))
    id_map = {old_id: i for i, old_id in enumerate(node_ids)}
    id_map_rev = {i: old_id for i, old_id in enumerate(node_ids)}
    
    x_list = []
    adj = {} # src_idx -> list of dicts
    edges_info = []

    for nid in node_ids:
        node = nodes_data[nid]
        if node['type'] == 'Pair':
            p = node['patient']; d = node['donors'][0]
            feat = [p['age']/100.0, p['cPRA'], 1.0 if p['hasBloodCompatibleDonor'] else 0.0] + \
                   get_one_hot_bt(p['bloodtype']) + [d['dage']/100.0] + get_one_hot_bt(d['bloodtype']) + [0.0]
        else:
            d = node['donor']
            feat = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, d['dage']/100.0] + get_one_hot_bt(d['bloodtype']) + [1.0]
        x_list.append(feat)
        adj[id_map[nid]] = []

    x = torch.tensor(x_list, dtype=torch.float)
    edge_indices = []
    edge_attrs = []
    
    edge_count = 0
    for src_id, node in nodes_data.items():
        for match in node['matches']:
            dst_id = match['recipient']
            src_idx = id_map[src_id]
            dst_idx = id_map[dst_id]
            edge_indices.append([src_idx, dst_idx])
            edge_attrs.append([match['utility'] / 100.0])
            adj[src_idx].append({'dst': dst_idx, 'edge_idx': edge_count})
            edges_info.append({'src': src_id, 'dst': dst_id})
            edge_count += 1
            
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    
    # Extract Ground Truth Labels for Oracle Mode
    gt_labels = []
    for src_id, node in nodes_data.items():
        for match in node['matches']:
            gt_labels.append(match.get('ground_truth_label', 0.0))
    
    return {
        'x': x, 'edge_index': edge_index, 'edge_attr': edge_attr,
        'adj': adj, 'nodes_data': nodes_data, 'id_map_rev': id_map_rev,
        'num_nodes': len(node_ids), 'gt_labels': gt_labels
    }

# ==========================================
# 2. DFS for Enumeration
# ==========================================

def find_all_cycles_and_chains(adj, nodes_data, id_map_rev, max_cycle=3, max_chain=5):
    cycles = []
    chains = []
    num_nodes = len(nodes_data)
    
    ndd_indices = [i for i in range(num_nodes) if nodes_data[id_map_rev[i]]['type'] != 'Pair']
    pair_indices = [i for i in range(num_nodes) if nodes_data[id_map_rev[i]]['type'] == 'Pair']

    def dfs_chain(u, current_path, current_edges):
        if len(current_edges) >= max_chain:
            return
        for edge in adj[u]:
            v = edge['dst']
            if v not in current_path:
                chains.append({'nodes': current_path + [v], 'edges': current_edges + [edge['edge_idx']]})
                dfs_chain(v, current_path + [v], current_edges + [edge['edge_idx']])

    for ndd in ndd_indices:
        dfs_chain(ndd, [ndd], [])

    def dfs_cycle(start_node, u, current_path, current_edges):
        if len(current_path) > max_cycle: return
        for edge in adj[u]:
            v = edge['dst']
            if v == start_node:
                if start_node == min(current_path):
                    cycles.append({'nodes': current_path, 'edges': current_edges + [edge['edge_idx']]})
            elif v not in current_path and v > start_node:
                if nodes_data[id_map_rev[v]]['type'] == 'Pair':
                    dfs_cycle(start_node, v, current_path + [v], current_edges + [edge['edge_idx']])

    for p_idx in pair_indices:
        dfs_cycle(p_idx, p_idx, [p_idx], [])
            
    return cycles, chains

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
    parser.add_argument("--data_dir", type=str, default="dataset/processed")
    parser.add_argument("--gt_mode", action="store_true", help="Oracle mode: use Ground Truth labels as weights")
    args = parser.parse_args()

    if args.gt_mode:
        model_type, model = "GroundTruth", None
        sol_out = os.path.join("solutions", "ground_truth")
        print("💡 Running in ORACLE mode (using Ground Truth labels)")

        # 创建 results/ground_truth/ 并复制 test_files.txt，使 4-evaluation 与预测模型使用相同测试集
        results_gt_dir = os.path.join("results", "ground_truth")
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
        sol_out = os.path.join("solutions", exp_id)
        print(f"🚀 Running in PREDICTION mode: {model_type}")

    os.makedirs(sol_out, exist_ok=True)
    print(f"📁 Solutions will be saved to: {sol_out}")
    
    files = sorted(glob.glob(os.path.join(args.data_dir, "G-*.json")))
    for f in files:
        solve_kep(f, model, model_type, sol_out, args.max_chain)
