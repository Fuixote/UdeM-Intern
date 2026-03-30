import os
import json
import glob
import argparse
import numpy as np

def evaluate_directory(sol_dir, data_dir, test_list_path=None, use_all_files=False):
    test_filter = None
    
    # 1. Try to find test list
    if test_list_path and os.path.exists(test_list_path):
        with open(test_list_path, 'r') as f:
            test_filter = set(line.strip() for line in f if line.strip())
    else:
        # Smart search: look for test_files.txt or all_files.txt in the corresponding results folder
        exp_id = os.path.basename(sol_dir)
        list_name = "all_files.txt" if use_all_files else "test_files.txt"
        auto_test_path = os.path.join("results", exp_id, list_name)
        if os.path.exists(auto_test_path):
            with open(auto_test_path, 'r') as f:
                test_filter = set(line.strip() for line in f if line.strip())
            print(f"💡 Auto-detected {list_name} for {exp_id} at {auto_test_path} ({len(test_filter)} graphs)")

    if test_filter:
        test_filter = {name.replace('.json', '') for name in test_filter}

    sol_files = sorted(glob.glob(os.path.join(sol_dir, "*_sol.json")))
    if test_filter:
        sol_files = [f for f in sol_files if os.path.basename(f).replace('_sol.json', '') in test_filter]

    if not sol_files:
        return None

    overall = {
        'total_gt_score': 0,
        'total_edges_selected': 0,
        'total_matches': 0,
        'graph_count': 0
    }

    for sol_path in sol_files:
        with open(sol_path, 'r') as f:
            sol_data = json.load(f)
        
        graph_name = sol_data.get('graph', os.path.basename(sol_path).replace('_sol.json', '.json'))
        graph_path = os.path.join(data_dir, graph_name)
        if not os.path.exists(graph_path):
            alt_path = os.path.join("dataset/processed", graph_name)
            if os.path.exists(alt_path): graph_path = alt_path
            else: continue

        with open(graph_path, 'r') as f:
            graph_json = json.load(f)
            vertices = graph_json.get('data', {})

        edge_lookup = {}
        for src_id, node in vertices.items():
            for match in node['matches']:
                edge_lookup[(src_id, match['recipient'])] = match['ground_truth_label']

        graph_gt_score = 0
        graph_edges_count = 0
        for match in sol_data.get('matches', []):
            node_ids = match['node_ids']
            path_edges = []
            if match['type'] == 'cycle':
                for i in range(len(node_ids)):
                    path_edges.append((node_ids[i], node_ids[(i + 1) % len(node_ids)]))
            else:
                for i in range(len(node_ids) - 1):
                    path_edges.append((node_ids[i], node_ids[i + 1]))
            
            for edge in path_edges:
                if edge in edge_lookup:
                    graph_gt_score += edge_lookup[edge]
                    graph_edges_count += 1

        overall['total_gt_score'] += graph_gt_score
        overall['total_edges_selected'] += graph_edges_count
        overall['total_matches'] += sol_data.get('num_matches', 0)
        overall['graph_count'] += 1

    return {
        'dir': os.path.basename(sol_dir),
        'graphs': overall['graph_count'],
        'matches': overall['total_matches'],
        'transplants': overall['total_edges_selected'],
        'total_gt': overall['total_gt_score'],
        'avg_per_graph': overall['total_gt_score'] / overall['graph_count'] if overall['graph_count'] > 0 else 0,
        'avg_per_edge': overall['total_gt_score'] / overall['total_edges_selected'] if overall['total_edges_selected'] > 0 else 0
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-model Evaluation Comparison")
    parser.add_argument("--sol_dir", type=str, default="solutions", help="Base directory containing model subfolders")
    parser.add_argument("--data_dir", type=str, default="dataset/processed", help="Original data directory")
    parser.add_argument("--test_list", type=str, default=None, help="Path to text file containing filenames to evaluate")
    parser.add_argument("--full_eval", action="store_true", help="Use all_files.txt instead of test_files.txt (evaluate full dataset)")
    args = parser.parse_args()
    
    comp_results = []
    if os.path.isdir(args.sol_dir):
        # Check subdirectories
        subdirs = sorted([os.path.join(args.sol_dir, d) for d in os.listdir(args.sol_dir) if os.path.isdir(os.path.join(args.sol_dir, d))])
        
        # 确定统一的测试集：优先 --test_list，否则从第一个有 test_files.txt 的 subdir 取用
        unified_test_list = args.test_list
        if not unified_test_list and subdirs:
            list_name = "all_files.txt" if args.full_eval else "test_files.txt"
            for sd in subdirs:
                exp_id = os.path.basename(sd)
                cand = os.path.join("results", exp_id, list_name)
                if os.path.exists(cand):
                    unified_test_list = cand
                    print(f"📌 Using unified test set for all experiments: {unified_test_list}")
                    break
            if not unified_test_list:
                print(f"⚠️ No {list_name} found in any results/*/, each experiment may use its own filter")
        
        # If no subdirectories, try evaluating the dir itself
        if not subdirs:
            res = evaluate_directory(args.sol_dir, args.data_dir, unified_test_list, args.full_eval)
            if res: comp_results.append(res)
        else:
            for sd in subdirs:
                res = evaluate_directory(sd, args.data_dir, unified_test_list, args.full_eval)
                if res: comp_results.append(res)

    if not comp_results:
        print("❌ No valid solution directories found.")
    else:
        print("\n" + "="*115)
        print(f"{'Experiment / Model':<35} | {'Graphs':<6} | {'Cycles/Chains':<13} | {'Transplants':<11} | {'Total GT Score':<14} | {'Avg/Graph':<9} | {'Avg/Transplant':<14}")
        print("-" * 115)
        for r in comp_results:
            print(f"{r['dir']:<35} | {r['graphs']:<6} | {r['matches']:<13} | {r['transplants']:<11} | {r['total_gt']:<10.2f} | {r['avg_per_graph']:<9.2f} | {r['avg_per_edge']:<14.4f}")
        print("="*115 + "\n")
