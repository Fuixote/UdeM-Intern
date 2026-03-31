import argparse
import json
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment_config import PROCESSED_DATA_DIR, RESULTS_ROOT, SOLUTIONS_ROOT, resolve_path, solution_dir_for_experiment
from formulations.common.backend_utils import infer_ndd_mask
from formulations.common.model_io import load_prediction_model, predict_edge_weights
from formulations.pief.backend import solve_pief
from model.graph_utils import parse_json_to_graph_info, resolve_graph_data_dir

FORMULATION_TAG = "pief"


def formulation_experiment_name(base_name):
    return f"{base_name}__{FORMULATION_TAG}"


def solve_kep(json_path, model, model_type, output_dir, max_cycle=3, max_chain=4):
    file_name = os.path.basename(json_path)
    graph_data = parse_json_to_graph_info(json_path)
    weights = predict_edge_weights(graph_data, model, model_type)
    is_ndd_mask = infer_ndd_mask(graph_data["x"])

    result = solve_pief(
        weights=weights,
        edge_index=graph_data["edge_index"],
        is_ndd_mask=is_ndd_mask,
        num_nodes=graph_data["num_nodes"],
        max_cycle=max_cycle,
        max_chain=max_chain,
        id_map_rev=graph_data["id_map_rev"],
    )

    if result["matches"]:
        payload = {
            "graph": file_name,
            "model_used": model_type,
            "formulation": FORMULATION_TAG,
            "total_predicted_w": float(result["objective"]),
            "num_matches": len(result["formatted_matches"]),
            "matches": result["formatted_matches"],
        }
        with open(os.path.join(output_dir, file_name.replace(".json", "_sol.json")), "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=4)
        print(f"✅ Saved PIEF solution for {file_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--max_cycle", type=int, default=3)
    parser.add_argument("--max_chain", type=int, default=4)
    parser.add_argument("--data_dir", type=str, default=str(PROCESSED_DATA_DIR))
    parser.add_argument("--results_root", type=str, default=str(RESULTS_ROOT))
    parser.add_argument("--solutions_root", type=str, default=str(SOLUTIONS_ROOT))
    parser.add_argument("--gt_mode", action="store_true")
    args = parser.parse_args()

    data_dir = str(resolve_path(args.data_dir))
    results_root = str(resolve_path(args.results_root))
    solutions_root = str(resolve_path(args.solutions_root))
    if args.model_path is not None:
        args.model_path = str(resolve_path(args.model_path))

    if args.gt_mode:
        model_type, model = "GroundTruth", None
        experiment_name = formulation_experiment_name("ground_truth")
        sol_out = str(solution_dir_for_experiment(experiment_name, solutions_root=solutions_root))
        print("💡 Running in ORACLE mode (using Ground Truth labels)")

        results_gt_dir = os.path.join(results_root, experiment_name)
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
        experiment_name = formulation_experiment_name(os.path.basename(model_dir))
        sol_out = str(solution_dir_for_experiment(experiment_name, solutions_root=solutions_root))
        print(f"🚀 Running in PREDICTION mode: {model_type} | formulation={FORMULATION_TAG}")

    os.makedirs(sol_out, exist_ok=True)
    print(f"📁 Solutions will be saved to: {sol_out}")

    _, files = resolve_graph_data_dir(data_dir, log_prefix="🔍 Solving")
    for graph_file in files:
        solve_kep(graph_file, model, model_type, sol_out, max_cycle=args.max_cycle, max_chain=args.max_chain)
