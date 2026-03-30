import argparse
import glob
import json
import os
import re

from experiment_config import PROCESSED_DATA_DIR, RESULTS_ROOT, SOLUTIONS_ROOT, resolve_path


BATCH_NAME_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}(?:__.+)?$")


def read_name_list(list_path):
    with open(list_path, "r", encoding="utf-8") as handle:
        names = [line.strip() for line in handle if line.strip()]
    return {name.replace(".json", "") for name in names}


def resolve_graph_data_dir(directory):
    direct_files = sorted(glob.glob(os.path.join(directory, "G-*.json")))
    if direct_files:
        return directory, direct_files, "direct"

    if not os.path.isdir(directory):
        return directory, [], "missing"

    candidate_batches = []
    for entry in os.listdir(directory):
        batch_path = os.path.join(directory, entry)
        if not os.path.isdir(batch_path) or not BATCH_NAME_PATTERN.match(entry):
            continue
        batch_files = sorted(glob.glob(os.path.join(batch_path, "G-*.json")))
        if batch_files:
            candidate_batches.append((entry, batch_path, batch_files))

    if not candidate_batches:
        return directory, [], "empty"

    candidate_batches.sort(key=lambda item: item[0], reverse=True)
    batch_name, batch_path, batch_files = candidate_batches[0]
    return batch_path, batch_files, f"latest_batch:{batch_name}"


def parse_dataset_path_from_summary(results_dir):
    summary_path = os.path.join(results_dir, "summary.txt")
    if not os.path.exists(summary_path):
        return None

    with open(summary_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("Dataset Path:"):
                return line.split(":", 1)[1].strip()
    return None


def infer_data_dir_for_experiment(sol_dir, default_data_dir, results_root):
    exp_id = os.path.basename(sol_dir)
    result_dir = os.path.join(results_root, exp_id)
    candidates = []

    dataset_path = parse_dataset_path_from_summary(result_dir)
    if dataset_path:
        candidates.append(("summary", str(resolve_path(dataset_path))))

    candidates.append(("argument", default_data_dir))

    tried = []
    for source, candidate in candidates:
        resolved_dir, graph_files, resolution_mode = resolve_graph_data_dir(candidate)
        tried.append(f"{source}={resolved_dir} ({resolution_mode})")
        if graph_files:
            return resolved_dir, graph_files, f"{source}:{resolution_mode}"

    return None, [], "; ".join(tried)


def auto_detect_filter_path(exp_id, results_root, use_all_files):
    list_name = "all_files.txt" if use_all_files else "test_files.txt"
    candidate = os.path.join(results_root, exp_id, list_name)
    if os.path.exists(candidate):
        return candidate
    return None


def build_edge_lookup(graph_path, cache):
    if graph_path in cache:
        return cache[graph_path]

    with open(graph_path, "r", encoding="utf-8") as handle:
        graph_json = json.load(handle)

    vertices = graph_json.get("data", {})
    edge_lookup = {}
    for src_id, node in vertices.items():
        for match in node.get("matches", []):
            edge_lookup[(src_id, match["recipient"])] = match.get("ground_truth_label", 0.0)

    cache[graph_path] = edge_lookup
    return edge_lookup


def evaluate_directory(
    sol_dir,
    default_data_dir,
    results_root,
    explicit_filter=None,
    explicit_filter_source=None,
    use_all_files=False,
    edge_lookup_cache=None,
):
    exp_id = os.path.basename(sol_dir)
    edge_lookup_cache = edge_lookup_cache if edge_lookup_cache is not None else {}

    resolved_data_dir, graph_files, data_source = infer_data_dir_for_experiment(
        sol_dir, default_data_dir, results_root
    )
    if not graph_files:
        return {
            "dir": exp_id,
            "error": f"Unable to locate graph data ({data_source})",
        }

    test_filter = explicit_filter
    filter_source = explicit_filter_source
    if test_filter is None:
        auto_filter_path = auto_detect_filter_path(exp_id, results_root, use_all_files)
        if auto_filter_path:
            test_filter = read_name_list(auto_filter_path)
            filter_source = auto_filter_path

    sol_files = sorted(glob.glob(os.path.join(sol_dir, "*_sol.json")))
    if test_filter is not None:
        sol_files = [
            sol_path
            for sol_path in sol_files
            if os.path.basename(sol_path).replace("_sol.json", "") in test_filter
        ]

    if not sol_files:
        return None

    overall = {
        "total_gt_score": 0.0,
        "total_edges_selected": 0,
        "total_matches": 0,
        "graph_count": 0,
        "skipped_missing_graph": 0,
        "missing_edges": 0,
        "attempted_edges": 0,
    }

    for sol_path in sol_files:
        with open(sol_path, "r", encoding="utf-8") as handle:
            sol_data = json.load(handle)

        graph_name = sol_data.get("graph", os.path.basename(sol_path).replace("_sol.json", ".json"))
        graph_path = os.path.join(resolved_data_dir, graph_name)
        if not os.path.exists(graph_path):
            overall["skipped_missing_graph"] += 1
            continue

        edge_lookup = build_edge_lookup(graph_path, edge_lookup_cache)

        graph_gt_score = 0.0
        graph_edges_count = 0
        for match in sol_data.get("matches", []):
            node_ids = match.get("node_ids", [])
            path_edges = []
            if match.get("type") == "cycle":
                for idx in range(len(node_ids)):
                    path_edges.append((node_ids[idx], node_ids[(idx + 1) % len(node_ids)]))
            else:
                for idx in range(len(node_ids) - 1):
                    path_edges.append((node_ids[idx], node_ids[idx + 1]))

            for edge in path_edges:
                overall["attempted_edges"] += 1
                if edge in edge_lookup:
                    graph_gt_score += edge_lookup[edge]
                    graph_edges_count += 1
                else:
                    overall["missing_edges"] += 1

        overall["total_gt_score"] += graph_gt_score
        overall["total_edges_selected"] += graph_edges_count
        overall["total_matches"] += sol_data.get("num_matches", 0)
        overall["graph_count"] += 1

    result = {
        "dir": exp_id,
        "graphs": overall["graph_count"],
        "matches": overall["total_matches"],
        "transplants": overall["total_edges_selected"],
        "total_gt": overall["total_gt_score"],
        "avg_per_graph": (
            overall["total_gt_score"] / overall["graph_count"] if overall["graph_count"] > 0 else 0.0
        ),
        "avg_per_edge": (
            overall["total_gt_score"] / overall["total_edges_selected"]
            if overall["total_edges_selected"] > 0
            else 0.0
        ),
        "data_dir": resolved_data_dir,
        "data_source": data_source,
        "filter_source": filter_source or ("all solution files" if use_all_files else "no filter found"),
        "solution_files": len(sol_files),
        "skipped_missing_graph": overall["skipped_missing_graph"],
        "missing_edges": overall["missing_edges"],
        "attempted_edges": overall["attempted_edges"],
    }

    if overall["missing_edges"] > 0:
        missing_ratio = overall["missing_edges"] / overall["attempted_edges"] if overall["attempted_edges"] else 0.0
        result["error"] = (
            "Solution paths do not match the resolved graph data "
            f"(missing_edges={overall['missing_edges']}/{overall['attempted_edges']}, "
            f"ratio={missing_ratio:.2%}). This usually means the experiment was run on a different "
            "processed batch, or the solution directory contains stale files from another run."
        )

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-model evaluation comparison")
    parser.add_argument(
        "--sol_dir",
        type=str,
        default=str(SOLUTIONS_ROOT),
        help="Base directory containing experiment solution subfolders",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(PROCESSED_DATA_DIR),
        help="Fallback processed-data directory if an experiment cannot infer its own dataset path",
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default=str(RESULTS_ROOT),
        help="Results directory used for auto-discovering dataset paths and test_files.txt",
    )
    parser.add_argument(
        "--test_list",
        type=str,
        default=None,
        help="Path to text file containing filenames to evaluate for every experiment",
    )
    parser.add_argument(
        "--full_eval",
        action="store_true",
        help="Use all_files.txt when available; otherwise evaluate every solution file",
    )
    args = parser.parse_args()

    args.sol_dir = str(resolve_path(args.sol_dir))
    args.data_dir = str(resolve_path(args.data_dir))
    args.results_root = str(resolve_path(args.results_root))

    explicit_filter = None
    explicit_filter_source = None
    if args.test_list:
        explicit_filter_source = str(resolve_path(args.test_list))
        explicit_filter = read_name_list(explicit_filter_source)
        print(f"📌 Using explicit evaluation list: {explicit_filter_source} ({len(explicit_filter)} graphs)")

    comp_results = []
    edge_lookup_cache = {}

    if os.path.isdir(args.sol_dir):
        subdirs = sorted(
            [
                os.path.join(args.sol_dir, entry)
                for entry in os.listdir(args.sol_dir)
                if os.path.isdir(os.path.join(args.sol_dir, entry))
            ]
        )

        if not subdirs:
            result = evaluate_directory(
                args.sol_dir,
                args.data_dir,
                args.results_root,
                explicit_filter=explicit_filter,
                explicit_filter_source=explicit_filter_source,
                use_all_files=args.full_eval,
                edge_lookup_cache=edge_lookup_cache,
            )
            if result:
                comp_results.append(result)
        else:
            unified_filter = explicit_filter
            unified_filter_source = explicit_filter_source
            if unified_filter is None and subdirs and not args.full_eval:
                preferred_candidates = []
                fallback_candidates = []
                for subdir in subdirs:
                    exp_id = os.path.basename(subdir)
                    candidate = auto_detect_filter_path(exp_id, args.results_root, use_all_files=False)
                    if not candidate:
                        continue
                    record = (os.path.getmtime(candidate), candidate)
                    if exp_id == "ground_truth":
                        fallback_candidates.append(record)
                    else:
                        preferred_candidates.append(record)

                ranked_candidates = sorted(preferred_candidates, reverse=True) + sorted(
                    fallback_candidates, reverse=True
                )
                if ranked_candidates:
                    _, candidate = ranked_candidates[0]
                    unified_filter = read_name_list(candidate)
                    unified_filter_source = candidate
                    print(
                        f"📌 Using unified test set for all experiments: "
                        f"{candidate} ({len(unified_filter)} graphs)"
                    )

            for subdir in subdirs:
                result = evaluate_directory(
                    subdir,
                    args.data_dir,
                    args.results_root,
                    explicit_filter=unified_filter,
                    explicit_filter_source=unified_filter_source,
                    use_all_files=args.full_eval,
                    edge_lookup_cache=edge_lookup_cache,
                )
                if result:
                    comp_results.append(result)

    if not comp_results:
        print("❌ No valid solution directories found.")
    else:
        valid_results = [result for result in comp_results if "error" not in result]
        failed_results = [result for result in comp_results if "error" in result]

        if valid_results:
            print("\n" + "=" * 158)
            print(
                f"{'Experiment / Model':<35} | {'Graphs':<6} | {'Cycles/Chains':<13} | "
                f"{'Transplants':<11} | {'Total GT Score':<14} | {'Avg/Graph':<9} | "
                f"{'Avg/Transplant':<14} | {'Skipped':<7}"
            )
            print("-" * 158)
            for result in valid_results:
                print(
                    f"{result['dir']:<35} | "
                    f"{result['graphs']:<6} | "
                    f"{result['matches']:<13} | "
                    f"{result['transplants']:<11} | "
                    f"{result['total_gt']:<14.2f} | "
                    f"{result['avg_per_graph']:<9.2f} | "
                    f"{result['avg_per_edge']:<14.4f} | "
                    f"{result['skipped_missing_graph']:<7}"
                )
            print("=" * 158)

            print("\nDetails:")
            for result in valid_results:
                print(
                    f"- {result['dir']}: data={result['data_dir']} "
                    f"[{result['data_source']}], filter={result['filter_source']}, "
                    f"solution_files={result['solution_files']}, missing_edges={result['missing_edges']}"
                )

        if failed_results:
            print("\nWarnings:")
            for result in failed_results:
                print(f"- {result['dir']}: {result['error']}")
