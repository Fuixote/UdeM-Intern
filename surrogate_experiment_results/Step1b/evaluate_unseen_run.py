"""Evaluate one Step1b run directory on an unseen processed dataset.

This script is intentionally not a sweep driver. It evaluates exactly one
training run directory, which should contain the three standard Step1b model
checkpoints under ``model_weights/``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import step1b_common as common
from evaluate_models import load_model_weight, summarize_evaluated_models, write_csv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "step1_noisy_linear_sigma010_unseen_test1000_seed20260513"
)

EXPECTED_WEIGHT_FILES = [
    "2stage_best_by_validation_mse_loss.npz",
    "e2e_best_by_validation_decision_gap.npz",
    "e2e_best_by_validation_fy_loss.npz",
]

SUMMARY_FIELDS = [
    "evaluation_dataset",
    "evaluation_graph_count",
    "method",
    "train_size",
    "selected_epoch",
    "theta_1",
    "theta_2",
    "selection_metric",
    "selection_value",
    "test_mean_decision_gap",
    "test_mean_normalized_gap",
    "test_median_normalized_gap",
    "test_mean_achieved_oracle_ratio",
    "paired_mean_improvement_over_2stage",
    "paired_median_improvement_over_2stage",
    "fraction_improved_over_2stage",
    "paired_mean_improvement_ci_low",
    "paired_mean_improvement_ci_high",
    "model_path",
]


def graph_sort_key(path: Path):
    match = re.fullmatch(r"G-(\d+)\.json", path.name)
    if match:
        return (0, int(match.group(1)))
    return (1, path.name)


def list_graph_paths(dataset_dir: Path, graph_limit: int | None = None):
    paths = sorted(dataset_dir.glob("G-*.json"), key=graph_sort_key)
    if graph_limit is not None:
        paths = paths[:graph_limit]
    return paths


def resolve_run_weights(run_dir: Path):
    weights_dir = run_dir / "model_weights"
    weights = [weights_dir / name for name in EXPECTED_WEIGHT_FILES]
    missing = [str(path) for path in weights if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing expected model weight files:\n" + "\n".join(missing)
        )
    return weights


def with_evaluation_context(summary_rows, dataset_dir: Path, graph_count: int):
    enriched = []
    for row in summary_rows:
        enriched.append(
            {
                "evaluation_dataset": str(dataset_dir),
                "evaluation_graph_count": graph_count,
                **row,
            }
        )
    return enriched


def per_graph_rows(models_and_evaluations, dataset_dir: Path):
    rows = []
    for model, evaluations in models_and_evaluations:
        for row in evaluations:
            rows.append(
                {
                    "evaluation_dataset": str(dataset_dir),
                    "method": model["method"],
                    "train_size": model["train_size"],
                    "selected_epoch": model["selected_epoch"],
                    "selection_metric": model["selection_metric"],
                    "selection_value": model["selection_value"],
                    "model_path": model["path"],
                    **row,
                }
            )
    return rows


def write_run_config(path: Path, args, graph_count: int, weights):
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": "surrogate_experiment_results/Step1b/evaluate_unseen_run.py",
        "run_dir": str(Path(args.run_dir)),
        "dataset_dir": str(Path(args.dataset_dir)),
        "graph_count": graph_count,
        "graph_limit": args.graph_limit,
        "weights": [str(path) for path in weights],
        "output_stem": args.output_stem,
        "gurobi_seed": args.gurobi_seed,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one Step1b run directory on an unseen processed dataset. "
            "The run directory must contain the three standard model_weights/*.npz files."
        )
    )
    parser.add_argument(
        "--run_dir",
        required=True,
        help="One Step1b run directory, e.g. .../formal_M16.../train_size=50",
    )
    parser.add_argument(
        "--dataset_dir",
        default=str(DEFAULT_DATASET_DIR),
        help="Processed unseen dataset directory containing G-*.json files.",
    )
    parser.add_argument(
        "--output_stem",
        default="unseen_test",
        help="Prefix for output files written under RUN_DIR/metrics/.",
    )
    parser.add_argument(
        "--graph_limit",
        type=int,
        default=None,
        help="Optional first-N graph limit for smoke testing.",
    )
    parser.add_argument("--gurobi_seed", type=int, default=42)
    parser.add_argument("--bootstrap_samples", type=int, default=1000)
    parser.add_argument("--bootstrap_seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    dataset_dir = Path(args.dataset_dir)
    metrics_dir = run_dir / "metrics"

    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    weights = resolve_run_weights(run_dir)
    graph_paths = list_graph_paths(dataset_dir, args.graph_limit)
    if not graph_paths:
        raise FileNotFoundError(f"No G-*.json files found under {dataset_dir}")

    import gurobipy as gp

    metrics_dir.mkdir(parents=True, exist_ok=True)
    write_run_config(
        metrics_dir / f"{args.output_stem}_run_config.json",
        args,
        len(graph_paths),
        weights,
    )

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Seed", args.gurobi_seed)
    env.start()

    test_graphs = []
    try:
        print(f"Run dir: {run_dir}", flush=True)
        print(f"Dataset: {dataset_dir}", flush=True)
        print(f"Loading unseen graphs: n={len(graph_paths)}", flush=True)
        test_graphs = common.load_graph_records(graph_paths, env)

        models_and_evaluations = []
        for weight_path in weights:
            model = load_model_weight(weight_path)
            print(
                "Evaluating "
                f"{model['method']} selected_by={model['selection_metric']} "
                f"epoch={model['selected_epoch']} theta={model['theta']}",
                flush=True,
            )
            evaluations = common.evaluate_theta(model["theta"], test_graphs, env)
            models_and_evaluations.append((model, evaluations))

        summary_rows = summarize_evaluated_models(
            models_and_evaluations,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        summary_rows = with_evaluation_context(
            summary_rows,
            dataset_dir=dataset_dir,
            graph_count=len(graph_paths),
        )
        graph_rows = per_graph_rows(models_and_evaluations, dataset_dir)

        summary_csv = metrics_dir / f"{args.output_stem}_summary.csv"
        summary_json = metrics_dir / f"{args.output_stem}_summary.json"
        per_graph_csv = metrics_dir / f"{args.output_stem}_per_graph.csv"

        write_csv(summary_csv, summary_rows, SUMMARY_FIELDS)
        write_csv(per_graph_csv, graph_rows)
        summary_json.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

        print(f"Saved summary: {summary_csv}", flush=True)
        print(f"Saved per-graph metrics: {per_graph_csv}", flush=True)
        for row in summary_rows:
            paired = row.get("paired_mean_improvement_over_2stage", "")
            print(
                f"  {row['method']} selected_by={row['selection_metric']} "
                f"gap={row['test_mean_decision_gap']:.6f} "
                f"norm_gap={row['test_mean_normalized_gap']:.6f} "
                f"paired_improvement={paired}",
                flush=True,
            )
    finally:
        common.dispose_graph_records(test_graphs)
        env.dispose()


if __name__ == "__main__":
    main()
