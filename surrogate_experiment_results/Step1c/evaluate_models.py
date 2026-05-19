"""Evaluate trained Step1c 2stage and e2e model weights on the test split."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import step1c_common as common
from split_dataset import read_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT_PATH = PROJECT_ROOT / "results" / "step1b_splits" / "master_split_seed=42.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "step1c_runs" / "train_size=50"


def load_model_weight(path):
    payload = np.load(path, allow_pickle=False)
    return {
        "path": str(path),
        "theta": payload["theta"].astype(float),
        "method": payload["method"].item(),
        "train_size": int(payload["train_size"].item()),
        "selection_metric": payload["selection_metric"].item(),
        "selection_value": float(payload["selection_value"].item()),
        "selected_epoch": int(payload["selected_epoch"].item()),
    }


def summarize_model_evaluations(
    method,
    train_size,
    theta,
    selection_metric,
    selection_value,
    evaluations,
    selected_epoch=None,
    model_path=None,
    paired_stats=None,
):
    gaps = np.asarray([row["gap"] for row in evaluations], dtype=float)
    normalized = np.asarray([row["normalized_gap"] for row in evaluations], dtype=float)
    ratios = np.asarray([row["ratio"] for row in evaluations], dtype=float)
    theta = np.asarray(theta, dtype=float)
    row = {
        "method": method,
        "train_size": int(train_size),
        "selected_epoch": "" if selected_epoch is None else int(selected_epoch),
        "theta_1": float(theta[0]),
        "theta_2": float(theta[1]),
        "selection_metric": selection_metric,
        "selection_value": float(selection_value),
        "test_mean_decision_gap": float(np.mean(gaps)),
        "test_mean_normalized_gap": float(np.mean(normalized)),
        "test_median_normalized_gap": float(np.median(normalized)),
        "test_mean_achieved_oracle_ratio": float(np.nanmean(ratios)),
        "model_path": "" if model_path is None else str(model_path),
    }
    if paired_stats is not None:
        row.update(paired_stats)
    return row


def paired_improvement_stats(candidate_evaluations, baseline_evaluations, n_bootstrap=1000, seed=42):
    baseline_by_graph = {row["graph"]: float(row["gap"]) for row in baseline_evaluations}
    improvements = np.asarray(
        [
            baseline_by_graph[row["graph"]] - float(row["gap"])
            for row in candidate_evaluations
            if row["graph"] in baseline_by_graph
        ],
        dtype=float,
    )
    if len(improvements) == 0:
        return {
            "paired_mean_improvement_over_2stage": np.nan,
            "paired_median_improvement_over_2stage": np.nan,
            "fraction_improved_over_2stage": np.nan,
            "paired_mean_improvement_ci_low": np.nan,
            "paired_mean_improvement_ci_high": np.nan,
        }

    rng = np.random.RandomState(seed)
    boot_means = np.asarray(
        [
            float(np.mean(rng.choice(improvements, size=len(improvements), replace=True)))
            for _ in range(n_bootstrap)
        ],
        dtype=float,
    )
    return {
        "paired_mean_improvement_over_2stage": float(np.mean(improvements)),
        "paired_median_improvement_over_2stage": float(np.median(improvements)),
        "fraction_improved_over_2stage": float(np.mean(improvements > 0)),
        "paired_mean_improvement_ci_low": float(np.percentile(boot_means, 2.5)),
        "paired_mean_improvement_ci_high": float(np.percentile(boot_means, 97.5)),
    }


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prefixed_evaluations(model, evaluations):
    return [
        {
            "method": model["method"],
            "selection_metric": model["selection_metric"],
            **row,
        }
        for row in evaluations
    ]


def summarize_evaluated_models(models_and_evaluations, bootstrap_samples=1000, bootstrap_seed=42):
    baseline_evaluations = None
    for model, evaluations in models_and_evaluations:
        if model["method"] == "2stage":
            baseline_evaluations = evaluations
            break

    summary_rows = []
    for model, evaluations in models_and_evaluations:
        paired_stats = None
        if model["method"] != "2stage" and baseline_evaluations is not None:
            paired_stats = paired_improvement_stats(
                evaluations,
                baseline_evaluations,
                n_bootstrap=bootstrap_samples,
                seed=bootstrap_seed,
            )
        summary_rows.append(
            summarize_model_evaluations(
                method=model["method"],
                train_size=model["train_size"],
                theta=model["theta"],
                selection_metric=model["selection_metric"],
                selection_value=model["selection_value"],
                evaluations=evaluations,
                selected_epoch=model["selected_epoch"],
                model_path=model["path"],
                paired_stats=paired_stats,
            )
        )
    return summary_rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate Step1c model weights.")
    parser.add_argument("--split_path", default=str(DEFAULT_SPLIT_PATH))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--weights",
        nargs="+",
        required=True,
        help="One or more .npz model weight files to evaluate.",
    )
    parser.add_argument("--gurobi_seed", type=int, default=42)
    parser.add_argument("--bootstrap_samples", type=int, default=1000)
    parser.add_argument("--bootstrap_seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv=None):
    import gurobipy as gp

    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    metrics_dir = out_dir / "metrics"
    split = read_json(args.split_path)

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Seed", args.gurobi_seed)
    env.start()

    test_graphs = []
    try:
        print(f"Loading test split: n={len(split['test'])}")
        test_graphs = common.load_graph_records([entry["path"] for entry in split["test"]], env)

        models_and_evaluations = []
        for weight_path in args.weights:
            model = load_model_weight(weight_path)
            evaluations = common.evaluate_theta(model["theta"], test_graphs, env)
            models_and_evaluations.append((model, evaluations))

        summary_rows = summarize_evaluated_models(
            models_and_evaluations,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        per_graph_rows = []
        for model, evaluations in models_and_evaluations:
            per_graph_rows.extend(prefixed_evaluations(model, evaluations))

        summary_fields = [
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
        write_csv(metrics_dir / "test_summary.csv", summary_rows, summary_fields)
        write_csv(metrics_dir / "test_per_graph.csv", per_graph_rows)
        with (metrics_dir / "test_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary_rows, handle, indent=2)

        print(f"Saved test metrics under {metrics_dir}")
        for row in summary_rows:
            print(
                f"  {row['method']} train_size={row['train_size']} "
                f"gap={row['test_mean_decision_gap']:.6f} "
                f"norm_gap={row['test_mean_normalized_gap']:.6f}"
            )
    finally:
        common.dispose_graph_records(test_graphs)
        env.dispose()


if __name__ == "__main__":
    main()
