"""
Step 1b sample-size generalization experiment.

This script turns the Step1a in-sample diagnostic into a small held-out
experiment:

1. Create disjoint train/validation/test graph splits from the same synthetic
   reward benchmark.
2. Fit an MSE reward-fitting baseline on the training split.
3. Run decision-focused FY training on the training split.
4. Select FY checkpoints on validation decision gap and, optionally, validation
   FY objective.
5. Evaluate selected checkpoints on unseen test graphs.

The default settings are intentionally small so the pipeline can be smoke-tested
before scaling train/validation/test sizes.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STEP1A_PATH = PROJECT_ROOT / "surrogate_experiment_results" / "Step1a" / "Step1.py"
DEFAULT_DATA_DIR = PROJECT_ROOT / "dataset" / "processed" / "step1_noisy_linear_sigma010"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "step1b_runs" / "smoke"

PROBE = {
    "feature_names": ["utility", "recipient_cPRA"],
    "feature_labels": [r"$\theta_1$ (utility weight)", r"$\theta_2$ (cPRA weight)"],
}


def load_step1a_module():
    """Load Step1a helpers without requiring Step1a to be a Python package."""
    spec = importlib.util.spec_from_file_location("step1a_training", STEP1A_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def list_graph_files(data_dir):
    """Return G-*.json files in stable numeric order when possible."""
    paths = list(Path(data_dir).glob("G-*.json"))
    if not paths:
        raise FileNotFoundError(f"No G-*.json files found in {data_dir}")

    def graph_id(path):
        stem = path.stem
        try:
            return int(stem.split("-", 1)[1])
        except (IndexError, ValueError):
            return stem

    return sorted(paths, key=graph_id)


def make_split(files, train_size, val_size, test_size, seed):
    """Create deterministic disjoint train/validation/test filename splits."""
    total_needed = train_size + val_size + test_size
    if len(files) < total_needed:
        raise ValueError(
            f"Need {total_needed} graphs for split, found {len(files)}"
        )
    shuffled = [Path(path) for path in files]
    random.Random(seed).shuffle(shuffled)
    train_end = train_size
    val_end = train_end + val_size
    return {
        "train": [str(path) for path in shuffled[:train_end]],
        "validation": [str(path) for path in shuffled[train_end:val_end]],
        "test": [str(path) for path in shuffled[val_end:total_needed]],
    }


def load_graph_records(paths, env):
    """Load graphs and attach reusable KEP solvers."""
    step1a = load_step1a_module()
    records = []
    for idx, path in enumerate(paths, start=1):
        graph = step1a.load_graph(path)
        graph["cached_solver"] = step1a.CachedHybridKepModel(graph, env)
        y_optimal = step1a.solve_once(graph["w_true"], graph, env)
        records.append(
            {
                "path": str(path),
                "filename": Path(path).name,
                "X": step1a.feature_matrix(graph, PROBE),
                "w_true": graph["w_true"],
                "y_optimal": y_optimal,
                "graph": graph,
            }
        )
        print(f"  loaded {idx}/{len(paths)} {Path(path).name}", flush=True)
    return records


def dispose_graph_records(records):
    for record in records:
        solver = record.get("graph", {}).get("cached_solver")
        if solver is not None:
            solver.dispose()


def compute_ols(graphs):
    X = np.vstack([record["X"] for record in graphs])
    w = np.concatenate([record["w_true"] for record in graphs])
    theta, *_ = np.linalg.lstsq(X, w, rcond=None)
    return theta.astype(float)


class Adam:
    def __init__(self, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.m = np.zeros(2)
        self.v = np.zeros(2)
        self.t = 0

    def step(self, theta, grad):
        self.t += 1
        self.m = self.b1 * self.m + (1.0 - self.b1) * grad
        self.v = self.b2 * self.v + (1.0 - self.b2) * (grad ** 2)
        m_hat = self.m / (1.0 - self.b1 ** self.t)
        v_hat = self.v / (1.0 - self.b2 ** self.t)
        return theta - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def grad_fy(graphs, theta, eps_abs, M, rng, env):
    step1a = load_step1a_module()
    grad = np.zeros(2)
    for record in graphs:
        w_hat = (record["X"] @ theta).astype(np.float32)
        perturbations = step1a.make_antithetic_perturbations(
            rng, M, len(w_hat), eps_abs
        )
        y_bar = np.mean(
            [
                step1a.solve_once(w_hat + perturbation, record["graph"], env)
                for perturbation in perturbations
            ],
            axis=0,
        )
        grad += record["X"].T @ (y_bar - record["y_optimal"])
    return grad / len(graphs)


def run_fy_trajectory(graphs, theta_init, n_epochs, lr, eps_abs, M, seed, env):
    opt = Adam(lr)
    rng = np.random.RandomState(seed)
    theta = np.asarray(theta_init, dtype=float).copy()
    trajectory = [theta.copy()]
    for epoch in range(n_epochs):
        theta = opt.step(theta, grad_fy(graphs, theta, eps_abs, M, rng, env))
        trajectory.append(theta.copy())
        print(f"  [FY] epoch {epoch + 1:>4} theta={np.round(theta, 4)}")
    return np.asarray(trajectory, dtype=float)


def evaluate_theta(theta, graphs, env, normalizer_epsilon=1e-9):
    """Evaluate one theta on a graph split using synthetic-label decision gap."""
    step1a = load_step1a_module()
    theta = np.asarray(theta, dtype=float)
    rows = []
    for record in graphs:
        w_hat = record["X"] @ theta
        y_pred = step1a.solve_once(w_hat, record["graph"], env)
        optimal_obj = float(np.dot(record["w_true"], record["y_optimal"]))
        achieved_obj = float(np.dot(record["w_true"], y_pred))
        gap = optimal_obj - achieved_obj
        denominator = abs(optimal_obj) + normalizer_epsilon
        ratio = achieved_obj / optimal_obj if abs(optimal_obj) > normalizer_epsilon else np.nan
        rows.append(
            {
                "graph": record["filename"],
                "optimal_obj": optimal_obj,
                "achieved_obj": achieved_obj,
                "gap": gap,
                "normalized_gap": gap / denominator,
                "ratio": ratio,
            }
        )
    return rows


def mean_gap(theta, graphs, env):
    evaluations = evaluate_theta(theta, graphs, env)
    return float(np.mean([row["gap"] for row in evaluations]))


def make_perturbations_by_graph(graphs, eps_abs, M, seed):
    step1a = load_step1a_module()
    rng = np.random.RandomState(seed)
    return [
        step1a.make_antithetic_perturbations(rng, M, record["w_true"].shape[0], eps_abs)
        for record in graphs
    ]


def average_fy_objective(theta, graphs, perturbations_by_graph, env):
    """Approximate the perturbed FY objective, up to theta-independent terms."""
    step1a = load_step1a_module()
    theta = np.asarray(theta, dtype=float)
    total = 0.0
    for record, perturbations in zip(graphs, perturbations_by_graph):
        w_hat = (record["X"] @ theta).astype(np.float32)
        target_score = float(np.dot(w_hat, record["y_optimal"]))
        loss_sum = 0.0
        for perturbation in perturbations:
            perturbed_weights = w_hat + perturbation
            y_perturbed = step1a.solve_once(perturbed_weights, record["graph"], env)
            loss_sum += float(np.dot(perturbed_weights, y_perturbed)) - target_score
        total += loss_sum / len(perturbations)
    return float(total / len(graphs))


def trajectory_epoch_indices(length, stride):
    if stride <= 0:
        raise ValueError("--checkpoint_stride must be positive")
    indices = list(range(0, length, stride))
    if indices[-1] != length - 1:
        indices.append(length - 1)
    return np.asarray(indices, dtype=int)


def evaluate_trajectory_decision_gap(trajectory, graphs, env, indices=None):
    if indices is None:
        indices = np.arange(len(trajectory), dtype=int)
    return np.asarray([mean_gap(trajectory[idx], graphs, env) for idx in indices])


def evaluate_trajectory_fy_objective(
    trajectory, graphs, perturbations_by_graph, env, indices=None
):
    if indices is None:
        indices = np.arange(len(trajectory), dtype=int)
    return np.asarray(
        [
            average_fy_objective(trajectory[idx], graphs, perturbations_by_graph, env)
            for idx in indices
        ]
    )


def select_checkpoint(trajectory, metrics, epochs=None):
    metrics = np.asarray(metrics, dtype=float)
    if epochs is None:
        epochs = np.arange(len(metrics), dtype=int)
    epochs = np.asarray(epochs, dtype=int)
    if len(trajectory) != len(metrics) or len(metrics) != len(epochs):
        raise ValueError("trajectory, metrics, and epochs must have the same length")
    best_idx = int(np.nanargmin(metrics))
    return {
        "epoch": int(epochs[best_idx]),
        "theta": np.asarray(trajectory[best_idx], dtype=float).copy(),
        "metric": float(metrics[best_idx]),
    }


def paired_gap_improvement(evaluations, baseline_evaluations):
    if baseline_evaluations is None:
        return np.nan
    baseline_by_graph = {row["graph"]: row["gap"] for row in baseline_evaluations}
    improvements = [
        baseline_by_graph[row["graph"]] - row["gap"]
        for row in evaluations
        if row["graph"] in baseline_by_graph
    ]
    return float(np.mean(improvements)) if improvements else np.nan


def summarize_test_metrics(
    method, checkpoint_rule, checkpoint, evaluations, baseline_evaluations=None
):
    gaps = np.asarray([row["gap"] for row in evaluations], dtype=float)
    normalized = np.asarray([row["normalized_gap"] for row in evaluations], dtype=float)
    ratios = np.asarray([row["ratio"] for row in evaluations], dtype=float)
    theta = np.asarray(checkpoint["theta"], dtype=float)
    return {
        "method": method,
        "checkpoint_rule": checkpoint_rule,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "theta_1": float(theta[0]),
        "theta_2": float(theta[1]),
        "validation_selection_metric": float(checkpoint["metric"]),
        "test_mean_decision_gap": float(np.mean(gaps)),
        "test_mean_normalized_gap": float(np.mean(normalized)),
        "test_median_normalized_gap": float(np.median(normalized)),
        "test_mean_achieved_oracle_ratio": float(np.nanmean(ratios)),
        "paired_gap_improvement_over_mse": paired_gap_improvement(
            evaluations, baseline_evaluations
        ),
    }


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"Cannot JSON serialize {type(value).__name__}")


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=json_default)


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


def prefixed_evaluations(method, checkpoint_rule, evaluations):
    rows = []
    for row in evaluations:
        rows.append(
            {
                "method": method,
                "checkpoint_rule": checkpoint_rule,
                **row,
            }
        )
    return rows


def validation_rows(method, epochs, trajectory_subset, decision_gap, fy_objective=None):
    rows = []
    for idx, epoch in enumerate(epochs):
        row = {
            "method": method,
            "epoch": int(epoch),
            "theta_1": float(trajectory_subset[idx, 0]),
            "theta_2": float(trajectory_subset[idx, 1]),
            "validation_decision_gap": float(decision_gap[idx]),
            "validation_fy_objective": "",
        }
        if fy_objective is not None:
            row["validation_fy_objective"] = float(fy_objective[idx])
        rows.append(row)
    return rows


def run_experiment(args):
    import gurobipy as gp

    out_dir = Path(args.out_dir)
    trajectories_dir = out_dir / "trajectories"
    metrics_dir = out_dir / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    graph_files = list_graph_files(args.data_dir)
    split = make_split(
        graph_files,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    write_json(out_dir / "split.json", split)
    write_json(out_dir / "config.json", vars(args))

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Seed", args.seed)
    env.start()

    train_graphs = []
    validation_graphs = []
    test_graphs = []
    try:
        print("Loading train graphs")
        train_graphs = load_graph_records(split["train"], env)
        print("Loading validation graphs")
        validation_graphs = load_graph_records(split["validation"], env)
        print("Loading test graphs")
        test_graphs = load_graph_records(split["test"], env)

        theta_mse = compute_ols(train_graphs)
        np.save(trajectories_dir / "trajectory_mse.npy", theta_mse[None, :])
        print(f"MSE/OLS theta: {np.round(theta_mse, 4)}")

        mse_evaluations = evaluate_theta(theta_mse, test_graphs, env)
        mse_checkpoint = {"epoch": 0, "theta": theta_mse, "metric": mean_gap(theta_mse, validation_graphs, env)}
        summary_rows = [
            summarize_test_metrics(
                method="mse",
                checkpoint_rule="ols_train_solution",
                checkpoint=mse_checkpoint,
                evaluations=mse_evaluations,
                baseline_evaluations=None,
            )
        ]
        per_graph_rows = prefixed_evaluations(
            "mse", "ols_train_solution", mse_evaluations
        )
        validation_metric_rows = [
            {
                "method": "mse",
                "epoch": 0,
                "theta_1": float(theta_mse[0]),
                "theta_2": float(theta_mse[1]),
                "validation_decision_gap": float(mse_checkpoint["metric"]),
                "validation_fy_objective": "",
            }
        ]

        rng0 = np.random.RandomState(args.seed)
        theta_random = (
            np.asarray(args.theta_init, dtype=float)
            if args.theta_init is not None
            else rng0.uniform(0.5, 3.5, size=2)
        )
        method_initializers = {
            "fy_random": theta_random,
            "fy_warm": theta_mse,
        }

        for method in args.methods:
            if method == "mse":
                continue
            theta_init = method_initializers[method]
            print(f"{method} initial theta: {np.round(theta_init, 4)}")
            trajectory = run_fy_trajectory(
                train_graphs,
                theta_init=theta_init,
                n_epochs=args.n_epochs,
                lr=args.lr_fy,
                eps_abs=args.fy_epsilon,
                M=args.fy_M,
                seed=args.seed,
                env=env,
            )
            trajectory_path = trajectories_dir / f"trajectory_{method}.npy"
            np.save(trajectory_path, trajectory)

            eval_indices = trajectory_epoch_indices(len(trajectory), args.checkpoint_stride)
            trajectory_subset = trajectory[eval_indices]
            val_gaps = evaluate_trajectory_decision_gap(
                trajectory, validation_graphs, env, indices=eval_indices
            )

            val_fy = None
            if "validation_fy_objective" in args.checkpoint_rules:
                perturbations = make_perturbations_by_graph(
                    validation_graphs, args.fy_epsilon, args.fy_M, args.seed
                )
                val_fy = evaluate_trajectory_fy_objective(
                    trajectory,
                    validation_graphs,
                    perturbations,
                    env,
                    indices=eval_indices,
                )

            validation_metric_rows.extend(
                validation_rows(method, eval_indices, trajectory_subset, val_gaps, val_fy)
            )

            checkpoint_inputs = {
                "validation_decision_gap": val_gaps,
            }
            if val_fy is not None:
                checkpoint_inputs["validation_fy_objective"] = val_fy

            for checkpoint_rule in args.checkpoint_rules:
                checkpoint = select_checkpoint(
                    trajectory_subset,
                    checkpoint_inputs[checkpoint_rule],
                    epochs=eval_indices,
                )
                evaluations = evaluate_theta(checkpoint["theta"], test_graphs, env)
                summary_rows.append(
                    summarize_test_metrics(
                        method=method,
                        checkpoint_rule=checkpoint_rule,
                        checkpoint=checkpoint,
                        evaluations=evaluations,
                        baseline_evaluations=mse_evaluations,
                    )
                )
                per_graph_rows.extend(
                    prefixed_evaluations(method, checkpoint_rule, evaluations)
                )

        summary_fields = [
            "method",
            "checkpoint_rule",
            "checkpoint_epoch",
            "theta_1",
            "theta_2",
            "validation_selection_metric",
            "test_mean_decision_gap",
            "test_mean_normalized_gap",
            "test_median_normalized_gap",
            "test_mean_achieved_oracle_ratio",
            "paired_gap_improvement_over_mse",
        ]
        write_csv(metrics_dir / "test_summary.csv", summary_rows, summary_fields)
        write_csv(metrics_dir / "test_per_graph.csv", per_graph_rows)
        validation_fields = [
            "method",
            "epoch",
            "theta_1",
            "theta_2",
            "validation_decision_gap",
            "validation_fy_objective",
        ]
        write_csv(
            metrics_dir / "validation_trajectory_metrics.csv",
            validation_metric_rows,
            validation_fields,
        )

        print(f"Saved split/config/results under {out_dir}")
        print("Test summary:")
        for row in summary_rows:
            print(
                f"  {row['method']} [{row['checkpoint_rule']}] "
                f"gap={row['test_mean_decision_gap']:.6f} "
                f"norm_gap={row['test_mean_normalized_gap']:.6f} "
                f"paired_impr={row['paired_gap_improvement_over_mse']:.6f}"
            )
    finally:
        dispose_graph_records(train_graphs)
        dispose_graph_records(validation_graphs)
        dispose_graph_records(test_graphs)
        env.dispose()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Step1b held-out generalization smoke experiment"
    )
    parser.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--train_size", type=int, default=2)
    parser.add_argument("--val_size", type=int, default=2)
    parser.add_argument("--test_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--lr_fy", type=float, default=0.1)
    parser.add_argument("--fy_epsilon", type=float, default=1.0)
    parser.add_argument("--fy_M", type=int, default=2)
    parser.add_argument("--checkpoint_stride", type=int, default=1)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["mse", "fy_random", "fy_warm"],
        default=["mse", "fy_warm"],
    )
    parser.add_argument(
        "--checkpoint_rules",
        nargs="+",
        choices=["validation_decision_gap", "validation_fy_objective"],
        default=["validation_decision_gap"],
    )
    parser.add_argument("--theta_init", type=float, nargs=2, default=None)
    args = parser.parse_args(argv)

    if args.train_size <= 0 or args.val_size <= 0 or args.test_size <= 0:
        raise ValueError("train_size, val_size, and test_size must be positive")
    if args.n_epochs < 0:
        raise ValueError("--n_epochs must be non-negative")
    if args.fy_M <= 0:
        raise ValueError("--fy_M must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    run_experiment(args)


if __name__ == "__main__":
    main()
