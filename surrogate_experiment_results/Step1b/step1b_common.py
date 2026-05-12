"""Shared Step1b helpers built on Step1a graph and solver utilities."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STEP1A_PATH = PROJECT_ROOT / "surrogate_experiment_results" / "Step1a" / "Step1.py"

PROBE = {
    "feature_names": ["utility", "recipient_cPRA"],
    "feature_labels": [r"$\theta_1$ (utility weight)", r"$\theta_2$ (cPRA weight)"],
}


def load_step1a_module():
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    spec = importlib.util.spec_from_file_location("step1a_training", STEP1A_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_graph_records(paths, env):
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
        raise ValueError("--metric_stride must be positive")
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
