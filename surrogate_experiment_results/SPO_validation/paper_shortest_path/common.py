"""Synthetic shortest-path benchmark utilities from the SPO paper.

The module keeps the benchmark dependency-light for local smoke tests.  The
default implementation uses NumPy plus the existing deterministic grid oracle
and the Step1c SPO+ algebra.  PyEPO support is optional and imported only when
the PyEPO method is requested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
SPO_DIR = SCRIPT_DIR.parent
STEP1C_DIR = SPO_DIR.parent / "Step1c"
for candidate in (SPO_DIR, STEP1C_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from spoplus_core import cost_min_prediction_gradient, cost_min_spoplus_loss  # noqa: E402
from spoplus_shortest_path import grid_edges, solve_shortest_path  # noqa: E402


GridShape = Tuple[int, int]
DEFAULT_GRID_SHAPE: GridShape = (5, 5)
DEFAULT_DEGREES = (1, 2, 4, 6, 8)
DEFAULT_NOISE_HALF_WIDTHS = (0.0, 0.5)
DEFAULT_LAMBDA_GRID = tuple(float(v) for v in np.logspace(-6, 0, 10))
METRIC_FIELDS = (
    "normalized_spo_loss",
    "avg_regret",
    "avg_relative_regret",
    "path_accuracy",
    "optimality_ratio",
)


@dataclass(frozen=True)
class PaperShortestPathConfig:
    grid_shape: GridShape = DEFAULT_GRID_SHAPE
    feature_dim: int = 5
    n_train: int = 1000
    n_val: int = 250
    n_test: int = 10000


@dataclass(frozen=True)
class PaperSplit:
    features: np.ndarray
    costs: np.ndarray
    opt_solutions: np.ndarray
    opt_objectives: np.ndarray


@dataclass(frozen=True)
class PaperTrialInstance:
    train: PaperSplit
    val: PaperSplit
    test: PaperSplit
    b_star: np.ndarray
    degree: int
    noise_half_width: float
    trial: int
    seed: int


@dataclass(frozen=True)
class ModelResult:
    method: str
    coefficients: np.ndarray
    selected_lambda: float
    val_metrics: Mapping[str, float]
    train_loss: float | None = None
    diagnostics: Mapping[str, object] = field(default_factory=dict)
    lambda_diagnostics: Tuple[Mapping[str, object], ...] = field(default_factory=tuple)
    reference_coefficients: np.ndarray | None = None


@dataclass(frozen=True)
class DependencyStatus:
    available: bool
    message: str
    details: Mapping[str, str]


def check_pyepo_dependencies() -> DependencyStatus:
    """Return import-level availability for the PyEPO reference path."""

    details: Dict[str, str] = {}
    available = True
    for name in ("torch", "pyepo", "gurobipy"):
        try:
            module = importlib.import_module(name)
        except Exception as exc:  # pragma: no cover - exact env dependent
            available = False
            details[name] = f"{type(exc).__name__}: {exc}"
        else:
            details[name] = getattr(module, "__version__", "importable")
    if available:
        return DependencyStatus(True, "PyEPO/Torch/Gurobi imports are available", details)
    return DependencyStatus(False, "PyEPO reference dependencies are not importable", details)


def paper_edge_count(grid_shape: Sequence[int] = DEFAULT_GRID_SHAPE) -> int:
    return len(grid_edges(grid_shape))


def _as_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def regime_seed(seed: int, trial: int, degree: int, noise_half_width: float) -> int:
    noise_code = int(round(float(noise_half_width) * 1000.0))
    return int(seed) + 1_000_003 * int(trial) + 10_007 * int(degree) + 101 * noise_code


def sample_b_star(
    rng: np.random.Generator,
    n_edges: int,
    feature_dim: int,
) -> np.ndarray:
    return rng.binomial(1, 0.5, size=(int(n_edges), int(feature_dim))).astype(float)


def make_split(
    rng: np.random.Generator,
    b_star: np.ndarray,
    n_samples: int,
    degree: int,
    noise_half_width: float,
    grid_shape: Sequence[int] = DEFAULT_GRID_SHAPE,
) -> PaperSplit:
    """Generate one split from the SPO paper synthetic shortest-path design."""

    n_samples = int(n_samples)
    feature_dim = b_star.shape[1]
    features = rng.normal(loc=0.0, scale=1.0, size=(n_samples, feature_dim))
    base = features @ b_star.T / np.sqrt(float(feature_dim)) + 3.0
    epsilon = rng.uniform(
        low=1.0 - float(noise_half_width),
        high=1.0 + float(noise_half_width),
        size=base.shape,
    )
    costs = (np.power(base, int(degree)) + 1.0) * epsilon
    opt_solutions = np.zeros_like(costs)
    opt_objectives = np.zeros(n_samples, dtype=float)
    for idx, cost in enumerate(costs):
        solution, objective = solve_shortest_path(cost, grid_shape)
        opt_solutions[idx] = solution
        opt_objectives[idx] = objective
    return PaperSplit(
        features=features.astype(float),
        costs=costs.astype(float),
        opt_solutions=opt_solutions.astype(float),
        opt_objectives=opt_objectives.astype(float),
    )


def make_trial_instance(
    degree: int,
    noise_half_width: float,
    trial: int,
    seed: int,
    config: PaperShortestPathConfig = PaperShortestPathConfig(),
) -> PaperTrialInstance:
    rng = _as_rng(regime_seed(seed, trial, degree, noise_half_width))
    n_edges = paper_edge_count(config.grid_shape)
    b_star = sample_b_star(rng, n_edges=n_edges, feature_dim=config.feature_dim)
    train = make_split(
        rng,
        b_star,
        config.n_train,
        degree,
        noise_half_width,
        grid_shape=config.grid_shape,
    )
    val = make_split(
        rng,
        b_star,
        config.n_val,
        degree,
        noise_half_width,
        grid_shape=config.grid_shape,
    )
    test = make_split(
        rng,
        b_star,
        config.n_test,
        degree,
        noise_half_width,
        grid_shape=config.grid_shape,
    )
    return PaperTrialInstance(
        train=train,
        val=val,
        test=test,
        b_star=b_star,
        degree=int(degree),
        noise_half_width=float(noise_half_width),
        trial=int(trial),
        seed=regime_seed(seed, trial, degree, noise_half_width),
    )


def augment_features(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=float)
    return np.hstack([values, np.ones((values.shape[0], 1), dtype=float)])


def predict_costs(features: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Predict edge costs from a linear model with an unregularized intercept."""

    features_aug = augment_features(features)
    coef = np.asarray(coefficients, dtype=float)
    if coef.ndim != 2:
        raise ValueError("coefficients must have shape (n_edges, feature_dim + 1)")
    if features_aug.shape[1] != coef.shape[1]:
        raise ValueError(
            f"feature dimension mismatch: features imply {features_aug.shape[1]}, "
            f"coefficients have {coef.shape[1]}"
        )
    return features_aug @ coef.T


def evaluate_predictions(
    pred_costs: np.ndarray,
    split: PaperSplit,
    grid_shape: Sequence[int] = DEFAULT_GRID_SHAPE,
) -> Dict[str, float]:
    """Evaluate predictions with the paper's ratio-of-sums normalized SPO loss."""

    pred = np.asarray(pred_costs, dtype=float)
    if pred.shape != split.costs.shape:
        raise ValueError(f"pred_costs shape {pred.shape} does not match {split.costs.shape}")

    total_regret = 0.0
    total_opt_cost = 0.0
    relative_regrets: List[float] = []
    accuracies: List[float] = []
    optimal_flags: List[float] = []
    for idx in range(pred.shape[0]):
        pred_solution, _ = solve_shortest_path(pred[idx], grid_shape)
        achieved_obj = float(np.dot(split.costs[idx], pred_solution))
        true_obj = float(split.opt_objectives[idx])
        regret = achieved_obj - true_obj
        if -1e-9 < regret < 0.0:
            regret = 0.0
        total_regret += regret
        total_opt_cost += true_obj
        relative_regrets.append(regret / (abs(true_obj) + 1e-12))
        accuracies.append(float(np.mean(np.abs(pred_solution - split.opt_solutions[idx]) < 0.5)))
        optimal_flags.append(float(abs(regret) < 1e-7))

    return {
        "normalized_spo_loss": float(total_regret / (total_opt_cost + 1e-12)),
        "avg_regret": float(total_regret / pred.shape[0]),
        "avg_relative_regret": float(np.mean(relative_regrets)),
        "path_accuracy": float(np.mean(accuracies)),
        "optimality_ratio": float(np.mean(optimal_flags)),
    }


def _soft_threshold(value: float, penalty: float) -> float:
    if value > penalty:
        return value - penalty
    if value < -penalty:
        return value + penalty
    return 0.0


def _fit_lasso_single_output(
    x_aug: np.ndarray,
    y: np.ndarray,
    lambda_value: float,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    n_samples, n_features = x_aug.shape
    weights = np.zeros(n_features, dtype=float)
    weights[-1] = float(np.mean(y))
    residual = y - x_aug @ weights
    column_norms = (x_aug * x_aug).sum(axis=0) / n_samples
    for _ in range(max_iter):
        max_delta = 0.0
        for feature_idx in range(n_features):
            column = x_aug[:, feature_idx]
            residual += column * weights[feature_idx]
            rho = float(np.dot(column, residual) / n_samples)
            denom = float(column_norms[feature_idx]) + 1e-12
            if feature_idx == n_features - 1:
                updated = rho / denom
            else:
                updated = _soft_threshold(rho, lambda_value) / denom
            residual -= column * updated
            max_delta = max(max_delta, abs(updated - weights[feature_idx]))
            weights[feature_idx] = updated
        if max_delta < tol:
            break
    return weights


def fit_least_squares(
    split: PaperSplit,
    lambda_value: float = 0.0,
    max_iter: int = 5000,
    tol: float = 1e-10,
) -> np.ndarray:
    """Fit multi-output LS/Lasso with an unregularized intercept column."""

    x_aug = augment_features(split.features)
    costs = np.asarray(split.costs, dtype=float)
    lambda_value = float(lambda_value)
    if lambda_value == 0.0:
        coef_aug, *_ = np.linalg.lstsq(x_aug, costs, rcond=None)
        return coef_aug.T.astype(float)

    coefficients = np.zeros((costs.shape[1], x_aug.shape[1]), dtype=float)
    for edge_idx in range(costs.shape[1]):
        coefficients[edge_idx] = _fit_lasso_single_output(
            x_aug,
            costs[:, edge_idx],
            lambda_value=lambda_value,
            max_iter=max_iter,
            tol=tol,
        )
    return coefficients


def select_least_squares_model(
    train: PaperSplit,
    val: PaperSplit,
    lambdas: Sequence[float] = DEFAULT_LAMBDA_GRID,
    grid_shape: Sequence[int] = DEFAULT_GRID_SHAPE,
) -> ModelResult:
    best: ModelResult | None = None
    for lambda_value in lambdas:
        coefficients = fit_least_squares(train, lambda_value=float(lambda_value))
        val_predictions = predict_costs(val.features, coefficients)
        val_metrics = evaluate_predictions(val_predictions, val, grid_shape=grid_shape)
        candidate = ModelResult(
            method="ls",
            coefficients=coefficients,
            selected_lambda=float(lambda_value),
            val_metrics=val_metrics,
        )
        if best is None or (
            candidate.val_metrics["normalized_spo_loss"]
            < best.val_metrics["normalized_spo_loss"]
        ):
            best = candidate
    if best is None:
        raise ValueError("at least one lambda must be provided")
    return best


def _spoplus_batch_gradient(
    coefficients: np.ndarray,
    split: PaperSplit,
    batch_indices: np.ndarray,
    lambda_value: float,
    grid_shape: Sequence[int],
) -> Tuple[np.ndarray, float]:
    grad = np.zeros_like(coefficients)
    losses: List[float] = []
    predictions = predict_costs(split.features[batch_indices], coefficients)
    for row, sample_idx in enumerate(batch_indices):
        c_hat = predictions[row]
        c_true = split.costs[sample_idx]
        z_true = split.opt_solutions[sample_idx]
        shifted_solution, _ = solve_shortest_path(2.0 * c_hat - c_true, grid_shape)
        loss = cost_min_spoplus_loss(c_hat, c_true, z_true, shifted_solution)
        grad_pred = cost_min_prediction_gradient(z_true, shifted_solution)
        grad[:, :-1] += np.outer(grad_pred, split.features[sample_idx])
        grad[:, -1] += grad_pred
        losses.append(float(loss))

    grad /= float(len(batch_indices))
    if lambda_value > 0.0:
        grad[:, :-1] += lambda_value * np.sign(coefficients[:, :-1])
    train_loss = float(np.mean(losses) + lambda_value * np.abs(coefficients[:, :-1]).sum())
    return grad, train_loss


def path_change_rate_from_coefficients(
    split: PaperSplit,
    reference_coefficients: np.ndarray,
    candidate_coefficients: np.ndarray,
    grid_shape: Sequence[int] = DEFAULT_GRID_SHAPE,
) -> float:
    """Return the fraction of instances whose chosen path changes."""

    reference_predictions = predict_costs(split.features, reference_coefficients)
    candidate_predictions = predict_costs(split.features, candidate_coefficients)
    changed = 0
    for reference_costs, candidate_costs in zip(reference_predictions, candidate_predictions):
        reference_solution, _ = solve_shortest_path(reference_costs, grid_shape)
        candidate_solution, _ = solve_shortest_path(candidate_costs, grid_shape)
        if not np.array_equal(reference_solution, candidate_solution):
            changed += 1
    return float(changed) / float(split.features.shape[0])


def train_spoplus_ours(
    train: PaperSplit,
    val: PaperSplit,
    lambda_value: float = 0.0,
    iterations: int = 500,
    batch_size: int = 32,
    learning_rate: float = 0.05,
    seed: int = 0,
    grid_shape: Sequence[int] = DEFAULT_GRID_SHAPE,
    eval_period: int = 50,
    spoplus_iterate: str = "raw",
    spoplus_init: str = "ls",
) -> ModelResult:
    """Train a linear model with the Step1c-core cost-min SPO+ formula."""

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if spoplus_iterate not in {"raw", "averaged"}:
        raise ValueError("spoplus_iterate must be 'raw' or 'averaged'")
    if spoplus_init not in {"ls", "zero"}:
        raise ValueError("spoplus_init must be 'ls' or 'zero'")
    rng = _as_rng(seed)
    ls_coefficients = fit_least_squares(train, lambda_value=0.0)
    if spoplus_init == "ls":
        coefficients = ls_coefficients.copy()
    else:
        coefficients = np.zeros_like(ls_coefficients)
    averaged_coefficients = coefficients.copy()
    averaged_count = 0
    best_coefficients = coefficients.copy()
    best_step = 0
    best_metrics = evaluate_predictions(
        predict_costs(val.features, best_coefficients),
        val,
        grid_shape=grid_shape,
    )
    initial_metrics = best_metrics
    final_metrics = best_metrics
    last_loss: float | None = None
    n_samples = train.features.shape[0]
    eval_period = max(1, int(eval_period))
    for step in range(1, int(iterations) + 1):
        size = min(int(batch_size), n_samples)
        batch_indices = rng.choice(n_samples, size=size, replace=n_samples < size)
        grad, last_loss = _spoplus_batch_gradient(
            coefficients,
            train,
            batch_indices,
            lambda_value=float(lambda_value),
            grid_shape=grid_shape,
        )
        step_size = float(learning_rate) / np.sqrt(float(step))
        coefficients -= step_size * grad
        averaged_count += 1
        averaged_coefficients += (coefficients - averaged_coefficients) / float(averaged_count)
        eval_coefficients = (
            averaged_coefficients if spoplus_iterate == "averaged" else coefficients
        )
        if step % eval_period == 0 or step == iterations:
            metrics = evaluate_predictions(
                predict_costs(val.features, eval_coefficients),
                val,
                grid_shape=grid_shape,
            )
            final_metrics = metrics
            if metrics["normalized_spo_loss"] < best_metrics["normalized_spo_loss"]:
                best_metrics = metrics
                best_coefficients = eval_coefficients.copy()
                best_step = step

    diagnostics = {
        "lambda_value": float(lambda_value),
        "spoplus_variant": spoplus_iterate,
        "spoplus_init": spoplus_init,
        "initial_val_norm_spo": float(initial_metrics["normalized_spo_loss"]),
        "best_val_norm_spo": float(best_metrics["normalized_spo_loss"]),
        "final_val_norm_spo": float(final_metrics["normalized_spo_loss"]),
        "best_step": int(best_step),
        "train_loss_last": "" if last_loss is None else float(last_loss),
        "coef_delta_norm_from_ls": float(np.linalg.norm(best_coefficients - ls_coefficients)),
        "val_path_change_rate_from_ls": path_change_rate_from_coefficients(
            val,
            ls_coefficients,
            best_coefficients,
            grid_shape=grid_shape,
        ),
    }

    return ModelResult(
        method="ours-spoplus",
        coefficients=best_coefficients,
        selected_lambda=float(lambda_value),
        val_metrics=best_metrics,
        train_loss=last_loss,
        diagnostics=diagnostics,
        reference_coefficients=ls_coefficients,
    )


def select_spoplus_ours_model(
    train: PaperSplit,
    val: PaperSplit,
    lambdas: Sequence[float] = DEFAULT_LAMBDA_GRID,
    iterations: int = 500,
    batch_size: int = 32,
    learning_rate: float = 0.05,
    seed: int = 0,
    grid_shape: Sequence[int] = DEFAULT_GRID_SHAPE,
    eval_period: int = 50,
    spoplus_iterate: str = "raw",
    spoplus_init: str = "ls",
) -> ModelResult:
    best: ModelResult | None = None
    best_idx = -1
    candidates: List[ModelResult] = []
    for lambda_idx, lambda_value in enumerate(lambdas):
        candidate = train_spoplus_ours(
            train,
            val,
            lambda_value=float(lambda_value),
            iterations=iterations,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=int(seed) + 7919 * lambda_idx,
            grid_shape=grid_shape,
            eval_period=eval_period,
            spoplus_iterate=spoplus_iterate,
            spoplus_init=spoplus_init,
        )
        candidates.append(candidate)
        if best is None or (
            candidate.val_metrics["normalized_spo_loss"]
            < best.val_metrics["normalized_spo_loss"]
        ):
            best = candidate
            best_idx = lambda_idx
    if best is None:
        raise ValueError("at least one lambda must be provided")
    lambda_diagnostics = []
    for lambda_idx, candidate in enumerate(candidates):
        row = dict(candidate.diagnostics)
        row["selected"] = lambda_idx == best_idx
        lambda_diagnostics.append(row)
    return ModelResult(
        method=best.method,
        coefficients=best.coefficients,
        selected_lambda=best.selected_lambda,
        val_metrics=best.val_metrics,
        train_loss=best.train_loss,
        diagnostics=best.diagnostics,
        lambda_diagnostics=tuple(lambda_diagnostics),
        reference_coefficients=best.reference_coefficients,
    )


def _build_pyepo_grid_optmodel(grid_shape: Sequence[int] = DEFAULT_GRID_SHAPE):
    import gurobipy as gp
    from gurobipy import GRB
    from pyepo.model.grb.grbmodel import optGrbModel

    class PaperShortestPathModel(optGrbModel):
        def __init__(self, shape: Sequence[int]):
            self.grid_shape = (int(shape[0]), int(shape[1]))
            self.edges = grid_edges(self.grid_shape)
            self.nodes = [
                (row, col)
                for row in range(self.grid_shape[0])
                for col in range(self.grid_shape[1])
            ]
            super().__init__()

        def _getModel(self):
            model = gp.Model("paper_shortest_path")
            x = model.addVars(range(len(self.edges)), lb=0.0, ub=1.0, name="x")
            model.modelSense = GRB.MINIMIZE
            for node in self.nodes:
                outgoing = gp.quicksum(
                    x[idx] for idx, edge in enumerate(self.edges) if edge[0] == node
                )
                incoming = gp.quicksum(
                    x[idx] for idx, edge in enumerate(self.edges) if edge[1] == node
                )
                if node == (0, 0):
                    model.addConstr(outgoing - incoming == 1.0)
                elif node == (self.grid_shape[0] - 1, self.grid_shape[1] - 1):
                    model.addConstr(incoming - outgoing == 1.0)
                else:
                    model.addConstr(outgoing - incoming == 0.0)
            return model, x

        def setObj(self, c):
            costs = np.asarray(c, dtype=float).reshape(-1)
            if costs.shape[0] != len(self.edges):
                raise ValueError(f"expected {len(self.edges)} costs, got {costs.shape[0]}")
            objective = gp.quicksum(costs[idx] * self.x[idx] for idx in range(len(self.edges)))
            self._model.setObjective(objective)

        def solve(self):
            self._model.update()
            self._model.optimize()
            solution = np.zeros(len(self.edges), dtype=float)
            for idx in range(len(self.edges)):
                solution[idx] = float(self.x[idx].x > 0.5)
            return solution, float(self._model.objVal)

    return PaperShortestPathModel(grid_shape)


def train_spoplus_pyepo(
    train: PaperSplit,
    val: PaperSplit,
    lambdas: Sequence[float] = DEFAULT_LAMBDA_GRID,
    iterations: int = 500,
    batch_size: int = 32,
    learning_rate: float = 0.05,
    seed: int = 0,
    grid_shape: Sequence[int] = DEFAULT_GRID_SHAPE,
) -> ModelResult:
    """Optional PyEPO SPO+ trainer for Orange/Gurobi environments."""

    import torch
    from torch import nn
    import pyepo

    torch.manual_seed(int(seed))
    rng = _as_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optmodel = _build_pyepo_grid_optmodel(grid_shape)
    loss_module = pyepo.func.SPOPlus(optmodel, processes=1, reduction="none")

    x_train = torch.as_tensor(train.features, dtype=torch.float32, device=device)
    c_train = torch.as_tensor(train.costs, dtype=torch.float32, device=device)
    sol_train = torch.as_tensor(train.opt_solutions, dtype=torch.float32, device=device)
    obj_train = torch.as_tensor(
        train.opt_objectives.reshape(-1, 1),
        dtype=torch.float32,
        device=device,
    )

    best_result: ModelResult | None = None
    for lambda_idx, lambda_value in enumerate(lambdas):
        torch.manual_seed(int(seed) + 7919 * lambda_idx)
        model = nn.Linear(train.features.shape[1], train.costs.shape[1], bias=True).to(device)
        init = fit_least_squares(train, lambda_value=0.0)
        with torch.no_grad():
            model.weight.copy_(torch.as_tensor(init[:, :-1], dtype=torch.float32, device=device))
            model.bias.copy_(torch.as_tensor(init[:, -1], dtype=torch.float32, device=device))
        optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
        n_samples = train.features.shape[0]
        for _ in range(int(iterations)):
            size = min(int(batch_size), n_samples)
            batch_indices = rng.choice(n_samples, size=size, replace=n_samples < size)
            index = torch.as_tensor(batch_indices, dtype=torch.long, device=device)
            pred = model(x_train[index])
            loss = loss_module(pred, c_train[index], sol_train[index], obj_train[index]).mean()
            if float(lambda_value) > 0.0:
                loss = loss + float(lambda_value) * model.weight.abs().sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            coefficients = np.hstack(
                [
                    model.weight.detach().cpu().numpy(),
                    model.bias.detach().cpu().numpy().reshape(-1, 1),
                ]
            )
        metrics = evaluate_predictions(
            predict_costs(val.features, coefficients),
            val,
            grid_shape=grid_shape,
        )
        candidate = ModelResult(
            method="pyepo-spoplus",
            coefficients=coefficients,
            selected_lambda=float(lambda_value),
            val_metrics=metrics,
        )
        if best_result is None or (
            candidate.val_metrics["normalized_spo_loss"]
            < best_result.val_metrics["normalized_spo_loss"]
        ):
            best_result = candidate

    if best_result is None:
        raise ValueError("at least one lambda must be provided")
    return best_result


def parse_float_list(values: Iterable[str]) -> Tuple[float, ...]:
    parsed: List[float] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                parsed.append(float(part))
    if not parsed:
        raise ValueError("expected at least one float value")
    return tuple(parsed)


def parse_int_list(values: Iterable[str]) -> Tuple[int, ...]:
    return tuple(int(round(value)) for value in parse_float_list(values))
