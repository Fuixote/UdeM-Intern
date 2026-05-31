"""Utilities for Step1c-vs-PyEPO shortest-path validation.

This module is intentionally an adapter/test harness. It imports Step1c's
shared SPO+ algebra as read-only code and does not modify Step1c experiment
files.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
SPO_VALIDATION_DIR = REPO_ROOT / "surrogate_experiment_results" / "SPO_validation"
PYEPO_DIR = SPO_VALIDATION_DIR / "PyEPO"
STEP1C_CORE_PATH = (
    REPO_ROOT / "surrogate_experiment_results" / "Step1c" / "spoplus_core.py"
)
DEFAULT_DATA_ROOT = SPO_VALIDATION_DIR / "data"
DEFAULT_RESULT_ROOT = SPO_VALIDATION_DIR / "res_step1c_vs_pyepo"
DEFAULT_GRID = (5, 5)
DEFAULT_TEST_SIZE = 1000
DEFAULT_TRAIN_SIZES = (100, 1000, 5000)
DEFAULT_DEGREES = (1, 2, 4, 6)
DEFAULT_NOISES = (0.0, 0.5)
RESULT_COLUMNS = ["True SPO", "Unamb SPO", "MSE", "Elapsed", "Epochs"]


@dataclass(frozen=True)
class FixedSplit:
    x_train: np.ndarray
    c_train: np.ndarray
    x_test: np.ndarray
    c_test: np.ndarray
    metadata: dict


@dataclass(frozen=True)
class LinearParams:
    weight: np.ndarray
    bias: np.ndarray


@dataclass(frozen=True)
class SPSetting:
    train_size: int
    deg: int
    noise: float
    seed: int


def ensure_local_pyepo_on_path() -> None:
    """Prefer the local PyEPO submodule package over any site-package copy."""

    pkg_path = str(PYEPO_DIR / "pkg")
    if pkg_path not in sys.path:
        sys.path.insert(0, pkg_path)


def load_step1c_spoplus_core():
    """Load Step1c's shared SPO+ algebra without importing Step1c runners."""

    spec = importlib.util.spec_from_file_location("step1c_spoplus_core", STEP1C_CORE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixed_dataset_path(
    data_root=DEFAULT_DATA_ROOT,
    grid=DEFAULT_GRID,
    train_size=100,
    test_size=DEFAULT_TEST_SIZE,
    feat=5,
    deg=1,
    noise=0.0,
    seed=0,
) -> Path:
    return (
        Path(data_root)
        / "sp"
        / "h{}w{}".format(*tuple(grid))
        / "train{}-test{}-p{}-d{}-e{}-seed{}.npz".format(
            train_size,
            test_size,
            feat,
            deg,
            noise,
            seed,
        )
    )


def result_csv_path(
    result_root=DEFAULT_RESULT_ROOT,
    method_slug="my-2stage-lr",
    grid=DEFAULT_GRID,
    train_size=100,
    feat=5,
    deg=1,
    noise=0.0,
    lan="gurobi",
) -> Path:
    path = Path(result_root) / "sp" / "h{}w{}".format(*tuple(grid)) / lan
    filename = "n{}p{}-d{}-e{}_{}.csv".format(
        train_size,
        feat,
        deg,
        noise,
        method_slug,
    )
    return path / filename


def iter_sp_settings(
    expnum=10,
    train_sizes=DEFAULT_TRAIN_SIZES,
    degs=DEFAULT_DEGREES,
    noises=DEFAULT_NOISES,
):
    for train_size in train_sizes:
        for noise in noises:
            for deg in degs:
                for seed in range(expnum):
                    yield SPSetting(
                        train_size=int(train_size),
                        deg=int(deg),
                        noise=float(noise),
                        seed=int(seed),
                    )


def load_result_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.exists():
        df = pd.read_csv(path)
        missing = [col for col in RESULT_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(
                "Result CSV {} is missing required columns: {}".format(path, missing)
            )
        return df[RESULT_COLUMNS]
    return pd.DataFrame(columns=RESULT_COLUMNS)


def completed_seed_count(path: Path) -> int:
    return len(load_result_table(path))


def append_result_row(path: Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = load_result_table(path)
    normalized = {col: row[col] for col in RESULT_COLUMNS}
    row_df = pd.DataFrame([normalized], columns=RESULT_COLUMNS)
    if df.empty:
        df = row_df
    else:
        df = pd.concat([df, row_df], ignore_index=True)
    df.to_csv(path, index=False)


def load_fixed_split(path: Path) -> FixedSplit:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        metadata_raw = data["metadata"].item()
        import json

        return FixedSplit(
            x_train=np.asarray(data["x_train"], dtype=np.float64),
            c_train=np.asarray(data["c_train"], dtype=np.float64),
            x_test=np.asarray(data["x_test"], dtype=np.float64),
            c_test=np.asarray(data["c_test"], dtype=np.float64),
            metadata=json.loads(metadata_raw),
        )


def solve_costs(optmodel, costs):
    """Solve one PyEPO optimization model for each cost vector."""

    sols, objs = [], []
    for cost in np.asarray(costs):
        optmodel.setObj(cost)
        sol, obj = optmodel.solve()
        sols.append(np.asarray(sol, dtype=np.float64))
        objs.append(float(obj))
    return np.asarray(sols, dtype=np.float64), np.asarray(objs, dtype=np.float64)


def linear_predict(x, weight, bias):
    """Return C_hat = X @ W + b for W shaped as (features, costs)."""

    return np.asarray(x) @ np.asarray(weight) + np.asarray(bias)


def deterministic_linear_params(feat_dim, cost_dim, seed=0, scale=0.1) -> LinearParams:
    rng = np.random.default_rng(seed)
    return LinearParams(
        weight=rng.normal(loc=0.0, scale=scale, size=(feat_dim, cost_dim)),
        bias=rng.normal(loc=0.0, scale=scale, size=(cost_dim,)),
    )


def multioutput_linear_params(model) -> LinearParams:
    """Extract C_hat = X @ W + b from sklearn MultiOutputRegressor."""

    coefs = [estimator.coef_ for estimator in model.estimators_]
    intercepts = [estimator.intercept_ for estimator in model.estimators_]
    return LinearParams(
        weight=np.asarray(coefs, dtype=np.float64).T,
        bias=np.asarray(intercepts, dtype=np.float64),
    )


def mean_linear_parameter_gradients(x, grad_pred):
    """Return gradients for mean loss wrt W and b in C_hat = X @ W + b.

    `grad_pred` is the per-instance dL_i/dC_hat_i before the batch mean.
    """

    x = np.asarray(x, dtype=np.float64)
    grad_pred = np.asarray(grad_pred, dtype=np.float64)
    batch_size = x.shape[0]
    return x.T @ grad_pred / batch_size, grad_pred.mean(axis=0)


def step1c_cost_min_loss(pred_cost, true_cost, true_sol, adversarial_sol):
    core = load_step1c_spoplus_core()
    return np.asarray(
        core.cost_min_spoplus_loss(pred_cost, true_cost, true_sol, adversarial_sol),
        dtype=np.float64,
    )


def step1c_cost_min_grad_pred(y_optimal, y_adversarial, mean=False):
    core = load_step1c_spoplus_core()
    grad = np.asarray(
        core.cost_min_prediction_gradient(y_optimal, y_adversarial),
        dtype=np.float64,
    )
    if mean:
        grad = grad / grad.shape[0]
    return grad


def sgd_step(weight, bias, grad_weight, grad_bias, lr):
    return (
        np.asarray(weight, dtype=np.float64) - float(lr) * np.asarray(grad_weight),
        np.asarray(bias, dtype=np.float64) - float(lr) * np.asarray(grad_bias),
    )


def mse(pred_cost, true_cost):
    return float(np.mean((np.asarray(pred_cost) - np.asarray(true_cost)) ** 2))


def normalized_regret(optmodel, pred_costs, true_costs, true_objs):
    regret_sum = 0.0
    for pred_cost, true_cost, true_obj in zip(pred_costs, true_costs, true_objs):
        optmodel.setObj(pred_cost)
        pred_sol, _ = optmodel.solve()
        achieved_obj = float(np.dot(np.asarray(pred_sol), true_cost))
        regret_sum += achieved_obj - float(true_obj)
    return regret_sum / abs(float(np.sum(true_objs)) + 1e-3)


def prediction_metrics_2stage(optmodel, pred_costs, true_costs, true_objs):
    """Evaluate array predictions with the same normalization as PyEPO 2-stage."""

    ensure_local_pyepo_on_path()
    import pyepo

    true_spo = 0.0
    unamb_spo = 0.0
    for pred_cost, true_cost, true_obj in zip(pred_costs, true_costs, true_objs):
        true_spo += pyepo.metric.calRegret(
            optmodel,
            pred_cost,
            true_cost,
            float(np.asarray(true_obj).reshape(-1)[0]),
        )
        unamb_spo += pyepo.metric.calUnambRegret(
            optmodel,
            pred_cost,
            true_cost,
            float(np.asarray(true_obj).reshape(-1)[0]),
        )
    denom = abs(float(np.asarray(true_objs).sum()) + 1e-3)
    return {
        "True SPO": true_spo / denom,
        "Unamb SPO": unamb_spo / denom,
        "MSE": mse(pred_costs, true_costs),
    }


def max_abs_diff(left, right):
    return float(np.max(np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64))))
