"""Level 1.5 validation for the current Step1c KEP SPO+ code path.

This module deliberately tests the exact public Step1c function
``step1c_common.spo_plus_loss_and_grad`` while keeping the oracle small and
deterministic.  The fake graph stores a finite list of feasible KEP-like
selection vectors; ``EnumerationRewardOracle.solve_once`` implements the same
reward-max interface as Step1a's ``solve_once``.

The goal is not to validate Gurobi here.  The goal is to validate that Step1c
uses the reward-max SPO+ formula and propagates the gradient through
``w_hat = X @ theta`` correctly.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
STEP1C_COMMON_PATH = (
    REPO_ROOT / "surrogate_experiment_results" / "Step1c" / "step1c_common.py"
)

DEFAULT_THETA = np.array([1.1, -0.4], dtype=float)
DEFAULT_DIRECTION = np.array([0.3, -0.2], dtype=float)


class EnumerationRewardOracle:
    """Deterministic reward-max oracle with the Step1a ``solve_once`` API."""

    def __init__(self):
        self.seen_weights = []

    def solve_once(self, weights, graph, env):
        weights = np.asarray(weights, dtype=float)
        self.seen_weights.append(weights.copy())
        solutions = np.asarray(graph["solutions"], dtype=float)
        scores = solutions @ weights
        # np.argmax returns the first maximum, giving deterministic tie-breaking.
        return solutions[int(np.argmax(scores))].copy()


def make_toy_kep_record() -> Dict[str, np.ndarray]:
    """Return a tiny reward-max record shaped like Step1c graph records."""

    solutions = np.asarray(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    graph = {"solutions": solutions}
    x = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, -1.0],
        ],
        dtype=float,
    )
    w_true = np.asarray([3.0, 2.0, 4.5, 0.5], dtype=float)
    y_optimal = EnumerationRewardOracle().solve_once(w_true, graph, env=None)
    return {
        "filename": "toy-kep-spoplus-record",
        "X": x,
        "w_true": w_true,
        "y_optimal": y_optimal,
        "graph": graph,
    }


def predicted_reward_weights(record, theta) -> np.ndarray:
    x = np.asarray(record["X"], dtype=float)
    theta = np.asarray(theta, dtype=float)
    return x @ theta


def shifted_reward_weights(record, theta) -> np.ndarray:
    w_hat = predicted_reward_weights(record, theta)
    w_true = np.asarray(record["w_true"], dtype=float)
    return 2.0 * w_hat - w_true


def reference_spoplus_reward_max_loss_and_grad(
    record,
    theta,
    oracle: EnumerationRewardOracle,
) -> Tuple[float, np.ndarray]:
    """Independent reward-max SPO+ reference for the Step1c theta parameter.

    Formula checked:

    L(w_hat, w) = max_y (2 w_hat - w)^T y
                  - 2 w_hat^T y*(w)
                  + w^T y*(w)

    dL/dtheta = 2 X^T (y*(2 w_hat - w) - y*(w)).
    """

    x = np.asarray(record["X"], dtype=float)
    w_true = np.asarray(record["w_true"], dtype=float)
    y_optimal = np.asarray(record["y_optimal"], dtype=float)
    w_hat = predicted_reward_weights(record, theta)
    shifted_w = 2.0 * w_hat - w_true
    y_adv = np.asarray(oracle.solve_once(shifted_w, record["graph"], env=None), dtype=float)

    loss = (
        float(np.dot(shifted_w, y_adv))
        - 2.0 * float(np.dot(w_hat, y_optimal))
        + float(np.dot(w_true, y_optimal))
    )
    grad = 2.0 * (x.T @ (y_adv - y_optimal))
    return float(loss), np.asarray(grad, dtype=float)


def finite_difference_directional_derivative(
    loss_fn: Callable[[np.ndarray], float],
    theta,
    direction,
    epsilon: float = 1e-6,
) -> float:
    theta = np.asarray(theta, dtype=float)
    direction = np.asarray(direction, dtype=float)
    loss_plus = float(loss_fn(theta + epsilon * direction))
    loss_minus = float(loss_fn(theta - epsilon * direction))
    return (loss_plus - loss_minus) / (2.0 * epsilon)


def load_step1c_common():
    spec = importlib.util.spec_from_file_location(
        "step1c_common_level15_validation", STEP1C_COMMON_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_step1c_code_path() -> Dict[str, object]:
    """Run the Level 1.5 Step1c code-path checks and return diagnostics."""

    common = load_step1c_common()
    record = make_toy_kep_record()
    theta = DEFAULT_THETA.copy()

    reference_oracle = EnumerationRewardOracle()
    expected_loss, expected_grad = reference_spoplus_reward_max_loss_and_grad(
        record, theta, reference_oracle
    )

    step1c_oracle = EnumerationRewardOracle()
    common.load_step1a_module = lambda: step1c_oracle
    actual_loss, actual_grad = common.spo_plus_loss_and_grad(
        record, theta=theta, env=None
    )

    if not np.allclose(actual_loss, expected_loss, atol=1e-12):
        raise AssertionError(f"Step1c loss {actual_loss} != reference {expected_loss}")
    if not np.allclose(actual_grad, expected_grad, atol=1e-12):
        raise AssertionError(f"Step1c grad {actual_grad} != reference {expected_grad}")
    if not np.allclose(step1c_oracle.seen_weights[0], shifted_reward_weights(record, theta)):
        raise AssertionError("Step1c did not call the oracle with 2 * w_hat - w_true")

    common.load_step1a_module = EnumerationRewardOracle
    _, grad = common.spo_plus_loss_and_grad(record, theta=theta, env=None)
    finite_diff = finite_difference_directional_derivative(
        lambda value: common.spo_plus_loss_and_grad(record, theta=value, env=None)[0],
        theta,
        DEFAULT_DIRECTION,
    )
    analytic_directional = float(np.dot(grad, DEFAULT_DIRECTION))
    if not np.allclose(analytic_directional, finite_diff, atol=1e-7):
        raise AssertionError(
            f"directional derivative {analytic_directional} != finite diff {finite_diff}"
        )

    return {
        "theta": theta.tolist(),
        "shifted_w": shifted_reward_weights(record, theta).tolist(),
        "loss": float(actual_loss),
        "grad": np.asarray(actual_grad, dtype=float).tolist(),
        "direction": DEFAULT_DIRECTION.tolist(),
        "analytic_directional_derivative": analytic_directional,
        "finite_difference_directional_derivative": float(finite_diff),
    }
