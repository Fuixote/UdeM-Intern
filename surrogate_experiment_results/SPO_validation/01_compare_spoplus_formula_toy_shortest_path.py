#!/usr/bin/env python3
"""Formula-level toy shortest-path SPO+ checks.

This script is intentionally small.  It checks the local cost-min SPO+ formula,
the reward-max sign adapter, the upper-bound property, and a stable
finite-difference gradient direction.  It also delegates to the PyEPO
comparison.  Missing PyEPO/Gurobi dependencies or Gurobi license/network
failures are hard failures.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compare_with_pyepo_spoplus import compare_with_pyepo  # noqa: E402
from spoplus_shortest_path import (  # noqa: E402
    grid_edges,
    solve_shortest_path,
    spo_plus_max_loss_and_grad,
    spo_plus_min_loss_and_grad,
)


def _check_perfect_prediction() -> None:
    grid_shape = (3, 3)
    c = np.array([1.0, 0.2, 1.4, 2.0, 0.1, 0.8, 1.3, 0.4, 1.1, 0.7, 0.6, 0.9])
    loss, grad = spo_plus_min_loss_and_grad(c, c, grid_shape)
    np.testing.assert_allclose(loss, 0.0, atol=1e-10)
    np.testing.assert_allclose(grad, np.zeros_like(c), atol=1e-10)


def _check_upper_bound() -> None:
    grid_shape = (4, 3)
    rng = np.random.RandomState(7)
    n_edges = len(grid_edges(grid_shape))
    for _ in range(20):
        c = rng.uniform(-1.5, 2.0, size=n_edges)
        c_hat = rng.uniform(-2.0, 2.0, size=n_edges)
        _, true_obj = solve_shortest_path(c, grid_shape)
        z_pred, _ = solve_shortest_path(c_hat, grid_shape)
        loss, _ = spo_plus_min_loss_and_grad(c_hat, c, grid_shape)
        decision_gap = float(np.dot(c, z_pred) - true_obj)
        if loss + 1e-9 < decision_gap:
            raise AssertionError(
                f"SPO+ loss {loss} did not upper-bound gap {decision_gap}"
            )


def _check_finite_difference_gradient() -> None:
    grid_shape = (3, 3)
    c = np.array([1.0, 4.0, 1.0, 4.0, 1.0, 4.0, 4.0, 1.0, 4.0, 1.0, 4.0, 1.0])
    c_hat = np.array(
        [1.0, 3.5, 1.0, 3.5, 1.0, 3.5, 3.5, 1.0, 3.5, 1.0, 3.5, 1.0]
    )
    direction = np.array(
        [0.3, -0.2, 0.1, 0.05, -0.1, 0.2, -0.15, 0.25, 0.12, -0.08, 0.18, -0.05]
    )
    _, grad = spo_plus_min_loss_and_grad(c_hat, c, grid_shape)
    eps = 1e-6
    loss_plus, _ = spo_plus_min_loss_and_grad(c_hat + eps * direction, c, grid_shape)
    loss_minus, _ = spo_plus_min_loss_and_grad(c_hat - eps * direction, c, grid_shape)
    finite_difference = (loss_plus - loss_minus) / (2.0 * eps)
    np.testing.assert_allclose(float(np.dot(grad, direction)), finite_difference, atol=1e-7)


def _check_reward_max_sign_conversion() -> None:
    grid_shape = (4, 3)
    rng = np.random.RandomState(11)
    n_edges = len(grid_edges(grid_shape))

    for _ in range(30):
        w = rng.uniform(-2.0, 2.0, size=n_edges)
        w_hat = rng.uniform(-2.0, 2.0, size=n_edges)

        max_loss, max_grad = spo_plus_max_loss_and_grad(w_hat, w, grid_shape)
        min_loss, min_grad = spo_plus_min_loss_and_grad(-w_hat, -w, grid_shape)

        np.testing.assert_allclose(max_loss, min_loss, atol=1e-10)
        np.testing.assert_allclose(max_grad, -min_grad, atol=1e-10)

    print("Reward-max / cost-min SPO+ sign conversion checks passed.")


def main() -> int:
    _check_perfect_prediction()
    _check_upper_bound()
    _check_finite_difference_gradient()
    _check_reward_max_sign_conversion()
    print("Local cost-min SPO+ toy formula checks passed.")
    return compare_with_pyepo()


if __name__ == "__main__":
    raise SystemExit(main())
