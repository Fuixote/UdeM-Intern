"""Shared SPO+ algebra for Step1c-style validation code.

The functions here contain only the linear-objective SPO+ formulas.  They do
not know about KEP, shortest path, Gurobi, PyEPO, or model parameters; callers
provide the true optimal solution and the adversarial shifted-objective
solution from their own oracle.
"""

from __future__ import annotations

import numpy as np


def _is_torch_tensor(value) -> bool:
    return value.__class__.__module__.split(".", maxsplit=1)[0] == "torch"


def _sum_product(left, right):
    if _is_torch_tensor(left) or _is_torch_tensor(right):
        product = left * right
        if product.ndim <= 1:
            return product.sum()
        return product.reshape(product.shape[0], -1).sum(dim=1)

    product = np.asarray(left, dtype=float) * np.asarray(right, dtype=float)
    if product.ndim <= 1:
        return float(np.sum(product))
    return np.sum(product.reshape(product.shape[0], -1), axis=1)


def reward_max_spoplus_loss(
    pred_reward,
    true_reward,
    optimal_solution,
    adversarial_solution,
):
    """Return maximization-form SPO+ loss.

    Formula:

        max_y (2 * pred_reward - true_reward)^T y
        - 2 * pred_reward^T y*(true_reward)
        + true_reward^T y*(true_reward)
    """

    shifted_reward = 2.0 * pred_reward - true_reward
    return (
        _sum_product(shifted_reward, adversarial_solution)
        - 2.0 * _sum_product(pred_reward, optimal_solution)
        + _sum_product(true_reward, optimal_solution)
    )


def reward_max_prediction_gradient(optimal_solution, adversarial_solution):
    """Return d loss / d pred_reward for maximization-form SPO+."""

    return 2.0 * (adversarial_solution - optimal_solution)


def cost_min_spoplus_loss(
    pred_cost,
    true_cost,
    optimal_solution,
    adversarial_solution,
):
    """Return minimization-form SPO+ loss via the reward-max sign adapter."""

    return reward_max_spoplus_loss(
        -pred_cost,
        -true_cost,
        optimal_solution,
        adversarial_solution,
    )


def cost_min_prediction_gradient(optimal_solution, adversarial_solution):
    """Return d loss / d pred_cost for minimization-form SPO+."""

    return -reward_max_prediction_gradient(optimal_solution, adversarial_solution)
