#!/usr/bin/env python3
"""Compare PyEPO SPOPlus forward loss against the local Step1c-core formula."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

import numpy as np

import common


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare PyEPO SPOPlus forward loss with the local SPO+ formula."
    )
    parser.add_argument("--allow-missing-pyepo", action="store_true")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260529)
    return parser


def local_forward_losses(
    pred_costs: np.ndarray,
    true_costs: np.ndarray,
    true_solutions: np.ndarray,
    grid_shape=common.DEFAULT_GRID_SHAPE,
) -> np.ndarray:
    losses = []
    for c_hat, c_true, z_true in zip(pred_costs, true_costs, true_solutions):
        shifted_solution, _ = common.solve_shortest_path(2.0 * c_hat - c_true, grid_shape)
        loss = common.cost_min_spoplus_loss(c_hat, c_true, z_true, shifted_solution)
        losses.append(float(loss))
    return np.asarray(losses, dtype=float)


def pyepo_forward_losses(
    pred_costs: np.ndarray,
    true_costs: np.ndarray,
    true_solutions: np.ndarray,
    true_objectives: np.ndarray,
    grid_shape=common.DEFAULT_GRID_SHAPE,
) -> np.ndarray:
    import torch
    import pyepo

    optmodel = common._build_pyepo_grid_optmodel(grid_shape)
    spoplus = pyepo.func.SPOPlus(optmodel, processes=1, reduction="none")
    pred_tensor = torch.as_tensor(pred_costs, dtype=torch.float32)
    true_tensor = torch.as_tensor(true_costs, dtype=torch.float32)
    sol_tensor = torch.as_tensor(true_solutions, dtype=torch.float32)
    obj_tensor = torch.as_tensor(true_objectives.reshape(-1, 1), dtype=torch.float32)
    losses = spoplus(pred_tensor, true_tensor, sol_tensor, obj_tensor)
    return losses.detach().cpu().numpy().reshape(-1).astype(float)


def make_fixed_batch(samples: int, seed: int):
    config = common.PaperShortestPathConfig(n_train=1, n_val=1, n_test=int(samples))
    instance = common.make_trial_instance(
        degree=2,
        noise_half_width=0.5,
        trial=0,
        seed=int(seed),
        config=config,
    )
    rng = np.random.default_rng(int(seed) + 17)
    pred_costs = instance.test.costs + rng.normal(
        loc=0.0,
        scale=0.5,
        size=instance.test.costs.shape,
    )
    return instance.test, pred_costs


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    status = common.check_pyepo_dependencies()
    if not status.available:
        payload = {
            "pyepo_available": False,
            "message": status.message,
            "details": dict(status.details),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        if args.allow_missing_pyepo:
            return 0
        raise SystemExit(
            "PyEPO/Torch/Gurobi are required for forward-loss comparison. "
            "Pass --allow-missing-pyepo only for dependency smoke checks."
        )

    split, pred_costs = make_fixed_batch(samples=args.samples, seed=args.seed)
    local_losses = local_forward_losses(
        pred_costs,
        split.costs,
        split.opt_solutions,
        grid_shape=common.DEFAULT_GRID_SHAPE,
    )
    pyepo_losses = pyepo_forward_losses(
        pred_costs,
        split.costs,
        split.opt_solutions,
        split.opt_objectives,
        grid_shape=common.DEFAULT_GRID_SHAPE,
    )
    diff = np.abs(pyepo_losses - local_losses)
    payload = {
        "pyepo_available": True,
        "number_of_samples": int(len(diff)),
        "max_abs_loss_diff": float(np.max(diff)),
        "mean_abs_loss_diff": float(np.mean(diff)),
        "local_losses": [float(value) for value in local_losses],
        "pyepo_losses": [float(value) for value in pyepo_losses],
        "convention": (
            "Both paths use reduction=none and the cost-min SPO+ convention "
            "L(c_hat,c)=max_z (c-2*c_hat)^T z + 2*c_hat^T z*(c) - c^T z*(c)."
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
