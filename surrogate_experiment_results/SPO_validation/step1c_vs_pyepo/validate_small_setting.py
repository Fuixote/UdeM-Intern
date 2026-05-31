#!/usr/bin/env python3
"""Run the smallest Step1c-vs-PyEPO shortest-path validation.

Default setting follows README.md:
grid=5x5, train_size=100, noise=0.0, degree=1, seed=0.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import validation_core as core


@dataclass(frozen=True)
class CheckResult:
    name: str
    value: float
    threshold: float

    @property
    def passed(self):
        return self.value <= self.threshold


def build_pyepo_model(grid):
    core.ensure_local_pyepo_on_path()
    import pyepo

    return pyepo.model.grb.shortestPathModel(tuple(grid))


def level0_oracle_alignment(model, split, count):
    core.ensure_local_pyepo_on_path()
    import pyepo

    x = split.x_train[:count]
    c = split.c_train[:count]
    dataset = pyepo.data.dataset.optDataset(model, x, c)
    adapter_sols, adapter_objs = core.solve_costs(model, c)
    return [
        CheckResult(
            "level0_solution_max_abs_diff",
            core.max_abs_diff(adapter_sols, dataset.sols),
            0.0,
        ),
        CheckResult(
            "level0_objective_max_abs_diff",
            core.max_abs_diff(adapter_objs, dataset.objs.reshape(-1)),
            1e-8,
        ),
    ]


def level1_lr_alignment(model, split):
    core.ensure_local_pyepo_on_path()
    import pyepo
    from sklearn.linear_model import LinearRegression

    pyepo_lr = pyepo.twostage.sklearnPred(LinearRegression())
    step1c_lr = pyepo.twostage.sklearnPred(LinearRegression())
    pyepo_lr.fit(split.x_train, split.c_train)
    step1c_lr.fit(split.x_train, split.c_train)

    pyepo_params = core.multioutput_linear_params(pyepo_lr)
    step1c_params = core.multioutput_linear_params(step1c_lr)
    pyepo_pred = pyepo_lr.predict(split.x_test)
    step1c_pred = core.linear_predict(split.x_test, step1c_params.weight, step1c_params.bias)

    true_sols, true_objs = core.solve_costs(model, split.c_test)
    _ = true_sols
    pyepo_regret = core.normalized_regret(model, pyepo_pred, split.c_test, true_objs)
    step1c_regret = core.normalized_regret(model, step1c_pred, split.c_test, true_objs)
    return [
        CheckResult(
            "level1_lr_weight_max_abs_diff",
            core.max_abs_diff(pyepo_params.weight, step1c_params.weight),
            1e-10,
        ),
        CheckResult(
            "level1_lr_bias_max_abs_diff",
            core.max_abs_diff(pyepo_params.bias, step1c_params.bias),
            1e-10,
        ),
        CheckResult(
            "level1_lr_test_pred_cost_max_abs_diff",
            core.max_abs_diff(pyepo_pred, step1c_pred),
            1e-8,
        ),
        CheckResult(
            "level1_lr_test_mse_diff",
            abs(core.mse(pyepo_pred, split.c_test) - core.mse(step1c_pred, split.c_test)),
            1e-10,
        ),
        CheckResult(
            "level1_lr_test_norm_regret_diff",
            abs(pyepo_regret - step1c_regret),
            1e-10,
        ),
    ]


def torch_spoplus_loss_and_grads(model, x, true_cost, true_sol, true_obj, params):
    import torch
    import pyepo

    x_t = torch.tensor(x.astype(np.float32), dtype=torch.float32)
    c_t = torch.tensor(true_cost.astype(np.float32), dtype=torch.float32)
    w_t = torch.tensor(true_sol.astype(np.float32), dtype=torch.float32)
    z_t = torch.tensor(true_obj.reshape(-1, 1).astype(np.float32), dtype=torch.float32)
    weight_t = torch.tensor(params.weight.T.astype(np.float32), requires_grad=True)
    bias_t = torch.tensor(params.bias.astype(np.float32), requires_grad=True)

    pred_t = torch.nn.functional.linear(x_t, weight_t, bias_t)
    pred_t.retain_grad()
    spop = pyepo.func.SPOPlus(model, processes=1)
    loss_vec = spop(pred_t, c_t, w_t, z_t)
    loss_mean = loss_vec.mean()
    loss_mean.backward()
    return {
        "pred": pred_t.detach().cpu().numpy(),
        "loss_vec": loss_vec.detach().cpu().numpy().reshape(-1),
        "grad_pred": pred_t.grad.detach().cpu().numpy(),
        "grad_weight": weight_t.grad.detach().cpu().numpy().T,
        "grad_bias": bias_t.grad.detach().cpu().numpy(),
    }


def level2_3_4_spoplus_alignment(model, split, batch_size, param_seed, lr):
    x = split.x_train[:batch_size].astype(np.float32)
    true_cost = split.c_train[:batch_size].astype(np.float32)
    true_sol, true_obj = core.solve_costs(model, true_cost)
    params = core.deterministic_linear_params(
        feat_dim=x.shape[1],
        cost_dim=true_cost.shape[1],
        seed=param_seed,
        scale=0.1,
    )
    params = core.LinearParams(
        weight=params.weight.astype(np.float32),
        bias=params.bias.astype(np.float32),
    )

    pyepo_values = torch_spoplus_loss_and_grads(
        model, x, true_cost, true_sol, true_obj, params
    )

    pred = pyepo_values["pred"].astype(np.float32)
    shifted_cost = (2.0 * pred - true_cost).astype(np.float32)
    adv_sol, _ = core.solve_costs(model, shifted_cost)

    step1c_loss_vec = core.step1c_cost_min_loss(pred, true_cost, true_sol, adv_sol)
    step1c_grad_pred_unscaled = core.step1c_cost_min_grad_pred(
        true_sol, adv_sol, mean=False
    )
    step1c_grad_pred_mean = step1c_grad_pred_unscaled / batch_size
    step1c_grad_weight, step1c_grad_bias = core.mean_linear_parameter_gradients(
        x, step1c_grad_pred_unscaled
    )
    step1c_next_weight, step1c_next_bias = core.sgd_step(
        params.weight,
        params.bias,
        step1c_grad_weight,
        step1c_grad_bias,
        lr=lr,
    )
    pyepo_next_weight, pyepo_next_bias = core.sgd_step(
        params.weight,
        params.bias,
        pyepo_values["grad_weight"],
        pyepo_values["grad_bias"],
        lr=lr,
    )

    return [
        CheckResult(
            "level2_forward_loss_max_abs_diff",
            core.max_abs_diff(pyepo_values["loss_vec"], step1c_loss_vec),
            1e-5,
        ),
        CheckResult(
            "level3_grad_pred_max_abs_diff",
            core.max_abs_diff(pyepo_values["grad_pred"], step1c_grad_pred_mean),
            1e-6,
        ),
        CheckResult(
            "level3_grad_weight_max_abs_diff",
            core.max_abs_diff(pyepo_values["grad_weight"], step1c_grad_weight),
            1e-6,
        ),
        CheckResult(
            "level3_grad_bias_max_abs_diff",
            core.max_abs_diff(pyepo_values["grad_bias"], step1c_grad_bias),
            1e-6,
        ),
        CheckResult(
            "level4_sgd_weight_update_max_abs_diff",
            core.max_abs_diff(pyepo_next_weight, step1c_next_weight),
            1e-6,
        ),
        CheckResult(
            "level4_sgd_bias_update_max_abs_diff",
            core.max_abs_diff(pyepo_next_bias, step1c_next_bias),
            1e-6,
        ),
    ]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--feat", type=int, default=5)
    parser.add_argument("--degree", type=int, default=1)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid", type=int, nargs=2, default=(5, 5))
    parser.add_argument("--data-root", type=Path, default=core.DEFAULT_DATA_ROOT)
    parser.add_argument("--oracle-count", type=int, default=8)
    parser.add_argument("--spoplus-batch-size", type=int, default=8)
    parser.add_argument("--param-seed", type=int, default=123)
    parser.add_argument("--sgd-lr", type=float, default=0.05)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--no-strict", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = core.fixed_dataset_path(
        data_root=args.data_root,
        grid=tuple(args.grid),
        train_size=args.train_size,
        test_size=args.test_size,
        feat=args.feat,
        deg=args.degree,
        noise=args.noise,
        seed=args.seed,
    )
    if not data_path.exists():
        raise FileNotFoundError(
            "Fixed dataset not found: {}. Run PyEPO/generate_sp_datasets.py first.".format(
                data_path
            )
        )

    split = core.load_fixed_split(data_path)
    model = build_pyepo_model(tuple(args.grid))
    results = []
    results.extend(level0_oracle_alignment(model, split, args.oracle_count))
    results.extend(level1_lr_alignment(model, split))
    results.extend(
        level2_3_4_spoplus_alignment(
            model,
            split,
            batch_size=args.spoplus_batch_size,
            param_seed=args.param_seed,
            lr=args.sgd_lr,
        )
    )

    print("Step1c vs PyEPO shortest-path validation")
    print("dataset: {}".format(data_path))
    print()
    for item in results:
        status = "PASS" if item.passed else "FAIL"
        print(
            "{:<42} {:>12.4e} <= {:<10.4e} {}".format(
                item.name,
                item.value,
                item.threshold,
                status,
            )
        )

    payload = {
        "dataset": str(data_path),
        "results": [
            {
                "name": item.name,
                "value": item.value,
                "threshold": item.threshold,
                "passed": item.passed,
            }
            for item in results
        ],
        "passed": all(item.passed for item in results),
    }
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if not args.no_strict and not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
