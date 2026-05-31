#!/usr/bin/env python3
"""Run a Step1c-compatible cost-min SPO+ linear adapter on PyEPO SP data."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import validation_core as core


METHOD_SLUG = "my-spoplus"


class Step1cLinearModel(nn.Module):
    """Linear C_hat = X @ W + b with numpy W shaped (features, costs)."""

    def __init__(self, feat_dim: int, cost_dim: int):
        super().__init__()
        self.linear = nn.Linear(feat_dim, cost_dim)

    def forward(self, x):
        return self.linear(x)

    def set_numpy_params(self, weight, bias) -> None:
        weight = np.asarray(weight, dtype=np.float32)
        bias = np.asarray(bias, dtype=np.float32)
        with torch.no_grad():
            self.linear.weight.copy_(torch.as_tensor(weight.T))
            self.linear.bias.copy_(torch.as_tensor(bias))

    def predict_numpy(self, x) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            pred = self(torch.as_tensor(np.asarray(x, dtype=np.float32)))
        self.train()
        return pred.detach().cpu().numpy()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def build_model(grid):
    core.ensure_local_pyepo_on_path()
    import pyepo

    return pyepo.model.grb.shortestPathModel(tuple(grid))


def build_dataset(optmodel, split):
    core.ensure_local_pyepo_on_path()
    import pyepo

    trainset = pyepo.data.dataset.optDataset(optmodel, split.x_train, split.c_train)
    testset = pyepo.data.dataset.optDataset(optmodel, split.x_test, split.c_test)
    return trainset, testset


def default_epochs(train_size: int) -> int:
    if train_size == 100:
        return 200
    if train_size == 1000:
        return 20
    if train_size == 5000:
        return 4
    raise ValueError("No default epoch schedule for train_size={}".format(train_size))


def make_optimizer(model, optm: str, lr: float):
    if optm == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    if optm == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr)
    raise ValueError("Unsupported optimizer: {}".format(optm))


def step1c_spoplus_backward(pred_cost, true_cost, true_sol, true_obj, optmodel):
    pred_np = pred_cost.detach().cpu().numpy()
    true_cost_np = true_cost.detach().cpu().numpy()
    true_sol_np = true_sol.detach().cpu().numpy()
    shifted_cost = 2.0 * pred_np - true_cost_np
    adversarial_sol, _ = core.solve_costs(optmodel, shifted_cost)

    loss_vec = core.step1c_cost_min_loss(
        pred_np,
        true_cost_np,
        true_sol_np,
        adversarial_sol,
    )
    grad_pred = core.step1c_cost_min_grad_pred(
        true_sol_np,
        adversarial_sol,
        mean=True,
    )
    pred_cost.backward(
        torch.as_tensor(grad_pred, dtype=pred_cost.dtype, device=pred_cost.device)
    )
    return float(np.mean(loss_vec))


def train_spoplus(model, optmodel, trainset, batch_size, epochs, optm, lr):
    optimizer = make_optimizer(model, optm=optm, lr=lr)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    model.train()
    last_loss = None
    for epoch in range(epochs):
        for x, c, w, z in trainloader:
            optimizer.zero_grad()
            pred = model(x)
            last_loss = step1c_spoplus_backward(pred, c, w, z, optmodel)
            optimizer.step()
        print("    epoch {}/{} loss {:.6g}".format(epoch + 1, epochs, last_loss))
    return model


def evaluate_torch_model(model, optmodel, testset, batch_size):
    core.ensure_local_pyepo_on_path()
    import pyepo

    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return {
        "True SPO": float(pyepo.metric.regret(model, optmodel, testloader)),
        "Unamb SPO": float(pyepo.metric.unambRegret(model, optmodel, testloader)),
        "MSE": float(pyepo.metric.MSE(model, testloader)),
    }


def run_one_seed(args, setting, optmodel):
    seed_everything(setting.seed)
    data_path = core.fixed_dataset_path(
        data_root=args.data_root,
        grid=tuple(args.grid),
        train_size=setting.train_size,
        test_size=args.test_size,
        feat=args.feat,
        deg=setting.deg,
        noise=setting.noise,
        seed=setting.seed,
    )
    split = core.load_fixed_split(data_path)
    trainset, testset = build_dataset(optmodel, split)

    model = Step1cLinearModel(
        feat_dim=split.x_train.shape[1],
        cost_dim=split.c_train.shape[1],
    )
    epochs = args.epochs if args.epochs is not None else default_epochs(setting.train_size)
    if args.smoke and args.epochs is None:
        epochs = args.smoke_epochs

    tick = time.time()
    train_spoplus(
        model,
        optmodel,
        trainset,
        batch_size=args.batch,
        epochs=epochs,
        optm=args.optm,
        lr=args.lr,
    )
    elapsed = time.time() - tick

    metrics = evaluate_torch_model(model, optmodel, testset, batch_size=args.batch)
    return {
        "True SPO": metrics["True SPO"],
        "Unamb SPO": metrics["Unamb SPO"],
        "MSE": metrics["MSE"],
        "Elapsed": elapsed,
        "Epochs": epochs,
    }


def iter_setting_groups(args):
    train_sizes = [100] if args.smoke else args.train_sizes
    degs = [1] if args.smoke else args.degs
    noises = [0.0] if args.smoke else args.noises
    expnum = 1 if args.smoke else args.expnum
    for train_size in train_sizes:
        for noise in noises:
            for deg in degs:
                yield train_size, deg, noise, expnum


def run(args):
    optmodel = build_model(tuple(args.grid))
    total_new_rows = 0
    for train_size, deg, noise, expnum in iter_setting_groups(args):
        save_path = core.result_csv_path(
            result_root=args.result_root,
            method_slug=METHOD_SLUG,
            grid=tuple(args.grid),
            train_size=train_size,
            feat=args.feat,
            deg=deg,
            noise=noise,
            lan=args.lan,
        )
        if args.overwrite and save_path.exists():
            save_path.unlink()
        completed = core.completed_seed_count(save_path)
        print("Setting n={} d={} e={} -> {}".format(train_size, deg, noise, save_path))
        for seed in range(expnum):
            if seed < completed:
                print("  seed {}: skip existing row".format(seed))
                continue
            setting = core.SPSetting(
                train_size=train_size,
                deg=deg,
                noise=noise,
                seed=seed,
            )
            row = run_one_seed(args, setting, optmodel)
            core.append_result_row(save_path, row)
            total_new_rows += 1
            print(
                "  seed {}: True SPO={:.6g}, Unamb SPO={:.6g}, MSE={:.6g}".format(
                    seed,
                    row["True SPO"],
                    row["Unamb SPO"],
                    row["MSE"],
                )
            )
    print("my_spoplus complete: new rows={}".format(total_new_rows))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expnum", type=int, default=10)
    parser.add_argument("--data-root", type=Path, default=core.DEFAULT_DATA_ROOT)
    parser.add_argument("--result-root", type=Path, default=core.DEFAULT_RESULT_ROOT)
    parser.add_argument("--grid", type=int, nargs=2, default=core.DEFAULT_GRID)
    parser.add_argument("--feat", type=int, default=5)
    parser.add_argument("--test-size", type=int, default=core.DEFAULT_TEST_SIZE)
    parser.add_argument("--train-sizes", type=int, nargs="+", default=list(core.DEFAULT_TRAIN_SIZES))
    parser.add_argument("--degs", type=int, nargs="+", default=list(core.DEFAULT_DEGREES))
    parser.add_argument("--noises", type=float, nargs="+", default=list(core.DEFAULT_NOISES))
    parser.add_argument("--lan", type=str, default="gurobi")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--smoke-epochs", type=int, default=1)
    parser.add_argument("--optm", type=str, choices=["adam", "sgd"], default="adam")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
