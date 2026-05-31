#!/usr/bin/env python3
"""Run the Step1c-compatible two-stage LR adapter on fixed PyEPO SP data."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys
import time

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import validation_core as core


METHOD_SLUG = "my-2stage-lr"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def build_model(grid):
    core.ensure_local_pyepo_on_path()
    import pyepo

    return pyepo.model.grb.shortestPathModel(tuple(grid))


def fit_lr(split):
    core.ensure_local_pyepo_on_path()
    import pyepo
    from sklearn.linear_model import LinearRegression

    predictor = pyepo.twostage.sklearnPred(LinearRegression())
    predictor.fit(split.x_train, split.c_train)
    return predictor


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

    tick = time.time()
    predictor = fit_lr(split)
    elapsed = time.time() - tick

    pred_costs = predictor.predict(split.x_test)
    _, true_objs = core.solve_costs(optmodel, split.c_test)
    metrics = core.prediction_metrics_2stage(
        optmodel,
        pred_costs,
        split.c_test,
        true_objs,
    )
    return {
        "True SPO": metrics["True SPO"],
        "Unamb SPO": metrics["Unamb SPO"],
        "MSE": metrics["MSE"],
        "Elapsed": elapsed,
        "Epochs": 0,
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
    print("my_2stage_lr complete: new rows={}".format(total_new_rows))


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
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
