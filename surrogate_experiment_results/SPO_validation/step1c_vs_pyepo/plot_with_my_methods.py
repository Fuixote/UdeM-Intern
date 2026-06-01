#!/usr/bin/env python3
"""Plot PyEPO shortest-path results together with Step1c-compatible runners."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import validation_core as core


PYEPO_FILENAMES = {
    "pyepo-lr": "n{n}p{p}-d{d}-e{e}_2s-lr.csv",
    "pyepo-spo": "n{n}p{p}-d{d}-e{e}_spo_lr_adam0.01_bs32_l10.0l20.0_c1.csv",
}
METHODS = [
    ("pyepo-lr", "PyEPO LR"),
    ("pyepo-spo", "PyEPO SPO+"),
    ("my-2stage-lr", "my 2stage LR"),
    ("my-spoplus", "my SPO+"),
]


def pyepo_csv_path(root, grid, lan, train_size, feat, deg, noise, method):
    filename = PYEPO_FILENAMES[method].format(
        n=train_size,
        p=feat,
        d=deg,
        e=noise,
    )
    return Path(root) / "sp" / "h{}w{}".format(*tuple(grid)) / lan / filename


def method_csv_path(args, method, train_size, deg, noise):
    if method in PYEPO_FILENAMES:
        return pyepo_csv_path(
            args.pyepo_result_root,
            args.grid,
            args.lan,
            train_size,
            args.feat,
            deg,
            noise,
            method,
        )
    return core.result_csv_path(
        result_root=args.my_result_root,
        method_slug=method,
        grid=args.grid,
        train_size=train_size,
        feat=args.feat,
        deg=deg,
        noise=noise,
        lan=args.lan,
    )


def load_method_degree_values(args, method, train_size, noise):
    values = []
    for deg in args.degs:
        path = method_csv_path(args, method, train_size, deg, noise)
        if not path.exists():
            if args.allow_missing:
                return None
            raise FileNotFoundError("Missing result CSV: {}".format(path))
        df = pd.read_csv(path)
        values.append(df[args.column].to_numpy())
    return values


def plot_one(args, train_size, noise):
    available = []
    for method, label in METHODS:
        values = load_method_degree_values(args, method, train_size, noise)
        if values is not None:
            available.append((method, label, values))
    if not available:
        print("No complete methods for n={} noise={}; skip".format(train_size, noise))
        return None

    fig, ax = plt.subplots(figsize=(16, 6))
    width = min(0.1, 0.8 / max(len(available), 1))
    offsets = np.linspace(
        -width * (len(available) - 1) / 2,
        width * (len(available) - 1) / 2,
        len(available),
    )
    cmap = plt.get_cmap("tab10")
    handles = []
    labels = []
    for idx, (_, label, values) in enumerate(available):
        positions = np.arange(len(args.degs)) + offsets[idx]
        color = cmap(idx % 10)
        bp = ax.boxplot(
            values,
            positions=positions,
            widths=width * 0.85,
            patch_artist=True,
            boxprops={"facecolor": color, "color": color, "linewidth": 1.5},
            medianprops={"color": "white", "linewidth": 1.5},
            whiskerprops={"color": color, "linewidth": 1.2},
            capprops={"color": color, "linewidth": 1.2},
            flierprops={
                "marker": "o",
                "markersize": 3,
                "markeredgecolor": color,
            },
        )
        handles.append(bp["boxes"][0])
        labels.append(label)

    ax.set_xticks(range(len(args.degs)))
    ax.set_xticklabels(args.degs)
    ax.set_xlabel("Polynomial Degree")
    ax.set_ylabel("Normalized Regret")
    ax.set_title(
        "Shortest Path: train size = {}, noise half-width = {}".format(
            train_size,
            noise,
        )
    )
    ax.grid(axis="y", color="0.85", linewidth=0.8)
    ax.legend(handles, labels, loc="upper left", ncol=3, fontsize=9)
    fig.tight_layout()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "my-methods-sp-n{}e{}.png".format(
        train_size,
        int(10 * noise),
    )
    fig.savefig(output, dpi=args.dpi)
    plt.close(fig)
    print("Saved to {}".format(output))
    return output


def run(args):
    outputs = []
    for train_size in args.train_sizes:
        for noise in args.noises:
            output = plot_one(args, train_size, noise)
            if output is not None:
                outputs.append(output)
    print("Generated {} plot(s).".format(len(outputs)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prob", choices=["sp"], default="sp")
    parser.add_argument("--pyepo-result-root", type=Path, default=core.SPO_VALIDATION_DIR / "res")
    parser.add_argument("--my-result-root", type=Path, default=core.DEFAULT_RESULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "images")
    parser.add_argument("--grid", type=int, nargs=2, default=core.DEFAULT_GRID)
    parser.add_argument("--feat", type=int, default=5)
    parser.add_argument("--lan", type=str, default="gurobi")
    parser.add_argument("--train-sizes", type=int, nargs="+", default=list(core.DEFAULT_TRAIN_SIZES))
    parser.add_argument("--degs", type=int, nargs="+", default=list(core.DEFAULT_DEGREES))
    parser.add_argument("--noises", type=float, nargs="+", default=list(core.DEFAULT_NOISES))
    parser.add_argument("--column", type=str, default="True SPO")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
