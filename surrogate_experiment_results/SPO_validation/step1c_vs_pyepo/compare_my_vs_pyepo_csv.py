#!/usr/bin/env python3
"""Row-by-row CSV comparison for PyEPO and Step1c-compatible SP runners."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import validation_core as core


PYEPO_METHOD_PATTERNS = {
    "pyepo-lr": "n{n}p{p}-d{d}-e{e}_2s-lr.csv",
    "pyepo-spo": "n{n}p{p}-d{d}-e{e}_spo_lr_adam0.01_bs32_l10.0l20.0_c1.csv",
}
DEFAULT_COLUMNS = ["True SPO", "Unamb SPO", "MSE", "Epochs"]
DEFAULT_TOLERANCES = {
    "True SPO": 1e-10,
    "Unamb SPO": 1e-10,
    "MSE": 1e-10,
    "Epochs": 0.0,
}
SPO_TOLERANCES = {
    "True SPO": 1e-5,
    "Unamb SPO": 1e-5,
    "MSE": 1e-4,
    "Epochs": 0.0,
}


@dataclass(frozen=True)
class ColumnDiff:
    max_abs_diff: float
    mean_abs_diff: float
    worst_row: int
    tolerance: float

    @property
    def passed(self) -> bool:
        return self.max_abs_diff <= self.tolerance


@dataclass(frozen=True)
class PairReport:
    left_path: Path
    right_path: Path
    row_count: int
    stats: dict[str, ColumnDiff]

    @property
    def passed(self) -> bool:
        return all(item.passed for item in self.stats.values())


@dataclass(frozen=True)
class SettingReport:
    train_size: int
    deg: int
    noise: float
    report: PairReport


def pyepo_csv_path(root, grid, lan, train_size, feat, deg, noise, method):
    pattern = PYEPO_METHOD_PATTERNS[method]
    filename = pattern.format(n=train_size, p=feat, d=deg, e=noise)
    return Path(root) / "sp" / "h{}w{}".format(*tuple(grid)) / lan / filename


def compare_csv_pair(
    left_path,
    right_path,
    columns=DEFAULT_COLUMNS,
    tolerances=None,
    limit_rows=None,
) -> PairReport:
    left_path = Path(left_path)
    right_path = Path(right_path)
    tolerances = dict(DEFAULT_TOLERANCES if tolerances is None else tolerances)
    left = pd.read_csv(left_path)
    right = pd.read_csv(right_path)
    if limit_rows is not None:
        limit_rows = int(limit_rows)
        if limit_rows <= 0:
            raise ValueError("limit_rows must be positive.")
        if len(left) < limit_rows or len(right) < limit_rows:
            raise ValueError(
                "Cannot compare first {} rows: {} has {}, {} has {}".format(
                    limit_rows,
                    left_path,
                    len(left),
                    right_path,
                    len(right),
                )
            )
        left = left.iloc[:limit_rows].copy()
        right = right.iloc[:limit_rows].copy()
    elif len(left) != len(right):
        raise ValueError(
            "Row count mismatch: {} has {}, {} has {}".format(
                left_path,
                len(left),
                right_path,
                len(right),
            )
        )
    stats = {}
    for col in columns:
        if col not in left.columns:
            raise ValueError("{} missing column {}".format(left_path, col))
        if col not in right.columns:
            raise ValueError("{} missing column {}".format(right_path, col))
        diff = np.abs(left[col].to_numpy(dtype=float) - right[col].to_numpy(dtype=float))
        worst_row = int(np.argmax(diff)) if len(diff) else -1
        stats[col] = ColumnDiff(
            max_abs_diff=float(diff[worst_row]) if len(diff) else 0.0,
            mean_abs_diff=float(np.mean(diff)) if len(diff) else 0.0,
            worst_row=worst_row,
            tolerance=float(tolerances[col]),
        )
    return PairReport(
        left_path=left_path,
        right_path=right_path,
        row_count=len(left),
        stats=stats,
    )


def compare_settings(args):
    reports = []
    tolerances = SPO_TOLERANCES if args.pair == "spo" else DEFAULT_TOLERANCES
    for train_size in args.train_sizes:
        for noise in args.noises:
            for deg in args.degs:
                left = pyepo_csv_path(
                    args.pyepo_result_root,
                    args.grid,
                    args.lan,
                    train_size,
                    args.feat,
                    deg,
                    noise,
                    args.pyepo_method,
                )
                right = core.result_csv_path(
                    result_root=args.my_result_root,
                    method_slug=args.my_method,
                    grid=args.grid,
                    train_size=train_size,
                    feat=args.feat,
                    deg=deg,
                    noise=noise,
                    lan=args.lan,
                )
                if not left.exists() or not right.exists():
                    if args.allow_missing:
                        print("SKIP missing: {} | {}".format(left, right))
                        continue
                    missing = left if not left.exists() else right
                    raise FileNotFoundError("Missing result CSV: {}".format(missing))
                report = compare_csv_pair(
                    left,
                    right,
                    columns=args.columns,
                    tolerances=tolerances,
                    limit_rows=args.limit_rows,
                )
                reports.append(
                    SettingReport(
                        train_size=train_size,
                        deg=deg,
                        noise=noise,
                        report=report,
                    )
                )
    return reports


def print_reports(reports):
    global_stats = {}
    for setting in reports:
        print(
            "n={}, e={}, d={} rows={} status={}".format(
                setting.train_size,
                setting.noise,
                setting.deg,
                setting.report.row_count,
                "PASS" if setting.report.passed else "FAIL",
            )
        )
        for col, stat in setting.report.stats.items():
            print(
                "  {:<10} max_abs={:.6g} mean_abs={:.6g} worst_row={} tol={:.2g} {}".format(
                    col,
                    stat.max_abs_diff,
                    stat.mean_abs_diff,
                    stat.worst_row,
                    stat.tolerance,
                    "PASS" if stat.passed else "FAIL",
                )
            )
            current = global_stats.get(col, (0.0, None))
            if stat.max_abs_diff >= current[0]:
                global_stats[col] = (stat.max_abs_diff, setting)
    if reports:
        print()
        print("global:")
        for col, (value, setting) in global_stats.items():
            print(
                "  {:<10} max_abs={:.6g} at n={}, e={}, d={}".format(
                    col,
                    value,
                    setting.train_size,
                    setting.noise,
                    setting.deg,
                )
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", choices=["lr", "spo"], default="lr")
    parser.add_argument("--pyepo-result-root", type=Path, default=core.SPO_VALIDATION_DIR / "res")
    parser.add_argument("--my-result-root", type=Path, default=core.DEFAULT_RESULT_ROOT)
    parser.add_argument("--grid", type=int, nargs=2, default=core.DEFAULT_GRID)
    parser.add_argument("--feat", type=int, default=5)
    parser.add_argument("--lan", type=str, default="gurobi")
    parser.add_argument("--train-sizes", type=int, nargs="+", default=list(core.DEFAULT_TRAIN_SIZES))
    parser.add_argument("--degs", type=int, nargs="+", default=list(core.DEFAULT_DEGREES))
    parser.add_argument("--noises", type=float, nargs="+", default=list(core.DEFAULT_NOISES))
    parser.add_argument("--columns", nargs="+", default=DEFAULT_COLUMNS)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    if args.pair == "lr":
        args.pyepo_method = "pyepo-lr"
        args.my_method = "my-2stage-lr"
    else:
        args.pyepo_method = "pyepo-spo"
        args.my_method = "my-spoplus"
    return args


def main():
    reports = compare_settings(parse_args())
    print_reports(reports)
    if not reports:
        raise SystemExit("No result pairs compared.")
    if not all(item.report.passed for item in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
