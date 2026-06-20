#!/usr/bin/env python3
"""Train Step3 2stage with native validation-MSE early stopping.

This script intentionally lives under Step3 so Step1c training code remains
unchanged. It reuses Step1c data loading and MSE helpers, but owns the
early-stopping training loop and its audit JSON.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
STEP1C_DIR = PROJECT_ROOT / "surrogate_experiment_results" / "Step1c"
if str(STEP1C_DIR) not in sys.path:
    sys.path.insert(0, str(STEP1C_DIR))

import train_2stage as step1c_2stage  # noqa: E402
from split_dataset import graph_entries_from_data_dir, read_json, select_train_subset  # noqa: E402


DEFAULT_SPLIT_PATH = PROJECT_ROOT / "results" / "step1b_splits" / "master_split_seed=42.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "step3_runs" / "train_size=50"


class EarlyStoppingTracker:
    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        if patience <= 0:
            raise ValueError("early stopping patience must be positive")
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best_epoch: int | None = None
        self.best_value: float | None = None
        self.stop_epoch: int | None = None
        self.num_bad_checks = 0
        self.should_stop = False

    def update(self, epoch: int, value: float) -> bool:
        value = float(value)
        if self.best_value is None or value < self.best_value - self.min_delta:
            self.best_epoch = int(epoch)
            self.best_value = value
            self.num_bad_checks = 0
            return False

        self.num_bad_checks += 1
        if self.num_bad_checks >= self.patience:
            self.should_stop = True
            self.stop_epoch = int(epoch)
        return self.should_stop

    def to_dict(self) -> dict[str, int | float | bool | None]:
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best_epoch": self.best_epoch,
            "best_value": self.best_value,
            "stop_epoch": self.stop_epoch,
            "num_bad_checks": self.num_bad_checks,
            "should_stop": self.should_stop,
        }


def run_mse_trajectory_with_early_stopping(
    train_graphs: list[dict],
    validation_graphs: list[dict],
    *,
    theta_init: np.ndarray,
    n_epochs: int,
    lr: float,
    metric_stride: int,
    patience: int,
    min_delta: float,
) -> tuple[np.ndarray, dict]:
    if metric_stride <= 0:
        raise ValueError("--metric_stride must be positive")

    opt = step1c_2stage.Adam(lr)
    theta = np.asarray(theta_init, dtype=float).copy()
    trajectory = [theta.copy()]
    tracker = EarlyStoppingTracker(patience=patience, min_delta=min_delta)

    validation_mse = step1c_2stage.mse_loss(theta, validation_graphs)
    tracker.update(epoch=0, value=validation_mse)
    print(
        "  [2stage/early-stop] epoch    0 "
        f"validation_mse_loss={validation_mse:.6f} best_epoch={tracker.best_epoch}",
        flush=True,
    )

    for epoch in range(1, int(n_epochs) + 1):
        theta = opt.step(theta, step1c_2stage.grad_mse(train_graphs, theta))
        trajectory.append(theta.copy())
        print(f"  [2stage/MSE] epoch {epoch:>4} theta={np.round(theta, 4)}", flush=True)

        if epoch % int(metric_stride) != 0 and epoch != int(n_epochs):
            continue

        validation_mse = step1c_2stage.mse_loss(theta, validation_graphs)
        should_stop = tracker.update(epoch=epoch, value=validation_mse)
        print(
            f"  [2stage/early-stop] epoch {epoch:>4} "
            f"validation_mse_loss={validation_mse:.6f} "
            f"best_epoch={tracker.best_epoch} bad_checks={tracker.num_bad_checks}",
            flush=True,
        )
        if should_stop:
            print(
                f"  [2stage/early-stop] stopping at epoch {epoch}; "
                f"best validation_mse_loss={tracker.best_value:.6f} "
                f"at epoch {tracker.best_epoch}",
                flush=True,
            )
            break

    stopped_epoch = len(trajectory) - 1
    summary = tracker.to_dict()
    summary.update(
        {
            "enabled": True,
            "source": "step3_native",
            "metric": "validation_mse_loss",
            "max_epochs": int(n_epochs),
            "metric_stride": int(metric_stride),
            "stopped_epoch": int(stopped_epoch),
            "stopped_early": bool(stopped_epoch < int(n_epochs)),
        }
    )
    return np.asarray(trajectory, dtype=float), summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split_path", default=str(DEFAULT_SPLIT_PATH))
    parser.add_argument("--validation_data_dir", default=None)
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--train_size", type=int, default=50)
    parser.add_argument("--subset_seed", type=int, default=42)
    parser.add_argument("--theta_seed", type=int, default=42)
    parser.add_argument("--theta_init", type=float, nargs=2, default=None)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--train_graph_limit", type=int, default=None)
    parser.add_argument("--validation_limit", type=int, default=None)
    parser.add_argument("--metric_stride", type=int, default=1)
    parser.add_argument("--early_stop_patience", type=int, required=True)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0)
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    metrics_dir = out_dir / "metrics"
    trajectories_dir = out_dir / "trajectories"
    model_weights_dir = out_dir / "model_weights"
    plots_dir = out_dir / "plots"
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    split = read_json(args.split_path)
    train_entries = select_train_subset(
        split["train_pool"], train_size=args.train_size, seed=args.subset_seed
    )
    if args.train_graph_limit is not None:
        train_entries = train_entries[: args.train_graph_limit]
    validation_entries = (
        graph_entries_from_data_dir(args.validation_data_dir)
        if args.validation_data_dir
        else split["validation"]
    )
    if args.validation_limit is not None:
        validation_entries = validation_entries[: args.validation_limit]

    step1c_2stage.write_json(out_dir / "train_subset.json", train_entries)
    step1c_2stage.write_json(out_dir / "validation_set.json", validation_entries)
    print(f"Loading 2stage train subset: n={len(train_entries)}")
    train_graphs = step1c_2stage.load_reward_records(train_entries)
    validation_source = args.validation_data_dir or "split validation"
    print(f"Loading 2stage validation set: n={len(validation_entries)} source={validation_source}")
    validation_graphs = step1c_2stage.load_reward_records(validation_entries)

    rng = np.random.RandomState(args.theta_seed)
    theta_init = (
        np.asarray(args.theta_init, dtype=float)
        if args.theta_init is not None
        else rng.uniform(0.5, 3.5, size=2)
    )
    print(f"2stage theta_init={np.round(theta_init, 4)}")

    trajectory, early_stopping_summary = run_mse_trajectory_with_early_stopping(
        train_graphs,
        validation_graphs,
        theta_init=theta_init,
        n_epochs=args.n_epochs,
        lr=args.lr,
        metric_stride=args.metric_stride,
        patience=args.early_stop_patience,
        min_delta=args.early_stop_min_delta,
    )
    np.save(trajectories_dir / "trajectory_2stage.npy", trajectory)
    step1c_2stage.write_json(metrics_dir / "early_stopping_2stage.json", early_stopping_summary)

    train_mse = step1c_2stage.evaluate_mse_losses(trajectory, train_graphs)
    validation_mse = step1c_2stage.evaluate_mse_losses(trajectory, validation_graphs)
    rows = []
    for epoch, theta in enumerate(trajectory):
        rows.append(
            {
                "epoch": epoch,
                "theta_1": float(theta[0]),
                "theta_2": float(theta[1]),
                "train_mse_loss": float(train_mse[epoch]),
                "validation_mse_loss": float(validation_mse[epoch]),
            }
        )
    loss_csv = metrics_dir / "2stage_loss_curve.csv"
    step1c_2stage.write_csv(
        loss_csv,
        rows,
        ["epoch", "theta_1", "theta_2", "train_mse_loss", "validation_mse_loss"],
    )

    best_epoch = int(early_stopping_summary["best_epoch"])
    checkpoint = {
        "method": "2stage",
        "epoch": best_epoch,
        "theta": np.asarray(trajectory[best_epoch], dtype=float).copy(),
        "selection_metric": "validation_mse_loss",
        "selection_value": float(validation_mse[best_epoch]),
    }
    weights_path = step1c_2stage.write_model_weights(model_weights_dir, checkpoint, args.train_size)
    print(
        "Saved 2stage model weights "
        f"{weights_path} at epoch {checkpoint['epoch']} "
        f"validation_mse_loss={checkpoint['selection_value']:.6f}"
    )

    if args.plot:
        step1c_2stage.plot_loss_curve(
            loss_csv,
            plots_dir / "2stage_mse_loss.png",
            train_column="train_mse_loss",
            validation_column="validation_mse_loss",
            ylabel="MSE loss",
            title="Step3 2stage reward-fitting MSE loss with early stopping",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
