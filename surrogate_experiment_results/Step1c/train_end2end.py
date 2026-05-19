"""Train the Step1c end-to-end FY decision-focused model."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import step1c_common as common
from split_dataset import graph_entries_from_data_dir, read_json, select_train_subset
from plot_training_curves import plot_loss_curve


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT_PATH = PROJECT_ROOT / "results" / "step1b_splits" / "master_split_seed=42.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "step1c_runs" / "train_size=50"
MODEL_FORMULA = "w_hat_e = theta_1 * utility_e + theta_2 * recipient_cPRA_e"


def select_best_decision_gap_checkpoint(trajectory, validation_decision_gap):
    validation_decision_gap = np.asarray(validation_decision_gap, dtype=float)
    best_idx = int(np.nanargmin(validation_decision_gap))
    return {
        "method": "e2e",
        "epoch": best_idx,
        "theta": np.asarray(trajectory[best_idx], dtype=float).copy(),
        "selection_metric": "validation_decision_gap",
        "selection_value": float(validation_decision_gap[best_idx]),
    }


def select_best_fy_loss_checkpoint(trajectory, validation_fy_loss):
    validation_fy_loss = np.asarray(validation_fy_loss, dtype=float)
    best_idx = int(np.nanargmin(validation_fy_loss))
    return {
        "method": "e2e",
        "epoch": best_idx,
        "theta": np.asarray(trajectory[best_idx], dtype=float).copy(),
        "selection_metric": "validation_fy_loss",
        "selection_value": float(validation_fy_loss[best_idx]),
    }


class EarlyStoppingTracker:
    def __init__(self, patience, min_delta=0.0):
        if patience <= 0:
            raise ValueError("early stopping patience must be positive")
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best_epoch = None
        self.best_value = None
        self.stop_epoch = None
        self.num_bad_checks = 0
        self.should_stop = False

    def update(self, epoch, value):
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

    def to_dict(self):
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best_epoch": self.best_epoch,
            "best_value": self.best_value,
            "stop_epoch": self.stop_epoch,
            "num_bad_checks": self.num_bad_checks,
            "should_stop": self.should_stop,
        }


def run_fy_trajectory_with_early_stopping(
    train_graphs,
    validation_graphs,
    theta_init,
    n_epochs,
    lr,
    eps_abs,
    M,
    seed,
    metric_stride,
    patience,
    min_delta,
    env,
):
    if metric_stride <= 0:
        raise ValueError("--metric_stride must be positive")

    opt = common.Adam(lr)
    rng = np.random.RandomState(seed)
    theta = np.asarray(theta_init, dtype=float).copy()
    trajectory = [theta.copy()]
    tracker = EarlyStoppingTracker(patience=patience, min_delta=min_delta)
    validation_perturbations = common.make_perturbations_by_graph(
        validation_graphs, eps_abs, M, seed
    )

    validation_fy = common.average_fy_objective(
        theta, validation_graphs, validation_perturbations, env
    )
    tracker.update(epoch=0, value=validation_fy)
    print(
        "  [early-stop] epoch    0 "
        f"validation_fy_loss={validation_fy:.6f} best_epoch={tracker.best_epoch}"
    )

    for epoch in range(1, n_epochs + 1):
        theta = opt.step(theta, common.grad_fy(train_graphs, theta, eps_abs, M, rng, env))
        trajectory.append(theta.copy())
        print(f"  [FY] epoch {epoch:>4} theta={np.round(theta, 4)}")

        if epoch % metric_stride != 0 and epoch != n_epochs:
            continue

        validation_fy = common.average_fy_objective(
            theta, validation_graphs, validation_perturbations, env
        )
        should_stop = tracker.update(epoch=epoch, value=validation_fy)
        print(
            f"  [early-stop] epoch {epoch:>4} "
            f"validation_fy_loss={validation_fy:.6f} "
            f"best_epoch={tracker.best_epoch} bad_checks={tracker.num_bad_checks}"
        )
        if should_stop:
            print(
                f"  [early-stop] stopping at epoch {epoch}; "
                f"best validation_fy_loss={tracker.best_value:.6f} "
                f"at epoch {tracker.best_epoch}"
            )
            break

    summary = tracker.to_dict()
    summary.update(
        {
            "enabled": True,
            "metric": "validation_fy_loss",
            "max_epochs": int(n_epochs),
            "metric_stride": int(metric_stride),
            "stopped_epoch": len(trajectory) - 1,
        }
    )
    return np.asarray(trajectory, dtype=float), summary


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_model_weights(out_dir, checkpoint, train_size):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    theta = np.asarray(checkpoint["theta"], dtype=float)
    stem = f"e2e_best_by_{checkpoint['selection_metric']}"
    npz_path = out_dir / f"{stem}.npz"
    np.savez_compressed(
        npz_path,
        theta=theta,
        method=np.asarray("e2e"),
        model_type=np.asarray("linear_probe"),
        model_formula=np.asarray(MODEL_FORMULA),
        feature_names=np.asarray(common.PROBE["feature_names"]),
        train_size=np.asarray(int(train_size)),
        selected_epoch=np.asarray(int(checkpoint["epoch"])),
        selection_metric=np.asarray(checkpoint["selection_metric"]),
        selection_value=np.asarray(float(checkpoint["selection_value"])),
    )
    write_json(
        out_dir / f"{stem}.json",
        {
            "method": "e2e",
            "model_type": "linear_probe",
            "model_formula": MODEL_FORMULA,
            "feature_names": common.PROBE["feature_names"],
            "train_size": int(train_size),
            "selected_epoch": int(checkpoint["epoch"]),
            "selection_metric": checkpoint["selection_metric"],
            "selection_value": float(checkpoint["selection_value"]),
            "theta": theta.tolist(),
        },
    )
    return npz_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train Step1c end-to-end FY model.")
    parser.add_argument("--split_path", default=str(DEFAULT_SPLIT_PATH))
    parser.add_argument(
        "--validation_data_dir",
        default=None,
        help=(
            "Optional processed G-*.json directory used for validation/checkpoint "
            "selection instead of the validation split in --split_path."
        ),
    )
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--train_size", type=int, default=50)
    parser.add_argument("--subset_seed", type=int, default=42)
    parser.add_argument("--theta_seed", type=int, default=42)
    parser.add_argument("--theta_init", type=float, nargs=2, default=None)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--fy_epsilon", type=float, default=1.0)
    parser.add_argument("--fy_M", type=int, default=4)
    parser.add_argument("--metric_stride", type=int, default=1)
    parser.add_argument(
        "--early_stop_metric",
        choices=["validation_fy_loss"],
        default=None,
        help="Enable early stopping using the selected validation metric.",
    )
    parser.add_argument("--early_stop_patience", type=int, default=0)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0)
    parser.add_argument("--gurobi_seed", type=int, default=42)
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    import gurobipy as gp

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
    validation_entries = (
        graph_entries_from_data_dir(args.validation_data_dir)
        if args.validation_data_dir
        else split["validation"]
    )
    write_json(out_dir / "validation_set.json", validation_entries)

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Seed", args.gurobi_seed)
    env.start()

    train_graphs = []
    validation_graphs = []
    try:
        print(f"Loading e2e train subset: n={len(train_entries)}")
        train_graphs = common.load_graph_records(
            [entry["path"] for entry in train_entries], env
        )
        validation_source = args.validation_data_dir or "split validation"
        print(f"Loading e2e validation set: n={len(validation_entries)} source={validation_source}")
        validation_graphs = common.load_graph_records(
            [entry["path"] for entry in validation_entries], env
        )

        rng = np.random.RandomState(args.theta_seed)
        theta_init = (
            np.asarray(args.theta_init, dtype=float)
            if args.theta_init is not None
            else rng.uniform(0.5, 3.5, size=2)
        )
        print(f"e2e theta_init={np.round(theta_init, 4)}")

        early_stopping_summary = {
            "enabled": False,
            "metric": args.early_stop_metric,
            "max_epochs": int(args.n_epochs),
            "metric_stride": int(args.metric_stride),
            "stopped_epoch": int(args.n_epochs),
        }
        if args.early_stop_metric == "validation_fy_loss":
            trajectory, early_stopping_summary = run_fy_trajectory_with_early_stopping(
                train_graphs,
                validation_graphs,
                theta_init=theta_init,
                n_epochs=args.n_epochs,
                lr=args.lr,
                eps_abs=args.fy_epsilon,
                M=args.fy_M,
                seed=args.theta_seed,
                metric_stride=args.metric_stride,
                patience=args.early_stop_patience,
                min_delta=args.early_stop_min_delta,
                env=env,
            )
        else:
            trajectory = common.run_fy_trajectory(
                train_graphs,
                theta_init=theta_init,
                n_epochs=args.n_epochs,
                lr=args.lr,
                eps_abs=args.fy_epsilon,
                M=args.fy_M,
                seed=args.theta_seed,
                env=env,
            )
        np.save(trajectories_dir / "trajectory_e2e.npy", trajectory)
        write_json(metrics_dir / "early_stopping.json", early_stopping_summary)

        eval_indices = common.trajectory_epoch_indices(len(trajectory), args.metric_stride)
        trajectory_subset = trajectory[eval_indices]
        train_gaps = common.evaluate_trajectory_decision_gap(
            trajectory, train_graphs, env, indices=eval_indices
        )
        validation_gaps = common.evaluate_trajectory_decision_gap(
            trajectory, validation_graphs, env, indices=eval_indices
        )
        train_perturbations = common.make_perturbations_by_graph(
            train_graphs, args.fy_epsilon, args.fy_M, args.theta_seed
        )
        validation_perturbations = common.make_perturbations_by_graph(
            validation_graphs, args.fy_epsilon, args.fy_M, args.theta_seed
        )
        train_fy = common.evaluate_trajectory_fy_objective(
            trajectory, train_graphs, train_perturbations, env, indices=eval_indices
        )
        validation_fy = common.evaluate_trajectory_fy_objective(
            trajectory,
            validation_graphs,
            validation_perturbations,
            env,
            indices=eval_indices,
        )

        rows = []
        for row_idx, epoch in enumerate(eval_indices):
            theta = trajectory_subset[row_idx]
            rows.append(
                {
                    "epoch": int(epoch),
                    "theta_1": float(theta[0]),
                    "theta_2": float(theta[1]),
                    "train_fy_loss": float(train_fy[row_idx]),
                    "validation_fy_loss": float(validation_fy[row_idx]),
                    "train_decision_gap": float(train_gaps[row_idx]),
                    "validation_decision_gap": float(validation_gaps[row_idx]),
                }
            )
        loss_csv = metrics_dir / "e2e_loss_curve.csv"
        write_csv(
            loss_csv,
            rows,
            [
                "epoch",
                "theta_1",
                "theta_2",
                "train_fy_loss",
                "validation_fy_loss",
                "train_decision_gap",
                "validation_decision_gap",
            ],
        )

        gap_checkpoint = select_best_decision_gap_checkpoint(
            trajectory_subset, validation_gaps
        )
        gap_checkpoint["epoch"] = int(eval_indices[gap_checkpoint["epoch"]])
        gap_weights_path = write_model_weights(
            model_weights_dir, gap_checkpoint, args.train_size
        )
        fy_checkpoint = select_best_fy_loss_checkpoint(
            trajectory_subset, validation_fy
        )
        fy_checkpoint["epoch"] = int(eval_indices[fy_checkpoint["epoch"]])
        fy_weights_path = write_model_weights(
            model_weights_dir, fy_checkpoint, args.train_size
        )
        print(
            "Saved e2e model weights "
            f"{gap_weights_path} at epoch {gap_checkpoint['epoch']} "
            f"validation_decision_gap={gap_checkpoint['selection_value']:.6f}"
        )
        print(
            "Saved e2e model weights "
            f"{fy_weights_path} at epoch {fy_checkpoint['epoch']} "
            f"validation_fy_loss={fy_checkpoint['selection_value']:.6f}"
        )

        if args.plot:
            plot_loss_curve(
                loss_csv,
                plots_dir / "e2e_fy_loss.png",
                train_column="train_fy_loss",
                validation_column="validation_fy_loss",
                ylabel="Perturbed FY objective",
                title="End-to-end FY surrogate loss",
            )
    finally:
        common.dispose_graph_records(train_graphs)
        common.dispose_graph_records(validation_graphs)
        env.dispose()


if __name__ == "__main__":
    main()
