"""Train the Step1c two-stage MSE reward-fitting model."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from split_dataset import graph_entries_from_data_dir, read_json, select_train_subset
from plot_training_curves import plot_loss_curve


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STEP1A_PATH = PROJECT_ROOT / "surrogate_experiment_results" / "Step1a" / "Step1.py"
DEFAULT_SPLIT_PATH = PROJECT_ROOT / "results" / "step1b_splits" / "master_split_seed=42.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "step1c_runs" / "train_size=50"

PROBE = {
    "feature_names": ["utility", "recipient_cPRA"],
    "feature_labels": [r"$\theta_1$ (utility weight)", r"$\theta_2$ (cPRA weight)"],
}
MODEL_FORMULA = "w_hat_e = theta_1 * utility_e + theta_2 * recipient_cPRA_e"


def load_step1a_module():
    spec = importlib.util.spec_from_file_location("step1a_training", STEP1A_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_reward_records(entries):
    step1a = load_step1a_module()
    records = []
    for idx, entry in enumerate(entries, start=1):
        graph = step1a.load_graph(entry["path"])
        records.append(
            {
                "path": entry["path"],
                "filename": Path(entry["path"]).name,
                "index": entry["index"],
                "X": step1a.feature_matrix(graph, PROBE),
                "w_true": graph["w_true"],
            }
        )
        print(f"  loaded {idx}/{len(entries)} {Path(entry['path']).name}", flush=True)
    return records


def grad_mse(graphs, theta):
    grad = np.zeros(2)
    for record in graphs:
        grad += record["X"].T @ (record["X"] @ theta - record["w_true"])
    return grad / len(graphs)


class Adam:
    def __init__(self, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.m = np.zeros(2)
        self.v = np.zeros(2)
        self.t = 0

    def step(self, theta, grad):
        self.t += 1
        self.m = self.b1 * self.m + (1.0 - self.b1) * grad
        self.v = self.b2 * self.v + (1.0 - self.b2) * (grad ** 2)
        m_hat = self.m / (1.0 - self.b1 ** self.t)
        v_hat = self.v / (1.0 - self.b2 ** self.t)
        return theta - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def run_mse_trajectory(graphs, theta_init, n_epochs, lr):
    opt = Adam(lr)
    theta = np.asarray(theta_init, dtype=float).copy()
    trajectory = [theta.copy()]
    for epoch in range(n_epochs):
        theta = opt.step(theta, grad_mse(graphs, theta))
        trajectory.append(theta.copy())
        print(f"  [2stage/MSE] epoch {epoch + 1:>4} theta={np.round(theta, 4)}")
    return np.asarray(trajectory, dtype=float)


def mse_loss(theta, graphs):
    theta = np.asarray(theta, dtype=float)
    losses = [
        float(np.mean((record["X"] @ theta - record["w_true"]) ** 2))
        for record in graphs
    ]
    return float(np.mean(losses))


def evaluate_mse_losses(trajectory, graphs):
    return np.asarray([mse_loss(theta, graphs) for theta in trajectory], dtype=float)


def select_best_mse_checkpoint(trajectory, validation_mse_loss):
    validation_mse_loss = np.asarray(validation_mse_loss, dtype=float)
    best_idx = int(np.nanargmin(validation_mse_loss))
    return {
        "method": "2stage",
        "epoch": best_idx,
        "theta": np.asarray(trajectory[best_idx], dtype=float).copy(),
        "selection_metric": "validation_mse_loss",
        "selection_value": float(validation_mse_loss[best_idx]),
    }


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
    stem = "2stage_best_by_validation_mse_loss"
    npz_path = out_dir / f"{stem}.npz"
    np.savez_compressed(
        npz_path,
        theta=theta,
        method=np.asarray("2stage"),
        model_type=np.asarray("linear_probe"),
        model_formula=np.asarray(MODEL_FORMULA),
        feature_names=np.asarray(PROBE["feature_names"]),
        train_size=np.asarray(int(train_size)),
        selected_epoch=np.asarray(int(checkpoint["epoch"])),
        selection_metric=np.asarray(checkpoint["selection_metric"]),
        selection_value=np.asarray(float(checkpoint["selection_value"])),
    )
    write_json(
        out_dir / f"{stem}.json",
        {
            "method": "2stage",
            "model_type": "linear_probe",
            "model_formula": MODEL_FORMULA,
            "feature_names": PROBE["feature_names"],
            "train_size": int(train_size),
            "selected_epoch": int(checkpoint["epoch"]),
            "selection_metric": checkpoint["selection_metric"],
            "selection_value": float(checkpoint["selection_value"]),
            "theta": theta.tolist(),
        },
    )
    return npz_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train Step1c 2stage baseline.")
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
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
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

    write_json(out_dir / "train_subset.json", train_entries)
    write_json(out_dir / "validation_set.json", validation_entries)
    print(f"Loading 2stage train subset: n={len(train_entries)}")
    train_graphs = load_reward_records(train_entries)
    validation_source = args.validation_data_dir or "split validation"
    print(f"Loading 2stage validation set: n={len(validation_entries)} source={validation_source}")
    validation_graphs = load_reward_records(validation_entries)

    rng = np.random.RandomState(args.theta_seed)
    theta_init = (
        np.asarray(args.theta_init, dtype=float)
        if args.theta_init is not None
        else rng.uniform(0.5, 3.5, size=2)
    )
    print(f"2stage theta_init={np.round(theta_init, 4)}")

    trajectory = run_mse_trajectory(train_graphs, theta_init, args.n_epochs, args.lr)
    np.save(trajectories_dir / "trajectory_2stage.npy", trajectory)

    train_mse = evaluate_mse_losses(trajectory, train_graphs)
    validation_mse = evaluate_mse_losses(trajectory, validation_graphs)
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
    write_csv(
        loss_csv,
        rows,
        ["epoch", "theta_1", "theta_2", "train_mse_loss", "validation_mse_loss"],
    )

    checkpoint = select_best_mse_checkpoint(trajectory, validation_mse)
    weights_path = write_model_weights(model_weights_dir, checkpoint, args.train_size)
    print(
        "Saved 2stage model weights "
        f"{weights_path} at epoch {checkpoint['epoch']} "
        f"validation_mse_loss={checkpoint['selection_value']:.6f}"
    )

    if args.plot:
        plot_loss_curve(
            loss_csv,
            plots_dir / "2stage_mse_loss.png",
            train_column="train_mse_loss",
            validation_column="validation_mse_loss",
            ylabel="MSE loss",
            title="2stage reward-fitting MSE loss",
        )


if __name__ == "__main__":
    main()
