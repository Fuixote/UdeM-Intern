"""
Plot Step1 training metrics over epochs.

Inputs:
    trajectory_fy_with_fy_loss_and_regret.npy
        columns: theta_1, theta_2, fy_objective, decision_gap

Output:
    trajectory_epoch_metrics.png
"""

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_array(path, expected_cols, name):
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] < expected_cols:
        raise ValueError(
            f"{name} must have shape (n, >={expected_cols}), got {arr.shape}: {path}"
        )
    return arr


def annotate_endpoint(ax, epochs, values, label, color):
    min_idx = int(np.argmin(values))
    final_idx = len(values) - 1
    ax.scatter(epochs[min_idx], values[min_idx], s=38, color=color, zorder=5)
    ax.scatter(
        epochs[final_idx],
        values[final_idx],
        s=78,
        marker="*",
        color=color,
        edgecolor="white",
        linewidth=0.8,
        zorder=6,
    )
    ax.annotate(
        f"{label} min={values[min_idx]:.4g}",
        xy=(epochs[min_idx], values[min_idx]),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=8,
        color=color,
    )


def style_axis(ax, title, ylabel):
    ax.set_title(title, fontsize=12, pad=8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_epoch_metrics(fy_path, out_path, title=None):
    fy = load_array(fy_path, 4, "FY trajectory")

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fy_epochs = np.arange(fy.shape[0])
    fy_loss = fy[:, 2]
    fy_regret = fy[:, 3]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    ax_loss, ax_regret = axes

    ax_loss.plot(fy_epochs, fy_loss, color="tab:green", linewidth=2.0, label="Perturbed FY objective")
    annotate_endpoint(ax_loss, fy_epochs, fy_loss, "FY obj.", "tab:green")
    style_axis(ax_loss, "(a) Perturbed FY objective vs epoch", "Perturbed FY objective")
    ax_loss.legend(fontsize=8)

    ax_regret.plot(
        fy_epochs,
        fy_regret,
        color="tab:orange",
        linewidth=2.0,
        label="Synthetic-label decision gap",
    )
    annotate_endpoint(ax_regret, fy_epochs, fy_regret, "Decision gap", "tab:orange")
    style_axis(ax_regret, "(b) Synthetic-label decision gap vs epoch", "Decision gap")
    ax_regret.legend(fontsize=8)

    if title:
        fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Loaded FY: {fy.shape} from {fy_path}")
    print(f"Saved: {out_path}")
    plt.close(fig)


def main():
    here = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fy_path",
        default=str(here / "trajectory_fy_with_fy_loss_and_regret.npy"),
    )
    parser.add_argument(
        "--out_path",
        default=str(here / "trajectory_epoch_metrics.png"),
    )
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    plot_epoch_metrics(
        fy_path=args.fy_path,
        out_path=args.out_path,
        title=args.title,
    )


if __name__ == "__main__":
    main()
