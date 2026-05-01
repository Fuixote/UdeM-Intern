"""
Plot Step1 trajectories as three 3D curves in one figure.

Inputs:
    trajectory_mse_with_regret.npy
        columns: theta_1, theta_2, true_regret
    trajectory_fy_with_fy_loss_and_regret.npy
        columns: theta_1, theta_2, fy_loss, true_regret

Output:
    trajectory_3d_metrics.png
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


def draw_3d_curve(ax, theta_1, theta_2, metric, title, z_label, cmap_name):
    cmap = plt.get_cmap(cmap_name)
    n = len(theta_1)

    for i in range(n - 1):
        color = cmap(0.2 + 0.8 * i / max(n - 2, 1))
        ax.plot(
            theta_1[i : i + 2],
            theta_2[i : i + 2],
            metric[i : i + 2],
            color=color,
            lw=1.8,
        )

    ax.scatter(theta_1[0], theta_2[0], metric[0], s=45, color=cmap(0.2), label="start")
    ax.scatter(
        theta_1[-1],
        theta_2[-1],
        metric[-1],
        s=90,
        marker="*",
        color=cmap(1.0),
        label="final",
    )

    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel(r"$\theta_1$", labelpad=7)
    ax.set_ylabel(r"$\theta_2$", labelpad=7)
    ax.set_zlabel(z_label, labelpad=7)
    ax.view_init(elev=24, azim=-56)
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8, loc="upper left")


def main():
    here = Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mse_path",
        default=str(here / "trajectory_mse_with_regret.npy"),
    )
    parser.add_argument(
        "--fy_path",
        default=str(here / "trajectory_fy_with_fy_loss_and_regret.npy"),
    )
    parser.add_argument(
        "--out_path",
        default=str(here / "trajectory_3d_metrics.png"),
    )
    args = parser.parse_args()

    mse = load_array(args.mse_path, 3, "MSE trajectory")
    fy = load_array(args.fy_path, 4, "FY trajectory")
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    fig = plt.figure(figsize=(18, 6.2))
    ax_mse_regret = fig.add_subplot(1, 3, 1, projection="3d")
    ax_fy_regret = fig.add_subplot(1, 3, 2, projection="3d")
    ax_fy_loss = fig.add_subplot(1, 3, 3, projection="3d")

    draw_3d_curve(
        ax_mse_regret,
        mse[:, 0],
        mse[:, 1],
        mse[:, 2],
        "2-stage MSE trajectory",
        "True Regret",
        "Blues_r",
    )
    draw_3d_curve(
        ax_fy_regret,
        fy[:, 0],
        fy[:, 1],
        fy[:, 3],
        "End-to-end FY trajectory",
        "True Regret",
        "Oranges_r",
    )
    draw_3d_curve(
        ax_fy_loss,
        fy[:, 0],
        fy[:, 1],
        fy[:, 2],
        "End-to-end FY trajectory",
        "FY Loss",
        "Greens_r",
    )

    fig.suptitle(
        r"Step1 trajectory metrics  ·  $\hat{w}_e = \theta_1 u_e + \theta_2 c_e$",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(args.out_path, dpi=180, bbox_inches="tight")
    print(f"Loaded MSE: {mse.shape} from {args.mse_path}")
    print(f"Loaded FY: {fy.shape} from {args.fy_path}")
    print(f"Saved: {args.out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
