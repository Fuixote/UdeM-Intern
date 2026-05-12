"""Plot Step1b train/validation loss curves from a metrics CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def load_rows(csv_path):
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_loss_curve(
    csv_path,
    out_path,
    train_column,
    validation_column,
    ylabel,
    title,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load_rows(csv_path)
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    epochs = [int(row["epoch"]) for row in rows]
    train_values = [float(row[train_column]) for row in rows]
    validation_values = [float(row[validation_column]) for row in rows]

    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    ax.plot(epochs, train_values, label="train", marker="o", linewidth=1.8)
    ax.plot(
        epochs,
        validation_values,
        label="validation",
        marker="s",
        linewidth=1.8,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Plot a Step1b loss curve CSV.")
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--train_column", required=True)
    parser.add_argument("--validation_column", required=True)
    parser.add_argument("--ylabel", required=True)
    parser.add_argument("--title", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    written = plot_loss_curve(
        args.csv_path,
        args.out_path,
        train_column=args.train_column,
        validation_column=args.validation_column,
        ylabel=args.ylabel,
        title=args.title,
    )
    print(f"Saved loss curve to {written}")


if __name__ == "__main__":
    main()
