#!/usr/bin/env python3
"""Plot Step2b degree bridge LR/SPO+ overlay for a fixed train-size setting."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BRIDGE_JSON = SCRIPT_DIR / "step2b_bridge_results" / "latest_step2b_bridge.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "step2b_bridge_results" / "plots"

METHODS = [
    ("pyepo_lr", "PyEPO LR", "#1f77b4", "o", "-"),
    ("pyepo_spoplus", "PyEPO SPO+", "#ff7f0e", "s", "-"),
    ("step1c_lr", "my LR", "#2ca02c", "^", "--"),
    ("step1c_spoplus", "my SPO+", "#d62728", "D", "--"),
]
METHOD_LABELS = {method: label for method, label, _, _, _ in METHODS}
METRIC_LABELS = {
    "validation_normalized_gap": "Validation Normalized Decision Gap",
    "train_normalized_gap": "Train Normalized Decision Gap",
    "validation_decision_gap": "Validation Decision Gap",
    "train_decision_gap": "Train Decision Gap",
    "validation_normalized_spoplus_loss": "Validation Normalized SPO+ Loss",
    "train_normalized_spoplus_loss": "Train Normalized SPO+ Loss",
    "test_mean_normalized_gap": "Test Mean Normalized Decision Gap",
    "test_mean_decision_gap": "Test Mean Decision Gap",
    "test_median_normalized_gap": "Test Median Normalized Decision Gap",
}
LR_METRIC_KEYS = {
    "validation_normalized_gap": "validation_normalized_gap",
    "train_normalized_gap": "train_normalized_gap",
    "validation_decision_gap": "validation_decision_gap",
    "train_decision_gap": "train_decision_gap",
}
SPOPLUS_METRIC_KEYS = {
    "validation_normalized_gap": "validation_normalized_decision_gap",
    "train_normalized_gap": "train_normalized_decision_gap",
    "validation_decision_gap": "validation_decision_gap",
    "train_decision_gap": "train_decision_gap",
    "validation_normalized_spoplus_loss": "validation_normalized_spoplus_loss",
    "train_normalized_spoplus_loss": "train_normalized_spoplus_loss",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-json", type=Path, default=DEFAULT_BRIDGE_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--source",
        choices=["formal-test-summary", "bridge"],
        default="formal-test-summary",
        help="Use full formal train_size summary or the small correctness bridge.",
    )
    parser.add_argument(
        "--metric",
        choices=sorted(set(LR_METRIC_KEYS) | set(SPOPLUS_METRIC_KEYS) | set(METRIC_LABELS)),
        default="test_mean_normalized_gap",
    )
    parser.add_argument(
        "--spoplus-selection-metric",
        choices=["validation_spoplus_loss", "validation_decision_gap"],
        default="validation_spoplus_loss",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["png", "pdf", "svg"],
        default=["png"],
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser


def read_json(path: Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv_rows(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require_float(row: dict, key: str, source: Path | str) -> float:
    if key not in row:
        raise KeyError("Missing column '{}' in {}".format(key, source))
    return float(row[key])


def final_spoplus_value(degree_payload: dict, source: str, metric: str) -> tuple[float, int]:
    if metric not in SPOPLUS_METRIC_KEYS:
        raise ValueError("Metric '{}' is not available for SPO+ curves".format(metric))
    artifact_key = {
        "pyepo_spoplus": "pyepo_loss_curve_csv",
        "step1c_spoplus": "step1c_loss_curve_csv",
    }[source]
    path = Path(degree_payload["artifacts"][artifact_key])
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError("Empty SPO+ curve CSV: {}".format(path))
    last = rows[-1]
    return require_float(last, SPOPLUS_METRIC_KEYS[metric], path), int(last["epoch"])


def lr_value(degree_payload: dict, source: str, metric: str) -> float:
    if metric not in LR_METRIC_KEYS:
        raise ValueError("Metric '{}' is not available for LR summaries".format(metric))
    summary_key = {
        "pyepo_lr": "pyepo_summary",
        "step1c_lr": "step1c_summary",
    }[source]
    return require_float(
        degree_payload["lr"][summary_key],
        LR_METRIC_KEYS[metric],
        "degree {}".format(degree_payload["degree"]),
    )


def collect_plot_rows(payload: dict, metric: str) -> list[dict]:
    rows = []
    for degree_payload in sorted(payload["degree_results"], key=lambda item: int(item["degree"])):
        degree = int(degree_payload["degree"])
        for method, label, _, _, _ in METHODS:
            if method.endswith("_lr"):
                value = lr_value(degree_payload, method, metric)
                epoch = ""
                source = "lr_summary"
            else:
                value, epoch = final_spoplus_value(degree_payload, method, metric)
                source = "spoplus_final_epoch"
            rows.append(
                {
                    "degree": degree,
                    "method": method,
                    "label": label,
                    "value": value,
                    "metric": metric,
                    "source": source,
                    "epoch": epoch,
                    "source_train_size": int(payload.get("source_train_size", 0)),
                    "bridge_train_size": int(payload.get("train_size", 0)),
                    "validation_size": int(payload.get("validation_size", 0)),
                    "selection_metric": "",
                    "evaluation": "bridge",
                }
            )
    return rows


def find_summary_row(rows: list[dict], method: str, selection_metric: str) -> dict:
    matches = [
        row
        for row in rows
        if row.get("method") == method and row.get("selection_metric") == selection_metric
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected one row for method={} selection_metric={}, found {}".format(
                method,
                selection_metric,
                len(matches),
            )
        )
    return matches[0]


def formal_summary_value(summary_row: dict, metric: str, source: Path) -> float:
    if metric not in {
        "test_mean_normalized_gap",
        "test_mean_decision_gap",
        "test_median_normalized_gap",
    }:
        raise ValueError("Metric '{}' is not available in formal summary".format(metric))
    return require_float(summary_row, metric, source)


def collect_formal_summary_rows(
    payload: dict,
    metric: str,
    spoplus_selection_metric: str,
) -> list[dict]:
    rows = []
    for degree_payload in sorted(payload["degree_results"], key=lambda item: int(item["degree"])):
        degree = int(degree_payload["degree"])
        summary_path = Path(degree_payload["source_run_dir"]) / "metrics" / "test_summary.csv"
        summary_rows = read_csv_rows(summary_path)
        lr_summary = find_summary_row(
            summary_rows,
            method="2stage",
            selection_metric="validation_mse_loss",
        )
        spoplus_summary = find_summary_row(
            summary_rows,
            method="spoplus",
            selection_metric=spoplus_selection_metric,
        )
        method_summaries = {
            "pyepo_lr": lr_summary,
            "pyepo_spoplus": spoplus_summary,
            "step1c_lr": lr_summary,
            "step1c_spoplus": spoplus_summary,
        }
        for method, label, _, _, _ in METHODS:
            summary = method_summaries[method]
            rows.append(
                {
                    "degree": degree,
                    "method": method,
                    "label": label,
                    "value": formal_summary_value(summary, metric, summary_path),
                    "metric": metric,
                    "source": "formal_test_summary_mirrored_by_bridge",
                    "epoch": summary.get("selected_epoch", ""),
                    "source_train_size": int(summary.get("train_size", payload.get("source_train_size", 0))),
                    "bridge_train_size": "",
                    "validation_size": "",
                    "selection_metric": summary.get("selection_metric", ""),
                    "evaluation": "formal_test_summary",
                }
            )
    return rows


def write_plot_table(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "degree",
        "method",
        "label",
        "value",
        "metric",
        "source",
        "epoch",
        "source_train_size",
        "bridge_train_size",
        "validation_size",
        "selection_metric",
        "evaluation",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(rows: list[dict], output_path: Path, metric: str, dpi: int) -> None:
    if not rows:
        raise ValueError("No plot rows to render")

    degrees = sorted({int(row["degree"]) for row in rows})
    degree_positions = {degree: idx for idx, degree in enumerate(degrees)}
    source_train_size = rows[0]["source_train_size"]
    evaluation = rows[0].get("evaluation", "bridge")

    fig, ax = plt.subplots(figsize=(13.5, 5.5))
    offsets = np.linspace(-0.18, 0.18, len(METHODS))
    by_method = {
        method: [row for row in rows if row["method"] == method]
        for method, _, _, _, _ in METHODS
    }
    for method_idx, (method, label, color, marker, linestyle) in enumerate(METHODS):
        method_rows = sorted(by_method[method], key=lambda row: int(row["degree"]))
        xs = [degree_positions[int(row["degree"])] + offsets[method_idx] for row in method_rows]
        ys = [float(row["value"]) for row in method_rows]
        ax.plot(
            xs,
            ys,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.8,
            markersize=6,
            label=label,
        )

    ax.set_xticks(range(len(degrees)))
    ax.set_xticklabels([str(degree) for degree in degrees])
    ax.set_xlabel("Polynomial Degree")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    if evaluation == "formal_test_summary":
        title = "KEP Step2b: train size = {}, formal test summary".format(
            source_train_size
        )
    else:
        title = "KEP Step2b: source train size = {}, bridge train/val = {}/{}".format(
            source_train_size,
            rows[0]["bridge_train_size"],
            rows[0]["validation_size"],
        )
    ax.set_title(title)
    ax.grid(axis="y", color="0.86", linewidth=0.8)
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def output_stem(payload: dict, metric: str) -> str:
    return "step2b_degree_overlay_train_size={}_{}".format(
        int(payload.get("source_train_size", 0)),
        metric,
    )


def run(args: argparse.Namespace) -> dict:
    payload = read_json(args.bridge_json)
    if args.source == "formal-test-summary":
        rows = collect_formal_summary_rows(
            payload,
            metric=args.metric,
            spoplus_selection_metric=args.spoplus_selection_metric,
        )
    else:
        rows = collect_plot_rows(payload, metric=args.metric)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem(payload, args.metric)
    csv_path = args.output_dir / "{}.csv".format(stem)
    write_plot_table(csv_path, rows)

    plot_paths = []
    for fmt in args.formats:
        plot_path = args.output_dir / "{}.{}".format(stem, fmt)
        plot_rows(rows, plot_path, metric=args.metric, dpi=args.dpi)
        plot_paths.append(plot_path)
        print("Saved {}".format(plot_path))
    print("Saved {}".format(csv_path))
    return {"csv": csv_path, "plots": plot_paths, "rows": rows}


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
