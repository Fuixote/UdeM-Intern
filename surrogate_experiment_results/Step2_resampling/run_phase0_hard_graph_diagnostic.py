#!/usr/bin/env python3
"""Run the Step2 resampling Phase 0 hard-graph diagnostic.

This script does not train models. It reuses existing Step2 per-graph
unseen10000 evaluations, selects the graphs with the largest 2stage normalized
gap, and measures the paired SPO+ improvement on those MSE-hard graphs.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STEP2_ROOT = PROJECT_ROOT / "surrogate_experiment_results" / "Step2"
DEFAULT_OUT_ROOT = Path(__file__).resolve().parent
DEFAULT_RUN_TAG = "formal_2stage500_spoplus500_s10"
DEFAULT_BASELINE_LABEL = "2stage_val_mse"
DEFAULT_CANDIDATE_LABEL = "spoplus_val_spoplus_loss"


SUMMARY_FIELDS = [
    "block",
    "regime",
    "degree",
    "train_size",
    "evaluation_dataset",
    "graph_count",
    "top_fraction",
    "top_graph_count",
    "baseline_checkpoint_label",
    "spoplus_checkpoint_label",
    "all_mean_norm_gap_2stage",
    "all_mean_norm_gap_spoplus",
    "all_mean_norm_gap_improvement",
    "all_fraction_graphs_improved",
    "top10_mean_norm_gap_2stage",
    "top10_mean_norm_gap_spoplus",
    "top10_mean_norm_gap_improvement",
    "top10_median_norm_gap_improvement",
    "top10_mean_raw_gap_improvement",
    "top10_fraction_graphs_improved",
    "source_per_graph_csv",
]

TOP_GRAPH_FIELDS = [
    "block",
    "regime",
    "degree",
    "train_size",
    "rank_by_2stage_norm_gap",
    "graph",
    "optimal_obj",
    "2stage_gap",
    "spoplus_gap",
    "raw_gap_improvement",
    "2stage_normalized_gap",
    "spoplus_normalized_gap",
    "normalized_gap_improvement",
    "spoplus_improved",
    "2stage_model_path",
    "spoplus_model_path",
]


@dataclass(frozen=True)
class RegimeSetting:
    block: str
    regime: str
    degree: str
    remote_results_root: Path


def default_settings(step2_root: Path) -> list[RegimeSetting]:
    return [
        RegimeSetting(
            block="Step2b",
            regime="step2b_poly_d8",
            degree="8",
            remote_results_root=(
                step2_root
                / "Step2b_polynomial_degree_noiseless"
                / "remote_results"
                / "step2b_poly_d8"
            ),
        ),
        RegimeSetting(
            block="Step2c",
            regime="step2c_poly_d8_mult_eps050",
            degree="8",
            remote_results_root=(
                step2_root
                / "Step2c_polynomial_degree_multiplicative_noise"
                / "remote_results"
                / "step2c_poly_d8_mult_eps050"
            ),
        ),
    ]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else float("nan")


def median(values) -> float:
    values = sorted(values)
    if not values:
        return float("nan")
    mid = len(values) // 2
    if len(values) % 2:
        return float(values[mid])
    return float((values[mid - 1] + values[mid]) / 2)


def by_checkpoint_label(rows: list[dict[str, str]]) -> dict[str, dict[str, dict[str, str]]]:
    grouped: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["checkpoint_label"], {})[row["graph"]] = row
    return grouped


def compute_diagnostic(
    rows: list[dict[str, str]],
    *,
    top_fraction: float,
    baseline_label: str,
    candidate_label: str,
) -> tuple[dict, list[dict]]:
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1]")

    grouped = by_checkpoint_label(rows)
    if baseline_label not in grouped:
        raise KeyError(f"missing baseline checkpoint_label: {baseline_label}")
    if candidate_label not in grouped:
        raise KeyError(f"missing candidate checkpoint_label: {candidate_label}")

    baseline = grouped[baseline_label]
    candidate = grouped[candidate_label]
    graphs = sorted(set(baseline) & set(candidate))
    if not graphs:
        raise ValueError("no paired graphs found for baseline and candidate")

    ranked = sorted(
        graphs,
        key=lambda graph: float(baseline[graph]["normalized_gap"]),
        reverse=True,
    )
    top_n = max(1, int(round(len(ranked) * top_fraction)))
    top_graphs = ranked[:top_n]

    def values(graph_set, source, field):
        return [float(source[graph][field]) for graph in graph_set]

    all_base_norm = values(graphs, baseline, "normalized_gap")
    all_candidate_norm = values(graphs, candidate, "normalized_gap")
    top_base_norm = values(top_graphs, baseline, "normalized_gap")
    top_candidate_norm = values(top_graphs, candidate, "normalized_gap")
    all_delta_norm = [a - b for a, b in zip(all_base_norm, all_candidate_norm)]
    top_delta_norm = [a - b for a, b in zip(top_base_norm, top_candidate_norm)]
    top_base_raw = values(top_graphs, baseline, "gap")
    top_candidate_raw = values(top_graphs, candidate, "gap")
    top_delta_raw = [a - b for a, b in zip(top_base_raw, top_candidate_raw)]

    summary = {
        "evaluation_dataset": baseline[graphs[0]].get("evaluation_dataset", ""),
        "graph_count": len(graphs),
        "top_fraction": top_fraction,
        "top_graph_count": len(top_graphs),
        "baseline_checkpoint_label": baseline_label,
        "spoplus_checkpoint_label": candidate_label,
        "all_mean_norm_gap_2stage": mean(all_base_norm),
        "all_mean_norm_gap_spoplus": mean(all_candidate_norm),
        "all_mean_norm_gap_improvement": mean(all_delta_norm),
        "all_fraction_graphs_improved": mean(
            a > b for a, b in zip(all_base_norm, all_candidate_norm)
        ),
        "top10_mean_norm_gap_2stage": mean(top_base_norm),
        "top10_mean_norm_gap_spoplus": mean(top_candidate_norm),
        "top10_mean_norm_gap_improvement": mean(top_delta_norm),
        "top10_median_norm_gap_improvement": median(top_delta_norm),
        "top10_mean_raw_gap_improvement": mean(top_delta_raw),
        "top10_fraction_graphs_improved": mean(
            a > b for a, b in zip(top_base_raw, top_candidate_raw)
        ),
    }

    top_rows = []
    for rank, graph in enumerate(top_graphs, start=1):
        base = baseline[graph]
        cand = candidate[graph]
        top_rows.append(
            {
                "rank_by_2stage_norm_gap": rank,
                "graph": graph,
                "optimal_obj": base.get("optimal_obj", ""),
                "2stage_gap": base["gap"],
                "spoplus_gap": cand["gap"],
                "raw_gap_improvement": float(base["gap"]) - float(cand["gap"]),
                "2stage_normalized_gap": base["normalized_gap"],
                "spoplus_normalized_gap": cand["normalized_gap"],
                "normalized_gap_improvement": (
                    float(base["normalized_gap"]) - float(cand["normalized_gap"])
                ),
                "spoplus_improved": float(base["gap"]) > float(cand["gap"]),
                "2stage_model_path": base.get("model_path", ""),
                "spoplus_model_path": cand.get("model_path", ""),
            }
        )
    return summary, top_rows


def per_graph_path(setting: RegimeSetting, train_size: int, run_tag: str) -> Path:
    return (
        setting.remote_results_root
        / "step1c_spoplus"
        / run_tag
        / f"train_size={train_size}"
        / "metrics"
        / "unseen10000_per_graph.csv"
    )


def enrich_summary(
    summary: dict,
    *,
    setting: RegimeSetting,
    train_size: int,
    source_per_graph_csv: Path,
) -> dict:
    return {
        "block": setting.block,
        "regime": setting.regime,
        "degree": setting.degree,
        "train_size": int(train_size),
        **summary,
        "source_per_graph_csv": str(source_per_graph_csv),
    }


def enrich_top_rows(
    rows: list[dict],
    *,
    setting: RegimeSetting,
    train_size: int,
) -> list[dict]:
    return [
        {
            "block": setting.block,
            "regime": setting.regime,
            "degree": setting.degree,
            "train_size": int(train_size),
            **row,
        }
        for row in rows
    ]


def write_plot(summary_rows: list[dict], plot_path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    labels = [
        f"{row['block']} d{row['degree']}\nn={row['train_size']}"
        for row in summary_rows
    ]
    all_improvements = [
        float(row["all_mean_norm_gap_improvement"]) for row in summary_rows
    ]
    top_improvements = [
        float(row["top10_mean_norm_gap_improvement"]) for row in summary_rows
    ]
    x_positions = list(range(len(labels)))
    width = 0.36

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(
        [x - width / 2 for x in x_positions],
        all_improvements,
        width,
        label="All unseen10000 graphs",
        color="#4c78a8",
    )
    ax.bar(
        [x + width / 2 for x in x_positions],
        top_improvements,
        width,
        label="Top 10% MSE-hard graphs",
        color="#f58518",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean normalized-gap improvement\n2stage MSE - SPO+")
    ax.set_title("Phase 0 hard-graph diagnostic")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return True


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Compute the Step2_resampling Phase 0 hard-graph diagnostic from "
            "existing unseen10000 per-graph CSV files."
        )
    )
    parser.add_argument("--step2_root", default=str(DEFAULT_STEP2_ROOT))
    parser.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--train_sizes", nargs="+", type=int, default=[50, 1200])
    parser.add_argument("--top_fraction", type=float, default=0.10)
    parser.add_argument("--run_tag", default=DEFAULT_RUN_TAG)
    parser.add_argument("--baseline_label", default=DEFAULT_BASELINE_LABEL)
    parser.add_argument("--candidate_label", default=DEFAULT_CANDIDATE_LABEL)
    parser.add_argument(
        "--regimes",
        nargs="+",
        default=None,
        help="Optional regime names. Defaults to Step2b d8 and Step2c d8 eps050.",
    )
    parser.add_argument("--skip_plot", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    step2_root = Path(args.step2_root)
    out_root = Path(args.out_root)
    settings = default_settings(step2_root)
    if args.regimes is not None:
        wanted = set(args.regimes)
        settings = [setting for setting in settings if setting.regime in wanted]
        missing = sorted(wanted - {setting.regime for setting in settings})
        if missing:
            raise ValueError(f"Unknown regimes requested: {', '.join(missing)}")
    if not settings:
        raise ValueError("No regimes selected")

    summary_rows: list[dict] = []
    top_rows: list[dict] = []
    for setting in settings:
        for train_size in args.train_sizes:
            source_path = per_graph_path(setting, train_size, args.run_tag)
            if not source_path.is_file():
                raise FileNotFoundError(f"Missing per-graph CSV: {source_path}")
            summary, top = compute_diagnostic(
                read_csv(source_path),
                top_fraction=args.top_fraction,
                baseline_label=args.baseline_label,
                candidate_label=args.candidate_label,
            )
            summary_rows.append(
                enrich_summary(
                    summary,
                    setting=setting,
                    train_size=train_size,
                    source_per_graph_csv=source_path,
                )
            )
            top_rows.extend(enrich_top_rows(top, setting=setting, train_size=train_size))

    results_dir = out_root / "results"
    plot_dir = out_root / "plot_results"
    summary_path = results_dir / "phase0_hard_graph_diagnostic_summary.csv"
    top_path = results_dir / "phase0_top_decile_graphs.csv"
    plot_path = plot_dir / "phase0_top_decile_improvement.png"
    write_csv(summary_path, summary_rows, SUMMARY_FIELDS)
    write_csv(top_path, top_rows, TOP_GRAPH_FIELDS)
    print(f"wrote {summary_path}")
    print(f"wrote {top_path}")
    if not args.skip_plot:
        if write_plot(summary_rows, plot_path):
            print(f"wrote {plot_path}")
        else:
            print(
                "warning: matplotlib is not available; skipped plot generation",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
