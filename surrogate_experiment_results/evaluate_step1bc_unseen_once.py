"""Evaluate Step1b/Step1c model archives on one unseen dataset in one pass.

The older per-run evaluation scripts load the unseen graph dataset once per run
directory. This script loads the graph records once, computes the oracle
solutions once, evaluates all discovered model checkpoints, and then writes the
usual ``<stem>_summary`` and ``<stem>_per_graph`` files back into each run
directory.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
STEP1B_DIR = SCRIPT_DIR / "Step1b"

if str(STEP1B_DIR) not in sys.path:
    sys.path.insert(0, str(STEP1B_DIR))

import step1b_common as common  # noqa: E402
from evaluate_models import load_model_weight, summarize_evaluated_models, write_csv  # noqa: E402


DEFAULT_DATASET_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "step1_noisy_linear_sigma010_unseen_test10000_seed20260520"
)
DEFAULT_STEP1B_ROOT = (
    SCRIPT_DIR
    / "Step1b"
    / "remote_results"
    / "formal_M16_2stage500_e2e500_s10_val2000"
)
DEFAULT_STEP1C_ROOT = (
    SCRIPT_DIR
    / "Step1c"
    / "remote_results"
    / "formal_spoplus_ablation_val2000"
)
DEFAULT_TRAIN_SIZES = [50, 200, 600, 1200]

STEP1B_WEIGHT_FILES = (
    "2stage_best_by_validation_mse_loss.npz",
    "e2e_best_by_validation_decision_gap.npz",
    "e2e_best_by_validation_fy_loss.npz",
)
STEP1C_WEIGHT_FILES = (
    "2stage_best_by_validation_mse_loss.npz",
    "spoplus_best_by_validation_decision_gap.npz",
    "spoplus_best_by_validation_spoplus_loss.npz",
)

SUMMARY_FIELDS = [
    "evaluation_dataset",
    "evaluation_graph_count",
    "source",
    "method",
    "train_size",
    "selected_epoch",
    "theta_1",
    "theta_2",
    "selection_metric",
    "selection_value",
    "test_mean_decision_gap",
    "test_mean_normalized_gap",
    "test_median_normalized_gap",
    "test_mean_achieved_oracle_ratio",
    "paired_mean_improvement_over_2stage",
    "paired_median_improvement_over_2stage",
    "fraction_improved_over_2stage",
    "paired_mean_improvement_ci_low",
    "paired_mean_improvement_ci_high",
    "model_path",
]


@dataclass(frozen=True)
class RunSpec:
    source: str
    train_size: int
    run_dir: Path
    weights: tuple[Path, ...]


def graph_sort_key(path: Path):
    match = re.fullmatch(r"G-(\d+)\.json", path.name)
    if match:
        return (0, int(match.group(1)))
    return (1, path.name)


def list_graph_paths(dataset_dir: Path, graph_limit: int | None = None) -> list[Path]:
    paths = sorted(dataset_dir.glob("G-*.json"), key=graph_sort_key)
    if graph_limit is not None:
        paths = paths[:graph_limit]
    return paths


def complete_run_spec(
    source: str,
    root: Path,
    train_size: int,
    expected_weight_files: tuple[str, ...],
) -> tuple[RunSpec | None, str | None]:
    run_dir = root / f"train_size={train_size}"
    weights_dir = run_dir / "model_weights"
    weights = tuple(weights_dir / filename for filename in expected_weight_files)
    missing = [path for path in weights if not path.exists()]
    if missing:
        return None, (
            f"Skipping {source} train_size={train_size}: missing "
            + ", ".join(str(path) for path in missing)
        )
    return RunSpec(source=source, train_size=train_size, run_dir=run_dir, weights=weights), None


def discover_run_specs(
    step1b_root: Path,
    step1c_root: Path,
    train_sizes: list[int],
    include_step1b: bool = True,
    include_step1c: bool = True,
) -> tuple[list[RunSpec], list[str]]:
    specs: list[RunSpec] = []
    warnings: list[str] = []

    for train_size in train_sizes:
        if include_step1b:
            spec, warning = complete_run_spec(
                "step1b",
                step1b_root,
                train_size,
                STEP1B_WEIGHT_FILES,
            )
            if spec is None:
                warnings.append(warning or f"Skipping step1b train_size={train_size}")
            else:
                specs.append(spec)

        if include_step1c:
            spec, warning = complete_run_spec(
                "step1c",
                step1c_root,
                train_size,
                STEP1C_WEIGHT_FILES,
            )
            if spec is None:
                warnings.append(warning or f"Skipping step1c train_size={train_size}")
            else:
                specs.append(spec)

    return specs, warnings


def summary_rows_with_context(
    rows: list[dict],
    *,
    source: str,
    dataset_dir: Path,
    graph_count: int,
) -> list[dict]:
    return [
        {
            "evaluation_dataset": str(dataset_dir),
            "evaluation_graph_count": graph_count,
            "source": source,
            **row,
        }
        for row in rows
    ]


def per_graph_rows_for_model(
    *,
    model: dict,
    evaluations: list[dict],
    dataset_dir: Path,
    graph_count: int,
) -> list[dict]:
    rows = []
    for row in evaluations:
        rows.append(
            {
                "evaluation_dataset": str(dataset_dir),
                "evaluation_graph_count": graph_count,
                "method": model["method"],
                "train_size": model["train_size"],
                "selected_epoch": model["selected_epoch"],
                "selection_metric": model["selection_metric"],
                "selection_value": model["selection_value"],
                "model_path": model["path"],
                **row,
            }
        )
    return rows


def write_run_config(
    path: Path,
    *,
    script_args: argparse.Namespace,
    spec: RunSpec,
    dataset_dir: Path,
    graph_count: int,
) -> None:
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": "surrogate_experiment_results/evaluate_step1bc_unseen_once.py",
        "source": spec.source,
        "run_dir": str(spec.run_dir),
        "dataset_dir": str(dataset_dir),
        "graph_count": graph_count,
        "graph_limit": script_args.graph_limit,
        "weights": [str(path) for path in spec.weights],
        "output_stem": script_args.output_stem,
        "gurobi_seed": script_args.gurobi_seed,
        "bootstrap_samples": script_args.bootstrap_samples,
        "bootstrap_seed": script_args.bootstrap_seed,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def evaluate_run_spec(
    *,
    spec: RunSpec,
    graphs: list[dict],
    env,
    dataset_dir: Path,
    graph_count: int,
    output_stem: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
    script_args: argparse.Namespace,
) -> None:
    print(
        f"Evaluating {spec.source} train_size={spec.train_size}: "
        f"{len(spec.weights)} checkpoints",
        flush=True,
    )
    models_and_evaluations = []
    for weight_path in spec.weights:
        model = load_model_weight(weight_path)
        print(
            "  checkpoint "
            f"{model['method']} selected_by={model['selection_metric']} "
            f"epoch={model['selected_epoch']} theta={model['theta']}",
            flush=True,
        )
        evaluations = common.evaluate_theta(model["theta"], graphs, env)
        models_and_evaluations.append((model, evaluations))

    summary_rows = summarize_evaluated_models(
        models_and_evaluations,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    summary_rows = summary_rows_with_context(
        summary_rows,
        source=spec.source,
        dataset_dir=dataset_dir,
        graph_count=graph_count,
    )

    per_graph_rows = []
    for model, evaluations in models_and_evaluations:
        per_graph_rows.extend(
            per_graph_rows_for_model(
                model=model,
                evaluations=evaluations,
                dataset_dir=dataset_dir,
                graph_count=graph_count,
            )
        )

    metrics_dir = spec.run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    write_run_config(
        metrics_dir / f"{output_stem}_run_config.json",
        script_args=script_args,
        spec=spec,
        dataset_dir=dataset_dir,
        graph_count=graph_count,
    )
    write_csv(metrics_dir / f"{output_stem}_summary.csv", summary_rows, SUMMARY_FIELDS)
    write_csv(metrics_dir / f"{output_stem}_per_graph.csv", per_graph_rows)
    (metrics_dir / f"{output_stem}_summary.json").write_text(
        json.dumps(summary_rows, indent=2),
        encoding="utf-8",
    )
    print(f"  saved {metrics_dir / f'{output_stem}_summary.csv'}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Load one unseen processed dataset once, evaluate discovered Step1b/Step1c "
            "model checkpoints, and write per-run unseen metrics."
        )
    )
    parser.add_argument("--dataset_dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--step1b_root", default=str(DEFAULT_STEP1B_ROOT))
    parser.add_argument("--step1c_root", default=str(DEFAULT_STEP1C_ROOT))
    parser.add_argument(
        "--train_sizes",
        nargs="+",
        type=int,
        default=DEFAULT_TRAIN_SIZES,
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["step1b", "step1c"],
        default=["step1b", "step1c"],
    )
    parser.add_argument("--output_stem", default="unseen10000")
    parser.add_argument("--graph_limit", type=int, default=None)
    parser.add_argument("--gurobi_seed", type=int, default=42)
    parser.add_argument("--bootstrap_samples", type=int, default=1000)
    parser.add_argument("--bootstrap_seed", type=int, default=42)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only list complete run directories that would be evaluated.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    dataset_dir = Path(args.dataset_dir)
    step1b_root = Path(args.step1b_root)
    step1c_root = Path(args.step1c_root)

    specs, warnings = discover_run_specs(
        step1b_root=step1b_root,
        step1c_root=step1c_root,
        train_sizes=list(args.train_sizes),
        include_step1b="step1b" in args.sources,
        include_step1c="step1c" in args.sources,
    )
    for warning in warnings:
        print(f"warning: {warning}", flush=True)

    if not specs:
        raise FileNotFoundError("No complete Step1b/Step1c run directories found.")

    print("Complete run directories:", flush=True)
    for spec in specs:
        print(f"  {spec.source} train_size={spec.train_size}: {spec.run_dir}", flush=True)

    if args.dry_run:
        return 0

    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
    graph_paths = list_graph_paths(dataset_dir, args.graph_limit)
    if not graph_paths:
        raise FileNotFoundError(f"No G-*.json files found under {dataset_dir}")

    import gurobipy as gp

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Seed", args.gurobi_seed)
    env.start()

    graphs = []
    try:
        print(f"Loading unseen graphs once: n={len(graph_paths)}", flush=True)
        graphs = common.load_graph_records(graph_paths, env)
        for spec in specs:
            evaluate_run_spec(
                spec=spec,
                graphs=graphs,
                env=env,
                dataset_dir=dataset_dir,
                graph_count=len(graph_paths),
                output_stem=args.output_stem,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
                script_args=args,
            )
    finally:
        common.dispose_graph_records(graphs)
        env.dispose()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
