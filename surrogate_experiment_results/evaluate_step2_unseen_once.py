"""Evaluate Step2 FY/SPO+ archives on matching unseen10000 datasets.

This is the Step2 analogue of ``evaluate_step1bc_unseen_once.py``.  It groups
trained runs by label regime, loads the matching unseen10000 dataset once,
evaluates all train sizes and checkpoint rules for that regime, and writes the
usual ``<stem>_summary`` / ``<stem>_per_graph`` files back under each run's
``metrics/`` directory.

The default plan evaluates 9 regimes x 4 train sizes x 5 checkpoints:

* 2stage selected by validation MSE
* FY selected by validation decision gap
* FY selected by validation FY loss
* SPO+ selected by validation decision gap
* SPO+ selected by validation SPO+ loss
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
STEP2_DIR = SCRIPT_DIR / "Step2"

if str(STEP1B_DIR) not in sys.path:
    sys.path.insert(0, str(STEP1B_DIR))

import step1b_common as common  # noqa: E402
from evaluate_models import load_model_weight, summarize_evaluated_models, write_csv  # noqa: E402


DEFAULT_TRAIN_SIZES = [50, 200, 600, 1200]
DEFAULT_FY_RUN_TAG = "formal_2stage500_fyM16_e500_s10"
DEFAULT_SPOPLUS_RUN_TAG = "formal_2stage500_spoplus500_s10"

FY_WEIGHT_FILES = (
    "2stage_best_by_validation_mse_loss.npz",
    "e2e_best_by_validation_decision_gap.npz",
    "e2e_best_by_validation_fy_loss.npz",
)
SPOPLUS_WEIGHT_FILES = (
    "2stage_best_by_validation_mse_loss.npz",
    "spoplus_best_by_validation_decision_gap.npz",
    "spoplus_best_by_validation_spoplus_loss.npz",
)

CHECKPOINT_LABELS = {
    "2stage_best_by_validation_mse_loss.npz": "2stage_val_mse",
    "e2e_best_by_validation_decision_gap.npz": "fy_val_decision_gap",
    "e2e_best_by_validation_fy_loss.npz": "fy_val_fy_loss",
    "spoplus_best_by_validation_decision_gap.npz": "spoplus_val_decision_gap",
    "spoplus_best_by_validation_spoplus_loss.npz": "spoplus_val_spoplus_loss",
}

SUMMARY_FIELDS = [
    "evaluation_dataset",
    "evaluation_graph_count",
    "block",
    "regime",
    "degree",
    "checkpoint_label",
    "checkpoint_source",
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
class RegimeSpec:
    block: str
    regime: str
    degree: str
    dataset_dir: Path
    remote_results_root: Path


@dataclass(frozen=True)
class WeightSpec:
    label: str
    source: str
    path: Path


@dataclass(frozen=True)
class SettingSpec:
    regime_spec: RegimeSpec
    train_size: int
    fy_run_dir: Path
    spoplus_run_dir: Path
    weights: tuple[WeightSpec, ...]

    @property
    def block(self) -> str:
        return self.regime_spec.block

    @property
    def regime(self) -> str:
        return self.regime_spec.regime

    @property
    def degree(self) -> str:
        return self.regime_spec.degree


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


def default_regime_specs() -> list[RegimeSpec]:
    processed = PROJECT_ROOT / "dataset" / "processed"
    return [
        RegimeSpec(
            block="Step2a",
            regime="step2a_additive_rho050",
            degree="-",
            dataset_dir=processed / "step2a_additive_rho050_unseen10000_seed20260523",
            remote_results_root=(
                STEP2_DIR
                / "Step2a_additive_linear_gaussian"
                / "remote_results"
                / "step2a_additive_rho050"
            ),
        ),
        *[
            RegimeSpec(
                block="Step2b",
                regime=f"step2b_poly_d{degree}",
                degree=str(degree),
                dataset_dir=processed / f"step2b_poly_d{degree}_unseen10000_seed20260523",
                remote_results_root=(
                    STEP2_DIR
                    / "Step2b_polynomial_degree_noiseless"
                    / "remote_results"
                    / f"step2b_poly_d{degree}"
                ),
            )
            for degree in (1, 2, 4, 8)
        ],
        *[
            RegimeSpec(
                block="Step2c",
                regime=f"step2c_poly_d{degree}_mult_eps050",
                degree=str(degree),
                dataset_dir=(
                    processed
                    / f"step2c_poly_d{degree}_mult_eps050_unseen10000_seed20260523"
                ),
                remote_results_root=(
                    STEP2_DIR
                    / "Step2c_polynomial_degree_multiplicative_noise"
                    / "remote_results"
                    / f"step2c_poly_d{degree}_mult_eps050"
                ),
            )
            for degree in (1, 2, 4, 8)
        ],
    ]


def build_setting_spec(regime_spec: RegimeSpec, train_size: int) -> tuple[SettingSpec | None, str | None]:
    fy_run_dir = (
        regime_spec.remote_results_root
        / "step1b_fy"
        / DEFAULT_FY_RUN_TAG
        / f"train_size={train_size}"
    )
    spoplus_run_dir = (
        regime_spec.remote_results_root
        / "step1c_spoplus"
        / DEFAULT_SPOPLUS_RUN_TAG
        / f"train_size={train_size}"
    )

    # Use the FY-run 2stage checkpoint as the canonical baseline, then evaluate
    # the two FY checkpoints and the two SPO+ checkpoints.  The duplicated
    # 2stage checkpoint in the SPO+ run is intentionally not evaluated again.
    weight_specs = [
        WeightSpec(
            label=CHECKPOINT_LABELS["2stage_best_by_validation_mse_loss.npz"],
            source="2stage",
            path=fy_run_dir / "model_weights" / "2stage_best_by_validation_mse_loss.npz",
        ),
        *[
            WeightSpec(
                label=CHECKPOINT_LABELS[filename],
                source="step1b_fy",
                path=fy_run_dir / "model_weights" / filename,
            )
            for filename in FY_WEIGHT_FILES[1:]
        ],
        *[
            WeightSpec(
                label=CHECKPOINT_LABELS[filename],
                source="step1c_spoplus",
                path=spoplus_run_dir / "model_weights" / filename,
            )
            for filename in SPOPLUS_WEIGHT_FILES[1:]
        ],
    ]
    missing = [str(weight.path) for weight in weight_specs if not weight.path.exists()]
    if missing:
        return None, (
            f"Skipping {regime_spec.regime} train_size={train_size}: missing "
            + ", ".join(missing)
        )
    return (
        SettingSpec(
            regime_spec=regime_spec,
            train_size=train_size,
            fy_run_dir=fy_run_dir,
            spoplus_run_dir=spoplus_run_dir,
            weights=tuple(weight_specs),
        ),
        None,
    )


def discover_setting_specs(
    regime_specs: list[RegimeSpec],
    train_sizes: list[int],
) -> tuple[list[SettingSpec], list[str]]:
    settings: list[SettingSpec] = []
    warnings: list[str] = []
    for regime_spec in regime_specs:
        for train_size in train_sizes:
            setting, warning = build_setting_spec(regime_spec, train_size)
            if setting is None:
                warnings.append(warning or f"Skipping {regime_spec.regime} train_size={train_size}")
            else:
                settings.append(setting)
    return settings, warnings


def enrich_summary_rows(
    rows: list[dict],
    *,
    setting: SettingSpec,
    models: list[dict],
    dataset_dir: Path,
    graph_count: int,
) -> list[dict]:
    enriched = []
    for row, model in zip(rows, models):
        enriched.append(
            {
                "evaluation_dataset": str(dataset_dir),
                "evaluation_graph_count": graph_count,
                "block": setting.block,
                "regime": setting.regime,
                "degree": setting.degree,
                "checkpoint_label": model["checkpoint_label"],
                "checkpoint_source": model["checkpoint_source"],
                **row,
            }
        )
    return enriched


def per_graph_rows_for_model(
    *,
    setting: SettingSpec,
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
                "block": setting.block,
                "regime": setting.regime,
                "degree": setting.degree,
                "checkpoint_label": model["checkpoint_label"],
                "checkpoint_source": model["checkpoint_source"],
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


def rows_for_run_dir(rows: list[dict], source: str) -> list[dict]:
    if source == "step1b_fy":
        labels = {"2stage_val_mse", "fy_val_decision_gap", "fy_val_fy_loss"}
    elif source == "step1c_spoplus":
        labels = {
            "2stage_val_mse",
            "spoplus_val_decision_gap",
            "spoplus_val_spoplus_loss",
        }
    else:
        raise ValueError(f"Unknown source: {source}")
    return [row for row in rows if row["checkpoint_label"] in labels]


def write_run_config(
    path: Path,
    *,
    args: argparse.Namespace,
    setting: SettingSpec,
    dataset_dir: Path,
    graph_count: int,
) -> None:
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": "surrogate_experiment_results/evaluate_step2_unseen_once.py",
        "block": setting.block,
        "regime": setting.regime,
        "degree": setting.degree,
        "train_size": setting.train_size,
        "fy_run_dir": str(setting.fy_run_dir),
        "spoplus_run_dir": str(setting.spoplus_run_dir),
        "dataset_dir": str(dataset_dir),
        "graph_count": graph_count,
        "graph_limit": args.graph_limit,
        "weights": [str(weight.path) for weight in setting.weights],
        "output_stem": args.output_stem,
        "gurobi_seed": args.gurobi_seed,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "skip_per_graph": args.skip_per_graph,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_setting_outputs(
    *,
    setting: SettingSpec,
    summary_rows: list[dict],
    per_graph_rows: list[dict],
    args: argparse.Namespace,
    dataset_dir: Path,
    graph_count: int,
) -> None:
    for source, run_dir in [
        ("step1b_fy", setting.fy_run_dir),
        ("step1c_spoplus", setting.spoplus_run_dir),
    ]:
        metrics_dir = run_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        source_summary = rows_for_run_dir(summary_rows, source)
        write_csv(metrics_dir / f"{args.output_stem}_summary.csv", source_summary, SUMMARY_FIELDS)
        (metrics_dir / f"{args.output_stem}_summary.json").write_text(
            json.dumps(source_summary, indent=2),
            encoding="utf-8",
        )
        write_run_config(
            metrics_dir / f"{args.output_stem}_run_config.json",
            args=args,
            setting=setting,
            dataset_dir=dataset_dir,
            graph_count=graph_count,
        )
        if not args.skip_per_graph:
            source_per_graph = rows_for_run_dir(per_graph_rows, source)
            write_csv(metrics_dir / f"{args.output_stem}_per_graph.csv", source_per_graph)


def evaluate_setting(
    *,
    setting: SettingSpec,
    graphs: list[dict],
    env,
    dataset_dir: Path,
    graph_count: int,
    args: argparse.Namespace,
) -> list[dict]:
    print(
        f"Evaluating {setting.regime} train_size={setting.train_size}: "
        f"{len(setting.weights)} checkpoints",
        flush=True,
    )
    models_and_evaluations = []
    models = []
    for weight_spec in setting.weights:
        model = load_model_weight(weight_spec.path)
        model["checkpoint_label"] = weight_spec.label
        model["checkpoint_source"] = weight_spec.source
        models.append(model)
        print(
            "  checkpoint "
            f"{weight_spec.label} method={model['method']} "
            f"selected_by={model['selection_metric']} "
            f"epoch={model['selected_epoch']} theta={model['theta']}",
            flush=True,
        )
        evaluations = common.evaluate_theta(model["theta"], graphs, env)
        models_and_evaluations.append((model, evaluations))

    summary_rows = summarize_evaluated_models(
        models_and_evaluations,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    summary_rows = enrich_summary_rows(
        summary_rows,
        setting=setting,
        models=models,
        dataset_dir=dataset_dir,
        graph_count=graph_count,
    )

    per_graph_rows = []
    if not args.skip_per_graph:
        for model, evaluations in models_and_evaluations:
            per_graph_rows.extend(
                per_graph_rows_for_model(
                    setting=setting,
                    model=model,
                    evaluations=evaluations,
                    dataset_dir=dataset_dir,
                    graph_count=graph_count,
                )
            )

    write_setting_outputs(
        setting=setting,
        summary_rows=summary_rows,
        per_graph_rows=per_graph_rows,
        args=args,
        dataset_dir=dataset_dir,
        graph_count=graph_count,
    )
    return summary_rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Step2 FY/SPO+ model checkpoints on matching unseen10000 "
            "processed datasets. Each regime is loaded once."
        )
    )
    parser.add_argument(
        "--regimes",
        nargs="+",
        default=None,
        help="Optional regime names to evaluate. Defaults to all Step2a/b/c regimes.",
    )
    parser.add_argument(
        "--train_sizes",
        nargs="+",
        type=int,
        default=DEFAULT_TRAIN_SIZES,
    )
    parser.add_argument("--output_stem", default="unseen10000")
    parser.add_argument(
        "--combined_summary_out",
        default=str(STEP2_DIR / "step2_unseen10000_all_checkpoints_summary.csv"),
    )
    parser.add_argument("--graph_limit", type=int, default=None)
    parser.add_argument("--gurobi_seed", type=int, default=42)
    parser.add_argument("--bootstrap_samples", type=int, default=1000)
    parser.add_argument("--bootstrap_seed", type=int, default=42)
    parser.add_argument(
        "--skip_per_graph",
        action="store_true",
        help="Only write summary outputs. Useful for quick smoke tests.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only list regimes, run directories, and checkpoint weights.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    regime_specs = default_regime_specs()
    if args.regimes is not None:
        wanted = set(args.regimes)
        regime_specs = [spec for spec in regime_specs if spec.regime in wanted]
        missing = sorted(wanted - {spec.regime for spec in regime_specs})
        if missing:
            raise ValueError(f"Unknown regimes requested: {', '.join(missing)}")

    settings, warnings = discover_setting_specs(
        regime_specs=regime_specs,
        train_sizes=list(args.train_sizes),
    )
    for warning in warnings:
        print(f"warning: {warning}", flush=True)
    if not settings:
        raise FileNotFoundError("No complete Step2 settings found.")

    print("Complete Step2 settings:", flush=True)
    for setting in settings:
        print(
            f"  {setting.regime} train_size={setting.train_size} "
            f"weights={len(setting.weights)}",
            flush=True,
        )
        if args.dry_run:
            for weight in setting.weights:
                print(f"    {weight.label}: {weight.path}", flush=True)

    if args.dry_run:
        print(
            f"Dry run: {len(settings)} settings, "
            f"{sum(len(setting.weights) for setting in settings)} model evaluations.",
            flush=True,
        )
        return 0

    import gurobipy as gp

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Seed", args.gurobi_seed)
    env.start()

    all_summary_rows: list[dict] = []
    try:
        by_regime: dict[str, list[SettingSpec]] = {}
        for setting in settings:
            by_regime.setdefault(setting.regime, []).append(setting)

        for regime in sorted(by_regime):
            regime_settings = sorted(by_regime[regime], key=lambda item: item.train_size)
            regime_spec = regime_settings[0].regime_spec
            dataset_dir = regime_spec.dataset_dir
            if not dataset_dir.is_dir():
                raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
            graph_paths = list_graph_paths(dataset_dir, args.graph_limit)
            if not graph_paths:
                raise FileNotFoundError(f"No G-*.json files found under {dataset_dir}")

            graphs = []
            try:
                print(
                    f"Loading unseen dataset once for {regime}: n={len(graph_paths)} "
                    f"dataset={dataset_dir}",
                    flush=True,
                )
                graphs = common.load_graph_records(graph_paths, env)
                for setting in regime_settings:
                    all_summary_rows.extend(
                        evaluate_setting(
                            setting=setting,
                            graphs=graphs,
                            env=env,
                            dataset_dir=dataset_dir,
                            graph_count=len(graph_paths),
                            args=args,
                        )
                    )
            finally:
                common.dispose_graph_records(graphs)

        combined_summary_out = Path(args.combined_summary_out)
        write_csv(combined_summary_out, all_summary_rows, SUMMARY_FIELDS)
        combined_summary_out.with_suffix(".json").write_text(
            json.dumps(all_summary_rows, indent=2),
            encoding="utf-8",
        )
        print(
            f"Saved combined summary: {combined_summary_out} "
            f"rows={len(all_summary_rows)}",
            flush=True,
        )
    finally:
        env.dispose()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
