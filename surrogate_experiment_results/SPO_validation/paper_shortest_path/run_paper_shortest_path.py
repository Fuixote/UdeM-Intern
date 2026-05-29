#!/usr/bin/env python3
"""Run the SPO-paper synthetic shortest-path validation experiment."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import common


DependencyStatus = common.DependencyStatus

OUTPUT_FIELDS = (
    "implementation",
    "method",
    "trial",
    "degree",
    "noise_half_width",
    "seed",
    "selected_lambda",
    "val_norm_spo",
    "val_avg_regret",
    "val_path_accuracy",
    "val_optimality_ratio",
    "test_norm_spo",
    "test_avg_regret",
    "test_avg_relative_regret",
    "test_path_accuracy",
    "test_optimality_ratio",
    "n_train",
    "n_val",
    "n_test",
    "spoplus_iterations",
    "batch_size",
    "learning_rate",
)

PRESET_NAMES = (
    "smoke",
    "pilot",
    "pyepo-pilot",
    "middle-row",
    "middle-row-pyepo",
    "full",
)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _preset_defaults(preset: str) -> Dict[str, object]:
    if preset == "smoke":
        return {
            "degrees": (1,),
            "noise_half_widths": (0.0,),
            "trials": 1,
            "n_train": 20,
            "n_val": 8,
            "n_test": 12,
            "lambda_grid": (0.0,),
            "spoplus_iterations": 3,
            "batch_size": 5,
            "learning_rate": 0.05,
            "methods": ("ls", "ours-spoplus"),
        }
    if preset == "pilot":
        return {
            "degrees": (1, 8),
            "noise_half_widths": (0.0, 0.5),
            "trials": 3,
            "n_train": 1000,
            "n_val": 250,
            "n_test": 2000,
            "lambda_grid": common.DEFAULT_LAMBDA_GRID,
            "spoplus_iterations": 300,
            "batch_size": 32,
            "learning_rate": 0.05,
            "methods": ("ls", "ours-spoplus"),
        }
    if preset == "pyepo-pilot":
        options = _preset_defaults("pilot")
        options["methods"] = ("ls", "ours-spoplus", "pyepo-spoplus")
        return options
    if preset in {"middle-row", "full"}:
        return {
            "degrees": common.DEFAULT_DEGREES,
            "noise_half_widths": common.DEFAULT_NOISE_HALF_WIDTHS,
            "trials": 50,
            "n_train": 1000,
            "n_val": 250,
            "n_test": 10000,
            "lambda_grid": common.DEFAULT_LAMBDA_GRID,
            "spoplus_iterations": 1000,
            "batch_size": 32,
            "learning_rate": 0.05,
            "methods": ("ls", "ours-spoplus"),
        }
    if preset == "middle-row-pyepo":
        options = _preset_defaults("middle-row")
        options["methods"] = ("ls", "ours-spoplus", "pyepo-spoplus")
        return options
    raise ValueError(f"unknown preset: {preset}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the synthetic shortest-path benchmark from the SPO paper."
    )
    parser.add_argument("--preset", choices=PRESET_NAMES, default="smoke")
    parser.add_argument("--degrees", nargs="+", default=None)
    parser.add_argument("--noise-half-widths", nargs="+", default=None)
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--n-val", type=int, default=None)
    parser.add_argument("--n-test", type=int, default=None)
    parser.add_argument("--lambda-grid", nargs="+", default=None)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("ls", "ours-spoplus", "pyepo-spoplus"),
        default=None,
    )
    parser.add_argument("--spoplus-iterations", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-if-pyepo-missing", action="store_true")
    return parser


def resolve_options(args: argparse.Namespace) -> Dict[str, object]:
    options = _preset_defaults(args.preset)
    if args.degrees is not None:
        options["degrees"] = common.parse_int_list(args.degrees)
    if args.noise_half_widths is not None:
        options["noise_half_widths"] = common.parse_float_list(args.noise_half_widths)
    if args.trials is not None:
        options["trials"] = args.trials
    if args.n_train is not None:
        options["n_train"] = args.n_train
    if args.n_val is not None:
        options["n_val"] = args.n_val
    if args.n_test is not None:
        options["n_test"] = args.n_test
    if args.lambda_grid is not None:
        options["lambda_grid"] = common.parse_float_list(args.lambda_grid)
    if args.methods is not None:
        options["methods"] = tuple(args.methods)
    if args.spoplus_iterations is not None:
        options["spoplus_iterations"] = args.spoplus_iterations
    if args.batch_size is not None:
        options["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        options["learning_rate"] = args.learning_rate
    options["seed"] = args.seed
    if args.output_dir is None:
        options["output_dir"] = (
            Path(__file__).resolve().parent / "results" / f"{args.preset}_{_timestamp()}"
        )
    else:
        options["output_dir"] = args.output_dir
    options["preset"] = args.preset
    return options


def pyepo_requested(options: Mapping[str, object]) -> bool:
    return "pyepo-spoplus" in tuple(options["methods"])


def validate_pyepo_request(
    options: Mapping[str, object],
    fail_if_missing: bool,
    status_checker=common.check_pyepo_dependencies,
) -> DependencyStatus:
    status = status_checker()
    if pyepo_requested(options) and fail_if_missing and not status.available:
        raise RuntimeError(
            "PyEPO SPO+ was requested but the reference dependencies are unavailable: "
            f"{status.message}; details={dict(status.details)}"
        )
    return status


def estimate_model_fits(options: Mapping[str, object]) -> int:
    regimes = (
        len(tuple(options["degrees"]))
        * len(tuple(options["noise_half_widths"]))
        * int(options["trials"])
    )
    return regimes * len(tuple(options["methods"])) * len(tuple(options["lambda_grid"]))


def estimate_ours_spoplus_oracle_calls(options: Mapping[str, object]) -> int:
    if "ours-spoplus" not in tuple(options["methods"]):
        return 0
    regimes = (
        len(tuple(options["degrees"]))
        * len(tuple(options["noise_half_widths"]))
        * int(options["trials"])
    )
    return (
        regimes
        * len(tuple(options["lambda_grid"]))
        * int(options["spoplus_iterations"])
        * int(options["batch_size"])
    )


def print_dry_run_plan(
    options: Mapping[str, object],
    pyepo_status: DependencyStatus,
) -> None:
    print("Resolved SPO paper shortest-path plan")
    print(f"preset: {options['preset']}")
    print(f"degrees: {tuple(options['degrees'])}")
    print(f"noise_half_widths: {tuple(options['noise_half_widths'])}")
    print(f"trials: {options['trials']}")
    print(f"n_train/n_val/n_test: {options['n_train']}/{options['n_val']}/{options['n_test']}")
    print(f"methods: {tuple(options['methods'])}")
    print(f"lambda_grid_length: {len(tuple(options['lambda_grid']))}")
    print(f"estimated_model_fits: {estimate_model_fits(options)}")
    print(
        "estimated_ours_spoplus_oracle_calls: "
        f"{estimate_ours_spoplus_oracle_calls(options)}"
    )
    print(f"pyepo_requested: {pyepo_requested(options)}")
    print(f"pyepo_available: {pyepo_status.available}")
    if pyepo_requested(options):
        print(f"pyepo_status: {pyepo_status.message}")


def build_metadata(
    options: Mapping[str, object],
    config: common.PaperShortestPathConfig,
    pyepo_status: DependencyStatus,
) -> Dict[str, object]:
    return {
        "paper_experiment": "shortest_path_middle_row",
        "normalized_spo_definition": "sum_regret_over_sum_oracle_cost",
        "grid_shape": list(config.grid_shape),
        "feature_dim": config.feature_dim,
        "edge_dim": common.paper_edge_count(config.grid_shape),
        "lambda_grid": [float(value) for value in options["lambda_grid"]],
        "methods": list(options["methods"]),
        "optimizer_ours_spoplus": (
            "stochastic subgradient with step_size=learning_rate/sqrt(iteration), "
            "L1 subgradient on non-intercept weights"
        ),
        "optimizer_pyepo_spoplus": "torch Adam on linear model with optional L1 penalty",
        "pyepo_requested": pyepo_requested(options),
        "pyepo_available": bool(pyepo_status.available),
        "pyepo_status": pyepo_status.message,
        "pyepo_dependency_details": dict(pyepo_status.details),
        "estimated_model_fits": estimate_model_fits(options),
        "estimated_ours_spoplus_oracle_calls": estimate_ours_spoplus_oracle_calls(options),
        **{
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in options.items()
        },
    }


def _method_result(
    method: str,
    instance: common.PaperTrialInstance,
    lambdas: Sequence[float],
    spoplus_iterations: int,
    batch_size: int,
    learning_rate: float,
) -> common.ModelResult:
    if method == "ls":
        return common.select_least_squares_model(
            instance.train,
            instance.val,
            lambdas=lambdas,
        )
    if method == "ours-spoplus":
        return common.select_spoplus_ours_model(
            instance.train,
            instance.val,
            lambdas=lambdas,
            iterations=spoplus_iterations,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=instance.seed,
        )
    if method == "pyepo-spoplus":
        return common.train_spoplus_pyepo(
            instance.train,
            instance.val,
            lambdas=lambdas,
            iterations=spoplus_iterations,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=instance.seed,
        )
    raise ValueError(f"unknown method: {method}")


def _row_for_result(
    result: common.ModelResult,
    instance: common.PaperTrialInstance,
    options: Mapping[str, object],
) -> Dict[str, object]:
    predictions = common.predict_costs(instance.test.features, result.coefficients)
    test_metrics = common.evaluate_predictions(predictions, instance.test)
    return {
        "implementation": result.method,
        "method": "SPO+" if result.method.endswith("spoplus") else "LS",
        "trial": instance.trial,
        "degree": instance.degree,
        "noise_half_width": instance.noise_half_width,
        "seed": instance.seed,
        "selected_lambda": result.selected_lambda,
        "val_norm_spo": result.val_metrics["normalized_spo_loss"],
        "val_avg_regret": result.val_metrics["avg_regret"],
        "val_path_accuracy": result.val_metrics["path_accuracy"],
        "val_optimality_ratio": result.val_metrics["optimality_ratio"],
        "test_norm_spo": test_metrics["normalized_spo_loss"],
        "test_avg_regret": test_metrics["avg_regret"],
        "test_avg_relative_regret": test_metrics["avg_relative_regret"],
        "test_path_accuracy": test_metrics["path_accuracy"],
        "test_optimality_ratio": test_metrics["optimality_ratio"],
        "n_train": options["n_train"],
        "n_val": options["n_val"],
        "n_test": options["n_test"],
        "spoplus_iterations": options["spoplus_iterations"],
        "batch_size": options["batch_size"],
        "learning_rate": options["learning_rate"],
    }


def write_summary(output_dir: Path, rows: Sequence[Mapping[str, object]]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return summary_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    options = resolve_options(args)
    try:
        pyepo_status = validate_pyepo_request(
            options,
            fail_if_missing=args.fail_if_pyepo_missing or not args.dry_run,
        )
    except RuntimeError as exc:
        parser.exit(2, f"{exc}\n")

    if args.dry_run:
        print_dry_run_plan(options, pyepo_status)
        return 0

    output_dir = Path(options["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    config = common.PaperShortestPathConfig(
        n_train=int(options["n_train"]),
        n_val=int(options["n_val"]),
        n_test=int(options["n_test"]),
    )
    lambdas = tuple(float(value) for value in options["lambda_grid"])
    rows: List[Dict[str, object]] = []
    for trial in range(int(options["trials"])):
        for degree in options["degrees"]:
            for noise_half_width in options["noise_half_widths"]:
                instance = common.make_trial_instance(
                    degree=int(degree),
                    noise_half_width=float(noise_half_width),
                    trial=trial,
                    seed=int(options["seed"]),
                    config=config,
                )
                for method in options["methods"]:
                    result = _method_result(
                        str(method),
                        instance,
                        lambdas=lambdas,
                        spoplus_iterations=int(options["spoplus_iterations"]),
                        batch_size=int(options["batch_size"]),
                        learning_rate=float(options["learning_rate"]),
                    )
                    row = _row_for_result(result, instance, options)
                    rows.append(row)
                    print(
                        "completed",
                        row["implementation"],
                        "trial",
                        row["trial"],
                        "degree",
                        row["degree"],
                        "noise",
                        row["noise_half_width"],
                        "test_norm_spo",
                        f"{float(row['test_norm_spo']):.6g}",
                    )

    summary_path = write_summary(output_dir, rows)
    metadata = build_metadata(options, config, pyepo_status)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
