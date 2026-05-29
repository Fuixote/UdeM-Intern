#!/usr/bin/env python3
"""Run small SPO+ training-protocol sweeps for the paper shortest-path setup."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence, Tuple

import common
import run_paper_shortest_path as runner


@dataclass(frozen=True)
class ProtocolVariant:
    name: str
    spoplus_iterations: int
    batch_size: int
    learning_rate: float
    lambda_grid: Tuple[float, ...]
    spoplus_iterate: str = "raw"
    spoplus_init: str = "ls"


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def protocol_variants() -> Tuple[ProtocolVariant, ...]:
    return (
        ProtocolVariant(
            name="baseline-current",
            spoplus_iterations=1000,
            batch_size=32,
            learning_rate=0.05,
            lambda_grid=common.DEFAULT_LAMBDA_GRID,
        ),
        ProtocolVariant(
            name="smaller-batch",
            spoplus_iterations=3000,
            batch_size=10,
            learning_rate=0.01,
            lambda_grid=common.DEFAULT_LAMBDA_GRID,
        ),
        ProtocolVariant(
            name="no-l1",
            spoplus_iterations=1000,
            batch_size=32,
            learning_rate=0.05,
            lambda_grid=(0.0,),
        ),
        ProtocolVariant(
            name="averaged-iterate",
            spoplus_iterations=1000,
            batch_size=32,
            learning_rate=0.05,
            lambda_grid=common.DEFAULT_LAMBDA_GRID,
            spoplus_iterate="averaged",
        ),
        ProtocolVariant(
            name="zero-init-diagnostic",
            spoplus_iterations=1000,
            batch_size=32,
            learning_rate=0.05,
            lambda_grid=common.DEFAULT_LAMBDA_GRID,
            spoplus_init="zero",
        ),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run fixed SPO+ protocol sweep variants on high-degree regimes."
    )
    parser.add_argument("--degrees", nargs="+", default=("6", "8"))
    parser.add_argument("--noise-half-widths", nargs="+", default=("0", "0.5"))
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-val", type=int, default=250)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(variant.name for variant in protocol_variants()),
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / f"protocol_sweep_{_timestamp()}",
    )
    return parser


def _variant_args(
    variant: ProtocolVariant,
    args: argparse.Namespace,
    output_dir: Path,
) -> list[str]:
    values = [
        "--preset",
        "middle-row",
        "--methods",
        "ls",
        "ours-spoplus",
        "--degrees",
        *[str(value) for value in args.degrees],
        "--noise-half-widths",
        *[str(value) for value in args.noise_half_widths],
        "--trials",
        str(args.trials),
        "--n-train",
        str(args.n_train),
        "--n-val",
        str(args.n_val),
        "--n-test",
        str(args.n_test),
        "--seed",
        str(args.seed),
        "--spoplus-iterations",
        str(variant.spoplus_iterations),
        "--batch-size",
        str(variant.batch_size),
        "--learning-rate",
        str(variant.learning_rate),
        "--spoplus-iterate",
        variant.spoplus_iterate,
        "--spoplus-init",
        variant.spoplus_init,
        "--lambda-grid",
        *[f"{value:.16g}" for value in variant.lambda_grid],
        "--output-dir",
        str(output_dir / variant.name),
    ]
    return values


def write_sweep_index(output_dir: Path, variants: Sequence[ProtocolVariant]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "sweep_index.csv"
    fields = (
        "variant",
        "spoplus_iterations",
        "batch_size",
        "learning_rate",
        "lambda_grid_length",
        "lambda_grid",
        "spoplus_iterate",
        "spoplus_init",
        "output_dir",
    )
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for variant in variants:
            writer.writerow(
                {
                    "variant": variant.name,
                    "spoplus_iterations": variant.spoplus_iterations,
                    "batch_size": variant.batch_size,
                    "learning_rate": variant.learning_rate,
                    "lambda_grid_length": len(variant.lambda_grid),
                    "lambda_grid": " ".join(f"{value:.16g}" for value in variant.lambda_grid),
                    "spoplus_iterate": variant.spoplus_iterate,
                    "spoplus_init": variant.spoplus_init,
                    "output_dir": str(output_dir / variant.name),
                }
            )
    return index_path


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    selected_names = set(args.variants) if args.variants is not None else None
    variants = tuple(
        variant
        for variant in protocol_variants()
        if selected_names is None or variant.name in selected_names
    )
    output_dir = Path(args.output_dir)
    index_path = write_sweep_index(output_dir, variants)
    print(f"wrote {index_path}")
    for variant in variants:
        print(f"running protocol variant: {variant.name}")
        code = runner.main(_variant_args(variant, args, output_dir))
        if code != 0:
            return int(code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
