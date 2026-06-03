#!/usr/bin/env python3
"""Select representative subset seeds for decision-level analysis."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "Step2_resampling"
    / "results"
    / "phase1_heldout400_paired_main.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "decision_analysis"
    / "results"
    / "selected_case_seeds.csv"
)


@dataclass(frozen=True)
class PairedSeedRow:
    regime: str
    block: str
    degree: int
    subset_seed: int
    mse_norm_gap: float
    spoplus_norm_gap: float
    norm_gap_reduction: float
    raw_gap_reduction: float
    spoplus_better_norm_gap: bool
    mse_selected_epoch: int
    spoplus_selected_epoch: int


@dataclass(frozen=True)
class SelectedCaseSeed:
    case_type: str
    subset_seed: int
    regime: str
    block: str
    degree: int
    mse_norm_gap: float
    spoplus_norm_gap: float
    norm_gap_reduction: float
    raw_gap_reduction: float
    spoplus_better_norm_gap: bool
    mse_selected_epoch: int
    spoplus_selected_epoch: int
    selection_reason: str


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def parse_row(row: dict[str, str]) -> PairedSeedRow:
    return PairedSeedRow(
        regime=row["regime"],
        block=row["block"],
        degree=int(row["degree"]),
        subset_seed=int(row["subset_seed"]),
        mse_norm_gap=float(row["mse_norm_gap"]),
        spoplus_norm_gap=float(row["spoplus_norm_gap"]),
        norm_gap_reduction=float(row["norm_gap_reduction"]),
        raw_gap_reduction=float(row["raw_gap_reduction"]),
        spoplus_better_norm_gap=parse_bool(row["spoplus_better_norm_gap"]),
        mse_selected_epoch=int(row["mse_selected_epoch"]),
        spoplus_selected_epoch=int(row["spoplus_selected_epoch"]),
    )


def load_rows(path: str | Path) -> list[PairedSeedRow]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [parse_row(row) for row in csv.DictReader(handle)]


def selected_case(
    row: PairedSeedRow,
    case_type: str,
    selection_reason: str,
) -> SelectedCaseSeed:
    return SelectedCaseSeed(
        case_type=case_type,
        subset_seed=row.subset_seed,
        regime=row.regime,
        block=row.block,
        degree=row.degree,
        mse_norm_gap=row.mse_norm_gap,
        spoplus_norm_gap=row.spoplus_norm_gap,
        norm_gap_reduction=row.norm_gap_reduction,
        raw_gap_reduction=row.raw_gap_reduction,
        spoplus_better_norm_gap=row.spoplus_better_norm_gap,
        mse_selected_epoch=row.mse_selected_epoch,
        spoplus_selected_epoch=row.spoplus_selected_epoch,
        selection_reason=selection_reason,
    )


def choose_rows(
    rows: list[PairedSeedRow],
    count: int,
    selected_seeds: set[int],
    sort_key,
) -> list[PairedSeedRow]:
    candidates = [row for row in rows if row.subset_seed not in selected_seeds]
    chosen = sorted(candidates, key=sort_key)[:count]
    if len(chosen) != count:
        raise ValueError(
            f"Requested {count} seeds but only found {len(chosen)} available candidates"
        )
    selected_seeds.update(row.subset_seed for row in chosen)
    return chosen


def select_case_seeds(
    rows: list[PairedSeedRow],
    regime: str,
    large_count: int = 3,
    weak_count: int = 3,
    easy_count: int = 3,
) -> list[SelectedCaseSeed]:
    regime_rows = [row for row in rows if row.regime == regime]
    if not regime_rows:
        raise ValueError(f"No rows found for regime: {regime}")

    selected_seeds: set[int] = set()
    large = choose_rows(
        regime_rows,
        large_count,
        selected_seeds,
        sort_key=lambda row: (-row.norm_gap_reduction, row.subset_seed),
    )
    easy = choose_rows(
        regime_rows,
        easy_count,
        selected_seeds,
        sort_key=lambda row: (
            max(row.mse_norm_gap, row.spoplus_norm_gap),
            row.mse_norm_gap,
            row.subset_seed,
        ),
    )
    weak = choose_rows(
        regime_rows,
        weak_count,
        selected_seeds,
        sort_key=lambda row: (row.norm_gap_reduction, row.mse_norm_gap, row.subset_seed),
    )

    selected: list[SelectedCaseSeed] = []
    selected.extend(
        selected_case(
            row,
            "large_improvement",
            "highest paired normalized-gap reduction",
        )
        for row in large
    )
    selected.extend(
        selected_case(
            row,
            "weak_borderline_improvement",
            "lowest paired normalized-gap reduction after excluding easy seeds",
        )
        for row in weak
    )
    selected.extend(
        selected_case(
            row,
            "easy_low_gap",
            "lowest max normalized gap across 2stage and SPO+",
        )
        for row in easy
    )
    return selected


def write_selected_csv(path: str | Path, rows: list[SelectedCaseSeed]) -> None:
    fieldnames = [
        "case_type",
        "subset_seed",
        "regime",
        "block",
        "degree",
        "mse_norm_gap",
        "spoplus_norm_gap",
        "norm_gap_reduction",
        "raw_gap_reduction",
        "spoplus_better_norm_gap",
        "mse_selected_epoch",
        "spoplus_selected_epoch",
        "selection_reason",
    ]
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fieldnames})


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Select Step2 resampling subset seeds for decision analysis."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--regime", default="step2b_poly_d8")
    parser.add_argument("--large-count", type=int, default=3)
    parser.add_argument("--weak-count", type=int, default=3)
    parser.add_argument("--easy-count", type=int, default=3)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    rows = load_rows(args.input)
    selected = select_case_seeds(
        rows,
        regime=args.regime,
        large_count=args.large_count,
        weak_count=args.weak_count,
        easy_count=args.easy_count,
    )
    write_selected_csv(args.output, selected)
    print(f"Saved {len(selected)} selected case seeds to {args.output}")
    for row in selected:
        print(
            f"{row.case_type}: seed={row.subset_seed} "
            f"mse_gap={row.mse_norm_gap:.6g} "
            f"spoplus_gap={row.spoplus_norm_gap:.6g} "
            f"reduction={row.norm_gap_reduction:.6g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
