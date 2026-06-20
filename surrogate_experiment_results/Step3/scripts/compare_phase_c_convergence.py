#!/usr/bin/env python3
"""Compare Step3 Phase-C review topology training at 100 versus 500 epochs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
STEP3_ROOT = PROJECT_ROOT / "surrogate_experiment_results" / "Step3" / "pairs20_ndd2"
DEFAULT_STATUS_100 = STEP3_ROOT / "phase_b" / "results" / "phase_b_training_status_e100.csv"
DEFAULT_STATUS_500 = (
    STEP3_ROOT
    / "phase_c"
    / "convergence"
    / "results"
    / "phase_c_convergence_status_e500.csv"
)
DEFAULT_RUNS_100 = STEP3_ROOT / "phase_b" / "runs_e100"
DEFAULT_RUNS_500 = STEP3_ROOT / "phase_c" / "convergence" / "runs_e500"
DEFAULT_TOPOLOGY_IDS = STEP3_ROOT / "phase_c" / "preflight" / "phase_c_review_topology_ids.txt"
DEFAULT_OUTPUT_DIR = STEP3_ROOT / "phase_c" / "convergence" / "comparison"
COMPLETED_STATUSES = {"success", "skipped"}
TOLERANCE = 1e-9

SEED_FIELDS = [
    "topology_id",
    "train_seed",
    "status_100",
    "status_500",
    "seed_outcome_100",
    "seed_outcome_500",
    "seed_outcome_changed",
    "improvement_gap_100",
    "improvement_gap_500",
    "delta_improvement_gap",
    "normalized_improvement_gap_100",
    "normalized_improvement_gap_500",
    "delta_normalized_improvement_gap",
    "test_gap_2stage_100",
    "test_gap_2stage_500",
    "delta_test_gap_2stage",
    "test_gap_spoplus_100",
    "test_gap_spoplus_500",
    "delta_test_gap_spoplus",
    "selected_epoch_2stage_100",
    "selected_epoch_2stage_500",
    "selected_epoch_spoplus_100",
    "selected_epoch_spoplus_500",
    "epoch_2stage_at_max_100",
    "epoch_2stage_at_max_500",
    "epoch_spoplus_at_max_100",
    "epoch_spoplus_at_max_500",
]

TOPOLOGY_FIELDS = [
    "topology_id",
    "matched_seed_count",
    "mean_improvement_gap_100",
    "mean_improvement_gap_500",
    "delta_mean_improvement_gap",
    "mean_abs_delta_improvement_gap",
    "max_abs_delta_improvement_gap",
    "std_improvement_gap_100",
    "std_improvement_gap_500",
    "mean_normalized_improvement_gap_100",
    "mean_normalized_improvement_gap_500",
    "delta_mean_normalized_improvement_gap",
    "fraction_better_100",
    "fraction_better_500",
    "fraction_worse_100",
    "fraction_worse_500",
    "fraction_tied_100",
    "fraction_tied_500",
    "changed_seed_outcome_count",
    "fraction_seed_outcome_changed",
    "topology_pattern_100",
    "topology_pattern_500",
    "topology_pattern_changed",
    "fraction_2stage_selected_at_max_epoch_100",
    "fraction_2stage_selected_at_max_epoch_500",
    "fraction_spoplus_selected_at_max_epoch_100",
    "fraction_spoplus_selected_at_max_epoch_500",
    "selected_epoch_values_2stage_100",
    "selected_epoch_values_2stage_500",
    "selected_epoch_values_spoplus_100",
    "selected_epoch_values_spoplus_500",
]


def int_or_text_key(value: Any) -> tuple[int, int | str]:
    text = str(value)
    if text.startswith("G-"):
        text = text[2:]
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def to_float(value: Any, default: float = 0.0) -> float:
    if value in (None, "", "None", "nan"):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(result) else result


def to_optional_float(value: Any) -> float | None:
    if value in (None, "", "None", "nan"):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(result) else result


def mean_or_zero(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def pstdev_or_zero(values: list[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) >= 2 else 0.0


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_read_json(path: Path) -> Any:
    if not Path(path).exists():
        return None
    return read_json(Path(path))


def read_topology_ids(path: Path) -> list[str] | None:
    if not Path(path).exists():
        return None
    ids = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()]
    ids = [value for value in ids if value]
    return ids or None


def status_key(row: dict[str, Any]) -> tuple[str, int]:
    return (str(row.get("topology_id", "")), int(row.get("train_seed", 0) or 0))


def read_status_index(
    path: Path,
    topology_ids: list[str] | None = None,
) -> dict[tuple[str, int], dict[str, Any]]:
    allowed = None if topology_ids is None else set(topology_ids)
    rows = {}
    for row in read_csv_rows(path):
        topology_id = str(row.get("topology_id", ""))
        if allowed is not None and topology_id not in allowed:
            continue
        if str(row.get("status", "")).strip().lower() not in COMPLETED_STATUSES:
            continue
        rows[status_key(row)] = row
    return rows


def seed_outcome(value: float | None) -> str:
    if value is None:
        return "missing"
    if value > TOLERANCE:
        return "better"
    if value < -TOLERANCE:
        return "worse"
    return "tie"


def weight_json_path(runs_dir: Path, topology_id: str, train_seed: int, filename: str) -> Path:
    return (
        Path(runs_dir)
        / topology_id
        / f"train_seed={int(train_seed):06d}"
        / "model_weights"
        / filename
    )


def selected_epoch(runs_dir: Path, topology_id: str, train_seed: int, filename: str) -> int | str:
    payload = safe_read_json(weight_json_path(runs_dir, topology_id, train_seed, filename))
    if not isinstance(payload, dict):
        return ""
    value = payload.get("selected_epoch", "")
    if value in ("", None):
        return ""
    return int(float(value))


def epoch_at_max(epoch: int | str, max_epoch: int) -> bool | str:
    if epoch in ("", None):
        return ""
    return int(epoch) >= int(max_epoch)


def metric(row: dict[str, Any], field: str) -> float | None:
    return to_optional_float(row.get(field))


def compare_seed(
    topology_id: str,
    train_seed: int,
    row_100: dict[str, Any],
    row_500: dict[str, Any],
    runs_100_dir: Path,
    runs_500_dir: Path,
) -> dict[str, Any]:
    improvement_100 = metric(row_100, "spoplus_improvement_gap")
    improvement_500 = metric(row_500, "spoplus_improvement_gap")
    norm_100 = metric(row_100, "spoplus_improvement_normalized_gap")
    norm_500 = metric(row_500, "spoplus_improvement_normalized_gap")
    gap_2stage_100 = metric(row_100, "test_mean_decision_gap_2stage")
    gap_2stage_500 = metric(row_500, "test_mean_decision_gap_2stage")
    gap_spoplus_100 = metric(row_100, "test_mean_decision_gap_spoplus")
    gap_spoplus_500 = metric(row_500, "test_mean_decision_gap_spoplus")
    epochs_100 = int(to_float(row_100.get("epochs_2stage"), 100.0))
    epochs_500 = int(to_float(row_500.get("epochs_2stage"), 500.0))
    spop_epochs_100 = int(to_float(row_100.get("epochs_spoplus"), 100.0))
    spop_epochs_500 = int(to_float(row_500.get("epochs_spoplus"), 500.0))
    epoch_2stage_100 = selected_epoch(
        runs_100_dir,
        topology_id,
        train_seed,
        "2stage_best_by_validation_mse_loss.json",
    )
    epoch_2stage_500 = selected_epoch(
        runs_500_dir,
        topology_id,
        train_seed,
        "2stage_best_by_validation_mse_loss.json",
    )
    epoch_spoplus_100 = selected_epoch(
        runs_100_dir,
        topology_id,
        train_seed,
        "spoplus_best_by_validation_spoplus_loss.json",
    )
    epoch_spoplus_500 = selected_epoch(
        runs_500_dir,
        topology_id,
        train_seed,
        "spoplus_best_by_validation_spoplus_loss.json",
    )
    outcome_100 = seed_outcome(improvement_100)
    outcome_500 = seed_outcome(improvement_500)
    return {
        "topology_id": topology_id,
        "train_seed": int(train_seed),
        "status_100": row_100.get("status", ""),
        "status_500": row_500.get("status", ""),
        "seed_outcome_100": outcome_100,
        "seed_outcome_500": outcome_500,
        "seed_outcome_changed": outcome_100 != outcome_500,
        "improvement_gap_100": "" if improvement_100 is None else improvement_100,
        "improvement_gap_500": "" if improvement_500 is None else improvement_500,
        "delta_improvement_gap": ""
        if improvement_100 is None or improvement_500 is None
        else improvement_500 - improvement_100,
        "normalized_improvement_gap_100": "" if norm_100 is None else norm_100,
        "normalized_improvement_gap_500": "" if norm_500 is None else norm_500,
        "delta_normalized_improvement_gap": ""
        if norm_100 is None or norm_500 is None
        else norm_500 - norm_100,
        "test_gap_2stage_100": "" if gap_2stage_100 is None else gap_2stage_100,
        "test_gap_2stage_500": "" if gap_2stage_500 is None else gap_2stage_500,
        "delta_test_gap_2stage": ""
        if gap_2stage_100 is None or gap_2stage_500 is None
        else gap_2stage_500 - gap_2stage_100,
        "test_gap_spoplus_100": "" if gap_spoplus_100 is None else gap_spoplus_100,
        "test_gap_spoplus_500": "" if gap_spoplus_500 is None else gap_spoplus_500,
        "delta_test_gap_spoplus": ""
        if gap_spoplus_100 is None or gap_spoplus_500 is None
        else gap_spoplus_500 - gap_spoplus_100,
        "selected_epoch_2stage_100": epoch_2stage_100,
        "selected_epoch_2stage_500": epoch_2stage_500,
        "selected_epoch_spoplus_100": epoch_spoplus_100,
        "selected_epoch_spoplus_500": epoch_spoplus_500,
        "epoch_2stage_at_max_100": epoch_at_max(epoch_2stage_100, epochs_100),
        "epoch_2stage_at_max_500": epoch_at_max(epoch_2stage_500, epochs_500),
        "epoch_spoplus_at_max_100": epoch_at_max(epoch_spoplus_100, spop_epochs_100),
        "epoch_spoplus_at_max_500": epoch_at_max(epoch_spoplus_500, spop_epochs_500),
    }


def fraction(rows: list[dict[str, Any]], predicate) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if predicate(row)) / len(rows)


def numeric_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    values = []
    for row in rows:
        value = to_optional_float(row.get(field))
        if value is not None:
            values.append(value)
    return values


def epoch_values(rows: list[dict[str, Any]], field: str) -> str:
    values = sorted(
        {int(row[field]) for row in rows if row.get(field, "") not in ("", None)}
    )
    return "|".join(str(value) for value in values)


def topology_pattern(values: list[float]) -> str:
    if not values:
        return "missing"
    better = sum(1 for value in values if value > TOLERANCE)
    worse = sum(1 for value in values if value < -TOLERANCE)
    tied = sum(1 for value in values if abs(value) <= TOLERANCE)
    if better == len(values):
        return "all_better"
    if worse == len(values):
        return "all_worse"
    if tied == len(values):
        return "all_tie"
    if better > 0 and worse > 0:
        return "mixed_better_worse"
    if better > 0 and tied > 0:
        return "better_tie"
    if worse > 0 and tied > 0:
        return "worse_tie"
    return "mixed"


def summarize_topology(topology_id: str, seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    imp_100 = numeric_values(seed_rows, "improvement_gap_100")
    imp_500 = numeric_values(seed_rows, "improvement_gap_500")
    norm_100 = numeric_values(seed_rows, "normalized_improvement_gap_100")
    norm_500 = numeric_values(seed_rows, "normalized_improvement_gap_500")
    deltas = numeric_values(seed_rows, "delta_improvement_gap")
    norm_deltas = numeric_values(seed_rows, "delta_normalized_improvement_gap")
    pattern_100 = topology_pattern(imp_100)
    pattern_500 = topology_pattern(imp_500)
    return {
        "topology_id": topology_id,
        "matched_seed_count": len(seed_rows),
        "mean_improvement_gap_100": mean_or_zero(imp_100),
        "mean_improvement_gap_500": mean_or_zero(imp_500),
        "delta_mean_improvement_gap": mean_or_zero(imp_500) - mean_or_zero(imp_100),
        "mean_abs_delta_improvement_gap": mean_or_zero([abs(value) for value in deltas]),
        "max_abs_delta_improvement_gap": max((abs(value) for value in deltas), default=0.0),
        "std_improvement_gap_100": pstdev_or_zero(imp_100),
        "std_improvement_gap_500": pstdev_or_zero(imp_500),
        "mean_normalized_improvement_gap_100": mean_or_zero(norm_100),
        "mean_normalized_improvement_gap_500": mean_or_zero(norm_500),
        "delta_mean_normalized_improvement_gap": mean_or_zero(norm_deltas),
        "fraction_better_100": fraction(seed_rows, lambda row: row["seed_outcome_100"] == "better"),
        "fraction_better_500": fraction(seed_rows, lambda row: row["seed_outcome_500"] == "better"),
        "fraction_worse_100": fraction(seed_rows, lambda row: row["seed_outcome_100"] == "worse"),
        "fraction_worse_500": fraction(seed_rows, lambda row: row["seed_outcome_500"] == "worse"),
        "fraction_tied_100": fraction(seed_rows, lambda row: row["seed_outcome_100"] == "tie"),
        "fraction_tied_500": fraction(seed_rows, lambda row: row["seed_outcome_500"] == "tie"),
        "changed_seed_outcome_count": sum(
            1 for row in seed_rows if bool(row.get("seed_outcome_changed"))
        ),
        "fraction_seed_outcome_changed": fraction(
            seed_rows, lambda row: bool(row.get("seed_outcome_changed"))
        ),
        "topology_pattern_100": pattern_100,
        "topology_pattern_500": pattern_500,
        "topology_pattern_changed": pattern_100 != pattern_500,
        "fraction_2stage_selected_at_max_epoch_100": fraction(
            seed_rows, lambda row: bool(row.get("epoch_2stage_at_max_100"))
        ),
        "fraction_2stage_selected_at_max_epoch_500": fraction(
            seed_rows, lambda row: bool(row.get("epoch_2stage_at_max_500"))
        ),
        "fraction_spoplus_selected_at_max_epoch_100": fraction(
            seed_rows, lambda row: bool(row.get("epoch_spoplus_at_max_100"))
        ),
        "fraction_spoplus_selected_at_max_epoch_500": fraction(
            seed_rows, lambda row: bool(row.get("epoch_spoplus_at_max_500"))
        ),
        "selected_epoch_values_2stage_100": epoch_values(seed_rows, "selected_epoch_2stage_100"),
        "selected_epoch_values_2stage_500": epoch_values(seed_rows, "selected_epoch_2stage_500"),
        "selected_epoch_values_spoplus_100": epoch_values(seed_rows, "selected_epoch_spoplus_100"),
        "selected_epoch_values_spoplus_500": epoch_values(seed_rows, "selected_epoch_spoplus_500"),
    }


def compare_convergence(
    *,
    status_100_csv: Path,
    status_500_csv: Path,
    runs_100_dir: Path,
    runs_500_dir: Path,
    topology_ids: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    status_100 = read_status_index(status_100_csv, topology_ids=topology_ids)
    status_500 = read_status_index(status_500_csv, topology_ids=topology_ids)
    matched_keys = sorted(set(status_100) & set(status_500), key=lambda item: (int_or_text_key(item[0]), item[1]))
    seed_rows = [
        compare_seed(
            topology_id=topology_id,
            train_seed=train_seed,
            row_100=status_100[(topology_id, train_seed)],
            row_500=status_500[(topology_id, train_seed)],
            runs_100_dir=runs_100_dir,
            runs_500_dir=runs_500_dir,
        )
        for topology_id, train_seed in matched_keys
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        grouped[str(row["topology_id"])].append(row)
    topology_rows = [
        summarize_topology(topology_id, grouped[topology_id])
        for topology_id in sorted(grouped, key=int_or_text_key)
    ]
    return seed_rows, topology_rows


def ordered_fieldnames(rows: list[dict[str, Any]], preferred: list[str]) -> list[str]:
    present = set()
    for row in rows:
        present.update(row)
    fields = [field for field in preferred if field in present]
    fields.extend(sorted(present - set(fields)))
    return fields


def write_convergence_outputs(
    output_dir: Path,
    seed_rows: list[dict[str, Any]],
    topology_rows: list[dict[str, Any]],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "phase_c_convergence_seed_compare.csv",
        seed_rows,
        ordered_fieldnames(seed_rows, SEED_FIELDS),
    )
    write_csv(
        output_dir / "phase_c_convergence_topology_summary.csv",
        topology_rows,
        ordered_fieldnames(topology_rows, TOPOLOGY_FIELDS),
    )
    pattern_changes = sum(1 for row in topology_rows if bool(row.get("topology_pattern_changed")))
    summary = {
        "generated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "num_topologies": len(topology_rows),
        "num_seed_rows": len(seed_rows),
        "topology_pattern_changed_count": pattern_changes,
        "mean_abs_delta_improvement_gap": mean_or_zero(
            [to_float(row.get("mean_abs_delta_improvement_gap")) for row in topology_rows]
        ),
        "max_abs_delta_improvement_gap": max(
            (to_float(row.get("max_abs_delta_improvement_gap")) for row in topology_rows),
            default=0.0,
        ),
        "topology_ids": [row["topology_id"] for row in topology_rows],
        "pattern_counts_100": dict(Counter(str(row.get("topology_pattern_100", "")) for row in topology_rows)),
        "pattern_counts_500": dict(Counter(str(row.get("topology_pattern_500", "")) for row in topology_rows)),
    }
    (output_dir / "phase_c_convergence_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-100", type=Path, default=DEFAULT_STATUS_100)
    parser.add_argument("--status-500", type=Path, default=DEFAULT_STATUS_500)
    parser.add_argument("--runs-100-dir", type=Path, default=DEFAULT_RUNS_100)
    parser.add_argument("--runs-500-dir", type=Path, default=DEFAULT_RUNS_500)
    parser.add_argument("--topology-ids", type=Path, default=DEFAULT_TOPOLOGY_IDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    topology_ids = read_topology_ids(args.topology_ids)
    seed_rows, topology_rows = compare_convergence(
        status_100_csv=args.status_100,
        status_500_csv=args.status_500,
        runs_100_dir=args.runs_100_dir,
        runs_500_dir=args.runs_500_dir,
        topology_ids=topology_ids,
    )
    if topology_ids and len({row["topology_id"] for row in topology_rows}) != len(topology_ids):
        found = {row["topology_id"] for row in topology_rows}
        missing = [topology_id for topology_id in topology_ids if topology_id not in found]
        raise ValueError(f"Missing compared topology ids: {missing}")
    write_convergence_outputs(args.output_dir, seed_rows, topology_rows)
    print(
        json.dumps(
            {
                "num_topologies": len(topology_rows),
                "num_seed_rows": len(seed_rows),
                "topology_pattern_changed_count": sum(
                    1 for row in topology_rows if bool(row.get("topology_pattern_changed"))
                ),
                "output_dir": str(args.output_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
