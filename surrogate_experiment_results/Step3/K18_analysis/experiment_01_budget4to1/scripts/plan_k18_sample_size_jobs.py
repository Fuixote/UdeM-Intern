#!/usr/bin/env python3
"""Plan K18 sample-size learning-curve jobs without launching them."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[5]
STEP3_SCRIPTS = PROJECT_ROOT / "surrogate_experiment_results" / "Step3" / "scripts"
if str(STEP3_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(STEP3_SCRIPTS))

import fixed_topology_xy_common as common  # noqa: E402


RUN_ONE_JOB = STEP3_SCRIPTS / "run_one_job.py"
DEFAULT_REGIME = "step2c_poly_d8_mult_eps050"
DEFAULT_SAMPLE_SIZE_SPLITS = {
    50: {"training_size": 40, "validation_size": 10},
    100: {"training_size": 80, "validation_size": 20},
    500: {"training_size": 400, "validation_size": 100},
}

JOB_FIELDS = [
    "job_id",
    "topology_id",
    "role",
    "regime",
    "protocol",
    "data_seed",
    "trainer_train_seed_arg",
    "sample_size",
    "training_size",
    "validation_size",
    "trainer_train_size_arg",
    "train_bank_path",
    "eval_manifest_path",
    "validation_path",
    "test_path",
    "output_dir",
    "expected_training_hash",
    "validation_hash",
    "test_hash",
    "theta_seed",
    "gurobi_seed",
    "max_epochs",
    "metric_stride",
    "early_stop_patience",
    "early_stop_min_delta",
    "runtime_class",
    "run_one_job_command",
    "status",
]


def read_csv_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in str(raw).split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one integer")
    return values


def parse_sample_size_splits(raw: str | None) -> dict[int, dict[str, int]]:
    if raw is None or not str(raw).strip():
        return {key: dict(value) for key, value in DEFAULT_SAMPLE_SIZE_SPLITS.items()}
    payload = json.loads(raw)
    return {
        int(sample_size): {
            "training_size": int(row["training_size"]),
            "validation_size": int(row["validation_size"]),
        }
        for sample_size, row in payload.items()
    }


def selected_topology_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError(f"No topology rows found in {path}")
    if "selection_rank" in rows[0]:
        return sorted(rows, key=lambda row: int(row["selection_rank"]))
    return rows


def read_train_prefix_hash(train_bank_path: Path, training_size: int) -> str | None:
    if not train_bank_path.exists():
        return None
    dataset = common.read_npz_dataset(train_bank_path)
    return dataset["manifest"].get("prefix_hashes", {}).get(str(int(training_size)))


def read_eval_manifest(eval_manifest_path: Path) -> dict[str, Any]:
    if not eval_manifest_path.exists():
        return {}
    return json.loads(eval_manifest_path.read_text(encoding="utf-8"))


def build_run_one_job_command(job: dict[str, Any], *, python_bin: str | None = None) -> str:
    command = [
        python_bin or sys.executable,
        str(RUN_ONE_JOB),
        "--train-bank",
        str(job["train_bank_path"]),
        "--eval-manifest",
        str(job["eval_manifest_path"]),
        "--topology-id",
        str(job["topology_id"]),
        "--regime",
        str(job["regime"]),
        "--protocol",
        str(job["protocol"]),
        "--train-seed",
        str(job["trainer_train_seed_arg"]),
        "--train-size",
        str(job["trainer_train_size_arg"]),
        "--sample-size",
        str(job["sample_size"]),
        "--theta-seed",
        str(job["theta_seed"]),
        "--gurobi-seed",
        str(job["gurobi_seed"]),
        "--max-epochs",
        str(job["max_epochs"]),
        "--metric-stride",
        str(job["metric_stride"]),
        "--early-stop-patience",
        str(job["early_stop_patience"]),
        "--early-stop-min-delta",
        str(job["early_stop_min_delta"]),
        "--output-dir",
        str(job["output_dir"]),
        "--dry-run",
    ]
    return shlex.join(command)


def build_plan(
    *,
    topology_rows: list[dict[str, Any]],
    output_root: str | Path,
    regime: str = DEFAULT_REGIME,
    data_seeds: list[int] | tuple[int, ...],
    sample_size_splits: dict[int, dict[str, int]],
    protocol: str = "screen",
    theta_seed: int = 42,
    gurobi_seed: int = 42,
    max_epochs: int = 3000,
    metric_stride: int = 1,
    early_stop_patience: int = 20,
    early_stop_min_delta: float = 0.0001,
    long_topologies: set[str] | None = None,
    python_bin: str | None = None,
) -> dict[str, Any]:
    output_root = Path(output_root)
    long_topologies = long_topologies or {"G-237", "G-670", "G-970"}
    jobs: list[dict[str, Any]] = []
    for topology_row in topology_rows:
        topology_id = str(topology_row["topology_id"])
        role = str(topology_row.get("role") or topology_row.get("selection_bucket") or "")
        for data_seed in [int(seed) for seed in data_seeds]:
            data_dir = output_root / "data" / str(regime) / topology_id / f"data_seed={data_seed:06d}"
            train_bank_path = data_dir / "train_bank.npz"
            for sample_size in sorted(int(size) for size in sample_size_splits):
                split = sample_size_splits[int(sample_size)]
                training_size = int(split["training_size"])
                validation_size = int(split["validation_size"])
                eval_manifest_path = data_dir / f"eval_manifest_sample_size{sample_size:03d}.json"
                eval_manifest = read_eval_manifest(eval_manifest_path)
                job = {
                    "job_id": (
                        f"{topology_id}|data_seed={data_seed:06d}|"
                        f"sample_size={sample_size:03d}|training={training_size:03d}|"
                        f"validation={validation_size:03d}"
                    ),
                    "topology_id": topology_id,
                    "role": role,
                    "regime": str(regime),
                    "protocol": str(protocol),
                    "data_seed": data_seed,
                    "trainer_train_seed_arg": data_seed,
                    "sample_size": sample_size,
                    "training_size": training_size,
                    "validation_size": validation_size,
                    "trainer_train_size_arg": training_size,
                    "train_bank_path": str(train_bank_path),
                    "eval_manifest_path": str(eval_manifest_path),
                    "validation_path": str(eval_manifest.get("validation_path", "")),
                    "test_path": str(eval_manifest.get("test_path", "")),
                    "output_dir": str(
                        output_root
                        / "jobs"
                        / str(regime)
                        / topology_id
                        / f"data_seed={data_seed:06d}"
                        / f"sample_size={sample_size:03d}"
                    ),
                    "expected_training_hash": read_train_prefix_hash(train_bank_path, training_size),
                    "validation_hash": eval_manifest.get("validation_hash"),
                    "test_hash": eval_manifest.get("test_hash"),
                    "theta_seed": int(theta_seed),
                    "gurobi_seed": int(gurobi_seed),
                    "max_epochs": int(max_epochs),
                    "metric_stride": int(metric_stride),
                    "early_stop_patience": int(early_stop_patience),
                    "early_stop_min_delta": float(early_stop_min_delta),
                    "runtime_class": "long" if topology_id in long_topologies else "normal",
                    "status": "planned",
                }
                job["run_one_job_command"] = build_run_one_job_command(job, python_bin=python_bin)
                jobs.append(job)
    return {
        "status": "planned",
        "job_count": len(jobs),
        "topology_count": len(topology_rows),
        "data_seed_count": len(set(int(seed) for seed in data_seeds)),
        "sample_sizes": sorted(int(size) for size in sample_size_splits),
        "jobs": jobs,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topologies-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--regime", default=DEFAULT_REGIME)
    parser.add_argument("--data-seeds", default="101,102,103,104,105")
    parser.add_argument("--sample-size-splits-json", default=None)
    parser.add_argument("--protocol", choices=("screen", "confirm"), default="screen")
    parser.add_argument("--theta-seed", type=int, default=42)
    parser.add_argument("--gurobi-seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=3000)
    parser.add_argument("--metric-stride", type=int, default=1)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0001)
    parser.add_argument("--python", default=None)
    parser.add_argument("--plan-output", type=Path, default=None)
    parser.add_argument("--jobs-csv-output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan = build_plan(
        topology_rows=selected_topology_rows(args.topologies_csv),
        output_root=args.output_root,
        regime=args.regime,
        data_seeds=parse_int_list(args.data_seeds),
        sample_size_splits=parse_sample_size_splits(args.sample_size_splits_json),
        protocol=args.protocol,
        theta_seed=args.theta_seed,
        gurobi_seed=args.gurobi_seed,
        max_epochs=args.max_epochs,
        metric_stride=args.metric_stride,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        python_bin=args.python,
    )
    plan_output = args.plan_output or Path(args.output_root) / "sample_size_plan.json"
    jobs_output = args.jobs_csv_output or Path(args.output_root) / "sample_size_jobs.csv"
    common.atomic_write_json(plan_output, plan)
    write_csv(jobs_output, plan["jobs"], JOB_FIELDS)
    print(json.dumps({"plan_output": str(plan_output), "jobs_csv_output": str(jobs_output), **{k: plan[k] for k in ("job_count", "topology_count", "data_seed_count")}}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
