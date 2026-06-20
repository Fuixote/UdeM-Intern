#!/usr/bin/env python3
"""Evaluate Step3 fixed-topology checkpoints on a fixed eval NPZ set."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
STEP1C_EVALUATE = PROJECT_ROOT / "surrogate_experiment_results" / "Step1c" / "evaluate_models.py"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import fixed_topology_xy_common as common  # noqa: E402


def build_command(args: argparse.Namespace, split_path: Path) -> list[str]:
    python_bin = args.python or os.environ.get("KEP_PYTHON") or sys.executable
    command = [
        python_bin,
        str(STEP1C_EVALUATE),
        "--split_path",
        str(split_path),
        "--out_dir",
        str(args.out_dir),
        "--gurobi_seed",
        str(args.gurobi_seed),
        "--bootstrap_samples",
        str(args.bootstrap_samples),
        "--bootstrap_seed",
        str(args.bootstrap_seed),
        "--weights",
    ]
    command.extend(str(path) for path in args.weights)
    return command


def _validate_expected_hash(name: str, actual: str, expected: str | None) -> None:
    if expected is not None and str(actual) != str(expected):
        raise ValueError(f"expected {name} {expected}, observed {actual}")


def prepare_inputs(args: argparse.Namespace) -> tuple[Path, list[str], dict]:
    materialized = Path(args.out_dir) / "_materialized" / "evaluation"
    test_dir = materialized / "test"
    split_path = materialized / "split.json"
    eval_set = common.read_npz_dataset(args.eval_set)
    test_hash = eval_set["manifest"]["dataset_hash"]
    _validate_expected_hash("test hash", test_hash, args.expected_test_hash)
    test_entries = common.materialize_npz_payloads_to_dir(args.eval_set, test_dir, clear=True)
    common.atomic_write_json(
        split_path,
        {
            "seed": 0,
            "fixed_topology_eval": True,
            "train_pool_size": 0,
            "validation_size": 0,
            "test_size": len(test_entries),
            "train_pool": [],
            "validation": [],
            "test": test_entries,
        },
    )
    input_manifest = {
        "eval_set_path": str(args.eval_set),
        "test_hash": test_hash,
        "expected_test_hash": args.expected_test_hash,
        "paired_job_manifest": None if args.paired_job_manifest is None else str(args.paired_job_manifest),
        "split_path": str(split_path),
        "test_dir": str(test_dir),
        "weights": [str(path) for path in args.weights],
    }
    return split_path, build_command(args, split_path), input_manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, nargs="+", required=True)
    parser.add_argument("--gurobi-seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--python", default=None)
    parser.add_argument("--expected-test-hash", default=None)
    parser.add_argument("--paired-job-manifest", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    split_path, command, input_manifest = prepare_inputs(args)
    input_manifest["command"] = command
    common.atomic_write_json(Path(args.out_dir) / "evaluation_input_manifest.json", input_manifest)
    common.atomic_write_json(
        Path(args.out_dir) / "evaluate_fixed_topology_command.json",
        {"split_path": str(split_path), "command": command},
    )
    if args.dry_run:
        print(json.dumps({"command": command}, indent=2))
        return 0
    return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
