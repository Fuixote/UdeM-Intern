#!/usr/bin/env python3
"""Train Step3 fixed-topology SPO+ on an exact nested-bank prefix."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
STEP1C_SPOPLUS = PROJECT_ROOT / "surrogate_experiment_results" / "Step1c" / "train_spoplus.py"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import fixed_topology_xy_common as common  # noqa: E402


def write_split(train_entries: list[dict], split_path: Path) -> None:
    common.atomic_write_json(
        split_path,
        {
            "seed": 0,
            "fixed_order_prefix": True,
            "train_pool_size": len(train_entries),
            "validation_size": 0,
            "test_size": 0,
            "train_pool": train_entries,
            "validation": [],
            "test": [],
        },
    )


def build_command(args: argparse.Namespace, split_path: Path, validation_dir: Path) -> list[str]:
    python_bin = args.python or os.environ.get("KEP_PYTHON") or sys.executable
    return [
        python_bin,
        str(STEP1C_SPOPLUS),
        "--split_path",
        str(split_path),
        "--validation_data_dir",
        str(validation_dir),
        "--out_dir",
        str(args.out_dir),
        "--train_size",
        str(args.train_size),
        "--subset_seed",
        "0",
        "--theta_seed",
        str(args.theta_seed),
        "--gurobi_seed",
        str(args.gurobi_seed),
        "--n_epochs",
        str(args.max_epochs),
        "--lr",
        str(args.lr),
        "--metric_stride",
        str(args.metric_stride),
        "--early_stop_metric",
        "validation_spoplus_loss",
        "--early_stop_patience",
        str(args.early_stop_patience),
        "--early_stop_min_delta",
        str(args.early_stop_min_delta),
    ]


def _validate_expected_hash(name: str, actual: str, expected: str | None) -> None:
    if expected is not None and str(actual) != str(expected):
        raise ValueError(f"expected {name} {expected}, observed {actual}")


def prepare_inputs(args: argparse.Namespace) -> tuple[Path, Path, list[str], dict]:
    materialized = Path(args.out_dir) / "_materialized" / "spoplus"
    train_dir = materialized / "train_prefix"
    validation_dir = materialized / "validation"
    split_path = materialized / "split.json"
    train_bank = common.read_npz_dataset(args.train_bank)
    train_manifest = train_bank["manifest"]
    prefix_key = str(int(args.train_size))
    prefix_hashes = train_manifest.get("prefix_hashes", {})
    if prefix_key not in prefix_hashes:
        raise ValueError(f"train_size={args.train_size} missing from train bank prefix_hashes")
    train_prefix_hash = prefix_hashes[prefix_key]
    validation = common.read_npz_dataset(args.validation_set)
    validation_hash = validation["manifest"]["dataset_hash"]
    _validate_expected_hash("train prefix hash", train_prefix_hash, args.expected_train_prefix_hash)
    _validate_expected_hash("validation hash", validation_hash, args.expected_validation_hash)
    train_entries = common.materialize_npz_payloads_to_dir(
        args.train_bank,
        train_dir,
        limit=args.train_size,
        clear=True,
    )
    common.materialize_npz_payloads_to_dir(args.validation_set, validation_dir, clear=True)
    write_split(train_entries, split_path)
    input_manifest = {
        "method": "SPO+",
        "train_bank_path": str(args.train_bank),
        "validation_set_path": str(args.validation_set),
        "train_size": int(args.train_size),
        "train_bank_hash": train_manifest.get("bank_hash"),
        "train_prefix_hash": train_prefix_hash,
        "validation_hash": validation_hash,
        "expected_train_prefix_hash": args.expected_train_prefix_hash,
        "expected_validation_hash": args.expected_validation_hash,
        "paired_job_manifest": None if args.paired_job_manifest is None else str(args.paired_job_manifest),
        "split_path": str(split_path),
        "train_dir": str(train_dir),
        "validation_dir": str(validation_dir),
    }
    return split_path, validation_dir, build_command(args, split_path, validation_dir), input_manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-bank", type=Path, required=True)
    parser.add_argument("--validation-set", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-size", type=int, required=True)
    parser.add_argument("--theta-seed", type=int, default=42)
    parser.add_argument("--gurobi-seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=1500)
    parser.add_argument("--metric-stride", type=int, default=1)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0001)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--python", default=None)
    parser.add_argument("--expected-train-prefix-hash", default=None)
    parser.add_argument("--expected-validation-hash", default=None)
    parser.add_argument("--paired-job-manifest", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    split_path, validation_dir, command, input_manifest = prepare_inputs(args)
    input_manifest["command"] = command
    common.atomic_write_json(Path(args.out_dir) / "method_input_manifest.json", input_manifest)
    common.atomic_write_json(
        Path(args.out_dir) / "train_spoplus_fixed_topology_command.json",
        {
            "split_path": str(split_path),
            "validation_dir": str(validation_dir),
            "command": command,
            "train_size": int(args.train_size),
        },
    )
    if args.dry_run:
        print(json.dumps({"command": command}, indent=2))
        return 0
    return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
