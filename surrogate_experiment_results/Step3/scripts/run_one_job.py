#!/usr/bin/env python3
"""Prepare or run one paired Step3 fixed-topology training job."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import fixed_topology_xy_common as common  # noqa: E402


DEFAULT_MAX_EPOCHS = 1500
DEFAULT_METRIC_STRIDE = 1
DEFAULT_EARLY_STOP_PATIENCE = 20
DEFAULT_EARLY_STOP_MIN_DELTA = 0.0001


def theta_init_from_seed(theta_seed: int) -> list[float]:
    rng = np.random.RandomState(int(theta_seed))
    return [float(value) for value in rng.uniform(0.5, 3.5, size=2)]


def prepare_paired_job_manifest(
    *,
    topology_id: str,
    regime: str,
    train_seed: int,
    train_size: int,
    train_bank_manifest: dict[str, Any],
    eval_manifest: dict[str, Any],
    output_dir: str | Path,
    theta_seed: int,
    gurobi_seed: int,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    metric_stride: int = DEFAULT_METRIC_STRIDE,
    early_stop_patience: int = DEFAULT_EARLY_STOP_PATIENCE,
    early_stop_min_delta: float = DEFAULT_EARLY_STOP_MIN_DELTA,
) -> dict[str, Any]:
    prefix_hashes = train_bank_manifest.get("prefix_hashes", {})
    key = str(int(train_size))
    if key not in prefix_hashes:
        raise ValueError(f"train_size={train_size} not available in train bank prefix_hashes")
    theta_init = theta_init_from_seed(theta_seed)
    output_dir = Path(output_dir)
    method_common = {
        "train_prefix_hash": prefix_hashes[key],
        "validation_hash": eval_manifest["validation_hash"],
        "test_hash": eval_manifest["test_hash"],
        "theta_init": theta_init,
    }
    manifest = {
        "job_id": f"{topology_id}|{regime}|seed={int(train_seed):06d}|n={int(train_size)}",
        "topology_id": str(topology_id),
        "regime": str(regime),
        "train_seed": int(train_seed),
        "train_size": int(train_size),
        "train_bank_hash": train_bank_manifest["bank_hash"],
        "train_prefix_hash": prefix_hashes[key],
        "validation_hash": eval_manifest["validation_hash"],
        "test_hash": eval_manifest["test_hash"],
        "theta_seed": int(theta_seed),
        "theta_init": theta_init,
        "gurobi_seed": int(gurobi_seed),
        "max_epochs": int(max_epochs),
        "metric_stride": int(metric_stride),
        "early_stop_patience": int(early_stop_patience),
        "early_stop_min_delta": float(early_stop_min_delta),
        "generator_config_hash": train_bank_manifest.get("generator_config_hash", ""),
        "output_directory": str(output_dir),
        "status": "dry_run_ready",
        "methods": {
            "2stage": {
                **method_common,
                "status": "pending",
                "checkpoint_metric": "validation_mse_loss",
            },
            "SPO+": {
                **method_common,
                "status": "pending",
                "checkpoint_metric": "validation_spoplus_loss",
            },
        },
    }
    validate_paired_job_manifest(manifest)
    return manifest


def validate_paired_job_manifest(manifest: dict[str, Any]) -> None:
    two_stage = manifest["methods"]["2stage"]
    spoplus = manifest["methods"]["SPO+"]
    for key in ("train_prefix_hash", "validation_hash", "test_hash", "theta_init"):
        if two_stage[key] != spoplus[key]:
            raise AssertionError(f"Paired job mismatch for {key}")
    if manifest["train_prefix_hash"] != two_stage["train_prefix_hash"]:
        raise AssertionError("job train_prefix_hash does not match method hash")
    if manifest["validation_hash"] != two_stage["validation_hash"]:
        raise AssertionError("job validation_hash does not match method hash")
    if manifest["test_hash"] != two_stage["test_hash"]:
        raise AssertionError("job test_hash does not match method hash")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-bank", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--topology-id", required=True)
    parser.add_argument("--regime", required=True)
    parser.add_argument("--train-seed", type=int, required=True)
    parser.add_argument("--train-size", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--theta-seed", type=int, default=42)
    parser.add_argument("--gurobi-seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--metric-stride", type=int, default=DEFAULT_METRIC_STRIDE)
    parser.add_argument("--early-stop-patience", type=int, default=DEFAULT_EARLY_STOP_PATIENCE)
    parser.add_argument("--early-stop-min-delta", type=float, default=DEFAULT_EARLY_STOP_MIN_DELTA)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    train_bank = common.read_npz_dataset(args.train_bank)
    eval_manifest = json.loads(args.eval_manifest.read_text(encoding="utf-8"))
    manifest = prepare_paired_job_manifest(
        topology_id=args.topology_id,
        regime=args.regime,
        train_seed=args.train_seed,
        train_size=args.train_size,
        train_bank_manifest=train_bank["manifest"],
        eval_manifest=eval_manifest,
        output_dir=args.output_dir,
        theta_seed=args.theta_seed,
        gurobi_seed=args.gurobi_seed,
        max_epochs=args.max_epochs,
        metric_stride=args.metric_stride,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    common.atomic_write_json(args.output_dir / "paired_job_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if not args.dry_run:
        raise SystemExit(
            "This wrapper currently prepares the paired formal job manifest. "
            "Use train_2stage_fixed_topology.py and train_spoplus_fixed_topology.py "
            "with the emitted hashes for execution."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
