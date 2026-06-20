#!/usr/bin/env python3
"""Prepare or run one paired Step3 fixed-topology training job."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import fixed_topology_xy_common as common  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAIN_2STAGE = SCRIPT_DIR / "train_2stage_fixed_topology.py"
TRAIN_SPOPLUS = SCRIPT_DIR / "train_spoplus_fixed_topology.py"
EVALUATE = SCRIPT_DIR / "evaluate_fixed_topology.py"
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


def _python_bin(selected: str | None = None) -> str:
    return selected or os.environ.get("KEP_PYTHON") or sys.executable


def build_job_commands(
    *,
    train_bank_path: str | Path,
    eval_manifest_path: str | Path,
    eval_manifest: dict[str, Any],
    manifest: dict[str, Any],
    python_bin: str | None = None,
) -> dict[str, list[str]]:
    py = _python_bin(python_bin)
    output_dir = Path(manifest["output_directory"])
    paired_manifest_path = output_dir / "paired_job_manifest.json"
    two_stage_out = output_dir / "2stage"
    spoplus_out = output_dir / "spoplus"
    evaluation_out = output_dir / "evaluation"
    eval_manifest_dir = Path(eval_manifest_path).parent
    validation_path = Path(eval_manifest["validation_path"])
    test_path = Path(eval_manifest["test_path"])
    if not validation_path.is_absolute():
        validation_path = eval_manifest_dir / validation_path
    if not test_path.is_absolute():
        test_path = eval_manifest_dir / test_path
    common_train_args = [
        "--train-bank",
        str(train_bank_path),
        "--validation-set",
        str(validation_path),
        "--train-size",
        str(manifest["train_size"]),
        "--theta-seed",
        str(manifest["theta_seed"]),
        "--max-epochs",
        str(manifest["max_epochs"]),
        "--metric-stride",
        str(manifest["metric_stride"]),
        "--early-stop-patience",
        str(manifest["early_stop_patience"]),
        "--early-stop-min-delta",
        str(manifest["early_stop_min_delta"]),
        "--expected-train-prefix-hash",
        str(manifest["train_prefix_hash"]),
        "--expected-validation-hash",
        str(manifest["validation_hash"]),
        "--paired-job-manifest",
        str(paired_manifest_path),
    ]
    two_stage_weights = two_stage_out / "model_weights" / "2stage_best_by_validation_mse_loss.npz"
    spoplus_weights = spoplus_out / "model_weights" / "spoplus_best_by_validation_spoplus_loss.npz"
    return {
        "2stage": [
            py,
            str(TRAIN_2STAGE),
            *common_train_args,
            "--out-dir",
            str(two_stage_out),
        ],
        "SPO+": [
            py,
            str(TRAIN_SPOPLUS),
            *common_train_args,
            "--out-dir",
            str(spoplus_out),
            "--gurobi-seed",
            str(manifest["gurobi_seed"]),
        ],
        "evaluation": [
            py,
            str(EVALUATE),
            "--eval-set",
            str(test_path),
            "--out-dir",
            str(evaluation_out),
            "--weights",
            str(two_stage_weights),
            str(spoplus_weights),
            "--gurobi-seed",
            str(manifest["gurobi_seed"]),
            "--expected-test-hash",
            str(manifest["test_hash"]),
            "--paired-job-manifest",
            str(paired_manifest_path),
        ],
    }


def run_command(label: str, command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    stdout = getattr(completed, "stdout", "") or ""
    stderr = getattr(completed, "stderr", "") or ""
    if not isinstance(stdout, str):
        stdout = str(stdout)
    if not isinstance(stderr, str):
        stderr = str(stderr)
    return {
        "label": label,
        "command": command,
        "returncode": int(completed.returncode),
        "status": "success" if int(completed.returncode) == 0 else "failed",
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-4000:],
    }


def execute_paired_job(output_dir: Path, manifest: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    commands = manifest["commands"]
    results: list[dict[str, Any]] = []
    status = {
        "job_id": manifest["job_id"],
        "status": "running",
        "2stage status": "pending",
        "SPO+ status": "pending",
        "evaluation status": "pending",
        "commands": commands,
        "results": results,
    }
    for label in ("2stage", "SPO+", "evaluation"):
        result = run_command(label, commands[label])
        results.append(result)
        status[f"{label} status"] = result["status"]
        if result["returncode"] != 0:
            status["status"] = "failed"
            status["failure_reason"] = (
                f"{label} command failed with returncode {result['returncode']}: "
                f"{result.get('stderr_tail', '')}"
            )
            common.atomic_write_json(output_dir / "job_status.json", status)
            return int(result["returncode"]), status
    status["status"] = "success"
    common.atomic_write_json(output_dir / "job_status.json", status)
    return 0, status


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
    parser.add_argument("--python", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
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
    commands = build_job_commands(
        train_bank_path=args.train_bank,
        eval_manifest_path=args.eval_manifest,
        eval_manifest=eval_manifest,
        manifest=manifest,
        python_bin=args.python,
    )
    manifest["commands"] = commands
    args.output_dir.mkdir(parents=True, exist_ok=True)
    common.atomic_write_json(args.output_dir / "paired_job_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if not args.execute:
        return 0
    manifest["status"] = "running"
    common.atomic_write_json(args.output_dir / "paired_job_manifest.json", manifest)
    returncode, status = execute_paired_job(args.output_dir, manifest)
    manifest["methods"]["2stage"]["status"] = status["2stage status"]
    manifest["methods"]["SPO+"]["status"] = status["SPO+ status"]
    manifest["status"] = status["status"]
    manifest["evaluation_status"] = status["evaluation status"]
    common.atomic_write_json(args.output_dir / "paired_job_manifest.json", manifest)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
