#!/usr/bin/env python3
"""Validate and plan Step3 formal confirmation jobs without launching them."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import audit_fixed_topology_xy  # noqa: E402
import fixed_topology_xy_common as common  # noqa: E402
import sample_fixed_topology_context as context_sampler  # noqa: E402


def load_json_or_simple_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if config_path.suffix.lower() == ".json":
        return json.loads(config_path.read_text(encoding="utf-8"))
    return context_sampler.load_simple_yaml(config_path)


def validate_confirmation_config(
    config: dict[str, Any],
    *,
    generator_config: dict[str, Any],
) -> None:
    audit_fixed_topology_xy.assert_formal_config_locked(generator_config)
    training = config.get("training", {})
    if training.get("nested_train_sets") is False:
        raise ValueError("formal confirmation requires nested_train_sets=true")
    eval_config = config.get("evaluation", {})
    if eval_config.get("validation_namespace", "confirm_validation") != "confirm_validation":
        raise ValueError("formal validation namespace must be confirm_validation")
    if eval_config.get("test_namespace", "confirm_test") != "confirm_test":
        raise ValueError("formal test namespace must be confirm_test")


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def topology_entries(config: dict[str, Any]) -> list[dict[str, Any]]:
    if "topologies" in config:
        entries = config["topologies"]
    else:
        manifest_config = config.get("topology_manifest", {})
        manifest_path = manifest_config.get("path")
        if not manifest_path:
            raise ValueError("confirmation plan requires topologies or topology_manifest.path")
        manifest = _load_json(manifest_path)
        entries = manifest.get("topologies", manifest.get("topology_ids", []))
    normalized: list[dict[str, Any]] = []
    for entry in entries:
        if isinstance(entry, str):
            normalized.append({"topology_id": entry})
        else:
            normalized.append(dict(entry))
    if not normalized:
        raise ValueError("confirmation plan requires at least one topology")
    return normalized


def _train_bank_path_for_entry(
    *,
    config: dict[str, Any],
    entry: dict[str, Any],
    regime: str,
    train_seed: int,
) -> Path:
    if "train_bank_path_template" in entry:
        return Path(str(entry["train_bank_path_template"]).format(
            topology_id=entry["topology_id"],
            regime=regime,
            train_seed=int(train_seed),
        ))
    if "train_bank_dir" in entry:
        return Path(entry["train_bank_dir"]) / f"train_seed={int(train_seed):06d}.npz"
    data_config = config.get("data", {})
    train_bank_root = data_config.get("train_bank_root")
    if train_bank_root:
        return Path(train_bank_root) / str(regime) / str(entry["topology_id"]) / f"train_seed={int(train_seed):06d}.npz"
    raise ValueError(f"topology {entry.get('topology_id')} requires train_bank_dir or train_bank_path_template")


def _eval_manifest_path_for_entry(
    *,
    config: dict[str, Any],
    entry: dict[str, Any],
    regime: str,
) -> Path:
    if "eval_manifest" in entry:
        return Path(entry["eval_manifest"])
    if "eval_manifest_path" in entry:
        return Path(entry["eval_manifest_path"])
    data_config = config.get("data", {})
    eval_root = data_config.get("eval_root")
    if eval_root:
        return Path(eval_root) / str(regime) / str(entry["topology_id"]) / "eval_manifest.json"
    raise ValueError(f"topology {entry.get('topology_id')} requires eval_manifest")


def _output_dir_for_entry(
    *,
    config: dict[str, Any],
    entry: dict[str, Any],
    regime: str,
    train_seed: int,
    train_size: int,
) -> Path:
    base = Path(entry.get("output_dir") or config.get("output_root") or config.get("run_root") or "surrogate_experiment_results/Step3/runs/confirmation")
    return base / str(regime) / f"train_seed={int(train_seed):06d}" / f"train_size={int(train_size)}"


def _read_prefix_hash(train_bank_path: Path, train_size: int) -> str | None:
    if not train_bank_path.exists():
        return None
    bank = common.read_npz_dataset(train_bank_path)
    return bank["manifest"].get("prefix_hashes", {}).get(str(int(train_size)))


def _read_eval_hashes(eval_manifest_path: Path) -> dict[str, Any]:
    if not eval_manifest_path.exists():
        return {
            "validation_path": None,
            "test_path": None,
            "validation_hash": None,
            "test_hash": None,
        }
    manifest = _load_json(eval_manifest_path)
    return {
        "validation_path": manifest.get("validation_path"),
        "test_path": manifest.get("test_path"),
        "validation_hash": manifest.get("validation_hash"),
        "test_hash": manifest.get("test_hash"),
    }


def build_confirmation_job_plan(
    config: dict[str, Any],
    *,
    generator_config: dict[str, Any],
) -> dict[str, Any]:
    validate_confirmation_config(config, generator_config=generator_config)
    regimes = list(config.get("regimes", []))
    if not regimes:
        raise ValueError("confirmation plan requires at least one regime")
    training = config.get("training", {})
    train_sizes = [int(size) for size in training.get("train_sizes", [50, 100, 500])]
    train_seed_start = int(training.get("train_seed_start", 1))
    train_seed_count = int(training.get("train_seed_count", 1000))
    entries = topology_entries(config)
    jobs: list[dict[str, Any]] = []
    for regime in regimes:
        for entry in entries:
            topology_id = str(entry["topology_id"])
            eval_manifest_path = _eval_manifest_path_for_entry(config=config, entry=entry, regime=str(regime))
            eval_hashes = _read_eval_hashes(eval_manifest_path)
            for train_seed in range(train_seed_start, train_seed_start + train_seed_count):
                train_bank_path = _train_bank_path_for_entry(
                    config=config,
                    entry=entry,
                    regime=str(regime),
                    train_seed=train_seed,
                )
                for train_size in train_sizes:
                    jobs.append(
                        {
                            "topology_id": topology_id,
                            "regime": str(regime),
                            "train_seed": int(train_seed),
                            "train_size": int(train_size),
                            "train_bank_path": str(train_bank_path),
                            "validation_path": eval_hashes["validation_path"],
                            "test_path": eval_hashes["test_path"],
                            "eval_manifest_path": str(eval_manifest_path),
                            "output_dir": str(
                                _output_dir_for_entry(
                                    config=config,
                                    entry=entry,
                                    regime=str(regime),
                                    train_seed=train_seed,
                                    train_size=train_size,
                                )
                            ),
                            "expected_train_prefix_hash": _read_prefix_hash(train_bank_path, train_size),
                            "validation_hash": eval_hashes["validation_hash"],
                            "test_hash": eval_hashes["test_hash"],
                        }
                    )
    return {
        "status": "planned",
        "plan_only": True,
        "job_count": len(jobs),
        "topology_count": len(entries),
        "regime_count": len(regimes),
        "train_seed_count": train_seed_count,
        "train_sizes": train_sizes,
        "jobs": jobs,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--context-generator-config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_json_or_simple_yaml(args.config)
    generator_config = context_sampler.load_generator_config(args.context_generator_config)
    plan = build_confirmation_job_plan(config, generator_config=generator_config)
    if args.output:
        common.atomic_write_json(args.output, plan)
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
