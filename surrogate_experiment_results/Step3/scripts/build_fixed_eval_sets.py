#!/usr/bin/env python3
"""Build fixed Step3 validation/test sets for one fixed topology."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import fixed_topology_xy_common as common  # noqa: E402
import sample_fixed_topology_context as context_sampler  # noqa: E402
import sample_fixed_topology_xy as xy_sampler  # noqa: E402


def eval_namespaces_for_protocol(protocol: str) -> tuple[str, str]:
    if protocol == "confirm":
        return "confirm_validation", "confirm_test"
    if protocol == "screen":
        return "screen_validation", "screen_test"
    raise ValueError("protocol must be screen or confirm")


def expected_eval_manifest(
    *,
    topology_template: dict[str, Any],
    topology_id: str,
    regime: str,
    validation_size: int,
    test_size: int,
    protocol: str,
    validation_namespace: str,
    test_namespace: str,
    experiment_version: str,
    master_label_seed: int,
    generator_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "topology_id": str(topology_id),
        "regime": str(regime),
        "validation_size": int(validation_size),
        "test_size": int(test_size),
        "protocol": str(protocol),
        "validation_namespace": str(validation_namespace),
        "test_namespace": str(test_namespace),
        "train_seed_sentinel": common.EVAL_TRAIN_SEED_SENTINEL,
        "experiment_version": str(experiment_version),
        "master_label_seed": int(master_label_seed),
        "generator_version": str(generator_config["generator_version"]),
        "generator_config_hash": common.generator_config_hash(generator_config),
        **common.template_hashes(topology_template),
    }


def compatible_existing_manifest(path: Path, expected: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    existing = json.loads(path.read_text(encoding="utf-8"))
    checked_keys = [
        "topology_id",
        "regime",
        "validation_size",
        "test_size",
        "protocol",
        "validation_namespace",
        "test_namespace",
        "train_seed_sentinel",
        "experiment_version",
        "master_label_seed",
        "generator_config_hash",
        "topology_hash",
        "arc_order_hash",
        "feasible_set_hash",
    ]
    for key in checked_keys:
        existing_value = existing.get(key)
        if key == "protocol":
            existing_value = existing.get(key, "confirm")
        if existing_value != expected.get(key):
            return False
    return True


def build_fixed_eval_sets_for_topology(
    *,
    topology_template: dict[str, Any],
    base_payload: dict[str, Any],
    output_dir: str | Path,
    topology_id: str,
    regime: str,
    validation_size: int,
    test_size: int,
    experiment_version: str,
    master_label_seed: int,
    generator_config: dict[str, Any],
    force: bool = False,
    dry_run: bool = False,
    protocol: str = "confirm",
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    validation_namespace, test_namespace = eval_namespaces_for_protocol(protocol)
    validation_path = output_dir / "validation.npz"
    test_path = output_dir / "test.npz"
    manifest_path = output_dir / "eval_manifest.json"
    expected = expected_eval_manifest(
        topology_template=topology_template,
        topology_id=topology_id,
        regime=regime,
        validation_size=validation_size,
        test_size=test_size,
        protocol=protocol,
        validation_namespace=validation_namespace,
        test_namespace=test_namespace,
        experiment_version=experiment_version,
        master_label_seed=master_label_seed,
        generator_config=generator_config,
    )
    if dry_run:
        return {
            "dry_run": True,
            "validation_path": str(validation_path),
            "test_path": str(test_path),
            "eval_manifest_path": str(manifest_path),
            "expected_manifest": expected,
        }
    if manifest_path.exists() or validation_path.exists() or test_path.exists():
        if not force:
            if compatible_existing_manifest(manifest_path, expected):
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
                return {
                    "dry_run": False,
                    "validation_path": str(validation_path),
                    "test_path": str(test_path),
                    "eval_manifest_path": str(manifest_path),
                    "validation_hash": existing["validation_hash"],
                    "test_hash": existing["test_hash"],
                    "manifest": existing,
                }
            raise ValueError("existing eval set does not match requested config; pass force to overwrite")

    validation_samples = xy_sampler.generate_samples(
        topology_template=topology_template,
        base_payload=base_payload,
        topology_id=topology_id,
        regime=regime,
        split_namespace=validation_namespace,
        train_seed=None,
        num_samples=int(validation_size),
        experiment_version=experiment_version,
        master_label_seed=int(master_label_seed),
        generator_config=generator_config,
    )
    test_samples = xy_sampler.generate_samples(
        topology_template=topology_template,
        base_payload=base_payload,
        topology_id=topology_id,
        regime=regime,
        split_namespace=test_namespace,
        train_seed=None,
        num_samples=int(test_size),
        experiment_version=experiment_version,
        master_label_seed=int(master_label_seed),
        generator_config=generator_config,
    )
    validation_manifest = xy_sampler.dataset_manifest(
        topology_template=topology_template,
        samples=validation_samples,
        topology_id=topology_id,
        regime=regime,
        split_namespace=validation_namespace,
        train_seed=None,
        experiment_version=experiment_version,
        master_label_seed=int(master_label_seed),
        generator_config=generator_config,
    )
    test_manifest = xy_sampler.dataset_manifest(
        topology_template=topology_template,
        samples=test_samples,
        topology_id=topology_id,
        regime=regime,
        split_namespace=test_namespace,
        train_seed=None,
        experiment_version=experiment_version,
        master_label_seed=int(master_label_seed),
        generator_config=generator_config,
    )
    common.write_npz_dataset(validation_path, samples=validation_samples, manifest=validation_manifest)
    common.write_npz_dataset(test_path, samples=test_samples, manifest=test_manifest)
    manifest = {
        **expected,
        "validation_path": str(validation_path),
        "test_path": str(test_path),
        "validation_hash": validation_manifest["dataset_hash"],
        "test_hash": test_manifest["dataset_hash"],
        "validation_samples": validation_manifest["samples"],
        "test_samples": test_manifest["samples"],
    }
    common.atomic_write_json(manifest_path, manifest)
    return {
        "dry_run": False,
        "validation_path": str(validation_path),
        "test_path": str(test_path),
        "eval_manifest_path": str(manifest_path),
        "validation_hash": manifest["validation_hash"],
        "test_hash": manifest["test_hash"],
        "manifest": manifest,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--base-payload", type=Path, required=True)
    parser.add_argument("--topology-id", required=True)
    parser.add_argument("--regime", default=xy_sampler.DEFAULT_REGIME)
    parser.add_argument("--protocol", choices=("screen", "confirm"), default="confirm")
    parser.add_argument("--validation-size", type=int, required=True)
    parser.add_argument("--test-size", type=int, required=True)
    parser.add_argument("--experiment-version", required=True)
    parser.add_argument("--master-label-seed", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    template = json.loads(args.topology.read_text(encoding="utf-8"))
    base_payload = json.loads(args.base_payload.read_text(encoding="utf-8"))
    generator_config = context_sampler.load_generator_config(args.config)
    result = build_fixed_eval_sets_for_topology(
        topology_template=template,
        base_payload=base_payload,
        output_dir=args.output_dir,
        topology_id=args.topology_id,
        regime=args.regime,
        protocol=args.protocol,
        validation_size=args.validation_size,
        test_size=args.test_size,
        experiment_version=args.experiment_version,
        master_label_seed=args.master_label_seed,
        generator_config=generator_config,
        force=args.force,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
