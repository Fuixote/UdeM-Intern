#!/usr/bin/env python3
"""Build one independent K18 fixed-topology test bank."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[5]
STEP3_SCRIPTS = PROJECT_ROOT / "surrogate_experiment_results" / "Step3" / "scripts"
if str(STEP3_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(STEP3_SCRIPTS))

import fixed_topology_xy_common as common  # noqa: E402
import sample_fixed_topology_context as context_sampler  # noqa: E402
import sample_fixed_topology_xy as xy_sampler  # noqa: E402


def test_namespace_for_protocol(protocol: str) -> str:
    if protocol == "screen":
        return "screen_test"
    if protocol == "confirm":
        return "confirm_test"
    raise ValueError("protocol must be screen or confirm")


def _provenance_fields(
    *,
    topology_template: dict[str, Any],
    generator_config: dict[str, Any],
    experiment_version: str,
    master_label_seed: int,
) -> dict[str, Any]:
    return {
        "experiment_version": str(experiment_version),
        "master_label_seed": int(master_label_seed),
        "generator_version": str(generator_config["generator_version"]),
        "generator_config_hash": common.generator_config_hash(generator_config),
        **common.template_hashes(topology_template),
    }


def write_test_bank_from_samples(
    *,
    samples: list[dict[str, Any]],
    output_dir: str | Path,
    topology_id: str,
    regime: str,
    protocol: str,
    topology_template: dict[str, Any],
    generator_config: dict[str, Any],
    experiment_version: str,
    master_label_seed: int,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_namespace = test_namespace_for_protocol(protocol)
    sample_rows = [sample["manifest"] for sample in samples]
    namespaces = {str(row.get("split_namespace", "")) for row in sample_rows}
    if namespaces and namespaces != {split_namespace}:
        raise ValueError(f"test samples must use {split_namespace}, observed={sorted(namespaces)}")
    train_seeds = {row.get("train_seed") for row in sample_rows}
    if train_seeds - {None}:
        raise ValueError("test samples must not vary by train_seed")

    provenance = _provenance_fields(
        topology_template=topology_template,
        generator_config=generator_config,
        experiment_version=experiment_version,
        master_label_seed=int(master_label_seed),
    )
    test_hash = common.sample_manifest_hashes(sample_rows)
    dataset_manifest = {
        "topology_id": str(topology_id),
        "regime": str(regime),
        "protocol": str(protocol),
        "split_namespace": split_namespace,
        "train_seed": None,
        "train_seed_sentinel": common.EVAL_TRAIN_SEED_SENTINEL,
        "sample_count": len(samples),
        "test_size": len(samples),
        "samples": sample_rows,
        "dataset_hash": test_hash,
        **provenance,
    }
    test_path = output_dir / "test.npz"
    common.write_npz_dataset(test_path, samples=samples, manifest=dataset_manifest)
    manifest = {
        "topology_id": str(topology_id),
        "regime": str(regime),
        "protocol": str(protocol),
        "test_namespace": split_namespace,
        "test_size": len(samples),
        "test_path": str(test_path),
        "test_hash": test_hash,
        "test_samples": sample_rows,
        **provenance,
    }
    manifest_path = output_dir / "test_manifest.json"
    common.atomic_write_json(manifest_path, manifest)
    return {
        "test_path": str(test_path),
        "test_manifest_path": str(manifest_path),
        "test_hash": test_hash,
        "test_size": len(samples),
    }


def build_test_bank(
    *,
    topology_template: dict[str, Any],
    base_payload: dict[str, Any],
    output_dir: str | Path,
    topology_id: str,
    regime: str,
    protocol: str,
    test_size: int,
    experiment_version: str,
    master_label_seed: int,
    generator_config: dict[str, Any],
) -> dict[str, Any]:
    samples = xy_sampler.generate_samples(
        topology_template=topology_template,
        base_payload=base_payload,
        topology_id=topology_id,
        regime=regime,
        split_namespace=test_namespace_for_protocol(protocol),
        train_seed=None,
        num_samples=int(test_size),
        experiment_version=experiment_version,
        master_label_seed=int(master_label_seed),
        generator_config=generator_config,
    )
    return write_test_bank_from_samples(
        samples=samples,
        output_dir=output_dir,
        topology_id=topology_id,
        regime=regime,
        protocol=protocol,
        topology_template=topology_template,
        generator_config=generator_config,
        experiment_version=experiment_version,
        master_label_seed=int(master_label_seed),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--base-payload", type=Path, required=True)
    parser.add_argument("--topology-id", required=True)
    parser.add_argument("--regime", default=xy_sampler.DEFAULT_REGIME)
    parser.add_argument("--protocol", choices=("screen", "confirm"), default="screen")
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--experiment-version", required=True)
    parser.add_argument("--master-label-seed", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    template = json.loads(args.topology.read_text(encoding="utf-8"))
    base_payload = json.loads(args.base_payload.read_text(encoding="utf-8"))
    generator_config = context_sampler.load_generator_config(args.config)
    result = build_test_bank(
        topology_template=template,
        base_payload=base_payload,
        output_dir=args.output_dir,
        topology_id=args.topology_id,
        regime=args.regime,
        protocol=args.protocol,
        test_size=args.test_size,
        experiment_version=args.experiment_version,
        master_label_seed=args.master_label_seed,
        generator_config=generator_config,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
