#!/usr/bin/env python3
"""Build one deterministic nested Step3 fixed-topology training bank."""

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


def train_namespace_for_protocol(protocol: str) -> str:
    if protocol == "confirm":
        return "confirm_train"
    if protocol == "screen":
        return "screen_train"
    raise ValueError("protocol must be screen or confirm")


def validate_prefix_sizes(prefix_sizes: tuple[int, ...] | list[int], max_train_size: int) -> list[int]:
    sizes = [int(size) for size in prefix_sizes]
    if not sizes:
        raise ValueError("prefix_sizes must be non-empty")
    if sizes != sorted(sizes):
        raise ValueError("prefix_sizes must be strictly increasing")
    if len(set(sizes)) != len(sizes):
        raise ValueError("prefix_sizes must be strictly increasing")
    if sizes[-1] != int(max_train_size):
        raise ValueError("largest prefix_sizes entry must equal max_train_size")
    if sizes[0] <= 0:
        raise ValueError("prefix_sizes must be positive")
    return sizes


def build_manifest(
    *,
    topology_template: dict[str, Any],
    samples: list[dict[str, Any]],
    topology_id: str,
    regime: str,
    train_seed: int,
    protocol: str,
    split_namespace: str,
    max_train_size: int,
    prefix_sizes: list[int],
    experiment_version: str,
    master_label_seed: int,
    generator_config: dict[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    sample_rows = [sample["manifest"] for sample in samples]
    prefix_hashes = {
        str(size): common.prefix_hash(sample_rows, int(size))
        for size in prefix_sizes
    }
    return {
        "topology_id": str(topology_id),
        "regime": str(regime),
        "protocol": str(protocol),
        "split_namespace": str(split_namespace),
        "train_seed": int(train_seed),
        "experiment_version": str(experiment_version),
        "master_label_seed": int(master_label_seed),
        "max_train_size": int(max_train_size),
        "prefix_sizes": prefix_sizes,
        "D50": f"bank[0:{prefix_sizes[0]}]",
        "D100": f"bank[0:{prefix_sizes[1]}]" if len(prefix_sizes) > 1 else "",
        "D500": f"bank[0:{prefix_sizes[-1]}]",
        "bank_hash": common.sample_manifest_hashes(sample_rows),
        "prefix_hashes": prefix_hashes,
        "samples": sample_rows,
        "output_path": str(output_path),
        "generator_version": str(generator_config["generator_version"]),
        "generator_config_hash": common.generator_config_hash(generator_config),
        **common.template_hashes(topology_template),
    }


def verify_nested_prefixes(manifest: dict[str, Any]) -> bool:
    samples = list(manifest.get("samples", []))
    prefix_hashes = manifest.get("prefix_hashes", {})
    for raw_size, expected_hash in prefix_hashes.items():
        size = int(raw_size)
        if size > len(samples):
            return False
        if common.prefix_hash(samples, size) != expected_hash:
            return False
    max_size = int(manifest.get("max_train_size", len(samples)))
    return common.sample_manifest_hashes(samples[:max_size]) == manifest.get("bank_hash")


def build_nested_train_bank(
    *,
    topology_template: dict[str, Any],
    base_payload: dict[str, Any],
    output_path: str | Path,
    topology_id: str,
    regime: str,
    train_seed: int,
    max_train_size: int = 500,
    prefix_sizes: tuple[int, ...] | list[int] = (50, 100, 500),
    experiment_version: str,
    master_label_seed: int,
    generator_config: dict[str, Any],
    mode: str = "materialized",
    protocol: str = "confirm",
) -> dict[str, Any]:
    if mode not in {"materialized", "lazy", "cache_on_first_use"}:
        raise ValueError("mode must be materialized, lazy, or cache_on_first_use")
    split_namespace = train_namespace_for_protocol(protocol)
    sizes = validate_prefix_sizes(prefix_sizes, max_train_size)
    samples = xy_sampler.generate_samples(
        topology_template=topology_template,
        base_payload=base_payload,
        topology_id=topology_id,
        regime=regime,
        split_namespace=split_namespace,
        train_seed=int(train_seed),
        num_samples=int(max_train_size),
        experiment_version=experiment_version,
        master_label_seed=int(master_label_seed),
        generator_config=generator_config,
    )
    manifest = build_manifest(
        topology_template=topology_template,
        samples=samples,
        topology_id=topology_id,
        regime=regime,
        train_seed=int(train_seed),
        protocol=protocol,
        split_namespace=split_namespace,
        max_train_size=int(max_train_size),
        prefix_sizes=sizes,
        experiment_version=experiment_version,
        master_label_seed=int(master_label_seed),
        generator_config=generator_config,
        output_path=output_path,
    )
    if not verify_nested_prefixes(manifest):
        raise AssertionError("Nested prefix verification failed")
    if mode == "materialized":
        common.write_npz_dataset(output_path, samples=samples, manifest=manifest)
    return manifest


def load_train_bank(path: str | Path) -> dict[str, Any]:
    return common.read_npz_dataset(path)


def parse_prefix_sizes(raw: str) -> tuple[int, ...]:
    return tuple(int(part) for part in str(raw).split(",") if part)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--base-payload", type=Path, required=True)
    parser.add_argument("--topology-id", required=True)
    parser.add_argument("--regime", default=xy_sampler.DEFAULT_REGIME)
    parser.add_argument("--train-seed", type=int, required=True)
    parser.add_argument("--protocol", choices=("screen", "confirm"), default="confirm")
    parser.add_argument("--max-train-size", type=int, default=500)
    parser.add_argument("--prefix-sizes", default="50,100,500")
    parser.add_argument("--experiment-version", required=True)
    parser.add_argument("--master-label-seed", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", default="materialized")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    template = json.loads(args.topology.read_text(encoding="utf-8"))
    base_payload = json.loads(args.base_payload.read_text(encoding="utf-8"))
    generator_config = context_sampler.load_generator_config(args.config)
    manifest = build_nested_train_bank(
        topology_template=template,
        base_payload=base_payload,
        output_path=args.output,
        topology_id=args.topology_id,
        regime=args.regime,
        train_seed=args.train_seed,
        protocol=args.protocol,
        max_train_size=args.max_train_size,
        prefix_sizes=parse_prefix_sizes(args.prefix_sizes),
        experiment_version=args.experiment_version,
        master_label_seed=args.master_label_seed,
        generator_config=generator_config,
        mode=args.mode,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
