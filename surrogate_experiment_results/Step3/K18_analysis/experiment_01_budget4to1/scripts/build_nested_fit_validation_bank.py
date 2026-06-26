#!/usr/bin/env python3
"""Build K18 sample-size 4:1 training/validation artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[5]
STEP3_SCRIPTS = PROJECT_ROOT / "surrogate_experiment_results" / "Step3" / "scripts"
if str(STEP3_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(STEP3_SCRIPTS))

import fixed_topology_xy_common as common  # noqa: E402
import sample_fixed_topology_context as context_sampler  # noqa: E402
import sample_fixed_topology_xy as xy_sampler  # noqa: E402


DEFAULT_SAMPLE_SIZES = (50, 100, 500)


def train_namespace_for_protocol(protocol: str) -> str:
    if protocol == "screen":
        return "screen_train"
    if protocol == "confirm":
        return "confirm_train"
    raise ValueError("protocol must be screen or confirm")


def parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in str(raw).split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one integer")
    return values


def validate_sample_sizes(sample_sizes: list[int] | tuple[int, ...], fit_sample_count: int) -> list[int]:
    sizes = [int(size) for size in sample_sizes]
    if sizes != sorted(sizes) or len(set(sizes)) != len(sizes):
        raise ValueError("sample_sizes must be strictly increasing")
    if sizes[0] <= 0:
        raise ValueError("sample_sizes must be positive")
    if sizes[-1] > int(fit_sample_count):
        raise ValueError("largest sample_size exceeds fit sample count")
    if any(size % 5 != 0 for size in sizes):
        raise ValueError("4:1 sample-size splits require sample_sizes divisible by 5")
    return sizes


def is_validation_fit_index(index: int) -> bool:
    return (int(index) + 1) % 5 == 0


def _fit_index(sample: dict[str, Any], fallback: int) -> int:
    return int(sample.get("manifest", {}).get("sample_index", fallback))


def _provenance_fields(
    *,
    topology_template: dict[str, Any] | None = None,
    generator_config: dict[str, Any] | None = None,
    experiment_version: str | None = None,
    master_label_seed: int | None = None,
) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    if experiment_version is not None:
        fields["experiment_version"] = str(experiment_version)
    if master_label_seed is not None:
        fields["master_label_seed"] = int(master_label_seed)
    if generator_config is not None:
        fields["generator_version"] = str(generator_config["generator_version"])
        fields["generator_config_hash"] = common.generator_config_hash(generator_config)
    if topology_template is not None:
        fields.update(common.template_hashes(topology_template))
    return fields


def split_fit_samples(
    fit_samples: list[dict[str, Any]],
    *,
    sample_sizes: list[int] | tuple[int, ...] = DEFAULT_SAMPLE_SIZES,
) -> dict[str, Any]:
    sizes = validate_sample_sizes(sample_sizes, len(fit_samples))
    max_sample_size = sizes[-1]
    fit_prefix = list(fit_samples[:max_sample_size])
    training_samples = [
        sample
        for fallback, sample in enumerate(fit_prefix)
        if not is_validation_fit_index(_fit_index(sample, fallback))
    ]
    validation_samples_by_sample_size: dict[str, list[dict[str, Any]]] = {}
    sample_size_splits: dict[str, dict[str, Any]] = {}
    fit_role_rows: list[dict[str, Any]] = []
    for fallback, sample in enumerate(fit_prefix):
        fit_index = _fit_index(sample, fallback)
        role = "validation" if is_validation_fit_index(fit_index) else "training"
        fit_role_rows.append(
            {
                "fit_index": fit_index,
                "sample_index": fit_index,
                "sample_id": sample.get("manifest", {}).get("sample_id", ""),
                "role": role,
            }
        )
    for sample_size in sizes:
        prefix = fit_prefix[:sample_size]
        training_indices = [
            _fit_index(sample, fallback)
            for fallback, sample in enumerate(prefix)
            if not is_validation_fit_index(_fit_index(sample, fallback))
        ]
        validation_indices = [
            _fit_index(sample, fallback)
            for fallback, sample in enumerate(prefix)
            if is_validation_fit_index(_fit_index(sample, fallback))
        ]
        validation_samples_by_sample_size[str(sample_size)] = [
            sample
            for fallback, sample in enumerate(prefix)
            if is_validation_fit_index(_fit_index(sample, fallback))
        ]
        sample_size_splits[str(sample_size)] = {
            "sample_size": int(sample_size),
            "training_size": len(training_indices),
            "validation_size": len(validation_indices),
            "trainer_train_size_arg": len(training_indices),
            "training_indices": training_indices,
            "validation_indices": validation_indices,
        }
    return {
        "sample_sizes": sizes,
        "max_sample_size": max_sample_size,
        "training_samples": training_samples,
        "training_prefix_sizes": [
            sample_size_splits[str(sample_size)]["training_size"]
            for sample_size in sizes
        ],
        "validation_samples_by_sample_size": validation_samples_by_sample_size,
        "sample_size_splits": sample_size_splits,
        "fit_role_rows": fit_role_rows,
    }


def _dataset_manifest(
    *,
    samples: list[dict[str, Any]],
    topology_id: str,
    regime: str,
    split_namespace: str,
    data_seed: int,
    protocol: str,
    sample_size: int | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sample_rows = [sample["manifest"] for sample in samples]
    manifest = {
        "topology_id": str(topology_id),
        "regime": str(regime),
        "protocol": str(protocol),
        "split_namespace": str(split_namespace),
        "data_seed": int(data_seed),
        "sample_count": len(samples),
        "samples": sample_rows,
        "dataset_hash": common.sample_manifest_hashes(sample_rows),
    }
    if extra_fields:
        manifest.update(extra_fields)
    if sample_size is not None:
        manifest["sample_size"] = int(sample_size)
    return manifest


def write_artifacts_from_fit_samples(
    *,
    fit_samples: list[dict[str, Any]],
    output_dir: str | Path,
    topology_id: str,
    regime: str,
    data_seed: int,
    protocol: str,
    sample_sizes: list[int] | tuple[int, ...] = DEFAULT_SAMPLE_SIZES,
    test_path: str | Path,
    test_hash: str,
    topology_template: dict[str, Any] | None = None,
    generator_config: dict[str, Any] | None = None,
    experiment_version: str | None = None,
    master_label_seed: int | None = None,
) -> dict[str, Any]:
    split = split_fit_samples(fit_samples, sample_sizes=sample_sizes)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = _provenance_fields(
        topology_template=topology_template,
        generator_config=generator_config,
        experiment_version=experiment_version,
        master_label_seed=master_label_seed,
    )
    source_namespace = train_namespace_for_protocol(protocol)
    validation_scheme = "every_fifth_sample"
    fit_prefix = list(fit_samples[: int(split["max_sample_size"])])
    fit_rows = [sample["manifest"] for sample in fit_prefix]
    fit_bank_hash = common.sample_manifest_hashes(fit_rows)
    fit_manifest_path = output_dir / "fit_manifest.json"
    fit_manifest = {
        "topology_id": str(topology_id),
        "regime": str(regime),
        "protocol": str(protocol),
        "source_namespace": source_namespace,
        "data_seed": int(data_seed),
        "train_seed": int(data_seed),
        "sample_count": len(fit_prefix),
        "max_sample_size": int(split["max_sample_size"]),
        "fit_bank_hash": fit_bank_hash,
        "samples": fit_rows,
        "fit_role_rows": split["fit_role_rows"],
        "validation_scheme": validation_scheme,
        **provenance,
    }
    common.atomic_write_json(fit_manifest_path, fit_manifest)

    training_samples = split["training_samples"]
    training_rows = [sample["manifest"] for sample in training_samples]
    prefix_hashes = {
        str(size): common.prefix_hash(training_rows, size)
        for size in split["training_prefix_sizes"]
    }
    train_bank_manifest = {
        "topology_id": str(topology_id),
        "regime": str(regime),
        "protocol": str(protocol),
        "split_namespace": train_namespace_for_protocol(protocol),
        "data_seed": int(data_seed),
        "train_seed": int(data_seed),
        "max_sample_size": int(split["max_sample_size"]),
        "max_train_size": len(training_samples),
        "sample_sizes": split["sample_sizes"],
        "prefix_sizes": split["training_prefix_sizes"],
        "bank_hash": common.sample_manifest_hashes(training_rows),
        "prefix_hashes": prefix_hashes,
        "samples": training_rows,
        "source_namespace": source_namespace,
        "validation_scheme": validation_scheme,
        "fit_manifest_path": str(fit_manifest_path),
        "fit_bank_hash": fit_bank_hash,
        **provenance,
    }
    train_bank_path = output_dir / "train_bank.npz"
    common.write_npz_dataset(train_bank_path, samples=training_samples, manifest=train_bank_manifest)

    eval_manifest_paths: dict[str, str] = {}
    validation_paths: dict[str, str] = {}
    split_manifest_splits: dict[str, dict[str, Any]] = {}
    for sample_size in split["sample_sizes"]:
        key = str(sample_size)
        validation_samples = split["validation_samples_by_sample_size"][key]
        validation_path = output_dir / f"validation_sample_size{sample_size:03d}.npz"
        validation_manifest = _dataset_manifest(
            samples=validation_samples,
            topology_id=topology_id,
            regime=regime,
            split_namespace=source_namespace,
            data_seed=data_seed,
            protocol=protocol,
            sample_size=sample_size,
            extra_fields={
                "source_namespace": source_namespace,
                "fit_role": "validation",
                "validation_scheme": validation_scheme,
                **provenance,
            },
        )
        common.write_npz_dataset(validation_path, samples=validation_samples, manifest=validation_manifest)
        split_row = dict(split["sample_size_splits"][key])
        split_row["training_hash"] = prefix_hashes[str(split_row["training_size"])]
        split_row["validation_hash"] = validation_manifest["dataset_hash"]
        split_manifest_splits[key] = split_row
        eval_manifest = {
            "topology_id": str(topology_id),
            "regime": str(regime),
            "protocol": str(protocol),
            "data_seed": int(data_seed),
            "train_seed": int(data_seed),
            "sample_size": int(sample_size),
            "training_size": int(split_row["training_size"]),
            "validation_size": int(split_row["validation_size"]),
            "trainer_train_size_arg": int(split_row["trainer_train_size_arg"]),
            "validation_path": validation_path.name,
            "validation_hash": validation_manifest["dataset_hash"],
            "test_path": str(test_path),
            "test_hash": str(test_hash),
            "source_namespace": source_namespace,
            "validation_scheme": validation_scheme,
            "fit_manifest_path": str(fit_manifest_path),
            "fit_bank_hash": fit_bank_hash,
            **provenance,
        }
        eval_manifest_path = output_dir / f"eval_manifest_sample_size{sample_size:03d}.json"
        common.atomic_write_json(eval_manifest_path, eval_manifest)
        eval_manifest_paths[key] = str(eval_manifest_path)
        validation_paths[key] = str(validation_path)

    split_manifest = {
        "topology_id": str(topology_id),
        "regime": str(regime),
        "protocol": str(protocol),
        "data_seed": int(data_seed),
        "sample_sizes": split["sample_sizes"],
        "assignment_rule": "every_fifth_sample_is_validation",
        "training_bank_path": str(train_bank_path),
        "fit_manifest_path": str(fit_manifest_path),
        "fit_bank_hash": fit_bank_hash,
        "sample_size_splits": split_manifest_splits,
        "fit_role_rows": split["fit_role_rows"],
        "source_namespace": source_namespace,
        "validation_scheme": validation_scheme,
        **provenance,
    }
    split_manifest_path = output_dir / "split_manifest.json"
    common.atomic_write_json(split_manifest_path, split_manifest)
    return {
        "train_bank_path": str(train_bank_path),
        "fit_manifest_path": str(fit_manifest_path),
        "split_manifest_path": str(split_manifest_path),
        "validation_paths": validation_paths,
        "eval_manifest_paths": eval_manifest_paths,
        "sample_size_splits": split_manifest_splits,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--base-payload", type=Path, required=True)
    parser.add_argument("--topology-id", required=True)
    parser.add_argument("--regime", default=xy_sampler.DEFAULT_REGIME)
    parser.add_argument("--data-seed", type=int, required=True)
    parser.add_argument("--protocol", choices=("screen", "confirm"), default="screen")
    parser.add_argument("--sample-sizes", default="50,100,500")
    parser.add_argument("--experiment-version", required=True)
    parser.add_argument("--master-label-seed", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, required=True)
    parser.add_argument("--test-hash", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sample_sizes = parse_int_list(args.sample_sizes)
    template = json.loads(args.topology.read_text(encoding="utf-8"))
    base_payload = json.loads(args.base_payload.read_text(encoding="utf-8"))
    generator_config = context_sampler.load_generator_config(args.config)
    fit_samples = xy_sampler.generate_samples(
        topology_template=template,
        base_payload=base_payload,
        topology_id=args.topology_id,
        regime=args.regime,
        split_namespace=train_namespace_for_protocol(args.protocol),
        train_seed=int(args.data_seed),
        num_samples=max(sample_sizes),
        experiment_version=args.experiment_version,
        master_label_seed=int(args.master_label_seed),
        generator_config=generator_config,
    )
    result = write_artifacts_from_fit_samples(
        fit_samples=fit_samples,
        output_dir=args.output_dir,
        topology_id=args.topology_id,
        regime=args.regime,
        data_seed=args.data_seed,
        protocol=args.protocol,
        sample_sizes=sample_sizes,
        test_path=args.test_path,
        test_hash=args.test_hash,
        topology_template=template,
        generator_config=generator_config,
        experiment_version=args.experiment_version,
        master_label_seed=int(args.master_label_seed),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
