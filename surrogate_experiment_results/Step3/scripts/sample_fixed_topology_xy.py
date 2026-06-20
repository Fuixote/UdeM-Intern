#!/usr/bin/env python3
"""Generate Step3 fixed-topology full-(X,y) samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DECISION_ANALYSIS_DIR = PROJECT_ROOT / "surrogate_experiment_results" / "decision_analysis" / "scripts"
for path in (SCRIPT_DIR, DECISION_ANALYSIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import fixed_topology_xy_common as common  # noqa: E402
import sample_fixed_topology_context as context_sampler  # noqa: E402
from audit_fixed_topology_label_seed import relabel_payload_step2c  # noqa: E402


DEFAULT_REGIME = "step2c_poly_d8_mult_eps050"


def epsilon_bar_for_regime(regime: str, generator_config: dict[str, Any]) -> float:
    label_config = generator_config.get("label", {}) if isinstance(generator_config, dict) else {}
    if "epsilon_bar" in label_config:
        return float(label_config["epsilon_bar"])
    if str(regime).endswith("eps050"):
        return 0.5
    raise ValueError(f"Cannot infer epsilon_bar for regime {regime!r}; set label.epsilon_bar")


def generate_sample(
    *,
    topology_template: dict[str, Any],
    base_payload: dict[str, Any],
    topology_id: str,
    regime: str,
    split_namespace: str,
    train_seed: int | None,
    sample_index: int,
    experiment_version: str,
    master_label_seed: int,
    generator_config: dict[str, Any],
    context_seed_override: int | None = None,
    label_noise_seed_override: int | None = None,
) -> dict[str, Any]:
    common.normalize_train_seed(split_namespace, train_seed)
    seeds = common.derive_sample_seeds(
        experiment_version=experiment_version,
        regime=regime,
        topology_id=topology_id,
        split_namespace=split_namespace,
        train_seed=train_seed,
        sample_index=sample_index,
        master_label_seed=master_label_seed,
    )
    context_seed = int(context_seed_override if context_seed_override is not None else seeds["context_seed"])
    label_noise_seed = int(
        label_noise_seed_override
        if label_noise_seed_override is not None
        else seeds["label_noise_seed"]
    )

    context_payload = context_sampler.sample_context(
        topology_template=topology_template,
        base_payload=base_payload,
        context_seed=context_seed,
        generator_config=generator_config,
    )
    sample_payload = relabel_payload_step2c(
        context_payload,
        label_seed=label_noise_seed,
        epsilon_bar=epsilon_bar_for_regime(regime, generator_config),
    )
    common.assert_structure_matches_template(sample_payload, topology_template)
    common.assert_context_label_consistency(sample_payload, topology_template)

    X = common.feature_matrix_from_payload(sample_payload, topology_template)
    y = common.labels_from_payload(sample_payload, topology_template)
    hashes = common.template_hashes(topology_template)
    sample_id = (
        f"{topology_id}|{regime}|{split_namespace}|"
        f"{common.seed_key_train_seed(split_namespace, train_seed)}|{int(sample_index):06d}"
    )
    manifest = {
        "sample_id": sample_id,
        "topology_id": str(topology_id),
        "regime": str(regime),
        "split_namespace": str(split_namespace),
        "train_seed": None if train_seed is None else int(train_seed),
        "sample_index": int(sample_index),
        "context_seed": int(context_seed),
        "label_noise_seed": int(label_noise_seed),
        "x_hash": common.matrix_hash(X),
        "label_hash": common.vector_hash(y),
        "generator_version": str(generator_config["generator_version"]),
        "generator_config_hash": common.generator_config_hash(generator_config),
        "master_label_seed": int(master_label_seed),
        "edge_count": int(X.shape[0]),
        **hashes,
    }
    sample_payload.setdefault("metadata", {})
    sample_payload["metadata"]["step3_fixed_topology_xy"] = manifest
    return {
        "payload": sample_payload,
        "X": X,
        "y": y,
        "manifest": manifest,
    }


def generate_samples(
    *,
    topology_template: dict[str, Any],
    base_payload: dict[str, Any],
    topology_id: str,
    regime: str,
    split_namespace: str,
    train_seed: int | None,
    num_samples: int,
    experiment_version: str,
    master_label_seed: int,
    generator_config: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        generate_sample(
            topology_template=topology_template,
            base_payload=base_payload,
            topology_id=topology_id,
            regime=regime,
            split_namespace=split_namespace,
            train_seed=train_seed,
            sample_index=sample_index,
            experiment_version=experiment_version,
            master_label_seed=master_label_seed,
            generator_config=generator_config,
        )
        for sample_index in range(int(num_samples))
    ]


def dataset_manifest(
    *,
    topology_template: dict[str, Any],
    samples: list[dict[str, Any]],
    topology_id: str,
    regime: str,
    split_namespace: str,
    train_seed: int | None,
    experiment_version: str,
    master_label_seed: int,
    generator_config: dict[str, Any],
) -> dict[str, Any]:
    sample_rows = [sample["manifest"] for sample in samples]
    return {
        "topology_id": str(topology_id),
        "regime": str(regime),
        "split_namespace": str(split_namespace),
        "train_seed": None if train_seed is None else int(train_seed),
        "experiment_version": str(experiment_version),
        "master_label_seed": int(master_label_seed),
        "sample_count": len(samples),
        "samples": sample_rows,
        "dataset_hash": common.sample_manifest_hashes(sample_rows),
        "generator_version": str(generator_config["generator_version"]),
        "generator_config_hash": common.generator_config_hash(generator_config),
        **common.template_hashes(topology_template),
    }


def write_samples(
    *,
    output_path: str | Path,
    topology_template: dict[str, Any],
    samples: list[dict[str, Any]],
    topology_id: str,
    regime: str,
    split_namespace: str,
    train_seed: int | None,
    experiment_version: str,
    master_label_seed: int,
    generator_config: dict[str, Any],
) -> dict[str, Any]:
    manifest = dataset_manifest(
        topology_template=topology_template,
        samples=samples,
        topology_id=topology_id,
        regime=regime,
        split_namespace=split_namespace,
        train_seed=train_seed,
        experiment_version=experiment_version,
        master_label_seed=master_label_seed,
        generator_config=generator_config,
    )
    common.write_npz_dataset(output_path, samples=samples, manifest=manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--base-payload", type=Path, required=True)
    parser.add_argument("--regime", default=DEFAULT_REGIME)
    parser.add_argument("--split-namespace", required=True)
    parser.add_argument("--train-seed", type=int, default=None)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--experiment-version", required=True)
    parser.add_argument("--master-label-seed", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    topology_template = json.loads(args.topology.read_text(encoding="utf-8"))
    base_payload = json.loads(args.base_payload.read_text(encoding="utf-8"))
    generator_config = context_sampler.load_generator_config(args.config)
    topology_id = str(topology_template.get("topology_id", args.topology.parent.name))
    samples = generate_samples(
        topology_template=topology_template,
        base_payload=base_payload,
        topology_id=topology_id,
        regime=args.regime,
        split_namespace=args.split_namespace,
        train_seed=args.train_seed,
        num_samples=args.num_samples,
        experiment_version=args.experiment_version,
        master_label_seed=args.master_label_seed,
        generator_config=generator_config,
    )
    manifest = write_samples(
        output_path=args.output,
        topology_template=topology_template,
        samples=samples,
        topology_id=topology_id,
        regime=args.regime,
        split_namespace=args.split_namespace,
        train_seed=args.train_seed,
        experiment_version=args.experiment_version,
        master_label_seed=args.master_label_seed,
        generator_config=generator_config,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
