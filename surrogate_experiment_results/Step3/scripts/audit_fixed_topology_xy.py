#!/usr/bin/env python3
"""Audit Step3 fixed-topology full-(X,y) data artifacts."""

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


def assert_formal_config_locked(generator_config: dict[str, Any]) -> None:
    if generator_config.get("status") != "locked":
        raise ValueError("formal confirmation requires generator_config status='locked'")


def _append_failure(failures: list[str], name: str) -> None:
    if name not in failures:
        failures.append(name)


def audit_sample_rows(
    *,
    sample_rows: list[dict[str, Any]],
    payloads: list[dict[str, Any]],
    topology_template: dict[str, Any],
    base_payload: dict[str, Any],
    generator_config: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    expected_hashes = common.template_hashes(topology_template)
    expected_config_hash = common.generator_config_hash(generator_config)
    seen_context_hashes: set[str] = set()
    seen_sample_keys: set[tuple[str, str, str]] = set()
    for row, payload in zip(sample_rows, payloads):
        for key, expected in expected_hashes.items():
            if expected and row.get(key) != expected:
                _append_failure(failures, f"{key}_mismatch")
        if row.get("generator_config_hash") != expected_config_hash:
            _append_failure(failures, "generator_config_hash_mismatch")
        try:
            common.assert_structure_matches_template(payload, topology_template)
        except AssertionError:
            _append_failure(failures, "sample_structure_changed")
        try:
            common.assert_context_label_consistency(payload, topology_template)
        except AssertionError:
            _append_failure(failures, "context_label_inconsistent")
        if common.x_hash(payload, topology_template) != row.get("x_hash"):
            _append_failure(failures, "x_hash_mismatch")
        if common.label_hash(payload, topology_template) != row.get("label_hash"):
            _append_failure(failures, "label_hash_mismatch")
        seen_context_hashes.add(str(row.get("x_hash")))
        seen_sample_keys.add(
            (
                str(row.get("split_namespace")),
                str(row.get("train_seed")),
                str(row.get("sample_index")),
            )
        )

        regenerated = xy_sampler.generate_sample(
            topology_template=topology_template,
            base_payload=base_payload,
            topology_id=str(row["topology_id"]),
            regime=str(row["regime"]),
            split_namespace=str(row["split_namespace"]),
            train_seed=row.get("train_seed"),
            sample_index=int(row["sample_index"]),
            experiment_version=str(row.get("experiment_version", "")),
            master_label_seed=int(row["master_label_seed"]),
            generator_config=generator_config,
            context_seed_override=int(row["context_seed"]),
            label_noise_seed_override=int(row["label_noise_seed"]),
        )
        if regenerated["manifest"]["x_hash"] != row.get("x_hash"):
            _append_failure(failures, "same_seed_x_not_reproducible")
        if regenerated["manifest"]["label_hash"] != row.get("label_hash"):
            _append_failure(failures, "same_seed_y_not_reproducible")

    if len(seen_sample_keys) != len(sample_rows):
        _append_failure(failures, "duplicate_sample_key")
    if len(sample_rows) > 1 and len(seen_context_hashes) <= 1:
        _append_failure(failures, "context_seeds_do_not_change_x")
    return failures


def audit_train_bank(
    *,
    train_bank_path: str | Path,
    topology_template: dict[str, Any],
    base_payload: dict[str, Any],
    generator_config: dict[str, Any],
) -> dict[str, Any]:
    dataset = common.read_npz_dataset(train_bank_path)
    manifest = dataset["manifest"]
    sample_rows = list(manifest.get("samples", []))
    failures = audit_sample_rows(
        sample_rows=sample_rows,
        payloads=dataset["payloads"],
        topology_template=topology_template,
        base_payload=base_payload,
        generator_config=generator_config,
    )
    if manifest.get("generator_config_hash") != common.generator_config_hash(generator_config):
        _append_failure(failures, "generator_config_hash_mismatch")
    prefix_hashes = manifest.get("prefix_hashes", {})
    for raw_size, expected_hash in prefix_hashes.items():
        if common.prefix_hash(sample_rows, int(raw_size)) != expected_hash:
            _append_failure(failures, "nested_prefix_hash_mismatch")
    if common.sample_manifest_hashes(sample_rows[: int(manifest["max_train_size"])]) != manifest.get("bank_hash"):
        _append_failure(failures, "bank_hash_mismatch")
    if any(str(row.get("split_namespace")) not in common.TRAIN_NAMESPACES for row in sample_rows):
        _append_failure(failures, "train_bank_namespace_invalid")
    return {
        "passed": not failures,
        "failures": failures,
        "manifest": manifest,
    }


def audit_eval_manifest(eval_manifest_path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(eval_manifest_path).read_text(encoding="utf-8"))
    failures: list[str] = []
    validation_rows = manifest.get("validation_samples", [])
    test_rows = manifest.get("test_samples", [])
    if any(row.get("train_seed") is not None for row in validation_rows + test_rows):
        _append_failure(failures, "eval_train_seed_varies")
    if {row.get("split_namespace") for row in validation_rows} != {"confirm_validation"}:
        _append_failure(failures, "validation_namespace_invalid")
    if {row.get("split_namespace") for row in test_rows} != {"confirm_test"}:
        _append_failure(failures, "test_namespace_invalid")
    train_keys = {
        (row.get("split_namespace"), row.get("train_seed"), row.get("sample_index"))
        for row in validation_rows + test_rows
    }
    if len(train_keys) != len(validation_rows) + len(test_rows):
        _append_failure(failures, "eval_namespace_overlap")
    return {
        "passed": not failures,
        "failures": failures,
        "manifest": manifest,
    }


def audit_fixed_topology_xy(
    *,
    train_bank_path: str | Path,
    eval_manifest_path: str | Path,
    topology_template: dict[str, Any],
    base_payload: dict[str, Any],
    generator_config: dict[str, Any],
) -> dict[str, Any]:
    train_result = audit_train_bank(
        train_bank_path=train_bank_path,
        topology_template=topology_template,
        base_payload=base_payload,
        generator_config=generator_config,
    )
    eval_result = audit_eval_manifest(eval_manifest_path)
    failures = list(dict.fromkeys(train_result["failures"] + eval_result["failures"]))
    train_namespaces = {
        row.get("split_namespace")
        for row in train_result["manifest"].get("samples", [])
    }
    eval_namespaces = {
        row.get("split_namespace")
        for row in eval_result["manifest"].get("validation_samples", [])
        + eval_result["manifest"].get("test_samples", [])
    }
    if train_namespaces & eval_namespaces:
        _append_failure(failures, "train_eval_namespace_overlap")
    return {
        "passed": not failures,
        "failures": failures,
        "train": train_result,
        "eval": eval_result,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-bank", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--base-payload", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--require-locked", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    template = json.loads(args.topology.read_text(encoding="utf-8"))
    base_payload = json.loads(args.base_payload.read_text(encoding="utf-8"))
    generator_config = context_sampler.load_generator_config(args.config)
    if args.require_locked:
        assert_formal_config_locked(generator_config)
    result = audit_fixed_topology_xy(
        train_bank_path=args.train_bank,
        eval_manifest_path=args.eval_manifest,
        topology_template=template,
        base_payload=base_payload,
        generator_config=generator_config,
    )
    print(json.dumps({"passed": result["passed"], "failures": result["failures"]}, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
