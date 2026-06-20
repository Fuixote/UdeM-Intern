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


def namespaces_for_protocol(protocol: str) -> dict[str, str]:
    if protocol == "confirm":
        return {
            "train": "confirm_train",
            "validation": "confirm_validation",
            "test": "confirm_test",
        }
    if protocol == "screen":
        return {
            "train": "screen_train",
            "validation": "screen_validation",
            "test": "screen_test",
        }
    raise ValueError("protocol must be screen or confirm")


def namespace_protocol(namespace: Any) -> str | None:
    text = str(namespace)
    if text.startswith("screen_"):
        return "screen"
    if text.startswith("confirm_"):
        return "confirm"
    return None


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
    if len(sample_rows) != len(payloads):
        _append_failure(failures, "sample_payload_count_mismatch")
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


def _prefix_failures(prefix: str, failures: list[str]) -> list[str]:
    return [f"{prefix}_{failure}" for failure in failures]


def _resolve_manifest_path(eval_manifest_path: str | Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return Path(eval_manifest_path).parent / path


def audit_train_bank(
    *,
    train_bank_path: str | Path,
    topology_template: dict[str, Any],
    base_payload: dict[str, Any],
    generator_config: dict[str, Any],
    protocol: str = "confirm",
) -> dict[str, Any]:
    expected_namespace = namespaces_for_protocol(protocol)["train"]
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
    if manifest.get("split_namespace") != expected_namespace:
        _append_failure(failures, "train_bank_namespace_invalid")
    if any(str(row.get("split_namespace")) != expected_namespace for row in sample_rows):
        _append_failure(failures, "train_bank_namespace_invalid")
    if manifest.get("protocol", protocol) != protocol:
        _append_failure(failures, "protocol_namespace_mismatch")
    if any(namespace_protocol(row.get("split_namespace")) != protocol for row in sample_rows):
        _append_failure(failures, "protocol_namespace_mismatch")
    return {
        "passed": not failures,
        "failures": failures,
        "manifest": manifest,
    }


def audit_eval_dataset(
    *,
    dataset_path: str | Path,
    eval_manifest_path: str | Path,
    eval_manifest: dict[str, Any],
    split_label: str,
    expected_namespace: str,
    expected_dataset_hash: str,
    topology_template: dict[str, Any],
    base_payload: dict[str, Any],
    generator_config: dict[str, Any],
) -> dict[str, Any]:
    dataset = common.read_npz_dataset(dataset_path)
    dataset_manifest = dataset["manifest"]
    sample_rows = list(dataset_manifest.get("samples", []))
    failures = _prefix_failures(
        split_label,
        audit_sample_rows(
            sample_rows=sample_rows,
            payloads=dataset["payloads"],
            topology_template=topology_template,
            base_payload=base_payload,
            generator_config=generator_config,
        ),
    )
    if dataset_manifest.get("samples") != dataset.get("sample_manifests"):
        _append_failure(failures, f"{split_label}_sample_manifest_array_mismatch")
    if dataset_manifest.get("split_namespace") != expected_namespace:
        _append_failure(failures, f"{split_label}_dataset_namespace_invalid")
    if dataset_manifest.get("train_seed") is not None:
        _append_failure(failures, f"{split_label}_dataset_train_seed_varies")
    if dataset_manifest.get("topology_id") != eval_manifest.get("topology_id"):
        _append_failure(failures, f"{split_label}_dataset_topology_mismatch")
    if any(row.get("split_namespace") != expected_namespace for row in sample_rows):
        _append_failure(failures, f"{split_label}_namespace_invalid")
    if any(row.get("train_seed") is not None for row in sample_rows):
        _append_failure(failures, f"{split_label}_train_seed_varies")
    if any(row.get("topology_id") != eval_manifest.get("topology_id") for row in sample_rows):
        _append_failure(failures, f"{split_label}_topology_mismatch")
    for key, expected in common.template_hashes(topology_template).items():
        if expected and dataset_manifest.get(key) != expected:
            _append_failure(failures, f"{split_label}_{key}_mismatch")
    recomputed_dataset_hash = common.sample_manifest_hashes(sample_rows)
    if dataset_manifest.get("dataset_hash") != recomputed_dataset_hash:
        _append_failure(failures, f"{split_label}_npz_dataset_hash_mismatch")
    if str(expected_dataset_hash) != str(recomputed_dataset_hash):
        _append_failure(failures, f"{split_label}_dataset_hash_mismatch")
    manifest_rows = eval_manifest.get(f"{split_label}_samples", [])
    if manifest_rows and manifest_rows != sample_rows:
        _append_failure(failures, f"{split_label}_manifest_samples_mismatch")
    for index, payload in enumerate(dataset["payloads"]):
        if index >= len(dataset["X"]) or index >= len(dataset["y"]):
            _append_failure(failures, f"{split_label}_array_count_mismatch")
            break
        if common.matrix_hash(dataset["X"][index]) != common.x_hash(payload, topology_template):
            _append_failure(failures, f"{split_label}_X_array_mismatch")
        if common.vector_hash(dataset["y"][index]) != common.label_hash(payload, topology_template):
            _append_failure(failures, f"{split_label}_y_array_mismatch")
    return {
        "passed": not failures,
        "failures": failures,
        "path": str(dataset_path),
        "manifest": dataset_manifest,
    }


def audit_eval_manifest(
    eval_manifest_path: str | Path,
    *,
    topology_template: dict[str, Any],
    base_payload: dict[str, Any],
    generator_config: dict[str, Any],
    protocol: str = "confirm",
) -> dict[str, Any]:
    expected = namespaces_for_protocol(protocol)
    manifest = json.loads(Path(eval_manifest_path).read_text(encoding="utf-8"))
    failures: list[str] = []
    validation_rows = manifest.get("validation_samples", [])
    test_rows = manifest.get("test_samples", [])
    if any(row.get("train_seed") is not None for row in validation_rows + test_rows):
        _append_failure(failures, "eval_train_seed_varies")
    if {row.get("split_namespace") for row in validation_rows} != {expected["validation"]}:
        _append_failure(failures, "validation_namespace_invalid")
    if {row.get("split_namespace") for row in test_rows} != {expected["test"]}:
        _append_failure(failures, "test_namespace_invalid")
    train_keys = {
        (row.get("split_namespace"), row.get("train_seed"), row.get("sample_index"))
        for row in validation_rows + test_rows
    }
    if len(train_keys) != len(validation_rows) + len(test_rows):
        _append_failure(failures, "eval_namespace_overlap")
    if manifest.get("validation_namespace") != expected["validation"]:
        _append_failure(failures, "validation_namespace_invalid")
    if manifest.get("test_namespace") != expected["test"]:
        _append_failure(failures, "test_namespace_invalid")
    if manifest.get("protocol", protocol) != protocol:
        _append_failure(failures, "protocol_namespace_mismatch")
    if any(namespace_protocol(row.get("split_namespace")) != protocol for row in validation_rows + test_rows):
        _append_failure(failures, "protocol_namespace_mismatch")
    validation_path = _resolve_manifest_path(eval_manifest_path, manifest["validation_path"])
    test_path = _resolve_manifest_path(eval_manifest_path, manifest["test_path"])
    validation_result = audit_eval_dataset(
        dataset_path=validation_path,
        eval_manifest_path=eval_manifest_path,
        eval_manifest=manifest,
        split_label="validation",
        expected_namespace=expected["validation"],
        expected_dataset_hash=str(manifest.get("validation_hash")),
        topology_template=topology_template,
        base_payload=base_payload,
        generator_config=generator_config,
    )
    test_result = audit_eval_dataset(
        dataset_path=test_path,
        eval_manifest_path=eval_manifest_path,
        eval_manifest=manifest,
        split_label="test",
        expected_namespace=expected["test"],
        expected_dataset_hash=str(manifest.get("test_hash")),
        topology_template=topology_template,
        base_payload=base_payload,
        generator_config=generator_config,
    )
    failures.extend(validation_result["failures"])
    failures.extend(test_result["failures"])
    return {
        "passed": not failures,
        "failures": list(dict.fromkeys(failures)),
        "manifest": manifest,
        "validation": validation_result,
        "test": test_result,
    }


def audit_fixed_topology_xy(
    *,
    train_bank_path: str | Path,
    eval_manifest_path: str | Path,
    topology_template: dict[str, Any],
    base_payload: dict[str, Any],
    generator_config: dict[str, Any],
    protocol: str = "confirm",
) -> dict[str, Any]:
    train_result = audit_train_bank(
        train_bank_path=train_bank_path,
        topology_template=topology_template,
        base_payload=base_payload,
        generator_config=generator_config,
        protocol=protocol,
    )
    eval_result = audit_eval_manifest(
        eval_manifest_path,
        topology_template=topology_template,
        base_payload=base_payload,
        generator_config=generator_config,
        protocol=protocol,
    )
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
    all_namespaces = train_namespaces | eval_namespaces
    if any(namespace_protocol(namespace) != protocol for namespace in all_namespaces):
        _append_failure(failures, "protocol_namespace_mismatch")
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
    parser.add_argument("--protocol", choices=("screen", "confirm"), default="confirm")
    parser.add_argument("--require-locked", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    template = json.loads(args.topology.read_text(encoding="utf-8"))
    base_payload = json.loads(args.base_payload.read_text(encoding="utf-8"))
    generator_config = context_sampler.load_generator_config(args.config)
    if args.require_locked:
        if args.protocol != "confirm":
            raise ValueError("--require-locked is only valid for protocol=confirm")
        assert_formal_config_locked(generator_config)
    result = audit_fixed_topology_xy(
        train_bank_path=args.train_bank,
        eval_manifest_path=args.eval_manifest,
        topology_template=template,
        base_payload=base_payload,
        generator_config=generator_config,
        protocol=args.protocol,
    )
    print(json.dumps({"passed": result["passed"], "failures": result["failures"]}, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
