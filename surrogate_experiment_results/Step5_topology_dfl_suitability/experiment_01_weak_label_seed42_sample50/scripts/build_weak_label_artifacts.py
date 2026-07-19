#!/usr/bin/env python3
"""Build deterministic Step5 train/validation/test artifacts per topology."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
STEP3_SCRIPTS = PROJECT_ROOT / "surrogate_experiment_results" / "Step3" / "scripts"
if str(STEP3_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(STEP3_SCRIPTS))

import fixed_topology_xy_common as common  # noqa: E402
import sample_fixed_topology_context as context_sampler  # noqa: E402
import sample_fixed_topology_xy as xy_sampler  # noqa: E402


DEFAULT_TOPOLOGIES = EXPERIMENT_ROOT / "configs" / "topologies.locked.csv"
DEFAULT_EXPERIMENT_VERSION = "step5_exp1_weak_label_seed42_sample50_v1"
DEFAULT_REGIME = "step2c_poly_d8_mult_eps050"
DEFAULT_DATA_SEED = 42
DEFAULT_SAMPLE_SIZE = 50
DEFAULT_TEST_SIZE = 1000


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def resolve_project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def selected_topology_rows(
    path: str | Path,
    *,
    topology_ids: list[str] | tuple[str, ...] | None = None,
    limit: int | None = None,
) -> list[dict[str, str]]:
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError(f"No topology rows found in {path}")
    ids = [str(value) for value in (topology_ids or [])]
    if len(ids) != len(set(ids)):
        raise ValueError("--topology-id values must be unique")
    if ids:
        row_by_id = {str(row["topology_id"]): row for row in rows}
        missing = [topology_id for topology_id in ids if topology_id not in row_by_id]
        if missing:
            raise ValueError(f"Unknown topology ids: {missing}")
        rows = [row_by_id[topology_id] for topology_id in ids]
    if limit is not None:
        if int(limit) <= 0:
            raise ValueError("--limit must be positive")
        rows = rows[: int(limit)]
    return rows


def train_namespace(protocol: str) -> str:
    if protocol not in {"screen", "confirm"}:
        raise ValueError("protocol must be screen or confirm")
    return f"{protocol}_train"


def test_namespace(protocol: str) -> str:
    if protocol not in {"screen", "confirm"}:
        raise ValueError("protocol must be screen or confirm")
    return f"{protocol}_test"


def validate_sample_size(sample_size: int) -> tuple[int, int]:
    sample_size = int(sample_size)
    if sample_size <= 0 or sample_size % 5:
        raise ValueError("4:1 split requires a positive sample_size divisible by 5")
    validation_size = sample_size // 5
    return sample_size - validation_size, validation_size


def is_validation_index(index: int) -> bool:
    return (int(index) + 1) % 5 == 0


def _sample_index(sample: dict[str, Any], fallback: int) -> int:
    return int(sample.get("manifest", {}).get("sample_index", fallback))


def split_fit_samples(
    fit_samples: list[dict[str, Any]],
    *,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> dict[str, Any]:
    training_size, validation_size = validate_sample_size(sample_size)
    if len(fit_samples) < int(sample_size):
        raise ValueError(f"Need {sample_size} fit samples, observed {len(fit_samples)}")
    prefix = list(fit_samples[: int(sample_size)])
    training_samples = [
        sample
        for fallback, sample in enumerate(prefix)
        if not is_validation_index(_sample_index(sample, fallback))
    ]
    validation_samples = [
        sample
        for fallback, sample in enumerate(prefix)
        if is_validation_index(_sample_index(sample, fallback))
    ]
    if len(training_samples) != training_size or len(validation_samples) != validation_size:
        raise ValueError(
            "4:1 split count mismatch: "
            f"training={len(training_samples)} validation={len(validation_samples)}"
        )
    training_indices = [
        _sample_index(sample, fallback)
        for fallback, sample in enumerate(prefix)
        if not is_validation_index(_sample_index(sample, fallback))
    ]
    validation_indices = [
        _sample_index(sample, fallback)
        for fallback, sample in enumerate(prefix)
        if is_validation_index(_sample_index(sample, fallback))
    ]
    return {
        "sample_size": int(sample_size),
        "training_size": training_size,
        "validation_size": validation_size,
        "trainer_train_size_arg": training_size,
        "training_samples": training_samples,
        "validation_samples": validation_samples,
        "training_indices": training_indices,
        "validation_indices": validation_indices,
        "fit_samples": prefix,
    }


def _protocol_record(
    *,
    data_seed: int,
    sample_size: int,
    training_size: int,
    validation_size: int,
    test_size: int,
    protocol: str,
) -> dict[str, Any]:
    return {
        "weak_label": True,
        "weak_label_target": "test_gap_2stage_minus_test_gap_spoplus",
        "weak_label_threshold": 0.1,
        "data_seed": int(data_seed),
        "sample_size": int(sample_size),
        "training_size": int(training_size),
        "validation_size": int(validation_size),
        "test_size": int(test_size),
        "protocol": str(protocol),
        "split_assignment_rule": "every_fifth_sample_is_validation",
    }


def _provenance(
    *,
    template: dict[str, Any],
    generator_config: dict[str, Any],
    experiment_version: str,
    master_label_seed: int,
) -> dict[str, Any]:
    return {
        "experiment_version": str(experiment_version),
        "master_label_seed": int(master_label_seed),
        "generator_version": str(generator_config["generator_version"]),
        "generator_config_hash": common.generator_config_hash(generator_config),
        **common.template_hashes(template),
    }


def _validate_template_row(row: dict[str, str], template: dict[str, Any]) -> None:
    topology_id = str(row["topology_id"])
    if str(template.get("topology_id", "")) != topology_id:
        raise ValueError(
            f"template topology_id mismatch for {topology_id}: {template.get('topology_id')}"
        )
    for field in ("topology_hash", "arc_order_hash", "feasible_set_hash"):
        if str(template.get(field, "")) != str(row.get(field, "")):
            raise ValueError(f"template {field} mismatch for {topology_id}")


def artifact_paths(
    output_root: str | Path,
    *,
    regime: str,
    topology_id: str,
    data_seed: int,
    sample_size: int,
) -> dict[str, Path]:
    topology_root = Path(output_root) / "data" / str(regime) / str(topology_id)
    data_dir = topology_root / f"data_seed={int(data_seed):06d}"
    test_dir = topology_root / "test"
    return {
        "topology_root": topology_root,
        "data_dir": data_dir,
        "test_dir": test_dir,
        "train_bank": data_dir / "train_bank.npz",
        "validation": data_dir / f"validation_sample_size{int(sample_size):03d}.npz",
        "eval_manifest": data_dir / f"eval_manifest_sample_size{int(sample_size):03d}.json",
        "fit_manifest": data_dir / "fit_manifest.json",
        "split_manifest": data_dir / "split_manifest.json",
        "test": test_dir / "test.npz",
        "test_manifest": test_dir / "test_manifest.json",
    }


def write_test_artifacts(
    *,
    samples: list[dict[str, Any]],
    paths: dict[str, Path],
    topology_id: str,
    regime: str,
    protocol: str,
    provenance: dict[str, Any],
    protocol_record: dict[str, Any],
) -> dict[str, Any]:
    sample_rows = [sample["manifest"] for sample in samples]
    expected_namespace = test_namespace(protocol)
    namespaces = {str(row.get("split_namespace", "")) for row in sample_rows}
    if namespaces != {expected_namespace}:
        raise ValueError(f"test namespace mismatch: {sorted(namespaces)}")
    if {row.get("train_seed") for row in sample_rows} - {None}:
        raise ValueError("test samples must not vary by train_seed")
    test_hash = common.sample_manifest_hashes(sample_rows)
    dataset_manifest = {
        "topology_id": str(topology_id),
        "regime": str(regime),
        "protocol": str(protocol),
        "split_namespace": expected_namespace,
        "train_seed": None,
        "train_seed_sentinel": common.EVAL_TRAIN_SEED_SENTINEL,
        "sample_count": len(samples),
        "dataset_hash": test_hash,
        "samples": sample_rows,
        "step5_protocol": protocol_record,
        **provenance,
    }
    common.write_npz_dataset(paths["test"], samples=samples, manifest=dataset_manifest)
    test_manifest = {
        "topology_id": str(topology_id),
        "regime": str(regime),
        "protocol": str(protocol),
        "test_namespace": expected_namespace,
        "test_size": len(samples),
        "test_path": str(paths["test"]),
        "test_hash": test_hash,
        "test_samples": sample_rows,
        "step5_protocol": protocol_record,
        **provenance,
    }
    common.atomic_write_json(paths["test_manifest"], test_manifest)
    return test_manifest


def write_fit_artifacts(
    *,
    fit_samples: list[dict[str, Any]],
    paths: dict[str, Path],
    topology_id: str,
    regime: str,
    data_seed: int,
    sample_size: int,
    protocol: str,
    test_manifest: dict[str, Any],
    provenance: dict[str, Any],
    protocol_record: dict[str, Any],
) -> dict[str, Any]:
    split = split_fit_samples(fit_samples, sample_size=sample_size)
    fit_rows = [sample["manifest"] for sample in split["fit_samples"]]
    training_rows = [sample["manifest"] for sample in split["training_samples"]]
    validation_rows = [sample["manifest"] for sample in split["validation_samples"]]
    fit_hash = common.sample_manifest_hashes(fit_rows)
    training_hash = common.sample_manifest_hashes(training_rows)
    validation_hash = common.sample_manifest_hashes(validation_rows)
    namespace = train_namespace(protocol)

    fit_manifest = {
        "topology_id": str(topology_id),
        "regime": str(regime),
        "protocol": str(protocol),
        "source_namespace": namespace,
        "data_seed": int(data_seed),
        "train_seed": int(data_seed),
        "sample_count": len(fit_rows),
        "fit_bank_hash": fit_hash,
        "samples": fit_rows,
        "fit_role_rows": [
            {
                "fit_index": _sample_index(sample, fallback),
                "sample_index": _sample_index(sample, fallback),
                "sample_id": sample.get("manifest", {}).get("sample_id", ""),
                "role": (
                    "validation"
                    if is_validation_index(_sample_index(sample, fallback))
                    else "training"
                ),
            }
            for fallback, sample in enumerate(split["fit_samples"])
        ],
        "validation_scheme": "every_fifth_sample",
        "step5_protocol": protocol_record,
        **provenance,
    }
    common.atomic_write_json(paths["fit_manifest"], fit_manifest)

    train_manifest = {
        "topology_id": str(topology_id),
        "regime": str(regime),
        "protocol": str(protocol),
        "split_namespace": namespace,
        "data_seed": int(data_seed),
        "train_seed": int(data_seed),
        "max_sample_size": int(sample_size),
        "max_train_size": int(split["training_size"]),
        "sample_sizes": [int(sample_size)],
        "prefix_sizes": [int(split["training_size"])],
        "bank_hash": training_hash,
        "prefix_hashes": {str(split["training_size"]): training_hash},
        "samples": training_rows,
        "source_namespace": namespace,
        "validation_scheme": "every_fifth_sample",
        "fit_manifest_path": str(paths["fit_manifest"]),
        "fit_bank_hash": fit_hash,
        "step5_protocol": protocol_record,
        **provenance,
    }
    common.write_npz_dataset(
        paths["train_bank"], samples=split["training_samples"], manifest=train_manifest
    )

    validation_manifest = {
        "topology_id": str(topology_id),
        "regime": str(regime),
        "protocol": str(protocol),
        "split_namespace": namespace,
        "source_namespace": namespace,
        "fit_role": "validation",
        "validation_scheme": "every_fifth_sample",
        "data_seed": int(data_seed),
        "train_seed": int(data_seed),
        "sample_size": int(sample_size),
        "sample_count": int(split["validation_size"]),
        "dataset_hash": validation_hash,
        "samples": validation_rows,
        "step5_protocol": protocol_record,
        **provenance,
    }
    common.write_npz_dataset(
        paths["validation"], samples=split["validation_samples"], manifest=validation_manifest
    )

    eval_manifest = {
        "topology_id": str(topology_id),
        "regime": str(regime),
        "protocol": str(protocol),
        "data_seed": int(data_seed),
        "train_seed": int(data_seed),
        "sample_size": int(sample_size),
        "training_size": int(split["training_size"]),
        "validation_size": int(split["validation_size"]),
        "trainer_train_size_arg": int(split["training_size"]),
        "validation_path": paths["validation"].name,
        "validation_hash": validation_hash,
        "test_path": str(paths["test"]),
        "test_hash": str(test_manifest["test_hash"]),
        "source_namespace": namespace,
        "validation_scheme": "every_fifth_sample",
        "fit_manifest_path": str(paths["fit_manifest"]),
        "fit_bank_hash": fit_hash,
        "step5_protocol": protocol_record,
        **provenance,
    }
    common.atomic_write_json(paths["eval_manifest"], eval_manifest)

    split_manifest = {
        "topology_id": str(topology_id),
        "regime": str(regime),
        "protocol": str(protocol),
        "data_seed": int(data_seed),
        "sample_sizes": [int(sample_size)],
        "assignment_rule": "every_fifth_sample_is_validation",
        "training_bank_path": str(paths["train_bank"]),
        "fit_manifest_path": str(paths["fit_manifest"]),
        "fit_bank_hash": fit_hash,
        "sample_size_splits": {
            str(sample_size): {
                "sample_size": int(sample_size),
                "training_size": int(split["training_size"]),
                "validation_size": int(split["validation_size"]),
                "trainer_train_size_arg": int(split["training_size"]),
                "training_indices": split["training_indices"],
                "validation_indices": split["validation_indices"],
                "training_hash": training_hash,
                "validation_hash": validation_hash,
            }
        },
        "source_namespace": namespace,
        "validation_scheme": "every_fifth_sample",
        "step5_protocol": protocol_record,
        **provenance,
    }
    common.atomic_write_json(paths["split_manifest"], split_manifest)
    return {
        "train_bank_path": str(paths["train_bank"]),
        "validation_path": str(paths["validation"]),
        "eval_manifest_path": str(paths["eval_manifest"]),
        "fit_manifest_path": str(paths["fit_manifest"]),
        "split_manifest_path": str(paths["split_manifest"]),
        "test_path": str(paths["test"]),
        "test_manifest_path": str(paths["test_manifest"]),
        "training_hash": training_hash,
        "validation_hash": validation_hash,
        "test_hash": str(test_manifest["test_hash"]),
    }


def _read_existing_test(paths: dict[str, Path], expected_test_size: int) -> dict[str, Any] | None:
    existing = [paths["test"].exists(), paths["test_manifest"].exists()]
    if not any(existing):
        return None
    if not all(existing):
        raise ValueError(f"Partial test artifacts under {paths['test_dir']}; pass --force")
    manifest = json.loads(paths["test_manifest"].read_text(encoding="utf-8"))
    if int(manifest.get("test_size", -1)) != int(expected_test_size):
        raise ValueError(
            f"Existing test_size={manifest.get('test_size')} differs from {expected_test_size}; pass --force"
        )
    return manifest


def build_one_topology(
    row: dict[str, str],
    *,
    output_root: str | Path,
    generator_config: dict[str, Any],
    regime: str,
    protocol: str,
    data_seed: int,
    sample_size: int,
    test_size: int,
    experiment_version: str,
    master_label_seed: int,
    force: bool = False,
) -> dict[str, Any]:
    topology_id = str(row["topology_id"])
    template_path = resolve_project_path(row["template_path"])
    source_path = resolve_project_path(row["source_path"])
    template = json.loads(template_path.read_text(encoding="utf-8"))
    base_payload = json.loads(source_path.read_text(encoding="utf-8"))
    _validate_template_row(row, template)
    training_size, validation_size = validate_sample_size(sample_size)
    paths = artifact_paths(
        output_root,
        regime=regime,
        topology_id=topology_id,
        data_seed=data_seed,
        sample_size=sample_size,
    )
    fit_keys = ("train_bank", "validation", "eval_manifest", "fit_manifest", "split_manifest")
    fit_existing = [paths[key].exists() for key in fit_keys]
    if all(fit_existing) and not force:
        return {"topology_id": topology_id, "status": "skipped_existing", **{k: str(v) for k, v in paths.items()}}
    if any(fit_existing) and not force:
        raise ValueError(f"Partial fit artifacts under {paths['data_dir']}; pass --force")

    provenance = _provenance(
        template=template,
        generator_config=generator_config,
        experiment_version=experiment_version,
        master_label_seed=master_label_seed,
    )
    protocol_record = _protocol_record(
        data_seed=data_seed,
        sample_size=sample_size,
        training_size=training_size,
        validation_size=validation_size,
        test_size=test_size,
        protocol=protocol,
    )
    test_manifest = None if force else _read_existing_test(paths, test_size)
    if test_manifest is None:
        test_samples = xy_sampler.generate_samples(
            topology_template=template,
            base_payload=base_payload,
            topology_id=topology_id,
            regime=regime,
            split_namespace=test_namespace(protocol),
            train_seed=None,
            num_samples=int(test_size),
            experiment_version=experiment_version,
            master_label_seed=int(master_label_seed),
            generator_config=generator_config,
        )
        test_manifest = write_test_artifacts(
            samples=test_samples,
            paths=paths,
            topology_id=topology_id,
            regime=regime,
            protocol=protocol,
            provenance=provenance,
            protocol_record=protocol_record,
        )

    fit_samples = xy_sampler.generate_samples(
        topology_template=template,
        base_payload=base_payload,
        topology_id=topology_id,
        regime=regime,
        split_namespace=train_namespace(protocol),
        train_seed=int(data_seed),
        num_samples=int(sample_size),
        experiment_version=experiment_version,
        master_label_seed=int(master_label_seed),
        generator_config=generator_config,
    )
    result = write_fit_artifacts(
        fit_samples=fit_samples,
        paths=paths,
        topology_id=topology_id,
        regime=regime,
        data_seed=data_seed,
        sample_size=sample_size,
        protocol=protocol,
        test_manifest=test_manifest,
        provenance=provenance,
        protocol_record=protocol_record,
    )
    return {"topology_id": topology_id, "status": "built", **result}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topologies-csv", type=Path, default=DEFAULT_TOPOLOGIES)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--master-label-seed", type=int, required=True)
    parser.add_argument("--experiment-version", default=DEFAULT_EXPERIMENT_VERSION)
    parser.add_argument("--regime", default=DEFAULT_REGIME)
    parser.add_argument("--protocol", choices=("screen", "confirm"), default="screen")
    parser.add_argument("--data-seed", type=int, default=DEFAULT_DATA_SEED)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--test-size", type=int, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--topology-id", action="append", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    training_size, validation_size = validate_sample_size(args.sample_size)
    rows = selected_topology_rows(
        args.topologies_csv,
        topology_ids=args.topology_id,
        limit=args.limit,
    )
    generator_config = context_sampler.load_generator_config(args.config)
    preview = {
        "topology_count": len(rows),
        "topology_ids": [row["topology_id"] for row in rows],
        "output_root": str(args.output_root),
        "regime": args.regime,
        "protocol": args.protocol,
        "data_seed": int(args.data_seed),
        "sample_size": int(args.sample_size),
        "training_size": training_size,
        "validation_size": validation_size,
        "test_size": int(args.test_size),
        "experiment_version": args.experiment_version,
        "master_label_seed": int(args.master_label_seed),
        "generator_config_hash": common.generator_config_hash(generator_config),
        "dry_run": bool(args.dry_run),
    }
    if args.dry_run:
        print(json.dumps(preview, indent=2, sort_keys=True))
        return 0

    results: list[dict[str, Any]] = []
    failures: list[str] = []
    for row in rows:
        try:
            result = build_one_topology(
                row,
                output_root=args.output_root,
                generator_config=generator_config,
                regime=args.regime,
                protocol=args.protocol,
                data_seed=args.data_seed,
                sample_size=args.sample_size,
                test_size=args.test_size,
                experiment_version=args.experiment_version,
                master_label_seed=args.master_label_seed,
                force=args.force,
            )
        except Exception as exc:  # pragma: no cover - real generator failures vary.
            result = {
                "topology_id": str(row.get("topology_id", "")),
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(f"{result['topology_id']}:{result['error']}")
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    summary = {
        **preview,
        "dry_run": False,
        "passed": not failures,
        "built": sum(row["status"] == "built" for row in results),
        "skipped_existing": sum(row["status"] == "skipped_existing" for row in results),
        "failed": len(failures),
        "failures": failures,
        "results": results,
    }
    summary_output = args.summary_output or Path(args.output_root) / "results" / "artifact_build_summary.json"
    common.atomic_write_json(summary_output, summary)
    print(json.dumps({**{key: summary[key] for key in ("passed", "topology_count", "built", "skipped_existing", "failed")}, "summary_output": str(summary_output)}, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
