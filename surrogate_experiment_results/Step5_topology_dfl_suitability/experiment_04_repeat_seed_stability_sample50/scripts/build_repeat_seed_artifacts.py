#!/usr/bin/env python3
"""Build seed-43/44 train/validation data while reusing the seed-42 fixed test bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import repeat_seed_common as common


builder = common.base_builder()


def selected_rows(path: Path, shard_index: int | None, shard_count: int | None) -> list[dict[str, str]]:
    rows = common.read_csv(path)
    if (shard_index is None) != (shard_count is None):
        raise ValueError("--shard-index and --shard-count must be used together")
    if shard_count is not None:
        if shard_count <= 0 or shard_index is None or not 0 <= shard_index < shard_count:
            raise ValueError("invalid shard specification")
        rows = [row for index, row in enumerate(rows) if index % shard_count == shard_index]
    return rows


def reference_test_manifest(topology_id: str, formal_output_root: Path) -> tuple[Path, Path, dict[str, Any]]:
    test_path, manifest_path = common.reference_test_paths(topology_id, formal_output_root=formal_output_root)
    if not test_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"missing fixed formal test bank for {topology_id}: {test_path}")
    manifest = common.read_json(manifest_path)
    expected = {
        "topology_id": topology_id,
        "test_size": common.TEST_SIZE,
        "test_hash": str(manifest.get("test_hash", "")),
    }
    if str(manifest.get("topology_id")) != topology_id:
        raise ValueError(f"reference topology mismatch for {topology_id}")
    if int(manifest.get("test_size", -1)) != common.TEST_SIZE:
        raise ValueError(f"reference test size mismatch for {topology_id}")
    if not expected["test_hash"]:
        raise ValueError(f"reference test hash missing for {topology_id}")
    sample_seeds = {sample.get("train_seed") for sample in manifest.get("test_samples", [])}
    if sample_seeds - {None}:
        raise ValueError(f"reference test varies by train seed for {topology_id}")
    return test_path, manifest_path, manifest


def build_one(row: dict[str, str], *, train_seed: int, output_root: Path, formal_output_root: Path, generator_config: dict[str, Any], force: bool) -> dict[str, Any]:
    topology_id = str(row["topology_id"])
    template_path = builder.resolve_project_path(row["template_path"])
    source_path = builder.resolve_project_path(row["source_path"])
    template = json.loads(template_path.read_text(encoding="utf-8"))
    base_payload = json.loads(source_path.read_text(encoding="utf-8"))
    builder._validate_template_row(row, template)

    paths = builder.artifact_paths(
        output_root,
        regime=common.DEFAULT_REGIME,
        topology_id=topology_id,
        data_seed=train_seed,
        sample_size=common.SAMPLE_SIZE,
    )
    fit_keys = ("train_bank", "validation", "eval_manifest", "fit_manifest", "split_manifest")
    present = [paths[key].exists() for key in fit_keys]
    if all(present) and not force:
        return {"topology_id": topology_id, "train_seed": train_seed, "status": "skipped_existing"}
    if any(present) and not force:
        raise ValueError(f"partial fit artifacts under {paths['data_dir']}; pass --force")

    test_path, test_manifest_path, test_manifest = reference_test_manifest(topology_id, formal_output_root)
    selected_hash = str(row.get("test_hash", ""))
    if selected_hash and selected_hash != str(test_manifest["test_hash"]):
        raise ValueError(f"selected/reference test hash mismatch for {topology_id}")
    paths["test"] = Path(common.project_relative(test_path))
    paths["test_manifest"] = Path(common.project_relative(test_manifest_path))
    paths["test_dir"] = paths["test"].parent

    provenance = builder._provenance(
        template=template,
        generator_config=generator_config,
        experiment_version=common.EXPERIMENT_VERSION,
        master_label_seed=common.MASTER_LABEL_SEED,
    )
    protocol_record = builder._protocol_record(
        data_seed=train_seed,
        sample_size=common.SAMPLE_SIZE,
        training_size=common.TRAINING_SIZE,
        validation_size=common.VALIDATION_SIZE,
        test_size=common.TEST_SIZE,
        protocol=common.DEFAULT_PROTOCOL,
    )
    fit_samples = builder.xy_sampler.generate_samples(
        topology_template=template,
        base_payload=base_payload,
        topology_id=topology_id,
        regime=common.DEFAULT_REGIME,
        split_namespace=builder.train_namespace(common.DEFAULT_PROTOCOL),
        train_seed=train_seed,
        num_samples=common.SAMPLE_SIZE,
        experiment_version=common.EXPERIMENT_VERSION,
        master_label_seed=common.MASTER_LABEL_SEED,
        generator_config=generator_config,
    )
    result = builder.write_fit_artifacts(
        fit_samples=fit_samples,
        paths=paths,
        topology_id=topology_id,
        regime=common.DEFAULT_REGIME,
        data_seed=train_seed,
        sample_size=common.SAMPLE_SIZE,
        protocol=common.DEFAULT_PROTOCOL,
        test_manifest=test_manifest,
        provenance=provenance,
        protocol_record=protocol_record,
    )
    eval_path = Path(result["eval_manifest_path"])
    eval_manifest = common.read_json(eval_path)
    eval_manifest["repeat_seed_protocol"] = {
        "train_seed": train_seed,
        "reference_label_seed": common.REFERENCE_SEED,
        "fixed_test_bank": True,
        "reference_test_path": common.project_relative(test_path),
        "reference_test_manifest_path": common.project_relative(test_manifest_path),
        "reference_test_hash": str(test_manifest["test_hash"]),
        "test_bank_varies_with_train_seed": False,
    }
    builder.common.atomic_write_json(eval_path, eval_manifest)
    return {
        "topology_id": topology_id,
        "train_seed": train_seed,
        "status": "built",
        "test_hash": str(test_manifest["test_hash"]),
        **result,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topologies-csv", type=Path, default=common.DEFAULT_SELECTED_TOPOLOGIES)
    parser.add_argument("--output-root", type=Path, default=common.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--formal-output-root", type=Path, default=common.DEFAULT_FORMAL_OUTPUT_ROOT)
    parser.add_argument("--config", type=Path, default=common.DEFAULT_GENERATOR_CONFIG)
    parser.add_argument("--train-seed", type=int, action="append", choices=common.TRAIN_SEEDS)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    seeds = tuple(args.train_seed or common.TRAIN_SEEDS)
    rows = selected_rows(args.topologies_csv, args.shard_index, args.shard_count)
    preview = {
        "topology_count": len(rows),
        "train_seeds": list(seeds),
        "new_artifact_count": len(rows) * len(seeds),
        "sample_size": common.SAMPLE_SIZE,
        "training_size": common.TRAINING_SIZE,
        "validation_size": common.VALIDATION_SIZE,
        "test_size": common.TEST_SIZE,
        "fixed_reference_test_seed": common.REFERENCE_SEED,
        "formal_output_root": str(args.formal_output_root),
        "output_root": str(args.output_root),
        "dry_run": args.dry_run,
    }
    if args.dry_run:
        print(json.dumps(preview, indent=2, sort_keys=True))
        return 0
    generator_config = builder.context_sampler.load_generator_config(args.config)
    results: list[dict[str, Any]] = []
    failures: list[str] = []
    for row in rows:
        for seed in seeds:
            try:
                result = build_one(
                    row,
                    train_seed=seed,
                    output_root=args.output_root,
                    formal_output_root=args.formal_output_root,
                    generator_config=generator_config,
                    force=args.force,
                )
            except Exception as exc:  # pragma: no cover - generator/solver failures vary.
                result = {"topology_id": row.get("topology_id", ""), "train_seed": seed, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                failures.append(f"{result['topology_id']}@{seed}:{result['error']}")
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
    summary_path = args.output_root / "results" / f"artifact_build_shard_{args.shard_index if args.shard_index is not None else 'all'}.json"
    builder.common.atomic_write_json(summary_path, summary)
    print(json.dumps({key: summary[key] for key in ("passed", "new_artifact_count", "built", "skipped_existing", "failed")}, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
