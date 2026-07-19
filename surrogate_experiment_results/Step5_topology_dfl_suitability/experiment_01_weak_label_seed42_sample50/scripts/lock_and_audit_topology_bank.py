#!/usr/bin/env python3
"""Validate and lock the Step5 1000-topology manifest without building datasets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_BANK = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "Step3"
    / "pairs20_ndd2"
    / "data"
    / "topologies"
    / "topology_bank.csv"
)
DEFAULT_LOCKED_MANIFEST = EXPERIMENT_ROOT / "configs" / "topologies.locked.csv"
DEFAULT_AUDIT_OUTPUT = EXPERIMENT_ROOT / "results" / "topology_bank_audit.json"
REQUIRED_FIELDS = (
    "topology_id",
    "topology_hash",
    "arc_order_hash",
    "feasible_set_hash",
    "template_path",
    "source_path",
)
UNIQUE_FIELDS = (
    "topology_id",
    "topology_hash",
    "arc_order_hash",
    "feasible_set_hash",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def resolve_project_path(raw_path: str | Path, project_root: str | Path = PROJECT_ROOT) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else Path(project_root) / path


def portable_path(path: str | Path, project_root: str | Path = PROJECT_ROOT) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(Path(project_root).resolve()))
    except ValueError:
        return str(resolved)


def _duplicates(rows: list[dict[str, str]], field: str) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for row in rows:
        value = str(row.get(field, ""))
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


def audit_topology_bank(
    source_bank: str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
    expected_count: int = 1000,
    validate_json: bool = True,
) -> dict[str, Any]:
    source_bank = Path(source_bank)
    fieldnames, rows = read_csv(source_bank)
    failures: list[str] = []
    missing_fields = [field for field in REQUIRED_FIELDS if field not in fieldnames]
    if missing_fields:
        failures.append("missing_fields:" + ",".join(missing_fields))
    if len(rows) != int(expected_count):
        failures.append(f"row_count_mismatch:{len(rows)}!={int(expected_count)}")

    blank_counts = {
        field: sum(not str(row.get(field, "")).strip() for row in rows)
        for field in REQUIRED_FIELDS
    }
    for field, count in blank_counts.items():
        if count:
            failures.append(f"blank_{field}:{count}")

    duplicate_values = {field: _duplicates(rows, field) for field in UNIQUE_FIELDS}
    for field, values in duplicate_values.items():
        if values:
            failures.append(f"duplicate_{field}:{len(values)}")

    missing_templates: list[str] = []
    missing_sources: list[str] = []
    invalid_templates: list[str] = []
    invalid_sources: list[str] = []
    template_metadata_mismatches: list[str] = []
    for row in rows:
        topology_id = str(row.get("topology_id", ""))
        template_path = resolve_project_path(row.get("template_path", ""), project_root)
        source_path = resolve_project_path(row.get("source_path", ""), project_root)
        if not template_path.is_file():
            missing_templates.append(topology_id)
        elif validate_json:
            try:
                template = json.loads(template_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                invalid_templates.append(topology_id)
            else:
                for field in UNIQUE_FIELDS:
                    expected = str(row.get(field, ""))
                    observed = str(template.get(field, ""))
                    if expected and observed != expected:
                        template_metadata_mismatches.append(
                            f"{topology_id}:{field}:{observed}!={expected}"
                        )
        if not source_path.is_file():
            missing_sources.append(topology_id)
        elif validate_json:
            try:
                json.loads(source_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                invalid_sources.append(topology_id)

    if missing_templates:
        failures.append(f"missing_templates:{len(missing_templates)}")
    if missing_sources:
        failures.append(f"missing_sources:{len(missing_sources)}")
    if invalid_templates:
        failures.append(f"invalid_template_json:{len(invalid_templates)}")
    if invalid_sources:
        failures.append(f"invalid_source_json:{len(invalid_sources)}")
    if template_metadata_mismatches:
        failures.append(f"template_metadata_mismatches:{len(template_metadata_mismatches)}")

    return {
        "passed": not failures,
        "failures": failures,
        "source_bank": portable_path(source_bank, project_root),
        "source_bank_sha256": sha256_file(source_bank),
        "expected_row_count": int(expected_count),
        "row_count": len(rows),
        "column_count": len(fieldnames),
        "fieldnames": fieldnames,
        "blank_counts": blank_counts,
        "unique_counts": {
            field: len({str(row.get(field, "")) for row in rows})
            for field in UNIQUE_FIELDS
        },
        "duplicate_values": duplicate_values,
        "missing_template_count": len(missing_templates),
        "missing_template_topology_ids": missing_templates,
        "missing_source_count": len(missing_sources),
        "missing_source_topology_ids": missing_sources,
        "invalid_template_json_topology_ids": invalid_templates,
        "invalid_source_json_topology_ids": invalid_sources,
        "template_metadata_mismatches": template_metadata_mismatches,
    }


def atomic_write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_name = handle.name
    os.replace(temp_name, output)


def atomic_write_csv(
    path: str | Path,
    *,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temp_name = handle.name
    os.replace(temp_name, output)


def lock_manifest(
    source_bank: str | Path,
    locked_manifest: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    source_bank = Path(source_bank)
    locked_manifest = Path(locked_manifest)
    fieldnames, rows = read_csv(source_bank)
    source_hash = sha256_file(source_bank)
    if locked_manifest.exists():
        locked_hash = sha256_file(locked_manifest)
        if locked_hash == source_hash:
            return {"status": "already_locked", "locked_manifest_sha256": locked_hash}
        if not overwrite:
            raise ValueError(
                f"Locked manifest differs from source: {locked_manifest}; pass --overwrite to replace it"
            )
    atomic_write_csv(locked_manifest, fieldnames=fieldnames, rows=rows)
    return {
        "status": "locked",
        "locked_manifest_sha256": sha256_file(locked_manifest),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bank", type=Path, default=DEFAULT_SOURCE_BANK)
    parser.add_argument("--locked-manifest", type=Path, default=DEFAULT_LOCKED_MANIFEST)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT_OUTPUT)
    parser.add_argument("--expected-count", type=int, default=1000)
    parser.add_argument("--skip-json-validation", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    audit = audit_topology_bank(
        args.source_bank,
        expected_count=args.expected_count,
        validate_json=not args.skip_json_validation,
    )
    if audit["passed"]:
        try:
            lock_result = lock_manifest(
                args.source_bank,
                args.locked_manifest,
                overwrite=args.overwrite,
            )
        except ValueError as exc:
            audit["passed"] = False
            audit["failures"].append(f"lock_refused:{exc}")
        else:
            audit.update(lock_result)
            audit["locked_manifest"] = portable_path(args.locked_manifest)
    atomic_write_json(args.audit_output, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
