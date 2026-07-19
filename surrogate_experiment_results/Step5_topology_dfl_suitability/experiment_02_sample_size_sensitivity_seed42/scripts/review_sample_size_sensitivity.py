#!/usr/bin/env python3
"""Audit and summarize the six-topology Step5 sample-size sensitivity run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


REGIME = "step2c_poly_d8_mult_eps050"
DATA_SEED = 42
EXPECTED_SAMPLE_SIZES = (50, 100, 250)
SUCCESS_FIELDS = {
    "status": "success",
    "2stage status": "success",
    "SPO+ status": "success",
    "evaluation status": "success",
}
FIELDS = [
    "topology_id",
    "diagnostic_role",
    "sample_size",
    "training_size",
    "validation_size",
    "max_epochs",
    "selected_epoch_2stage",
    "selected_epoch_spoplus",
    "two_stage_cap_hit",
    "spoplus_cap_hit",
    "test_normalized_gap_2stage",
    "test_normalized_gap_spoplus",
    "normalized_improvement_pp",
    "relative_improvement_pct",
    "test_gap_2stage",
    "test_gap_spoplus",
    "raw_gap_improvement",
    "fraction_improved_over_2stage",
    "exact_achieved_objective_matches",
    "exact_gap_matches",
    "test_row_pairs",
    "all_achieved_objectives_exactly_equal",
    "max_abs_achieved_objective_difference",
    "test_hash",
    "train_prefix_hash",
    "validation_hash",
    "job_dir",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_name = handle.name
    os.replace(temp_name, path)


def atomic_write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        temp_name = handle.name
    os.replace(temp_name, path)


def parse_root(value: str) -> tuple[int, Path]:
    sample_text, separator, path_text = value.partition("=")
    if not separator or not path_text:
        raise argparse.ArgumentTypeError("sample roots must use SAMPLE_SIZE=PATH")
    try:
        sample_size = int(sample_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("sample size must be an integer") from exc
    return sample_size, Path(path_text)


def topology_roles(path: Path) -> list[tuple[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [(str(row["topology_id"]), str(row["diagnostic_role"])) for row in rows]


def job_dir(root: Path, topology_id: str, sample_size: int) -> Path:
    return (
        root
        / "jobs"
        / REGIME
        / topology_id
        / f"data_seed={DATA_SEED:06d}"
        / f"sample_size={sample_size:03d}"
    )


def fit_manifest_path(root: Path, topology_id: str) -> Path:
    return (
        root
        / "data"
        / REGIME
        / topology_id
        / f"data_seed={DATA_SEED:06d}"
        / "fit_manifest.json"
    )


def canonical_method(value: Any) -> str:
    normalized = str(value).strip().lower().replace("_", "").replace("-", "")
    return "2stage" if normalized in {"2stage", "twostage"} else (
        "spoplus" if normalized in {"spo+", "spoplus", "spo"} else normalized
    )


def method_summary(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_json(path)
    return {canonical_method(row["method"]): row for row in rows}


def per_graph_equality(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_graph: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        by_graph.setdefault(str(row["graph"]), {})[canonical_method(row["method"])] = row
    achieved_matches = 0
    gap_matches = 0
    max_abs_achieved_difference = 0.0
    for graph, methods in by_graph.items():
        if set(methods) != {"2stage", "spoplus"}:
            raise ValueError(f"{path}: incomplete method pair for {graph}")
        achieved_2stage = float(methods["2stage"]["achieved_obj"])
        achieved_spoplus = float(methods["spoplus"]["achieved_obj"])
        gap_2stage = float(methods["2stage"]["gap"])
        gap_spoplus = float(methods["spoplus"]["gap"])
        difference = abs(achieved_2stage - achieved_spoplus)
        max_abs_achieved_difference = max(max_abs_achieved_difference, difference)
        achieved_matches += int(difference == 0.0)
        gap_matches += int(gap_2stage == gap_spoplus)
    pair_count = len(by_graph)
    return {
        "exact_achieved_objective_matches": achieved_matches,
        "exact_gap_matches": gap_matches,
        "test_row_pairs": pair_count,
        "all_achieved_objectives_exactly_equal": achieved_matches == pair_count,
        "max_abs_achieved_objective_difference": max_abs_achieved_difference,
    }


def collect_one(
    root: Path,
    topology_id: str,
    role: str,
    sample_size: int,
    expected_max_epochs: int,
) -> dict[str, Any]:
    current_job_dir = job_dir(root, topology_id, sample_size)
    status = read_json(current_job_dir / "job_status.json")
    for field, expected in SUCCESS_FIELDS.items():
        if status.get(field) != expected:
            raise ValueError(f"{current_job_dir}: {field}={status.get(field)!r}")
    manifest = read_json(current_job_dir / "paired_job_manifest.json")
    expected_manifest = {
        "topology_id": topology_id,
        "regime": REGIME,
        "protocol": "screen",
        "train_seed": DATA_SEED,
        "sample_size": sample_size,
        "training_size": sample_size * 4 // 5,
        "validation_size": sample_size // 5,
        "theta_seed": 42,
        "gurobi_seed": 42,
        "max_epochs": expected_max_epochs,
        "metric_stride": 1,
        "early_stop_patience": 20,
    }
    for field, expected in expected_manifest.items():
        if manifest.get(field) != expected:
            raise ValueError(
                f"{current_job_dir}: manifest {field}={manifest.get(field)!r}, expected {expected!r}"
            )
    if not math.isclose(float(manifest.get("early_stop_min_delta", -1)), 0.0001):
        raise ValueError(f"{current_job_dir}: early_stop_min_delta mismatch")
    methods = method_summary(current_job_dir / "evaluation" / "metrics" / "test_summary.json")
    if set(methods) != {"2stage", "spoplus"}:
        raise ValueError(f"{current_job_dir}: expected paired method summaries")
    two_stage = methods["2stage"]
    spoplus = methods["spoplus"]
    normalized_2stage = float(two_stage["test_mean_normalized_gap"])
    normalized_spoplus = float(spoplus["test_mean_normalized_gap"])
    normalized_delta = normalized_2stage - normalized_spoplus
    relative_improvement = ""
    if normalized_2stage > 0:
        relative_improvement = 100.0 * normalized_delta / normalized_2stage
    selected_2stage = int(two_stage["selected_epoch"])
    selected_spoplus = int(spoplus["selected_epoch"])
    gap_2stage = float(two_stage["test_mean_decision_gap"])
    gap_spoplus = float(spoplus["test_mean_decision_gap"])
    return {
        "topology_id": topology_id,
        "diagnostic_role": role,
        "sample_size": sample_size,
        "training_size": sample_size * 4 // 5,
        "validation_size": sample_size // 5,
        "max_epochs": expected_max_epochs,
        "selected_epoch_2stage": selected_2stage,
        "selected_epoch_spoplus": selected_spoplus,
        "two_stage_cap_hit": selected_2stage == expected_max_epochs,
        "spoplus_cap_hit": selected_spoplus == expected_max_epochs,
        "test_normalized_gap_2stage": normalized_2stage,
        "test_normalized_gap_spoplus": normalized_spoplus,
        "normalized_improvement_pp": 100.0 * normalized_delta,
        "relative_improvement_pct": relative_improvement,
        "test_gap_2stage": gap_2stage,
        "test_gap_spoplus": gap_spoplus,
        "raw_gap_improvement": gap_2stage - gap_spoplus,
        "fraction_improved_over_2stage": float(spoplus["fraction_improved_over_2stage"]),
        **per_graph_equality(current_job_dir / "evaluation" / "metrics" / "test_per_graph.csv"),
        "test_hash": str(manifest["test_hash"]),
        "train_prefix_hash": str(manifest["train_prefix_hash"]),
        "validation_hash": str(manifest["validation_hash"]),
        "job_dir": str(current_job_dir),
    }


def run_review(
    roles: list[tuple[str, str]],
    sample_roots: dict[int, Path],
    output_dir: Path,
    cap_roots: dict[int, Path] | None = None,
) -> dict[str, Any]:
    if tuple(sorted(sample_roots)) != EXPECTED_SAMPLE_SIZES:
        raise ValueError(f"sample roots must be exactly {EXPECTED_SAMPLE_SIZES}")
    rows = [
        collect_one(sample_roots[size], topology_id, role, size, 1500)
        for topology_id, role in roles
        for size in EXPECTED_SAMPLE_SIZES
    ]
    hash_failures: list[str] = []
    hashes_by_topology: dict[str, list[str]] = {}
    fit_prefix_nested_by_topology: dict[str, bool] = {}
    for topology_id, _ in roles:
        hashes = [str(row["test_hash"]) for row in rows if row["topology_id"] == topology_id]
        hashes_by_topology[topology_id] = hashes
        if len(set(hashes)) != 1:
            hash_failures.append(f"{topology_id}:test_hash_mismatch")
        fit_samples = {
            size: read_json(fit_manifest_path(sample_roots[size], topology_id))["samples"]
            for size in EXPECTED_SAMPLE_SIZES
        }
        nested = (
            fit_samples[100][:50] == fit_samples[50]
            and fit_samples[250][:50] == fit_samples[50]
            and fit_samples[250][:100] == fit_samples[100]
        )
        fit_prefix_nested_by_topology[topology_id] = nested
        if not nested:
            hash_failures.append(f"{topology_id}:fit_prefix_not_nested")

    cap_rows: list[dict[str, Any]] = []
    cap_roots = cap_roots or {}
    if cap_roots:
        unknown_sizes = sorted(set(cap_roots) - set(EXPECTED_SAMPLE_SIZES))
        if unknown_sizes:
            raise ValueError(f"unexpected cap-check sample sizes: {unknown_sizes}")
        role_by_id = dict(roles)
        for size in sorted(cap_roots):
            baseline = next(
                row
                for row in rows
                if row["topology_id"] == "G-15" and row["sample_size"] == size
            )
            comparison = collect_one(
                cap_roots[size], "G-15", role_by_id["G-15"], size, 3000
            )
            cap_rows.extend((baseline, comparison))
            for field, failure_name in (
                ("test_hash", "test_hash_mismatch"),
                ("train_prefix_hash", "train_hash_mismatch"),
                ("validation_hash", "validation_hash_mismatch"),
            ):
                if baseline[field] != comparison[field]:
                    hash_failures.append(f"G-15:sample{size}:cap_check_{failure_name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "sample_size_sensitivity.csv"
    cap_path = output_dir / "g15_epoch_cap_check.csv"
    audit_path = output_dir / "sample_size_sensitivity_audit.json"
    atomic_write_csv(result_path, rows, FIELDS)
    if cap_rows:
        atomic_write_csv(cap_path, cap_rows, FIELDS)
    audit = {
        "passed": not hash_failures,
        "failures": hash_failures,
        "topology_count": len(roles),
        "sample_sizes": list(EXPECTED_SAMPLE_SIZES),
        "expected_rows": len(roles) * len(EXPECTED_SAMPLE_SIZES),
        "observed_rows": len(rows),
        "cap_check_rows": len(cap_rows),
        "test_hashes_by_topology": hashes_by_topology,
        "fit_prefix_nested_by_topology": fit_prefix_nested_by_topology,
        "primary_target": (
            "normalized_improvement_pp=100*(test_mean_normalized_gap_2stage-"
            "test_mean_normalized_gap_spoplus)"
        ),
        "outputs": {
            "sample_size_sensitivity_csv": str(result_path),
            "g15_epoch_cap_check_csv": str(cap_path) if cap_rows else None,
            "audit_json": str(audit_path),
        },
    }
    atomic_write_json(audit_path, audit)
    return audit


def parse_args() -> argparse.Namespace:
    experiment_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topologies-csv",
        type=Path,
        default=experiment_root / "configs" / "topologies.selected.csv",
    )
    parser.add_argument("--sample-root", action="append", type=parse_root, required=True)
    parser.add_argument(
        "--cap-root",
        action="append",
        type=parse_root,
        default=[],
        help="Optional repeated SAMPLE_SIZE=PATH roots for G-15 3000-epoch checks.",
    )
    parser.add_argument("--output-dir", type=Path, default=experiment_root / "results")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    roots = dict(args.sample_root)
    if len(roots) != len(args.sample_root):
        raise ValueError("duplicate --sample-root sample size")
    cap_roots = dict(args.cap_root)
    if len(cap_roots) != len(args.cap_root):
        raise ValueError("duplicate --cap-root sample size")
    audit = run_review(topology_roles(args.topologies_csv), roots, args.output_dir, cap_roots)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
