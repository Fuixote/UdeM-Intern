#!/usr/bin/env python3
"""Review Step5 paired jobs and export deterministic topology weak labels."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_weak_label_artifacts as builder  # noqa: E402


DEFAULT_TOPOLOGIES = EXPERIMENT_ROOT / "configs" / "topologies.locked.csv"
DEFAULT_MAX_EPOCHS = 1500
DEFAULT_METRIC_STRIDE = 1
DEFAULT_EARLY_STOP_PATIENCE = 20
DEFAULT_EARLY_STOP_MIN_DELTA = 0.0001
SUCCESS_FIELDS = {
    "status": "success",
    "2stage status": "success",
    "SPO+ status": "success",
    "evaluation status": "success",
}
JOB_METRIC_FIELDS = [
    "job_id",
    "topology_id",
    "manifest_index",
    "regime",
    "protocol",
    "data_seed",
    "sample_size",
    "training_size",
    "validation_size",
    "test_size",
    "theta_seed",
    "gurobi_seed",
    "max_epochs",
    "metric_stride",
    "early_stop_patience",
    "early_stop_min_delta",
    "status",
    "two_stage_status",
    "spoplus_status",
    "evaluation_status",
    "test_gap_2stage",
    "test_gap_spoplus",
    "delta",
    "paired_mean_improvement_over_2stage",
    "paired_improvement_matches_delta",
    "test_normalized_gap_2stage",
    "test_normalized_gap_spoplus",
    "fraction_improved_over_2stage",
    "weak_label_class",
    "weak_label",
    "weak_label_threshold",
    "label_protocol",
    "test_hash",
    "train_prefix_hash",
    "validation_hash",
    "job_dir",
    "failure",
]
LABEL_FIELDS = [
    "manifest_index",
    "data_seed",
    "sample_size",
    "training_size",
    "validation_size",
    "test_size",
    "theta_seed",
    "gurobi_seed",
    "max_epochs",
    "metric_stride",
    "early_stop_patience",
    "early_stop_min_delta",
    "protocol",
    "regime",
    "test_gap_2stage",
    "test_gap_spoplus",
    "delta",
    "weak_label_class",
    "weak_label",
    "weak_label_threshold",
    "label_protocol",
    "test_hash",
    "train_prefix_hash",
    "validation_hash",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
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
    rows: list[dict[str, Any]],
    fieldnames: list[str],
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
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        temp_name = handle.name
    os.replace(temp_name, output)


def to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def to_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def canonical_method(value: Any) -> str:
    normalized = str(value).strip().lower().replace("_", "").replace("-", "")
    if normalized in {"2stage", "twostage"}:
        return "2stage"
    if normalized in {"spo+", "spoplus", "spo"}:
        return "spoplus"
    return normalized


def method_rows_from_summary(path: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError("test_summary.json must contain a list")
    rows: dict[str, dict[str, Any]] = {}
    for row in payload:
        method = canonical_method(row.get("method"))
        if method in rows:
            raise ValueError(f"duplicate method {method}")
        rows[method] = row
    return rows


def classify_delta(delta: float, threshold: float = 0.1) -> str:
    if float(delta) > float(threshold):
        return "helpful"
    if float(delta) < -float(threshold):
        return "harmful"
    return "near_neutral"


def label_protocol_text(threshold: float) -> str:
    return (
        "delta=test_gap_2stage-test_gap_spoplus; "
        f"helpful if delta>{float(threshold)}; harmful if delta<{-float(threshold)}; "
        f"near_neutral if abs(delta)<={float(threshold)}; weak_label=true"
    )


def expected_job_dir(
    output_root: str | Path,
    *,
    regime: str,
    topology_id: str,
    data_seed: int,
    sample_size: int,
) -> Path:
    return (
        Path(output_root)
        / "jobs"
        / str(regime)
        / str(topology_id)
        / f"data_seed={int(data_seed):06d}"
        / f"sample_size={int(sample_size):03d}"
    )


def review_one_job(
    topology_row: dict[str, str],
    *,
    manifest_index: int,
    output_root: str | Path,
    regime: str,
    protocol: str,
    data_seed: int,
    sample_size: int,
    test_size: int,
    theta_seed: int,
    gurobi_seed: int,
    max_epochs: int,
    metric_stride: int,
    early_stop_patience: int,
    early_stop_min_delta: float,
    threshold: float,
) -> tuple[dict[str, Any], dict[str, Any] | None, list[str]]:
    topology_id = str(topology_row["topology_id"])
    training_size, validation_size = builder.validate_sample_size(sample_size)
    job_dir = expected_job_dir(
        output_root,
        regime=regime,
        topology_id=topology_id,
        data_seed=data_seed,
        sample_size=sample_size,
    )
    status_path = job_dir / "job_status.json"
    manifest_path = job_dir / "paired_job_manifest.json"
    summary_path = job_dir / "evaluation" / "metrics" / "test_summary.json"
    failures: list[str] = []
    status: dict[str, Any] = {}
    manifest: dict[str, Any] = {}
    methods: dict[str, dict[str, Any]] = {}
    for path, label in (
        (status_path, "job_status_missing"),
        (manifest_path, "paired_job_manifest_missing"),
        (summary_path, "test_summary_missing"),
    ):
        if not path.is_file():
            failures.append(label)
    if status_path.is_file():
        try:
            status = read_json(status_path)
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"job_status_invalid:{exc}")
    if manifest_path.is_file():
        try:
            manifest = read_json(manifest_path)
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"paired_job_manifest_invalid:{exc}")
    if summary_path.is_file():
        try:
            methods = method_rows_from_summary(summary_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            failures.append(f"test_summary_invalid:{exc}")

    for field, expected in SUCCESS_FIELDS.items():
        if status.get(field) != expected:
            failures.append(f"{field}={status.get(field)}")
    expected_manifest = {
        "topology_id": topology_id,
        "regime": str(regime),
        "protocol": str(protocol),
        "train_seed": int(data_seed),
        "sample_size": int(sample_size),
        "training_size": training_size,
        "validation_size": validation_size,
        "trainer_train_size_arg": training_size,
        "theta_seed": int(theta_seed),
        "gurobi_seed": int(gurobi_seed),
        "max_epochs": int(max_epochs),
        "metric_stride": int(metric_stride),
        "early_stop_patience": int(early_stop_patience),
    }
    for field, expected in expected_manifest.items():
        observed = manifest.get(field)
        if isinstance(expected, int):
            observed = to_int(observed)
        else:
            observed = str(observed or "")
        if observed != expected:
            failures.append(f"manifest_{field}_mismatch:{observed}!={expected}")
    observed_min_delta = to_float(manifest.get("early_stop_min_delta"))
    if observed_min_delta is None or not math.isclose(
        observed_min_delta,
        float(early_stop_min_delta),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        failures.append(
            "manifest_early_stop_min_delta_mismatch:"
            f"{observed_min_delta}!={float(early_stop_min_delta)}"
        )
    if set(methods) != {"2stage", "spoplus"}:
        failures.append(f"test_methods_mismatch:{sorted(methods)}")

    two_stage = methods.get("2stage", {})
    spoplus = methods.get("spoplus", {})
    gap_2stage = to_float(two_stage.get("test_mean_decision_gap"))
    gap_spoplus = to_float(spoplus.get("test_mean_decision_gap"))
    delta = None if gap_2stage is None or gap_spoplus is None else gap_2stage - gap_spoplus
    if delta is None:
        failures.append("delta_unavailable")
    paired_improvement = to_float(spoplus.get("paired_mean_improvement_over_2stage"))
    paired_matches = None
    if delta is not None and paired_improvement is not None:
        paired_matches = math.isclose(delta, paired_improvement, rel_tol=1e-9, abs_tol=1e-9)
        if not paired_matches:
            failures.append("paired_improvement_delta_mismatch")
    elif delta is not None:
        failures.append("paired_improvement_missing")

    weak_class = "" if delta is None else classify_delta(delta, threshold)
    normalized_gap_2stage = to_float(two_stage.get("test_mean_normalized_gap"))
    normalized_gap_spoplus = to_float(spoplus.get("test_mean_normalized_gap"))
    fraction_improved = to_float(spoplus.get("fraction_improved_over_2stage"))
    label_protocol = label_protocol_text(threshold)
    row = {
        "job_id": manifest.get("job_id", status.get("job_id", "")),
        "topology_id": topology_id,
        "manifest_index": int(manifest_index),
        "regime": str(regime),
        "protocol": str(protocol),
        "data_seed": int(data_seed),
        "sample_size": int(sample_size),
        "training_size": training_size,
        "validation_size": validation_size,
        "test_size": int(test_size),
        "theta_seed": int(theta_seed),
        "gurobi_seed": int(gurobi_seed),
        "max_epochs": int(max_epochs),
        "metric_stride": int(metric_stride),
        "early_stop_patience": int(early_stop_patience),
        "early_stop_min_delta": float(early_stop_min_delta),
        "status": status.get("status", "missing"),
        "two_stage_status": status.get("2stage status", "missing"),
        "spoplus_status": status.get("SPO+ status", "missing"),
        "evaluation_status": status.get("evaluation status", "missing"),
        "test_gap_2stage": "" if gap_2stage is None else gap_2stage,
        "test_gap_spoplus": "" if gap_spoplus is None else gap_spoplus,
        "delta": "" if delta is None else delta,
        "paired_mean_improvement_over_2stage": (
            "" if paired_improvement is None else paired_improvement
        ),
        "paired_improvement_matches_delta": "" if paired_matches is None else paired_matches,
        "test_normalized_gap_2stage": (
            "" if normalized_gap_2stage is None else normalized_gap_2stage
        ),
        "test_normalized_gap_spoplus": (
            "" if normalized_gap_spoplus is None else normalized_gap_spoplus
        ),
        "fraction_improved_over_2stage": "" if fraction_improved is None else fraction_improved,
        "weak_label_class": weak_class,
        "weak_label": True,
        "weak_label_threshold": float(threshold),
        "label_protocol": label_protocol,
        "test_hash": manifest.get("test_hash", ""),
        "train_prefix_hash": manifest.get("train_prefix_hash", ""),
        "validation_hash": manifest.get("validation_hash", ""),
        "job_dir": str(job_dir),
        "failure": ";".join(failures),
    }
    label_row = None
    if not failures and delta is not None:
        label_row = {
            **topology_row,
            "manifest_index": int(manifest_index),
            "data_seed": int(data_seed),
            "sample_size": int(sample_size),
            "training_size": training_size,
            "validation_size": validation_size,
            "test_size": int(test_size),
            "theta_seed": int(theta_seed),
            "gurobi_seed": int(gurobi_seed),
            "max_epochs": int(max_epochs),
            "metric_stride": int(metric_stride),
            "early_stop_patience": int(early_stop_patience),
            "early_stop_min_delta": float(early_stop_min_delta),
            "protocol": str(protocol),
            "regime": str(regime),
            "test_gap_2stage": gap_2stage,
            "test_gap_spoplus": gap_spoplus,
            "delta": delta,
            "weak_label_class": weak_class,
            "weak_label": True,
            "weak_label_threshold": float(threshold),
            "label_protocol": label_protocol,
            "test_hash": manifest.get("test_hash", ""),
            "train_prefix_hash": manifest.get("train_prefix_hash", ""),
            "validation_hash": manifest.get("validation_hash", ""),
        }
    return row, label_row, failures


def review_results(
    topology_rows: list[dict[str, str]],
    *,
    output_root: str | Path,
    output_dir: str | Path,
    regime: str,
    protocol: str,
    data_seed: int,
    sample_size: int,
    test_size: int,
    theta_seed: int,
    gurobi_seed: int,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    metric_stride: int = DEFAULT_METRIC_STRIDE,
    early_stop_patience: int = DEFAULT_EARLY_STOP_PATIENCE,
    early_stop_min_delta: float = DEFAULT_EARLY_STOP_MIN_DELTA,
    threshold: float = 0.1,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    job_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for index, topology_row in enumerate(topology_rows):
        job_row, label_row, job_failures = review_one_job(
            topology_row,
            manifest_index=index,
            output_root=output_root,
            regime=regime,
            protocol=protocol,
            data_seed=data_seed,
            sample_size=sample_size,
            test_size=test_size,
            theta_seed=theta_seed,
            gurobi_seed=gurobi_seed,
            max_epochs=max_epochs,
            metric_stride=metric_stride,
            early_stop_patience=early_stop_patience,
            early_stop_min_delta=early_stop_min_delta,
            threshold=threshold,
        )
        job_rows.append(job_row)
        if label_row is not None:
            label_rows.append(label_row)
        failures.extend(f"{topology_row['topology_id']}:{failure}" for failure in job_failures)

    expected_status_paths = {
        expected_job_dir(
            output_root,
            regime=regime,
            topology_id=row["topology_id"],
            data_seed=data_seed,
            sample_size=sample_size,
        )
        / "job_status.json"
        for row in topology_rows
    }
    observed_status_paths = set((Path(output_root) / "jobs").glob("**/job_status.json"))
    extra_status_paths = sorted(str(path) for path in observed_status_paths - expected_status_paths)
    if extra_status_paths:
        failures.append(f"unexpected_job_status_files:{len(extra_status_paths)}")

    success_count = sum(
        all(row[field] == expected for field, expected in {
            "status": "success",
            "two_stage_status": "success",
            "spoplus_status": "success",
            "evaluation_status": "success",
        }.items())
        for row in job_rows
    )
    topology_ids = [str(row["topology_id"]) for row in label_rows]
    if len(topology_ids) != len(set(topology_ids)):
        failures.append("duplicate_topology_labels")
    if len(label_rows) != len(topology_rows):
        failures.append(f"label_count_mismatch:{len(label_rows)}!={len(topology_rows)}")

    topology_fields = list(topology_rows[0].keys()) if topology_rows else ["topology_id"]
    label_fieldnames = [*topology_fields, *[field for field in LABEL_FIELDS if field not in topology_fields]]
    job_metrics_path = output_dir / "weak_label_job_metrics.csv"
    topology_summary_path = output_dir / "weak_label_topology_summary.csv"
    audit_path = output_dir / "weak_label_integrity_audit.json"
    atomic_write_csv(job_metrics_path, job_rows, JOB_METRIC_FIELDS)
    atomic_write_csv(topology_summary_path, label_rows, label_fieldnames)

    class_counts = dict(sorted(Counter(row["weak_label_class"] for row in label_rows).items()))
    audit = {
        "passed": not failures,
        "failures": failures,
        "expected_jobs": len(topology_rows),
        "job_rows": sum(Path(row["job_dir"]).joinpath("job_status.json").is_file() for row in job_rows),
        "success": success_count,
        "label_rows": len(label_rows),
        "topology_count": len({row["topology_id"] for row in label_rows}),
        "class_counts": class_counts,
        "unexpected_job_status_files": extra_status_paths,
        "label_protocol": {
            "continuous_label": "delta=test_gap_2stage-test_gap_spoplus",
            "helpful": f"delta>{float(threshold)}",
            "harmful": f"delta<{-float(threshold)}",
            "near_neutral": f"abs(delta)<={float(threshold)}",
            "weak_label": True,
            "data_seed": int(data_seed),
            "sample_size": int(sample_size),
            "training_size": builder.validate_sample_size(sample_size)[0],
            "validation_size": builder.validate_sample_size(sample_size)[1],
            "test_size": int(test_size),
            "theta_seed": int(theta_seed),
            "gurobi_seed": int(gurobi_seed),
            "max_epochs": int(max_epochs),
            "metric_stride": int(metric_stride),
            "early_stop_patience": int(early_stop_patience),
            "early_stop_min_delta": float(early_stop_min_delta),
            "protocol": str(protocol),
            "regime": str(regime),
        },
        "outputs": {
            "job_metrics_csv": str(job_metrics_path),
            "topology_summary_csv": str(topology_summary_path),
            "integrity_audit_json": str(audit_path),
        },
    }
    atomic_write_json(audit_path, audit)
    return audit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topologies-csv", type=Path, default=DEFAULT_TOPOLOGIES)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--regime", default=builder.DEFAULT_REGIME)
    parser.add_argument("--protocol", choices=("screen", "confirm"), default="screen")
    parser.add_argument("--data-seed", type=int, default=builder.DEFAULT_DATA_SEED)
    parser.add_argument("--sample-size", type=int, default=builder.DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--test-size", type=int, default=builder.DEFAULT_TEST_SIZE)
    parser.add_argument("--theta-seed", type=int, default=42)
    parser.add_argument("--gurobi-seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--metric-stride", type=int, default=DEFAULT_METRIC_STRIDE)
    parser.add_argument("--early-stop-patience", type=int, default=DEFAULT_EARLY_STOP_PATIENCE)
    parser.add_argument("--early-stop-min-delta", type=float, default=DEFAULT_EARLY_STOP_MIN_DELTA)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--topology-id", action="append", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.threshold < 0:
        raise ValueError("threshold must be non-negative")
    rows = builder.selected_topology_rows(
        args.topologies_csv,
        topology_ids=args.topology_id,
        limit=args.limit,
    )
    output_dir = args.output_dir or Path(args.output_root) / "results"
    audit = review_results(
        rows,
        output_root=args.output_root,
        output_dir=output_dir,
        regime=args.regime,
        protocol=args.protocol,
        data_seed=args.data_seed,
        sample_size=args.sample_size,
        test_size=args.test_size,
        theta_seed=args.theta_seed,
        gurobi_seed=args.gurobi_seed,
        max_epochs=args.max_epochs,
        metric_stride=args.metric_stride,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        threshold=args.threshold,
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
