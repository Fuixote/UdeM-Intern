#!/usr/bin/env python3
"""Post-run integrity audit and metric aggregation for K18-E1 formal jobs."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import re
from statistics import mean, median, pstdev
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[5]
EXPERIMENT_ROOT = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "Step3"
    / "K18_analysis"
    / "experiment_01_budget4to1"
)
DEFAULT_OUTPUT_ROOT = EXPERIMENT_ROOT / "results" / "formal_270_full_epoch_20260626"
EXPECTED_SAMPLE_MAP = {
    50: (40, 10),
    100: (80, 20),
    500: (400, 100),
}
SUCCESS_STATUS_KEYS = ("status", "2stage status", "SPO+ status", "evaluation status")
SUCCESS_VALUE = "success"
FINISHED_RE = re.compile(
    r"\[formal-launcher\] finished (?P<job_id>.+?) "
    r"status=(?P<status>\S+) rc=(?P<rc>-?\d+) elapsed=(?P<elapsed>[0-9.]+)s"
)
START_RE = re.compile(r"\[formal-launcher\] start (?P<timestamp>\S+) (?P<job_id>.+)")
LAUNCHER_JOB_ID_RE = re.compile(
    r"(?P<topology_id>G-\d+)\|data_seed=(?P<data_seed>\d+)\|"
    r"sample_size=(?P<sample_size>\d+)\|training=(?P<training_size>\d+)\|"
    r"validation=(?P<validation_size>\d+)"
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def to_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_int(value: Any) -> int | None:
    if value in ("", None):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def mean_or_none(values: list[float]) -> float | None:
    return mean(values) if values else None


def median_or_none(values: list[float]) -> float | None:
    return median(values) if values else None


def pstdev_or_none(values: list[float]) -> float | None:
    return pstdev(values) if len(values) > 1 else 0.0 if values else None


def runtime_key(topology_id: Any, data_seed: Any, sample_size: Any) -> tuple[str, int, int]:
    return str(topology_id), int(data_seed), int(sample_size)


def parse_launcher_runtimes(log_path: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    runtimes: dict[tuple[str, int, int], dict[str, Any]] = {}
    if not log_path.exists():
        return runtimes
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = FINISHED_RE.search(line)
        if not match:
            continue
        id_match = LAUNCHER_JOB_ID_RE.fullmatch(match.group("job_id"))
        if not id_match:
            continue
        key = runtime_key(
            id_match.group("topology_id"),
            id_match.group("data_seed"),
            id_match.group("sample_size"),
        )
        runtimes[key] = {
            "runtime_seconds": float(match.group("elapsed")),
            "runtime_source": "formal_launcher_log",
            "launcher_status": match.group("status"),
            "launcher_returncode": int(match.group("rc")),
        }
    return runtimes


def parse_job_log_start(log_path: Path) -> tuple[str | None, datetime | None]:
    if not log_path.exists():
        return None, None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()[:5]:
        match = START_RE.search(line)
        if not match:
            continue
        timestamp = match.group("timestamp")
        try:
            return timestamp, datetime.fromisoformat(timestamp)
        except ValueError:
            return timestamp, None
    return None, None


def relative_job_dir(job_status_path: Path, jobs_root: Path) -> Path:
    return job_status_path.parent.relative_to(jobs_root)


def runtime_from_mtime(
    *,
    topology_id: Any,
    data_seed: Any,
    sample_size: Any,
    training_size: Any,
    validation_size: Any,
    job_status_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    label = (
        f"{topology_id}_data_seed{int(data_seed):06d}_sample_size{int(sample_size):03d}_"
        f"training{int(training_size):03d}_validation{int(validation_size):03d}"
    )
    log_path = output_root / "logs" / "jobs" / f"{label}.log"
    start_text, start_dt = parse_job_log_start(log_path)
    if start_dt is None:
        return {
            "runtime_seconds": "",
            "runtime_source": "missing",
            "job_log_start": start_text or "",
        }
    end_dt = datetime.fromtimestamp(job_status_path.stat().st_mtime, tz=start_dt.tzinfo)
    return {
        "runtime_seconds": max(0.0, (end_dt - start_dt).total_seconds()),
        "runtime_source": "job_status_mtime_minus_job_log_start",
        "job_log_start": start_text or "",
    }


def method_rows_from_test_summary(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_json(path)
    return {str(row["method"]): row for row in rows}


def early_stop_record(path: Path) -> dict[str, Any]:
    data = read_json(path)
    max_epochs = to_int(data.get("max_epochs"))
    stop_epoch = to_int(data.get("stop_epoch", data.get("stopped_epoch")))
    best_epoch = to_int(data.get("best_epoch"))
    stopped_early = data.get("stopped_early")
    if stopped_early is None:
        stopped_early = bool(data.get("should_stop")) and stop_epoch is not None and (
            max_epochs is None or stop_epoch < max_epochs
        )
    hit_max_epoch = False
    if max_epochs is not None:
        hit_max_epoch = not bool(stopped_early) or (stop_epoch is not None and stop_epoch >= max_epochs)
    return {
        "enabled": data.get("enabled", ""),
        "metric": data.get("metric", ""),
        "best_epoch": "" if best_epoch is None else best_epoch,
        "stop_epoch": "" if stop_epoch is None else stop_epoch,
        "stopped_early": bool(stopped_early),
        "hit_max_epoch": bool(hit_max_epoch),
        "best_value": data.get("best_value", ""),
        "max_epochs": "" if max_epochs is None else max_epochs,
        "patience": data.get("patience", ""),
        "min_delta": data.get("min_delta", ""),
    }


def aggregate_rows(rows: list[dict[str, Any]], group_fields: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)
    output: list[dict[str, Any]] = []
    for key in sorted(groups):
        items = groups[key]
        record = {field: value for field, value in zip(group_fields, key)}
        record["job_count"] = len(items)
        for field in [
            "runtime_seconds",
            "test_gap_2stage",
            "test_gap_spoplus",
            "test_gap_improvement",
            "test_normalized_gap_2stage",
            "test_normalized_gap_spoplus",
            "fraction_improved_over_2stage",
        ]:
            values = [float(row[field]) for row in items if row.get(field) not in ("", None)]
            record[f"mean_{field}"] = mean_or_none(values)
            record[f"median_{field}"] = median_or_none(values)
            record[f"std_{field}"] = pstdev_or_none(values)
        for method in ("2stage", "spoplus"):
            prefix = "two_stage" if method == "2stage" else "spoplus"
            stop_values = [
                float(row[f"{prefix}_stop_epoch"])
                for row in items
                if row.get(f"{prefix}_stop_epoch") not in ("", None)
            ]
            best_values = [
                float(row[f"{prefix}_best_epoch"])
                for row in items
                if row.get(f"{prefix}_best_epoch") not in ("", None)
            ]
            record[f"{prefix}_stopped_early_count"] = sum(
                1 for row in items if str(row.get(f"{prefix}_stopped_early")) == "True"
            )
            record[f"{prefix}_hit_max_epoch_count"] = sum(
                1 for row in items if str(row.get(f"{prefix}_hit_max_epoch")) == "True"
            )
            record[f"mean_{prefix}_stop_epoch"] = mean_or_none(stop_values)
            record[f"median_{prefix}_stop_epoch"] = median_or_none(stop_values)
            record[f"mean_{prefix}_best_epoch"] = mean_or_none(best_values)
            record[f"median_{prefix}_best_epoch"] = median_or_none(best_values)
        output.append(record)
    return output


def topology_test_hash_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    hashes: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        hashes[str(row["topology_id"])].add(str(row["test_hash"]))
    return {topology: len(values) for topology, values in sorted(hashes.items())}


def review(output_root: Path, output_dir: Path, expected_jobs: int) -> dict[str, Any]:
    jobs_root = output_root / "jobs"
    launcher_log = output_root.parent / "logs" / f"{output_root.name}.log"
    if not launcher_log.exists():
        launcher_log = (
            EXPERIMENT_ROOT
            / "results"
            / "logs"
            / "formal_270_full_epoch_20260626.log"
        )
    launcher_runtimes = parse_launcher_runtimes(launcher_log)
    job_rows: list[dict[str, Any]] = []
    method_rows: list[dict[str, Any]] = []
    failures: list[str] = []

    job_status_paths = sorted(jobs_root.glob("**/job_status.json"))
    for status_path in job_status_paths:
        job_dir = status_path.parent
        status = read_json(status_path)
        manifest_path = job_dir / "paired_job_manifest.json"
        test_summary_path = job_dir / "evaluation" / "metrics" / "test_summary.json"
        two_stage_early_path = job_dir / "2stage" / "metrics" / "early_stopping_2stage.json"
        spoplus_early_path = job_dir / "spoplus" / "metrics" / "early_stopping.json"
        for path, label in [
            (manifest_path, "paired_manifest_missing"),
            (test_summary_path, "test_summary_missing"),
            (two_stage_early_path, "two_stage_early_stopping_missing"),
            (spoplus_early_path, "spoplus_early_stopping_missing"),
        ]:
            if not path.exists():
                failures.append(f"{label}:{job_dir}")
        manifest = read_json(manifest_path) if manifest_path.exists() else {}
        job_id = str(manifest.get("job_id") or status.get("job_id") or relative_job_dir(status_path, jobs_root))
        topology_id = manifest.get("topology_id", "")
        regime = manifest.get("regime", "")
        protocol = manifest.get("protocol", "")
        data_seed = manifest.get("train_seed", "")
        sample_size = to_int(manifest.get("sample_size"))
        training_size = to_int(manifest.get("training_size", manifest.get("train_size")))
        validation_size = to_int(manifest.get("validation_size"))
        trainer_train_size = to_int(manifest.get("trainer_train_size_arg"))
        expected_training, expected_validation = EXPECTED_SAMPLE_MAP.get(sample_size or -1, (None, None))
        if sample_size not in EXPECTED_SAMPLE_MAP:
            failures.append(f"unexpected_sample_size:{job_id}:{sample_size}")
        if expected_training != training_size or expected_validation != validation_size:
            failures.append(
                f"sample_training_validation_mismatch:{job_id}:"
                f"{sample_size}/{training_size}/{validation_size}"
            )
        if trainer_train_size != training_size:
            failures.append(f"trainer_train_size_mismatch:{job_id}:{trainer_train_size}!={training_size}")
        for key in SUCCESS_STATUS_KEYS:
            if status.get(key) != SUCCESS_VALUE:
                failures.append(f"job_status_not_success:{job_id}:{key}={status.get(key)}")

        test_rows = method_rows_from_test_summary(test_summary_path) if test_summary_path.exists() else {}
        if set(test_rows) != {"2stage", "spoplus"}:
            failures.append(f"test_summary_methods_mismatch:{job_id}:{sorted(test_rows)}")
        two_stage_early = early_stop_record(two_stage_early_path) if two_stage_early_path.exists() else {}
        spoplus_early = early_stop_record(spoplus_early_path) if spoplus_early_path.exists() else {}
        key = runtime_key(topology_id, data_seed, sample_size)
        runtime = launcher_runtimes.get(key)
        if runtime is None:
            runtime = runtime_from_mtime(
                topology_id=topology_id,
                data_seed=data_seed,
                sample_size=sample_size,
                training_size=training_size,
                validation_size=validation_size,
                job_status_path=status_path,
                output_root=output_root,
            )

        two_stage = test_rows.get("2stage", {})
        spoplus = test_rows.get("spoplus", {})
        gap_2stage = to_float(two_stage.get("test_mean_decision_gap"))
        gap_spoplus = to_float(spoplus.get("test_mean_decision_gap"))
        improvement = to_float(spoplus.get("paired_mean_improvement_over_2stage"))
        if improvement is None and gap_2stage is not None and gap_spoplus is not None:
            improvement = gap_2stage - gap_spoplus
        row = {
            "job_id": job_id,
            "topology_id": topology_id,
            "regime": regime,
            "protocol": protocol,
            "data_seed": data_seed,
            "sample_size": sample_size,
            "training_size": training_size,
            "validation_size": validation_size,
            "trainer_train_size_arg": trainer_train_size,
            "max_epochs": manifest.get("max_epochs", ""),
            "early_stop_patience": manifest.get("early_stop_patience", ""),
            "early_stop_min_delta": manifest.get("early_stop_min_delta", ""),
            "test_hash": manifest.get("test_hash", ""),
            "train_prefix_hash": manifest.get("train_prefix_hash", ""),
            "validation_hash": manifest.get("validation_hash", ""),
            "status": status.get("status", ""),
            "two_stage_status": status.get("2stage status", ""),
            "spoplus_status": status.get("SPO+ status", ""),
            "evaluation_status": status.get("evaluation status", ""),
            "runtime_seconds": runtime.get("runtime_seconds", ""),
            "runtime_source": runtime.get("runtime_source", ""),
            "test_gap_2stage": gap_2stage,
            "test_gap_spoplus": gap_spoplus,
            "test_gap_improvement": improvement,
            "test_normalized_gap_2stage": to_float(two_stage.get("test_mean_normalized_gap")),
            "test_normalized_gap_spoplus": to_float(spoplus.get("test_mean_normalized_gap")),
            "fraction_improved_over_2stage": to_float(spoplus.get("fraction_improved_over_2stage")),
            "two_stage_selected_epoch": two_stage.get("selected_epoch", ""),
            "spoplus_selected_epoch": spoplus.get("selected_epoch", ""),
            "two_stage_best_epoch": two_stage_early.get("best_epoch", ""),
            "two_stage_stop_epoch": two_stage_early.get("stop_epoch", ""),
            "two_stage_stopped_early": two_stage_early.get("stopped_early", ""),
            "two_stage_hit_max_epoch": two_stage_early.get("hit_max_epoch", ""),
            "two_stage_best_value": two_stage_early.get("best_value", ""),
            "spoplus_best_epoch": spoplus_early.get("best_epoch", ""),
            "spoplus_stop_epoch": spoplus_early.get("stop_epoch", ""),
            "spoplus_stopped_early": spoplus_early.get("stopped_early", ""),
            "spoplus_hit_max_epoch": spoplus_early.get("hit_max_epoch", ""),
            "spoplus_best_value": spoplus_early.get("best_value", ""),
            "job_dir": str(job_dir),
        }
        job_rows.append(row)
        for method, metrics, early in [
            ("2stage", two_stage, two_stage_early),
            ("spoplus", spoplus, spoplus_early),
        ]:
            method_rows.append(
                {
                    "job_id": job_id,
                    "topology_id": row["topology_id"],
                    "data_seed": row["data_seed"],
                    "sample_size": sample_size,
                    "training_size": training_size,
                    "validation_size": validation_size,
                    "method": method,
                    "selected_epoch": metrics.get("selected_epoch", ""),
                    "selection_metric": metrics.get("selection_metric", ""),
                    "selection_value": metrics.get("selection_value", ""),
                    "test_mean_decision_gap": metrics.get("test_mean_decision_gap", ""),
                    "test_mean_normalized_gap": metrics.get("test_mean_normalized_gap", ""),
                    "best_epoch": early.get("best_epoch", ""),
                    "stop_epoch": early.get("stop_epoch", ""),
                    "stopped_early": early.get("stopped_early", ""),
                    "hit_max_epoch": early.get("hit_max_epoch", ""),
                    "best_value": early.get("best_value", ""),
                    "max_epochs": early.get("max_epochs", ""),
                }
            )

    counts = {
        "job_status_files": len(job_status_paths),
        "job_rows": len(job_rows),
        "topology_count": len({row["topology_id"] for row in job_rows}),
        "data_seed_count": len({row["data_seed"] for row in job_rows}),
        "sample_size_counts": dict(sorted(Counter(int(row["sample_size"]) for row in job_rows).items())),
        "training_size_counts": dict(sorted(Counter(int(row["training_size"]) for row in job_rows).items())),
        "validation_size_counts": dict(sorted(Counter(int(row["validation_size"]) for row in job_rows).items())),
        "status_counts": dict(sorted(Counter(str(row["status"]) for row in job_rows).items())),
        "runtime_source_counts": dict(sorted(Counter(str(row["runtime_source"]) for row in job_rows).items())),
        "test_hash_unique_count": len({row["test_hash"] for row in job_rows}),
        "test_hashes_per_topology": topology_test_hash_counts(job_rows),
    }
    if len(job_rows) != expected_jobs:
        failures.append(f"job_count_mismatch:{len(job_rows)}!={expected_jobs}")
    if counts["sample_size_counts"] != {50: 90, 100: 90, 500: 90}:
        failures.append(f"sample_size_counts_mismatch:{counts['sample_size_counts']}")
    if any(count != 1 for count in counts["test_hashes_per_topology"].values()):
        failures.append(f"test_hash_not_fixed_per_topology:{counts['test_hashes_per_topology']}")
    if counts["test_hash_unique_count"] != 18:
        failures.append(f"test_hash_unique_count_mismatch:{counts['test_hash_unique_count']}")

    job_fieldnames = [
        "job_id",
        "topology_id",
        "regime",
        "protocol",
        "data_seed",
        "sample_size",
        "training_size",
        "validation_size",
        "trainer_train_size_arg",
        "max_epochs",
        "early_stop_patience",
        "early_stop_min_delta",
        "status",
        "two_stage_status",
        "spoplus_status",
        "evaluation_status",
        "runtime_seconds",
        "runtime_source",
        "test_gap_2stage",
        "test_gap_spoplus",
        "test_gap_improvement",
        "test_normalized_gap_2stage",
        "test_normalized_gap_spoplus",
        "fraction_improved_over_2stage",
        "two_stage_selected_epoch",
        "spoplus_selected_epoch",
        "two_stage_best_epoch",
        "two_stage_stop_epoch",
        "two_stage_stopped_early",
        "two_stage_hit_max_epoch",
        "two_stage_best_value",
        "spoplus_best_epoch",
        "spoplus_stop_epoch",
        "spoplus_stopped_early",
        "spoplus_hit_max_epoch",
        "spoplus_best_value",
        "test_hash",
        "train_prefix_hash",
        "validation_hash",
        "job_dir",
    ]
    method_fieldnames = [
        "job_id",
        "topology_id",
        "data_seed",
        "sample_size",
        "training_size",
        "validation_size",
        "method",
        "selected_epoch",
        "selection_metric",
        "selection_value",
        "test_mean_decision_gap",
        "test_mean_normalized_gap",
        "best_epoch",
        "stop_epoch",
        "stopped_early",
        "hit_max_epoch",
        "best_value",
        "max_epochs",
    ]
    aggregate_fieldnames = [
        "sample_size",
        "job_count",
        "mean_runtime_seconds",
        "median_runtime_seconds",
        "std_runtime_seconds",
        "mean_test_gap_2stage",
        "mean_test_gap_spoplus",
        "mean_test_gap_improvement",
        "mean_test_normalized_gap_2stage",
        "mean_test_normalized_gap_spoplus",
        "mean_fraction_improved_over_2stage",
        "two_stage_stopped_early_count",
        "two_stage_hit_max_epoch_count",
        "mean_two_stage_stop_epoch",
        "median_two_stage_stop_epoch",
        "spoplus_stopped_early_count",
        "spoplus_hit_max_epoch_count",
        "mean_spoplus_stop_epoch",
        "median_spoplus_stop_epoch",
    ]
    topology_aggregate_fieldnames = ["topology_id", *aggregate_fieldnames]

    sample_summary = aggregate_rows(job_rows, ["sample_size"])
    topology_sample_summary = aggregate_rows(job_rows, ["topology_id", "sample_size"])
    method_sample_summary = aggregate_rows(
        [
            {
                **row,
                "runtime_seconds": "",
                "test_gap_2stage": row["test_mean_decision_gap"] if row["method"] == "2stage" else "",
                "test_gap_spoplus": row["test_mean_decision_gap"] if row["method"] == "spoplus" else "",
                "test_gap_improvement": "",
                "test_normalized_gap_2stage": row["test_mean_normalized_gap"] if row["method"] == "2stage" else "",
                "test_normalized_gap_spoplus": row["test_mean_normalized_gap"] if row["method"] == "spoplus" else "",
                "fraction_improved_over_2stage": "",
                "two_stage_stop_epoch": row["stop_epoch"] if row["method"] == "2stage" else "",
                "two_stage_best_epoch": row["best_epoch"] if row["method"] == "2stage" else "",
                "two_stage_stopped_early": row["stopped_early"] if row["method"] == "2stage" else False,
                "two_stage_hit_max_epoch": row["hit_max_epoch"] if row["method"] == "2stage" else False,
                "spoplus_stop_epoch": row["stop_epoch"] if row["method"] == "spoplus" else "",
                "spoplus_best_epoch": row["best_epoch"] if row["method"] == "spoplus" else "",
                "spoplus_stopped_early": row["stopped_early"] if row["method"] == "spoplus" else False,
                "spoplus_hit_max_epoch": row["hit_max_epoch"] if row["method"] == "spoplus" else False,
            }
            for row in method_rows
        ],
        ["method", "sample_size"],
    )

    write_csv(output_dir / "formal_post_run_job_metrics.csv", job_rows, job_fieldnames)
    write_csv(output_dir / "formal_post_run_method_metrics.csv", method_rows, method_fieldnames)
    write_csv(output_dir / "formal_post_run_sample_size_summary.csv", sample_summary, aggregate_fieldnames)
    write_csv(
        output_dir / "formal_post_run_topology_sample_summary.csv",
        topology_sample_summary,
        topology_aggregate_fieldnames,
    )
    write_csv(
        output_dir / "formal_post_run_method_sample_summary.csv",
        method_sample_summary,
        [
            "method",
            *aggregate_fieldnames,
        ],
    )

    audit = {
        "passed": not failures,
        "failures": failures,
        "counts": counts,
        "output_root": str(output_root),
        "output_dir": str(output_dir),
        "launcher_log": str(launcher_log),
    }
    write_json(output_dir / "formal_post_run_integrity_audit.json", audit)

    summary = {
        "audit_passed": audit["passed"],
        "failure_count": len(failures),
        "counts": counts,
        "sample_size_summary": sample_summary,
        "method_sample_summary": method_sample_summary,
        "topology_sample_summary_rows": len(topology_sample_summary),
        "outputs": {
            "job_metrics_csv": str(output_dir / "formal_post_run_job_metrics.csv"),
            "method_metrics_csv": str(output_dir / "formal_post_run_method_metrics.csv"),
            "sample_size_summary_csv": str(output_dir / "formal_post_run_sample_size_summary.csv"),
            "topology_sample_summary_csv": str(output_dir / "formal_post_run_topology_sample_summary.csv"),
            "method_sample_summary_csv": str(output_dir / "formal_post_run_method_sample_summary.csv"),
            "integrity_audit_json": str(output_dir / "formal_post_run_integrity_audit.json"),
        },
    }
    write_json(output_dir / "formal_post_run_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-jobs", type=int, default=270)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root
    output_dir = args.output_dir or output_root / "post_run_review"
    summary = review(output_root, output_dir, args.expected_jobs)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
