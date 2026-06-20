#!/usr/bin/env python3
"""Run Step3 Phase-B fixed-topology cheap DFL screening training jobs."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
STEP1C_DIR = PROJECT_ROOT / "surrogate_experiment_results" / "Step1c"
STEP3_DIR = PROJECT_ROOT / "surrogate_experiment_results" / "Step3"
TRAIN_2STAGE_SCRIPT = STEP1C_DIR / "train_2stage.py"
TRAIN_2STAGE_EARLYSTOP_SCRIPT = STEP3_DIR / "scripts" / "train_2stage_earlystop.py"
TRAIN_SPOPLUS_SCRIPT = STEP1C_DIR / "train_spoplus.py"
EVALUATE_MODELS_SCRIPT = STEP1C_DIR / "evaluate_models.py"

DEFAULT_DATASET_DIR = (
    PROJECT_ROOT / "surrogate_experiment_results" / "Step3" / "pairs20_ndd2" / "phase_b" / "datasets"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "surrogate_experiment_results" / "Step3" / "pairs20_ndd2" / "phase_b" / "runs"
)
DEFAULT_SPLIT_DIR = (
    PROJECT_ROOT / "surrogate_experiment_results" / "Step3" / "pairs20_ndd2" / "phase_b" / "splits"
)
DEFAULT_STATUS_PATH = (
    PROJECT_ROOT / "surrogate_experiment_results" / "Step3" / "pairs20_ndd2" / "phase_b" / "results" / "phase_b_training_status.csv"
)
DEFAULT_MANIFEST_PATH = (
    PROJECT_ROOT / "surrogate_experiment_results" / "Step3" / "pairs20_ndd2" / "phase_b" / "results" / "phase_b_training_manifest.csv"
)
DEFAULT_EARLY_STOP_PATIENCE = 20
DEFAULT_EARLY_STOP_MIN_DELTA = 0.0001

PRIMARY_WEIGHT_FILES = [
    "2stage_best_by_validation_mse_loss.npz",
    "spoplus_best_by_validation_spoplus_loss.npz",
]
GENERATED_WEIGHT_FILES = [
    "2stage_best_by_validation_mse_loss.npz",
    "spoplus_best_by_validation_decision_gap.npz",
    "spoplus_best_by_validation_spoplus_loss.npz",
]

MANIFEST_FIELDS = [
    "job_index",
    "topology_id",
    "train_seed",
    "train_sample_count",
    "validation_sample_count",
    "test_sample_count",
    "topology_hash",
    "topology_bank_hash",
    "feasible_set_hash",
    "train_dir",
    "validation_dir",
    "test_dir",
    "split_path",
    "run_dir",
]

STATUS_FIELDS = [
    "job_index",
    "topology_id",
    "train_seed",
    "status",
    "return_code",
    "elapsed_seconds",
    "split_seconds",
    "train_2stage_seconds",
    "train_spoplus_seconds",
    "posthoc_early_stop_seconds",
    "evaluate_seconds",
    "train_sample_count",
    "validation_sample_count",
    "test_sample_count",
    "validation_limit",
    "test_limit",
    "epochs_2stage",
    "epochs_spoplus",
    "early_stop_patience_2stage",
    "early_stop_min_delta_2stage",
    "early_stop_patience_spoplus",
    "early_stop_min_delta_spoplus",
    "test_mean_decision_gap_2stage",
    "test_mean_decision_gap_spoplus",
    "test_mean_normalized_gap_2stage",
    "test_mean_normalized_gap_spoplus",
    "spoplus_improvement_gap",
    "spoplus_improvement_normalized_gap",
    "run_dir",
    "message",
]


class PhaseBJob:
    def __init__(
        self,
        *,
        index: int,
        topology_id: str,
        train_seed: int,
        topology_dir: Path,
        train_dir: Path,
        validation_dir: Path,
        test_dir: Path,
        split_path: Path,
        run_dir: Path,
        train_sample_count: int,
        validation_sample_count: int,
        test_sample_count: int,
        topology_hash: str,
        topology_bank_hash: str,
        feasible_set_hash: str,
    ) -> None:
        self.index = int(index)
        self.topology_id = str(topology_id)
        self.train_seed = int(train_seed)
        self.topology_dir = Path(topology_dir)
        self.train_dir = Path(train_dir)
        self.validation_dir = Path(validation_dir)
        self.test_dir = Path(test_dir)
        self.split_path = Path(split_path)
        self.run_dir = Path(run_dir)
        self.train_sample_count = int(train_sample_count)
        self.validation_sample_count = int(validation_sample_count)
        self.test_sample_count = int(test_sample_count)
        self.topology_hash = str(topology_hash)
        self.topology_bank_hash = str(topology_bank_hash)
        self.feasible_set_hash = str(feasible_set_hash)


class PhaseBOptions:
    def __init__(
        self,
        *,
        project_root: Path = PROJECT_ROOT,
        python_bin: str | None = None,
        epochs_2stage: int = 100,
        epochs_spoplus: int = 100,
        lr_2stage: float = 0.05,
        lr_spoplus: float = 0.1,
        metric_stride: int = 10,
        theta_seed: int = 42,
        gurobi_seed: int = 42,
        bootstrap_samples: int = 1000,
        bootstrap_seed: int = 42,
        validation_limit: int | None = None,
        test_limit: int | None = None,
        thread_count: int = 1,
        include_decision_gap_checkpoint: bool = False,
        skip_completed: bool = True,
        early_stop_patience_2stage: int = DEFAULT_EARLY_STOP_PATIENCE,
        early_stop_min_delta_2stage: float = DEFAULT_EARLY_STOP_MIN_DELTA,
        early_stop_patience_spoplus: int = DEFAULT_EARLY_STOP_PATIENCE,
        early_stop_min_delta_spoplus: float = DEFAULT_EARLY_STOP_MIN_DELTA,
    ) -> None:
        self.project_root = Path(project_root)
        self.python_bin = python_bin or os.environ.get("KEP_PYTHON") or sys.executable
        self.epochs_2stage = int(epochs_2stage)
        self.epochs_spoplus = int(epochs_spoplus)
        self.lr_2stage = float(lr_2stage)
        self.lr_spoplus = float(lr_spoplus)
        self.metric_stride = int(metric_stride)
        self.theta_seed = int(theta_seed)
        self.gurobi_seed = int(gurobi_seed)
        self.bootstrap_samples = int(bootstrap_samples)
        self.bootstrap_seed = int(bootstrap_seed)
        self.validation_limit = validation_limit
        self.test_limit = test_limit
        self.thread_count = int(thread_count)
        self.include_decision_gap_checkpoint = bool(include_decision_gap_checkpoint)
        self.skip_completed = bool(skip_completed)
        self.early_stop_patience_2stage = int(early_stop_patience_2stage)
        self.early_stop_min_delta_2stage = float(early_stop_min_delta_2stage)
        self.early_stop_patience_spoplus = int(early_stop_patience_spoplus)
        self.early_stop_min_delta_spoplus = float(early_stop_min_delta_spoplus)

    @property
    def early_stop_enabled_2stage(self) -> bool:
        return self.early_stop_patience_2stage > 0

    @property
    def early_stop_enabled_spoplus(self) -> bool:
        return self.early_stop_patience_spoplus > 0

    @property
    def early_stop_enabled(self) -> bool:
        return self.early_stop_enabled_2stage or self.early_stop_enabled_spoplus


class JobResult:
    def __init__(
        self,
        *,
        job: PhaseBJob,
        status: str,
        return_code: int,
        elapsed_seconds: float,
        split_seconds: float = 0.0,
        train_2stage_seconds: float = 0.0,
        train_spoplus_seconds: float = 0.0,
        posthoc_early_stop_seconds: float = 0.0,
        evaluate_seconds: float = 0.0,
        metrics: dict[str, float | str] | None = None,
        message: str = "",
    ) -> None:
        self.job = job
        self.status = status
        self.return_code = int(return_code)
        self.elapsed_seconds = float(elapsed_seconds)
        self.split_seconds = float(split_seconds)
        self.train_2stage_seconds = float(train_2stage_seconds)
        self.train_spoplus_seconds = float(train_spoplus_seconds)
        self.posthoc_early_stop_seconds = float(posthoc_early_stop_seconds)
        self.evaluate_seconds = float(evaluate_seconds)
        self.metrics = metrics or {}
        self.message = message


def int_or_text_key(value: Any) -> tuple[int, int | str]:
    text = str(value)
    if text.startswith("G-"):
        text = text[2:]
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def read_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def simulate_native_early_stopping(
    *,
    epochs: list[int],
    values: list[float],
    patience: int,
    min_delta: float,
    metric_name: str,
    max_epochs: int,
    metric_stride: int,
) -> dict[str, Any]:
    if patience <= 0:
        raise ValueError("early stopping patience must be positive")
    if len(epochs) != len(values) or not epochs:
        raise ValueError("epochs and values must be non-empty and have the same length")
    best_epoch = int(epochs[0])
    best_value = float(values[0])
    bad_checks = 0
    stop_epoch = int(epochs[-1])
    stopped_early = False
    evaluated_epochs = []
    for index, (epoch, value) in enumerate(zip(epochs, values)):
        epoch = int(epoch)
        value = float(value)
        evaluated_epochs.append(epoch)
        if index == 0:
            continue
        if value < best_value - float(min_delta):
            best_epoch = epoch
            best_value = value
            bad_checks = 0
            continue
        bad_checks += 1
        if bad_checks >= patience:
            stop_epoch = epoch
            stopped_early = epoch < int(max_epochs)
            break
    return {
        "enabled": True,
        "source": "step3_native",
        "metric": metric_name,
        "patience": int(patience),
        "min_delta": float(min_delta),
        "max_epochs": int(max_epochs),
        "metric_stride": int(metric_stride),
        "best_epoch": int(best_epoch),
        "best_value": float(best_value),
        "stop_epoch": int(stop_epoch),
        "stopped_epoch": int(stop_epoch),
        "stopped_early": bool(stopped_early),
        "num_bad_checks_at_stop": int(bad_checks),
        "evaluated_epochs": evaluated_epochs,
    }


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def graph_sort_key(path: Path) -> tuple[int, int | str]:
    stem = path.stem
    if stem.startswith("G-"):
        stem = stem[2:]
    try:
        return (0, int(stem))
    except ValueError:
        return (1, stem)


def graph_entry(path: Path) -> dict[str, Any]:
    key = graph_sort_key(path)[1]
    return {
        "index": key,
        "graph_id": key,
        "path": str(path),
    }


def list_graph_files(path: Path) -> list[Path]:
    files = sorted(Path(path).glob("G-*.json"), key=graph_sort_key)
    if not files:
        raise FileNotFoundError(f"No G-*.json files found in {path}")
    return files


def train_seed_dirs(topology_dir: Path) -> list[tuple[int, Path]]:
    pairs: list[tuple[int, Path]] = []
    for train_path in topology_dir.glob("train_seed=*/train"):
        name = train_path.parent.name
        try:
            seed = int(name.split("=", 1)[1])
        except (IndexError, ValueError):
            continue
        pairs.append((seed, train_path))
    return sorted(pairs)


def select_index_rows(
    rows: list[dict[str, Any]],
    topology_ids: list[str] | None = None,
    max_topologies: int | None = None,
) -> list[dict[str, Any]]:
    selected = sorted(rows, key=lambda row: int_or_text_key(row["topology_id"]))
    if topology_ids:
        requested = list(dict.fromkeys(str(value) for value in topology_ids))
        by_id = {str(row["topology_id"]): row for row in selected}
        missing = [topology_id for topology_id in requested if topology_id not in by_id]
        if missing:
            raise ValueError(f"Requested topology IDs missing from dataset index: {missing}")
        selected = [by_id[topology_id] for topology_id in requested]
    if max_topologies is not None:
        selected = selected[: int(max_topologies)]
    return selected


def discover_phase_b_jobs(
    *,
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    split_dir: str | Path = DEFAULT_SPLIT_DIR,
    topology_ids: list[str] | None = None,
    max_topologies: int | None = None,
    train_seeds: list[int] | None = None,
    max_train_seeds: int | None = None,
) -> list[PhaseBJob]:
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    split_dir = Path(split_dir)
    index_path = dataset_dir / "phase_b_dataset_index.csv"
    rows = select_index_rows(
        read_csv_rows(index_path),
        topology_ids=topology_ids,
        max_topologies=max_topologies,
    )
    jobs: list[PhaseBJob] = []
    for row in rows:
        topology_id = str(row["topology_id"])
        topology_dir = Path(row.get("output_dir") or dataset_dir / topology_id)
        if not topology_dir.is_absolute() and not topology_dir.exists():
            topology_dir = dataset_dir / topology_id
        available_seed_dirs = train_seed_dirs(topology_dir)
        available_seeds = [seed for seed, _ in available_seed_dirs]
        seed_to_train_dir = dict(available_seed_dirs)
        selected_seeds = train_seeds if train_seeds is not None else available_seeds
        selected_seeds = [int(seed) for seed in selected_seeds if int(seed) in seed_to_train_dir]
        if max_train_seeds is not None:
            selected_seeds = selected_seeds[: int(max_train_seeds)]
        for seed in selected_seeds:
            jobs.append(
                PhaseBJob(
                    index=len(jobs) + 1,
                    topology_id=topology_id,
                    train_seed=seed,
                    topology_dir=topology_dir,
                    train_dir=seed_to_train_dir[seed],
                    validation_dir=topology_dir / "validation",
                    test_dir=topology_dir / "test",
                    split_path=split_dir / topology_id / f"train_seed={seed:06d}.json",
                    run_dir=output_dir / topology_id / f"train_seed={seed:06d}",
                    train_sample_count=int(row["train_sample_count"]),
                    validation_sample_count=int(row["validation_sample_count"]),
                    test_sample_count=int(row["test_sample_count"]),
                    topology_hash=str(row.get("topology_hash", "")),
                    topology_bank_hash=str(row.get("topology_bank_hash", "")),
                    feasible_set_hash=str(row.get("feasible_set_hash", "")),
                )
            )
    return jobs


def make_split_payload(job: PhaseBJob) -> dict[str, Any]:
    train_files = list_graph_files(job.train_dir)
    test_files = list_graph_files(job.test_dir)
    return {
        "seed": int(job.train_seed),
        "topology_id": job.topology_id,
        "train_pool_size": len(train_files),
        "validation_size": 0,
        "test_size": len(test_files),
        "train_pool": [graph_entry(path) for path in train_files],
        "validation": [],
        "test": [graph_entry(path) for path in test_files],
    }


def write_job_split(job: PhaseBJob) -> dict[str, Any]:
    payload = make_split_payload(job)
    write_json(job.split_path, payload)
    return payload


def optional_int_arg(command: list[str], flag: str, value: int | None) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def build_train_2stage_command(job: PhaseBJob, options: PhaseBOptions) -> list[str]:
    train_script = TRAIN_2STAGE_EARLYSTOP_SCRIPT if options.early_stop_enabled_2stage else TRAIN_2STAGE_SCRIPT
    command = [
        options.python_bin,
        str(train_script),
        "--split_path",
        str(job.split_path),
        "--validation_data_dir",
        str(job.validation_dir),
        "--out_dir",
        str(job.run_dir),
        "--train_size",
        str(job.train_sample_count),
        "--subset_seed",
        str(job.train_seed),
        "--theta_seed",
        str(options.theta_seed),
        "--n_epochs",
        str(options.epochs_2stage),
        "--lr",
        str(options.lr_2stage),
    ]
    optional_int_arg(command, "--validation_limit", options.validation_limit)
    if options.early_stop_enabled_2stage:
        command.extend(
            [
                "--metric_stride",
                str(options.metric_stride),
                "--early_stop_patience",
                str(options.early_stop_patience_2stage),
                "--early_stop_min_delta",
                str(options.early_stop_min_delta_2stage),
            ]
        )
    return command


def build_train_spoplus_command(job: PhaseBJob, options: PhaseBOptions) -> list[str]:
    command = [
        options.python_bin,
        str(TRAIN_SPOPLUS_SCRIPT),
        "--split_path",
        str(job.split_path),
        "--validation_data_dir",
        str(job.validation_dir),
        "--out_dir",
        str(job.run_dir),
        "--train_size",
        str(job.train_sample_count),
        "--subset_seed",
        str(job.train_seed),
        "--theta_seed",
        str(options.theta_seed),
        "--gurobi_seed",
        str(options.gurobi_seed),
        "--n_epochs",
        str(options.epochs_spoplus),
        "--lr",
        str(options.lr_spoplus),
        "--metric_stride",
        str(options.metric_stride),
    ]
    optional_int_arg(command, "--validation_limit", options.validation_limit)
    if options.early_stop_enabled_spoplus:
        command.extend(
            [
                "--early_stop_metric",
                "validation_spoplus_loss",
                "--early_stop_patience",
                str(options.early_stop_patience_spoplus),
                "--early_stop_min_delta",
                str(options.early_stop_min_delta_spoplus),
            ]
        )
    return command


def evaluation_weight_paths(job: PhaseBJob, options: PhaseBOptions) -> list[Path]:
    weights = [
        job.run_dir / "model_weights" / PRIMARY_WEIGHT_FILES[0],
        job.run_dir / "model_weights" / PRIMARY_WEIGHT_FILES[1],
    ]
    if options.include_decision_gap_checkpoint:
        weights.append(job.run_dir / "model_weights" / "spoplus_best_by_validation_decision_gap.npz")
    return weights


def build_evaluate_command(job: PhaseBJob, options: PhaseBOptions) -> list[str]:
    command = [
        options.python_bin,
        str(EVALUATE_MODELS_SCRIPT),
        "--split_path",
        str(job.split_path),
        "--out_dir",
        str(job.run_dir),
        "--gurobi_seed",
        str(options.gurobi_seed),
        "--bootstrap_samples",
        str(options.bootstrap_samples),
        "--bootstrap_seed",
        str(options.bootstrap_seed),
    ]
    optional_int_arg(command, "--test_limit", options.test_limit)
    command.append("--weights")
    command.extend(str(path) for path in evaluation_weight_paths(job, options))
    return command


def expected_artifacts(job: PhaseBJob, options: PhaseBOptions | None = None) -> list[Path]:
    artifacts = [job.run_dir / "model_weights" / filename for filename in GENERATED_WEIGHT_FILES]
    if options is not None and options.early_stop_enabled_2stage:
        artifacts.append(job.run_dir / "metrics" / "early_stopping_2stage.json")
    if options is not None and options.early_stop_enabled_spoplus:
        artifacts.append(job.run_dir / "metrics" / "early_stopping.json")
    artifacts.append(job.run_dir / "metrics" / "test_summary.csv")
    return artifacts


def is_job_complete(job: PhaseBJob, options: PhaseBOptions | None = None) -> bool:
    return all(path.exists() for path in expected_artifacts(job, options))


def run_environment(options: PhaseBOptions) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "MPLCONFIGDIR": os.environ.get("MPLCONFIGDIR", "/tmp/matplotlib"),
            "OMP_NUM_THREADS": str(options.thread_count),
            "MKL_NUM_THREADS": str(options.thread_count),
            "OPENBLAS_NUM_THREADS": str(options.thread_count),
            "NUMEXPR_NUM_THREADS": str(options.thread_count),
        }
    )
    return env


def run_logged_command(command: list[str], cwd: Path, env: dict[str, str], log_path: Path) -> tuple[int, float]:
    start = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        print("", file=log_file)
        print(f"=== {datetime.now().isoformat(timespec='seconds')} ===", file=log_file)
        print(f"$ {shlex.join(command)}", file=log_file, flush=True)
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        elapsed = time.monotonic() - start
        print(f"[exit {completed.returncode}] elapsed={elapsed:.2f}s", file=log_file, flush=True)
    return completed.returncode, elapsed


def read_test_summary_rows(run_dir: Path) -> list[dict[str, Any]]:
    path = Path(run_dir) / "metrics" / "test_summary.csv"
    if not path.exists():
        return []
    return read_csv_rows(path)


def float_or_blank(value: Any) -> float | str:
    if value in (None, ""):
        return ""
    return float(value)


def summarize_primary_metrics(rows: list[dict[str, Any]]) -> dict[str, float | str]:
    two_stage = None
    spoplus = None
    for row in rows:
        if row.get("method") == "2stage":
            two_stage = row
        if row.get("method") == "spoplus" and row.get("selection_metric") == "validation_spoplus_loss":
            spoplus = row
    if two_stage is None or spoplus is None:
        return {
            "test_mean_decision_gap_2stage": "",
            "test_mean_decision_gap_spoplus": "",
            "test_mean_normalized_gap_2stage": "",
            "test_mean_normalized_gap_spoplus": "",
            "spoplus_improvement_gap": "",
            "spoplus_improvement_normalized_gap": "",
        }
    gap_2stage = float(two_stage["test_mean_decision_gap"])
    gap_spoplus = float(spoplus["test_mean_decision_gap"])
    norm_2stage = float(two_stage["test_mean_normalized_gap"])
    norm_spoplus = float(spoplus["test_mean_normalized_gap"])
    return {
        "test_mean_decision_gap_2stage": gap_2stage,
        "test_mean_decision_gap_spoplus": gap_spoplus,
        "test_mean_normalized_gap_2stage": norm_2stage,
        "test_mean_normalized_gap_spoplus": norm_spoplus,
        "spoplus_improvement_gap": gap_2stage - gap_spoplus,
        "spoplus_improvement_normalized_gap": norm_2stage - norm_spoplus,
    }


def write_job_config(job: PhaseBJob, options: PhaseBOptions) -> None:
    write_json(
        job.run_dir / "phase_b_job_config.json",
        {
            "topology_id": job.topology_id,
            "train_seed": job.train_seed,
            "train_sample_count": job.train_sample_count,
            "validation_sample_count": job.validation_sample_count,
            "test_sample_count": job.test_sample_count,
            "topology_hash": job.topology_hash,
            "topology_bank_hash": job.topology_bank_hash,
            "feasible_set_hash": job.feasible_set_hash,
            "train_dir": str(job.train_dir),
            "validation_dir": str(job.validation_dir),
            "test_dir": str(job.test_dir),
            "split_path": str(job.split_path),
            "epochs_2stage": options.epochs_2stage,
            "epochs_spoplus": options.epochs_spoplus,
            "metric_stride": options.metric_stride,
            "theta_seed": options.theta_seed,
            "gurobi_seed": options.gurobi_seed,
            "validation_limit": options.validation_limit,
            "test_limit": options.test_limit,
            "early_stop_patience_2stage": options.early_stop_patience_2stage,
            "early_stop_min_delta_2stage": options.early_stop_min_delta_2stage,
            "early_stop_patience_spoplus": options.early_stop_patience_spoplus,
            "early_stop_min_delta_spoplus": options.early_stop_min_delta_spoplus,
        },
    )


def run_job(job: PhaseBJob, options: PhaseBOptions) -> JobResult:
    start = time.monotonic()
    if options.skip_completed and is_job_complete(job, options):
        metrics = summarize_primary_metrics(read_test_summary_rows(job.run_dir))
        return JobResult(
            job=job,
            status="skipped",
            return_code=0,
            elapsed_seconds=0.0,
            metrics=metrics,
            message="expected artifacts already exist",
        )

    timings = {
        "split_seconds": 0.0,
        "train_2stage_seconds": 0.0,
        "train_spoplus_seconds": 0.0,
        "posthoc_early_stop_seconds": 0.0,
        "evaluate_seconds": 0.0,
    }
    env = run_environment(options)
    job.run_dir.mkdir(parents=True, exist_ok=True)
    try:
        split_start = time.monotonic()
        write_job_split(job)
        timings["split_seconds"] = time.monotonic() - split_start
        write_job_config(job, options)

        train_stages = [
            ("train_2stage", build_train_2stage_command(job, options), "train_2stage_seconds"),
            ("train_spoplus", build_train_spoplus_command(job, options), "train_spoplus_seconds"),
        ]
        for stage_name, command, timing_key in train_stages:
            code, elapsed = run_logged_command(
                command,
                cwd=options.project_root,
                env=env,
                log_path=job.run_dir / "logs" / f"{stage_name}.log",
            )
            timings[timing_key] = elapsed
            if code != 0:
                return JobResult(
                    job=job,
                    status="failed",
                    return_code=code,
                    elapsed_seconds=time.monotonic() - start,
                    split_seconds=timings["split_seconds"],
                    train_2stage_seconds=timings["train_2stage_seconds"],
                    train_spoplus_seconds=timings["train_spoplus_seconds"],
                    posthoc_early_stop_seconds=timings["posthoc_early_stop_seconds"],
                    evaluate_seconds=timings["evaluate_seconds"],
                    message=f"{stage_name} failed",
                )

        code, elapsed = run_logged_command(
            build_evaluate_command(job, options),
            cwd=options.project_root,
            env=env,
            log_path=job.run_dir / "logs" / "evaluate.log",
        )
        timings["evaluate_seconds"] = elapsed
        if code != 0:
            return JobResult(
                job=job,
                status="failed",
                return_code=code,
                elapsed_seconds=time.monotonic() - start,
                split_seconds=timings["split_seconds"],
                train_2stage_seconds=timings["train_2stage_seconds"],
                train_spoplus_seconds=timings["train_spoplus_seconds"],
                posthoc_early_stop_seconds=timings["posthoc_early_stop_seconds"],
                evaluate_seconds=timings["evaluate_seconds"],
                message="evaluate failed",
            )

        missing = [str(path) for path in expected_artifacts(job, options) if not path.exists()]
        if missing:
            return JobResult(
                job=job,
                status="failed",
                return_code=1,
                elapsed_seconds=time.monotonic() - start,
                split_seconds=timings["split_seconds"],
                train_2stage_seconds=timings["train_2stage_seconds"],
                train_spoplus_seconds=timings["train_spoplus_seconds"],
                posthoc_early_stop_seconds=timings["posthoc_early_stop_seconds"],
                evaluate_seconds=timings["evaluate_seconds"],
                message="missing expected artifacts: " + "; ".join(missing),
            )
    except Exception as exc:  # noqa: BLE001 - keep failures in status CSV.
        return JobResult(
            job=job,
            status="failed",
            return_code=1,
            elapsed_seconds=time.monotonic() - start,
            split_seconds=timings["split_seconds"],
            train_2stage_seconds=timings["train_2stage_seconds"],
            train_spoplus_seconds=timings["train_spoplus_seconds"],
            posthoc_early_stop_seconds=timings["posthoc_early_stop_seconds"],
            evaluate_seconds=timings["evaluate_seconds"],
            message=f"{type(exc).__name__}: {exc}",
        )
    metrics = summarize_primary_metrics(read_test_summary_rows(job.run_dir))
    return JobResult(
        job=job,
        status="success",
        return_code=0,
        elapsed_seconds=time.monotonic() - start,
        split_seconds=timings["split_seconds"],
        train_2stage_seconds=timings["train_2stage_seconds"],
        train_spoplus_seconds=timings["train_spoplus_seconds"],
        posthoc_early_stop_seconds=timings["posthoc_early_stop_seconds"],
        evaluate_seconds=timings["evaluate_seconds"],
        metrics=metrics,
        message="completed",
    )


def result_row(result: JobResult, options: PhaseBOptions) -> dict[str, Any]:
    job = result.job
    row = {
        "job_index": job.index,
        "topology_id": job.topology_id,
        "train_seed": job.train_seed,
        "status": result.status,
        "return_code": result.return_code,
        "elapsed_seconds": f"{result.elapsed_seconds:.2f}",
        "split_seconds": f"{result.split_seconds:.2f}",
        "train_2stage_seconds": f"{result.train_2stage_seconds:.2f}",
        "train_spoplus_seconds": f"{result.train_spoplus_seconds:.2f}",
        "posthoc_early_stop_seconds": f"{result.posthoc_early_stop_seconds:.2f}",
        "evaluate_seconds": f"{result.evaluate_seconds:.2f}",
        "train_sample_count": job.train_sample_count,
        "validation_sample_count": job.validation_sample_count,
        "test_sample_count": job.test_sample_count,
        "validation_limit": "" if options.validation_limit is None else options.validation_limit,
        "test_limit": "" if options.test_limit is None else options.test_limit,
        "epochs_2stage": options.epochs_2stage,
        "epochs_spoplus": options.epochs_spoplus,
        "early_stop_patience_2stage": options.early_stop_patience_2stage,
        "early_stop_min_delta_2stage": options.early_stop_min_delta_2stage,
        "early_stop_patience_spoplus": options.early_stop_patience_spoplus,
        "early_stop_min_delta_spoplus": options.early_stop_min_delta_spoplus,
        "run_dir": str(job.run_dir),
        "message": result.message,
    }
    row.update(result.metrics)
    return row


def write_manifest(jobs: list[PhaseBJob], path: Path) -> None:
    rows = [
        {
            "job_index": job.index,
            "topology_id": job.topology_id,
            "train_seed": job.train_seed,
            "train_sample_count": job.train_sample_count,
            "validation_sample_count": job.validation_sample_count,
            "test_sample_count": job.test_sample_count,
            "topology_hash": job.topology_hash,
            "topology_bank_hash": job.topology_bank_hash,
            "feasible_set_hash": job.feasible_set_hash,
            "train_dir": str(job.train_dir),
            "validation_dir": str(job.validation_dir),
            "test_dir": str(job.test_dir),
            "split_path": str(job.split_path),
            "run_dir": str(job.run_dir),
        }
        for job in jobs
    ]
    write_csv(path, rows, MANIFEST_FIELDS)


def write_status(results: list[JobResult], path: Path, options: PhaseBOptions) -> None:
    rows = [result_row(result, options) for result in sorted(results, key=lambda item: item.job.index)]
    write_csv(path, rows, STATUS_FIELDS)


def parse_int_values(values: list[str] | None) -> list[int] | None:
    if not values:
        return None
    output: list[int] = []
    for value in values:
        output.extend(int(part) for part in str(value).split(",") if part)
    return output


def run_jobs(
    jobs: list[PhaseBJob],
    options: PhaseBOptions,
    *,
    workers: int,
    status_path: Path,
) -> list[JobResult]:
    results: list[JobResult] = []
    if workers <= 1:
        for job in jobs:
            result = run_job(job, options)
            results.append(result)
            write_status(results, status_path, options)
            print(
                f"[{len(results)}/{len(jobs)}] {job.topology_id} seed={job.train_seed} "
                f"{result.status} elapsed={result.elapsed_seconds:.1f}s",
                flush=True,
            )
        return results

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_job = {pool.submit(run_job, job, options): job for job in jobs}
        for future in concurrent.futures.as_completed(future_to_job):
            result = future.result()
            results.append(result)
            write_status(results, status_path, options)
            print(
                f"[{len(results)}/{len(jobs)}] {result.job.topology_id} "
                f"seed={result.job.train_seed} {result.status} "
                f"elapsed={result.elapsed_seconds:.1f}s",
                flush=True,
            )
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--status-path", type=Path, default=DEFAULT_STATUS_PATH)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--topology-id", action="append", default=None)
    parser.add_argument("--max-topologies", type=int, default=None)
    parser.add_argument("--train-seed", action="append", default=None)
    parser.add_argument("--max-train-seeds", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--thread-count", type=int, default=1)
    parser.add_argument("--python", dest="python_bin", default=os.environ.get("KEP_PYTHON"))
    parser.add_argument("--epochs-2stage", type=int, default=100)
    parser.add_argument("--epochs-spoplus", type=int, default=100)
    parser.add_argument("--lr-2stage", type=float, default=0.05)
    parser.add_argument("--lr-spoplus", type=float, default=0.1)
    parser.add_argument("--metric-stride", type=int, default=10)
    parser.add_argument("--theta-seed", type=int, default=42)
    parser.add_argument("--gurobi-seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--include-decision-gap-checkpoint", action="store_true")
    parser.add_argument(
        "--early-stop-patience-2stage",
        type=int,
        default=DEFAULT_EARLY_STOP_PATIENCE,
        help="Enable Step3 native 2stage early stopping after this many bad validation checks.",
    )
    parser.add_argument(
        "--early-stop-min-delta-2stage",
        type=float,
        default=DEFAULT_EARLY_STOP_MIN_DELTA,
        help="Minimum validation MSE improvement required to reset Step3 native 2stage early stopping.",
    )
    parser.add_argument(
        "--early-stop-patience-spoplus",
        type=int,
        default=DEFAULT_EARLY_STOP_PATIENCE,
        help="Enable native SPO+ early stopping after this many bad validation checks.",
    )
    parser.add_argument(
        "--early-stop-min-delta-spoplus",
        type=float,
        default=DEFAULT_EARLY_STOP_MIN_DELTA,
        help="Minimum validation SPO+ loss improvement required to reset native SPO+ early stopping.",
    )
    parser.add_argument("--no-skip-completed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def print_plan(jobs: list[PhaseBJob], args: argparse.Namespace, options: PhaseBOptions) -> None:
    topology_count = len({job.topology_id for job in jobs})
    print("Step3 Phase-B fixed-topology training")
    print(f"  topology count: {topology_count}")
    print(f"  jobs: {len(jobs)}")
    print(f"  methods: 2stage, SPO+")
    print(f"  train/validation/test: 40/10/1000 per topology")
    print(f"  epochs: 2stage={options.epochs_2stage}, spoplus={options.epochs_spoplus}")
    print(
        "  early stop: "
        f"2stage_patience={options.early_stop_patience_2stage}, "
        f"spoplus_patience={options.early_stop_patience_spoplus}, "
        f"min_delta_2stage={options.early_stop_min_delta_2stage}, "
        f"min_delta_spoplus={options.early_stop_min_delta_spoplus}"
    )
    print(f"  validation_limit: {options.validation_limit}")
    print(f"  test_limit: {options.test_limit}")
    print(f"  workers: {args.workers}")
    print(f"  output_dir: {args.output_dir}")
    print(f"  status_path: {args.status_path}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    options = PhaseBOptions(
        project_root=PROJECT_ROOT,
        python_bin=args.python_bin,
        epochs_2stage=args.epochs_2stage,
        epochs_spoplus=args.epochs_spoplus,
        lr_2stage=args.lr_2stage,
        lr_spoplus=args.lr_spoplus,
        metric_stride=args.metric_stride,
        theta_seed=args.theta_seed,
        gurobi_seed=args.gurobi_seed,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        validation_limit=args.validation_limit,
        test_limit=args.test_limit,
        thread_count=args.thread_count,
        include_decision_gap_checkpoint=args.include_decision_gap_checkpoint,
        skip_completed=not args.no_skip_completed,
        early_stop_patience_2stage=args.early_stop_patience_2stage,
        early_stop_min_delta_2stage=args.early_stop_min_delta_2stage,
        early_stop_patience_spoplus=args.early_stop_patience_spoplus,
        early_stop_min_delta_spoplus=args.early_stop_min_delta_spoplus,
    )
    jobs = discover_phase_b_jobs(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        split_dir=args.split_dir,
        topology_ids=args.topology_id,
        max_topologies=args.max_topologies,
        train_seeds=parse_int_values(args.train_seed),
        max_train_seeds=args.max_train_seeds,
    )
    if not jobs:
        raise ValueError("No Phase-B training jobs selected")
    print_plan(jobs, args, options)
    write_manifest(jobs, args.manifest_path)
    if args.dry_run:
        print("DRY RUN: no training commands executed.")
        for job in jobs[:10]:
            print(f"  {job.topology_id} train_seed={job.train_seed} run_dir={job.run_dir}")
        if len(jobs) > 10:
            print(f"  ... {len(jobs) - 10} more jobs")
        return 0
    results = run_jobs(jobs, options, workers=int(args.workers), status_path=args.status_path)
    failed = [result for result in results if result.status == "failed"]
    print(f"Completed jobs: {len(results)}; failed: {len(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
