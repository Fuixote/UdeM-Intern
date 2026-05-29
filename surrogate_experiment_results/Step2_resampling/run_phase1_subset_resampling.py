#!/usr/bin/env python3
"""Run Phase 1 fixed-pool subset resampling for Step2b/Step2c.

Phase 1 keeps the processed graph pools fixed and varies only subset_seed.
Each wrapper job delegates training to Step1c/run_step1c.sh, which trains
2stage MSE and DFL SPO+, then this driver evaluates the resulting checkpoints
on the matching unseen10000 dataset.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STEP2_DIR = Path("surrogate_experiment_results/Step2_resampling")
STEP1C_DIR = Path("surrogate_experiment_results/Step1c")
STEP1C_RUN_SCRIPT = STEP1C_DIR / "run_step1c.sh"
STEP1C_SPLIT_SCRIPT = STEP1C_DIR / "split_dataset.py"
STEP1C_UNSEEN_SCRIPT = STEP1C_DIR / "evaluate_unseen_run.py"

DEFAULT_LABEL_SEED = 20260523
DEFAULT_SPLIT_SEED = 42
DEFAULT_THETA_SEED = 42
DEFAULT_GUROBI_SEED = 42
DEFAULT_DEGREES = (1, 2, 4, 8)
DEFAULT_BLOCKS = ("step2b", "step2c")

EXPECTED_WEIGHT_FILES = [
    "2stage_best_by_validation_mse_loss.npz",
    "spoplus_best_by_validation_decision_gap.npz",
    "spoplus_best_by_validation_spoplus_loss.npz",
]


@dataclass(frozen=True)
class Regime:
    name: str
    block: str
    degree: int
    dataset_base: str
    main_dir: Path
    val_dir: Path
    unseen_dir: Path


@dataclass(frozen=True)
class Phase1Job:
    index: int
    regime: Regime
    subset_seed: int
    train_size: int
    split_path: Path
    run_dir: Path


@dataclass(frozen=True)
class Phase1Options:
    project_root: Path
    workers: int
    thread_count: int
    dry_run: bool
    skip_completed: bool
    status_path: Path
    manifest_path: Path
    log_root: Path
    skip_unseen_eval: bool = False
    python_bin: str | None = None
    split_seed: int = DEFAULT_SPLIT_SEED
    theta_seed: int = DEFAULT_THETA_SEED
    gurobi_seed: int = DEFAULT_GUROBI_SEED
    train_pool_size: int = 1200
    validation_size: int = 400
    test_size: int = 400
    epochs_2stage: int = 500
    epochs_spoplus: int = 500
    metric_stride: int = 10
    bootstrap_samples: int = 1000
    bootstrap_seed: int = 42
    train_graph_limit: int | None = None
    validation_limit: int | None = None
    test_limit: int | None = None
    unseen_graph_limit: int | None = None
    dry_run_limit: int = 20


@dataclass(frozen=True)
class JobResult:
    job: Phase1Job
    status: str
    return_code: int
    elapsed_seconds: float
    log_path: Path
    message: str


def dataset_base_for(block: str, degree: int) -> str:
    if block == "step2b":
        return f"step2b_poly_d{degree}"
    if block == "step2c":
        return f"step2c_poly_d{degree}_mult_eps050"
    raise ValueError(f"Unsupported block: {block}")


def build_default_regimes(
    dataset_root: Path = Path("dataset/processed"),
    blocks: tuple[str, ...] | list[str] = DEFAULT_BLOCKS,
    degrees: tuple[int, ...] | list[int] = DEFAULT_DEGREES,
    label_seed: int = DEFAULT_LABEL_SEED,
) -> list[Regime]:
    regimes: list[Regime] = []
    for block in blocks:
        for degree in degrees:
            base = dataset_base_for(block, degree)
            regimes.append(
                Regime(
                    name=base,
                    block=block,
                    degree=degree,
                    dataset_base=base,
                    main_dir=dataset_root / f"{base}_main2000_seed{label_seed}",
                    val_dir=dataset_root / f"{base}_val2000_seed{label_seed}",
                    unseen_dir=dataset_root / f"{base}_unseen10000_seed{label_seed}",
                )
            )
    return regimes


def build_jobs(
    regimes: list[Regime],
    subset_seeds,
    train_size: int,
    output_root: Path,
    split_root: Path,
    split_seed: int,
) -> list[Phase1Job]:
    jobs: list[Phase1Job] = []
    for regime in regimes:
        split_path = split_root / regime.name / f"master_split_seed={split_seed}.json"
        for subset_seed in subset_seeds:
            jobs.append(
                Phase1Job(
                    index=len(jobs) + 1,
                    regime=regime,
                    subset_seed=int(subset_seed),
                    train_size=train_size,
                    split_path=split_path,
                    run_dir=output_root / regime.name / f"subset_seed={int(subset_seed)}",
                )
            )
    return jobs


def path_arg(path: Path) -> str:
    return path.as_posix()


def optional_int_env(name: str, value: int | None, env: dict[str, str]) -> None:
    if value is not None:
        env[name] = str(value)


def build_step1c_environment(job: Phase1Job, options: Phase1Options) -> dict[str, str]:
    env = {
        "PYTHONUNBUFFERED": "1",
        "MPLCONFIGDIR": os.environ.get("MPLCONFIGDIR", "/tmp/matplotlib"),
        "OMP_NUM_THREADS": str(options.thread_count),
        "MKL_NUM_THREADS": str(options.thread_count),
        "OPENBLAS_NUM_THREADS": str(options.thread_count),
        "NUMEXPR_NUM_THREADS": str(options.thread_count),
        "STEP1C_DATA_DIR": path_arg(job.regime.main_dir),
        "STEP1C_VALIDATION_DATA_DIR": path_arg(job.regime.val_dir),
        "STEP1C_SPLIT_PATH": path_arg(job.split_path),
        "STEP1C_OUTPUT_DIR": path_arg(job.run_dir),
        "STEP1C_SPLIT_SEED": str(options.split_seed),
        "STEP1C_SUBSET_SEED": str(job.subset_seed),
        "STEP1C_THETA_SEED": str(options.theta_seed),
        "STEP1C_GUROBI_SEED": str(options.gurobi_seed),
        "STEP1C_TRAIN_POOL_SIZE": str(options.train_pool_size),
        "STEP1C_VAL_SIZE": str(options.validation_size),
        "STEP1C_TEST_SIZE": str(options.test_size),
        "STEP1C_TRAIN_SIZE": str(job.train_size),
        "STEP1C_2STAGE_N_EPOCHS": str(options.epochs_2stage),
        "STEP1C_SPOPLUS_N_EPOCHS": str(options.epochs_spoplus),
        "STEP1C_METRIC_STRIDE": str(options.metric_stride),
        "STEP1C_BOOTSTRAP_SAMPLES": str(options.bootstrap_samples),
        "STEP1C_BOOTSTRAP_SEED": str(options.bootstrap_seed),
    }
    if options.python_bin:
        env["KEP_PYTHON"] = options.python_bin
    optional_int_env("STEP1C_TRAIN_GRAPH_LIMIT", options.train_graph_limit, env)
    optional_int_env("STEP1C_VALIDATION_LIMIT", options.validation_limit, env)
    optional_int_env("STEP1C_TEST_LIMIT", options.test_limit, env)
    return env


def python_bin(options: Phase1Options) -> str:
    return options.python_bin or os.environ.get("KEP_PYTHON") or sys.executable


def build_step1c_command() -> list[str]:
    return ["bash", path_arg(STEP1C_RUN_SCRIPT)]


def build_prepare_split_command(job: Phase1Job, options: Phase1Options) -> list[str]:
    return [
        python_bin(options),
        path_arg(STEP1C_SPLIT_SCRIPT),
        "--data_dir",
        path_arg(job.regime.main_dir),
        "--split_path",
        path_arg(job.split_path),
        "--train_pool_size",
        str(options.train_pool_size),
        "--val_size",
        str(options.validation_size),
        "--test_size",
        str(options.test_size),
        "--seed",
        str(options.split_seed),
        "--reuse_if_exists",
    ]


def build_unseen_command(job: Phase1Job, options: Phase1Options) -> list[str]:
    command = [
        python_bin(options),
        path_arg(STEP1C_UNSEEN_SCRIPT),
        "--run_dir",
        path_arg(job.run_dir),
        "--dataset_dir",
        path_arg(job.regime.unseen_dir),
        "--output_stem",
        "unseen10000",
        "--gurobi_seed",
        str(options.gurobi_seed),
        "--bootstrap_samples",
        str(options.bootstrap_samples),
        "--bootstrap_seed",
        str(options.bootstrap_seed),
    ]
    if options.unseen_graph_limit is not None:
        command.extend(["--graph_limit", str(options.unseen_graph_limit)])
    return command


def expected_artifacts(job: Phase1Job, include_unseen: bool = True) -> list[Path]:
    weights_dir = job.run_dir / "model_weights"
    metrics_dir = job.run_dir / "metrics"
    artifacts = [weights_dir / filename for filename in EXPECTED_WEIGHT_FILES]
    artifacts.append(metrics_dir / "test_summary.csv")
    if include_unseen:
        artifacts.append(metrics_dir / "unseen10000_summary.csv")
    return artifacts


def is_job_complete(job: Phase1Job, include_unseen: bool = True) -> bool:
    return all(path.exists() for path in expected_artifacts(job, include_unseen))


def log_path_for(job: Phase1Job, options: Phase1Options) -> Path:
    return options.log_root / job.regime.name / f"subset_seed={job.subset_seed}.log"


def resolved(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def missing_required_paths(jobs: list[Phase1Job], options: Phase1Options) -> list[Path]:
    required = {STEP1C_RUN_SCRIPT, STEP1C_SPLIT_SCRIPT, STEP1C_UNSEEN_SCRIPT}
    for job in jobs:
        required.add(job.regime.main_dir)
        required.add(job.regime.val_dir)
        required.add(job.regime.unseen_dir)
    missing = [path for path in sorted(required, key=path_arg) if not resolved(options.project_root, path).exists()]
    return missing


def unique_split_jobs(jobs: list[Phase1Job]) -> list[Phase1Job]:
    seen: set[Path] = set()
    selected: list[Phase1Job] = []
    for job in jobs:
        if job.split_path in seen:
            continue
        seen.add(job.split_path)
        selected.append(job)
    return selected


def run_logged_command(
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    log_file,
) -> int:
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
    print(f"[exit {completed.returncode}] {shlex.join(command)}", file=log_file, flush=True)
    return completed.returncode


def run_job(job: Phase1Job, options: Phase1Options) -> JobResult:
    start = time.monotonic()
    log_path = log_path_for(job, options)

    include_unseen = not options.skip_unseen_eval
    if options.skip_completed and is_job_complete(job, include_unseen=include_unseen):
        return JobResult(
            job=job,
            status="skipped",
            return_code=0,
            elapsed_seconds=0.0,
            log_path=log_path,
            message="all expected artifacts already exist",
        )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(build_step1c_environment(job, options))

    try:
        with log_path.open("a", encoding="utf-8") as log_file:
            print("", file=log_file)
            print(f"=== Phase1 job started {datetime.now().isoformat(timespec='seconds')} ===", file=log_file)
            print(f"job={job.regime.name} subset_seed={job.subset_seed}", file=log_file)
            print(f"run_dir={job.run_dir}", file=log_file)

            code = run_logged_command(build_step1c_command(), options.project_root, env, log_file)
            if code != 0:
                return JobResult(
                    job=job,
                    status="failed",
                    return_code=code,
                    elapsed_seconds=time.monotonic() - start,
                    log_path=log_path,
                    message="Step1c training/evaluation failed",
                )

            if options.skip_unseen_eval:
                print("[skip] unseen10000 evaluation disabled for this run", file=log_file, flush=True)
            else:
                code = run_logged_command(build_unseen_command(job, options), options.project_root, env, log_file)
                if code != 0:
                    return JobResult(
                        job=job,
                        status="failed",
                        return_code=code,
                        elapsed_seconds=time.monotonic() - start,
                        log_path=log_path,
                        message="unseen10000 evaluation failed",
                    )

            if not is_job_complete(job, include_unseen=include_unseen):
                missing = [
                    path_arg(path)
                    for path in expected_artifacts(job, include_unseen=include_unseen)
                    if not path.exists()
                ]
                return JobResult(
                    job=job,
                    status="failed",
                    return_code=1,
                    elapsed_seconds=time.monotonic() - start,
                    log_path=log_path,
                    message="missing expected artifacts: " + "; ".join(missing),
                )
    except Exception as exc:  # noqa: BLE001 - keep worker failures in status CSV.
        return JobResult(
            job=job,
            status="failed",
            return_code=1,
            elapsed_seconds=time.monotonic() - start,
            log_path=log_path,
            message=f"{type(exc).__name__}: {exc}",
        )

    return JobResult(
        job=job,
        status="success",
        return_code=0,
        elapsed_seconds=time.monotonic() - start,
        log_path=log_path,
        message="completed",
    )


def write_manifest(jobs: list[Phase1Job], path: Path) -> None:
    fields = [
        "job_index",
        "regime",
        "block",
        "degree",
        "subset_seed",
        "train_size",
        "main_dir",
        "validation_dir",
        "unseen_dir",
        "split_path",
        "run_dir",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for job in jobs:
            writer.writerow(
                {
                    "job_index": job.index,
                    "regime": job.regime.name,
                    "block": job.regime.block,
                    "degree": job.regime.degree,
                    "subset_seed": job.subset_seed,
                    "train_size": job.train_size,
                    "main_dir": path_arg(job.regime.main_dir),
                    "validation_dir": path_arg(job.regime.val_dir),
                    "unseen_dir": path_arg(job.regime.unseen_dir),
                    "split_path": path_arg(job.split_path),
                    "run_dir": path_arg(job.run_dir),
                }
            )


def write_status(results: list[JobResult], path: Path) -> None:
    fields = [
        "job_index",
        "regime",
        "degree",
        "subset_seed",
        "train_size",
        "status",
        "return_code",
        "elapsed_seconds",
        "log_path",
        "run_dir",
        "message",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in sorted(results, key=lambda item: item.job.index):
            job = result.job
            writer.writerow(
                {
                    "job_index": job.index,
                    "regime": job.regime.name,
                    "degree": job.regime.degree,
                    "subset_seed": job.subset_seed,
                    "train_size": job.train_size,
                    "status": result.status,
                    "return_code": result.return_code,
                    "elapsed_seconds": f"{result.elapsed_seconds:.1f}",
                    "log_path": path_arg(result.log_path),
                    "run_dir": path_arg(job.run_dir),
                    "message": result.message,
                }
            )


def flatten_values(values: list[str] | None) -> list[str]:
    if not values:
        return []
    flattened: list[str] = []
    for value in values:
        flattened.extend(part for part in value.split(",") if part)
    return flattened


def parse_int_values(values: list[str] | None) -> list[int]:
    return [int(value) for value in flatten_values(values)]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run Step2_resampling Phase 1 fixed-pool subset resampling."
    )
    parser.add_argument("--project_root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--dataset_root", type=Path, default=Path("dataset/processed"))
    parser.add_argument("--output_root", type=Path, default=STEP2_DIR / "phase1_runs")
    parser.add_argument("--split_root", type=Path, default=STEP2_DIR / "splits")
    parser.add_argument("--log_root", type=Path, default=STEP2_DIR / "logs/phase1")
    parser.add_argument(
        "--manifest_path",
        type=Path,
        default=STEP2_DIR / "results/phase1_job_manifest.csv",
    )
    parser.add_argument(
        "--status_path",
        type=Path,
        default=STEP2_DIR / "results/phase1_job_status.csv",
    )
    parser.add_argument("--blocks", nargs="+", default=list(DEFAULT_BLOCKS))
    parser.add_argument("--degrees", nargs="+", default=[str(degree) for degree in DEFAULT_DEGREES])
    parser.add_argument(
        "--regimes",
        nargs="+",
        help="Optional explicit regime names, e.g. step2b_poly_d8 step2c_poly_d8_mult_eps050.",
    )
    parser.add_argument("--label_seed", type=int, default=DEFAULT_LABEL_SEED)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--seed_count", type=int, default=50)
    parser.add_argument("--subset_seeds", nargs="+")
    parser.add_argument("--train_size", type=int, default=50)
    parser.add_argument("--split_seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--theta_seed", type=int, default=DEFAULT_THETA_SEED)
    parser.add_argument("--gurobi_seed", type=int, default=DEFAULT_GUROBI_SEED)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--thread_count", type=int, default=1)
    parser.add_argument("--python", dest="python_bin", default=os.environ.get("KEP_PYTHON"))
    parser.add_argument("--epochs_2stage", type=int, default=500)
    parser.add_argument("--epochs_spoplus", type=int, default=500)
    parser.add_argument("--metric_stride", type=int, default=10)
    parser.add_argument("--bootstrap_samples", type=int, default=1000)
    parser.add_argument("--bootstrap_seed", type=int, default=42)
    parser.add_argument("--train_graph_limit", type=int)
    parser.add_argument("--validation_limit", type=int)
    parser.add_argument("--test_limit", type=int)
    parser.add_argument("--unseen_graph_limit", type=int)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--skip_unseen_eval",
        action="store_true",
        help=(
            "Train and run Step1c heldout400 evaluation only. Defer matching "
            "unseen10000 final evaluation to a later batch step."
        ),
    )
    parser.add_argument("--dry_run_limit", type=int, default=20)
    parser.add_argument("--no_skip_completed", action="store_true")
    parser.add_argument(
        "--no_prepare_splits",
        action="store_true",
        help="Skip sequential master-split preparation before launching workers.",
    )
    return parser.parse_args(argv)


def print_plan(jobs: list[Phase1Job], options: Phase1Options) -> None:
    regimes = sorted({job.regime.name for job in jobs})
    print("Phase 1 fixed graph pool + subset resampling")
    print(f"  regimes: {len(regimes)}")
    for regime in regimes:
        count = sum(1 for job in jobs if job.regime.name == regime)
        print(f"    {regime}: {count} subset jobs")
    print(f"  wrapper jobs: {len(jobs)}")
    print(f"  training runs represented: {len(jobs) * 2}")
    print(f"  unseen10000 evaluation: {'skipped' if options.skip_unseen_eval else 'enabled'}")
    print(f"  workers: {options.workers}")
    print(f"  per-worker BLAS threads: {options.thread_count}")
    print(f"  status: {options.status_path}")
    print(f"  manifest: {options.manifest_path}")


def print_dry_run(jobs: list[Phase1Job], options: Phase1Options) -> None:
    print("DRY RUN: no training or evaluation commands will be executed.")
    for job in jobs[: options.dry_run_limit]:
        env = build_step1c_environment(job, options)
        print()
        print(f"[{job.index}/{len(jobs)}] {job.regime.name} subset_seed={job.subset_seed}")
        print(f"  run_dir={job.run_dir}")
        print(f"  STEP1C_DATA_DIR={env['STEP1C_DATA_DIR']}")
        print(f"  STEP1C_VALIDATION_DATA_DIR={env['STEP1C_VALIDATION_DATA_DIR']}")
        print(f"  train: {shlex.join(build_step1c_command())}")
        if options.skip_unseen_eval:
            print("  unseen: skipped")
        else:
            print(f"  unseen: {shlex.join(build_unseen_command(job, options))}")
    if len(jobs) > options.dry_run_limit:
        print()
        print(f"... omitted {len(jobs) - options.dry_run_limit} dry-run jobs")


def prepare_splits(jobs: list[Phase1Job], options: Phase1Options) -> int:
    split_jobs = unique_split_jobs(jobs)
    log_path = options.log_root / "_prepare_splits.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": str(options.thread_count),
            "MKL_NUM_THREADS": str(options.thread_count),
            "OPENBLAS_NUM_THREADS": str(options.thread_count),
            "NUMEXPR_NUM_THREADS": str(options.thread_count),
        }
    )
    print(f"Preparing {len(split_jobs)} shared master splits before worker launch.")
    with log_path.open("a", encoding="utf-8") as log_file:
        print("", file=log_file)
        print(f"=== split preparation {datetime.now().isoformat(timespec='seconds')} ===", file=log_file)
        for job in split_jobs:
            code = run_logged_command(build_prepare_split_command(job, options), options.project_root, env, log_file)
            if code != 0:
                print(f"Split preparation failed for {job.regime.name}; see {log_path}")
                return code
    return 0


def run_jobs(jobs: list[Phase1Job], options: Phase1Options) -> list[JobResult]:
    results: list[JobResult] = []
    completed = 0

    if options.workers == 1:
        for job in jobs:
            result = run_job(job, options)
            results.append(result)
            completed += 1
            write_status(results, options.status_path)
            print_progress(result, completed, len(jobs))
        return results

    with concurrent.futures.ThreadPoolExecutor(max_workers=options.workers) as executor:
        futures = {executor.submit(run_job, job, options): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            write_status(results, options.status_path)
            print_progress(result, completed, len(jobs))
    return results


def print_progress(result: JobResult, completed: int, total: int) -> None:
    job = result.job
    print(
        f"[{completed}/{total}] {result.status.upper()} "
        f"{job.regime.name} subset_seed={job.subset_seed} "
        f"elapsed={result.elapsed_seconds:.1f}s log={result.log_path}"
    )
    if result.status == "failed":
        print(f"  {result.message}")


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.thread_count < 1:
        raise SystemExit("--thread_count must be >= 1")
    if args.seed_count < 1:
        raise SystemExit("--seed_count must be >= 1")

    blocks = tuple(flatten_values(args.blocks))
    degrees = tuple(parse_int_values(args.degrees))
    regimes = build_default_regimes(
        dataset_root=args.dataset_root,
        blocks=blocks,
        degrees=degrees,
        label_seed=args.label_seed,
    )
    requested_regimes = set(flatten_values(args.regimes))
    if requested_regimes:
        known = {regime.name for regime in regimes}
        unknown = sorted(requested_regimes - known)
        if unknown:
            raise SystemExit(f"Unknown regimes: {', '.join(unknown)}")
        regimes = [regime for regime in regimes if regime.name in requested_regimes]
    if not regimes:
        raise SystemExit("No regimes selected")

    if args.subset_seeds:
        subset_seeds = parse_int_values(args.subset_seeds)
    else:
        subset_seeds = list(range(args.seed_start, args.seed_start + args.seed_count))

    jobs = build_jobs(
        regimes,
        subset_seeds=subset_seeds,
        train_size=args.train_size,
        output_root=args.output_root,
        split_root=args.split_root,
        split_seed=args.split_seed,
    )

    options = Phase1Options(
        project_root=args.project_root,
        workers=args.workers,
        thread_count=args.thread_count,
        dry_run=args.dry_run,
        skip_unseen_eval=args.skip_unseen_eval,
        skip_completed=not args.no_skip_completed,
        status_path=args.status_path,
        manifest_path=args.manifest_path,
        log_root=args.log_root,
        python_bin=args.python_bin,
        split_seed=args.split_seed,
        theta_seed=args.theta_seed,
        gurobi_seed=args.gurobi_seed,
        epochs_2stage=args.epochs_2stage,
        epochs_spoplus=args.epochs_spoplus,
        metric_stride=args.metric_stride,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        train_graph_limit=args.train_graph_limit,
        validation_limit=args.validation_limit,
        test_limit=args.test_limit,
        unseen_graph_limit=args.unseen_graph_limit,
        dry_run_limit=args.dry_run_limit,
    )

    print_plan(jobs, options)
    missing = missing_required_paths(jobs, options)
    if missing:
        print()
        print("Missing required paths:")
        for path in missing:
            print(f"  {path}")
        if not options.dry_run:
            return 2

    if options.dry_run:
        print_dry_run(jobs, options)
        return 0

    write_manifest(jobs, options.manifest_path)

    if not args.no_prepare_splits:
        code = prepare_splits(jobs, options)
        if code != 0:
            return code

    results = run_jobs(jobs, options)
    failed = [result for result in results if result.status == "failed"]
    skipped = [result for result in results if result.status == "skipped"]
    succeeded = [result for result in results if result.status == "success"]
    print()
    print(
        f"Phase1 finished: success={len(succeeded)} "
        f"skipped={len(skipped)} failed={len(failed)}"
    )
    print(f"Status CSV: {options.status_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
