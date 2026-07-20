#!/usr/bin/env python3
"""Preview or execute one dependency-resolved Experiment 06 job stage."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import tempfile
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class Job:
    job_id: str
    stage: str
    fold: int
    seed: int
    output_dir: Path
    log_path: Path
    dependency_path: Path | None
    threads: int
    command: list[str]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def parse_job(row: dict[str, str], output_root: Path) -> Job:
    if row.get("status") not in {"ready", "ready_after_classifier"}:
        raise ValueError(f"job is not launchable:{row.get('job_id')}:{row.get('status')}")
    stage = row["stage"]
    command = shlex.split(row["command_preview"])
    if "--execute" in command:
        raise ValueError(f"preview command already executes:{row.get('job_id')}")
    expected_script = "train_hurdle_classifier.py" if stage == "classifier" else "train_hurdle_regressor.py"
    if not any(Path(token).name == expected_script for token in command):
        raise ValueError(f"unexpected trainer:{row.get('job_id')}")
    output_dir = Path(row["output_dir"])
    try:
        output_dir.resolve().relative_to(output_root.resolve())
    except ValueError as exc:
        raise ValueError(f"output escapes root:{output_dir}") from exc
    dependency_path = Path(row["dependency_path"]) if row.get("dependency_path") else None
    command.append("--execute")
    return Job(
        job_id=row["job_id"],
        stage=stage,
        fold=int(row["fold"]),
        seed=int(row["seed"]),
        output_dir=output_dir,
        log_path=Path(row["log_path"]),
        dependency_path=dependency_path,
        threads=int(row["threads"]),
        command=command,
    )


def successful(job: Job) -> bool:
    result_path = job.output_dir / "run_result.json"
    predictions_path = job.output_dir / "test_predictions.csv"
    if not result_path.is_file() or not predictions_path.is_file():
        return False
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        predictions = read_csv(predictions_path)
        identity = int(result.get("fold", -1)) == job.fold and int(result.get("seed", -1)) == job.seed
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False
    expected_task = "zero_nonzero_classifier" if job.stage == "classifier" else "subset_regressor"
    return (
        result.get("status") == "success"
        and result.get("formal") is True
        and result.get("task") == expected_task
        and identity
        and len(predictions) == 200
        and len({row.get("topology_id") for row in predictions}) == 200
    )


def launch(job: Job) -> dict[str, Any]:
    if job.dependency_path is not None and not job.dependency_path.is_file():
        raise ValueError(f"missing dependency:{job.job_id}:{job.dependency_path}")
    job.output_dir.mkdir(parents=True, exist_ok=True)
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = job.log_path.open("a", encoding="utf-8")
    handle.write(f"[hurdle-launcher] start {time.strftime('%Y-%m-%dT%H:%M:%S%z')} {job.job_id}\n")
    handle.write(f"[hurdle-launcher] command {shlex.join(job.command)}\n")
    handle.flush()
    environment = os.environ.copy()
    threads = str(job.threads)
    environment.update(
        {
            "OMP_NUM_THREADS": threads,
            "MKL_NUM_THREADS": threads,
            "OPENBLAS_NUM_THREADS": threads,
            "NUMEXPR_NUM_THREADS": threads,
            "PYTHONUNBUFFERED": "1",
        }
    )
    process = subprocess.Popen(
        job.command,
        cwd=PROJECT_ROOT,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        env=environment,
    )
    return {"job": job, "process": process, "handle": handle, "started_at": time.time()}


def run(jobs: list[Job], output_root: Path, workers: int, monitor_interval: int) -> int:
    stage = jobs[0].stage
    status_path = output_root / f"{stage}_launcher_status.json"
    summary_path = output_root / f"{stage}_launcher_summary.json"
    monitor_path = output_root / f"{stage}_launcher_monitor.jsonl"
    skipped = [{"job_id": job.job_id, "status": "skipped_success"} for job in jobs if successful(job)]
    pending = [job for job in jobs if not successful(job)]
    active: dict[str, dict[str, Any]] = {}
    finished: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    started_at = time.time()
    last_monitor = 0.0

    def snapshot(status: str) -> dict[str, Any]:
        return {
            "status": status,
            "stage": stage,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "job_count": len(jobs),
            "workers": workers,
            "active_jobs": len(active),
            "finished_jobs": len(finished),
            "skipped_jobs": len(skipped),
            "failed_jobs": len(failed),
            "pending_jobs": len(pending),
            "elapsed_seconds": round(time.time() - started_at, 3),
            "loadavg": list(os.getloadavg()),
            "summary_path": str(summary_path),
        }

    atomic_write_json(status_path, snapshot("running"))
    try:
        while pending or active:
            while pending and len(active) < workers:
                job = pending.pop(0)
                active[job.job_id] = launch(job)
                print(f"[hurdle-launcher] launched {job.job_id}", flush=True)
            for job_id, item in list(active.items()):
                returncode = item["process"].poll()
                if returncode is None:
                    continue
                item["handle"].write(
                    f"[hurdle-launcher] end {time.strftime('%Y-%m-%dT%H:%M:%S%z')} returncode={returncode}\n"
                )
                item["handle"].close()
                row = {
                    "job_id": job_id,
                    "returncode": int(returncode),
                    "elapsed_seconds": round(time.time() - item["started_at"], 3),
                    "output_dir": str(item["job"].output_dir),
                    "log_path": str(item["job"].log_path),
                }
                if returncode == 0 and successful(item["job"]):
                    row["status"] = "success"
                    finished.append(row)
                else:
                    row["status"] = "failed"
                    failed.append(row)
                del active[job_id]
                print(f"[hurdle-launcher] finished {job_id} status={row['status']}", flush=True)
            if time.time() - last_monitor >= monitor_interval:
                payload = snapshot("running")
                atomic_write_json(status_path, payload)
                append_jsonl(monitor_path, payload)
                print(
                    f"[hurdle-launcher] monitor stage={stage} active={payload['active_jobs']} "
                    f"finished={payload['finished_jobs']} failed={payload['failed_jobs']} pending={payload['pending_jobs']}",
                    flush=True,
                )
                last_monitor = time.time()
            time.sleep(1)
    except KeyboardInterrupt:
        for item in active.values():
            item["process"].terminate()
        for item in active.values():
            try:
                item["process"].wait(timeout=10)
            except subprocess.TimeoutExpired:
                item["process"].kill()
                item["process"].wait()
            item["handle"].close()
        atomic_write_json(status_path, snapshot("interrupted"))
        raise
    final_status = "success" if not failed and len(finished) + len(skipped) == len(jobs) else "failed"
    summary = {**snapshot(final_status), "finished": finished, "skipped": skipped, "failed": failed}
    atomic_write_json(summary_path, summary)
    atomic_write_json(status_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if final_status == "success" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-job-count", type=int, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--monitor-interval", type=int, default=15)
    parser.add_argument("--require-hostname")
    parser.add_argument("--require-dependencies", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    rows = read_csv(args.jobs_csv)
    if len(rows) != args.expected_job_count:
        raise ValueError(f"expected {args.expected_job_count} jobs, observed {len(rows)}")
    jobs = [parse_job(row, args.output_root) for row in rows]
    if len({job.job_id for job in jobs}) != len(jobs):
        raise ValueError("duplicate job ids")
    if len({job.stage for job in jobs}) != 1:
        raise ValueError("one launcher invocation must contain one stage")
    hostname = socket.gethostname()
    if args.require_hostname and args.require_hostname not in hostname:
        raise ValueError(f"hostname safety check failed:{hostname}")
    dependency_count = sum(job.dependency_path is not None for job in jobs)
    available_dependencies = sum(job.dependency_path is not None and job.dependency_path.is_file() for job in jobs)
    if (args.execute or args.require_dependencies) and available_dependencies != dependency_count:
        raise ValueError(f"dependency check failed:{available_dependencies}/{dependency_count}")
    preview = {
        "passed": True,
        "execute": args.execute,
        "hostname": hostname,
        "stage": jobs[0].stage,
        "job_count": len(jobs),
        "workers": args.workers,
        "threads_per_job": sorted({job.threads for job in jobs}),
        "maximum_requested_threads": args.workers * max(job.threads for job in jobs),
        "dependency_count": dependency_count,
        "available_dependencies": available_dependencies,
        "already_successful_jobs": sum(successful(job) for job in jobs),
    }
    print(json.dumps(preview, indent=2, sort_keys=True))
    if not args.execute:
        return 0
    return run(jobs, args.output_root, args.workers, args.monitor_interval)


if __name__ == "__main__":
    raise SystemExit(main())
