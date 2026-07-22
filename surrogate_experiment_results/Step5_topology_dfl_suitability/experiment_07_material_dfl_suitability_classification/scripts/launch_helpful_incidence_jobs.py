#!/usr/bin/env python3
"""Safely preview or execute planned Experiment 07 incidence jobs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import time
from typing import Any

import material_common as common


PROJECT_ROOT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class Job:
    job_id: str
    fold: int
    seed: int
    output_dir: Path
    log_path: Path
    threads: int
    command: list[str]


def parse_job(row: dict[str, str], output_root: Path) -> Job:
    if row.get("status") != "ready":
        raise ValueError(f"job is not ready:{row.get('job_id')}:{row.get('status')}")
    command = shlex.split(row["command_preview"])
    if "--execute" in command:
        raise ValueError(f"planned command already executes:{row['job_id']}")
    if not any(Path(token).name == "train_helpful_incidence_classifier.py" for token in command):
        raise ValueError(f"unexpected trainer:{row['job_id']}")
    output_dir = Path(row["output_dir"])
    try:
        output_dir.resolve().relative_to(output_root.resolve())
    except ValueError as exc:
        raise ValueError(f"output escapes root:{output_dir}") from exc
    return Job(
        job_id=row["job_id"],
        fold=int(row["fold"]),
        seed=int(row["seed"]),
        output_dir=output_dir,
        log_path=Path(row["log_path"]),
        threads=int(row["threads"]),
        command=[*command, "--execute"],
    )


def successful(job: Job) -> bool:
    result_path = job.output_dir / "run_result.json"
    predictions_path = job.output_dir / "test_predictions.csv"
    if not result_path.is_file() or not predictions_path.is_file():
        return False
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        predictions = common.read_csv(predictions_path)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False
    return (
        result.get("status") == "success"
        and result.get("formal") is True
        and result.get("task") == "material_helpful_vs_non_helpful"
        and int(result.get("fold", -1)) == job.fold
        and int(result.get("seed", -1)) == job.seed
        and len(predictions) == 200
        and len({row.get("topology_id") for row in predictions}) == 200
    )


def launch(job: Job) -> dict[str, Any]:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = job.log_path.open("a", encoding="utf-8")
    handle.write(
        f"[exp07-launcher] start {time.strftime('%Y-%m-%dT%H:%M:%S%z')} {job.job_id}\n"
    )
    handle.write(f"[exp07-launcher] command {shlex.join(job.command)}\n")
    handle.flush()
    environment = os.environ.copy()
    threads = str(job.threads)
    environment.update({
        "OMP_NUM_THREADS": threads,
        "MKL_NUM_THREADS": threads,
        "OPENBLAS_NUM_THREADS": threads,
        "NUMEXPR_NUM_THREADS": threads,
        "PYTHONUNBUFFERED": "1",
    })
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
    status_path = output_root / "launcher_status.json"
    summary_path = output_root / "launcher_summary.json"
    skipped = [job for job in jobs if successful(job)]
    pending = [job for job in jobs if not successful(job)]
    active: dict[str, dict[str, Any]] = {}
    finished: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    started_at = time.time()
    last_monitor = 0.0

    def snapshot(status: str) -> dict[str, Any]:
        return {
            "status": status,
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

    common.atomic_write_json(status_path, snapshot("running"))
    try:
        while pending or active:
            while pending and len(active) < workers:
                job = pending.pop(0)
                active[job.job_id] = launch(job)
                print(f"[exp07-launcher] launched {job.job_id}", flush=True)
            for job_id, item in list(active.items()):
                returncode = item["process"].poll()
                if returncode is None:
                    continue
                item["handle"].write(
                    f"[exp07-launcher] end {time.strftime('%Y-%m-%dT%H:%M:%S%z')} returncode={returncode}\n"
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
                print(f"[exp07-launcher] finished {job_id} status={row['status']}", flush=True)
            if time.time() - last_monitor >= monitor_interval:
                common.atomic_write_json(status_path, snapshot("running"))
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
        common.atomic_write_json(status_path, snapshot("interrupted"))
        raise
    final_status = (
        "success"
        if not failed and len(finished) + len(skipped) == len(jobs)
        else "failed"
    )
    summary = {
        **snapshot(final_status),
        "finished": finished,
        "skipped": [job.job_id for job in skipped],
        "failed": failed,
    }
    common.atomic_write_json(summary_path, summary)
    common.atomic_write_json(status_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if final_status == "success" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-job-count", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--monitor-interval", type=int, default=15)
    parser.add_argument("--require-hostname")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    rows = common.read_csv(args.jobs_csv)
    if len(rows) != args.expected_job_count:
        raise ValueError(
            f"expected {args.expected_job_count} jobs, observed {len(rows)}"
        )
    jobs = [parse_job(row, args.output_root) for row in rows]
    if len({job.job_id for job in jobs}) != len(jobs):
        raise ValueError("duplicate job ids")
    hostname = socket.gethostname()
    if args.require_hostname and args.require_hostname not in hostname:
        raise ValueError(f"hostname safety check failed:{hostname}")
    preview = {
        "passed": True,
        "execute": args.execute,
        "hostname": hostname,
        "job_count": len(jobs),
        "workers": args.workers,
        "threads_per_job": sorted({job.threads for job in jobs}),
        "maximum_requested_threads": args.workers * max(job.threads for job in jobs),
        "already_successful_jobs": sum(successful(job) for job in jobs),
    }
    print(json.dumps(preview, indent=2, sort_keys=True))
    if not args.execute:
        return 0
    return run(jobs, args.output_root, args.workers, args.monitor_interval)


if __name__ == "__main__":
    raise SystemExit(main())
