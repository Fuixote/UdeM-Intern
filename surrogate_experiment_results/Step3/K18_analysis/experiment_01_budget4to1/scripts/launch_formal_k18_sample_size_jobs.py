#!/usr/bin/env python3
"""Launch K18-E1 formal sample-size jobs with bounded normal/long queues."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_JOBS_CSV = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "Step3"
    / "K18_analysis"
    / "experiment_01_budget4to1"
    / "results"
    / "materialized"
    / "sample_size_jobs.csv"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "Step3"
    / "K18_analysis"
    / "experiment_01_budget4to1"
    / "results"
    / "formal_270_full_epoch_20260626"
)
SUCCESS_STATUSES = {
    "status": "success",
    "2stage status": "success",
    "SPO+ status": "success",
    "evaluation status": "success",
}


@dataclass(frozen=True)
class PlannedJob:
    job_id: str
    topology_id: str
    queue: str
    output_dir: Path
    command: list[str]
    log_path: Path


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_jobs_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def suffix_after_jobs(path: Path) -> Path:
    parts = path.parts
    if "jobs" in parts:
        index = parts.index("jobs")
        return Path(*parts[index + 1 :])
    if len(parts) >= 3:
        return Path(*parts[-3:])
    return Path(path.name)


def replace_flag_value(command: list[str], flag: str, value: str) -> list[str]:
    if flag not in command:
        return [*command, flag, value]
    index = command.index(flag)
    if index + 1 >= len(command):
        raise ValueError(f"{flag} is missing a value")
    return [*command[: index + 1], value, *command[index + 2 :]]


def command_for_execute(raw_command: str, output_dir: Path) -> list[str]:
    command = [part for part in shlex.split(raw_command) if part != "--dry-run"]
    command = replace_flag_value(command, "--output-dir", str(output_dir))
    if "--execute" not in command:
        command.append("--execute")
    return command


def sanitize_label(value: str) -> str:
    return (
        str(value)
        .replace("|", "_")
        .replace("/", "_")
        .replace("=", "")
        .replace(" ", "_")
        .replace(":", "_")
    )


def job_from_plan_row(row: dict[str, str], *, output_root: Path) -> PlannedJob:
    queue = "long" if str(row.get("runtime_class", "")).lower() == "long" else "normal"
    suffix = suffix_after_jobs(Path(row["output_dir"]))
    output_dir = output_root / "jobs" / suffix
    command = command_for_execute(row["run_one_job_command"], output_dir)
    label = sanitize_label(row.get("job_id") or f"{row.get('topology_id', 'job')}_{suffix}")
    log_path = output_root / "logs" / "jobs" / f"{label}.log"
    return PlannedJob(
        job_id=str(row["job_id"]),
        topology_id=str(row["topology_id"]),
        queue=queue,
        output_dir=output_dir,
        command=command,
        log_path=log_path,
    )


def is_job_success(job: PlannedJob) -> bool:
    status_path = job.output_dir / "job_status.json"
    if not status_path.exists():
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return all(status.get(key) == expected for key, expected in SUCCESS_STATUSES.items())


def split_queues(jobs: list[PlannedJob]) -> dict[str, list[PlannedJob]]:
    queues = {"normal": [], "long": []}
    for job in jobs:
        queues[job.queue].append(job)
    return queues


def loadavg() -> list[float] | None:
    try:
        return [float(part) for part in os.getloadavg()]
    except OSError:
        return None


def process_snapshot() -> list[dict[str, Any]]:
    try:
        completed = subprocess.run(
            [
                "ps",
                "-eo",
                "pid,pcpu,pmem,stat,comm,args",
                "--sort=-pcpu",
            ],
            check=False,
            text=True,
            capture_output=True,
        )
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines()[1:16]:
        parts = line.split(None, 5)
        if len(parts) < 6:
            continue
        pid, pcpu, pmem, stat, comm, args = parts
        if "run_one_job.py" not in args and "train_spoplus" not in args and "train_2stage" not in args:
            continue
        rows.append(
            {
                "pid": int(pid),
                "pcpu": float(pcpu),
                "pmem": float(pmem),
                "stat": stat,
                "comm": comm,
                "args": args[:500],
            }
        )
    return rows


def write_planned_jobs_csv(path: Path, jobs: list[PlannedJob]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["job_id", "topology_id", "queue", "output_dir", "log_path", "command"],
        )
        writer.writeheader()
        for job in jobs:
            writer.writerow(
                {
                    "job_id": job.job_id,
                    "topology_id": job.topology_id,
                    "queue": job.queue,
                    "output_dir": str(job.output_dir),
                    "log_path": str(job.log_path),
                    "command": shlex.join(job.command),
                }
            )


def monitor_payload(
    *,
    started_at: float,
    active: dict[str, dict[str, Any]],
    finished: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    failed: list[dict[str, Any]],
    pending: dict[str, list[PlannedJob]],
    normal_workers: int,
    long_workers: int,
) -> dict[str, Any]:
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_seconds": round(time.time() - started_at, 3),
        "active_jobs": len(active),
        "active_normal": sum(1 for item in active.values() if item["job"].queue == "normal"),
        "active_long": sum(1 for item in active.values() if item["job"].queue == "long"),
        "finished_jobs": len(finished),
        "skipped_jobs": len(skipped),
        "failed_jobs": len(failed),
        "pending_normal": len(pending["normal"]),
        "pending_long": len(pending["long"]),
        "normal_workers": int(normal_workers),
        "long_workers": int(long_workers),
        "loadavg": loadavg(),
        "processes": process_snapshot(),
    }


def append_monitor(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def launch_job(job: PlannedJob, env: dict[str, str]) -> dict[str, Any]:
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    job.output_dir.mkdir(parents=True, exist_ok=True)
    log_handle = job.log_path.open("a", encoding="utf-8")
    log_handle.write(f"[formal-launcher] start {time.strftime('%Y-%m-%dT%H:%M:%S%z')} {job.job_id}\n")
    log_handle.write(f"[formal-launcher] command {shlex.join(job.command)}\n")
    log_handle.flush()
    process = subprocess.Popen(
        job.command,
        cwd=PROJECT_ROOT,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    return {"job": job, "process": process, "log_handle": log_handle, "started_at": time.time()}


def run_scheduler(
    *,
    jobs: list[PlannedJob],
    output_root: Path,
    normal_workers: int,
    long_workers: int,
    monitor_interval: int,
    fail_fast: bool = False,
) -> int:
    output_root.mkdir(parents=True, exist_ok=True)
    monitor_path = output_root / "formal_launcher_monitor.jsonl"
    status_path = output_root / "formal_launcher_status.json"
    summary_path = output_root / "formal_launcher_summary.json"
    planned_csv_path = output_root / "formal_launcher_jobs.csv"
    write_planned_jobs_csv(planned_csv_path, jobs)

    skipped: list[dict[str, Any]] = []
    runnable: list[PlannedJob] = []
    for job in jobs:
        if is_job_success(job):
            skipped.append({"job_id": job.job_id, "queue": job.queue, "status": "skipped_success"})
        else:
            runnable.append(job)

    pending = split_queues(runnable)
    active: dict[str, dict[str, Any]] = {}
    finished: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    started_at = time.time()
    last_monitor = 0.0
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )

    atomic_write_json(
        status_path,
        {
            "status": "running",
            "job_count": len(jobs),
            "runnable_jobs": len(runnable),
            "skipped_jobs": len(skipped),
            "failed_jobs": 0,
            "normal_workers": int(normal_workers),
            "long_workers": int(long_workers),
            "thread_limits": {key: env[key] for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
            "monitor_path": str(monitor_path),
            "summary_path": str(summary_path),
        },
    )
    print(
        "[formal-launcher] start "
        f"jobs={len(jobs)} runnable={len(runnable)} skipped={len(skipped)} "
        f"normal_workers={normal_workers} long_workers={long_workers}",
        flush=True,
    )

    def maybe_launch(queue_name: str, limit: int) -> None:
        while sum(1 for item in active.values() if item["job"].queue == queue_name) < int(limit):
            if not pending[queue_name]:
                return
            job = pending[queue_name].pop(0)
            active[job.job_id] = launch_job(job, env)
            print(f"[formal-launcher] launched {queue_name} {job.job_id}", flush=True)

    while pending["normal"] or pending["long"] or active:
        maybe_launch("normal", normal_workers)
        maybe_launch("long", long_workers)

        for job_id, item in list(active.items()):
            process: subprocess.Popen = item["process"]
            returncode = process.poll()
            if returncode is None:
                continue
            item["log_handle"].write(
                f"[formal-launcher] end {time.strftime('%Y-%m-%dT%H:%M:%S%z')} "
                f"returncode={returncode}\n"
            )
            item["log_handle"].close()
            elapsed = round(time.time() - item["started_at"], 3)
            row = {
                "job_id": job_id,
                "queue": item["job"].queue,
                "returncode": int(returncode),
                "elapsed_seconds": elapsed,
                "output_dir": str(item["job"].output_dir),
                "log_path": str(item["job"].log_path),
            }
            if returncode == 0 and is_job_success(item["job"]):
                row["status"] = "success"
                finished.append(row)
            else:
                row["status"] = "failed"
                failed.append(row)
                if fail_fast:
                    pending["normal"].clear()
                    pending["long"].clear()
            del active[job_id]
            print(
                f"[formal-launcher] finished {job_id} status={row['status']} "
                f"rc={returncode} elapsed={elapsed}s",
                flush=True,
            )

        now = time.time()
        if now - last_monitor >= int(monitor_interval):
            payload = monitor_payload(
                started_at=started_at,
                active=active,
                finished=finished,
                skipped=skipped,
                failed=failed,
                pending=pending,
                normal_workers=normal_workers,
                long_workers=long_workers,
            )
            append_monitor(monitor_path, payload)
            atomic_write_json(
                status_path,
                {
                    "status": "running",
                    "job_count": len(jobs),
                    "finished_jobs": len(finished),
                    "skipped_jobs": len(skipped),
                    "failed_jobs": len(failed),
                    "active_jobs": len(active),
                    "pending_normal": len(pending["normal"]),
                    "pending_long": len(pending["long"]),
                    "elapsed_seconds": payload["elapsed_seconds"],
                    "normal_workers": int(normal_workers),
                    "long_workers": int(long_workers),
                    "monitor_path": str(monitor_path),
                    "summary_path": str(summary_path),
                },
            )
            print(
                "[formal-launcher] monitor "
                f"active={payload['active_jobs']} finished={payload['finished_jobs']} "
                f"skipped={payload['skipped_jobs']} failed={payload['failed_jobs']} "
                f"pending_normal={payload['pending_normal']} pending_long={payload['pending_long']} "
                f"load={payload['loadavg']}",
                flush=True,
            )
            last_monitor = now
        time.sleep(1)

    elapsed = round(time.time() - started_at, 3)
    status = "success" if not failed else "failed"
    summary = {
        "status": status,
        "job_count": len(jobs),
        "finished_jobs": len(finished),
        "skipped_jobs": len(skipped),
        "failed_jobs": len(failed),
        "elapsed_seconds": elapsed,
        "normal_workers": int(normal_workers),
        "long_workers": int(long_workers),
        "finished": finished,
        "skipped": skipped,
        "failed": failed,
    }
    atomic_write_json(summary_path, summary)
    atomic_write_json(
        status_path,
        {
            "status": status,
            "job_count": len(jobs),
            "finished_jobs": len(finished),
            "skipped_jobs": len(skipped),
            "failed_jobs": len(failed),
            "elapsed_seconds": elapsed,
            "normal_workers": int(normal_workers),
            "long_workers": int(long_workers),
            "monitor_path": str(monitor_path),
            "summary_path": str(summary_path),
        },
    )
    print(f"[formal-launcher] complete status={status} elapsed={elapsed}s failed={len(failed)}", flush=True)
    return 0 if status == "success" else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-csv", type=Path, default=DEFAULT_JOBS_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--normal-workers", type=int, default=16)
    parser.add_argument("--long-workers", type=int, default=4)
    parser.add_argument("--monitor-interval", type=int, default=60)
    parser.add_argument("--expected-job-count", type=int, default=270)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = read_jobs_csv(args.jobs_csv)
    if len(rows) != int(args.expected_job_count):
        raise ValueError(
            f"expected {args.expected_job_count} rows in {args.jobs_csv}, observed {len(rows)}"
        )
    jobs = [job_from_plan_row(row, output_root=args.output_root) for row in rows]
    duplicate_ids = sorted({job.job_id for job in jobs if sum(1 for other in jobs if other.job_id == job.job_id) > 1})
    if duplicate_ids:
        raise ValueError(f"duplicate job ids: {duplicate_ids[:5]}")
    queues = split_queues(jobs)
    print(
        json.dumps(
            {
                "jobs_csv": str(args.jobs_csv),
                "output_root": str(args.output_root),
                "job_count": len(jobs),
                "normal_jobs": len(queues["normal"]),
                "long_jobs": len(queues["long"]),
                "normal_workers": int(args.normal_workers),
                "long_workers": int(args.long_workers),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.dry_run:
        return 0
    return run_scheduler(
        jobs=jobs,
        output_root=args.output_root,
        normal_workers=args.normal_workers,
        long_workers=args.long_workers,
        monitor_interval=args.monitor_interval,
        fail_fast=args.fail_fast,
    )


if __name__ == "__main__":
    raise SystemExit(main())
