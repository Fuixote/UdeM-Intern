#!/usr/bin/env python3
"""Launch Step5 paired jobs with bounded, resumable normal/long queues."""

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
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
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


def atomic_write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)


def read_jobs_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def suffix_after_jobs(path: Path) -> Path:
    if "jobs" not in path.parts:
        raise ValueError(f"planned output_dir does not contain a jobs component: {path}")
    index = path.parts.index("jobs")
    suffix = Path(*path.parts[index + 1 :])
    if not suffix.parts:
        raise ValueError(f"planned output_dir has no suffix after jobs: {path}")
    return suffix


def replace_flag_value(command: list[str], flag: str, value: str) -> list[str]:
    if flag not in command:
        raise ValueError(f"planned command is missing {flag}")
    index = command.index(flag)
    if index + 1 >= len(command):
        raise ValueError(f"{flag} is missing its value")
    return [*command[: index + 1], value, *command[index + 2 :]]


def command_for_execute(raw_command: str, output_dir: Path) -> list[str]:
    command = shlex.split(raw_command)
    if "--dry-run" not in command:
        raise ValueError("refusing to launch a plan command that was not dry-run-only")
    if "--execute" in command:
        raise ValueError("refusing a plan command that already contains --execute")
    if not any(Path(token).name == "run_one_job.py" for token in command):
        raise ValueError("planned command does not invoke run_one_job.py")
    command = [part for part in command if part != "--dry-run"]
    command = replace_flag_value(command, "--output-dir", str(output_dir))
    command.append("--execute")
    return command


def sanitize_label(value: str) -> str:
    output = str(value)
    for old, new in (("|", "_"), ("/", "_"), ("=", ""), (" ", "_"), (":", "_")):
        output = output.replace(old, new)
    return output


def job_from_plan_row(row: dict[str, str], *, output_root: str | Path) -> PlannedJob:
    if str(row.get("status", "")) != "ready":
        raise ValueError(f"job is not ready: {row.get('job_id')} status={row.get('status')}")
    if str(row.get("weak_label", "")).lower() not in {"true", "1"}:
        raise ValueError(f"job is not marked weak_label=true: {row.get('job_id')}")
    queue = "long" if str(row.get("runtime_class", "")).lower() == "long" else "normal"
    suffix = suffix_after_jobs(Path(row["output_dir"]))
    output_dir = Path(output_root) / "jobs" / suffix
    command = command_for_execute(row["run_one_job_command"], output_dir)
    job_id = str(row["job_id"])
    return PlannedJob(
        job_id=job_id,
        topology_id=str(row["topology_id"]),
        queue=queue,
        output_dir=output_dir,
        command=command,
        log_path=Path(output_root) / "logs" / "jobs" / f"{sanitize_label(job_id)}.log",
    )


def is_job_success(job: PlannedJob) -> bool:
    status_path = job.output_dir / "job_status.json"
    if not status_path.is_file():
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return all(status.get(field) == value for field, value in SUCCESS_STATUSES.items())


def split_queues(jobs: list[PlannedJob]) -> dict[str, list[PlannedJob]]:
    queues = {"normal": [], "long": []}
    for job in jobs:
        queues[job.queue].append(job)
    return queues


def loadavg() -> list[float] | None:
    try:
        return [float(value) for value in os.getloadavg()]
    except OSError:
        return None


def process_snapshot() -> list[dict[str, Any]]:
    try:
        completed = subprocess.run(
            ["ps", "-eo", "pid,pcpu,pmem,stat,comm,args", "--sort=-pcpu"],
            check=False,
            text=True,
            capture_output=True,
        )
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines()[1:21]:
        parts = line.split(None, 5)
        if len(parts) != 6:
            continue
        pid, pcpu, pmem, state, command, args = parts
        if not any(name in args for name in ("run_one_job.py", "train_spoplus", "train_2stage")):
            continue
        rows.append(
            {
                "pid": int(pid),
                "pcpu": float(pcpu),
                "pmem": float(pmem),
                "state": state,
                "command": command,
                "args": args[:500],
            }
        )
    return rows


def write_launch_jobs_csv(path: Path, jobs: list[PlannedJob]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["job_id", "topology_id", "queue", "output_dir", "log_path", "command"]
        writer = csv.DictWriter(handle, fieldnames=fields)
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


def launch_job(job: PlannedJob, env: dict[str, str]) -> dict[str, Any]:
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    job.output_dir.mkdir(parents=True, exist_ok=True)
    log_handle = job.log_path.open("a", encoding="utf-8")
    log_handle.write(f"[step5-launcher] start {time.strftime('%Y-%m-%dT%H:%M:%S%z')} {job.job_id}\n")
    log_handle.write(f"[step5-launcher] command {shlex.join(job.command)}\n")
    log_handle.flush()
    process = subprocess.Popen(
        job.command,
        cwd=PROJECT_ROOT,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    return {
        "job": job,
        "process": process,
        "log_handle": log_handle,
        "started_at": time.time(),
    }


def _monitor_payload(
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
        "active_normal": sum(item["job"].queue == "normal" for item in active.values()),
        "active_long": sum(item["job"].queue == "long" for item in active.values()),
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


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _stop_active_jobs(active: dict[str, dict[str, Any]]) -> None:
    for item in active.values():
        process: subprocess.Popen = item["process"]
        if process.poll() is None:
            process.terminate()
    for item in active.values():
        process = item["process"]
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        item["log_handle"].close()


def run_scheduler(
    jobs: list[PlannedJob],
    *,
    output_root: str | Path,
    normal_workers: int,
    long_workers: int,
    monitor_interval: int,
    fail_fast: bool = False,
) -> int:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    monitor_path = output_root / "launcher_monitor.jsonl"
    status_path = output_root / "launcher_status.json"
    summary_path = output_root / "launcher_summary.json"
    write_launch_jobs_csv(output_root / "launcher_jobs.csv", jobs)

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

    def write_status(status: str) -> None:
        atomic_write_json(
            status_path,
            {
                "status": status,
                "job_count": len(jobs),
                "finished_jobs": len(finished),
                "skipped_jobs": len(skipped),
                "failed_jobs": len(failed),
                "active_jobs": len(active),
                "pending_normal": len(pending["normal"]),
                "pending_long": len(pending["long"]),
                "elapsed_seconds": round(time.time() - started_at, 3),
                "normal_workers": int(normal_workers),
                "long_workers": int(long_workers),
                "monitor_path": str(monitor_path),
                "summary_path": str(summary_path),
            },
        )

    def maybe_launch(queue_name: str, limit: int) -> None:
        while sum(item["job"].queue == queue_name for item in active.values()) < int(limit):
            if not pending[queue_name]:
                return
            job = pending[queue_name].pop(0)
            active[job.job_id] = launch_job(job, env)
            print(f"[step5-launcher] launched {queue_name} {job.job_id}", flush=True)

    write_status("running")
    try:
        while pending["normal"] or pending["long"] or active:
            maybe_launch("normal", normal_workers)
            maybe_launch("long", long_workers)
            for job_id, item in list(active.items()):
                process: subprocess.Popen = item["process"]
                returncode = process.poll()
                if returncode is None:
                    continue
                item["log_handle"].write(
                    f"[step5-launcher] end {time.strftime('%Y-%m-%dT%H:%M:%S%z')} returncode={returncode}\n"
                )
                item["log_handle"].close()
                row = {
                    "job_id": job_id,
                    "queue": item["job"].queue,
                    "returncode": int(returncode),
                    "elapsed_seconds": round(time.time() - item["started_at"], 3),
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
                print(f"[step5-launcher] finished {job_id} status={row['status']}", flush=True)

            if time.time() - last_monitor >= int(monitor_interval):
                payload = _monitor_payload(
                    started_at=started_at,
                    active=active,
                    finished=finished,
                    skipped=skipped,
                    failed=failed,
                    pending=pending,
                    normal_workers=normal_workers,
                    long_workers=long_workers,
                )
                _append_jsonl(monitor_path, payload)
                write_status("running")
                print(
                    "[step5-launcher] monitor "
                    f"active={payload['active_jobs']} finished={payload['finished_jobs']} "
                    f"failed={payload['failed_jobs']} pending_normal={payload['pending_normal']} "
                    f"pending_long={payload['pending_long']}",
                    flush=True,
                )
                last_monitor = time.time()
            time.sleep(1)
    except KeyboardInterrupt:
        _stop_active_jobs(active)
        write_status("interrupted")
        raise

    final_status = "success" if not failed else "failed"
    summary = {
        "status": final_status,
        "job_count": len(jobs),
        "finished_jobs": len(finished),
        "skipped_jobs": len(skipped),
        "failed_jobs": len(failed),
        "elapsed_seconds": round(time.time() - started_at, 3),
        "normal_workers": int(normal_workers),
        "long_workers": int(long_workers),
        "finished": finished,
        "skipped": skipped,
        "failed": failed,
    }
    atomic_write_json(summary_path, summary)
    write_status(final_status)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if final_status == "success" else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-job-count", type=int, required=True)
    parser.add_argument("--normal-workers", type=int, default=16)
    parser.add_argument("--long-workers", type=int, default=4)
    parser.add_argument("--monitor-interval", type=int, default=60)
    parser.add_argument("--require-hostname", default=None)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.normal_workers < 1 or args.long_workers < 0:
        raise ValueError("normal-workers must be >=1 and long-workers must be >=0")
    rows = read_jobs_csv(args.jobs_csv)
    if len(rows) != int(args.expected_job_count):
        raise ValueError(
            f"expected {args.expected_job_count} jobs, observed {len(rows)} in {args.jobs_csv}"
        )
    jobs = [job_from_plan_row(row, output_root=args.output_root) for row in rows]
    ids = [job.job_id for job in jobs]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate job ids in plan")
    if args.require_hostname and args.require_hostname not in socket.gethostname():
        raise ValueError(
            f"hostname safety check failed: required substring {args.require_hostname!r}, "
            f"observed {socket.gethostname()!r}"
        )
    queues = split_queues(jobs)
    if queues["long"] and args.long_workers == 0:
        raise ValueError("long-workers must be >=1 when the plan contains long jobs")
    preview = {
        "jobs_csv": str(args.jobs_csv),
        "output_root": str(args.output_root),
        "job_count": len(jobs),
        "normal_jobs": len(queues["normal"]),
        "long_jobs": len(queues["long"]),
        "normal_workers": int(args.normal_workers),
        "long_workers": int(args.long_workers),
        "hostname": socket.gethostname(),
        "execute": bool(args.execute),
    }
    print(json.dumps(preview, indent=2, sort_keys=True))
    if not args.execute:
        return 0
    return run_scheduler(
        jobs,
        output_root=args.output_root,
        normal_workers=args.normal_workers,
        long_workers=args.long_workers,
        monitor_interval=args.monitor_interval,
        fail_fast=args.fail_fast,
    )


if __name__ == "__main__":
    raise SystemExit(main())
