#!/usr/bin/env python3
"""Preview or execute the locked formal GNN plan with bounded CPU workers."""

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
    fold: int
    seed: int
    output_dir: Path
    log_path: Path
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
    if row.get("status") != "ready":
        raise ValueError(f"job is not ready:{row.get('job_id')}:{row.get('status')}")
    command = shlex.split(row["command_preview"])
    if "--execute" in command:
        raise ValueError(f"preview command already contains --execute:{row.get('job_id')}")
    if not any(Path(token).name == "train_formal_gnn.py" for token in command):
        raise ValueError(f"unexpected trainer command:{row.get('job_id')}")
    output_dir = Path(row["output_dir"])
    try:
        output_dir.resolve().relative_to(output_root.resolve())
    except ValueError as exc:
        raise ValueError(f"output dir escapes output root:{output_dir}") from exc
    command.append("--execute")
    return Job(
        job_id=row["job_id"],
        fold=int(row["fold"]),
        seed=int(row["seed"]),
        output_dir=output_dir,
        log_path=Path(row["log_path"]),
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
    except (OSError, json.JSONDecodeError, KeyError, ValueError):
        return False
    topology_ids = [row.get("topology_id") for row in predictions]
    try:
        identity_matches = int(result.get("fold", -1)) == job.fold and int(result.get("seed", -1)) == job.seed
    except (TypeError, ValueError):
        return False
    return (
        result.get("status") == "success"
        and result.get("formal") is True
        and identity_matches
        and result.get("target") == "formal_label_mean_pp"
        and len(predictions) == 200
        and len(set(topology_ids)) == 200
    )


def process_snapshot() -> list[dict[str, Any]]:
    completed = subprocess.run(
        ["ps", "-eo", "pid,pcpu,pmem,stat,comm,args", "--sort=-pcpu"],
        check=False,
        text=True,
        capture_output=True,
    )
    output = []
    for line in completed.stdout.splitlines()[1:31]:
        parts = line.split(None, 5)
        if len(parts) != 6 or "train_formal_gnn.py" not in parts[5]:
            continue
        output.append(
            {
                "pid": int(parts[0]),
                "pcpu": float(parts[1]),
                "pmem": float(parts[2]),
                "state": parts[3],
                "command": parts[4],
                "args": parts[5][:500],
            }
        )
    return output


def launch(job: Job) -> dict[str, Any]:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = job.log_path.open("a", encoding="utf-8")
    handle.write(f"[formal-gnn-launcher] start {time.strftime('%Y-%m-%dT%H:%M:%S%z')} {job.job_id}\n")
    handle.write(f"[formal-gnn-launcher] command {shlex.join(job.command)}\n")
    handle.flush()
    environment = os.environ.copy()
    thread_count = str(job.threads)
    environment.update(
        {
            "OMP_NUM_THREADS": thread_count,
            "MKL_NUM_THREADS": thread_count,
            "OPENBLAS_NUM_THREADS": thread_count,
            "NUMEXPR_NUM_THREADS": thread_count,
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
    status_path = output_root / "launcher_status.json"
    monitor_path = output_root / "launcher_monitor.jsonl"
    summary_path = output_root / "launcher_summary.json"
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
            "processes": process_snapshot(),
            "summary_path": str(summary_path),
        }

    atomic_write_json(status_path, snapshot("running"))
    try:
        while pending or active:
            while pending and len(active) < workers:
                job = pending.pop(0)
                active[job.job_id] = launch(job)
                print(f"[formal-gnn-launcher] launched {job.job_id}", flush=True)
            for job_id, item in list(active.items()):
                returncode = item["process"].poll()
                if returncode is None:
                    continue
                item["handle"].write(
                    f"[formal-gnn-launcher] end {time.strftime('%Y-%m-%dT%H:%M:%S%z')} returncode={returncode}\n"
                )
                item["handle"].close()
                row = {
                    "job_id": job_id,
                    "fold": item["job"].fold,
                    "seed": item["job"].seed,
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
                print(f"[formal-gnn-launcher] finished {job_id} status={row['status']}", flush=True)
            if time.time() - last_monitor >= monitor_interval:
                payload = snapshot("running")
                atomic_write_json(status_path, payload)
                append_jsonl(monitor_path, payload)
                print(
                    "[formal-gnn-launcher] monitor "
                    f"active={payload['active_jobs']} finished={payload['finished_jobs']} "
                    f"failed={payload['failed_jobs']} pending={payload['pending_jobs']}",
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
    summary = {
        **snapshot(final_status),
        "finished": finished,
        "skipped": skipped,
        "failed": failed,
    }
    atomic_write_json(summary_path, summary)
    atomic_write_json(status_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if final_status == "success" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-job-count", type=int, default=15)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--monitor-interval", type=int, default=15)
    parser.add_argument("--require-hostname")
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
    if {(job.fold, job.seed) for job in jobs} != {(fold, seed) for fold in range(5) for seed in (42, 43, 44)}:
        raise ValueError("job matrix is not exactly 5 folds x seeds 42/43/44")
    hostname = socket.gethostname()
    if args.require_hostname and args.require_hostname not in hostname:
        raise ValueError(f"hostname safety check failed:{hostname}")
    preview = {
        "passed": True,
        "execute": args.execute,
        "hostname": hostname,
        "jobs_csv": str(args.jobs_csv),
        "output_root": str(args.output_root),
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
