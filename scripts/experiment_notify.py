#!/usr/bin/env python3
"""
Send Brevo emails when a tmux experiment watcher starts and finishes.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import urllib.error
import urllib.request


DEFAULT_ENV_PATH = Path.home() / ".config" / "experiment-notify" / "brevo.env"
ERROR_PATTERNS = [
    "Traceback",
    "GurobiError",
    "FileNotFoundError",
    "No module named",
    "RuntimeError",
    "Killed",
]


def load_env_file(path):
    env = {}
    path = Path(path)
    if not path.exists():
        return env
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        parts = shlex.split(value.strip())
        env[key] = parts[0] if parts else ""
    return env


def tmux_session_exists(session):
    result = subprocess.run(
        ["tmux", "has-session", "-t", "={}".format(session)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def active_sessions(sessions):
    return [session for session in sessions if tmux_session_exists(session)]


def count_csv_rows(path):
    with Path(path).open(newline="") as handle:
        rows = sum(1 for _ in csv.reader(handle))
    return max(rows - 1, 0)


def collect_summary(result_dir, log_dir):
    result_dir = Path(result_dir)
    log_dir = Path(log_dir)
    csv_files = sorted(result_dir.rglob("*.csv")) if result_dir.exists() else []
    log_files = sorted(log_dir.rglob("*.log")) if log_dir.exists() else []
    error_logs = []
    for log_file in log_files:
        text = log_file.read_text(errors="replace")
        if any(pattern in text for pattern in ERROR_PATTERNS):
            error_logs.append(str(log_file))
    return {
        "csv_files": len(csv_files),
        "result_rows": sum(count_csv_rows(path) for path in csv_files),
        "log_files": len(log_files),
        "error_logs": error_logs,
    }


def format_message(project, sessions, summary):
    errors = "none" if not summary["error_logs"] else "\n".join(summary["error_logs"])
    return (
        "{} finished.\n\n"
        "Watched tmux sessions: {}\n"
        "Result CSV files: {}\n"
        "Result rows: {}\n"
        "Log files: {}\n"
        "Logs with error patterns: {}\n".format(
            project,
            ", ".join(sessions),
            summary["csv_files"],
            summary["result_rows"],
            summary["log_files"],
            errors,
        )
    )


def format_start_message(project, sessions, result_dir, log_dir, interval, running_sessions):
    running = ", ".join(running_sessions) if running_sessions else "none"
    return (
        "Watcher started successfully.\n\n"
        "Project: {}\n"
        "Watched tmux sessions: {}\n"
        "Currently running sessions: {}\n"
        "Check interval seconds: {}\n"
        "Result dir: {}\n"
        "Log dir: {}\n".format(
            project,
            ", ".join(sessions),
            running,
            interval,
            result_dir,
            log_dir,
        )
    )


def send_brevo_email(api_key, sender, recipient, subject, text):
    payload = {
        "sender": {"email": sender},
        "to": [{"email": recipient}],
        "subject": subject,
        "textContent": text,
    }
    request = urllib.request.Request(
        "https://api.brevo.com/v3/smtp/email",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "accept": "application/json",
            "api-key": api_key,
            "content-type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError("Brevo API failed with HTTP {}: {}".format(exc.code, body))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=Path, default=DEFAULT_ENV_PATH)
    parser.add_argument("--result-dir", type=Path, default=Path("results"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--session", action="append", dest="sessions", required=True)
    parser.add_argument("--project", default=Path.cwd().name)
    parser.add_argument("--subject")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def require_notification_env(env, env_path):
    required = ["BREVO_API_KEY", "SPO_NOTIFY_FROM", "SPO_NOTIFY_TO"]
    missing = [key for key in required if not env.get(key)]
    if missing:
        raise RuntimeError(
            "Missing notification settings in {}: {}".format(
                env_path,
                ", ".join(missing),
            )
        )
    return env


def deliver_email(send_email, env, subject, message):
    status, body = send_email(
        env["BREVO_API_KEY"],
        env["SPO_NOTIFY_FROM"],
        env["SPO_NOTIFY_TO"],
        subject,
        message,
    )
    print("Brevo notification sent: HTTP {}".format(status))
    if body:
        print(body)


def run_watcher(
    args,
    env=None,
    send_email=send_brevo_email,
    active_sessions_fn=active_sessions,
    sleep_fn=time.sleep,
    collect_summary_fn=collect_summary,
):
    running = active_sessions_fn(args.sessions)
    start_subject = "{} watcher started".format(args.project)
    start_message = format_start_message(
        args.project,
        args.sessions,
        args.result_dir,
        args.log_dir,
        args.interval,
        running,
    )

    if args.dry_run:
        output = "Subject: {}\n{}\n".format(start_subject, start_message)
    else:
        notify_env = env
        if notify_env is None:
            notify_env = os.environ.copy()
            notify_env.update(load_env_file(args.env))
        require_notification_env(notify_env, args.env)
        deliver_email(send_email, notify_env, start_subject, start_message)
        output = ""

    while running:
        print("Still running: {}".format(", ".join(running)), flush=True)
        sleep_fn(args.interval)
        running = active_sessions_fn(args.sessions)

    subject = args.subject or "{} experiments finished".format(args.project)
    summary = collect_summary_fn(args.result_dir, args.log_dir)
    message = format_message(args.project, args.sessions, summary)
    if args.dry_run:
        output += "Subject: {}\n{}\n".format(subject, message)
        print(output, end="")
        return output

    deliver_email(send_email, notify_env, subject, message)
    return output


def main():
    run_watcher(parse_args())


if __name__ == "__main__":
    main()
