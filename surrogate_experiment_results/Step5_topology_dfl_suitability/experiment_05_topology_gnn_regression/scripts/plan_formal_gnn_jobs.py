#!/usr/bin/env python3
"""Create the locked 5-fold x 3-seed formal GNN job plan."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shlex
import sys
import tempfile
from typing import Any


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRAPHS = (
    EXPERIMENT_ROOT
    / "results"
    / "multiseed_completion1880"
    / "results"
    / "formal_topology_incidence_graphs.jsonl"
)
DEFAULT_FOLDS = EXPERIMENT_ROOT / "results" / "formal_three_seed" / "splits" / "folds.csv"
DEFAULT_OUTPUT_ROOT = EXPERIMENT_ROOT / "results" / "formal_three_seed" / "gnn_formal15"
TRAINER = EXPERIMENT_ROOT / "scripts" / "train_formal_gnn.py"
LOCKED_GRAPH_SHA256 = "8a41232110bf7f151c0192c6723fe5e0d7f4b9dd83aa70aba7fa142e503fd522"
LOCKED_FOLD_SHA256 = "b66a9c2e529e1c23fd96762340c07ddc9d3aa4f8c0a25aa16cc77f3dac7c3b39"
SEEDS = (42, 43, 44)
FOLDS = tuple(range(5))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    temporary.replace(path)


def audit_inputs(graph_path: Path, fold_path: Path) -> dict[str, Any]:
    failures: list[str] = []
    graph_rows = [json.loads(line) for line in graph_path.read_text(encoding="utf-8").splitlines() if line]
    fold_rows = read_csv(fold_path)
    graph_ids = [str(row.get("topology_id")) for row in graph_rows]
    fold_ids = [row.get("topology_id", "") for row in fold_rows]
    graph_sha256 = sha256_file(graph_path)
    fold_sha256 = sha256_file(fold_path)
    if graph_sha256 != LOCKED_GRAPH_SHA256:
        failures.append(f"graph_sha256_mismatch:{graph_sha256}")
    if fold_sha256 != LOCKED_FOLD_SHA256:
        failures.append(f"fold_sha256_mismatch:{fold_sha256}")
    if len(graph_rows) != 1000 or len(set(graph_ids)) != 1000:
        failures.append(f"graph_count_or_uniqueness_mismatch:{len(graph_rows)}/{len(set(graph_ids))}")
    if len(fold_rows) != 1000 or len(set(fold_ids)) != 1000:
        failures.append(f"fold_count_or_uniqueness_mismatch:{len(fold_rows)}/{len(set(fold_ids))}")
    if set(graph_ids) != set(fold_ids):
        failures.append("graph_fold_topology_sets_differ")
    fold_sizes = {fold: 0 for fold in FOLDS}
    for row in fold_rows:
        try:
            fold = int(row["fold"])
        except (KeyError, TypeError, ValueError):
            failures.append(f"invalid_fold:{row.get('topology_id')}")
            continue
        if fold not in fold_sizes:
            failures.append(f"out_of_range_fold:{row.get('topology_id')}:{fold}")
            continue
        fold_sizes[fold] += 1
        if row.get("target_name") != "formal_label_mean_pp":
            failures.append(f"fold_target_name_mismatch:{row.get('topology_id')}")
    if any(count != 200 for count in fold_sizes.values()):
        failures.append(f"fold_size_mismatch:{fold_sizes}")
    for row in graph_rows:
        target = row.get("target", {})
        if target.get("formal") is not True or target.get("name") != "formal_label_mean_pp":
            failures.append(f"nonformal_graph_target:{row.get('topology_id')}")
            continue
        try:
            value = float(target["value"])
        except (KeyError, TypeError, ValueError):
            failures.append(f"invalid_graph_target:{row.get('topology_id')}")
        else:
            if not math.isfinite(value):
                failures.append(f"nonfinite_graph_target:{row.get('topology_id')}")
    return {
        "passed": not failures,
        "graph_count": len(graph_rows),
        "fold_assignment_count": len(fold_rows),
        "fold_sizes": fold_sizes,
        "graph_sha256": graph_sha256,
        "fold_sha256": fold_sha256,
        "target": "formal_label_mean_pp",
        "failures": failures,
    }


def build_jobs(args: argparse.Namespace, ready: bool) -> list[dict[str, Any]]:
    jobs = []
    for fold in FOLDS:
        for seed in SEEDS:
            job_id = f"fold{fold}_seed{seed}"
            output_dir = args.output_root / "jobs" / f"fold{fold}" / f"seed{seed}"
            command = [
                args.python,
                str(TRAINER),
                "--graph-jsonl",
                str(args.graph_jsonl),
                "--folds",
                str(args.folds),
                "--output-dir",
                str(output_dir),
                "--fold",
                str(fold),
                "--seed",
                str(seed),
                "--node-input-dim",
                str(args.node_input_dim),
                "--hidden-dim",
                str(args.hidden_dim),
                "--layers",
                str(args.layers),
                "--relation-count",
                str(args.relation_count),
                "--dropout",
                str(args.dropout),
                "--batch-size",
                str(args.batch_size),
                "--learning-rate",
                str(args.learning_rate),
                "--weight-decay",
                str(args.weight_decay),
                "--max-epochs",
                str(args.max_epochs),
                "--early-stop-patience",
                str(args.early_stop_patience),
                "--early-stop-min-delta",
                str(args.early_stop_min_delta),
                "--threads",
                str(args.threads),
            ]
            jobs.append(
                {
                    "manifest_index": len(jobs),
                    "job_id": job_id,
                    "fold": fold,
                    "validation_fold": (fold + 1) % 5,
                    "seed": seed,
                    "status": "ready" if ready else "blocked",
                    "output_dir": str(output_dir),
                    "log_path": str(args.output_root / "logs" / "jobs" / f"{job_id}.log"),
                    "threads": args.threads,
                    "max_epochs": args.max_epochs,
                    "early_stop_patience": args.early_stop_patience,
                    "early_stop_min_delta": args.early_stop_min_delta,
                    "command_preview": shlex.join(command),
                }
            )
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-jsonl", type=Path, default=DEFAULT_GRAPHS)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--plan-output", type=Path)
    parser.add_argument("--jobs-csv-output", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--node-input-dim", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--relation-count", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--early-stop-patience", type=int, default=30)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0001)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--expected-job-count", type=int, default=15)
    args = parser.parse_args()
    if args.max_epochs < 1 or args.early_stop_patience < 1 or args.threads < 1:
        raise ValueError("max epochs, patience, and threads must be positive")
    audit = audit_inputs(args.graph_jsonl, args.folds)
    jobs = build_jobs(args, audit["passed"])
    failures = list(audit["failures"])
    if len(jobs) != args.expected_job_count:
        failures.append(f"job_count_mismatch:{len(jobs)}!={args.expected_job_count}")
    if len({row["job_id"] for row in jobs}) != len(jobs):
        failures.append("duplicate_job_ids")
    passed = not failures and all(row["status"] == "ready" for row in jobs)
    plan = {
        "passed": passed,
        "status": "ready" if passed else "blocked",
        "experiment": "step5_exp5_topology_gnn_regression_v1",
        "target": "formal_label_mean_pp",
        "folds": list(FOLDS),
        "seeds": list(SEEDS),
        "job_count": len(jobs),
        "expected_job_count": args.expected_job_count,
        "commands_are_preview_only": True,
        "execute_flag_present": any("--execute" in shlex.split(row["command_preview"]) for row in jobs),
        "input_audit": audit,
        "hyperparameters": {
            "node_input_dim": args.node_input_dim,
            "hidden_dim": args.hidden_dim,
            "layers": args.layers,
            "relation_count": args.relation_count,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_epochs": args.max_epochs,
            "early_stop_patience": args.early_stop_patience,
            "early_stop_min_delta": args.early_stop_min_delta,
            "threads": args.threads,
        },
        "failures": failures,
        "jobs": jobs,
    }
    plan_output = args.plan_output or args.output_root / "plans" / "formal_gnn15_plan.json"
    jobs_output = args.jobs_csv_output or args.output_root / "plans" / "formal_gnn15_jobs.csv"
    atomic_write_json(plan_output, plan)
    atomic_write_csv(jobs_output, jobs)
    print(
        json.dumps(
            {
                "passed": passed,
                "status": plan["status"],
                "job_count": len(jobs),
                "execute_flag_present": plan["execute_flag_present"],
                "plan_output": str(plan_output),
                "jobs_csv_output": str(jobs_output),
                "input_audit": audit,
                "failures": failures,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
