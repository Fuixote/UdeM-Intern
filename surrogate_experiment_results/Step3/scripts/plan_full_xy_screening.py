#!/usr/bin/env python3
"""Plan Phase-B' full-(X,y) fixed-topology screening jobs without launching them."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import fixed_topology_xy_common as common  # noqa: E402
import sample_fixed_topology_context as context_sampler  # noqa: E402


DEFAULT_SELECTED_TOPOLOGIES_CSV = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "Step3"
    / "pairs20_ndd2"
    / "screening"
    / "phase_b_topologies.csv"
)
DEFAULT_TOPOLOGY_ROOT = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "Step3"
    / "pairs20_ndd2"
    / "data"
    / "topologies"
)
DEFAULT_BASE_PAYLOAD_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "step3_pairs20_ndd2_step2c_poly_d8_mult_eps050_seed20260619"
)
DEFAULT_OUTPUT_ROOT = Path("/tmp/step3_full_xy_screening_plan")
DEFAULT_REGIME = "step2c_poly_d8_mult_eps050"
RUN_ONE_JOB = SCRIPT_DIR / "run_one_job.py"

JOB_FIELDS = [
    "topology_id",
    "regime",
    "protocol",
    "train_seed",
    "train_size",
    "topology_template_path",
    "base_payload_path",
    "train_bank_path",
    "eval_manifest_path",
    "validation_path",
    "test_path",
    "output_dir",
    "expected_train_prefix_hash",
    "validation_hash",
    "test_hash",
    "run_one_job_command",
]


def read_csv_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        field_set: set[str] = set()
        for row in rows:
            field_set.update(row)
        fieldnames = sorted(field_set)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def int_or_text_key(value: Any) -> tuple[int, int | str]:
    text = str(value)
    if text.startswith("G-"):
        text = text[2:]
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def selected_topology_rows(path: str | Path, max_topologies: int | None = None) -> list[dict[str, Any]]:
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError(f"No selected topologies found in {path}")
    if "selection_rank" in rows[0]:
        rows = sorted(rows, key=lambda row: int(row.get("selection_rank") or 0))
    else:
        rows = sorted(rows, key=lambda row: int_or_text_key(row.get("topology_id", "")))
    if max_topologies is not None:
        rows = rows[: int(max_topologies)]
    return rows


def resolve_topology_template_path(topology_root: str | Path, topology_id: str) -> Path:
    root = Path(topology_root)
    candidates = [
        root / topology_id / "template.json",
        root / f"{topology_id}.template.json",
        root / f"{topology_id}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_base_payload_path(base_payload_dir: str | Path, topology_id: str) -> Path:
    root = Path(base_payload_dir)
    candidates = [
        root / f"{topology_id}.json",
        root / topology_id / "base_payload.json",
        root / topology_id / "payload.json",
        root / topology_id / f"{topology_id}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def read_train_prefix_hash(train_bank_path: Path, train_size: int) -> str | None:
    if not train_bank_path.exists():
        return None
    dataset = common.read_npz_dataset(train_bank_path)
    return dataset["manifest"].get("prefix_hashes", {}).get(str(int(train_size)))


def read_eval_hashes(eval_manifest_path: Path) -> dict[str, Any]:
    if not eval_manifest_path.exists():
        return {
            "validation_path": None,
            "test_path": None,
            "validation_hash": None,
            "test_hash": None,
        }
    manifest = json.loads(eval_manifest_path.read_text(encoding="utf-8"))
    return {
        "validation_path": manifest.get("validation_path"),
        "test_path": manifest.get("test_path"),
        "validation_hash": manifest.get("validation_hash"),
        "test_hash": manifest.get("test_hash"),
    }


def build_run_one_job_command(job: dict[str, Any]) -> str:
    command = [
        sys.executable,
        str(RUN_ONE_JOB),
        "--train-bank",
        str(job["train_bank_path"]),
        "--eval-manifest",
        str(job["eval_manifest_path"]),
        "--topology-id",
        str(job["topology_id"]),
        "--regime",
        str(job["regime"]),
        "--protocol",
        str(job["protocol"]),
        "--train-seed",
        str(job["train_seed"]),
        "--train-size",
        str(job["train_size"]),
        "--output-dir",
        str(job["output_dir"]),
        "--dry-run",
    ]
    return shlex.join(command)


def build_screening_plan(
    *,
    selected_topologies_csv: str | Path = DEFAULT_SELECTED_TOPOLOGIES_CSV,
    topology_root: str | Path = DEFAULT_TOPOLOGY_ROOT,
    base_payload_dir: str | Path = DEFAULT_BASE_PAYLOAD_DIR,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    regime: str = DEFAULT_REGIME,
    train_seed_start: int = 1,
    train_seed_count: int = 20,
    train_size: int = 50,
    validation_size: int = 100,
    test_size: int = 500,
    max_topologies: int | None = None,
    protocol: str = "screen",
    context_generator_config: str | Path | None = None,
) -> dict[str, Any]:
    if protocol not in {"screen", "confirm"}:
        raise ValueError("protocol must be screen or confirm")
    if context_generator_config is None:
        raise ValueError("context_generator_config is required")
    output_root = Path(output_root)
    generator_config = context_sampler.load_generator_config(context_generator_config)
    selected_rows = selected_topology_rows(selected_topologies_csv, max_topologies=max_topologies)
    jobs: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for row in selected_rows:
        topology_id = str(row["topology_id"])
        template_path = resolve_topology_template_path(topology_root, topology_id)
        base_payload_path = resolve_base_payload_path(base_payload_dir, topology_id)
        data_dir = output_root / "data" / str(regime) / topology_id
        eval_manifest_path = data_dir / "eval" / "eval_manifest.json"
        eval_hashes = read_eval_hashes(eval_manifest_path)
        validation_path = eval_hashes["validation_path"] or str(data_dir / "eval" / "validation.npz")
        test_path = eval_hashes["test_path"] or str(data_dir / "eval" / "test.npz")
        summary = {
            **row,
            "protocol": protocol,
            "topology_template_path": str(template_path),
            "topology_template_exists": template_path.exists(),
            "base_payload_path": str(base_payload_path),
            "base_payload_exists": base_payload_path.exists(),
            "eval_manifest_path": str(eval_manifest_path),
            "eval_manifest_exists": eval_manifest_path.exists(),
        }
        summary_rows.append(summary)
        for train_seed in range(int(train_seed_start), int(train_seed_start) + int(train_seed_count)):
            train_bank_path = data_dir / "train_banks" / f"train_seed={int(train_seed):06d}.npz"
            job = {
                "topology_id": topology_id,
                "regime": str(regime),
                "protocol": str(protocol),
                "train_seed": int(train_seed),
                "train_size": int(train_size),
                "topology_template_path": str(template_path),
                "base_payload_path": str(base_payload_path),
                "train_bank_path": str(train_bank_path),
                "eval_manifest_path": str(eval_manifest_path),
                "validation_path": str(validation_path),
                "test_path": str(test_path),
                "output_dir": str(
                    output_root
                    / "jobs"
                    / str(regime)
                    / topology_id
                    / f"train_seed={int(train_seed):06d}"
                    / f"train_size={int(train_size)}"
                ),
                "expected_train_prefix_hash": read_train_prefix_hash(train_bank_path, train_size),
                "validation_hash": eval_hashes["validation_hash"],
                "test_hash": eval_hashes["test_hash"],
            }
            job["run_one_job_command"] = build_run_one_job_command(job)
            jobs.append(job)
    return {
        "status": "planned",
        "plan_only": True,
        "dry_run": True,
        "protocol": str(protocol),
        "selected_topologies_csv": str(selected_topologies_csv),
        "topology_root": str(topology_root),
        "base_payload_dir": str(base_payload_dir),
        "context_generator_config": str(context_generator_config),
        "generator_config_hash": common.generator_config_hash(generator_config),
        "output_root": str(output_root),
        "regime": str(regime),
        "topology_count": len(selected_rows),
        "train_seed_start": int(train_seed_start),
        "train_seed_count": int(train_seed_count),
        "train_size": int(train_size),
        "validation_size": int(validation_size),
        "test_size": int(test_size),
        "job_count": len(jobs),
        "jobs": jobs,
        "selected_topologies": summary_rows,
    }


def write_plan_outputs(plan: dict[str, Any], output_root: str | Path) -> None:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    common.atomic_write_json(output_root / "screening_plan.json", plan)
    write_csv(output_root / "screening_jobs.csv", list(plan.get("jobs", [])), JOB_FIELDS)
    summary_rows = list(plan.get("selected_topologies", []))
    write_csv(output_root / "selected_topology_summary.csv", summary_rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-topologies-csv", type=Path, default=DEFAULT_SELECTED_TOPOLOGIES_CSV)
    parser.add_argument("--topology-root", type=Path, default=DEFAULT_TOPOLOGY_ROOT)
    parser.add_argument("--base-payload-dir", type=Path, default=DEFAULT_BASE_PAYLOAD_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--regime", default=DEFAULT_REGIME)
    parser.add_argument("--train-seed-start", type=int, default=1)
    parser.add_argument("--train-seed-count", type=int, default=20)
    parser.add_argument("--train-size", type=int, default=50)
    parser.add_argument("--validation-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--max-topologies", type=int, default=None)
    parser.add_argument("--protocol", choices=("screen", "confirm"), default="screen")
    parser.add_argument("--context-generator-config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", default=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan = build_screening_plan(
        selected_topologies_csv=args.selected_topologies_csv,
        topology_root=args.topology_root,
        base_payload_dir=args.base_payload_dir,
        output_root=args.output_root,
        regime=args.regime,
        train_seed_start=args.train_seed_start,
        train_seed_count=args.train_seed_count,
        train_size=args.train_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
        max_topologies=args.max_topologies,
        protocol=args.protocol,
        context_generator_config=args.context_generator_config,
    )
    write_plan_outputs(plan, args.output_root)
    print(json.dumps({key: plan[key] for key in ("status", "protocol", "topology_count", "job_count", "output_root")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
