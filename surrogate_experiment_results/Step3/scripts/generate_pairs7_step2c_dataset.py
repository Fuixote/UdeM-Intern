#!/usr/bin/env python3
"""Generate and audit an isolated Step3 pairs=7 Step2c dataset."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple


PROJECT_ROOT = Path(__file__).resolve().parents[3]
STEP3_ROOT = PROJECT_ROOT / "surrogate_experiment_results" / "Step3"
EXPERIMENT_ROOT = STEP3_ROOT / "pairs7"
RAW_GENERATOR = PROJECT_ROOT / "0-data-generation.py"
STEP2C_PROCESSOR = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "Step2"
    / "Step2c_polynomial_degree_multiplicative_noise"
    / "data-processing.py"
)
LABEL_MODE_STEP2C = "step2c_polynomial_degree_multiplicative_noise"
BATCH_NAME_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}(?:__.+)?$")

DEFAULT_PATIENTS = 7
DEFAULT_PROB_NDD = 0.20
DEFAULT_SEED = 20260619
DEFAULT_INSTANCES = 1000


class Commands(NamedTuple):
    raw: list[str]
    process: list[str]


def default_raw_output_dir(seed: int) -> Path:
    return PROJECT_ROOT / "dataset" / "raw" / f"step3_pairs7_raw_seed{seed}"


def default_processed_output_dir(seed: int) -> Path:
    return (
        PROJECT_ROOT
        / "dataset"
        / "processed"
        / f"step3_pairs7_step2c_poly_d8_mult_eps050_seed{seed}"
    )


def raw_batch_dir(raw_output_dir: Path, seed: int) -> Path:
    raw_output_dir = Path(raw_output_dir)
    if BATCH_NAME_PATTERN.match(raw_output_dir.name):
        return raw_output_dir
    return raw_output_dir / f"2026-06-19_000000__step3_pairs7_raw_seed{seed}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an isolated Step3 dataset with 7 paired patients per graph, "
            "high NDD prevalence, and Step2c d8 multiplicative labels."
        )
    )
    parser.add_argument("--instances", type=int, default=DEFAULT_INSTANCES)
    parser.add_argument("--patients", type=int, default=DEFAULT_PATIENTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--prob-ndd", "--prob_ndd", dest="prob_ndd", type=float, default=DEFAULT_PROB_NDD)
    parser.add_argument("--raw-output-dir", type=Path, default=None)
    parser.add_argument("--processed-output-dir", type=Path, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--step2c-degree", dest="step2c_degree", type=int, default=8)
    parser.add_argument("--step2c-kappa", dest="step2c_kappa", type=float, default=3.0)
    parser.add_argument("--step2c-delta", dest="step2c_delta", type=float, default=1e-12)
    parser.add_argument("--step2c-epsilon-bar", dest="step2c_epsilon_bar", type=float, default=0.5)
    parser.add_argument("--label-seed", dest="label_seed", type=int, default=DEFAULT_SEED)

    args = parser.parse_args(argv)
    if args.patients != DEFAULT_PATIENTS:
        raise ValueError("Step3 pairs7 requires --patients 7")
    if args.instances <= 0:
        raise ValueError("--instances must be positive")
    if not (0.0 <= args.prob_ndd < 1.0):
        raise ValueError("--prob-ndd must be in [0, 1)")
    if args.raw_output_dir is None:
        args.raw_output_dir = default_raw_output_dir(args.seed)
    if args.processed_output_dir is None:
        args.processed_output_dir = default_processed_output_dir(args.seed)
    return args


def format_float(value: float) -> str:
    return f"{value:g}"


def build_commands(args: argparse.Namespace) -> Commands:
    raw_input_dir = raw_batch_dir(args.raw_output_dir, args.seed)
    raw_cmd = [
        args.python,
        str(RAW_GENERATOR),
        "--instances",
        str(args.instances),
        "--patients",
        str(args.patients),
        "--prob_ndd",
        format_float(args.prob_ndd),
        "--seed",
        str(args.seed),
        "--output_dir",
        str(raw_input_dir),
        "--no_tune",
        "--force",
    ]

    process_cmd = [
        args.python,
        str(STEP2C_PROCESSOR),
        str(raw_input_dir),
        str(args.processed_output_dir),
        "--all",
        "--force",
        "--label_mode",
        LABEL_MODE_STEP2C,
        "--step2c_degree",
        str(args.step2c_degree),
        "--step2c_kappa",
        format_float(args.step2c_kappa),
        "--step2c_delta",
        format_float(args.step2c_delta),
        "--step2c_epsilon_bar",
        format_float(args.step2c_epsilon_bar),
        "--label_seed",
        str(args.label_seed),
        "--output_as_batch_dir",
    ]
    return Commands(raw=raw_cmd, process=process_cmd)


def graph_counts(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    data = payload.get("data", {})
    if not isinstance(data, dict):
        raise ValueError(f"{path.name}: data must be an object")

    pair_count = 0
    ndd_count = 0
    arc_count = 0
    for vertex in data.values():
        vertex_type = vertex.get("type", "Pair")
        if vertex_type == "NDD":
            ndd_count += 1
        else:
            pair_count += 1
        matches = vertex.get("matches", [])
        if not isinstance(matches, list):
            raise ValueError(f"{path.name}: matches must be a list")
        arc_count += len(matches)

    metadata_vertices = payload.get("metadata", {}).get("total_vertices")
    total_vertices = int(metadata_vertices) if metadata_vertices is not None else len(data)
    return {
        "graph_file": path.name,
        "pair_count": pair_count,
        "ndd_count": ndd_count,
        "total_vertices": total_vertices,
        "arc_count": arc_count,
    }


def audit_processed_dataset(processed_dir: Path, expected_pairs: int = DEFAULT_PATIENTS) -> dict[str, Any]:
    processed_dir = Path(processed_dir)
    graph_paths = sorted(processed_dir.glob("G-*.json"))
    if not graph_paths:
        raise ValueError(f"No processed G-*.json files found in {processed_dir}")

    rows = [graph_counts(path) for path in graph_paths]
    bad_rows = [row for row in rows if row["pair_count"] != expected_pairs]
    if bad_rows:
        details = ", ".join(
            f"{row['graph_file']} has {row['pair_count']}" for row in bad_rows[:5]
        )
        raise ValueError(f"Processed dataset expected {expected_pairs} Pair vertices per graph: {details}")

    ndd_counts = [row["ndd_count"] for row in rows]
    arc_counts = [row["arc_count"] for row in rows]
    summary = {
        "processed_dir": str(processed_dir),
        "graph_count": len(rows),
        "expected_pair_count": expected_pairs,
        "min_ndd_count": min(ndd_counts),
        "max_ndd_count": max(ndd_counts),
        "mean_ndd_count": sum(ndd_counts) / len(ndd_counts),
        "min_arc_count": min(arc_counts),
        "max_arc_count": max(arc_counts),
        "mean_arc_count": sum(arc_counts) / len(arc_counts),
        "rows": rows,
    }
    (processed_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def manifest_payload(args: argparse.Namespace, commands: Commands, audit: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "experiment": "step3_pairs7_step2c_poly_d8_mult_eps050",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dry_run": bool(args.dry_run),
        "config": {
            "instances": args.instances,
            "patients": args.patients,
            "prob_ndd": args.prob_ndd,
            "seed": args.seed,
            "raw_output_dir": str(args.raw_output_dir),
            "raw_batch_dir": str(raw_batch_dir(args.raw_output_dir, args.seed)),
            "processed_output_dir": str(args.processed_output_dir),
            "label_mode": LABEL_MODE_STEP2C,
            "step2c_degree": args.step2c_degree,
            "step2c_kappa": args.step2c_kappa,
            "step2c_delta": args.step2c_delta,
            "step2c_epsilon_bar": args.step2c_epsilon_bar,
            "label_seed": args.label_seed,
        },
        "commands": {
            "raw": commands.raw,
            "process": commands.process,
        },
        "audit": audit,
    }


def write_manifest(args: argparse.Namespace, commands: Commands, audit: dict[str, Any] | None) -> Path:
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_name = "dry_run_manifest.json" if args.dry_run else "run_manifest.json"
    manifest_path = EXPERIMENT_ROOT / manifest_name
    manifest_path.write_text(
        json.dumps(manifest_payload(args, commands, audit), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def print_command(label: str, command: list[str]) -> None:
    print(f"[{label}]")
    print(" ".join(command))


def run_command(command: list[str]) -> None:
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    commands = build_commands(args)

    print_command("raw generation", commands.raw)
    print_command("step2c processing", commands.process)

    audit = None
    if not args.dry_run:
        run_command(commands.raw)
        run_command(commands.process)
        audit = audit_processed_dataset(args.processed_output_dir, expected_pairs=args.patients)

    manifest_path = write_manifest(args, commands, audit)
    print(f"manifest: {manifest_path}")
    if audit is not None:
        print(f"audited graphs: {audit['graph_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
