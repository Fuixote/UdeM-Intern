#!/usr/bin/env python3
"""Run Phase 0 KEP-vs-PyEPO migration preflight.

Phase 0 checks the real KEP data and Step1c oracle/algebra boundary before any
PyEPO wrapper, trajectory, or plot work.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

try:
    from . import kep_validation_core as core
except ImportError:  # pragma: no cover - direct script execution path
    SCRIPT_DIR = Path(__file__).resolve().parent
    REPO_ROOT = Path(__file__).resolve().parents[3]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    import kep_validation_core as core


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=core.DEFAULT_DATA_DIR)
    parser.add_argument("--graph-count", type=int, default=2)
    parser.add_argument("--theta-mode", choices=["random", "ols"], default="random")
    parser.add_argument("--theta-seed", type=int, default=42)
    parser.add_argument("--gurobi-seed", type=int, default=42)
    parser.add_argument("--json", type=Path, default=core.DEFAULT_JSON_PATH)
    parser.add_argument("--no-json", action="store_true")
    parser.add_argument("--no-strict", action="store_true")
    return parser


def run_phase0(args):
    import gurobipy as gp

    graph_paths = core.select_graph_paths(args.data_dir, args.graph_count)
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Seed", args.gurobi_seed)
    env.start()

    records = []
    try:
        records = core.step1c_common.load_graph_records(graph_paths, env)
        theta = core.make_probe_theta(
            records,
            mode=args.theta_mode,
            seed=args.theta_seed,
        )
        step1a = core.step1c_common.load_step1a_module()
        record_checks = [
            core.step1c_code_path_check(record, theta, step1a=step1a, env=env)
            for record in records
        ]
        checks = core.summarize_phase0(record_checks)
        return core.build_payload(
            args.data_dir,
            graph_paths,
            theta,
            record_checks,
            checks,
        )
    finally:
        if records:
            core.step1c_common.dispose_graph_records(records)
        env.dispose()


def print_payload(payload):
    print("KEP-vs-PyEPO Phase 0 preflight")
    print("data_dir: {}".format(payload["data_dir"]))
    print("graphs: {}".format(len(payload["graphs"])))
    print("theta: {}".format([round(value, 6) for value in payload["theta"]]))
    print()
    for item in payload["results"]:
        status = "PASS" if item["passed"] else "FAIL"
        print(
            "{:<46} {:>12.4e} <= {:<10.4e} {}".format(
                item["name"],
                item["value"],
                item["threshold"],
                status,
            )
        )


def main(argv=None):
    args = build_parser().parse_args(argv)
    payload = run_phase0(args)
    print_payload(payload)
    if not args.no_json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if not args.no_strict and not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
