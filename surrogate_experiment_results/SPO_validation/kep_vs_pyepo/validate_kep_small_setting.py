#!/usr/bin/env python3
"""Run Phase 1 KEP small validation with PyEPO SPOPlus reference."""

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
    parser.add_argument("--graph-count", type=int, default=5)
    parser.add_argument("--theta-mode", choices=["random", "ols"], default="random")
    parser.add_argument("--theta-seed", type=int, default=42)
    parser.add_argument("--gurobi-seed", type=int, default=42)
    parser.add_argument("--sgd-lr", type=float, default=0.05)
    parser.add_argument("--json", type=Path, default=core.DEFAULT_SMALL_SETTING_JSON_PATH)
    parser.add_argument("--no-json", action="store_true")
    parser.add_argument("--no-strict", action="store_true")
    return parser


def run_small_setting(args):
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

        checks = []
        checks.extend(core.phase1_level0_checks(records, step1a=step1a, env=env))
        checks.extend(core.phase1_level1_checks(records, step1a=step1a, env=env))
        record_checks, spoplus_checks = core.phase1_level2_3_4_checks(
            records,
            theta,
            step1a=step1a,
            env=env,
            sgd_lr=args.sgd_lr,
        )
        checks.extend(spoplus_checks)
        return core.build_phase1_payload(
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
    print("KEP-vs-PyEPO Phase 1 small validation")
    print("data_dir: {}".format(payload["data_dir"]))
    print("graphs: {}".format(len(payload["graphs"])))
    print("theta: {}".format([round(value, 6) for value in payload["theta"]]))
    print()
    for item in payload["results"]:
        status = "PASS" if item["passed"] else "FAIL"
        print(
            "{:<42} {:>12.4e} <= {:<10.4e} {}".format(
                item["name"],
                item["value"],
                item["threshold"],
                status,
            )
        )


def main(argv=None):
    args = build_parser().parse_args(argv)
    payload = run_small_setting(args)
    print_payload(payload)
    if not args.no_json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if not args.no_strict and not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
