#!/usr/bin/env python3
"""Run Step2b degree bridge alignment against PyEPO SPOPlus."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np

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


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STEP2B_ROOT = (
    REPO_ROOT
    / "surrogate_experiment_results"
    / "Step2"
    / "Step2b_polynomial_degree_noiseless"
    / "remote_results"
)
DEFAULT_FORMAL_RUN = "formal_2stage500_spoplus500_s10"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "step2b_bridge_results"


@dataclass(frozen=True)
class Step2bArtifacts:
    degree: int
    regime: str
    run_dir: Path
    train_subset_json: Path
    validation_set_json: Path
    run_config_json: Path


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step2b-root", type=Path, default=DEFAULT_STEP2B_ROOT)
    parser.add_argument("--formal-run", type=str, default=DEFAULT_FORMAL_RUN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--json", type=Path, default=DEFAULT_OUTPUT_ROOT / "latest_step2b_bridge.json")
    parser.add_argument("--degrees", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--source-train-size", type=int, default=50)
    parser.add_argument("--train-size", type=int, default=5)
    parser.add_argument("--validation-size", type=int, default=5)
    parser.add_argument("--theta-seed", type=int, default=42)
    parser.add_argument("--theta-init", type=float, nargs=2, default=None)
    parser.add_argument("--gurobi-seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--n-epochs", type=int, default=3)
    parser.add_argument("--metric-stride", type=int, default=1)
    parser.add_argument("--no-json", action="store_true")
    parser.add_argument("--no-strict", action="store_true")
    return parser


def step2b_artifacts_for_degree(
    degree,
    source_train_size,
    step2b_root=DEFAULT_STEP2B_ROOT,
    formal_run=DEFAULT_FORMAL_RUN,
):
    degree = int(degree)
    regime = "step2b_poly_d{}".format(degree)
    run_dir = (
        Path(step2b_root)
        / regime
        / "step1c_spoplus"
        / formal_run
        / "train_size={}".format(int(source_train_size))
    )
    return Step2bArtifacts(
        degree=degree,
        regime=regime,
        run_dir=run_dir,
        train_subset_json=run_dir / "train_subset.json",
        validation_set_json=run_dir / "validation_set.json",
        run_config_json=run_dir / "run_config.json",
    )


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_and_truncate_entries(path, limit):
    entries = read_json(path)
    if limit <= 0:
        raise ValueError("limit must be positive")
    if len(entries) < limit:
        raise ValueError(
            "Requested {} entries from {}, found {}".format(
                limit,
                path,
                len(entries),
            )
        )
    return list(entries[:limit])


def degree_output_dir(output_root, regime):
    return Path(output_root) / regime


def serializable_degree_payload(
    args,
    artifacts,
    train_entries,
    validation_entries,
    phase2,
    lr_bridge,
):
    output_dir = degree_output_dir(args.output_root, artifacts.regime)
    spoplus_passed = bool(phase2["summary"]["passed"])
    lr_passed = bool(lr_bridge["passed"])
    return {
        "phase": "step2b_bridge_degree",
        "degree": int(artifacts.degree),
        "regime": artifacts.regime,
        "source_run_dir": str(artifacts.run_dir),
        "output_dir": str(output_dir),
        "source_train_size": int(args.source_train_size),
        "train_size": int(args.train_size),
        "validation_size": int(args.validation_size),
        "theta_seed": int(args.theta_seed),
        "theta_init": np.asarray(phase2["pyepo_trajectory"][0], dtype=float).tolist(),
        "gurobi_seed": int(args.gurobi_seed),
        "optimizer": "adam",
        "lr": float(args.lr),
        "n_epochs": int(args.n_epochs),
        "metric_stride": int(args.metric_stride),
        "epoch_indices": np.asarray(phase2["epoch_indices"], dtype=int).tolist(),
        "train_subset": list(train_entries),
        "validation_set": list(validation_entries),
        "train_graphs": [entry["path"] for entry in train_entries],
        "validation_graphs": [entry["path"] for entry in validation_entries],
        "artifacts": {
            "pyepo_loss_curve_csv": str(output_dir / "pyepo_spoplus_loss_curve.csv"),
            "step1c_loss_curve_csv": str(output_dir / "step1c_spoplus_loss_curve.csv"),
            "diff_summary_csv": str(output_dir / "trajectory_diff_summary.csv"),
            "diff_summary_json": str(output_dir / "trajectory_diff_summary.json"),
            "run_config_json": str(output_dir / "run_config.json"),
            "train_subset_json": str(output_dir / "train_subset.json"),
            "validation_set_json": str(output_dir / "validation_set.json"),
            "pyepo_trajectory_npy": str(output_dir / "trajectory_pyepo_spoplus.npy"),
            "step1c_trajectory_npy": str(output_dir / "trajectory_step1c_spoplus.npy"),
            "pyepo_lr_summary_csv": str(output_dir / "pyepo_lr_summary.csv"),
            "step1c_lr_summary_csv": str(output_dir / "step1c_lr_summary.csv"),
            "lr_diff_summary_csv": str(output_dir / "lr_diff_summary.csv"),
            "lr_diff_summary_json": str(output_dir / "lr_diff_summary.json"),
            "pyepo_lr_theta_npy": str(output_dir / "theta_pyepo_lr.npy"),
            "step1c_lr_theta_npy": str(output_dir / "theta_step1c_lr.npy"),
        },
        "results": phase2["summary"]["results"],
        "spoplus_passed": spoplus_passed,
        "lr_results": lr_bridge["results"],
        "lr_passed": lr_passed,
        "lr": lr_bridge,
        "passed": spoplus_passed and lr_passed,
    }


def run_degree_bridge(args, artifacts, env):
    for path in (
        artifacts.train_subset_json,
        artifacts.validation_set_json,
        artifacts.run_config_json,
    ):
        if not path.exists():
            raise FileNotFoundError("Missing Step2b bridge artifact: {}".format(path))

    train_entries = load_and_truncate_entries(
        artifacts.train_subset_json,
        args.train_size,
    )
    validation_entries = load_and_truncate_entries(
        artifacts.validation_set_json,
        args.validation_size,
    )
    train_records = []
    validation_records = []
    try:
        print(
            "Loading {} train subset: n={}".format(
                artifacts.regime,
                len(train_entries),
            ),
            flush=True,
        )
        train_records = core.step1c_common.load_graph_records(
            [entry["path"] for entry in train_entries],
            env,
        )
        print(
            "Loading {} validation set: n={}".format(
                artifacts.regime,
                len(validation_entries),
            ),
            flush=True,
        )
        validation_records = core.step1c_common.load_graph_records(
            [entry["path"] for entry in validation_entries],
            env,
        )

        rng = np.random.RandomState(args.theta_seed)
        theta_init = (
            np.asarray(args.theta_init, dtype=float)
            if args.theta_init is not None
            else rng.uniform(0.5, 3.5, size=2)
        )
        step1a = core.step1c_common.load_step1a_module()
        phase2 = core.run_phase2_paired_spoplus_trajectory(
            train_records,
            validation_records,
            theta_init=theta_init,
            n_epochs=args.n_epochs,
            lr=args.lr,
            metric_stride=args.metric_stride,
            step1a=step1a,
            env=env,
        )
        lr_bridge = core.run_paired_lr_bridge(
            train_records,
            validation_records,
            step1a=step1a,
            env=env,
        )
        payload = serializable_degree_payload(
            args,
            artifacts,
            train_entries,
            validation_entries,
            phase2,
            lr_bridge,
        )
        payload["_phase2_rows"] = {
            "pyepo": phase2["pyepo_rows"],
            "step1c": phase2["step1c_rows"],
        }
        payload["_phase2_trajectories"] = {
            "pyepo": phase2["pyepo_trajectory"],
            "step1c": phase2["step1c_trajectory"],
        }
        return payload
    finally:
        if train_records:
            core.step1c_common.dispose_graph_records(train_records)
        if validation_records:
            core.step1c_common.dispose_graph_records(validation_records)


def write_degree_artifacts(payload):
    rows = payload.pop("_phase2_rows")
    trajectories = payload.pop("_phase2_trajectories")
    output_dir = Path(payload["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(
        output_dir / "pyepo_spoplus_loss_curve.csv",
        rows["pyepo"],
        core.PHASE2_LOSS_CURVE_FIELDS,
    )
    write_csv(
        output_dir / "step1c_spoplus_loss_curve.csv",
        rows["step1c"],
        core.PHASE2_LOSS_CURVE_FIELDS,
    )
    write_csv(
        output_dir / "trajectory_diff_summary.csv",
        payload["results"],
        ["name", "value", "threshold", "passed"],
    )
    write_json(output_dir / "trajectory_diff_summary.json", payload["results"])
    write_csv(
        output_dir / "pyepo_lr_summary.csv",
        [payload["lr"]["pyepo_summary"]],
        core.LR_SUMMARY_FIELDS,
    )
    write_csv(
        output_dir / "step1c_lr_summary.csv",
        [payload["lr"]["step1c_summary"]],
        core.LR_SUMMARY_FIELDS,
    )
    write_csv(
        output_dir / "lr_diff_summary.csv",
        payload["lr_results"],
        ["name", "value", "threshold", "passed"],
    )
    write_json(output_dir / "lr_diff_summary.json", payload["lr_results"])
    write_json(output_dir / "run_config.json", payload)
    write_json(output_dir / "train_subset.json", payload["train_subset"])
    write_json(output_dir / "validation_set.json", payload["validation_set"])
    np.save(output_dir / "trajectory_pyepo_spoplus.npy", trajectories["pyepo"])
    np.save(output_dir / "trajectory_step1c_spoplus.npy", trajectories["step1c"])
    np.save(output_dir / "theta_pyepo_lr.npy", np.asarray(payload["lr"]["pyepo_theta"]))
    np.save(output_dir / "theta_step1c_lr.npy", np.asarray(payload["lr"]["step1c_theta"]))


def aggregate_bridge_payload(degree_payloads, args):
    degree_payloads = list(degree_payloads)
    return {
        "phase": "step2b_bridge",
        "purpose": "Step2b degree PyEPO-vs-Step1c LR and SPO+ bridge",
        "methods": ["lr", "spoplus"],
        "degrees": [int(payload["degree"]) for payload in degree_payloads],
        "source_train_size": int(args.source_train_size),
        "train_size": int(args.train_size),
        "validation_size": int(args.validation_size),
        "theta_seed": int(args.theta_seed),
        "gurobi_seed": int(args.gurobi_seed),
        "optimizer": "adam",
        "lr": float(args.lr),
        "n_epochs": int(args.n_epochs),
        "metric_stride": int(args.metric_stride),
        "output_root": str(args.output_root),
        "degree_results": degree_payloads,
        "lr_passed": all(bool(payload.get("lr_passed", False)) for payload in degree_payloads),
        "spoplus_passed": all(
            bool(payload.get("spoplus_passed", payload.get("passed", False)))
            for payload in degree_payloads
        ),
        "passed": all(
            bool(payload.get("lr_passed", False))
            and bool(payload.get("spoplus_passed", payload.get("passed", False)))
            for payload in degree_payloads
        ),
    }


def print_degree_summary(payload):
    status = "PASS" if payload["passed"] else "FAIL"
    print("{} degree {} {}".format(payload["regime"], payload["degree"], status))
    if "lr_results" in payload:
        print("  LR bridge:")
        for item in payload["lr_results"]:
            item_status = "PASS" if item["passed"] else "FAIL"
            print(
                "    {:<52} {:>12.4e} <= {:<10.4e} {}".format(
                    item["name"],
                    item["value"],
                    item["threshold"],
                    item_status,
                )
            )
    print("  SPO+ bridge:")
    for item in payload["results"]:
        item_status = "PASS" if item["passed"] else "FAIL"
        print(
            "    {:<52} {:>12.4e} <= {:<10.4e} {}".format(
                item["name"],
                item["value"],
                item["threshold"],
                item_status,
            )
        )


def main(argv=None):
    import gurobipy as gp

    args = build_parser().parse_args(argv)
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Seed", args.gurobi_seed)
    env.start()
    degree_payloads = []
    try:
        for degree in args.degrees:
            artifacts = step2b_artifacts_for_degree(
                degree=degree,
                source_train_size=args.source_train_size,
                step2b_root=args.step2b_root,
                formal_run=args.formal_run,
            )
            payload = run_degree_bridge(args, artifacts, env)
            write_degree_artifacts(payload)
            print_degree_summary(payload)
            degree_payloads.append(payload)
    finally:
        env.dispose()

    aggregate = aggregate_bridge_payload(degree_payloads, args)
    if not args.no_json:
        write_json(args.json, aggregate)
    print("Step2b bridge passed: {}".format(aggregate["passed"]))
    if not args.no_strict and not aggregate["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
