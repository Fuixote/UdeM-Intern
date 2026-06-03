#!/usr/bin/env python3
"""Run Phase 2 KEP full trajectory equivalence against PyEPO SPOPlus."""

from __future__ import annotations

import argparse
import csv
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

from surrogate_experiment_results.Step1c import split_dataset


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=core.DEFAULT_DATA_DIR)
    parser.add_argument("--split-path", type=Path, default=core.DEFAULT_SPLIT_PATH)
    parser.add_argument("--output-dir", type=Path, default=core.DEFAULT_FULL_TRAJECTORY_OUTPUT_DIR)
    parser.add_argument("--train-size", type=int, default=5)
    parser.add_argument("--validation-size", type=int, default=5)
    parser.add_argument("--subset-seed", type=int, default=42)
    parser.add_argument("--theta-seed", type=int, default=42)
    parser.add_argument("--theta-init", type=float, nargs=2, default=None)
    parser.add_argument("--gurobi-seed", type=int, default=42)
    parser.add_argument("--optimizer", choices=["adam"], default="adam")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--n-epochs", type=int, default=3)
    parser.add_argument("--metric-stride", type=int, default=1)
    parser.add_argument("--json", type=Path, default=core.DEFAULT_FULL_TRAJECTORY_JSON_PATH)
    parser.add_argument("--no-json", action="store_true")
    parser.add_argument("--no-strict", action="store_true")
    return parser


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def graph_entries_from_phase2_args(args):
    if args.split_path.exists():
        split = split_dataset.read_json(args.split_path)
        train_entries = split_dataset.select_train_subset(
            split["train_pool"],
            train_size=args.train_size,
            seed=args.subset_seed,
        )
        validation_entries = split["validation"][: args.validation_size]
        return train_entries, validation_entries

    entries = split_dataset.graph_entries_from_data_dir(args.data_dir)
    required = args.train_size + args.validation_size
    if len(entries) < required:
        raise FileNotFoundError(
            "Need at least {} G-*.json files under {}, found {}".format(
                required,
                args.data_dir,
                len(entries),
            )
        )
    return entries[: args.train_size], entries[args.train_size : required]


def serializable_phase2_payload(args, train_entries, validation_entries, phase2):
    output_dir = Path(args.output_dir)
    return {
        "phase": "phase2",
        "purpose": "KEP full trajectory equivalence",
        "data_dir": str(args.data_dir),
        "split_path": str(args.split_path),
        "output_dir": str(output_dir),
        "train_size": int(args.train_size),
        "validation_size": int(args.validation_size),
        "subset_seed": int(args.subset_seed),
        "theta_seed": int(args.theta_seed),
        "theta_init": np.asarray(phase2["pyepo_trajectory"][0], dtype=float).tolist(),
        "gurobi_seed": int(args.gurobi_seed),
        "optimizer": args.optimizer,
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
        },
        "results": phase2["summary"]["results"],
        "passed": bool(phase2["summary"]["passed"]),
    }


def run_full_trajectory(args):
    import gurobipy as gp

    train_entries, validation_entries = graph_entries_from_phase2_args(args)
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Seed", args.gurobi_seed)
    env.start()

    train_records = []
    validation_records = []
    try:
        print("Loading Phase 2 train subset: n={}".format(len(train_entries)))
        train_records = core.step1c_common.load_graph_records(
            [entry["path"] for entry in train_entries],
            env,
        )
        print("Loading Phase 2 validation set: n={}".format(len(validation_entries)))
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
        print("Phase 2 theta_init={}".format(np.round(theta_init, 6)))
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
        payload = serializable_phase2_payload(
            args,
            train_entries,
            validation_entries,
            phase2,
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
        env.dispose()


def write_phase2_artifacts(args, payload):
    output_dir = Path(args.output_dir)
    rows = payload.pop("_phase2_rows")
    trajectories = payload.pop("_phase2_trajectories")

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
    write_json(output_dir / "run_config.json", payload)
    write_json(output_dir / "train_subset.json", payload["train_subset"])
    write_json(output_dir / "validation_set.json", payload["validation_set"])
    np.save(output_dir / "trajectory_pyepo_spoplus.npy", trajectories["pyepo"])
    np.save(output_dir / "trajectory_step1c_spoplus.npy", trajectories["step1c"])
    if not args.no_json:
        write_json(args.json, payload)


def print_payload(payload):
    print("KEP-vs-PyEPO Phase 2 full trajectory equivalence")
    print("output_dir: {}".format(payload["output_dir"]))
    print("train graphs: {}".format(len(payload["train_graphs"])))
    print("validation graphs: {}".format(len(payload["validation_graphs"])))
    print("epochs: {} stride: {}".format(payload["n_epochs"], payload["metric_stride"]))
    print()
    for item in payload["results"]:
        status = "PASS" if item["passed"] else "FAIL"
        print(
            "{:<52} {:>12.4e} <= {:<10.4e} {}".format(
                item["name"],
                item["value"],
                item["threshold"],
                status,
            )
        )


def main(argv=None):
    args = build_parser().parse_args(argv)
    payload = run_full_trajectory(args)
    write_phase2_artifacts(args, payload)
    print_payload(payload)
    if not args.no_strict and not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
