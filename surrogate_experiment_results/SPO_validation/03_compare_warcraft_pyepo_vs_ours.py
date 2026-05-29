#!/usr/bin/env python3
"""Run the Warcraft PyEPO SPO+ vs local SPO+ comparison."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import warcraft_level2_common as common  # noqa: E402


OUTPUT_FIELDS = common.OUTPUT_FIELDS


def build_arg_parser():
    return common.build_arg_parser(
        "Compare PyEPO SPO+ against the local Step1c-core SPO+ adapter on Warcraft."
    )


def _load_metrics(output_dir: str):
    with (Path(output_dir) / "metrics.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_comparison_summary(output_root: Path, rows) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "level2_warcraft_comparison_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in OUTPUT_FIELDS})
    return path


def run_comparison(config: common.Level2Config) -> Path:
    pyepo_result = common.train_warcraft_spoplus(
        config=config,
        method="warcraft_pyepo_spoplus_reference",
        loss_builder=common.build_pyepo_spoplus,
        output_subdir="level2_pyepo_reference",
    )
    our_result = common.train_warcraft_spoplus(
        config=config,
        method="warcraft_step1c_core_spoplus",
        loss_builder=common.build_our_spoplus,
        output_subdir="level2_our_spoplus",
    )
    rows = [_load_metrics(pyepo_result["output_dir"]), _load_metrics(our_result["output_dir"])]
    return _write_comparison_summary(Path(config.output_root), rows)


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = common.config_from_args(args)
    summary_path = run_comparison(config)
    print(f"Saved Warcraft comparison summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
