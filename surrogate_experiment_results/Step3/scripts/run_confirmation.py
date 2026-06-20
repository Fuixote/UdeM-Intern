#!/usr/bin/env python3
"""Validate and plan Step3 formal confirmation jobs without launching them."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import audit_fixed_topology_xy  # noqa: E402
import sample_fixed_topology_context as context_sampler  # noqa: E402


def load_json_or_simple_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if config_path.suffix.lower() == ".json":
        return json.loads(config_path.read_text(encoding="utf-8"))
    return context_sampler.load_simple_yaml(config_path)


def validate_confirmation_config(
    config: dict[str, Any],
    *,
    generator_config: dict[str, Any],
) -> None:
    audit_fixed_topology_xy.assert_formal_config_locked(generator_config)
    training = config.get("training", {})
    if training.get("nested_train_sets") is False:
        raise ValueError("formal confirmation requires nested_train_sets=true")
    eval_config = config.get("evaluation", {})
    if eval_config.get("validation_namespace", "confirm_validation") != "confirm_validation":
        raise ValueError("formal validation namespace must be confirm_validation")
    if eval_config.get("test_namespace", "confirm_test") != "confirm_test":
        raise ValueError("formal test namespace must be confirm_test")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--context-generator-config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_json_or_simple_yaml(args.config)
    generator_config = context_sampler.load_generator_config(args.context_generator_config)
    validate_confirmation_config(config, generator_config=generator_config)
    if not args.dry_run:
        raise SystemExit(
            "run_confirmation.py currently validates the locked formal config and "
            "job plan only. Use run_one_job.py for explicit small jobs."
        )
    print(json.dumps({"status": "dry_run_validated"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
