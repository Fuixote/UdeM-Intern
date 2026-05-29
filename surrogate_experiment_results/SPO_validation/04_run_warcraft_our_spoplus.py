#!/usr/bin/env python3
"""Level 2: run the Warcraft experiment with the local SPO+ wrapper."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import warcraft_level2_common as common  # noqa: E402


OUTPUT_FIELDS = common.OUTPUT_FIELDS


def build_arg_parser():
    return common.build_arg_parser(
        "Run the Level-2 Warcraft reproduction with the local SPO+ wrapper."
    )


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = common.config_from_args(args)
    result = common.train_warcraft_spoplus(
        config=config,
        method="warcraft_our_spoplus",
        loss_builder=common.build_our_spoplus,
        output_subdir="level2_our_spoplus",
    )
    print(f"Saved Level-2 local SPO+ outputs to {result['output_dir']}")
    print(f"final_normalized_regret={result['final_normalized_regret']:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
