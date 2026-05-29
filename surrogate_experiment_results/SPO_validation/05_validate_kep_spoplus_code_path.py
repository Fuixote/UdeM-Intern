#!/usr/bin/env python3
"""Validate the current Step1c KEP SPO+ code path."""

from __future__ import annotations

import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from spoplus_kep_path_validation import validate_step1c_code_path  # noqa: E402


def main() -> int:
    diagnostics = validate_step1c_code_path()
    print("KEP SPO+ Step1c code-path validation passed.")
    print(json.dumps(diagnostics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
