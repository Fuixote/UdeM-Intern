#!/usr/bin/env python3
"""Experiment-local wrapper for predicted top-M solution enumeration."""

from __future__ import annotations

import runpy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TARGET = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "decision_analysis"
    / "scripts"
    / "compute_second_best_solutions.py"
)


if __name__ == "__main__":
    runpy.run_path(str(TARGET), run_name="__main__")
