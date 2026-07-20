#!/usr/bin/env python3
"""Shared constants and import helpers for Experiment 04."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
STEP5_ROOT = EXPERIMENT_ROOT.parent
EXP1_ROOT = STEP5_ROOT / "experiment_01_weak_label_seed42_sample50"
EXP3_ROOT = STEP5_ROOT / "experiment_03_formal_continuous_label_seed42_sample50"

DEFAULT_SELECTED_TOPOLOGIES = EXPERIMENT_ROOT / "configs" / "topologies.repeat60.csv"
DEFAULT_FORMAL_SUMMARY = (
    EXP3_ROOT / "results" / "formal1000" / "results" / "weak_label_topology_summary.csv"
)
DEFAULT_FORMAL_OUTPUT_ROOT = EXP3_ROOT / "results" / "formal1000"
DEFAULT_OUTPUT_ROOT = EXPERIMENT_ROOT / "results" / "repeat_seed120"
DEFAULT_GENERATOR_CONFIG = EXP1_ROOT / "configs" / "context_generator.locked.yaml"
DEFAULT_REGIME = "step2c_poly_d8_mult_eps050"
DEFAULT_PROTOCOL = "screen"
REFERENCE_SEED = 42
TRAIN_SEEDS = (43, 44)
ALL_LABEL_SEEDS = (42, 43, 44)
SAMPLE_SIZE = 50
TRAINING_SIZE = 40
VALIDATION_SIZE = 10
TEST_SIZE = 1000
MASTER_LABEL_SEED = 20260719
MAX_EPOCHS = 10000
METRIC_STRIDE = 1
EARLY_STOP_PATIENCE = 20
EARLY_STOP_MIN_DELTA = 0.0001
THETA_SEED = 42
GUROBI_SEED = 42
EXPERIMENT_VERSION = "step5_exp4_repeat_seed_stability_sample50_v1"


def import_script(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def base_builder() -> Any:
    return import_script("step5_exp1_builder", EXP1_ROOT / "scripts" / "build_weak_label_artifacts.py")


def base_planner() -> Any:
    return import_script("step5_exp1_planner", EXP1_ROOT / "scripts" / "plan_weak_label_jobs.py")


def base_reviewer() -> Any:
    return import_script("step5_exp1_reviewer", EXP1_ROOT / "scripts" / "review_weak_label_results.py")


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def project_relative(path: str | Path) -> str:
    candidate = Path(path).resolve()
    try:
        return str(candidate.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(candidate)


def reference_test_paths(topology_id: str, *, formal_output_root: str | Path = DEFAULT_FORMAL_OUTPUT_ROOT) -> tuple[Path, Path]:
    test_dir = Path(formal_output_root) / "data" / DEFAULT_REGIME / topology_id / "test"
    return test_dir / "test.npz", test_dir / "test_manifest.json"
