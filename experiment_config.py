import os
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_path(path_like, base_dir=PROJECT_ROOT):
    path = Path(path_like)
    if not path.is_absolute():
        path = Path(base_dir) / path
    return path.resolve()


def env_path(env_name, default_relative_path):
    raw_value = os.environ.get(env_name)
    if raw_value:
        return resolve_path(raw_value)
    return resolve_path(default_relative_path)


RAW_DATA_DIR = env_path("KEP_RAW_DATA_DIR", "dataset/raw")
PROCESSED_DATA_DIR = env_path("KEP_DATA_DIR", "dataset/processed")
RESULTS_ROOT = env_path("KEP_RESULTS_DIR", "results")
SOLUTIONS_ROOT = env_path("KEP_SOLUTIONS_DIR", "solutions")


def timestamp_now():
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def make_results_dir(prefix, timestamp=None, results_root=RESULTS_ROOT):
    ts = timestamp or timestamp_now()
    result_dir = resolve_path(Path(results_root) / f"{prefix}{ts}")
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


def solution_dir_for_experiment(experiment_name, solutions_root=SOLUTIONS_ROOT):
    solution_dir = resolve_path(Path(solutions_root) / experiment_name)
    solution_dir.mkdir(parents=True, exist_ok=True)
    return solution_dir


def solution_dir_for_result_dir(result_dir, solutions_root=SOLUTIONS_ROOT):
    return solution_dir_for_experiment(Path(result_dir).name, solutions_root=solutions_root)


def latest_result_dir(prefix, results_root=RESULTS_ROOT):
    root = resolve_path(results_root)
    if not root.exists():
        return None
    matches = sorted(
        [path for path in root.iterdir() if path.is_dir() and path.name.startswith(prefix)],
        key=lambda path: path.name,
        reverse=True,
    )
    return matches[0] if matches else None


def latest_checkpoint(prefix, filename, results_root=RESULTS_ROOT):
    latest_dir = latest_result_dir(prefix, results_root=results_root)
    if latest_dir is None:
        return None
    checkpoint_path = latest_dir / filename
    if checkpoint_path.exists():
        return checkpoint_path
    return None
