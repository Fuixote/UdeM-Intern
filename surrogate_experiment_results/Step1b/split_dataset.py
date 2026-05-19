"""Create and reuse the fixed Step1b train/validation/test master split."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "dataset" / "processed" / "step1_noisy_linear_sigma010"
DEFAULT_SPLIT_PATH = PROJECT_ROOT / "results" / "step1b_splits" / "master_split_seed42.json"


def graph_id(path):
    stem = Path(path).stem
    try:
        return int(stem.split("-", 1)[1])
    except (IndexError, ValueError):
        return stem


def graph_entry(path):
    gid = graph_id(path)
    return {
        "index": gid,
        "graph_id": gid,
        "path": str(Path(path)),
    }


def list_graph_files(data_dir):
    paths = list(Path(data_dir).glob("G-*.json"))
    if not paths:
        raise FileNotFoundError(f"No G-*.json files found in {data_dir}")
    return sorted(paths, key=graph_id)


def graph_entries_from_data_dir(data_dir):
    return [graph_entry(path) for path in list_graph_files(data_dir)]


def make_master_split(files, train_pool_size, val_size, test_size, seed):
    total_needed = train_pool_size + val_size + test_size
    if len(files) < total_needed:
        raise ValueError(f"Need {total_needed} graphs for split, found {len(files)}")

    shuffled = [Path(path) for path in files]
    random.Random(seed).shuffle(shuffled)
    train_end = train_pool_size
    val_end = train_end + val_size
    return {
        "seed": int(seed),
        "train_pool_size": int(train_pool_size),
        "validation_size": int(val_size),
        "test_size": int(test_size),
        "train_pool": [graph_entry(path) for path in shuffled[:train_end]],
        "validation": [graph_entry(path) for path in shuffled[train_end:val_end]],
        "test": [graph_entry(path) for path in shuffled[val_end:total_needed]],
    }


def select_train_subset(train_pool, train_size, seed):
    if train_size <= 0:
        raise ValueError("train_size must be positive")
    if train_size > len(train_pool):
        raise ValueError(
            f"train_size={train_size} exceeds train pool size {len(train_pool)}"
        )

    shuffled = list(train_pool)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:train_size]


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Create the fixed Step1b split.")
    parser.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--split_path", default=str(DEFAULT_SPLIT_PATH))
    parser.add_argument("--train_pool_size", type=int, default=1200)
    parser.add_argument("--val_size", type=int, default=400)
    parser.add_argument("--test_size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reuse_if_exists",
        action="store_true",
        help="Do not rewrite an existing split file.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    split_path = Path(args.split_path)
    if args.reuse_if_exists and split_path.exists():
        print(f"Reusing existing split: {split_path}")
        return

    split = make_master_split(
        list_graph_files(args.data_dir),
        train_pool_size=args.train_pool_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    write_json(split_path, split)
    print(
        f"Saved split to {split_path} "
        f"(train_pool={len(split['train_pool'])}, "
        f"validation={len(split['validation'])}, test={len(split['test'])})"
    )


if __name__ == "__main__":
    main()
