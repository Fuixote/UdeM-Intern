"""Step1c run directory and config helpers."""

from __future__ import annotations

import json
from pathlib import Path


def default_run_dir(
    output_root,
    train_size,
    split_seed,
    subset_seed,
    theta_seed,
    fy_epsilon,
    fy_M,
    e2e_epochs,
    metric_stride,
    early_stop_metric=None,
    early_stop_patience=None,
    early_stop_min_delta=None,
):
    early_stop_suffix = ""
    if early_stop_metric:
        early_stop_suffix = (
            f"_earlystop={early_stop_metric}"
            f"_patience={early_stop_patience}"
            f"_mindelta={early_stop_min_delta}"
        )
    return str(
        Path(output_root)
        / f"train_size={train_size}"
        / f"split_seed={split_seed}"
        / f"subset_seed={subset_seed}"
        / f"theta_seed={theta_seed}"
        / (
            f"eps={fy_epsilon}_M={fy_M}_e2e_epochs={e2e_epochs}"
            f"_stride={metric_stride}{early_stop_suffix}"
        )
    )


def write_run_config(path, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
