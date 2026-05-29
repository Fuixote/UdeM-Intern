#!/usr/bin/env python3
"""README Step B: verify cost-min and reward-max SPO+ sign conversion."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from spoplus_shortest_path import (  # noqa: E402
    grid_edges,
    spo_plus_max_loss_and_grad,
    spo_plus_min_loss_and_grad,
)


def main() -> int:
    grid_shape = (4, 3)
    rng = np.random.RandomState(11)
    n_edges = len(grid_edges(grid_shape))

    for _ in range(30):
        w = rng.uniform(-2.0, 2.0, size=n_edges)
        w_hat = rng.uniform(-2.0, 2.0, size=n_edges)

        max_loss, max_grad = spo_plus_max_loss_and_grad(w_hat, w, grid_shape)
        min_loss, min_grad = spo_plus_min_loss_and_grad(-w_hat, -w, grid_shape)

        np.testing.assert_allclose(max_loss, min_loss, atol=1e-10)
        np.testing.assert_allclose(max_grad, -min_grad, atol=1e-10)

    print("Reward-max / cost-min SPO+ sign conversion checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
