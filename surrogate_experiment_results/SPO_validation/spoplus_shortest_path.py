"""Auditable SPO+ formulas on a tiny deterministic shortest-path oracle.

This module is intentionally small and dependency-light.  It validates the
mathematics used by the KEP SPO+ experiments on a standard cost-minimization
shortest-path problem before touching any large Warcraft or KEP training runs.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


GridShape = Tuple[int, int]
Edge = Tuple[Tuple[int, int], Tuple[int, int]]


def _validate_grid_shape(grid_shape: Sequence[int]) -> GridShape:
    if len(grid_shape) != 2:
        raise ValueError("grid_shape must be a pair: (n_rows, n_cols)")
    n_rows, n_cols = int(grid_shape[0]), int(grid_shape[1])
    if n_rows <= 0 or n_cols <= 0:
        raise ValueError("grid_shape entries must be positive")
    return n_rows, n_cols


def grid_edges(grid_shape: Sequence[int]) -> List[Edge]:
    """Return the fixed edge order for a right/down rectangular grid DAG.

    Edges are ordered by row-major source node.  For each node, the right edge
    is emitted before the down edge.  This order is part of the validation
    contract because tie-breaking returns the first minimum-cost path induced by
    this right-before-down dynamic program.
    """

    n_rows, n_cols = _validate_grid_shape(grid_shape)
    edges: List[Edge] = []
    for row in range(n_rows):
        for col in range(n_cols):
            if col + 1 < n_cols:
                edges.append(((row, col), (row, col + 1)))
            if row + 1 < n_rows:
                edges.append(((row, col), (row + 1, col)))
    return edges


def _as_cost_vector(values: Sequence[float], grid_shape: GridShape, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float).reshape(-1)
    expected = len(grid_edges(grid_shape))
    if vector.shape != (expected,):
        raise ValueError(f"{name} must have shape ({expected},), got {vector.shape}")
    return vector


def solve_shortest_path(
    cost_vector: Sequence[float], grid_shape: Sequence[int]
) -> Tuple[np.ndarray, float]:
    """Solve a monotone grid shortest-path problem without Gurobi.

    The feasible set contains paths from the top-left node to the bottom-right
    node using only right and down moves.  Because this is a DAG, arbitrary real
    edge costs are safe, including negative SPO+ shifted costs.

    Tie-breaking is deterministic: at each node, if right and down moves have
    equal total cost-to-go, the right edge is chosen.  This matches the fixed
    edge order from :func:`grid_edges` and keeps tests reproducible.
    """

    shape = _validate_grid_shape(grid_shape)
    n_rows, n_cols = shape
    costs = _as_cost_vector(cost_vector, shape, "cost_vector")
    edges = grid_edges(shape)
    edge_to_index: Dict[Edge, int] = {edge: idx for idx, edge in enumerate(edges)}

    # best_cost[row, col] stores the minimum remaining cost from this node to
    # the sink.  next_edge stores the chosen outgoing edge index for recovery.
    best_cost = np.full(shape, np.inf, dtype=float)
    next_edge = np.full(shape, -1, dtype=int)
    best_cost[n_rows - 1, n_cols - 1] = 0.0

    for row in range(n_rows - 1, -1, -1):
        for col in range(n_cols - 1, -1, -1):
            if row == n_rows - 1 and col == n_cols - 1:
                continue

            best_value = np.inf
            best_index = -1

            if col + 1 < n_cols:
                edge = ((row, col), (row, col + 1))
                edge_index = edge_to_index[edge]
                value = costs[edge_index] + best_cost[row, col + 1]
                best_value = value
                best_index = edge_index

            if row + 1 < n_rows:
                edge = ((row, col), (row + 1, col))
                edge_index = edge_to_index[edge]
                value = costs[edge_index] + best_cost[row + 1, col]
                # Strict comparison preserves right-before-down ties, because
                # the right option is visited first when it exists.
                if value < best_value:
                    best_value = value
                    best_index = edge_index

            best_cost[row, col] = best_value
            next_edge[row, col] = best_index

    path = np.zeros(len(edges), dtype=float)
    row = col = 0
    while not (row == n_rows - 1 and col == n_cols - 1):
        edge_index = int(next_edge[row, col])
        if edge_index < 0:
            raise RuntimeError("failed to recover a top-left to bottom-right path")
        path[edge_index] = 1.0
        (_, _), (row, col) = edges[edge_index]

    objective = float(np.dot(costs, path))
    return path, objective


def spo_plus_min_loss_and_grad(
    c_hat: Sequence[float], c: Sequence[float], grid_shape: Sequence[int]
) -> Tuple[float, np.ndarray]:
    """Return cost-minimization SPO+ loss and a subgradient.

    For a minimization oracle

        z*(c) in argmin_z c^T z,

    the SPO+ surrogate is

        L_min(c_hat, c)
          = max_z (c - 2 c_hat)^T z
            + 2 c_hat^T z*(c)
            - c^T z*(c).

    The max oracle is evaluated through the equivalent minimization

        argmax_z (c - 2 c_hat)^T z
        = argmin_z (2 c_hat - c)^T z.

    Therefore one valid subgradient with respect to c_hat is

        2 * (z*(c) - z*(2 c_hat - c)).
    """

    shape = _validate_grid_shape(grid_shape)
    c_hat_vec = _as_cost_vector(c_hat, shape, "c_hat")
    c_vec = _as_cost_vector(c, shape, "c")

    z_true, true_obj = solve_shortest_path(c_vec, shape)
    shifted_cost = 2.0 * c_hat_vec - c_vec
    z_shifted, _ = solve_shortest_path(shifted_cost, shape)

    loss = (
        float(np.dot(c_vec - 2.0 * c_hat_vec, z_shifted))
        + 2.0 * float(np.dot(c_hat_vec, z_true))
        - true_obj
    )
    grad = 2.0 * (z_true - z_shifted)
    return float(loss), np.asarray(grad, dtype=float)


def _solve_max_path(weights: np.ndarray, grid_shape: GridShape) -> Tuple[np.ndarray, float]:
    path, neg_objective = solve_shortest_path(-weights, grid_shape)
    return path, -float(neg_objective)


def spo_plus_max_loss_and_grad(
    w_hat: Sequence[float], w: Sequence[float], grid_shape: Sequence[int]
) -> Tuple[float, np.ndarray]:
    """Return reward-maximization SPO+ loss and a subgradient.

    KEP is a reward maximization problem:

        y*(w) in argmax_y w^T y.

    With the sign conversion c = -w and c_hat = -w_hat, the cost-min SPO+
    formula becomes

        L_max(w_hat, w)
          = max_y (2 w_hat - w)^T y
            - 2 w_hat^T y*(w)
            + w^T y*(w),

    with subgradient

        2 * (y*(2 w_hat - w) - y*(w)).

    Chain rule check:

        L_max(w_hat, w) = L_min(-w_hat, -w)
        grad_w_hat L_max = - grad_c_hat L_min.
    """

    shape = _validate_grid_shape(grid_shape)
    w_hat_vec = _as_cost_vector(w_hat, shape, "w_hat")
    w_vec = _as_cost_vector(w, shape, "w")

    y_true, true_obj = _solve_max_path(w_vec, shape)
    shifted_w = 2.0 * w_hat_vec - w_vec
    y_shifted, _ = _solve_max_path(shifted_w, shape)

    loss = (
        float(np.dot(shifted_w, y_shifted))
        - 2.0 * float(np.dot(w_hat_vec, y_true))
        + true_obj
    )
    grad = 2.0 * (y_shifted - y_true)
    return float(loss), np.asarray(grad, dtype=float)
