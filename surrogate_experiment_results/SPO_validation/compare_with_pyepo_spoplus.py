#!/usr/bin/env python3
"""Required PyEPO comparison for the Level-1 SPO+ toy oracle.

This script compares the toy cost-min SPO+ loss/gradient against
``pyepo.func.SPOPlus``.  Missing dependencies, Gurobi license/network failures,
and numerical mismatches are intentional hard failures.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from spoplus_shortest_path import (  # noqa: E402
    grid_edges,
    solve_shortest_path,
    spo_plus_min_loss_and_grad,
)


def _build_pyepo_model_class():
    import gurobipy as gp
    from gurobipy import GRB
    from pyepo.model.grb.grbmodel import optGrbModel

    class MonotoneGridShortestPathModel(optGrbModel):
        """PyEPO/Gurobi model using the same edge order as the toy oracle."""

        def __init__(self, grid_shape):
            self.grid_shape = tuple(grid_shape)
            self.edges = grid_edges(self.grid_shape)
            super().__init__()

        def _getModel(self):
            model = gp.Model("monotone_grid_shortest_path")
            model.Params.OutputFlag = 0
            x = model.addVars(self.edges, lb=0.0, ub=1.0, name="x")
            model.modelSense = GRB.MINIMIZE

            n_rows, n_cols = self.grid_shape
            for row in range(n_rows):
                for col in range(n_cols):
                    node = (row, col)
                    flow = 0.0
                    for source, target in self.edges:
                        if target == node:
                            flow += x[source, target]
                        elif source == node:
                            flow -= x[source, target]
                    if node == (0, 0):
                        model.addConstr(flow == -1.0)
                    elif node == (n_rows - 1, n_cols - 1):
                        model.addConstr(flow == 1.0)
                    else:
                        model.addConstr(flow == 0.0)
            return model, x

        def setObj(self, c):
            c = np.asarray(c, dtype=float).reshape(-1)
            if c.shape != (len(self.edges),):
                raise ValueError(f"expected {len(self.edges)} costs, got {c.shape}")
            obj = gp.quicksum(
                float(c[idx]) * self.x[source, target]
                for idx, (source, target) in enumerate(self.edges)
            )
            self._model.setObjective(obj)

        def solve(self):
            self._model.update()
            self._model.optimize()
            sol = np.array(
                [
                    float(self.x[source, target].x)
                    for source, target in self.edges
                ],
                dtype=float,
            )
            return sol, float(self._model.objVal)

    return MonotoneGridShortestPathModel


def compare_with_pyepo() -> int:
    import torch
    import pyepo

    grid_shape = (3, 3)
    c = np.array(
        [
            1.48234593,
            -0.56930333,
            -0.86574273,
            0.75657385,
            1.59734485,
            0.11553230,
            2.90382099,
            1.42414869,
            0.40465951,
            -0.03941241,
            -0.28410992,
            1.64524854,
        ]
    )
    c_hat = np.array(
        [
            -0.36856653,
            -2.64193262,
            -0.61173447,
            1.42797243,
            -1.90504962,
            -1.94728946,
            0.18930824,
            0.19096552,
            0.80640575,
            2.09659076,
            1.34673195,
            0.66614106,
        ]
    )

    ours_loss, ours_grad = spo_plus_min_loss_and_grad(c_hat, c, grid_shape)
    z_true, true_obj = solve_shortest_path(c, grid_shape)

    Model = _build_pyepo_model_class()
    optmodel = Model(grid_shape)
    spoplus = pyepo.func.SPOPlus(optmodel, processes=1)

    cp = torch.tensor(c_hat[None, :], dtype=torch.float32, requires_grad=True)
    c_tensor = torch.tensor(c[None, :], dtype=torch.float32)
    w_tensor = torch.tensor(z_true[None, :], dtype=torch.float32)
    z_tensor = torch.tensor([[true_obj]], dtype=torch.float32)

    loss = spoplus(cp, c_tensor, w_tensor, z_tensor).mean()
    loss.backward()
    pyepo_loss = float(loss.detach().cpu().item())
    pyepo_grad = cp.grad.detach().cpu().numpy().reshape(-1)

    np.testing.assert_allclose(pyepo_loss, ours_loss, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(pyepo_grad, ours_grad, atol=1e-5, rtol=1e-5)

    print("PyEPO SPO+ comparison passed.")
    print(f"loss={ours_loss:.8f}")
    print(f"grad_max_abs_diff={np.max(np.abs(pyepo_grad - ours_grad)):.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(compare_with_pyepo())
