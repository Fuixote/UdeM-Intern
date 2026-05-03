import unittest
from unittest import mock

import numpy as np

try:
    from gurobipy import GRB, GurobiError
except ImportError:  # pragma: no cover - exercised only without gurobipy
    GRB = None
    GurobiError = Exception

from formulations.hybrid.backend import CachedHybridKepModel, solve_cf_cycle_pief_chain


def small_hybrid_instance():
    edge_index = np.array(
        [
            [0, 1, 2, 2, 3],
            [1, 2, 1, 3, 1],
        ],
        dtype=np.int64,
    )
    is_ndd_mask = np.array([True, False, False, False], dtype=bool)
    cycle_candidates = [
        {"type": "cycle", "nodes": [1, 2], "edges": [1, 2]},
        {"type": "cycle", "nodes": [1, 2, 3], "edges": [1, 3, 4]},
    ]
    return edge_index, is_ndd_mask, 4, cycle_candidates


class ReusableHybridSolverTest(unittest.TestCase):
    def setUp(self):
        if GRB is None:
            self.skipTest("gurobipy is not installed")

    def test_cached_hybrid_matches_rebuild_objective_over_updates(self):
        edge_index, is_ndd_mask, num_nodes, cycle_candidates = small_hybrid_instance()
        rng = np.random.default_rng(7)

        try:
            solver = CachedHybridKepModel(
                edge_index=edge_index,
                is_ndd_mask=is_ndd_mask,
                num_nodes=num_nodes,
                cycle_candidates=cycle_candidates,
                max_chain=4,
            )
        except GurobiError as exc:
            self.skipTest(f"Gurobi is unavailable: {exc}")

        try:
            for weights in rng.normal(size=(20, edge_index.shape[1])).astype(np.float32):
                rebuild = solve_cf_cycle_pief_chain(
                    weights=weights,
                    edge_index=edge_index,
                    is_ndd_mask=is_ndd_mask,
                    num_nodes=num_nodes,
                    cycle_candidates=cycle_candidates,
                    max_chain=4,
                )
                cached = solver.solve(
                    weights,
                    time_limit=None,
                    reset_before_solve=True,
                )

                self.assertIn(cached["status"], (GRB.OPTIMAL, GRB.TIME_LIMIT))
                self.assertAlmostEqual(rebuild["objective"], cached["objective"], places=6)

                if not np.array_equal(rebuild["edge_selection"], cached["edge_selection"]):
                    rebuild_value = float(np.dot(weights, rebuild["edge_selection"]))
                    cached_value = float(np.dot(weights, cached["edge_selection"]))
                    self.assertAlmostEqual(rebuild_value, cached_value, places=6)
        finally:
            solver.dispose()

    def test_cached_hybrid_updates_time_limit_per_solve(self):
        edge_index, is_ndd_mask, num_nodes, cycle_candidates = small_hybrid_instance()

        try:
            solver = CachedHybridKepModel(
                edge_index=edge_index,
                is_ndd_mask=is_ndd_mask,
                num_nodes=num_nodes,
                cycle_candidates=cycle_candidates,
            )
        except GurobiError as exc:
            self.skipTest(f"Gurobi is unavailable: {exc}")

        try:
            result = solver.solve(
                np.array([1.0, 2.0, 1.0, 1.5, 0.5], dtype=np.float32),
                time_limit=3.0,
                reset_before_solve=True,
            )
            self.assertIn(result["status"], (GRB.OPTIMAL, GRB.TIME_LIMIT))
            self.assertAlmostEqual(solver.model.Params.TimeLimit, 3.0, places=6)
        finally:
            solver.dispose()


class ObjectiveUpdateImplementationTest(unittest.TestCase):
    def test_cached_hybrid_uses_batched_objective_update(self):
        class Var:
            def __init__(self):
                self.X = 0.0

        solver = object.__new__(CachedHybridKepModel)
        solver.cycle_candidates = [
            {"type": "cycle", "nodes": [1, 2], "edges": [1, 2]},
        ]
        solver.valid_chain_keys = [(0, 1), (3, 2)]
        solver.num_edges = 4
        solver.max_chain = 4
        solver.src = np.array([0, 1, 2, 2], dtype=np.int64)
        solver.dst = np.array([1, 2, 1, 3], dtype=np.int64)
        solver.cycle_vars = {0: Var()}
        solver.chain_vars = {
            (0, 1): Var(),
            (3, 2): Var(),
        }
        solver.obj_vars = [solver.cycle_vars[0], solver.chain_vars[(0, 1)], solver.chain_vars[(3, 2)]]
        solver.obj_edge_groups = [(1, 2), (0,), (3,)]
        solver.default_time_limit = None
        solver.model = mock.Mock()
        solver.model.status = GRB.OPTIMAL if GRB is not None else 2
        solver.model.SolCount = 1
        solver.model.ObjVal = 0.0

        solver.solve(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))

        solver.model.setAttr.assert_called_once()
        attr_name, variables, values = solver.model.setAttr.call_args.args
        self.assertEqual(attr_name, GRB.Attr.Obj if GRB is not None else "Obj")
        self.assertEqual(variables, [solver.cycle_vars[0], solver.chain_vars[(0, 1)], solver.chain_vars[(3, 2)]])
        self.assertEqual(values, [5.0, 1.0, 4.0])
        solver.model.update.assert_called()
        for var in variables:
            self.assertFalse(hasattr(var, "Obj"))


if __name__ == "__main__":
    unittest.main()
