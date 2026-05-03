import unittest
from unittest import mock

import numpy as np
import torch

from formulations.hybrid import end2end_gnn


class HybridGnnSolverCacheTest(unittest.TestCase):
    def test_blackbox_uses_cached_solver_when_provided(self):
        cached_solver = mock.Mock()
        cached_solver.solve.return_value = {
            "status": end2end_gnn.GRB.OPTIMAL,
            "edge_selection": np.array([1.0, 0.0], dtype=np.float32),
        }

        selection = end2end_gnn.blackbox_kep_solver(
            np.array([2.0, 3.0], dtype=np.float32),
            cycle_candidates=[],
            edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
            node_is_ndd=np.array([False, False]),
            num_edges=2,
            num_nodes=2,
            cached_solver=cached_solver,
        )

        np.testing.assert_array_equal(selection, np.array([1.0, 0.0], dtype=np.float32))
        cached_solver.solve.assert_called_once()

    def test_get_expected_y_passes_cached_solver_to_each_sample(self):
        cached_solver = mock.Mock()
        cached_solver.solve.side_effect = [
            {"status": end2end_gnn.GRB.OPTIMAL, "edge_selection": np.array([1.0, 0.0], dtype=np.float32)},
            {"status": end2end_gnn.GRB.OPTIMAL, "edge_selection": np.array([0.0, 1.0], dtype=np.float32)},
        ]

        y_soft = end2end_gnn.get_expected_y(
            torch.tensor([[1.0], [2.0]], dtype=torch.float32),
            cycle_candidates=[],
            edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
            node_is_ndd=np.array([False, False]),
            num_nodes=2,
            num_edges=2,
            epsilon=0.0,
            M=2,
            cached_solver=cached_solver,
        )

        self.assertEqual(cached_solver.solve.call_count, 2)
        torch.testing.assert_close(y_soft.cpu(), torch.tensor([[0.5], [0.5]], dtype=torch.float32))


if __name__ == "__main__":
    unittest.main()
