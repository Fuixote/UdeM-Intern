import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from scripts import benchmark_hybrid_model_reuse_on_dataset as bench


class HybridDatasetBenchmarkTest(unittest.TestCase):
    def test_random_graph_has_fixed_nodes_and_random_edges(self):
        graph = bench.random_graph(num_nodes=50, num_edges=80, num_ndds=5, seed=3)

        self.assertEqual(graph["num_nodes"], 50)
        self.assertEqual(graph["edge_index"].shape, (2, 80))
        self.assertEqual(graph["is_ndd"].shape, (50,))
        self.assertEqual(graph["is_ndd"].sum(), 5)
        self.assertTrue((graph["edge_index"][1] >= 5).all())

    def test_random_integer_weights_are_between_0_and_100(self):
        weights = bench.random_weights(num_vectors=4, num_edges=3, seed=7)

        self.assertEqual(weights.shape, (4, 3))
        self.assertTrue(np.issubdtype(weights.dtype, np.integer))
        self.assertGreaterEqual(weights.min(), 0)
        self.assertLessEqual(weights.max(), 100)

    def test_benchmark_graph_rebuilds_each_time_and_reuses_once(self):
        graph = {"edge_index": "edges", "is_ndd": "ndd", "num_nodes": 3, "cycles": []}
        weights = np.array([[1, 2], [3, 4]], dtype=np.int32)
        cached = mock.Mock()
        cached.solve.side_effect = [{"objective": 1.0}, {"objective": 2.0}]

        with (
            mock.patch.object(bench, "solve_cf_cycle_pief_chain", return_value={"objective": 1.0}) as rebuild,
            mock.patch.object(bench, "CachedHybridKepModel", return_value=cached) as cached_cls,
        ):
            result = bench.benchmark_graph(graph, weights, max_chain=4, env="env")

        self.assertEqual(rebuild.call_count, 2)
        cached_cls.assert_called_once()
        self.assertEqual(cached.solve.call_count, 2)
        cached.dispose.assert_called_once()
        self.assertIn("rebuild_seconds", result)
        self.assertIn("reuse_seconds", result)
        self.assertIn("speedup", result)


if __name__ == "__main__":
    unittest.main()
