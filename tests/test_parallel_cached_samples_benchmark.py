import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from scripts import benchmark_parallel_cached_samples as bench


class ParallelCachedSamplesBenchmarkTest(unittest.TestCase):
    def test_split_indexed_items_assigns_round_robin_chunks(self):
        chunks = bench.split_indexed_items(list(range(10)), num_chunks=4)

        self.assertEqual(chunks[0], [(0, 0), (4, 4), (8, 8)])
        self.assertEqual(chunks[1], [(1, 1), (5, 5), (9, 9)])
        self.assertEqual(chunks[2], [(2, 2), (6, 6)])
        self.assertEqual(chunks[3], [(3, 3), (7, 7)])

    def test_compare_results_tracks_objective_and_selection_mismatches(self):
        serial = [
            {"objective": 1.0, "edge_selection": np.array([1, 0, 0])},
            {"objective": 2.0, "edge_selection": np.array([0, 1, 0])},
        ]
        parallel = [
            {"objective": 1.0, "edge_selection": np.array([1, 0, 0])},
            {"objective": 2.5, "edge_selection": np.array([0, 0, 1])},
        ]

        comparison = bench.compare_results(serial, parallel, tolerance=1e-6)

        self.assertEqual(comparison["max_obj_diff"], 0.5)
        self.assertEqual(comparison["obj_mismatches"], 1)
        self.assertEqual(comparison["selection_mismatches"], 1)

    def test_select_graph_paths_defaults_to_multiple_graphs_from_start_index(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ["G-0.json", "G-1.json", "G-2.json", "G-3.json"]:
                (root / name).write_text("{}", encoding="utf-8")

            paths = bench.select_graph_paths(
                root,
                graph_name=None,
                graph_index=1,
                num_graphs=2,
            )

            self.assertEqual([path.name for path in paths], ["G-1.json", "G-2.json"])

    def test_select_graph_paths_graph_name_overrides_num_graphs(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ["G-0.json", "G-1.json", "G-2.json"]:
                (root / name).write_text("{}", encoding="utf-8")

            paths = bench.select_graph_paths(
                root,
                graph_name="G-2.json",
                graph_index=0,
                num_graphs=2,
            )

            self.assertEqual([path.name for path in paths], ["G-2.json"])


if __name__ == "__main__":
    unittest.main()
