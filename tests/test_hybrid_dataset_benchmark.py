import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from scripts import benchmark_hybrid_model_reuse_on_dataset as bench


class HybridDatasetBenchmarkHelperTest(unittest.TestCase):
    def test_cycle_candidates_only_handles_flat_and_batched_candidates(self):
        candidates = [
            {"type": "cycle", "edges": [0, 1]},
            {"type": "chain", "edges": [2, 3]},
        ]
        self.assertEqual(bench.cycle_candidates_only(candidates), [candidates[0]])
        self.assertEqual(bench.cycle_candidates_only([candidates]), [candidates[0]])

    def test_make_weight_vectors_generates_uniform_objective_coefficients(self):
        class Data:
            edge_index = torch.zeros((2, 3), dtype=torch.long)

        rng = np.random.default_rng(123)
        weights = bench.make_weight_vectors(
            Data(),
            num_weight_vectors=4,
            rng=rng,
        )

        expected = np.random.default_rng(123).uniform(0.0, 1.0, size=(4, 3)).astype(np.float32)
        self.assertEqual(weights.shape, (4, 3))
        self.assertEqual(weights.dtype, np.float32)
        np.testing.assert_allclose(weights, expected)

    def test_make_weight_vectors_handles_empty_graph(self):
        class Data:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        weights = bench.make_weight_vectors(
            Data(),
            num_weight_vectors=2,
            rng=np.random.default_rng(1),
        )

        self.assertEqual(weights.shape, (2, 0))
        self.assertEqual(weights.dtype, np.float32)

    def test_parse_args_defaults_to_one_graph(self):
        with mock.patch("sys.argv", ["benchmark_hybrid_model_reuse_on_dataset.py"]):
            args = bench.parse_args()

        self.assertEqual(args.num_graphs, 1)
        self.assertFalse(hasattr(args, "noise_scale"))

    def test_save_outputs_writes_csvs_and_pngs(self):
        summary_rows = [
            {
                "graph": "G-00001.json",
                "num_nodes": 3,
                "num_edges": 2,
                "num_cycles": 1,
                "num_weight_vectors": 2,
                "rebuild_total_time": 2.0,
                "reuse_build_time": 0.5,
                "reuse_solve_time": 0.5,
                "reuse_total_time": 1.0,
                "speedup_including_build": 2.0,
                "speedup_excluding_build": 4.0,
                "max_obj_diff": 0.0,
                "mean_obj_diff": 0.0,
                "obj_mismatches": 0,
                "selection_mismatches": 0,
            }
        ]
        per_solve_rows = [
            {
                "graph": "G-00001.json",
                "solve_idx": 0,
                "rebuild_time": 1.0,
                "reuse_time": 0.2,
                "rebuild_obj": 1.5,
                "reuse_obj": 1.5,
                "obj_diff": 0.0,
                "rebuild_status": 2,
                "reuse_status": 2,
                "selection_mismatch": False,
            },
            {
                "graph": "G-00001.json",
                "solve_idx": 1,
                "rebuild_time": 1.0,
                "reuse_time": 0.3,
                "rebuild_obj": 2.5,
                "reuse_obj": 2.5,
                "obj_diff": 0.0,
                "rebuild_status": 2,
                "reuse_status": 2,
                "selection_mismatch": False,
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            bench.save_outputs(summary_rows, per_solve_rows, Path(tmp))
            self.assertTrue((Path(tmp) / "hybrid_reuse_summary.csv").exists())
            self.assertTrue((Path(tmp) / "hybrid_reuse_per_solve.csv").exists())
            self.assertTrue((Path(tmp) / "per_solve_time_curve.png").exists())
            self.assertTrue((Path(tmp) / "total_time_bar.png").exists())
            self.assertTrue((Path(tmp) / "speedup_by_graph.png").exists())

    def test_select_data_dir_prefers_root_then_populated_subdir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            empty = root / "empty"
            populated = root / "realistic_synthetic_dataset"
            empty.mkdir()
            populated.mkdir()
            (populated / "G-2.json").write_text("{}", encoding="utf-8")
            (populated / "G-1.json").write_text("{}", encoding="utf-8")

            self.assertEqual(bench.select_data_dir(root), populated)

            (root / "G-0.json").write_text("{}", encoding="utf-8")
            self.assertEqual(bench.select_data_dir(root), root)


if __name__ == "__main__":
    unittest.main()
