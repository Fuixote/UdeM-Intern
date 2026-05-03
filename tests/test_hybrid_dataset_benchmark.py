import tempfile
import unittest
from pathlib import Path

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

    def test_make_weight_vectors_uses_scaled_labels_and_nonnegative_noise(self):
        class Data:
            y = torch.tensor([0.5, 1.0], dtype=torch.float32)
            edge_attr = torch.tensor([[0.1], [0.2]], dtype=torch.float32)

        rng = np.random.default_rng(123)
        weights = bench.make_weight_vectors(
            Data(),
            num_weight_vectors=4,
            noise_scale=0.2,
            rng=rng,
        )

        self.assertEqual(weights.shape, (4, 2))
        self.assertEqual(weights.dtype, np.float32)
        self.assertTrue(np.all(weights >= 0.0))
        self.assertTrue(np.all(weights.mean(axis=0) > np.array([5.0, 10.0], dtype=np.float32)))

    def test_make_weight_vectors_falls_back_to_edge_attr_when_labels_are_zero(self):
        class Data:
            y = torch.zeros(2, dtype=torch.float32)
            edge_attr = torch.tensor([[0.1], [0.2]], dtype=torch.float32)

        weights = bench.make_weight_vectors(
            Data(),
            num_weight_vectors=2,
            noise_scale=0.0,
            rng=np.random.default_rng(1),
        )

        np.testing.assert_allclose(weights, np.array([[0.1, 0.2], [0.1, 0.2]], dtype=np.float32))

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
