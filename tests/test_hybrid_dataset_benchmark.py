import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
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

    def test_solve_by_rebuilding_calls_solver_for_each_objective_vector(self):
        data = SimpleNamespace(edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long))
        args = SimpleNamespace(max_chain=4, time_limit=9.0)
        node_is_ndd = np.array([True, False, False])
        cycle_candidates = [{"type": "cycle", "edges": [0, 1], "nodes": [1, 2]}]
        weight_vectors = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        fake_results = [
            {"objective": 1.0, "edge_selection": np.array([1, 0]), "status": 2},
            {"objective": 2.0, "edge_selection": np.array([0, 1]), "status": 2},
        ]

        with (
            mock.patch.object(bench, "solve_cf_cycle_pief_chain", side_effect=fake_results) as solver,
            mock.patch.object(bench.time, "perf_counter", side_effect=[1.0, 1.5, 2.0, 2.75]),
        ):
            results, times = bench.solve_by_rebuilding(
                data=data,
                node_is_ndd=node_is_ndd,
                num_nodes=3,
                cycle_candidates=cycle_candidates,
                weight_vectors=weight_vectors,
                args=args,
                env="env",
            )

        self.assertEqual(results, fake_results)
        self.assertEqual(times, [0.5, 0.75])
        self.assertEqual(solver.call_count, 2)
        first_call = solver.call_args_list[0].kwargs
        np.testing.assert_array_equal(first_call["weights"], weight_vectors[0])
        self.assertIs(first_call["edge_index"], data.edge_index)
        self.assertIs(first_call["is_ndd_mask"], node_is_ndd)
        self.assertEqual(first_call["num_nodes"], 3)
        self.assertEqual(first_call["cycle_candidates"], cycle_candidates)
        self.assertEqual(first_call["max_chain"], 4)
        self.assertEqual(first_call["env"], "env")
        self.assertEqual(first_call["time_limit"], 9.0)

    def test_solve_by_reusing_builds_once_and_updates_objectives(self):
        data = SimpleNamespace(edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long))
        args = SimpleNamespace(max_chain=4, time_limit=9.0, threads=2, reset_before_solve=True)
        node_is_ndd = np.array([True, False, False])
        cycle_candidates = [{"type": "cycle", "edges": [0, 1], "nodes": [1, 2]}]
        weight_vectors = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        fake_results = [
            {"objective": 1.0, "edge_selection": np.array([1, 0]), "status": 2},
            {"objective": 2.0, "edge_selection": np.array([0, 1]), "status": 2},
        ]
        cached_model = mock.Mock()
        cached_model.solve.side_effect = fake_results

        with (
            mock.patch.object(bench, "CachedHybridKepModel", return_value=cached_model) as model_cls,
            mock.patch.object(bench.time, "perf_counter", side_effect=[10.0, 10.25, 11.0, 11.4, 12.0, 12.6]),
        ):
            results, times, build_time = bench.solve_by_reusing(
                data=data,
                node_is_ndd=node_is_ndd,
                num_nodes=3,
                cycle_candidates=cycle_candidates,
                weight_vectors=weight_vectors,
                args=args,
                env="env",
                graph="G-0.json",
            )

        self.assertEqual(results, fake_results)
        np.testing.assert_allclose(times, [0.4, 0.6])
        self.assertAlmostEqual(build_time, 0.25)
        model_cls.assert_called_once()
        model_kwargs = model_cls.call_args.kwargs
        self.assertIs(model_kwargs["edge_index"], data.edge_index)
        self.assertIs(model_kwargs["is_ndd_mask"], node_is_ndd)
        self.assertEqual(model_kwargs["num_nodes"], 3)
        self.assertEqual(model_kwargs["cycle_candidates"], cycle_candidates)
        self.assertEqual(model_kwargs["max_chain"], 4)
        self.assertEqual(model_kwargs["env"], "env")
        self.assertEqual(model_kwargs["time_limit"], 9.0)
        self.assertEqual(model_kwargs["threads"], 2)
        self.assertEqual(model_kwargs["name"], "cached_hybrid_G-0.json")
        self.assertEqual(cached_model.solve.call_count, 2)
        np.testing.assert_array_equal(cached_model.solve.call_args_list[0].args[0], weight_vectors[0])
        self.assertEqual(cached_model.solve.call_args_list[0].kwargs["time_limit"], 9.0)
        self.assertTrue(cached_model.solve.call_args_list[0].kwargs["reset_before_solve"])
        cached_model.dispose.assert_called_once()

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
