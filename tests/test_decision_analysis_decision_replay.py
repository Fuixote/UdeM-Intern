from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "decision_analysis"
    / "scripts"
    / "compare_decisions_per_graph.py"
)


class DecisionAnalysisReplayTests(unittest.TestCase):
    def load_module(self):
        self.assertTrue(SCRIPT_PATH.exists(), f"Missing script: {SCRIPT_PATH}")
        spec = importlib.util.spec_from_file_location(
            "compare_decisions_per_graph", SCRIPT_PATH
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def test_solution_overlap_metrics_for_partial_overlap(self):
        module = self.load_module()
        y_pred = np.asarray([1, 0, 1, 0, 0], dtype=float)
        y_opt = np.asarray([1, 1, 0, 0, 0], dtype=float)

        metrics = module.solution_overlap_metrics(y_pred, y_opt)

        self.assertFalse(metrics["same_solution_as_opt"])
        self.assertEqual(metrics["num_edges_pred"], 2)
        self.assertEqual(metrics["num_edges_opt"], 2)
        self.assertEqual(metrics["edge_overlap_count"], 1)
        self.assertAlmostEqual(metrics["edge_jaccard_with_opt"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["edge_hamming_with_opt"], 2.0 / 5.0)

    def test_solution_overlap_metrics_for_two_empty_solutions(self):
        module = self.load_module()
        y_pred = np.zeros(4, dtype=float)
        y_opt = np.zeros(4, dtype=float)

        metrics = module.solution_overlap_metrics(y_pred, y_opt)

        self.assertTrue(metrics["same_solution_as_opt"])
        self.assertEqual(metrics["edge_jaccard_with_opt"], 1.0)
        self.assertEqual(metrics["edge_hamming_with_opt"], 0.0)

    def test_normalized_gap_uses_absolute_optimal_objective(self):
        module = self.load_module()

        self.assertAlmostEqual(module.normalized_gap(3.0, 12.0), 0.25)
        self.assertAlmostEqual(module.normalized_gap(3.0, -12.0), 0.25)

    def test_resolve_run_dir_matches_phase1_layout(self):
        module = self.load_module()
        run_root = Path("surrogate_experiment_results/Step2_resampling/phase1_runs")

        self.assertEqual(
            module.resolve_run_dir(run_root, "step2b_poly_d8", 25),
            run_root / "step2b_poly_d8" / "subset_seed=25",
        )

    def test_metric_tolerance_accepts_observed_replay_roundoff_only(self):
        module = self.load_module()

        self.assertTrue(
            module.metric_diffs_within_tolerance(
                gap_diff=2.5e-5,
                normalized_gap_diff=1.2e-7,
            )
        )
        self.assertFalse(
            module.metric_diffs_within_tolerance(
                gap_diff=2.0e-4,
                normalized_gap_diff=1.2e-7,
            )
        )
        self.assertFalse(
            module.metric_diffs_within_tolerance(
                gap_diff=2.5e-5,
                normalized_gap_diff=2.0e-6,
            )
        )

    def test_filter_split_entries_by_graph_filename(self):
        module = self.load_module()
        entries = [
            {"path": "dataset/G-1.json"},
            {"path": "dataset/G-43.json"},
            {"path": "dataset/G-7.json"},
        ]

        filtered = module.filter_split_entries_by_graphs(entries, {"G-43.json", "G-7.json"})

        self.assertEqual([Path(row["path"]).name for row in filtered], ["G-43.json", "G-7.json"])

    def test_weight_loading_order_matches_original_step1c_evaluator(self):
        module = self.load_module()

        self.assertEqual(
            module.WEIGHT_FILENAMES,
            (
                "2stage_best_by_validation_mse_loss.npz",
                "spoplus_best_by_validation_decision_gap.npz",
                "spoplus_best_by_validation_spoplus_loss.npz",
            ),
        )


if __name__ == "__main__":
    unittest.main()
