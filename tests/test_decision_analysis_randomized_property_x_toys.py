from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "decision_analysis"
    / "scripts"
    / "run_randomized_property_x_toy_experiments.py"
)


class RandomizedPropertyXToyTests(unittest.TestCase):
    def load_module(self):
        self.assertTrue(SCRIPT_PATH.exists(), f"Missing script: {SCRIPT_PATH}")
        spec = importlib.util.spec_from_file_location(
            "run_randomized_property_x_toy_experiments", SCRIPT_PATH
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def test_core_metrics_match_hand_checkable_packing_and_path_cases(self):
        module = self.load_module()

        packing = module.evaluate_partition_instance(
            true_values=((10.0, 9.9), (10.0, 9.9)),
            predicted_values=((9.8, 10.1), (9.8, 10.1)),
        )
        self.assertTrue(packing.identity_mismatch)
        self.assertEqual(packing.changed_component_count, 2)
        self.assertAlmostEqual(packing.oracle_objective, 20.0)
        self.assertAlmostEqual(packing.selected_true_objective, 19.8)
        self.assertAlmostEqual(packing.normalized_regret, 0.01)
        self.assertAlmostEqual(packing.true_second_best_normalized_gap, 0.005)

        path = module.evaluate_parallel_path_instance(
            true_path_costs=(5.0, 8.0),
            predicted_path_costs=(5.0, 4.8),
            path_length=3,
        )
        self.assertTrue(path.identity_mismatch)
        self.assertEqual(path.changed_component_count, 3)
        self.assertAlmostEqual(path.oracle_objective, 5.0)
        self.assertAlmostEqual(path.selected_true_objective, 8.0)
        self.assertAlmostEqual(path.normalized_regret, 0.6)
        self.assertAlmostEqual(path.true_second_best_normalized_gap, 0.6)

    def test_sweep_is_deterministic_and_covers_both_problem_families(self):
        module = self.load_module()

        first = module.run_sweep(
            tau_values=(0.05, 0.30),
            sigma_values=(0.0, 0.10),
            num_instances=12,
            base_seed=7,
            packing_blocks=4,
            packing_choices=3,
            path_count=3,
            path_length=4,
        )
        second = module.run_sweep(
            tau_values=(0.05, 0.30),
            sigma_values=(0.0, 0.10),
            num_instances=12,
            base_seed=7,
            packing_blocks=4,
            packing_choices=3,
            path_count=3,
            path_length=4,
        )

        self.assertEqual(first.summary_rows, second.summary_rows)
        self.assertEqual(len(first.packing_summary_rows), 4)
        self.assertEqual(len(first.shortest_path_summary_rows), 4)
        self.assertEqual(len(first.summary_rows), 8)

        families = {row["problem_family"] for row in first.summary_rows}
        self.assertEqual(families, {"decomposable_packing", "parallel_shortest_path"})

        sigma_zero_rows = [
            row for row in first.summary_rows if abs(row["sigma"] - 0.0) < 1e-12
        ]
        self.assertTrue(sigma_zero_rows)
        self.assertTrue(all(row["identity_mismatch_rate"] == 0.0 for row in sigma_zero_rows))
        self.assertTrue(all(row["mean_normalized_regret"] == 0.0 for row in sigma_zero_rows))

    def test_main_writes_expected_randomized_toy_csvs_without_plots(self):
        module = self.load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            result_dir = tmp_path / "results"
            plot_dir = tmp_path / "plots"
            argv = [
                "run_randomized_property_x_toy_experiments.py",
                "--result-dir",
                str(result_dir),
                "--plot-dir",
                str(plot_dir),
                "--num-instances",
                "10",
                "--tau-values",
                "0.05",
                "0.30",
                "--sigma-values",
                "0.0",
                "0.10",
                "--skip-plots",
            ]
            with mock.patch.object(sys, "argv", argv):
                exit_code = module.main()

            self.assertEqual(exit_code, 0)
            expected_files = {
                "randomized_packing_summary.csv",
                "randomized_shortest_path_summary.csv",
                "randomized_property_x_comparison.csv",
            }
            self.assertEqual({path.name for path in result_dir.iterdir()}, expected_files)

            with (result_dir / "randomized_property_x_comparison.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 8)
            self.assertIn("identity_mismatch_rate", rows[0])
            self.assertIn("mean_true_second_best_normalized_gap", rows[0])
            self.assertIn("mean_dfl_improvement_ceiling", rows[0])


if __name__ == "__main__":
    unittest.main()
