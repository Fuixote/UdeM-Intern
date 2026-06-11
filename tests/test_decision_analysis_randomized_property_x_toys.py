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

    def test_core_metrics_match_hand_checkable_packing_stable_and_path_cases(self):
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

        stable = module.evaluate_clustered_stable_set_instance(
            true_values=((10.0, 9.9), (10.0, 9.9)),
            predicted_values=((9.8, 10.1), (9.8, 10.1)),
        )
        self.assertTrue(stable.identity_mismatch)
        self.assertEqual(stable.changed_component_count, 2)
        self.assertAlmostEqual(stable.oracle_objective, 20.0)
        self.assertAlmostEqual(stable.selected_true_objective, 19.8)
        self.assertAlmostEqual(stable.normalized_regret, 0.01)
        self.assertAlmostEqual(stable.true_second_best_normalized_gap, 0.005)

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

    def test_kep_set_packing_metrics_match_hand_checkable_cycle_case(self):
        module = self.load_module()

        cycles = (
            module.KepCycle(
                name="C12",
                vertices=(1, 2),
                edges=((1, 2), (2, 1)),
                true_value=10.0,
                predicted_value=9.5,
            ),
            module.KepCycle(
                name="C34",
                vertices=(3, 4),
                edges=((3, 4), (4, 3)),
                true_value=10.0,
                predicted_value=9.5,
            ),
            module.KepCycle(
                name="C13",
                vertices=(1, 3),
                edges=((1, 3), (3, 1)),
                true_value=9.8,
                predicted_value=10.1,
            ),
            module.KepCycle(
                name="C24",
                vertices=(2, 4),
                edges=((2, 4), (4, 2)),
                true_value=9.8,
                predicted_value=10.1,
            ),
        )

        metrics = module.evaluate_kep_set_packing_instance(cycles)

        self.assertTrue(metrics.identity_mismatch)
        self.assertEqual(metrics.changed_component_count, 4)
        self.assertEqual(metrics.oracle_solution, "C12+C34")
        self.assertEqual(metrics.selected_solution, "C13+C24")
        self.assertAlmostEqual(metrics.oracle_objective, 20.0)
        self.assertAlmostEqual(metrics.selected_true_objective, 19.6)
        self.assertAlmostEqual(metrics.normalized_regret, 0.02)
        self.assertAlmostEqual(metrics.true_second_best_normalized_gap, 0.02)

    def test_kep_cycle_enumeration_finds_two_and_three_cycles_once(self):
        module = self.load_module()

        arcs = {
            (1, 2),
            (2, 1),
            (2, 3),
            (3, 4),
            (4, 2),
            (1, 3),
            (3, 1),
        }

        cycles = module.enumerate_directed_cycles(vertices=(1, 2, 3, 4), arcs=arcs, max_cycle=3)

        self.assertEqual(cycles, ((1, 2), (1, 3), (1, 2, 3), (2, 3, 4)))

    def test_sweep_is_deterministic_and_covers_all_problem_families(self):
        module = self.load_module()

        first = module.run_sweep(
            tau_values=(0.05, 0.30),
            sigma_values=(0.0, 0.10),
            num_instances=12,
            base_seed=7,
            packing_blocks=4,
            packing_choices=3,
            kep_vertices=8,
            kep_arc_probability=0.35,
            kep_max_cycle=3,
            stable_cliques=4,
            stable_vertices_per_clique=3,
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
            kep_vertices=8,
            kep_arc_probability=0.35,
            kep_max_cycle=3,
            stable_cliques=4,
            stable_vertices_per_clique=3,
            path_count=3,
            path_length=4,
        )

        self.assertEqual(first.summary_rows, second.summary_rows)
        self.assertEqual(len(first.packing_summary_rows), 4)
        self.assertEqual(len(first.kep_set_packing_summary_rows), 4)
        self.assertEqual(len(first.stable_set_summary_rows), 4)
        self.assertEqual(len(first.shortest_path_summary_rows), 4)
        self.assertEqual(len(first.summary_rows), 16)

        families = {row["problem_family"] for row in first.summary_rows}
        self.assertEqual(
            families,
            {
                "decomposable_packing",
                "random_kep_set_packing",
                "clustered_stable_set",
                "parallel_shortest_path",
            },
        )

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
                "randomized_kep_set_packing_summary.csv",
                "randomized_stable_set_summary.csv",
                "randomized_shortest_path_summary.csv",
                "randomized_property_x_comparison.csv",
            }
            self.assertEqual({path.name for path in result_dir.iterdir()}, expected_files)

            with (result_dir / "randomized_property_x_comparison.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 16)
            self.assertIn("identity_mismatch_rate", rows[0])
            self.assertIn("mean_true_second_best_normalized_gap", rows[0])
            self.assertIn("mean_dfl_improvement_ceiling", rows[0])


if __name__ == "__main__":
    unittest.main()
