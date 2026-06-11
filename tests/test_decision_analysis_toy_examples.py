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
    / "build_toy_property_x_examples.py"
)


class DecisionAnalysisToyExampleTests(unittest.TestCase):
    def load_module(self):
        self.assertTrue(SCRIPT_PATH.exists(), f"Missing script: {SCRIPT_PATH}")
        spec = importlib.util.spec_from_file_location("build_toy_property_x_examples", SCRIPT_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def selected_row(self, summary_rows: list[dict], example: str, selected_by: str) -> dict:
        return next(
            row
            for row in summary_rows
            if row["example"] == example and row["selected_by"] == selected_by
        )

    def test_toy_family_covers_packing_families_parametric_case_and_path_controls(self):
        module = self.load_module()

        detail_tables, summary_rows = module.build_all_toy_examples()
        examples = {row["example"] for row in summary_rows}

        positive_examples = {
            "toy_kep_packing",
            "toy_stable_set",
            "toy_weighted_matching",
            "toy_knapsack_capacity",
            "toy_partition_matroid",
        }
        negative_controls = {
            "toy_shortest_path",
            "toy_serial_path",
        }
        parametric_examples = {
            example for example in examples if example.startswith("toy_parametric_epsilon_")
        }

        self.assertTrue(positive_examples.issubset(examples))
        self.assertTrue(negative_controls.issubset(examples))
        self.assertGreaterEqual(len(parametric_examples), 4)
        self.assertTrue(positive_examples.issubset(detail_tables))
        self.assertIn("toy_parametric_epsilon_solutions", detail_tables)

        for example in positive_examples:
            oracle = self.selected_row(summary_rows, example, "oracle_by_true")
            two_stage = self.selected_row(summary_rows, example, "2stage_rank1_by_prediction")
            dfl = self.selected_row(summary_rows, example, "dfl_rank1_by_prediction")
            self.assertNotEqual(two_stage["solution"], oracle["solution"])
            self.assertLessEqual(two_stage["normalized_gap_percent"], 5.0)
            self.assertEqual(dfl["solution"], oracle["solution"])
            self.assertAlmostEqual(dfl["normalized_gap_percent"], 0.0)

        for example in negative_controls:
            oracle = self.selected_row(summary_rows, example, "oracle_by_true")
            two_stage = self.selected_row(summary_rows, example, "2stage_rank1_by_prediction")
            self.assertNotEqual(two_stage["solution"], oracle["solution"])
            self.assertGreaterEqual(two_stage["normalized_gap_percent"], 40.0)

    def test_parametric_close_substitute_family_keeps_identity_switch_with_vanishing_regret(self):
        module = self.load_module()

        _, summary_rows = module.build_all_toy_examples()
        parametric_two_stage = [
            row
            for row in summary_rows
            if row["example"].startswith("toy_parametric_epsilon_")
            and row["selected_by"] == "2stage_rank1_by_prediction"
        ]
        parametric_dfl = [
            row
            for row in summary_rows
            if row["example"].startswith("toy_parametric_epsilon_")
            and row["selected_by"] == "dfl_rank1_by_prediction"
        ]

        self.assertGreaterEqual(len(parametric_two_stage), 4)
        self.assertTrue(
            all(row["solution"] == "S_epsilon" for row in parametric_two_stage)
        )
        self.assertEqual(
            sorted(round(row["normalized_gap_percent"], 3) for row in parametric_two_stage),
            [0.1, 1.0, 2.0, 5.0],
        )
        self.assertTrue(all(row["solution"] == "S_star" for row in parametric_dfl))
        self.assertTrue(
            all(abs(row["normalized_gap_percent"]) < 1e-12 for row in parametric_dfl)
        )

    def test_main_writes_expanded_toy_family_outputs(self):
        module = self.load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            result_dir = tmp_path / "results"
            plot_dir = tmp_path / "plots"
            argv = [
                "build_toy_property_x_examples.py",
                "--result-dir",
                str(result_dir),
                "--plot-dir",
                str(plot_dir),
                "--skip-plots",
            ]
            with mock.patch.object(sys, "argv", argv):
                exit_code = module.main()

            self.assertEqual(exit_code, 0)

            expected_files = {
                "toy_kep_packing_solutions.csv",
                "toy_stable_set_solutions.csv",
                "toy_weighted_matching_solutions.csv",
                "toy_knapsack_capacity_solutions.csv",
                "toy_partition_matroid_solutions.csv",
                "toy_shortest_path_solutions.csv",
                "toy_serial_path_solutions.csv",
                "toy_parametric_epsilon_solutions.csv",
                "toy_policy_summary.csv",
                "toy_summary_for_paper.tex",
                "toy_explanation.md",
            }
            self.assertEqual(
                {path.name for path in result_dir.iterdir()},
                expected_files,
            )

            with (result_dir / "toy_policy_summary.csv").open(newline="", encoding="utf-8") as handle:
                summary_rows = list(csv.DictReader(handle))
            examples = {row["example"] for row in summary_rows}
            self.assertIn("toy_weighted_matching", examples)
            self.assertIn("toy_serial_path", examples)

            explanation = (result_dir / "toy_explanation.md").read_text(encoding="utf-8")
            self.assertIn("weighted matching", explanation)
            self.assertIn("epsilon", explanation)
            self.assertIn("negative controls", explanation)

            latex = (result_dir / "toy_summary_for_paper.tex").read_text(encoding="utf-8")
            self.assertIn("Weighted matching", latex)
            self.assertIn("Parametric packing family", latex)
            self.assertIn("S\\_star", latex)
            self.assertIn("S\\_epsilon", latex)
            self.assertNotIn("S_star", latex)
            self.assertNotIn("S_epsilon", latex)


if __name__ == "__main__":
    unittest.main()
