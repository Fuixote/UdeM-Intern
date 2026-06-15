from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "decision_analysis"
    / "scripts"
    / "compute_second_best_solutions.py"
)


class DecisionAnalysisSecondBestSolutionTests(unittest.TestCase):
    def load_module(self):
        self.assertTrue(SCRIPT_PATH.exists(), f"Missing script: {SCRIPT_PATH}")
        spec = importlib.util.spec_from_file_location(
            "compute_second_best_solutions", SCRIPT_PATH
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def test_parse_args_accepts_cycle_length_controls(self):
        module = self.load_module()

        defaults = module.parse_args([])
        self.assertEqual(defaults.max_cycle, 3)
        self.assertEqual(defaults.max_chain, 4)

        args = module.parse_args(["--max-cycle", "5", "--max-chain", "6"])
        self.assertEqual(args.max_cycle, 5)
        self.assertEqual(args.max_chain, 6)

    def test_selected_cases_can_be_generated_from_subset_seed_range(self):
        module = self.load_module()

        args = module.parse_args(
            [
                "--regime",
                "step2c_poly_d8_mult_eps050",
                "--subset-seed-start",
                "0",
                "--subset-seed-stop",
                "2",
                "--case-type-prefix",
                "step2c_fixed_graph",
            ]
        )
        cases = module.selected_cases_for_args(args)

        self.assertEqual(
            cases,
            [
                {
                    "regime": "step2c_poly_d8_mult_eps050",
                    "case_type": "step2c_fixed_graph_seed0",
                    "subset_seed": "0",
                },
                {
                    "regime": "step2c_poly_d8_mult_eps050",
                    "case_type": "step2c_fixed_graph_seed1",
                    "subset_seed": "1",
                },
                {
                    "regime": "step2c_poly_d8_mult_eps050",
                    "case_type": "step2c_fixed_graph_seed2",
                    "subset_seed": "2",
                },
            ],
        )

    def test_output_schemas_record_cycle_length_controls(self):
        module = self.load_module()

        self.assertIn("max_cycle", module.CSV_FIELDS)
        self.assertIn("max_chain", module.CSV_FIELDS)
        self.assertIn("num_cycle_candidates", module.CSV_FIELDS)
        self.assertIn("num_chain_candidates", module.CSV_FIELDS)
        self.assertIn("max_cycle", module.SUMMARY_FIELDS)
        self.assertIn("max_chain", module.SUMMARY_FIELDS)

    def test_compute_rows_passes_cycle_controls_to_graph_loader(self):
        module = self.load_module()
        args = module.parse_args(["--max-cycle", "4", "--max-chain", "7"])

        fake_common = types.SimpleNamespace(
            load_graph_records=mock.Mock(return_value=[]),
            dispose_graph_records=mock.Mock(),
        )
        fake_env = mock.Mock()
        fake_gp = types.SimpleNamespace(Env=mock.Mock(return_value=fake_env))

        with mock.patch.object(
            module,
            "ensure_step1c_imports",
            return_value=(fake_common, object(), None, None, None),
        ), mock.patch.object(
            module,
            "load_selected_cases",
            return_value=[{"subset_seed": "1", "case_type": "synthetic"}],
        ), mock.patch.object(
            module, "load_or_make_split_entries", return_value=[{"path": "G-1.json"}]
        ), mock.patch.object(
            module, "resolve_graph_path", return_value=Path("G-1.json")
        ), mock.patch.object(
            module, "resolve_run_dir", return_value=Path("run")
        ), mock.patch.object(
            module, "load_models", return_value=[]
        ), mock.patch.dict(
            sys.modules, {"gurobipy": fake_gp}
        ):
            rows = module.compute_second_best_rows(args)

        self.assertEqual(rows, [])
        fake_common.load_graph_records.assert_called_once_with(
            [Path("G-1.json")],
            fake_env,
            max_cycle=4,
            max_chain=7,
        )
        fake_common.dispose_graph_records.assert_called_once_with([])
        fake_env.dispose.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
