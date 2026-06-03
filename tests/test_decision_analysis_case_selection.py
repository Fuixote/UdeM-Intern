from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "decision_analysis"
    / "scripts"
    / "select_case_seeds.py"
)
PAIRED_MAIN = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "Step2_resampling"
    / "results"
    / "phase1_heldout400_paired_main.csv"
)


class DecisionAnalysisCaseSelectionTests(unittest.TestCase):
    def load_module(self):
        self.assertTrue(SCRIPT_PATH.exists(), f"Missing script: {SCRIPT_PATH}")
        spec = importlib.util.spec_from_file_location("select_case_seeds", SCRIPT_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def test_select_case_seeds_matches_step2b_d8_plan(self):
        module = self.load_module()
        rows = module.load_rows(PAIRED_MAIN)

        selected = module.select_case_seeds(rows, regime="step2b_poly_d8")
        by_type = {}
        for row in selected:
            by_type.setdefault(row.case_type, []).append(row.subset_seed)

        self.assertEqual(
            by_type,
            {
                "large_improvement": [1, 25, 22],
                "weak_borderline_improvement": [27, 21, 30],
                "easy_low_gap": [41, 16, 33],
            },
        )

    def test_write_selected_case_seeds_csv(self):
        module = self.load_module()
        rows = module.load_rows(PAIRED_MAIN)
        selected = module.select_case_seeds(rows, regime="step2b_poly_d8")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "selected_case_seeds.csv"
            module.write_selected_csv(output_path, selected)

            with output_path.open(newline="", encoding="utf-8") as handle:
                written = list(csv.DictReader(handle))

        self.assertEqual(written[0]["case_type"], "large_improvement")
        self.assertEqual(written[0]["subset_seed"], "1")
        self.assertEqual(
            written[0]["selection_reason"],
            "highest paired normalized-gap reduction",
        )
        self.assertEqual(len(written), 9)


if __name__ == "__main__":
    unittest.main()
