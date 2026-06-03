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
    / "analyze_edge_error_criticality.py"
)


class DecisionAnalysisEdgeCriticalityTests(unittest.TestCase):
    def load_module(self):
        self.assertTrue(SCRIPT_PATH.exists(), f"Missing script: {SCRIPT_PATH}")
        spec = importlib.util.spec_from_file_location(
            "analyze_edge_error_criticality", SCRIPT_PATH
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def test_descending_error_ranks_start_at_one(self):
        module = self.load_module()

        ranks = module.descending_error_ranks(np.asarray([0.2, 1.5, 0.1, 0.7]))

        self.assertEqual(ranks.tolist(), [3, 1, 4, 2])

    def test_membership_flags_identify_selected_and_symdiff_edges(self):
        module = self.load_module()
        y_opt = np.asarray([1, 0, 1, 0], dtype=float)
        y_2stage = np.asarray([1, 1, 0, 0], dtype=float)
        y_spoplus = np.asarray([1, 0, 1, 1], dtype=float)

        flags = module.edge_membership_flags(y_opt, y_2stage, y_spoplus)

        self.assertEqual(flags["in_opt"].tolist(), [True, False, True, False])
        self.assertEqual(flags["in_2stage"].tolist(), [True, True, False, False])
        self.assertEqual(flags["in_spoplus"].tolist(), [True, False, True, True])
        self.assertEqual(
            flags["in_2stage_symdiff"].tolist(),
            [False, True, True, False],
        )
        self.assertEqual(
            flags["in_spoplus_symdiff"].tolist(),
            [False, False, False, True],
        )
        self.assertEqual(
            flags["in_any_selected"].tolist(),
            [True, True, True, True],
        )

    def test_safe_mean_returns_nan_for_empty_masks(self):
        module = self.load_module()

        value = module.safe_masked_mean(
            np.asarray([1.0, 2.0]), np.asarray([False, False])
        )

        self.assertTrue(np.isnan(value))


if __name__ == "__main__":
    unittest.main()
