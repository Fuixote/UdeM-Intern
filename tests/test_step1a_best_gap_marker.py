import importlib.util
import unittest
from pathlib import Path

import numpy as np


def load_plot_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "surrogate_experiment_results"
        / "Step1a"
        / "plot_trajectories_2D.py"
    )
    spec = importlib.util.spec_from_file_location("step1a_plot_2d", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Step1aBestGapMarkerTest(unittest.TestCase):
    def test_best_gap_marker_selects_lowest_decision_gap_epoch(self):
        plot_2d = load_plot_module()
        metrics = {"decision_gap": np.array([0.8, 0.3, 0.5])}

        marker = plot_2d.best_gap_marker(metrics)

        self.assertEqual(marker["epoch"], 1)
        self.assertAlmostEqual(marker["gap"], 0.3)

    def test_best_gap_marker_returns_none_without_decision_gap(self):
        plot_2d = load_plot_module()

        self.assertIsNone(plot_2d.best_gap_marker({}))


if __name__ == "__main__":
    unittest.main()
