import importlib.util
import unittest
from pathlib import Path

import numpy as np


def load_step1b_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "surrogate_experiment_results"
        / "Step1b"
        / "generalization_experiment.py"
    )
    spec = importlib.util.spec_from_file_location("step1b_generalization", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Step1bGeneralizationHelpersTest(unittest.TestCase):
    def test_make_split_is_deterministic_and_disjoint(self):
        step1b = load_step1b_module()
        files = [Path(f"G-{idx}.json") for idx in range(20)]

        first = step1b.make_split(files, train_size=4, val_size=3, test_size=5, seed=7)
        second = step1b.make_split(files, train_size=4, val_size=3, test_size=5, seed=7)

        self.assertEqual(first, second)
        self.assertEqual(len(first["train"]), 4)
        self.assertEqual(len(first["validation"]), 3)
        self.assertEqual(len(first["test"]), 5)

        train = set(first["train"])
        validation = set(first["validation"])
        test = set(first["test"])
        self.assertFalse(train & validation)
        self.assertFalse(train & test)
        self.assertFalse(validation & test)

    def test_make_split_rejects_insufficient_files(self):
        step1b = load_step1b_module()
        files = [Path(f"G-{idx}.json") for idx in range(3)]

        with self.assertRaisesRegex(ValueError, "Need 4 graphs"):
            step1b.make_split(files, train_size=2, val_size=1, test_size=1, seed=1)

    def test_select_checkpoint_uses_minimum_metric(self):
        step1b = load_step1b_module()
        trajectory = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        metrics = np.array([3.0, 0.5, 0.8])

        selected = step1b.select_checkpoint(trajectory, metrics)

        self.assertEqual(selected["epoch"], 1)
        np.testing.assert_allclose(selected["theta"], np.array([1.0, 1.0]))
        self.assertEqual(selected["metric"], 0.5)

    def test_summarize_test_metrics_reports_normalized_gap_and_pairing(self):
        step1b = load_step1b_module()
        mse = [
            {"graph": "G-1.json", "gap": 2.0, "normalized_gap": 0.20, "ratio": 0.80},
            {"graph": "G-2.json", "gap": 1.0, "normalized_gap": 0.10, "ratio": 0.90},
        ]
        fy = [
            {"graph": "G-1.json", "gap": 1.5, "normalized_gap": 0.15, "ratio": 0.85},
            {"graph": "G-2.json", "gap": 0.8, "normalized_gap": 0.08, "ratio": 0.92},
        ]

        summary = step1b.summarize_test_metrics(
            method="fy_warm",
            checkpoint_rule="validation_decision_gap",
            checkpoint={"epoch": 3, "theta": np.array([2.0, 1.0]), "metric": 0.4},
            evaluations=fy,
            baseline_evaluations=mse,
        )

        self.assertEqual(summary["method"], "fy_warm")
        self.assertEqual(summary["checkpoint_epoch"], 3)
        self.assertAlmostEqual(summary["test_mean_decision_gap"], 1.15)
        self.assertAlmostEqual(summary["test_mean_normalized_gap"], 0.115)
        self.assertAlmostEqual(summary["test_median_normalized_gap"], 0.115)
        self.assertAlmostEqual(summary["test_mean_achieved_oracle_ratio"], 0.885)
        self.assertAlmostEqual(summary["paired_gap_improvement_over_mse"], 0.35)


if __name__ == "__main__":
    unittest.main()
