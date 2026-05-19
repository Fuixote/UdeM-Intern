import importlib.util
import unittest
from pathlib import Path


STEP1B_DIR = (
    Path(__file__).resolve().parents[1]
    / "surrogate_experiment_results"
    / "Step1b"
)


def load_module():
    module_path = STEP1B_DIR / "plot_posthoc_diagnostics.py"
    spec = importlib.util.spec_from_file_location("step1b_posthoc_diagnostics", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Step1bPosthocDiagnosticsTest(unittest.TestCase):
    def test_best_epoch_selects_minimum_metric(self):
        diagnostics = load_module()
        rows = [
            {"epoch": "0", "validation_fy_loss": "3.0"},
            {"epoch": "10", "validation_fy_loss": "2.0"},
            {"epoch": "20", "validation_fy_loss": "2.5"},
        ]

        selected = diagnostics.best_epoch(rows, "validation_fy_loss")

        self.assertEqual(selected, 10)

    def test_selection_suboptimality_uses_selected_minus_best(self):
        diagnostics = load_module()
        rows = [
            {"epoch": "0", "test_mean_decision_gap": "4.0"},
            {"epoch": "10", "test_mean_decision_gap": "2.5"},
            {"epoch": "20", "test_mean_decision_gap": "2.0"},
        ]

        value = diagnostics.selection_suboptimality(
            rows,
            selected_epoch=10,
            metric="test_mean_decision_gap",
        )

        self.assertAlmostEqual(value, 0.5)

    def test_paired_delta_is_baseline_gap_minus_candidate_gap(self):
        diagnostics = load_module()
        rows = [
            {
                "method": "2stage",
                "selection_metric": "validation_mse_loss",
                "graph": "G-1.json",
                "gap": "3.0",
            },
            {
                "method": "e2e",
                "selection_metric": "validation_fy_loss",
                "graph": "G-1.json",
                "gap": "1.5",
            },
        ]

        deltas = diagnostics.paired_gap_deltas(
            rows,
            candidate_key=("e2e", "validation_fy_loss"),
        )

        self.assertEqual(deltas, [1.5])

    def test_nonzero_mean_ignores_zero_gap_graphs(self):
        diagnostics = load_module()

        self.assertAlmostEqual(diagnostics.nonzero_mean([0.0, 2.0, 4.0]), 3.0)
        self.assertEqual(diagnostics.nonzero_mean([0.0, 0.0]), 0.0)


if __name__ == "__main__":
    unittest.main()
