import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "surrogate_experiment_results"
    / "Step2"
    / "plot_step2_summary.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("plot_step2_summary", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Step2SummaryPlotTest(unittest.TestCase):
    def test_regime_sort_key_orders_step2_abc(self):
        module = load_module()

        regimes = [
            "step2c_poly_d8_mult_eps050",
            "step2b_poly_d1",
            "step2a_additive_rho050",
            "step2b_poly_d4",
        ]

        self.assertEqual(
            sorted(regimes, key=module.regime_sort_key),
            [
                "step2a_additive_rho050",
                "step2b_poly_d1",
                "step2b_poly_d4",
                "step2c_poly_d8_mult_eps050",
            ],
        )

    def test_selector_delta_positive_means_surrogate_selector_better(self):
        module = load_module()

        rows = [
            {
                "regime": "step2b_poly_d8",
                "train_size": "50",
                "checkpoint_label": "fy_val_decision_gap",
                "test_mean_normalized_gap": "0.20",
            },
            {
                "regime": "step2b_poly_d8",
                "train_size": "50",
                "checkpoint_label": "fy_val_fy_loss",
                "test_mean_normalized_gap": "0.15",
            },
        ]

        deltas = module.compute_selector_deltas(
            rows=rows,
            direct_label="fy_val_decision_gap",
            surrogate_label="fy_val_fy_loss",
        )

        self.assertEqual(len(deltas), 1)
        self.assertAlmostEqual(deltas[0]["selector_delta"], 0.05)

    def test_primary_checkpoint_labels_are_stable(self):
        module = load_module()

        self.assertEqual(
            list(module.PRIMARY_CHECKPOINTS),
            ["2stage_val_mse", "fy_val_fy_loss", "spoplus_val_spoplus_loss"],
        )
        self.assertEqual(module.METHOD_LABELS["spoplus_val_spoplus_loss"], "SPO+ (val SPO+)")


if __name__ == "__main__":
    unittest.main()
