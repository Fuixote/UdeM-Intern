import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "surrogate_experiment_results"
    / "Step2_resampling"
    / "plot_phase1_heldout400.py"
)


def load_plot_module():
    spec = importlib.util.spec_from_file_location("phase1_heldout400_plot", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Step2ResamplingPhase1PlotTest(unittest.TestCase):
    def test_groups_main_gap_boxes_in_phase1_regime_order(self):
        module = load_plot_module()
        rows = [
            {
                "regime": "step2c_poly_d1_mult_eps050",
                "method_label": "spoplus_val_spoplus_loss",
                "test_mean_normalized_gap": "0.4",
            },
            {
                "regime": "step2b_poly_d1",
                "method_label": "2stage_val_mse",
                "test_mean_normalized_gap": "0.2",
            },
            {
                "regime": "step2b_poly_d1",
                "method_label": "spoplus_val_spoplus_loss",
                "test_mean_normalized_gap": "0.1",
            },
            {
                "regime": "step2c_poly_d1_mult_eps050",
                "method_label": "2stage_val_mse",
                "test_mean_normalized_gap": "0.5",
            },
        ]

        grouped = module.group_main_gap_boxes(rows)

        self.assertEqual(grouped.labels, ["Step2b d1", "Step2c d1 eps050"])
        self.assertEqual(grouped.regimes, ["step2b_poly_d1", "step2c_poly_d1_mult_eps050"])
        self.assertEqual(grouped.mse_values, [[0.2], [0.5]])
        self.assertEqual(grouped.spoplus_values, [[0.1], [0.4]])

    def test_groups_paired_reduction_boxes_in_phase1_regime_order(self):
        module = load_plot_module()
        rows = [
            {
                "regime": "step2c_poly_d1_mult_eps050",
                "norm_gap_reduction": "-0.1",
            },
            {
                "regime": "step2b_poly_d1",
                "norm_gap_reduction": "0.2",
            },
        ]

        grouped = module.group_paired_reduction_boxes(rows)

        self.assertEqual(grouped.labels, ["Step2b d1", "Step2c d1 eps050"])
        self.assertEqual(grouped.values, [[0.2], [-0.1]])

    def test_groups_by_block_gap_boxes_for_two_panel_plot(self):
        module = load_plot_module()
        rows = [
            {
                "regime": "step2b_poly_d2",
                "method_label": "2stage_val_mse",
                "test_mean_normalized_gap": "0.2",
            },
            {
                "regime": "step2b_poly_d2",
                "method_label": "spoplus_val_spoplus_loss",
                "test_mean_normalized_gap": "0.1",
            },
            {
                "regime": "step2c_poly_d4_mult_eps050",
                "method_label": "2stage_val_mse",
                "test_mean_normalized_gap": "0.5",
            },
            {
                "regime": "step2c_poly_d4_mult_eps050",
                "method_label": "spoplus_val_spoplus_loss",
                "test_mean_normalized_gap": "0.4",
            },
        ]

        panels = module.group_by_block_gap_boxes(rows)

        self.assertEqual([panel.block for panel in panels], ["step2b", "step2c"])
        self.assertEqual(panels[0].degrees, [2])
        self.assertEqual(panels[0].mse_values, [[0.2]])
        self.assertEqual(panels[0].spoplus_values, [[0.1]])
        self.assertEqual(panels[1].degrees, [4])
        self.assertEqual(panels[1].mse_values, [[0.5]])
        self.assertEqual(panels[1].spoplus_values, [[0.4]])

    def test_to_percent_boxes_multiplies_nested_values_by_100(self):
        module = load_plot_module()

        self.assertEqual(
            module.to_percent_boxes([[0.061, 0.049], [0.001]]),
            [[6.1, 4.9], [0.1]],
        )

    def test_groups_relative_reduction_by_block_as_percent_of_2stage_gap(self):
        module = load_plot_module()
        rows = [
            {
                "regime": "step2b_poly_d8",
                "mse_norm_gap": "0.06",
                "spoplus_norm_gap": "0.048",
            },
            {
                "regime": "step2c_poly_d2_mult_eps050",
                "mse_norm_gap": "0.04",
                "spoplus_norm_gap": "0.038",
            },
        ]

        panels = module.group_by_block_relative_reduction_boxes(rows)

        self.assertEqual([panel.block for panel in panels], ["step2b", "step2c"])
        self.assertEqual(panels[0].degrees, [8])
        self.assertAlmostEqual(panels[0].values[0][0], 20.0)
        self.assertEqual(panels[1].degrees, [2])
        self.assertAlmostEqual(panels[1].values[0][0], 5.0)


if __name__ == "__main__":
    unittest.main()
