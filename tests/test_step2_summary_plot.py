import importlib.util
import sys
import tempfile
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

    def test_load_heldout_primary_expands_wide_rows_to_primary_checkpoints(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "heldout.csv"
            path.write_text(
                "\n".join(
                    [
                        "block,regime,degree,noise,train_size,2stage_val_mse,fy_val_fy,spoplus_val_spoplus,best_method,best_mean_norm_gap",
                        "Step2b,step2b_poly_d8,8,,200,0.30,0.20,0.25,fy,0.20",
                    ]
                ),
                encoding="utf-8",
            )

            rows = module.load_heldout_primary(path)

        self.assertEqual([row["checkpoint_label"] for row in rows], list(module.PRIMARY_CHECKPOINTS))
        self.assertEqual([row["train_size_int"] for row in rows], [200, 200, 200])
        self.assertEqual([row["heldout_mean_normalized_gap_float"] for row in rows], [0.30, 0.20, 0.25])

    def test_best_primary_checkpoint_by_setting_uses_lowest_normalized_gap(self):
        module = load_module()

        rows = [
            {
                "regime": "step2c_poly_d8_mult_eps050",
                "train_size_int": 1200,
                "checkpoint_label": "2stage_val_mse",
                "test_mean_normalized_gap_float": 0.10,
            },
            {
                "regime": "step2c_poly_d8_mult_eps050",
                "train_size_int": 1200,
                "checkpoint_label": "fy_val_fy_loss",
                "test_mean_normalized_gap_float": 0.07,
            },
            {
                "regime": "step2c_poly_d8_mult_eps050",
                "train_size_int": 1200,
                "checkpoint_label": "spoplus_val_spoplus_loss",
                "test_mean_normalized_gap_float": 0.08,
            },
        ]

        winners = module.best_primary_checkpoint_by_setting(rows)

        self.assertEqual(winners[("step2c_poly_d8_mult_eps050", 1200)]["checkpoint_label"], "fy_val_fy_loss")
        self.assertAlmostEqual(winners[("step2c_poly_d8_mult_eps050", 1200)]["test_mean_normalized_gap_float"], 0.07)


if __name__ == "__main__":
    unittest.main()
