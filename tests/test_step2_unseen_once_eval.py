import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "surrogate_experiment_results"
    / "evaluate_step2_unseen_once.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("evaluate_step2_unseen_once", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Step2UnseenOnceEvalTest(unittest.TestCase):
    def test_default_regime_specs_cover_step2_abc(self):
        module = load_module()

        specs = module.default_regime_specs()
        regimes = {spec.regime for spec in specs}

        self.assertEqual(len(specs), 9)
        self.assertIn("step2a_additive_rho050", regimes)
        self.assertIn("step2b_poly_d8", regimes)
        self.assertIn("step2c_poly_d8_mult_eps050", regimes)
        self.assertTrue(
            all("unseen10000" in spec.dataset_dir.name for spec in specs)
        )

    def test_discover_setting_specs_builds_five_checkpoint_evaluation_plan(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            regime = "step2b_poly_d8"
            remote_root = root / "remote_results" / regime
            fy_weights = (
                remote_root
                / "step1b_fy"
                / module.DEFAULT_FY_RUN_TAG
                / "train_size=50"
                / "model_weights"
            )
            spo_weights = (
                remote_root
                / "step1c_spoplus"
                / module.DEFAULT_SPOPLUS_RUN_TAG
                / "train_size=50"
                / "model_weights"
            )
            fy_weights.mkdir(parents=True)
            spo_weights.mkdir(parents=True)

            for filename in module.FY_WEIGHT_FILES:
                (fy_weights / filename).write_text("", encoding="utf-8")
            for filename in module.SPOPLUS_WEIGHT_FILES:
                (spo_weights / filename).write_text("", encoding="utf-8")

            spec = module.RegimeSpec(
                block="Step2b",
                regime=regime,
                degree="8",
                dataset_dir=root / "dataset" / f"{regime}_unseen10000_seed20260523",
                remote_results_root=remote_root,
            )
            settings, warnings = module.discover_setting_specs(
                regime_specs=[spec],
                train_sizes=[50],
            )

        self.assertEqual(warnings, [])
        self.assertEqual(len(settings), 1)
        self.assertEqual(settings[0].regime, regime)
        self.assertEqual(settings[0].train_size, 50)
        self.assertEqual(len(settings[0].weights), 5)
        self.assertEqual(
            [weight.label for weight in settings[0].weights],
            [
                "2stage_val_mse",
                "fy_val_decision_gap",
                "fy_val_fy_loss",
                "spoplus_val_decision_gap",
                "spoplus_val_spoplus_loss",
            ],
        )


if __name__ == "__main__":
    unittest.main()
