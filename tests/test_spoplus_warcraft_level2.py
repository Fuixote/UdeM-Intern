import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SPO_DIR = REPO_ROOT / "surrogate_experiment_results" / "SPO_validation"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class WarcraftLevel2FilesTest(unittest.TestCase):
    def test_canonical_level2_entrypoint_exists(self):
        self.assertTrue((SPO_DIR / "03_compare_warcraft_pyepo_vs_ours.py").exists())
        self.assertTrue((SPO_DIR / "warcraft_level2_common.py").exists())

    def test_default_config_matches_warcraft_notebook(self):
        common = load_module(
            SPO_DIR / "warcraft_level2_common.py", "warcraft_level2_common_defaults"
        )

        config = common.Level2Config()

        self.assertEqual(config.grid_size, 12)
        self.assertEqual(config.batch_size, 70)
        self.assertEqual(config.epochs, 50)
        self.assertEqual(config.seed, 135)
        self.assertAlmostEqual(config.learning_rate, 5e-4)
        self.assertEqual(config.log_step, 1)
        self.assertIsNone(config.train_limit)
        self.assertIsNone(config.test_limit)
        self.assertEqual(
            config.data_root,
            SPO_DIR / "warcraft_shortest_path_oneskin",
        )

    def test_level2_entrypoint_exposes_expected_cli(self):
        comparison = load_module(
            SPO_DIR / "03_compare_warcraft_pyepo_vs_ours.py",
            "warcraft_level2_comparison_cli",
        )

        comparison_options = {
            action.dest for action in comparison.build_arg_parser()._actions
        }
        expected = {
            "batch_size",
            "data_root",
            "device",
            "epochs",
            "grid_size",
            "learning_rate",
            "log_step",
            "output_root",
            "seed",
            "test_limit",
            "train_limit",
        }

        self.assertTrue(expected.issubset(comparison_options))

    def test_missing_default_warcraft_data_is_hard_failure(self):
        common = load_module(
            SPO_DIR / "warcraft_level2_common.py", "warcraft_level2_common_missing_data"
        )

        with self.assertRaises(FileNotFoundError):
            common.load_warcraft_arrays(SPO_DIR / "does-not-exist", grid_size=12)

    def test_output_schema_is_shared_by_comparison_entrypoint(self):
        common = load_module(
            SPO_DIR / "warcraft_level2_common.py", "warcraft_level2_common_schema"
        )
        comparison = load_module(
            SPO_DIR / "03_compare_warcraft_pyepo_vs_ours.py",
            "warcraft_level2_comparison_schema",
        )

        self.assertEqual(comparison.OUTPUT_FIELDS, common.OUTPUT_FIELDS)

    def test_our_spoplus_reuses_shared_step1c_core(self):
        source = (SPO_DIR / "warcraft_level2_common.py").read_text(encoding="utf-8")

        self.assertIn("cost_min_spoplus_loss", source)
        self.assertNotIn("class OurSPOPlusFunction", source)

    def test_level2_scripts_have_no_failure_fallback_wording(self):
        for filename in [
            "03_compare_warcraft_pyepo_vs_ours.py",
            "warcraft_level2_common.py",
        ]:
            source = (SPO_DIR / filename).read_text(encoding="utf-8").lower()
            self.assertNotIn("sk" + "ip", source)
            self.assertNotIn("missing " + "optional", source)
            self.assertNotIn("graceful", source)
            self.assertNotIn("fallback", source)


if __name__ == "__main__":
    unittest.main()
