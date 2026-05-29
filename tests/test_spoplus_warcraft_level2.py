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
    def test_readme_level2_entrypoints_exist(self):
        self.assertTrue((SPO_DIR / "03_run_warcraft_pyepo_reference.py").exists())
        self.assertTrue((SPO_DIR / "04_run_warcraft_our_spoplus.py").exists())
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

    def test_level2_entrypoints_expose_consistent_cli(self):
        pyepo_ref = load_module(
            SPO_DIR / "03_run_warcraft_pyepo_reference.py",
            "warcraft_level2_pyepo_reference_cli",
        )
        ours = load_module(
            SPO_DIR / "04_run_warcraft_our_spoplus.py",
            "warcraft_level2_our_spoplus_cli",
        )

        pyepo_options = {
            action.dest for action in pyepo_ref.build_arg_parser()._actions
        }
        ours_options = {action.dest for action in ours.build_arg_parser()._actions}
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

        self.assertEqual(pyepo_options, ours_options)
        self.assertTrue(expected.issubset(pyepo_options))

    def test_missing_default_warcraft_data_is_hard_failure(self):
        common = load_module(
            SPO_DIR / "warcraft_level2_common.py", "warcraft_level2_common_missing_data"
        )

        with self.assertRaises(FileNotFoundError):
            common.load_warcraft_arrays(SPO_DIR / "does-not-exist", grid_size=12)

    def test_output_schema_is_shared_by_reference_and_ours(self):
        common = load_module(
            SPO_DIR / "warcraft_level2_common.py", "warcraft_level2_common_schema"
        )
        pyepo_ref = load_module(
            SPO_DIR / "03_run_warcraft_pyepo_reference.py",
            "warcraft_level2_pyepo_reference_schema",
        )
        ours = load_module(
            SPO_DIR / "04_run_warcraft_our_spoplus.py",
            "warcraft_level2_our_spoplus_schema",
        )

        self.assertEqual(pyepo_ref.OUTPUT_FIELDS, common.OUTPUT_FIELDS)
        self.assertEqual(ours.OUTPUT_FIELDS, common.OUTPUT_FIELDS)

    def test_level2_scripts_have_no_failure_fallback_wording(self):
        for filename in [
            "03_run_warcraft_pyepo_reference.py",
            "04_run_warcraft_our_spoplus.py",
            "warcraft_level2_common.py",
        ]:
            source = (SPO_DIR / filename).read_text(encoding="utf-8").lower()
            self.assertNotIn("sk" + "ip", source)
            self.assertNotIn("missing " + "optional", source)
            self.assertNotIn("graceful", source)
            self.assertNotIn("fallback", source)


if __name__ == "__main__":
    unittest.main()
