import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STEP2A_SCRIPT = (
    ROOT
    / "surrogate_experiment_results"
    / "Step2"
    / "Step2a_additive_linear_gaussian"
    / "data-processing.py"
)
STEP2C_SCRIPT = (
    ROOT
    / "surrogate_experiment_results"
    / "Step2"
    / "Step2c_polynomial_degree_multiplicative_noise"
    / "data-processing.py"
)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Step2NoiseKeyProvenanceTest(unittest.TestCase):
    def test_step2a_label_source_key_includes_raw_batch_name(self):
        dp = load_module("step2a_data_processing", STEP2A_SCRIPT)

        key_a = dp.build_label_source_key(Path("/tmp/raw_a/genjson-0.json"), 1, "P2", 80)
        key_b = dp.build_label_source_key(Path("/tmp/raw_b/genjson-0.json"), 1, "P2", 80)

        self.assertEqual(key_a, "raw_a|genjson-0.json|1|P2|80")
        self.assertEqual(key_b, "raw_b|genjson-0.json|1|P2|80")
        self.assertNotEqual(key_a, key_b)

    def test_step2c_label_source_key_includes_raw_batch_name(self):
        dp = load_module("step2c_data_processing", STEP2C_SCRIPT)

        key_a = dp.build_label_source_key(Path("/tmp/raw_a/genjson-0.json"), 1, "P2", 80)
        key_b = dp.build_label_source_key(Path("/tmp/raw_b/genjson-0.json"), 1, "P2", 80)

        self.assertEqual(key_a, "raw_a|genjson-0.json|1|P2|80")
        self.assertEqual(key_b, "raw_b|genjson-0.json|1|P2|80")
        self.assertNotEqual(key_a, key_b)


if __name__ == "__main__":
    unittest.main()
