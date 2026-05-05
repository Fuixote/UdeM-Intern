import importlib.util
import tempfile
import unittest
from pathlib import Path


def load_data_processing_module():
    module_path = Path(__file__).resolve().parents[1] / "1-data-processing.py"
    spec = importlib.util.spec_from_file_location("data_processing", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dp = load_data_processing_module()


class NoisyCleanLinearLabelTest(unittest.TestCase):
    def test_noisy_clean_linear_label_is_deterministic_multiplicative(self):
        label_config = {
            "label_mode": dp.LABEL_MODE_NOISY_CLEAN_LINEAR_UTILITY_CPRA,
            "clean_linear_utility_weight": 10.0,
            "clean_linear_cpra_weight": 5.0,
            "clean_linear_noise_sigma": 0.10,
        }
        source_key = "genjson-0.json|1|P2|80"

        latent = dp.clean_linear_utility_cpra_label(80, 0.4, label_config)
        epsilon = dp.get_deterministic_epsilon(f"{source_key}|clean_linear_noise", 0.10)
        expected = round(max(0.0, latent * (1.0 + epsilon)), 4)

        first = dp.compute_ground_truth_label(
            label_config,
            expected_transplant_count=0.5,
            qaly=10.0,
            priority_multiplier=1.0,
            source_key=source_key,
            utility=80,
            cpra=0.4,
        )
        second = dp.compute_ground_truth_label(
            label_config,
            expected_transplant_count=0.9,
            qaly=20.0,
            priority_multiplier=1.5,
            source_key=source_key,
            utility=80,
            cpra=0.4,
        )

        self.assertEqual(first, expected)
        self.assertEqual(second, expected)

    def test_processing_config_advertises_noisy_clean_linear_metadata(self):
        label_config = {
            "label_mode": dp.LABEL_MODE_NOISY_CLEAN_LINEAR_UTILITY_CPRA,
            "clean_linear_utility_weight": 10.0,
            "clean_linear_cpra_weight": 5.0,
            "clean_linear_noise_sigma": 0.10,
        }

        config = dp.processing_config(label_config)

        self.assertIn(dp.LABEL_MODE_NOISY_CLEAN_LINEAR_UTILITY_CPRA, config["available_label_modes"])
        self.assertEqual(config["ground_truth_label_mode"], dp.LABEL_MODE_NOISY_CLEAN_LINEAR_UTILITY_CPRA)
        self.assertEqual(config["clean_linear_noise_sigma"], 0.10)
        self.assertEqual(
            config["clean_linear_noise_mode"],
            "deterministic_multiplicative_per_edge",
        )
        self.assertIn("deterministic_noise", config["label_formula"])

    def test_processed_payload_preserves_latent_and_noise_for_noisy_edges(self):
        label_config = {
            "label_mode": dp.LABEL_MODE_NOISY_CLEAN_LINEAR_UTILITY_CPRA,
            "clean_linear_utility_weight": 10.0,
            "clean_linear_cpra_weight": 5.0,
            "clean_linear_noise_sigma": 0.10,
        }
        raw_payload = {
            "data": {
                "1": {
                    "altruistic": False,
                    "sources": ["P1"],
                    "dage": 40,
                    "bloodtype": "A",
                    "matches": [
                        {
                            "recipient": "P2",
                            "utility": 80,
                            "recipient_cpra": 0.4,
                            "recipient_age": 50,
                            "donor_age": 40,
                            "donor_bt": "A",
                            "recipient_bt": "A",
                        }
                    ],
                },
                "2": {
                    "altruistic": False,
                    "sources": ["P2"],
                    "dage": 45,
                    "bloodtype": "A",
                    "matches": [],
                },
            },
            "recipients": {
                "P1": {
                    "age": 45,
                    "bloodtype": "A",
                    "cPRA": 0.2,
                    "hasBloodCompatibleDonor": True,
                },
                "P2": {
                    "age": 50,
                    "bloodtype": "A",
                    "cPRA": 0.4,
                    "hasBloodCompatibleDonor": True,
                },
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            input_file = Path(tmp) / "genjson-0.json"
            input_file.write_text(dp.json.dumps(raw_payload), encoding="utf-8")
            payload = dp.build_processed_payload(input_file, label_config)

        match = payload["data"]["P1"]["matches"][0]
        source_key = "genjson-0.json|1|P2|80"
        latent = dp.clean_linear_utility_cpra_label(80, 0.4, label_config)
        epsilon = dp.get_deterministic_epsilon(f"{source_key}|clean_linear_noise", 0.10)
        noisy = round(max(0.0, latent * (1.0 + epsilon)), 4)

        self.assertEqual(match["latent_clean_linear_label"], latent)
        self.assertEqual(match["label_noise_epsilon"], round(epsilon, 6))
        self.assertEqual(match["ground_truth_label"], noisy)


if __name__ == "__main__":
    unittest.main()
