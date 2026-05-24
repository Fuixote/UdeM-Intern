import importlib.util
import tempfile
import unittest
from pathlib import Path


STEP2A_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "surrogate_experiment_results"
    / "Step2"
    / "Step2a_additive_linear_gaussian"
    / "data-processing.py"
)


def load_step2a_module():
    spec = importlib.util.spec_from_file_location("step2a_data_processing", STEP2A_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dp = load_step2a_module()


class Step2aAdditiveLabelTest(unittest.TestCase):
    def make_label_config(self):
        return {
            "label_mode": dp.LABEL_MODE_STEP2A_ADDITIVE_LINEAR_GAUSSIAN,
            "clean_linear_utility_weight": 10.0,
            "clean_linear_cpra_weight": 5.0,
            "clean_linear_noise_sigma": 0.08,
            "step2a_noise_rho": 0.5,
            "label_seed": 123,
        }

    def test_processing_config_advertises_step2a_additive_metadata(self):
        config = dp.processing_config(self.make_label_config())

        self.assertIn(
            dp.LABEL_MODE_STEP2A_ADDITIVE_LINEAR_GAUSSIAN,
            config["available_label_modes"],
        )
        self.assertEqual(
            config["ground_truth_label_mode"],
            dp.LABEL_MODE_STEP2A_ADDITIVE_LINEAR_GAUSSIAN,
        )
        self.assertEqual(config["step2a_noise_rho"], 0.5)
        self.assertEqual(config["label_seed"], 123)
        self.assertEqual(config["step2a_noise_mode"], "deterministic_additive_gaussian_per_edge")
        self.assertIn("rho * mu_G", config["label_formula"])

    def test_step2a_label_uses_additive_noise_scaled_by_graph_mean(self):
        label_config = self.make_label_config()
        source_key = "genjson-0.json|1|P2|80"
        graph_label_context = {
            "clean_linear_mean": 7.0,
            "clean_linear_edge_count": 3,
        }
        latent = dp.clean_linear_utility_cpra_label(80, 0.4, label_config)
        sigma = label_config["step2a_noise_rho"] * graph_label_context["clean_linear_mean"]
        noise = dp.get_deterministic_epsilon(
            f"{source_key}|step2a_additive_noise|label_seed=123",
            sigma,
        )
        expected = round(max(0.0, latent + noise), 4)

        fields = dp.compute_ground_truth_label_fields(
            label_config,
            expected_transplant_count=0.5,
            qaly=10.0,
            priority_multiplier=1.0,
            source_key=source_key,
            utility=80,
            cpra=0.4,
            graph_label_context=graph_label_context,
        )

        self.assertEqual(fields["ground_truth_label"], expected)
        self.assertEqual(fields["latent_clean_linear_label"], latent)
        self.assertEqual(fields["label_noise_value"], round(noise, 6))
        self.assertEqual(fields["step2a_graph_clean_linear_mean"], 7.0)
        self.assertEqual(fields["step2a_noise_sigma"], 3.5)

    def test_processed_payload_preserves_step2a_label_diagnostics(self):
        label_config = self.make_label_config()
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
        self.assertEqual(match["latent_clean_linear_label"], 10.0)
        self.assertEqual(match["step2a_graph_clean_linear_mean"], 10.0)
        self.assertIn("label_noise_value", match)
        self.assertIn("step2a_noise_sigma", match)
        self.assertEqual(
            payload["metadata"]["ground_truth_label_mode"],
            dp.LABEL_MODE_STEP2A_ADDITIVE_LINEAR_GAUSSIAN,
        )


if __name__ == "__main__":
    unittest.main()

