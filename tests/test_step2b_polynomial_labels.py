import importlib.util
import tempfile
import unittest
from pathlib import Path


STEP2B_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "surrogate_experiment_results"
    / "Step2"
    / "Step2b_polynomial_degree_noiseless"
    / "data-processing.py"
)


def load_step2b_module():
    spec = importlib.util.spec_from_file_location("step2b_data_processing", STEP2B_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dp = load_step2b_module()


class Step2bPolynomialLabelTest(unittest.TestCase):
    def make_label_config(self, degree=2, delta=0.0):
        return {
            "label_mode": dp.LABEL_MODE_STEP2B_POLYNOMIAL_DEGREE_NOISELESS,
            "clean_linear_utility_weight": 10.0,
            "clean_linear_cpra_weight": 5.0,
            "clean_linear_noise_sigma": 0.08,
            "step2b_degree": degree,
            "step2b_kappa": 3.0,
            "step2b_delta": delta,
            "label_seed": 20260523,
        }

    def test_processing_config_advertises_step2b_polynomial_metadata(self):
        config = dp.processing_config(self.make_label_config(degree=4))

        self.assertIn(
            dp.LABEL_MODE_STEP2B_POLYNOMIAL_DEGREE_NOISELESS,
            config["available_label_modes"],
        )
        self.assertEqual(
            config["ground_truth_label_mode"],
            dp.LABEL_MODE_STEP2B_POLYNOMIAL_DEGREE_NOISELESS,
        )
        self.assertEqual(config["step2b_degree"], 4)
        self.assertEqual(config["step2b_kappa"], 3.0)
        self.assertEqual(config["step2b_rescale_mode"], "graph_mean_to_clean_linear_mean")
        self.assertIn("(q_e + kappa)^degree", config["label_formula"])

    def test_degree_one_recovers_clean_linear_label_under_rescaling(self):
        label_config = self.make_label_config(degree=1, delta=0.0)
        graph_label_context = {
            "clean_linear_mean": 6.0,
            "polynomial_score_mean": 1.0,
            "clean_linear_edge_count": 3,
        }

        fields = dp.compute_ground_truth_label_fields(
            label_config,
            expected_transplant_count=0.5,
            qaly=10.0,
            priority_multiplier=1.0,
            source_key="genjson-0.json|1|P2|80",
            utility=80,
            cpra=0.4,
            graph_label_context=graph_label_context,
        )

        self.assertEqual(fields["ground_truth_label"], 10.0)
        self.assertEqual(fields["latent_clean_linear_label"], 10.0)
        self.assertEqual(fields["step2b_polynomial_label"], 10.0)

    def test_degree_two_uses_graph_level_rescaling(self):
        label_config = self.make_label_config(degree=2, delta=0.0)
        clean_values = [3.0, 6.0, 9.0]
        mu = sum(clean_values) / len(clean_values)
        kappa = label_config["step2b_kappa"]
        polynomial_scores = [((b / mu) + kappa) ** 2 - kappa**2 for b in clean_values]
        mean_r = sum(polynomial_scores) / len(polynomial_scores)
        graph_label_context = {
            "clean_linear_mean": mu,
            "polynomial_score_mean": mean_r,
            "clean_linear_edge_count": len(clean_values),
        }

        labels = []
        for latent in clean_values:
            fields = dp.step2b_polynomial_degree_noiseless_components(
                latent,
                label_config,
                graph_label_context,
            )
            labels.append(fields["ground_truth_label"])

        self.assertAlmostEqual(sum(labels) / len(labels), mu, places=4)
        self.assertEqual(labels[-1], round(mu * polynomial_scores[-1] / mean_r, 4))
        self.assertGreater(labels[-1], clean_values[-1])

    def test_processed_payload_preserves_step2b_rescaling_diagnostics(self):
        label_config = self.make_label_config(degree=2, delta=0.0)
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
        self.assertEqual(match["step2b_graph_clean_linear_mean"], 10.0)
        self.assertEqual(match["step2b_graph_polynomial_score_mean"], 7.0)
        self.assertIn("step2b_q_value", match)
        self.assertIn("step2b_polynomial_score", match)
        self.assertEqual(
            payload["metadata"]["ground_truth_label_mode"],
            dp.LABEL_MODE_STEP2B_POLYNOMIAL_DEGREE_NOISELESS,
        )


if __name__ == "__main__":
    unittest.main()

