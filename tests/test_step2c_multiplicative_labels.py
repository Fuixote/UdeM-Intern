import importlib.util
import tempfile
import unittest
from pathlib import Path


STEP2C_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "surrogate_experiment_results"
    / "Step2"
    / "Step2c_polynomial_degree_multiplicative_noise"
    / "data-processing.py"
)


def load_step2c_module():
    spec = importlib.util.spec_from_file_location("step2c_data_processing", STEP2C_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dp = load_step2c_module()


class Step2cMultiplicativeLabelTest(unittest.TestCase):
    def make_label_config(self, degree=2, epsilon_bar=0.5, delta=0.0):
        return {
            "label_mode": dp.LABEL_MODE_STEP2C_POLYNOMIAL_DEGREE_MULTIPLICATIVE_NOISE,
            "clean_linear_utility_weight": 10.0,
            "clean_linear_cpra_weight": 5.0,
            "clean_linear_noise_sigma": 0.08,
            "step2c_degree": degree,
            "step2c_kappa": 3.0,
            "step2c_delta": delta,
            "step2c_epsilon_bar": epsilon_bar,
            "label_seed": 20260523,
        }

    def test_processing_config_advertises_step2c_metadata(self):
        config = dp.processing_config(self.make_label_config(degree=4))

        self.assertIn(
            dp.LABEL_MODE_STEP2C_POLYNOMIAL_DEGREE_MULTIPLICATIVE_NOISE,
            config["available_label_modes"],
        )
        self.assertEqual(
            config["ground_truth_label_mode"],
            dp.LABEL_MODE_STEP2C_POLYNOMIAL_DEGREE_MULTIPLICATIVE_NOISE,
        )
        self.assertEqual(config["step2c_degree"], 4)
        self.assertEqual(config["step2c_kappa"], 3.0)
        self.assertEqual(config["step2c_epsilon_bar"], 0.5)
        self.assertEqual(config["step2c_rescale_mode"], "graph_mean_to_clean_linear_mean_before_noise")
        self.assertIn("eta_e", config["label_formula"])

    def test_zero_epsilon_bar_matches_noiseless_polynomial_label(self):
        label_config = self.make_label_config(degree=2, epsilon_bar=0.0, delta=0.0)
        graph_label_context = {
            "clean_linear_mean": 6.0,
            "polynomial_score_mean": 7.0,
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

        self.assertEqual(fields["step2c_multiplier"], 1.0)
        self.assertEqual(fields["ground_truth_label"], fields["step2c_polynomial_label"])

    def test_multiplicative_noise_uses_deterministic_uniform_multiplier(self):
        label_config = self.make_label_config(degree=2, epsilon_bar=0.5, delta=0.0)
        source_key = "genjson-0.json|1|P2|80"
        graph_label_context = {
            "clean_linear_mean": 6.0,
            "polynomial_score_mean": 7.0,
            "clean_linear_edge_count": 3,
        }
        latent = dp.clean_linear_utility_cpra_label(80, 0.4, label_config)
        q_value, polynomial_score = dp.step2c_polynomial_score(
            latent,
            graph_label_context["clean_linear_mean"],
            label_config,
        )
        polynomial_label = round(
            graph_label_context["clean_linear_mean"]
            * polynomial_score
            / graph_label_context["polynomial_score_mean"],
            4,
        )
        multiplier = dp.get_deterministic_uniform(
            f"{source_key}|step2c_multiplicative_noise|label_seed=20260523",
            0.5,
            1.5,
        )
        expected = round(max(0.0, polynomial_label * multiplier), 4)

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
        self.assertEqual(fields["step2c_polynomial_label"], polynomial_label)
        self.assertEqual(fields["step2c_multiplier"], round(multiplier, 6))
        self.assertEqual(fields["step2c_epsilon_bar"], 0.5)
        self.assertEqual(fields["step2c_q_value"], round(q_value, 6))

    def test_processed_payload_preserves_step2c_diagnostics(self):
        label_config = self.make_label_config(degree=2, epsilon_bar=0.5, delta=0.0)
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
        self.assertEqual(match["step2c_graph_clean_linear_mean"], 10.0)
        self.assertEqual(match["step2c_graph_polynomial_score_mean"], 7.0)
        self.assertIn("step2c_polynomial_label", match)
        self.assertIn("step2c_multiplier", match)
        self.assertIn("step2c_noisy_polynomial_label", match)
        self.assertEqual(
            payload["metadata"]["ground_truth_label_mode"],
            dp.LABEL_MODE_STEP2C_POLYNOMIAL_DEGREE_MULTIPLICATIVE_NOISE,
        )


if __name__ == "__main__":
    unittest.main()

