import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


RAW_PAYLOAD = {
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


class Step2ProcessingSmokeIntegrationTest(unittest.TestCase):
    def run_processor(self, relative_script, label_mode, extra_args, expected_fields):
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw_batch"
            out_dir = Path(tmp) / "processed"
            raw_dir.mkdir()
            (raw_dir / "genjson-0.json").write_text(json.dumps(RAW_PAYLOAD), encoding="utf-8")

            cmd = [
                "python",
                str(ROOT / relative_script),
                str(raw_dir),
                str(out_dir),
                "--file",
                "genjson-0.json",
                "--label_mode",
                label_mode,
                "--label_seed",
                "20260523",
                "--output_as_batch_dir",
                *extra_args,
            ]
            result = subprocess.run(
                cmd,
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr + result.stdout)
            for artifact in ("G-0.json", "run_info.json", "batch_summary.json", "batch_report.md"):
                self.assertTrue((out_dir / artifact).exists(), msg=f"missing {artifact}")
            run_info = json.loads((out_dir / "run_info.json").read_text(encoding="utf-8"))
            self.assertEqual(run_info["processor_script"], relative_script)
            payload = json.loads((out_dir / "G-0.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["metadata"]["ground_truth_label_mode"], label_mode)
            match = payload["data"]["P1"]["matches"][0]
            self.assertIsNotNone(match["ground_truth_label"])
            for field in expected_fields:
                self.assertIn(field, match)

    def test_step2a_cli_smoke_writes_artifacts(self):
        self.run_processor(
            "surrogate_experiment_results/Step2/Step2a_additive_linear_gaussian/data-processing.py",
            "step2a_additive_linear_gaussian",
            ["--step2a_noise_rho", "0.5"],
            ["latent_clean_linear_label", "label_noise_value", "step2a_noise_sigma"],
        )

    def test_step2b_cli_smoke_writes_artifacts(self):
        self.run_processor(
            "surrogate_experiment_results/Step2/Step2b_polynomial_degree_noiseless/data-processing.py",
            "step2b_polynomial_degree_noiseless",
            ["--step2b_degree", "2", "--step2b_kappa", "3", "--step2b_delta", "0"],
            ["latent_clean_linear_label", "step2b_polynomial_label", "step2b_polynomial_score"],
        )

    def test_step2c_cli_smoke_writes_artifacts(self):
        self.run_processor(
            "surrogate_experiment_results/Step2/Step2c_polynomial_degree_multiplicative_noise/data-processing.py",
            "step2c_polynomial_degree_multiplicative_noise",
            [
                "--step2c_degree",
                "2",
                "--step2c_kappa",
                "3",
                "--step2c_delta",
                "0",
                "--step2c_epsilon_bar",
                "0.5",
            ],
            ["latent_clean_linear_label", "step2c_polynomial_label", "step2c_multiplier"],
        )


if __name__ == "__main__":
    unittest.main()
