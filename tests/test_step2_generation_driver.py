import os
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATION_SCRIPT = ROOT / "surrogate_experiment_results" / "Step2" / "run_generate_step2abc_datasets.sh"


class Step2GenerationDriverTest(unittest.TestCase):
    def test_dry_run_lists_expected_step2abc_datasets(self):
        result = subprocess.run(
            ["bash", str(GENERATION_SCRIPT)],
            cwd=ROOT,
            env={**os.environ, "DRY_RUN": "1"},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        expected_names = [
            "step2a_additive_rho050_main2000_seed20260523",
            "step2a_additive_rho050_val2000_seed20260523",
            "step2a_additive_rho050_unseen10000_seed20260523",
            "step2b_poly_d1_main2000_seed20260523",
            "step2b_poly_d2_val2000_seed20260523",
            "step2b_poly_d4_unseen10000_seed20260523",
            "step2b_poly_d8_unseen10000_seed20260523",
            "step2c_poly_d1_mult_eps050_main2000_seed20260523",
            "step2c_poly_d2_mult_eps050_val2000_seed20260523",
            "step2c_poly_d4_mult_eps050_unseen10000_seed20260523",
            "step2c_poly_d8_mult_eps050_unseen10000_seed20260523",
        ]
        for name in expected_names:
            self.assertIn(name, result.stdout)
        self.assertIn("validate_step2_processed_dataset.py", result.stdout)
        self.assertIn("--strict --expected_graph_count 2000 --expected_label_mode step2a_additive_linear_gaussian", result.stdout)
        self.assertIn("--strict --expected_graph_count 10000 --expected_label_mode step2c_polynomial_degree_multiplicative_noise", result.stdout)

    def test_dry_run_uses_parameterized_rho_and_epsilon_tags(self):
        result = subprocess.run(
            ["bash", str(GENERATION_SCRIPT)],
            cwd=ROOT,
            env={**os.environ, "DRY_RUN": "1", "STEP2_RHO": "0.25", "STEP2_EPSILON_BAR": "0.25"},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("step2a_additive_rho025_main2000_seed20260523", result.stdout)
        self.assertIn("step2c_poly_d1_mult_eps025_main2000_seed20260523", result.stdout)
        self.assertNotIn("step2a_additive_rho050_main2000_seed20260523", result.stdout)
        self.assertNotIn("step2c_poly_d1_mult_eps050_main2000_seed20260523", result.stdout)


if __name__ == "__main__":
    unittest.main()
