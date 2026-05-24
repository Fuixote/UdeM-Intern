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
            env={"DRY_RUN": "1", **dict()},
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
            "step2c_poly_d1_mult_eps050_main2000_seed20260523",
            "step2c_poly_d2_mult_eps050_val2000_seed20260523",
            "step2c_poly_d4_mult_eps050_unseen10000_seed20260523",
        ]
        for name in expected_names:
            self.assertIn(name, result.stdout)
        self.assertIn("validate_step2_processed_dataset.py", result.stdout)


if __name__ == "__main__":
    unittest.main()
