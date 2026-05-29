import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SPO_DIR = REPO_ROOT / "surrogate_experiment_results" / "SPO_validation"
STEP1C_COMMON_PATH = (
    REPO_ROOT / "surrogate_experiment_results" / "Step1c" / "step1c_common.py"
)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class KepSpoPlusCodePathValidationTest(unittest.TestCase):
    def test_reference_matches_step1c_code_path_loss_and_gradient(self):
        validation = load_module(
            SPO_DIR / "spoplus_kep_path_validation.py",
            "spoplus_kep_path_validation_match",
        )
        common = load_module(STEP1C_COMMON_PATH, "step1c_common_kep_path_match")
        record = validation.make_toy_kep_record()
        theta = np.array([1.1, -0.4], dtype=float)

        reference_oracle = validation.EnumerationRewardOracle()
        expected_loss, expected_grad = (
            validation.reference_spoplus_reward_max_loss_and_grad(
                record, theta, reference_oracle
            )
        )

        step1c_oracle = validation.EnumerationRewardOracle()
        common.load_step1a_module = lambda: step1c_oracle
        actual_loss, actual_grad = common.spo_plus_loss_and_grad(
            record, theta=theta, env=None
        )

        np.testing.assert_allclose(actual_loss, expected_loss, atol=1e-12)
        np.testing.assert_allclose(actual_grad, expected_grad, atol=1e-12)
        np.testing.assert_allclose(
            step1c_oracle.seen_weights[0],
            validation.shifted_reward_weights(record, theta),
            atol=1e-12,
        )

    def test_step1c_theta_gradient_matches_stable_finite_difference(self):
        validation = load_module(
            SPO_DIR / "spoplus_kep_path_validation.py",
            "spoplus_kep_path_validation_fd",
        )
        common = load_module(STEP1C_COMMON_PATH, "step1c_common_kep_path_fd")
        record = validation.make_toy_kep_record()
        theta = np.array([1.1, -0.4], dtype=float)
        direction = np.array([0.3, -0.2], dtype=float)

        common.load_step1a_module = validation.EnumerationRewardOracle

        _, grad = common.spo_plus_loss_and_grad(record, theta=theta, env=None)
        finite_difference = validation.finite_difference_directional_derivative(
            lambda value: common.spo_plus_loss_and_grad(
                record, theta=value, env=None
            )[0],
            theta,
            direction,
            epsilon=1e-6,
        )

        self.assertAlmostEqual(
            float(np.dot(grad, direction)), finite_difference, places=7
        )

    def test_level15_cli_entrypoint_runs(self):
        script = SPO_DIR / "05_validate_kep_spoplus_code_path.py"

        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=REPO_ROOT,
            check=False,
            text=True,
            capture_output=True,
        )

        self.assertEqual(
            result.returncode,
            0,
            f"Level 1.5 script failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        self.assertIn("KEP SPO+ Step1c code-path validation passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
