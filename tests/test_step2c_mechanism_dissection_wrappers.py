from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = PROJECT_ROOT / "surrogate_experiment_results" / "Step2c Mechanism Dissection Audit"


class Step2cMechanismDissectionWrapperTests(unittest.TestCase):
    def test_experiment_directory_exposes_predicted_topm_entrypoint(self):
        script = AUDIT_DIR / "scripts" / "compute_predicted_topm_solutions.py"
        self.assertTrue(script.exists(), f"Missing wrapper script: {script}")

        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=PROJECT_ROOT,
            check=False,
            text=True,
            capture_output=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Compute best and second-best", result.stdout)

    def test_experiment_directory_exposes_oracle_landscape_entrypoint(self):
        script = AUDIT_DIR / "scripts" / "compute_true_oracle_landscape.py"
        self.assertTrue(script.exists(), f"Missing wrapper script: {script}")

        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=PROJECT_ROOT,
            check=False,
            text=True,
            capture_output=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Enumerate true-label top-M", result.stdout)

    def test_experiment_directory_exposes_mechanism_summary_entrypoint(self):
        script = AUDIT_DIR / "scripts" / "summarize_step2c_mechanism_dissection.py"
        self.assertTrue(script.exists(), f"Missing wrapper script: {script}")

        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=PROJECT_ROOT,
            check=False,
            text=True,
            capture_output=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Build Step2c mechanism-dissection", result.stdout)

    def test_experiment_directory_exposes_presentation_entrypoint(self):
        script = AUDIT_DIR / "scripts" / "build_presentation_artifacts.py"
        self.assertTrue(script.exists(), f"Missing wrapper script: {script}")

        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=PROJECT_ROOT,
            check=False,
            text=True,
            capture_output=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Build presentation tables and figures", result.stdout)


if __name__ == "__main__":
    unittest.main()
