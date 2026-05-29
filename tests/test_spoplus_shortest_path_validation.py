import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "surrogate_experiment_results"
    / "SPO_validation"
    / "spoplus_shortest_path.py"
)


def load_validation_module():
    spec = importlib.util.spec_from_file_location(
        "spoplus_shortest_path_validation", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SpoPlusShortestPathValidationTest(unittest.TestCase):
    def test_perfect_prediction_has_zero_loss_and_gradient(self):
        sp = load_validation_module()
        grid_shape = (3, 3)
        c = np.array([1.0, 0.2, 1.4, 2.0, 0.1, 0.8, 1.3, 0.4, 1.1, 0.7, 0.6, 0.9])

        loss, grad = sp.spo_plus_min_loss_and_grad(c, c, grid_shape)

        self.assertAlmostEqual(loss, 0.0, places=10)
        np.testing.assert_allclose(grad, np.zeros_like(c), atol=1e-10)

    def test_spoplus_upper_bounds_spo_decision_gap(self):
        sp = load_validation_module()
        grid_shape = (4, 3)
        rng = np.random.RandomState(7)
        n_edges = len(sp.grid_edges(grid_shape))

        for _ in range(20):
            c = rng.uniform(-1.5, 2.0, size=n_edges)
            c_hat = rng.uniform(-2.0, 2.0, size=n_edges)
            z_true, true_obj = sp.solve_shortest_path(c, grid_shape)
            z_pred, _ = sp.solve_shortest_path(c_hat, grid_shape)

            loss, _ = sp.spo_plus_min_loss_and_grad(c_hat, c, grid_shape)
            decision_gap = float(np.dot(c, z_pred) - true_obj)

            self.assertGreaterEqual(loss + 1e-9, decision_gap)
            self.assertGreaterEqual(decision_gap + 1e-9, 0.0)
            self.assertEqual(float(np.dot(c, z_true)), true_obj)

    def test_analytical_gradient_matches_stable_directional_derivative(self):
        sp = load_validation_module()
        grid_shape = (3, 3)
        c = np.array([1.0, 4.0, 1.0, 4.0, 1.0, 4.0, 4.0, 1.0, 4.0, 1.0, 4.0, 1.0])
        c_hat = np.array([1.0, 3.5, 1.0, 3.5, 1.0, 3.5, 3.5, 1.0, 3.5, 1.0, 3.5, 1.0])
        direction = np.array([0.3, -0.2, 0.1, 0.05, -0.1, 0.2, -0.15, 0.25, 0.12, -0.08, 0.18, -0.05])

        loss, grad = sp.spo_plus_min_loss_and_grad(c_hat, c, grid_shape)
        eps = 1e-6
        loss_plus, _ = sp.spo_plus_min_loss_and_grad(c_hat + eps * direction, c, grid_shape)
        loss_minus, _ = sp.spo_plus_min_loss_and_grad(c_hat - eps * direction, c, grid_shape)
        finite_difference = (loss_plus - loss_minus) / (2.0 * eps)

        self.assertTrue(np.isfinite(loss))
        self.assertAlmostEqual(
            float(np.dot(grad, direction)), finite_difference, places=7
        )

    def test_reward_max_and_cost_min_sign_conversion(self):
        sp = load_validation_module()
        grid_shape = (3, 3)
        w = np.array([1.0, -0.2, 1.5, 0.4, 1.2, -0.8, 0.3, 1.7, -0.4, 1.1, 0.6, -0.1])
        w_hat = np.array([0.8, 0.1, 1.2, -0.3, 1.4, -0.6, 0.2, 1.3, -0.2, 0.9, 0.4, 0.3])

        max_loss, max_grad = sp.spo_plus_max_loss_and_grad(w_hat, w, grid_shape)
        min_loss, min_grad = sp.spo_plus_min_loss_and_grad(-w_hat, -w, grid_shape)

        self.assertAlmostEqual(max_loss, min_loss, places=10)
        np.testing.assert_allclose(max_grad, -min_grad, atol=1e-10)

    def test_negative_shifted_costs_are_supported(self):
        sp = load_validation_module()
        grid_shape = (3, 2)
        c = np.array([3.0, 2.0, 1.5, 0.8, 1.0, 2.5, 0.4])
        c_hat = np.array([-1.0, 0.2, -0.7, 0.3, -0.4, 0.1, -0.5])

        shifted = 2.0 * c_hat - c
        self.assertLess(float(np.min(shifted)), 0.0)

        loss, grad = sp.spo_plus_min_loss_and_grad(c_hat, c, grid_shape)

        self.assertTrue(np.isfinite(loss))
        self.assertTrue(np.all(np.isfinite(grad)))

    def test_tie_breaking_is_deterministic_and_documented(self):
        sp = load_validation_module()
        grid_shape = (2, 2)
        c = np.zeros(len(sp.grid_edges(grid_shape)))

        z_first, obj_first = sp.solve_shortest_path(c, grid_shape)
        z_second, obj_second = sp.solve_shortest_path(c, grid_shape)

        np.testing.assert_array_equal(z_first, z_second)
        self.assertEqual(obj_first, obj_second)
        np.testing.assert_array_equal(z_first, np.array([1.0, 0.0, 1.0, 0.0]))

    def test_readme_level1_scripts_exist_and_run(self):
        repo_root = Path(__file__).resolve().parents[1]
        script_dir = repo_root / "surrogate_experiment_results" / "SPO_validation"
        scripts = [
            "01_compare_spoplus_formula_toy_shortest_path.py",
            "02_compare_reward_max_sign_conversion.py",
        ]

        for script_name in scripts:
            script_path = script_dir / script_name
            self.assertTrue(script_path.exists(), f"missing {script_path}")
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=repo_root,
                check=False,
                text=True,
                capture_output=True,
            )
            self.assertEqual(
                result.returncode,
                0,
                f"{script_name} failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
            )

    def test_pyepo_comparison_has_no_failure_fallback(self):
        repo_root = Path(__file__).resolve().parents[1]
        script_path = (
            repo_root
            / "surrogate_experiment_results"
            / "SPO_validation"
            / "compare_with_pyepo_spoplus.py"
        )

        source = script_path.read_text(encoding="utf-8").lower()

        self.assertNotIn("sk" + "ip", source)
        self.assertNotIn("missing " + "optional", source)
        self.assertNotIn("un" + "available", source)


if __name__ == "__main__":
    unittest.main()
