import tempfile
import unittest
from pathlib import Path

import numpy as np

from surrogate_experiment_results.SPO_validation.kep_vs_pyepo import (
    kep_validation_core,
    validate_kep_phase0,
)


class FakeStep1a:
    def solve_once(self, weights, graph, env):
        weights = np.asarray(weights, dtype=float)
        solutions = np.asarray(graph["solutions"], dtype=float)
        return solutions[int(np.argmax(solutions @ weights))].copy()


def make_record():
    solutions = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    w_true = np.array([2.0, 0.5, 1.0], dtype=float)
    y_optimal = solutions[int(np.argmax(solutions @ w_true))].copy()
    return {
        "filename": "fake-G-1.json",
        "X": np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float),
        "w_true": w_true,
        "y_optimal": y_optimal,
        "graph": {"solutions": solutions},
    }


class KepVsPyepoPhase0Tests(unittest.TestCase):
    def test_select_graph_paths_uses_numeric_graph_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ["G-10.json", "G-2.json", "G-1.json", "not-a-graph.json"]:
                (root / name).write_text("{}", encoding="utf-8")

            paths = kep_validation_core.select_graph_paths(root, graph_count=3)

        self.assertEqual([path.name for path in paths], ["G-1.json", "G-2.json", "G-10.json"])

    def test_sign_adapter_check_matches_reward_max_loss_and_theta_grad(self):
        record = make_record()
        theta = np.array([0.25, 1.25], dtype=float)

        check = kep_validation_core.sign_adapter_check(
            record,
            theta,
            step1a=FakeStep1a(),
            env=None,
        )

        self.assertEqual(check["num_edges"], 3)
        self.assertEqual(check["theta_dim"], 2)
        self.assertLessEqual(check["loss_abs_diff"], 1e-12)
        self.assertLessEqual(check["grad_pred_max_abs_diff"], 1e-12)
        self.assertLessEqual(check["grad_theta_max_abs_diff"], 1e-12)

    def test_phase0_cli_parser_defaults_to_small_kep_smoke(self):
        args = validate_kep_phase0.build_parser().parse_args([])

        self.assertEqual(args.graph_count, 2)
        self.assertEqual(args.gurobi_seed, 42)
        self.assertEqual(args.theta_seed, 42)
        self.assertTrue(str(args.data_dir).endswith("dataset/processed/step1_noisy_linear_sigma010"))


if __name__ == "__main__":
    unittest.main()
