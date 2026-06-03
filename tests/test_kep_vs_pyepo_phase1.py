import unittest

import numpy as np

from surrogate_experiment_results.SPO_validation.kep_vs_pyepo import (
    kep_validation_core,
    validate_kep_small_setting,
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


class KepVsPyepoPhase1Tests(unittest.TestCase):
    def test_kep_cost_min_wrapper_views_reward_oracle_through_negative_cost(self):
        record = make_record()
        step1a = FakeStep1a()
        model = kep_validation_core.KepCostMinOptModel(record, step1a=step1a, env=None)
        cost = np.array([-0.25, -2.0, -1.0], dtype=float)

        model.setObj(cost)
        sol, obj = model.solve()

        expected = step1a.solve_once(-cost, record["graph"], env=None)
        np.testing.assert_allclose(sol, expected)
        self.assertAlmostEqual(obj, float(np.dot(cost, expected)))

    def test_phase1_spoplus_reference_matches_step1c_on_fake_kep_record(self):
        record = make_record()
        theta = np.array([0.25, 1.25], dtype=float)

        check = kep_validation_core.phase1_spoplus_record_check(
            record,
            theta,
            step1a=FakeStep1a(),
            env=None,
            sgd_lr=0.05,
        )

        self.assertLessEqual(check["level2_forward_loss_abs_diff"], 1e-6)
        self.assertLessEqual(check["level3_grad_pred_max_abs_diff"], 1e-6)
        self.assertLessEqual(check["level3_grad_theta_max_abs_diff"], 1e-6)
        self.assertLessEqual(check["level4_sgd_theta_update_max_abs_diff"], 1e-6)

    def test_phase1_cli_parser_uses_guided_small_setting_defaults(self):
        args = validate_kep_small_setting.build_parser().parse_args([])

        self.assertEqual(args.graph_count, 5)
        self.assertEqual(args.theta_seed, 42)
        self.assertEqual(args.gurobi_seed, 42)
        self.assertEqual(args.sgd_lr, 0.05)


if __name__ == "__main__":
    unittest.main()
