import unittest

import numpy as np

from surrogate_experiment_results.SPO_validation.kep_vs_pyepo import (
    kep_validation_core,
    validate_kep_full_trajectory,
)


class FakeStep1a:
    def solve_once(self, weights, graph, env):
        weights = np.asarray(weights, dtype=float)
        solutions = np.asarray(graph["solutions"], dtype=float)
        return solutions[int(np.argmax(solutions @ weights))].copy()


def make_record(name, w_true):
    solutions = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    w_true = np.asarray(w_true, dtype=float)
    y_optimal = solutions[int(np.argmax(solutions @ w_true))].copy()
    return {
        "filename": name,
        "X": np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            dtype=float,
        ),
        "w_true": w_true,
        "y_optimal": y_optimal,
        "graph": {"solutions": solutions},
    }


class KepVsPyepoPhase2Tests(unittest.TestCase):
    def test_phase2_loss_curve_rows_use_step1c_spoplus_schema(self):
        trajectory = np.asarray([[1.0, 2.0], [1.5, 2.5]], dtype=float)
        train_diagnostics = [
            {
                "spoplus_loss": 10.0,
                "normalized_spoplus_loss": 0.1,
                "decision_gap": 1.0,
                "normalized_decision_gap": 0.01,
                "y_adv_oracle_equal_rate": 0.2,
                "y_pred_oracle_equal_rate": 0.3,
            },
            {
                "spoplus_loss": 8.0,
                "normalized_spoplus_loss": 0.08,
                "decision_gap": 0.5,
                "normalized_decision_gap": 0.005,
                "y_adv_oracle_equal_rate": 0.4,
                "y_pred_oracle_equal_rate": 0.5,
            },
        ]
        validation_diagnostics = [
            {
                "spoplus_loss": 11.0,
                "normalized_spoplus_loss": 0.11,
                "decision_gap": 1.1,
                "normalized_decision_gap": 0.011,
                "y_adv_oracle_equal_rate": 0.6,
                "y_pred_oracle_equal_rate": 0.7,
            },
            {
                "spoplus_loss": 9.0,
                "normalized_spoplus_loss": 0.09,
                "decision_gap": 0.6,
                "normalized_decision_gap": 0.006,
                "y_adv_oracle_equal_rate": 0.8,
                "y_pred_oracle_equal_rate": 0.9,
            },
        ]

        rows = kep_validation_core.phase2_loss_curve_rows(
            trajectory,
            train_diagnostics,
            validation_diagnostics,
            epoch_indices=np.asarray([0, 2], dtype=int),
        )

        self.assertEqual(list(rows[0].keys()), kep_validation_core.PHASE2_LOSS_CURVE_FIELDS)
        self.assertEqual(rows[1]["epoch"], 2)
        self.assertEqual(rows[1]["theta_1"], 1.5)
        self.assertEqual(rows[1]["theta_2"], 2.5)
        self.assertAlmostEqual(rows[1]["validation_decision_gap"], 0.6)
        self.assertAlmostEqual(rows[1]["validation_normalized_decision_gap"], 0.006)

    def test_phase2_compare_loss_curves_reports_field_max_abs_diffs(self):
        pyepo_rows = [
            {
                "epoch": 0,
                "theta_1": 1.0,
                "theta_2": 2.0,
                "theta_norm": np.sqrt(5.0),
                "train_spoplus_loss": 4.0,
                "validation_spoplus_loss": 5.0,
                "train_normalized_spoplus_loss": 0.4,
                "validation_normalized_spoplus_loss": 0.5,
                "train_decision_gap": 1.0,
                "validation_decision_gap": 2.0,
                "train_normalized_decision_gap": 0.1,
                "validation_normalized_decision_gap": 0.2,
                "train_y_adv_oracle_equal_rate": 0.0,
                "validation_y_adv_oracle_equal_rate": 0.0,
                "train_y_pred_oracle_equal_rate": 1.0,
                "validation_y_pred_oracle_equal_rate": 1.0,
            }
        ]
        step1c_rows = [dict(pyepo_rows[0])]
        step1c_rows[0]["validation_spoplus_loss"] += 2.5e-6
        step1c_rows[0]["theta_2"] += 5.0e-7

        summary = kep_validation_core.phase2_compare_loss_curves(
            pyepo_rows,
            step1c_rows,
        )

        self.assertTrue(summary["passed"])
        by_name = {item["name"]: item for item in summary["results"]}
        self.assertAlmostEqual(
            by_name["phase2_validation_spoplus_loss_max_abs_diff"]["value"],
            2.5e-6,
        )
        self.assertAlmostEqual(
            by_name["phase2_theta_2_max_abs_diff"]["value"],
            5.0e-7,
        )

    def test_phase2_fake_kep_paired_adam_trajectory_matches(self):
        train_records = [
            make_record("fake-G-1.json", [2.0, 0.5, 1.0]),
            make_record("fake-G-2.json", [0.4, 1.5, 1.0]),
        ]
        validation_records = [make_record("fake-G-3.json", [1.0, 2.0, 0.25])]

        payload = kep_validation_core.run_phase2_paired_spoplus_trajectory(
            train_records,
            validation_records,
            theta_init=np.asarray([0.25, 1.25], dtype=float),
            n_epochs=2,
            lr=0.05,
            metric_stride=1,
            step1a=FakeStep1a(),
            env=None,
        )

        self.assertTrue(payload["summary"]["passed"])
        self.assertEqual(len(payload["pyepo_rows"]), 3)
        by_name = {item["name"]: item for item in payload["summary"]["results"]}
        self.assertLessEqual(by_name["phase2_theta_1_max_abs_diff"]["value"], 1e-6)
        self.assertLessEqual(
            by_name["phase2_validation_spoplus_loss_max_abs_diff"]["value"],
            1e-5,
        )

    def test_phase2_cli_parser_uses_guided_defaults(self):
        args = validate_kep_full_trajectory.build_parser().parse_args([])

        self.assertEqual(args.train_size, 5)
        self.assertEqual(args.validation_size, 5)
        self.assertEqual(args.n_epochs, 3)
        self.assertEqual(args.metric_stride, 1)
        self.assertEqual(args.theta_seed, 42)
        self.assertEqual(args.gurobi_seed, 42)
        self.assertEqual(args.optimizer, "adam")


if __name__ == "__main__":
    unittest.main()
