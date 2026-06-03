import tempfile
import unittest
from pathlib import Path

import numpy as np

from surrogate_experiment_results.SPO_validation.kep_vs_pyepo import (
    kep_validation_core,
    validate_step2b_degree_trajectory,
)


class FakeStep1a:
    def solve_once(self, weights, graph, env):
        weights = np.asarray(weights, dtype=float)
        solutions = np.asarray(graph["solutions"], dtype=float)
        return solutions[int(np.argmax(solutions @ weights))].copy()


def make_lr_record(filename, x, theta_true):
    x = np.asarray(x, dtype=float)
    theta_true = np.asarray(theta_true, dtype=float)
    solutions = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    w_true = x @ theta_true
    y_optimal = solutions[int(np.argmax(solutions @ w_true))].copy()
    return {
        "filename": filename,
        "X": x,
        "w_true": w_true,
        "y_optimal": y_optimal,
        "graph": {"solutions": solutions},
    }


class Step2bBridgeTests(unittest.TestCase):
    def test_cli_defaults_match_small_bridge_plan(self):
        args = validate_step2b_degree_trajectory.build_parser().parse_args([])

        self.assertEqual(args.degrees, [1, 2, 4, 8])
        self.assertEqual(args.source_train_size, 50)
        self.assertEqual(args.train_size, 5)
        self.assertEqual(args.validation_size, 5)
        self.assertEqual(args.n_epochs, 3)
        self.assertEqual(args.metric_stride, 1)
        self.assertEqual(args.lr, 0.1)
        self.assertEqual(args.theta_seed, 42)
        self.assertEqual(args.gurobi_seed, 42)

    def test_step2b_artifacts_resolve_existing_formal_split_files(self):
        artifacts = validate_step2b_degree_trajectory.step2b_artifacts_for_degree(
            degree=4,
            source_train_size=50,
        )

        self.assertEqual(artifacts.regime, "step2b_poly_d4")
        self.assertTrue(artifacts.train_subset_json.exists())
        self.assertTrue(artifacts.validation_set_json.exists())
        self.assertTrue(artifacts.run_config_json.exists())
        self.assertIn("step2b_poly_d4", str(artifacts.run_dir))
        self.assertIn("train_size=50", str(artifacts.run_dir))

    def test_load_and_truncate_entries_preserves_saved_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "entries.json"
            path.write_text(
                "["
                '{"index": 7, "graph_id": 7, "path": "G-7.json"},'
                '{"index": 3, "graph_id": 3, "path": "G-3.json"},'
                '{"index": 9, "graph_id": 9, "path": "G-9.json"}'
                "]",
                encoding="utf-8",
            )

            entries = validate_step2b_degree_trajectory.load_and_truncate_entries(
                path,
                limit=2,
            )

        self.assertEqual([entry["graph_id"] for entry in entries], [7, 3])
        self.assertEqual([entry["path"] for entry in entries], ["G-7.json", "G-3.json"])

    def test_aggregate_payload_passes_only_when_all_degrees_pass(self):
        payload = validate_step2b_degree_trajectory.aggregate_bridge_payload(
            degree_payloads=[
                {"degree": 1, "passed": True, "results": []},
                {"degree": 2, "passed": True, "results": []},
                {"degree": 4, "passed": False, "results": []},
            ],
            args=validate_step2b_degree_trajectory.build_parser().parse_args([]),
        )

        self.assertEqual(payload["phase"], "step2b_bridge")
        self.assertEqual(payload["degrees"], [1, 2, 4])
        self.assertFalse(payload["passed"])

    def test_aggregate_payload_tracks_lr_as_a_bridge_method(self):
        payload = validate_step2b_degree_trajectory.aggregate_bridge_payload(
            degree_payloads=[
                {
                    "degree": 1,
                    "passed": True,
                    "spoplus_passed": True,
                    "lr_passed": False,
                    "results": [],
                    "lr_results": [],
                },
            ],
            args=validate_step2b_degree_trajectory.build_parser().parse_args([]),
        )

        self.assertEqual(payload["methods"], ["lr", "spoplus"])
        self.assertFalse(payload["passed"])

    def test_paired_lr_bridge_checks_theta_predictions_mse_and_gap(self):
        theta_true = np.array([2.0, 0.5], dtype=float)
        train_records = [
            make_lr_record(
                "train-1",
                [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                theta_true,
            ),
            make_lr_record(
                "train-2",
                [[2.0, 0.0], [0.0, 2.0], [1.0, 2.0]],
                theta_true,
            ),
        ]
        validation_records = [
            make_lr_record(
                "validation-1",
                [[1.0, 0.5], [0.5, 1.0], [1.5, 0.5]],
                theta_true,
            )
        ]

        bridge = kep_validation_core.run_paired_lr_bridge(
            train_records,
            validation_records,
            step1a=FakeStep1a(),
            env=None,
        )

        self.assertTrue(bridge["passed"])
        result_names = {item["name"] for item in bridge["results"]}
        self.assertIn("lr_theta_max_abs_diff", result_names)
        self.assertIn("lr_train_prediction_max_abs_diff", result_names)
        self.assertIn("lr_validation_prediction_max_abs_diff", result_names)
        self.assertIn("lr_train_mse_diff", result_names)
        self.assertIn("lr_validation_mse_diff", result_names)
        self.assertIn("lr_validation_decision_gap_diff", result_names)
        self.assertIn("lr_validation_normalized_gap_diff", result_names)
        np.testing.assert_allclose(bridge["pyepo_theta"], theta_true, atol=1e-12)
        np.testing.assert_allclose(bridge["step1c_theta"], theta_true, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
