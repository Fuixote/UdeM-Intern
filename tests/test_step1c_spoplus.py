import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


STEP1C_DIR = (
    Path(__file__).resolve().parents[1]
    / "surrogate_experiment_results"
    / "Step1c"
)


def load_module(filename, name):
    module_path = STEP1C_DIR / filename
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeStep1a:
    def __init__(self):
        self.seen_weights = []

    def solve_once(self, weights, graph, env):
        weights = np.asarray(weights, dtype=float)
        self.seen_weights.append(weights.copy())
        solutions = graph["solutions"]
        scores = solutions @ weights
        return solutions[int(np.argmax(scores))].copy()


def decision_gap(record, theta, oracle):
    w_hat = record["X"] @ np.asarray(theta, dtype=float)
    y_pred = oracle.solve_once(w_hat, record["graph"], env=None)
    optimal = float(np.dot(record["w_true"], record["y_optimal"]))
    achieved = float(np.dot(record["w_true"], y_pred))
    return optimal - achieved


def make_two_edge_record(w_true=(3.0, 1.0)):
    solutions = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    w_true = np.asarray(w_true, dtype=float)
    graph = {"solutions": solutions}
    y_optimal = solutions[int(np.argmax(solutions @ w_true))].copy()
    return {
        "filename": "fake-G-0.json",
        "X": np.eye(2, dtype=float),
        "w_true": w_true,
        "y_optimal": y_optimal,
        "graph": graph,
    }


class Step1cSpoPlusTest(unittest.TestCase):
    def test_spoplus_loss_zero_at_perfect_prediction_on_small_graph(self):
        common = load_module("step1c_common.py", "step1c_common_spoplus_zero")
        fake = FakeStep1a()
        common.load_step1a_module = lambda: fake
        record = make_two_edge_record(w_true=(3.0, 1.0))

        loss, grad = common.spo_plus_loss_and_grad(
            record, theta=np.array([3.0, 1.0]), env=None
        )

        self.assertAlmostEqual(loss, 0.0, places=9)
        np.testing.assert_allclose(grad, np.zeros(2), atol=1e-9)

    def test_spoplus_upper_bounds_decision_gap_for_random_theta(self):
        common = load_module("step1c_common.py", "step1c_common_spoplus_bound")
        fake = FakeStep1a()
        common.load_step1a_module = lambda: fake
        record = make_two_edge_record(w_true=(3.0, 1.0))
        theta = np.array([-1.0, 2.0])

        loss, _ = common.spo_plus_loss_and_grad(record, theta=theta, env=None)
        gap = decision_gap(record, theta, fake)

        self.assertGreaterEqual(loss + 1e-9, gap)

    def test_spoplus_shifted_weights_can_be_negative(self):
        common = load_module("step1c_common.py", "step1c_common_spoplus_negative")
        fake = FakeStep1a()
        common.load_step1a_module = lambda: fake
        record = make_two_edge_record(w_true=(3.0, 1.0))

        common.spo_plus_loss_and_grad(record, theta=np.array([-1.0, 2.0]), env=None)

        shifted_weights = fake.seen_weights[0]
        self.assertLess(float(np.min(shifted_weights)), 0.0)

    def test_spoplus_gradient_shape_and_finite_values(self):
        common = load_module("step1c_common.py", "step1c_common_spoplus_grad")
        fake = FakeStep1a()
        common.load_step1a_module = lambda: fake
        record = make_two_edge_record(w_true=(3.0, 1.0))

        loss, grad = common.spo_plus_loss_and_grad(
            record, theta=np.array([-1.0, 2.0]), env=None
        )

        self.assertTrue(np.isfinite(loss))
        self.assertEqual(grad.shape, (2,))
        self.assertTrue(np.all(np.isfinite(grad)))

    def test_combined_spoplus_diagnostics_reuses_pred_and_adversarial_solves(self):
        common = load_module("step1c_common.py", "step1c_common_spoplus_diag")
        fake = FakeStep1a()
        common.load_step1a_module = lambda: fake
        record = make_two_edge_record(w_true=(3.0, 1.0))
        trajectory = np.array([[3.0, 1.0], [-1.0, 2.0]], dtype=float)

        rows = common.evaluate_trajectory_spoplus_diagnostics(
            trajectory,
            [record],
            env=None,
            indices=np.array([0, 1]),
            label="test",
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(len(fake.seen_weights), 4)
        self.assertIn("decision_gap", rows[0])
        self.assertIn("spoplus_loss", rows[0])
        self.assertIn("normalized_spoplus_loss", rows[0])
        self.assertIn("y_adv_oracle_equal_rate", rows[0])
        self.assertIn("y_pred_oracle_equal_rate", rows[0])

    def test_spoplus_checkpoint_selection_uses_validation_spoplus_loss(self):
        train_spoplus = load_module("train_spoplus.py", "step1c_train_spoplus_select")
        trajectory = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        validation_spoplus_loss = np.array([0.7, 0.2, 0.5])

        checkpoint = train_spoplus.select_best_spoplus_loss_checkpoint(
            trajectory, validation_spoplus_loss
        )

        self.assertEqual(checkpoint["method"], "spoplus")
        self.assertEqual(checkpoint["epoch"], 1)
        np.testing.assert_allclose(checkpoint["theta"], np.array([1.0, 1.0]))
        self.assertEqual(checkpoint["selection_metric"], "validation_spoplus_loss")
        self.assertEqual(checkpoint["selection_value"], 0.2)

    def test_spoplus_model_weights_use_spoplus_method_and_filename(self):
        train_spoplus = load_module("train_spoplus.py", "step1c_train_spoplus_weights")
        checkpoint = {
            "method": "spoplus",
            "epoch": 3,
            "theta": np.array([9.0, 4.0]),
            "selection_metric": "validation_spoplus_loss",
            "selection_value": 1.25,
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = train_spoplus.write_model_weights(
                Path(tmp), checkpoint, train_size=50
            )
            payload = np.load(path, allow_pickle=False)

        self.assertEqual(path.name, "spoplus_best_by_validation_spoplus_loss.npz")
        self.assertEqual(payload["method"].item(), "spoplus")
        self.assertEqual(payload["selection_metric"].item(), "validation_spoplus_loss")
        self.assertEqual(int(payload["selected_epoch"].item()), 3)
        np.testing.assert_allclose(payload["theta"], np.array([9.0, 4.0]))

    def test_run_step1c_calls_spoplus_trainer_and_evaluates_spoplus_weights(self):
        script = (STEP1C_DIR / "run_step1c.sh").read_text(encoding="utf-8")

        self.assertIn("train_spoplus.py", script)
        self.assertIn("spoplus_best_by_validation_decision_gap.npz", script)
        self.assertIn("spoplus_best_by_validation_spoplus_loss.npz", script)
        self.assertNotIn("train_end2end.py", script)
        self.assertNotIn("e2e_best_by_validation_fy_loss.npz", script)

    def test_evaluate_unseen_run_expects_spoplus_weights(self):
        evaluate_unseen = load_module(
            "evaluate_unseen_run.py", "step1c_evaluate_unseen_spoplus"
        )

        self.assertEqual(
            evaluate_unseen.EXPECTED_WEIGHT_FILES,
            [
                "2stage_best_by_validation_mse_loss.npz",
                "spoplus_best_by_validation_decision_gap.npz",
                "spoplus_best_by_validation_spoplus_loss.npz",
            ],
        )

    def test_2stage_parser_accepts_smoke_graph_limits(self):
        train_2stage = load_module("train_2stage.py", "step1c_train_2stage_limits")

        args = train_2stage.parse_args(
            ["--train_graph_limit", "2", "--validation_limit", "3"]
        )

        self.assertEqual(args.train_graph_limit, 2)
        self.assertEqual(args.validation_limit, 3)

    def test_evaluate_models_parser_accepts_test_limit(self):
        evaluate_models = load_module("evaluate_models.py", "step1c_eval_limit")

        args = evaluate_models.parse_args(
            ["--weights", "a.npz", "b.npz", "--test_limit", "4"]
        )

        self.assertEqual(args.test_limit, 4)


if __name__ == "__main__":
    unittest.main()
