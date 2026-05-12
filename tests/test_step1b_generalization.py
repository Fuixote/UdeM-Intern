import importlib.util
import unittest
from pathlib import Path

import numpy as np


STEP1B_DIR = (
    Path(__file__).resolve().parents[1]
    / "surrogate_experiment_results"
    / "Step1b"
)


def load_module(filename, name):
    module_path = STEP1B_DIR / filename
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Step1bGeneralizationHelpersTest(unittest.TestCase):
    def test_master_split_is_deterministic_and_disjoint(self):
        split_step1b = load_module("split_dataset.py", "step1b_split_dataset")
        files = [Path(f"G-{idx}.json") for idx in range(20)]

        first = split_step1b.make_master_split(
            files, train_pool_size=4, val_size=3, test_size=5, seed=7
        )
        second = split_step1b.make_master_split(
            files, train_pool_size=4, val_size=3, test_size=5, seed=7
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first["train_pool"]), 4)
        self.assertEqual(len(first["validation"]), 3)
        self.assertEqual(len(first["test"]), 5)

        train = {item["path"] for item in first["train_pool"]}
        validation = {item["path"] for item in first["validation"]}
        test = {item["path"] for item in first["test"]}
        self.assertFalse(train & validation)
        self.assertFalse(train & test)
        self.assertFalse(validation & test)

    def test_train_subset_rejects_oversized_request(self):
        split_step1b = load_module("split_dataset.py", "step1b_split_dataset")
        train_pool = [{"index": idx, "graph_id": idx, "path": f"G-{idx}.json"} for idx in range(3)]

        with self.assertRaisesRegex(ValueError, "exceeds train pool size"):
            split_step1b.select_train_subset(train_pool, train_size=4, seed=1)

    def test_run_mse_trajectory_starts_from_given_theta(self):
        train_2stage = load_module("train_2stage.py", "step1b_train_2stage")
        theta_init = np.array([0.5, 0.5])
        graphs = [{"X": np.eye(2), "w_true": np.array([3.0, 1.0])}]

        trajectory = train_2stage.run_mse_trajectory(
            graphs, theta_init=theta_init, n_epochs=2, lr=0.1
        )

        self.assertEqual(trajectory.shape, (3, 2))
        np.testing.assert_allclose(trajectory[0], theta_init)
        self.assertFalse(np.allclose(trajectory[-1], theta_init))

    def test_2stage_checkpoint_uses_validation_mse_loss(self):
        train_2stage = load_module("train_2stage.py", "step1b_train_2stage")
        trajectory = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        metrics = np.array([3.0, 0.5, 0.8])

        selected = train_2stage.select_best_mse_checkpoint(trajectory, metrics)

        self.assertEqual(selected["epoch"], 1)
        self.assertEqual(selected["selection_metric"], "validation_mse_loss")
        np.testing.assert_allclose(selected["theta"], np.array([1.0, 1.0]))

    def test_e2e_checkpoint_uses_validation_decision_gap(self):
        train_end2end = load_module("train_end2end.py", "step1b_train_end2end")
        trajectory = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        metrics = np.array([3.0, 0.5, 0.8])

        selected = train_end2end.select_best_decision_gap_checkpoint(
            trajectory, metrics
        )

        self.assertEqual(selected["epoch"], 1)
        self.assertEqual(selected["selection_metric"], "validation_decision_gap")
        np.testing.assert_allclose(selected["theta"], np.array([1.0, 1.0]))


if __name__ == "__main__":
    unittest.main()
