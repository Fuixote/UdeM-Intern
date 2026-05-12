import importlib.util
import tempfile
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


class Step1bRedesignedPipelineTest(unittest.TestCase):
    def test_master_split_keeps_1200_400_400_indices_reproducibly(self):
        split_step1b = load_module("split_dataset.py", "step1b_split_dataset")
        files = [Path(f"G-{idx}.json") for idx in range(2000)]

        first = split_step1b.make_master_split(
            files, train_pool_size=1200, val_size=400, test_size=400, seed=42
        )
        second = split_step1b.make_master_split(
            files, train_pool_size=1200, val_size=400, test_size=400, seed=42
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first["train_pool"]), 1200)
        self.assertEqual(len(first["validation"]), 400)
        self.assertEqual(len(first["test"]), 400)
        self.assertIn("graph_id", first["train_pool"][0])
        all_paths = [
            item["path"]
            for split_name in ("train_pool", "validation", "test")
            for item in first[split_name]
        ]
        self.assertEqual(len(all_paths), len(set(all_paths)))

    def test_training_subset_is_reproducible_subset_of_train_pool(self):
        split_step1b = load_module("split_dataset.py", "step1b_split_dataset")
        train_pool = [
            {"index": idx, "graph_id": idx, "path": f"G-{idx}.json"}
            for idx in range(20)
        ]

        first = split_step1b.select_train_subset(train_pool, train_size=5, seed=7)
        second = split_step1b.select_train_subset(train_pool, train_size=5, seed=7)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        self.assertTrue({item["path"] for item in first} <= {item["path"] for item in train_pool})

    def test_mse_checkpoint_selection_uses_validation_mse_loss(self):
        train_2stage = load_module("train_2stage.py", "step1b_train_2stage")
        trajectory = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        val_mse = np.array([0.4, 0.1, 0.3])

        checkpoint = train_2stage.select_best_mse_checkpoint(trajectory, val_mse)

        self.assertEqual(checkpoint["epoch"], 1)
        np.testing.assert_allclose(checkpoint["theta"], np.array([1.0, 1.0]))
        self.assertEqual(checkpoint["selection_metric"], "validation_mse_loss")
        self.assertEqual(checkpoint["selection_value"], 0.1)

    def test_end2end_checkpoint_selection_uses_validation_decision_gap(self):
        train_end2end = load_module("train_end2end.py", "step1b_train_end2end")
        trajectory = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        val_gap = np.array([0.8, 0.9, 0.2])

        checkpoint = train_end2end.select_best_decision_gap_checkpoint(
            trajectory, val_gap
        )

        self.assertEqual(checkpoint["epoch"], 2)
        np.testing.assert_allclose(checkpoint["theta"], np.array([2.0, 2.0]))
        self.assertEqual(checkpoint["selection_metric"], "validation_decision_gap")
        self.assertEqual(checkpoint["selection_value"], 0.2)

    def test_end2end_checkpoint_selection_can_use_validation_fy_loss(self):
        train_end2end = load_module("train_end2end.py", "step1b_train_end2end")
        trajectory = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        val_fy = np.array([0.8, 0.1, 0.2])

        checkpoint = train_end2end.select_best_fy_loss_checkpoint(trajectory, val_fy)

        self.assertEqual(checkpoint["epoch"], 1)
        np.testing.assert_allclose(checkpoint["theta"], np.array([1.0, 1.0]))
        self.assertEqual(checkpoint["selection_metric"], "validation_fy_loss")
        self.assertEqual(checkpoint["selection_value"], 0.1)

    def test_evaluation_summary_marks_train_size_and_method(self):
        evaluate_models = load_module("evaluate_models.py", "step1b_evaluate_models")
        evaluations = [
            {"graph": "G-1.json", "gap": 2.0, "normalized_gap": 0.2, "ratio": 0.8},
            {"graph": "G-2.json", "gap": 1.0, "normalized_gap": 0.1, "ratio": 0.9},
        ]

        summary = evaluate_models.summarize_model_evaluations(
            method="2stage",
            train_size=50,
            theta=np.array([10.0, 5.0]),
            selection_metric="validation_mse_loss",
            selection_value=0.12,
            evaluations=evaluations,
        )

        self.assertEqual(summary["method"], "2stage")
        self.assertEqual(summary["train_size"], 50)
        self.assertEqual(summary["selection_metric"], "validation_mse_loss")
        self.assertAlmostEqual(summary["test_mean_decision_gap"], 1.5)
        self.assertAlmostEqual(summary["test_mean_normalized_gap"], 0.15)

    def test_paired_stats_compare_against_2stage_by_graph(self):
        evaluate_models = load_module("evaluate_models.py", "step1b_evaluate_models")
        baseline = [
            {"graph": "G-1.json", "gap": 3.0},
            {"graph": "G-2.json", "gap": 2.0},
            {"graph": "G-3.json", "gap": 1.0},
        ]
        candidate = [
            {"graph": "G-1.json", "gap": 1.0},
            {"graph": "G-2.json", "gap": 2.5},
            {"graph": "G-3.json", "gap": 0.0},
        ]

        stats = evaluate_models.paired_improvement_stats(
            candidate, baseline, n_bootstrap=200, seed=3
        )

        self.assertAlmostEqual(stats["paired_mean_improvement_over_2stage"], (2.0 - 0.5 + 1.0) / 3.0)
        self.assertAlmostEqual(stats["paired_median_improvement_over_2stage"], 1.0)
        self.assertAlmostEqual(stats["fraction_improved_over_2stage"], 2.0 / 3.0)
        self.assertLessEqual(
            stats["paired_mean_improvement_ci_low"],
            stats["paired_mean_improvement_over_2stage"],
        )
        self.assertGreaterEqual(
            stats["paired_mean_improvement_ci_high"],
            stats["paired_mean_improvement_over_2stage"],
        )

    def test_evaluation_summary_keeps_each_model_path(self):
        evaluate_models = load_module("evaluate_models.py", "step1b_evaluate_models")
        evaluations = [
            {"graph": "G-1.json", "gap": 2.0, "normalized_gap": 0.2, "ratio": 0.8},
        ]
        models_and_evaluations = [
            (
                {
                    "path": "model_weights/2stage_best_by_validation_mse_loss.npz",
                    "method": "2stage",
                    "train_size": 50,
                    "theta": np.array([1.0, 2.0]),
                    "selection_metric": "validation_mse_loss",
                    "selection_value": 0.5,
                    "selected_epoch": 1,
                },
                evaluations,
            ),
            (
                {
                    "path": "model_weights/e2e_best_by_validation_fy_loss.npz",
                    "method": "e2e",
                    "train_size": 50,
                    "theta": np.array([3.0, 4.0]),
                    "selection_metric": "validation_fy_loss",
                    "selection_value": 0.4,
                    "selected_epoch": 2,
                },
                evaluations,
            ),
        ]

        rows = evaluate_models.summarize_evaluated_models(
            models_and_evaluations,
            bootstrap_samples=10,
            bootstrap_seed=1,
        )

        self.assertEqual(
            rows[0]["model_path"],
            "model_weights/2stage_best_by_validation_mse_loss.npz",
        )
        self.assertEqual(
            rows[1]["model_path"],
            "model_weights/e2e_best_by_validation_fy_loss.npz",
        )

    def test_run_dir_name_includes_key_hyperparameters(self):
        run_step1b = load_module("run_config.py", "step1b_run_config")

        run_dir = run_step1b.default_run_dir(
            output_root="results/step1b_runs",
            train_size=50,
            split_seed=42,
            subset_seed=7,
            theta_seed=11,
            fy_epsilon=1.0,
            fy_M=4,
            e2e_epochs=100,
            metric_stride=5,
        )

        self.assertIn("train_size=50", run_dir)
        self.assertIn("split_seed=42", run_dir)
        self.assertIn("subset_seed=7", run_dir)
        self.assertIn("theta_seed=11", run_dir)
        self.assertIn("eps=1.0", run_dir)
        self.assertIn("M=4", run_dir)
        self.assertIn("e2e_epochs=100", run_dir)
        self.assertIn("stride=5", run_dir)

    def test_training_curve_plot_writes_png_from_loss_csv(self):
        try:
            import matplotlib  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("matplotlib is not installed in this Python environment")

        plot_curves = load_module("plot_training_curves.py", "step1b_plot_curves")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "loss_curve.csv"
            out_path = tmp_path / "loss_curve.png"
            csv_path.write_text(
                "epoch,train_mse_loss,validation_mse_loss\n"
                "0,2.0,2.5\n"
                "1,1.0,1.8\n",
                encoding="utf-8",
            )

            written = plot_curves.plot_loss_curve(
                csv_path,
                out_path,
                train_column="train_mse_loss",
                validation_column="validation_mse_loss",
                ylabel="MSE loss",
                title="2stage MSE loss",
            )

            self.assertEqual(written, out_path)
            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
