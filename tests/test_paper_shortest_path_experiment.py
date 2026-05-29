import csv
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = (
    REPO_ROOT
    / "surrogate_experiment_results"
    / "SPO_validation"
    / "paper_shortest_path"
)


def load_common():
    spec = importlib.util.spec_from_file_location(
        "paper_shortest_path_common", PAPER_DIR / "common.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PaperShortestPathExperimentTest(unittest.TestCase):
    def test_paper_grid_and_data_generation_match_expected_shapes(self):
        common = load_common()
        config = common.PaperShortestPathConfig(n_train=5, n_val=3, n_test=4)

        instance = common.make_trial_instance(
            degree=2,
            noise_half_width=0.0,
            trial=0,
            seed=11,
            config=config,
        )

        self.assertEqual(common.paper_edge_count(config.grid_shape), 40)
        self.assertEqual(instance.b_star.shape, (40, 5))
        self.assertTrue(set(np.unique(instance.b_star)).issubset({0.0, 1.0}))
        self.assertEqual(instance.train.features.shape, (5, 5))
        self.assertEqual(instance.train.costs.shape, (5, 40))
        self.assertEqual(instance.train.opt_solutions.shape, (5, 40))
        self.assertEqual(instance.train.opt_objectives.shape, (5,))
        self.assertTrue(np.all(instance.train.costs > 0.0))

        manual = (
            (
                instance.train.features[0] @ instance.b_star.T / np.sqrt(config.feature_dim)
                + 3.0
            )
            ** 2
            + 1.0
        )
        np.testing.assert_allclose(instance.train.costs[0], manual, atol=1e-12)

    def test_normalized_spo_loss_is_zero_for_true_cost_predictions(self):
        common = load_common()
        config = common.PaperShortestPathConfig(n_train=3, n_val=2, n_test=4)
        instance = common.make_trial_instance(
            degree=1,
            noise_half_width=0.0,
            trial=0,
            seed=17,
            config=config,
        )

        metrics = common.evaluate_predictions(instance.test.costs, instance.test)

        self.assertAlmostEqual(metrics["normalized_spo_loss"], 0.0, places=12)
        self.assertAlmostEqual(metrics["avg_regret"], 0.0, places=12)
        self.assertAlmostEqual(metrics["optimality_ratio"], 1.0, places=12)

    def test_least_squares_recovers_linear_degree_one_without_noise(self):
        common = load_common()
        config = common.PaperShortestPathConfig(n_train=80, n_val=20, n_test=30)
        instance = common.make_trial_instance(
            degree=1,
            noise_half_width=0.0,
            trial=1,
            seed=23,
            config=config,
        )

        result = common.select_least_squares_model(
            instance.train,
            instance.val,
            lambdas=(0.0,),
        )
        predictions = common.predict_costs(instance.test.features, result.coefficients)
        metrics = common.evaluate_predictions(predictions, instance.test)

        self.assertLess(metrics["normalized_spo_loss"], 1e-10)
        self.assertEqual(result.selected_lambda, 0.0)

    def test_runner_writes_smoke_summary_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    str(PAPER_DIR / "run_paper_shortest_path.py"),
                    "--preset",
                    "smoke",
                    "--degrees",
                    "1",
                    "--noise-half-widths",
                    "0",
                    "--trials",
                    "1",
                    "--n-train",
                    "20",
                    "--n-val",
                    "8",
                    "--n-test",
                    "12",
                    "--lambda-grid",
                    "0",
                    "--methods",
                    "ls",
                    "ours-spoplus",
                    "--spoplus-iterations",
                    "3",
                    "--batch-size",
                    "5",
                    "--output-dir",
                    temp_dir,
                ],
                cwd=REPO_ROOT,
                check=False,
                text=True,
                capture_output=True,
            )

            self.assertEqual(
                result.returncode,
                0,
                f"runner failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
            )
            summary_path = Path(temp_dir) / "summary.csv"
            self.assertTrue(summary_path.exists())
            with summary_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(
                {row["implementation"] for row in rows},
                {"ls", "ours-spoplus"},
            )
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertEqual(row["degree"], "1")
                self.assertEqual(row["noise_half_width"], "0.0")
                self.assertTrue(float(row["test_norm_spo"]) >= 0.0)


if __name__ == "__main__":
    unittest.main()
