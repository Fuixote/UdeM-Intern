import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from surrogate_experiment_results.SPO_validation.step1c_vs_pyepo import (
    validation_core,
)
from surrogate_experiment_results.SPO_validation.step1c_vs_pyepo import (
    run_my_spoplus,
)
from surrogate_experiment_results.SPO_validation.step1c_vs_pyepo import (
    compare_my_vs_pyepo_csv,
)


class Step1cVsPyepoRunnerTests(unittest.TestCase):
    def test_result_csv_path_matches_planned_layout(self):
        path = validation_core.result_csv_path(
            result_root=Path("/tmp/results"),
            method_slug="my-2stage-lr",
            grid=(5, 5),
            train_size=100,
            feat=5,
            deg=1,
            noise=0.0,
            lan="gurobi",
        )

        self.assertEqual(
            path,
            Path(
                "/tmp/results/sp/h5w5/gurobi/"
                "n100p5-d1-e0.0_my-2stage-lr.csv"
            ),
        )

    def test_default_sp_settings_cover_240_seeded_runs(self):
        settings = list(validation_core.iter_sp_settings(expnum=10))

        self.assertEqual(len(settings), 240)
        self.assertEqual(
            settings[0],
            validation_core.SPSetting(train_size=100, deg=1, noise=0.0, seed=0),
        )
        self.assertEqual(
            settings[-1],
            validation_core.SPSetting(train_size=5000, deg=6, noise=0.5, seed=9),
        )

    def test_append_result_row_preserves_pyepo_schema_and_resume_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "res.csv"
            validation_core.append_result_row(
                path,
                {
                    "True SPO": 0.1,
                    "Unamb SPO": 0.2,
                    "MSE": 0.3,
                    "Elapsed": 0.4,
                    "Epochs": 0,
                },
            )
            validation_core.append_result_row(
                path,
                {
                    "True SPO": 1.1,
                    "Unamb SPO": 1.2,
                    "MSE": 1.3,
                    "Elapsed": 1.4,
                    "Epochs": 10,
                },
            )

            df = pd.read_csv(path)
            self.assertEqual(list(df.columns), validation_core.RESULT_COLUMNS)
            self.assertEqual(validation_core.completed_seed_count(path), 2)
            np.testing.assert_allclose(df["True SPO"].to_numpy(), [0.1, 1.1])

    def test_step1c_linear_model_uses_numpy_weight_convention(self):
        model = run_my_spoplus.Step1cLinearModel(feat_dim=2, cost_dim=3)
        weight = np.array([[0.5, -1.0, 2.0], [1.5, 0.25, -0.5]])
        bias = np.array([0.1, -0.2, 0.3])
        model.set_numpy_params(weight, bias)

        x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        pred = model.predict_numpy(x)

        np.testing.assert_allclose(
            pred,
            validation_core.linear_predict(x, weight, bias),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_compare_csv_pair_reports_max_mean_and_worst_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            left = Path(tmp) / "left.csv"
            right = Path(tmp) / "right.csv"
            pd.DataFrame(
                {
                    "True SPO": [0.1, 0.2],
                    "Unamb SPO": [0.3, 0.4],
                    "MSE": [1.0, 2.0],
                    "Elapsed": [10.0, 11.0],
                    "Epochs": [0, 0],
                }
            ).to_csv(left, index=False)
            pd.DataFrame(
                {
                    "True SPO": [0.1, 0.25],
                    "Unamb SPO": [0.33, 0.4],
                    "MSE": [1.5, 2.0],
                    "Elapsed": [9.0, 12.0],
                    "Epochs": [0, 1],
                }
            ).to_csv(right, index=False)

            report = compare_my_vs_pyepo_csv.compare_csv_pair(
                left,
                right,
                columns=["True SPO", "Unamb SPO", "MSE", "Epochs"],
                tolerances={
                    "True SPO": 0.1,
                    "Unamb SPO": 0.1,
                    "MSE": 0.6,
                    "Epochs": 0.0,
                },
            )

            self.assertEqual(report.row_count, 2)
            self.assertAlmostEqual(report.stats["True SPO"].max_abs_diff, 0.05)
            self.assertAlmostEqual(report.stats["True SPO"].mean_abs_diff, 0.025)
            self.assertEqual(report.stats["True SPO"].worst_row, 1)
            self.assertFalse(report.passed)
            self.assertFalse(report.stats["Epochs"].passed)

    def test_compare_csv_pair_rejects_row_count_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            left = Path(tmp) / "left.csv"
            right = Path(tmp) / "right.csv"
            pd.DataFrame(
                {
                    "True SPO": [0.1],
                    "Unamb SPO": [0.3],
                    "MSE": [1.0],
                    "Elapsed": [10.0],
                    "Epochs": [0],
                }
            ).to_csv(left, index=False)
            pd.DataFrame(
                {
                    "True SPO": [0.1, 0.2],
                    "Unamb SPO": [0.3, 0.4],
                    "MSE": [1.0, 2.0],
                    "Elapsed": [10.0, 11.0],
                    "Epochs": [0, 0],
                }
            ).to_csv(right, index=False)

            with self.assertRaises(ValueError):
                compare_my_vs_pyepo_csv.compare_csv_pair(left, right)

    def test_compare_csv_pair_can_limit_rows_for_smoke_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            left = Path(tmp) / "left.csv"
            right = Path(tmp) / "right.csv"
            pd.DataFrame(
                {
                    "True SPO": [0.1, 0.9],
                    "Unamb SPO": [0.3, 0.9],
                    "MSE": [1.0, 9.0],
                    "Elapsed": [10.0, 11.0],
                    "Epochs": [0, 0],
                }
            ).to_csv(left, index=False)
            pd.DataFrame(
                {
                    "True SPO": [0.1],
                    "Unamb SPO": [0.3],
                    "MSE": [1.0],
                    "Elapsed": [10.0],
                    "Epochs": [0],
                }
            ).to_csv(right, index=False)

            report = compare_my_vs_pyepo_csv.compare_csv_pair(
                left,
                right,
                limit_rows=1,
            )

            self.assertEqual(report.row_count, 1)
            self.assertTrue(report.passed)


if __name__ == "__main__":
    unittest.main()
