import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "surrogate_experiment_results"
    / "plot_step1bc_final_comparison.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("plot_step1bc_final", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Step1bcFinalComparisonPlotTest(unittest.TestCase):
    def test_canonical_method_mapping_keeps_five_final_methods(self):
        plot = load_module()

        self.assertEqual(
            plot.canonical_method_id(
                "step1b",
                {"method": "2stage", "selection_metric": "validation_mse_loss"},
            ),
            "2stage_val_mse",
        )
        self.assertEqual(
            plot.canonical_method_id(
                "step1b",
                {"method": "e2e", "selection_metric": "validation_fy_loss"},
            ),
            "fy_val_fy",
        )
        self.assertEqual(
            plot.canonical_method_id(
                "step1c",
                {"method": "spoplus", "selection_metric": "validation_spoplus_loss"},
            ),
            "spoplus_val_spoplus",
        )
        self.assertIsNone(
            plot.canonical_method_id(
                "step1c",
                {"method": "e2e", "selection_metric": "validation_fy_loss"},
            )
        )

    def test_collect_summary_rows_prefers_step1b_2stage_and_uses_step1c_fallback(self):
        plot = load_module()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            step1b_root = tmp_path / "step1b"
            step1c_root = tmp_path / "step1c"

            b50 = step1b_root / "train_size=50" / "metrics"
            c50 = step1c_root / "train_size=50" / "metrics"
            c200 = step1c_root / "train_size=200" / "metrics"
            b50.mkdir(parents=True)
            c50.mkdir(parents=True)
            c200.mkdir(parents=True)

            header = (
                "method,train_size,selected_epoch,theta_1,theta_2,"
                "selection_metric,test_mean_normalized_gap\n"
            )
            (b50 / "test_summary.csv").write_text(
                header
                + "2stage,50,500,9.0,5.0,validation_mse_loss,0.010\n"
                + "e2e,50,40,6.0,3.0,validation_decision_gap,0.009\n",
                encoding="utf-8",
            )
            (c50 / "test_summary.csv").write_text(
                header
                + "2stage,50,500,99.0,99.0,validation_mse_loss,0.999\n"
                + "spoplus,50,80,7.0,4.0,validation_spoplus_loss,0.008\n",
                encoding="utf-8",
            )
            (c200 / "test_summary.csv").write_text(
                header
                + "2stage,200,500,10.0,5.0,validation_mse_loss,0.007\n",
                encoding="utf-8",
            )

            datasets = plot.default_dataset_specs("unseen10000")
            rows, warnings = plot.collect_summary_rows(
                [
                    plot.ResultSource("step1b", step1b_root),
                    plot.ResultSource("step1c", step1c_root),
                ],
                datasets,
                train_sizes=[50, 200],
            )

        self.assertFalse([warning for warning in warnings if "test_summary.csv" in warning])

        by_key = {
            (row["dataset_key"], int(row["train_size"]), row["method_id"]): row
            for row in rows
        }

        row_50_2stage = by_key[("heldout400", 50, "2stage_val_mse")]
        self.assertEqual(row_50_2stage["source"], "step1b")
        self.assertEqual(row_50_2stage["theta_1"], "9.0")

        row_200_2stage = by_key[("heldout400", 200, "2stage_val_mse")]
        self.assertEqual(row_200_2stage["source"], "step1c")
        self.assertEqual(row_200_2stage["theta_1"], "10.0")

        self.assertIn(("heldout400", 50, "fy_val_gap"), by_key)
        self.assertIn(("heldout400", 50, "spoplus_val_spoplus"), by_key)

    def test_plot_mean_without_error_bars_writes_png_and_pdf(self):
        plot = load_module()

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "figure0_mean_normalized_gap_no_error_bars.png"
            datasets = plot.default_dataset_specs("unseen10000")
            summary_rows = [
                {
                    "dataset_key": "heldout400",
                    "train_size": "50",
                    "method_id": "2stage_val_mse",
                    "test_mean_normalized_gap": "0.006",
                },
                {
                    "dataset_key": "unseen10000",
                    "train_size": "50",
                    "method_id": "2stage_val_mse",
                    "test_mean_normalized_gap": "0.005",
                },
            ]

            plot.plot_mean_normalized_gap_no_error_bar(
                summary_rows,
                datasets,
                train_sizes=[50],
                out_path=out_path,
            )

            self.assertTrue(out_path.exists())
            self.assertTrue(out_path.with_suffix(".pdf").exists())

    def test_plot_mean_with_error_bars_writes_grouped_bar_figure(self):
        plot = load_module()

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "figure1_mean_normalized_gap_heldout400_unseen10000.png"
            datasets = plot.default_dataset_specs("unseen10000")
            summary_rows = [
                {
                    "dataset_key": "heldout400",
                    "train_size": "50",
                    "method_id": "2stage_val_mse",
                    "test_mean_normalized_gap": "0.006",
                },
                {
                    "dataset_key": "heldout400",
                    "train_size": "50",
                    "method_id": "fy_val_gap",
                    "test_mean_normalized_gap": "0.0058",
                },
                {
                    "dataset_key": "unseen10000",
                    "train_size": "50",
                    "method_id": "2stage_val_mse",
                    "test_mean_normalized_gap": "0.005",
                },
            ]
            per_graph_rows = []
            for row in summary_rows:
                for idx, value in enumerate([0.0, float(row["test_mean_normalized_gap"]) * 2.0]):
                    per_graph_rows.append(
                        {
                            "dataset_key": row["dataset_key"],
                            "train_size": row["train_size"],
                            "method_id": row["method_id"],
                            "normalized_gap": str(value),
                            "graph": f"G-{idx}.json",
                        }
                    )

            plot.plot_mean_normalized_gap(
                summary_rows,
                per_graph_rows,
                datasets,
                train_sizes=[50],
                out_path=out_path,
                n_bootstrap=10,
                seed=1,
            )

            self.assertTrue(out_path.exists())
            self.assertTrue(out_path.with_suffix(".pdf").exists())


if __name__ == "__main__":
    unittest.main()
