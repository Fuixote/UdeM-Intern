import importlib.util
import tempfile
import unittest
from pathlib import Path


STEP1B_DIR = (
    Path(__file__).resolve().parents[1]
    / "surrogate_experiment_results"
    / "Step1b"
)


def load_module():
    module_path = STEP1B_DIR / "plot_result_summary.py"
    spec = importlib.util.spec_from_file_location("step1b_result_summary_plot", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Step1bResultSummaryPlotTest(unittest.TestCase):
    def test_bootstrap_mean_ci_centers_on_sample_mean(self):
        plot_summary = load_module()

        mean_value, low, high = plot_summary.bootstrap_mean_ci(
            [0.0, 1.0, 2.0, 3.0],
            n_bootstrap=200,
            seed=7,
        )

        self.assertAlmostEqual(mean_value, 1.5)
        self.assertLessEqual(low, mean_value)
        self.assertGreaterEqual(high, mean_value)

    def test_direct_bar_error_stats_reads_per_graph_normalized_gap(self):
        plot_summary = load_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics_dir = root / "train_size=50" / "metrics"
            metrics_dir.mkdir(parents=True)
            (metrics_dir / "test_per_graph.csv").write_text(
                "method,selection_metric,graph,normalized_gap\n"
                "2stage,validation_mse_loss,G-1.json,0.10\n"
                "2stage,validation_mse_loss,G-2.json,0.30\n"
                "e2e,validation_fy_loss,G-1.json,0.05\n",
                encoding="utf-8",
            )

            stats = plot_summary.direct_bar_error_stats(
                root,
                [50],
                n_bootstrap=50,
                seed=3,
            )

        mean_value, low, high = stats[
            ("heldout400", 50, ("2stage", "validation_mse_loss"))
        ]
        self.assertAlmostEqual(mean_value, 0.20)
        self.assertLessEqual(low, mean_value)
        self.assertGreaterEqual(high, mean_value)
        self.assertIn(
            ("heldout400", 50, ("e2e", "validation_fy_loss")),
            stats,
        )


if __name__ == "__main__":
    unittest.main()
