import importlib.util
import tempfile
import unittest
from pathlib import Path


def load_plot_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "surrogate_experiment_results"
        / "Step1b"
        / "plot_training_curves.py"
    )
    spec = importlib.util.spec_from_file_location("step1b_plot_curves", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Step1bPlotMetricsTest(unittest.TestCase):
    def test_plot_metrics_writes_png_from_train_validation_csv(self):
        try:
            import matplotlib  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("matplotlib is not installed in this Python environment")

        plot_curves = load_plot_module()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "loss_curve.csv"
            output_path = tmp_path / "loss_curve.png"
            csv_path.write_text(
                "epoch,train_loss,validation_loss\n"
                "0,2.0,2.5\n"
                "1,1.0,1.7\n",
                encoding="utf-8",
            )

            returned_path = plot_curves.plot_loss_curve(
                csv_path,
                output_path,
                train_column="train_loss",
                validation_column="validation_loss",
                ylabel="Loss",
                title="Loss curve",
            )

            self.assertEqual(returned_path, output_path)
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
