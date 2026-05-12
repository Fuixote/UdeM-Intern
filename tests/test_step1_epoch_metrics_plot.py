import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


def load_plot_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "surrogate_experiment_results"
        / "Step1a"
        / "plot_epoch_metrics.py"
    )
    spec = importlib.util.spec_from_file_location("plot_epoch_metrics", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Step1EpochMetricsPlotTest(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("matplotlib"),
        "matplotlib is required for Step1 epoch metric plotting",
    )
    def test_plot_epoch_metrics_writes_png_from_fy_trajectory_only(self):
        plot_epoch_metrics = load_plot_module()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fy_path = tmp_path / "trajectory_fy_with_fy_loss_and_regret.npy"
            out_path = tmp_path / "trajectory_epoch_metrics.png"

            np.save(
                fy_path,
                np.array(
                    [
                        [1.0, 3.0, 5.0, 4.0],
                        [2.0, 2.0, 3.0, 1.5],
                        [3.0, 1.5, 2.0, 1.0],
                    ]
                ),
            )

            plot_epoch_metrics.plot_epoch_metrics(
                fy_path=fy_path,
                out_path=out_path,
                title="epsilon=unit-test",
            )

            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
