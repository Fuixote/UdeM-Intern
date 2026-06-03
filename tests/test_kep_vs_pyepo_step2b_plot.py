import csv
import json
import tempfile
import unittest
from pathlib import Path

from surrogate_experiment_results.SPO_validation.kep_vs_pyepo import (
    plot_step2b_degree_overlay,
)


def write_spoplus_curve(path, values):
    rows = []
    for epoch, value in enumerate(values):
        rows.append(
            {
                "epoch": epoch,
                "theta_1": 1.0,
                "theta_2": 2.0,
                "theta_norm": 2.2360679,
                "validation_decision_gap": value * 100.0,
                "validation_normalized_decision_gap": value,
            }
        )
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


class Step2bOverlayPlotTests(unittest.TestCase):
    def test_collect_plot_rows_uses_lr_summary_and_final_spoplus_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pyepo_curve = tmp_path / "pyepo_spoplus.csv"
            step1c_curve = tmp_path / "step1c_spoplus.csv"
            write_spoplus_curve(pyepo_curve, [0.30, 0.20])
            write_spoplus_curve(step1c_curve, [0.31, 0.21])
            payload = {
                "source_train_size": 50,
                "train_size": 5,
                "validation_size": 5,
                "degree_results": [
                    {
                        "degree": 1,
                        "artifacts": {
                            "pyepo_loss_curve_csv": str(pyepo_curve),
                            "step1c_loss_curve_csv": str(step1c_curve),
                        },
                        "lr": {
                            "pyepo_summary": {"validation_normalized_gap": 0.10},
                            "step1c_summary": {"validation_normalized_gap": 0.10},
                        },
                    }
                ],
            }

            rows = plot_step2b_degree_overlay.collect_plot_rows(
                payload,
                metric="validation_normalized_gap",
            )

        self.assertEqual(
            [(row["method"], row["degree"], row["value"]) for row in rows],
            [
                ("pyepo_lr", 1, 0.10),
                ("pyepo_spoplus", 1, 0.20),
                ("step1c_lr", 1, 0.10),
                ("step1c_spoplus", 1, 0.21),
            ],
        )

    def test_run_writes_plot_and_csv_for_train_size_50_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output_dir = tmp_path / "plots"
            pyepo_curve = tmp_path / "pyepo_spoplus.csv"
            step1c_curve = tmp_path / "step1c_spoplus.csv"
            write_spoplus_curve(pyepo_curve, [0.40, 0.25])
            write_spoplus_curve(step1c_curve, [0.40, 0.25])
            bridge_json = tmp_path / "latest_step2b_bridge.json"
            bridge_json.write_text(
                json.dumps(
                    {
                        "source_train_size": 50,
                        "train_size": 5,
                        "validation_size": 5,
                        "degree_results": [
                            {
                                "degree": 1,
                                "artifacts": {
                                    "pyepo_loss_curve_csv": str(pyepo_curve),
                                    "step1c_loss_curve_csv": str(step1c_curve),
                                },
                                "lr": {
                                    "pyepo_summary": {
                                        "validation_normalized_gap": 0.15
                                    },
                                    "step1c_summary": {
                                        "validation_normalized_gap": 0.15
                                    },
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            args = plot_step2b_degree_overlay.build_parser().parse_args(
                [
                    "--bridge-json",
                    str(bridge_json),
                    "--output-dir",
                    str(output_dir),
                    "--source",
                    "bridge",
                    "--metric",
                    "validation_normalized_gap",
                    "--formats",
                    "png",
                ]
            )
            outputs = plot_step2b_degree_overlay.run(args)

            self.assertTrue(outputs["csv"].exists())
            self.assertEqual(len(outputs["plots"]), 1)
            self.assertTrue(outputs["plots"][0].exists())
            csv_text = outputs["csv"].read_text(encoding="utf-8")
            self.assertIn("PyEPO LR", csv_text)
            self.assertIn("my SPO+", csv_text)

    def test_collect_formal_summary_rows_mirrors_step1c_values_for_pyepo_overlay(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run_dir = tmp_path / "step2b_poly_d1" / "train_size=50"
            metrics_dir = run_dir / "metrics"
            metrics_dir.mkdir(parents=True)
            (metrics_dir / "test_summary.csv").write_text(
                "method,train_size,selected_epoch,theta_1,theta_2,selection_metric,"
                "selection_value,test_mean_decision_gap,test_mean_normalized_gap\n"
                "2stage,50,500,1.0,2.0,validation_mse_loss,0.1,10.0,0.10\n"
                "spoplus,50,200,3.0,4.0,validation_spoplus_loss,0.2,8.0,0.08\n",
                encoding="utf-8",
            )
            payload = {
                "source_train_size": 50,
                "degree_results": [
                    {
                        "degree": 1,
                        "source_run_dir": str(run_dir),
                    }
                ],
            }

            rows = plot_step2b_degree_overlay.collect_formal_summary_rows(
                payload,
                metric="test_mean_normalized_gap",
                spoplus_selection_metric="validation_spoplus_loss",
            )

        self.assertEqual(
            [(row["method"], row["degree"], row["value"]) for row in rows],
            [
                ("pyepo_lr", 1, 0.10),
                ("pyepo_spoplus", 1, 0.08),
                ("step1c_lr", 1, 0.10),
                ("step1c_spoplus", 1, 0.08),
            ],
        )


if __name__ == "__main__":
    unittest.main()
