import csv
import tempfile
import unittest
from pathlib import Path

from surrogate_experiment_results.SPO_validation.kep_vs_pyepo import (
    plot_step2b_seed_boxplots,
)


def make_seed_rows(degrees=(1, 2), seeds=(0, 1)):
    rows = []
    for degree in degrees:
        for seed in seeds:
            base = {
                "regime": f"step2b_poly_d{degree}",
                "block": "step2b",
                "degree": str(degree),
                "subset_seed": str(seed),
                "method": "",
                "selection_metric": "",
                "selected_epoch": "500",
                "test_mean_normalized_gap": "",
            }
            rows.append(
                {
                    **base,
                    "method_label": "2stage_val_mse",
                    "method": "2stage",
                    "selection_metric": "validation_mse_loss",
                    "test_mean_normalized_gap": str(0.01 * degree + 0.001 * seed),
                }
            )
            rows.append(
                {
                    **base,
                    "method_label": "spoplus_val_spoplus_loss",
                    "method": "spoplus",
                    "selection_metric": "validation_spoplus_loss",
                    "test_mean_normalized_gap": str(0.005 * degree + 0.0005 * seed),
                }
            )
            rows.append(
                {
                    **base,
                    "method_label": "spoplus_val_decision_gap",
                    "method": "spoplus",
                    "selection_metric": "validation_decision_gap",
                    "test_mean_normalized_gap": "999",
                }
            )
    rows.append(
        {
            "regime": "step2c_poly_d1_mult_eps050",
            "block": "step2c",
            "degree": "1",
            "subset_seed": "0",
            "method_label": "2stage_val_mse",
            "method": "2stage",
            "selection_metric": "validation_mse_loss",
            "selected_epoch": "500",
            "test_mean_normalized_gap": "888",
        }
    )
    return rows


def write_csv(path, rows):
    path = Path(path)
    fieldnames = [
        "regime",
        "block",
        "degree",
        "subset_seed",
        "method_label",
        "method",
        "selection_metric",
        "selected_epoch",
        "test_mean_normalized_gap",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class Step2bSeedBoxplotTests(unittest.TestCase):
    def test_default_per_seed_csv_points_to_existing_step2_resampling_summary(self):
        self.assertTrue(
            plot_step2b_seed_boxplots.DEFAULT_PER_SEED_CSV.exists(),
            plot_step2b_seed_boxplots.DEFAULT_PER_SEED_CSV,
        )
        self.assertIn(
            "surrogate_experiment_results/Step2_resampling",
            str(plot_step2b_seed_boxplots.DEFAULT_PER_SEED_CSV),
        )

    def test_collect_boxplot_rows_filters_step2b_seed_window_and_mirrors_methods(self):
        rows = plot_step2b_seed_boxplots.collect_boxplot_rows(
            make_seed_rows(),
            degrees=[1, 2],
            subset_seeds=[0, 1],
            metric="test_mean_normalized_gap",
            train_size=50,
            spoplus_method_label="spoplus_val_spoplus_loss",
        )

        self.assertEqual(len(rows), 16)
        self.assertEqual(
            sorted({(row["degree"], row["subset_seed"]) for row in rows}),
            [(1, 0), (1, 1), (2, 0), (2, 1)],
        )
        self.assertEqual(
            sorted({row["method"] for row in rows}),
            ["pyepo_lr", "pyepo_spoplus", "step1c_lr", "step1c_spoplus"],
        )
        d1s0 = {
            row["method"]: row
            for row in rows
            if row["degree"] == 1 and row["subset_seed"] == 0
        }
        self.assertAlmostEqual(d1s0["pyepo_lr"]["value"], 0.01)
        self.assertAlmostEqual(d1s0["step1c_lr"]["value"], 0.01)
        self.assertAlmostEqual(d1s0["pyepo_spoplus"]["value"], 0.005)
        self.assertAlmostEqual(d1s0["step1c_spoplus"]["value"], 0.005)
        self.assertEqual(
            d1s0["pyepo_spoplus"]["source"],
            "step2_resampling_heldout400_mirrored_by_bridge",
        )

    def test_collect_boxplot_rows_requires_complete_lr_and_spoplus_pairs(self):
        rows = [
            row
            for row in make_seed_rows(degrees=(1,), seeds=(0,))
            if row["method_label"] != "spoplus_val_spoplus_loss"
        ]

        with self.assertRaisesRegex(ValueError, "Missing Step2b seed rows"):
            plot_step2b_seed_boxplots.collect_boxplot_rows(
                rows,
                degrees=[1],
                subset_seeds=[0],
                metric="test_mean_normalized_gap",
                train_size=50,
                spoplus_method_label="spoplus_val_spoplus_loss",
            )

    def test_run_writes_seed_boxplot_and_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            per_seed_csv = tmp_path / "per_seed.csv"
            output_dir = tmp_path / "plots"
            write_csv(per_seed_csv, make_seed_rows())

            args = plot_step2b_seed_boxplots.build_parser().parse_args(
                [
                    "--per-seed-csv",
                    str(per_seed_csv),
                    "--output-dir",
                    str(output_dir),
                    "--degrees",
                    "1",
                    "2",
                    "--subset-seeds",
                    "0",
                    "1",
                    "--formats",
                    "png",
                ]
            )
            outputs = plot_step2b_seed_boxplots.run(args)

            self.assertTrue(outputs["csv"].exists())
            self.assertEqual(len(outputs["plots"]), 1)
            self.assertTrue(outputs["plots"][0].exists())
            csv_text = outputs["csv"].read_text(encoding="utf-8")
            self.assertIn("PyEPO LR", csv_text)
            self.assertIn("my SPO+", csv_text)
            self.assertIn("subset_seed", csv_text)


if __name__ == "__main__":
    unittest.main()
