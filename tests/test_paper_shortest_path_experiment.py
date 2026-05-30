import csv
import inspect
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

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


def load_module(filename: str, name: str):
    if str(PAPER_DIR) not in sys.path:
        sys.path.insert(0, str(PAPER_DIR))
    spec = importlib.util.spec_from_file_location(name, PAPER_DIR / filename)
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
            diagnostics_path = Path(temp_dir) / "training_diagnostics.csv"
            metadata_path = Path(temp_dir) / "metadata.json"
            self.assertTrue(summary_path.exists())
            self.assertTrue(diagnostics_path.exists())
            self.assertTrue(metadata_path.exists())
            with summary_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            with diagnostics_path.open(newline="", encoding="utf-8") as handle:
                diagnostics = list(csv.DictReader(handle))
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

            self.assertEqual(
                {row["implementation"] for row in rows},
                {"ls", "ours-spoplus"},
            )
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertEqual(row["degree"], "1")
                self.assertEqual(row["noise_half_width"], "0.0")
                self.assertTrue(float(row["test_norm_spo"]) >= 0.0)
            ours_row = next(row for row in rows if row["implementation"] == "ours-spoplus")
            self.assertEqual(ours_row["spoplus_variant"], "raw")
            self.assertEqual(ours_row["spoplus_init"], "ls")
            self.assertIn("best_step", ours_row)
            self.assertIn("coef_delta_norm_from_ls", ours_row)
            self.assertIn("val_path_change_rate_from_ls", ours_row)
            self.assertIn("test_path_change_rate_from_ls", ours_row)
            self.assertEqual(len(diagnostics), 1)
            self.assertEqual(diagnostics[0]["implementation"], "ours-spoplus")
            self.assertEqual(diagnostics[0]["lambda_value"], "0.0")
            self.assertEqual(diagnostics[0]["selected"], "True")
            self.assertIn("initial_val_norm_spo", diagnostics[0])
            self.assertIn("best_val_norm_spo", diagnostics[0])
            self.assertIn("final_val_norm_spo", diagnostics[0])
            self.assertEqual(
                metadata["normalized_spo_definition"],
                "sum_regret_over_sum_oracle_cost",
            )
            self.assertEqual(metadata["paper_experiment"], "shortest_path_middle_row")
            self.assertEqual(metadata["grid_shape"], [5, 5])
            self.assertEqual(metadata["feature_dim"], 5)
            self.assertEqual(metadata["edge_dim"], 40)
            self.assertFalse(metadata["pyepo_requested"])

    def test_middle_row_preset_resolves_to_paper_middle_row(self):
        runner = load_module("run_paper_shortest_path.py", "paper_runner_presets")
        args = runner.build_arg_parser().parse_args(["--preset", "middle-row"])

        options = runner.resolve_options(args)

        self.assertEqual(options["degrees"], (1, 2, 4, 6, 8))
        self.assertEqual(options["noise_half_widths"], (0.0, 0.5))
        self.assertEqual(options["trials"], 50)
        self.assertEqual(options["n_train"], 1000)
        self.assertEqual(options["n_val"], 250)
        self.assertEqual(options["n_test"], 10000)
        self.assertEqual(len(options["lambda_grid"]), 10)
        self.assertEqual(options["methods"], ("ls", "ours-spoplus"))
        self.assertEqual(options["spoplus_iterate"], "raw")
        self.assertEqual(options["spoplus_init"], "ls")
        self.assertEqual(options["eval_period"], 50)

    def test_spoplus_training_protocol_options_resolve(self):
        runner = load_module("run_paper_shortest_path.py", "paper_runner_protocol_options")
        args = runner.build_arg_parser().parse_args(
            [
                "--preset",
                "pilot",
                "--spoplus-iterate",
                "averaged",
                "--spoplus-init",
                "zero",
                "--eval-period",
                "7",
            ]
        )

        options = runner.resolve_options(args)

        self.assertEqual(options["spoplus_iterate"], "averaged")
        self.assertEqual(options["spoplus_init"], "zero")
        self.assertEqual(options["eval_period"], 7)

    def test_protocol_sweep_defines_expected_variants(self):
        sweep = load_module("run_protocol_sweep.py", "paper_protocol_sweep")

        variants = sweep.protocol_variants()
        by_name = {variant.name: variant for variant in variants}

        self.assertEqual(
            set(by_name),
            {
                "baseline-current",
                "smaller-batch",
                "no-l1",
                "averaged-iterate",
                "zero-init-diagnostic",
            },
        )
        self.assertEqual(by_name["baseline-current"].spoplus_iterate, "raw")
        self.assertEqual(by_name["baseline-current"].spoplus_init, "ls")
        self.assertEqual(by_name["smaller-batch"].batch_size, 10)
        self.assertEqual(by_name["smaller-batch"].learning_rate, 0.01)
        self.assertEqual(by_name["smaller-batch"].spoplus_iterations, 3000)
        self.assertEqual(by_name["no-l1"].lambda_grid, (0.0,))
        self.assertEqual(by_name["averaged-iterate"].spoplus_iterate, "averaged")
        self.assertEqual(by_name["zero-init-diagnostic"].spoplus_init, "zero")

    def test_pyepo_pilot_preset_includes_pyepo_spoplus(self):
        runner = load_module("run_paper_shortest_path.py", "paper_runner_pyepo_preset")
        args = runner.build_arg_parser().parse_args(["--preset", "pyepo-pilot"])

        options = runner.resolve_options(args)

        self.assertEqual(options["degrees"], (1, 8))
        self.assertEqual(options["noise_half_widths"], (0.0, 0.5))
        self.assertIn("pyepo-spoplus", options["methods"])

    def test_pyepo_trainer_accepts_init_and_checkpoint_options(self):
        common = load_common()

        signature = inspect.signature(common.train_spoplus_pyepo)

        self.assertIn("spoplus_init", signature.parameters)
        self.assertEqual(signature.parameters["spoplus_init"].default, "ls")
        self.assertIn("eval_period", signature.parameters)
        self.assertEqual(signature.parameters["eval_period"].default, 50)

    def test_pyepo_method_receives_spoplus_protocol_options(self):
        runner = load_module("run_paper_shortest_path.py", "paper_runner_pyepo_protocol")
        sentinel = object()
        instance = SimpleNamespace(train="train-split", val="val-split", seed=101)

        with mock.patch.object(
            runner.common,
            "train_spoplus_pyepo",
            return_value=sentinel,
        ) as trainer:
            result = runner._method_result(
                "pyepo-spoplus",
                instance,
                lambdas=(0.0,),
                spoplus_iterations=30,
                batch_size=7,
                learning_rate=0.02,
                spoplus_iterate="raw",
                spoplus_init="zero",
                eval_period=10,
            )

        self.assertIs(result, sentinel)
        trainer.assert_called_once_with(
            "train-split",
            "val-split",
            lambdas=(0.0,),
            iterations=30,
            batch_size=7,
            learning_rate=0.02,
            spoplus_init="zero",
            eval_period=10,
            seed=101,
        )

    def test_dry_run_reports_middle_row_plan_without_writing_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "dry-run-output"
            result = subprocess.run(
                [
                    sys.executable,
                    str(PAPER_DIR / "run_paper_shortest_path.py"),
                    "--preset",
                    "middle-row",
                    "--dry-run",
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=REPO_ROOT,
                check=False,
                text=True,
                capture_output=True,
            )

            self.assertEqual(
                result.returncode,
                0,
                f"dry-run failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
            )
            self.assertIn("preset: middle-row", result.stdout)
            self.assertIn("degrees: (1, 2, 4, 6, 8)", result.stdout)
            self.assertIn("noise_half_widths: (0.0, 0.5)", result.stdout)
            self.assertIn("n_train/n_val/n_test: 1000/250/10000", result.stdout)
            self.assertIn("lambda_grid_length: 10", result.stdout)
            self.assertIn("estimated_model_fits: 10000", result.stdout)
            self.assertIn("estimated_ours_spoplus_oracle_calls: 160000000", result.stdout)
            self.assertIn("spoplus_iterate: raw", result.stdout)
            self.assertIn("spoplus_init: ls", result.stdout)
            self.assertIn("eval_period: 50", result.stdout)
            self.assertFalse(output_dir.exists())

    def test_fail_if_pyepo_missing_uses_clear_dependency_error(self):
        runner = load_module("run_paper_shortest_path.py", "paper_runner_pyepo_gate")
        missing = runner.DependencyStatus(
            available=False,
            message="forced missing dependency",
            details={"torch": "missing"},
        )

        with self.assertRaisesRegex(RuntimeError, "PyEPO SPO\\+ was requested"):
            runner.validate_pyepo_request(
                {"methods": ("ls", "pyepo-spoplus")},
                fail_if_missing=True,
                status_checker=lambda: missing,
            )

    def test_plot_script_runs_on_tiny_summary_csv_when_matplotlib_is_available(self):
        if importlib.util.find_spec("matplotlib") is None:
            self.skipTest("matplotlib is not installed in this Python environment")
        with tempfile.TemporaryDirectory() as temp_dir:
            summary_path = Path(temp_dir) / "summary.csv"
            fields = [
                "implementation",
                "trial",
                "degree",
                "noise_half_width",
                "test_norm_spo",
            ]
            with summary_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for noise in (0.0, 0.5):
                    for implementation, value in (
                        ("ls", 0.20 + noise),
                        ("ours-spoplus", 0.12 + noise),
                    ):
                        writer.writerow(
                            {
                                "implementation": implementation,
                                "trial": 0,
                                "degree": 1,
                                "noise_half_width": noise,
                                "test_norm_spo": value,
                            }
                        )

            result = subprocess.run(
                [
                    sys.executable,
                    str(PAPER_DIR / "plot_paper_shortest_path.py"),
                    str(summary_path),
                    "--output-dir",
                    str(Path(temp_dir) / "plots"),
                ],
                cwd=REPO_ROOT,
                check=False,
                text=True,
                capture_output=True,
            )

            self.assertEqual(
                result.returncode,
                0,
                f"plotter failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
            )
            self.assertTrue(
                (Path(temp_dir) / "plots" / "paper_shortest_path_middle_row.png").exists()
            )

    def test_forward_loss_comparison_allows_missing_pyepo_when_requested(self):
        if all(
            importlib.util.find_spec(name) is not None
            for name in ("torch", "pyepo", "gurobipy")
        ):
            self.skipTest("PyEPO imports are present; missing-dependency path is not active")
        result = subprocess.run(
            [
                sys.executable,
                str(PAPER_DIR / "compare_pyepo_forward_loss.py"),
                "--allow-missing-pyepo",
            ],
            cwd=REPO_ROOT,
            check=False,
            text=True,
            capture_output=True,
        )

        self.assertEqual(
            result.returncode,
            0,
            f"forward-loss script failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        self.assertIn("pyepo_available", result.stdout)


if __name__ == "__main__":
    unittest.main()
