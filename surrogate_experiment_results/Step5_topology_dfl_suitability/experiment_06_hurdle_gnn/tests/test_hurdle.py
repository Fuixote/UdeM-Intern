from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import sys
import tempfile
import unittest

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import hurdle_common as common
import launch_hurdle_jobs as launcher
import plan_hurdle_jobs as planner
import review_hurdle_results as reviewer


class HurdleTests(unittest.TestCase):
    def test_nested_split_is_600_200_200(self) -> None:
        rows = [{"topology_id": f"G-{index}", "fold": str(index % 5)} for index in range(1000)]
        splits = common.split_topology_ids(rows, test_fold=4)
        self.assertEqual({name: len(ids) for name, ids in splits.items()}, {"train": 600, "validation": 200, "test": 200})
        self.assertEqual(splits["validation"], {f"G-{index}" for index in range(1000) if index % 5 == 0})

    def test_binary_metrics_are_perfect_for_separated_probabilities(self) -> None:
        target = np.asarray([0, 0, 1, 1])
        probability = np.asarray([0.1, 0.2, 0.8, 0.9])
        result = common.binary_metrics(target, probability)
        self.assertEqual(result["balanced_accuracy"], 1.0)
        self.assertEqual(result["f1"], 1.0)
        self.assertEqual(result["auroc"], 1.0)
        self.assertEqual(result["average_precision"], 1.0)

    def test_regression_metrics_are_perfect_for_identity(self) -> None:
        target = np.asarray([-2.0, 0.0, 1.0, 5.0])
        result = common.regression_metrics(target, target.copy())
        self.assertEqual(result["mae"], 0.0)
        self.assertEqual(result["rmse"], 0.0)
        self.assertEqual(result["r2"], 1.0)
        self.assertEqual(result["spearman"], 1.0)

    def test_smoke_plan_is_one_classifier_plus_ten_regressors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "smoke"
            args = argparse.Namespace(
                graph_jsonl=Path("graphs.jsonl"),
                folds=Path("folds.csv"),
                output_root=output_root,
                selected_folds=[0],
                selected_seeds=[42],
                python="python",
                max_epochs=300,
                early_stop_patience=30,
                threads=4,
            )
            classifiers, regressors = planner.build_jobs(args, input_ready=True)
            self.assertEqual(len(classifiers), 1)
            self.assertEqual(len(regressors), 10)
            self.assertEqual(
                {(row["regression_subset"], row["objective"]) for row in regressors},
                {(subset, objective) for subset in common.REGRESSION_SUBSETS for objective in common.OBJECTIVES},
            )
            self.assertTrue(all("--execute" not in shlex.split(row["command_preview"]) for row in classifiers + regressors))
            parsed_classifier = launcher.parse_job(
                {key: str(value) for key, value in classifiers[0].items()},
                output_root,
            )
            parsed_regressor = launcher.parse_job(
                {key: str(value) for key, value in regressors[0].items()},
                output_root,
            )
            self.assertEqual(parsed_classifier.command[-1], "--execute")
            self.assertEqual(parsed_regressor.command[-1], "--execute")
            self.assertEqual(parsed_regressor.dependency_path, Path(regressors[0]["dependency_path"]))

    def test_oof_aggregation_pools_disjoint_folds(self) -> None:
        classifier_rows = []
        regressor_rows = []
        for fold, targets in ((0, [0.0, 2.0]), (1, [0.0, -1.0])):
            for index, target in enumerate(targets):
                common_fields = {
                    "topology_id": f"G-{fold * 2 + index}",
                    "fold": fold,
                    "seed": 42,
                }
                classifier_rows.append(
                    {
                        **common_fields,
                        "target_is_nonzero": int(target != 0.0),
                        "probability_nonzero": 0.9 if target != 0.0 else 0.1,
                    }
                )
                regressor_rows.append(
                    {
                        **common_fields,
                        "regression_subset": "nonzero",
                        "objective": "mse",
                        "target_formal_label_mean_pp": target,
                        "raw_regression_prediction_pp": target,
                        "hard_hurdle_prediction_pp": target,
                        "soft_hurdle_prediction_pp": target,
                        "oracle_nonzero_gate_prediction_pp": target,
                    }
                )
        classifier_metrics = reviewer.aggregate_classifier_predictions(classifier_rows)
        self.assertEqual(classifier_metrics[0]["fold_count"], 2)
        self.assertEqual(classifier_metrics[0]["count"], 4)
        self.assertEqual(classifier_metrics[0]["auroc"], 1.0)
        regression_metrics = reviewer.aggregate_regressor_predictions(regressor_rows)
        all_hard = next(
            row
            for row in regression_metrics
            if row["prediction_mode"] == "hard_hurdle" and row["subset"] == "all"
        )
        self.assertEqual(all_hard["fold_count"], 2)
        self.assertEqual(all_hard["count"], 4)
        self.assertEqual(all_hard["mae"], 0.0)
        self.assertEqual(all_hard["r2"], 1.0)


if __name__ == "__main__":
    unittest.main()
