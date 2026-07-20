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

import gnn_data_common as common
import launch_formal_gnn_jobs as formal_launcher
import plan_formal_gnn_jobs as formal_planner
import plan_stratified_folds as folds
import review_formal_gnn_results as formal_review
import run_scalar_baselines as baselines
import train_formal_gnn as formal_gnn


class GNNScaffoldTests(unittest.TestCase):
    def test_formal_plan_is_exact_preview_only_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "formal15"
            args = argparse.Namespace(
                python="python",
                graph_jsonl=Path("graphs.jsonl"),
                folds=Path("folds.csv"),
                output_root=output_root,
                node_input_dim=10,
                hidden_dim=64,
                layers=3,
                relation_count=3,
                dropout=0.1,
                batch_size=32,
                learning_rate=0.001,
                weight_decay=0.0001,
                max_epochs=500,
                early_stop_patience=30,
                early_stop_min_delta=0.0001,
                threads=4,
            )
            jobs = formal_planner.build_jobs(args, ready=True)
            self.assertEqual(len(jobs), 15)
            self.assertEqual(
                {(int(row["fold"]), int(row["seed"])) for row in jobs},
                {(fold, seed) for fold in range(5) for seed in (42, 43, 44)},
            )
            self.assertTrue(all("--execute" not in shlex.split(row["command_preview"]) for row in jobs))
            parsed = formal_launcher.parse_job(
                {key: str(value) for key, value in jobs[0].items()},
                output_root,
            )
            self.assertEqual(parsed.command[-1], "--execute")
            self.assertEqual(parsed.threads, 4)

    def test_formal_review_metrics_detect_perfect_predictions(self) -> None:
        target = np.asarray([-1.0, 0.0, 2.0, 3.0])
        result = formal_review.metrics(target, target.copy())
        self.assertEqual(result["mae"], 0.0)
        self.assertEqual(result["rmse"], 0.0)
        self.assertEqual(result["r2"], 1.0)
        self.assertEqual(result["spearman"], 1.0)

    def test_formal_gnn_outer_test_and_validation_folds_are_disjoint(self) -> None:
        fold_rows = [
            {"topology_id": f"G-{index}", "fold": str(index % 5)}
            for index in range(1000)
        ]
        splits = formal_gnn.split_topology_ids(fold_rows, test_fold=3)
        self.assertEqual({name: len(values) for name, values in splits.items()}, {"train": 600, "validation": 200, "test": 200})
        self.assertEqual(splits["test"], {f"G-{index}" for index in range(1000) if index % 5 == 3})
        self.assertEqual(splits["validation"], {f"G-{index}" for index in range(1000) if index % 5 == 4})
        self.assertFalse(splits["train"] & splits["validation"])
        self.assertFalse(splits["train"] & splits["test"])

    def test_graph_contains_compatibility_and_bidirectional_incidence(self) -> None:
        template = {
            "topology_id": "G-X",
            "vertices": [{"id": "0", "type": "Pair"}, {"id": "1", "type": "Pair"}, {"id": "2", "type": "NDD"}],
            "arcs": [{"source": "2", "target": "0"}, {"source": "0", "target": "1"}],
            "feasible_candidates": [{"type": "chain", "length": 2, "nodes": ["2", "0", "1"]}],
        }
        row = {field: "1" for field in common.SCALAR_FEATURES}
        row.update({
            "topology_id": "G-X", "topology_hash": "th", "feasible_set_hash": "fh",
            "template_path": "template.json", "normalized_improvement_pp": "2.5",
        })
        record = common.build_graph_record(row, template)
        self.assertEqual(len(record["node_ids"]), 4)
        self.assertEqual(record["edge_type"].count(common.RELATION_TYPES["compatibility"]), 2)
        self.assertEqual(record["edge_type"].count(common.RELATION_TYPES["vertex_to_candidate"]), 3)
        self.assertEqual(record["edge_type"].count(common.RELATION_TYPES["candidate_to_vertex"]), 3)
        self.assertEqual(common.validate_no_target_leakage(record), [])

    def test_formal_graph_target_uses_mean_and_keeps_uncertainty_separate(self) -> None:
        template = {
            "topology_id": "G-X",
            "vertices": [{"id": "0", "type": "Pair"}],
            "arcs": [],
            "feasible_candidates": [],
        }
        row = {field: "1" for field in common.SCALAR_FEATURES}
        row.update({
            "topology_id": "G-X", "topology_hash": "th", "feasible_set_hash": "fh",
            "template_path": "template.json", "normalized_improvement_pp": "99.0",
        })
        target = {
            "formal_label_ready": "True",
            "formal_label_mean_pp": "2.5",
            "label_uncertainty_std_pp": "0.75",
            "uncertainty_ddof": "0",
        }
        record = common.build_graph_record(row, template, formal_target_row=target)
        self.assertEqual(record["target"]["name"], "formal_label_mean_pp")
        self.assertEqual(record["target"]["value"], 2.5)
        self.assertTrue(record["target"]["formal"])
        self.assertEqual(record["label_uncertainty"]["value"], 0.75)
        self.assertNotIn("label_uncertainty_std_pp", record["scalar_topology_features"])

    def test_stratified_assignment_is_deterministic(self) -> None:
        rows = []
        values = [0.0, 0.02, -0.02, 0.5, -0.5, 5.0, -5.0] * 10
        for index, value in enumerate(values):
            rows.append({"topology_id": f"G-{index}", "topology_hash": f"t{index}", "feasible_set_hash": f"f{index}", "normalized_improvement_pp": str(value)})
        first = folds.assign_folds(rows, folds=5, seed=7)
        second = folds.assign_folds(rows, folds=5, seed=7)
        self.assertEqual(first, second)
        self.assertEqual({row["fold"] for row in first}, set(range(5)))

    def test_formal_folds_and_baselines_use_three_seed_mean(self) -> None:
        feature_rows = []
        target_rows = []
        for index in range(1000):
            topology_id = f"G-{index}"
            feature = {
                field: str((index + feature_index) % 7)
                for feature_index, field in enumerate(common.SCALAR_FEATURES)
            }
            feature.update(
                {
                    "topology_id": topology_id,
                    "topology_hash": f"topology-{index}",
                    "feasible_set_hash": f"feasible-{index}",
                    "normalized_improvement_pp": "999.0",
                }
            )
            target_rows.append(
                {
                    "topology_id": topology_id,
                    "topology_hash": f"topology-{index}",
                    "feasible_set_hash": f"feasible-{index}",
                    "formal_label_mean_pp": str((index % 17 - 8) / 10.0),
                    "label_uncertainty_std_pp": "0.25",
                    "formal_label_ready": "True",
                }
            )
            feature_rows.append(feature)

        folds.validate_targets(
            target_rows,
            target_field="formal_label_mean_pp",
            require_formal_targets=True,
        )
        fold_rows = folds.assign_folds(
            target_rows,
            folds=5,
            seed=7,
            target_field="formal_label_mean_pp",
        )
        merged = baselines.merge_feature_targets(
            feature_rows,
            target_rows,
            target_field="formal_label_mean_pp",
            require_formal_targets=True,
        )
        prediction_rows, audit = baselines.run(
            merged,
            [{key: str(value) for key, value in row.items()} for row in fold_rows],
            target_field="formal_label_mean_pp",
            require_formal_targets=True,
        )
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["target_field"], "formal_label_mean_pp")
        self.assertEqual(prediction_rows[0]["target_name"], "formal_label_mean_pp")
        self.assertNotEqual(prediction_rows[0]["target_value"], 999.0)

    def test_ridge_predicts_linear_signal(self) -> None:
        x = np.arange(20, dtype=float).reshape(-1, 1)
        y = 1.0 + 2.0 * x[:, 0]
        prediction = baselines.ridge_predict(x, y, x, alpha=0.0)
        self.assertTrue(np.allclose(prediction, y))


if __name__ == "__main__":
    unittest.main()
