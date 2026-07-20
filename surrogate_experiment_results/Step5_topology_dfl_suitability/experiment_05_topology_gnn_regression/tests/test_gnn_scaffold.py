from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import gnn_data_common as common
import plan_stratified_folds as folds
import run_scalar_baselines as baselines


class GNNScaffoldTests(unittest.TestCase):
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
