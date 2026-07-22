from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_material_labels as label_builder
import build_candidate_conflict_graphs as conflict_builder
import helpful_policy_common as helpful_policy
import material_common as common
import plan_material_folds as fold_planner
import run_scalar_classification_baselines as baselines
import train_incidence_classifier as incidence


def source_row(values: list[float], topology_id: str = "G-1") -> dict[str, str]:
    array = np.asarray(values, dtype=float)
    return {
        "topology_id": topology_id,
        "topology_hash": f"topology-{topology_id}",
        "feasible_set_hash": f"feasible-{topology_id}",
        "test_hash": f"test-{topology_id}",
        "seed42_normalized_improvement_pp": str(values[0]),
        "seed43_normalized_improvement_pp": str(values[1]),
        "seed44_normalized_improvement_pp": str(values[2]),
        "formal_label_mean_pp": str(float(np.mean(array))),
        "label_uncertainty_std_pp": str(float(np.std(array, ddof=0))),
        "formal_label_ready": "True",
        "label_seed_count": "3",
    }


class MaterialClassificationTests(unittest.TestCase):
    def test_primary_threshold_is_strict(self) -> None:
        self.assertEqual(common.primary_label(0.5), "neutral_or_uncertain")
        self.assertEqual(common.primary_label(-0.5), "neutral_or_uncertain")
        self.assertEqual(common.primary_label(0.500001), "material_helpful")
        self.assertEqual(common.primary_label(-0.500001), "material_harmful")

    def test_confidence_fields_separate_primary_and_uncertain_labels(self) -> None:
        confident = common.derive_label_row(source_row([1.0, 1.0, 1.0]))
        self.assertEqual(confident["primary_label"], "material_helpful")
        self.assertEqual(confident["confidence_label"], "material_helpful")
        self.assertEqual(confident["confidence_state"], "confident_material")

        unstable = common.derive_label_row(source_row([10.0, -1.0, -1.0]))
        self.assertEqual(unstable["primary_label"], "material_helpful")
        self.assertFalse(unstable["sign_agreement_passed"])
        self.assertEqual(unstable["confidence_label"], "neutral_or_uncertain")
        self.assertEqual(unstable["confidence_state"], "uncertain_material")

    def test_exact_bootstrap_is_deterministic(self) -> None:
        first = common.exact_three_seed_bootstrap_ci([1.0, 2.0, 3.0])
        second = common.exact_three_seed_bootstrap_ci([1.0, 2.0, 3.0])
        self.assertEqual(first, second)
        self.assertLess(first[0], 2.0)
        self.assertGreater(first[1], 2.0)

    def test_candidate_conflict_builder_uses_shared_membership_and_drops_target(self) -> None:
        graph = {
            "topology_id": "G-1",
            "topology_hash": "topology-1",
            "feasible_set_hash": "feasible-1",
            "node_ids": ["v:0", "v:1", "c:0", "c:1"],
            "node_features": [
                [1, 0, 1, 0, 0, 0, 0, 1, 2, 2],
                [1, 0, 0, 1, 0, 0, 0, 0, 1, 1],
                [0, 1, 0, 0, 1, 0, 2, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 2, 0, 0, 0],
            ],
            "edge_source": [0, 0, 1],
            "edge_target": [2, 3, 3],
            "edge_type": [1, 1, 1],
            "scalar_topology_features": {"candidate_conflict_edges": 1},
            "target": {"value": 9.0},
        }
        result = conflict_builder.build_record(graph)
        self.assertEqual(result["candidate_count"], 2)
        self.assertEqual(result["undirected_conflict_edge_count"], 1)
        self.assertEqual(result["edge_source"], [0, 1])
        self.assertEqual(result["edge_target"], [1, 0])
        self.assertTrue(result["target_free"])
        self.assertNotIn("target", result)

    def test_actual_locked_target_distribution(self) -> None:
        rows, failures = label_builder.build(
            common.read_csv(common.DEFAULT_TARGET_TABLE),
            material_threshold_pp=0.5,
            high_variance_std_pp=0.5,
            bootstrap_alpha=0.05,
        )
        self.assertEqual(failures, [])
        self.assertEqual(
            common.label_counts(rows, "primary_label"),
            {
                "material_harmful": 11,
                "neutral_or_uncertain": 827,
                "material_helpful": 162,
            },
        )
        self.assertEqual(
            common.label_counts(rows, "confidence_label"),
            {
                "material_harmful": 4,
                "neutral_or_uncertain": 877,
                "material_helpful": 119,
            },
        )

    def test_material_folds_are_deterministic_and_balanced(self) -> None:
        rows = []
        labels = (
            ["material_harmful"] * 11
            + ["neutral_or_uncertain"] * 827
            + ["material_helpful"] * 162
        )
        for index, label in enumerate(labels):
            rows.append(
                {
                    "topology_id": f"G-{index}",
                    "topology_hash": f"topology-{index}",
                    "feasible_set_hash": f"feasible-{index}",
                    "formal_label_mean_pp": "0.0",
                    "primary_label": label,
                }
            )
        first = fold_planner.assign_folds(rows)
        second = fold_planner.assign_folds(rows)
        self.assertEqual(first, second)
        result = fold_planner.audit(first, folds=5, label_field="primary_label")
        self.assertTrue(result["passed"])
        self.assertEqual(set(result["fold_sizes"].values()), {200})
        self.assertEqual(
            sorted(result["label_fold_counts"]["material_harmful"].values()),
            [2, 2, 2, 2, 3],
        )

    def test_policy_metrics_reward_correct_helpful_selection(self) -> None:
        result = common.policy_metrics(
            np.asarray([-1.0, 0.0, 2.0]),
            np.asarray([False, False, True]),
            np.asarray([0.1, 0.1, 0.9]),
        )
        self.assertEqual(result["selected_count"], 1)
        self.assertEqual(result["helpful_precision"], 1.0)
        self.assertEqual(result["policy_regret_pp"], 0.0)
        self.assertEqual(result["oracle_improvement_captured"], 1.0)

    def test_incidence_nested_split_is_600_200_200(self) -> None:
        fold_rows = [
            {"topology_id": f"G-{index}", "fold": str(index % 5)}
            for index in range(1000)
        ]
        splits = incidence.split_topology_ids(fold_rows, test_fold=3)
        self.assertEqual(
            {name: len(values) for name, values in splits.items()},
            {"train": 600, "validation": 200, "test": 200},
        )
        self.assertFalse(splits["train"] & splits["validation"])
        self.assertFalse(splits["train"] & splits["test"])
        self.assertFalse(splits["validation"] & splits["test"])

    def test_temperature_scaling_is_validation_nll_optimal(self) -> None:
        target = np.asarray([True, False, True, False])
        logits = np.asarray([10.0, -10.0, -10.0, 10.0])
        fitted = helpful_policy.fit_temperature(target, logits)
        self.assertGreater(fitted["temperature"], 1.0)
        self.assertLessEqual(
            fitted["validation_nll_after"], fitted["validation_nll_before"]
        )

    def test_regret_threshold_uses_realized_validation_utility(self) -> None:
        result = helpful_policy.select_regret_threshold(
            np.asarray([-2.0, 3.0]),
            np.asarray([False, True]),
            np.asarray([0.1, 0.9]),
            compute_cost_pp=0.0,
            compute_costs_pp=(0.0,),
        )
        self.assertEqual(result["threshold"], 0.9)
        self.assertEqual(result["validation_policy"]["selected_count"], 1)
        self.assertEqual(result["validation_policy"]["helpful_precision"], 1.0)

    def test_precision_constraint_can_abstain(self) -> None:
        result = helpful_policy.select_precision_constrained_threshold(
            np.asarray([1.0, 0.0]),
            np.asarray([True, False]),
            np.asarray([0.1, 0.9]),
            minimum_precision=0.75,
            compute_costs_pp=(0.0,),
        )
        self.assertEqual(result["validation_policy"]["selected_count"], 0)
        self.assertEqual(result["validation_helpful_recall"], 0.0)

    def test_perfect_multiclass_probabilities_have_perfect_metrics(self) -> None:
        target = np.asarray(common.CLASS_LABELS, dtype=object)
        probabilities = np.eye(3, dtype=float)
        metrics = baselines.evaluate_model(
            target,
            np.asarray([-1.0, 0.0, 2.0]),
            probabilities,
            helpful_threshold=0.5,
            compute_costs_pp=[0.0, 0.1],
        )
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["balanced_accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["macro_auroc_ovr"], 1.0)
        self.assertEqual(metrics["macro_auprc_ovr"], 1.0)
        self.assertEqual(metrics["multiclass_brier"], 0.0)


if __name__ == "__main__":
    unittest.main()
