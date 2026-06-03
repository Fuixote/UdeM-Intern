from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "decision_analysis"
    / "scripts"
    / "analyze_margin_near_ties.py"
)


def per_graph_row(method_label: str, achieved_obj: float, normalized_gap: float) -> dict[str, str]:
    method = "2stage" if method_label == "2stage_val_mse" else "spoplus"
    return {
        "regime": "step2b_poly_d8",
        "case_type": "synthetic_seed",
        "subset_seed": "3",
        "graph_id": "G-test.json",
        "method_label": method_label,
        "method": method,
        "selection_metric": "validation_metric",
        "optimal_obj": "100.0",
        "achieved_obj": str(achieved_obj),
        "decision_gap": str(100.0 - achieved_obj),
        "normalized_gap": str(normalized_gap),
        "same_solution_as_opt": "False",
        "edge_jaccard_with_opt": "0.5",
    }


def edge_row(edge_id: int, in_opt: bool, in_2stage: bool, in_spoplus: bool) -> dict[str, str]:
    return {
        "regime": "step2b_poly_d8",
        "case_type": "synthetic_seed",
        "subset_seed": "3",
        "graph_id": "G-test.json",
        "edge_id": str(edge_id),
        "in_opt": str(in_opt),
        "in_2stage": str(in_2stage),
        "in_spoplus": str(in_spoplus),
    }


class DecisionAnalysisMarginNearTieTests(unittest.TestCase):
    def load_module(self):
        self.assertTrue(SCRIPT_PATH.exists(), f"Missing script: {SCRIPT_PATH}")
        spec = importlib.util.spec_from_file_location("analyze_margin_near_ties", SCRIPT_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def test_solution_overlap_metrics_from_edge_flags(self):
        module = self.load_module()
        rows = [
            edge_row(0, True, True, True),
            edge_row(1, True, False, True),
            edge_row(2, False, True, False),
            edge_row(3, False, False, False),
        ]

        metrics = module.solution_overlap_metrics(rows, "in_2stage", "in_spoplus")

        self.assertFalse(metrics["same_solution"])
        self.assertEqual(metrics["left_count"], 2)
        self.assertEqual(metrics["right_count"], 2)
        self.assertEqual(metrics["intersection_count"], 1)
        self.assertAlmostEqual(metrics["edge_jaccard"], 1.0 / 3.0)

    def test_build_margin_and_candidate_rows_marks_near_ties_and_ranks_candidates(self):
        module = self.load_module()
        per_graph_rows = [
            per_graph_row("2stage_val_mse", achieved_obj=98.0, normalized_gap=0.02),
            per_graph_row(
                "spoplus_val_spoplus_loss", achieved_obj=99.0, normalized_gap=0.01
            ),
        ]
        edge_rows = [
            edge_row(0, True, True, True),
            edge_row(1, True, False, True),
            edge_row(2, False, True, False),
            edge_row(3, False, False, False),
        ]

        margin_rows, candidate_rows, summary_rows = module.analyze_margin_near_ties(
            per_graph_rows,
            edge_rows,
            near_tie_threshold=0.03,
        )

        self.assertEqual(len(margin_rows), 1)
        margin = margin_rows[0]
        self.assertAlmostEqual(float(margin["abs_obj_2stage_minus_spoplus"]), 1.0)
        self.assertAlmostEqual(float(margin["edge_jaccard_2stage_spoplus"]), 1.0 / 3.0)
        self.assertEqual(margin["two_stage_different_solution_near_tie"], True)
        self.assertEqual(margin["spoplus_different_solution_near_tie"], True)
        self.assertEqual(margin["same_2stage_spoplus"], False)

        by_label = {row["candidate_label"]: row for row in candidate_rows}
        self.assertEqual(by_label["y_opt"]["rank_true_obj"], 1)
        self.assertEqual(by_label["y_spoplus"]["rank_true_obj"], 2)
        self.assertEqual(by_label["y_2stage"]["rank_true_obj"], 3)
        self.assertAlmostEqual(float(by_label["y_2stage"]["edge_jaccard_with_spoplus"]), 1.0 / 3.0)

        self.assertEqual(summary_rows[0]["graph_count"], 1)
        self.assertEqual(summary_rows[0]["two_stage_near_tie_count"], 1)
        self.assertEqual(summary_rows[0]["spoplus_near_tie_count"], 1)


if __name__ == "__main__":
    unittest.main()
