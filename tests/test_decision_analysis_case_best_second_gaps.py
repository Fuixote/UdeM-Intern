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
    / "summarize_case_best_second_gaps.py"
)


def case_row(case_id: str, case_label: str, subset_seed: int, graph_id: str) -> dict[str, str]:
    return {
        "case_id": case_id,
        "case_label": case_label,
        "regime": "step2b_poly_d8",
        "case_type": "synthetic_case_type",
        "subset_seed": str(subset_seed),
        "graph_id": graph_id,
    }


def second_row(
    subset_seed: int,
    graph_id: str,
    method_label: str,
    solution_rank: int,
    gap: float,
    normalized_gap: float,
    same_oracle: bool,
    jaccard: float,
    predicted_margin: float,
) -> dict[str, str]:
    return {
        "regime": "step2b_poly_d8",
        "case_type": "synthetic_case_type",
        "subset_seed": str(subset_seed),
        "graph_id": graph_id,
        "method_label": method_label,
        "solution_rank": str(solution_rank),
        "gap_to_oracle": str(gap),
        "normalized_gap_to_oracle": str(normalized_gap),
        "same_solution_as_oracle": str(same_oracle),
        "edge_jaccard_with_oracle": str(jaccard),
        "predicted_margin_from_best": str(predicted_margin),
    }


class DecisionAnalysisCaseBestSecondGapTests(unittest.TestCase):
    def load_module(self):
        self.assertTrue(SCRIPT_PATH.exists(), f"Missing script: {SCRIPT_PATH}")
        spec = importlib.util.spec_from_file_location(
            "summarize_case_best_second_gaps", SCRIPT_PATH
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def test_build_case_best_second_rows_pivots_rank1_and_rank2_by_case_and_method(self):
        module = self.load_module()
        case_rows = [
            case_row("case_b_001", "case_b_different_solution_near_optimal", 7, "G-1.json"),
            case_row("case_c_001", "case_c_spoplus_fixes_2stage", 8, "G-2.json"),
        ]
        second_rows = [
            second_row(7, "G-1.json", "2stage_val_mse", 1, 0.2, 0.002, False, 0.7, 0.0),
            second_row(7, "G-1.json", "2stage_val_mse", 2, 0.4, 0.004, False, 0.6, 1.5),
            second_row(
                7, "G-1.json", "spoplus_val_spoplus_loss", 1, 0.2, 0.002, False, 0.7, 0.0
            ),
            second_row(
                7, "G-1.json", "spoplus_val_spoplus_loss", 2, 0.3, 0.003, True, 1.0, 1.2
            ),
            second_row(8, "G-2.json", "2stage_val_mse", 1, 30.0, 0.3, False, 0.2, 0.0),
            second_row(8, "G-2.json", "2stage_val_mse", 2, 5.0, 0.05, False, 0.8, 2.0),
            second_row(
                9, "G-not-selected.json", "2stage_val_mse", 1, 1.0, 0.01, False, 0.5, 0.0
            ),
        ]

        rows = module.build_case_best_second_rows(case_rows, second_rows)

        self.assertEqual(
            [(row["case_id"], row["method_label"]) for row in rows],
            [
                ("case_b_001", "2stage_val_mse"),
                ("case_b_001", "spoplus_val_spoplus_loss"),
                ("case_c_001", "2stage_val_mse"),
            ],
        )

        first = rows[0]
        self.assertEqual(first["subset_seed"], 7)
        self.assertEqual(first["graph_id"], "G-1.json")
        self.assertAlmostEqual(first["rank1_normalized_gap"], 0.002)
        self.assertAlmostEqual(first["rank2_normalized_gap"], 0.004)
        self.assertAlmostEqual(first["rank2_minus_rank1_normalized_gap"], 0.002)
        self.assertAlmostEqual(first["rank1_gap_to_oracle"], 0.2)
        self.assertAlmostEqual(first["rank2_gap_to_oracle"], 0.4)
        self.assertAlmostEqual(first["rank2_minus_rank1_gap_to_oracle"], 0.2)
        self.assertEqual(first["rank1_same_oracle"], False)
        self.assertEqual(first["rank2_same_oracle"], False)
        self.assertAlmostEqual(first["rank1_jaccard_oracle"], 0.7)
        self.assertAlmostEqual(first["rank2_jaccard_oracle"], 0.6)
        self.assertAlmostEqual(first["rank2_predicted_margin_from_best"], 1.5)

    def test_build_case_best_second_summary_counts_near_rank2_cases(self):
        module = self.load_module()
        rows = [
            {
                "case_label": "case_a_bad_prediction_irrelevant",
                "method_label": "2stage_val_mse",
                "rank1_normalized_gap": 0.0,
                "rank2_normalized_gap": 0.03,
                "rank2_minus_rank1_normalized_gap": 0.03,
            },
            {
                "case_label": "case_a_bad_prediction_irrelevant",
                "method_label": "2stage_val_mse",
                "rank1_normalized_gap": 0.0,
                "rank2_normalized_gap": 0.08,
                "rank2_minus_rank1_normalized_gap": 0.08,
            },
            {
                "case_label": "case_a_bad_prediction_irrelevant",
                "method_label": "spoplus_val_spoplus_loss",
                "rank1_normalized_gap": 0.01,
                "rank2_normalized_gap": 0.02,
                "rank2_minus_rank1_normalized_gap": 0.01,
            },
        ]

        summary = module.build_case_best_second_summary(rows, near_threshold=0.05)

        by_key = {(row["case_label"], row["method_label"]): row for row in summary}
        two_stage = by_key[("case_a_bad_prediction_irrelevant", "2stage_val_mse")]
        self.assertEqual(two_stage["row_count"], 2)
        self.assertEqual(two_stage["rank2_near_threshold_count"], 1)
        self.assertAlmostEqual(two_stage["rank2_near_threshold_rate"], 0.5)
        self.assertAlmostEqual(two_stage["mean_rank2_minus_rank1_normalized_gap"], 0.055)


if __name__ == "__main__":
    unittest.main()
