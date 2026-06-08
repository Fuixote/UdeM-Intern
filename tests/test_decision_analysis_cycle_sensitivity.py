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
    / "summarize_cycle_sensitivity.py"
)


def solution_row(
    max_cycle: int,
    subset_seed: int,
    graph_id: str,
    method_label: str,
    solution_rank: int,
    normalized_gap: float,
    same_oracle: bool = False,
) -> dict[str, str]:
    return {
        "regime": "step2b_poly_d8",
        "max_cycle": str(max_cycle),
        "max_chain": "4",
        "subset_seed": str(subset_seed),
        "graph_id": graph_id,
        "method_label": method_label,
        "solution_rank": str(solution_rank),
        "gap_to_oracle": str(100.0 * normalized_gap),
        "normalized_gap_to_oracle": str(normalized_gap),
        "same_solution_as_oracle": str(same_oracle),
        "edge_jaccard_with_oracle": "0.5",
        "edge_jaccard_with_rank1": "0.4",
        "predicted_margin_from_best": "1.25",
        "true_obj_diff_from_rank1": str(100.0 * normalized_gap),
        "num_cycle_candidates": str(10 * max_cycle),
        "num_chain_candidates": "40",
    }


def case_index_row(case_id: str, case_label: str, subset_seed: int, graph_id: str):
    return {
        "case_id": case_id,
        "case_label": case_label,
        "case_type": "synthetic_case_type",
        "subset_seed": str(subset_seed),
        "graph_id": graph_id,
    }


class DecisionAnalysisCycleSensitivityTests(unittest.TestCase):
    def load_module(self):
        self.assertTrue(SCRIPT_PATH.exists(), f"Missing script: {SCRIPT_PATH}")
        spec = importlib.util.spec_from_file_location("summarize_cycle_sensitivity", SCRIPT_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def test_build_second_best_summary_groups_by_cycle_method_and_rank(self):
        module = self.load_module()
        rows = [
            solution_row(3, 1, "G-1.json", "2stage_val_mse", 2, 0.04),
            solution_row(3, 1, "G-2.json", "2stage_val_mse", 2, 0.08),
            solution_row(3, 1, "G-1.json", "2stage_val_mse", 1, 0.00, True),
            solution_row(4, 1, "G-1.json", "2stage_val_mse", 2, 0.10),
            solution_row(4, 1, "G-2.json", "2stage_val_mse", 2, 0.20),
            solution_row(4, 1, "G-1.json", "spoplus_val_spoplus_loss", 2, 0.03),
        ]

        summary = module.build_second_best_summary(rows)
        by_key = {
            (int(row["max_cycle"]), row["method_label"], int(row["solution_rank"])): row
            for row in summary
        }

        two_stage_rank2_cycle3 = by_key[(3, "2stage_val_mse", 2)]
        self.assertEqual(two_stage_rank2_cycle3["row_count"], 2)
        self.assertAlmostEqual(two_stage_rank2_cycle3["mean_normalized_gap_to_oracle"], 0.06)
        self.assertAlmostEqual(two_stage_rank2_cycle3["median_normalized_gap_to_oracle"], 0.06)
        self.assertAlmostEqual(two_stage_rank2_cycle3["near_5pct_rate"], 0.5)
        self.assertAlmostEqual(two_stage_rank2_cycle3["near_1pct_rate"], 0.0)
        self.assertEqual(two_stage_rank2_cycle3["mean_num_cycle_candidates"], 30.0)

    def test_build_rank2_gap_by_case_pairs_rank1_and_rank2_for_each_cycle(self):
        module = self.load_module()
        rows = [
            solution_row(3, 7, "G-1.json", "2stage_val_mse", 1, 0.01),
            solution_row(3, 7, "G-1.json", "2stage_val_mse", 2, 0.04),
            solution_row(4, 7, "G-1.json", "2stage_val_mse", 1, 0.03),
            solution_row(4, 7, "G-1.json", "2stage_val_mse", 2, 0.12),
            solution_row(4, 7, "G-2.json", "2stage_val_mse", 1, 0.02),
        ]

        output = module.build_rank2_gap_by_case(rows)

        self.assertEqual(len(output), 2)
        first = output[0]
        self.assertEqual(first["max_cycle"], 3)
        self.assertEqual(first["subset_seed"], 7)
        self.assertEqual(first["graph_id"], "G-1.json")
        self.assertAlmostEqual(first["rank1_normalized_gap"], 0.01)
        self.assertAlmostEqual(first["rank2_normalized_gap"], 0.04)
        self.assertAlmostEqual(first["rank2_minus_rank1_normalized_gap"], 0.03)

    def test_build_case_summary_uses_case_index_and_cycle_case_rows(self):
        module = self.load_module()
        case_rows = [
            case_index_row("case_b_001", "case_b_different_solution_near_optimal", 7, "G-1.json"),
            case_index_row("case_c_001", "case_c_spoplus_fixes_2stage", 8, "G-2.json"),
        ]
        rank_rows = [
            {
                "max_cycle": 3,
                "method_label": "2stage_val_mse",
                "subset_seed": 7,
                "graph_id": "G-1.json",
                "rank1_normalized_gap": 0.01,
                "rank2_normalized_gap": 0.04,
                "rank2_minus_rank1_normalized_gap": 0.03,
            },
            {
                "max_cycle": 3,
                "method_label": "2stage_val_mse",
                "subset_seed": 8,
                "graph_id": "G-2.json",
                "rank1_normalized_gap": 0.20,
                "rank2_normalized_gap": 0.05,
                "rank2_minus_rank1_normalized_gap": -0.15,
            },
        ]

        summary = module.build_case_summary(case_rows, rank_rows)
        by_label = {row["case_label"]: row for row in summary}

        case_b = by_label["case_b_different_solution_near_optimal"]
        self.assertEqual(case_b["row_count"], 1)
        self.assertEqual(case_b["near_5pct_rank2_count"], 1)
        self.assertAlmostEqual(case_b["rank2_near_5pct_rate"], 1.0)
        self.assertAlmostEqual(case_b["mean_rank2_minus_rank1_normalized_gap"], 0.03)

    def test_build_rank2_paired_delta_rows_compares_k4_and_k5_against_k3(self):
        module = self.load_module()
        rank_rows = [
            {
                "regime": "step2b_poly_d8",
                "max_cycle": 3,
                "case_type": "synthetic_case_type",
                "subset_seed": 7,
                "graph_id": "G-1.json",
                "method_label": "2stage_val_mse",
                "rank2_gap_to_oracle": 4.0,
                "rank2_normalized_gap": 0.04,
                "rank2_same_oracle": False,
                "num_cycle_candidates": 30.0,
            },
            {
                "regime": "step2b_poly_d8",
                "max_cycle": 4,
                "case_type": "synthetic_case_type",
                "subset_seed": 7,
                "graph_id": "G-1.json",
                "method_label": "2stage_val_mse",
                "rank2_gap_to_oracle": 3.0,
                "rank2_normalized_gap": 0.03,
                "rank2_same_oracle": False,
                "num_cycle_candidates": 40.0,
            },
            {
                "regime": "step2b_poly_d8",
                "max_cycle": 5,
                "case_type": "synthetic_case_type",
                "subset_seed": 7,
                "graph_id": "G-1.json",
                "method_label": "2stage_val_mse",
                "rank2_gap_to_oracle": 6.0,
                "rank2_normalized_gap": 0.06,
                "rank2_same_oracle": False,
                "num_cycle_candidates": 50.0,
            },
        ]

        deltas = module.build_rank2_paired_delta_rows(rank_rows, baseline_cycle=3)

        self.assertEqual(len(deltas), 2)
        first = deltas[0]
        self.assertEqual(first["baseline_max_cycle"], 3)
        self.assertEqual(first["comparison_max_cycle"], 4)
        self.assertEqual(first["subset_seed"], 7)
        self.assertEqual(first["graph_id"], "G-1.json")
        self.assertAlmostEqual(first["baseline_rank2_normalized_gap"], 0.04)
        self.assertAlmostEqual(first["comparison_rank2_normalized_gap"], 0.03)
        self.assertAlmostEqual(first["delta_rank2_normalized_gap"], -0.01)
        self.assertAlmostEqual(first["delta_rank2_gap_to_oracle"], -1.0)

    def test_build_rank2_paired_delta_summary_reports_sign_fractions_and_quartiles(self):
        module = self.load_module()
        delta_rows = [
            {
                "comparison_max_cycle": 4,
                "method_label": "2stage_val_mse",
                "baseline_rank2_normalized_gap": 0.10,
                "comparison_rank2_normalized_gap": 0.08,
                "delta_rank2_normalized_gap": -0.02,
            },
            {
                "comparison_max_cycle": 4,
                "method_label": "2stage_val_mse",
                "baseline_rank2_normalized_gap": 0.10,
                "comparison_rank2_normalized_gap": 0.10,
                "delta_rank2_normalized_gap": 0.00,
            },
            {
                "comparison_max_cycle": 4,
                "method_label": "2stage_val_mse",
                "baseline_rank2_normalized_gap": 0.10,
                "comparison_rank2_normalized_gap": 0.13,
                "delta_rank2_normalized_gap": 0.03,
            },
            {
                "comparison_max_cycle": 4,
                "method_label": "2stage_val_mse",
                "baseline_rank2_normalized_gap": 0.10,
                "comparison_rank2_normalized_gap": 0.14,
                "delta_rank2_normalized_gap": 0.04,
            },
        ]

        summary = module.build_rank2_paired_delta_summary(delta_rows)

        self.assertEqual(len(summary), 1)
        row = summary[0]
        self.assertEqual(row["paired_count"], 4)
        self.assertAlmostEqual(row["fraction_delta_lt_0"], 0.25)
        self.assertAlmostEqual(row["fraction_delta_eq_0"], 0.25)
        self.assertAlmostEqual(row["fraction_delta_gt_0"], 0.50)
        self.assertAlmostEqual(row["mean_delta_rank2_normalized_gap"], 0.0125)
        self.assertAlmostEqual(row["median_delta_rank2_normalized_gap"], 0.015)
        self.assertAlmostEqual(row["q25_delta_rank2_normalized_gap"], -0.005)
        self.assertAlmostEqual(row["q75_delta_rank2_normalized_gap"], 0.0325)


if __name__ == "__main__":
    unittest.main()
