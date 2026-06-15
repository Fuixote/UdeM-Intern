from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "decision_analysis"
    / "scripts"
    / "audit_rank_reversal_critical_edges.py"
)
WRAPPER_PATH = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "Step2c Mechanism Dissection Audit"
    / "scripts"
    / "audit_rank_reversal_critical_edges.py"
)


def solution_row(rank: int, signature: str, gap: float) -> dict[str, str]:
    return {
        "solution_rank": str(rank),
        "solution_edge_signature": signature,
        "normalized_gap_to_oracle": str(gap),
        "edge_jaccard_with_oracle": "0.0",
    }


class Step2cRankReversalCriticalEdgeTests(unittest.TestCase):
    def load_module(self):
        self.assertTrue(SCRIPT_PATH.exists(), f"Missing script: {SCRIPT_PATH}")
        spec = importlib.util.spec_from_file_location(
            "audit_rank_reversal_critical_edges", SCRIPT_PATH
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def test_build_rank_reversal_rows_selects_best_near_oracle_and_margins(self):
        module = self.load_module()
        two_stage = {
            1: solution_row(1, "0|1", 0.25),
            2: solution_row(2, "2|3", 0.20),
            8: solution_row(8, "4|5", 0.01),
        }
        spoplus = {
            1: solution_row(1, "4|5", 0.01),
            2: solution_row(2, "0|1", 0.25),
        }

        rows, roles = module.build_rank_reversal_rows_for_seed(
            graph_id="G-x.json",
            subset_seed=7,
            two_stage_rows=two_stage,
            spoplus_rows=spoplus,
            true_rank_by_signature={"4|5": 3, "0|1": 40},
            w_true=np.array([10.0, 5.0, 1.0, 1.0, 9.0, 8.0]),
            w_hat_2stage=np.array([10.0, 10.0, 4.0, 4.0, 3.0, 2.0]),
            w_hat_spoplus=np.array([2.0, 2.0, 1.0, 1.0, 10.0, 9.0]),
            y_oracle=module.mask_from_signature("4|5", 6),
            num_edges=6,
            near_oracle_gap=0.05,
        )

        by_role = {row["solution_role"]: row for row in rows}
        self.assertEqual(by_role["best_2stage_near_oracle_top20"]["rank_under_2stage"], 8)
        self.assertEqual(by_role["best_2stage_near_oracle_top20"]["rank_under_spoplus"], 1)
        self.assertEqual(by_role["best_2stage_near_oracle_top20"]["rank_under_true_oracle_top50"], 3)
        self.assertAlmostEqual(
            by_role["best_2stage_near_oracle_top20"]["two_stage_margin_vs_rank1"],
            15.0,
        )
        self.assertAlmostEqual(
            by_role["two_stage_rank1"]["spoplus_margin_vs_rank1"],
            15.0,
        )
        self.assertTrue(by_role["best_2stage_near_oracle_top20"]["found"])
        self.assertEqual(
            roles["best_2stage_near_oracle_top20"]["solution_edge_signature"],
            "4|5",
        )

    def test_missing_near_oracle_role_is_explicit(self):
        module = self.load_module()
        rows, roles = module.build_rank_reversal_rows_for_seed(
            graph_id="G-x.json",
            subset_seed=0,
            two_stage_rows={1: solution_row(1, "0", 0.30), 2: solution_row(2, "1", 0.20)},
            spoplus_rows={1: solution_row(1, "0", 0.30)},
            true_rank_by_signature={},
            w_true=np.array([1.0, 1.0]),
            w_hat_2stage=np.array([2.0, 1.0]),
            w_hat_spoplus=np.array([2.0, 1.0]),
            y_oracle=module.mask_from_signature("1", 2),
            num_edges=2,
            near_oracle_gap=0.05,
        )

        by_role = {row["solution_role"]: row for row in rows}
        self.assertFalse(by_role["best_2stage_near_oracle_top20"]["found"])
        self.assertEqual(roles["best_2stage_near_oracle_top20"], {})

    def test_critical_edge_rows_cover_only_solution_symdiff_with_signed_contributions(self):
        module = self.load_module()
        roles = {
            "two_stage_rank1": {"solution_edge_signature": "0|1"},
            "spoplus_rank1": {"solution_edge_signature": "1|2"},
        }

        rows = module.build_critical_edge_rows_for_comparison(
            graph_id="G-x.json",
            subset_seed=3,
            comparison="two_stage_rank1_vs_spoplus_rank1",
            left_role="two_stage_rank1",
            right_role="spoplus_rank1",
            roles=roles,
            w_true=np.array([5.0, 7.0, 11.0]),
            w_hat_2stage=np.array([8.0, 6.0, 1.0]),
            w_hat_spoplus=np.array([1.0, 6.5, 12.0]),
            y_oracle=module.mask_from_signature("1|2", 3),
            edge_src=np.array([10, 11, 12]),
            edge_dst=np.array([20, 21, 22]),
        )

        self.assertEqual([row["edge_id"] for row in rows], [0, 2])
        left_only = rows[0]
        right_only = rows[1]
        self.assertTrue(left_only["edge_in_left"])
        self.assertFalse(left_only["edge_in_right"])
        self.assertAlmostEqual(left_only["signed_true_value_delta_right_minus_left"], -5.0)
        self.assertFalse(right_only["edge_in_left"])
        self.assertTrue(right_only["edge_in_right"])
        self.assertTrue(right_only["edge_in_oracle"])
        self.assertAlmostEqual(right_only["signed_true_value_delta_right_minus_left"], 11.0)

    def test_summary_rows_aggregate_rank_and_edge_reversal_metrics(self):
        module = self.load_module()
        rank_rows = [
            {
                "graph_id": "G-x.json",
                "subset_seed": 0,
                "solution_role": "two_stage_rank1",
                "found": True,
                "rank_under_2stage": 1,
                "rank_under_true_oracle_top50": 40,
                "true_gap_pct": 20.0,
                "spoplus_margin_vs_rank1": 5.0,
                "solution_edge_signature": "0",
            },
            {
                "graph_id": "G-x.json",
                "subset_seed": 0,
                "solution_role": "best_2stage_near_oracle_top20",
                "found": True,
                "rank_under_2stage": 8,
                "rank_under_true_oracle_top50": 3,
                "true_gap_pct": 1.0,
                "solution_edge_signature": "1",
            },
            {
                "graph_id": "G-x.json",
                "subset_seed": 0,
                "solution_role": "spoplus_rank1",
                "found": True,
                "rank_under_2stage": 8,
                "rank_under_true_oracle_top50": 3,
                "true_gap_pct": 1.0,
                "two_stage_margin_vs_rank1": 4.0,
                "solution_edge_signature": "1",
            },
            {
                "graph_id": "G-x.json",
                "subset_seed": 1,
                "solution_role": "two_stage_rank1",
                "found": True,
                "rank_under_2stage": 1,
                "rank_under_true_oracle_top50": 10,
                "true_gap_pct": 10.0,
                "spoplus_margin_vs_rank1": 7.0,
                "solution_edge_signature": "2",
            },
            {
                "graph_id": "G-x.json",
                "subset_seed": 1,
                "solution_role": "best_2stage_near_oracle_top20",
                "found": False,
                "rank_under_2stage": "",
                "rank_under_true_oracle_top50": "",
                "true_gap_pct": "",
                "solution_edge_signature": "",
            },
            {
                "graph_id": "G-x.json",
                "subset_seed": 1,
                "solution_role": "spoplus_rank1",
                "found": True,
                "rank_under_2stage": 5,
                "rank_under_true_oracle_top50": "",
                "true_gap_pct": 8.0,
                "two_stage_margin_vs_rank1": 2.0,
                "solution_edge_signature": "3",
            },
        ]
        edge_rows = [
            {
                "graph_id": "G-x.json",
                "subset_seed": 0,
                "comparison": "two_stage_rank1_vs_spoplus_rank1",
                "signed_true_value_delta_right_minus_left": 10.0,
                "signed_pred_2stage_delta_right_minus_left": -2.0,
                "signed_pred_spoplus_delta_right_minus_left": 3.0,
            },
            {
                "graph_id": "G-x.json",
                "subset_seed": 0,
                "comparison": "two_stage_rank1_vs_spoplus_rank1",
                "signed_true_value_delta_right_minus_left": 5.0,
                "signed_pred_2stage_delta_right_minus_left": -1.0,
                "signed_pred_spoplus_delta_right_minus_left": 2.0,
            },
            {
                "graph_id": "G-x.json",
                "subset_seed": 1,
                "comparison": "two_stage_rank1_vs_spoplus_rank1",
                "signed_true_value_delta_right_minus_left": -4.0,
                "signed_pred_2stage_delta_right_minus_left": -3.0,
                "signed_pred_spoplus_delta_right_minus_left": 1.0,
            },
        ]

        rank_summary = module.summarize_rank_reversal_rows(rank_rows)
        edge_summary = module.summarize_critical_edge_rows(edge_rows)

        self.assertEqual(len(rank_summary), 1)
        self.assertAlmostEqual(rank_summary[0]["best_near_found_rate"], 0.5)
        self.assertAlmostEqual(rank_summary[0]["spoplus_equals_best_near_rate"], 0.5)
        self.assertAlmostEqual(rank_summary[0]["median_spoplus_rank_under_2stage"], 6.5)
        self.assertAlmostEqual(rank_summary[0]["median_two_stage_margin_to_spoplus"], 3.0)
        self.assertAlmostEqual(rank_summary[0]["median_spoplus_margin_to_two_stage"], 6.0)
        self.assertEqual(len(edge_summary), 1)
        self.assertAlmostEqual(edge_summary[0]["median_signed_true_value_delta"], 5.5)
        self.assertAlmostEqual(edge_summary[0]["median_signed_pred_2stage_delta"], -3.0)
        self.assertAlmostEqual(edge_summary[0]["median_signed_pred_spoplus_delta"], 3.0)

    def test_experiment_directory_exposes_rank_reversal_entrypoint(self):
        self.assertTrue(WRAPPER_PATH.exists(), f"Missing wrapper: {WRAPPER_PATH}")


if __name__ == "__main__":
    unittest.main()
