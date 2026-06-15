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
    / "summarize_fixed_graph_model_seed_audit.py"
)


def solution_row(
    graph_id: str,
    subset_seed: int,
    method_label: str,
    solution_rank: int,
    normalized_gap: float,
    signature: str,
) -> dict[str, str]:
    return {
        "regime": "step2c_poly_d8_mult_eps050",
        "graph_id": graph_id,
        "subset_seed": str(subset_seed),
        "method_label": method_label,
        "solution_rank": str(solution_rank),
        "normalized_gap_to_oracle": str(normalized_gap),
        "gap_to_oracle": str(100.0 * normalized_gap),
        "solution_edge_signature": signature,
    }


class FixedGraphModelSeedAuditTests(unittest.TestCase):
    def load_module(self):
        self.assertTrue(SCRIPT_PATH.exists(), f"Missing script: {SCRIPT_PATH}")
        spec = importlib.util.spec_from_file_location(
            "summarize_fixed_graph_model_seed_audit",
            SCRIPT_PATH,
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def test_build_seed_audit_rows_marks_correction_and_promotion_mechanisms(self):
        module = self.load_module()
        rows = [
            solution_row("G-392.json", 1, "2stage_val_mse", 1, 0.25, "g392_2s_r1"),
            solution_row("G-392.json", 1, "2stage_val_mse", 2, 0.12, "g392_2s_r2"),
            solution_row("G-392.json", 1, "spoplus_val_spoplus_loss", 1, 0.01, "g392_spo_r1"),
            solution_row("G-392.json", 1, "spoplus_val_spoplus_loss", 2, 0.02, "g392_spo_r2"),
            solution_row("G-1560.json", 30, "2stage_val_mse", 1, 0.25, "g1560_2s_r1"),
            solution_row("G-1560.json", 30, "2stage_val_mse", 2, 0.005, "g1560_2s_r2"),
            solution_row("G-1560.json", 30, "2stage_val_mse", 3, 0.03, "g1560_2s_r3"),
            solution_row("G-1560.json", 30, "spoplus_val_spoplus_loss", 1, 0.005, "g1560_2s_r2"),
        ]

        audit_rows = module.build_seed_audit_rows(
            rows,
            discovery_seeds={"G-392.json": 1, "G-1560.json": 30},
            top_k=5,
        )

        by_graph = {row["graph_id"]: row for row in audit_rows}
        g392 = by_graph["G-392.json"]
        self.assertAlmostEqual(g392["delta_pp"], 24.0)
        self.assertTrue(g392["strict_case_c"])
        self.assertTrue(g392["strong_case_c"])
        self.assertTrue(g392["correction_preserved"])
        self.assertFalse(g392["spo_rank1_equals_2stage_rank2"])
        self.assertFalse(g392["rank2_promotion_preserved"])

        g1560 = by_graph["G-1560.json"]
        self.assertAlmostEqual(g1560["two_stage_rank2_gap_pct"], 0.5)
        self.assertTrue(g1560["strict_case_c"])
        self.assertTrue(g1560["spo_rank1_equals_2stage_rank2"])
        self.assertEqual(g1560["spoplus_matches_2stage_rank"], 2)
        self.assertTrue(g1560["rank2_promotion_preserved"])
        self.assertTrue(g1560["topk_promotion_preserved"])

    def test_topk_promotion_allows_non_rank2_match(self):
        module = self.load_module()
        rows = [
            solution_row("G-1560.json", 31, "2stage_val_mse", 1, 0.22, "r1"),
            solution_row("G-1560.json", 31, "2stage_val_mse", 2, 0.03, "r2"),
            solution_row("G-1560.json", 31, "2stage_val_mse", 3, 0.02, "r3"),
            solution_row("G-1560.json", 31, "spoplus_val_spoplus_loss", 1, 0.02, "r3"),
        ]

        audit_rows = module.build_seed_audit_rows(
            rows,
            discovery_seeds={"G-1560.json": 30},
            top_k=5,
        )

        self.assertEqual(len(audit_rows), 1)
        row = audit_rows[0]
        self.assertTrue(row["strict_case_c"])
        self.assertFalse(row["spo_rank1_equals_2stage_rank2"])
        self.assertEqual(row["spoplus_matches_2stage_rank"], 3)
        self.assertFalse(row["rank2_promotion_preserved"])
        self.assertTrue(row["topk_promotion_preserved"])

    def test_summary_includes_discovery_seed_excluded_sensitivity(self):
        module = self.load_module()
        audit_rows = [
            {
                "regime": "step2c_poly_d8_mult_eps050",
                "graph_id": "G-392.json",
                "subset_seed": 1,
                "discovery_seed": 1,
                "is_discovery_seed": True,
                "spo_better": True,
                "meaningful_spo_benefit": True,
                "strict_case_c": True,
                "strong_case_c": True,
                "correction_preserved": True,
                "rank2_promotion_preserved": False,
                "topk_promotion_preserved": False,
                "spo_rank1_equals_2stage_rank2": False,
                "delta_pp": 24.0,
                "two_stage_rank1_gap_pct": 25.0,
                "spoplus_rank1_gap_pct": 1.0,
                "two_stage_rank2_gap_pct": 12.0,
                "spoplus_rank2_gap_pct": 2.0,
            },
            {
                "regime": "step2c_poly_d8_mult_eps050",
                "graph_id": "G-392.json",
                "subset_seed": 2,
                "discovery_seed": 1,
                "is_discovery_seed": False,
                "spo_better": True,
                "meaningful_spo_benefit": False,
                "strict_case_c": False,
                "strong_case_c": False,
                "correction_preserved": False,
                "rank2_promotion_preserved": False,
                "topk_promotion_preserved": False,
                "spo_rank1_equals_2stage_rank2": False,
                "delta_pp": 2.0,
                "two_stage_rank1_gap_pct": 8.0,
                "spoplus_rank1_gap_pct": 6.0,
                "two_stage_rank2_gap_pct": 9.0,
                "spoplus_rank2_gap_pct": 7.0,
            },
        ]

        summary = module.summarize_audit_rows(audit_rows)

        by_label = {row["seed_filter"]: row for row in summary}
        all_row = by_label["all"]
        excl_row = by_label["exclude_discovery_seed"]
        self.assertEqual(all_row["seed_count"], 2)
        self.assertEqual(all_row["strict_case_c_count"], 1)
        self.assertAlmostEqual(all_row["strict_case_c_rate"], 0.5)
        self.assertAlmostEqual(all_row["median_delta_pp"], 13.0)
        self.assertEqual(excl_row["seed_count"], 1)
        self.assertEqual(excl_row["strict_case_c_count"], 0)
        self.assertAlmostEqual(excl_row["median_delta_pp"], 2.0)

    def test_wilson_interval_handles_empty_and_nonempty_counts(self):
        module = self.load_module()

        self.assertEqual(module.wilson_interval(0, 0), ("", ""))
        low, high = module.wilson_interval(5, 10)

        self.assertAlmostEqual(low, 0.2366, places=3)
        self.assertAlmostEqual(high, 0.7634, places=3)


if __name__ == "__main__":
    unittest.main()
