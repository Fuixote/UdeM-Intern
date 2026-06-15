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
    / "summarize_all400_model_seed_baseline.py"
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


class All400ModelSeedBaselineTests(unittest.TestCase):
    def load_module(self):
        self.assertTrue(SCRIPT_PATH.exists(), f"Missing script: {SCRIPT_PATH}")
        spec = importlib.util.spec_from_file_location(
            "summarize_all400_model_seed_baseline",
            SCRIPT_PATH,
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def test_seed_rows_compute_generic_case_and_promotion_flags(self):
        module = self.load_module()
        rows = [
            solution_row("G-a.json", 0, "2stage_val_mse", 1, 0.25, "two-a-r1"),
            solution_row("G-a.json", 0, "2stage_val_mse", 2, 0.01, "near-r2"),
            solution_row("G-a.json", 0, "2stage_val_mse", 3, 0.03, "near-r3"),
            solution_row("G-a.json", 0, "spoplus_val_spoplus_loss", 1, 0.01, "near-r2"),
            solution_row("G-b.json", 0, "2stage_val_mse", 1, 0.25, "two-b-r1"),
            solution_row("G-b.json", 0, "2stage_val_mse", 2, 0.15, "two-b-r2"),
            solution_row("G-b.json", 0, "spoplus_val_spoplus_loss", 1, 0.01, "spo-b-r1"),
        ]

        audit_rows = module.build_seed_audit_rows(rows, top_k=5)

        by_graph = {row["graph_id"]: row for row in audit_rows}
        promotion = by_graph["G-a.json"]
        self.assertTrue(promotion["strict_case_c"])
        self.assertTrue(promotion["strong_case_c"])
        self.assertTrue(promotion["exact_rank2_promotion"])
        self.assertTrue(promotion["topk_promotion"])
        self.assertFalse(promotion["correction"])
        self.assertEqual(promotion["spoplus_matches_2stage_rank"], 2)

        correction = by_graph["G-b.json"]
        self.assertTrue(correction["strict_case_c"])
        self.assertTrue(correction["correction"])
        self.assertFalse(correction["exact_rank2_promotion"])
        self.assertFalse(correction["topk_promotion"])
        self.assertEqual(correction["spoplus_matches_2stage_rank"], "")

    def test_graph_summary_aggregates_rates_and_continuous_metrics(self):
        module = self.load_module()
        rows = [
            {
                "regime": "step2c_poly_d8_mult_eps050",
                "graph_id": "G-a.json",
                "subset_seed": 0,
                "spo_better": True,
                "meaningful_spo_benefit": True,
                "strict_case_c": True,
                "strong_case_c": True,
                "correction": False,
                "exact_rank2_promotion": True,
                "topk_promotion": True,
                "delta_pp": 20.0,
                "two_stage_rank1_gap_pct": 25.0,
                "spoplus_rank1_gap_pct": 5.0,
                "two_stage_rank2_gap_pct": 1.0,
            },
            {
                "regime": "step2c_poly_d8_mult_eps050",
                "graph_id": "G-a.json",
                "subset_seed": 1,
                "spo_better": True,
                "meaningful_spo_benefit": False,
                "strict_case_c": False,
                "strong_case_c": False,
                "correction": False,
                "exact_rank2_promotion": False,
                "topk_promotion": True,
                "delta_pp": 2.0,
                "two_stage_rank1_gap_pct": 8.0,
                "spoplus_rank1_gap_pct": 6.0,
                "two_stage_rank2_gap_pct": 3.0,
            },
        ]

        summary = module.summarize_by_graph(rows)

        self.assertEqual(len(summary), 1)
        row = summary[0]
        self.assertEqual(row["seed_count"], 2)
        self.assertEqual(row["strict_case_c_count"], 1)
        self.assertAlmostEqual(row["strict_case_c_rate"], 0.5)
        self.assertAlmostEqual(row["topk_promotion_rate"], 1.0)
        self.assertAlmostEqual(row["median_delta_pp"], 11.0)
        self.assertAlmostEqual(row["median_two_stage_rank1_gap_pct"], 16.5)

    def test_target_percentiles_use_midrank_and_leq(self):
        module = self.load_module()
        graph_rows = [
            {"graph_id": "G-low.json", "strict_case_c_rate": 0.0, "median_delta_pp": 0.0},
            {"graph_id": "G-392.json", "strict_case_c_rate": 1.0, "median_delta_pp": 20.0},
            {"graph_id": "G-tie.json", "strict_case_c_rate": 1.0, "median_delta_pp": 20.0},
            {"graph_id": "G-high.json", "strict_case_c_rate": 1.0, "median_delta_pp": 30.0},
        ]

        rows = module.build_target_percentile_rows(
            graph_rows,
            target_graphs=["G-392.json"],
            metrics=["strict_case_c_rate", "median_delta_pp"],
        )

        by_metric = {row["metric"]: row for row in rows}
        strict = by_metric["strict_case_c_rate"]
        self.assertEqual(strict["tie_count"], 3)
        self.assertAlmostEqual(strict["percentile_midrank"], 62.5)
        self.assertAlmostEqual(strict["percentile_leq"], 100.0)
        self.assertEqual(strict["rank_desc"], 1)

        delta = by_metric["median_delta_pp"]
        self.assertEqual(delta["tie_count"], 2)
        self.assertAlmostEqual(delta["percentile_midrank"], 50.0)
        self.assertAlmostEqual(delta["percentile_leq"], 75.0)
        self.assertEqual(delta["rank_desc"], 2)


if __name__ == "__main__":
    unittest.main()
