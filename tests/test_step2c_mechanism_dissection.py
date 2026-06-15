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
    / "summarize_step2c_mechanism_dissection.py"
)


def predicted_row(
    graph_id: str,
    subset_seed: int,
    method_label: str,
    solution_rank: int,
    normalized_gap: float,
    signature: str,
    jaccard_oracle: float,
) -> dict[str, str]:
    return {
        "regime": "step2c_poly_d8_mult_eps050",
        "graph_id": graph_id,
        "subset_seed": str(subset_seed),
        "method_label": method_label,
        "solution_rank": str(solution_rank),
        "normalized_gap_to_oracle": str(normalized_gap),
        "solution_edge_signature": signature,
        "edge_jaccard_with_oracle": str(jaccard_oracle),
    }


class Step2cMechanismDissectionTests(unittest.TestCase):
    def load_module(self):
        self.assertTrue(SCRIPT_PATH.exists(), f"Missing script: {SCRIPT_PATH}")
        spec = importlib.util.spec_from_file_location(
            "summarize_step2c_mechanism_dissection", SCRIPT_PATH
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def test_candidate_basin_diagnostic_tracks_top20_and_structural_overlap(self):
        module = self.load_module()
        rows = [
            predicted_row("G-x.json", 0, "2stage_val_mse", 1, 0.25, "1|2", 0.1),
            predicted_row("G-x.json", 0, "2stage_val_mse", 2, 0.20, "3|4", 0.2),
            predicted_row("G-x.json", 0, "2stage_val_mse", 6, 0.01, "5|6|7", 0.9),
            predicted_row("G-x.json", 0, "spoplus_val_spoplus_loss", 1, 0.01, "5|6|7", 0.9),
            predicted_row("G-x.json", 1, "2stage_val_mse", 1, 0.30, "10|11", 0.0),
            predicted_row("G-x.json", 1, "2stage_val_mse", 2, 0.15, "12|13", 0.2),
            predicted_row("G-x.json", 1, "spoplus_val_spoplus_loss", 1, 0.02, "12|13|14", 0.5),
        ]
        oracle_rows = [
            {"graph_id": "G-x.json", "solution_rank": "1", "solution_edge_signature": "5|6|7"},
            {"graph_id": "G-x.json", "solution_rank": "2", "solution_edge_signature": "12|13|14"},
        ]

        diagnostics = module.build_candidate_basin_diagnostics(
            rows,
            oracle_rows,
            top_k_values=(5, 20),
            near_oracle_gap=0.05,
        )

        self.assertEqual(len(diagnostics), 1)
        row = diagnostics[0]
        self.assertEqual(row["seed_count"], 2)
        self.assertAlmostEqual(row["rate_2stage_top5_contains_near_oracle"], 0.0)
        self.assertAlmostEqual(row["rate_2stage_top20_contains_near_oracle"], 0.5)
        self.assertAlmostEqual(row["rate_spoplus_rank1_in_2stage_top5"], 0.0)
        self.assertAlmostEqual(row["rate_spoplus_rank1_in_2stage_top20"], 0.5)
        self.assertAlmostEqual(row["rate_spoplus_rank1_in_true_top50"], 1.0)
        self.assertAlmostEqual(row["median_rank_of_best_near_oracle_under_2stage"], 6.0)
        self.assertAlmostEqual(
            row["median_jaccard_spoplus_rank1_to_nearest_2stage_top20"],
            5.0 / 6.0,
        )

    def test_mechanism_atlas_preserves_explicit_selected_graph_family_labels(self):
        module = self.load_module()
        graph_summary_rows = [
            {
                "graph_id": "G-392.json",
                "strict_case_c_rate": "1.0",
                "strong_case_c_rate": "1.0",
                "correction_rate": "1.0",
                "exact_rank2_promotion_rate": "0.0",
                "topk_promotion_rate": "0.0",
                "median_delta_pp": "24.18",
                "median_two_stage_rank1_gap_pct": "25.01",
                "median_spoplus_rank1_gap_pct": "0.83",
                "median_two_stage_rank2_gap_pct": "28.63",
            },
            {
                "graph_id": "G-1285.json",
                "strict_case_c_rate": "1.0",
                "strong_case_c_rate": "1.0",
                "correction_rate": "0.0",
                "exact_rank2_promotion_rate": "1.0",
                "topk_promotion_rate": "1.0",
                "median_delta_pp": "22.36",
                "median_two_stage_rank1_gap_pct": "22.36",
                "median_spoplus_rank1_gap_pct": "0.0",
                "median_two_stage_rank2_gap_pct": "0.0",
            },
        ]
        basin_rows = [
            {
                "graph_id": "G-392.json",
                "rate_2stage_top20_contains_near_oracle": 0.0,
                "rate_spoplus_rank1_in_2stage_top20": 0.0,
            },
            {
                "graph_id": "G-1285.json",
                "rate_2stage_top20_contains_near_oracle": 1.0,
                "rate_spoplus_rank1_in_2stage_top20": 1.0,
            },
        ]

        atlas = module.build_mechanism_atlas(graph_summary_rows, basin_rows)

        by_graph = {row["graph_id"]: row for row in atlas}
        self.assertEqual(by_graph["G-392.json"]["assigned_family"], "clean_correction")
        self.assertEqual(by_graph["G-1285.json"]["assigned_family"], "clean_exact_rank2_promotion")
        self.assertAlmostEqual(
            by_graph["G-1285.json"]["rate_spoplus_rank1_in_2stage_top20"], 1.0
        )


if __name__ == "__main__":
    unittest.main()
