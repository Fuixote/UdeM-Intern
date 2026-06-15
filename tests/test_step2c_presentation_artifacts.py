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
    / "build_step2c_presentation_artifacts.py"
)
WRAPPER_PATH = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "Step2c Mechanism Dissection Audit"
    / "scripts"
    / "build_presentation_artifacts.py"
)


class Step2cPresentationArtifactTests(unittest.TestCase):
    def load_module(self):
        self.assertTrue(SCRIPT_PATH.exists(), f"Missing script: {SCRIPT_PATH}")
        spec = importlib.util.spec_from_file_location(
            "build_step2c_presentation_artifacts", SCRIPT_PATH
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def test_paper_mechanism_rows_merge_rank_and_edge_summaries(self):
        module = self.load_module()
        rows = module.build_paper_mechanism_rows(
            atlas_rows=[
                {
                    "graph_id": "G-392.json",
                    "median_two_stage_rank1_gap_pct": "25.01",
                    "median_spoplus_rank1_gap_pct": "0.83",
                    "median_delta_pp": "24.18",
                }
            ],
            rank_summary_rows=[
                {
                    "graph_id": "G-392.json",
                    "best_near_found_rate": "1.0",
                    "spoplus_equals_best_near_rate": "1.0",
                    "median_spoplus_rank_under_2stage": "8",
                    "median_spoplus_true_rank": "3",
                }
            ],
            edge_summary_rows=[
                {
                    "graph_id": "G-392.json",
                    "comparison": "two_stage_rank1_vs_spoplus_rank1",
                    "median_signed_true_value_delta": "61.58",
                    "median_signed_pred_2stage_delta": "-4.24",
                    "median_signed_pred_spoplus_delta": "3.49",
                }
            ],
            graph_order=["G-392.json"],
            mechanism_labels={"G-392.json": "Deep-candidate correction"},
        )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["graph_id"], "G-392")
        self.assertEqual(row["mechanism_label"], "Deep-candidate correction")
        self.assertEqual(row["rank_reversal_pattern"], "helpful_reversal")
        self.assertEqual(row["c_rank_under_2stage"], "8")
        self.assertEqual(row["c_true_rank"], "3")
        self.assertEqual(row["true_delta_a_to_c"], "61.58")
        self.assertEqual(row["two_stage_pred_delta_a_to_c"], "-4.24")
        self.assertEqual(row["spoplus_pred_delta_a_to_c"], "3.49")

    def test_top_critical_edges_are_frequency_weighted_and_role_labeled(self):
        module = self.load_module()
        rows = [
            {
                "graph_id": "G-x.json",
                "subset_seed": "0",
                "comparison": "two_stage_rank1_vs_spoplus_rank1",
                "edge_id": "5",
                "src": "1",
                "dst": "2",
                "edge_in_left": "True",
                "edge_in_right": "False",
                "edge_in_oracle": "False",
                "signed_true_value_delta_right_minus_left": "-10",
                "signed_pred_2stage_delta_right_minus_left": "-2",
                "signed_pred_spoplus_delta_right_minus_left": "-1",
                "delta_prediction_spoplus_minus_2stage": "1",
                "error_2stage": "4",
                "error_spoplus": "1",
            },
            {
                "graph_id": "G-x.json",
                "subset_seed": "1",
                "comparison": "two_stage_rank1_vs_spoplus_rank1",
                "edge_id": "5",
                "src": "1",
                "dst": "2",
                "edge_in_left": "True",
                "edge_in_right": "False",
                "edge_in_oracle": "False",
                "signed_true_value_delta_right_minus_left": "-10",
                "signed_pred_2stage_delta_right_minus_left": "-2",
                "signed_pred_spoplus_delta_right_minus_left": "-1",
                "delta_prediction_spoplus_minus_2stage": "1",
                "error_2stage": "4",
                "error_spoplus": "1",
            },
            {
                "graph_id": "G-x.json",
                "subset_seed": "0",
                "comparison": "two_stage_rank1_vs_spoplus_rank1",
                "edge_id": "9",
                "src": "3",
                "dst": "4",
                "edge_in_left": "False",
                "edge_in_right": "True",
                "edge_in_oracle": "True",
                "signed_true_value_delta_right_minus_left": "15",
                "signed_pred_2stage_delta_right_minus_left": "3",
                "signed_pred_spoplus_delta_right_minus_left": "6",
                "delta_prediction_spoplus_minus_2stage": "3",
                "error_2stage": "-2",
                "error_spoplus": "0",
            },
        ]

        top_edges = module.build_top_critical_edge_rows(
            rows,
            graph_order=["G-x.json"],
            top_n=2,
        )

        self.assertEqual(len(top_edges), 2)
        self.assertEqual(top_edges[0]["edge_id"], "5")
        self.assertEqual(top_edges[0]["edge_role"], "removed_from_2stage_rank1")
        self.assertEqual(top_edges[0]["edge_frequency"], "1.00")
        self.assertEqual(top_edges[0]["mean_signed_true_delta"], "-10.00")
        self.assertEqual(top_edges[1]["edge_id"], "9")
        self.assertEqual(top_edges[1]["edge_role"], "added_by_spoplus_rank1")

    def test_case_panel_rows_keep_requested_graph_order(self):
        module = self.load_module()
        paper_rows = [
            {"graph_id": "G-14", "mechanism_label": "Harmful", "true_delta_a_to_c": "-28.14"},
            {"graph_id": "G-392", "mechanism_label": "Deep correction", "true_delta_a_to_c": "61.58"},
        ]

        panels = module.build_case_panel_rows(paper_rows, ["G-392", "G-14"])

        self.assertEqual([row["graph_id"] for row in panels], ["G-392", "G-14"])
        self.assertEqual(panels[0]["panel"], "A")
        self.assertEqual(panels[1]["panel"], "B")
        self.assertEqual(panels[0]["outcome"], "SPO+ helps")
        self.assertEqual(panels[1]["outcome"], "SPO+ hurts")

    def test_experiment_directory_exposes_presentation_entrypoint(self):
        self.assertTrue(WRAPPER_PATH.exists(), f"Missing wrapper: {WRAPPER_PATH}")


if __name__ == "__main__":
    unittest.main()
