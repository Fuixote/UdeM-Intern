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
    / "build_step2c_prediction_boundary_suitability.py"
)
WRAPPER_PATH = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "Step2c Graph-Level DFL Suitability Audit"
    / "scripts"
    / "build_phase2_prediction_boundary.py"
)


def top5_row(
    graph_id: str,
    seed: int,
    rank: int,
    predicted_obj: float,
    margin: float,
    signature: str,
    jaccard_rank1: float,
    method: str = "2stage_val_mse",
) -> dict[str, str]:
    return {
        "graph_id": graph_id,
        "subset_seed": str(seed),
        "method_label": method,
        "solution_rank": str(rank),
        "predicted_obj": str(predicted_obj),
        "predicted_margin_from_best": str(margin),
        "solution_edge_signature": signature,
        "edge_jaccard_with_rank1": str(jaccard_rank1),
    }


class Step2cPredictionBoundarySuitabilityTests(unittest.TestCase):
    def load_module(self):
        self.assertTrue(SCRIPT_PATH.exists(), f"Missing script: {SCRIPT_PATH}")
        spec = importlib.util.spec_from_file_location(
            "build_step2c_prediction_boundary_suitability", SCRIPT_PATH
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def test_boundary_rows_summarize_2stage_top5_margins_diversity_and_stability(self):
        module = self.load_module()
        rows = [
            top5_row("G-x.json", 0, 1, 100.0, 0.0, "a", 1.0),
            top5_row("G-x.json", 0, 2, 99.0, 1.0, "b", 0.5),
            top5_row("G-x.json", 0, 3, 98.0, 2.0, "c", 0.25),
            top5_row("G-x.json", 0, 4, 95.0, 5.0, "d", 0.0),
            top5_row("G-x.json", 0, 5, 90.0, 10.0, "e", 0.0),
            top5_row("G-x.json", 1, 1, 200.0, 0.0, "a", 1.0),
            top5_row("G-x.json", 1, 2, 198.0, 2.0, "f", 0.25),
            top5_row("G-x.json", 1, 3, 196.0, 4.0, "g", 0.25),
            top5_row("G-x.json", 1, 4, 190.0, 10.0, "h", 0.25),
            top5_row("G-x.json", 1, 5, 180.0, 20.0, "i", 0.25),
            top5_row("G-x.json", 1, 1, 999.0, 0.0, "spo", 1.0, method="spoplus_val_spoplus_loss"),
        ]

        output = module.build_prediction_boundary_rows(rows)

        self.assertEqual(len(output), 1)
        row = output[0]
        self.assertEqual(row["graph_id"], "G-x.json")
        self.assertEqual(row["seed_count"], 2)
        self.assertAlmostEqual(row["median_2stage_top1_top2_pred_margin"], 1.5)
        self.assertAlmostEqual(row["median_2stage_top1_top5_pred_margin"], 15.0)
        self.assertAlmostEqual(row["median_2stage_top1_top2_pred_margin_pct"], 0.01)
        self.assertAlmostEqual(row["mean_2stage_top5_within_1pct_count"], 2.0)
        self.assertAlmostEqual(row["mean_2stage_top5_within_5pct_count"], 4.0)
        self.assertAlmostEqual(row["median_2stage_top5_diversity_from_rank1"], 0.78125)
        self.assertEqual(row["rank1_unique_signature_count"], 1)
        self.assertAlmostEqual(row["rank1_modal_signature_rate"], 1.0)

    def test_join_boundary_with_phase1_keeps_labels_and_adds_ambiguity_score(self):
        module = self.load_module()
        phase1_rows = [
            {
                "graph_id": "G-a.json",
                "median_delta_pp": "12",
                "helpful_graph": "1",
                "harmful_graph": "0",
            },
            {
                "graph_id": "G-b.json",
                "median_delta_pp": "-12",
                "helpful_graph": "0",
                "harmful_graph": "1",
            },
        ]
        boundary_rows = [
            {
                "graph_id": "G-a.json",
                "median_2stage_top1_top2_pred_margin_pct": 0.01,
                "median_2stage_top1_top5_pred_margin_pct": 0.05,
                "mean_2stage_top5_within_1pct_count": 2.0,
                "median_2stage_top5_diversity_from_rank1": 0.8,
            },
            {
                "graph_id": "G-b.json",
                "median_2stage_top1_top2_pred_margin_pct": 0.20,
                "median_2stage_top1_top5_pred_margin_pct": 0.50,
                "mean_2stage_top5_within_1pct_count": 1.0,
                "median_2stage_top5_diversity_from_rank1": 0.1,
            },
        ]

        joined = module.join_phase1_with_boundary(phase1_rows, boundary_rows)
        by_graph = {row["graph_id"]: row for row in joined}

        self.assertEqual(by_graph["G-a.json"]["helpful_graph"], "1")
        self.assertGreater(
            by_graph["G-a.json"]["ranking_ambiguity_score"],
            by_graph["G-b.json"]["ranking_ambiguity_score"],
        )

    def test_phase2_association_includes_prediction_boundary_family(self):
        module = self.load_module()
        rows = [
            {
                "graph_id": "G-1.json",
                "median_delta_pp": "20",
                "helpful_graph": "1",
                "harmful_graph": "0",
                "median_2stage_top1_top2_pred_margin_pct": 0.01,
                "ranking_ambiguity_score": 2.0,
            },
            {
                "graph_id": "G-2.json",
                "median_delta_pp": "10",
                "helpful_graph": "1",
                "harmful_graph": "0",
                "median_2stage_top1_top2_pred_margin_pct": 0.02,
                "ranking_ambiguity_score": 1.0,
            },
            {
                "graph_id": "G-3.json",
                "median_delta_pp": "0",
                "helpful_graph": "0",
                "harmful_graph": "0",
                "median_2stage_top1_top2_pred_margin_pct": 0.20,
                "ranking_ambiguity_score": -1.0,
            },
            {
                "graph_id": "G-4.json",
                "median_delta_pp": "-10",
                "helpful_graph": "0",
                "harmful_graph": "1",
                "median_2stage_top1_top2_pred_margin_pct": 0.30,
                "ranking_ambiguity_score": -2.0,
            },
        ]

        associations = module.build_phase2_association_rows(rows)
        by_feature = {row["feature"]: row for row in associations}

        self.assertEqual(
            by_feature["ranking_ambiguity_score"]["feature_family"],
            "prediction_boundary",
        )
        self.assertAlmostEqual(
            by_feature["ranking_ambiguity_score"]["spearman_median_delta_pp"],
            1.0,
        )
        self.assertAlmostEqual(by_feature["ranking_ambiguity_score"]["auroc_helpful"], 1.0)

    def test_experiment_directory_exposes_phase2_entrypoint(self):
        self.assertTrue(WRAPPER_PATH.exists(), f"Missing wrapper: {WRAPPER_PATH}")


if __name__ == "__main__":
    unittest.main()
