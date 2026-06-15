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
    / "build_step2c_top20_prediction_boundary_suitability.py"
)
WRAPPER_PATH = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "Step2c Graph-Level DFL Suitability Audit"
    / "scripts"
    / "build_phase4_top20_prediction_boundary.py"
)


def top20_row(
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


def seed_rows(graph_id: str, seed: int, rank1_obj: float, scale: float) -> list[dict[str, str]]:
    rows = []
    for rank in range(1, 21):
        signature = "0" if rank == 1 else str(rank)
        rows.append(
            top20_row(
                graph_id,
                seed,
                rank,
                rank1_obj - scale * (rank - 1),
                scale * (rank - 1),
                signature,
                1.0 if rank == 1 else 0.5,
            )
        )
    return rows


class Step2cTop20PredictionBoundarySuitabilityTests(unittest.TestCase):
    def load_module(self):
        self.assertTrue(SCRIPT_PATH.exists(), f"Missing script: {SCRIPT_PATH}")
        spec = importlib.util.spec_from_file_location(
            "build_step2c_top20_prediction_boundary_suitability", SCRIPT_PATH
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def test_top20_boundary_rows_summarize_margins_near_ties_and_diversity(self):
        module = self.load_module()
        rows = [
            *seed_rows("G-x.json", 0, rank1_obj=100.0, scale=1.0),
            *seed_rows("G-x.json", 1, rank1_obj=200.0, scale=2.0),
            top20_row("G-x.json", 1, 1, 999.0, 0.0, "spo", 1.0, method="spoplus_val_spoplus_loss"),
        ]

        output = module.build_top20_boundary_rows(rows)

        self.assertEqual(len(output), 1)
        row = output[0]
        self.assertEqual(row["graph_id"], "G-x.json")
        self.assertEqual(row["seed_count"], 2)
        self.assertAlmostEqual(row["median_2stage_top1_top20_pred_margin"], 28.5)
        self.assertAlmostEqual(row["median_2stage_top1_top20_pred_margin_pct"], 0.19)
        self.assertAlmostEqual(row["mean_2stage_top20_within_1pct_count"], 2.0)
        self.assertAlmostEqual(row["mean_2stage_top20_within_5pct_count"], 6.0)
        self.assertAlmostEqual(row["median_2stage_top20_mean_jaccard_to_rank1"], 0.5)
        self.assertAlmostEqual(row["median_2stage_top20_diversity_from_rank1"], 0.5)
        self.assertAlmostEqual(row["median_2stage_top20_mean_pairwise_jaccard"], 0.0)
        self.assertAlmostEqual(row["median_2stage_top20_pairwise_diversity"], 1.0)
        self.assertEqual(row["rank1_unique_signature_count"], 1)
        self.assertAlmostEqual(row["rank1_modal_signature_rate"], 1.0)

    def test_join_top20_with_phase2_keeps_existing_fields_and_adds_top20_score(self):
        module = self.load_module()
        phase2_rows = [
            {"graph_id": "G-a.json", "median_delta_pp": "20", "helpful_graph": "1", "harmful_graph": "0"},
            {"graph_id": "G-b.json", "median_delta_pp": "-10", "helpful_graph": "0", "harmful_graph": "1"},
        ]
        top20_rows = [
            {
                "graph_id": "G-a.json",
                "median_2stage_top1_top20_pred_margin_pct": 0.01,
                "mean_2stage_top20_within_1pct_count": 4.0,
                "median_2stage_top20_pairwise_diversity": 0.8,
            },
            {
                "graph_id": "G-b.json",
                "median_2stage_top1_top20_pred_margin_pct": 0.30,
                "mean_2stage_top20_within_1pct_count": 1.0,
                "median_2stage_top20_pairwise_diversity": 0.1,
            },
        ]

        joined = module.join_phase2_with_top20(phase2_rows, top20_rows)
        by_graph = {row["graph_id"]: row for row in joined}

        self.assertEqual(by_graph["G-a.json"]["helpful_graph"], "1")
        self.assertGreater(
            by_graph["G-a.json"]["ranking_ambiguity_top20_score"],
            by_graph["G-b.json"]["ranking_ambiguity_top20_score"],
        )

    def test_phase4_association_includes_top20_prediction_boundary_family(self):
        module = self.load_module()
        rows = [
            {
                "graph_id": "G-1.json",
                "median_delta_pp": "20",
                "helpful_graph": "1",
                "harmful_graph": "0",
                "ranking_ambiguity_top20_score": 2.0,
            },
            {
                "graph_id": "G-2.json",
                "median_delta_pp": "10",
                "helpful_graph": "1",
                "harmful_graph": "0",
                "ranking_ambiguity_top20_score": 1.0,
            },
            {
                "graph_id": "G-3.json",
                "median_delta_pp": "0",
                "helpful_graph": "0",
                "harmful_graph": "0",
                "ranking_ambiguity_top20_score": -1.0,
            },
            {
                "graph_id": "G-4.json",
                "median_delta_pp": "-10",
                "helpful_graph": "0",
                "harmful_graph": "1",
                "ranking_ambiguity_top20_score": -2.0,
            },
        ]

        associations = module.build_phase4_association_rows(rows)
        by_feature = {row["feature"]: row for row in associations}

        self.assertEqual(
            by_feature["ranking_ambiguity_top20_score"]["feature_family"],
            "top20_prediction_boundary",
        )
        self.assertAlmostEqual(
            by_feature["ranking_ambiguity_top20_score"]["spearman_median_delta_pp"],
            1.0,
        )
        self.assertAlmostEqual(by_feature["ranking_ambiguity_top20_score"]["auroc_helpful"], 1.0)

    def test_experiment_directory_exposes_phase4_entrypoint(self):
        self.assertTrue(WRAPPER_PATH.exists(), f"Missing wrapper: {WRAPPER_PATH}")


if __name__ == "__main__":
    unittest.main()
