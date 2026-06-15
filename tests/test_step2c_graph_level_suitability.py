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
    / "build_step2c_graph_level_suitability.py"
)
WRAPPER_PATH = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "Step2c Graph-Level DFL Suitability Audit"
    / "scripts"
    / "build_phase1_graph_features.py"
)


def toy_graph() -> dict:
    return {
        "metadata": {"total_vertices": 4},
        "data": {
            "0": {"type": "Pair", "matches": [{"recipient": "1"}, {"recipient": "2"}]},
            "1": {"type": "Pair", "matches": [{"recipient": "0"}, {"recipient": "2"}]},
            "2": {"type": "Pair", "matches": [{"recipient": "0"}]},
            "3": {"type": "NDD", "matches": [{"recipient": "0"}, {"recipient": "2"}]},
        },
    }


class Step2cGraphLevelSuitabilityTests(unittest.TestCase):
    def load_module(self):
        self.assertTrue(SCRIPT_PATH.exists(), f"Missing script: {SCRIPT_PATH}")
        spec = importlib.util.spec_from_file_location(
            "build_step2c_graph_level_suitability", SCRIPT_PATH
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def test_graph_feature_rows_include_topology_exchange_and_conflict_features(self):
        module = self.load_module()

        row = module.compute_graph_feature_row(
            graph_id="G-toy.json",
            graph_json=toy_graph(),
            regime="test_regime",
            max_cycle=3,
            max_chain=4,
        )

        self.assertEqual(row["graph_id"], "G-toy.json")
        self.assertEqual(row["num_vertices"], 4)
        self.assertEqual(row["num_arcs"], 7)
        self.assertAlmostEqual(row["density"], 7 / 12)
        self.assertEqual(row["num_2cycles"], 2)
        self.assertEqual(row["num_3cycles"], 1)
        self.assertEqual(row["num_chain_len1"], 2)
        self.assertEqual(row["num_chain_len2"], 3)
        self.assertEqual(row["num_chain_len3"], 2)
        self.assertEqual(row["num_chains_total"], 7)
        self.assertEqual(row["num_exchange_candidates"], 10)
        self.assertGreater(row["exchange_size_entropy"], 0.0)
        self.assertEqual(row["max_vertex_exchange_participation"], 9)
        self.assertEqual(row["conflict_graph_num_nodes"], 10)
        self.assertGreater(row["conflict_graph_density"], 0.0)

    def test_join_features_with_outcomes_adds_phase1_labels_and_scores(self):
        module = self.load_module()
        feature_rows = [
            {
                "graph_id": "G-help.json",
                "num_exchange_candidates": 100,
                "num_cycles_total": 20,
                "exchange_size_entropy": 1.5,
                "vertex_exchange_participation_gini": 0.4,
            },
            {
                "graph_id": "G-neutral.json",
                "num_exchange_candidates": 10,
                "num_cycles_total": 1,
                "exchange_size_entropy": 0.2,
                "vertex_exchange_participation_gini": 0.1,
            },
            {
                "graph_id": "G-hurt.json",
                "num_exchange_candidates": 50,
                "num_cycles_total": 5,
                "exchange_size_entropy": 0.7,
                "vertex_exchange_participation_gini": 0.3,
            },
        ]
        outcome_rows = [
            {
                "graph_id": "G-help.json",
                "median_delta_pp": "12",
                "strict_case_c_rate": "0.4",
                "meaningful_spo_benefit_rate": "1.0",
            },
            {
                "graph_id": "G-neutral.json",
                "median_delta_pp": "0",
                "strict_case_c_rate": "0",
                "meaningful_spo_benefit_rate": "0",
            },
            {
                "graph_id": "G-hurt.json",
                "median_delta_pp": "-11",
                "strict_case_c_rate": "0",
                "meaningful_spo_benefit_rate": "0",
            },
        ]

        joined = module.join_features_with_outcomes(feature_rows, outcome_rows)
        by_graph = {row["graph_id"]: row for row in joined}

        self.assertEqual(by_graph["G-help.json"]["helpful_graph"], 1)
        self.assertEqual(by_graph["G-help.json"]["extreme_helpful_graph"], 1)
        self.assertEqual(by_graph["G-neutral.json"]["neutral_graph"], 1)
        self.assertEqual(by_graph["G-hurt.json"]["harmful_graph"], 1)
        self.assertIn("feasible_set_richness_score", by_graph["G-help.json"])

    def test_feature_associations_report_family_spearman_and_auroc(self):
        module = self.load_module()
        rows = [
            {
                "graph_id": "G-1.json",
                "median_delta_pp": "20",
                "helpful_graph": 1,
                "harmful_graph": 0,
                "density": 0.4,
                "num_exchange_candidates": 100,
            },
            {
                "graph_id": "G-2.json",
                "median_delta_pp": "10",
                "helpful_graph": 1,
                "harmful_graph": 0,
                "density": 0.3,
                "num_exchange_candidates": 80,
            },
            {
                "graph_id": "G-3.json",
                "median_delta_pp": "0",
                "helpful_graph": 0,
                "harmful_graph": 0,
                "density": 0.2,
                "num_exchange_candidates": 20,
            },
            {
                "graph_id": "G-4.json",
                "median_delta_pp": "-10",
                "helpful_graph": 0,
                "harmful_graph": 1,
                "density": 0.1,
                "num_exchange_candidates": 10,
            },
        ]

        associations = module.build_feature_association_rows(rows)
        by_feature = {row["feature"]: row for row in associations}

        self.assertEqual(by_feature["density"]["feature_family"], "raw_topology")
        self.assertEqual(
            by_feature["num_exchange_candidates"]["feature_family"],
            "exchange_geometry",
        )
        self.assertAlmostEqual(
            by_feature["num_exchange_candidates"]["spearman_median_delta_pp"],
            1.0,
        )
        self.assertAlmostEqual(by_feature["num_exchange_candidates"]["auroc_helpful"], 1.0)

    def test_selected_case_overlay_reports_feature_percentiles(self):
        module = self.load_module()
        rows = [
            {"graph_id": "G-1.json", "density": 0.1, "median_delta_pp": 0},
            {"graph_id": "G-392.json", "density": 0.3, "median_delta_pp": 24},
            {"graph_id": "G-3.json", "density": 0.2, "median_delta_pp": 5},
        ]

        overlay = module.build_selected_case_overlay_rows(
            rows,
            selected_graphs=["G-392.json"],
            feature_keys=["density"],
        )

        self.assertEqual(len(overlay), 1)
        self.assertEqual(overlay[0]["graph_id"], "G-392.json")
        self.assertAlmostEqual(overlay[0]["density_percentile"], 1.0)

    def test_experiment_directory_exposes_phase1_entrypoint(self):
        self.assertTrue(WRAPPER_PATH.exists(), f"Missing wrapper: {WRAPPER_PATH}")


if __name__ == "__main__":
    unittest.main()
