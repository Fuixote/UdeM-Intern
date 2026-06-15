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
    / "compute_true_oracle_landscape.py"
)


class Step2cOracleLandscapeTests(unittest.TestCase):
    def load_module(self):
        self.assertTrue(SCRIPT_PATH.exists(), f"Missing script: {SCRIPT_PATH}")
        spec = importlib.util.spec_from_file_location("compute_true_oracle_landscape", SCRIPT_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def test_topology_metrics_count_directed_edges_reciprocals_and_sccs(self):
        module = self.load_module()
        graph_json = {
            "metadata": {"total_vertices": 4},
            "data": {
                "0": {"type": "Pair", "matches": [{"recipient": "1"}, {"recipient": "2"}]},
                "1": {"type": "Pair", "matches": [{"recipient": "0"}, {"recipient": "2"}]},
                "2": {"type": "Pair", "matches": [{"recipient": "0"}]},
                "3": {"type": "NDD", "matches": [{"recipient": "2"}]},
            },
        }

        metrics = module.compute_raw_topology_metrics(graph_json)

        self.assertEqual(metrics["num_vertices"], 4)
        self.assertEqual(metrics["num_edges"], 6)
        self.assertAlmostEqual(metrics["density"], 0.5)
        self.assertEqual(metrics["reciprocal_edge_count"], 4)
        self.assertEqual(metrics["num_2cycles"], 2)
        self.assertEqual(metrics["num_3cycles"], 1)
        self.assertEqual(metrics["number_of_sccs"], 2)
        self.assertAlmostEqual(metrics["largest_scc_fraction"], 0.75)
        self.assertAlmostEqual(metrics["out_degree_mean"], 1.5)
        self.assertAlmostEqual(metrics["in_degree_mean"], 1.5)

    def test_oracle_landscape_summary_uses_observed_top_m_not_full_feasible_set(self):
        module = self.load_module()
        rows = [
            {
                "graph_id": "G-x.json",
                "solution_rank": 1,
                "normalized_gap_to_oracle": 0.0,
                "edge_jaccard_with_oracle": 1.0,
                "solution_edge_signature": "1|2|3",
                "oracle_obj": 100.0,
                "true_obj": 100.0,
                "num_edges": 10,
                "num_cycle_candidates": 5,
                "num_chain_candidates": 7,
            },
            {
                "graph_id": "G-x.json",
                "solution_rank": 2,
                "normalized_gap_to_oracle": 0.004,
                "edge_jaccard_with_oracle": 0.8,
                "solution_edge_signature": "1|2|4",
                "oracle_obj": 100.0,
                "true_obj": 99.6,
                "num_edges": 10,
                "num_cycle_candidates": 5,
                "num_chain_candidates": 7,
            },
            {
                "graph_id": "G-x.json",
                "solution_rank": 3,
                "normalized_gap_to_oracle": 0.06,
                "edge_jaccard_with_oracle": 0.2,
                "solution_edge_signature": "8",
                "oracle_obj": 100.0,
                "true_obj": 94.0,
                "num_edges": 10,
                "num_cycle_candidates": 5,
                "num_chain_candidates": 7,
            },
        ]

        summary = module.summarize_oracle_landscape_rows(rows, top_m=50)

        self.assertEqual(len(summary), 1)
        row = summary[0]
        self.assertEqual(row["observed_top_m"], 3)
        self.assertAlmostEqual(row["oracle_second_best_gap_pct"], 0.4)
        self.assertEqual(row["num_observed_solutions_within_1pct"], 2)
        self.assertEqual(row["num_observed_solutions_within_5pct"], 2)
        self.assertEqual(row["num_observed_solutions_within_10pct"], 3)
        self.assertAlmostEqual(row["near_oracle_jaccard_mean"], 0.9)
        self.assertAlmostEqual(row["near_oracle_jaccard_min"], 0.8)


if __name__ == "__main__":
    unittest.main()
