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
    / "plot_decision_analysis.py"
)


class DecisionAnalysisPlotTests(unittest.TestCase):
    def load_module(self):
        self.assertTrue(SCRIPT_PATH.exists(), f"Missing script: {SCRIPT_PATH}")
        spec = importlib.util.spec_from_file_location("plot_decision_analysis", SCRIPT_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def test_join_prediction_mse_to_per_graph_rows(self):
        module = self.load_module()
        per_graph = [
            {
                "regime": "step2b_poly_d8",
                "subset_seed": "1",
                "graph_id": "G-1.json",
                "method_label": "2stage_val_mse",
                "normalized_gap": "0.12",
            }
        ]
        summary = [
            {
                "regime": "step2b_poly_d8",
                "subset_seed": "1",
                "graph_id": "G-1.json",
                "method_label": "2stage_val_mse",
                "mse_all_edges": "3.5",
            }
        ]

        joined = module.join_prediction_mse(per_graph, summary)

        self.assertEqual(len(joined), 1)
        self.assertEqual(joined[0]["prediction_mse"], 3.5)
        self.assertEqual(joined[0]["normalized_gap"], 0.12)

    def test_error_percentile_bin_places_highest_error_first(self):
        module = self.load_module()

        self.assertEqual(module.error_percentile_bin(rank=1, num_edges=100), 0)
        self.assertEqual(module.error_percentile_bin(rank=10, num_edges=100), 0)
        self.assertEqual(module.error_percentile_bin(rank=11, num_edges=100), 1)
        self.assertEqual(module.error_percentile_bin(rank=100, num_edges=100), 9)

    def test_error_percentile_rates_aggregate_selected_and_symdiff(self):
        module = self.load_module()
        edge_rows = [
            {
                "regime": "r",
                "subset_seed": "1",
                "graph_id": "G-1.json",
                "rank_err_2stage": "1",
                "rank_err_spoplus": "2",
                "in_2stage": "True",
                "in_spoplus": "False",
                "in_2stage_symdiff": "False",
                "in_spoplus_symdiff": "True",
            },
            {
                "regime": "r",
                "subset_seed": "1",
                "graph_id": "G-1.json",
                "rank_err_2stage": "2",
                "rank_err_spoplus": "1",
                "in_2stage": "False",
                "in_spoplus": "True",
                "in_2stage_symdiff": "True",
                "in_spoplus_symdiff": "False",
            },
        ]

        rows = module.error_percentile_rate_rows(edge_rows, bin_count=2)

        by_key = {(row["method_label"], row["bin_index"]): row for row in rows}
        self.assertEqual(by_key[("2stage_val_mse", 0)]["selected_rate"], 1.0)
        self.assertEqual(by_key[("2stage_val_mse", 1)]["symdiff_rate"], 1.0)
        self.assertEqual(by_key[("spoplus_val_spoplus_loss", 0)]["selected_rate"], 1.0)
        self.assertEqual(by_key[("spoplus_val_spoplus_loss", 1)]["symdiff_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
