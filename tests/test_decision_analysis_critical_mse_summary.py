from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "decision_analysis"
    / "scripts"
    / "summarize_decision_critical_mse.py"
)


def per_graph_row(graph_id: str, method_label: str, normalized_gap: float) -> dict[str, str]:
    method = "2stage" if method_label == "2stage_val_mse" else "spoplus"
    return {
        "regime": "step2b_poly_d8",
        "case_type": "synthetic_seed",
        "subset_seed": "5",
        "graph_id": graph_id,
        "method_label": method_label,
        "method": method,
        "selection_metric": "validation_metric",
        "normalized_gap": str(normalized_gap),
    }


def edge_summary_row(
    graph_id: str,
    method_label: str,
    mse_all_edges: float,
    mse_edges_in_pred: float,
    mse_edges_in_symdiff: float | str,
    top10_symdiff_rate: float,
) -> dict[str, str]:
    method = "2stage" if method_label == "2stage_val_mse" else "spoplus"
    return {
        "regime": "step2b_poly_d8",
        "case_type": "synthetic_seed",
        "subset_seed": "5",
        "graph_id": graph_id,
        "method_label": method_label,
        "method": method,
        "selection_metric": "validation_metric",
        "mse_all_edges": str(mse_all_edges),
        "mse_edges_in_opt": "0.0",
        "mse_edges_in_pred": str(mse_edges_in_pred),
        "mse_edges_in_symdiff": str(mse_edges_in_symdiff),
        "mse_edges_not_selected": "0.0",
        "top10_error_edges_in_symdiff_rate": str(top10_symdiff_rate),
    }


class DecisionCriticalMseSummaryTests(unittest.TestCase):
    def load_module(self):
        self.assertTrue(SCRIPT_PATH.exists(), f"Missing script: {SCRIPT_PATH}")
        spec = importlib.util.spec_from_file_location(
            "summarize_decision_critical_mse", SCRIPT_PATH
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def test_pearson_and_spearman_correlations(self):
        module = self.load_module()

        self.assertAlmostEqual(module.pearson_corr([0, 1, 2], [0, 2, 4]), 1.0)
        self.assertAlmostEqual(module.pearson_corr([0, 1, 2], [4, 2, 0]), -1.0)
        self.assertAlmostEqual(module.spearman_corr([0, 1, 2], [10, 20, 30]), 1.0)
        self.assertAlmostEqual(module.spearman_corr([0, 1, 2], [30, 20, 10]), -1.0)
        self.assertTrue(math.isnan(module.pearson_corr([1, 1, 1], [0, 1, 2])))

    def test_summarize_correlations_joins_gap_and_edge_summary_rows(self):
        module = self.load_module()
        per_graph_rows = [
            per_graph_row("G-1.json", "2stage_val_mse", 0.0),
            per_graph_row("G-2.json", "2stage_val_mse", 1.0),
            per_graph_row("G-3.json", "2stage_val_mse", 2.0),
        ]
        edge_summary_rows = [
            edge_summary_row("G-1.json", "2stage_val_mse", 0.0, 2.0, "nan", 0.0),
            edge_summary_row("G-2.json", "2stage_val_mse", 1.0, 1.0, 10.0, 0.5),
            edge_summary_row("G-3.json", "2stage_val_mse", 2.0, 0.0, 20.0, 1.0),
        ]

        rows = module.summarize_correlations(per_graph_rows, edge_summary_rows)
        by_predictor = {row["predictor"]: row for row in rows}

        self.assertAlmostEqual(
            float(by_predictor["mse_all_edges"]["pearson_corr_with_normalized_gap"]),
            1.0,
        )
        self.assertAlmostEqual(
            float(by_predictor["mse_edges_in_pred"]["pearson_corr_with_normalized_gap"]),
            -1.0,
        )
        self.assertEqual(by_predictor["mse_edges_in_symdiff"]["n_pairs"], 2)
        self.assertAlmostEqual(
            float(
                by_predictor["top10_error_edges_in_symdiff_rate"][
                    "spearman_corr_with_normalized_gap"
                ]
            ),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
