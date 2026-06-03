from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "decision_analysis"
    / "scripts"
    / "make_case_study_tables.py"
)


def per_graph_row(
    graph_id: str,
    method_label: str,
    normalized_gap: float,
    edge_jaccard: float,
    same_solution: bool,
    subset_seed: int = 7,
) -> dict[str, str]:
    method = "2stage" if method_label == "2stage_val_mse" else "spoplus"
    return {
        "regime": "step2b_poly_d8",
        "case_type": "synthetic_seed",
        "subset_seed": str(subset_seed),
        "graph_id": graph_id,
        "method_label": method_label,
        "method": method,
        "selection_metric": "validation_mse_loss",
        "optimal_obj": "100.0",
        "achieved_obj": str(100.0 - 100.0 * normalized_gap),
        "decision_gap": str(100.0 * normalized_gap),
        "normalized_gap": str(normalized_gap),
        "same_solution_as_opt": str(same_solution),
        "edge_jaccard_with_opt": str(edge_jaccard),
    }


def edge_summary_row(
    graph_id: str,
    method_label: str,
    mse_all_edges: float,
    top_opt: float,
    top_pred: float,
    top_symdiff: float,
    subset_seed: int = 7,
) -> dict[str, str]:
    method = "2stage" if method_label == "2stage_val_mse" else "spoplus"
    return {
        "regime": "step2b_poly_d8",
        "case_type": "synthetic_seed",
        "subset_seed": str(subset_seed),
        "graph_id": graph_id,
        "method_label": method_label,
        "method": method,
        "selection_metric": "validation_mse_loss",
        "num_edges": "5",
        "num_top_error_edges": "5",
        "mse_all_edges": str(mse_all_edges),
        "mse_edges_in_opt": "1.0",
        "mse_edges_in_pred": "1.0",
        "mse_edges_in_symdiff": "1.0",
        "mse_edges_not_selected": "1.0",
        "top10_error_edges_in_opt_rate": str(top_opt),
        "top10_error_edges_in_pred_rate": str(top_pred),
        "top10_error_edges_in_symdiff_rate": str(top_symdiff),
    }


class DecisionAnalysisCaseStudyTests(unittest.TestCase):
    def load_module(self):
        self.assertTrue(SCRIPT_PATH.exists(), f"Missing script: {SCRIPT_PATH}")
        spec = importlib.util.spec_from_file_location("make_case_study_tables", SCRIPT_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def test_select_case_studies_finds_a_b_c_examples(self):
        module = self.load_module()
        per_graph_rows = [
            per_graph_row("A.json", "2stage_val_mse", 0.004, 1.0, True),
            per_graph_row("A.json", "spoplus_val_spoplus_loss", 0.005, 1.0, True),
            per_graph_row("B.json", "2stage_val_mse", 0.006, 0.52, False),
            per_graph_row("B.json", "spoplus_val_spoplus_loss", 0.005, 0.55, False),
            per_graph_row("C.json", "2stage_val_mse", 0.25, 0.20, False),
            per_graph_row("C.json", "spoplus_val_spoplus_loss", 0.015, 0.90, False),
            per_graph_row("D.json", "2stage_val_mse", 0.20, 0.25, False),
            per_graph_row("D.json", "spoplus_val_spoplus_loss", 0.18, 0.30, False),
        ]
        edge_summary_rows = [
            edge_summary_row("A.json", "2stage_val_mse", 100.0, 0.0, 0.0, 0.0),
            edge_summary_row("A.json", "spoplus_val_spoplus_loss", 20.0, 0.0, 0.0, 0.0),
            edge_summary_row("B.json", "2stage_val_mse", 10.0, 0.2, 0.2, 0.2),
            edge_summary_row("B.json", "spoplus_val_spoplus_loss", 11.0, 0.2, 0.2, 0.2),
            edge_summary_row("C.json", "2stage_val_mse", 30.0, 0.2, 0.2, 0.8),
            edge_summary_row("C.json", "spoplus_val_spoplus_loss", 12.0, 0.1, 0.1, 0.0),
            edge_summary_row("D.json", "2stage_val_mse", 25.0, 0.2, 0.2, 0.3),
            edge_summary_row("D.json", "spoplus_val_spoplus_loss", 12.0, 0.2, 0.2, 0.2),
        ]

        selected = module.select_case_studies(
            per_graph_rows,
            edge_summary_rows,
            per_case_count=1,
        )

        self.assertEqual(
            [(row["case_label"], row["graph_id"]) for row in selected],
            [
                ("case_a_bad_prediction_irrelevant", "A.json"),
                ("case_b_different_solution_near_optimal", "B.json"),
                ("case_c_spoplus_fixes_2stage", "C.json"),
            ],
        )

    def test_select_case_studies_prefers_distinct_graphs_within_each_case_label(self):
        module = self.load_module()
        per_graph_rows = [
            per_graph_row("A.json", "2stage_val_mse", 0.001, 1.0, True, subset_seed=1),
            per_graph_row(
                "A.json",
                "spoplus_val_spoplus_loss",
                0.001,
                1.0,
                True,
                subset_seed=1,
            ),
            per_graph_row("A.json", "2stage_val_mse", 0.002, 1.0, True, subset_seed=2),
            per_graph_row(
                "A.json",
                "spoplus_val_spoplus_loss",
                0.002,
                1.0,
                True,
                subset_seed=2,
            ),
            per_graph_row("E.json", "2stage_val_mse", 0.003, 1.0, True, subset_seed=3),
            per_graph_row(
                "E.json",
                "spoplus_val_spoplus_loss",
                0.003,
                1.0,
                True,
                subset_seed=3,
            ),
            per_graph_row("B.json", "2stage_val_mse", 0.004, 0.52, False, subset_seed=4),
            per_graph_row(
                "B.json",
                "spoplus_val_spoplus_loss",
                0.004,
                0.52,
                False,
                subset_seed=4,
            ),
            per_graph_row("F.json", "2stage_val_mse", 0.005, 0.56, False, subset_seed=5),
            per_graph_row(
                "F.json",
                "spoplus_val_spoplus_loss",
                0.005,
                0.56,
                False,
                subset_seed=5,
            ),
            per_graph_row("C.json", "2stage_val_mse", 0.25, 0.20, False, subset_seed=6),
            per_graph_row(
                "C.json",
                "spoplus_val_spoplus_loss",
                0.01,
                0.90,
                False,
                subset_seed=6,
            ),
            per_graph_row("D.json", "2stage_val_mse", 0.20, 0.25, False, subset_seed=7),
            per_graph_row(
                "D.json",
                "spoplus_val_spoplus_loss",
                0.02,
                0.85,
                False,
                subset_seed=7,
            ),
        ]
        edge_summary_rows = [
            edge_summary_row("A.json", "2stage_val_mse", 100.0, 0.0, 0.0, 0.0, 1),
            edge_summary_row("A.json", "spoplus_val_spoplus_loss", 10.0, 0.0, 0.0, 0.0, 1),
            edge_summary_row("A.json", "2stage_val_mse", 99.0, 0.0, 0.0, 0.0, 2),
            edge_summary_row("A.json", "spoplus_val_spoplus_loss", 10.0, 0.0, 0.0, 0.0, 2),
            edge_summary_row("E.json", "2stage_val_mse", 90.0, 0.0, 0.0, 0.0, 3),
            edge_summary_row("E.json", "spoplus_val_spoplus_loss", 10.0, 0.0, 0.0, 0.0, 3),
            edge_summary_row("B.json", "2stage_val_mse", 10.0, 0.1, 0.1, 0.1, 4),
            edge_summary_row("B.json", "spoplus_val_spoplus_loss", 10.0, 0.1, 0.1, 0.1, 4),
            edge_summary_row("F.json", "2stage_val_mse", 11.0, 0.1, 0.1, 0.1, 5),
            edge_summary_row("F.json", "spoplus_val_spoplus_loss", 11.0, 0.1, 0.1, 0.1, 5),
            edge_summary_row("C.json", "2stage_val_mse", 30.0, 0.1, 0.1, 0.9, 6),
            edge_summary_row("C.json", "spoplus_val_spoplus_loss", 10.0, 0.1, 0.1, 0.0, 6),
            edge_summary_row("D.json", "2stage_val_mse", 25.0, 0.1, 0.1, 0.8, 7),
            edge_summary_row("D.json", "spoplus_val_spoplus_loss", 10.0, 0.1, 0.1, 0.1, 7),
        ]

        selected = module.select_case_studies(
            per_graph_rows,
            edge_summary_rows,
            per_case_count=2,
        )

        case_a_graphs = [
            row["graph_id"]
            for row in selected
            if row["case_label"] == "case_a_bad_prediction_irrelevant"
        ]
        self.assertEqual(case_a_graphs, ["A.json", "E.json"])

    def test_write_case_study_outputs_uses_required_edge_columns(self):
        module = self.load_module()
        selected = [
            {
                "case_id": "case_c_spoplus_fixes_2stage_001",
                "case_label": "case_c_spoplus_fixes_2stage",
                "regime": "step2b_poly_d8",
                "subset_seed": "7",
                "graph_id": "C.json",
            }
        ]
        edge_rows = [
            {
                "regime": "step2b_poly_d8",
                "subset_seed": "7",
                "graph_id": "C.json",
                "edge_id": "1",
                "src": "0",
                "dst": "1",
                "w_true": "3.0",
                "w_hat_2stage": "10.0",
                "w_hat_spoplus": "3.2",
                "abs_err_2stage": "7.0",
                "abs_err_spoplus": "0.2",
                "rank_err_2stage": "1",
                "rank_err_spoplus": "2",
                "in_opt": "True",
                "in_2stage": "False",
                "in_spoplus": "True",
                "in_2stage_symdiff": "True",
                "in_spoplus_symdiff": "False",
                "utility": "0.8",
                "recipient_cPRA": "0.9",
            },
            {
                "regime": "step2b_poly_d8",
                "subset_seed": "7",
                "graph_id": "C.json",
                "edge_id": "2",
                "src": "2",
                "dst": "3",
                "w_true": "1.0",
                "w_hat_2stage": "1.1",
                "w_hat_spoplus": "1.1",
                "abs_err_2stage": "0.1",
                "abs_err_spoplus": "0.1",
                "rank_err_2stage": "2",
                "rank_err_spoplus": "1",
                "in_opt": "False",
                "in_2stage": "False",
                "in_spoplus": "False",
                "in_2stage_symdiff": "False",
                "in_spoplus_symdiff": "False",
                "utility": "0.1",
                "recipient_cPRA": "0.2",
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            index_rows = module.write_case_study_outputs(selected, edge_rows, output_dir)
            table_path = output_dir / index_rows[0]["case_table_file"]

            with table_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(list(rows[0].keys()), module.CASE_EDGE_FIELDS)
        self.assertEqual(rows[0]["edge_id"], "1")
        self.assertEqual(rows[0]["src_dst"], "0 -> 1")
        self.assertEqual(rows[0]["in_2stage_symdiff"], "True")


if __name__ == "__main__":
    unittest.main()
