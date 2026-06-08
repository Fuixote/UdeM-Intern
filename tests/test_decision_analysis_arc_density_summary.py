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
    / "summarize_arc_density_sensitivity.py"
)


def solution_row(
    *,
    variant: str,
    rank: int,
    solution_sig: str,
    oracle_sig: str,
    true_obj: float,
    oracle_obj: float,
    gap: float,
    norm_gap: float,
    method_label: str = "2stage_val_mse",
    added: str = "",
    removed: str = "",
    variant_num_arcs: int = 4,
    arc_delta: int = 0,
    perturb_seed: int = 42,
) -> dict[str, object]:
    return {
        "case_id": "case_test_001",
        "case_label": "case_test",
        "base_graph_id": "G-test.json",
        "variant_id": f"G-test__{variant}__seed{perturb_seed}",
        "variant_graph_path": f"graphs/G-test__{variant}__seed{perturb_seed}.json",
        "density_variant": variant,
        "arc_delta_type": "none" if variant == "original" else "perturb",
        "original_num_arcs": 4,
        "variant_num_arcs": variant_num_arcs,
        "arc_delta": arc_delta,
        "added_arc_count": 0 if not added else len(added.split("|")),
        "removed_arc_count": 0 if not removed else len(removed.split("|")),
        "added_arc_keys": added,
        "removed_arc_keys": removed,
        "perturb_seed": perturb_seed,
        "regime": "step2b_poly_d8",
        "max_cycle": 3,
        "max_chain": 4,
        "case_type": "case_test",
        "subset_seed": 7,
        "graph_id": f"G-test__{variant}__seed{perturb_seed}.json",
        "method_label": method_label,
        "solution_rank": rank,
        "true_obj": true_obj,
        "oracle_obj": oracle_obj,
        "gap_to_oracle": gap,
        "normalized_gap_to_oracle": norm_gap,
        "predicted_margin_from_best": 0.0 if rank == 1 else 1.5,
        "num_edges": variant_num_arcs,
        "edge_count": len(solution_sig.split("|")) if solution_sig else 0,
        "num_cycle_candidates": 2,
        "num_chain_candidates": 3,
        "same_solution_as_oracle": str(solution_sig == oracle_sig),
        "edge_jaccard_with_oracle": 1.0 if solution_sig == oracle_sig else 0.5,
        "solution_edge_signature": "0|1",
        "solution_selected_edge_count": len(solution_sig.split("|")) if solution_sig else 0,
        "solution_arc_key_signature": solution_sig,
        "oracle_arc_key_signature": oracle_sig,
        "rank1_arc_key_signature": "0->1|1->2" if variant == "original" else "0->1|2->3",
        "original_oracle_arc_key_signature": "0->1|1->2",
    }


def fixture_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rows.extend(
        [
            solution_row(
                variant="original",
                rank=1,
                solution_sig="0->1|1->2",
                oracle_sig="0->1|1->2",
                true_obj=100.0,
                oracle_obj=100.0,
                gap=0.0,
                norm_gap=0.0,
            ),
            solution_row(
                variant="original",
                rank=2,
                solution_sig="0->1",
                oracle_sig="0->1|1->2",
                true_obj=95.0,
                oracle_obj=100.0,
                gap=5.0,
                norm_gap=0.05,
            ),
            solution_row(
                variant="add25arcs",
                rank=1,
                solution_sig="0->1|2->3",
                oracle_sig="0->1|2->3",
                true_obj=110.0,
                oracle_obj=110.0,
                gap=0.0,
                norm_gap=0.0,
                added="2->3|3->0",
                variant_num_arcs=6,
                arc_delta=2,
            ),
            solution_row(
                variant="add25arcs",
                rank=2,
                solution_sig="2->3",
                oracle_sig="0->1|2->3",
                true_obj=108.0,
                oracle_obj=110.0,
                gap=2.0,
                norm_gap=0.0181818181818,
                added="2->3|3->0",
                variant_num_arcs=6,
                arc_delta=2,
            ),
            solution_row(
                variant="remove25arcs",
                rank=1,
                solution_sig="0->1",
                oracle_sig="0->1",
                true_obj=90.0,
                oracle_obj=90.0,
                gap=0.0,
                norm_gap=0.0,
                removed="1->2",
                variant_num_arcs=3,
                arc_delta=-1,
            ),
            solution_row(
                variant="remove25arcs",
                rank=2,
                solution_sig="2->3",
                oracle_sig="0->1",
                true_obj=80.0,
                oracle_obj=90.0,
                gap=10.0,
                norm_gap=0.111111111111,
                removed="1->2",
                variant_num_arcs=3,
                arc_delta=-1,
            ),
        ]
    )
    return rows


def fixture_rows_for_seed(perturb_seed: int) -> list[dict[str, object]]:
    rows = []
    for row in fixture_rows():
        copied = dict(row)
        copied["perturb_seed"] = perturb_seed
        copied["variant_id"] = f"G-test__{copied['density_variant']}__seed{perturb_seed}"
        copied["variant_graph_path"] = (
            f"graphs/G-test__{copied['density_variant']}__seed{perturb_seed}.json"
        )
        copied["graph_id"] = f"G-test__{copied['density_variant']}__seed{perturb_seed}.json"
        rows.append(copied)
    return rows


class ArcDensitySummaryTests(unittest.TestCase):
    def load_module(self):
        self.assertTrue(SCRIPT_PATH.exists(), f"Missing script: {SCRIPT_PATH}")
        spec = importlib.util.spec_from_file_location("summarize_arc_density_sensitivity", SCRIPT_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def test_case_summary_uses_arc_keys_for_changes_and_added_removed_overlaps(self):
        module = self.load_module()

        rows = module.build_case_summary(fixture_rows())
        by_variant = {row["density_variant"]: row for row in rows}

        added = by_variant["add25arcs"]
        self.assertEqual(added["oracle_obj_delta_vs_original"], 10.0)
        self.assertEqual(added["oracle_solution_changed_vs_original"], True)
        self.assertEqual(added["rank1_solution_changed_vs_original"], True)
        self.assertEqual(added["rank2_solution_changed_vs_original"], True)
        self.assertEqual(added["num_added_arcs_in_oracle"], 1)
        self.assertEqual(added["num_added_arcs_in_rank1"], 1)
        self.assertEqual(added["num_added_arcs_in_rank2"], 1)

        removed = by_variant["remove25arcs"]
        self.assertEqual(removed["oracle_obj_delta_vs_original"], -10.0)
        self.assertEqual(removed["num_removed_arcs_from_original_oracle"], 1)
        self.assertEqual(removed["num_removed_arcs_from_original_rank1"], 1)
        self.assertEqual(removed["num_removed_arcs_from_original_rank2"], 0)
        self.assertEqual(removed["rank2_minus_rank1_normalized_gap"], 0.111111111111)

    def test_case_summary_keeps_perturb_seed_replicates_separate(self):
        module = self.load_module()

        rows = fixture_rows_for_seed(0) + fixture_rows_for_seed(1)
        case_rows = module.build_case_summary(rows)
        add_rows = [
            row
            for row in case_rows
            if row["density_variant"] == "add25arcs"
            and row["method_label"] == "2stage_val_mse"
        ]

        self.assertEqual(len(add_rows), 2)
        self.assertEqual(sorted(int(row["perturb_seed"]) for row in add_rows), [0, 1])

    def test_second_best_summary_groups_by_variant_method_and_rank(self):
        module = self.load_module()

        rows = module.build_second_best_summary(fixture_rows())
        by_key = {
            (row["density_variant"], row["method_label"], row["solution_rank"]): row
            for row in rows
        }

        rank2_added = by_key[("add25arcs", "2stage_val_mse", 2)]
        self.assertEqual(rank2_added["row_count"], 1)
        self.assertAlmostEqual(rank2_added["mean_normalized_gap_to_oracle"], 0.0181818181818)
        self.assertEqual(rank2_added["near_5pct_rate"], 1.0)
        self.assertEqual(rank2_added["solution_changed_vs_original_rate"], 1.0)
        self.assertEqual(rank2_added["mean_num_added_arcs_in_solution"], 1.0)

    def test_delta_and_oracle_summaries_compare_variants_to_original(self):
        module = self.load_module()

        delta_rows = module.build_delta_vs_original(fixture_rows())
        delta_by_variant = {row["density_variant"]: row for row in delta_rows}
        self.assertEqual(delta_by_variant["add25arcs"]["delta_rank2_normalized_gap"], -0.0318181818182)
        self.assertEqual(delta_by_variant["remove25arcs"]["delta_rank2_normalized_gap"], 0.061111111111)

        oracle_rows = module.build_oracle_change_summary(fixture_rows())
        oracle_by_variant = {row["density_variant"]: row for row in oracle_rows}
        self.assertEqual(oracle_by_variant["add25arcs"]["graph_count"], 1)
        self.assertEqual(oracle_by_variant["add25arcs"]["mean_oracle_obj_delta"], 10.0)
        self.assertEqual(oracle_by_variant["add25arcs"]["fraction_oracle_solution_changed"], 1.0)
        self.assertEqual(oracle_by_variant["add25arcs"]["fraction_rank2_solution_changed"], 1.0)

    def test_main_writes_four_expected_outputs_from_formal_input_only(self):
        module = self.load_module()
        fieldnames = list(fixture_rows()[0])

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "arc_density_second_best_gap.csv"
            with input_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(fixture_rows())

            smoke_path = tmp / "arc_density_second_best_gap_smoke_G696_original_2stage.csv"
            with smoke_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                extra_row = solution_row(
                    variant="original",
                    rank=1,
                    solution_sig="9->9",
                    oracle_sig="9->9",
                    true_obj=999.0,
                    oracle_obj=999.0,
                    gap=0.0,
                    norm_gap=0.0,
                )
                writer.writerow(extra_row)

            module.main(
                [
                    "--input",
                    str(input_path),
                    "--summary-output",
                    str(tmp / "arc_density_second_best_summary.csv"),
                    "--case-summary-output",
                    str(tmp / "arc_density_case_summary.csv"),
                    "--delta-output",
                    str(tmp / "arc_density_delta_vs_original.csv"),
                    "--oracle-change-summary-output",
                    str(tmp / "arc_density_oracle_change_summary.csv"),
                ]
            )

            for name in [
                "arc_density_second_best_summary.csv",
                "arc_density_case_summary.csv",
                "arc_density_delta_vs_original.csv",
                "arc_density_oracle_change_summary.csv",
            ]:
                self.assertTrue((tmp / name).exists(), name)

            with (tmp / "arc_density_second_best_summary.csv").open(newline="", encoding="utf-8") as handle:
                loaded = list(csv.DictReader(handle))
            self.assertEqual(sum(int(row["row_count"]) for row in loaded), len(fixture_rows()))


if __name__ == "__main__":
    unittest.main()
