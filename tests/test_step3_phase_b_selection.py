import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "surrogate_experiment_results"
    / "Step3"
    / "scripts"
    / "select_phase_b_topologies.py"
)


def load_selection_module():
    spec = importlib.util.spec_from_file_location("step3_select_phase_b_topologies", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(
    topology_id,
    candidates,
    cycles,
    entropy,
    distinct,
    proxy_diff,
    norm_gap,
    margin=2.0,
):
    return {
        "topology_id": topology_id,
        "num_exchange_candidates": str(candidates),
        "num_cycles_total": str(cycles),
        "num_3cycles": "1" if cycles else "0",
        "num_chains_total": str(max(1, candidates - cycles)),
        "candidate_conflict_density": "0.7",
        "mean_candidates_per_vertex": "2.0",
        "num_distinct_oracle_solutions": str(distinct),
        "oracle_solution_entropy": str(entropy),
        "dominant_oracle_solution_fraction": "0.8",
        "fraction_linear_proxy_differs_from_oracle": str(proxy_diff),
        "mean_linear_proxy_normalized_gap_to_oracle": str(norm_gap),
        "median_top1_top2_margin": str(margin),
        "mean_pairwise_oracle_jaccard": "0.7",
    }


class Step3PhaseBSelectionTests(unittest.TestCase):
    def test_classifies_complexity_and_landscape_regimes(self):
        module = load_selection_module()

        self.assertEqual(module.complexity_bin(row("G-a", 8, 0, 0, 1, 0, 0)), "sparse_simple")
        self.assertEqual(module.complexity_bin(row("G-b", 20, 1, 0, 1, 0, 0)), "low_medium")
        self.assertEqual(module.complexity_bin(row("G-c", 40, 1, 0, 1, 0, 0)), "medium_rich")
        self.assertEqual(module.complexity_bin(row("G-d", 80, 1, 0, 1, 0, 0)), "rich")
        self.assertEqual(module.complexity_bin(row("G-e", 81, 1, 0, 1, 0, 0)), "extreme")

        self.assertEqual(
            module.landscape_regime(row("G-easy", 6, 0, 0.0, 1, 0.0, 0.0)),
            "easy_control",
        )
        self.assertEqual(
            module.landscape_regime(row("G-hard", 25, 1, 0.6, 3, 0.7, 0.1)),
            "proxy_hard",
        )
        self.assertEqual(
            module.landscape_regime(row("G-var", 25, 1, 1.1, 6, 0.2, 0.02)),
            "high_variance",
        )

    def test_selects_unique_topologies_by_complexity_quotas(self):
        module = load_selection_module()
        rows = []
        bins = [
            ("sparse", 6),
            ("low", 14),
            ("medium", 30),
            ("rich", 60),
            ("extreme", 100),
        ]
        for bin_name, candidates in bins:
            for idx in range(6):
                rows.append(
                    row(
                        f"G-{bin_name}-{idx}",
                        candidates,
                        cycles=idx % 2,
                        entropy=0.0 if idx == 0 else 0.4 + idx / 10,
                        distinct=1 if idx == 0 else 2 + idx,
                        proxy_diff=0.0 if idx == 0 else 0.15 * idx,
                        norm_gap=0.0 if idx == 0 else 0.02 * idx,
                    )
                )

        selected = module.select_phase_b_topologies(
            rows,
            quotas={
                "sparse_simple": 2,
                "low_medium": 2,
                "medium_rich": 2,
                "rich": 2,
                "extreme": 2,
            },
        )

        self.assertEqual(len(selected), 10)
        self.assertEqual(len({item["topology_id"] for item in selected}), 10)
        counts = {}
        for item in selected:
            counts[item["complexity_bin"]] = counts.get(item["complexity_bin"], 0) + 1
            self.assertIn("selection_rank", item)
            self.assertIn("screening_score", item)
            self.assertIn("selection_reason", item)
        self.assertEqual(
            counts,
            {
                "sparse_simple": 2,
                "low_medium": 2,
                "medium_rich": 2,
                "rich": 2,
                "extreme": 2,
            },
        )

    def test_write_outputs_creates_phase_b_artifacts(self):
        module = load_selection_module()
        selected = [
            module.derive_selection_fields(row("G-1", 6, 0, 0.0, 1, 0.0, 0.0)),
            module.derive_selection_fields(row("G-2", 30, 1, 0.9, 4, 0.6, 0.08)),
        ]
        for rank, item in enumerate(selected, start=1):
            item["selection_rank"] = rank
            item["selection_reason"] = "test"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            module.write_selection_outputs(output_dir, selected)

            csv_path = output_dir / "phase_b_topologies.csv"
            txt_path = output_dir / "phase_b_topology_ids.txt"
            summary_path = output_dir / "phase_b_selection_summary.json"
            self.assertTrue(csv_path.exists())
            self.assertTrue(txt_path.exists())
            self.assertTrue(summary_path.exists())

            with csv_path.open(newline="") as handle:
                csv_rows = list(csv.DictReader(handle))
            summary = json.loads(summary_path.read_text())
            self.assertEqual([row["topology_id"] for row in csv_rows], ["G-1", "G-2"])
            self.assertEqual(txt_path.read_text().splitlines(), ["G-1", "G-2"])
            self.assertEqual(summary["num_selected"], 2)


if __name__ == "__main__":
    unittest.main()
