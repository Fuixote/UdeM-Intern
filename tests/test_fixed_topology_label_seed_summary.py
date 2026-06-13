import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "surrogate_experiment_results"
    / "decision_analysis"
    / "scripts"
    / "summarize_fixed_topology_label_seed.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "summarize_fixed_topology_label_seed",
        SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(graph, seed, method, rank, gap, signature, case_flag=True):
    return {
        "base_graph_id": graph,
        "label_seed": str(seed),
        "topology_hash": f"topology-{graph}",
        "label_hash": f"label-{graph}-{seed}",
        "case_c_signature_for_label_seed": str(case_flag),
        "method_label": method,
        "solution_rank": str(rank),
        "normalized_gap_to_oracle": str(gap),
        "solution_arc_key_signature": signature,
        "oracle_arc_key_signature": f"oracle-{graph}-{seed}",
    }


class FixedTopologyLabelSeedSummaryTest(unittest.TestCase):
    def test_correction_persistence_requires_bad_2stage_top2_and_good_spoplus_rank1(self):
        mod = load_module()
        rows = [
            row("G-392.json", 0, "2stage_val_mse", 1, 0.30, "a"),
            row("G-392.json", 0, "2stage_val_mse", 2, 0.24, "b"),
            row("G-392.json", 0, "spoplus_val_spoplus_loss", 1, 0.01, "c"),
            row("G-392.json", 0, "spoplus_val_spoplus_loss", 2, 0.20, "a"),
            row("G-392.json", 1, "2stage_val_mse", 1, 0.30, "a"),
            row("G-392.json", 1, "2stage_val_mse", 2, 0.01, "b"),
            row("G-392.json", 1, "spoplus_val_spoplus_loss", 1, 0.01, "c"),
            row("G-392.json", 1, "spoplus_val_spoplus_loss", 2, 0.20, "a"),
        ]

        summary = mod.summarize(rows)[0]

        self.assertEqual(summary["label_seed_count"], 2)
        self.assertAlmostEqual(summary["correction_persistence_rate"], 0.5)
        self.assertAlmostEqual(summary["rank2_promotion_persistence_rate"], 0.0)

    def test_rank2_promotion_requires_2stage_rank2_matching_spoplus_rank1(self):
        mod = load_module()
        rows = [
            row("G-1560.json", 0, "2stage_val_mse", 1, 0.29, "bad-a"),
            row("G-1560.json", 0, "2stage_val_mse", 2, 0.01, "good"),
            row("G-1560.json", 0, "spoplus_val_spoplus_loss", 1, 0.01, "good"),
            row("G-1560.json", 0, "spoplus_val_spoplus_loss", 2, 0.25, "bad-a"),
            row("G-1560.json", 1, "2stage_val_mse", 1, 0.30, "bad-b"),
            row("G-1560.json", 1, "2stage_val_mse", 2, 0.015, "good-b"),
            row("G-1560.json", 1, "spoplus_val_spoplus_loss", 1, 0.012, "good-b"),
            row("G-1560.json", 1, "spoplus_val_spoplus_loss", 2, 0.24, "bad-b"),
        ]

        summary = mod.summarize(rows)[0]

        self.assertEqual(summary["label_seed_count"], 2)
        self.assertAlmostEqual(summary["case_c_preserved_rate"], 1.0)
        self.assertAlmostEqual(summary["rank2_promotion_persistence_rate"], 1.0)
        self.assertAlmostEqual(summary["correction_persistence_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
