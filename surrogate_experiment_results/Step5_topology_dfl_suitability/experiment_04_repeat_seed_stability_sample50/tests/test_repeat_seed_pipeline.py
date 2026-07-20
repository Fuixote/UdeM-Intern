from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_repeat_seed_artifacts as artifacts
import review_repeat_seed_results as review
import select_repeat_seed_topologies as selection


def row(topology_id: str, value: float, gap: float, complexity: int) -> dict[str, str]:
    return {
        "topology_id": topology_id,
        "normalized_improvement_pp": str(value),
        "test_normalized_gap_2stage": str(gap + value / 100.0),
        "test_normalized_gap_spoplus": str(gap),
        "num_feasible_candidates": str(complexity),
        "candidate_conflict_edges": str(complexity * 2),
        "num_arcs": str(complexity + 1),
        "test_hash": f"hash-{topology_id}",
    }


class RepeatSeedPipelineTests(unittest.TestCase):
    def test_selection_has_locked_category_counts_and_unique_topologies(self) -> None:
        rows = []
        rows += [row(f"cz-{i}", 0.0, 0.0, i) for i in range(10)]
        rows += [row(f"pz-{i}", 0.0, 0.2, i) for i in range(10)]
        rows += [row(f"sm-{i}", 0.01 + i * 0.001, 0.2, i) for i in range(10)]
        rows += [row(f"pos-{i}", 0.2 + i, 0.2, i) for i in range(15)]
        rows += [row(f"neg-{i}", -0.2 - i, 0.2, i) for i in range(15)]
        selected = selection.select_panel(rows)
        self.assertEqual(len(selected), 60)
        self.assertEqual(len({item["topology_id"] for item in selected}), 60)
        observed = {
            category: sum(item["selection_category"] == category for item in selected)
            for category in selection.CATEGORY_COUNTS
        }
        self.assertEqual(observed, selection.CATEGORY_COUNTS)

    def test_reference_test_manifest_rejects_train_seeded_test(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test, manifest = artifacts.common.reference_test_paths("G-X", formal_output_root=root)
            test.parent.mkdir(parents=True)
            test.write_bytes(b"placeholder")
            manifest.write_text(json.dumps({
                "topology_id": "G-X",
                "test_size": 1000,
                "test_hash": "abc",
                "test_samples": [{"train_seed": 43}],
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "varies by train seed"):
                artifacts.reference_test_manifest("G-X", root)

    def test_rank_metrics_handle_ties(self) -> None:
        left = [0.0, 0.0, 1.0, 2.0]
        right = [0.0, 0.0, 2.0, 4.0]
        self.assertAlmostEqual(review.spearman(left, right) or 0.0, 1.0)
        self.assertEqual(review.sign_class(-0.2), "negative")
        self.assertEqual(review.sign_class(0.0), "zero")


if __name__ == "__main__":
    unittest.main()
