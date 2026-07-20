from __future__ import annotations

from pathlib import Path
import sys
import unittest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_multiseed_targets as targets


def formal_row(topology_id: str, value: float) -> dict[str, str]:
    return {
        "topology_id": topology_id,
        "topology_hash": f"topology-{topology_id}",
        "feasible_set_hash": f"feasible-{topology_id}",
        "test_hash": f"test-{topology_id}",
        "normalized_improvement_pp": str(value),
    }


def repeat_row(topology_id: str, seed: int, value: float) -> dict[str, str]:
    return {
        "topology_id": topology_id,
        "train_seed": str(seed),
        "normalized_improvement_pp": str(value),
        "test_hash": f"test-{topology_id}",
    }


class MultiseedTargetTests(unittest.TestCase):
    def test_formal_target_requires_all_three_seeds(self) -> None:
        formal = [formal_row("G-0", 1.0), formal_row("G-1", 4.0)]
        repeat = [
            repeat_row("G-0", 42, 1.0),
            repeat_row("G-0", 43, 2.0),
            repeat_row("G-0", 44, 3.0),
        ]
        rows, missing, audit = targets.aggregate_targets(formal, repeat)
        by_id = {row["topology_id"]: row for row in rows}
        self.assertEqual(by_id["G-0"]["formal_label_mean_pp"], 2.0)
        self.assertAlmostEqual(by_id["G-0"]["label_uncertainty_std_pp"], (2.0 / 3.0) ** 0.5)
        self.assertEqual(by_id["G-0"]["uncertainty_ddof"], 0)
        self.assertEqual(by_id["G-1"]["formal_label_mean_pp"], "")
        self.assertEqual(by_id["G-1"]["missing_label_seeds"], "43;44")
        self.assertEqual([row["topology_id"] for row in missing], ["G-1"])
        self.assertEqual(audit["complete_three_seed_topology_count"], 1)
        self.assertEqual(audit["missing_seed_job_count"], 2)
        self.assertFalse(audit["formal_ready"])


if __name__ == "__main__":
    unittest.main()
