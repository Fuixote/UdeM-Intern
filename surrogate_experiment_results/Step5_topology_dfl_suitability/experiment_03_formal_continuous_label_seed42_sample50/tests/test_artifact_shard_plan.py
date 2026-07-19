#!/usr/bin/env python3

import importlib.util
from pathlib import Path
import sys
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "plan_formal_artifact_shards.py"
SPEC = importlib.util.spec_from_file_location("plan_formal_artifact_shards", SCRIPT)
assert SPEC and SPEC.loader
planner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = planner
SPEC.loader.exec_module(planner)


class ArtifactShardPlanTests(unittest.TestCase):
    def test_balanced_shards_cover_each_topology_once(self):
        rows = [
            {
                "topology_id": f"G-{index}",
                "num_arcs": str(10 + index),
                "num_feasible_candidates": str(index * 3),
                "candidate_conflict_edges": str(index * 100),
            }
            for index in range(17)
        ]

        shards = planner.build_shard_plan(rows, 4)
        flattened = [topology_id for shard in shards for topology_id in shard["topology_ids"]]

        self.assertEqual(len(shards), 4)
        self.assertEqual(len(flattened), 17)
        self.assertEqual(set(flattened), {row["topology_id"] for row in rows})
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertLessEqual(
            max(shard["topology_count"] for shard in shards)
            - min(shard["topology_count"] for shard in shards),
            1,
        )


if __name__ == "__main__":
    unittest.main()
