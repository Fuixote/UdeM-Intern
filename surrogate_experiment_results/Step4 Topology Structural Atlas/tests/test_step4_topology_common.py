#!/usr/bin/env python3
"""Unit tests for Step4 topology-first audit helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


STRUCTURAL_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = STRUCTURAL_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import step4_topology_common as common  # noqa: E402


class Step4TopologyCommonTest(unittest.TestCase):
    def test_candidate_rows_preserve_candidate_geometry(self) -> None:
        template = {
            "topology_id": "G-test",
            "vertices": [
                {"id": "N0", "type": "NDD"},
                {"id": "1", "type": "Pair"},
                {"id": "2", "type": "Pair"},
            ],
            "arcs": [
                {"edge_idx": 0, "source": "N0", "target": "1"},
                {"edge_idx": 1, "source": "1", "target": "2"},
                {"edge_idx": 2, "source": "2", "target": "1"},
            ],
            "feasible_candidates": [
                {"type": "chain", "length": 1, "nodes": ["N0", "1"], "edges": [0], "signature": "chain:N0->1"},
                {"type": "chain", "length": 2, "nodes": ["N0", "1", "2"], "edges": [0, 1], "signature": "chain:N0->1->2"},
                {"type": "cycle", "length": 2, "nodes": ["1", "2"], "edges": [1, 2], "signature": "cycle:1->2->1"},
            ],
        }

        rows = common.candidate_rows_from_template(template)

        self.assertEqual([row["candidate_id"] for row in rows], ["G-test:c0000", "G-test:c0001", "G-test:c0002"])
        self.assertEqual(rows[1]["candidate_type"], "chain")
        self.assertEqual(rows[1]["edge_set"], "0|1")
        self.assertEqual(rows[2]["node_set"], "1|2")

    def test_candidate_conflicts_are_vertex_intersections(self) -> None:
        candidates = [
            {"topology_id": "G-test", "candidate_id": "a", "node_set": "N0|1"},
            {"topology_id": "G-test", "candidate_id": "b", "node_set": "N0|1|2"},
            {"topology_id": "G-test", "candidate_id": "c", "node_set": "3|4"},
        ]

        rows = common.conflict_rows_from_candidates(candidates)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["left_candidate_id"], "a")
        self.assertEqual(rows[0]["right_candidate_id"], "b")
        self.assertEqual(rows[0]["shared_vertices"], "1|N0")

    def test_selected_candidate_ids_use_maximal_edge_sets(self) -> None:
        candidates = [
            {"candidate_id": "chain_prefix", "edge_set": "0", "length": 1},
            {"candidate_id": "chain_full", "edge_set": "0|1", "length": 2},
            {"candidate_id": "cycle", "edge_set": "2|3", "length": 2},
            {"candidate_id": "unselected", "edge_set": "4", "length": 1},
        ]

        selected = common.selected_candidate_ids_for_edge_signature("0|1|2|3", candidates)

        self.assertEqual(selected, ["chain_full", "cycle"])


if __name__ == "__main__":
    unittest.main()
