#!/usr/bin/env python3
"""Unit tests for Step4 rank-reversal detail target planning."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


DETAIL_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = DETAIL_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import build_rank_reversal_detail_plan as planner  # noqa: E402


class RankReversalDetailPlanTest(unittest.TestCase):
    def test_targets_capture_contexts_where_methods_select_different_candidates(self) -> None:
        rows = [
            {"topology_id": "G-test", "data_seed": "101", "sample_size": "50", "test_sample_index": "1", "solution_source": "oracle", "selected_candidate_ids": "c2", "true_obj": "100", "gap_to_oracle": "0"},
            {"topology_id": "G-test", "data_seed": "101", "sample_size": "50", "test_sample_index": "1", "solution_source": "2stage", "selected_candidate_ids": "c1", "true_obj": "80", "gap_to_oracle": "20"},
            {"topology_id": "G-test", "data_seed": "101", "sample_size": "50", "test_sample_index": "1", "solution_source": "spoplus", "selected_candidate_ids": "c2", "true_obj": "100", "gap_to_oracle": "0"},
            {"topology_id": "G-test", "data_seed": "101", "sample_size": "50", "test_sample_index": "2", "solution_source": "oracle", "selected_candidate_ids": "c3", "true_obj": "90", "gap_to_oracle": "0"},
            {"topology_id": "G-test", "data_seed": "101", "sample_size": "50", "test_sample_index": "2", "solution_source": "2stage", "selected_candidate_ids": "c3", "true_obj": "90", "gap_to_oracle": "0"},
            {"topology_id": "G-test", "data_seed": "101", "sample_size": "50", "test_sample_index": "2", "solution_source": "spoplus", "selected_candidate_ids": "c3", "true_obj": "90", "gap_to_oracle": "0"},
        ]

        targets = planner.build_rank_reversal_targets(rows, targets_per_topology=5)

        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0]["test_sample_index"], 1)
        self.assertEqual(targets[0]["two_stage_candidate_ids"], "c1")
        self.assertEqual(targets[0]["spoplus_candidate_ids"], "c2")
        self.assertEqual(targets[0]["true_delta_spoplus_minus_2stage"], 20.0)
        self.assertEqual(targets[0]["case_direction"], "beneficial_reversal")


if __name__ == "__main__":
    unittest.main()
