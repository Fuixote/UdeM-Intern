#!/usr/bin/env python3
"""Unit tests for Step4 rank-reversal summary tables."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


DETAIL_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = DETAIL_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import summarize_rank_reversal_detail as summary  # noqa: E402


class RankReversalSummaryTest(unittest.TestCase):
    def test_summaries_use_total_contexts_and_group_switch_patterns(self) -> None:
        decision_rows = [
            {"topology_id": "G-test", "sample_size": "50", "data_seed": "101", "test_sample_index": "1", "solution_source": "oracle"},
            {"topology_id": "G-test", "sample_size": "50", "data_seed": "101", "test_sample_index": "1", "solution_source": "2stage"},
            {"topology_id": "G-test", "sample_size": "50", "data_seed": "101", "test_sample_index": "1", "solution_source": "spoplus"},
            {"topology_id": "G-test", "sample_size": "50", "data_seed": "101", "test_sample_index": "2", "solution_source": "oracle"},
            {"topology_id": "G-test", "sample_size": "50", "data_seed": "101", "test_sample_index": "2", "solution_source": "2stage"},
            {"topology_id": "G-test", "sample_size": "50", "data_seed": "101", "test_sample_index": "2", "solution_source": "spoplus"},
            {"topology_id": "G-test", "sample_size": "50", "data_seed": "101", "test_sample_index": "3", "solution_source": "oracle"},
            {"topology_id": "G-test", "sample_size": "50", "data_seed": "101", "test_sample_index": "3", "solution_source": "2stage"},
            {"topology_id": "G-test", "sample_size": "50", "data_seed": "101", "test_sample_index": "3", "solution_source": "spoplus"},
        ]
        target_rows = [
            {
                "topology_id": "G-test",
                "sample_size": "50",
                "data_seed": "101",
                "test_sample_index": "1",
                "graph": "G-000001.json",
                "oracle_candidate_ids": "c2",
                "two_stage_candidate_ids": "c1",
                "spoplus_candidate_ids": "c2",
                "gap_2stage": "20",
                "gap_spoplus": "0",
                "true_delta_spoplus_minus_2stage": "20",
                "case_direction": "beneficial_reversal",
                "abs_true_delta": "20",
            },
            {
                "topology_id": "G-test",
                "sample_size": "50",
                "data_seed": "101",
                "test_sample_index": "3",
                "graph": "G-000003.json",
                "oracle_candidate_ids": "c4",
                "two_stage_candidate_ids": "c3",
                "spoplus_candidate_ids": "c4",
                "gap_2stage": "0",
                "gap_spoplus": "5",
                "true_delta_spoplus_minus_2stage": "-5",
                "case_direction": "harmful_reversal",
                "abs_true_delta": "5",
            },
        ]

        case_rows, switch_rows = summary.summarize_reversal_detail(decision_rows, target_rows)

        self.assertEqual(len(case_rows), 1)
        self.assertEqual(case_rows[0]["total_contexts"], 3)
        self.assertEqual(case_rows[0]["different_decision_contexts"], 2)
        self.assertAlmostEqual(case_rows[0]["different_decision_rate"], 2 / 3)
        self.assertEqual(case_rows[0]["beneficial_reversal_count"], 1)
        self.assertEqual(case_rows[0]["harmful_reversal_count"], 1)
        self.assertAlmostEqual(case_rows[0]["mean_delta_different"], 7.5)

        self.assertEqual(len(switch_rows), 2)
        first = switch_rows[0]
        self.assertEqual(first["two_stage_candidate_ids"], "c1")
        self.assertEqual(first["spoplus_candidate_ids"], "c2")
        self.assertEqual(first["count"], 1)
        self.assertAlmostEqual(first["rate_total"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
