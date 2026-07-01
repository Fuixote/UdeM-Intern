#!/usr/bin/env python3
"""Unit tests for Step4 decision overlay summaries."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


OVERLAY_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = OVERLAY_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import compute_decision_overlay as overlay  # noqa: E402


class DecisionOverlayTest(unittest.TestCase):
    def test_parse_args_accepts_gurobi_thread_limit(self) -> None:
        args = overlay.parse_args(["--gurobi-threads", "1"])

        self.assertEqual(args.gurobi_threads, 1)

    def test_candidate_summary_rates_are_grouped_by_topology_and_sample_size(self) -> None:
        candidate_rows = [
            {"topology_id": "G-test", "candidate_id": "c1", "candidate_type": "chain", "length": "1"},
            {"topology_id": "G-test", "candidate_id": "c2", "candidate_type": "cycle", "length": "2"},
        ]
        decision_rows = [
            {"topology_id": "G-test", "sample_size": "50", "solution_source": "oracle", "selected_candidate_ids": "c1"},
            {"topology_id": "G-test", "sample_size": "50", "solution_source": "oracle", "selected_candidate_ids": "c1|c2"},
            {"topology_id": "G-test", "sample_size": "50", "solution_source": "2stage", "selected_candidate_ids": "c1"},
            {"topology_id": "G-test", "sample_size": "50", "solution_source": "2stage", "selected_candidate_ids": "c1"},
            {"topology_id": "G-test", "sample_size": "50", "solution_source": "spoplus", "selected_candidate_ids": "c2"},
            {"topology_id": "G-test", "sample_size": "50", "solution_source": "spoplus", "selected_candidate_ids": "c1|c2"},
        ]

        summary = overlay.summarize_candidate_overlay(decision_rows, candidate_rows)
        by_candidate = {row["candidate_id"]: row for row in summary}

        self.assertEqual(by_candidate["c1"]["oracle_selection_rate"], 1.0)
        self.assertEqual(by_candidate["c1"]["two_stage_selection_rate"], 1.0)
        self.assertEqual(by_candidate["c1"]["spoplus_selection_rate"], 0.5)
        self.assertEqual(by_candidate["c1"]["spoplus_minus_2stage_rate"], -0.5)
        self.assertEqual(by_candidate["c2"]["oracle_selection_rate"], 0.5)
        self.assertEqual(by_candidate["c2"]["two_stage_selection_rate"], 0.0)
        self.assertEqual(by_candidate["c2"]["spoplus_selection_rate"], 1.0)
        self.assertEqual(by_candidate["c2"]["spoplus_minus_2stage_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
