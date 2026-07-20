from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import review_multiseed_completion_results as review


class CompletionReviewTests(unittest.TestCase):
    def test_parse_cap_hit_requires_topology_and_integer_seed(self) -> None:
        self.assertEqual(review.parse_cap_hit("G-122@44"), ("G-122", 44))
        with self.assertRaises(Exception):
            review.parse_cap_hit("G-122")
        with self.assertRaises(Exception):
            review.parse_cap_hit("G-122@seed44")

    def test_explicit_successful_max_epoch_cap_hit_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job_dir = Path(temporary)
            record_path = job_dir / "spoplus" / "metrics" / "early_stopping.json"
            record_path.parent.mkdir(parents=True)
            record_path.write_text(
                json.dumps(
                    {
                        "enabled": True,
                        "should_stop": False,
                        "metric": "validation_spoplus_loss",
                        "max_epochs": 10000,
                        "stopped_epoch": 10000,
                    }
                ),
                encoding="utf-8",
            )
            row = {
                "job_dir": str(job_dir),
                "status": "success",
                "two_stage_status": "success",
                "spoplus_status": "success",
                "evaluation_status": "success",
            }
            accepted, failure = review.validate_accepted_spoplus_cap_hit(
                row,
                topology_id="G-122",
                seed=44,
                accepted_cap_hits={("G-122", 44)},
            )
            self.assertTrue(accepted)
            self.assertIsNone(failure)

            row["evaluation_status"] = "failed"
            accepted, failure = review.validate_accepted_spoplus_cap_hit(
                row,
                topology_id="G-122",
                seed=44,
                accepted_cap_hits={("G-122", 44)},
            )
            self.assertFalse(accepted)
            self.assertEqual(failure, "accepted_spoplus_cap_hit_status_not_success")


if __name__ == "__main__":
    unittest.main()
