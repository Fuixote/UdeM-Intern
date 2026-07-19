#!/usr/bin/env python3

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "review_sample_size_sensitivity.py"
SPEC = importlib.util.spec_from_file_location("review_sample_size_sensitivity", SCRIPT)
assert SPEC and SPEC.loader
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


def materialize_job(root: Path, sample_size: int, max_epochs: int, improvement: float) -> None:
    job = review.job_dir(root, "G-3", sample_size)
    metrics = job / "evaluation" / "metrics"
    metrics.mkdir(parents=True)
    (job / "job_status.json").write_text(
        json.dumps(
            {
                "status": "success",
                "2stage status": "success",
                "SPO+ status": "success",
                "evaluation status": "success",
            }
        ),
        encoding="utf-8",
    )
    (job / "paired_job_manifest.json").write_text(
        json.dumps(
            {
                "topology_id": "G-3",
                "regime": review.REGIME,
                "protocol": "screen",
                "train_seed": 42,
                "sample_size": sample_size,
                "training_size": sample_size * 4 // 5,
                "validation_size": sample_size // 5,
                "theta_seed": 42,
                "gurobi_seed": 42,
                "max_epochs": max_epochs,
                "metric_stride": 1,
                "early_stop_patience": 20,
                "early_stop_min_delta": 0.0001,
                "test_hash": "same-test-hash",
                "train_prefix_hash": f"train-{sample_size}",
                "validation_hash": f"validation-{sample_size}",
            }
        ),
        encoding="utf-8",
    )
    fit_manifest = review.fit_manifest_path(root, "G-3")
    fit_manifest.parent.mkdir(parents=True, exist_ok=True)
    fit_manifest.write_text(
        json.dumps({"samples": [{"sample_index": index} for index in range(sample_size)]}),
        encoding="utf-8",
    )
    (metrics / "test_summary.json").write_text(
        json.dumps(
            [
                {
                    "method": "2stage",
                    "selected_epoch": 100,
                    "test_mean_normalized_gap": 0.2,
                    "test_mean_decision_gap": 2.0,
                },
                {
                    "method": "spoplus",
                    "selected_epoch": max_epochs,
                    "test_mean_normalized_gap": 0.2 - improvement,
                    "test_mean_decision_gap": 2.0 - 10.0 * improvement,
                    "fraction_improved_over_2stage": 0.5,
                },
            ]
        ),
        encoding="utf-8",
    )
    with (metrics / "test_per_graph.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "graph", "achieved_obj", "gap"],
        )
        writer.writeheader()
        writer.writerow({"method": "2stage", "graph": "G-0.json", "achieved_obj": 8, "gap": 2})
        writer.writerow(
            {
                "method": "spoplus",
                "graph": "G-0.json",
                "achieved_obj": 8 + 10 * improvement,
                "gap": 2 - 10 * improvement,
            }
        )


class SampleSizeSensitivityReviewTests(unittest.TestCase):
    def test_review_computes_pp_and_checks_test_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            roots = {size: base / str(size) for size in review.EXPECTED_SAMPLE_SIZES}
            for size, root in roots.items():
                materialize_job(root, size, 1500, improvement=size / 10000.0)
            audit = review.run_review(
                [("G-3", "ordinary_zero")],
                roots,
                base / "out",
            )
            with (base / "out" / "sample_size_sensitivity.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))

        self.assertTrue(audit["passed"], audit["failures"])
        self.assertEqual(len(rows), 3)
        self.assertAlmostEqual(float(rows[0]["normalized_improvement_pp"]), 0.5)
        self.assertEqual(rows[0]["spoplus_cap_hit"], "True")
        self.assertEqual(rows[0]["all_achieved_objectives_exactly_equal"], "False")
        self.assertTrue(audit["fit_prefix_nested_by_topology"]["G-3"])


if __name__ == "__main__":
    unittest.main()
