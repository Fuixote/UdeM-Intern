import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "surrogate_experiment_results"
    / "Step3"
    / "scripts"
    / "aggregate_phase_b_results.py"
)


def load_aggregation_module():
    spec = importlib.util.spec_from_file_location("step3_aggregate_phase_b_results", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def status_row(topology_id, seed, improvement, status="success"):
    gap_2stage = 10.0
    gap_spoplus = gap_2stage - improvement
    return {
        "topology_id": topology_id,
        "train_seed": str(seed),
        "status": status,
        "elapsed_seconds": "20.0",
        "train_sample_count": "40",
        "validation_sample_count": "10",
        "test_sample_count": "1000",
        "test_mean_decision_gap_2stage": str(gap_2stage),
        "test_mean_decision_gap_spoplus": str(gap_spoplus),
        "test_mean_normalized_gap_2stage": "0.10",
        "test_mean_normalized_gap_spoplus": str(0.10 - improvement / 100.0),
        "spoplus_improvement_gap": str(improvement),
        "spoplus_improvement_normalized_gap": str(improvement / 100.0),
    }


def descriptor_row(topology_id, complexity, landscape, candidates):
    return {
        "selection_rank": "1",
        "topology_id": topology_id,
        "complexity_bin": complexity,
        "structural_type": "cycle_chain",
        "landscape_regime": landscape,
        "screening_score": "0.5",
        "selection_reason": "test",
        "num_exchange_candidates": str(candidates),
        "num_cycles_total": "1",
        "num_3cycles": "1",
        "num_chains_total": str(max(1, candidates - 1)),
        "candidate_conflict_density": "0.5",
        "mean_candidates_per_vertex": "2.0",
        "num_distinct_oracle_solutions": "3",
        "oracle_solution_entropy": "0.4",
        "dominant_oracle_solution_fraction": "0.7",
        "fraction_linear_proxy_differs_from_oracle": "0.2",
        "mean_linear_proxy_normalized_gap_to_oracle": "0.02",
        "median_top1_top2_margin": "1.0",
        "mean_pairwise_oracle_jaccard": "0.8",
    }


class Step3PhaseBResultAggregationTests(unittest.TestCase):
    def test_summarizes_training_results_and_classifies_outcomes(self):
        module = load_aggregation_module()
        rows = []
        rows.extend(status_row("G-help", seed, 1.0) for seed in range(1, 6))
        rows.extend(status_row("G-harm", seed, -1.0) for seed in range(1, 6))
        rows.extend(status_row("G-neutral", seed, 0.0) for seed in range(1, 6))
        rows.extend(status_row("G-control", seed, 0.0) for seed in range(1, 6))
        rows.append(status_row("G-help", 99, 100.0, status="failed"))

        descriptors = {
            "G-help": descriptor_row("G-help", "rich", "proxy_hard", 60),
            "G-harm": descriptor_row("G-harm", "medium_rich", "neutral", 30),
            "G-neutral": descriptor_row("G-neutral", "low_medium", "neutral", 12),
            "G-control": descriptor_row("G-control", "sparse_simple", "easy_control", 6),
        }

        summaries = module.merge_phase_b_descriptors(
            module.summarize_training_results(rows),
            descriptors,
        )
        by_id = {row["topology_id"]: row for row in summaries}

        self.assertEqual(by_id["G-help"]["completed_jobs"], 5)
        self.assertEqual(by_id["G-help"]["failed_jobs"], 1)
        self.assertAlmostEqual(float(by_id["G-help"]["mean_spoplus_improvement_gap"]), 1.0)
        self.assertEqual(by_id["G-help"]["phase_b_outcome"], "helpful")
        self.assertEqual(by_id["G-harm"]["phase_b_outcome"], "harmful")
        self.assertEqual(by_id["G-neutral"]["phase_b_outcome"], "neutral")
        self.assertEqual(by_id["G-control"]["phase_b_outcome"], "control")
        self.assertEqual(by_id["G-help"]["complexity_bin"], "rich")

    def test_selects_phase_c_candidates_by_outcome_quotas(self):
        module = load_aggregation_module()
        rows = [
            {"topology_id": "G-help-1", "phase_b_outcome": "helpful", "mean_spoplus_improvement_gap": "3.0", "fraction_spoplus_better": "1.0", "fraction_spoplus_worse": "0.0", "max_abs_spoplus_improvement_gap": "3.0"},
            {"topology_id": "G-help-2", "phase_b_outcome": "helpful", "mean_spoplus_improvement_gap": "2.0", "fraction_spoplus_better": "1.0", "fraction_spoplus_worse": "0.0", "max_abs_spoplus_improvement_gap": "2.0"},
            {"topology_id": "G-harm-1", "phase_b_outcome": "harmful", "mean_spoplus_improvement_gap": "-3.0", "fraction_spoplus_better": "0.0", "fraction_spoplus_worse": "1.0", "max_abs_spoplus_improvement_gap": "3.0"},
            {"topology_id": "G-neutral-1", "phase_b_outcome": "neutral", "mean_spoplus_improvement_gap": "0.0", "fraction_spoplus_better": "0.0", "fraction_spoplus_worse": "0.0", "max_abs_spoplus_improvement_gap": "0.0"},
            {"topology_id": "G-control-1", "phase_b_outcome": "control", "mean_spoplus_improvement_gap": "0.0", "fraction_spoplus_better": "0.0", "fraction_spoplus_worse": "0.0", "max_abs_spoplus_improvement_gap": "0.0"},
        ]

        selected = module.select_phase_c_candidates(
            rows,
            quotas={"helpful": 1, "harmful": 1, "neutral": 1, "control": 1},
        )

        self.assertEqual(
            [row["topology_id"] for row in selected],
            ["G-help-1", "G-harm-1", "G-neutral-1", "G-control-1"],
        )
        self.assertEqual([row["phase_c_rank"] for row in selected], [1, 2, 3, 4])
        self.assertEqual(len({row["topology_id"] for row in selected}), 4)

    def test_write_outputs_creates_aggregation_artifacts(self):
        module = load_aggregation_module()
        summaries = [
            {
                "topology_id": "G-1",
                "phase_b_outcome": "helpful",
                "mean_spoplus_improvement_gap": "1.0",
                "fraction_spoplus_better": "1.0",
                "fraction_spoplus_worse": "0.0",
                "max_abs_spoplus_improvement_gap": "1.0",
                "completed_jobs": "5",
                "failed_jobs": "0",
            }
        ]
        selected = [dict(summaries[0], phase_c_rank=1, phase_c_selection_reason="quota_helpful")]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            module.write_aggregation_outputs(output_dir, summaries, selected)

            summary_csv = output_dir / "phase_b_topology_training_summary.csv"
            selected_csv = output_dir / "phase_c_candidate_topologies.csv"
            selected_txt = output_dir / "phase_c_topology_ids.txt"
            counts_json = output_dir / "phase_b_outcome_counts.json"
            result_json = output_dir / "phase_b_result_summary.json"

            self.assertTrue(summary_csv.exists())
            self.assertTrue(selected_csv.exists())
            self.assertTrue(selected_txt.exists())
            self.assertTrue(counts_json.exists())
            self.assertTrue(result_json.exists())

            with summary_csv.open(newline="") as handle:
                csv_rows = list(csv.DictReader(handle))
            counts = json.loads(counts_json.read_text())
            result = json.loads(result_json.read_text())
            self.assertEqual(csv_rows[0]["topology_id"], "G-1")
            self.assertEqual(selected_txt.read_text().splitlines(), ["G-1"])
            self.assertEqual(counts["outcome_counts"], {"helpful": 1})
            self.assertEqual(result["num_phase_c_candidates"], 1)


if __name__ == "__main__":
    unittest.main()
