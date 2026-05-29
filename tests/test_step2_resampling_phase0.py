import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "surrogate_experiment_results"
    / "Step2_resampling"
    / "run_phase0_hard_graph_diagnostic.py"
)


def load_phase0_module():
    spec = importlib.util.spec_from_file_location("phase0_diagnostic", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Step2ResamplingPhase0Test(unittest.TestCase):
    def test_compute_diagnostic_uses_top_mse_normalized_gap_and_paired_improvement(self):
        module = load_phase0_module()
        rows = [
            {
                "checkpoint_label": "2stage_val_mse",
                "graph": "G-1.json",
                "gap": "40.0",
                "normalized_gap": "0.40",
                "evaluation_dataset": "toy",
                "optimal_obj": "100.0",
                "model_path": "mse.npz",
            },
            {
                "checkpoint_label": "spoplus_val_spoplus_loss",
                "graph": "G-1.json",
                "gap": "10.0",
                "normalized_gap": "0.10",
                "model_path": "spo.npz",
            },
            {
                "checkpoint_label": "2stage_val_mse",
                "graph": "G-2.json",
                "gap": "30.0",
                "normalized_gap": "0.30",
                "evaluation_dataset": "toy",
                "optimal_obj": "100.0",
                "model_path": "mse.npz",
            },
            {
                "checkpoint_label": "spoplus_val_spoplus_loss",
                "graph": "G-2.json",
                "gap": "35.0",
                "normalized_gap": "0.35",
                "model_path": "spo.npz",
            },
            {
                "checkpoint_label": "2stage_val_mse",
                "graph": "G-3.json",
                "gap": "20.0",
                "normalized_gap": "0.20",
                "evaluation_dataset": "toy",
                "optimal_obj": "100.0",
                "model_path": "mse.npz",
            },
            {
                "checkpoint_label": "spoplus_val_spoplus_loss",
                "graph": "G-3.json",
                "gap": "15.0",
                "normalized_gap": "0.15",
                "model_path": "spo.npz",
            },
            {
                "checkpoint_label": "2stage_val_mse",
                "graph": "G-4.json",
                "gap": "10.0",
                "normalized_gap": "0.10",
                "evaluation_dataset": "toy",
                "optimal_obj": "100.0",
                "model_path": "mse.npz",
            },
            {
                "checkpoint_label": "spoplus_val_spoplus_loss",
                "graph": "G-4.json",
                "gap": "5.0",
                "normalized_gap": "0.05",
                "model_path": "spo.npz",
            },
        ]

        summary, top_rows = module.compute_diagnostic(
            rows,
            top_fraction=0.5,
            baseline_label="2stage_val_mse",
            candidate_label="spoplus_val_spoplus_loss",
        )

        self.assertEqual(summary["graph_count"], 4)
        self.assertEqual(summary["top_graph_count"], 2)
        self.assertAlmostEqual(summary["all_mean_norm_gap_improvement"], 0.0875)
        self.assertAlmostEqual(summary["top10_mean_norm_gap_improvement"], 0.125)
        self.assertAlmostEqual(summary["top10_fraction_graphs_improved"], 0.5)
        self.assertEqual([row["graph"] for row in top_rows], ["G-1.json", "G-2.json"])
        self.assertEqual([row["rank_by_2stage_norm_gap"] for row in top_rows], [1, 2])


if __name__ == "__main__":
    unittest.main()
