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
    / "compare_phase_c_convergence.py"
)


def load_compare_module():
    spec = importlib.util.spec_from_file_location("step3_compare_phase_c_convergence", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def status_row(topology_id, train_seed, improvement, epochs):
    return {
        "topology_id": topology_id,
        "train_seed": str(train_seed),
        "status": "success",
        "epochs_2stage": str(epochs),
        "epochs_spoplus": str(epochs),
        "test_mean_decision_gap_2stage": "5.0",
        "test_mean_decision_gap_spoplus": str(5.0 - improvement),
        "test_mean_normalized_gap_2stage": "0.50",
        "test_mean_normalized_gap_spoplus": str(0.50 - improvement / 10.0),
        "spoplus_improvement_gap": str(improvement),
        "spoplus_improvement_normalized_gap": str(improvement / 10.0),
    }


def make_weight(run_root, topology_id, seed, method_file, epoch, theta):
    write_json(
        run_root
        / topology_id
        / f"train_seed={seed:06d}"
        / "model_weights"
        / method_file,
        {"selected_epoch": epoch, "theta": theta},
    )


class Step3PhaseCConvergenceCompareTests(unittest.TestCase):
    def test_compares_seed_level_100_vs_500_results_and_epochs(self):
        module = load_compare_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            status_100 = root / "status100.csv"
            status_500 = root / "status500.csv"
            runs_100 = root / "runs100"
            runs_500 = root / "runs500"
            fields = [
                "topology_id",
                "train_seed",
                "status",
                "epochs_2stage",
                "epochs_spoplus",
                "test_mean_decision_gap_2stage",
                "test_mean_decision_gap_spoplus",
                "test_mean_normalized_gap_2stage",
                "test_mean_normalized_gap_spoplus",
                "spoplus_improvement_gap",
                "spoplus_improvement_normalized_gap",
            ]
            write_csv(
                status_100,
                [status_row("G-1", 1, 1.0, 100), status_row("G-1", 2, 0.0, 100)],
                fields,
            )
            write_csv(
                status_500,
                [status_row("G-1", 1, 1.5, 500), status_row("G-1", 2, -0.5, 500)],
                fields,
            )
            for seed in [1, 2]:
                make_weight(
                    runs_100,
                    "G-1",
                    seed,
                    "2stage_best_by_validation_mse_loss.json",
                    100,
                    [1.0 + seed, 2.0],
                )
                make_weight(
                    runs_100,
                    "G-1",
                    seed,
                    "spoplus_best_by_validation_spoplus_loss.json",
                    90,
                    [3.0 + seed, 4.0],
                )
                make_weight(
                    runs_500,
                    "G-1",
                    seed,
                    "2stage_best_by_validation_mse_loss.json",
                    500,
                    [1.5 + seed, 2.0],
                )
                make_weight(
                    runs_500,
                    "G-1",
                    seed,
                    "spoplus_best_by_validation_spoplus_loss.json",
                    450,
                    [3.5 + seed, 4.0],
                )

            seed_rows, topology_rows = module.compare_convergence(
                status_100_csv=status_100,
                status_500_csv=status_500,
                runs_100_dir=runs_100,
                runs_500_dir=runs_500,
            )

        self.assertEqual(len(seed_rows), 2)
        self.assertEqual(len(topology_rows), 1)
        first = seed_rows[0]
        self.assertEqual(first["seed_outcome_100"], "better")
        self.assertEqual(first["seed_outcome_500"], "better")
        self.assertAlmostEqual(float(first["delta_improvement_gap"]), 0.5)
        self.assertEqual(first["selected_epoch_2stage_100"], 100)
        self.assertEqual(first["selected_epoch_2stage_500"], 500)
        summary = topology_rows[0]
        self.assertEqual(summary["topology_id"], "G-1")
        self.assertEqual(summary["matched_seed_count"], 2)
        self.assertAlmostEqual(float(summary["mean_improvement_gap_100"]), 0.5)
        self.assertAlmostEqual(float(summary["mean_improvement_gap_500"]), 0.5)
        self.assertEqual(summary["changed_seed_outcome_count"], 1)
        self.assertEqual(summary["fraction_2stage_selected_at_max_epoch_500"], 1.0)

    def test_write_outputs_creates_convergence_artifacts(self):
        module = load_compare_module()
        seed_rows = [{"topology_id": "G-1", "train_seed": 1}]
        topology_rows = [{"topology_id": "G-1", "matched_seed_count": 1}]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            module.write_convergence_outputs(output_dir, seed_rows, topology_rows)

            self.assertTrue((output_dir / "phase_c_convergence_seed_compare.csv").exists())
            self.assertTrue((output_dir / "phase_c_convergence_topology_summary.csv").exists())
            summary = json.loads((output_dir / "phase_c_convergence_summary.json").read_text())
            self.assertEqual(summary["num_topologies"], 1)
            self.assertEqual(summary["num_seed_rows"], 1)


if __name__ == "__main__":
    unittest.main()
