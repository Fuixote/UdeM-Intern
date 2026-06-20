import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "surrogate_experiment_results"
    / "Step3"
    / "scripts"
    / "run_phase_b_training.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("run_phase_b_training", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_materialized_dataset(root):
    dataset_dir = root / "datasets"
    topology_dir = dataset_dir / "G-1"
    for idx in range(10):
        write_json(topology_dir / "validation" / f"G-{idx:06d}.json", {"split": "validation"})
    for idx in range(1000):
        write_json(topology_dir / "test" / f"G-{idx:06d}.json", {"split": "test"})
    for train_seed in [1, 2]:
        for idx in range(40):
            write_json(
                topology_dir / f"train_seed={train_seed:06d}" / "train" / f"G-{idx:06d}.json",
                {"split": "train", "train_seed": train_seed},
            )
    write_json(
        topology_dir / "dataset_manifest.json",
        {
            "topology_id": "G-1",
            "train_seed_count": 2,
            "train_sample_count": 40,
            "validation_sample_count": 10,
            "test_sample_count": 1000,
            "topology_hash": "processed-hash",
            "topology_bank_hash": "bank-hash",
            "feasible_set_hash": "feasible-hash",
        },
    )
    write_csv(
        dataset_dir / "phase_b_dataset_index.csv",
        [
            {
                "topology_id": "G-1",
                "output_dir": str(topology_dir),
                "train_seed_count": 2,
                "training_size_budget": 50,
                "train_sample_count": 40,
                "validation_sample_count": 10,
                "test_sample_count": 1000,
                "topology_hash": "processed-hash",
                "topology_bank_hash": "bank-hash",
                "feasible_set_hash": "feasible-hash",
                "status": "materialized",
            }
        ],
        [
            "topology_id",
            "output_dir",
            "train_seed_count",
            "training_size_budget",
            "train_sample_count",
            "validation_sample_count",
            "test_sample_count",
            "topology_hash",
            "topology_bank_hash",
            "feasible_set_hash",
            "status",
        ],
    )
    return dataset_dir


class Step3PhaseBTrainingRunnerTests(unittest.TestCase):
    def test_discovers_fixed_topology_train_seed_jobs(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = make_materialized_dataset(root)

            jobs = module.discover_phase_b_jobs(
                dataset_dir=dataset_dir,
                output_dir=root / "runs",
                split_dir=root / "splits",
                max_topologies=1,
                max_train_seeds=2,
            )

            self.assertEqual(len(jobs), 2)
            self.assertEqual(jobs[0].topology_id, "G-1")
            self.assertEqual(jobs[0].train_seed, 1)
            self.assertEqual(jobs[1].train_seed, 2)
            self.assertEqual(jobs[0].train_sample_count, 40)
            self.assertEqual(jobs[0].validation_sample_count, 10)
            self.assertEqual(jobs[0].test_sample_count, 1000)
            self.assertEqual(jobs[0].train_dir.name, "train")
            self.assertEqual(jobs[0].validation_dir.name, "validation")
            self.assertEqual(jobs[0].test_dir.name, "test")

    def test_writes_step1c_split_for_train_seed_and_fixed_test(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = make_materialized_dataset(root)
            job = module.discover_phase_b_jobs(
                dataset_dir=dataset_dir,
                output_dir=root / "runs",
                split_dir=root / "splits",
                max_topologies=1,
                max_train_seeds=1,
            )[0]

            split = module.write_job_split(job)

            self.assertEqual(split["train_pool_size"], 40)
            self.assertEqual(split["validation_size"], 0)
            self.assertEqual(split["test_size"], 1000)
            self.assertEqual(len(split["train_pool"]), 40)
            self.assertEqual(len(split["validation"]), 0)
            self.assertEqual(len(split["test"]), 1000)
            self.assertTrue(Path(split["train_pool"][0]["path"]).as_posix().endswith("train/G-000000.json"))
            self.assertTrue(Path(split["test"][0]["path"]).as_posix().endswith("test/G-000000.json"))

    def test_builds_training_and_evaluation_commands_with_step3_paths(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = make_materialized_dataset(root)
            job = module.discover_phase_b_jobs(
                dataset_dir=dataset_dir,
                output_dir=root / "runs",
                split_dir=root / "splits",
                max_topologies=1,
                max_train_seeds=1,
            )[0]
            options = module.PhaseBOptions(
                project_root=Path("/repo"),
                python_bin="/env/bin/python",
                epochs_2stage=2,
                epochs_spoplus=3,
                metric_stride=1,
                validation_limit=5,
                test_limit=20,
                bootstrap_samples=10,
            )

            two_stage = module.build_train_2stage_command(job, options)
            spoplus = module.build_train_spoplus_command(job, options)
            evaluate = module.build_evaluate_command(job, options)

            self.assertEqual(two_stage[0], "/env/bin/python")
            self.assertIn("train_2stage_earlystop.py", two_stage[1])
            self.assertIn("--validation_data_dir", two_stage)
            self.assertIn(str(job.validation_dir), two_stage)
            self.assertIn("--train_size", two_stage)
            self.assertIn("40", two_stage)
            self.assertIn("--early_stop_patience", two_stage)
            self.assertIn("20", two_stage)
            self.assertIn("--early_stop_min_delta", two_stage)
            self.assertIn("0.0001", two_stage)
            self.assertIn("--n_epochs", spoplus)
            self.assertIn("3", spoplus)
            self.assertIn("--early_stop_metric", spoplus)
            self.assertIn("validation_spoplus_loss", spoplus)
            self.assertIn("--early_stop_patience", spoplus)
            self.assertIn("20", spoplus)
            self.assertIn("--early_stop_min_delta", spoplus)
            self.assertIn("0.0001", spoplus)
            self.assertIn("--test_limit", evaluate)
            self.assertIn("20", evaluate)
            self.assertIn("spoplus_best_by_validation_spoplus_loss.npz", evaluate[-1])
            self.assertNotIn("spoplus_best_by_validation_decision_gap.npz", evaluate)

    def test_summarizes_primary_2stage_vs_spoplus_metrics(self):
        module = load_module()
        rows = [
            {
                "method": "2stage",
                "selection_metric": "validation_mse_loss",
                "test_mean_decision_gap": "4.0",
                "test_mean_normalized_gap": "0.20",
            },
            {
                "method": "spoplus",
                "selection_metric": "validation_spoplus_loss",
                "test_mean_decision_gap": "1.5",
                "test_mean_normalized_gap": "0.05",
            },
        ]

        summary = module.summarize_primary_metrics(rows)

        self.assertEqual(summary["test_mean_decision_gap_2stage"], 4.0)
        self.assertEqual(summary["test_mean_decision_gap_spoplus"], 1.5)
        self.assertAlmostEqual(summary["spoplus_improvement_gap"], 2.5)
        self.assertAlmostEqual(summary["spoplus_improvement_normalized_gap"], 0.15)


if __name__ == "__main__":
    unittest.main()
