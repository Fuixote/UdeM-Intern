import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "surrogate_experiment_results" / "Step3" / "scripts" / "run_phase_b_training.py"
TRAIN_2STAGE_EARLYSTOP_SCRIPT = (
    ROOT / "surrogate_experiment_results" / "Step3" / "scripts" / "train_2stage_earlystop.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("run_phase_b_training", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_2stage_earlystop_module():
    spec = importlib.util.spec_from_file_location("train_2stage_earlystop", TRAIN_2STAGE_EARLYSTOP_SCRIPT)
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
    for idx in range(40):
        write_json(
            topology_dir / "train_seed=000001" / "train" / f"G-{idx:06d}.json",
            {"split": "train", "train_seed": 1},
        )
    write_csv(
        dataset_dir / "phase_b_dataset_index.csv",
        [
            {
                "topology_id": "G-1",
                "output_dir": str(topology_dir),
                "train_seed_count": 1,
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


class Step3PhaseBTrainingEarlyStopTests(unittest.TestCase):
    def test_default_early_stop_hyperparameters_are_step3_standard(self):
        module = load_module()

        options = module.PhaseBOptions()
        args = module.parse_args([])

        self.assertEqual(options.early_stop_patience_2stage, 20)
        self.assertEqual(options.early_stop_min_delta_2stage, 0.0001)
        self.assertEqual(options.early_stop_patience_spoplus, 20)
        self.assertEqual(options.early_stop_min_delta_spoplus, 0.0001)
        self.assertEqual(args.early_stop_patience_2stage, 20)
        self.assertEqual(args.early_stop_min_delta_2stage, 0.0001)
        self.assertEqual(args.early_stop_patience_spoplus, 20)
        self.assertEqual(args.early_stop_min_delta_spoplus, 0.0001)

    def test_native_early_stopping_selects_best_epoch_before_patience_stop(self):
        module = load_module()

        summary = module.simulate_native_early_stopping(
            epochs=[0, 10, 20, 30, 40, 50],
            values=[10.0, 8.0, 7.0, 7.05, 7.07, 6.5],
            patience=2,
            min_delta=0.01,
            metric_name="validation_mse_loss",
            max_epochs=50,
            metric_stride=10,
        )

        self.assertTrue(summary["enabled"])
        self.assertEqual(summary["source"], "step3_native")
        self.assertTrue(summary["stopped_early"])
        self.assertEqual(summary["best_epoch"], 20)
        self.assertEqual(summary["stop_epoch"], 40)
        self.assertEqual(summary["best_value"], 7.0)

    def test_2stage_command_uses_step3_native_trainer_when_early_stop_enabled(self):
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
                early_stop_patience_2stage=20,
                early_stop_min_delta_2stage=0.001,
                epochs_2stage=1500,
                metric_stride=1,
            )

            command = module.build_train_2stage_command(job, options)

            self.assertIn("train_2stage_earlystop.py", command[1])
            self.assertIn("--early_stop_patience", command)
            self.assertIn("20", command)
            self.assertIn("--early_stop_min_delta", command)
            self.assertIn("0.001", command)
            self.assertIn("--metric_stride", command)
            self.assertIn("1", command)

    def test_2stage_native_trainer_stops_before_max_epoch(self):
        module = load_2stage_earlystop_module()
        train_graphs = [{"X": module.np.zeros((1, 2)), "w_true": module.np.zeros(1)}]
        validation_graphs = [{"X": module.np.zeros((1, 2)), "w_true": module.np.zeros(1)}]

        trajectory, summary = module.run_mse_trajectory_with_early_stopping(
            train_graphs,
            validation_graphs,
            theta_init=module.np.asarray([1.0, 1.0], dtype=float),
            n_epochs=10,
            lr=0.05,
            metric_stride=1,
            patience=2,
            min_delta=0.0,
        )

        self.assertEqual(len(trajectory), 3)
        self.assertEqual(summary["source"], "step3_native")
        self.assertEqual(summary["best_epoch"], 0)
        self.assertEqual(summary["stop_epoch"], 2)
        self.assertEqual(summary["stopped_epoch"], 2)
        self.assertTrue(summary["stopped_early"])

    def test_spoplus_command_uses_step1c_native_early_stop_when_enabled(self):
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
                early_stop_patience_spoplus=20,
                early_stop_min_delta_spoplus=0.002,
            )

            command = module.build_train_spoplus_command(job, options)

            self.assertIn("train_spoplus.py", command[1])
            self.assertIn("--early_stop_metric", command)
            self.assertIn("validation_spoplus_loss", command)
            self.assertIn("--early_stop_patience", command)
            self.assertIn("20", command)
            self.assertIn("--early_stop_min_delta", command)
            self.assertIn("0.002", command)

    def test_evaluation_uses_primary_weights_when_native_early_stop_enabled(self):
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
                early_stop_patience_2stage=2,
                early_stop_patience_spoplus=3,
            )

            weights = module.evaluation_weight_paths(job, options)

            self.assertEqual(
                [path.name for path in weights],
                [
                    "2stage_best_by_validation_mse_loss.npz",
                    "spoplus_best_by_validation_spoplus_loss.npz",
                ],
            )

    def test_cli_and_status_row_include_early_stop_hyperparameters(self):
        module = load_module()
        args = module.parse_args(
            [
                "--early-stop-patience-2stage",
                "4",
                "--early-stop-min-delta-2stage",
                "0.001",
                "--early-stop-patience-spoplus",
                "5",
                "--early-stop-min-delta-spoplus",
                "0.002",
            ]
        )
        options = module.PhaseBOptions(
            early_stop_patience_2stage=args.early_stop_patience_2stage,
            early_stop_min_delta_2stage=args.early_stop_min_delta_2stage,
            early_stop_patience_spoplus=args.early_stop_patience_spoplus,
            early_stop_min_delta_spoplus=args.early_stop_min_delta_spoplus,
        )
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

            row = module.result_row(
                module.JobResult(
                    job=job,
                    status="success",
                    return_code=0,
                    elapsed_seconds=1.0,
                    posthoc_early_stop_seconds=0.25,
                ),
                options,
            )

            self.assertEqual(row["early_stop_patience_2stage"], 4)
            self.assertEqual(row["early_stop_min_delta_2stage"], 0.001)
            self.assertEqual(row["early_stop_patience_spoplus"], 5)
            self.assertEqual(row["early_stop_min_delta_spoplus"], 0.002)
            self.assertEqual(row["posthoc_early_stop_seconds"], "0.25")


if __name__ == "__main__":
    unittest.main()
