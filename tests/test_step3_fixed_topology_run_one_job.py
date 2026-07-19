import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
JOB_SCRIPT = (
    ROOT
    / "surrogate_experiment_results"
    / "Step3"
    / "scripts"
    / "run_one_job.py"
)
BANK_SCRIPT = (
    ROOT
    / "surrogate_experiment_results"
    / "Step3"
    / "scripts"
    / "build_nested_train_bank.py"
)
EVAL_SCRIPT = (
    ROOT
    / "surrogate_experiment_results"
    / "Step3"
    / "scripts"
    / "build_fixed_eval_sets.py"
)
TRAIN_2STAGE_SCRIPT = (
    ROOT
    / "surrogate_experiment_results"
    / "Step3"
    / "scripts"
    / "train_2stage_fixed_topology.py"
)
EVALUATE_SCRIPT = (
    ROOT
    / "surrogate_experiment_results"
    / "Step3"
    / "scripts"
    / "evaluate_fixed_topology.py"
)
BUILDER = (
    ROOT
    / "surrogate_experiment_results"
    / "Step3"
    / "scripts"
    / "build_topology_bank.py"
)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def payload():
    return {
        "metadata": {"source_file": "G-test.json"},
        "data": {
            "0": {
                "type": "Pair",
                "patient": {"age": 40, "bloodtype": "A", "cPRA": 0.2, "hasBloodCompatibleDonor": True},
                "donors": [{"original_node_id": "0", "dage": 35, "bloodtype": "A"}],
                "matches": [{"recipient": "1", "utility": 60.0, "recipient_cpra": 0.5}],
            },
            "1": {
                "type": "Pair",
                "patient": {"age": 50, "bloodtype": "B", "cPRA": 0.5, "hasBloodCompatibleDonor": False},
                "donors": [{"original_node_id": "1", "dage": 45, "bloodtype": "O"}],
                "matches": [{"recipient": "0", "utility": 70.0, "recipient_cpra": 0.2}],
            },
        },
    }


def config():
    return {
        "status": "pilot_not_locked",
        "generator_version": "test-v1",
        "method": "bounded_additive_uniform",
        "recipient_cpra": {"lower": 0.0, "upper": 1.0, "half_width": 0.05},
        "utility": {"lower": 0.0, "upper": 100.0, "half_width": 5.0},
    }


def build_tiny_artifacts(root: Path):
    bank_mod = load_module(BANK_SCRIPT, f"build_nested_train_bank_for_job_{id(root)}")
    eval_mod = load_module(EVAL_SCRIPT, f"build_fixed_eval_sets_for_job_{id(root)}")
    builder = load_module(BUILDER, f"build_topology_bank_job_{id(root)}")
    template = builder.build_topology_template("G-test", payload(), max_cycle=3, max_chain=4)
    bank_path = root / "bank.npz"
    bank_manifest = bank_mod.build_nested_train_bank(
        topology_template=template,
        base_payload=payload(),
        output_path=bank_path,
        topology_id="G-test",
        regime="step2c_poly_d8_mult_eps050",
        train_seed=17,
        max_train_size=4,
        prefix_sizes=(2, 4),
        experiment_version="v-test",
        master_label_seed=20260619,
        generator_config=config(),
    )
    eval_result = eval_mod.build_fixed_eval_sets_for_topology(
        topology_template=template,
        base_payload=payload(),
        output_dir=root / "eval",
        topology_id="G-test",
        regime="step2c_poly_d8_mult_eps050",
        validation_size=1,
        test_size=1,
        experiment_version="v-test",
        master_label_seed=20260619,
        generator_config=config(),
    )
    return bank_path, bank_manifest, eval_result


class Step3FixedTopologyRunOneJobTests(unittest.TestCase):
    def test_dry_run_manifest_pairs_methods_on_same_prefix_eval_and_theta(self):
        job_mod = load_module(JOB_SCRIPT, "run_one_job")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _bank_path, bank_manifest, eval_result = build_tiny_artifacts(root)

            manifest = job_mod.prepare_paired_job_manifest(
                topology_id="G-test",
                regime="step2c_poly_d8_mult_eps050",
                train_seed=17,
                train_size=2,
                train_bank_manifest=bank_manifest,
                eval_manifest=eval_result["manifest"],
                output_dir=root / "job",
                theta_seed=42,
                gurobi_seed=42,
            )

            self.assertEqual(manifest["status"], "dry_run_ready")
            self.assertEqual(manifest["train_prefix_hash"], bank_manifest["prefix_hashes"]["2"])
            self.assertEqual(manifest["methods"]["2stage"]["train_prefix_hash"], manifest["methods"]["SPO+"]["train_prefix_hash"])
            self.assertEqual(manifest["methods"]["2stage"]["validation_hash"], manifest["methods"]["SPO+"]["validation_hash"])
            self.assertEqual(manifest["methods"]["2stage"]["test_hash"], manifest["methods"]["SPO+"]["test_hash"])
            self.assertEqual(manifest["methods"]["2stage"]["theta_init"], manifest["methods"]["SPO+"]["theta_init"])
            job_mod.validate_paired_job_manifest(manifest)

    def test_dry_run_writes_paired_manifest_and_commands(self):
        job_mod = load_module(JOB_SCRIPT, "run_one_job_dry_run_commands")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bank_path, bank_manifest, eval_result = build_tiny_artifacts(root)
            output_dir = root / "job"

            rc = job_mod.main(
                [
                    "--train-bank", str(bank_path),
                    "--eval-manifest", eval_result["eval_manifest_path"],
                    "--topology-id", "G-test",
                    "--regime", "step2c_poly_d8_mult_eps050",
                    "--train-seed", "17",
                    "--train-size", "2",
                    "--output-dir", str(output_dir),
                    "--max-epochs", "2",
                    "--dry-run",
                ]
            )

            self.assertEqual(rc, 0)
            manifest = json.loads((output_dir / "paired_job_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["train_prefix_hash"], bank_manifest["prefix_hashes"]["2"])
            self.assertIn("commands", manifest)
            self.assertIn("2stage", manifest["commands"])
            self.assertIn("SPO+", manifest["commands"])
            self.assertIn("evaluation", manifest["commands"])
            self.assertEqual(manifest["status"], "dry_run_ready")

    def test_commands_resolve_manifest_and_project_relative_eval_paths(self):
        job_mod = load_module(JOB_SCRIPT, "run_one_job_mixed_eval_paths")
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            eval_dir = project_root / "artifacts" / "G-1" / "data_seed=000042"
            eval_dir.mkdir(parents=True)
            eval_manifest_path = eval_dir / "eval_manifest.json"
            validation_path = eval_dir / "validation.npz"
            validation_path.touch()
            test_path = project_root / "artifacts" / "G-1" / "test" / "test.npz"
            test_path.parent.mkdir(parents=True)
            test_path.touch()
            eval_manifest = {
                "validation_path": "validation.npz",
                "test_path": "artifacts/G-1/test/test.npz",
            }
            manifest = {
                "output_directory": str(project_root / "job"),
                "train_size": 40,
                "theta_seed": 42,
                "max_epochs": 1500,
                "metric_stride": 1,
                "early_stop_patience": 20,
                "early_stop_min_delta": 0.0001,
                "train_prefix_hash": "train-hash",
                "validation_hash": "validation-hash",
                "test_hash": "test-hash",
                "gurobi_seed": 42,
            }

            with mock.patch.object(job_mod, "PROJECT_ROOT", project_root):
                commands = job_mod.build_job_commands(
                    train_bank_path=project_root / "train_bank.npz",
                    eval_manifest_path=eval_manifest_path,
                    eval_manifest=eval_manifest,
                    manifest=manifest,
                )

        two_stage = commands["2stage"]
        evaluation = commands["evaluation"]
        self.assertEqual(
            Path(two_stage[two_stage.index("--validation-set") + 1]),
            validation_path,
        )
        self.assertEqual(
            Path(evaluation[evaluation.index("--eval-set") + 1]),
            test_path,
        )

    def test_refuses_train_size_not_present_in_prefix_hashes(self):
        job_mod = load_module(JOB_SCRIPT, "run_one_job_missing_prefix")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _bank_path, bank_manifest, eval_result = build_tiny_artifacts(root)

            with self.assertRaisesRegex(ValueError, "prefix_hashes"):
                job_mod.prepare_paired_job_manifest(
                    topology_id="G-test",
                    regime="step2c_poly_d8_mult_eps050",
                    train_seed=17,
                    train_size=3,
                    train_bank_manifest=bank_manifest,
                    eval_manifest=eval_result["manifest"],
                    output_dir=root / "job",
                    theta_seed=42,
                    gurobi_seed=42,
                )

    def test_execute_orchestration_records_mocked_subprocess_statuses(self):
        job_mod = load_module(JOB_SCRIPT, "run_one_job_execute_mock")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bank_path, _bank_manifest, eval_result = build_tiny_artifacts(root)
            output_dir = root / "job"

            completed = mock.Mock(returncode=0)
            with mock.patch.object(job_mod.subprocess, "run", return_value=completed) as run_mock:
                rc = job_mod.main(
                    [
                        "--train-bank", str(bank_path),
                        "--eval-manifest", eval_result["eval_manifest_path"],
                        "--topology-id", "G-test",
                        "--regime", "step2c_poly_d8_mult_eps050",
                        "--train-seed", "17",
                        "--train-size", "2",
                        "--output-dir", str(output_dir),
                        "--max-epochs", "2",
                        "--execute",
                    ]
                )

            self.assertEqual(rc, 0)
            self.assertEqual(run_mock.call_count, 3)
            status = json.loads((output_dir / "job_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "success")
            self.assertEqual(status["2stage status"], "success")
            self.assertEqual(status["SPO+ status"], "success")
            self.assertEqual(status["evaluation status"], "success")

    def test_train_wrapper_validates_expected_hashes_and_writes_input_manifest(self):
        train_mod = load_module(TRAIN_2STAGE_SCRIPT, "train_2stage_fixed_topology_hash_contract")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bank_path, bank_manifest, eval_result = build_tiny_artifacts(root)
            out_dir = root / "train_2stage"

            rc = train_mod.main(
                [
                    "--train-bank", str(bank_path),
                    "--validation-set", eval_result["validation_path"],
                    "--out-dir", str(out_dir),
                    "--train-size", "2",
                    "--max-epochs", "2",
                    "--expected-train-prefix-hash", bank_manifest["prefix_hashes"]["2"],
                    "--expected-validation-hash", eval_result["validation_hash"],
                    "--dry-run",
                ]
            )

            self.assertEqual(rc, 0)
            manifest = json.loads((out_dir / "method_input_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["train_prefix_hash"], bank_manifest["prefix_hashes"]["2"])
            self.assertEqual(manifest["validation_hash"], eval_result["validation_hash"])
            with self.assertRaisesRegex(ValueError, "expected train prefix hash"):
                train_mod.main(
                    [
                        "--train-bank", str(bank_path),
                        "--validation-set", eval_result["validation_path"],
                        "--out-dir", str(root / "bad_train"),
                        "--train-size", "2",
                        "--expected-train-prefix-hash", "bad-hash",
                        "--dry-run",
                    ]
                )

    def test_evaluate_wrapper_validates_expected_test_hash_and_writes_input_manifest(self):
        eval_wrapper = load_module(EVALUATE_SCRIPT, "evaluate_fixed_topology_hash_contract")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _bank_path, _bank_manifest, eval_result = build_tiny_artifacts(root)
            weights = root / "dummy_weights.npz"
            weights.write_bytes(b"placeholder")
            out_dir = root / "eval_job"

            rc = eval_wrapper.main(
                [
                    "--eval-set", eval_result["test_path"],
                    "--out-dir", str(out_dir),
                    "--weights", str(weights),
                    "--expected-test-hash", eval_result["test_hash"],
                    "--dry-run",
                ]
            )

            self.assertEqual(rc, 0)
            manifest = json.loads((out_dir / "evaluation_input_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["test_hash"], eval_result["test_hash"])
            with self.assertRaisesRegex(ValueError, "expected test hash"):
                eval_wrapper.main(
                    [
                        "--eval-set", eval_result["test_path"],
                        "--out-dir", str(root / "bad_eval"),
                        "--weights", str(weights),
                        "--expected-test-hash", "bad-hash",
                        "--dry-run",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
