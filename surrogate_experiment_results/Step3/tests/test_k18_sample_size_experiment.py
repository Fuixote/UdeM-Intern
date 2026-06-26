import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = (
    ROOT
    / "surrogate_experiment_results"
    / "Step3"
    / "K18_analysis"
    / "experiment_01_budget4to1"
)
EXPERIMENT_SCRIPTS = EXPERIMENT_DIR / "scripts"
STEP3_SCRIPTS = ROOT / "surrogate_experiment_results" / "Step3" / "scripts"
RUN_ONE_JOB_SCRIPT = STEP3_SCRIPTS / "run_one_job.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def common_module():
    return load_module("fixed_topology_xy_common_test", STEP3_SCRIPTS / "fixed_topology_xy_common.py")


def fake_sample(index):
    manifest = {
        "sample_id": f"S-{index:06d}",
        "topology_id": "G-1",
        "regime": "step2c_poly_d8_mult_eps050",
        "split_namespace": "screen_train",
        "train_seed": 101,
        "sample_index": index,
        "x_hash": f"x-{index}",
        "label_hash": f"y-{index}",
    }
    return {
        "payload": {"id": index},
        "X": np.asarray([[float(index), float(index + 1)]], dtype=float),
        "y": np.asarray([float(index)], dtype=float),
        "manifest": manifest,
    }


def write_fake_train_bank(path, prefix_hashes):
    common = common_module()
    samples = [fake_sample(index) for index in range(8)]
    manifest = {
        "topology_id": "G-1",
        "regime": "step2c_poly_d8_mult_eps050",
        "protocol": "screen",
        "split_namespace": "screen_train",
        "train_seed": 101,
        "max_train_size": 8,
        "prefix_sizes": sorted(int(size) for size in prefix_hashes),
        "bank_hash": "bank-hash",
        "prefix_hashes": {str(key): value for key, value in prefix_hashes.items()},
        "samples": [sample["manifest"] for sample in samples],
    }
    common.write_npz_dataset(path, samples=samples, manifest=manifest)


class K18SampleSizeExperimentTests(unittest.TestCase):
    def test_split_fit_samples_keeps_training_and_validation_roles_nested(self):
        module = load_module(
            "build_nested_fit_validation_bank",
            EXPERIMENT_SCRIPTS / "build_nested_fit_validation_bank.py",
        )
        samples = [fake_sample(index) for index in range(10)]

        split = module.split_fit_samples(samples, sample_sizes=[5, 10])

        self.assertEqual(
            [sample["manifest"]["sample_index"] for sample in split["training_samples"]],
            [0, 1, 2, 3, 5, 6, 7, 8],
        )
        self.assertEqual(split["sample_size_splits"]["5"]["training_size"], 4)
        self.assertEqual(split["sample_size_splits"]["5"]["validation_size"], 1)
        self.assertEqual(split["sample_size_splits"]["10"]["training_size"], 8)
        self.assertEqual(split["sample_size_splits"]["10"]["validation_size"], 2)
        self.assertEqual(
            [
                sample["manifest"]["sample_index"]
                for sample in split["validation_samples_by_sample_size"]["5"]
            ],
            [4],
        )
        self.assertEqual(
            [
                sample["manifest"]["sample_index"]
                for sample in split["validation_samples_by_sample_size"]["10"]
            ],
            [4, 9],
        )
        self.assertEqual(split["training_prefix_sizes"], [4, 8])

    def test_plan_rows_use_sample_size_fields_and_training_cli_argument(self):
        module = load_module(
            "plan_k18_sample_size_jobs",
            EXPERIMENT_SCRIPTS / "plan_k18_sample_size_jobs.py",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data" / "step2c_poly_d8_mult_eps050" / "G-1" / "data_seed=000101"
            data_dir.mkdir(parents=True)
            write_fake_train_bank(data_dir / "train_bank.npz", {4: "train-4", 8: "train-8"})
            test_path = root / "data" / "step2c_poly_d8_mult_eps050" / "G-1" / "test" / "test.npz"
            for sample_size, validation_hash in [(5, "val-5"), (10, "val-10")]:
                manifest_path = data_dir / f"eval_manifest_sample_size{sample_size:03d}.json"
                manifest_path.write_text(
                    json.dumps(
                        {
                            "topology_id": "G-1",
                            "protocol": "screen",
                            "data_seed": 101,
                            "sample_size": sample_size,
                            "training_size": 4 if sample_size == 5 else 8,
                            "validation_size": 1 if sample_size == 5 else 2,
                            "trainer_train_size_arg": 4 if sample_size == 5 else 8,
                            "validation_path": f"validation_sample_size{sample_size:03d}.npz",
                            "validation_hash": validation_hash,
                            "test_path": str(test_path),
                            "test_hash": "test-hash",
                        }
                    ),
                    encoding="utf-8",
                )

            plan = module.build_plan(
                topology_rows=[{"topology_id": "G-1", "role": "test"}],
                output_root=root,
                regime="step2c_poly_d8_mult_eps050",
                data_seeds=[101],
                sample_size_splits={
                    5: {"training_size": 4, "validation_size": 1},
                    10: {"training_size": 8, "validation_size": 2},
                },
                protocol="screen",
            )

        self.assertEqual(plan["job_count"], 2)
        first = plan["jobs"][0]
        self.assertEqual(first["sample_size"], 5)
        self.assertEqual(first["training_size"], 4)
        self.assertEqual(first["trainer_train_size_arg"], 4)
        self.assertEqual(first["expected_training_hash"], "train-4")
        self.assertIn("--train-size 4", first["run_one_job_command"])
        self.assertIn("--sample-size 5", first["run_one_job_command"])
        self.assertNotIn("sample_budget", first)

    def test_audit_accepts_nested_training_and_validation_artifacts(self):
        builder = load_module(
            "build_nested_fit_validation_bank",
            EXPERIMENT_SCRIPTS / "build_nested_fit_validation_bank.py",
        )
        audit = load_module(
            "audit_k18_sample_size_artifacts",
            EXPERIMENT_SCRIPTS / "audit_k18_sample_size_artifacts.py",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            test_path = root / "test.npz"
            result = builder.write_artifacts_from_fit_samples(
                fit_samples=[fake_sample(index) for index in range(10)],
                output_dir=root,
                topology_id="G-1",
                regime="step2c_poly_d8_mult_eps050",
                data_seed=101,
                protocol="screen",
                sample_sizes=[5, 10],
                test_path=test_path,
                test_hash="test-hash",
            )

            audit_result = audit.audit_artifacts(
                train_bank_path=Path(result["train_bank_path"]),
                split_manifest_path=Path(result["split_manifest_path"]),
                eval_manifest_paths=[
                    Path(result["eval_manifest_paths"]["5"]),
                    Path(result["eval_manifest_paths"]["10"]),
                ],
            )

        self.assertTrue(audit_result["passed"], audit_result["failures"])
        self.assertEqual(audit_result["sample_size_count"], 2)
        self.assertEqual(audit_result["training_prefix_sizes"], [4, 8])

    def test_run_one_job_records_sample_size_metadata_without_changing_train_size(self):
        run_one_job = load_module("run_one_job_test", RUN_ONE_JOB_SCRIPT)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_bank = root / "train_bank.npz"
            write_fake_train_bank(train_bank, {4: "train-4"})
            eval_manifest = root / "eval_manifest_sample_size005.json"
            eval_manifest.write_text(
                json.dumps(
                    {
                        "validation_path": "validation_sample_size005.npz",
                        "validation_hash": "val-5",
                        "test_path": "test.npz",
                        "test_hash": "test-hash",
                        "sample_size": 5,
                        "training_size": 4,
                        "validation_size": 1,
                    }
                ),
                encoding="utf-8",
            )
            output_dir = root / "job"

            rc = run_one_job.main(
                [
                    "--train-bank",
                    str(train_bank),
                    "--eval-manifest",
                    str(eval_manifest),
                    "--topology-id",
                    "G-1",
                    "--regime",
                    "step2c_poly_d8_mult_eps050",
                    "--protocol",
                    "screen",
                    "--train-seed",
                    "101",
                    "--train-size",
                    "4",
                    "--sample-size",
                    "5",
                    "--output-dir",
                    str(output_dir),
                ]
            )

            paired_manifest = json.loads((output_dir / "paired_job_manifest.json").read_text())

        self.assertEqual(rc, 0)
        self.assertEqual(paired_manifest["sample_size"], 5)
        self.assertEqual(paired_manifest["train_size"], 4)
        self.assertEqual(paired_manifest["methods"]["2stage"]["train_prefix_hash"], "train-4")


if __name__ == "__main__":
    unittest.main()
