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
TOPOLOGY_TEMPLATE = {
    "topology_id": "G-1",
    "topology_hash": "topology-hash",
    "arc_order_hash": "arc-order-hash",
    "feasible_set_hash": "feasible-set-hash",
}
GENERATOR_CONFIG = {
    "generator_version": "fixed_topology_context_v1_test",
    "label": {"epsilon_bar": 0.5},
}


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


def fake_test_sample(index):
    sample = fake_sample(index)
    sample["manifest"] = {
        **sample["manifest"],
        "sample_id": f"T-{index:06d}",
        "split_namespace": "screen_test",
        "train_seed": None,
    }
    return sample


def write_fake_train_bank(path, prefix_hashes):
    common = common_module()
    samples = [fake_sample(index) for index in range(8)]
    manifest = {
        "topology_id": "G-1",
        "regime": "step2c_poly_d8_mult_eps050",
        "protocol": "screen",
        "split_namespace": "screen_train",
        "data_seed": 101,
        "train_seed": 101,
        "max_train_size": 8,
        "prefix_sizes": sorted(int(size) for size in prefix_hashes),
        "bank_hash": "bank-hash",
        "prefix_hashes": {str(key): value for key, value in prefix_hashes.items()},
        "samples": [sample["manifest"] for sample in samples],
        "experiment_version": "exp-v1",
        "master_label_seed": 20260626,
        "generator_version": GENERATOR_CONFIG["generator_version"],
        "generator_config_hash": common.generator_config_hash(GENERATOR_CONFIG),
        "topology_hash": TOPOLOGY_TEMPLATE["topology_hash"],
        "arc_order_hash": TOPOLOGY_TEMPLATE["arc_order_hash"],
        "feasible_set_hash": TOPOLOGY_TEMPLATE["feasible_set_hash"],
    }
    common.write_npz_dataset(path, samples=samples, manifest=manifest)


def write_fake_dataset(path, samples, extra_manifest=None):
    common = common_module()
    sample_rows = [sample["manifest"] for sample in samples]
    manifest = {
        "topology_id": "G-1",
        "regime": "step2c_poly_d8_mult_eps050",
        "protocol": "screen",
        "split_namespace": samples[0]["manifest"]["split_namespace"],
        "train_seed": samples[0]["manifest"].get("train_seed"),
        "sample_count": len(samples),
        "samples": sample_rows,
        "dataset_hash": common.sample_manifest_hashes(sample_rows),
        "experiment_version": "exp-v1",
        "master_label_seed": 20260626,
        "generator_version": GENERATOR_CONFIG["generator_version"],
        "generator_config_hash": common.generator_config_hash(GENERATOR_CONFIG),
        "topology_hash": TOPOLOGY_TEMPLATE["topology_hash"],
        "arc_order_hash": TOPOLOGY_TEMPLATE["arc_order_hash"],
        "feasible_set_hash": TOPOLOGY_TEMPLATE["feasible_set_hash"],
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    common.write_npz_dataset(path, samples=samples, manifest=manifest)
    return manifest


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
            test_path.parent.mkdir(parents=True)
            write_fake_dataset(test_path, [fake_test_sample(index) for index in range(2)])
            for sample_size, validation_hash in [(5, "val-5"), (10, "val-10")]:
                validation_path = data_dir / f"validation_sample_size{sample_size:03d}.npz"
                write_fake_dataset(
                    validation_path,
                    [fake_sample(index) for index in range(1 if sample_size == 5 else 2)],
                    {
                        "sample_size": sample_size,
                        "source_namespace": "screen_train",
                        "fit_role": "validation",
                        "validation_scheme": "every_fifth_sample",
                        "dataset_hash": validation_hash,
                    },
                )
                manifest_path = data_dir / f"eval_manifest_sample_size{sample_size:03d}.json"
                manifest_path.write_text(
                    json.dumps(
                        {
                            "topology_id": "G-1",
                            "regime": "step2c_poly_d8_mult_eps050",
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
        self.assertEqual(first["status"], "ready")

    def test_plan_fails_in_strict_mode_when_artifacts_are_missing(self):
        module = load_module(
            "plan_k18_sample_size_jobs_strict",
            EXPERIMENT_SCRIPTS / "plan_k18_sample_size_jobs.py",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                module.build_plan(
                    topology_rows=[{"topology_id": "G-1", "role": "test"}],
                    output_root=root,
                    regime="step2c_poly_d8_mult_eps050",
                    data_seeds=[101],
                    sample_size_splits={5: {"training_size": 4, "validation_size": 1}},
                    protocol="screen",
                )

    def test_builder_writes_fit_manifest_provenance_and_source_namespace(self):
        builder = load_module(
            "build_nested_fit_validation_bank_provenance",
            EXPERIMENT_SCRIPTS / "build_nested_fit_validation_bank.py",
        )
        common = common_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            test_path = root / "test.npz"
            test_manifest = write_fake_dataset(test_path, [fake_test_sample(index) for index in range(2)])
            result = builder.write_artifacts_from_fit_samples(
                fit_samples=[fake_sample(index) for index in range(10)],
                output_dir=root,
                topology_id="G-1",
                regime="step2c_poly_d8_mult_eps050",
                data_seed=101,
                protocol="screen",
                sample_sizes=[5, 10],
                test_path=test_path,
                test_hash=test_manifest["dataset_hash"],
                topology_template=TOPOLOGY_TEMPLATE,
                generator_config=GENERATOR_CONFIG,
                experiment_version="exp-v1",
                master_label_seed=20260626,
            )

            train_manifest = common.read_npz_dataset(result["train_bank_path"])["manifest"]
            validation_manifest = common.read_npz_dataset(result["validation_paths"]["5"])["manifest"]
            split_manifest = json.loads(Path(result["split_manifest_path"]).read_text(encoding="utf-8"))
            fit_manifest = json.loads(Path(result["fit_manifest_path"]).read_text(encoding="utf-8"))

        for manifest in (train_manifest, validation_manifest, fit_manifest):
            self.assertEqual(manifest["experiment_version"], "exp-v1")
            self.assertEqual(manifest["master_label_seed"], 20260626)
            self.assertEqual(manifest["generator_version"], "fixed_topology_context_v1_test")
            self.assertEqual(manifest["generator_config_hash"], common.generator_config_hash(GENERATOR_CONFIG))
            self.assertEqual(manifest["topology_hash"], "topology-hash")
            self.assertEqual(manifest["arc_order_hash"], "arc-order-hash")
            self.assertEqual(manifest["feasible_set_hash"], "feasible-set-hash")
        self.assertEqual(validation_manifest["split_namespace"], "screen_train")
        self.assertEqual(validation_manifest["source_namespace"], "screen_train")
        self.assertEqual(validation_manifest["fit_role"], "validation")
        self.assertEqual(validation_manifest["validation_scheme"], "every_fifth_sample")
        self.assertEqual(fit_manifest["sample_count"], 10)
        self.assertEqual(split_manifest["fit_manifest_path"], result["fit_manifest_path"])
        self.assertEqual(split_manifest["fit_bank_hash"], fit_manifest["fit_bank_hash"])

    def test_test_bank_builder_writes_verifiable_screen_test_artifact(self):
        module = load_module(
            "build_k18_test_bank",
            EXPERIMENT_SCRIPTS / "build_k18_test_bank.py",
        )
        common = common_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = module.write_test_bank_from_samples(
                samples=[fake_test_sample(index) for index in range(3)],
                output_dir=root,
                topology_id="G-1",
                regime="step2c_poly_d8_mult_eps050",
                protocol="screen",
                topology_template=TOPOLOGY_TEMPLATE,
                generator_config=GENERATOR_CONFIG,
                experiment_version="exp-v1",
                master_label_seed=20260626,
            )
            dataset = common.read_npz_dataset(result["test_path"])
            manifest = json.loads(Path(result["test_manifest_path"]).read_text(encoding="utf-8"))

        self.assertEqual(dataset["manifest"]["split_namespace"], "screen_test")
        self.assertEqual(dataset["manifest"]["sample_count"], 3)
        self.assertEqual(result["test_hash"], dataset["manifest"]["dataset_hash"])
        self.assertEqual(manifest["test_path"], result["test_path"])
        self.assertEqual(manifest["test_hash"], result["test_hash"])
        self.assertEqual(manifest["test_size"], 3)
        self.assertEqual(manifest["generator_config_hash"], common.generator_config_hash(GENERATOR_CONFIG))

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
            test_manifest = write_fake_dataset(test_path, [fake_test_sample(index) for index in range(2)])
            result = builder.write_artifacts_from_fit_samples(
                fit_samples=[fake_sample(index) for index in range(10)],
                output_dir=root,
                topology_id="G-1",
                regime="step2c_poly_d8_mult_eps050",
                data_seed=101,
                protocol="screen",
                sample_sizes=[5, 10],
                test_path=test_path,
                test_hash=test_manifest["dataset_hash"],
                topology_template=TOPOLOGY_TEMPLATE,
                generator_config=GENERATOR_CONFIG,
                experiment_version="exp-v1",
                master_label_seed=20260626,
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

    def test_audit_fails_when_test_npz_or_fit_manifest_is_not_real(self):
        builder = load_module(
            "build_nested_fit_validation_bank_missing_test",
            EXPERIMENT_SCRIPTS / "build_nested_fit_validation_bank.py",
        )
        audit = load_module(
            "audit_k18_sample_size_artifacts_missing_test",
            EXPERIMENT_SCRIPTS / "audit_k18_sample_size_artifacts.py",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = builder.write_artifacts_from_fit_samples(
                fit_samples=[fake_sample(index) for index in range(10)],
                output_dir=root,
                topology_id="G-1",
                regime="step2c_poly_d8_mult_eps050",
                data_seed=101,
                protocol="screen",
                sample_sizes=[5],
                test_path=root / "missing_test.npz",
                test_hash="missing-test-hash",
                topology_template=TOPOLOGY_TEMPLATE,
                generator_config=GENERATOR_CONFIG,
                experiment_version="exp-v1",
                master_label_seed=20260626,
            )
            split_manifest_path = Path(result["split_manifest_path"])
            split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
            split_manifest["fit_manifest_path"] = str(root / "missing_fit_manifest.json")
            split_manifest_path.write_text(json.dumps(split_manifest), encoding="utf-8")

            audit_result = audit.audit_artifacts(
                train_bank_path=Path(result["train_bank_path"]),
                split_manifest_path=split_manifest_path,
                eval_manifest_paths=[Path(result["eval_manifest_paths"]["5"])],
            )

        self.assertFalse(audit_result["passed"])
        self.assertIn("test_npz_missing_5", audit_result["failures"])
        self.assertIn("fit_manifest_missing", audit_result["failures"])

    def test_audit_fails_when_test_npz_topology_metadata_mismatches(self):
        builder = load_module(
            "build_nested_fit_validation_bank_bad_test_topology",
            EXPERIMENT_SCRIPTS / "build_nested_fit_validation_bank.py",
        )
        audit = load_module(
            "audit_k18_sample_size_artifacts_bad_test_topology",
            EXPERIMENT_SCRIPTS / "audit_k18_sample_size_artifacts.py",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            test_path = root / "test.npz"
            test_manifest = write_fake_dataset(
                test_path,
                [fake_test_sample(index) for index in range(2)],
                {"topology_id": "G-2"},
            )
            result = builder.write_artifacts_from_fit_samples(
                fit_samples=[fake_sample(index) for index in range(10)],
                output_dir=root,
                topology_id="G-1",
                regime="step2c_poly_d8_mult_eps050",
                data_seed=101,
                protocol="screen",
                sample_sizes=[5],
                test_path=test_path,
                test_hash=test_manifest["dataset_hash"],
                topology_template=TOPOLOGY_TEMPLATE,
                generator_config=GENERATOR_CONFIG,
                experiment_version="exp-v1",
                master_label_seed=20260626,
            )

            audit_result = audit.audit_artifacts(
                train_bank_path=Path(result["train_bank_path"]),
                split_manifest_path=Path(result["split_manifest_path"]),
                eval_manifest_paths=[Path(result["eval_manifest_paths"]["5"])],
            )

        self.assertFalse(audit_result["passed"])
        self.assertIn("test_topology_id_mismatch_5", audit_result["failures"])

    def test_global_audit_catches_test_hash_drift_across_data_seeds(self):
        audit = load_module(
            "audit_k18_sample_size_artifacts_global",
            EXPERIMENT_SCRIPTS / "audit_k18_sample_size_artifacts.py",
        )
        result = audit.audit_plan_jobs(
            [
                {
                    "job_id": "G-1|data_seed=000101|sample_size=005",
                    "topology_id": "G-1",
                    "data_seed": 101,
                    "sample_size": 5,
                    "status": "ready",
                    "test_hash": "test-A",
                    "expected_training_hash": "train-4",
                    "validation_hash": "val-1",
                },
                {
                    "job_id": "G-1|data_seed=000102|sample_size=005",
                    "topology_id": "G-1",
                    "data_seed": 102,
                    "sample_size": 5,
                    "status": "ready",
                    "test_hash": "test-B",
                    "expected_training_hash": "train-4",
                    "validation_hash": "val-1",
                },
            ],
            expected_topology_count=1,
            expected_data_seed_count=2,
            expected_sample_sizes=[5],
            expected_job_count=2,
        )

        self.assertFalse(result["passed"])
        self.assertIn("test_hash_not_shared_for_topology_G-1", result["failures"])

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
                        "topology_id": "G-1",
                        "regime": "step2c_poly_d8_mult_eps050",
                        "protocol": "screen",
                        "data_seed": 101,
                        "trainer_train_size_arg": 4,
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

    def test_run_one_job_rejects_cross_manifest_mismatches(self):
        run_one_job = load_module("run_one_job_mismatch_test", RUN_ONE_JOB_SCRIPT)
        train_bank_manifest = {
            "topology_id": "G-1",
            "regime": "step2c_poly_d8_mult_eps050",
            "protocol": "screen",
            "data_seed": 101,
            "train_seed": 101,
            "bank_hash": "bank-hash",
            "prefix_hashes": {"4": "train-4"},
        }
        base_eval_manifest = {
            "topology_id": "G-1",
            "regime": "step2c_poly_d8_mult_eps050",
            "protocol": "screen",
            "data_seed": 101,
            "sample_size": 5,
            "training_size": 4,
            "validation_size": 1,
            "trainer_train_size_arg": 4,
            "validation_hash": "val-5",
            "test_hash": "test-hash",
        }
        cases = [
            ("training_size", {"training_size": 3}),
            ("trainer_train_size_arg", {"trainer_train_size_arg": 3}),
            ("data_seed", {"data_seed": 102}),
            ("topology_id", {"topology_id": "G-2"}),
            ("regime", {"regime": "other"}),
            ("protocol", {"protocol": "confirm"}),
            ("sample_size_sum", {"validation_size": 2}),
        ]
        for label, patch in cases:
            with self.subTest(label=label):
                eval_manifest = {**base_eval_manifest, **patch}
                with self.assertRaises(ValueError):
                    run_one_job.prepare_paired_job_manifest(
                        topology_id="G-1",
                        regime="step2c_poly_d8_mult_eps050",
                        protocol="screen",
                        train_seed=101,
                        train_size=4,
                        train_bank_manifest=train_bank_manifest,
                        eval_manifest=eval_manifest,
                        output_dir="job",
                        theta_seed=42,
                        gurobi_seed=42,
                        sample_size=5,
                    )


    def test_formal_launcher_builds_execute_commands_and_skips_successful_jobs(self):
        launcher = load_module(
            "launch_formal_k18_sample_size_jobs_test",
            EXPERIMENT_SCRIPTS / "launch_formal_k18_sample_size_jobs.py",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_root = root / "formal"
            row = {
                "job_id": "G-1|data_seed=000101|sample_size=050|training=040|validation=010",
                "topology_id": "G-1",
                "runtime_class": "normal",
                "output_dir": "old/results/jobs/G-1/data_seed=000101/sample_size=050",
                "run_one_job_command": (
                    "/usr/bin/python /repo/surrogate_experiment_results/Step3/scripts/run_one_job.py "
                    "--output-dir old/results/jobs/G-1/data_seed=000101/sample_size=050 "
                    "--max-epochs 3000 --dry-run"
                ),
            }

            planned = launcher.job_from_plan_row(row, output_root=output_root)

            self.assertEqual(planned.queue, "normal")
            self.assertEqual(
                planned.output_dir,
                output_root / "jobs" / "G-1" / "data_seed=000101" / "sample_size=050",
            )
            self.assertIn("--execute", planned.command)
            self.assertNotIn("--dry-run", planned.command)
            self.assertEqual(
                planned.command[planned.command.index("--output-dir") + 1],
                str(planned.output_dir),
            )
            self.assertFalse(launcher.is_job_success(planned))

            planned.output_dir.mkdir(parents=True)
            (planned.output_dir / "job_status.json").write_text(
                json.dumps({"status": "success", "2stage status": "success", "SPO+ status": "success", "evaluation status": "success"}),
                encoding="utf-8",
            )

            self.assertTrue(launcher.is_job_success(planned))

    def test_formal_launcher_splits_normal_and_long_queues(self):
        launcher = load_module(
            "launch_formal_k18_sample_size_jobs_queue_test",
            EXPERIMENT_SCRIPTS / "launch_formal_k18_sample_size_jobs.py",
        )
        rows = [
            {
                "job_id": "normal-job",
                "topology_id": "G-1",
                "runtime_class": "normal",
                "output_dir": "old/G-1/data_seed=000101/sample_size=050",
                "run_one_job_command": "python run_one_job.py --output-dir old/G-1/data_seed=000101/sample_size=050 --dry-run",
            },
            {
                "job_id": "long-job",
                "topology_id": "G-237",
                "runtime_class": "long",
                "output_dir": "old/G-237/data_seed=000101/sample_size=050",
                "run_one_job_command": "python run_one_job.py --output-dir old/G-237/data_seed=000101/sample_size=050 --dry-run",
            },
        ]

        queues = launcher.split_queues(
            [launcher.job_from_plan_row(row, output_root=Path("formal")) for row in rows]
        )

        self.assertEqual([job.job_id for job in queues["normal"]], ["normal-job"])
        self.assertEqual([job.job_id for job in queues["long"]], ["long-job"])


if __name__ == "__main__":
    unittest.main()
