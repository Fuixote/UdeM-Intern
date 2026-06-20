import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "surrogate_experiment_results" / "Step3" / "scripts"
COMMON_SCRIPT = SCRIPT_DIR / "fixed_topology_xy_common.py"
BUILDER_SCRIPT = SCRIPT_DIR / "build_topology_bank.py"
BANK_SCRIPT = SCRIPT_DIR / "build_nested_train_bank.py"
EVAL_SCRIPT = SCRIPT_DIR / "build_fixed_eval_sets.py"
AUDIT_SCRIPT = SCRIPT_DIR / "audit_fixed_topology_xy.py"
JOB_SCRIPT = SCRIPT_DIR / "run_one_job.py"
CONFIRMATION_SCRIPT = SCRIPT_DIR / "run_confirmation.py"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tiny_payload():
    return {
        "metadata": {
            "source_file": "G-smoke.json",
            "step2c_degree": 8,
            "step2c_kappa": 3.0,
            "step2c_delta": 1e-12,
            "clean_linear_utility_weight": 10.0,
            "clean_linear_cpra_weight": 5.0,
        },
        "data": {
            "0": {
                "type": "Pair",
                "patient": {
                    "age": 40,
                    "bloodtype": "A",
                    "cPRA": 0.2,
                    "hasBloodCompatibleDonor": True,
                },
                "donors": [{"original_node_id": "0", "dage": 35, "bloodtype": "A"}],
                "matches": [{"recipient": "1", "utility": 60.0, "recipient_cpra": 0.5}],
            },
            "1": {
                "type": "Pair",
                "patient": {
                    "age": 50,
                    "bloodtype": "B",
                    "cPRA": 0.5,
                    "hasBloodCompatibleDonor": False,
                },
                "donors": [{"original_node_id": "1", "dage": 45, "bloodtype": "O"}],
                "matches": [{"recipient": "0", "utility": 70.0, "recipient_cpra": 0.2}],
            },
            "2": {
                "type": "NDD",
                "donor": {"original_node_id": "2", "dage": 55, "bloodtype": "O"},
                "matches": [
                    {"recipient": "0", "utility": 40.0, "recipient_cpra": 0.2},
                    {"recipient": "1", "utility": 50.0, "recipient_cpra": 0.5},
                ],
            },
        },
    }


def pilot_config():
    return {
        "status": "pilot_not_locked",
        "generator_version": "smoke_test_context_v1",
        "method": "bounded_additive_uniform",
        "recipient_cpra": {"lower": 0.0, "upper": 1.0, "half_width": 0.05},
        "utility": {"lower": 0.0, "upper": 100.0, "half_width": 5.0},
        "label": {"epsilon_bar": 0.5},
    }


def locked_config():
    config = copy.deepcopy(pilot_config())
    config["status"] = "locked"
    return config


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_config(path, config):
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_artifacts(root: Path, train_seed: int = 1):
    builder = load_module(BUILDER_SCRIPT, f"smoke_builder_{id(root)}_{train_seed}")
    bank_mod = load_module(BANK_SCRIPT, f"smoke_bank_{id(root)}_{train_seed}")
    eval_mod = load_module(EVAL_SCRIPT, f"smoke_eval_{id(root)}_{train_seed}")
    payload = tiny_payload()
    template = builder.build_topology_template("G-smoke", payload, max_cycle=3, max_chain=4)
    template_path = root / "G-smoke.template.json"
    payload_path = root / "G-smoke.base.json"
    config_path = root / "context_config.json"
    write_json(template_path, template)
    write_json(payload_path, payload)
    write_config(config_path, pilot_config())

    bank_path = root / "train_banks" / f"train_seed={train_seed:06d}.npz"
    bank_manifest = bank_mod.build_nested_train_bank(
        topology_template=template,
        base_payload=payload,
        output_path=bank_path,
        topology_id="G-smoke",
        regime="step2c_poly_d8_mult_eps050",
        train_seed=train_seed,
        max_train_size=5,
        prefix_sizes=(2, 3, 5),
        experiment_version="v-smoke",
        master_label_seed=20260619,
        generator_config=pilot_config(),
    )
    eval_result = eval_mod.build_fixed_eval_sets_for_topology(
        topology_template=template,
        base_payload=payload,
        output_dir=root / "eval",
        topology_id="G-smoke",
        regime="step2c_poly_d8_mult_eps050",
        validation_size=2,
        test_size=2,
        experiment_version="v-smoke",
        master_label_seed=20260619,
        generator_config=pilot_config(),
    )
    return {
        "payload": payload,
        "template": template,
        "template_path": template_path,
        "payload_path": payload_path,
        "config_path": config_path,
        "bank_path": bank_path,
        "bank_manifest": bank_manifest,
        "eval_result": eval_result,
    }


def rewrite_dataset_with_payloads(common, dataset_path: Path, payloads):
    dataset = common.read_npz_dataset(dataset_path)
    samples = []
    for index, payload in enumerate(payloads):
        samples.append(
            {
                "X": dataset["X"][index],
                "y": dataset["y"][index],
                "payload": payload,
                "manifest": dataset["sample_manifests"][index],
            }
        )
    common.write_npz_dataset(dataset_path, samples=samples, manifest=dataset["manifest"])


class Step3FixedTopologyTinySmokeTests(unittest.TestCase):
    def test_tiny_full_xy_smoke_audit_corruption_dry_run_execute_and_plan(self):
        common = load_module(COMMON_SCRIPT, "smoke_common")
        audit_mod = load_module(AUDIT_SCRIPT, "smoke_audit")
        job_mod = load_module(JOB_SCRIPT, "smoke_run_one_job")
        confirmation_mod = load_module(CONFIRMATION_SCRIPT, "smoke_run_confirmation")
        bank_mod = load_module(BANK_SCRIPT, "smoke_bank_seed2")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = build_artifacts(root, train_seed=1)
            template = artifacts["template"]
            payload = artifacts["payload"]
            bank_path = artifacts["bank_path"]
            eval_manifest_path = Path(artifacts["eval_result"]["eval_manifest_path"])

            result = audit_mod.audit_fixed_topology_xy(
                train_bank_path=bank_path,
                eval_manifest_path=eval_manifest_path,
                topology_template=template,
                base_payload=payload,
                generator_config=pilot_config(),
            )
            self.assertTrue(result["passed"], result["failures"])

            manifest = json.loads(eval_manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(all(row["train_seed"] is None for row in manifest["validation_samples"]))
            self.assertTrue(all(row["train_seed"] is None for row in manifest["test_samples"]))

            corrupt_manifest = copy.deepcopy(manifest)
            corrupt_manifest["validation_hash"] = "bad-validation-hash"
            corrupt_manifest_path = root / "eval" / "bad_eval_manifest.json"
            write_json(corrupt_manifest_path, corrupt_manifest)
            corrupt_result = audit_mod.audit_fixed_topology_xy(
                train_bank_path=bank_path,
                eval_manifest_path=corrupt_manifest_path,
                topology_template=template,
                base_payload=payload,
                generator_config=pilot_config(),
            )
            self.assertFalse(corrupt_result["passed"])
            self.assertIn("validation_dataset_hash_mismatch", corrupt_result["failures"])

            broken_eval_dir = root / "broken_eval"
            broken_eval_dir.mkdir()
            validation_copy = broken_eval_dir / "validation.npz"
            test_copy = broken_eval_dir / "test.npz"
            validation_copy.write_bytes(Path(manifest["validation_path"]).read_bytes())
            test_copy.write_bytes(Path(manifest["test_path"]).read_bytes())
            broken_manifest = copy.deepcopy(manifest)
            broken_manifest["validation_path"] = str(validation_copy)
            broken_manifest["test_path"] = str(test_copy)
            broken_manifest_path = broken_eval_dir / "eval_manifest.json"
            write_json(broken_manifest_path, broken_manifest)
            validation_dataset = common.read_npz_dataset(validation_copy)
            broken_payloads = copy.deepcopy(validation_dataset["payloads"])
            broken_payloads[0]["data"]["0"]["patient"]["cPRA"] = 0.999
            rewrite_dataset_with_payloads(common, validation_copy, broken_payloads)
            broken_result = audit_mod.audit_fixed_topology_xy(
                train_bank_path=bank_path,
                eval_manifest_path=broken_manifest_path,
                topology_template=template,
                base_payload=payload,
                generator_config=pilot_config(),
            )
            self.assertFalse(broken_result["passed"])
            self.assertIn("validation_context_label_inconsistent", broken_result["failures"])

            job_dir = root / "job"
            rc = job_mod.main(
                [
                    "--train-bank", str(bank_path),
                    "--eval-manifest", str(eval_manifest_path),
                    "--topology-id", "G-smoke",
                    "--regime", "step2c_poly_d8_mult_eps050",
                    "--train-seed", "1",
                    "--train-size", "2",
                    "--output-dir", str(job_dir),
                    "--max-epochs", "2",
                    "--dry-run",
                ]
            )
            self.assertEqual(rc, 0)
            paired_manifest = json.loads((job_dir / "paired_job_manifest.json").read_text(encoding="utf-8"))
            self.assertIn("2stage", paired_manifest["commands"])
            self.assertIn("SPO+", paired_manifest["commands"])
            self.assertIn("evaluation", paired_manifest["commands"])
            self.assertEqual(
                paired_manifest["methods"]["2stage"]["train_prefix_hash"],
                paired_manifest["methods"]["SPO+"]["train_prefix_hash"],
            )
            self.assertEqual(
                paired_manifest["methods"]["2stage"]["validation_hash"],
                paired_manifest["methods"]["SPO+"]["validation_hash"],
            )
            self.assertEqual(
                paired_manifest["methods"]["2stage"]["theta_init"],
                paired_manifest["methods"]["SPO+"]["theta_init"],
            )
            self.assertEqual(paired_manifest["test_hash"], artifacts["eval_result"]["test_hash"])

            execute_dir = root / "job_execute"
            completed = mock.Mock(returncode=0, stdout="", stderr="")
            with mock.patch.object(job_mod.subprocess, "run", return_value=completed) as run_mock:
                execute_rc = job_mod.main(
                    [
                        "--train-bank", str(bank_path),
                        "--eval-manifest", str(eval_manifest_path),
                        "--topology-id", "G-smoke",
                        "--regime", "step2c_poly_d8_mult_eps050",
                        "--train-seed", "1",
                        "--train-size", "2",
                        "--output-dir", str(execute_dir),
                        "--max-epochs", "2",
                        "--execute",
                    ]
                )
            self.assertEqual(execute_rc, 0)
            self.assertEqual(run_mock.call_count, 3)
            commands = [call.args[0] for call in run_mock.call_args_list]
            self.assertTrue(commands[0][1].endswith("train_2stage_fixed_topology.py"))
            self.assertTrue(commands[1][1].endswith("train_spoplus_fixed_topology.py"))
            self.assertTrue(commands[2][1].endswith("evaluate_fixed_topology.py"))
            status = json.loads((execute_dir / "job_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "success")

            fail_dir = root / "job_execute_failure"
            failed_spoplus = mock.Mock(returncode=9, stdout="", stderr="mock gurobi failure")
            with mock.patch.object(job_mod.subprocess, "run", side_effect=[completed, failed_spoplus]) as fail_mock:
                fail_rc = job_mod.main(
                    [
                        "--train-bank", str(bank_path),
                        "--eval-manifest", str(eval_manifest_path),
                        "--topology-id", "G-smoke",
                        "--regime", "step2c_poly_d8_mult_eps050",
                        "--train-seed", "1",
                        "--train-size", "2",
                        "--output-dir", str(fail_dir),
                        "--max-epochs", "2",
                        "--execute",
                    ]
                )
            self.assertEqual(fail_rc, 9)
            self.assertEqual(fail_mock.call_count, 2)
            failure_status = json.loads((fail_dir / "job_status.json").read_text(encoding="utf-8"))
            self.assertEqual(failure_status["status"], "failed")
            self.assertEqual(failure_status["SPO+ status"], "failed")
            self.assertEqual(failure_status["evaluation status"], "pending")
            self.assertIn("mock gurobi failure", failure_status["failure_reason"])

            seed2_bank_path = root / "train_banks" / "train_seed=000002.npz"
            bank_mod.build_nested_train_bank(
                topology_template=template,
                base_payload=payload,
                output_path=seed2_bank_path,
                topology_id="G-smoke",
                regime="step2c_poly_d8_mult_eps050",
                train_seed=2,
                max_train_size=5,
                prefix_sizes=(2, 3, 5),
                experiment_version="v-smoke",
                master_label_seed=20260619,
                generator_config=pilot_config(),
            )
            confirmation_config = {
                "regimes": ["step2c_poly_d8_mult_eps050"],
                "topologies": [
                    {
                        "topology_id": "G-smoke",
                        "train_bank_dir": str(root / "train_banks"),
                        "eval_manifest": str(eval_manifest_path),
                        "output_dir": str(root / "planned_runs"),
                    }
                ],
                "training": {
                    "nested_train_sets": True,
                    "train_seed_start": 1,
                    "train_seed_count": 2,
                    "train_sizes": [2, 3],
                },
            }
            plan = confirmation_mod.build_confirmation_job_plan(
                confirmation_config,
                generator_config=locked_config(),
            )
            self.assertEqual(plan["job_count"], 4)
            for job in plan["jobs"]:
                self.assertEqual(job["validation_hash"], artifacts["eval_result"]["validation_hash"])
                self.assertEqual(job["test_hash"], artifacts["eval_result"]["test_hash"])
                self.assertIsNotNone(job["expected_train_prefix_hash"])
                self.assertIn("train_bank_path", job)
                self.assertIn("eval_manifest_path", job)
                self.assertIn("output_dir", job)
            with self.assertRaisesRegex(ValueError, "status='locked'"):
                confirmation_mod.build_confirmation_job_plan(
                    confirmation_config,
                    generator_config=pilot_config(),
                )


if __name__ == "__main__":
    unittest.main()
