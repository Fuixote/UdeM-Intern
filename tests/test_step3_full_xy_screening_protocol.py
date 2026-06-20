import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "surrogate_experiment_results" / "Step3" / "scripts"
COMMON_SCRIPT = SCRIPT_DIR / "fixed_topology_xy_common.py"
BUILDER_SCRIPT = SCRIPT_DIR / "build_topology_bank.py"
BANK_SCRIPT = SCRIPT_DIR / "build_nested_train_bank.py"
EVAL_SCRIPT = SCRIPT_DIR / "build_fixed_eval_sets.py"
AUDIT_SCRIPT = SCRIPT_DIR / "audit_fixed_topology_xy.py"
JOB_SCRIPT = SCRIPT_DIR / "run_one_job.py"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def payload():
    return {
        "metadata": {
            "source_file": "G-screen.json",
            "step2c_degree": 8,
            "step2c_kappa": 3.0,
            "step2c_delta": 1e-12,
            "clean_linear_utility_weight": 10.0,
            "clean_linear_cpra_weight": 5.0,
        },
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


def config(status="pilot_not_locked"):
    return {
        "status": status,
        "generator_version": "screen-test-v1",
        "method": "bounded_additive_uniform",
        "recipient_cpra": {"lower": 0.0, "upper": 1.0, "half_width": 0.05},
        "utility": {"lower": 0.0, "upper": 100.0, "half_width": 5.0},
        "label": {"epsilon_bar": 0.5},
    }


def template():
    builder = load_module(BUILDER_SCRIPT, "screen_protocol_builder")
    return builder.build_topology_template("G-screen", payload(), max_cycle=3, max_chain=4)


def build_artifacts(root: Path, *, protocol: str = "screen"):
    bank_mod = load_module(BANK_SCRIPT, f"screen_protocol_bank_{protocol}_{id(root)}")
    eval_mod = load_module(EVAL_SCRIPT, f"screen_protocol_eval_{protocol}_{id(root)}")
    topo = template()
    bank_path = root / protocol / "train_seed=000001.npz"
    bank_manifest = bank_mod.build_nested_train_bank(
        topology_template=topo,
        base_payload=payload(),
        output_path=bank_path,
        topology_id="G-screen",
        regime="step2c_poly_d8_mult_eps050",
        train_seed=1,
        max_train_size=5,
        prefix_sizes=(2, 3, 5),
        experiment_version="v-screen",
        master_label_seed=20260619,
        generator_config=config(),
        protocol=protocol,
    )
    eval_result = eval_mod.build_fixed_eval_sets_for_topology(
        topology_template=topo,
        base_payload=payload(),
        output_dir=root / protocol / "eval",
        topology_id="G-screen",
        regime="step2c_poly_d8_mult_eps050",
        validation_size=2,
        test_size=2,
        experiment_version="v-screen",
        master_label_seed=20260619,
        generator_config=config(),
        protocol=protocol,
    )
    return topo, bank_path, bank_manifest, eval_result


class Step3FullXYScreeningProtocolTests(unittest.TestCase):
    def test_build_nested_train_bank_protocol_screen_produces_screen_train_samples(self):
        common = load_module(COMMON_SCRIPT, "screen_protocol_common_bank")
        with tempfile.TemporaryDirectory() as tmp:
            _topo, bank_path, manifest, _eval_result = build_artifacts(Path(tmp), protocol="screen")
            dataset = common.read_npz_dataset(bank_path)

            self.assertEqual(manifest["protocol"], "screen")
            self.assertEqual(manifest["split_namespace"], "screen_train")
            self.assertEqual(dataset["manifest"]["protocol"], "screen")
            self.assertEqual(
                {row["split_namespace"] for row in dataset["manifest"]["samples"]},
                {"screen_train"},
            )

    def test_build_fixed_eval_sets_protocol_screen_records_screen_namespaces_and_no_train_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            _topo, _bank_path, _manifest, eval_result = build_artifacts(Path(tmp), protocol="screen")
            manifest = eval_result["manifest"]

            self.assertEqual(manifest["protocol"], "screen")
            self.assertEqual(manifest["validation_namespace"], "screen_validation")
            self.assertEqual(manifest["test_namespace"], "screen_test")
            self.assertEqual(manifest["train_seed_sentinel"], "EVAL")
            self.assertEqual({row["split_namespace"] for row in manifest["validation_samples"]}, {"screen_validation"})
            self.assertEqual({row["split_namespace"] for row in manifest["test_samples"]}, {"screen_test"})
            self.assertTrue(all(row["train_seed"] is None for row in manifest["validation_samples"]))
            self.assertTrue(all(row["train_seed"] is None for row in manifest["test_samples"]))

    def test_audit_protocol_screen_passes_and_protocol_confirm_rejects_screen_artifacts(self):
        audit = load_module(AUDIT_SCRIPT, "screen_protocol_audit_pass_reject")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            topo, bank_path, _manifest, eval_result = build_artifacts(root, protocol="screen")

            screen_result = audit.audit_fixed_topology_xy(
                train_bank_path=bank_path,
                eval_manifest_path=eval_result["eval_manifest_path"],
                topology_template=topo,
                base_payload=payload(),
                generator_config=config(),
                protocol="screen",
            )
            self.assertTrue(screen_result["passed"], screen_result["failures"])

            confirm_result = audit.audit_fixed_topology_xy(
                train_bank_path=bank_path,
                eval_manifest_path=eval_result["eval_manifest_path"],
                topology_template=topo,
                base_payload=payload(),
                generator_config=config(),
                protocol="confirm",
            )
            self.assertFalse(confirm_result["passed"])
            self.assertIn("train_bank_namespace_invalid", confirm_result["failures"])
            self.assertIn("validation_namespace_invalid", confirm_result["failures"])
            self.assertIn("test_namespace_invalid", confirm_result["failures"])

    def test_audit_protocol_screen_rejects_mixed_screen_confirm_namespaces(self):
        audit = load_module(AUDIT_SCRIPT, "screen_protocol_audit_mixed")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            topo, screen_bank_path, _manifest, _eval_result = build_artifacts(root / "screen", protocol="screen")
            _confirm_topo, _confirm_bank_path, _confirm_manifest, confirm_eval = build_artifacts(
                root / "confirm",
                protocol="confirm",
            )

            result = audit.audit_fixed_topology_xy(
                train_bank_path=screen_bank_path,
                eval_manifest_path=confirm_eval["eval_manifest_path"],
                topology_template=topo,
                base_payload=payload(),
                generator_config=config(),
                protocol="screen",
            )

            self.assertFalse(result["passed"])
            self.assertIn("validation_namespace_invalid", result["failures"])
            self.assertIn("test_namespace_invalid", result["failures"])
            self.assertIn("protocol_namespace_mismatch", result["failures"])

    def test_screening_audit_allows_pilot_not_locked_but_confirmation_lock_still_required(self):
        audit = load_module(AUDIT_SCRIPT, "screen_protocol_lock")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            topo, bank_path, _manifest, eval_result = build_artifacts(root, protocol="screen")

            screen_result = audit.audit_fixed_topology_xy(
                train_bank_path=bank_path,
                eval_manifest_path=eval_result["eval_manifest_path"],
                topology_template=topo,
                base_payload=payload(),
                generator_config=config("pilot_not_locked"),
                protocol="screen",
            )
            self.assertTrue(screen_result["passed"], screen_result["failures"])

            with self.assertRaisesRegex(ValueError, "locked"):
                audit.assert_formal_config_locked(config("pilot_not_locked"))

    def test_run_one_job_records_protocol_in_manifest_without_changing_commands(self):
        job_mod = load_module(JOB_SCRIPT, "screen_protocol_run_one_job")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _topo, bank_path, _manifest, eval_result = build_artifacts(root, protocol="screen")
            output_dir = root / "job"

            rc = job_mod.main(
                [
                    "--train-bank", str(bank_path),
                    "--eval-manifest", eval_result["eval_manifest_path"],
                    "--topology-id", "G-screen",
                    "--regime", "step2c_poly_d8_mult_eps050",
                    "--train-seed", "1",
                    "--train-size", "2",
                    "--output-dir", str(output_dir),
                    "--max-epochs", "2",
                    "--protocol", "screen",
                    "--dry-run",
                ]
            )

            self.assertEqual(rc, 0)
            paired_manifest = json.loads((output_dir / "paired_job_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(paired_manifest["protocol"], "screen")
            self.assertIn("train_2stage_fixed_topology.py", paired_manifest["commands"]["2stage"][1])
            self.assertIn("train_spoplus_fixed_topology.py", paired_manifest["commands"]["SPO+"][1])


if __name__ == "__main__":
    unittest.main()
