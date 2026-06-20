import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCRIPT = (
    ROOT
    / "surrogate_experiment_results"
    / "Step3"
    / "scripts"
    / "audit_fixed_topology_xy.py"
)
CONFIRMATION_SCRIPT = (
    ROOT
    / "surrogate_experiment_results"
    / "Step3"
    / "scripts"
    / "run_confirmation.py"
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


def config(status="pilot_not_locked"):
    return {
        "status": status,
        "generator_version": "test-v1",
        "method": "bounded_additive_uniform",
        "recipient_cpra": {"lower": 0.0, "upper": 1.0, "half_width": 0.05},
        "utility": {"lower": 0.0, "upper": 100.0, "half_width": 5.0},
    }


class Step3FixedTopologyXYAuditTests(unittest.TestCase):
    def test_audits_bank_eval_namespaces_prefixes_and_config_hashes(self):
        audit = load_module(AUDIT_SCRIPT, "audit_fixed_topology_xy")
        bank_mod = load_module(BANK_SCRIPT, "build_nested_train_bank_for_audit")
        eval_mod = load_module(EVAL_SCRIPT, "build_fixed_eval_sets_for_audit")
        builder = load_module(BUILDER, "build_topology_bank_audit")
        template = builder.build_topology_template("G-test", payload(), max_cycle=3, max_chain=4)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bank_path = root / "train_seed=000017.npz"
            bank_mod.build_nested_train_bank(
                topology_template=template,
                base_payload=payload(),
                output_path=bank_path,
                topology_id="G-test",
                regime="step2c_poly_d8_mult_eps050",
                train_seed=17,
                max_train_size=4,
                prefix_sizes=(1, 2, 4),
                experiment_version="v-test",
                master_label_seed=20260619,
                generator_config=config(),
            )
            eval_mod.build_fixed_eval_sets_for_topology(
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

            result = audit.audit_fixed_topology_xy(
                train_bank_path=bank_path,
                eval_manifest_path=root / "eval" / "eval_manifest.json",
                topology_template=template,
                base_payload=payload(),
                generator_config=config(),
            )

            self.assertTrue(result["passed"])
            self.assertEqual(result["failures"], [])

    def test_formal_confirmation_requires_locked_generator_config(self):
        audit = load_module(AUDIT_SCRIPT, "audit_fixed_topology_xy_lock")

        with self.assertRaisesRegex(ValueError, "locked"):
            audit.assert_formal_config_locked(config(status="pilot_not_locked"))

        audit.assert_formal_config_locked(config(status="locked"))

    def test_audit_reports_generator_config_hash_mismatch(self):
        audit = load_module(AUDIT_SCRIPT, "audit_fixed_topology_xy_hash_mismatch")
        bank_mod = load_module(BANK_SCRIPT, "build_nested_train_bank_for_audit_hash")
        builder = load_module(BUILDER, "build_topology_bank_audit_hash")
        template = builder.build_topology_template("G-test", payload(), max_cycle=3, max_chain=4)
        with tempfile.TemporaryDirectory() as tmp:
            bank_path = Path(tmp) / "bank.npz"
            bank_mod.build_nested_train_bank(
                topology_template=template,
                base_payload=payload(),
                output_path=bank_path,
                topology_id="G-test",
                regime="step2c_poly_d8_mult_eps050",
                train_seed=17,
                max_train_size=2,
                prefix_sizes=(1, 2),
                experiment_version="v-test",
                master_label_seed=20260619,
                generator_config=config(),
            )
            changed = config()
            changed["utility"]["half_width"] = 10.0

            result = audit.audit_train_bank(
                train_bank_path=bank_path,
                topology_template=template,
                base_payload=payload(),
                generator_config=changed,
            )

            self.assertFalse(result["passed"])
            self.assertIn("generator_config_hash_mismatch", result["failures"])

    def test_confirmation_validator_refuses_unlocked_context_generator(self):
        module = load_module(CONFIRMATION_SCRIPT, "run_confirmation_lock_test")

        with self.assertRaisesRegex(ValueError, "locked"):
            module.validate_confirmation_config(
                {"execution": {"dry_run": True}},
                generator_config=config(status="pilot_not_locked"),
            )

        module.validate_confirmation_config(
            {"execution": {"dry_run": True}},
            generator_config=config(status="locked"),
        )


if __name__ == "__main__":
    unittest.main()
