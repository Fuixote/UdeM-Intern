import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "surrogate_experiment_results"
    / "Step3"
    / "scripts"
    / "build_nested_train_bank.py"
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


def sample_payload():
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


class Step3NestedTrainBankTests(unittest.TestCase):
    def test_builds_one_bank_and_records_strict_prefix_hashes(self):
        module = load_module(SCRIPT, "build_nested_train_bank")
        builder = load_module(BUILDER, "build_topology_bank_nested")
        template = builder.build_topology_template("G-test", sample_payload(), max_cycle=3, max_chain=4)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "train_seed=000017.npz"

            manifest = module.build_nested_train_bank(
                topology_template=template,
                base_payload=sample_payload(),
                output_path=output,
                topology_id="G-test",
                regime="step2c_poly_d8_mult_eps050",
                train_seed=17,
                max_train_size=5,
                prefix_sizes=(2, 3, 5),
                experiment_version="v-test",
                master_label_seed=20260619,
                generator_config=config(),
            )

            self.assertTrue(output.exists())
            self.assertEqual(manifest["max_train_size"], 5)
            self.assertEqual(manifest["prefix_sizes"], [2, 3, 5])
            self.assertEqual(manifest["D50"], "bank[0:2]")
            self.assertEqual(manifest["D100"], "bank[0:3]")
            self.assertEqual(manifest["D500"], "bank[0:5]")
            self.assertNotEqual(manifest["prefix_hashes"]["2"], manifest["prefix_hashes"]["3"])
            self.assertNotEqual(manifest["prefix_hashes"]["3"], manifest["bank_hash"])

            bank = module.load_train_bank(output)
            self.assertEqual(len(bank["manifest"]["samples"]), 5)
            self.assertTrue(module.verify_nested_prefixes(bank["manifest"]))

    def test_rejects_non_nested_prefix_sizes(self):
        module = load_module(SCRIPT, "build_nested_train_bank")
        builder = load_module(BUILDER, "build_topology_bank_nested_reject")
        template = builder.build_topology_template("G-test", sample_payload(), max_cycle=3, max_chain=4)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "prefix_sizes"):
                module.build_nested_train_bank(
                    topology_template=template,
                    base_payload=sample_payload(),
                    output_path=Path(tmp) / "bad.npz",
                    topology_id="G-test",
                    regime="step2c_poly_d8_mult_eps050",
                    train_seed=17,
                    max_train_size=5,
                    prefix_sizes=(3, 2, 5),
                    experiment_version="v-test",
                    master_label_seed=20260619,
                    generator_config=config(),
                )

    def test_materialized_manifest_contains_serializable_samples(self):
        module = load_module(SCRIPT, "build_nested_train_bank")
        builder = load_module(BUILDER, "build_topology_bank_nested_json")
        template = builder.build_topology_template("G-test", sample_payload(), max_cycle=3, max_chain=4)
        with tempfile.TemporaryDirectory() as tmp:
            manifest = module.build_nested_train_bank(
                topology_template=template,
                base_payload=sample_payload(),
                output_path=Path(tmp) / "bank.npz",
                topology_id="G-test",
                regime="step2c_poly_d8_mult_eps050",
                train_seed=17,
                max_train_size=3,
                prefix_sizes=(1, 2, 3),
                experiment_version="v-test",
                master_label_seed=20260619,
                generator_config=config(),
            )

            json.dumps(manifest)
            self.assertEqual([row["sample_index"] for row in manifest["samples"]], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
