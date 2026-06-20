import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
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


def sample_payload(cpra=0.2):
    return {
        "metadata": {"source_file": "G-test.json"},
        "data": {
            "0": {
                "type": "Pair",
                "patient": {"age": 40, "bloodtype": "A", "cPRA": cpra, "hasBloodCompatibleDonor": True},
                "donors": [{"original_node_id": "0", "dage": 35, "bloodtype": "A"}],
                "matches": [{"recipient": "1", "utility": 60.0, "recipient_cpra": 0.5}],
            },
            "1": {
                "type": "Pair",
                "patient": {"age": 50, "bloodtype": "B", "cPRA": 0.5, "hasBloodCompatibleDonor": False},
                "donors": [{"original_node_id": "1", "dage": 45, "bloodtype": "O"}],
                "matches": [{"recipient": "0", "utility": 70.0, "recipient_cpra": cpra}],
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


class Step3FixedEvalSetTests(unittest.TestCase):
    def test_builds_fixed_validation_and_test_without_train_seed_variation(self):
        module = load_module(SCRIPT, "build_fixed_eval_sets")
        builder = load_module(BUILDER, "build_topology_bank_eval")
        template = builder.build_topology_template("G-test", sample_payload(), max_cycle=3, max_chain=4)
        with tempfile.TemporaryDirectory() as tmp:
            result = module.build_fixed_eval_sets_for_topology(
                topology_template=template,
                base_payload=sample_payload(),
                output_dir=Path(tmp),
                topology_id="G-test",
                regime="step2c_poly_d8_mult_eps050",
                validation_size=2,
                test_size=3,
                experiment_version="v-test",
                master_label_seed=20260619,
                generator_config=config(),
            )

            self.assertTrue(Path(result["validation_path"]).exists())
            self.assertTrue(Path(result["test_path"]).exists())
            self.assertTrue(Path(result["eval_manifest_path"]).exists())
            self.assertEqual(result["validation_hash"], result["manifest"]["validation_hash"])
            self.assertEqual(result["test_hash"], result["manifest"]["test_hash"])
            self.assertNotEqual(result["validation_hash"], result["test_hash"])
            for row in result["manifest"]["validation_samples"] + result["manifest"]["test_samples"]:
                self.assertIsNone(row["train_seed"])
                self.assertIn(row["split_namespace"], {"confirm_validation", "confirm_test"})

    def test_reuses_existing_eval_sets_only_when_manifest_matches(self):
        module = load_module(SCRIPT, "build_fixed_eval_sets")
        builder = load_module(BUILDER, "build_topology_bank_eval_reuse")
        template = builder.build_topology_template("G-test", sample_payload(), max_cycle=3, max_chain=4)
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            first = module.build_fixed_eval_sets_for_topology(
                topology_template=template,
                base_payload=sample_payload(),
                output_dir=output_dir,
                topology_id="G-test",
                regime="step2c_poly_d8_mult_eps050",
                validation_size=1,
                test_size=1,
                experiment_version="v-test",
                master_label_seed=20260619,
                generator_config=config(),
            )
            second = module.build_fixed_eval_sets_for_topology(
                topology_template=template,
                base_payload=sample_payload(),
                output_dir=output_dir,
                topology_id="G-test",
                regime="step2c_poly_d8_mult_eps050",
                validation_size=1,
                test_size=1,
                experiment_version="v-test",
                master_label_seed=20260619,
                generator_config=config(),
            )

            self.assertEqual(first["manifest"], second["manifest"])
            with self.assertRaisesRegex(ValueError, "existing eval set"):
                module.build_fixed_eval_sets_for_topology(
                    topology_template=template,
                    base_payload=sample_payload(),
                    output_dir=output_dir,
                    topology_id="G-test",
                    regime="step2c_poly_d8_mult_eps050",
                    validation_size=2,
                    test_size=1,
                    experiment_version="v-test",
                    master_label_seed=20260619,
                    generator_config=config(),
                )

    def test_dry_run_does_not_write_eval_files(self):
        module = load_module(SCRIPT, "build_fixed_eval_sets")
        builder = load_module(BUILDER, "build_topology_bank_eval_dry")
        template = builder.build_topology_template("G-test", sample_payload(), max_cycle=3, max_chain=4)
        with tempfile.TemporaryDirectory() as tmp:
            result = module.build_fixed_eval_sets_for_topology(
                topology_template=template,
                base_payload=sample_payload(),
                output_dir=Path(tmp),
                topology_id="G-test",
                regime="step2c_poly_d8_mult_eps050",
                validation_size=1,
                test_size=1,
                experiment_version="v-test",
                master_label_seed=20260619,
                generator_config=config(),
                dry_run=True,
            )

            self.assertTrue(result["dry_run"])
            self.assertFalse((Path(tmp) / "validation.npz").exists())
            self.assertFalse((Path(tmp) / "test.npz").exists())


if __name__ == "__main__":
    unittest.main()
