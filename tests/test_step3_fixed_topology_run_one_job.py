import importlib.util
import tempfile
import unittest
from pathlib import Path


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


class Step3FixedTopologyRunOneJobTests(unittest.TestCase):
    def test_dry_run_manifest_pairs_methods_on_same_prefix_eval_and_theta(self):
        job_mod = load_module(JOB_SCRIPT, "run_one_job")
        bank_mod = load_module(BANK_SCRIPT, "build_nested_train_bank_for_job")
        eval_mod = load_module(EVAL_SCRIPT, "build_fixed_eval_sets_for_job")
        builder = load_module(BUILDER, "build_topology_bank_job")
        template = builder.build_topology_template("G-test", payload(), max_cycle=3, max_chain=4)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
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


if __name__ == "__main__":
    unittest.main()
