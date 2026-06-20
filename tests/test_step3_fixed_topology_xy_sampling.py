import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
XY_SCRIPT = (
    ROOT
    / "surrogate_experiment_results"
    / "Step3"
    / "scripts"
    / "sample_fixed_topology_xy.py"
)
COMMON_SCRIPT = (
    ROOT
    / "surrogate_experiment_results"
    / "Step3"
    / "scripts"
    / "fixed_topology_xy_common.py"
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
        "metadata": {
            "source_file": "G-test.json",
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


def generator_config():
    return {
        "status": "pilot_not_locked",
        "generator_version": "test-v1",
        "method": "bounded_additive_uniform",
        "recipient_cpra": {"lower": 0.0, "upper": 1.0, "half_width": 0.05},
        "utility": {"lower": 0.0, "upper": 100.0, "half_width": 5.0},
    }


class Step3FixedTopologyXYSamplingTests(unittest.TestCase):
    def test_same_sample_key_is_bitwise_reproducible(self):
        xy = load_module(XY_SCRIPT, "sample_fixed_topology_xy")
        builder = load_module(BUILDER, "build_topology_bank_xy_repro")
        template = builder.build_topology_template("G-test", sample_payload(), max_cycle=3, max_chain=4)

        first = xy.generate_sample(
            topology_template=template,
            base_payload=sample_payload(),
            topology_id="G-test",
            regime="step2c_poly_d8_mult_eps050",
            split_namespace="confirm_train",
            train_seed=17,
            sample_index=0,
            experiment_version="v-test",
            master_label_seed=20260619,
            generator_config=generator_config(),
        )
        second = xy.generate_sample(
            topology_template=template,
            base_payload=sample_payload(),
            topology_id="G-test",
            regime="step2c_poly_d8_mult_eps050",
            split_namespace="confirm_train",
            train_seed=17,
            sample_index=0,
            experiment_version="v-test",
            master_label_seed=20260619,
            generator_config=generator_config(),
        )

        self.assertEqual(first["manifest"], second["manifest"])
        self.assertEqual(first["payload"], second["payload"])
        self.assertEqual(first["X"].tolist(), second["X"].tolist())
        self.assertEqual(first["y"].tolist(), second["y"].tolist())

    def test_different_context_seed_changes_x_hash(self):
        xy = load_module(XY_SCRIPT, "sample_fixed_topology_xy")
        builder = load_module(BUILDER, "build_topology_bank_xy_context")
        template = builder.build_topology_template("G-test", sample_payload(), max_cycle=3, max_chain=4)

        first = xy.generate_sample(
            topology_template=template,
            base_payload=sample_payload(),
            topology_id="G-test",
            regime="step2c_poly_d8_mult_eps050",
            split_namespace="confirm_train",
            train_seed=17,
            sample_index=0,
            experiment_version="v-test",
            master_label_seed=20260619,
            generator_config=generator_config(),
        )
        second = xy.generate_sample(
            topology_template=template,
            base_payload=sample_payload(),
            topology_id="G-test",
            regime="step2c_poly_d8_mult_eps050",
            split_namespace="confirm_train",
            train_seed=17,
            sample_index=1,
            experiment_version="v-test",
            master_label_seed=20260619,
            generator_config=generator_config(),
        )

        self.assertNotEqual(first["manifest"]["context_seed"], second["manifest"]["context_seed"])
        self.assertNotEqual(first["manifest"]["x_hash"], second["manifest"]["x_hash"])

    def test_same_x_different_label_noise_changes_only_label_hash(self):
        xy = load_module(XY_SCRIPT, "sample_fixed_topology_xy")
        builder = load_module(BUILDER, "build_topology_bank_xy_label")
        template = builder.build_topology_template("G-test", sample_payload(), max_cycle=3, max_chain=4)

        first = xy.generate_sample(
            topology_template=template,
            base_payload=sample_payload(),
            topology_id="G-test",
            regime="step2c_poly_d8_mult_eps050",
            split_namespace="confirm_train",
            train_seed=17,
            sample_index=0,
            experiment_version="v-test",
            master_label_seed=20260619,
            generator_config=generator_config(),
            context_seed_override=111,
            label_noise_seed_override=222,
        )
        second = xy.generate_sample(
            topology_template=template,
            base_payload=sample_payload(),
            topology_id="G-test",
            regime="step2c_poly_d8_mult_eps050",
            split_namespace="confirm_train",
            train_seed=17,
            sample_index=0,
            experiment_version="v-test",
            master_label_seed=20260619,
            generator_config=generator_config(),
            context_seed_override=111,
            label_noise_seed_override=333,
        )

        self.assertEqual(first["manifest"]["x_hash"], second["manifest"]["x_hash"])
        self.assertNotEqual(first["manifest"]["label_hash"], second["manifest"]["label_hash"])

    def test_x_hash_uses_model_visible_utility_and_target_node_cpra(self):
        xy = load_module(XY_SCRIPT, "sample_fixed_topology_xy")
        common = load_module(COMMON_SCRIPT, "fixed_topology_xy_common_for_hash_test")
        builder = load_module(BUILDER, "build_topology_bank_xy_hash")
        template = builder.build_topology_template("G-test", sample_payload(), max_cycle=3, max_chain=4)
        sample = xy.generate_sample(
            topology_template=template,
            base_payload=sample_payload(),
            topology_id="G-test",
            regime="step2c_poly_d8_mult_eps050",
            split_namespace="confirm_train",
            train_seed=17,
            sample_index=0,
            experiment_version="v-test",
            master_label_seed=20260619,
            generator_config=generator_config(),
        )

        matrix = common.feature_matrix_from_payload(sample["payload"], template)

        self.assertEqual(sample["X"].tolist(), matrix.tolist())
        self.assertEqual(sample["manifest"]["x_hash"], common.x_hash(sample["payload"], template))


if __name__ == "__main__":
    unittest.main()
