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
    / "materialize_phase_b_step2c_datasets.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "materialize_phase_b_step2c_datasets",
        SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample_payload():
    return {
        "metadata": {
            "source_file": "G-1.json",
            "ground_truth_label_mode": "step2c_polynomial_degree_multiplicative_noise",
            "step2c_degree": 8,
            "step2c_kappa": 3.0,
            "step2c_delta": 1e-12,
            "step2c_epsilon_bar": 0.5,
            "clean_linear_utility_weight": 10.0,
            "clean_linear_cpra_weight": 5.0,
        },
        "data": {
            "0": {
                "type": "Pair",
                "matches": [
                    {"recipient": "1", "utility": 80.0, "recipient_cpra": 0.2},
                    {"recipient": "2", "utility": 50.0, "recipient_cpra": 0.6},
                ],
            },
            "1": {
                "type": "Pair",
                "matches": [{"recipient": "0", "utility": 70.0, "recipient_cpra": 0.4}],
            },
            "2": {"type": "NDD", "matches": [{"recipient": "0", "utility": 40.0, "recipient_cpra": 0.1}]},
        },
    }


class Step3PhaseBMaterializationTests(unittest.TestCase):
    def test_training_size_budget_splits_four_to_one(self):
        module = load_module()

        self.assertEqual(module.phase_b_train_validation_counts(50), (40, 10))
        self.assertEqual(module.phase_b_train_validation_counts(100), (80, 20))
        self.assertEqual(module.phase_b_train_validation_counts(500), (400, 100))

    def test_label_seed_namespaces_make_test_unseen(self):
        module = load_module()

        train_seed = module.phase_b_label_seed("G-47", "train", sample_idx=0, train_seed=1)
        same_train_seed = module.phase_b_label_seed("G-47", "train", sample_idx=0, train_seed=1)
        different_train_seed = module.phase_b_label_seed("G-47", "train", sample_idx=0, train_seed=2)
        validation_seed = module.phase_b_label_seed("G-47", "validation", sample_idx=0)
        test_seed = module.phase_b_label_seed("G-47", "test", sample_idx=0)
        phase_c_train_seed = module.phase_b_label_seed("G-47", "train", sample_idx=0, train_seed=1000)
        next_topology_train_seed = module.phase_b_label_seed("G-48", "train", sample_idx=0, train_seed=1)

        self.assertEqual(train_seed, same_train_seed)
        self.assertEqual(
            len(
                {
                    train_seed,
                    different_train_seed,
                    validation_seed,
                    test_seed,
                    phase_c_train_seed,
                    next_topology_train_seed,
                }
            ),
            6,
        )
        self.assertGreater(test_seed, validation_seed)

    def test_materializes_fixed_topology_train_validation_test_tree(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_graph = root / "source" / "G-1.json"
            source_graph.parent.mkdir()
            source_graph.write_text(json.dumps(sample_payload()), encoding="utf-8")
            template = root / "topologies" / "G-1" / "template.json"
            template.parent.mkdir(parents=True)
            template.write_text(
                json.dumps(
                    {
                        "topology_id": "G-1",
                        "topology_hash": "topology-bank-hash",
                        "feasible_set_hash": "feasible-set-hash",
                        "arc_order_hash": "arc-order-hash",
                        "num_exchange_candidates": 3,
                    }
                ),
                encoding="utf-8",
            )
            output_dir = root / "phase_b" / "datasets" / "G-1"

            manifest = module.materialize_topology_dataset(
                topology_id="G-1",
                source_graph_path=source_graph,
                output_dir=output_dir,
                topology_template_path=template,
                train_seed_start=1,
                train_seed_count=2,
                training_size_budget=5,
                test_size=3,
                epsilon_bar=0.5,
            )

            self.assertEqual(manifest["train_sample_count"], 4)
            self.assertEqual(manifest["validation_sample_count"], 1)
            self.assertEqual(manifest["test_sample_count"], 3)
            self.assertEqual(manifest["topology_bank_hash"], "topology-bank-hash")
            self.assertEqual(manifest["feasible_set_hash"], "feasible-set-hash")
            self.assertEqual(len(list((output_dir / "validation").glob("G-*.json"))), 1)
            self.assertEqual(len(list((output_dir / "test").glob("G-*.json"))), 3)
            self.assertEqual(len(list((output_dir / "train_seed=000001" / "train").glob("G-*.json"))), 4)
            self.assertEqual(len(list((output_dir / "train_seed=000002" / "train").glob("G-*.json"))), 4)

            first_train = json.loads(
                (output_dir / "train_seed=000001" / "train" / "G-000000.json").read_text(
                    encoding="utf-8"
                )
            )
            first_test = json.loads(
                (output_dir / "test" / "G-000000.json").read_text(encoding="utf-8")
            )

            self.assertEqual(
                first_train["metadata"]["step3_phase_b"]["split"],
                "train",
            )
            self.assertEqual(
                first_train["metadata"]["step3_phase_b"]["train_seed"],
                1,
            )
            self.assertEqual(
                first_test["metadata"]["step3_phase_b"]["split"],
                "test",
            )
            self.assertIsNone(first_test["metadata"]["step3_phase_b"]["train_seed"])
            self.assertNotEqual(
                first_train["metadata"]["fixed_topology_relabel_seed"],
                first_test["metadata"]["fixed_topology_relabel_seed"],
            )


if __name__ == "__main__":
    unittest.main()
