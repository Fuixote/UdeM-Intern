import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "surrogate_experiment_results"
    / "decision_analysis"
    / "scripts"
    / "audit_fixed_topology_label_seed.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "audit_fixed_topology_label_seed",
        SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample_payload():
    return {
        "metadata": {
            "source_file": "genjson-sample.json",
            "ground_truth_label_mode": "step2b_polynomial_degree_noiseless",
            "step2b_degree": 8,
            "step2b_kappa": 3.0,
            "step2b_delta": 1e-12,
            "clean_linear_utility_weight": 10.0,
            "clean_linear_cpra_weight": 5.0,
        },
        "data": {
            "1": {
                "type": "Pair",
                "matches": [
                    {"recipient": "2", "utility": 80, "recipient_cpra": 0.4},
                    {"recipient": "3", "utility": 40, "recipient_cpra": 0.2},
                ],
            },
            "2": {
                "type": "Pair",
                "matches": [{"recipient": "3", "utility": 70, "recipient_cpra": 0.1}],
            },
            "3": {"type": "Pair", "matches": []},
        },
    }


class FixedTopologyLabelSeedTest(unittest.TestCase):
    def test_topology_hash_ignores_labels(self):
        mod = load_module()
        payload = sample_payload()
        relabeled_a = mod.relabel_payload_step2c(payload, label_seed=1, epsilon_bar=0.5)
        relabeled_b = mod.relabel_payload_step2c(payload, label_seed=2, epsilon_bar=0.5)

        self.assertEqual(mod.topology_hash(relabeled_a), mod.topology_hash(relabeled_b))
        self.assertEqual(mod.edge_count(relabeled_a), 3)

    def test_label_seed_changes_labels_under_step2c(self):
        mod = load_module()
        payload = sample_payload()
        relabeled_a = mod.relabel_payload_step2c(payload, label_seed=1, epsilon_bar=0.5)
        relabeled_b = mod.relabel_payload_step2c(payload, label_seed=2, epsilon_bar=0.5)

        labels_a = mod.edge_labels(relabeled_a)
        labels_b = mod.edge_labels(relabeled_b)

        self.assertNotEqual(labels_a, labels_b)
        self.assertTrue(all(value >= 0.0 for value in labels_a))
        self.assertTrue(all(value >= 0.0 for value in labels_b))

    def test_case_c_signature_requires_large_2stage_gap_and_small_spoplus_gap(self):
        mod = load_module()
        rows = [
            {
                "method_label": "2stage_val_mse",
                "solution_rank": 1,
                "normalized_gap_to_oracle": 0.31,
            },
            {
                "method_label": "spoplus_val_spoplus_loss",
                "solution_rank": 1,
                "normalized_gap_to_oracle": 0.01,
            },
        ]

        self.assertTrue(
            mod.case_c_signature(
                rows,
                two_stage_min_gap=0.05,
                spoplus_max_gap=0.02,
            )
        )

    def test_parse_args_expands_inclusive_label_seed_range(self):
        mod = load_module()

        args = mod.parse_args(["--label-seed-start", "2", "--label-seed-stop", "5"])

        self.assertEqual(args.label_seeds, [2, 3, 4, 5])


if __name__ == "__main__":
    unittest.main()
