import copy
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "surrogate_experiment_results"
    / "Step3"
    / "scripts"
    / "sample_fixed_topology_context.py"
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
    def pair(node_id, cpra):
        return {
            "id": str(node_id),
            "type": "Pair",
            "patient": {
                "age": 40,
                "bloodtype": "A",
                "cPRA": cpra,
                "hasBloodCompatibleDonor": True,
            },
            "donors": [{"original_node_id": str(node_id), "dage": 35, "bloodtype": "A"}],
            "matches": [],
        }

    data = {
        "0": pair("0", 0.2),
        "1": pair("1", 0.5),
        "2": {
            "id": "2",
            "type": "NDD",
            "donor": {"original_node_id": "2", "dage": 45, "bloodtype": "O"},
            "matches": [],
        },
    }
    data["0"]["matches"] = [
        {"recipient": "1", "utility": 60.0, "recipient_cpra": 0.5},
    ]
    data["1"]["matches"] = [
        {"recipient": "0", "utility": 70.0, "recipient_cpra": 0.2},
    ]
    data["2"]["matches"] = [
        {"recipient": "0", "utility": 40.0, "recipient_cpra": 0.2},
        {"recipient": "1", "utility": 50.0, "recipient_cpra": 0.5},
    ]
    return {
        "metadata": {"source_file": "G-test.json"},
        "data": data,
    }


def pilot_config():
    return {
        "status": "pilot_not_locked",
        "generator_version": "test-v1",
        "method": "bounded_additive_uniform",
        "recipient_cpra": {"lower": 0.0, "upper": 1.0, "half_width": 0.05},
        "utility": {"lower": 0.0, "upper": 100.0, "half_width": 5.0},
    }


class Step3FixedTopologyContextTests(unittest.TestCase):
    def test_same_seed_reproduces_context_and_different_seed_changes_x(self):
        module = load_module(SCRIPT, "sample_fixed_topology_context")
        builder = load_module(BUILDER, "build_topology_bank_for_context_test")
        base = sample_payload()
        template = builder.build_topology_template("G-test", base, max_cycle=3, max_chain=4)

        first = module.sample_context(template, base, context_seed=123, generator_config=pilot_config())
        second = module.sample_context(template, base, context_seed=123, generator_config=pilot_config())
        third = module.sample_context(template, base, context_seed=124, generator_config=pilot_config())

        self.assertEqual(first, second)
        self.assertNotEqual(first, third)
        self.assertEqual(base["data"]["0"]["patient"]["cPRA"], 0.2)

    def test_recipient_cpra_is_sampled_once_and_copied_to_all_incoming_arcs(self):
        module = load_module(SCRIPT, "sample_fixed_topology_context")
        builder = load_module(BUILDER, "build_topology_bank_for_context_sync_test")
        template = builder.build_topology_template("G-test", sample_payload(), max_cycle=3, max_chain=4)

        sampled = module.sample_context(
            template,
            sample_payload(),
            context_seed=7,
            generator_config=pilot_config(),
        )

        cpra_0 = sampled["data"]["0"]["patient"]["cPRA"]
        cpra_1 = sampled["data"]["1"]["patient"]["cPRA"]
        incoming_0 = [
            match["recipient_cpra"]
            for node in sampled["data"].values()
            for match in node.get("matches", [])
            if str(match["recipient"]) == "0"
        ]
        incoming_1 = [
            match["recipient_cpra"]
            for node in sampled["data"].values()
            for match in node.get("matches", [])
            if str(match["recipient"]) == "1"
        ]

        self.assertTrue(incoming_0)
        self.assertTrue(incoming_1)
        self.assertTrue(all(value == cpra_0 for value in incoming_0))
        self.assertTrue(all(value == cpra_1 for value in incoming_1))
        self.assertGreaterEqual(cpra_0, 0.0)
        self.assertLessEqual(cpra_0, 1.0)

    def test_context_sampler_preserves_structure_and_utility_bounds(self):
        module = load_module(SCRIPT, "sample_fixed_topology_context")
        builder = load_module(BUILDER, "build_topology_bank_for_context_structure_test")
        base = sample_payload()
        template = builder.build_topology_template("G-test", base, max_cycle=3, max_chain=4)
        base_copy = copy.deepcopy(base)

        sampled = module.sample_context(template, base, context_seed=99, generator_config=pilot_config())

        self.assertEqual(builder.build_topology_template("G-test", sampled)["topology_hash"], template["topology_hash"])
        self.assertEqual(builder.build_topology_template("G-test", sampled)["arc_order_hash"], template["arc_order_hash"])
        self.assertEqual(builder.build_topology_template("G-test", sampled)["feasible_set_hash"], template["feasible_set_hash"])
        for node in sampled["data"].values():
            for match in node.get("matches", []):
                self.assertGreaterEqual(match["utility"], 0.0)
                self.assertLessEqual(match["utility"], 100.0)
        self.assertEqual(base, base_copy)


if __name__ == "__main__":
    unittest.main()
