import csv
import importlib.util
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = (
    ROOT
    / "surrogate_experiment_results"
    / "Step3"
    / "scripts"
    / "build_topology_bank.py"
)


def load_builder():
    spec = importlib.util.spec_from_file_location("step3_topology_bank", BUILDER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample_graph(label_value=1.0):
    data = {}
    for index in range(7):
        data[str(index)] = {
            "type": "Pair",
            "id": str(index),
            "patient": {"age": 40, "bloodtype": "A", "cPRA": 0.2, "hasBloodCompatibleDonor": True},
            "donors": [{"original_node_id": str(index), "dage": 35, "bloodtype": "A"}],
            "matches": [],
        }
    data["7"] = {
        "type": "NDD",
        "id": "7",
        "donor": {"original_node_id": "7", "dage": 44, "bloodtype": "O"},
        "matches": [],
    }

    def add_edge(source, target, utility):
        data[str(source)]["matches"].append(
            {
                "recipient": str(target),
                "utility": utility,
                "ground_truth_label": label_value,
                "recipient_cpra": 0.2,
                "donor_age": 35,
                "recipient_age": 40,
                "donor_bt": "A",
                "recipient_bt": "A",
            }
        )

    add_edge(0, 1, 80)
    add_edge(1, 0, 70)
    add_edge(1, 2, 60)
    add_edge(2, 0, 50)
    add_edge(7, 3, 90)
    add_edge(3, 4, 40)
    return {
        "metadata": {
            "total_vertices": 8,
            "ground_truth_label_mode": "step2c_polynomial_degree_multiplicative_noise",
        },
        "data": data,
    }


def write_graph(path, graph):
    path.write_text(json.dumps(graph), encoding="utf-8")


class Step3TopologyBankTest(unittest.TestCase):
    def test_template_hashes_ignore_label_values(self):
        module = load_builder()
        graph_a = sample_graph(label_value=1.0)
        graph_b = sample_graph(label_value=99.0)

        template_a = module.build_topology_template("G-0", graph_a, max_cycle=3, max_chain=4)
        template_b = module.build_topology_template("G-0", graph_b, max_cycle=3, max_chain=4)

        self.assertEqual(template_a["topology_hash"], template_b["topology_hash"])
        self.assertEqual(template_a["arc_order_hash"], template_b["arc_order_hash"])
        self.assertEqual(template_a["feasible_set_hash"], template_b["feasible_set_hash"])
        self.assertEqual(template_a["num_pairs"], 7)
        self.assertEqual(template_a["num_ndds"], 1)
        self.assertEqual(template_a["num_arcs"], 6)
        self.assertEqual(template_a["num_cycles_total"], 2)
        self.assertEqual(template_a["num_chains_total"], 2)

    def test_topology_hash_changes_when_arc_changes(self):
        module = load_builder()
        graph_a = sample_graph()
        graph_b = deepcopy(graph_a)
        graph_b["data"]["4"]["matches"].append(
            {
                "recipient": "5",
                "utility": 30,
                "ground_truth_label": 1.0,
            }
        )

        template_a = module.build_topology_template("G-0", graph_a, max_cycle=3, max_chain=4)
        template_b = module.build_topology_template("G-0", graph_b, max_cycle=3, max_chain=4)

        self.assertNotEqual(template_a["topology_hash"], template_b["topology_hash"])
        self.assertNotEqual(template_a["arc_order_hash"], template_b["arc_order_hash"])

    def test_feasible_set_hash_changes_when_max_chain_changes(self):
        module = load_builder()
        graph = sample_graph()

        short_chain = module.build_topology_template("G-0", graph, max_cycle=3, max_chain=1)
        long_chain = module.build_topology_template("G-0", graph, max_cycle=3, max_chain=4)

        self.assertNotEqual(short_chain["feasible_set_hash"], long_chain["feasible_set_hash"])
        self.assertEqual(short_chain["topology_hash"], long_chain["topology_hash"])
        self.assertEqual(short_chain["num_chains_total"], 1)
        self.assertEqual(long_chain["num_chains_total"], 2)

    def test_template_includes_candidate_structure_descriptors(self):
        module = load_builder()
        graph = sample_graph()

        template = module.build_topology_template("G-0", graph, max_cycle=3, max_chain=4)

        self.assertEqual(template["num_exchange_candidates"], 4)
        self.assertEqual(template["num_chains_len1"], 1)
        self.assertEqual(template["num_chains_len2"], 1)
        self.assertEqual(template["num_chains_len3"], 0)
        self.assertEqual(template["num_chains_len4"], 0)
        self.assertEqual(template["candidate_conflict_edges"], 2)
        self.assertAlmostEqual(template["candidate_conflict_density"], 2 / 6)
        self.assertAlmostEqual(template["mean_conflict_degree"], 1.0)
        self.assertEqual(template["max_conflict_degree"], 1)
        self.assertEqual(template["num_conflict_components"], 2)
        self.assertAlmostEqual(template["largest_conflict_component_fraction"], 0.5)
        self.assertEqual(template["num_vertices_in_any_candidate"], 6)
        self.assertAlmostEqual(template["fraction_vertices_in_any_candidate"], 6 / 8)
        self.assertAlmostEqual(template["mean_candidates_per_vertex"], 10 / 8)
        self.assertEqual(template["max_candidates_per_vertex"], 2)

    def test_build_topology_bank_writes_templates_and_csvs(self):
        module = load_builder()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "processed"
            output_dir = root / "topologies"
            source_dir.mkdir()
            write_graph(source_dir / "G-0.json", sample_graph())

            summary = module.build_topology_bank(
                source_dir=source_dir,
                output_dir=output_dir,
                max_cycle=3,
                max_chain=4,
                expected_pairs=7,
                min_candidates=1,
                force=True,
            )

            self.assertEqual(summary["accepted"], 1)
            self.assertEqual(summary["rejected"], 0)
            self.assertTrue((output_dir / "G-0" / "template.json").exists())
            self.assertTrue((output_dir / "topology_bank.csv").exists())
            self.assertTrue((output_dir / "topology_hashes.csv").exists())
            self.assertTrue((output_dir / "rejected_topologies.csv").exists())

            template = json.loads((output_dir / "G-0" / "template.json").read_text(encoding="utf-8"))
            self.assertEqual(template["topology_id"], "G-0")
            self.assertEqual(template["max_cycle"], 3)
            self.assertEqual(template["max_chain"], 4)
            self.assertEqual(len(template["arcs"]), 6)
            self.assertEqual(len(template["feasible_candidates"]), 4)

            with (output_dir / "topology_bank.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["topology_id"], "G-0")
            self.assertEqual(rows[0]["num_pairs"], "7")
            self.assertEqual(rows[0]["num_chains_total"], "2")
            self.assertEqual(rows[0]["num_exchange_candidates"], "4")
            self.assertEqual(rows[0]["num_chains_len1"], "1")
            self.assertEqual(rows[0]["candidate_conflict_edges"], "2")
            self.assertEqual(rows[0]["num_vertices_in_any_candidate"], "6")


if __name__ == "__main__":
    unittest.main()
