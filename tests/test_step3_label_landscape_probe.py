import importlib.util
import math
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "surrogate_experiment_results"
    / "Step3"
    / "scripts"
    / "probe_label_landscape.py"
)


def load_probe_module():
    spec = importlib.util.spec_from_file_location("step3_probe_label_landscape", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample_template():
    return {
        "topology_id": "G-test",
        "vertices": [
            {"id": "0", "type": "pair"},
            {"id": "1", "type": "pair"},
            {"id": "2", "type": "ndd"},
        ],
        "arcs": [
            {"edge_idx": 0, "source": "0", "target": "1"},
            {"edge_idx": 1, "source": "1", "target": "0"},
            {"edge_idx": 2, "source": "2", "target": "0"},
            {"edge_idx": 3, "source": "2", "target": "1"},
        ],
        "feasible_candidates": [
            {"type": "cycle", "nodes": ["0", "1"], "edges": [0, 1]},
            {"type": "chain", "nodes": ["2", "0"], "edges": [2]},
            {"type": "chain", "nodes": ["2", "1"], "edges": [3]},
        ],
    }


class Step3LabelLandscapeProbeTests(unittest.TestCase):
    def test_step2c_label_realizations_are_reproducible_and_seed_dependent(self):
        module = load_probe_module()
        edge_rows = [
            {"edge_idx": 0, "source": "0", "target": "1", "utility": 50.0, "recipient_cpra": 0.2},
            {"edge_idx": 1, "source": "1", "target": "0", "utility": 80.0, "recipient_cpra": 0.6},
            {"edge_idx": 2, "source": "2", "target": "0", "utility": 20.0, "recipient_cpra": 0.4},
        ]

        labels_a = module.compute_step2c_labels(edge_rows, label_seed=11, topology_id="G-test")
        labels_a_repeat = module.compute_step2c_labels(edge_rows, label_seed=11, topology_id="G-test")
        labels_b = module.compute_step2c_labels(edge_rows, label_seed=12, topology_id="G-test")

        self.assertEqual(labels_a, labels_a_repeat)
        self.assertEqual(len(labels_a), len(edge_rows))
        self.assertTrue(all(label >= 0.0 for label in labels_a))
        self.assertTrue(any(abs(a - b) > 1e-9 for a, b in zip(labels_a, labels_b)))

    def test_candidate_packing_solver_selects_vertex_disjoint_solution(self):
        try:
            import gurobipy  # noqa: F401
        except ImportError:
            self.skipTest("gurobipy is not installed")

        module = load_probe_module()
        try:
            solver = module.CandidatePackingSolver(sample_template())
        except Exception as exc:
            if "Gurobi" in type(exc).__name__ or "token.gurobi.com" in str(exc):
                self.skipTest(f"Gurobi license unavailable: {exc}")
            raise
        try:
            solutions = solver.solve_top_k([5.0, 4.0, 1.0, 1.0], top_k=2)
        finally:
            solver.dispose()

        self.assertGreaterEqual(len(solutions), 2)
        self.assertEqual(solutions[0]["edge_signature"], "0|1")
        self.assertAlmostEqual(solutions[0]["objective"], 9.0)
        self.assertNotEqual(solutions[1]["edge_signature"], "0|1")

    def test_topology_summary_tracks_entropy_and_proxy_gap(self):
        module = load_probe_module()
        rows = [
            {
                "oracle_solution_signature": "0|1",
                "oracle_objective": 10.0,
                "oracle_top1_top2_margin": 2.0,
                "oracle_top1_top5_margin": 5.0,
                "linear_proxy_differs_from_oracle": False,
                "linear_proxy_gap_to_oracle": 0.0,
                "linear_proxy_normalized_gap_to_oracle": 0.0,
                "linear_proxy_jaccard_with_oracle": 1.0,
            },
            {
                "oracle_solution_signature": "0|1",
                "oracle_objective": 12.0,
                "oracle_top1_top2_margin": 3.0,
                "oracle_top1_top5_margin": 6.0,
                "linear_proxy_differs_from_oracle": True,
                "linear_proxy_gap_to_oracle": 2.0,
                "linear_proxy_normalized_gap_to_oracle": 2.0 / 12.0,
                "linear_proxy_jaccard_with_oracle": 0.5,
            },
            {
                "oracle_solution_signature": "2",
                "oracle_objective": 7.0,
                "oracle_top1_top2_margin": 1.0,
                "oracle_top1_top5_margin": None,
                "linear_proxy_differs_from_oracle": True,
                "linear_proxy_gap_to_oracle": 1.0,
                "linear_proxy_normalized_gap_to_oracle": 1.0 / 7.0,
                "linear_proxy_jaccard_with_oracle": 0.0,
            },
        ]

        summary = module.summarize_topology_samples(
            "G-test", rows, topology_descriptor={"num_exchange_candidates": 3}
        )

        expected_entropy = -((2 / 3) * math.log(2 / 3) + (1 / 3) * math.log(1 / 3))
        self.assertEqual(summary["num_distinct_oracle_solutions"], 2)
        self.assertAlmostEqual(summary["oracle_solution_entropy"], expected_entropy)
        self.assertAlmostEqual(summary["dominant_oracle_solution_fraction"], 2 / 3)
        self.assertAlmostEqual(summary["fraction_linear_proxy_differs_from_oracle"], 2 / 3)
        self.assertAlmostEqual(summary["mean_pairwise_oracle_jaccard"], 1 / 3)
        self.assertEqual(summary["num_exchange_candidates"], 3)


if __name__ == "__main__":
    unittest.main()
