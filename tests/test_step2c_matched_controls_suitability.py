from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "decision_analysis"
    / "scripts"
    / "build_step2c_matched_controls_suitability.py"
)
WRAPPER_PATH = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "Step2c Graph-Level DFL Suitability Audit"
    / "scripts"
    / "build_phase3_matched_controls.py"
)


def row(
    graph_id: str,
    *,
    vertices: float,
    arcs: float,
    density: float,
    two_cycles: float,
    three_cycles: float,
    scc: float,
    delta: float = 0.0,
    ambiguity: float = 0.0,
    richness: float = 0.0,
) -> dict[str, str]:
    return {
        "graph_id": graph_id,
        "num_vertices": str(vertices),
        "num_arcs": str(arcs),
        "density": str(density),
        "num_2cycles": str(two_cycles),
        "num_3cycles": str(three_cycles),
        "largest_scc_fraction": str(scc),
        "median_delta_pp": str(delta),
        "helpful_graph": str(int(delta >= 10)),
        "harmful_graph": str(int(delta <= -10)),
        "feasible_set_richness_score": str(richness),
        "ranking_ambiguity_score": str(ambiguity),
    }


class Step2cMatchedControlsSuitabilityTests(unittest.TestCase):
    def load_module(self):
        self.assertTrue(SCRIPT_PATH.exists(), f"Missing script: {SCRIPT_PATH}")
        spec = importlib.util.spec_from_file_location(
            "build_step2c_matched_controls_suitability", SCRIPT_PATH
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def test_match_rows_use_readme_topology_variables_and_exclude_target(self):
        module = self.load_module()
        rows = [
            row("G-target.json", vertices=10, arcs=30, density=0.30, two_cycles=4, three_cycles=2, scc=0.8),
            row("G-close.json", vertices=11, arcs=31, density=0.31, two_cycles=4, three_cycles=2, scc=0.8),
            row("G-far.json", vertices=30, arcs=90, density=0.70, two_cycles=20, three_cycles=10, scc=0.2),
        ]

        matches = module.build_matched_control_rows(
            rows,
            target_graphs=("G-target.json",),
            n_controls=2,
        )

        self.assertEqual([item["control_graph_id"] for item in matches], ["G-close.json", "G-far.json"])
        self.assertEqual(matches[0]["target_graph_id"], "G-target.json")
        self.assertEqual(matches[0]["match_rank"], 1)
        self.assertLess(matches[0]["match_distance"], matches[1]["match_distance"])
        self.assertEqual(module.MATCH_FEATURES, module.README_MATCH_FEATURES)

    def test_summary_compares_target_against_matched_control_outcomes_and_features(self):
        module = self.load_module()
        rows = [
            row("G-target.json", vertices=10, arcs=30, density=0.30, two_cycles=4, three_cycles=2, scc=0.8, delta=25, ambiguity=3.0, richness=2.0),
            row("G-c1.json", vertices=10, arcs=30, density=0.31, two_cycles=4, three_cycles=2, scc=0.8, delta=0, ambiguity=0.0, richness=0.5),
            row("G-c2.json", vertices=11, arcs=31, density=0.32, two_cycles=5, three_cycles=2, scc=0.7, delta=10, ambiguity=1.0, richness=1.0),
            row("G-c3.json", vertices=12, arcs=32, density=0.33, two_cycles=6, three_cycles=3, scc=0.7, delta=-10, ambiguity=2.0, richness=1.5),
        ]
        matches = module.build_matched_control_rows(
            rows,
            target_graphs=("G-target.json",),
            n_controls=3,
        )

        summary = module.build_target_vs_matched_summary_rows(rows, matches)

        self.assertEqual(len(summary), 1)
        target = summary[0]
        self.assertEqual(target["target_graph_id"], "G-target.json")
        self.assertEqual(target["n_controls"], 3)
        self.assertAlmostEqual(target["target_median_delta_pp"], 25.0)
        self.assertAlmostEqual(target["matched_median_delta_pp_median"], 0.0)
        self.assertAlmostEqual(target["target_delta_percentile_within_matched"], 1.0)
        self.assertAlmostEqual(target["target_ranking_ambiguity_percentile_within_matched"], 1.0)
        self.assertAlmostEqual(target["target_feasible_richness_percentile_within_matched"], 1.0)

    def test_target_catalog_uses_readme_case_groups(self):
        module = self.load_module()

        catalog = module.target_catalog_rows()
        labels = {item["graph_id"]: item["case_group"] for item in catalog}

        self.assertEqual(labels["G-392.json"], "helpful_success")
        self.assertEqual(labels["G-142.json"], "both_poor_control")
        self.assertEqual(labels["G-14.json"], "harmful_reranking_control")
        self.assertEqual(len(catalog), 9)

    def test_experiment_directory_exposes_phase3_entrypoint(self):
        self.assertTrue(WRAPPER_PATH.exists(), f"Missing wrapper: {WRAPPER_PATH}")


if __name__ == "__main__":
    unittest.main()
