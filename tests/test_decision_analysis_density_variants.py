from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "decision_analysis"
    / "scripts"
    / "make_arc_density_variants.py"
)


def pair_node(node_id: str, bloodtype: str = "A") -> dict[str, object]:
    numeric_id = int(node_id)
    return {
        "type": "Pair",
        "id": node_id,
        "vertex_failure_prob": 0.1,
        "vertex_available_prob": 0.9,
        "patient": {
            "age": 40 + numeric_id,
            "bloodtype": bloodtype,
            "cPRA": round(0.1 + 0.1 * numeric_id, 4),
            "hasBloodCompatibleDonor": True,
        },
        "donors": [
            {
                "original_node_id": node_id,
                "dage": 30 + numeric_id,
                "bloodtype": bloodtype,
            }
        ],
        "matches": [],
    }


def ndd_node(node_id: str = "100") -> dict[str, object]:
    return {
        "type": "NDD",
        "id": node_id,
        "vertex_failure_prob": 0.05,
        "vertex_available_prob": 0.95,
        "donor": {"dage": 52, "bloodtype": "O"},
        "matches": [],
    }


def match(dst: str, utility: int, cpra: float | None = None) -> dict[str, object]:
    cpra_value = cpra if cpra is not None else round(0.1 + 0.1 * int(dst), 4)
    return {
        "recipient": dst,
        "utility": utility,
        "graft_survival_time": 10.0,
        "qaly": 9.5,
        "medical_success_score": 0.7,
        "success_prob": 0.9,
        "arc_failure_prob": 0.1,
        "source_vertex_failure_prob": 0.1,
        "target_vertex_failure_prob": 0.1,
        "expected_transplant_count": 0.81,
        "recipient_in_degree": 1,
        "donor_out_degree": 1,
        "target_scarcity": 0.5,
        "donor_flexibility": 0.4,
        "has_reciprocal_edge": False,
        "priority_multiplier": 1.0,
        "ground_truth_label": round(utility / 10.0, 4),
        "donor_age": 35,
        "donor_bt": "A",
        "recipient_age": 44,
        "recipient_cpra": cpra_value,
        "recipient_bt": "A",
        "latent_clean_linear_label": round(utility / 10.0, 4),
        "step2b_polynomial_label": round(utility / 10.0, 4),
        "step2b_q_value": 1.0,
        "step2b_polynomial_score": 10.0,
        "step2b_graph_clean_linear_mean": 5.0,
        "step2b_graph_polynomial_score_mean": 10.0,
        "step2b_graph_clean_linear_edge_count": 6,
        "step2b_degree": 8,
        "step2b_kappa": 3.0,
        "step2b_delta": 1e-12,
        "winning_donor_id": "0",
    }


def toy_graph_payload() -> dict[str, object]:
    data = {
        "0": pair_node("0", "A"),
        "1": pair_node("1", "B"),
        "2": pair_node("2", "A"),
        "3": pair_node("3", "O"),
        "100": ndd_node("100"),
    }
    data["0"]["matches"] = [match("1", 60), match("2", 55)]
    data["1"]["matches"] = [match("2", 61)]
    data["2"]["matches"] = [match("3", 62)]
    data["3"]["matches"] = [match("0", 63)]
    data["100"]["matches"] = [match("0", 64)]
    return {
        "metadata": {
            "original_file": "G-test.json",
            "total_vertices": 5,
            "structure": "Unified Pair/NDD Graph",
            "ground_truth_label_mode": "step2b_polynomial_degree_noiseless",
            "clean_linear_utility_weight": 10.0,
            "clean_linear_cpra_weight": 5.0,
            "step2b_degree": 8,
            "step2b_kappa": 3.0,
            "step2b_delta": 1e-12,
            "label_seed": 42,
        },
        "data": data,
    }


def read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def arc_keys(payload: dict[str, object]) -> set[str]:
    output: set[str] = set()
    for src_id, node in payload["data"].items():
        for edge in node.get("matches", []):
            output.add(f"{src_id}->{edge['recipient']}")
    return output


def arc_count(payload: dict[str, object]) -> int:
    return sum(len(node.get("matches", [])) for node in payload["data"].values())


class DecisionAnalysisDensityVariantTests(unittest.TestCase):
    def load_module(self):
        self.assertTrue(SCRIPT_PATH.exists(), f"Missing script: {SCRIPT_PATH}")
        spec = importlib.util.spec_from_file_location("make_arc_density_variants", SCRIPT_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def write_toy_graph(self, root: Path) -> Path:
        graph_path = root / "G-test.json"
        with graph_path.open("w", encoding="utf-8") as handle:
            json.dump(toy_graph_payload(), handle)
        return graph_path

    def test_generate_variants_records_stable_arc_keys_and_arc_counts(self):
        module = self.load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            graph_path = self.write_toy_graph(tmp)
            output_dir = tmp / "variants"

            rows = module.generate_variants_for_graph(
                graph_path=graph_path,
                output_dir=output_dir,
                case={
                    "case_id": "case_test",
                    "case_label": "case_density",
                    "subset_seed": "7",
                    "graph_id": "G-test.json",
                },
                variants=["original", "add25pct", "add25arcs", "remove25arcs", "remove25pct"],
                perturb_seed=11,
                fixed_add_count=2,
                fixed_remove_count=2,
            )

            by_variant = {row["density_variant"]: row for row in rows}
            self.assertEqual(
                set(by_variant),
                {"original", "add25pct", "add25arcs", "remove25arcs", "remove25pct"},
            )

            original_payload = read_json(Path(by_variant["original"]["variant_graph_path"]))
            add_payload = read_json(Path(by_variant["add25pct"]["variant_graph_path"]))
            add_fixed_payload = read_json(Path(by_variant["add25arcs"]["variant_graph_path"]))
            remove_payload = read_json(Path(by_variant["remove25arcs"]["variant_graph_path"]))
            remove_pct_payload = read_json(Path(by_variant["remove25pct"]["variant_graph_path"]))

            self.assertEqual(by_variant["original"]["original_num_arcs"], 6)
            self.assertEqual(by_variant["original"]["variant_num_arcs"], 6)
            self.assertEqual(by_variant["add25pct"]["variant_num_arcs"], 8)
            self.assertEqual(by_variant["add25arcs"]["variant_num_arcs"], 8)
            self.assertEqual(by_variant["remove25arcs"]["variant_num_arcs"], 4)
            self.assertEqual(by_variant["remove25pct"]["variant_num_arcs"], 4)

            self.assertEqual(arc_count(original_payload), 6)
            self.assertEqual(arc_count(add_payload), 8)
            self.assertEqual(arc_count(add_fixed_payload), 8)
            self.assertEqual(arc_count(remove_payload), 4)
            self.assertEqual(arc_count(remove_pct_payload), 4)

            original_keys = arc_keys(original_payload)
            add_keys = arc_keys(add_payload)
            add_fixed_keys = arc_keys(add_fixed_payload)
            remove_keys = arc_keys(remove_payload)

            added_keys = set(by_variant["add25pct"]["added_arc_keys"].split("|"))
            added_fixed_keys = set(by_variant["add25arcs"]["added_arc_keys"].split("|"))
            removed_keys = set(by_variant["remove25arcs"]["removed_arc_keys"].split("|"))
            self.assertEqual(add_keys - original_keys, added_keys)
            self.assertEqual(add_fixed_keys - original_keys, added_fixed_keys)
            self.assertEqual(len(added_fixed_keys), 2)
            self.assertEqual(by_variant["add25arcs"]["arc_delta_type"], "add_fixed")
            self.assertEqual(original_keys - remove_keys, removed_keys)
            self.assertTrue(all("->" in key for key in added_keys | added_fixed_keys | removed_keys))

            for row in rows:
                self.assertIn("original_num_arcs", row)
                self.assertIn("variant_num_arcs", row)
                self.assertNotIn("edge_count", row)
                self.assertNotIn("original_edge_count", row)
                self.assertNotIn("variant_edge_count", row)

    def test_added_variant_is_parser_compatible_and_has_required_match_fields(self):
        module = self.load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            graph_path = self.write_toy_graph(tmp)
            rows = module.generate_variants_for_graph(
                graph_path=graph_path,
                output_dir=tmp / "variants",
                case={"case_id": "case_test", "case_label": "case_density", "subset_seed": "7", "graph_id": "G-test.json"},
                variants=["add25pct"],
                perturb_seed=17,
                fixed_remove_count=2,
            )
            add_row = rows[0]
            add_path = Path(add_row["variant_graph_path"])
            payload = read_json(add_path)
            added_keys = set(add_row["added_arc_keys"].split("|"))

            required_keys = {
                "recipient",
                "utility",
                "ground_truth_label",
                "donor_age",
                "donor_bt",
                "recipient_age",
                "recipient_cpra",
                "recipient_bt",
            }
            for src_id, node in payload["data"].items():
                for edge in node.get("matches", []):
                    if f"{src_id}->{edge['recipient']}" in added_keys:
                        self.assertTrue(required_keys.issubset(edge.keys()))
                        self.assertEqual(edge["label_policy"], "synthetic_step2b_original_scale")

            from model.graph_utils import parse_json_to_dfl_data

            data = parse_json_to_dfl_data(add_path, max_cycle=3, max_chain=4, label_scale=1.0)
            self.assertEqual(data.edge_index.shape[1], 8)

    def test_write_manifest_uses_density_field_names(self):
        module = self.load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            rows = [
                {
                    "case_id": "case_test",
                    "case_label": "case_density",
                    "subset_seed": "7",
                    "base_graph_id": "G-test.json",
                    "variant_id": "G-test__original__seed11",
                    "variant_graph_path": str(tmp / "G-test__original__seed11.json"),
                    "density_variant": "original",
                    "arc_delta_type": "none",
                    "original_num_arcs": 6,
                    "variant_num_arcs": 6,
                    "arc_delta": 0,
                    "added_arc_count": 0,
                    "removed_arc_count": 0,
                    "added_arc_keys": "",
                    "removed_arc_keys": "",
                    "perturb_seed": 11,
                    "generation_policy": "density_sensitivity_structural_perturbation",
                    "label_policy": "frozen_existing_edges_original_scale_new_edges",
                    "added_arc_source_policy": "sample_missing_vertex_arcs",
                    "added_arc_label_policy": "synthetic_step2b_original_scale",
                    "removed_arc_policy": "sample_existing_vertex_arcs",
                    "new_arc_label_mean": "",
                    "existing_arc_label_mean": 5.0,
                }
            ]
            manifest_path = tmp / "manifest.csv"
            module.write_manifest(manifest_path, rows)

            with manifest_path.open(newline="", encoding="utf-8") as handle:
                loaded = list(csv.DictReader(handle))

            self.assertEqual(len(loaded), 1)
            self.assertIn("original_num_arcs", loaded[0])
            self.assertIn("variant_num_arcs", loaded[0])
            self.assertIn("added_arc_keys", loaded[0])
            self.assertIn("removed_arc_keys", loaded[0])
            self.assertNotIn("edge_count", loaded[0])
            self.assertNotIn("original_edge_count", loaded[0])
            self.assertNotIn("variant_edge_count", loaded[0])

    def test_manifest_graph_path_is_repo_relative_for_repo_local_variants(self):
        module = self.load_module()
        variant_path = (
            module.PROJECT_ROOT
            / "surrogate_experiment_results"
            / "decision_analysis"
            / "results"
            / "density_sensitivity"
            / "graphs"
            / "G-test__original__seed42.json"
        )

        manifest_path = module.manifest_graph_path(variant_path)

        self.assertEqual(
            manifest_path,
            "surrogate_experiment_results/decision_analysis/results/density_sensitivity/graphs/G-test__original__seed42.json",
        )

    def test_generate_all_variants_accepts_multiple_perturb_seeds(self):
        module = self.load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset_dir = tmp / "dataset"
            dataset_dir.mkdir()
            self.write_toy_graph(dataset_dir)
            case_index = tmp / "case_study_index.csv"
            with case_index.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["case_id", "case_label", "subset_seed", "graph_id"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "case_id": "case_test",
                        "case_label": "case_density",
                        "subset_seed": "7",
                        "graph_id": "G-test.json",
                    }
                )

            args = module.parse_args(
                [
                    "--case-index",
                    str(case_index),
                    "--dataset-dir",
                    str(dataset_dir),
                    "--output-dir",
                    str(tmp / "robustness"),
                    "--graphs",
                    "G-test.json",
                    "--variants",
                    "original",
                    "add25arcs",
                    "--perturb-seeds",
                    "0",
                    "1",
                    "--fixed-add-count",
                    "1",
                ]
            )

            rows = module.generate_all_variants(args)

            self.assertEqual(len(rows), 4)
            self.assertEqual(sorted({int(row["perturb_seed"]) for row in rows}), [0, 1])
            self.assertEqual(
                sorted({row["variant_id"] for row in rows}),
                [
                    "G-test__add25arcs__seed0",
                    "G-test__add25arcs__seed1",
                    "G-test__original__seed0",
                    "G-test__original__seed1",
                ],
            )


if __name__ == "__main__":
    unittest.main()
