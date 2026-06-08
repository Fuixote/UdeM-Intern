from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "decision_analysis"
    / "scripts"
    / "compute_arc_density_sensitivity.py"
)


def manifest_row(
    graph_path: Path,
    *,
    density_variant: str = "original",
    subset_seed: str = "7",
    base_graph_id: str = "G-test.json",
) -> dict[str, str]:
    return {
        "case_id": f"case_{density_variant}",
        "case_label": "case_density",
        "subset_seed": subset_seed,
        "base_graph_id": base_graph_id,
        "variant_id": f"G-test__{density_variant}__seed42",
        "variant_graph_path": str(graph_path),
        "density_variant": density_variant,
        "arc_delta_type": "none" if density_variant == "original" else "add_fixed",
        "original_num_arcs": "3",
        "variant_num_arcs": "3",
        "arc_delta": "0",
        "added_arc_count": "0",
        "removed_arc_count": "0",
        "added_arc_keys": "",
        "removed_arc_keys": "",
        "perturb_seed": "42",
        "generation_policy": "density_sensitivity_structural_perturbation",
        "label_policy": "frozen_existing_edges_original_scale_new_edges",
        "added_arc_source_policy": "sample_missing_vertex_arcs",
        "added_arc_label_policy": "synthetic_step2b_original_scale",
        "removed_arc_policy": "sample_existing_vertex_arcs",
        "new_arc_label_mean": "",
        "existing_arc_label_mean": "5.0",
    }


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fake_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "filename": path.name,
        "X": np.zeros((3, 2)),
        "w_true": np.ones(3),
        "y_optimal": np.asarray([1.0, 0.0, 1.0]),
        "graph": {
            "edge_index": np.asarray([[0, 1, 2], [1, 2, 0]]),
            "id_map_rev": {0: "0", 1: "1", 2: "2"},
        },
    }


class DecisionAnalysisArcDensityComputeTests(unittest.TestCase):
    def load_module(self):
        self.assertTrue(SCRIPT_PATH.exists(), f"Missing script: {SCRIPT_PATH}")
        spec = importlib.util.spec_from_file_location("compute_arc_density_sensitivity", SCRIPT_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def test_arc_key_signature_uses_stable_vertex_ids_not_edge_indices(self):
        module = self.load_module()
        record = fake_record(Path("G-test__original__seed42.json"))

        edge_keys = module.edge_arc_keys(record)
        self.assertEqual(edge_keys, ["0->1", "1->2", "2->0"])

        self.assertEqual(module.arc_signature_from_y([1, 0, 1], edge_keys), "0->1|2->0")
        self.assertEqual(module.arc_signature_from_edge_signature("2|0", edge_keys), "0->1|2->0")

    def test_augment_rows_adds_density_metadata_and_arc_signatures(self):
        module = self.load_module()
        record = fake_record(Path("G-test__add25arcs__seed42.json"))
        row = {
            "method_label": "2stage_val_mse",
            "solution_rank": 1,
            "solution_edge_signature": "2|0",
            "edge_count": 2,
        }
        metadata = manifest_row(Path(record["path"]), density_variant="add25arcs")

        output = module.augment_solution_rows(
            rows=[row],
            record=record,
            manifest_row=metadata,
        )

        self.assertEqual(len(output), 1)
        augmented = output[0]
        self.assertEqual(augmented["density_variant"], "add25arcs")
        self.assertEqual(augmented["base_graph_id"], "G-test.json")
        self.assertEqual(augmented["original_num_arcs"], "3")
        self.assertEqual(augmented["variant_num_arcs"], "3")
        self.assertEqual(augmented["solution_selected_edge_count"], 2)
        self.assertEqual(augmented["solution_arc_key_signature"], "0->1|2->0")
        self.assertEqual(augmented["oracle_arc_key_signature"], "0->1|2->0")
        self.assertNotIn("original_edge_count", augmented)
        self.assertNotIn("variant_edge_count", augmented)

    def test_finalize_rows_adds_rank1_and_original_oracle_signatures(self):
        module = self.load_module()
        rows = [
            {
                "base_graph_id": "G-test.json",
                "density_variant": "original",
                "method_label": "2stage_val_mse",
                "solution_rank": 1,
                "solution_arc_key_signature": "0->1",
                "oracle_arc_key_signature": "0->1|2->0",
            },
            {
                "base_graph_id": "G-test.json",
                "density_variant": "add25arcs",
                "method_label": "2stage_val_mse",
                "solution_rank": 1,
                "solution_arc_key_signature": "0->1|1->2",
                "oracle_arc_key_signature": "0->1|1->2",
            },
            {
                "base_graph_id": "G-test.json",
                "density_variant": "add25arcs",
                "method_label": "2stage_val_mse",
                "solution_rank": 2,
                "solution_arc_key_signature": "1->2",
                "oracle_arc_key_signature": "0->1|1->2",
            },
        ]

        module.finalize_cross_variant_signatures(rows)

        rank2 = rows[2]
        self.assertEqual(rank2["rank1_arc_key_signature"], "0->1|1->2")
        self.assertEqual(rank2["original_oracle_arc_key_signature"], "0->1|2->0")

    def test_relative_variant_graph_path_resolves_from_project_root(self):
        module = self.load_module()
        relative_path = Path(
            "surrogate_experiment_results/decision_analysis/results/density_sensitivity/graphs/G-test__original__seed42.json"
        )

        resolved = module.resolve_variant_graph_path(str(relative_path))

        self.assertEqual(resolved, module.PROJECT_ROOT / relative_path)

    def test_compute_arc_density_rows_reuses_second_best_path_and_manifest_metadata(self):
        module = self.load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            graph_path = tmp / "G-test__original__seed42.json"
            graph_path.write_text("{}", encoding="utf-8")
            manifest_path = tmp / "manifest.csv"
            write_manifest(manifest_path, [manifest_row(graph_path)])

            args = module.parse_args(
                [
                    "--manifest",
                    str(manifest_path),
                    "--method-labels",
                    "2stage_val_mse",
                    "--max-cycle",
                    "3",
                    "--max-chain",
                    "4",
                ]
            )

            fake_common = types.SimpleNamespace(
                load_graph_records=mock.Mock(return_value=[fake_record(graph_path)]),
                dispose_graph_records=mock.Mock(),
            )
            fake_env = mock.Mock()
            fake_gp = types.SimpleNamespace(Env=mock.Mock(return_value=fake_env))
            fake_model = {
                "method": "2stage",
                "selection_metric": "validation_mse_loss",
                "theta": np.asarray([1.0, 2.0]),
                "selected_epoch": 5,
                "path": Path("model.npz"),
            }

            with mock.patch.object(
                module,
                "ensure_step1c_imports",
                return_value=(fake_common, object(), None, None, None),
            ), mock.patch.object(
                module, "resolve_run_dir", return_value=Path("run")
            ), mock.patch.object(
                module, "load_models", return_value=[fake_model]
            ), mock.patch.object(
                module, "method_label", return_value="2stage_val_mse"
            ), mock.patch.object(
                module,
                "rows_for_model_record",
                return_value=[
                    {
                        "method_label": "2stage_val_mse",
                        "solution_rank": 1,
                        "solution_edge_signature": "0|2",
                        "edge_count": 2,
                    }
                ],
            ), mock.patch.dict(sys.modules, {"gurobipy": fake_gp}):
                rows = module.compute_arc_density_rows(args)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["case_id"], "case_original")
            self.assertEqual(rows[0]["density_variant"], "original")
            self.assertEqual(rows[0]["solution_arc_key_signature"], "0->1|2->0")
            self.assertEqual(rows[0]["rank1_arc_key_signature"], "0->1|2->0")
            self.assertEqual(rows[0]["original_oracle_arc_key_signature"], "0->1|2->0")
            fake_common.load_graph_records.assert_called_once_with(
                [graph_path],
                fake_env,
                max_cycle=3,
                max_chain=4,
            )
            fake_common.dispose_graph_records.assert_called_once()
            fake_env.dispose.assert_called_once()


if __name__ == "__main__":
    unittest.main()
