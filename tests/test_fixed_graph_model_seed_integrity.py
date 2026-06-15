from __future__ import annotations

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
    / "audit_fixed_graph_model_seed_integrity.py"
)


def solution_row(
    graph_id: str,
    subset_seed: int,
    method_label: str,
    rank: int,
    theta_1: float,
    theta_2: float,
    signature: str,
    normalized_gap: float,
) -> dict[str, str]:
    return {
        "regime": "step2c_poly_d8_mult_eps050",
        "graph_id": graph_id,
        "subset_seed": str(subset_seed),
        "method_label": method_label,
        "theta_1": str(theta_1),
        "theta_2": str(theta_2),
        "solution_rank": str(rank),
        "solution_edge_signature": signature,
        "normalized_gap_to_oracle": str(normalized_gap),
        "model_path": (
            "surrogate_experiment_results/Step2_resampling/phase1_runs/"
            f"step2c_poly_d8_mult_eps050/subset_seed={subset_seed}/model_weights/"
            + (
                "2stage_best_by_validation_mse_loss.npz"
                if method_label == "2stage_val_mse"
                else "spoplus_best_by_validation_spoplus_loss.npz"
            )
        ),
    }


class FixedGraphModelSeedIntegrityTests(unittest.TestCase):
    def load_module(self):
        self.assertTrue(SCRIPT_PATH.exists(), f"Missing script: {SCRIPT_PATH}")
        spec = importlib.util.spec_from_file_location(
            "audit_fixed_graph_model_seed_integrity",
            SCRIPT_PATH,
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def test_build_integrity_rows_hashes_sources_and_prediction_vectors(self):
        module = self.load_module()
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            run_root = (
                project_root
                / "surrogate_experiment_results"
                / "Step2_resampling"
                / "phase1_runs"
                / "step2c_poly_d8_mult_eps050"
            )
            for seed in [0, 1]:
                run_dir = run_root / f"subset_seed={seed}"
                weights_dir = run_dir / "model_weights"
                weights_dir.mkdir(parents=True)
                (weights_dir / "2stage_best_by_validation_mse_loss.npz").write_bytes(
                    f"two-stage-{seed}".encode("utf-8")
                )
                (weights_dir / "spoplus_best_by_validation_spoplus_loss.npz").write_bytes(
                    f"spoplus-{seed}".encode("utf-8")
                )
                with (run_dir / "train_subset.json").open("w", encoding="utf-8") as handle:
                    json.dump(
                        [
                            {"graph_id": seed, "path": f"G-{seed}.json"},
                            {"graph_id": seed + 10, "path": f"G-{seed + 10}.json"},
                        ],
                        handle,
                    )

            rows = []
            for seed, theta in [(0, (1.0, 0.0)), (1, (2.0, 0.0))]:
                rows.extend(
                    [
                        solution_row("G-1.json", seed, "2stage_val_mse", 1, *theta, "a", 0.10),
                        solution_row("G-1.json", seed, "2stage_val_mse", 2, *theta, "b", 0.20),
                        solution_row("G-1.json", seed, "2stage_val_mse", 3, *theta, "c", 0.30),
                    ]
                )

            integrity_rows = module.build_integrity_rows(
                rows,
                feature_matrices={"G-1.json": [[1.0, 0.0], [2.0, 0.0]]},
                project_root=project_root,
                discovery_seeds={"G-1.json": 0},
                top_k=3,
            )

            self.assertEqual(len(integrity_rows), 2)
            discovery, candidate = integrity_rows
            self.assertEqual(discovery["subset_seed"], 0)
            self.assertEqual(candidate["subset_seed"], 1)
            self.assertNotEqual(
                discovery["checkpoint_sha256"], candidate["checkpoint_sha256"]
            )
            self.assertNotEqual(
                discovery["train_subset_hash"], candidate["train_subset_hash"]
            )
            self.assertNotEqual(
                discovery["prediction_vector_hash"], candidate["prediction_vector_hash"]
            )
            self.assertAlmostEqual(discovery["prediction_l2_from_discovery_seed"], 0.0)
            self.assertAlmostEqual(
                candidate["prediction_l2_from_discovery_seed"], 2.2360679, places=6
            )
            self.assertEqual(candidate["rank1_signature"], "a")
            self.assertEqual(candidate["rank2_signature"], "b")
            self.assertNotEqual(candidate["top3_signature_hash"], "")

    def test_summarize_integrity_rows_reports_unique_counts(self):
        module = self.load_module()
        rows = [
            {
                "regime": "step2c_poly_d8_mult_eps050",
                "graph_id": "G-1.json",
                "method_label": "2stage_val_mse",
                "checkpoint_sha256": "ckpt-a",
                "train_subset_hash": "train-a",
                "prediction_vector_hash": "pred-a",
                "rank1_signature": "rank1",
                "rank2_signature": "rank2",
                "top3_signature_hash": "top-a",
                "prediction_l2_from_discovery_seed": 0.0,
                "prediction_corr_with_discovery_seed": 1.0,
            },
            {
                "regime": "step2c_poly_d8_mult_eps050",
                "graph_id": "G-1.json",
                "method_label": "2stage_val_mse",
                "checkpoint_sha256": "ckpt-b",
                "train_subset_hash": "train-b",
                "prediction_vector_hash": "pred-b",
                "rank1_signature": "rank1",
                "rank2_signature": "rank2",
                "top3_signature_hash": "top-a",
                "prediction_l2_from_discovery_seed": 3.0,
                "prediction_corr_with_discovery_seed": 0.99,
            },
        ]

        summary = module.summarize_integrity_rows(rows, top_k=3)

        self.assertEqual(len(summary), 1)
        row = summary[0]
        self.assertEqual(row["unique_checkpoint_hashes"], 2)
        self.assertEqual(row["unique_train_subset_hashes"], 2)
        self.assertEqual(row["unique_prediction_hashes"], 2)
        self.assertEqual(row["unique_rank1_signatures"], 1)
        self.assertEqual(row["unique_rank2_signatures"], 1)
        self.assertEqual(row["unique_top3_signature_hashes"], 1)
        self.assertAlmostEqual(row["mean_prediction_l2_from_discovery_seed"], 1.5)
        self.assertAlmostEqual(row["max_prediction_l2_from_discovery_seed"], 3.0)


if __name__ == "__main__":
    unittest.main()
