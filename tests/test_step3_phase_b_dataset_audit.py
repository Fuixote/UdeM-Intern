import csv
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
    / "audit_phase_b_materialized_datasets.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "audit_phase_b_materialized_datasets",
        SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class Step3PhaseBDatasetAuditTests(unittest.TestCase):
    def test_audit_accepts_complete_materialized_topology(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "datasets"
            topology = base / "G-1"
            expected = module.ExpectedLayout(
                train_seed_count=2,
                train_sample_count=4,
                validation_sample_count=1,
                test_sample_count=3,
            )

            for split, count in [("validation", 1), ("test", 3)]:
                for idx in range(count):
                    write_json(topology / split / f"G-{idx:06d}.json", {"split": split})
            for seed in [1, 2]:
                for idx in range(4):
                    write_json(
                        topology / f"train_seed={seed:06d}" / "train" / f"G-{idx:06d}.json",
                        {"split": "train"},
                    )

            manifest = {
                "topology_id": "G-1",
                "topology_hash": "processed-hash",
                "topology_bank_hash": "bank-hash",
                "feasible_set_hash": "feasible-hash",
                "train_seed_count": 2,
                "train_sample_count": 4,
                "validation_sample_count": 1,
                "test_sample_count": 3,
                "num_sample_files": 12,
            }
            write_json(topology / "dataset_manifest.json", manifest)
            sample_rows = []
            for idx in range(1):
                sample_rows.append(
                    {
                        "topology_id": "G-1",
                        "split": "validation",
                        "train_seed": "",
                        "sample_idx": idx,
                        "label_seed": 200,
                        "path": str(topology / "validation" / f"G-{idx:06d}.json"),
                        "topology_hash": "processed-hash",
                        "label_hash": f"v{idx}",
                        "edge_count": 3,
                    }
                )
            for idx in range(3):
                sample_rows.append(
                    {
                        "topology_id": "G-1",
                        "split": "test",
                        "train_seed": "",
                        "sample_idx": idx,
                        "label_seed": 300 + idx,
                        "path": str(topology / "test" / f"G-{idx:06d}.json"),
                        "topology_hash": "processed-hash",
                        "label_hash": f"e{idx}",
                        "edge_count": 3,
                    }
                )
            for seed in [1, 2]:
                for idx in range(4):
                    sample_rows.append(
                        {
                            "topology_id": "G-1",
                            "split": "train",
                            "train_seed": seed,
                            "sample_idx": idx,
                            "label_seed": 1000 + seed * 10 + idx,
                            "path": str(
                                topology
                                / f"train_seed={seed:06d}"
                                / "train"
                                / f"G-{idx:06d}.json"
                            ),
                            "topology_hash": "processed-hash",
                            "label_hash": f"t{seed}-{idx}",
                            "edge_count": 3,
                        }
                    )
            write_csv(topology / "samples.csv", sample_rows, module.SAMPLE_FIELDS)
            write_csv(
                base / "phase_b_dataset_index.csv",
                [
                    {
                        "topology_id": "G-1",
                        "output_dir": str(topology),
                        "train_seed_count": 2,
                        "training_size_budget": 5,
                        "train_sample_count": 4,
                        "validation_sample_count": 1,
                        "test_sample_count": 3,
                        "topology_hash": "processed-hash",
                        "topology_bank_hash": "bank-hash",
                        "feasible_set_hash": "feasible-hash",
                        "status": "materialized",
                    }
                ],
                module.INDEX_FIELDS,
            )
            write_json(
                base / "phase_b_dataset_manifest.json",
                {
                    "status": "materialized",
                    "num_topologies": 1,
                    "train_seed_count": 2,
                    "train_sample_count": 4,
                    "validation_sample_count": 1,
                    "test_sample_count": 3,
                },
            )

            result = module.audit_phase_b_dataset(base, expected=expected)

            self.assertTrue(result["passed"])
            self.assertEqual(result["summary"]["num_topologies"], 1)
            self.assertEqual(result["summary"]["total_json_files"], 12)
            self.assertEqual(result["summary"]["num_failed_topologies"], 0)

    def test_audit_reports_missing_train_seed_and_hash_mismatch(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "datasets"
            topology = base / "G-1"
            expected = module.ExpectedLayout(
                train_seed_count=2,
                train_sample_count=4,
                validation_sample_count=1,
                test_sample_count=3,
            )
            write_json(topology / "validation" / "G-000000.json", {})
            for idx in range(3):
                write_json(topology / "test" / f"G-{idx:06d}.json", {})
            for idx in range(4):
                write_json(topology / "train_seed=000001" / "train" / f"G-{idx:06d}.json", {})
            write_json(
                topology / "dataset_manifest.json",
                {
                    "topology_id": "G-1",
                    "topology_hash": "processed-hash",
                    "topology_bank_hash": "bank-hash",
                    "feasible_set_hash": "wrong-feasible-hash",
                    "train_seed_count": 2,
                    "train_sample_count": 4,
                    "validation_sample_count": 1,
                    "test_sample_count": 3,
                    "num_sample_files": 8,
                },
            )
            write_csv(
                topology / "samples.csv",
                [
                    {
                        "topology_id": "G-1",
                        "split": "validation",
                        "train_seed": "",
                        "sample_idx": 0,
                        "label_seed": 200,
                        "path": str(topology / "validation" / "G-000000.json"),
                        "topology_hash": "different-hash",
                        "label_hash": "v",
                        "edge_count": 3,
                    }
                ],
                module.SAMPLE_FIELDS,
            )
            write_csv(
                base / "phase_b_dataset_index.csv",
                [
                    {
                        "topology_id": "G-1",
                        "output_dir": str(topology),
                        "train_seed_count": 2,
                        "training_size_budget": 5,
                        "train_sample_count": 4,
                        "validation_sample_count": 1,
                        "test_sample_count": 3,
                        "topology_hash": "processed-hash",
                        "topology_bank_hash": "bank-hash",
                        "feasible_set_hash": "feasible-hash",
                        "status": "materialized",
                    }
                ],
                module.INDEX_FIELDS,
            )
            write_json(
                base / "phase_b_dataset_manifest.json",
                {
                    "status": "materialized",
                    "num_topologies": 1,
                    "train_seed_count": 2,
                    "train_sample_count": 4,
                    "validation_sample_count": 1,
                    "test_sample_count": 3,
                },
            )

            result = module.audit_phase_b_dataset(base, expected=expected)

            self.assertFalse(result["passed"])
            failures = result["topology_rows"][0]["failure_reasons"]
            self.assertIn("train_seed_dir_count=1 expected=2", failures)
            self.assertIn("feasible_set_hash mismatch index=feasible-hash manifest=wrong-feasible-hash", failures)
            self.assertIn("sample_topology_hash_count=1 expected=processed-hash", failures)


if __name__ == "__main__":
    unittest.main()
