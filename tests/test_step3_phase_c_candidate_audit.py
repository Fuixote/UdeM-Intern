import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "surrogate_experiment_results"
    / "Step3"
    / "scripts"
    / "audit_phase_c_candidates.py"
)


def load_audit_module():
    spec = importlib.util.spec_from_file_location("step3_audit_phase_c_candidates", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def make_weight(path, theta, epoch):
    write_json(
        path,
        {
            "theta": theta,
            "selected_epoch": epoch,
            "selection_metric": path.stem,
            "selection_value": 1.0,
        },
    )


def make_test_summary(path, gap_2stage, gap_spoplus):
    write_json(
        path,
        [
            {
                "method": "2stage",
                "selected_epoch": 100,
                "theta_1": 1.0,
                "theta_2": 2.0,
                "test_mean_decision_gap": gap_2stage,
                "test_mean_normalized_gap": gap_2stage / 100.0,
            },
            {
                "method": "spoplus",
                "selected_epoch": 90,
                "theta_1": 2.0,
                "theta_2": 3.0,
                "test_mean_decision_gap": gap_spoplus,
                "test_mean_normalized_gap": gap_spoplus / 100.0,
            },
        ],
    )


def make_test_per_graph(path, method_gap_pairs):
    rows = []
    for method, gap in method_gap_pairs:
        rows.append(
            {
                "method": method,
                "selection_metric": "metric",
                "graph": "G-000000.json",
                "optimal_obj": "10.0",
                "achieved_obj": str(10.0 - gap),
                "gap": str(gap),
                "normalized_gap": str(gap / 10.0),
                "ratio": "0.9",
            }
        )
    write_csv(
        path,
        rows,
        [
            "method",
            "selection_metric",
            "graph",
            "optimal_obj",
            "achieved_obj",
            "gap",
            "normalized_gap",
            "ratio",
        ],
    )


class Step3PhaseCCandidateAuditTests(unittest.TestCase):
    def test_review_ids_merge_auto_candidates_and_seed_sensitive_extras(self):
        module = load_audit_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "phase_c_topology_ids.txt"
            path.write_text("G-1\nG-2\nG-47\n", encoding="utf-8")

            ids = module.review_topology_ids(path, extras=["G-47", "G-79"])

        self.assertEqual(ids, ["G-1", "G-2", "G-47", "G-79"])

    def test_audit_counts_data_theta_epoch_and_gap_variation(self):
        module = load_audit_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_dir = root / "datasets"
            runs_dir = root / "runs"
            status_csv = root / "status.csv"
            summary_csv = root / "summary.csv"

            write_csv(
                dataset_dir / "G-1" / "samples.csv",
                [
                    {
                        "topology_id": "G-1",
                        "split": "train",
                        "train_seed": "1",
                        "sample_idx": "0",
                        "label_seed": "101",
                        "path": "a",
                        "topology_hash": "t",
                        "label_hash": "h1",
                        "edge_count": "2",
                    },
                    {
                        "topology_id": "G-1",
                        "split": "train",
                        "train_seed": "1",
                        "sample_idx": "1",
                        "label_seed": "102",
                        "path": "b",
                        "topology_hash": "t",
                        "label_hash": "h2",
                        "edge_count": "2",
                    },
                    {
                        "topology_id": "G-1",
                        "split": "train",
                        "train_seed": "2",
                        "sample_idx": "0",
                        "label_seed": "201",
                        "path": "c",
                        "topology_hash": "t",
                        "label_hash": "h3",
                        "edge_count": "2",
                    },
                    {
                        "topology_id": "G-1",
                        "split": "validation",
                        "train_seed": "",
                        "sample_idx": "0",
                        "label_seed": "301",
                        "path": "v",
                        "topology_hash": "t",
                        "label_hash": "hv",
                        "edge_count": "2",
                    },
                    {
                        "topology_id": "G-1",
                        "split": "test",
                        "train_seed": "",
                        "sample_idx": "0",
                        "label_seed": "401",
                        "path": "te",
                        "topology_hash": "t",
                        "label_hash": "ht",
                        "edge_count": "2",
                    },
                ],
                [
                    "topology_id",
                    "split",
                    "train_seed",
                    "sample_idx",
                    "label_seed",
                    "path",
                    "topology_hash",
                    "label_hash",
                    "edge_count",
                ],
            )
            write_csv(
                status_csv,
                [
                    {
                        "topology_id": "G-1",
                        "train_seed": "1",
                        "status": "success",
                        "spoplus_improvement_gap": "1.0",
                        "spoplus_improvement_normalized_gap": "0.10",
                    },
                    {
                        "topology_id": "G-1",
                        "train_seed": "2",
                        "status": "success",
                        "spoplus_improvement_gap": "0.0",
                        "spoplus_improvement_normalized_gap": "0.00",
                    },
                ],
                [
                    "topology_id",
                    "train_seed",
                    "status",
                    "spoplus_improvement_gap",
                    "spoplus_improvement_normalized_gap",
                ],
            )
            write_csv(
                summary_csv,
                [
                    {
                        "topology_id": "G-1",
                        "phase_b_outcome": "helpful",
                        "complexity_bin": "sparse_simple",
                        "structural_type": "cycle_chain",
                        "landscape_regime": "proxy_hard",
                        "num_exchange_candidates": "8",
                    }
                ],
                [
                    "topology_id",
                    "phase_b_outcome",
                    "complexity_bin",
                    "structural_type",
                    "landscape_regime",
                    "num_exchange_candidates",
                ],
            )

            for seed, improvement in [(1, 1.0), (2, 0.0)]:
                run_dir = runs_dir / "G-1" / f"train_seed={seed:06d}"
                make_weight(
                    run_dir / "model_weights" / "2stage_best_by_validation_mse_loss.json",
                    [1.0 + seed, 2.0],
                    100,
                )
                make_weight(
                    run_dir / "model_weights" / "spoplus_best_by_validation_spoplus_loss.json",
                    [3.0 + seed, 4.0],
                    90,
                )
                make_test_summary(run_dir / "metrics" / "test_summary.json", 2.0, 2.0 - improvement)
                make_test_per_graph(
                    run_dir / "metrics" / "test_per_graph.csv",
                    [("2stage", 2.0), ("spoplus", 2.0 - improvement)],
                )

            summaries, seed_rows = module.audit_review_topologies(
                ["G-1"],
                dataset_dir=dataset_dir,
                runs_dir=runs_dir,
                status_csv=status_csv,
                phase_b_summary_csv=summary_csv,
            )

        self.assertEqual(len(summaries), 1)
        self.assertEqual(len(seed_rows), 2)
        row = summaries[0]
        self.assertEqual(row["topology_id"], "G-1")
        self.assertEqual(row["review_role"], "auto_phase_c")
        self.assertEqual(row["train_seed_count"], 2)
        self.assertEqual(row["train_label_hash_count"], 3)
        self.assertEqual(row["unique_theta_2stage"], 2)
        self.assertEqual(row["unique_theta_spoplus"], 2)
        self.assertEqual(row["unique_improvement_gap_count"], 2)
        self.assertEqual(row["spoplus_selected_epoch_values"], "90")
        self.assertGreater(float(row["fraction_better_wilson95_low"]), 0.0)

    def test_write_outputs_creates_review_audit_artifacts(self):
        module = load_audit_module()
        summaries = [{"topology_id": "G-1", "review_role": "auto_phase_c"}]
        seed_rows = [{"topology_id": "G-1", "train_seed": 1}]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            module.write_audit_outputs(output_dir, summaries, seed_rows)

            self.assertTrue((output_dir / "phase_c_review_topology_audit.csv").exists())
            self.assertTrue((output_dir / "phase_c_review_seed_audit.csv").exists())
            self.assertTrue((output_dir / "phase_c_review_audit_summary.json").exists())
            self.assertEqual(
                (output_dir / "phase_c_review_topology_ids.txt").read_text().splitlines(),
                ["G-1"],
            )


if __name__ == "__main__":
    unittest.main()
