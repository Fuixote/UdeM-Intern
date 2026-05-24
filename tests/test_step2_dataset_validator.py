import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


VALIDATOR_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "surrogate_experiment_results"
    / "Step2"
    / "validate_step2_processed_dataset.py"
)


def load_validator_module():
    spec = importlib.util.spec_from_file_location("step2_dataset_validator", VALIDATOR_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Step2DatasetValidatorTest(unittest.TestCase):
    def write_graph(self, root, name, labels):
        matches = []
        for idx, label in enumerate(labels):
            matches.append(
                {
                    "recipient": f"P{idx + 2}",
                    "ground_truth_label": label,
                    "latent_clean_linear_label": label,
                    "step2c_polynomial_label": label,
                    "step2c_multiplier": [1.0, 0.5, 1.5][idx % 3],
                }
            )
        payload = {
            "metadata": {
                "ground_truth_label_mode": "step2c_polynomial_degree_multiplicative_noise",
                "original_file": name.replace("G-", "genjson-"),
            },
            "data": {
                "P1": {
                    "type": "Pair",
                    "matches": matches,
                }
            },
        }
        (root / name).write_text(json.dumps(payload), encoding="utf-8")

    def test_summarize_processed_dataset_reports_label_quality_metrics(self):
        validator = load_validator_module()
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp)
            self.write_graph(dataset_dir, "G-0.json", [0.0, 2.0])
            self.write_graph(dataset_dir, "G-1.json", [4.0])

            summary, graph_rows = validator.summarize_processed_dataset(dataset_dir)

        self.assertEqual(summary["dataset"]["graph_count"], 2)
        self.assertEqual(summary["dataset"]["edge_count"], 3)
        self.assertEqual(summary["dataset"]["label_modes"], ["step2c_polynomial_degree_multiplicative_noise"])
        self.assertEqual(summary["labels"]["ground_truth_label"]["mean"], 2.0)
        self.assertEqual(summary["labels"]["ground_truth_label"]["min"], 0.0)
        self.assertEqual(summary["labels"]["ground_truth_label"]["max"], 4.0)
        self.assertAlmostEqual(summary["labels"]["ground_truth_label"]["fraction_zero"], 1 / 3)
        self.assertGreater(summary["correlations"]["latent_clean_linear_label_vs_ground_truth_label"], 0.99)
        self.assertAlmostEqual(
            summary["step2c"]["step2c_multiplier"]["mean"],
            (1.0 + 0.5 + 1.0) / 3,
            places=6,
        )
        self.assertEqual(len(graph_rows), 2)
        self.assertEqual(graph_rows[0]["edge_count"], 2)

    def test_write_diagnostics_creates_json_and_csv_artifacts(self):
        validator = load_validator_module()
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp) / "processed"
            dataset_dir.mkdir()
            self.write_graph(dataset_dir, "G-0.json", [1.0, 3.0])
            summary, graph_rows = validator.summarize_processed_dataset(dataset_dir)

            artifacts = validator.write_diagnostics(dataset_dir, summary, graph_rows)

            self.assertTrue(Path(artifacts["summary_json"]).exists())
            self.assertTrue(Path(artifacts["graph_csv"]).exists())
            loaded = json.loads(Path(artifacts["summary_json"]).read_text(encoding="utf-8"))
            self.assertEqual(loaded["dataset"]["edge_count"], 2)

    def test_strict_validation_reports_expected_count_and_label_mode_errors(self):
        validator = load_validator_module()
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp)
            self.write_graph(dataset_dir, "G-0.json", [0.0, 2.0])
            summary, _ = validator.summarize_processed_dataset(dataset_dir)

        errors = validator.validate_summary(
            summary,
            expected_graph_count=2,
            expected_label_mode="step2b_polynomial_degree_noiseless",
            max_fraction_zero=0.25,
        )

        self.assertIn("expected graph_count=2, got 1", errors)
        self.assertIn(
            "expected label mode step2b_polynomial_degree_noiseless, got step2c_polynomial_degree_multiplicative_noise",
            errors,
        )
        self.assertIn("fraction_zero 0.500000 exceeds max_fraction_zero 0.250000", errors)

    def test_strict_cli_fails_when_expected_graph_count_mismatches(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp)
            self.write_graph(dataset_dir, "G-0.json", [1.0, 2.0])

            import subprocess

            result = subprocess.run(
                [
                    "python",
                    str(VALIDATOR_SCRIPT),
                    str(dataset_dir),
                    "--strict",
                    "--expected_graph_count",
                    "2",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertEqual(result.returncode, 1)
        self.assertIn("expected graph_count=2, got 1", result.stderr)


if __name__ == "__main__":
    unittest.main()
