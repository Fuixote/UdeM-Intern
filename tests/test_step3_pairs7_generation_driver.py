import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DRIVER_PATH = (
    ROOT
    / "surrogate_experiment_results"
    / "Step3"
    / "scripts"
    / "generate_pairs7_step2c_dataset.py"
)


def load_driver():
    spec = importlib.util.spec_from_file_location("step3_pairs7_driver", DRIVER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_processed_graph(path, pair_count=7, ndd_count=2):
    data = {}
    for index in range(pair_count):
        data[str(index)] = {
            "type": "Pair",
            "matches": [
                {"recipient": str((index + 1) % pair_count), "ground_truth_label": 1.0}
            ],
        }
    for offset in range(ndd_count):
        data[str(pair_count + offset)] = {
            "type": "NDD",
            "matches": [{"recipient": "0", "ground_truth_label": 1.0}],
        }
    path.write_text(
        json.dumps(
            {
                "metadata": {
                    "total_vertices": pair_count + ndd_count,
                    "ground_truth_label_mode": "step2c_polynomial_degree_multiplicative_noise",
                },
                "data": data,
            }
        ),
        encoding="utf-8",
    )


class Step3Pairs7GenerationDriverTest(unittest.TestCase):
    def test_default_args_lock_pairs7_high_ndd_step2c_settings(self):
        module = load_driver()

        args = module.parse_args([])

        self.assertEqual(args.patients, 7)
        self.assertEqual(args.prob_ndd, 0.20)
        self.assertEqual(args.step2c_degree, 8)
        self.assertEqual(args.step2c_kappa, 3.0)
        self.assertEqual(args.step2c_delta, 1e-12)
        self.assertEqual(args.step2c_epsilon_bar, 0.5)
        self.assertEqual(args.label_seed, 20260619)

    def test_build_commands_target_existing_generators(self):
        module = load_driver()
        args = module.parse_args(
            [
                "--instances",
                "3",
                "--seed",
                "17",
                "--raw-output-dir",
                "/tmp/step3_raw",
                "--processed-output-dir",
                "/tmp/step3_processed",
            ]
        )

        commands = module.build_commands(args)
        expected_raw_batch = "/tmp/step3_raw/2026-06-19_000000__step3_pairs7_raw_seed17"

        self.assertEqual(commands.raw[1], str(ROOT / "0-data-generation.py"))
        self.assertIn("--patients", commands.raw)
        self.assertIn("7", commands.raw)
        self.assertIn("--prob_ndd", commands.raw)
        self.assertIn("0.2", commands.raw)
        self.assertIn("--instances", commands.raw)
        self.assertIn("3", commands.raw)
        self.assertIn("--output_dir", commands.raw)
        self.assertIn(expected_raw_batch, commands.raw)
        self.assertEqual(
            commands.process[1],
            str(
                ROOT
                / "surrogate_experiment_results"
                / "Step2"
                / "Step2c_polynomial_degree_multiplicative_noise"
                / "data-processing.py"
            ),
        )
        self.assertIn(expected_raw_batch, commands.process)
        self.assertIn("--label_mode", commands.process)
        self.assertIn("step2c_polynomial_degree_multiplicative_noise", commands.process)
        self.assertIn("--step2c_degree", commands.process)
        self.assertIn("8", commands.process)
        self.assertIn("--step2c_epsilon_bar", commands.process)
        self.assertIn("0.5", commands.process)

    def test_audit_processed_dataset_requires_seven_pairs(self):
        module = load_driver()
        with tempfile.TemporaryDirectory() as tmp:
            processed_dir = Path(tmp)
            write_processed_graph(processed_dir / "G-0.json", pair_count=6, ndd_count=2)

            with self.assertRaisesRegex(ValueError, "expected 7 Pair vertices"):
                module.audit_processed_dataset(processed_dir, expected_pairs=7)

    def test_audit_processed_dataset_writes_counts(self):
        module = load_driver()
        with tempfile.TemporaryDirectory() as tmp:
            processed_dir = Path(tmp)
            write_processed_graph(processed_dir / "G-0.json", pair_count=7, ndd_count=2)
            write_processed_graph(processed_dir / "G-1.json", pair_count=7, ndd_count=1)

            summary = module.audit_processed_dataset(processed_dir, expected_pairs=7)

            self.assertEqual(summary["graph_count"], 2)
            self.assertEqual(summary["expected_pair_count"], 7)
            self.assertEqual(summary["min_ndd_count"], 1)
            self.assertEqual(summary["max_ndd_count"], 2)
            self.assertEqual(summary["rows"][0]["pair_count"], 7)
            self.assertEqual(summary["rows"][0]["ndd_count"], 2)
            self.assertEqual(summary["rows"][0]["total_vertices"], 9)
            self.assertEqual(summary["rows"][0]["arc_count"], 9)
            self.assertTrue((processed_dir / "audit_summary.json").exists())

    def test_dry_run_manifest_does_not_overwrite_real_run_manifest(self):
        module = load_driver()
        with tempfile.TemporaryDirectory() as tmp:
            module.EXPERIMENT_ROOT = Path(tmp) / "pairs7"
            args = module.parse_args(["--dry-run"])
            commands = module.build_commands(args)

            dry_run_manifest = module.write_manifest(args, commands, audit=None)

            args.dry_run = False
            real_manifest = module.write_manifest(args, commands, audit={"graph_count": 0})

            self.assertEqual(dry_run_manifest.name, "dry_run_manifest.json")
            self.assertEqual(real_manifest.name, "run_manifest.json")
            self.assertTrue(dry_run_manifest.exists())
            self.assertTrue(real_manifest.exists())


if __name__ == "__main__":
    unittest.main()
