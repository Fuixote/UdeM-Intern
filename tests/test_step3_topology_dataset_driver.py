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
    / "generate_step3_topology_dataset.py"
)


def load_driver():
    spec = importlib.util.spec_from_file_location("step3_topology_dataset_driver", DRIVER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_processed_graph(path, pair_count=20, ndd_count=2):
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


class Step3TopologyDatasetDriverTest(unittest.TestCase):
    def test_explicit_pairs_and_ndds_derive_subexperiment_and_prob_ndd(self):
        module = load_driver()

        args = module.parse_args(
            [
                "--pairs",
                "20",
                "--num-ndds",
                "2",
                "--instances",
                "100",
                "--seed",
                "20260619",
            ]
        )

        self.assertEqual(args.pairs, 20)
        self.assertEqual(args.num_ndds, 2)
        self.assertEqual(args.patients, 20)
        self.assertAlmostEqual(args.prob_ndd, 2 / 22)
        self.assertAlmostEqual(module.prob_ndd_from_counts(20, 2), 2 / 22)
        self.assertEqual(args.subexperiment, "pairs20_ndd2")
        self.assertIn("step3_pairs20_ndd2_raw_seed20260619", str(args.raw_output_dir))
        self.assertIn(
            "step3_pairs20_ndd2_step2c_poly_d8_mult_eps050_seed20260619",
            str(args.processed_output_dir),
        )
        self.assertIn("pairs20_ndd2", str(args.output_root))

    def test_build_commands_use_existing_raw_and_step2c_tools(self):
        module = load_driver()
        args = module.parse_args(
            [
                "--pairs",
                "20",
                "--num-ndds",
                "2",
                "--instances",
                "3",
                "--seed",
                "17",
                "--raw-output-dir",
                "/tmp/step3_raw",
                "--processed-output-dir",
                "/tmp/step3_processed",
                "--output-root",
                "/tmp/step3_exp",
            ]
        )

        commands = module.build_commands(args)
        expected_raw_batch = "/tmp/step3_raw/2026-06-19_000000__step3_pairs20_ndd2_raw_seed17"

        self.assertEqual(commands.raw[1], str(ROOT / "0-data-generation.py"))
        self.assertIn("--patients", commands.raw)
        self.assertIn("20", commands.raw)
        self.assertIn("--prob_ndd", commands.raw)
        self.assertIn("0.0909091", commands.raw)
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
        self.assertIn("/tmp/step3_processed", commands.process)
        self.assertIn("--label_mode", commands.process)
        self.assertIn("step2c_polynomial_degree_multiplicative_noise", commands.process)
        self.assertIn("--step2c_degree", commands.process)
        self.assertIn("8", commands.process)
        self.assertIn("--step2c_epsilon_bar", commands.process)
        self.assertIn("0.5", commands.process)

    def test_audit_processed_dataset_requires_expected_pairs_and_ndds(self):
        module = load_driver()
        with tempfile.TemporaryDirectory() as tmp:
            processed_dir = Path(tmp)
            write_processed_graph(processed_dir / "G-0.json", pair_count=20, ndd_count=3)

            with self.assertRaisesRegex(ValueError, "expected 2 NDD vertices"):
                module.audit_processed_dataset(
                    processed_dir,
                    expected_pairs=20,
                    expected_ndds=2,
                )

    def test_audit_processed_dataset_writes_counts_and_config(self):
        module = load_driver()
        with tempfile.TemporaryDirectory() as tmp:
            processed_dir = Path(tmp)
            write_processed_graph(processed_dir / "G-0.json", pair_count=20, ndd_count=2)
            write_processed_graph(processed_dir / "G-1.json", pair_count=20, ndd_count=2)

            summary = module.audit_processed_dataset(
                processed_dir,
                expected_pairs=20,
                expected_ndds=2,
            )

            self.assertEqual(summary["graph_count"], 2)
            self.assertEqual(summary["expected_pair_count"], 20)
            self.assertEqual(summary["expected_ndd_count"], 2)
            self.assertEqual(summary["min_ndd_count"], 2)
            self.assertEqual(summary["max_ndd_count"], 2)
            self.assertEqual(summary["rows"][0]["pair_count"], 20)
            self.assertEqual(summary["rows"][0]["ndd_count"], 2)
            self.assertEqual(summary["rows"][0]["total_vertices"], 22)
            self.assertEqual(summary["rows"][0]["arc_count"], 22)
            self.assertTrue((processed_dir / "audit_summary.json").exists())

    def test_manifest_and_generation_config_are_written_under_output_root(self):
        module = load_driver()
        with tempfile.TemporaryDirectory() as tmp:
            args = module.parse_args(
                [
                    "--pairs",
                    "20",
                    "--num-ndds",
                    "2",
                    "--output-root",
                    tmp,
                    "--dry-run",
                ]
            )
            commands = module.build_commands(args)

            dry_manifest = module.write_run_files(args, commands, audit=None)

            self.assertEqual(dry_manifest.name, "dry_run_manifest.json")
            self.assertTrue(dry_manifest.exists())
            self.assertTrue((Path(tmp) / "generation_config.json").exists())
            payload = json.loads(dry_manifest.read_text(encoding="utf-8"))
            self.assertEqual(payload["experiment"], "step3_pairs20_ndd2_step2c_poly_d8_mult_eps050")
            self.assertEqual(payload["config"]["pairs"], 20)
            self.assertEqual(payload["config"]["num_ndds"], 2)
            self.assertAlmostEqual(payload["config"]["prob_ndd"], 2 / 22)


if __name__ == "__main__":
    unittest.main()
