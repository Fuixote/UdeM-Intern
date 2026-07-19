import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = EXPERIMENT_ROOT / "scripts"
STEP3_SCRIPTS = ROOT / "surrogate_experiment_results" / "Step3" / "scripts"
for import_path in (SCRIPTS, STEP3_SCRIPTS):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder = load_module("step5_builder_test", SCRIPTS / "build_weak_label_artifacts.py")
auditor = load_module("step5_auditor_test", SCRIPTS / "audit_weak_label_artifacts.py")
planner = load_module("step5_planner_test", SCRIPTS / "plan_weak_label_jobs.py")
launcher = load_module("step5_launcher_test", SCRIPTS / "launch_weak_label_jobs.py")
reviewer = load_module("step5_reviewer_test", SCRIPTS / "review_weak_label_results.py")
locker = load_module("step5_locker_test", SCRIPTS / "lock_and_audit_topology_bank.py")
common = load_module("step5_common_test", STEP3_SCRIPTS / "fixed_topology_xy_common.py")
context_sampler = load_module(
    "step5_context_sampler_test", STEP3_SCRIPTS / "sample_fixed_topology_context.py"
)


TOPOLOGY_ROW = {
    "topology_id": "G-1",
    "source_path": "source.json",
    "num_vertices": "3",
    "topology_hash": "topology-hash",
    "arc_order_hash": "arc-order-hash",
    "feasible_set_hash": "feasible-set-hash",
    "template_path": "template.json",
}
TOPOLOGY_TEMPLATE = {
    "topology_id": "G-1",
    "topology_hash": "topology-hash",
    "arc_order_hash": "arc-order-hash",
    "feasible_set_hash": "feasible-set-hash",
}
GENERATOR_CONFIG = {
    "generator_version": "step5_test_generator",
    "label": {"epsilon_bar": 0.5},
}
REGIME = "step2c_poly_d8_mult_eps050"


def fake_sample(index, *, test=False):
    namespace = "screen_test" if test else "screen_train"
    train_seed = None if test else 42
    prefix = "T" if test else "F"
    return {
        "payload": {"id": index},
        "X": np.asarray([[float(index), float(index + 1)]], dtype=float),
        "y": np.asarray([float(index)], dtype=float),
        "manifest": {
            "sample_id": f"{prefix}-{index:06d}",
            "topology_id": "G-1",
            "regime": REGIME,
            "split_namespace": namespace,
            "train_seed": train_seed,
            "sample_index": index,
            "x_hash": f"x-{prefix}-{index}",
            "label_hash": f"y-{prefix}-{index}",
        },
    }


def materialize_fake_bundle(root, *, sample_size=5, test_size=3):
    paths = builder.artifact_paths(
        root,
        regime=REGIME,
        topology_id="G-1",
        data_seed=42,
        sample_size=sample_size,
    )
    training_size, validation_size = builder.validate_sample_size(sample_size)
    provenance = builder._provenance(
        template=TOPOLOGY_TEMPLATE,
        generator_config=GENERATOR_CONFIG,
        experiment_version="step5-test-v1",
        master_label_seed=123,
    )
    protocol_record = builder._protocol_record(
        data_seed=42,
        sample_size=sample_size,
        training_size=training_size,
        validation_size=validation_size,
        test_size=test_size,
        protocol="screen",
    )
    test_manifest = builder.write_test_artifacts(
        samples=[fake_sample(index, test=True) for index in range(test_size)],
        paths=paths,
        topology_id="G-1",
        regime=REGIME,
        protocol="screen",
        provenance=provenance,
        protocol_record=protocol_record,
    )
    builder.write_fit_artifacts(
        fit_samples=[fake_sample(index) for index in range(sample_size)],
        paths=paths,
        topology_id="G-1",
        regime=REGIME,
        data_seed=42,
        sample_size=sample_size,
        protocol="screen",
        test_manifest=test_manifest,
        provenance=provenance,
        protocol_record=protocol_record,
    )
    return paths


class Step5WeakLabelScriptTests(unittest.TestCase):
    def test_sample_size_50_is_exactly_40_train_and_10_validation(self):
        split = builder.split_fit_samples([fake_sample(index) for index in range(50)])

        self.assertEqual(split["training_size"], 40)
        self.assertEqual(split["validation_size"], 10)
        self.assertEqual(len(split["training_indices"]), 40)
        self.assertEqual(len(split["validation_indices"]), 10)
        self.assertFalse(set(split["training_indices"]) & set(split["validation_indices"]))
        self.assertEqual(set(split["training_indices"]) | set(split["validation_indices"]), set(range(50)))

    def test_bundle_audit_accepts_hash_consistent_4_to_1_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            materialize_fake_bundle(root)

            result = auditor.audit_bundle(
                TOPOLOGY_ROW,
                output_root=root,
                regime=REGIME,
                protocol="screen",
                data_seed=42,
                sample_size=5,
                test_size=3,
            )

        self.assertTrue(result["passed"], result["failures"])
        self.assertEqual(result["training_size"], 4)
        self.assertEqual(result["validation_size"], 1)

    def test_planner_emits_dry_run_only_command_with_train_size_40(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            materialize_fake_bundle(root, sample_size=50, test_size=3)

            plan = planner.build_plan(
                [TOPOLOGY_ROW],
                output_root=root,
                regime=REGIME,
                protocol="screen",
                data_seed=42,
                sample_size=50,
                test_size=3,
            )

        self.assertTrue(plan["passed"])
        self.assertEqual(plan["job_count"], 1)
        command = plan["jobs"][0]["run_one_job_command"]
        self.assertIn("--train-size 40", command)
        self.assertIn("--sample-size 50", command)
        self.assertIn("--max-epochs 1500", command)
        self.assertIn("--dry-run", command)
        self.assertNotIn("--execute", command)

    def test_planner_resolves_both_eval_path_conventions(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            eval_dir = project_root / "results" / "G-1" / "data_seed=000042"
            eval_dir.mkdir(parents=True)
            eval_manifest = eval_dir / "eval_manifest.json"
            validation = eval_dir / "validation.npz"
            validation.touch()
            test = project_root / "results" / "G-1" / "test" / "test.npz"
            test.parent.mkdir(parents=True)
            test.touch()

            resolved_validation = planner._resolve_eval_path(
                eval_manifest,
                "validation.npz",
                project_root=project_root,
            )
            resolved_test = planner._resolve_eval_path(
                eval_manifest,
                "results/G-1/test/test.npz",
                project_root=project_root,
            )

        self.assertEqual(resolved_validation, validation)
        self.assertEqual(resolved_test, test)

    def test_launcher_requires_dry_run_plan_before_conversion(self):
        converted = launcher.command_for_execute(
            "python run_one_job.py --output-dir old/jobs/G-1 --dry-run",
            Path("new/jobs/G-1"),
        )
        self.assertIn("--execute", converted)
        self.assertNotIn("--dry-run", converted)
        self.assertEqual(converted[converted.index("--output-dir") + 1], "new/jobs/G-1")
        with self.assertRaises(ValueError):
            launcher.command_for_execute(
                "python run_one_job.py --output-dir old/jobs/G-1",
                Path("new/jobs/G-1"),
            )

    def test_reviewer_exports_helpful_weak_label_and_integrity_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job_dir = reviewer.expected_job_dir(
                root,
                regime=REGIME,
                topology_id="G-1",
                data_seed=42,
                sample_size=50,
            )
            metrics_dir = job_dir / "evaluation" / "metrics"
            metrics_dir.mkdir(parents=True)
            (job_dir / "job_status.json").write_text(
                json.dumps(
                    {
                        "job_id": "job-1",
                        "status": "success",
                        "2stage status": "success",
                        "SPO+ status": "success",
                        "evaluation status": "success",
                    }
                ),
                encoding="utf-8",
            )
            (job_dir / "paired_job_manifest.json").write_text(
                json.dumps(
                    {
                        "job_id": "job-1",
                        "topology_id": "G-1",
                        "regime": REGIME,
                        "protocol": "screen",
                        "train_seed": 42,
                        "sample_size": 50,
                        "training_size": 40,
                        "validation_size": 10,
                        "trainer_train_size_arg": 40,
                        "theta_seed": 42,
                        "gurobi_seed": 42,
                        "max_epochs": 1500,
                        "metric_stride": 1,
                        "early_stop_patience": 20,
                        "early_stop_min_delta": 0.0001,
                        "test_hash": "test-hash",
                        "train_prefix_hash": "train-hash",
                        "validation_hash": "validation-hash",
                    }
                ),
                encoding="utf-8",
            )
            (metrics_dir / "test_summary.json").write_text(
                json.dumps(
                    [
                        {
                            "method": "2stage",
                            "test_mean_decision_gap": 0.35,
                            "test_mean_normalized_gap": 0.2,
                        },
                        {
                            "method": "spoplus",
                            "test_mean_decision_gap": 0.15,
                            "test_mean_normalized_gap": 0.1,
                            "paired_mean_improvement_over_2stage": 0.2,
                            "fraction_improved_over_2stage": 0.7,
                        },
                    ]
                ),
                encoding="utf-8",
            )

            audit = reviewer.review_results(
                [TOPOLOGY_ROW],
                output_root=root,
                output_dir=root / "results",
                regime=REGIME,
                protocol="screen",
                data_seed=42,
                sample_size=50,
                test_size=1000,
                theta_seed=42,
                gurobi_seed=42,
            )
            with (root / "results" / "weak_label_topology_summary.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))

        self.assertTrue(audit["passed"], audit["failures"])
        self.assertEqual(audit["job_rows"], 1)
        self.assertEqual(audit["success"], 1)
        self.assertEqual(audit["label_rows"], 1)
        self.assertEqual(rows[0]["weak_label_class"], "helpful")
        self.assertAlmostEqual(float(rows[0]["delta"]), 0.2)

    def test_topology_bank_audit_checks_locked_hash_fields_and_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template = root / "template.json"
            source = root / "source.json"
            bank = root / "bank.csv"
            template.write_text(json.dumps(TOPOLOGY_TEMPLATE), encoding="utf-8")
            source.write_text(json.dumps({"data": {}}), encoding="utf-8")
            row = {
                **TOPOLOGY_ROW,
                "template_path": "template.json",
                "source_path": "source.json",
            }
            with bank.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)

            result = locker.audit_topology_bank(
                bank,
                project_root=root,
                expected_count=1,
            )

        self.assertTrue(result["passed"], result["failures"])
        self.assertEqual(result["unique_counts"]["topology_hash"], 1)

    def test_locked_configs_match_the_pre_smoke_protocol(self):
        context_path = EXPERIMENT_ROOT / "configs" / "context_generator.locked.yaml"
        experiment_path = EXPERIMENT_ROOT / "configs" / "experiment.yaml"

        generator = context_sampler.load_generator_config(context_path)
        context_sampler.validate_generator_config(generator)
        experiment = context_sampler.load_simple_yaml(experiment_path)

        self.assertEqual(generator["status"], "locked")
        self.assertEqual(generator["method"], "bounded_additive_uniform")
        self.assertEqual(generator["label"]["epsilon_bar"], 0.5)
        self.assertEqual(experiment["seeds"]["master_label_seed"], 20260719)
        self.assertEqual(experiment["seeds"]["data_seed"], 42)
        self.assertEqual(experiment["training"]["max_epochs"], 1500)
        self.assertEqual(experiment["training"]["metric_stride"], 1)
        self.assertEqual(experiment["training"]["early_stop_patience"], 20)
        self.assertEqual(experiment["training"]["early_stop_min_delta"], 0.0001)
        self.assertEqual(
            common.generator_config_hash(generator),
            experiment["paths"]["context_generator_config_hash"],
        )


if __name__ == "__main__":
    unittest.main()
