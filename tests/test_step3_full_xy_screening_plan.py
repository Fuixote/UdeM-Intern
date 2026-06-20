import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "surrogate_experiment_results" / "Step3" / "scripts"
BUILDER_SCRIPT = SCRIPT_DIR / "build_topology_bank.py"
PLAN_SCRIPT = SCRIPT_DIR / "plan_full_xy_screening.py"
PILOT_SCRIPT = SCRIPT_DIR / "run_full_xy_screening_pilot.py"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def payload(utility_offset=0.0):
    return {
        "metadata": {
            "source_file": "G-plan.json",
            "step2c_degree": 8,
            "step2c_kappa": 3.0,
            "step2c_delta": 1e-12,
            "clean_linear_utility_weight": 10.0,
            "clean_linear_cpra_weight": 5.0,
        },
        "data": {
            "0": {
                "type": "Pair",
                "patient": {"age": 40, "bloodtype": "A", "cPRA": 0.2, "hasBloodCompatibleDonor": True},
                "donors": [{"original_node_id": "0", "dage": 35, "bloodtype": "A"}],
                "matches": [{"recipient": "1", "utility": 60.0 + utility_offset, "recipient_cpra": 0.5}],
            },
            "1": {
                "type": "Pair",
                "patient": {"age": 50, "bloodtype": "B", "cPRA": 0.5, "hasBloodCompatibleDonor": False},
                "donors": [{"original_node_id": "1", "dage": 45, "bloodtype": "O"}],
                "matches": [{"recipient": "0", "utility": 70.0 + utility_offset, "recipient_cpra": 0.2}],
            },
            "2": {
                "type": "NDD",
                "donor": {"original_node_id": "2", "dage": 55, "bloodtype": "O"},
                "matches": [
                    {"recipient": "0", "utility": 40.0 + utility_offset, "recipient_cpra": 0.2},
                    {"recipient": "1", "utility": 50.0 + utility_offset, "recipient_cpra": 0.5},
                ],
            },
        },
    }


def config():
    return {
        "status": "pilot_not_locked",
        "generator_version": "plan-test-v1",
        "method": "bounded_additive_uniform",
        "recipient_cpra": {"lower": 0.0, "upper": 1.0, "half_width": 0.05},
        "utility": {"lower": 0.0, "upper": 100.0, "half_width": 5.0},
        "label": {"epsilon_bar": 0.5},
    }


def write_json(path: Path, payload_obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload_obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_topology_files(root: Path, topology_ids=("G-1", "G-2")):
    builder = load_module(BUILDER_SCRIPT, f"plan_builder_{id(root)}")
    topology_root = root / "topologies"
    payload_root = root / "payloads"
    selected_rows = []
    for index, topology_id in enumerate(topology_ids):
        base_payload = payload(utility_offset=float(index))
        template = builder.build_topology_template(topology_id, base_payload, max_cycle=3, max_chain=4)
        write_json(topology_root / topology_id / "template.json", template)
        write_json(payload_root / f"{topology_id}.json", base_payload)
        selected_rows.append(
            {
                "selection_rank": index + 1,
                "topology_id": topology_id,
                "complexity_bin": "tiny",
                "structural_type": "cycle_chain",
                "landscape_regime": "neutral",
                "screening_score": "0.1",
            }
        )
    selected_csv = root / "phase_b_topologies.csv"
    write_csv(selected_csv, selected_rows)
    config_path = root / "context_config.json"
    write_json(config_path, config())
    return selected_csv, topology_root, payload_root, config_path


class Step3FullXYScreeningPlanTests(unittest.TestCase):
    def test_plan_reads_selected_csv_counts_jobs_and_does_not_materialize_npz_or_runs(self):
        planner = load_module(PLAN_SCRIPT, "full_xy_plan")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selected_csv, topology_root, payload_root, config_path = build_topology_files(root, ("G-1", "G-2"))
            output_root = root / "plan"

            plan = planner.build_screening_plan(
                selected_topologies_csv=selected_csv,
                topology_root=topology_root,
                base_payload_dir=payload_root,
                output_root=output_root,
                regime="step2c_poly_d8_mult_eps050",
                train_seed_start=1,
                train_seed_count=3,
                train_size=50,
                validation_size=100,
                test_size=500,
                max_topologies=None,
                protocol="screen",
                context_generator_config=config_path,
            )
            planner.write_plan_outputs(plan, output_root)

            self.assertEqual(plan["job_count"], 6)
            self.assertEqual(len(plan["jobs"]), 6)
            self.assertTrue((output_root / "screening_plan.json").exists())
            self.assertTrue((output_root / "screening_jobs.csv").exists())
            self.assertTrue((output_root / "selected_topology_summary.csv").exists())
            self.assertFalse(list(output_root.rglob("*.npz")))
            self.assertFalse((output_root / "runs").exists())
            for job in plan["jobs"]:
                self.assertEqual(job["protocol"], "screen")
                self.assertEqual(job["train_size"], 50)
                self.assertIn("--protocol screen", job["run_one_job_command"])
                self.assertIsNone(job["expected_train_prefix_hash"])
                self.assertIsNone(job["validation_hash"])
                self.assertIsNone(job["test_hash"])

    def test_plan_max_topologies_limits_job_count(self):
        planner = load_module(PLAN_SCRIPT, "full_xy_plan_max_topologies")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selected_csv, topology_root, payload_root, config_path = build_topology_files(
                root,
                ("G-1", "G-2", "G-3"),
            )

            plan = planner.build_screening_plan(
                selected_topologies_csv=selected_csv,
                topology_root=topology_root,
                base_payload_dir=payload_root,
                output_root=root / "plan",
                regime="step2c_poly_d8_mult_eps050",
                train_seed_start=5,
                train_seed_count=2,
                train_size=50,
                validation_size=100,
                test_size=500,
                max_topologies=2,
                protocol="screen",
                context_generator_config=config_path,
            )

            self.assertEqual(plan["topology_count"], 2)
            self.assertEqual(plan["job_count"], 4)
            self.assertEqual({job["train_seed"] for job in plan["jobs"]}, {5, 6})

    def test_dry_run_pilot_creates_tiny_summary_and_does_not_execute_training(self):
        pilot = load_module(PILOT_SCRIPT, "full_xy_pilot_dry")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selected_csv, topology_root, payload_root, config_path = build_topology_files(root, ("G-1", "G-2"))
            output_root = root / "pilot"

            summary = pilot.run_pilot(
                selected_topologies_csv=selected_csv,
                topology_root=topology_root,
                base_payload_dir=payload_root,
                output_root=output_root,
                regime="step2c_poly_d8_mult_eps050",
                train_seed_start=1,
                train_seed_count=1,
                train_size=2,
                max_train_size=5,
                prefix_sizes=[2, 3, 5],
                validation_size=2,
                test_size=2,
                max_topologies=2,
                max_epochs=2,
                metric_stride=1,
                early_stop_patience=2,
                protocol="screen",
                context_generator_config=config_path,
                execute_one=False,
            )

            self.assertEqual(summary["status"], "success")
            self.assertEqual(summary["execute_one"], False)
            self.assertEqual(summary["executed_job_count"], 0)
            self.assertEqual(summary["job_count"], 2)
            self.assertTrue((output_root / "pilot_summary.json").exists())
            self.assertTrue((output_root / "pilot_jobs.csv").exists())
            self.assertTrue((output_root / "audit_results.json").exists())
            audits = json.loads((output_root / "audit_results.json").read_text(encoding="utf-8"))
            self.assertTrue(all(item["passed"] for item in audits))
            with (output_root / "pilot_jobs.csv").open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    self.assertEqual(row["protocol"], "screen")
                    self.assertEqual(row["dry_run_status"], "ready")
                    self.assertEqual(row["execute_status"], "not_executed")

    def test_execute_one_pilot_runs_exactly_one_mocked_job(self):
        pilot = load_module(PILOT_SCRIPT, "full_xy_pilot_execute_one")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selected_csv, topology_root, payload_root, config_path = build_topology_files(root, ("G-1", "G-2"))
            output_root = root / "pilot"
            completed = mock.Mock(returncode=0, stdout="", stderr="")

            with mock.patch.object(pilot.run_one_job.subprocess, "run", return_value=completed) as run_mock:
                summary = pilot.run_pilot(
                    selected_topologies_csv=selected_csv,
                    topology_root=topology_root,
                    base_payload_dir=payload_root,
                    output_root=output_root,
                    regime="step2c_poly_d8_mult_eps050",
                    train_seed_start=1,
                    train_seed_count=1,
                    train_size=2,
                    max_train_size=5,
                    prefix_sizes=[2, 3, 5],
                    validation_size=2,
                    test_size=2,
                    max_topologies=2,
                    max_epochs=2,
                    metric_stride=1,
                    early_stop_patience=2,
                    protocol="screen",
                    context_generator_config=config_path,
                    execute_one=True,
                )

            self.assertEqual(summary["status"], "success")
            self.assertEqual(summary["execute_one"], True)
            self.assertEqual(summary["executed_job_count"], 1)
            self.assertEqual(run_mock.call_count, 3)
            with (output_root / "pilot_jobs.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["execute_status"] for row in rows].count("success"), 1)
            self.assertEqual([row["execute_status"] for row in rows].count("not_executed"), 1)


if __name__ == "__main__":
    unittest.main()
