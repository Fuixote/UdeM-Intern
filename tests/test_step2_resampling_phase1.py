import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "surrogate_experiment_results"
    / "Step2_resampling"
    / "run_phase1_subset_resampling.py"
)


def load_phase1_module():
    spec = importlib.util.spec_from_file_location("phase1_subset_resampling", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Step2ResamplingPhase1Test(unittest.TestCase):
    def test_default_plan_has_eight_fixed_pool_regimes_and_fifty_subset_seeds(self):
        module = load_phase1_module()

        regimes = module.build_default_regimes(dataset_root=Path("dataset/processed"))
        jobs = module.build_jobs(
            regimes,
            subset_seeds=range(50),
            train_size=50,
            output_root=Path("surrogate_experiment_results/Step2_resampling/phase1_runs"),
            split_root=Path("surrogate_experiment_results/Step2_resampling/splits"),
            split_seed=42,
        )

        self.assertEqual(
            [regime.name for regime in regimes],
            [
                "step2b_poly_d1",
                "step2b_poly_d2",
                "step2b_poly_d4",
                "step2b_poly_d8",
                "step2c_poly_d1_mult_eps050",
                "step2c_poly_d2_mult_eps050",
                "step2c_poly_d4_mult_eps050",
                "step2c_poly_d8_mult_eps050",
            ],
        )
        self.assertEqual(len(jobs), 400)
        self.assertEqual(jobs[0].subset_seed, 0)
        self.assertEqual(jobs[-1].subset_seed, 49)
        self.assertEqual(
            jobs[-1].regime.unseen_dir.as_posix(),
            "dataset/processed/step2c_poly_d8_mult_eps050_unseen10000_seed20260523",
        )

    def test_step1c_environment_uses_phase1_fixed_seeds_and_external_val2000(self):
        module = load_phase1_module()
        regime = module.build_default_regimes(dataset_root=Path("dataset/processed"))[-1]
        job = module.build_jobs(
            [regime],
            subset_seeds=[49],
            train_size=50,
            output_root=Path("surrogate_experiment_results/Step2_resampling/phase1_runs"),
            split_root=Path("surrogate_experiment_results/Step2_resampling/splits"),
            split_seed=42,
        )[0]
        options = module.Phase1Options(
            project_root=Path("/repo"),
            workers=2,
            thread_count=1,
            dry_run=False,
            skip_completed=True,
            status_path=Path("status.csv"),
            manifest_path=Path("manifest.csv"),
            log_root=Path("logs"),
        )

        env = module.build_step1c_environment(job, options)

        self.assertEqual(
            env["STEP1C_DATA_DIR"],
            "dataset/processed/step2c_poly_d8_mult_eps050_main2000_seed20260523",
        )
        self.assertEqual(
            env["STEP1C_VALIDATION_DATA_DIR"],
            "dataset/processed/step2c_poly_d8_mult_eps050_val2000_seed20260523",
        )
        self.assertEqual(
            env["STEP1C_OUTPUT_DIR"],
            "surrogate_experiment_results/Step2_resampling/phase1_runs/"
            "step2c_poly_d8_mult_eps050/subset_seed=49",
        )
        self.assertEqual(env["STEP1C_TRAIN_SIZE"], "50")
        self.assertEqual(env["STEP1C_SUBSET_SEED"], "49")
        self.assertEqual(env["STEP1C_THETA_SEED"], "42")
        self.assertEqual(env["STEP1C_GUROBI_SEED"], "42")
        self.assertEqual(env["STEP1C_2STAGE_N_EPOCHS"], "500")
        self.assertEqual(env["STEP1C_SPOPLUS_N_EPOCHS"], "500")
        self.assertEqual(env["STEP1C_METRIC_STRIDE"], "10")
        self.assertEqual(env["STEP1C_TRAIN_POOL_SIZE"], "1200")
        self.assertEqual(env["STEP1C_VAL_SIZE"], "400")
        self.assertEqual(env["STEP1C_TEST_SIZE"], "400")
        self.assertEqual(env["OMP_NUM_THREADS"], "1")
        self.assertEqual(env["MKL_NUM_THREADS"], "1")
        self.assertEqual(env["OPENBLAS_NUM_THREADS"], "1")
        self.assertEqual(env["NUMEXPR_NUM_THREADS"], "1")

    def test_completion_requires_all_standard_weights_and_heldout_summary(self):
        module = load_phase1_module()
        regime = module.build_default_regimes(dataset_root=Path("dataset/processed"))[0]

        with tempfile.TemporaryDirectory() as tmp:
            job = module.build_jobs(
                [regime],
                subset_seeds=[0],
                train_size=50,
                output_root=Path(tmp) / "phase1_runs",
                split_root=Path(tmp) / "splits",
                split_seed=42,
            )[0]

            self.assertFalse(module.is_job_complete(job))

            weights_dir = job.run_dir / "model_weights"
            metrics_dir = job.run_dir / "metrics"
            weights_dir.mkdir(parents=True)
            metrics_dir.mkdir(parents=True)
            for filename in module.EXPECTED_WEIGHT_FILES[:-1]:
                (weights_dir / filename).write_text("placeholder", encoding="utf-8")
            (metrics_dir / "test_summary.csv").write_text(
                "method,test_mean_decision_gap\n", encoding="utf-8"
            )

            self.assertFalse(module.is_job_complete(job))

            (weights_dir / module.EXPECTED_WEIGHT_FILES[-1]).write_text(
                "placeholder", encoding="utf-8"
            )

            self.assertTrue(module.is_job_complete(job, include_unseen=False))
            self.assertFalse(module.is_job_complete(job, include_unseen=True))

            (metrics_dir / "unseen10000_summary.csv").write_text(
                "method,test_mean_decision_gap\n", encoding="utf-8"
            )

            self.assertTrue(module.is_job_complete(job, include_unseen=True))

    def test_skip_unseen_eval_dry_run_does_not_schedule_unseen_command(self):
        module = load_phase1_module()
        args = module.parse_args(
            [
                "--dry_run",
                "--skip_unseen_eval",
                "--regimes",
                "step2b_poly_d1",
                "--seed_count",
                "1",
            ]
        )

        self.assertTrue(args.skip_unseen_eval)


if __name__ == "__main__":
    unittest.main()
