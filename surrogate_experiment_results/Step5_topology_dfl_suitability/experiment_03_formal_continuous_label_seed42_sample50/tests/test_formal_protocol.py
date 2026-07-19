#!/usr/bin/env python3

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
STEP3_SCRIPTS = ROOT / "surrogate_experiment_results" / "Step3" / "scripts"
if str(STEP3_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(STEP3_SCRIPTS))

import sample_fixed_topology_context as context_sampler  # noqa: E402


class FormalProtocolTests(unittest.TestCase):
    def test_formal_training_is_early_stop_required_with_10000_safety_cap(self):
        config = context_sampler.load_simple_yaml(
            EXPERIMENT_ROOT / "configs" / "experiment.yaml"
        )

        self.assertEqual(config["training"]["max_epochs"], 10000)
        self.assertEqual(config["training"]["max_epochs_role"], "emergency_safety_cap")
        self.assertTrue(config["training"]["require_early_stop"])
        self.assertEqual(
            config["training"]["accepted_termination"],
            "both_methods_should_stop_true",
        )
        self.assertEqual(config["target"]["task"], "regression")
        self.assertEqual(config["target"]["primary"], "normalized_improvement_pp")
        self.assertEqual(config["execution"]["expected_job_count"], 1000)
        self.assertEqual(config["execution"]["normal_workers"], 20)
        self.assertTrue((ROOT / config["paths"]["topology_manifest"]).is_file())
        self.assertTrue((ROOT / config["paths"]["context_generator_config"]).is_file())


if __name__ == "__main__":
    unittest.main()
