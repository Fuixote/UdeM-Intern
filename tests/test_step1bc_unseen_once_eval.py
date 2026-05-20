import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "surrogate_experiment_results"
    / "evaluate_step1bc_unseen_once.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("evaluate_step1bc_unseen_once", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Step1bcUnseenOnceEvalTest(unittest.TestCase):
    def test_discover_run_specs_skips_incomplete_runs(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            step1b_root = root / "step1b"
            step1c_root = root / "step1c"

            complete_b50 = step1b_root / "train_size=50" / "model_weights"
            incomplete_b200 = step1b_root / "train_size=200" / "model_weights"
            complete_c50 = step1c_root / "train_size=50" / "model_weights"
            complete_c200 = step1c_root / "train_size=200" / "model_weights"

            for directory in [complete_b50, incomplete_b200, complete_c50, complete_c200]:
                directory.mkdir(parents=True)

            for filename in module.STEP1B_WEIGHT_FILES:
                (complete_b50 / filename).write_text("", encoding="utf-8")
            (incomplete_b200 / module.STEP1B_WEIGHT_FILES[0]).write_text(
                "", encoding="utf-8"
            )
            for directory in [complete_c50, complete_c200]:
                for filename in module.STEP1C_WEIGHT_FILES:
                    (directory / filename).write_text("", encoding="utf-8")

            specs, warnings = module.discover_run_specs(
                step1b_root=step1b_root,
                step1c_root=step1c_root,
                train_sizes=[50, 200],
            )

        found = {(spec.source, spec.train_size) for spec in specs}
        self.assertEqual(
            found,
            {
                ("step1b", 50),
                ("step1c", 50),
                ("step1c", 200),
            },
        )
        self.assertTrue(any("train_size=200" in warning for warning in warnings))

    def test_rows_keep_evaluation_context_and_model_metadata(self):
        module = load_module()
        model = {
            "method": "spoplus",
            "train_size": 50,
            "selected_epoch": 480,
            "selection_metric": "validation_spoplus_loss",
            "selection_value": 1.23,
            "path": "model_weights/spoplus_best_by_validation_spoplus_loss.npz",
        }
        evaluations = [
            {
                "graph": "G-0.json",
                "optimal_obj": 10.0,
                "achieved_obj": 9.0,
                "gap": 1.0,
                "normalized_gap": 0.1,
                "ratio": 0.9,
            }
        ]

        rows = module.per_graph_rows_for_model(
            model=model,
            evaluations=evaluations,
            dataset_dir=Path("dataset/processed/unseen10000"),
            graph_count=10000,
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["evaluation_dataset"], "dataset/processed/unseen10000")
        self.assertEqual(rows[0]["evaluation_graph_count"], 10000)
        self.assertEqual(rows[0]["method"], "spoplus")
        self.assertEqual(rows[0]["selection_metric"], "validation_spoplus_loss")
        self.assertEqual(rows[0]["graph"], "G-0.json")


if __name__ == "__main__":
    unittest.main()
