import subprocess
import sys
import unittest


class HybridModuleImportTest(unittest.TestCase):
    def test_end2end_modules_can_be_imported_in_one_process(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import formulations.hybrid.end2end_reg; import formulations.hybrid.end2end_gnn; print('ok')",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("ok", completed.stdout)


if __name__ == "__main__":
    unittest.main()
