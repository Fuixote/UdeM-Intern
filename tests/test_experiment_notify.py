import importlib.util
import io
from contextlib import redirect_stdout
import types
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "experiment_notify.py"
SPEC = importlib.util.spec_from_file_location("experiment_notify", MODULE_PATH)
experiment_notify = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(experiment_notify)


class ExperimentNotifyTest(unittest.TestCase):
    def make_args(self, dry_run=False):
        return types.SimpleNamespace(
            dry_run=dry_run,
            env=Path("/missing/env"),
            interval=1,
            log_dir=Path("logs"),
            project="Migration",
            result_dir=Path("results"),
            sessions=["rit_migration_all"],
            subject=None,
        )

    def test_run_watcher_sends_start_confirmation_before_completion_email(self):
        sent = []

        def fake_send(api_key, sender, recipient, subject, text):
            sent.append((subject, text))
            return 201, ""

        def fake_collect_summary(result_dir, log_dir):
            return {
                "csv_files": 0,
                "result_rows": 0,
                "log_files": 0,
                "error_logs": [],
            }

        with redirect_stdout(io.StringIO()):
            experiment_notify.run_watcher(
                self.make_args(),
                env={
                    "BREVO_API_KEY": "secret",
                    "SPO_NOTIFY_FROM": "from@example.com",
                    "SPO_NOTIFY_TO": "to@example.com",
                },
                send_email=fake_send,
                active_sessions_fn=lambda sessions: [],
                sleep_fn=lambda interval: None,
                collect_summary_fn=fake_collect_summary,
            )

        self.assertEqual(
            [subject for subject, _ in sent],
            ["Migration watcher started", "Migration experiments finished"],
        )
        self.assertIn("Watcher started successfully.", sent[0][1])
        self.assertIn("Watched tmux sessions: rit_migration_all", sent[0][1])
        self.assertIn("Migration finished.", sent[1][1])

    def test_dry_run_prints_start_and_completion_messages_without_sending(self):
        sent = []

        def fake_send(api_key, sender, recipient, subject, text):
            sent.append((subject, text))
            return 201, ""

        def fake_collect_summary(result_dir, log_dir):
            return {
                "csv_files": 1,
                "result_rows": 3,
                "log_files": 2,
                "error_logs": [],
            }

        with redirect_stdout(io.StringIO()):
            output = experiment_notify.run_watcher(
                self.make_args(dry_run=True),
                env={},
                send_email=fake_send,
                active_sessions_fn=lambda sessions: [],
                sleep_fn=lambda interval: None,
                collect_summary_fn=fake_collect_summary,
            )

        self.assertEqual(sent, [])
        self.assertIn("Migration watcher started", output)
        self.assertIn("Migration experiments finished", output)


if __name__ == "__main__":
    unittest.main()
