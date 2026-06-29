# Experiment Completion Notifications

This repository can send Brevo emails when long-running tmux experiment
watchers start successfully and when the watched experiments finish. The shared
helper is:

```text
scripts/experiment_notify.py
```

## Credentials

Secrets stay outside the repo in:

```text
~/.config/experiment-notify/brevo.env
```

Expected contents:

```bash
BREVO_API_KEY="xkeysib-..."
SPO_NOTIFY_FROM="brevo@fuixote.com"
SPO_NOTIFY_TO="don@fuixote.com"
```

Keep the file private:

```bash
chmod 700 ~/.config/experiment-notify
chmod 600 ~/.config/experiment-notify/brevo.env
```

Do not commit API keys, SMTP keys, app passwords, or screenshots containing
keys.

## Start A Watcher

From this repository root, start a watcher for one or more tmux sessions:

```bash
tmux new-session -d -s notify_my_experiment \
  'python3 scripts/experiment_notify.py \
    --project "My experiment name" \
    --session session_one \
    --session session_two \
    --result-dir results \
    --log-dir logs \
    > logs/notify_my_experiment.log 2>&1'
```

The watcher sends a startup confirmation email immediately after it has parsed
the request and loaded the Brevo settings. It then checks whether each tmux
session still exists. When all watched sessions have ended, it scans result
CSVs and log files, then sends the completion Brevo email.

## Dry Run

Use `--dry-run` to preview both email bodies without sending:

```bash
python3 scripts/experiment_notify.py \
  --dry-run \
  --project "Dry run" \
  --session definitely_not_running \
  --result-dir results \
  --log-dir logs
```

## Current SPO Validation Example

The PyEPO SPO validation uses a project-specific watcher script at:

```text
surrogate_experiment_results/SPO_validation/PyEPO/notify_when_done.py
```

The generic equivalent from the repository root is:

```bash
tmux new-session -d -s pyepo_notify \
  'python3 scripts/experiment_notify.py \
    --project "PyEPO SPO shortest-path experiments" \
    --session pyepo_sp_lr \
    --session pyepo_sp_rf \
    --session pyepo_sp_spo \
    --session pyepo_sp_dbb_pfyl_queue \
    --result-dir surrogate_experiment_results/SPO_validation/res \
    --log-dir surrogate_experiment_results/SPO_validation/logs \
    > surrogate_experiment_results/SPO_validation/logs/notify_when_done.log 2>&1'
```
