# Codex Operating Notes

## Remote Runtime

- Primary remote target: `ssh cirrelt`
- Current `cirrelt` alias points to `fuweik@garnet.umontreal.cirrelt.lan`.
- Garnet repo root: `/local1/fuweik/UdeM-Intern`
- Local repo root: `/home/weikang/projects/UdeM-Intern/Exps`
- Use garnet for CPU/Gurobi runs. It has no GPU.
- Do not assume Slurm is usable for `fuweik` yet. The Slurm frontend exists at
  `slurm.umontreal.cirrelt.lan`, but `sacctmgr` showed no user/account
  association, so `sbatch` failed with an account/partition error.

## Garnet Environment

On garnet, source the runtime file from the repo root:

```bash
cd /local1/fuweik/UdeM-Intern
source configs/runtime/garnet.env
```

The runtime file loads Anaconda and Gurobi, activates `KEPs`, sets `KEP_PYTHON`,
and redirects data/results/solutions to `/local1/fuweik/UdeM-Intern`.

Verify the environment with:

```bash
python -c "import gurobipy as gp; print(gp.gurobi.version())"
```

Expected Gurobi Python version: `(13, 0, 0)`.

## Data Sync

Remote `dataset/processed` may be empty after a fresh clone. Sync from local WSL:

```bash
rsync -av --info=progress2 \
  dataset/processed/ \
  cirrelt:/local1/fuweik/UdeM-Intern/dataset/processed/
```

Check remote data on garnet:

```bash
find dataset/processed -maxdepth 2 -name 'G-*.json' | head
du -sh dataset/processed
```

## Smoke Test

Run a dry-run first:

```bash
source configs/runtime/garnet.env
./run_experiment.sh --dry-run 2stg-dfl-lr --train_size 5
```

Then run the smallest real smoke:

```bash
./run_experiment.sh 2stg-dfl-lr --train_size 5 --epochs 1
```

If the smoke fails before loading data, check missing Python packages in the
`KEPs` conda environment first. `gurobipy==13.0.0` is already installed in:

```text
/home/fuweik/.conda/envs/KEPs/lib/python3.10/site-packages
```

## Resource Use

- Use `/local1/fuweik` for repo, data, results, and solutions.
- Avoid using `/home/fuweik` for large artifacts; it is a small network home.
- Garnet is a shared CPU server, so avoid saturating all cores for long jobs.
- `configs/runtime/garnet.env` sets conservative thread defaults:
  `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=4`.

## Git Hygiene

- Keep runtime environment files small and explicit.
- Do not store passwords, private keys, or tokens in the repo.
- Generated datasets and experiment outputs should remain ignored by git.

## Experiment Documentation

- When an experiment gains an important or iconic result, new artifact, changed
  protocol, or interpretation that future work will rely on, update the
  experiment's README before finishing the task. The README should stay current
  enough that a future AI agent can understand the experiment purpose, setup,
  outputs, and latest known results without reconstructing the context from chat
  history.

## Experiment Completion Notifications

- For long-running tmux experiments in this repo, prefer the shared Brevo
  completion-notification helper documented in `EXPERIMENT_NOTIFICATIONS.md`.
- The reusable script is `scripts/experiment_notify.py`; pass each watched tmux
  session with `--session`, and pass the experiment's result/log roots with
  `--result-dir` and `--log-dir`.
- Notification secrets live outside the repo in
  `~/.config/experiment-notify/brevo.env`. Do not store API keys, SMTP keys,
  app passwords, or token screenshots in git.
- If a subproject has its own watcher script, keep its behavior consistent with
  the shared helper: wait for tmux sessions, summarize CSV/log status, then send
  one Brevo transactional email.
