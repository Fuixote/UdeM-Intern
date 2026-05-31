# SPO Validation

This directory keeps local SPO validation artifacts separate from the upstream
PyEPO checkout.

## PyEPO Shortest Path Runs

The PyEPO `MPC` branch is checked out in `PyEPO/`.

First generate the fixed synthetic shortest-path datasets:

```bash
cd surrogate_experiment_results/SPO_validation/PyEPO

python3 generate_sp_datasets.py
```

This writes split datasets under:

```text
surrogate_experiment_results/SPO_validation/data/sp/h5w5/
```

Each file stores `x_train`, `c_train`, `x_test`, and `c_test` for one
`train_size`, degree, noise, and seed. Experiments load these files by default,
so all methods compare on the same saved train/test split.

Then run shortest-path experiments from the PyEPO checkout:

```bash
python3 experiments.py --prob sp --mthd lr   --expnum 10
python3 experiments.py --prob sp --mthd rf   --expnum 10
python3 experiments.py --prob sp --mthd spo  --expnum 10
python3 experiments.py --prob sp --mthd dbb  --expnum 10
python3 experiments.py --prob sp --mthd pfyl --expnum 10
```

`experiments.py` saves results to `../res` by default, so shortest-path CSVs are
written under:

```text
surrogate_experiment_results/SPO_validation/res/sp/h5w5/gurobi/
```

Use `--path PATH` to override the result root for ad hoc runs. Use
`--data-path PATH` to load fixed datasets from another location. Use
`--no-fixed-data` only for reproducing the original PyEPO behavior that
generates synthetic data in memory during each run.

## Plotting

From the PyEPO checkout, generate the shortest-path comparison plots with:

```bash
MPLCONFIGDIR=/tmp/matplotlib-pyepo python3 plot.py --plot cmp --prob sp
```

`plot.py` reads result CSVs from `../res` by default and writes six PNG files
under `PyEPO/images/`, one for each training-size and noise setting. Use
`--path PATH` to point the plotter at another result root.

If the plotting style dependencies are missing, install them into the active
environment:

```bash
python3 -m pip install SciencePlots tol_colors
```

## Completion Email Notification

`PyEPO/notify_when_done.py` can watch the experiment tmux sessions and send a
Brevo transactional email when all watched sessions finish.

Store Brevo settings outside the repo:

```bash
mkdir -p ~/.config/experiment-notify
chmod 700 ~/.config/experiment-notify
$EDITOR ~/.config/experiment-notify/brevo.env
chmod 600 ~/.config/experiment-notify/brevo.env
```

The env file should contain:

```bash
BREVO_API_KEY="xkeysib-..."
SPO_NOTIFY_FROM="notify@example.com"
SPO_NOTIFY_TO="you@example.com"
```

Start the watcher from the PyEPO checkout:

```bash
cd surrogate_experiment_results/SPO_validation/PyEPO
tmux new-session -d -s pyepo_notify \
  'python3 notify_when_done.py > ../logs/notify_when_done.log 2>&1'
```

The watcher checks these sessions by default:

```text
pyepo_sp_lr
pyepo_sp_rf
pyepo_sp_spo
pyepo_sp_dbb_pfyl_queue
```
