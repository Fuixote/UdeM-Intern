# SPO Paper Synthetic Shortest-Path Validation

This sub-experiment is a focused positive control for the Step1c/Step2 SPO+
implementation.  It is intentionally separate from the existing three
`SPO_validation` checks:

1. `01_compare_spoplus_formula_toy_shortest_path.py`
2. `05_validate_kep_spoplus_code_path.py`
3. `03_compare_warcraft_pyepo_vs_ours.py`

The purpose here is narrower: reproduce the synthetic shortest-path setting
from the SPO paper and compare LS vs SPO+ on the canonical cost-minimization
problem before returning to KEP multi-seed runs.

## Paper Setting

The benchmark uses a 5x5 directed grid from the northwest corner to the
southeast corner, with only east/south moves.  The edge-cost vector has
dimension `d = 40`, and the feature vector has dimension `p = 5`.

For each `(trial, degree, noise_half_width)`:

```text
x_i ~ N(0, I_p)
B*_{j,k} ~ Bernoulli(0.5)
c_ij = (((B* x_i)_j / sqrt(p) + 3)^degree + 1) * epsilon_ij
epsilon_ij ~ Uniform(1 - noise_half_width, 1 + noise_half_width)
```

The middle-row reproduction uses:

```text
n_train = 1000
n_val = 250
n_test = 10000
degree = {1, 2, 4, 6, 8}
noise_half_width = {0, 0.5}
trials = 50
lambda_grid = logspace(-6, 0, 10)
```

Some copies/screenshots label this plot as Figure 4, while the paper PDF I
checked labels the shortest-path experiment as Figure 2.  To avoid ambiguity,
use the name "SPO paper synthetic shortest-path middle row" in notes and
advisor updates.

## Implemented Files

```text
common.py
  Data generation, oracle evaluation, normalized SPO loss, LS/L1 baseline,
  our Step1c-core SPO+ trainer, and optional PyEPO SPO+ trainer.

run_paper_shortest_path.py
  Main experiment runner.  Writes summary.csv and metadata.json.

plot_paper_shortest_path.py
  Optional plotting script for LS-vs-SPO+ boxplots and PyEPO-vs-ours scatter.
```

The local default Python environment currently lacks `torch`, `pyepo`,
`sklearn`, `pytest`, and `matplotlib`, so the core implementation avoids
`sklearn` and can run smoke tests with only NumPy.  PyEPO is imported only when
`--methods pyepo-spoplus` is requested.

## Local Smoke

From repo root:

```bash
python -m unittest tests.test_paper_shortest_path_experiment -v
```

Run a tiny end-to-end smoke:

```bash
python surrogate_experiment_results/SPO_validation/paper_shortest_path/run_paper_shortest_path.py \
  --preset smoke \
  --degrees 1 \
  --noise-half-widths 0 \
  --trials 1 \
  --n-train 20 \
  --n-val 8 \
  --n-test 12 \
  --lambda-grid 0 \
  --methods ls ours-spoplus \
  --spoplus-iterations 3 \
  --batch-size 5 \
  --output-dir /tmp/spo_paper_smoke
```

Expected output:

```text
/tmp/spo_paper_smoke/summary.csv
/tmp/spo_paper_smoke/metadata.json
```

## Pilot and Full Runs

Pilot:

```bash
python surrogate_experiment_results/SPO_validation/paper_shortest_path/run_paper_shortest_path.py \
  --preset pilot \
  --output-dir surrogate_experiment_results/SPO_validation/paper_shortest_path/results/pilot_$(date +%Y%m%d_%H%M%S)
```

Full paper-style middle row:

```bash
python surrogate_experiment_results/SPO_validation/paper_shortest_path/run_paper_shortest_path.py \
  --preset full \
  --output-dir surrogate_experiment_results/SPO_validation/paper_shortest_path/results/full_$(date +%Y%m%d_%H%M%S)
```

On an environment with PyEPO/Gurobi/Torch installed, add PyEPO to the method
list:

```bash
python surrogate_experiment_results/SPO_validation/paper_shortest_path/run_paper_shortest_path.py \
  --preset pilot \
  --methods ls ours-spoplus pyepo-spoplus \
  --output-dir surrogate_experiment_results/SPO_validation/paper_shortest_path/results/pilot_pyepo_$(date +%Y%m%d_%H%M%S)
```

## Plotting

If `matplotlib` is available:

```bash
python surrogate_experiment_results/SPO_validation/paper_shortest_path/plot_paper_shortest_path.py \
  surrogate_experiment_results/SPO_validation/paper_shortest_path/results/<run>/summary.csv
```

This writes:

```text
plots/paper_shortest_path_ls_vs_spoplus.png
plots/pyepo_vs_ours_spoplus.png   # only when paired PyEPO and ours rows exist
```

## Output Schema

`summary.csv` contains one row per `(implementation, trial, degree, noise)`:

```text
implementation
method
trial
degree
noise_half_width
seed
selected_lambda
val_norm_spo
test_norm_spo
test_avg_regret
test_avg_relative_regret
test_path_accuracy
test_optimality_ratio
n_train
n_val
n_test
spoplus_iterations
batch_size
learning_rate
```

`method` is the paper-level method family (`LS` or `SPO+`).
`implementation` is the concrete code path (`ls`, `ours-spoplus`,
`pyepo-spoplus`).

## Success Criteria

This is not meant to be a pixel-perfect paper reproduction.  The validation is
successful if:

- cost-min SPO+ formula and oracle sanity tests pass;
- LS and SPO+ reproduce the qualitative paper trend on the synthetic
  shortest-path benchmark;
- PyEPO SPO+ and `ours-spoplus` are close in paired runs under the same data,
  model class, lambda grid, and seeds;
- there is no systematic sign reversal or gradient-direction failure.

If these checks pass, the appropriate advisor-facing conclusion is:

> Regarding the SPO+ implementation, the validation results look good so far.
> We verified it on the synthetic shortest-path problem from the SPO paper and
> compared the Step1c-core SPO+ adapter against the PyEPO/reference path under
> the same shortest-path setup.
