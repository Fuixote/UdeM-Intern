# SPO Paper Synthetic Shortest-Path Validation

This sub-experiment is a focused positive control for the Step1c/Step2 SPO+
implementation.  It reproduces the middle-row synthetic shortest-path setting
from Smart "Predict, then Optimize" and compares:

- Least Squares
- our Step1c-core SPO+ implementation
- PyEPO SPO+ reference implementation, when requested

It is intentionally separate from the existing three `SPO_validation` checks:

1. `01_compare_spoplus_formula_toy_shortest_path.py`
2. `05_validate_kep_spoplus_code_path.py`
3. `03_compare_warcraft_pyepo_vs_ours.py`

## Paper Setting

The benchmark uses a 5x5 directed grid from the northwest corner to the
southeast corner, with only east/south moves.  The edge-cost vector has
dimension `d = 40`, and the feature vector has dimension `p = 5`.

For each `(trial, degree, noise_half_width)`:

```text
x_i ~ N(0, I_p)
B*_{j,k} ~ Bernoulli(0.5)
c_ij = [((B* x_i)_j / sqrt(p) + 3)^degree + 1] * epsilon_ij
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
lambda_grid = 10 values log-spaced from 1e-6 to 1e0
```

The model class is linear cost prediction from features with an unregularized
intercept.  L1 regularization is selected on the validation set by normalized
SPO loss.

The evaluation metric is the paper's ratio-of-sums normalized SPO loss:

```text
sum_i [ c_i^T z*(c_hat_i) - c_i^T z*(c_i) ]
/
sum_i [ c_i^T z*(c_i) ]
```

Lower is better.  Some screenshots label this as Figure 4, while the paper PDF
I checked labels the shortest-path experiment as Figure 2.  To avoid ambiguity,
use "SPO paper synthetic shortest-path middle row" in notes and advisor updates.

## Implementation Audit

As of this version, `common.py` matches the requested paper setup on the core
data and metric definitions:

- `grid_shape = (5, 5)`, east/south only, source northwest, target southeast;
- `edge_dim = 40` through the shared `spoplus_shortest_path.grid_edges` order;
- `feature_dim = 5`;
- default middle-row sizes `n_train = 1000`, `n_val = 250`, `n_test = 10000`;
- default degrees `{1, 2, 4, 6, 8}` and noise half-widths `{0, 0.5}`;
- default lambda grid `logspace(-6, 0, 10)`;
- normalized SPO loss implemented as sum of regrets divided by sum of oracle
  costs, not mean per-instance relative regret.

## Important Scope Note

The original paper used a reformulation/JuMP/Gurobi approach for shortest-path
SPO+ ERM.  The default `ours-spoplus` path here uses the oracle-based stochastic
subgradient formula:

```text
z_shifted = z*(2*c_hat - c)
grad_c_hat = 2 * (z*(c) - z_shifted)
```

Therefore, the goal is qualitative paper-trend validation and implementation
validation against PyEPO, not pixel-perfect reproduction of the original paper
unless a paper-exact reformulation is added later.

## Files

```text
common.py
  Data generation, oracle evaluation, normalized SPO loss, LS/L1 baseline,
  our Step1c-core SPO+ trainer, and optional PyEPO SPO+ trainer.

run_paper_shortest_path.py
  Main experiment runner.  Writes summary.csv and metadata.json.

plot_paper_shortest_path.py
  Paper-style middle-row plot plus PyEPO-vs-ours diagnostics.

compare_pyepo_forward_loss.py
  Tiny direct PyEPO SPOPlus forward-loss comparison against our local formula.
```

## Presets

```text
smoke
  Tiny dependency-light local test.

pilot
  Small but meaningful LS vs ours-SPO+ run.

pyepo-pilot
  Same as pilot, with pyepo-spoplus included.

middle-row
  Paper middle-row reproduction with LS and ours-SPO+.

middle-row-pyepo
  Paper middle-row reproduction with LS, ours-SPO+, and PyEPO SPO+.
```

The old `full` preset is kept as an alias for `middle-row`.

## Commands

Advisor-facing dry run:

```bash
python surrogate_experiment_results/SPO_validation/paper_shortest_path/run_paper_shortest_path.py \
  --preset middle-row \
  --dry-run
```

Pilot:

```bash
python surrogate_experiment_results/SPO_validation/paper_shortest_path/run_paper_shortest_path.py \
  --preset pilot \
  --output-dir surrogate_experiment_results/SPO_validation/paper_shortest_path/results/pilot_$(date +%Y%m%d_%H%M%S)
```

PyEPO pilot:

```bash
python surrogate_experiment_results/SPO_validation/paper_shortest_path/run_paper_shortest_path.py \
  --preset pyepo-pilot \
  --fail-if-pyepo-missing \
  --output-dir surrogate_experiment_results/SPO_validation/paper_shortest_path/results/pyepo_pilot_$(date +%Y%m%d_%H%M%S)
```

Full middle-row run:

```bash
python surrogate_experiment_results/SPO_validation/paper_shortest_path/run_paper_shortest_path.py \
  --preset middle-row \
  --output-dir surrogate_experiment_results/SPO_validation/paper_shortest_path/results/middle_row_$(date +%Y%m%d_%H%M%S)
```

Full PyEPO middle-row run:

```bash
python surrogate_experiment_results/SPO_validation/paper_shortest_path/run_paper_shortest_path.py \
  --preset middle-row-pyepo \
  --fail-if-pyepo-missing \
  --output-dir surrogate_experiment_results/SPO_validation/paper_shortest_path/results/middle_row_pyepo_$(date +%Y%m%d_%H%M%S)
```

Direct PyEPO forward-loss check:

```bash
python surrogate_experiment_results/SPO_validation/paper_shortest_path/compare_pyepo_forward_loss.py
```

Use `--allow-missing-pyepo` only for local dependency smoke checks.  Real PyEPO
validation should fail clearly if PyEPO/Torch/Gurobi are not importable.

## Plotting

If `matplotlib` is available:

```bash
python surrogate_experiment_results/SPO_validation/paper_shortest_path/plot_paper_shortest_path.py \
  surrogate_experiment_results/SPO_validation/paper_shortest_path/results/<run>/summary.csv
```

This writes:

```text
plots/paper_shortest_path_middle_row.png
plots/pyepo_vs_ours_spoplus_scatter.png       # when paired PyEPO and ours rows exist
plots/pyepo_vs_ours_spoplus_difference.png    # when paired PyEPO and ours rows exist
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

`metadata.json` includes audit fields:

```text
paper_experiment = shortest_path_middle_row
grid_shape
feature_dim
edge_dim
lambda_grid
methods
optimizer_ours_spoplus
optimizer_pyepo_spoplus
pyepo_requested
pyepo_available
normalized_spo_definition = sum_regret_over_sum_oracle_cost
```

## Success Criteria

This validation is successful if:

- cost-min SPO+ formula and oracle sanity tests pass;
- LS and SPO+ reproduce the qualitative paper trend on the synthetic
  shortest-path benchmark;
- PyEPO SPO+ and `ours-spoplus` are close in paired runs under the same data,
  model class, lambda grid, and seeds;
- there is no systematic sign reversal or gradient-direction failure.

Advisor-facing summary if checks pass:

> Regarding the SPO+ implementation, the validation results look good so far.
> We verified it on the synthetic shortest-path problem from the SPO paper and
> compared the Step1c-core SPO+ adapter against the PyEPO/reference path under
> the same shortest-path setup.
