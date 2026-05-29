# SPO+ Validation

This folder is a small positive-control check for the Step2/Step1c SPO+
implementation.  It is not a full SPO paper reproduction.

The validation has three parts:

```text
01_compare_spoplus_formula_toy_shortest_path.py
05_validate_kep_spoplus_code_path.py
03_compare_warcraft_pyepo_vs_ours.py
```

## 01: Formula Check

`01_compare_spoplus_formula_toy_shortest_path.py` uses a deterministic toy
shortest-path oracle to check the cost-minimization SPO+ formula, finite
differences, the SPO+ upper-bound sanity check, and the reward-max/cost-min sign
conversion.  It also compares the toy loss and gradient against
`pyepo.func.SPOPlus`.

## 05: KEP Code-Path Check

`05_validate_kep_spoplus_code_path.py` directly calls the current
`surrogate_experiment_results/Step1c/step1c_common.py::spo_plus_loss_and_grad`
function on a deterministic toy reward-max oracle.  This is the most direct
check that the Step2/Step1c SPO+ path uses:

```text
shifted_w = 2 * w_hat - w_true
loss = shifted_w^T y_adv - 2 * w_hat^T y_optimal + w_true^T y_optimal
grad_theta = 2 * X.T @ (y_adv - y_optimal)
```

The shared algebra lives in:

```text
surrogate_experiment_results/Step1c/spoplus_core.py
```

## 03: Warcraft PyEPO Check

`03_compare_warcraft_pyepo_vs_ours.py` runs the Warcraft shortest-path
benchmark twice under matched data, model, optimizer, seed, and oracle settings:

```text
PyEPO SPOPlus
Step1c-core SPO+ with a shortest-path cost-min adapter
```

The local Warcraft SPO+ wrapper in `warcraft_level2_common.py` now uses the
same shared Step1c SPO+ algebra through a cost-minimization sign adapter.  The
script writes a single comparison CSV:

```text
surrogate_experiment_results/SPO_validation/plot_results/level2_warcraft_comparison_summary.csv
```

## Data

The Warcraft data is local-only and ignored by git:

```text
surrogate_experiment_results/SPO_validation/warcraft_maps.tar.gz
surrogate_experiment_results/SPO_validation/warcraft_shortest_path_oneskin/
```

The scripts expect the 12x12 split under:

```text
surrogate_experiment_results/SPO_validation/warcraft_shortest_path_oneskin/12x12/
```

## Commands

Run the test suite in the `KEPs` environment:

```bash
conda run -n KEPs python -m unittest \
  tests.test_spoplus_shortest_path_validation \
  tests.test_spoplus_kep_path_validation \
  tests.test_spoplus_warcraft_level2 -v
```

Run the three checks directly:

```bash
conda run -n KEPs python surrogate_experiment_results/SPO_validation/01_compare_spoplus_formula_toy_shortest_path.py
conda run -n KEPs python surrogate_experiment_results/SPO_validation/05_validate_kep_spoplus_code_path.py
conda run -n KEPs python surrogate_experiment_results/SPO_validation/03_compare_warcraft_pyepo_vs_ours.py
```
