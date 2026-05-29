# SPO+ Validation Status

Current scope is deliberately small.  The validation keeps only three checks:

```text
01 formula check
05 KEP Step2/Step1c code-path check
03 PyEPO Warcraft vs local SPO+ empirical check
```

No extra sweep experiment is part of this folder.

## Implemented

- `01_compare_spoplus_formula_toy_shortest_path.py` checks the toy shortest-path
  SPO+ formula, stable gradient direction, upper-bound sanity check,
  reward-max/cost-min sign conversion, and PyEPO `SPOPlus` agreement.
- `05_validate_kep_spoplus_code_path.py` calls the current
  `Step1c/step1c_common.py::spo_plus_loss_and_grad` code path directly.
- `surrogate_experiment_results/Step1c/spoplus_core.py` now holds the shared
  reward-max SPO+ algebra and the cost-min sign adapter.
- `warcraft_level2_common.py::OurSPOPlus` now uses that shared Step1c core
  through the cost-min adapter instead of maintaining a separate autograd
  formula.
- `03_compare_warcraft_pyepo_vs_ours.py` is the canonical Warcraft empirical
  comparison entrypoint.

## Interpretation

The strongest KEP-specific evidence is still the `05` code-path check, because
it exercises the Step2/Step1c SPO+ function directly.  The Warcraft check is an
external positive control: it asks whether the same SPO+ algebra behaves like
PyEPO's `SPOPlus` on a standard shortest-path benchmark when only the oracle and
objective direction change.

## Verification

Latest local verification command:

```bash
conda run -n KEPs python -m unittest \
  tests.test_spoplus_shortest_path_validation \
  tests.test_spoplus_kep_path_validation \
  tests.test_spoplus_warcraft_level2 -v
```
