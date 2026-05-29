# SPO+ Validation Status

## Implementation Note

The README now separates the validation into four levels:

1. **Level 1: formula and gradient checks.** Use a small shortest-path oracle to
   validate SPO+ loss, subgradient, shifted objectives, tie-breaking, and the
   cost-min to reward-max sign conversion.
2. **Level 1.5: KEP Step1c code-path checks.** Directly call the current
   `Step1c/step1c_common.py::spo_plus_loss_and_grad` implementation and compare
   it against an independent reward-max reference on a deterministic toy KEP
   oracle.
3. **Level 2: Warcraft/PyEPO empirical check.** Compare our SPO+ wrapper
   against the PyEPO Warcraft shortest-path notebook under matched data, model,
   optimizer, and seed settings.
4. **Level 3: synthetic degree sweep.** Later, reproduce the qualitative SPO
   paper-style observation that decision-focused training is most useful under
   increasing misspecification.

This workspace currently implements Level 1, Level 1.5, and the Level 2 runnable
files. The Warcraft archive has been downloaded into `SPO_validation/`, and the
12x12 split has been extracted for Level 2 smoke checks. Level 2 is an external
positive control, not a one-to-one reproduction of the KEP Step2 code path.

## What Was Validated

- Added a deterministic Gurobi-free monotone grid shortest-path oracle in
  `spoplus_shortest_path.py`.
- The oracle supports arbitrary real edge costs, including negative shifted
  costs, because paths live in a right/down directed acyclic graph.
- Tie-breaking is documented and deterministic: row-major edge order, with the
  right edge considered before the down edge at each node.
- Unit tests check perfect prediction, SPO+ upper bounding of the SPO decision
  gap, finite-difference agreement for a stable gradient direction, negative
  shifted costs, and deterministic ties.
- In the `KEPs` environment, `compare_with_pyepo_spoplus.py` compares a nonzero
  toy instance against `pyepo.func.SPOPlus`. Missing dependencies, Gurobi WLS
  license/network failures, and numerical mismatches are hard failures.
- Added Level 1.5 entrypoint `05_validate_kep_spoplus_code_path.py` and helper
  `spoplus_kep_path_validation.py`.
- Level 1.5 directly patches a deterministic reward-max oracle into
  `step1c_common.load_step1a_module`, calls
  `step1c_common.spo_plus_loss_and_grad(record, theta, env)`, and verifies:
  `w_hat = X @ theta`, `shifted_w = 2 w_hat - w_true`, the reward-max SPO+
  loss, and `grad_theta = 2 X.T @ (y_adv - y_optimal)`.
- Level 1.5 also checks a stable finite-difference directional derivative with
  respect to `theta`.
- Added Level 2 entrypoints `03_run_warcraft_pyepo_reference.py` and
  `04_run_warcraft_our_spoplus.py`, plus `warcraft_level2_common.py`.
- Level 2 defaults match the notebook protocol: `k=12`, `batch_size=70`,
  `epochs=50`, `lr=5e-4`, `seed=135`.
- Downloaded `warcraft_maps.tar.gz` from Edmond/PyEPO's referenced Warcraft
  dataset entry into `surrogate_experiment_results/SPO_validation/`.
  Dataset page: `https://edmond.mpdl.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.YJCQ5S`.
  Archive MD5: `acea5ea60a47664ff189923a84814e96`.
- Extracted the 12x12 split under
  `surrogate_experiment_results/SPO_validation/warcraft_shortest_path_oneskin/12x12/`.
- Level 2 requires local data under that path. Missing data remains a hard
  `FileNotFoundError`.

## Formulas Checked

For cost minimization:

```text
L_SPO+^min(c_hat, c)
  = max_z (c - 2 c_hat)^T z
    + 2 c_hat^T z*(c)
    - c^T z*(c)

gradient wrt c_hat:
  2 ( z*(c) - z*(2 c_hat - c) )
```

The implementation evaluates the max oracle through the equivalent minimization
oracle `argmin_z (2 c_hat - c)^T z`.

For reward maximization:

```text
L_SPO+^max(w_hat, w)
  = max_y (2 w_hat - w)^T y
    - 2 w_hat^T y*(w)
    + w^T y*(w)

gradient wrt w_hat:
  2 ( y*(2 w_hat - w) - y*(w) )
```

The two forms are related by:

```text
c = -w
c_hat = -w_hat
L_SPO+^max(w_hat, w) = L_SPO+^min(-w_hat, -w)
grad_w_hat L_max = - grad_c_hat L_min
```

## KEP SPO+ Audit

The existing Step1c implementation was directly checked against the reward-max
formula above:

- `shifted_w = 2.0 * w_hat - w_true`
- adversarial oracle: `solve_once(shifted_w, ...)`
- loss: `shifted_w^T y_adv - 2 w_hat^T y_optimal + w_true^T y_optimal`
- gradient: `2 * X.T @ (y_adv - y_optimal)`

No Step1c or Step2 KEP experiment code was modified for Level 1.5. The new test
calls the current Step1c code as-is and patches only the oracle provider in the
test process.

## Remaining Work

- Preserve the PyEPO/Gurobi comparison as a required Level-1 check. It requires
  Gurobi WLS license access to `token.gurobi.com`.
- Preserve Level 2 as an external positive control, but do not treat it as proof
  that the KEP Step2 code path is bug-free.
- Add a synthetic shortest-path degree/misspecification sweep for Level 3.
- For any future KEP solver-level audit, add a tiny real KEP graph check that
  uses Step1a's actual `solve_once` instead of the deterministic enumerating
  oracle.

## Commands

```bash
conda run -n KEPs python -m unittest tests.test_spoplus_shortest_path_validation -v
conda run -n KEPs python -m unittest tests.test_spoplus_kep_path_validation -v
conda run -n KEPs python -m unittest tests.test_step1c_spoplus -v
conda run -n KEPs python -m unittest tests.test_spoplus_warcraft_level2 -v
conda run -n KEPs python surrogate_experiment_results/SPO_validation/05_validate_kep_spoplus_code_path.py
conda run -n KEPs python surrogate_experiment_results/SPO_validation/compare_with_pyepo_spoplus.py
conda run -n KEPs python surrogate_experiment_results/SPO_validation/03_run_warcraft_pyepo_reference.py
conda run -n KEPs python surrogate_experiment_results/SPO_validation/04_run_warcraft_our_spoplus.py
```

If `pytest` is installed, the main Level-1 test can also be run as:

```bash
conda run -n KEPs pytest tests/test_spoplus_shortest_path_validation.py -q
```
