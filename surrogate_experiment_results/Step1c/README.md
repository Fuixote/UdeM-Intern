# Step1c: Surrogate Ablation Golden Rules

Step1c is the surrogate-ablation companion to Step1b.

The central rule is:

```text
Only change the surrogate. Keep the Step1b protocol fixed wherever possible.
```

Step1b studies sample-size generalization under a fixed decision-focused
training surrogate. Step1c should answer a narrower question:

```text
Holding the Step1b data, split, model class, initialization, training budget,
checkpoint protocol, and evaluation metrics fixed, does changing the surrogate
loss change held-out KEP decision quality?
```

The first surrogate to test should be **SPO+**. From the original Smart
Predict-then-Optimize framework, SPO loss measures the decision error induced
by a prediction, while SPO+ is a convex surrogate derived via duality and can be
used for linear-objective polyhedral, convex, and mixed-integer optimization
problems. This matches the Step1 KEP setting: the solver is a black-box
MIP/KEP oracle, and the predicted edge reward enters the KEP objective linearly.

References:

- Smart "Predict, then Optimize": <https://ar5iv.labs.arxiv.org/html/1710.08005>
- Decision-Focused Learning with Directional Gradients:
  <https://arxiv.org/abs/2402.03256>

## One-line Action Plan

Implement reward-max SPO+ for the same two-feature linear probe used in Step1b,
verify the sign and upper-bound properties with sanity checks, then run the
same Step1b train sizes and evaluation protocol while saving both
`best_by_validation_spoplus_loss` and `best_by_validation_decision_gap`
checkpoints.

## Current Scaffold Status

On 2026-05-19, Step1c was initialized by copying only the top-level Step1b
`.py` and `.sh` scripts, not any generated experiment outputs. The copied
scripts were renamed/path-separated so future Step1c runs write to Step1c
locations:

```text
surrogate_experiment_results/Step1c/
results/step1c_runs/
surrogate_experiment_results/Step1c/remote_results/
surrogate_experiment_results/Step1c/plot_results/
```

The shared master split intentionally remains:

```text
results/step1b_splits/master_split_seed=42.json
```

because Step1c should reuse the Step1b data protocol. The scaffold is not yet a
true SPO+ implementation: `train_end2end.py` still contains the copied FY logic.
The next implementation step is to replace or fork that training path with
reward-max SPO+ while preserving the surrounding Step1b evaluation protocol.

## Golden Rule 1: Step1c Is a Strict Surrogate Swap

Keep fixed:

```text
dataset: dataset/processed/step1_noisy_linear_sigma010
validation: dataset/processed/step1_noisy_linear_sigma010_validation2000_seed20260519
master split: results/step1b_splits/master_split_seed=42.json
train sizes: 50, 200, 600, 1200
model class: w_hat_e(theta) = theta_1 * utility_e + theta_2 * recipient_cPRA_e
initial theta seed: same as Step1b unless explicitly documented
evaluation metrics: synthetic-label decision gap and normalized gap
test sets: heldout400, unseen1000, realistic2000
metric stride: same as Step1b unless explicitly documented
Gurobi seed and solver settings: same as Step1b unless explicitly documented
```

Allow separate tuning only when necessary:

```text
learning rate
gradient clipping
loss normalization
weight decay / L2 regularization
early stopping patience
```

Reason: SPO+ and FY have different gradient scales. FY uses:

```text
grad_FY approx X^T (y_bar_perturbed - y_oracle)
```

SPO+ uses:

```text
grad_SPO+ = 2 X^T (y_adv - y_oracle)
```

Using the exact same learning rate may compare optimizer scale rather than
surrogate quality.

## Golden Rule 2: SPO Is Cost Minimization, KEP Is Reward Maximization

The original SPO paper is usually written for cost minimization:

```text
z*(c) in argmin_{z in Z} c^T z
```

The SPO loss is:

```text
ell_SPO(c_hat, c) = c^T z*(c_hat) - c^T z*(c)
```

This means: under the true cost vector, how much more expensive is the decision
induced by the predicted cost vector?

Step1 KEP is reward maximization:

```text
y*(w) in argmax_{y in Y} w^T y
```

The Step1b test metric is the synthetic-label decision gap:

```text
G(theta) = w^T y*(w) - w^T y*(w_hat_theta)
```

In code, this is exactly the pattern used by `evaluate_theta`: solve the KEP
under `w_hat`, then evaluate the achieved solution under `w_true` and subtract
from the synthetic-label oracle objective.

Do not copy the minimization SPO+ formula directly. Convert signs using:

```text
c = -w
c_hat = -w_hat
```

## Golden Rule 3: Reward-max SPO+ Formula

For graph instance `i`, define:

```text
w_i       = synthetic label rewards
X_i       = edge features
theta     = probe parameters
w_hat_i   = X_i theta
y_i*      = y*(w_i)
```

For reward-maximization KEP, use:

```text
L_i^SPO+(theta)
  = max_{y in Y_i} (2 w_hat_i - w_i)^T y
    - 2 w_hat_i^T y_i*
    + w_i^T y_i*
```

The adversarial shifted objective is:

```text
shifted_w_i = 2 w_hat_i - w_i
y_adv_i = y*(shifted_w_i)
```

The subgradient is:

```text
grad_theta L_i^SPO+(theta)
  = 2 X_i^T (y_adv_i - y_i*)
```

This is structurally close to the current FY gradient:

```text
grad_theta L_FY approx X^T (y_bar_perturbed - y*)
```

The scientific contrast is:

```text
FY challenger: stochastic perturbation-smoothed solution
SPO+ challenger: deterministic adversarial shifted-objective solution
```

This should be the central Step1c interpretation.

## Golden Rule 4: Minimal Implementation Shape

The core SPO+ function should look like this:

```python
def spo_plus_loss_and_grad(record, theta, env):
    X = record["X"]                  # shape: num_edges x 2
    w = record["w_true"]             # synthetic label rewards
    y_star = record["y_optimal"]     # solve_once(w)

    w_hat = X @ theta

    # Reward-max version of SPO+
    shifted_w = 2.0 * w_hat - w
    y_adv = solve_once(shifted_w, record["graph"], env)

    loss = (
        np.dot(shifted_w, y_adv)
        - 2.0 * np.dot(w_hat, y_star)
        + np.dot(w, y_star)
    )

    grad = 2.0 * (X.T @ (y_adv - y_star))
    return float(loss), grad
```

Train by averaging over graph instances:

```text
L(theta) = (1 / N) sum_i L_i^SPO+(theta)
```

Also consider recording a normalized version:

```text
L_i^SPO+_norm = L_i^SPO+ / (abs(w_i^T y_i*) + epsilon)
```

Use both names explicitly:

```text
raw_spoplus
normalized_spoplus
```

Raw SPO+ is closer to the original paper. Normalized SPO+ may be easier to
compare across heterogeneous graph scales.

## Golden Rule 5: Required Sanity Checks

Before any formal Step1c run, add and pass these checks.

### Check 1: Perfect Prediction

When:

```text
w_hat = w_true
```

Then:

```text
L_SPO+ approximately 0
grad approximately 0, if the oracle solution is unique or tie-breaking is stable
```

### Check 2: Upper Bound on Decision Gap

For several graphs and random `theta`, verify:

```text
SPO+ loss + tolerance >= synthetic-label decision gap
```

If this fails, the most likely error is a reward-max / cost-min sign mistake.

### Check 3: Negative Weights

SPO+ solves:

```text
max_y (2 w_hat - w)^T y
```

The shifted weights can be negative. The KEP solver path must not assume
nonnegative edge weights and must not pre-drop negative-weight edges. If it
does, the SPO+ subgradient is wrong.

### Check 4: Tie-breaking

KEP instances can have many optimal solutions, especially when many graph-level
decision gaps are zero. SPO and SPO+ behavior can depend on the oracle's
returned optimal solution. Keep Gurobi seed and solver settings fixed. If
needed, add tiny deterministic jitter only with explicit documentation.

## Golden Rule 6: Do Not Save Only One SPO+ Checkpoint

Step1b saves two FY checkpoints:

```text
e2e_best_by_validation_decision_gap.npz
e2e_best_by_validation_fy_loss.npz
```

Step1c should save symmetric SPO+ checkpoints:

```text
spoplus_best_by_validation_spoplus_loss.npz
spoplus_best_by_validation_decision_gap.npz
```

Optional diagnostics:

```text
spoplus_best_by_validation_normalized_gap.npz
spoplus_best_by_validation_mse_loss.npz
```

These checkpoints answer different questions:

```text
Trajectory quality:
  Does SPO+ training ever produce a low decision-gap model?

Selection quality:
  Does validation SPO+ loss itself select a model that generalizes?
```

The main Step1c result should emphasize `best_by_validation_spoplus_loss`,
because the question is whether SPO+ is useful as a training and model-selection
surrogate. `best_by_validation_decision_gap` is an oracle-style diagnostic.

## Golden Rule 7: Fairness Against Step1b FY

FY and SPO+ have different solver-call budgets.

FY with `M=16` needs many perturbed solves per graph per epoch. SPO+ typically
needs one shifted solve per graph per epoch. Therefore:

```text
Primary comparison:
  same Step1b-style epoch budget

Important caveat:
  solver-call budget differs

Optional later comparison:
  equal solver-call budget
```

Do not overclaim if SPO+ is faster or slower. Report both decision quality and
computational cost.

## Golden Rule 8: Suggested First Run Order

The first implementation should proceed in this order:

```text
1. Implement reward-max SPO+ loss and subgradient.
2. Add sanity checks for sign, upper bound, negative weights, and tie-breaking.
3. Run a tiny local smoke test.
4. Run train_size=200 on garnet as a single feasibility run.
5. Run train_size in {50, 200, 600, 1200}.
6. Evaluate heldout400, unseen1000, and realistic2000.
7. Plot full epoch diagnostics and final method-comparison bars.
```

Do not begin with a large surrogate zoo. First compare:

```text
2stage MSE baseline
FY surrogate from Step1b
SPO+ surrogate from Step1c
```

## Golden Rule 9: Warm-start Is Valuable but Not the Main Control

2stage warm-start may help SPO+ because SPO+ is a margin-style surrogate and
does not necessarily care about reward calibration. Starting from the MSE
solution near `[10, 5]` may be more stable than random initialization.

However, for the main strict surrogate-swap claim, keep initialization matched
to Step1b:

```text
main: same random init as FY
diagnostic/appendix: 2stage warm-start + SPO+ fine-tuning
```

This avoids mixing the surrogate comparison with an initialization comparison.

## Golden Rule 10: Add a Misspecification Panel Later

The noisy-linear Step1 benchmark is relatively well specified:

```text
w_syn = max(0, (10u + 5c) * (1 + noise))
w_hat = theta_1 u + theta_2 c
```

If truncation is rare and the noise is mean-zero, MSE is already a strong
baseline because the probe model can recover the conditional mean near `[10, 5]`.
SPO+ may therefore be close to 2stage on the noisy-linear benchmark. That is not
a failure.

SPO+ may be more valuable under model misspecification. A later Step1c panel can
test settings such as:

```text
utility-only probe: w_hat = theta_1 * utility
cPRA-only probe: w_hat = theta_2 * recipient_cPRA
wrong-scale / noisy feature probe
realistic_synthetic labels with the same 2-feature probe
nonlinear label regime with a linear probe
```

This should be treated as Step1c v2, not a blocker for the first SPO+ version.

## Golden Rule 11: Diagnostics to Plot

For each train size, record and plot:

```text
epoch vs train_spoplus_loss
epoch vs validation_spoplus_loss
epoch vs train_decision_gap
epoch vs validation_decision_gap
epoch vs heldout400_test_gap
epoch vs unseen1000_test_gap
epoch vs realistic2000_gap
epoch vs theta_1 and theta_2
epoch vs ||theta||
epoch vs y_adv == y_oracle fraction
epoch vs y_pred == y_oracle fraction
```

The two equality rates are especially useful:

```text
Pr[y*(2 w_hat - w) = y*(w)]
Pr[y*(w_hat) = y*(w)]
```

Interpretation:

```text
y_adv == y_oracle:
  SPO+ adversarial challenger has been defeated; the graph may have enough
  margin and contribute little or no gradient.

y_pred == y_oracle:
  The prediction-induced KEP solution already matches the synthetic-label
  oracle solution.

y_pred == y_oracle but y_adv != y_oracle:
  The current prediction gives the right decision, but the margin is not stable;
  SPO+ may keep pushing.
```

## Expected Outcomes

Do not expect SPO+ to automatically beat FY or 2stage.

Reasonable expectations:

```text
1. SPO+ should be cheaper than FY when FY uses many perturbations.
2. SPO+ may be a steadier checkpoint-selection loss than validation decision gap.
3. In the well-specified noisy-linear benchmark, SPO+ may only match 2stage.
4. SPO+ advantages are more likely in small-data or misspecified settings.
5. SPO+ may show scale or margin behavior, so track ||theta|| and reward
   calibration.
```

The final Step1c discussion should answer three questions:

```text
Q1: Trajectory quality
    Does SPO+ training produce low held-out decision-gap points?

Q2: Selection quality
    Does validation SPO+ loss select checkpoints that generalize better than
    validation decision gap?

Q3: Robustness under misspecification
    When reward fitting is insufficient, does SPO+ improve decision quality
    relative to MSE and FY?
```

## First Version Scope

The first Step1c version should stop at:

```text
1. Reward-max SPO+ implementation.
2. Sanity checks.
3. Same Step1b train sizes and validation2000 protocol.
4. Two SPO+ checkpoints:
   - selected by validation SPO+ loss
   - selected by validation decision gap
5. Heldout400 / unseen1000 / realistic2000 evaluation.
6. Full epoch diagnostics.
```

Do not expand into many surrogates or misspecification settings until this
strict SPO+ ablation is reproducible.
