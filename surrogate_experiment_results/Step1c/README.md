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

because Step1c should reuse the Step1b data protocol.

Current implementation status:

```text
split_dataset.py
step1c_common.py
train_2stage.py
train_end2end.py        # retained FY reference scaffold
train_spoplus.py        # reward-max SPO+ trainer
evaluate_models.py
evaluate_unseen_run.py  # expects Step1c SPO+ checkpoint names
plot_result_summary.py
plot_posthoc_diagnostics.py
run_step1c.sh
```

As of 2026-05-19, Step1c has a first reward-max SPO+ implementation:

```text
step1c_common.py:
  spo_plus_loss_and_grad
  grad_spoplus
  average_spoplus_objective
  evaluate_trajectory_spoplus_objective
  evaluate_trajectory_spoplus_diagnostics
  spoplus_solution_rates

train_spoplus.py:
  writes trajectory_spoplus.npy
  writes metrics/spoplus_loss_curve.csv
  writes spoplus_best_by_validation_decision_gap.npz
  writes spoplus_best_by_validation_spoplus_loss.npz
```

The first tiny smoke test was run on garnet, because local Gurobi attempted to
resolve `token.gurobi.com` and failed. The garnet smoke used:

```text
STEP1C_TRAIN_SIZE=5
STEP1C_2STAGE_N_EPOCHS=2
STEP1C_SPOPLUS_N_EPOCHS=2
STEP1C_METRIC_STRIDE=1
STEP1C_TRAIN_GRAPH_LIMIT=2
STEP1C_VALIDATION_LIMIT=3
STEP1C_TEST_LIMIT=3
STEP1C_OUTPUT_DIR=/tmp/step1c_spoplus_smoke
```

It completed and wrote:

```text
metrics/2stage_loss_curve.csv
metrics/spoplus_loss_curve.csv
metrics/test_summary.csv
model_weights/2stage_best_by_validation_mse_loss.npz
model_weights/spoplus_best_by_validation_decision_gap.npz
model_weights/spoplus_best_by_validation_spoplus_loss.npz
plots/2stage_mse_loss.png
plots/spoplus_loss.png
```

This smoke test is only an implementation check, not a scientific result.

On 2026-05-19, a first local formal `train_size=50` Step1c run was completed
with the matched Step1b-val2000 formal settings:

```text
STEP1C_TRAIN_SIZE=50
STEP1C_2STAGE_N_EPOCHS=500
STEP1C_SPOPLUS_N_EPOCHS=500
STEP1C_METRIC_STRIDE=10
validation override:
  dataset/processed/step1_noisy_linear_sigma010_validation2000_seed20260519
output:
  surrogate_experiment_results/Step1c/remote_results/
    formal_spoplus_ablation_val2000/train_size=50/
```

The first attempt exposed a performance issue in the SPO+ post-processing: the
code was separately re-solving KEP instances for validation decision gap, raw
SPO+ loss, normalized SPO+ loss, and solution equality rates. With 2000
validation graphs and 51 checkpoint epochs, this created several redundant
full passes over the same graph/checkpoint pairs. The fix was to add
`evaluate_trajectory_spoplus_diagnostics`, which computes decision gap,
raw/normalized SPO+ loss, and `y_adv` / `y_pred` equality rates in one pass.
This does not change the experimental definition; it only removes redundant
oracle calls and adds progress lines like:

```text
[SPO+ metrics] validation 24/51 epoch=230
```

First held-out 400-graph test result for `train_size=50`:

| method | selection metric | selected epoch | theta | test mean gap | test mean normalized gap | paired mean improvement over 2stage |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| 2stage | validation MSE | 500 | `[9.8707, 5.1130]` | 0.848028 | 0.006110 | -- |
| SPO+ | validation decision gap | 100 | `[6.7245, 3.4149]` | 0.807284 | 0.005809 | 0.040744 |
| SPO+ | validation SPO+ loss | 480 | `[9.1155, 4.1380]` | 0.740109 | 0.005442 | 0.107920 |

This is only one train-size run, so do not generalize yet. It is nevertheless
a useful feasibility signal: reward-max SPO+ trains stably from the same random
initialization, writes both checkpoint types, and the validation-SPO+-selected
checkpoint is best among the three methods on this held-out400 test split.

On 2026-05-19, a matched garnet server run was launched for all four train
sizes. This run is intended to be the strict Step1c counterpart to the Step1b
validation2000 FY rerun:

```text
train_size in {50, 200, 600, 1200}
STEP1C_2STAGE_N_EPOCHS=500
STEP1C_SPOPLUS_N_EPOCHS=500
STEP1C_SPOPLUS_LR=0.1
STEP1C_METRIC_STRIDE=10
validation override:
  dataset/processed/step1_noisy_linear_sigma010_validation2000_seed20260519
output root:
  surrogate_experiment_results/Step1c/remote_results/
    formal_spoplus_lr0.1_2stage500_spoplus500_s10_val2000_server/
logs:
  logs/step1c_spoplus_val2000_server_train_size_<n>.log
```

The output root is intentionally distinct from the local feasibility run
`formal_spoplus_ablation_val2000/` and from all Step1b FY outputs.

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

## Golden Rule 1b: Fair FY Baseline Must Use the Same Validation Protocol

Step1c defaults to the larger validation set:

```text
dataset/processed/step1_noisy_linear_sigma010_validation2000_seed20260519
```

This improves checkpoint-selection stability, but it also means old Step1b FY
archives that used the original 400-graph validation split are not a strict
baseline for Step1c.

For the primary Step1c comparison, use:

```text
2stage / FY / SPO+
same train split
same train size
same subset seed
same theta seed
same validation2000 set
same metric stride
same heldout400 / unseen1000 / realistic2000 evaluation
```

In other words:

```text
Strict comparison baseline = FY rerun under the Step1c validation2000 protocol.
Not strict baseline        = old Step1b FY archive selected with 400 validation graphs.
```

If an old Step1b result is shown, label it as historical context rather than as
the primary SPO+ baseline.

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

Recommended implementation targets:

```python
spo_plus_loss_and_grad(record, theta, env, normalize=False)
grad_spoplus(graphs, theta, env, normalize=False)
average_spoplus_objective(theta, graphs, env, normalize=False)
evaluate_trajectory_spoplus_objective(
    trajectory,
    graphs,
    env,
    indices=None,
    normalize=False,
)
```

Record both raw and normalized metrics:

```text
train_spoplus_loss
validation_spoplus_loss
train_normalized_spoplus_loss
validation_normalized_spoplus_loss
```

For the first version, select the main SPO+ checkpoint by raw
`validation_spoplus_loss`; keep normalized SPO+ as a diagnostic unless later
evidence shows raw scale is unstable.

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

Recommended test file:

```text
tests/test_step1c_spoplus.py
```

Minimum tests:

```text
test_spoplus_loss_zero_at_perfect_prediction_on_small_graphs
test_spoplus_upper_bounds_decision_gap_for_random_theta
test_spoplus_shifted_weights_can_be_negative
test_spoplus_gradient_shape_and_finite_values
```

The upper-bound test is the most important sign check:

```text
spo_plus_loss + tolerance >= decision_gap
```

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

Use explicit method names in `.npz` metadata:

```text
method = "spoplus"
selection_metric = "validation_spoplus_loss"
selection_metric = "validation_decision_gap"
```

Do not call SPO+ checkpoints `"e2e"` in saved metadata. FY and SPO+ are both
end-to-end decision-focused training paths, but the method label must identify
the surrogate to avoid confusing downstream plots.

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

Recommended plot/table labels:

```text
2stage (val MSE)
FY (val FY, validation2000 rerun)
FY (val gap, validation2000 rerun)          # diagnostic
SPO+ (val SPO+)
SPO+ (val gap)                             # diagnostic
```

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

## Golden Rule 8b: Concrete Engineering Order

Implement Step1c in this order:

```text
1. step1c_common.py
   Add SPO+ loss, gradient, trajectory-objective, and equality-rate helpers.

2. tests/test_step1c_spoplus.py
   Add sign, upper-bound, negative-weight, and finite-gradient tests.

3. train_spoplus.py
   Fork from train_end2end.py, replace FY training and checkpoint logic.

4. run_step1c.sh
   Call train_spoplus.py, remove FY-specific required weights from the Step1c
   evaluation call, and add SPO+ run-config fields.

5. evaluate_unseen_run.py
   Stop hard-coding FY checkpoint names. Either support explicit --weights or
   use Step1c SPO+ checkpoint names by default.

6. plot_result_summary.py / plot_posthoc_diagnostics.py
   Add SPO+ labels and diagnostics after the training artifacts exist.
```

Prefer forking `train_spoplus.py` instead of deleting `train_end2end.py`
immediately. The copied FY path is useful as a reference while implementing the
SPO+ path, but formal Step1c should call the SPO+ trainer.

Recommended Step1c runner variables:

```text
STEP1C_SPOPLUS_N_EPOCHS
STEP1C_SPOPLUS_LR
STEP1C_SPOPLUS_NORMALIZE_LOSS
STEP1C_SPOPLUS_GRAD_CLIP
STEP1C_SPOPLUS_WEIGHT_DECAY
```

The current copied variables `STEP1C_FY_EPSILON` and `STEP1C_FY_M` are FY
scaffold leftovers and should not define the formal SPO+ run.

## Golden Rule 8c: Smoke Runs Need Graph Limits

A tiny smoke test should not accidentally evaluate all 2000 validation graphs.
Before smoke testing, add a validation limit to the relevant training scripts:

```text
--validation_limit
```

Optionally also add:

```text
--train_graph_limit
```

Then use a true quick smoke:

```text
STEP1C_TRAIN_SIZE=5
STEP1C_2STAGE_N_EPOCHS=2
STEP1C_SPOPLUS_N_EPOCHS=2
STEP1C_METRIC_STRIDE=1
validation_limit=small
```

Smoke-test success means:

```text
SPO+ loss is finite
loss/metric CSVs are written
both SPO+ checkpoints are written
heldout test_summary.csv is written
negative shifted weights do not break the solver
```

Only after this should Step1c run `train_size=200` as a feasibility experiment.

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

The main SPO+ curve CSV should include at least:

```text
epoch
theta_1
theta_2
theta_norm
train_spoplus_loss
validation_spoplus_loss
train_normalized_spoplus_loss
validation_normalized_spoplus_loss
train_decision_gap
validation_decision_gap
train_y_adv_oracle_equal_rate
validation_y_adv_oracle_equal_rate
train_y_pred_oracle_equal_rate
validation_y_pred_oracle_equal_rate
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

## Full-run Gate

Do not start the full `{50, 200, 600, 1200}` Step1c run until all of these are
true:

```text
1. SPO+ sanity tests pass.
2. Tiny smoke run finishes and writes all expected artifacts.
3. train_size=200 feasibility run shows finite losses and stable theta.
4. SPO+ selected-by-loss and selected-by-gap checkpoints are both present.
5. Evaluation scripts accept the SPO+ checkpoint names.
6. FY baseline has been rerun under validation2000 if used in strict comparison.
```

If `theta_norm` grows without stabilizing, pause before the full run and test
gradient clipping, weight decay, or normalized SPO+ loss.
