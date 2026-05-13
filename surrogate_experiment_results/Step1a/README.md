# Step1a: In-Sample Landscape Diagnostic

Step1a is a controlled visualization and diagnostic experiment for the
perturbed Fenchel-Young (FY) surrogate on a two-dimensional linear reward probe.

It is not a standard generalization experiment. It uses the same sampled graph
instances to train trajectories, evaluate trajectory metrics, and draw the
decision-gap landscape. Its role is to explain the geometry of the surrogate,
not to prove held-out test performance.

Use Step1a to answer:

```text
On a fixed batch of synthetic KEP instances, do MSE and FY training move through
different regions of the two-dimensional decision landscape?
```

Use Step1b for:

```text
Does training on more graph instances improve held-out decision quality?
```

## Scientific Role

Step1a is best described as:

```text
an in-sample controlled diagnostic / visualization experiment
```

It should not be described as:

```text
a proof that FY generalizes
a real clinical regret experiment
a held-out ML benchmark
```

The experiment is useful because it makes the FY surrogate visually inspectable.
It shows how the FY training trajectory behaves in the same parameter plane as
the MSE reward-fitting baseline and the synthetic-label decision-gap landscape.

The core story:

- MSE reward fitting moves toward the noise-free linear signal.
- FY decision-focused training moves according to the induced KEP decision.
- The best decision-gap point along an FY trajectory need not be the final FY
  endpoint.
- The perturbation scale `epsilon` changes FY's parameter scale and path.

## Dataset

Current default dataset:

```text
dataset/processed/step1_noisy_linear_sigma010
```

This dataset contains 2000 processed synthetic graph instances in the current
workflow. Step1a normally samples only 100 of them:

```text
STEP1_N_TOTAL=100
```

The synthetic label is the controlled noisy-linear reward:

```text
w_syn_e = max(0, (10 * utility_e + 5 * recipient_cPRA_e) * (1 + noise_e))
```

with noise level:

```text
sigma = 0.10
```

The model is a two-feature linear probe:

```text
w_hat_e(theta) = theta_1 * utility_e + theta_2 * recipient_cPRA_e
```

Feature order is always:

```text
theta_1 -> utility
theta_2 -> recipient_cPRA
```

The vector `[10, 5]` should be called:

```text
noise-free linear coefficient [10,5]
```

Do not call it the "true parameter" in final figures or text. Because the labels
include multiplicative noise, `[10,5]` is the clean signal coefficient, not the
exact synthetic label generator after noise.

## Metrics and Naming

Older file and function names still use `regret` or `true_regret` for backward
compatibility. In figures, captions, and README text, use the safer language:

```text
synthetic-label decision gap
oracle objective gap
decision gap under synthetic rewards
```

The metric is:

```text
G(theta) = (w_syn)^T y_oracle - (w_syn)^T S(w_hat(theta))
```

where:

- `y_oracle = S(w_syn)` is the KEP solution optimized under synthetic label
  weights.
- `S(w_hat(theta))` is the KEP solution induced by predicted weights.
- Smaller is better.

This is not clinical or real-world transplant regret. It is an oracle objective
gap under synthetic labels.

The perturbed FY objective used for the FY trajectory is:

```text
L_FY(theta) ~= E_z[max_y <w_hat(theta) + z, y>] - <w_hat(theta), y_oracle>
```

up to constants independent of `theta`.

## Data Sampling

Step1a uses:

```python
random.Random(seed).sample(sorted(G-*.json), min(n_total, len(files)))
```

Therefore:

- MSE and FY use the same sampled graph instances.
- Offline FY objective and decision-gap augmentation use the same sampled graph
  instances when the same `STEP1_N_TOTAL` and `STEP1_SEED` are used.
- The landscape contour also uses the same sampled graph instances.

Default seed from `run_step1.sh`:

```text
STEP1_SEED=42
```

With seed 42, the default initialization is:

```text
theta_init = [1.6236, 3.3521]
```

## Scripts

Main wrapper:

```text
surrogate_experiment_results/Step1a/run_step1.sh
```

Pipeline:

```text
1. Step1.py
   Generates MSE and FY theta trajectories.

2. add_true_regret_to_trajectory.py
   Appends synthetic-label decision gap to the MSE trajectory.
   The legacy filename says "regret".

3. add_FY_loss_to_trajectory.py
   Appends perturbed FY objective to the FY trajectory.

4. add_true_regret_to_trajectory.py
   Appends synthetic-label decision gap to the FY trajectory after FY objective
   has already been appended.

5. plot_trajectories_2D.py
   Plots the 2D synthetic-label decision-gap landscape with MSE/FY
   trajectories, final iterates, and best-gap iterate markers.

6. plot_trajectories_3D.py
   Plots MSE decision gap, FY decision gap, and FY objective as 3D trajectory
   curves.

7. plot_epoch_metrics.py
   Plots FY objective and synthetic-label decision gap versus epoch.
```

Shared dependency:

```text
linear_probe_landscape.py
```

Step1a imports graph loading, cached Gurobi KEP solving, perturbation sampling,
feature construction, and decision-gap landscape computation from that file.

## Default Parameters

The recommended entrypoint is `run_step1.sh`, not direct calls to `Step1.py`,
because the wrapper runs all postprocessing and plotting.

Current wrapper defaults:

```text
STEP1_DATA_DIR     = dataset/processed/step1_noisy_linear_sigma010
STEP1_N_TOTAL      = 100
STEP1_N_EPOCHS     = 1000
STEP1_LR_MSE       = 0.1
STEP1_LR_FY        = 0.1
STEP1_FY_EPSILON   = 1.0
STEP1_FY_M         = 16
STEP1_SEED         = 42
STEP1_GRID_SIZE    = 25
STEP1_N_MILESTONES = 5
```

Direct `Step1.py` defaults are smaller and are mainly useful for debugging.

## Output Layout

Current wrapper default output path:

```text
surrogate_experiment_results/Step1a/epsilon=<STEP1_FY_EPSILON>/
```

For formal runs, prefer setting `STEP1_OUTPUT_DIR` explicitly so generated
artifacts stay out of the code directory:

```text
results/step1_runs/step1a/epsilon=<epsilon>/
```

Important outputs:

```text
trajectory_mse.npy
trajectory_fy.npy
trajectory_mse_with_regret.npy
trajectory_fy_with_fy_loss.npy
trajectory_fy_with_fy_loss_and_regret.npy
true_regret_surface.npz
trajectory_contour.png
trajectory_3d_metrics.png
trajectory_epoch_metrics.png
```

Array schemas:

```text
trajectory_mse.npy
  shape: (n_epochs + 1, 2)
  columns: theta_1, theta_2

trajectory_fy.npy
  shape: (n_epochs + 1, 2)
  columns: theta_1, theta_2

trajectory_mse_with_regret.npy
  shape: (n_epochs + 1, 3)
  columns: theta_1, theta_2, synthetic-label decision gap

trajectory_fy_with_fy_loss.npy
  shape: (n_epochs + 1, 3)
  columns: theta_1, theta_2, perturbed FY objective

trajectory_fy_with_fy_loss_and_regret.npy
  shape: (n_epochs + 1, 4)
  columns: theta_1, theta_2, perturbed FY objective,
           synthetic-label decision gap
```

`true_regret_surface.npz` is a legacy filename. It currently stores:

```text
T1
T2
decision_gap
true_regret       # legacy key, same surface as decision_gap
theta_1_grid
theta_2_grid
n_total
seed
grid_size
```

## Figures

`trajectory_contour.png`:

- Left panel: MSE reward-fitting baseline trajectory.
- Right panel: decision-focused FY training trajectory.
- Colorbar: average oracle objective gap.
- Marker for start.
- Marker for final iterate.
- Marker for best-gap iterate.
- Marker for noise-free linear coefficient `[10,5]`.
- Dashed ray for the clean coefficient ratio `theta_1 / theta_2 = 2`.

`trajectory_3d_metrics.png`:

- MSE trajectory with decision gap as height.
- FY trajectory with decision gap as height.
- FY trajectory with perturbed FY objective as height.

`trajectory_epoch_metrics.png`:

- Perturbed FY objective versus epoch.
- Synthetic-label decision gap versus epoch.

The epoch metrics plot is intentionally FY-only. MSE does not have an FY
objective, and mixing MSE loss with FY objective in the same panel invites a
wrong comparison of quantities with different meanings.

## Local Commands

Smoke test:

```bash
cd /home/weikang/projects/UdeM-Intern/Exps

MPLCONFIGDIR=/tmp/matplotlib \
PYTHONUNBUFFERED=1 \
STEP1_N_TOTAL=5 \
STEP1_N_EPOCHS=2 \
STEP1_FY_EPSILON=1.0 \
STEP1_FY_M=2 \
STEP1_GRID_SIZE=5 \
STEP1_OUTPUT_DIR=results/step1_runs/step1a_smoke/epsilon=1.0 \
surrogate_experiment_results/Step1a/run_step1.sh
```

Formal local run for `epsilon=0.2`:

```bash
cd /home/weikang/projects/UdeM-Intern/Exps

MPLCONFIGDIR=/tmp/matplotlib \
PYTHONUNBUFFERED=1 \
STEP1_DATA_DIR=dataset/processed/step1_noisy_linear_sigma010 \
STEP1_N_TOTAL=100 \
STEP1_N_EPOCHS=1000 \
STEP1_FY_EPSILON=0.2 \
STEP1_FY_M=16 \
STEP1_GRID_SIZE=25 \
STEP1_OUTPUT_DIR=results/step1_runs/step1a/epsilon=0.2 \
surrogate_experiment_results/Step1a/run_step1.sh
```

Formal local run for `epsilon=1.0`:

```bash
cd /home/weikang/projects/UdeM-Intern/Exps

MPLCONFIGDIR=/tmp/matplotlib \
PYTHONUNBUFFERED=1 \
STEP1_DATA_DIR=dataset/processed/step1_noisy_linear_sigma010 \
STEP1_N_TOTAL=100 \
STEP1_N_EPOCHS=1000 \
STEP1_FY_EPSILON=1.0 \
STEP1_FY_M=16 \
STEP1_GRID_SIZE=25 \
STEP1_OUTPUT_DIR=results/step1_runs/step1a/epsilon=1.0 \
surrogate_experiment_results/Step1a/run_step1.sh
```

To run the two epsilon settings in parallel locally, paste the two formal
commands into two different terminals. Avoid `nohup` locally if you want live
output.

`MPLCONFIGDIR=/tmp/matplotlib` is used because Matplotlib may not be able to
write its default config/cache directory in some environments.

## Garnet Remote Commands

Remote target:

```text
ssh cirrelt
```

Remote repo root:

```text
/local1/fuweik/UdeM-Intern
```

Runtime setup:

```bash
cd /local1/fuweik/UdeM-Intern
source configs/runtime/garnet.env
```

Check data:

```bash
find dataset/processed/step1_noisy_linear_sigma010 -name 'G-*.json' | head
```

Sync Step1a scripts from local if they changed:

```bash
rsync -av surrogate_experiment_results/Step1a/ \
  cirrelt:/local1/fuweik/UdeM-Intern/surrogate_experiment_results/Step1a/
```

Example remote run for `epsilon=0.2`:

```bash
ssh cirrelt 'bash -s' <<'REMOTE'
set -euo pipefail

cd /local1/fuweik/UdeM-Intern
source configs/runtime/garnet.env
mkdir -p logs /tmp/matplotlib

env \
  MPLCONFIGDIR=/tmp/matplotlib \
  PYTHONUNBUFFERED=1 \
  STEP1_DATA_DIR=dataset/processed/step1_noisy_linear_sigma010 \
  STEP1_N_TOTAL=100 \
  STEP1_N_EPOCHS=1000 \
  STEP1_FY_EPSILON=0.2 \
  STEP1_FY_M=16 \
  STEP1_GRID_SIZE=25 \
  STEP1_OUTPUT_DIR=results/step1_runs/step1a/epsilon=0.2 \
  surrogate_experiment_results/Step1a/run_step1.sh
REMOTE
```

For overnight remote runs, redirect to `logs/step1a_epsilon_<eps>.log` if live
output is not needed.

## Cost and Runtime Notes

Step1a is solver-heavy. The expensive parts are:

- FY training: each epoch solves perturbed KEP instances.
- Decision-gap augmentation: each trajectory point requires KEP solves.
- FY objective augmentation: each trajectory point and perturbation requires
  KEP solves.
- Landscape plotting: roughly `grid_size^2 * n_total` KEP solves.

Increasing any of these increases runtime:

```text
STEP1_N_TOTAL
STEP1_N_EPOCHS
STEP1_FY_M
STEP1_GRID_SIZE
```

Garnet is a CPU/Gurobi server, but the current outer loops are mostly serial
Python/Gurobi calls. Seeing near one-core utilization is expected unless the
code is parallelized.

For quick checks, reduce:

```text
STEP1_N_TOTAL=5
STEP1_N_EPOCHS=2
STEP1_FY_M=2
STEP1_GRID_SIZE=5
```

For formal Step1a diagnostics, the historical run used:

```text
STEP1_N_TOTAL=100
STEP1_N_EPOCHS=1000
STEP1_FY_M=16
STEP1_GRID_SIZE=25
```

## Historical Completed Results

The following numbers are the Step1a results recorded in
`surrogate_experiment_design.tex` from the completed noisy-linear diagnostic.
They should be treated as the current historical experiment record, not as live
files in this directory.

Configuration:

```text
dataset:        dataset/processed/step1_noisy_linear_sigma010
n_total:        100
n_epochs:       1000
seed:           42
theta_init:     [1.6236, 3.3521]
OLS theta:      [10.0017, 5.0147]
FY M:           16
epsilon values: 0.2 and 1.0
```

Initial synthetic-label decision gap:

```text
3.5402
```

Summary table:

| method | epsilon | final theta | final FY objective | final decision gap | best decision gap |
|---|---:|---:|---:|---:|---:|
| MSE | -- | (10.0017, 5.0147) | -- | 0.8759 | 0.8759 |
| FY | 0.2 | (2.8135, 1.3589) | 0.3609 | 0.8714 | 0.7183 at epoch 42 |
| FY | 1.0 | (13.9796, 6.7825) | 1.8042 | 0.8714 | 0.8668 at epoch 781 |

Interpretation:

- MSE recovered the noise-free linear coefficient, but did not achieve zero
  decision gap because the labels include multiplicative noise and the probe has
  only two features.
- FY with `epsilon=0.2` visited a lower-gap region than the final MSE endpoint,
  but the best-gap iterate occurred early.
- FY with `epsilon=1.0` moved to a much larger parameter scale and reached its
  best observed decision gap later.
- The final FY endpoint is not necessarily the best point for deterministic KEP
  decision quality.
- This motivates validation/model selection in Step1b.

## Current Local Artifact State

At the time this README was written, the local directories:

```text
surrogate_experiment_results/Step1a/epsilon=0.2/
surrogate_experiment_results/Step1a/epsilon=1.0/
```

exist but are empty. The local logs:

```text
logs/step1a_epsilon_0.2.log
logs/step1a_epsilon_1.0.log
```

are incomplete local attempts that stopped during MSE training and should not be
used as completed results.

To regenerate figures, rerun `run_step1.sh` with explicit `STEP1_OUTPUT_DIR`.

## Language Guidance for Figures and Text

Use:

```text
Synthetic-label Decision Gap Landscape
Avg. oracle objective gap
MSE reward-fitting baseline
Decision-focused FY training
Perturbed FY objective
Synthetic-label decision gap vs epoch
best-gap iterate
final iterate
noise-free linear coefficient [10,5]
```

Avoid or replace:

```text
True Regret -> Synthetic-label decision gap / Oracle objective gap
true weights -> synthetic label weights / oracle evaluation weights
true parameter -> noise-free linear coefficient
regret min -> best grid point by decision gap / best-gap iterate
end-to-end -> decision-focused FY training
low-regret region -> low decision-gap region
```

## Relationship to Step1b

Step1a supports a qualitative claim:

```text
The FY surrogate can move the two-feature probe through low synthetic-label
decision-gap regions on a fixed controlled batch.
```

Step1a does not support this stronger claim by itself:

```text
FY loss generalizes better than MSE on unseen KEP instances.
```

That stronger claim belongs to Step1b, which uses a fixed train/validation/test
split, model selection on validation data, and held-out test evaluation.

## Follow-Up Ideas

Useful future improvements:

1. Make `STEP1_OUTPUT_DIR=results/step1_runs/step1a/epsilon=<eps>` the default
   in `run_step1.sh` if the code directory should stay artifact-free.
2. Write a small manifest JSON after each run, recording all hyperparameters and
   final/best metrics.
3. Add a lightweight summary CSV with final theta, final gap, best gap, and best
   epoch.
4. Add optional surface-cache reuse so changing only trajectory labels or
   markers does not recompute the decision-gap landscape.
5. Add a cheaper "plot-only" mode that starts from existing `.npy` and `.npz`
   files.

