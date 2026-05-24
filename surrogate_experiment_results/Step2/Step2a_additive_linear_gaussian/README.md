# Step2a: Additive Linear Gaussian Label Regime

## Purpose

Step2a is the bridge experiment between Step1's noisy-linear benchmark and the later misspecification regimes in Step2. The graph structures, train/validation/test split, model class, and evaluation metric should stay the same as Step1b/Step1c. The only intended change is the synthetic label generation rule.

This regime asks:

> If the conditional mean is still linear but the label noise is additive instead of multiplicative, do FY or SPO+ gain anything beyond the 2stage MSE baseline?

Because the two-feature probe is still well specified in expectation, 2stage MSE is expected to be strong here. If FY or SPO+ win clearly, the result should be interpreted carefully: it may indicate a real decision-structure benefit, but it may also come from clipping or validation-selection effects.

## Label Definition

For each edge e in graph G, start from the clean linear score:

```text
b_e = 10 u_e + 5 c_e
```

Let the graph-level clean scale be:

```text
mu_G = mean_e b_e
```

Then generate additive Gaussian noise:

```text
epsilon_e ~ Normal(0, (rho * mu_G)^2)
```

and define the synthetic reward label:

```text
w_syn_e = max(0, b_e + epsilon_e)
```

The parameter rho controls the additive-noise strength.

## Recommended First Parameters

```text
main: rho = 0.5
optional sweep: rho in {0.25, 0.5, 1.0}
clip_mode: max0
label_seed: 20260523 for the first strict Step2a run
graph_source: reuse the same graph structures as Step1b/Step1c when possible
```

## Dataset Naming

Formal processed datasets should stay under `dataset/processed/`, not inside this experiment directory. Use the shared Step2 short naming convention:

```text
dataset/processed/step2a_additive_rho050_main2000_seed20260523
dataset/processed/step2a_additive_rho050_val2000_seed20260523
dataset/processed/step2a_additive_rho050_unseen10000_seed20260523
```

The intended raw sources are:

```text
main2000:
  dataset/raw/2026-04-17_135607

val2000:
  dataset/raw/2026-05-19_000000__step1_noisy_linear_sigma010_validation2000_seed20260519

unseen10000:
  dataset/raw/2026-05-20_000000__step1_noisy_linear_sigma010_unseen_test10000_seed20260520
```

The graph structures come from the raw source directories above; Step2a only changes `ground_truth_label`.

## Fixed Protocol

Use the same protocol as Step1b/Step1c unless there is an explicit reason to change it:

```text
model: w_hat_e = theta_1 u_e + theta_2 c_e
methods: 2stage MSE, FY, SPO+
train sizes: 50, 200, 600, 1200
validation: validation2000 if using the current strict comparison protocol
evaluation: synthetic-label decision gap / normalized decision gap
checkpoint rules:
  2stage selected by validation MSE
  FY selected by validation FY loss
  FY selected by validation decision gap as diagnostic
  SPO+ selected by validation SPO+ loss
  SPO+ selected by validation decision gap as diagnostic
```

## Current Implementation Status

This directory contains a Step2a-specific copy of the root processing script:

```text
data-processing.py
```

It currently supports:

```text
--label_mode step2a_additive_linear_gaussian
--step2a_noise_rho <float>
--label_seed <int>
```

The script preserves the existing root processing behavior, but adds Step2a label generation. For each graph, it computes `mu_G` as the mean clean-linear label over raw valid matches, then assigns each edge:

```text
w_syn_e = max(0, b_e + epsilon_e)
epsilon_e ~ Normal(0, (rho * mu_G)^2)
```

The noise is deterministic per edge through a hash key that includes the source edge and `label_seed`, so rerunning the same command gives the same labels.

Useful smoke command from the repo root:

```bash
python surrogate_experiment_results/Step2/Step2a_additive_linear_gaussian/data-processing.py \
  dataset/raw/2026-04-17_135607 \
  /tmp/step2a_smoke_processed \
  --file genjson-0.json \
  --label_mode step2a_additive_linear_gaussian \
  --step2a_noise_rho 0.5 \
  --label_seed 20260523 \
  --output_as_batch_dir
```

## Required Diagnostics

Before training on this regime, record label statistics:

```text
mean label
std label
min/max label
fraction clipped to zero
oracle objective distribution
```

The clipping fraction is important. If many labels are clipped to zero, Step2a becomes partly a censored/sparse-reward experiment rather than only an additive-noise experiment.

## Expected Interpretation

Expected qualitative behavior:

```text
2stage MSE should be strong because the conditional mean is linear.
FY/SPO+ may be close to 2stage rather than clearly better.
Large FY/SPO+ improvements should trigger a check of clipping, scale, and validation selection.
```
