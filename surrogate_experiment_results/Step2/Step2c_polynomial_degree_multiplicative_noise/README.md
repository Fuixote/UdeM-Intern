# Step2c: Polynomial Degree With Multiplicative Noise Label Regime

## Purpose

Step2c extends Step2b by adding SPO-style multiplicative noise on top of the degree-controlled nonlinear mean. This is the closest Step2 regime to the SPO paper's synthetic shortest-path setup: degree controls deterministic misspecification, while multiplicative noise controls stochastic uncertainty.

This regime asks:

> Under nonlinear misspecification plus edge-wise multiplicative noise, which validation surrogate selects checkpoints that generalize better: FY loss, SPO+ loss, or direct validation decision gap?

This regime is expected to be one of the most informative Step2 experiments.

## Label Definition

First compute the Step2b nonlinear mean:

```text
b_e = 10 u_e + 5 c_e
mu_G = mean_e b_e
q_e = b_e / (mu_G + delta)
r_e^(d) = (q_e + kappa)^d - kappa^d
m_e^(d) = mu_G * r_e^(d) / (mean_e r_e^(d) + delta)
```

Then draw independent multiplicative noise:

```text
eta_e ~ Uniform(1 - epsilon_bar, 1 + epsilon_bar)
```

and define:

```text
w_syn_e = max(0, m_e^(d) * eta_e)
```

If epsilon_bar <= 1 and m_e^(d) >= 0, clipping is usually inactive, but it is still kept for consistency with the rest of the synthetic-label pipeline.

## Recommended First Parameters

```text
kappa = 3
degree d in {1, 2, 4}
epsilon_bar = 0.5
optional later: epsilon_bar in {0.25, 0.5}
delta: small positive numerical stabilizer
clip_mode: max0
rescale_mode: graph_mean_to_clean_linear_mean before noise
label_seed: fixed and recorded
graph_source: reuse the same graph structures as Step1b/Step1c when possible
```

## Dataset Naming

Formal processed datasets should stay under `dataset/processed/`, not inside this experiment directory. Use the shared Step2 short naming convention:

```text
dataset/processed/step2c_poly_d1_mult_eps050_main2000_seed20260523
dataset/processed/step2c_poly_d1_mult_eps050_val2000_seed20260523
dataset/processed/step2c_poly_d1_mult_eps050_unseen10000_seed20260523

dataset/processed/step2c_poly_d2_mult_eps050_main2000_seed20260523
dataset/processed/step2c_poly_d2_mult_eps050_val2000_seed20260523
dataset/processed/step2c_poly_d2_mult_eps050_unseen10000_seed20260523

dataset/processed/step2c_poly_d4_mult_eps050_main2000_seed20260523
dataset/processed/step2c_poly_d4_mult_eps050_val2000_seed20260523
dataset/processed/step2c_poly_d4_mult_eps050_unseen10000_seed20260523
```

The graph structures should come from the same raw source directories used by Step2a/Step2b:

```text
main2000:
  dataset/raw/2026-04-17_135607

val2000:
  dataset/raw/2026-05-19_000000__step1_noisy_linear_sigma010_validation2000_seed20260519

unseen10000:
  dataset/raw/2026-05-20_000000__step1_noisy_linear_sigma010_unseen_test10000_seed20260520
```

## Current Implementation Status

This directory contains a Step2c-specific copy of the root processing script:

```text
data-processing.py
```

It currently supports:

```text
--label_mode step2c_polynomial_degree_multiplicative_noise
--step2c_degree <int>
--step2c_kappa <float>
--step2c_delta <float>
--step2c_epsilon_bar <float>
--label_seed <int>
```

For each graph, the script first computes the Step2b graph-rescaled polynomial label over raw valid matches:

```text
b_e = 10 * utility/100 + 5 * cPRA
q_e = b_e / (mu_G + delta)
r_e = (q_e + kappa)^degree - kappa^degree
m_e = max(0, mu_G * r_e / (mean_G(r) + delta))
```

It then applies deterministic per-edge uniform multiplicative noise:

```text
eta_e = Uniform(1 - epsilon_bar, 1 + epsilon_bar)
w_syn_e = max(0, m_e * eta_e)
```

The random draw is deterministic from `source_key`, `label_seed`, and the Step2c noise namespace, so repeated processing is reproducible.

The output preserves diagnostic fields:

```text
latent_clean_linear_label
step2c_polynomial_label
step2c_multiplier
step2c_noisy_polynomial_label
step2c_epsilon_bar
step2c_q_value
step2c_polynomial_score
step2c_graph_clean_linear_mean
step2c_graph_polynomial_score_mean
step2c_graph_clean_linear_edge_count
step2c_degree
step2c_kappa
step2c_delta
```

Useful smoke command from the repo root:

```bash
python surrogate_experiment_results/Step2/Step2c_polynomial_degree_multiplicative_noise/data-processing.py \
  dataset/raw/2026-04-17_135607 \
  /tmp/step2c_smoke_processed \
  --file genjson-0.json \
  --label_mode step2c_polynomial_degree_multiplicative_noise \
  --step2c_degree 2 \
  --step2c_kappa 3 \
  --step2c_delta 0 \
  --step2c_epsilon_bar 0.5 \
  --label_seed 20260523 \
  --output_as_batch_dir
```

## Fixed Protocol

Use the same method and split protocol for every degree/noise setting:

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

## Required Diagnostics

For every degree/noise setting, record:

```text
mean label
std label
min/max label
fraction clipped to zero
oracle objective distribution
correlation between nonlinear mean m_e^(d) and noisy label w_syn_e
```

Training diagnostics are especially important here:

```text
epoch vs validation surrogate loss
epoch vs validation decision gap
epoch vs heldout decision gap
epoch vs unseen decision gap
```

Step1b/Step1c already showed that direct validation decision gap can be high variance, so this regime should explicitly compare selection by surrogate loss versus selection by validation decision gap.

## Expected Interpretation

Expected qualitative behavior:

```text
degree high + epsilon_bar high makes checkpoint selection harder.
validation decision gap may be noisy.
FY loss and SPO+ loss should be evaluated as model-selection surrogates, not only as training objectives.
```

This regime should help separate deterministic misspecification effects from stochastic noise effects.
