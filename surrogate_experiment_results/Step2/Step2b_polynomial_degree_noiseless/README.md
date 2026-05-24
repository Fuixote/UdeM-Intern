# Step2b: Polynomial Degree Noiseless Label Regime

## Purpose

Step2b is the first explicit misspecification experiment. It keeps the downstream KEP problem and the two-feature linear probe fixed, but changes the synthetic label from a linear reward into a deterministic nonlinear reward controlled by a polynomial degree.

This follows the main experimental idea from the SPO paper: keep the training model class simple, then increase the degree of the true data-generating function to control misspecification.

This regime asks:

> When the true synthetic reward is a deterministic nonlinear function of the edge features, can FY or SPO+ learn a better decision-oriented linear proxy than 2stage MSE?

## Label Definition

For each edge e in graph G, start from the clean linear score:

```text
b_e = 10 u_e + 5 c_e
```

Let:

```text
mu_G = mean_e b_e
q_e = b_e / (mu_G + delta)
```

Define the shifted polynomial score:

```text
r_e^(d) = (q_e + kappa)^d - kappa^d
```

Then rescale within each graph so the average reward scale remains comparable across degrees:

```text
m_e^(d) = mu_G * r_e^(d) / (mean_e r_e^(d) + delta)
```

The synthetic reward label is:

```text
w_syn_e = max(0, m_e^(d))
```

Here d is the misspecification knob. Larger d makes high-score edges disproportionately more valuable and changes the KEP decision landscape while the model remains linear.

## Recommended First Parameters

```text
kappa = 3
degree d in {1, 2, 4}
optional later: degree d in {6, 8}
delta: small positive numerical stabilizer
clip_mode: max0
rescale_mode: graph_mean_to_clean_linear_mean
label_seed: 20260523 for the first strict Step2b run, although this noiseless version is deterministic after graph generation
graph_source: reuse the same graph structures as Step1b/Step1c when possible
```

## Dataset Naming

Formal processed datasets should stay under `dataset/processed/`, not inside this experiment directory. Use the shared Step2 short naming convention:

```text
dataset/processed/step2b_poly_d1_main2000_seed20260523
dataset/processed/step2b_poly_d1_val2000_seed20260523
dataset/processed/step2b_poly_d1_unseen10000_seed20260523

dataset/processed/step2b_poly_d2_main2000_seed20260523
dataset/processed/step2b_poly_d2_val2000_seed20260523
dataset/processed/step2b_poly_d2_unseen10000_seed20260523

dataset/processed/step2b_poly_d4_main2000_seed20260523
dataset/processed/step2b_poly_d4_val2000_seed20260523
dataset/processed/step2b_poly_d4_unseen10000_seed20260523
```

The graph structures should come from the same raw source directories used by Step2a:

```text
main2000:
  dataset/raw/2026-04-17_135607

val2000:
  dataset/raw/2026-05-19_000000__step1_noisy_linear_sigma010_validation2000_seed20260519

unseen10000:
  dataset/raw/2026-05-20_000000__step1_noisy_linear_sigma010_unseen_test10000_seed20260520
```

## Fixed Protocol

Use the same method and split protocol across all degrees:

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

This directory contains a Step2b-specific copy of the root processing script:

```text
data-processing.py
```

It currently supports:

```text
--label_mode step2b_polynomial_degree_noiseless
--step2b_degree <int>
--step2b_kappa <float>
--step2b_delta <float>
--label_seed <int>
```

For each graph, the script computes `mu_G` and `mean_G(r)` over raw valid matches, then assigns each edge:

```text
b_e = 10 * utility/100 + 5 * cPRA
q_e = b_e / (mu_G + delta)
r_e = (q_e + kappa)^degree - kappa^degree
w_syn_e = max(0, mu_G * r_e / (mean_G(r) + delta))
```

The output preserves diagnostic fields:

```text
latent_clean_linear_label
step2b_polynomial_label
step2b_q_value
step2b_polynomial_score
step2b_graph_clean_linear_mean
step2b_graph_polynomial_score_mean
step2b_graph_clean_linear_edge_count
step2b_degree
step2b_kappa
step2b_delta
```

Useful smoke command from the repo root:

```bash
python surrogate_experiment_results/Step2/Step2b_polynomial_degree_noiseless/data-processing.py \
  dataset/raw/2026-04-17_135607 \
  /tmp/step2b_smoke_processed \
  --file genjson-0.json \
  --label_mode step2b_polynomial_degree_noiseless \
  --step2b_degree 2 \
  --step2b_kappa 3 \
  --step2b_delta 0 \
  --label_seed 20260523 \
  --output_as_batch_dir
```

## Required Diagnostics

For every degree, record:

```text
mean label
std label
min/max label
fraction clipped to zero
oracle objective distribution
correlation between b_e and w_syn_e
```

The graph-level rescaling is mandatory for fair comparison. Without it, higher degrees can inflate reward scale and make raw decision gaps incomparable.

## Expected Interpretation

Expected qualitative behavior:

```text
degree = 1: close to the linear benchmark, so 2stage, FY, and SPO+ may be similar.
degree = 2: mild misspecification; decision-aware methods may start to help.
degree = 4+: stronger misspecification; SPO+ may become more competitive with or better than 2stage.
```

The key result is not a single best method. The key result is how the relative performance changes as degree increases.
