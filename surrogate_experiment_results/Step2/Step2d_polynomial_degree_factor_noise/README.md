# Step2d: Polynomial Degree With Graph-Correlated Factor Noise Label Regime

## Purpose

Step2d is the correlated-noise extension of Step2. It starts from the Step2b nonlinear mean, then adds graph-level latent factor noise. This is inspired by the SPO portfolio experiment, where returns are generated from a conditional mean plus correlated factor noise.

This regime asks:

> When reward noise is correlated within a graph rather than independent across edges, do FY and SPO+ behave differently?

This is expected to be more difficult to interpret than Step2a/2b/2c, so it should be treated as a second-stage experiment after the simpler regimes are working.

## Label Definition

First compute the Step2b nonlinear mean:

```text
b_e = 10 u_e + 5 c_e
mu_G = mean_e b_e
q_e = b_e / (mu_G + delta)
r_e^(d) = (q_e + kappa)^d - kappa^d
m_e^(d) = mu_G * r_e^(d) / (mean_e r_e^(d) + delta)
```

For each graph G, draw latent factors:

```text
f_G ~ Normal(0, I_K)
```

Use standardized edge loadings such as:

```text
ell_e = [u_e, c_e, u_e * c_e]
```

Then add correlated plus idiosyncratic noise:

```text
xi_e ~ Normal(0, 1)
nu_e = tau * mu_G * (ell_e^T f_G + sigma * xi_e)
```

and define:

```text
w_syn_e = max(0, m_e^(d) + nu_e)
```

The factor term makes errors correlated across edges in the same graph. The sigma term adds edge-wise residual noise.

## Recommended First Parameters

```text
K = 3
kappa = 3
degree d in {1, 4}
tau in {0.25, 0.5}
sigma = 0.1
delta: small positive numerical stabilizer
clip_mode: max0
rescale_mode: graph_mean_to_clean_linear_mean before noise
label_seed: fixed and recorded
graph_source: reuse the same graph structures as Step1b/Step1c when possible
```

## Fixed Protocol

Use the same method and split protocol if this regime is promoted to a full run:

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

For every factor-noise setting, record:

```text
mean label
std label
min/max label
fraction clipped to zero
oracle objective distribution
within-graph noise correlation summary if available
correlation between nonlinear mean m_e^(d) and noisy label w_syn_e
```

This regime can easily become a sparse/censored reward experiment if tau is too large and many edges clip to zero. The clipping fraction must be checked before using the results.

## Expected Interpretation

Expected qualitative behavior:

```text
Step2c tests independent multiplicative noise.
Step2d tests graph-correlated additive/factor noise.
SPO+ may react more strongly to adversarial margin changes.
FY may behave more like a robustness-smoothed optimizer.
The better method is not obvious in advance; the regime is diagnostic.
```

Step2d should come after Step2a/2b/2c, not before them. It is a richer stress test rather than the first proof-of-concept.

