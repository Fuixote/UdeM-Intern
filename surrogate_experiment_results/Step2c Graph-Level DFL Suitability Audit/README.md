# Step2c Graph-Level DFL Suitability Audit

## Purpose

This audit asks a graph-level diagnostic question:

```text
Given a Step2c KEP compatibility graph, can observable graph and feasible-set
descriptors help characterize when decision-focused learning is likely to help,
hurt, or be unnecessary?
```

This is not a topology-causality experiment. The safe target claim is:

```text
SPO+ gains are graph-instance / feasible-solution-landscape conditioned.
Raw topology may contain signal, but decision-focused gains should be interpreted
through feasible-set geometry and learned ranking boundaries.
```

The audit builds directly on the completed Step2c mechanism-dissection results:

```text
../Step2c Mechanism Dissection Audit/results/
../Step2c Mechanism Dissection Audit/presentation/
```

Those results show that selected success cases are mostly decision-critical
reranking cases: 2stage often has a near-oracle candidate in its predicted
candidate list but ranks it too low, while SPO+ promotes it to rank1. Negative
controls show that reranking can also hurt when the promoted candidate is not
near-oracle.

## Evidence Boundary

Keep three feature classes separate.

### A. Prospectively Available Graph Features

These use only the compatibility graph before model predictions or true labels:

```text
num_vertices
num_arcs
density
in/out degree distribution
degree Gini
reciprocity
weak/strong component structure
2-cycle and 3-cycle counts
source/sink counts
```

Allowed claim:

```text
Observable compatibility-graph structure is associated with DFL suitability.
```

### B. Feasible-Set Geometry Features

These use the feasible exchange candidates under the fixed KEP constraints
`max_cycle=3`, `max_chain=4`, but not true labels:

```text
number of feasible cycle candidates
number of feasible chain candidates by length
exchange size distribution
vertex exchange participation distribution
fraction of vertices participating in any / many exchanges
conflict graph density among exchange candidates
conflict degree distribution
largest conflict component fraction
```

Allowed claim:

```text
Feasible-set geometry is associated with whether DFL-style reranking can matter.
```

### C. Retrospective Mechanism Features

These use true labels, oracle landscapes, or observed method decisions:

```text
true oracle top50
near-oracle solution counts
2stage top20 contains true near-oracle candidate
true delta from 2stage rank1 to SPO+ rank1
SPO+ rank1 true rank
critical-edge signed true deltas
```

Allowed claim:

```text
These features explain the observed mechanism after the fact.
```

Do not use retrospective features to claim a deployable pre-match diagnostic.

## Phase 1 MVP

Phase 1 intentionally avoids true oracle landscapes and remote solver reruns.
It uses graph JSON files plus the existing all-400 model-seed outcome summary.

### Inputs

```text
dataset/processed/step2c_poly_d8_mult_eps050_main2000_seed20260523/G-*.json
surrogate_experiment_results/decision_analysis/results/all400_model_seed_baseline/
  step2c_all400_all50_graph_summary.csv
```

### Script

```text
scripts/build_phase1_graph_features.py
```

The experiment-local script is a thin wrapper around:

```text
surrogate_experiment_results/decision_analysis/scripts/
  build_step2c_graph_level_suitability.py
```

### Outputs

```text
results/step2c_all400_graph_features.csv
results/step2c_all400_graph_feature_outcome_table.csv
results/step2c_feature_family_association.csv
results/step2c_selected_case_feature_overlay.csv
presentation/step2c_dfl_suitability_story.md
```

### Phase 1 Questions

```text
Q1. Are raw graph features associated with median SPO+ improvement?
Q2. Are feasible-set geometry features more informative than raw topology?
Q3. Where do the selected mechanism cases sit relative to all 400 heldout graphs?
Q4. Are harmful controls structurally distinguishable from helpful reranking cases?
```

### Phase 1 Outcome Labels

Derived from the existing all-400 model-seed graph summary:

```text
helpful_graph:
  median_delta_pp >= 10 OR strict_case_c_rate >= 0.5

extreme_helpful_graph:
  strict_case_c_rate == 1.0 OR median_delta_pp is in the all-400 top 5%

harmful_graph:
  median_delta_pp <= -10

neutral_graph:
  abs(median_delta_pp) <= 0.1 AND strict_case_c_rate == 0
```

These labels are descriptive diagnostics, not causal labels.

### Phase 1 Feature Families

```text
raw_topology:
  num_vertices, num_arcs, density, degree features, components

cycle_chain:
  num_2cycles, num_3cycles, chain counts by length, cycle_to_chain_ratio

exchange_geometry:
  number of exchange candidates, exchange size entropy,
  vertex exchange participation features

conflict_geometry:
  conflict graph density, conflict degree features,
  conflict component structure
```

### Phase 1 Analysis

For each numeric feature:

```text
Spearman correlation with median_delta_pp
AUROC for helpful_graph
AUROC for harmful_graph
feature percentile for selected mechanism cases
```

For each feature family:

```text
best absolute Spearman feature
best helpful AUROC feature
best harmful AUROC feature
```

This is an association audit, not a predictive benchmark. With only 400 graphs,
avoid overinterpreting small differences.

## Phase 2: Prediction-Boundary Diagnostics

Only after Phase 1, add features from 2stage predicted candidate lists. Start
with existing all-400 top5 artifacts before considering an all-400 top20 rerun.

Phase 2 intentionally uses the existing all-400 top5 artifact. It does not
rerun the solver for all-400 top20.

### Inputs

```text
surrogate_experiment_results/decision_analysis/results/all400_model_seed_baseline/
  step2c_all400_all50_top5_second_best.csv

results/step2c_all400_graph_feature_outcome_table.csv
```

The top5 candidate artifact is a large generated input and is not committed to
git. It should be regenerated or synced separately when reproducing Phase 2.

### Script

```text
scripts/build_phase2_prediction_boundary.py
```

The experiment-local script is a thin wrapper around:

```text
surrogate_experiment_results/decision_analysis/scripts/
  build_step2c_prediction_boundary_suitability.py
```

### Outputs

```text
results/step2c_all400_prediction_boundary_features.csv
results/step2c_all400_graph_boundary_outcome_table.csv
results/step2c_phase2_feature_family_association.csv
results/step2c_phase2_selected_case_overlay.csv
presentation/step2c_phase2_dfl_suitability_story.md
```

### Phase 2 Features

```text
2stage top1-top2 predicted margin
2stage top1-top5 predicted margin
2stage top1-top2 / top1-top5 predicted margin normalized by rank1 score
number of top5 solutions within 1% / 5% predicted margin
top5 mean Jaccard-to-rank1 and diversity
rank1 unique-signature count and modal-signature rate across subset_seed
ranking_ambiguity_score
```

The `ranking_ambiguity_score` is a descriptive composite:

```text
smaller top1-top2 margin
+ smaller top1-top5 margin
+ more top5 candidates within 1%
+ higher top5 diversity from rank1
```

Allowed claim:

```text
After training a standard 2stage model, prediction-boundary diagnostics add
signal about whether DFL reranking may be useful.
```

Important boundary:

```text
High ambiguity is an opportunity/risk signal, not a sufficient condition for
SPO+ improvement. Harmful reranking controls can also have high ambiguity.
```

## Phase 3: Matched Controls

Use matched controls after the Phase 1/2 signal is known. Match first on simple
raw topology:

```text
num_vertices
num_arcs
density
num_2cycles
num_3cycles
largest_scc_fraction
```

Then compare target cases against matched controls on higher-level
feasible-set and prediction-boundary features.

Primary target cases:

```text
Helpful success:
  G-392
  G-1285
  G-1560
  G-1169
  G-1449

Both-poor controls:
  G-142
  G-946

Harmful reranking controls:
  G-14
  G-163
```

## Report-Safe Claim Templates

Strong result:

```text
Observable feasible-set geometry contains signal about when SPO+ helps,
especially through alternative-solution richness and exchange-conflict structure.
```

Medium result:

```text
Raw topology alone is weak, but feasible-set geometry better separates helpful
reranking cases from neutral or harmful controls.
```

Weak but useful result:

```text
We do not find a simple topology-only rule. This supports the mechanism result:
DFL advantage is not a property of density or cycle count alone, but of the
interaction between feasible-solution geometry and learned ranking errors.
```

Avoid:

```text
Topology causes SPO+ success.
SPO+ helps whenever the graph is dense.
The selected cases prove a population-level topology law.
```

## Current Status

```text
Status: protocol formalized; Phase 1 and Phase 2 implemented and run locally.
Primary regime: step2c_poly_d8_mult_eps050
Graph population: 400 heldout graphs from the existing all-400 model-seed audit
Solver constraints represented in features: max_cycle=3, max_chain=4
```

Latest Phase 1 run:

```text
graph feature rows: 400
feature-outcome rows: 400
feature association rows: 43
selected case overlay rows: 9
```

Initial readout:

```text
Single-feature Spearman associations with median_delta_pp are weak across all
feature families. The strongest useful signals in Phase 1 are modest binary
AUROC signals, especially raw density for helpful_graph and max_out_degree for
harmful_graph. This supports the conservative interpretation that topology and
feasible-set descriptors alone do not give a simple DFL-suitability rule.

Phase 2 should add prediction-boundary features from existing 2stage candidate
lists before making stronger suitability claims.
```

Latest Phase 2 run:

```text
prediction-boundary feature rows: 400
graph-boundary outcome rows: 400
Phase 2 feature association rows: 55
selected case overlay rows: 9
```

Initial Phase 2 readout:

```text
Prediction-boundary diagnostics add the clearest helpful-graph signal so far:
ranking_ambiguity_score has helpful AUROC 0.715. Its Spearman association with
median_delta_pp is still weak, so this is not a simple monotone effect-size
predictor.

The same ambiguity signal is also high for harmful reranking controls such as
G-14 and G-163. Therefore Phase 2 supports a boundary/instability
interpretation: 2stage candidate ambiguity identifies graphs where DFL
reranking can matter, but it does not determine whether the reranking will be
beneficial.

Selected helpful cases split into different boundary profiles. G-1169 and
G-1449 are high-ambiguity deep-reranking cases. G-1285 combines high diversity
with clean promotion. G-392 and G-1560 remain strong mechanism cases, but their
top5 boundary scores are not uniformly extreme, consistent with the earlier
top20/rank-reversal finding that top5 diagnostics are only a partial view of
the candidate landscape.
```
