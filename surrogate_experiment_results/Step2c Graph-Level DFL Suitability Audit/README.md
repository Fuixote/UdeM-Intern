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

## Paper-Facing Takeaway

The main result is not that raw graph topology alone predicts when SPO+ helps.
The stronger and safer conclusion is:

```text
Top20 prediction-boundary diagnostics identify graph instances where DFL-style
reranking can matter, but they do not determine whether reranking helps or
hurts. Helpful cases require near-oracle candidates inside the candidate list;
harmful cases show that ambiguity alone can also lead SPO+ to promote worse
candidates.
```

This gives a clean story linking this audit to the mechanism-dissection audit:

```text
1. Raw topology / feasible-set features give weak-to-modest association signal.
2. Post-training two-stage candidate-boundary features add clearer signal.
3. Top20 features are more informative than top5 for deep-reranking cases.
4. High ambiguity marks opportunity/risk, not guaranteed SPO+ improvement.
5. Candidate identity, true rank, and critical-edge diagnostics are still needed
   to decide whether reranking is beneficial.
```

### Results Worth Reporting

| Finding | Evidence | Safe interpretation |
| --- | --- | --- |
| No simple topology-only rule | Phase 1 single-feature Spearman correlations with `median_delta_pp` are weak across feature families. | DFL advantage is not explained by density, cycle count, or another single raw graph descriptor. |
| Boundary diagnostics add signal | Phase 2 top5 `ranking_ambiguity_score` has helpful AUROC `0.715`. | After training a standard two-stage model, its candidate-list boundary contains information about whether DFL reranking may matter. |
| Top20 is better than top5 for deep reranking | Phase 4 pilot improves the ambiguity signal from top5 harmful AUROC `0.577` to top20 harmful AUROC `0.732`. | Important cases such as `G-1169` and `G-1449` are deep top20 promotion cases that top5 under-describes. |
| Targeted all50 top20 run is the strongest denominator | Selected+matched run has `160,000 = 160 graphs x 50 subset seeds x 20 ranks` raw solution rows. | The key top20 claims are stable on the paper-critical target and matched-control set, not only on one selected seed. |
| Best helpful top20 feature | `mean_2stage_top20_within_1pct_count`: helpful AUROC `0.752`, harmful AUROC `0.601`. | Rich near-tie structure in the two-stage top20 list is associated with useful reranking opportunities. |
| Ambiguity is also a risk marker | `ranking_ambiguity_top20_score`: helpful AUROC `0.745`, harmful AUROC `0.810`. | Ambiguity means reranking can matter; it does not say the reranking will be beneficial. |
| Harmful controls validate the caution | `G-14` and `G-163` both have top20 ambiguity percentile `0.80` within matched controls. | SPO+ can strongly rerank on ambiguous graphs and still hurt if it promotes a non-near-oracle candidate. |
| Clean top20-boundary helpful cases | `G-1169`: ambiguity percentile `1.00`, within-1pct percentile `0.95`; `G-1449`: within-1pct percentile `0.80`. | These are the best examples for the top20 boundary story. |
| Some successes need mechanism-specific explanation | `G-392`, `G-1285`, and `G-1560` have top20 ambiguity percentiles `0.25`, `0.60`, and `0.45`. | These should be explained through candidate identity, rank reversal, and critical edges rather than a generic ambiguity rule. |

## Experiment Setting and Notation

Primary regime and solver settings:

| Item | Setting |
| --- | --- |
| Regime | `step2c_poly_d8_mult_eps050` |
| Dataset root | `dataset/processed/step2c_poly_d8_mult_eps050_main2000_seed20260523` |
| Base graph population | 400 heldout graphs from `master_split_seed=42` |
| Model-seed dimension | `subset_seed = 0..49`, each corresponding to a fixed-pool training subset and trained checkpoint |
| Compared methods | `2stage_val_mse` and `spoplus_val_spoplus_loss` for outcome summaries; Phase 2/4 boundary features are computed from two-stage predicted candidate lists |
| Solver constraints | `max_cycle=3`, `max_chain=4` |
| Phase 1/2/3 population | 400 heldout graphs |
| Phase 4 pilot | all 400 graphs, `subset_seed=0..4`, two-stage top20 |
| Phase 4 selected+matched run | 9 selected target graphs plus 151 unique topology-matched controls, all 50 subset seeds, two-stage top20 |

Notation used below:

```text
g: heldout KEP graph
s: subset_seed / trained-model seed
m: method, either 2stage or SPO+
r: predicted solution rank after repeated no-good cuts
y*(g): oracle solution under true labels
y_{m,r}(g,s): rank-r solution predicted by method m on graph g and seed s
V_g(y): true objective value of solution y on graph g
P_{m,r}(g,s): predicted objective score used by method m for rank-r solution
E(y): selected edge/cycle-chain signature represented as an edge set
```

The normalized oracle gap is reported in percentage points:

```text
gap_{m,r}(g,s) =
  100 * (V_g(y*(g)) - V_g(y_{m,r}(g,s))) / (abs(V_g(y*(g))) + 1e-9)
```

The method improvement used in the graph-level outcome tables is:

```text
delta_pp(g,s) = gap_{2stage,1}(g,s) - gap_{SPO+,1}(g,s)
```

Positive `delta_pp` means SPO+ has lower normalized gap than two-stage.
Graph-level summaries use the mean or median of `delta_pp(g,s)` over
`subset_seed = 0..49`.

The descriptive graph labels are:

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

These are diagnostic labels, not causal ground truth.

### Prediction-Boundary Feature Definitions

For a fixed graph `g` and seed `s`, Phase 4 computes two-stage top20 solutions
with the no-good-cut replay. Let `K=20`, let rank 1 be the highest predicted
score, and let `rank K` be the twentieth distinct candidate when it exists.

Predicted margin from the best candidate:

```text
margin_{1,r}(g,s) = P_{2stage,1}(g,s) - P_{2stage,r}(g,s)
```

Normalized top20 margin:

```text
top1_top20_margin_pct(g,s) =
  margin_{1,20}(g,s) / (abs(P_{2stage,1}(g,s)) + 1e-9)
```

Number of near-tie candidates inside top20:

```text
top20_within_tau_count(g,s) =
  count{r in 1..20 : margin_{1,r}(g,s) /
    (abs(P_{2stage,1}(g,s)) + 1e-9) <= tau}
```

The reported features use `tau = 0.01` and `tau = 0.05`, then average the count
over all available subset seeds.

Solution-overlap features use edge-set Jaccard:

```text
J(A,B) = |E(A) cap E(B)| / |E(A) cup E(B)|
```

Top20 diversity from rank1:

```text
diversity_from_rank1(g,s) =
  1 - mean_{r=2..20} J(y_{2stage,1}(g,s), y_{2stage,r}(g,s))
```

Top20 pairwise diversity:

```text
pairwise_diversity(g,s) =
  1 - mean_{1 <= i < j <= 20} J(y_{2stage,i}(g,s), y_{2stage,j}(g,s))
```

The composite top20 ambiguity score is implemented as a z-score composite over
graphs with top20 coverage:

```text
ranking_ambiguity_top20_score =
  - z(median_2stage_top1_top20_pred_margin_pct)
  + z(mean_2stage_top20_within_1pct_count)
  + z(median_2stage_top20_diversity_from_rank1)
  + z(median_2stage_top20_pairwise_diversity)
```

Larger values mean smaller predicted margins, more near ties, and more diverse
candidate solutions. Missing top20 coverage is assigned `NaN` for this score,
so partially generated graphs do not receive misleading finite ambiguity values.

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

### Phase 3 Inputs

```text
results/step2c_all400_graph_boundary_outcome_table.csv
```

This input is the Phase 2 joined all-400 table, containing Phase 1 graph /
feasible-set features, all-400 model-seed outcomes, and Phase 2
prediction-boundary features.

### Phase 3 Script

```text
scripts/build_phase3_matched_controls.py
```

The experiment-local script is a thin wrapper around:

```text
surrogate_experiment_results/decision_analysis/scripts/
  build_step2c_matched_controls_suitability.py
```

### Phase 3 Outputs

```text
results/step2c_phase3_matched_controls.csv
results/step2c_phase3_target_vs_matched_summary.csv
presentation/step2c_phase3_matched_controls_story.md
```

The default Phase 3 run uses 20 matched controls per README target graph. The
control pool excludes the README target set itself, so the selected success /
both-poor / harmful graphs do not serve as each other's matched controls.

Allowed claim:

```text
Selected target cases remain unusual, or do not remain unusual, relative to
graphs with similar coarse raw topology.
```

Important boundary:

```text
The matching variables are intentionally coarse raw-topology descriptors. A
target that remains extreme after this match supports graph-instance
specificity beyond these coarse descriptors, but it still does not prove
topology causality.
```

## Phase 4: All-400 Top20 Prediction-Boundary Extension

Phase 4 follows the expert-prioritized next step after Phase 1/2/3. Phase 2
used only existing all-400 2stage top5 candidate lists. The mechanism
dissection audit showed that several important cases, especially G-392,
G-1169, and G-1449, are top20/deep-reranking cases. Therefore Phase 4 extends
the prediction-boundary diagnostics from top5 to top20.

Phase 4 still uses post-training 2stage predicted candidates, so it is not a
topology-only diagnostic.

### Phase 4 Raw Candidate Generation

Generate the raw all-400 top20 2stage candidate artifact with the existing
no-good-cut solver path:

```bash
python surrogate_experiment_results/decision_analysis/scripts/compute_second_best_solutions.py \
  --regime step2c_poly_d8_mult_eps050 \
  --dataset-dir dataset/processed/step2c_poly_d8_mult_eps050_main2000_seed20260523 \
  --subset-seed-start 0 \
  --subset-seed-stop 49 \
  --case-type-prefix subset_seed \
  --method-labels 2stage_val_mse \
  --max-cycle 3 \
  --max-chain 4 \
  --max-solutions 20 \
  --max-cut-attempts 80 \
  --output surrogate_experiment_results/decision_analysis/results/all400_model_seed_baseline/step2c_all400_all50_top20_2stage.csv \
  --summary-output surrogate_experiment_results/decision_analysis/results/all400_model_seed_baseline/step2c_all400_all50_top20_2stage_summary.csv \
  --progress-every 25
```

The raw top20 artifact is expected to be large and must remain ignored by git.

### Phase 4 Targeted Selected+Matched Variant

The full all-400 x all-50 top20 run is expensive. Before committing to that
full run, use the Phase 3 target/matched-control graph set as a targeted
stability audit:

```text
graphs:
  9 README target graphs
  + all unique topology-matched controls from
    results/step2c_phase3_matched_controls.csv

current unique graph count:
  160 = 9 targets + 151 unique matched controls
```

Generate a graph list from the Phase 3 matched-control table, then run:

```bash
python surrogate_experiment_results/decision_analysis/scripts/compute_second_best_solutions.py \
  --regime step2c_poly_d8_mult_eps050 \
  --dataset-dir dataset/processed/step2c_poly_d8_mult_eps050_main2000_seed20260523 \
  --subset-seed-start 0 \
  --subset-seed-stop 49 \
  --case-type-prefix subset_seed \
  --method-labels 2stage_val_mse \
  --max-cycle 3 \
  --max-chain 4 \
  --max-solutions 20 \
  --max-cut-attempts 80 \
  --graphs $(cat surrogate_experiment_results/decision_analysis/results/all400_model_seed_baseline/step2c_phase4_selected_matched_graphs.txt) \
  --output surrogate_experiment_results/decision_analysis/results/all400_model_seed_baseline/step2c_selected_matched_all50_top20_2stage.csv \
  --summary-output surrogate_experiment_results/decision_analysis/results/all400_model_seed_baseline/step2c_selected_matched_all50_top20_2stage_summary.csv \
  --progress-every 20
```

This targeted run is intended to test whether top20 boundary diagnostics are
stable across all 50 model seeds for the cases that matter most to the paper:
selected helpful cases, both-poor controls, harmful controls, and their
topology-matched controls.

### Phase 4 Inputs

```text
surrogate_experiment_results/decision_analysis/results/all400_model_seed_baseline/
  step2c_all400_all50_top20_2stage.csv

results/step2c_all400_graph_boundary_outcome_table.csv
```

### Phase 4 Script

```text
scripts/build_phase4_top20_prediction_boundary.py
```

The experiment-local script is a thin wrapper around:

```text
surrogate_experiment_results/decision_analysis/scripts/
  build_step2c_top20_prediction_boundary_suitability.py
```

### Phase 4 Outputs

```text
results/step2c_all400_top20_prediction_boundary_features.csv
results/step2c_all400_graph_top20_boundary_outcome_table.csv
results/step2c_phase4_top20_feature_family_association.csv
results/step2c_phase4_top20_selected_case_overlay.csv
results/step2c_phase4_top20_target_vs_matched_summary.csv
presentation/step2c_phase4_top20_boundary_story.md
```

For the targeted selected+matched all50 variant, the curated outputs use the
same schema with the `step2c_selected_matched_all50_...` prefix:

```text
results/step2c_selected_matched_all50_top20_prediction_boundary_features.csv
results/step2c_selected_matched_all50_graph_top20_boundary_outcome_table.csv
results/step2c_selected_matched_all50_top20_feature_family_association.csv
results/step2c_selected_matched_all50_top20_selected_case_overlay.csv
results/step2c_selected_matched_all50_top20_target_vs_matched_summary.csv
presentation/step2c_selected_matched_all50_top20_boundary_story.md
```

### Phase 4 Features

```text
2stage top1-top20 predicted margin
2stage top1-top20 predicted margin normalized by rank1 score
number of top20 solutions within 1% / 5% predicted margin
top20 mean Jaccard-to-rank1 and diversity
top20 mean pairwise Jaccard and pairwise diversity
rank1 unique-signature count and modal-signature rate across subset_seed
top20 unique-signature count
ranking_ambiguity_top20_score
```

Allowed claim:

```text
Top20 prediction-boundary diagnostics test whether the broader 2stage candidate
landscape explains graph instances that top5 diagnostics under-rank.
```

Important boundary:

```text
Top20 boundary features are still post-training diagnostics. They can support
the learned-ranking-boundary story but cannot be used as pre-training topology
causal evidence.
```

## Follow-Up Roadmap

The expert review suggests several later steps. Keep them separate from Phase 4
so that each claim has a clear denominator and evidence boundary.

```text
Phase 5:
  Multivariate cross-validated diagnostic models.
  Compare feature groups A raw topology, B cycle/chain, C feasible/conflict
  geometry, D top5 boundary, E top20 boundary.

Phase 6:
  Helpful-vs-harmful modeling inside high-ambiguity graphs.
  Test what separates beneficial reranking from harmful reranking after the
  graph is already identified as reranking-sensitive.

Phase 7:
  Matched-control robustness.
  Repeat matching with richer specs: coarse topology only; topology plus
  feasible geometry; topology plus top20 ambiguity; k = 10, 20, 50 controls.

Phase 8:
  Cross-regime robustness.
  Replicate the graph-level suitability audit on Step2b or another Step2c
  sensitivity regime before making broader distribution-level claims.
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
Status: protocol formalized; Phase 1, Phase 2, and Phase 3 implemented and run locally. Phase 4 protocol and summarizer are implemented. A 5-seed all-400 pilot is complete, and the selected+matched all50 top20 run is complete. Full all-400 x all50 top20 generation remains deferred.
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

Latest Phase 3 run:

```text
matched-control rows: 180
target-vs-matched summary rows: 9
controls per target: 20
matching variables: num_vertices, num_arcs, density, num_2cycles,
  num_3cycles, largest_scc_fraction
```

Initial Phase 3 readout:

```text
The five helpful success targets remain high relative to topology-matched
controls on median_delta_pp. G-392, G-1285, G-1449, and G-1560 are at the top
of their matched-control sets, and G-1169 is at the 95th percentile. This
supports graph-instance specificity beyond the six coarse topology variables
used for matching.

The harmful reranking controls G-14 and G-163 sit at the bottom of their
matched-control sets on median_delta_pp while also having high ambiguity
percentiles. This reinforces the Phase 2 interpretation that ambiguity marks a
reranking-sensitive boundary, not guaranteed SPO+ improvement.

The both-poor controls G-142 and G-946 are not extreme on median_delta_pp
relative to their matched controls. They are useful as failure examples in the
mechanism story, but they do not appear graph-instance-extreme under this
coarse matched-control diagnostic.
```

Latest Phase 4 pilot:

```text
run type: all-400, subset_seed 0..4, 2stage top20
raw solution rows: 39,950
expected maximum rows: 40,000 = 400 graphs x 5 seeds x 20 ranks
missing top20 rows: G-592 only reached rank10 in each pilot seed
selected target graphs affected by missing top20 rows: none
top20 boundary rows: 400
graph-top20 rows: 400
Phase 4 association rows: 68
selected case overlay rows: 9
```

Initial Phase 4 pilot readout:

```text
Top20 prediction-boundary diagnostics modestly improve the helpful-graph signal
over top5 and strongly improve the harmful/reranking-sensitive signal.

top5 ranking_ambiguity_score:
  helpful AUROC = 0.715
  harmful AUROC = 0.577
  Spearman with median_delta_pp = 0.109

top20 ranking_ambiguity_top20_score:
  helpful AUROC = 0.739
  harmful AUROC = 0.732
  Spearman with median_delta_pp = 0.063

This supports the interpretation that top20 boundary features are better
reranking-sensitivity markers than top5 features. They still do not determine
whether reranking will help or hurt. The completed targeted run below uses
selected + matched controls x all 50 seeds x top20 to test whether this top20
signal is stable for the paper-critical graph set.
```

Latest Phase 4 selected+matched all50 run:

```text
run type: 9 selected targets + 151 unique topology-matched controls,
  subset_seed 0..49, 2stage top20
raw solution rows: 160,000
expected rows: 160,000 = 160 graphs x 50 seeds x 20 ranks
top20 boundary rows: 160
graph-top20 rows: 400 joined rows, with 160 top20-covered graphs
Phase 4 association rows: 68
selected case overlay rows: 9
target-vs-matched summary rows: 9
```

Selected+matched all50 top20 readout:

```text
Best helpful top20 feature:
  mean_2stage_top20_within_1pct_count
  helpful AUROC = 0.752
  harmful AUROC = 0.601
  Spearman with median_delta_pp = 0.134

Composite top20 ambiguity:
  ranking_ambiguity_top20_score
  helpful AUROC = 0.745
  harmful AUROC = 0.810
  Spearman with median_delta_pp = 0.061

Best harmful/reranking-sensitivity top20 feature:
  median_2stage_top20_pairwise_diversity
  harmful AUROC = 0.823
  helpful AUROC = 0.658
  Spearman with median_delta_pp = -0.046
```

Target-vs-matched top20 pattern:

```text
G-1169 is the clearest high-top20-boundary helpful target:
  target top20 ambiguity percentile within matched controls = 1.00
  target top20 within-1pct percentile within matched controls = 0.95

G-1449 is also top20 near-tie rich:
  target top20 within-1pct percentile within matched controls = 0.80
  target top20 ambiguity percentile within matched controls = 0.65

G-392, G-1285, and G-1560 remain strong SPO+ mechanism cases, but their generic
top20 ambiguity percentiles within matched controls are not uniformly extreme
(0.25, 0.60, and 0.45). Their explanation should stay tied to the earlier
candidate-identity and rank-reversal evidence rather than to a generic
top20-ambiguity rule.

The harmful reranking controls are also high on top20 boundary sensitivity:
  G-14 top20 ambiguity percentile within matched controls = 0.80
  G-163 top20 ambiguity percentile within matched controls = 0.80

The both-poor controls are not cleanly separated by top20 ambiguity:
  G-142 top20 ambiguity percentile within matched controls = 0.70
  G-946 top20 ambiguity percentile within matched controls = 0.55
```

Selected+matched all50 interpretation:

```text
Top20 boundary diagnostics are stable enough to support the ranking-boundary
story on the selected target/matched-control graph set, but they are still not
a sufficient predictor of SPO+ benefit. They mark reranking opportunity/risk.
The sign and quality of the reranking still require candidate-identity,
true-rank, and critical-edge diagnostics.
```
