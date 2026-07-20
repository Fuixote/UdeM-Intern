# Experiment 05 — topology-only GNN regression

This directory is a local scaffold, not a started GNN experiment. Formal model
training is blocked until Experiment 04 finishes all 120 repeat-seed jobs and
its integrity and stability gates are reviewed.

## Prediction contract

Input is pure topology. Each topology becomes one relation-aware homogeneous
graph containing the original pair/NDD vertices and one node for every feasible
cycle or chain candidate. Directed compatibility arcs retain the original
topology; bidirectional incidence edges connect each candidate to its member
vertices. No sampled train/validation/test context, objective gap, seed, or
label-derived statistic is an input feature.

The provisional target was Experiment 03's seed-42
`normalized_improvement_pp = 100 * (normalized_gap_2stage - normalized_gap_SPO+)`.
Following Experiment 04, the formal target policy is locked to the mean of that
quantity over train seeds 42, 43, and 44. Label uncertainty is the population
standard deviation over those same three values (`ddof=0`). Both fields remain
separate from graph inputs.

Only the 60 Experiment 04 audit topologies currently have all three seed
values. The other 940 topologies require two additional jobs each, so 1,880
label-generation jobs remain before a protocol-consistent 1,000-topology GNN
dataset exists. Missing topologies must have an empty formal target; the
pipeline is forbidden from silently substituting their seed-42 value.

## Implemented local scaffold

```bash
python surrogate_experiment_results/Step5_topology_dfl_suitability/experiment_05_topology_gnn_regression/scripts/build_incidence_dataset.py
python surrogate_experiment_results/Step5_topology_dfl_suitability/experiment_05_topology_gnn_regression/scripts/plan_stratified_folds.py
python surrogate_experiment_results/Step5_topology_dfl_suitability/experiment_05_topology_gnn_regression/scripts/run_scalar_baselines.py
python surrogate_experiment_results/Step5_topology_dfl_suitability/experiment_05_topology_gnn_regression/scripts/build_multiseed_targets.py
```

Once the target audit reports all 1,000 topologies complete, the formal graph
build must explicitly use the target table and strict mode:

```bash
python surrogate_experiment_results/Step5_topology_dfl_suitability/experiment_05_topology_gnn_regression/scripts/build_incidence_dataset.py \
  --target-table surrogate_experiment_results/Step5_topology_dfl_suitability/experiment_05_topology_gnn_regression/results/scaffold/targets/multiseed_targets.csv \
  --require-formal-targets
```

These commands create:

- `topology_incidence_graphs.jsonl` plus a no-leakage/uniqueness audit;
- deterministic five-fold assignments, balanced within each target stratum;
- out-of-fold zero, fold-mean, and ridge predictions and metrics on all,
  nonzero, and material-label subsets.

The model specification proposes a three-layer relation-aware incidence GNN,
type-conditioned graph pooling, and a regression head. It intentionally does
not contain executable PyTorch/PyG training code yet: dependency choice, final
target aggregation, and formal seeds are locked only after Experiment 04.

## Local scaffold validation (2026-07-20)

All 1,000 topology templates were converted successfully. The audit found
1,000 unique topology hashes and feasible-set hashes, all five folds contain
exactly 200 topologies, and every target stratum differs by at most one example
between folds. The resulting graph JSONL is a provisional feature artifact;
its target remains explicitly marked as seed-42-only.

The out-of-fold scalar baselines confirm the earlier exploratory conclusion:
the 21 aggregate topology statistics are weak predictors. On all 1,000 graphs,
the zero baseline has MAE 1.362 pp and RMSE 5.199 pp; ridge has MAE 2.215 pp,
RMSE 5.028 pp, R2 0.002, and Spearman 0.144. On the 241 material-label graphs,
ridge has MAE 5.260 pp and R2 -0.187. These results justify testing structural
message passing later, but do not authorize formal GNN training before the seed
audit.

## Start gate

Formal GNN training may start only after all conditions are true:

1. Experiment 04 artifact audit proves 120 fresh train/validation bundles use
   the exact Experiment 03 test hashes.
2. All 120 paired 2stage/SPO+ jobs and evaluations succeed and early-stop.
3. The three-seed review is interpreted and the target aggregation policy is
   recorded here and in `configs/experiment.yaml`.
4. All 1,000 topologies have seeds 42/43/44, and the strict multi-seed target
   audit reports `formal_ready=true`.
5. Fold and baseline audits are rebuilt and pass with the final target table.
