# Experiment 05 — topology-only GNN regression

This directory contains the topology-only GNN scaffold and the completed formal
three-seed label pipeline. The strict target and graph audits now pass for all
1,000 topologies. GNN model training has not started; the final folds and scalar
baselines must first be rebuilt against the formal target table.

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

At launch time, only the 60 Experiment 04 audit topologies had all three seed
values. The other 940 topologies required two additional jobs each. Those 1,880
label-generation jobs are complete, and no topology is missing a formal target.
The pipeline never substituted a seed-42-only value.

## Formal label completion run (completed 2026-07-20)

The Garnet run started at `2026-07-19T21:57:04-04:00` in tmux session
`step5_exp5_label_completion`. Its output root is
`results/multiseed_completion1880`. The pre-launch artifact audit passed all
1,880 seed bundles over 940 topologies with zero failures. The strict plan and
CSV contain 1,880 unique ready jobs: 940 for seed 43 and 940 for seed 44. All
stored commands are dry-run commands, and the launcher preview passed before
the execute launcher converted them in memory.

The execute launcher used 24 normal workers and no long-job queue. All 1,880
jobs completed successfully with zero launcher failures. Each job retained
the locked sample-50 protocol (40 train, 10 validation, fixed 1,000-sample test
bank), `max_epochs=10000`, early-stop patience 20, and min delta 0.0001. The
pipeline review explicitly accepts `G-122@44` and `G-670@44` as audited SPO+
epoch-cap checkpoints: both methods and evaluation succeeded, both stopped at
the locked 10,000 epoch cap, and their decision behavior was already valid.
No other cap hit is accepted implicitly.

The finalize-only run completed at `2026-07-20T01:39:42-04:00`. The completion
review passed 1,880/1,880 labels with zero failures. The formal target audit
reports 1,000 complete three-seed topologies, zero missing seeds, and
`formal_ready=true`. The graph audit reports 1,000 records, 1,000 unique
topology/feasible-set hashes, no target leakage, and zero failures. The target
CSV SHA-256 is `e1b2cd6cfb575ce7924d07a665fc7c0755dc1758aad7ceb8c8a18d1d17a29f8b`;
the graph JSONL SHA-256 is
`8a41232110bf7f151c0192c6723fe5e0d7f4b9dd83aa70aba7fa142e503fd522`.
The pipeline ended with `complete_gnn_not_started`; it did not launch a GNN.

The original Brevo watcher sent its final email when the first pipeline stopped
at the strict two-cap-hit review gate. A second external watcher was not started
for the finalize-only recovery because external result/log transmission was not
approved at that step. The original watcher log is
`results/multiseed_completion1880/logs/notify_step5_exp5_label_completion.log`.

## Implemented local scaffold

```bash
python surrogate_experiment_results/Step5_topology_dfl_suitability/experiment_05_topology_gnn_regression/scripts/build_incidence_dataset.py
python surrogate_experiment_results/Step5_topology_dfl_suitability/experiment_05_topology_gnn_regression/scripts/plan_stratified_folds.py
python surrogate_experiment_results/Step5_topology_dfl_suitability/experiment_05_topology_gnn_regression/scripts/run_scalar_baselines.py
python surrogate_experiment_results/Step5_topology_dfl_suitability/experiment_05_topology_gnn_regression/scripts/build_multiseed_targets.py
```

The completed formal graph build explicitly used the target table and strict
mode:

```bash
python surrogate_experiment_results/Step5_topology_dfl_suitability/experiment_05_topology_gnn_regression/scripts/build_incidence_dataset.py \
  --target-table surrogate_experiment_results/Step5_topology_dfl_suitability/experiment_05_topology_gnn_regression/results/multiseed_completion1880/results/multiseed_targets.csv \
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

Formal GNN training may start only after all conditions are true. Conditions
1-4 are complete; condition 5 remains pending:

1. Experiment 04 artifact audit proves 120 fresh train/validation bundles use
   the exact Experiment 03 test hashes.
2. All 120 paired 2stage/SPO+ jobs and evaluations succeed and early-stop.
3. The three-seed review is interpreted and the target aggregation policy is
   recorded here and in `configs/experiment.yaml`.
4. All 1,000 topologies have seeds 42/43/44, and the strict multi-seed target
   audit reports `formal_ready=true`.
5. Fold and baseline audits are rebuilt and pass with the final target table.
