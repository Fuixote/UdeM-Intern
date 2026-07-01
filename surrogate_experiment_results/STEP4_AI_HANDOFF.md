# Step4 AI Handoff

This handoff is for AI tools reviewing the K18-E1 topology-mechanism analysis.
Read the files in this order.

## 1. Experiment Context

Use these Step3 K18-E1 post-run review files for protocol and outcome context:

```text
surrogate_experiment_results/Step3/K18_analysis/experiment_01_budget4to1/results/formal_270_full_epoch_20260626/post_run_review/formal_post_run_integrity_audit.json
surrogate_experiment_results/Step3/K18_analysis/experiment_01_budget4to1/results/formal_270_full_epoch_20260626/post_run_review/formal_post_run_summary.json
surrogate_experiment_results/Step3/K18_analysis/experiment_01_budget4to1/results/formal_270_full_epoch_20260626/post_run_review/formal_post_run_sample_size_summary.csv
surrogate_experiment_results/Step3/K18_analysis/experiment_01_budget4to1/results/formal_270_full_epoch_20260626/post_run_review/formal_post_run_topology_sample_summary.csv
surrogate_experiment_results/Step3/K18_analysis/experiment_01_budget4to1/results/formal_270_full_epoch_20260626/post_run_review/formal_post_run_vertical_pattern_check.csv
```

Important convention:

```text
Delta = test_mean_decision_gap_2stage - test_mean_decision_gap_spoplus
Delta > 0 means SPO+ has lower downstream decision gap than 2stage.
sample_size 50/100/500 means train/validation splits 40/10, 80/20, 400/100.
```

## 2. Structural Substrate

Read all compact structural CSVs:

```text
surrogate_experiment_results/Step4 Topology Structural Atlas/results/topology_summary.csv
surrogate_experiment_results/Step4 Topology Structural Atlas/results/compatibility_arcs.csv
surrogate_experiment_results/Step4 Topology Structural Atlas/results/feasible_candidates.csv
surrogate_experiment_results/Step4 Topology Structural Atlas/results/candidate_conflicts.csv
```

These define the fixed compatibility graph, feasible cycle/chain candidates,
and candidate conflict graph. No model weights, data seed, sample size, or test
sample is needed for this layer.

## 3. Decision Overlay

Read:

```text
surrogate_experiment_results/Step4 Decision Overlay/results/full_8topology_5seed_3size/candidate_overlay_summary.csv
```

This table gives candidate selection rates under oracle, 2stage, and SPO+ by:

```text
topology_id x sample_size x candidate_id
```

It is the compact replacement for the large raw overlay file:

```text
surrogate_experiment_results/Step4 Decision Overlay/results/full_8topology_5seed_3size/decision_solution_rows.csv
```

The raw file is intentionally not tracked in git because it is about 199 MB and
exceeds GitHub's ordinary 100 MB file limit.

## 4. Rank-Reversal Detail

Read these summary/target tables:

```text
surrogate_experiment_results/Step4 Rank-Reversal Detail/results/full_8topology_5seed_3size/rank_reversal_case_summary_by_topology_sample.csv
surrogate_experiment_results/Step4 Rank-Reversal Detail/results/full_8topology_5seed_3size/candidate_set_switch_summary.csv
surrogate_experiment_results/Step4 Rank-Reversal Detail/results/full_8topology_5seed_3size/rank_reversal_top20_per_topology.csv
surrogate_experiment_results/Step4 Rank-Reversal Detail/results/full_8topology_5seed_3size/rank_reversal_all_different_contexts.csv
```

Use the case summary to compare reversal rates across topology/sample size.
Use the switch summary to identify recurring candidate-set substitutions.
Use top20 targets for detailed Top-M or visualization follow-up. Use the
all-different table only when re-filtering or auditing all non-identical
2stage/SPO+ decisions.

## 5. Interpretation Targets

Useful starting points:

```text
G-398: 2stage c0010|c0030 -> SPO+ c0002|c0009|c0032, clean beneficial case.
G-269: 2stage c0005|c0033 -> SPO+ c0003|c0010, strong beneficial case.
G-784: 2stage c0026|c0043 -> SPO+ c0027|c0042, rich candidate replacement.
G-970: sample-size-responsive positive; SPO+ gradually leaves a fixed 2stage solution.
G-364: sparse sample-size-responsive positive; switch toward c0006 increases with sample size.
G-79: harmful control; SPO+ increasingly favors a harmful candidate set.
G-836: small negative; 2stage and SPO+ usually select the same candidate set.
G-670: no-room neutral; oracle, 2stage, and SPO+ are decision-equivalent.
```

## 6. Reproduction Scripts

The Step4 scripts are:

```text
surrogate_experiment_results/Step4 Topology Structural Atlas/scripts/build_topology_structural_atlas.py
surrogate_experiment_results/Step4 Decision Overlay/scripts/compute_decision_overlay.py
surrogate_experiment_results/Step4 Decision Overlay/scripts/run_full_8topology_24batch_overlay.sh
surrogate_experiment_results/Step4 Rank-Reversal Detail/scripts/build_rank_reversal_detail_plan.py
surrogate_experiment_results/Step4 Rank-Reversal Detail/scripts/summarize_rank_reversal_detail.py
```

Do not assume the large raw full overlay files are present in a GitHub clone.
They are stored locally/lab-pc/garnet and should be regenerated or fetched from
external artifacts if needed.
