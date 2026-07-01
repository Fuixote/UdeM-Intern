# Step4 Rank-Reversal Detail

This is the third Step4 layer. It starts from Step4 Decision Overlay outputs and
selects contexts where 2stage and SPO+ choose different structural candidates.

The target planner does not replace Top-M analysis. It narrows the Top-M work to
contexts that are structurally meaningful:

```text
same topology
same fixed test context
different selected candidate set under 2stage vs SPO+
known true objective delta between the two selected decisions
```

Run after Step4 Decision Overlay has produced `decision_solution_rows.csv`:

```bash
python 'surrogate_experiment_results/Step4 Rank-Reversal Detail/scripts/build_rank_reversal_detail_plan.py'
```

For a smoke overlay output:

```bash
python 'surrogate_experiment_results/Step4 Rank-Reversal Detail/scripts/build_rank_reversal_detail_plan.py' \
  --decision-rows 'surrogate_experiment_results/Step4 Decision Overlay/results/smoke/g269_seed101_sample50_decisions.csv' \
  --output 'surrogate_experiment_results/Step4 Rank-Reversal Detail/results/smoke/g269_seed101_sample50_rank_reversal_targets.csv'
```

Output:

```text
results/rank_reversal_target_contexts.csv
```

## Full 8-Topology Run

After the full Step4 Decision Overlay run, build both the full set of differing
contexts and a small Top-20-per-topology inspection table:

```bash
python 'surrogate_experiment_results/Step4 Rank-Reversal Detail/scripts/build_rank_reversal_detail_plan.py' \
  --decision-rows 'surrogate_experiment_results/Step4 Decision Overlay/results/full_8topology_5seed_3size/decision_solution_rows.csv' \
  --output 'surrogate_experiment_results/Step4 Rank-Reversal Detail/results/full_8topology_5seed_3size/rank_reversal_all_different_contexts.csv' \
  --targets-per-topology 20000

python 'surrogate_experiment_results/Step4 Rank-Reversal Detail/scripts/build_rank_reversal_detail_plan.py' \
  --decision-rows 'surrogate_experiment_results/Step4 Decision Overlay/results/full_8topology_5seed_3size/decision_solution_rows.csv' \
  --output 'surrogate_experiment_results/Step4 Rank-Reversal Detail/results/full_8topology_5seed_3size/rank_reversal_top20_per_topology.csv' \
  --targets-per-topology 20
```

Then summarize reversal rates and recurring candidate-set substitutions:

```bash
python 'surrogate_experiment_results/Step4 Rank-Reversal Detail/scripts/summarize_rank_reversal_detail.py'
```

Full-run outputs:

```text
results/full_8topology_5seed_3size/rank_reversal_all_different_contexts.csv
results/full_8topology_5seed_3size/rank_reversal_top20_per_topology.csv
results/full_8topology_5seed_3size/rank_reversal_case_summary_by_topology_sample.csv
results/full_8topology_5seed_3size/candidate_set_switch_summary.csv
```

The target table is the bridge to detailed Top-M/rank-reversal computation:
for each selected context, compute predicted top-M under 2stage/SPO+ and true
oracle top-M, then inspect which candidate reordering caused the beneficial or
harmful decision difference.
