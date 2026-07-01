# Step4 Decision Overlay

This is the second Step4 layer. It overlays decisions on the pure topology
substrate from Step4 Topology Structural Atlas.

For each audited context it maps:

```text
oracle selected edge set
2stage selected edge set
SPO+ selected edge set
```

back to structural `candidate_id`s from:

```text
surrogate_experiment_results/Step4 Topology Structural Atlas/results/feasible_candidates.csv
```

The summary then reports candidate selection frequencies by:

```text
topology_id × sample_size × candidate_id
```

Default command is intentionally bounded to 25 test contexts per job. Full
1000-context overlays are possible with `--context-limit 1000`, but should be
launched deliberately on garnet.

## Local Dry Run

```bash
python 'surrogate_experiment_results/Step4 Decision Overlay/scripts/compute_decision_overlay.py' \
  --dry-run \
  --topology-ids G-269 \
  --data-seeds 101 \
  --sample-sizes 50
```

## Garnet Smoke

```bash
cd /local1/fuweik/UdeM-Intern
source configs/runtime/garnet.env

python 'surrogate_experiment_results/Step4 Decision Overlay/scripts/compute_decision_overlay.py' \
  --topology-ids G-269 \
  --data-seeds 101 \
  --sample-sizes 50 \
  --context-limit 5 \
  --decision-output 'surrogate_experiment_results/Step4 Decision Overlay/results/smoke/g269_seed101_sample50_decisions.csv' \
  --summary-output 'surrogate_experiment_results/Step4 Decision Overlay/results/smoke/g269_seed101_sample50_candidate_overlay.csv'
```

## Garnet Full 8-Topology Run

The full 8-topology overlay is launched as 24 parallel batches:

```text
8 topology IDs x 3 sample sizes = 24 batches
each batch = 5 data seeds x 1000 test contexts
Gurobi Threads = 1 per batch
```

Run from the repo root on garnet after sourcing the runtime environment:

```bash
bash 'surrogate_experiment_results/Step4 Decision Overlay/scripts/run_full_8topology_24batch_overlay.sh'
```

Outputs:

```text
results/full_8topology_5seed_3size/per_batch/
results/full_8topology_5seed_3size/decision_solution_rows.csv
results/full_8topology_5seed_3size/candidate_overlay_summary.csv
logs/full_8topology_5seed_3size_20260701/
```

Outputs:

```text
results/decision_solution_rows.csv
results/candidate_overlay_summary.csv
```

These outputs are the input to Step4 Rank-Reversal Detail.
