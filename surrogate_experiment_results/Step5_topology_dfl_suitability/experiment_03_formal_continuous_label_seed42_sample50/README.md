# Step5 Experiment 03: Formal continuous labels

## Status

Protocol locked on 2026-07-19; the formal 1000-topology run has not started.
The completed smoke and sample-size sensitivity experiments remain immutable
historical records with their original 1500/3000 epoch limits.

## Locked formal protocol

```text
topologies        1000
data_seed         42
sample_size       50 (40 train / 10 validation)
test_size         1000
theta_seed        42
gurobi_seed       42
master_label_seed 20260719
max_epochs        10000 (emergency safety cap only)
metric_stride     1
early_stop        patience=20, min_delta=0.0001
valid completion  both 2stage and SPO+ have should_stop=true
```

`max_epochs=10000` is not the intended stopping point. Training is expected to
stop through validation early stopping. If either method reaches 10000 without
`should_stop=true`, the job is rejected and must not contribute a formal label.

The primary label is continuous:

```text
normalized_improvement_pp = 100 * (
    test_mean_normalized_gap_2stage - test_mean_normalized_gap_spoplus
)
```

Positive values favor SPO+, negative values favor 2stage. This is the target
for the later topology-only GNN regression task; the legacy helpful/neutral/
harmful weak class is not the primary target.

## Execution contract

Use the existing Experiment 01 artifact/planner/launcher scripts with explicit
formal arguments. In particular, planning must pass `--max-epochs 10000`, and
final review must pass both `--max-epochs 10000` and
`--require-early-stop`. Omitting `--max-epochs` would silently restore the
historical planner default of 1500 and is not allowed for this experiment.

The formal reviewer exports `normalized_improvement_pp` and refuses label rows
when either early-stopping record is missing, invalid, or has
`should_stop != true`.

The critical planning and review arguments are:

```bash
PIPELINE=surrogate_experiment_results/Step5_topology_dfl_suitability/experiment_01_weak_label_seed42_sample50
FORMAL=surrogate_experiment_results/Step5_topology_dfl_suitability/experiment_03_formal_continuous_label_seed42_sample50
FORMAL_ROOT="$FORMAL/results/formal1000"

python "$PIPELINE/scripts/plan_weak_label_jobs.py" \
  --topologies-csv "$PIPELINE/configs/topologies.locked.csv" \
  --output-root "$FORMAL_ROOT" \
  --sample-size 50 \
  --max-epochs 10000 \
  --metric-stride 1 \
  --early-stop-patience 20 \
  --early-stop-min-delta 0.0001

python "$PIPELINE/scripts/review_weak_label_results.py" \
  --topologies-csv "$PIPELINE/configs/topologies.locked.csv" \
  --output-root "$FORMAL_ROOT" \
  --sample-size 50 \
  --max-epochs 10000 \
  --metric-stride 1 \
  --early-stop-patience 20 \
  --early-stop-min-delta 0.0001 \
  --require-early-stop
```

No formal artifacts, plans, training jobs, or labels have been created yet.
