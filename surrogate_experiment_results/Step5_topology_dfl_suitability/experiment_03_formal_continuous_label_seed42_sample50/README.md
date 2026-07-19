# Step5 Experiment 03: Formal continuous labels

## Status

Protocol locked on 2026-07-19. All 1000 formal artifact bundles have been built
and audited, the 1000-row dry-run job plan has been generated and independently
checked, and the launcher preview has passed. Formal training has not started.
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

Formal artifacts are a versioned dataset bundle, not model output. For each
topology they contain 40 training samples, 10 validation samples, the fixed
1000-sample test bank, and the manifests/hashes needed to prove the split and
generation provenance. The formal root rebuilds all 1000 bundles rather than
copying smoke files, avoiding stale paths while preserving the same deterministic
data namespace and hashes.

Artifact construction uses 16 deterministic, estimated-load-balanced shards.
The full artifact audit must pass 1000/1000 before any training plan is created.

The critical planning and review arguments are:

```bash
PIPELINE=surrogate_experiment_results/Step5_topology_dfl_suitability/experiment_01_weak_label_seed42_sample50
FORMAL=surrogate_experiment_results/Step5_topology_dfl_suitability/experiment_03_formal_continuous_label_seed42_sample50
FORMAL_ROOT="$FORMAL/results/formal1000"

python "$PIPELINE/scripts/plan_weak_label_jobs.py" \
  --topologies-csv "$PIPELINE/configs/topologies.locked.csv" \
  --output-root "$FORMAL_ROOT" \
  --data-seed 42 \
  --sample-size 50 \
  --test-size 1000 \
  --theta-seed 42 \
  --gurobi-seed 42 \
  --max-epochs 10000 \
  --metric-stride 1 \
  --early-stop-patience 20 \
  --early-stop-min-delta 0.0001 \
  --plan-output "$FORMAL_ROOT/plans/formal_plan.json" \
  --jobs-csv-output "$FORMAL_ROOT/plans/formal_jobs.csv"

python "$PIPELINE/scripts/launch_weak_label_jobs.py" \
  --jobs-csv "$FORMAL_ROOT/plans/formal_jobs.csv" \
  --output-root "$FORMAL_ROOT" \
  --expected-job-count 1000 \
  --normal-workers 16 \
  --long-workers 0 \
  --require-hostname garnet

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

## Preparation checkpoint: 2026-07-19

The formal artifact audit passed all 1000 topologies with zero failed
topologies and an empty failure list. Its SHA256 is:

```text
f273e4fa4ff301c72802f8c2b98d0ba6f944537aeff5ec1d8258acd626b4172d  formal_artifact_audit.json
```

The formal plan contains exactly 1000 unique jobs and topologies in the same
order as `configs/topologies.locked.csv`. All rows are `ready`, have empty
readiness failures, contain the locked 50/40/10 and test-1000 split, use seed
42, set `max_epochs=10000`, carry patience 20 and min delta 0.0001, and contain
`--dry-run` without `--execute`. An independent 30-check audit passed.

```text
27d670d44ca480a9d05aa94f91c39f956793017427583a012c1ada2686edf3df  formal_plan.json
7b25a45eb1cb020b2e9d3fa7a0af6088e352bd8f0073a847139ceb3e77919921  formal_jobs.csv
```

The launcher preview reported `execute=false`, 1000 normal jobs, zero long
jobs, 16 normal workers, and zero long workers on Garnet. A post-preview check
found zero formal `job_status.json` files and no formal training process. The
next step is an explicit launcher execution; it has not been performed yet.
