# Step5 Experiment 03: Formal continuous labels

## Status

Complete. All 1000 formal artifact bundles were built and audited, the 1000-row
job plan passed its independent checks, all paired training/evaluation jobs
succeeded, and the final continuous-label review passed with 1000 labels.
Formal training started on Garnet at 2026-07-19 10:22:34 -0400 with 20
concurrent normal workers and no long-worker queue.
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
  --normal-workers 20 \
  --long-workers 0 \
  --require-hostname garnet \
  --execute

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
later execution uses 20 workers following an explicit user override.

## Execution checkpoint: 2026-07-19

The formal launcher was started with `--execute`, `--normal-workers 20`, and
`--long-workers 0` in tmux session `step5_formal1000_train`. BLAS/OpenMP thread
environment variables were set to 1 per job to limit nested thread expansion.
The Brevo completion watcher is running in `step5_formal1000_notify`.

At the first 60-second launcher checkpoint there were 20 active jobs, 12
finished jobs, zero failures, and 968 pending jobs. A subsequent artifact-level
check found 19 completed `job_status.json` records; all 19 reported success for
the paired job, 2stage, SPO+, and evaluation stages. Both early-stopping files
were present for every completed job and all had `should_stop=true`.

## Final review checkpoint: 2026-07-20

The launcher finished all 1000 jobs in 5792.5 seconds with zero failed or
skipped jobs. The strict reviewer was run with `max_epochs=10000` and
`--require-early-stop`; it found 1000 successful paired jobs, 1000 unique label
rows, no unexpected status files, and an empty failure list. For every topology,
2stage, SPO+, and evaluation succeeded and both method records had
`should_stop=true`.

The primary GNN dataset is `weak_label_topology_summary.csv`. Its locked compact
outputs are:

```text
0e7b8554c63da668a2ad1b2ddd84386ec91970dd2a27d7986e417806af68deba  weak_label_topology_summary.csv
ad2b198910eb09b2c92c724370ca4792d416baf471d3ec975b730b75acbeb12d  weak_label_job_metrics.csv
a7bc853d44ebb7a0a4f72d2f2414ef6a68ee88821fc0cf19df76b67e02d942a5  weak_label_integrity_audit.json
```

The continuous target distribution, in normalized improvement percentage
points, is strongly zero-inflated:

```text
negative / exact zero / positive  85 / 641 / 274
mean / median                     1.3020 / 0.0000 pp
minimum / maximum                -9.7041 / 50.1005 pp
abs(label) <= 0.01 pp             679
abs(label) <= 0.10 pp             759
```

The legacy threshold classes are secondary metadata only: 20 harmful, 796
near-neutral, and 184 helpful. They must not replace the continuous regression
target. Future topology-only GNN work should report an all-zero baseline and
metrics on both the full set and the nonzero subset because 64.1% of labels are
exactly zero.

The 10000-epoch cap was not hit. The maximum stopped epoch was 2425 for 2stage
and 8649 for SPO+; their median stopped epochs were 740 and 359, respectively.

## Dataset readiness assessment: 2026-07-20

The dataset is internally valid and reproducible for the locked
`pairs20/ndd2`, sample-size-50 regime. All 1000 topology and source files are
present; topology and feasible-set hashes are unique; the label formula was
independently recomputed for every row; and no non-finite target was found.
The 1000 paired test samples and strict early-stop requirement make this a
strong controlled v1 dataset.

The exact zeros are real decision plateaus rather than rounding artifacts. Of
641 zero labels, 186 have zero normalized gap for both methods and 455 have the
same nonzero normalized gap. All 641 also have zero paired mean improvement and
zero fraction of samples improved over 2stage. The target is nevertheless
heavy-tailed: 241 rows have absolute improvement above 0.1 pp, 143 exceed 1 pp,
and 18 exceed 20 pp.

There is topology-complexity signal, especially for whether a label is nonzero,
but the available scalar summaries do not predict its magnitude or direction
well. Across the 21 nonconstant, topology-only scalar fields, the strongest
Spearman correlation with absolute target is 0.310. A deterministic exploratory
five-fold evaluation gave:

```text
model                     overall MAE   overall RMSE   overall R2
all-zero baseline              1.362          5.199       -0.067
fold-train mean                2.244          5.036       -0.001
ridge topology summaries       2.217          5.029        0.002
extra-trees summaries          2.171          4.991        0.017
```

On the 241 rows with `abs(label) > 0.1 pp`, the extra-trees R2 is -0.153.
Topology summaries identify exact nonzero labels only moderately (ROC AUC
0.669) and material labels weakly (ROC AUC 0.617); among the 241 material rows,
positive-versus-negative direction is effectively random (ROC AUC 0.509). The
material subset contains only 31 negative versus 210 positive rows.

This weak scalar baseline does not rule out a structural GNN. There are seven
graphs in three groups with identical values for all 21 scalar topology fields
but different labels, with a maximum within-group range of 18.744 pp. Their
connectivity or candidate-incidence structure therefore contains information
that aggregate counts discard. A candidate-vertex incidence GNN or
candidate-conflict GNN is better aligned with this signal than a vanilla GNN
using only the 22-vertex compatibility graph.

The main remaining validity limitation is replication. Every label uses the
same data, theta, and Gurobi seed 42, so this dataset cannot estimate
between-seed label variance or sign stability. It also contains only 20-pair,
2-NDD graphs and only sample size 50. The target must therefore be interpreted
as `DFL improvement under the locked seed-42, sample-50 protocol`, not yet as a
universal intrinsic topology property.

Before treating the labels as a stable GNN target, run a stratified repeat-seed
audit with the existing test bank fixed and independently regenerated
training/validation banks. Include exact-zero, small nonzero, material positive,
material negative, and extreme topologies. Report label standard deviation,
rank correlation, and sign/plateau consistency. For the first GNN baseline, use
stratified five-fold topology splits, Huber or another outlier-robust regression
loss, and report the all-zero baseline, full-set metrics, nonzero/material-subset
metrics, rank correlation, and downstream method-selection regret.
