# Step5 Experiment 02: Sample-size sensitivity at data seed 42

## Status

Protocol locked on 2026-07-19; execution is pending. This diagnostic reuses the
completed sample-size-50 smoke results and adds sample sizes 100 and 250 for six
fixed topologies. It is intentionally separate from the formal 1000-topology
sweep.

## Question and selected topologies

| Role | Topology |
| --- | --- |
| ceiling zero | G-0 |
| ordinary zero | G-3 |
| high-gap zero | G-14 |
| cap-hit zero | G-15 |
| positive control | G-9 |
| strong positive control | G-13 |

The sample-size comparison uses 50 = 40 train / 10 validation (existing),
100 = 80 / 20, and 250 = 200 / 50. All runs keep `data_seed=42`,
`theta_seed=42`, `gurobi_seed=42`, `master_label_seed=20260719`, test size 1000,
1500 epochs, metric stride 1, and early stop patience 20 / minimum delta 0.0001.

The generator deliberately keeps
`experiment_version=step5_exp1_weak_label_seed42_sample50_v1`. That string is
part of sample-seed derivation, so changing it would silently change both the
training prefix and test bank. The audit requires the topology-specific
`test_hash` to match across all three sample sizes.

The primary continuous target is:

```text
normalized_improvement_pp = 100 * (
    test_mean_normalized_gap_2stage - test_mean_normalized_gap_spoplus
)
```

Positive values favor SPO+, negative values favor 2stage, and zero means equal
mean normalized decision gap on the fixed test bank. The result audit also
counts exact per-test-instance achieved-objective matches.

G-15 receives a separate sample-size-50 3000-epoch cap check using the existing
artifacts. It is not mixed into the sample-size result roots.

## Planned result roots on Garnet

```text
results/sample_size100
results/sample_size250
results/g15_cap3000
```

Compact final outputs:

```text
results/sample_size_sensitivity.csv
results/sample_size_sensitivity_audit.json
results/g15_epoch_cap_check.csv
```
