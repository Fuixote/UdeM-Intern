# Step5 Experiment 02: Sample-size sensitivity at data seed 42

## Status

Complete on 2026-07-19. This diagnostic reused the completed sample-size-50
smoke results, trained sample sizes 100 and 250 for six fixed topologies, and
ran separate 3000-epoch G-15 checks at all three sample sizes. It is
intentionally separate from the formal 1000-topology sweep.

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

G-15 receives separate 3000-epoch cap checks at sample sizes 50, 100, and 250
using the corresponding existing artifacts. The larger-sample checks were added
because their 1500-epoch runs also reached the epoch boundary without triggering
early stopping. They are not mixed into the sample-size result roots.

## Result

| Topology | Role | 50 | 100 | 250 | Exact achieved-objective matches |
| --- | --- | ---: | ---: | ---: | --- |
| G-0 | ceiling zero | 0.0000 pp | 0.0000 pp | 0.0000 pp | 1000/1000 at every size |
| G-3 | ordinary zero | 0.0000 pp | 0.0000 pp | 0.0000 pp | 1000/1000 at every size |
| G-14 | high-gap zero | 0.0000 pp | 0.0000 pp | 0.0000 pp | 1000/1000 at every size |
| G-15 | cap-hit zero | 0.0000 pp | 0.0000 pp | 0.0000 pp | 1000/1000 at every size |
| G-9 | positive control | +7.6090 pp | +8.0720 pp | +8.0097 pp | 138/1000, 71/1000, 81/1000 |
| G-13 | strong positive control | +42.2631 pp | +42.2631 pp | +42.2631 pp | 0/1000 at every size |

The primary result is unambiguous under this fixed seed and protocol:

- G-3 and G-14 do not become positive with more data. Both methods retain
  exactly equal achieved objectives on every one of the 1000 fixed test
  instances at all three sample sizes. This supports a decision-plateau
  explanation rather than insufficient training samples for these two cases.
- G-0 remains a genuine ceiling case: both normalized gaps are exactly zero at
  every sample size.
- G-9 does not lose its advantage as sample size grows; its improvement is
  stable at roughly 7.6--8.1 pp. G-13 is exactly stable at +42.2631 pp. These
  controls do not support the hypothesis that SPO+'s benefit here is only a
  low-data effect.
- G-15 remains exactly zero after lifting the epoch cap. At sample sizes
  50/100/250, the 3000-epoch runs select SPO+ epochs 1711/1509/2034, compared
  with 1500/1495/1500 under the original cap, yet all six evaluations remain
  0 pp with 1000/1000 exact achieved-objective matches. The 1500 cap is not the
  cause of this topology's zero result.

This is a targeted six-topology, one-data-seed diagnostic. It supports the
mechanism interpretation above, but it is not a population estimate over the
1000-topology bank.

## Integrity and execution audit

- sample100 and sample250 artifact audits passed 6/6 with zero failures;
- all 12 sample-size jobs and all three 3000-epoch G-15 jobs completed with
  paired 2stage, SPO+, and evaluation status `success`;
- each topology's test hash is identical across sample sizes 50, 100, and 250;
- the generated fit samples are exact nested prefixes from 50 to 100 to 250;
- the final audit passed with 18/18 sensitivity rows and 6/6 cap-comparison
  rows;
- Garnet launcher elapsed times were approximately 406 seconds for sample100,
  966 seconds for sample250, and 112/169/539 seconds for the G-15 cap checks at
  sample sizes 50/100/250;
- the completion watcher sent the Brevo notification successfully (`HTTP 201`).

## Result roots on Garnet

```text
results/sample_size100
results/sample_size250
results/g15_cap3000
results/g15_cap3000_sample100
results/g15_cap3000_sample250
```

Compact final outputs:

```text
results/sample_size_sensitivity.csv
results/sample_size_sensitivity_audit.json
results/g15_epoch_cap_check.csv
```
