# Surrogate Experiment Results

This directory contains the Step1 diagnostic/generalization experiments and
report drafts. The current Step1 sequence is:

- `Step1a/`: in-sample two-dimensional landscape diagnostic.
- `Step1b/`: sample-size generalization with MSE two-stage and perturbed FY.
- `Step1c/`: matched Step1b protocol with the end-to-end surrogate swapped to
  reward-max SPO+.

## Combined Step1b/Step1c Plotting

Use `plot_step1bc_final_comparison.py` to combine the matched Step1b and Step1c
archives into final comparison figures.

Default inputs:

```text
Step1b: surrogate_experiment_results/Step1b/remote_results/formal_M16_2stage500_e2e500_s10_val2000
Step1c: surrogate_experiment_results/Step1c/remote_results/formal_spoplus_ablation_val2000
```

Default output:

```text
surrogate_experiment_results/plot_results_step1bc/
```

The script compares five selected checkpoints:

```text
2stage selected by validation MSE
FY selected by validation decision gap
FY selected by validation FY loss
SPO+ selected by validation decision gap
SPO+ selected by validation SPO+ loss
```

It currently expects held-out split test files named `test_summary.csv` and
`test_per_graph.csv`, and large unseen-test files named
`unseen10000_summary.csv` and `unseen10000_per_graph.csv`. Missing train sizes
or missing unseen-test files are skipped rather than treated as fatal, so the
script can be rerun as remote results arrive.

Run locally with the KEPs environment:

```bash
MPLCONFIGDIR=/tmp/matplotlib \
/home/weikang/miniconda3/envs/KEPs/bin/python \
  surrogate_experiment_results/plot_step1bc_final_comparison.py
```

Main figures:

```text
figure0_mean_normalized_gap_no_error_bars.{png,pdf}
figure1_mean_normalized_gap_heldout400_unseen10000.{png,pdf}
figure2_per_graph_gap_boxplots_heldout400_unseen10000.{png,pdf}
```

`figure0` is the mean-only line plot without uncertainty bars. `figure1` is the
grouped-bar version of the same mean normalized decision gap comparison, with
per-graph bootstrap 95% confidence intervals.

Diagnostics:

```text
figure3_checkpoint_epochs.{png,pdf}
figure4_selected_theta_endpoints.{png,pdf}
figure5_loss_curves.{png,pdf}
combined_step1bc_summary.csv
combined_step1bc_per_graph.csv
```

As of the latest local verification, held-out400 rows are available from Step1c
for train sizes 50/200/600/1200 and from Step1b for train sizes 50/200/600.
The 10,000-graph unseen noisy-linear dataset has been generated locally at:

```text
dataset/processed/step1_noisy_linear_sigma010_unseen_test10000_seed20260520
```

The unseen10000 panel currently includes Step1c train sizes 50/200/600/1200 and
Step1b train sizes 50/200/600. Step1b train size 1200 is still absent locally
until its remote run finishes and is synced back. The latest combined plotting
run wrote `36` summary rows and `187200` per-graph rows under
`surrogate_experiment_results/plot_results_step1bc/`.

## Combined Unseen Evaluation

Use `evaluate_step1bc_unseen_once.py` for large unseen-test evaluation across
Step1b/Step1c archives. This replaces the older pattern of launching one
`evaluate_unseen_run.py` process per run directory.

The reason is operational: per-run parallel evaluation loads the same 10,000
graphs repeatedly, builds multiple Gurobi environments, and recomputes the
synthetic-label oracle solution for the same graph in each process. On a local
machine this can saturate memory and I/O. The combined evaluator instead:

```text
load unseen graphs once
compute each graph's synthetic-label oracle once
load all complete Step1b/Step1c checkpoint sets
evaluate each theta sequentially against the cached graph records
write unseen10000_summary/per_graph files back under each run directory
```

Dry-run the discovery step without Gurobi:

```bash
/home/weikang/miniconda3/envs/KEPs/bin/python \
  surrogate_experiment_results/evaluate_step1bc_unseen_once.py \
  --dry_run \
  --train_sizes 50 200 600 1200
```

Run the current intended unseen10000 evaluation:

```bash
MPLCONFIGDIR=/tmp/matplotlib \
/home/weikang/miniconda3/envs/KEPs/bin/python \
  surrogate_experiment_results/evaluate_step1bc_unseen_once.py \
  --dataset_dir dataset/processed/step1_noisy_linear_sigma010_unseen_test10000_seed20260520 \
  --output_stem unseen10000 \
  --train_sizes 50 200 600 1200
```

As of the current local archive state, this evaluates seven complete run
directories: Step1b train sizes 50/200/600 and Step1c train sizes
50/200/600/1200. Step1b train size 1200 is skipped until its model weights are
synced back locally.

## Local Unseen-Test Timing

On 2026-05-19, a local timing run evaluated the first 1,000 graphs from the
unseen10000 noisy-linear dataset using the Step1c `train_size=50` archive and
three checkpoints:

```text
2stage selected by validation MSE
SPO+ selected by validation decision gap
SPO+ selected by validation SPO+ loss
```

Command shape:

```bash
MPLCONFIGDIR=/tmp/matplotlib \
/usr/bin/time -p /home/weikang/miniconda3/envs/KEPs/bin/python \
  surrogate_experiment_results/Step1c/evaluate_unseen_run.py \
  --run_dir surrogate_experiment_results/Step1c/remote_results/formal_spoplus_ablation_val2000/train_size=50 \
  --dataset_dir dataset/processed/step1_noisy_linear_sigma010_unseen_test10000_seed20260520 \
  --graph_limit 1000 \
  --output_stem timing_unseen10000_limit1000 \
  --bootstrap_samples 200
```

Observed wall time was `32.21s` for 1,000 graphs and three model checkpoints.
Outputs were written under:

```text
surrogate_experiment_results/Step1c/remote_results/formal_spoplus_ablation_val2000/train_size=50/metrics/timing_unseen10000_limit1000_summary.csv
surrogate_experiment_results/Step1c/remote_results/formal_spoplus_ablation_val2000/train_size=50/metrics/timing_unseen10000_limit1000_per_graph.csv
```
