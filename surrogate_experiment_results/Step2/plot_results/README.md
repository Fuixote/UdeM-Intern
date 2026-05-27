# Step2 Plotting Guide

This directory is the long-term plotting workspace for Step2.  The goal is not
to archive every raw experiment artifact here.  The goal is to keep a compact,
AI-readable plotting plan that turns Step2 data into figures for reports,
slides, and future diagnosis.

Step2 changes the synthetic label generation regime while keeping the downstream
KEP problem, two-feature linear probe, train sizes, seeds, and evaluation
protocol aligned with Step1b/Step1c.  The main scientific question is:

> As label misspecification and noise increase, do decision-focused surrogates
> such as FY and SPO+ produce better KEP decisions than the 2stage MSE baseline?

The expert guidance below should be treated as the default plotting roadmap.
It is intentionally broader than the final report needs: generate the core plots
first, then add diagnostics only when they clarify the result.

## Current Analysis Entry Points

Use root-level Step2 summaries and dataset diagnostics as the primary sources.
Do not depend on scattered `remote_results` files for the main narrative.

Primary performance summaries:

```text
surrogate_experiment_results/Step2/step2_heldout400_primary_summary.csv
surrogate_experiment_results/Step2/step2_unseen10000_all_checkpoints_summary.csv
surrogate_experiment_results/Step2/step2_unseen10000_all_checkpoints_summary.json
```

Dataset QA:

```text
dataset/processed/<step2 dataset>/label_diagnostics.json
dataset/processed/<step2 dataset>/label_graph_diagnostics.csv
```

Optional local diagnostics, if present:

```text
surrogate_experiment_results/Step2/**/remote_results/**/metrics/unseen10000_per_graph.csv
surrogate_experiment_results/Step2/**/remote_results/**/metrics/*loss_curve.csv
```

Those optional files are useful for tail and trajectory diagnosis, but they are
large and are not the main GitHub-facing evidence.

## Artifact Policy

Track on GitHub:

```text
README files
root-level summary CSV/JSON files
generated figures in plot_results/
small derived diagnostic summaries
```

Do not track:

```text
per-graph CSV/JSON files
model weights
per-run remote_results metrics/config dumps
large local-only archives
```

If a per-graph analysis is needed, derive a smaller summary table or figure from
the local per-graph CSV, then commit that summary/figure rather than the raw
per-graph file.

## Completed Experiment Scope

The current Step2 grid is:

```text
9 label regimes:
  Step2a additive linear Gaussian:
    step2a_additive_rho050

  Step2b noiseless polynomial:
    step2b_poly_d1
    step2b_poly_d2
    step2b_poly_d4
    step2b_poly_d8

  Step2c polynomial + multiplicative noise:
    step2c_poly_d1_mult_eps050
    step2c_poly_d2_mult_eps050
    step2c_poly_d4_mult_eps050
    step2c_poly_d8_mult_eps050

4 train sizes:
  50, 200, 600, 1200

2 training pipelines:
  Step1b-style FY
  Step1c-style SPO+

5 evaluation checkpoint rows:
  2stage selected by validation MSE
  FY selected by validation decision gap
  FY selected by validation FY loss
  SPO+ selected by validation decision gap
  SPO+ selected by validation SPO+ loss
```

The full unseen evaluation therefore has:

```text
9 regimes x 4 train sizes x 5 checkpoints = 180 rows
```

The root summary CSV should be the default input for performance plots.

## Known Result Direction

These are current interpretation anchors.  Re-check them whenever the summary
CSV is regenerated.

1. **Step2 label generation is controlled.**
   Graph structures are fixed, label scale is approximately controlled by
   graph-level rescaling, and degree/noise increase label variance while lowering
   clean-linear correlation.

2. **Step2a is a bridge benchmark.**
   The conditional mean is still close to linear, so 2stage MSE is expected to be
   strong.  FY/SPO+ should not be expected to dominate here.

3. **Step2b d1 is an easy sanity check.**
   Degree 1 is nearly clean-linear.  All methods should be close to oracle
   decision quality.

4. **Step2b higher degrees test deterministic misspecification.**
   As degree increases, the linear probe becomes more misspecified.  FY/SPO+
   should increasingly recover decision quality that 2stage loses, especially in
   absolute gap reduction.

5. **Step2c combines nonlinear misspecification and multiplicative noise.**
   This is one of the most informative regimes.  It is harder than Step2b, and
   it is where the difference between 2stage, FY, and SPO+ should be easiest to
   explain.

6. **FY vs SPO+ is close.**
   Current evidence suggests both decision-focused methods beat 2stage in high
   misspecification regimes, while neither is decisively dominant everywhere.
   Use selector, epoch, theta, and per-graph diagnostics to explain differences.

## Main Report Figures

These should be generated first.  They form the shortest coherent report.

### Figure 01: Label QA by Degree

Suggested output:

```text
01_label_std_and_corr_by_degree.png
```

Data source:

```text
dataset/processed/<step2 dataset>/label_diagnostics.json
```

Plot:

```text
x-axis: degree {1, 2, 4, 8}
y-axis left: label std
y-axis right: corr(clean-linear label, synthetic label)
panels: Step2b, Step2c
Step2a: horizontal/reference marker or separate annotation
```

Purpose:

> Show that polynomial degree increases label variance and lowers clean-linear
> correlation, and that Step2c is noisier because of multiplicative noise.

This figure validates the Step2 experimental design before showing model
performance.

### Figure 02: Main vs Unseen Label Alignment

Suggested output:

```text
02_main_vs_unseen_label_alignment.png
```

Data source:

```text
label_diagnostics.json for main2000 and unseen10000 datasets
```

Plot:

```text
scatter: x = main2000 label statistic
         y = unseen10000 label statistic
statistics: mean, std, clean-label correlation
one point per regime
reference: y = x diagonal
```

Purpose:

> Show that train-pool and unseen-test label distributions are aligned.  This
> makes unseen10000 a genuine larger test set rather than a different regime.

### Figure 03: Primary Normalized Gap vs Degree

Suggested output:

```text
03_unseen_normalized_gap_vs_degree.png
```

Data source:

```text
surrogate_experiment_results/Step2/step2_unseen10000_all_checkpoints_summary.csv
```

Plot:

```text
y-axis: test_mean_normalized_gap
x-axis: degree
panels: Step2b noiseless polynomial, Step2c multiplicative polynomial
primary lines:
  2stage_val_mse
  fy_val_fy_loss
  spoplus_val_spoplus_loss
optional dashed lines:
  fy_val_decision_gap
  spoplus_val_decision_gap
```

Purpose:

> This is the main performance plot.  It should show that d1 is easy, while
> higher-degree regimes expose larger gaps between 2stage and decision-focused
> training.

Use normalized gap for the main figure.  Raw gap can be reported in a secondary
table or appendix, but normalized gap is more comparable across label regimes.

### Figure 04: Paired Improvement Heatmap

Suggested output:

```text
04_paired_improvement_heatmap.png
```

Data source:

```text
step2_unseen10000_all_checkpoints_summary.csv
```

Plot:

```text
y-axis: train size {50, 200, 600, 1200}
x-axis: regime
color: paired_mean_improvement_over_2stage
panels:
  FY selected by validation FY loss
  SPO+ selected by validation SPO+ loss
optional panels:
  FY selected by validation decision gap
  SPO+ selected by validation decision gap
```

Purpose:

> Identify where decision-focused training actually improves paired test
> performance over 2stage.

This plot is usually more interpretable than many small line charts because it
shows regime, train size, and method at the same time.

### Figure 05: Selector Comparison

Suggested output:

```text
05_selector_delta_fy_spoplus.png
```

Data source:

```text
step2_unseen10000_all_checkpoints_summary.csv
```

Compute:

```text
delta_FY =
  gap(FY selected by validation decision gap)
  - gap(FY selected by validation FY loss)

delta_SPO+ =
  gap(SPO+ selected by validation decision gap)
  - gap(SPO+ selected by validation SPO+ loss)
```

Positive means the surrogate-loss selector is better than direct validation-gap
selection on the unseen test.

Purpose:

> Test whether validation decision gap is a noisy selector and whether FY/SPO+
> validation loss gives more stable model selection.

### Figure 06: Selected Epoch Heatmap

Suggested output:

```text
06_selected_epoch_heatmap.png
```

Data source:

```text
selected_epoch from step2_unseen10000_all_checkpoints_summary.csv
```

Plot:

```text
y-axis: train size
x-axis: regime
color: selected epoch
panels:
  FY selected by validation FY loss
  FY selected by validation decision gap
  SPO+ selected by validation SPO+ loss
  SPO+ selected by validation decision gap
```

Purpose:

> Show whether selectors choose early or late checkpoints.  This is important
> because FY and SPO+ often have different training dynamics and scaling
> behavior.

### Figure 07: Theta Endpoints

Suggested output:

```text
07_theta_endpoints.png
```

Data source:

```text
theta_1, theta_2 from step2_unseen10000_all_checkpoints_summary.csv
```

Plot:

```text
x-axis: theta_1
y-axis: theta_2
marker/color: method or checkpoint family
facets: selected regimes or degrees
reference point: clean-signal coefficient [10, 5]
```

Purpose:

> Show that FY and SPO+ are not simply recovering the clean reward coefficient.
> They learn different decision-oriented linear proxies.

Expected interpretation:

```text
2stage: tends to fit reward scale/calibration
FY: may use smaller-scale decision proxy
SPO+: may show margin-style scaling
```

### Figure 08: Theta Norm vs Gap

Suggested output:

```text
08_theta_norm_vs_gap.png
```

Data source:

```text
theta_1, theta_2, test_mean_normalized_gap
```

Plot:

```text
x-axis: sqrt(theta_1^2 + theta_2^2)
y-axis: test_mean_normalized_gap
color: checkpoint family
facet: block or degree
```

Purpose:

> Connect parameter scale to decision quality, especially for SPO+ where margin
> behavior can change theta scale without necessarily improving reward
> calibration.

## Optional Per-Graph Diagnostics

Use these only if local `unseen10000_per_graph.csv` files are present.  These
plots should generally be appendix or diagnostic figures, not the first result
shown to a reader.

### Figure 09: Paired Delta Histograms or ECDF

Suggested output:

```text
09_paired_delta_histograms.png
```

Compute per graph:

```text
delta_graph = gap_2stage - gap_candidate
```

Recommended regimes:

```text
Step2a additive rho050
Step2b d8
Step2c d8
```

Purpose:

> Determine whether the mean improvement comes from broad small gains or a small
> tail of hard graphs.

### Figure 10: Tail Gap ECDF

Suggested output:

```text
10_tail_gap_ecdf.png
```

Plot:

```text
x-axis: per-graph normalized gap
y-axis: empirical CDF or survival probability
methods: 2stage, FY, SPO+
regimes: Step2a, Step2b d8, Step2c d8
```

Purpose:

> Show whether FY/SPO+ reduce tail risk, not just mean gap.

### Figure 11: Hard Graph Attribution

Suggested output:

```text
11_hard_graph_attribution.png
```

Potential x-axes:

```text
graph label std
oracle objective
edge count
label-to-clean mean ratio
zero-label fraction
```

y-axis:

```text
candidate improvement over 2stage
```

Purpose:

> Identify what kind of graph benefits from decision-focused training.

## Optional Epoch-Level Diagnostics

Use these only if local `*loss_curve.csv` files are present.  They are useful for
explaining why a checkpoint was selected, but they are not needed for the first
main performance result.

### Figure 12-14: Training Trajectory Dashboards

Suggested outputs:

```text
12_training_trajectory_dashboard_step2a.png
13_training_trajectory_dashboard_step2b_d8.png
14_training_trajectory_dashboard_step2c_d8.png
```

Recommended panels:

```text
epoch vs validation surrogate loss
epoch vs validation decision gap
epoch vs theta_1
epoch vs theta_2
epoch vs theta norm
selected checkpoint markers
```

If epoch-level unseen diagnostics are available, add:

```text
epoch vs unseen10000 decision gap
```

Purpose:

> Explain training dynamics and checkpoint selection, especially when validation
> decision gap and surrogate-loss selection disagree.

### Figure 15: Selection Regret

Suggested output:

```text
15_selection_regret.png
```

If per-epoch validation/unseen gaps are available:

```text
selection_regret_D(selector)
  = gap_D(epoch selected by selector) - min_epoch gap_D(epoch)
```

Recommended selectors:

```text
validation decision gap
validation FY loss
validation SPO+ loss
```

Recommended datasets:

```text
validation
heldout400
unseen10000
```

Purpose:

> Quantify how far each checkpoint rule is from the best point on the trajectory.

## Report-Level Figure Order

For a concise report or advisor update, use this order:

1. **Label design / QA**
   Degree vs label std and clean-label correlation.

2. **Main/unseen alignment**
   Train-pool and unseen-test label distributions match.

3. **Primary performance**
   Normalized unseen10000 gap vs degree.

4. **Paired improvement**
   Heatmap of improvement over 2stage.

5. **Checkpoint selector comparison**
   Surrogate-selected vs validation-gap-selected checkpoint.

6. **Theta endpoint behavior**
   Show that FY and SPO+ learn different decision-oriented proxies.

7. **Per-graph tail analysis**
   Optional; include if the mean effect needs explanation.

8. **Trajectory dashboard**
   Optional; include if checkpoint selection or training dynamics are questioned.

## Recommended Plotting Script

Create or maintain:

```text
surrogate_experiment_results/Step2/plot_step2_summary.py
```

Recommended CLI:

```bash
MPLCONFIGDIR=/tmp/matplotlib \
python surrogate_experiment_results/Step2/plot_step2_summary.py \
  --summary_csv surrogate_experiment_results/Step2/step2_unseen10000_all_checkpoints_summary.csv \
  --heldout_csv surrogate_experiment_results/Step2/step2_heldout400_primary_summary.csv \
  --label_diagnostics_root dataset/processed \
  --out_dir surrogate_experiment_results/Step2/plot_results
```

The script should be robust to missing optional local files.  It should always
generate figures 01-08 from root summary and label diagnostics, and only produce
figures 09-15 when the required local per-graph or epoch files exist.

## Plotting Conventions

Use consistent method names:

```text
2stage (val MSE)
FY (val gap)
FY (val FY)
SPO+ (val gap)
SPO+ (val SPO+)
```

Use consistent metrics:

```text
Primary y-axis: mean normalized decision gap
Paired comparison: paired mean improvement over 2stage
Selection comparison: candidate gap difference on unseen10000
```

Use clear direction labels:

```text
For gaps: lower is better.
For paired improvement: higher is better.
For selector delta: positive means surrogate-loss selection is better.
```

Avoid overclaiming:

```text
Do not claim relative improvement is monotone in degree.
Do claim that absolute 2stage gap grows under high-degree misspecification, and
that FY/SPO+ recover meaningful decision quality in those regimes.
```

## Current Narrative to Test With Plots

The plots should test this narrative:

> Step2 constructs a controlled KEP synthetic-label benchmark ranging from
> nearly linear to high-degree nonlinear and noisy labels.  In nearly linear
> regimes, 2stage MSE is already strong.  As misspecification increases, the
> 2stage decision gap grows and decision-focused surrogates recover a meaningful
> portion of that loss.  FY and SPO+ are both strong in high-degree regimes, but
> they differ in checkpoint selection and parameter scaling behavior.

This narrative is plausible from the current summaries, but the plotting script
should verify it visually before it becomes a report claim.

