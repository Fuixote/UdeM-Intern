# SPO Paper Shortest-Path Run Log

## 2026-05-29 15:39 America/Toronto

Git commit:

```text
17a566b7907fa2b76d2d850dbfb74e1bfbc3d4de
```

### Commands Run

Local:

```bash
python -m unittest tests.test_spoplus_shortest_path_validation -v
python -m unittest tests.test_paper_shortest_path_experiment -v
python surrogate_experiment_results/SPO_validation/paper_shortest_path/run_paper_shortest_path.py --preset middle-row --dry-run
python surrogate_experiment_results/SPO_validation/paper_shortest_path/run_paper_shortest_path.py --preset smoke --output-dir surrogate_experiment_results/SPO_validation/paper_shortest_path/results/smoke_20260529_152226
```

Orange:

```bash
/part/01/Tmp/fuweikan/spo_validation_venv/bin/python -m unittest tests.test_paper_shortest_path_experiment -v
/part/01/Tmp/fuweikan/spo_validation_venv/bin/python -m unittest tests.test_spoplus_shortest_path_validation -v
/part/01/Tmp/fuweikan/spo_validation_venv/bin/python surrogate_experiment_results/SPO_validation/paper_shortest_path/compare_pyepo_forward_loss.py --allow-missing-pyepo
/part/01/Tmp/fuweikan/spo_validation_venv/bin/python surrogate_experiment_results/SPO_validation/paper_shortest_path/compare_pyepo_forward_loss.py
/part/01/Tmp/fuweikan/spo_validation_venv/bin/python surrogate_experiment_results/SPO_validation/paper_shortest_path/run_paper_shortest_path.py --preset pilot --output-dir surrogate_experiment_results/SPO_validation/paper_shortest_path/results/pilot_20260529_152538
/part/01/Tmp/fuweikan/spo_validation_venv/bin/python surrogate_experiment_results/SPO_validation/paper_shortest_path/run_paper_shortest_path.py --preset pyepo-pilot --fail-if-pyepo-missing --output-dir surrogate_experiment_results/SPO_validation/paper_shortest_path/results/pyepo_pilot_20260529_153225
```

Local plotting:

```bash
MPLCONFIGDIR=/tmp/matplotlib_codex conda run -n KEPs python surrogate_experiment_results/SPO_validation/paper_shortest_path/plot_paper_shortest_path.py surrogate_experiment_results/SPO_validation/paper_shortest_path/results/pyepo_pilot_20260529_153225/summary.csv
```

### Output Directories

```text
surrogate_experiment_results/SPO_validation/paper_shortest_path/results/smoke_20260529_152226
surrogate_experiment_results/SPO_validation/paper_shortest_path/results/pilot_20260529_152538
surrogate_experiment_results/SPO_validation/paper_shortest_path/results/pyepo_pilot_20260529_153225
```

Plots:

```text
surrogate_experiment_results/SPO_validation/paper_shortest_path/results/pyepo_pilot_20260529_153225/plots/paper_shortest_path_middle_row.png
surrogate_experiment_results/SPO_validation/paper_shortest_path/results/pyepo_pilot_20260529_153225/plots/pyepo_vs_ours_spoplus_scatter.png
surrogate_experiment_results/SPO_validation/paper_shortest_path/results/pyepo_pilot_20260529_153225/plots/pyepo_vs_ours_spoplus_difference.png
```

### Test Results

- Local `tests.test_paper_shortest_path_experiment`: passed, with matplotlib plot test skipped because local base Python lacks matplotlib.
- Local `tests.test_spoplus_shortest_path_validation`: local formula checks passed, but the full test suite failed in base Python because `torch` is missing for the PyEPO comparison path.
- Orange `tests.test_paper_shortest_path_experiment`: passed, with missing-PyEPO-path and matplotlib tests skipped.
- Orange `tests.test_spoplus_shortest_path_validation`: passed.

### Environment Notes

- Orange PyEPO/Torch/Gurobi imports were available.
- Gurobi version on Orange venv: `(13, 0, 2)`.
- Orange CUDA was not available during this run because `nvidia-smi` reported NVIDIA driver/library mismatch: kernel module `580.126.18`, NVML library `580.159`.
- The experiments therefore ran on CPU. This is acceptable for shortest-path sanity/pilot runs, but GPU should not be assumed available until Orange is rebooted or the NVIDIA driver/library mismatch is fixed.

### Forward-Loss Comparison

`compare_pyepo_forward_loss.py` on Orange:

```text
number_of_samples = 3
max_abs_loss_diff = 3.8146972585195726e-06
mean_abs_loss_diff = 1.5432000791311868e-06
```

Interpretation: PyEPO SPOPlus forward loss and the local Step1c-core cost-min
SPO+ formula agree to float/tolerance level. This does not indicate a sign or
convention mismatch.

### Pilot Summary

`pilot_20260529_152538` compared LS and ours-SPO+ only:

```text
LS mean test_norm_spo         = 0.128497
ours-SPO+ mean test_norm_spo  = 0.128863
negative normalized SPO loss  = 0 rows
```

The pilot did not show a clear SPO+ improvement over LS. Degree 1/noise 0 was
perfect for both methods, which is expected. Degree 8 results were essentially
identical for LS and ours-SPO+.

### PyEPO Pilot Summary

`pyepo_pilot_20260529_153225` compared LS, ours-SPO+, and PyEPO SPO+:

```text
LS mean test_norm_spo           = 0.128496997
ours-SPO+ mean test_norm_spo    = 0.128862565
PyEPO-SPO+ mean test_norm_spo   = 0.130814429
negative normalized SPO loss    = 0 rows
```

By regime:

```text
degree=1, noise=0.0:
  LS = 0.000000, ours-SPO+ = 0.000000, PyEPO-SPO+ = 0.000361

degree=1, noise=0.5:
  LS = 0.159355, ours-SPO+ = 0.160817, PyEPO-SPO+ = 0.168465

degree=8, noise=0.0:
  LS = 0.157113, ours-SPO+ = 0.157113, PyEPO-SPO+ = 0.156914

degree=8, noise=0.5:
  LS = 0.197521, ours-SPO+ = 0.197521, PyEPO-SPO+ = 0.197519
```

Paired differences:

```text
ours - LS:
  mean = 0.000365568, median = 0.000000000

PyEPO - LS:
  mean = 0.002317432, median = 0.000314178

ours - PyEPO:
  mean = -0.001951864, median = -0.000314178
```

### Current Interpretation

The pipeline appears to be numerically sane:

- no negative normalized SPO loss;
- formula-level local and PyEPO forward loss agree;
- no evidence of sign reversal;
- LS, ours-SPO+, and PyEPO-SPO+ are close on the pilot.

However, the pilot does **not** yet reproduce the expected qualitative paper
trend where SPO+ improves over LS at high degree. Since PyEPO SPO+ also does
not clearly improve, this looks more like an optimizer/pilot/protocol issue
than an isolated bug in the local SPO+ formula.

### Ready for Advisor Review?

Not yet as a final paper-style result. It is ready to report as an intermediate
validation result:

```text
Forward-loss/formula validation against PyEPO passes, but the current pilot
does not yet show the high-degree SPO+ advantage over LS. We should tune or
audit the training protocol before running full 50-trial middle-row results.
```

### Recommended Next Command

Do **not** run the full 50-trial middle row yet. A safer next step is a training
protocol audit or a small optimizer sweep on the pilot setting, for example:

```bash
python surrogate_experiment_results/SPO_validation/paper_shortest_path/run_paper_shortest_path.py \
  --preset pyepo-pilot \
  --spoplus-iterations 1000 \
  --learning-rate 0.01 \
  --fail-if-pyepo-missing \
  --output-dir surrogate_experiment_results/SPO_validation/paper_shortest_path/results/pyepo_pilot_iter1000_lr001_$(date +%Y%m%d_%H%M%S)
```

## 2026-05-29 16:05 America/Toronto

Git commit:

```text
17a566b7907fa2b76d2d850dbfb74e1bfbc3d4de
```

### Reduced Middle-Row Run

The user requested running a reduced middle-row locally.  This run used all
paper degrees and both noise levels, but only 5 trials:

```bash
python surrogate_experiment_results/SPO_validation/paper_shortest_path/run_paper_shortest_path.py \
  --preset middle-row \
  --trials 5 \
  --output-dir surrogate_experiment_results/SPO_validation/paper_shortest_path/results/middle_row_5trials_20260529_155029
```

Dry-run scale:

```text
degrees = (1, 2, 4, 6, 8)
noise_half_widths = (0.0, 0.5)
trials = 5
n_train/n_val/n_test = 1000/250/10000
methods = ('ls', 'ours-spoplus')
lambda_grid_length = 10
estimated_model_fits = 1000
estimated_ours_spoplus_oracle_calls = 16000000
```

Output:

```text
surrogate_experiment_results/SPO_validation/paper_shortest_path/results/middle_row_5trials_20260529_155029/summary.csv
surrogate_experiment_results/SPO_validation/paper_shortest_path/results/middle_row_5trials_20260529_155029/metadata.json
surrogate_experiment_results/SPO_validation/paper_shortest_path/results/middle_row_5trials_20260529_155029/plots/paper_shortest_path_middle_row.png
```

### Reduced Middle-Row Summary

```text
rows = 100
implementations = ['ls', 'ours-spoplus']
negative normalized SPO loss = 0 rows

LS mean test_norm_spo        = 0.092922181
ours-SPO+ mean test_norm_spo = 0.092502636
```

By degree and noise:

```text
degree=1, noise=0.0:
  LS = 0.000000, ours-SPO+ = 0.000000, ours-LS = 0.000000

degree=1, noise=0.5:
  LS = 0.157302, ours-SPO+ = 0.157872, ours-LS = 0.000570

degree=2, noise=0.0:
  LS = 0.001213, ours-SPO+ = 0.000531, ours-LS = -0.000682

degree=2, noise=0.5:
  LS = 0.109964, ours-SPO+ = 0.109513, ours-LS = -0.000451

degree=4, noise=0.0:
  LS = 0.017721, ours-SPO+ = 0.015697, ours-LS = -0.002024

degree=4, noise=0.5:
  LS = 0.088282, ours-SPO+ = 0.087041, ours-LS = -0.001241

degree=6, noise=0.0:
  LS = 0.060271, ours-SPO+ = 0.060055, ours-LS = -0.000217

degree=6, noise=0.5:
  LS = 0.126607, ours-SPO+ = 0.126456, ours-LS = -0.000151

degree=8, noise=0.0:
  LS = 0.154478, ours-SPO+ = 0.154478, ours-LS = 0.000000

degree=8, noise=0.5:
  LS = 0.213384, ours-SPO+ = 0.213384, ours-LS = 0.000000
```

Aggregated by degree:

```text
degree=1: LS = 0.078651, ours-SPO+ = 0.078936, ours-LS = 0.000285
degree=2: LS = 0.055589, ours-SPO+ = 0.055022, ours-LS = -0.000567
degree=4: LS = 0.053001, ours-SPO+ = 0.051369, ours-LS = -0.001632
degree=6: LS = 0.093439, ours-SPO+ = 0.093255, ours-LS = -0.000184
degree=8: LS = 0.183931, ours-SPO+ = 0.183931, ours-LS = 0.000000
```

### Interpretation

The reduced middle-row is numerically sane:

- no negative normalized SPO loss;
- degree 1/noise 0 is exactly solved by both methods;
- degree 2 and degree 4 show consistent but small SPO+ improvements;
- degree 6 shows very small improvements;
- degree 8 remains identical between LS and ours-SPO+ in this training setup.

This still does **not** reproduce the strong paper-style high-degree SPO+
advantage.  It suggests the data/oracle/metric pipeline is working, but the
current oracle-based stochastic SPO+ training protocol is likely too close to
the LS initialization/checkpoint or needs optimizer/protocol tuning before a
full 50-trial run.

### Current Recommendation

Do **not** run full 50 trials yet.  The next useful step is to audit/tune the
SPO+ training protocol, especially why degree 8 remains LS-identical.  Candidate
checks:

```text
1. Track selected lambda and validation SPO loss per lambda for LS vs SPO+.
2. Track how often the SPO+ best checkpoint differs from the LS initialization.
3. Try a small optimizer sweep on reduced middle-row or pyepo-pilot:
   learning_rate in {0.005, 0.01, 0.05}
   spoplus_iterations in {1000, 3000}
4. Compare no-L1 vs L1 behavior for SPO+ on degree 8.
```

## 2026-05-29 16:25 America/Toronto

### Training-Protocol Diagnostics Added

Implemented the next-stage audit recommended above.  The runner now writes:

```text
summary.csv
training_diagnostics.csv
metadata.json
```

New selected-model summary fields:

```text
spoplus_variant
spoplus_init
best_step
coef_delta_norm_from_ls
val_path_change_rate_from_ls
test_path_change_rate_from_ls
```

New per-lambda diagnostics in `training_diagnostics.csv`:

```text
initial_val_norm_spo
best_val_norm_spo
final_val_norm_spo
best_step
train_loss_last
coef_delta_norm_from_ls
val_path_change_rate_from_ls
```

The default behavior remains the previous protocol:

```text
spoplus_iterate = raw
spoplus_init = ls
eval_period = 50
```

Additional switches were added for audit runs:

```text
--spoplus-iterate raw|averaged
--spoplus-init ls|zero
--eval-period <int>
```

Also added `run_protocol_sweep.py`, which runs fixed high-degree diagnostic
variants: baseline-current, smaller-batch, no-l1, averaged-iterate, and
zero-init-diagnostic.

Current recommendation is unchanged: run diagnostics/sweep before any full
50-trial middle-row experiment.

## 2026-05-29 16:37 America/Toronto

### Local Protocol Sweep

Command:

```bash
python surrogate_experiment_results/SPO_validation/paper_shortest_path/run_protocol_sweep.py \
  --degrees 6 8 \
  --noise-half-widths 0 0.5 \
  --trials 3 \
  --n-test 2000 \
  --output-dir surrogate_experiment_results/SPO_validation/paper_shortest_path/results/protocol_sweep_20260529_163730
```

Output:

```text
surrogate_experiment_results/SPO_validation/paper_shortest_path/results/protocol_sweep_20260529_163730
```

Mean `test_norm_spo` by variant:

```text
baseline-current:
  degree=6, noise=0.0: LS=0.059904, ours=0.059537, ours-LS=-0.000368
  degree=6, noise=0.5: LS=0.133105, ours=0.132830, ours-LS=-0.000275
  degree=8, noise=0.0: LS=0.157113, ours=0.157113, ours-LS=+0.000000
  degree=8, noise=0.5: LS=0.197521, ours=0.197521, ours-LS=+0.000000

smaller-batch:
  degree=6, noise=0.0: LS=0.059904, ours=0.059683, ours-LS=-0.000222
  degree=6, noise=0.5: LS=0.133105, ours=0.133105, ours-LS=+0.000000
  degree=8, noise=0.0: LS=0.157113, ours=0.157113, ours-LS=+0.000000
  degree=8, noise=0.5: LS=0.197521, ours=0.197521, ours-LS=+0.000000

no-l1:
  degree=6, noise=0.0: LS=0.059893, ours=0.059542, ours-LS=-0.000351
  degree=6, noise=0.5: LS=0.133105, ours=0.132830, ours-LS=-0.000275
  degree=8, noise=0.0: LS=0.157113, ours=0.157113, ours-LS=+0.000000
  degree=8, noise=0.5: LS=0.197521, ours=0.197521, ours-LS=+0.000000

averaged-iterate:
  degree=6, noise=0.0: LS=0.059904, ours=0.059678, ours-LS=-0.000226
  degree=6, noise=0.5: LS=0.133105, ours=0.133102, ours-LS=-0.000003
  degree=8, noise=0.0: LS=0.157113, ours=0.157113, ours-LS=+0.000000
  degree=8, noise=0.5: LS=0.197521, ours=0.197521, ours-LS=+0.000000

zero-init-diagnostic:
  degree=6, noise=0.0: LS=0.059904, ours=0.047432, ours-LS=-0.012472
  degree=6, noise=0.5: LS=0.133105, ours=0.123755, ours-LS=-0.009350
  degree=8, noise=0.0: LS=0.157113, ours=0.101746, ours-LS=-0.055366
  degree=8, noise=0.5: LS=0.197521, ours=0.154887, ours-LS=-0.042634
```

Selected diagnostic summary:

```text
baseline-current: mean_coef_delta=1.752635, mean_val_path_change=0.001667, best_step_zero=8/12
smaller-batch: mean_coef_delta=0.090243, mean_val_path_change=0.000333, best_step_zero=11/12
no-l1: mean_coef_delta=1.128559, mean_val_path_change=0.001333, best_step_zero=9/12
averaged-iterate: mean_coef_delta=4.813177, mean_val_path_change=0.001333, best_step_zero=8/12
zero-init-diagnostic: mean_coef_delta=154632.286280, mean_val_path_change=0.565333, best_step_zero=0/12
```

Interpretation: the default LS-initialized protocol is often selecting the
initial LS-equivalent checkpoint, especially at degree 8.  The zero-init
diagnostic produces very different paths and substantially lower normalized SPO
loss in this 3-trial sweep.  This is strong evidence that the next issue is the
training/checkpoint protocol, not the SPO+ forward-loss formula.

Recommended next step: do **not** run full 50 trials yet.  Run a reduced
middle-row with a zero-init SPO+ variant, then compare it against the current
LS-init baseline and, if possible, PyEPO with matching initialization.

## 2026-05-29 17:51 America/Toronto

### Zero-Init 10-Trial Middle-Row Sanity Run

Command:

```bash
python surrogate_experiment_results/SPO_validation/paper_shortest_path/run_paper_shortest_path.py \
  --preset middle-row \
  --trials 10 \
  --spoplus-init zero \
  --spoplus-iterations 1000 \
  --batch-size 32 \
  --learning-rate 0.05 \
  --eval-period 10 \
  --output-dir surrogate_experiment_results/SPO_validation/paper_shortest_path/results/middle_row_zero_10trials_20260529_170844
```

Output:

```text
surrogate_experiment_results/SPO_validation/paper_shortest_path/results/middle_row_zero_10trials_20260529_170844/summary.csv
surrogate_experiment_results/SPO_validation/paper_shortest_path/results/middle_row_zero_10trials_20260529_170844/training_diagnostics.csv
surrogate_experiment_results/SPO_validation/paper_shortest_path/results/middle_row_zero_10trials_20260529_170844/metadata.json
surrogate_experiment_results/SPO_validation/paper_shortest_path/results/middle_row_zero_10trials_20260529_170844/plots/paper_shortest_path_middle_row.png
```

Overall:

```text
rows = 200
negative normalized SPO loss = 0 rows
LS mean test_norm_spo        = 0.093260
zero-init SPO+ mean          = 0.081545
ours-LS                     = -0.011715
```

By degree and noise:

```text
degree=1, noise=0.0: LS=0.000000, zero-init SPO+=0.000015, ours-LS=+0.000015, wins/ties/losses=0/0/10
degree=1, noise=0.5: LS=0.156219, zero-init SPO+=0.159823, ours-LS=+0.003604, wins/ties/losses=0/0/10
degree=2, noise=0.0: LS=0.001357, zero-init SPO+=0.001811, ours-LS=+0.000454, wins/ties/losses=0/0/10
degree=2, noise=0.5: LS=0.107533, zero-init SPO+=0.110418, ours-LS=+0.002884, wins/ties/losses=1/0/9
degree=4, noise=0.0: LS=0.018330, zero-init SPO+=0.024447, ours-LS=+0.006117, wins/ties/losses=1/0/9
degree=4, noise=0.5: LS=0.088756, zero-init SPO+=0.087798, ours-LS=-0.000958, wins/ties/losses=6/0/4
degree=6, noise=0.0: LS=0.066926, zero-init SPO+=0.054075, ours-LS=-0.012851, wins/ties/losses=9/0/1
degree=6, noise=0.5: LS=0.124977, zero-init SPO+=0.115845, ours-LS=-0.009132, wins/ties/losses=7/0/3
degree=8, noise=0.0: LS=0.159264, zero-init SPO+=0.104268, ours-LS=-0.054996, wins/ties/losses=8/0/2
degree=8, noise=0.5: LS=0.209239, zero-init SPO+=0.156953, ours-LS=-0.052286, wins/ties/losses=9/0/1
```

Aggregated by degree:

```text
degree=1: LS=0.078110, zero-init SPO+=0.079919, ours-LS=+0.001809, wins/ties/losses=0/0/20
degree=2: LS=0.054445, zero-init SPO+=0.056114, ours-LS=+0.001669, wins/ties/losses=1/0/19
degree=4: LS=0.053543, zero-init SPO+=0.056122, ours-LS=+0.002579, wins/ties/losses=7/0/13
degree=6: LS=0.095952, zero-init SPO+=0.084960, ours-LS=-0.010991, wins/ties/losses=16/0/4
degree=8: LS=0.184252, zero-init SPO+=0.130611, ours-LS=-0.053641, wins/ties/losses=17/0/3
```

Selected diagnostics:

```text
selected rows = 100
mean coef_delta_norm_from_ls = 59620.233359
mean val_path_change_rate_from_ls = 0.391920
best_step_zero = 0/100
```

Interpretation: zero-init is not a safe unconditional replacement for LS-init,
because it consistently hurts degree 1/2 and is mixed at degree 4.  However, it
strongly improves degree 6/8 and always selects a post-training checkpoint in
this run.  This supports the expert's recommendation to formalize a
validation-selected multi-start protocol that includes both LS-init and
zero-init candidates.

Recommended next step: implement multi-start SPO+ candidate selection and
matching PyEPO zero-init / validation-best checkpoint support before any full
50-trial middle-row run.

## 2026-05-29 18:05 America/Toronto

### Fair PyEPO Protocol Support

Implemented fair PyEPO trainer protocol support for the synthetic shortest-path
experiment:

```text
train_spoplus_pyepo(..., spoplus_init="ls|zero", eval_period=50)
```

The PyEPO path now:

```text
1. supports LS or zero initialization;
2. evaluates validation normalized SPO loss every eval_period steps;
3. returns the validation-best checkpoint instead of the final Adam iterate;
4. records the same diagnostic fields used by ours-spoplus:
   initial_val_norm_spo, best_val_norm_spo, final_val_norm_spo,
   best_step, coef_delta_norm_from_ls, val_path_change_rate_from_ls.
```

This makes the following small PyEPO zero-init comparison meaningful:

```bash
python surrogate_experiment_results/SPO_validation/paper_shortest_path/run_paper_shortest_path.py \
  --preset middle-row-pyepo \
  --trials 3 \
  --n-test 2000 \
  --spoplus-init zero \
  --spoplus-iterations 300 \
  --batch-size 32 \
  --learning-rate 0.05 \
  --eval-period 10 \
  --fail-if-pyepo-missing \
  --output-dir surrogate_experiment_results/SPO_validation/paper_shortest_path/results/middle_row_pyepo_zero_3trials_$(date +%Y%m%d_%H%M%S)
```

## 2026-05-29 18:44 America/Toronto

### PyEPO Zero-Init 3-Trial Paired Run

Command:

```bash
python surrogate_experiment_results/SPO_validation/paper_shortest_path/run_paper_shortest_path.py \
  --preset middle-row-pyepo \
  --trials 3 \
  --n-test 2000 \
  --spoplus-init zero \
  --spoplus-iterations 300 \
  --batch-size 32 \
  --learning-rate 0.05 \
  --eval-period 10 \
  --fail-if-pyepo-missing \
  --output-dir surrogate_experiment_results/SPO_validation/paper_shortest_path/results/middle_row_pyepo_zero_3trials_20260529_184401
```

Output:

```text
surrogate_experiment_results/SPO_validation/paper_shortest_path/results/middle_row_pyepo_zero_3trials_20260529_184401/summary.csv
surrogate_experiment_results/SPO_validation/paper_shortest_path/results/middle_row_pyepo_zero_3trials_20260529_184401/training_diagnostics.csv
surrogate_experiment_results/SPO_validation/paper_shortest_path/results/middle_row_pyepo_zero_3trials_20260529_184401/metadata.json
surrogate_experiment_results/SPO_validation/paper_shortest_path/results/middle_row_pyepo_zero_3trials_20260529_184401/plots/paper_shortest_path_middle_row.png
surrogate_experiment_results/SPO_validation/paper_shortest_path/results/middle_row_pyepo_zero_3trials_20260529_184401/plots/pyepo_vs_ours_spoplus_scatter.png
surrogate_experiment_results/SPO_validation/paper_shortest_path/results/middle_row_pyepo_zero_3trials_20260529_184401/plots/pyepo_vs_ours_spoplus_difference.png
```

Metadata:

```text
methods = [ls, ours-spoplus, pyepo-spoplus]
spoplus_init = zero
eval_period = 10
spoplus_iterations = 300
pyepo_available = true
negative normalized SPO loss = 0 rows
```

Overall mean `test_norm_spo`:

```text
LS          = 0.092844
ours-SPO+   = 0.082095  (ours-LS = -0.010749)
PyEPO-SPO+  = 0.087295  (PyEPO-LS = -0.005549)
ours-PyEPO  = -0.005200
```

Aggregated by degree:

```text
degree=1: LS=0.079677, ours=0.081658, PyEPO=0.083108
degree=2: LS=0.057809, ours=0.060655, PyEPO=0.058175
degree=4: LS=0.052913, ours=0.055216, PyEPO=0.052349
degree=6: LS=0.096505, ours=0.084793, PyEPO=0.101513
degree=8: LS=0.177317, ours=0.128155, PyEPO=0.141333
```

Paired wins/ties/losses over 30 `(trial, degree, noise)` regimes:

```text
ours vs LS:   13 / 0 / 17, mean_diff=-0.010749
PyEPO vs LS:  13 / 0 / 17, mean_diff=-0.005549
ours vs PyEPO: 19 / 0 / 11, mean_diff=-0.005200
```

Selected diagnostic summary:

```text
ours-spoplus:  mean_coef_delta=62054.323891, mean_val_path_change=0.407733, best_step_zero=0/30
pyepo-spoplus: mean_coef_delta=62043.008988, mean_val_path_change=0.422133, best_step_zero=0/30
```

Interpretation: both zero-init SPO+ implementations recover the qualitative
high-degree trend at degree 8.  Ours is lower than PyEPO on average in this
small run, especially at degree 6/noise 0.5 and degree 8/noise 0.5, but the
important sign check is now positive: PyEPO zero-init also improves over LS at
degree 8.  Low-degree regimes remain mixed or worse than LS, so the next
protocol improvement should be validation-selected multi-start rather than
using zero-init unconditionally.
