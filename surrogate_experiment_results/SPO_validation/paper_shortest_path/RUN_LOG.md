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
