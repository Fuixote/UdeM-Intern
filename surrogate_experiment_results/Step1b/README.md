# Step1b: Sample-Size Generalization Study

Step1b is the held-out generalization companion to the Step1a landscape
diagnostic.

Step1a asks an in-sample diagnostic question: on a fixed small batch of graph
instances, how do the MSE and Fenchel-Young (FY) training trajectories move
through the two-dimensional decision landscape?

Step1b asks a standard ML-style question: as the number of training graph
instances increases, do the trained reward models improve held-out kidney
exchange decision quality on validation/test graphs that were not used for
training?

The intended use of this folder is to make that question reproducible:

1. Create or reuse a fixed train/validation/test split.
2. Train a two-stage reward-fitting baseline with MSE.
3. Train an end-to-end decision-focused model with the perturbed FY surrogate.
4. Select checkpoints on validation data.
5. Evaluate all selected checkpoints on the same held-out test split.
6. Compare methods with paired test-set statistics.

## Scientific Role

This experiment should be described as a synthetic benchmark, not as evidence
about real clinical utility.

The evaluation labels are synthetic reward weights from:

```text
dataset/processed/step1_noisy_linear_sigma010
```

The label regime is the controlled noisy-linear benchmark used by Step1a:

```text
w_syn_e = max(0, (10 * utility_e + 5 * recipient_cPRA_e) * (1 + noise_e))
```

The probe model is deliberately simple:

```text
w_hat_e(theta) = theta_1 * utility_e + theta_2 * recipient_cPRA_e
```

This makes the experiment easy to interpret. MSE should try to recover the
noise-free linear signal near `[10, 5]`. FY training may move to a different
region if that region gives better downstream KEP decisions under the synthetic
label rewards.

The main evaluation metric is not "true regret" in a clinical sense. It is the
synthetic-label decision gap:

```text
gap(theta) = (w_syn)^T y_oracle - (w_syn)^T S(w_hat(theta))
```

where:

- `y_oracle = S(w_syn)` is the KEP solution optimized under the synthetic label
  weights.
- `S(w_hat(theta))` is the KEP solution induced by the predicted weights.
- Smaller gap is better.

For reporting, use names such as:

- `synthetic-label decision gap`
- `oracle objective gap`
- `decision gap under synthetic rewards`

Avoid `True Regret` unless the caption clearly defines that it is only relative
to synthetic labels.

## Data Split

The default master split is built from:

```text
dataset/processed/step1_noisy_linear_sigma010
```

Default sizes:

```text
train_pool = 1200 graphs
validation = 400 graphs
test       = 400 graphs
```

The default split file is:

```text
results/step1b_splits/master_split_seed=42.json
```

Each Step1b run uses `STEP1B_TRAIN_SIZE=n` to select a reproducible subset from
the 1200-graph train pool. With the same `STEP1B_SUBSET_SEED`, larger training
sizes are selected by reshuffling the same train pool and taking a longer
prefix, so the intended train-size comparison is stable.

Planned train sizes:

```text
n in {50, 200, 600, 1200}
```

Current runs focus on:

```text
n in {50, 200, 600, 1200}
```

## Unseen Test Dataset

On 2026-05-13, a separate 1000-graph unseen test batch was generated locally
for later out-of-split model evaluation. It keeps the same graph-generation
regime and the same synthetic label-processing regime as
`dataset/processed/step1_noisy_linear_sigma010`, but uses a different raw
generator seed and a different processed directory.

Raw batch:

```text
dataset/raw/2026-05-13_073333__step1_noisy_linear_sigma010_unseen_test1000_seed20260513
```

Processed batch:

```text
dataset/processed/step1_noisy_linear_sigma010_unseen_test1000_seed20260513
```

Remote synced processed copy on garnet:

```text
/local1/fuweik/UdeM-Intern/dataset/processed/step1_noisy_linear_sigma010_unseen_test1000_seed20260513
```

The raw unseen batch was intentionally removed from garnet after syncing,
because Step1b evaluation only consumes processed `G-*.json` files. The local
raw batch remains available for provenance.

Generation command:

```bash
python3 0-data-generation.py \
  --instances 1000 \
  --seed 20260513 \
  --run_name step1_noisy_linear_sigma010_unseen_test1000_seed20260513 \
  --output_root dataset/raw
```

Processing command:

```bash
python3 1-data-processing.py \
  dataset/raw/2026-05-13_073333__step1_noisy_linear_sigma010_unseen_test1000_seed20260513 \
  dataset/processed/step1_noisy_linear_sigma010_unseen_test1000_seed20260513 \
  --all \
  --label_mode noisy_clean_linear_utility_cpra \
  --clean_linear_utility_weight 10 \
  --clean_linear_cpra_weight 5 \
  --clean_linear_noise_sigma 0.1 \
  --output_as_batch_dir
```

Verification:

```text
raw genjson-*.json count: 1000
processed G-*.json count: 1000
old raw config seed/instances: 42 / 2000
new raw config seed/instances: 20260513 / 1000
changed raw config keys excluding seed and numberOfInstances: none
processed label mode: noisy_clean_linear_utility_cpra
processed label weights/noise: utility=10, recipient_cPRA=5, sigma=0.1
garnet processed count: 1000, size: 235M
```

Note: the raw generation run created the 1000 JSON files plus
`config.json` and `effective_config.json`, but the raw post-generation report
step returned a Python serialization warning under local Python 3.13 because an
internal argparse field was not JSON-serializable. The generator script was
patched afterward to omit internal argparse fields from JSON metadata, and a
1-instance smoke run verified that future raw reports write normally. This
specific raw batch still lacks raw `run_info.json` unless regenerated, but the
processed dataset completed normally and contains `run_info.json`,
`batch_summary.json`, and `batch_report.md`.

## Unseen Evaluation Script

Use `evaluate_unseen_run.py` to evaluate one completed Step1b training run
directory on the 1000-graph unseen processed dataset. This script is not a
sweep driver: it takes one `--run_dir`, automatically finds the three standard
model checkpoints under `model_weights/`, evaluates all three on the same
unseen graphs, and writes the summary back under that same run directory.

Expected model weights per run:

```text
model_weights/2stage_best_by_validation_mse_loss.npz
model_weights/e2e_best_by_validation_decision_gap.npz
model_weights/e2e_best_by_validation_fy_loss.npz
```

Example for the local copied `train_size=50` result:

```bash
python surrogate_experiment_results/Step1b/evaluate_unseen_run.py \
  --run_dir surrogate_experiment_results/Step1b/remote_results/formal_M16_2stage500_e2e500_s10/train_size=50
```

Example on garnet, using the live remote result directory:

```bash
cd /local1/fuweik/UdeM-Intern
source configs/runtime/garnet.env
python surrogate_experiment_results/Step1b/evaluate_unseen_run.py \
  --run_dir results/step1b_runs/train_size=50/split_seed=42/subset_seed=42/theta_seed=42/eps=1.0_M=16_2stage_epochs=500_e2e_epochs=500_stride=10
```

Outputs are written under `RUN_DIR/metrics/`:

```text
unseen_test_summary.csv
unseen_test_summary.json
unseen_test_per_graph.csv
unseen_test_run_config.json
```

For a tiny smoke test, add:

```text
--graph_limit 2 --output_stem unseen_test_smoke2 --bootstrap_samples 10
```

### 1000-graph unseen results

On 2026-05-13, the completed `train_size=50`, `200`, and `600` formal runs were
evaluated on the 1000-graph unseen processed dataset. The unseen evaluation ran
quickly because it only evaluates three fixed two-parameter models per
train-size directory; it does not perform FY training.

Local archived output paths:

```text
surrogate_experiment_results/Step1b/remote_results/formal_M16_2stage500_e2e500_s10/train_size=<n>/metrics/unseen_test_summary.csv
surrogate_experiment_results/Step1b/remote_results/formal_M16_2stage500_e2e500_s10/train_size=<n>/metrics/unseen_test_per_graph.csv
```

Unseen test summary:

| train_size | method | selected by | epoch | theta_1 | theta_2 | unseen mean gap | unseen norm gap | paired improvement over 2stage | 95% CI |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 50 | 2stage | val MSE | 500 | 9.8707 | 5.1130 | 0.7311 | 0.00531 | -- | -- |
| 50 | e2e | val decision gap | 50 | 5.4807 | 2.8132 | 0.7352 | 0.00534 | -0.0041 | [-0.0114, 0.0006] |
| 50 | e2e | val FY loss | 440 | 11.8304 | 5.0208 | 0.7591 | 0.00550 | -0.0280 | [-0.0878, 0.0319] |
| 200 | 2stage | val MSE | 500 | 9.8848 | 5.1038 | 0.7313 | 0.00532 | -- | -- |
| 200 | e2e | val decision gap | 150 | 8.8078 | 4.5582 | 0.7313 | 0.00532 | 0.0000 | [0.0000, 0.0000] |
| 200 | e2e | val FY loss | 500 | 12.2682 | 6.0649 | 0.7262 | 0.00528 | 0.0050 | [-0.0200, 0.0320] |
| 600 | 2stage | val MSE | 500 | 9.8899 | 5.0963 | 0.7309 | 0.00531 | -- | -- |
| 600 | e2e | val decision gap | 110 | 7.7838 | 4.0153 | 0.7329 | 0.00533 | -0.0019 | [-0.0078, 0.0000] |
| 600 | e2e | val FY loss | 490 | 12.2328 | 5.8586 | 0.7488 | 0.00545 | -0.0179 | [-0.0564, 0.0219] |

Interpretation:

- The 1000-graph unseen results do not replicate the stronger FY-loss-selected
  improvements seen on the original 400 held-out test split.
- The 2stage baseline is very stable across train sizes on this unseen dataset:
  mean gap stays near `0.731` and normalized gap near `0.00531`.
- The validation-FY-loss-selected e2e checkpoint is slightly better than 2stage
  only for `train_size=200`, and its paired bootstrap interval still crosses
  zero.
- More than half of unseen graphs have zero decision gap for each model, and
  many method pairs produce identical decisions on most graphs. For example,
  e2e selected by validation decision gap is identical to 2stage on all 1000
  unseen graphs for `train_size=200`, and identical on 999/1000 graphs for
  `train_size=600`. Therefore small mean differences are driven by a small
  subset of graphs where the induced KEP solution changes.

### Realistic-synthetic label stress test

On 2026-05-13, the same completed `train_size=50`, `200`, and `600` formal
runs were also evaluated on:

```text
dataset/processed/realistic_synthetic_dataset
```

This dataset has 2000 processed graphs and uses the more complex
`realistic_synthetic` label regime:

```text
expected_transplant_count * qaly * priority_multiplier * (1 + deterministic_noise)
```

Important caveat: this is best described as a label-regime or distribution-shift
stress test, not as a fully graph-unseen test. Both
`step1_noisy_linear_sigma010` and `realistic_synthetic_dataset` were processed
from the same raw batch:

```text
raw_batch_name = 2026-04-17_135607
```

Therefore graph structures may overlap with the original Step1b train,
validation, and held-out test split. What changes is the reward-label mechanism:
models trained on the noisy-linear synthetic rewards are evaluated under the
more complex realistic-synthetic rewards.

The remote processed copy is:

```text
/local1/fuweik/UdeM-Intern/dataset/processed/realistic_synthetic_dataset
```

It contains:

```text
G-*.json count: 2000
size on garnet: 451M
```

Evaluation command pattern on garnet:

```bash
cd /local1/fuweik/UdeM-Intern
source configs/runtime/garnet.env
python surrogate_experiment_results/Step1b/evaluate_unseen_run.py \
  --run_dir results/step1b_runs/train_size=50/split_seed=42/subset_seed=42/theta_seed=42/eps=1.0_M=16_2stage_epochs=500_e2e_epochs=500_stride=10 \
  --dataset_dir dataset/processed/realistic_synthetic_dataset \
  --output_stem realistic_unseen_test
```

Local archived output paths:

```text
surrogate_experiment_results/Step1b/remote_results/formal_M16_2stage500_e2e500_s10/train_size=<n>/metrics/realistic_unseen_test_summary.csv
surrogate_experiment_results/Step1b/remote_results/formal_M16_2stage500_e2e500_s10/train_size=<n>/metrics/realistic_unseen_test_per_graph.csv
```

Realistic-synthetic stress-test summary:

| train_size | method | selected by | epoch | theta_1 | theta_2 | mean gap | norm gap | achieved/oracle | paired improvement over 2stage | 95% CI |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 50 | 2stage | val MSE | 500 | 9.8707 | 5.1130 | 12.1103 | 0.06603 | 0.93397 | -- | -- |
| 50 | e2e | val decision gap | 50 | 5.4807 | 2.8132 | 12.1314 | 0.06613 | 0.93387 | -0.0210 | [-0.0750, 0.0239] |
| 50 | e2e | val FY loss | 440 | 11.8304 | 5.0208 | 12.3439 | 0.06718 | 0.93282 | -0.2335 | [-0.4036, -0.0678] |
| 200 | 2stage | val MSE | 500 | 9.8848 | 5.1038 | 12.0837 | 0.06588 | 0.93412 | -- | -- |
| 200 | e2e | val decision gap | 150 | 8.8078 | 4.5582 | 12.0879 | 0.06591 | 0.93409 | -0.0041 | [-0.0118, 0.0000] |
| 200 | e2e | val FY loss | 500 | 12.2682 | 6.0649 | 12.1739 | 0.06631 | 0.93369 | -0.0902 | [-0.1674, -0.0198] |
| 600 | 2stage | val MSE | 500 | 9.8899 | 5.0963 | 12.1108 | 0.06606 | 0.93394 | -- | -- |
| 600 | e2e | val decision gap | 110 | 7.7838 | 4.0153 | 12.1108 | 0.06606 | 0.93394 | 0.0000 | [0.0000, 0.0000] |
| 600 | e2e | val FY loss | 490 | 12.2328 | 5.8586 | 12.1670 | 0.06618 | 0.93382 | -0.0562 | [-0.1618, 0.0471] |

Interpretation:

- Under this shifted realistic-synthetic reward label, the noisy-linear-trained
  2stage model is slightly better than or tied with e2e for the completed
  `train_size=50`, `200`, and `600` runs.
- The e2e checkpoint selected by validation decision gap is almost tied with
  2stage, but does not show a positive paired improvement.
- The e2e checkpoint selected by validation FY loss is worse than 2stage for
  `train_size=50` and `200` with a negative paired bootstrap interval; for
  `train_size=600` the interval crosses zero.
- This result should not be read as a failure of FY on its original benchmark.
  It mainly says that training on the controlled noisy-linear labels does not
  transfer cleanly to a different, more complex synthetic reward definition
  within the same two-feature linear probe class.

## Methods

### 2stage: reward fitting plus downstream optimization

The two-stage baseline trains the same two-feature linear reward model with MSE:

```text
w_hat_e(theta) = theta_1 * utility_e + theta_2 * recipient_cPRA_e
```

Stage 1:

- Optimize MSE between `w_hat` and the synthetic label weights `w_syn`.
- Use Adam with default `STEP1B_2STAGE_LR=0.05`.
- Record `train_mse_loss` and `validation_mse_loss` for every epoch.
- Select the checkpoint with the lowest validation MSE.

Stage 2:

- During evaluation, solve the KEP problem with Gurobi using the learned
  predicted weights.
- Score the resulting solution under the synthetic label weights.

Saved checkpoint:

```text
model_weights/2stage_best_by_validation_mse_loss.npz
model_weights/2stage_best_by_validation_mse_loss.json
```

### e2e: decision-focused FY training

The end-to-end model uses the same linear probe, but trains it with the
perturbed FY surrogate. For each training graph, the gradient estimate compares
the average perturbed optimizer solution to the synthetic-label oracle solution:

```text
grad approx X^T (mean_y_perturbed - y_oracle)
```

Important FY controls:

- `STEP1B_FY_EPSILON`: perturbation scale. Default is `1.0`.
- `STEP1B_FY_M`: number of antithetic perturbation samples. Default is `4`.
- `STEP1B_E2E_N_EPOCHS`: FY training epochs. Default is `100`.
- `STEP1B_METRIC_STRIDE`: spacing of offline validation checkpoints. Default is
  `1`.

Two e2e checkpoints are saved:

```text
model_weights/e2e_best_by_validation_decision_gap.npz
model_weights/e2e_best_by_validation_decision_gap.json

model_weights/e2e_best_by_validation_fy_loss.npz
model_weights/e2e_best_by_validation_fy_loss.json
```

These two checkpoints answer different questions:

- `best_by_validation_decision_gap`: did the FY trajectory produce a model with
  good held-out downstream decision quality?
- `best_by_validation_fy_loss`: can the FY surrogate itself be used as a
  validation/model-selection loss?

They often differ. If they are identical in a run, that means the two validation
criteria happened to choose the same evaluated epoch.

## Validation Metrics

For e2e, the offline trajectory evaluation writes:

```text
metrics/e2e_loss_curve.csv
```

with columns:

```text
epoch
theta_1
theta_2
train_fy_loss
validation_fy_loss
train_decision_gap
validation_decision_gap
```

Interpretation:

- `validation_decision_gap` is the held-out downstream decision metric under
  synthetic label rewards. It is the metric we ultimately care about.
- `validation_fy_loss` is the perturbed FY surrogate objective. It is the proxy
  loss whose usefulness is being tested.

If `STEP1B_METRIC_STRIDE=10`, e2e checkpoints are only evaluated at:

```text
epoch 0, 10, 20, ..., final_epoch
```

Therefore the two e2e selected epochs will be multiples of 10, except for a
non-multiple final epoch if early stopping ends between stride points. This
stride does not change training; it only controls offline metric evaluation and
checkpoint selection granularity.

For 2stage, MSE evaluation is cheap. The script evaluates every epoch:

```text
epoch 0, 1, 2, ..., n_epochs_2stage
```

so the 2stage selected epoch is not tied to `STEP1B_METRIC_STRIDE`.

## Test Metrics

The test evaluator loads the selected model weights and solves KEP on the same
held-out test graphs for all methods.

Outputs:

```text
metrics/test_summary.csv
metrics/test_summary.json
metrics/test_per_graph.csv
```

`test_summary.csv` contains:

- `test_mean_decision_gap`
- `test_mean_normalized_gap`
- `test_median_normalized_gap`
- `test_mean_achieved_oracle_ratio`
- `paired_mean_improvement_over_2stage`
- `paired_median_improvement_over_2stage`
- `fraction_improved_over_2stage`
- `paired_mean_improvement_ci_low`
- `paired_mean_improvement_ci_high`

The paired improvement is:

```text
gap_i(2stage) - gap_i(candidate)
```

Positive values mean the candidate has a smaller decision gap than the 2stage
baseline on the same test graph.

## Scripts

Main wrapper:

```text
surrogate_experiment_results/Step1b/run_step1b.sh
```

Pipeline order:

```text
split_dataset.py
train_2stage.py
train_end2end.py
evaluate_models.py
```

Supporting files:

```text
step1b_common.py        shared graph loading, FY gradient, metrics
plot_training_curves.py plots train/validation curves
run_config.py           helper for run-directory naming
```

## Output Layout

Default output root:

```text
results/step1b_runs/
```

Default script-generated directory shape:

```text
results/step1b_runs/
  train_size=<n>/
    split_seed=<seed>/
      subset_seed=<seed>/
        theta_seed=<seed>/
          eps=<epsilon>_M=<M>_e2e_epochs=<epochs>_stride=<stride>/
```

For formal runs, prefer setting `STEP1B_OUTPUT_DIR` explicitly and include
important hyperparameters in the directory name, especially
`2stage_epochs`, because the default directory name currently does not include
the 2stage epoch count.

Recommended formal directory name:

```text
eps=1.0_M=16_2stage_epochs=500_e2e_epochs=500_stride=10
```

Important files inside a completed run:

```text
run_config.json
train_subset.json

trajectories/trajectory_2stage.npy
trajectories/trajectory_e2e.npy

model_weights/2stage_best_by_validation_mse_loss.npz
model_weights/2stage_best_by_validation_mse_loss.json
model_weights/e2e_best_by_validation_decision_gap.npz
model_weights/e2e_best_by_validation_decision_gap.json
model_weights/e2e_best_by_validation_fy_loss.npz
model_weights/e2e_best_by_validation_fy_loss.json

metrics/2stage_loss_curve.csv
metrics/e2e_loss_curve.csv
metrics/early_stopping.json
metrics/test_summary.csv
metrics/test_summary.json
metrics/test_per_graph.csv

plots/2stage_mse_loss.png
plots/e2e_fy_loss.png
```

The `.npz` weight files are intentionally small. The current model is linear,
so a checkpoint mainly stores:

```text
theta
method
model_type
model_formula
feature_names
train_size
selected_epoch
selection_metric
selection_value
```

## Local One-Shot Run

From the repo root:

```bash
MPLCONFIGDIR=/tmp/matplotlib \
STEP1B_TRAIN_SIZE=50 \
STEP1B_2STAGE_N_EPOCHS=500 \
STEP1B_E2E_N_EPOCHS=500 \
STEP1B_FY_EPSILON=1.0 \
STEP1B_FY_M=16 \
STEP1B_METRIC_STRIDE=10 \
STEP1B_OUTPUT_DIR=results/step1b_runs/train_size=50/split_seed=42/subset_seed=42/theta_seed=42/eps=1.0_M=16_2stage_epochs=500_e2e_epochs=500_stride=10 \
surrogate_experiment_results/Step1b/run_step1b.sh
```

## Garnet Remote Run

Primary remote target:

```text
ssh cirrelt
```

Remote repo root:

```text
/local1/fuweik/UdeM-Intern
```

Runtime setup on garnet:

```bash
cd /local1/fuweik/UdeM-Intern
source configs/runtime/garnet.env
```

The runtime activates the `KEPs` conda environment, loads Gurobi, and redirects
data/results/solutions to `/local1/fuweik/UdeM-Intern`.

Before running, confirm data and split exist:

```bash
find dataset/processed/step1_noisy_linear_sigma010 -name 'G-*.json' | head
test -f results/step1b_splits/master_split_seed=42.json
```

If the local Step1b scripts have changed, sync them first:

```bash
rsync -av surrogate_experiment_results/Step1b/ \
  cirrelt:/local1/fuweik/UdeM-Intern/surrogate_experiment_results/Step1b/
```

Start the three current formal train sizes in parallel:

```bash
ssh cirrelt 'bash -s' <<'REMOTE'
set -euo pipefail

cd /local1/fuweik/UdeM-Intern
source configs/runtime/garnet.env
mkdir -p logs /tmp/matplotlib

for n in 50 200 600; do
  outdir="results/step1b_runs/train_size=${n}/split_seed=42/subset_seed=42/theta_seed=42/eps=1.0_M=16_2stage_epochs=500_e2e_epochs=500_stride=10"
  logfile="logs/step1b_train_size_${n}_M16_2stg500_e2e500_s10.log"

  nohup env \
    MPLCONFIGDIR=/tmp/matplotlib \
    STEP1B_TRAIN_SIZE="${n}" \
    STEP1B_SPLIT_PATH="results/step1b_splits/master_split_seed=42.json" \
    STEP1B_2STAGE_N_EPOCHS=500 \
    STEP1B_E2E_N_EPOCHS=500 \
    STEP1B_FY_M=16 \
    STEP1B_METRIC_STRIDE=10 \
    STEP1B_OUTPUT_DIR="${outdir}" \
    surrogate_experiment_results/Step1b/run_step1b.sh \
    > "${logfile}" 2>&1 &

  echo "started train_size=${n}, pid=$!, log=${logfile}, out=${outdir}"
done
REMOTE
```

Status check:

```bash
ssh cirrelt 'ps -fu fuweik | grep -E "Step1b/run_step1b|train_2stage|train_end2end|evaluate_models" | grep -v grep'
```

Artifact check:

```bash
ssh cirrelt 'cd /local1/fuweik/UdeM-Intern && for n in 50 200 600 1200; do d=results/step1b_runs/train_size=${n}/split_seed=42/subset_seed=42/theta_seed=42/eps=1.0_M=16_2stage_epochs=500_e2e_epochs=500_stride=10; echo ==== $n ====; test -f "$d/model_weights/2stage_best_by_validation_mse_loss.json" && echo 2stage_done || echo 2stage_wait; test -f "$d/trajectories/trajectory_e2e.npy" && echo e2e_training_done || echo e2e_training_or_wait; test -f "$d/metrics/e2e_loss_curve.csv" && echo e2e_metrics_done || echo e2e_metrics_wait; test -f "$d/metrics/test_summary.csv" && echo DONE || echo test_wait; done'
```

Sync completed results back to local:

```bash
rsync -av \
  cirrelt:/local1/fuweik/UdeM-Intern/results/step1b_runs/train_size=50/ \
  results/step1b_runs/train_size=50/
```

Repeat for `train_size=200`, `train_size=600`, and `train_size=1200`, or sync the whole
`results/step1b_runs/` tree if the output size is acceptable.

## Early Stopping

The current local code supports e2e early stopping by validation FY loss:

```bash
STEP1B_E2E_EARLY_STOP_METRIC=validation_fy_loss
STEP1B_E2E_EARLY_STOP_PATIENCE=30
STEP1B_E2E_EARLY_STOP_MIN_DELTA=0.0001
```

This is intentionally not the default. Online early stopping is expensive
because every validation check requires perturbed FY objective solves over the
validation split. In a local test with 400 validation graphs, the first online
validation FY check was slow enough that the fixed-epoch/offline-evaluation
workflow remained preferable for the current formal runs.

Use early stopping only after deciding that the additional validation solve
cost is worth it.

## Completed Preliminary Run

The first completed remote comparison used:

```text
dataset:        dataset/processed/step1_noisy_linear_sigma010
split:          train_pool=1200, validation=400, test=400, seed=42
train_size:     50, 200, 600, 1200
theta_seed:     42
subset_seed:    42
gurobi_seed:    42
2stage epochs:  100
e2e epochs:     100
FY epsilon:     1.0
FY M:           4
metric_stride:  1
test size:      400
```

Path pattern on garnet:

```text
results/step1b_runs/train_size=<n>/split_seed=42/subset_seed=42/theta_seed=42/eps=1.0_M=4_e2e_epochs=100_stride=1/
```

Summary:

| train_size | method | selected by | epoch | theta_1 | theta_2 | test mean gap | test norm gap | paired improvement over 2stage |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 50 | 2stage | val MSE | 100 | 5.8025 | 6.9055 | 2.5684 | 0.01853 | -- |
| 50 | e2e | val decision gap | 51 | 5.5574 | 2.8700 | 0.8480 | 0.00611 | 1.7203 |
| 50 | e2e | val FY loss | 100 | 7.4683 | 3.3282 | 0.7311 | 0.00525 | 1.8373 |
| 200 | 2stage | val MSE | 100 | 5.8004 | 6.9065 | 2.5670 | 0.01851 | -- |
| 200 | e2e | val decision gap | 77 | 6.7171 | 3.6629 | 0.8713 | 0.00635 | 1.6957 |
| 200 | e2e | val FY loss | 100 | 7.4971 | 4.0502 | 0.8669 | 0.00632 | 1.7001 |
| 600 | 2stage | val MSE | 100 | 5.8031 | 6.9068 | 2.5670 | 0.01851 | -- |
| 600 | e2e | val decision gap | 94 | 7.2973 | 3.8497 | 0.8590 | 0.00619 | 1.7080 |
| 600 | e2e | val FY loss | 100 | 7.4886 | 3.8994 | 0.8604 | 0.00619 | 1.7066 |

Caveats:

- This run is useful as a pipeline validation, but it is not the final formal
  comparison.
- The 2stage baseline was undertrained: its best validation MSE checkpoint was
  the last epoch, and the learned theta stayed near `[5.8, 6.9]` instead of the
  expected clean-signal coefficient `[10, 5]`.
- The `train_size=50` e2e result being slightly better than `train_size=200`
  does not by itself imply a real sample-size trend. This run used only one
  seed, 100 epochs, `M=4`, and a 400-graph test split.
- The paired improvements were positive for e2e against this undertrained
  2stage baseline, but the next run strengthens the baseline and FY estimate.

## Current Formal Run Record

Started on garnet on 2026-05-12.

Configuration:

```text
dataset:        dataset/processed/step1_noisy_linear_sigma010
split:          results/step1b_splits/master_split_seed=42.json
train_size:     50, 200, 600, 1200
theta_seed:     42
subset_seed:    42
gurobi_seed:    42
2stage epochs:  500
e2e epochs:     500
FY epsilon:     1.0
FY M:           16
metric_stride:  10
test size:      400
```

Path pattern:

```text
results/step1b_runs/train_size=<n>/split_seed=42/subset_seed=42/theta_seed=42/eps=1.0_M=16_2stage_epochs=500_e2e_epochs=500_stride=10/
```

Local clean archive path for completed synced runs:

```text
surrogate_experiment_results/Step1b/remote_results/formal_M16_2stage500_e2e500_s10/train_size=<n>/
```

As of 2026-05-13 06:15 EDT, `train_size=50`, `200`, and `600` had completed
end to end; `1200` was still running in e2e training. The 2stage
validation-MSE checkpoints available at that point were:

| train_size | selected epoch | theta_1 | theta_2 | validation MSE |
|---:|---:|---:|---:|---:|
| 50 | 500 | 9.8707 | 5.1130 | 0.453802 |
| 200 | 500 | 9.8848 | 5.1038 | 0.453116 |
| 600 | 500 | 9.8899 | 5.0963 | 0.452829 |
| 1200 | 500 | 9.8848 | 5.0949 | 0.452999 |

These 2stage checkpoints are much closer to the expected clean-signal
coefficient `[10, 5]` than the 100-epoch preliminary checkpoints. However,
because the best validation MSE is still at epoch 500, a future 2stage-only
check with more epochs or MSE early stopping may be useful.

Completed test metrics:

| method | selected by | epoch | theta_1 | theta_2 | test mean gap | test norm gap | paired improvement over 2stage | 95% CI |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 50 / 2stage | val MSE | 500 | 9.8707 | 5.1130 | 0.8480 | 0.00611 | -- | -- |
| 50 / e2e | val decision gap | 50 | 5.4807 | 2.8132 | 0.8220 | 0.00592 | 0.0260 | [-0.0096, 0.0805] |
| 50 / e2e | val FY loss | 440 | 11.8304 | 5.0208 | 0.7587 | 0.00547 | 0.0894 | [-0.0057, 0.1908] |
| 200 / 2stage | val MSE | 500 | 9.8848 | 5.1038 | 0.8451 | 0.00610 | -- | -- |
| 200 / e2e | val decision gap | 150 | 8.8078 | 4.5582 | 0.8480 | 0.00611 | -0.0029 | [-0.0088, 0.0000] |
| 200 / e2e | val FY loss | 500 | 12.2682 | 6.0649 | 0.7730 | 0.00563 | 0.0721 | [0.0126, 0.1505] |
| 600 / 2stage | val MSE | 500 | 9.8899 | 5.0963 | 0.8223 | 0.00591 | -- | -- |
| 600 / e2e | val decision gap | 110 | 7.7838 | 4.0153 | 0.8223 | 0.00591 | 0.0000 | [0.0000, 0.0000] |
| 600 / e2e | val FY loss | 490 | 12.2328 | 5.8586 | 0.7507 | 0.00550 | 0.0716 | [0.0137, 0.1347] |

For `train_size=50`, both e2e checkpoints improve the mean test decision gap
relative to 2stage, but the paired bootstrap confidence intervals still cross
zero. For `train_size=200` and `600`, the validation-decision-gap-selected e2e
checkpoint does not improve over 2stage on test, while the
validation-FY-loss-selected checkpoint has a lower mean test gap and a positive
paired bootstrap interval. This makes the FY-loss-selected checkpoint especially
important for the final sample-size comparison.

Expected runtime:

```text
train_size=50   roughly 1.5-2.5 hours
train_size=200  roughly 4-6 hours
train_size=600  roughly 10-12 hours, likely overnight
train_size=1200 expected to be longer than 600; treat it as an overnight run
```

The total wall-clock time is now dominated by the `train_size=1200` FY run.

## Analysis Guidance

For the next analysis pass, inspect:

```text
metrics/test_summary.csv
metrics/e2e_loss_curve.csv
metrics/2stage_loss_curve.csv
model_weights/*.json
plots/*.png
```

Main questions:

1. Does `test_mean_normalized_gap` decrease as `train_size` grows?
2. Does e2e selected by validation decision gap beat 2stage on paired test
   graphs?
3. Does e2e selected by validation FY loss also beat 2stage?
4. Are the paired improvement confidence intervals positive?
5. Do the validation FY loss and validation decision gap select similar or
   different epochs?
6. Does the 2stage baseline appear sufficiently trained?

Interpretation rules:

- `e2e_best_by_validation_decision_gap` is allowed in the synthetic benchmark
  because validation synthetic labels are known. It answers whether the FY
  trajectory contains a decision-good model.
- `e2e_best_by_validation_fy_loss` is the more direct test of FY as a surrogate
  model-selection loss.
- Final plots should include all three curves/rows:
  `2stage selected by val MSE`,
  `e2e selected by val decision gap`,
  and `e2e selected by val FY loss`.
- Do not claim that Step1b proves real clinical regret improvement. The claim
  is about held-out synthetic-label decision quality on a controlled noisy
  linear synthetic reward benchmark.

## Next Extensions

The current 400-graph test split is appropriate for debugging and the first
formal pass. For a stronger final result, reuse the saved model weights and
evaluate on a much larger unseen test set, for example 10,000 newly generated
graphs from the same synthetic distribution.

Recommended future additions:

1. A standalone evaluation script that loads existing `.npz` model weights and
   evaluates them on a newly generated 10,000-graph test set.
2. A result aggregation script that reads multiple completed run directories and
   writes a sample-size summary table.
3. A plotting script for train size versus test normalized decision gap.
4. Multiple training seeds or subset seeds to quantify training variance.
5. A 2stage-only convergence check with 1000-2000 epochs or cheap validation-MSE
   early stopping.
