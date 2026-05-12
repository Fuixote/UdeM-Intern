# Step1b: Sample-Size Generalization Study

Step1b evaluates how the number of training graph instances affects held-out
decision quality.

The pipeline uses a fixed master split from
`dataset/processed/step1_noisy_linear_sigma010`:

- train pool: 1200 graphs
- validation: 400 graphs
- test: 400 graphs

For each run, `STEP1B_TRAIN_SIZE=n` selects a reproducible subset from the
1200-graph train pool. The intended sweep is:

```text
n in {50, 200, 600, 1200}
```

## Methods

`2stage`:

1. Stage 1 trains the linear reward model with MSE.
2. The checkpoint is selected by validation MSE loss.
3. Stage 2 is the downstream Gurobi KEP solve used during test evaluation.

`e2e`:

1. Trains the same linear model with the perturbed FY surrogate.
2. The checkpoint is selected by validation synthetic-label decision gap.
3. The selected model is evaluated on the same 400-graph test split.

Both methods can share an explicit initialization via:

```bash
STEP1B_THETA_INIT="2.0 1.0"
```

If omitted, the initialization is drawn reproducibly from `STEP1B_THETA_SEED`.

## One-Shot Run

Run one train size from the repo root:

```bash
STEP1B_TRAIN_SIZE=50 \
STEP1B_2STAGE_N_EPOCHS=100 \
STEP1B_E2E_N_EPOCHS=100 \
STEP1B_FY_M=4 \
surrogate_experiment_results/Step1b/run_step1b.sh
```

The script runs:

1. `split_dataset.py`
2. `train_2stage.py`
3. `train_end2end.py`
4. `evaluate_models.py`

Key outputs are under:

```text
results/step1b_runs/train_size=<n>/seed=<seed>/
```

including:

- `train_subset.json`
- `model_weights/2stage_best_by_validation_mse_loss.npz`
- `model_weights/e2e_best_by_validation_decision_gap.npz`
- `metrics/2stage_loss_curve.csv`
- `metrics/e2e_loss_curve.csv`
- `metrics/test_summary.csv`
- `plots/2stage_mse_loss.png`
- `plots/e2e_fy_loss.png`
