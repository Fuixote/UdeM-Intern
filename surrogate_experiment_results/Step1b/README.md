# Step1b: Sample-Size Generalization Smoke Experiment

This folder starts the held-out generalization layer for the Step1 noisy-linear
synthetic reward benchmark.

The current minimal protocol is intentionally small:

- train split: 2 graphs by default
- validation split: 2 graphs by default
- test split: 2 graphs by default
- methods: `mse` and `fy_warm` by default
- FY training: 1 epoch, `M=2` perturbations by default
- checkpoint rule: `validation_decision_gap`

Run the smoke test from the repo root:

```bash
surrogate_experiment_results/Step1b/run_step1b_smoke.sh
```

Scale it by overriding environment variables:

```bash
STEP1B_TRAIN_SIZE=50 \
STEP1B_VAL_SIZE=200 \
STEP1B_TEST_SIZE=500 \
STEP1B_N_EPOCHS=50 \
STEP1B_FY_M=4 \
STEP1B_METHODS="mse fy_random fy_warm" \
STEP1B_CHECKPOINT_RULES="validation_decision_gap validation_fy_objective" \
surrogate_experiment_results/Step1b/run_step1b_smoke.sh
```

Outputs are written under `results/step1b_runs/smoke/` unless
`STEP1B_OUTPUT_DIR` is set. The key files are:

- `split.json`
- `config.json`
- `trajectories/trajectory_mse.npy`
- `trajectories/trajectory_fy_warm.npy`
- `metrics/validation_trajectory_metrics.csv`
- `metrics/test_summary.csv`
- `metrics/test_per_graph.csv`
