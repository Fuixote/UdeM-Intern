# Step5 Experiment 01: Seed-42 Topology Weak Labels

## Status

The local pipeline implementation and pre-smoke protocol lock are complete as of 2026-07-19. The implementation was recorded in commit `6208873d53dfe4e624a80bfbb2aea8b1cb40b386`, synchronized to Garnet, and used to complete the smoke20 artifact build and audit. No model jobs have been launched yet.

The fixed experiment contract is:

```text
topologies       1000
data_seed        42
sample_size      50
training_size    40
validation_size  10
test_size        1000
theta_seed       42
gurobi_seed      42
master_label_seed 20260719
protocol         screen
regime           step2c_poly_d8_mult_eps050
max_epochs       1500
metric_stride    1
early_stop       patience=20, min_delta=0.0001
```

The `master_label_seed` fixes the label-generating process and is deliberately distinct from `data_seed=42`. The 1500-epoch cap follows the Step3 native-e1500 cap check: extending the reviewed cap-hit cases to 3000 epochs produced no decision-outcome changes, so the lower cap remains the frozen protocol.

Locked configuration artifacts:

- `configs/experiment.yaml`
- `configs/context_generator.locked.yaml`
- `configs/topologies.locked.csv`
- `results/topology_bank_audit.json`

The locked topology manifest has 1000 rows and SHA-256 `c621300e21d06b2217951abf007317bf774e9e01616a0bbb5c050e121f5847a2`. The parsed generator config hash recorded in generated manifests is `97d25caaf4383fecfe31ba2a03491d7bd30fbab0f869d9dff14b17bfe0042e49`.

## Smoke20 artifact checkpoint

The first 20 topologies (`G-0` through `G-19`) were materialized and audited on Garnet on 2026-07-19 under:

```text
/local1/fuweik/UdeM-Intern/surrogate_experiment_results/Step5_topology_dfl_suitability/experiment_01_weak_label_seed42_sample50/results/smoke20
```

Verified results:

- artifact build: `passed=true`, `built=20`, `failed=0`;
- artifact audit: `passed=true`, `passed_topologies=20`, `failed_topologies=0`, `failures=[]`;
- every topology has one 40-sample training bank, one 10-sample validation artifact, and one 1000-sample test artifact;
- inventory: 60 NPZ files, 80 manifest JSON files, approximately 137 MB total;
- protocol: `data_seed=42`, `master_label_seed=20260719`, `sample_size=50`, and regime `step2c_poly_d8_mult_eps050`.

The artifact gate was followed by the dry-run plan and launcher-preview checkpoint described below.

## Smoke20 plan and launcher-preview checkpoint

The initial Garnet planning attempt exposed a path-resolution mismatch: validation paths are manifest-relative while test paths are project-root-relative. Commit `630fb579da7af52bae18df8491ac24ac7ab6e6cb` fixed the planner to support both conventions and added a regression test. Local and Garnet test suites then passed 8/8.

The regenerated plan passed with 20 ready jobs. Independent token-level CSV review verified:

- exactly 20 unique jobs in topology order `G-0` through `G-19`;
- all 20 commands contain exactly one `--dry-run` and none contain `--execute`;
- all jobs are `ready` and `normal`;
- every command uses the locked 40/10/1000 split, seeds 42, `max_epochs=1500`, `metric_stride=1`, `early_stop_patience=20`, and `early_stop_min_delta=0.0001`;
- `weak_label_jobs.csv` SHA-256: `e845ff293b01ff52fa2078001443ac476bff900d7a0d4c56678661528f4e9ea0`.

The launcher preview reported `execute=false`, 20 normal jobs, 4 normal workers, 0 long jobs, and 1 reserved long worker. Post-preview checks found no `jobs/` directory, no job status files, and no paired-job manifests. No training has started. The next gate is an explicitly reviewed Garnet tmux launch with launcher-level `--execute`.

## Scripts

```text
scripts/
├── lock_and_audit_topology_bank.py
├── build_weak_label_artifacts.py
├── audit_weak_label_artifacts.py
├── plan_weak_label_jobs.py
├── launch_weak_label_jobs.py
└── review_weak_label_results.py
```

- `lock_and_audit_topology_bank.py` validates the 1000-row Step3 bank, all three graph hashes, template/source paths, and JSON metadata before writing a byte-stable locked manifest.
- `build_weak_label_artifacts.py` generates one shared test bank and one 40/10 fit bundle per topology. It supports `--limit` and repeated `--topology-id` selection for smoke runs.
- `audit_weak_label_artifacts.py` rejects missing files, wrong counts, hash drift, provenance drift, role overlap, and train/validation/test namespace errors.
- `plan_weak_label_jobs.py` produces `plans/weak_label_jobs.csv`. Every planned command contains `--dry-run` and never contains `--execute`.
- `launch_weak_label_jobs.py` converts reviewed dry-run commands only after an explicit launcher-level `--execute`. It supports bounded queues and resumes by skipping fully successful jobs.
- `review_weak_label_results.py` requires successful 2stage, SPO+, and evaluation outputs and writes the three canonical result summaries.

Unit tests live in `tests/test_step5_weak_label_scripts.py`.

## Safety and storage contract

- Run script development and tests locally.
- Materialize NPZ files and execute paired jobs only on Garnet under `/local1/fuweik/UdeM-Intern/...`.
- The planner is non-executing by construction.
- The launcher previews by default and requires `--execute` for mutations.
- For formal Garnet execution, pass `--require-hostname garnet` as an additional host safety check.
- Keep generated `data/`, `jobs/`, logs, model weights, and raw evaluation outputs out of Git.

## Intended Garnet workflow

Run these steps only after the scripts/configs have been reviewed and synchronized to Garnet.

From `/local1/fuweik/UdeM-Intern`:

```bash
source configs/runtime/garnet.env
```

Define the experiment path for readability:

```bash
STEP5_EXP=surrogate_experiment_results/Step5_topology_dfl_suitability/experiment_01_weak_label_seed42_sample50
```

First lock and audit the bank:

```bash
python "$STEP5_EXP/scripts/lock_and_audit_topology_bank.py"
```

Then build and audit the first 20 topologies using the locked label seed:

```bash
python "$STEP5_EXP/scripts/build_weak_label_artifacts.py" \
  --output-root "$STEP5_EXP/results/smoke20" \
  --config "$STEP5_EXP/configs/context_generator.locked.yaml" \
  --master-label-seed 20260719 \
  --limit 20

python "$STEP5_EXP/scripts/audit_weak_label_artifacts.py" \
  --output-root "$STEP5_EXP/results/smoke20" \
  --limit 20
```

Generate the smoke plan and preview the launcher:

```bash
python "$STEP5_EXP/scripts/plan_weak_label_jobs.py" \
  --output-root "$STEP5_EXP/results/smoke20" \
  --max-epochs 1500 \
  --metric-stride 1 \
  --early-stop-patience 20 \
  --early-stop-min-delta 0.0001 \
  --limit 20

python "$STEP5_EXP/scripts/launch_weak_label_jobs.py" \
  --jobs-csv "$STEP5_EXP/results/smoke20/plans/weak_label_jobs.csv" \
  --output-root "$STEP5_EXP/results/smoke20" \
  --expected-job-count 20 \
  --normal-workers 4 \
  --long-workers 1 \
  --require-hostname garnet
```

The last command is preview-only. Add `--execute` only inside the reviewed Garnet tmux run.

After completion:

```bash
python "$STEP5_EXP/scripts/review_weak_label_results.py" \
  --output-root "$STEP5_EXP/results/smoke20" \
  --max-epochs 1500 \
  --metric-stride 1 \
  --early-stop-patience 20 \
  --early-stop-min-delta 0.0001 \
  --limit 20
```

Smoke acceptance requires `passed=true`, `job_rows=20`, `success=20`, `label_rows=20`, and no failures in `results/weak_label_integrity_audit.json`. Only then remove `--limit 20`, generate the 1000-row formal plan, and use `--expected-job-count 1000` with the formal launcher.

## Canonical outputs

```text
results/weak_label_job_metrics.csv
results/weak_label_topology_summary.csv
results/weak_label_integrity_audit.json
```

The continuous label is:

```text
delta = test_gap_2stage - test_gap_spoplus
```

The weak class is `helpful` for `delta > 0.1`, `harmful` for `delta < -0.1`, and `near_neutral` otherwise. Every label row records the complete protocol and `weak_label=true`.
