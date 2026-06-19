# Step3 Pairs7 Data Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first Step3 data-generation slice: an isolated `pairs=7`, high-NDD Step2c dataset driver with audit output, without training models.

**Architecture:** Add one experiment-local Python driver under `surrogate_experiment_results/Step3/scripts/`. The driver wraps the existing root `0-data-generation.py` and Step2c processor, writes Step3-specific raw/processed directories, then audits processed `G-*.json` files for pair count, NDD count, total vertices, and arc count. Tests cover command construction, manifest/audit generation, and default experiment parameters without invoking the heavy generator.

**Tech Stack:** Python standard library, existing KEP generator, existing Step2c processor, `unittest`.

---

### Task 1: Tests for Pairs7 Driver Defaults and Audit

**Files:**
- Create: `tests/test_step3_pairs7_generation_driver.py`

- [ ] **Step 1: Write failing tests**

Add tests that import `surrogate_experiment_results/Step3/scripts/generate_pairs7_step2c_dataset.py` and verify:
- default config uses `patients=7`, `prob_ndd=0.20`, `step2c_degree=8`, `step2c_epsilon_bar=0.5`;
- generated command lists target the existing `0-data-generation.py` and Step2c processor;
- audit rejects processed graphs whose pair count is not 7;
- audit writes summary rows with pair/NDD/vertex/arc counts.

- [ ] **Step 2: Verify red**

Run:

```bash
python -m pytest tests/test_step3_pairs7_generation_driver.py -q
```

Expected: FAIL because the Step3 pairs7 driver module does not exist yet.

### Task 2: Implement Pairs7 Generation Driver

**Files:**
- Create: `surrogate_experiment_results/Step3/scripts/generate_pairs7_step2c_dataset.py`

- [ ] **Step 1: Implement minimal driver**

Add a Python CLI with:
- `--instances`;
- `--seed`;
- `--prob-ndd`, default `0.20`;
- `--raw-output-dir`;
- `--processed-output-dir`;
- `--dry-run`;
- `--force`;
- `--python`;
- Step2c options defaulting to degree 8, kappa 3, delta `1e-12`, epsilon `0.5`, label seed `20260619`.

The driver should:
- build and optionally run raw-generation command;
- build and optionally run Step2c processing command;
- audit processed graphs when not dry-run and write `audit_summary.json`;
- write `run_manifest.json` for both dry-run and real runs.

- [ ] **Step 2: Verify green**

Run:

```bash
python -m pytest tests/test_step3_pairs7_generation_driver.py -q
```

Expected: PASS.

### Task 3: Smoke the CLI

**Files:**
- No new files.

- [ ] **Step 1: Dry-run command**

Run:

```bash
python surrogate_experiment_results/Step3/scripts/generate_pairs7_step2c_dataset.py --instances 2 --dry-run
```

Expected: exit 0, manifest written under `surrogate_experiment_results/Step3/pairs7/`, and printed commands include `--patients 7` and `--prob_ndd 0.2`.

- [ ] **Step 2: Targeted tests**

Run:

```bash
python -m pytest tests/test_step3_pairs7_generation_driver.py -q
```

Expected: PASS.
