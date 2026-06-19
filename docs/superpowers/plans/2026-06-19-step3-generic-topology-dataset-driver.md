# Step3 Generic Topology Dataset Driver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the pairs7-only Step3 generation entry point with a generic, reproducible topology dataset driver that supports explicit pair and NDD counts.

**Architecture:** Add a new generic driver that wraps the existing raw generator and Step2c processor without changing either lower-level tool. The driver converts `--num-ndds` to the generator's donor-share `--prob_ndd`, writes manifests/config/audit files under a subexperiment root, and keeps the existing pairs7 driver as a compatibility wrapper.

**Tech Stack:** Python stdlib, `unittest`, existing `0-data-generation.py`, existing Step2c `data-processing.py`, existing Step3 `build_topology_bank.py`.

---

### Task 1: Generic Driver API And Tests

**Files:**
- Create: `surrogate_experiment_results/Step3/scripts/generate_step3_topology_dataset.py`
- Create: `tests/test_step3_topology_dataset_driver.py`
- Keep compatible: `surrogate_experiment_results/Step3/scripts/generate_pairs7_step2c_dataset.py`

- [x] **Step 1: Write failing tests**

Create tests that assert:

```python
args = module.parse_args(["--pairs", "20", "--num-ndds", "2", "--instances", "100", "--seed", "20260619"])
self.assertEqual(args.pairs, 20)
self.assertEqual(args.num_ndds, 2)
self.assertAlmostEqual(module.prob_ndd_from_counts(20, 2), 2 / 22)
self.assertEqual(args.subexperiment, "pairs20_ndd2")
```

Also assert that generated commands include `--patients 20`, `--prob_ndd 0.0909091`, and output directories named `step3_pairs20_ndd2_*`.

- [x] **Step 2: Verify tests fail**

Run:

```bash
python -m unittest tests.test_step3_topology_dataset_driver -v
```

Expected: failure because `generate_step3_topology_dataset.py` does not exist yet.

- [x] **Step 3: Implement generic driver**

Add the new driver with:

```text
--pairs
--num-ndds
--instances
--seed
--label-seed
--subexperiment
--raw-output-dir
--processed-output-dir
--output-root
--force
--dry-run
```

Use `prob_ndd = num_ndds / (pairs + num_ndds)` and require positive pairs, non-negative NDDs, and at least one total donor node.

- [x] **Step 4: Keep pairs7 compatibility**

Leave `generate_pairs7_step2c_dataset.py` available as the existing pairs7-specific
compatibility entry point for this slice. Verify its existing tests still pass so
older commands do not break while new subexperiments move to the generic driver.

- [x] **Step 5: Verify tests pass**

Run:

```bash
python -m unittest tests.test_step3_topology_dataset_driver tests.test_step3_pairs7_generation_driver -v
```

Expected: all tests pass.

### Task 2: Pilot Data And Topology Bank

**Files:**
- Generated: `dataset/raw/step3_pairs20_ndd2_raw_seed20260619/`
- Generated: `dataset/processed/step3_pairs20_ndd2_step2c_poly_d8_mult_eps050_seed20260619/`
- Generated: `surrogate_experiment_results/Step3/pairs20_ndd2/`
- Modify: `surrogate_experiment_results/Step3/README.md`

- [x] **Step 1: Run generic driver**

```bash
python surrogate_experiment_results/Step3/scripts/generate_step3_topology_dataset.py \
  --pairs 20 \
  --num-ndds 2 \
  --instances 100 \
  --seed 20260619 \
  --label-seed 20260619 \
  --force
```

Expected: 100 processed graphs, each with 20 Pair vertices and 2 NDD vertices.

- [x] **Step 2: Build topology bank**

```bash
python surrogate_experiment_results/Step3/scripts/build_topology_bank.py \
  --source-dir dataset/processed/step3_pairs20_ndd2_step2c_poly_d8_mult_eps050_seed20260619 \
  --output-dir surrogate_experiment_results/Step3/pairs20_ndd2/data/topologies \
  --expected-pairs 20 \
  --force
```

Expected: accepted topology count is reported, with rejected count recorded if any graph has no candidates.

- [x] **Step 3: Summarize smoke metrics**

Compute rows, pair/NDD counts, arcs, cycle counts, chain counts, and exchange candidate counts from `topology_bank.csv`.

- [x] **Step 4: Update README**

Add `pairs20_ndd2` to the current Step3 bootstrap section with paths and the observed smoke metrics. State that this is a topology-generation pilot, not a training result.

- [x] **Step 5: Final verification**

Run:

```bash
python -m unittest tests.test_step3_topology_dataset_driver tests.test_step3_pairs7_generation_driver tests.test_step3_topology_bank -v
```

Expected: all targeted Step3 tests pass.
