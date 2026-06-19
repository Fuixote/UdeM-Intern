# Step3 Topology Bank Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Step3 `pairs=7` topology templates from processed `G-*.json` files and compute stable topology, arc-order, and feasible-set hashes.

**Architecture:** Add one experiment-local script under `surrogate_experiment_results/Step3/scripts/`. The script reads processed Pair/NDD graph JSON, exports immutable templates under `surrogate_experiment_results/Step3/pairs7/data/topologies/`, writes `topology_bank.csv` and `topology_hashes.csv`, and rejects graphs that do not match the expected pair count or have too few feasible candidates. Candidate enumeration mirrors the existing `model.graph_utils.find_all_cycles_and_chains` semantics without importing torch.

**Tech Stack:** Python standard library, `unittest`, existing processed KEP JSON schema.

---

### Task 1: Tests for Template and Hash Semantics

**Files:**
- Create: `tests/test_step3_topology_bank.py`

- [ ] **Step 1: Write failing tests**

Tests should import `surrogate_experiment_results/Step3/scripts/build_topology_bank.py` and verify:
- templates contain `topology_id`, vertex list, canonical arc list, feasible candidates, and hashes;
- changing only labels leaves `topology_hash` and `feasible_set_hash` unchanged;
- adding an arc changes `topology_hash`;
- changing `max_chain` changes `feasible_set_hash`;
- `build_topology_bank()` writes per-topology `template.json`, `topology_bank.csv`, `topology_hashes.csv`, and `rejected_topologies.csv`.

- [ ] **Step 2: Verify red**

Run:

```bash
python -m unittest tests.test_step3_topology_bank -v
```

Expected: FAIL because `build_topology_bank.py` does not exist yet.

### Task 2: Implement Topology Bank Builder

**Files:**
- Create: `surrogate_experiment_results/Step3/scripts/build_topology_bank.py`

- [ ] **Step 1: Implement minimal builder**

Add a Python CLI with:
- `--source-dir`;
- `--output-dir`;
- `--max-cycle`, default `3`;
- `--max-chain`, default `4`;
- `--expected-pairs`, default `7`;
- `--min-candidates`, default `1`;
- `--force`.

The script should:
- parse processed `G-*.json`;
- canonicalize vertices and arcs;
- enumerate feasible cycles/chains;
- compute SHA-256 hashes over canonical JSON payloads;
- write one `template.json` per accepted graph;
- write accepted and rejected CSV summaries.

- [ ] **Step 2: Verify green**

Run:

```bash
python -m unittest tests.test_step3_topology_bank -v
```

Expected: PASS.

### Task 3: Build Smoke Topology Bank

**Files:**
- Generated local outputs under `surrogate_experiment_results/Step3/pairs7/data/topologies/`.

- [ ] **Step 1: Run builder on pairs7 smoke data**

Run:

```bash
python surrogate_experiment_results/Step3/scripts/build_topology_bank.py \
  --source-dir dataset/processed/step3_pairs7_step2c_poly_d8_mult_eps050_seed20260619 \
  --output-dir surrogate_experiment_results/Step3/pairs7/data/topologies \
  --force
```

Expected: exits 0 and writes topology templates plus CSV summaries.
