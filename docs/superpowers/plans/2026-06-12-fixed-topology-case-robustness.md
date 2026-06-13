# Fixed Topology Case Robustness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a bounded experiment that tests whether the G-392 and G-1560 Case C behavior is topology-conditioned, label/noise-realization-conditioned, or tied to one trained parameter realization.

**Architecture:** Add a fixed-topology relabel audit under `surrogate_experiment_results/decision_analysis`: keep the graph topology and trained Step2b d8 models fixed, regenerate only Step2c-style multiplicative labels over the same arcs, replay oracle/rank-1/rank-2 decisions, then summarize whether the Case C signature persists across label seeds. If the audit is promising, run a small consistent relabel+retrain extension on garnet so the final claim is not based only on out-of-training-regime evaluation.

**Tech Stack:** Python 3.10, NumPy, NetworkX optional for hashes, Matplotlib, existing Step1c Gurobi KEP loader, `surrogate_experiment_results/Step2/Step2c_polynomial_degree_multiplicative_noise/data-processing.py`, existing `compute_second_best_solutions.rows_for_model_record`.

---

## File Structure

- Create: `surrogate_experiment_results/decision_analysis/scripts/audit_fixed_topology_label_seed.py`
  - Relabels selected processed graph JSONs with Step2c multiplicative label seeds while preserving topology.
  - Loads the existing selected Step2b d8 2stage/SPO+ model weights for the graph's case seed.
  - Replays oracle, rank-1, and rank-2 decisions using existing KEP/Gurobi code.
  - Writes graph-level rows to `surrogate_experiment_results/decision_analysis/results/fixed_topology_label_seed/fixed_topology_label_seed_rows.csv`.

- Create: `surrogate_experiment_results/decision_analysis/scripts/summarize_fixed_topology_label_seed.py`
  - Converts replay rows into per-graph stability metrics: Case C preservation rate, SPO+ improvement rate, oracle-solution stability, rank-2 promotion stability, and topology/label hash checks.
  - Writes CSV and Markdown summaries under `surrogate_experiment_results/decision_analysis/results/fixed_topology_label_seed/`.

- Create: `surrogate_experiment_results/decision_analysis/scripts/plot_fixed_topology_label_seed.py`
  - Plots normalized gap across label seeds for G-392 and G-1560.
  - Plots per-graph oracle solution stability and Case C preservation.
  - Writes PNGs under `surrogate_experiment_results/decision_analysis/plots/fixed_topology_label_seed/`.

- Create: `tests/test_fixed_topology_label_seed.py`
  - Unit-tests deterministic relabeling, topology preservation, label-seed variation, and Case C signature logic without requiring Gurobi.

- Modify: `surrogate_experiment_results/decision_analysis/README.md`
  - Documents scope, commands, outputs, and report-safe interpretation.

- Optional later, no first-pass edit: `surrogate_experiment_results/Step2_resampling/run_phase1_subset_resampling.py`
  - Use existing `--label_seed` support for a consistent relabel+retrain extension only after the fixed-model audit says the effect is worth the compute.

---

### Task 1: Define Fixed-Topology Relabel Helpers

**Files:**
- Create: `tests/test_fixed_topology_label_seed.py`
- Create: `surrogate_experiment_results/decision_analysis/scripts/audit_fixed_topology_label_seed.py`

- [ ] **Step 1: Write failing unit tests for topology hash and relabel behavior**

Create `tests/test_fixed_topology_label_seed.py` with these tests:

```python
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "surrogate_experiment_results"
    / "decision_analysis"
    / "scripts"
    / "audit_fixed_topology_label_seed.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("audit_fixed_topology_label_seed", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample_payload():
    return {
        "metadata": {
            "source_file": "genjson-sample.json",
            "ground_truth_label_mode": "step2b_polynomial_degree_noiseless",
            "step2b_degree": 8,
            "step2b_kappa": 3.0,
            "step2b_delta": 1e-12,
            "clean_linear_utility_weight": 10.0,
            "clean_linear_cpra_weight": 5.0,
        },
        "data": {
            "1": {
                "type": "Pair",
                "matches": [
                    {"recipient": "2", "utility": 80, "recipient_cpra": 0.4},
                    {"recipient": "3", "utility": 40, "recipient_cpra": 0.2},
                ],
            },
            "2": {
                "type": "Pair",
                "matches": [{"recipient": "3", "utility": 70, "recipient_cpra": 0.1}],
            },
            "3": {"type": "Pair", "matches": []},
        },
    }


def test_topology_hash_ignores_labels():
    mod = load_module()
    payload = sample_payload()
    relabeled_a = mod.relabel_payload_step2c(payload, label_seed=1, epsilon_bar=0.5)
    relabeled_b = mod.relabel_payload_step2c(payload, label_seed=2, epsilon_bar=0.5)

    assert mod.topology_hash(relabeled_a) == mod.topology_hash(relabeled_b)
    assert mod.edge_count(relabeled_a) == 3


def test_label_seed_changes_labels_under_step2c():
    mod = load_module()
    payload = sample_payload()
    relabeled_a = mod.relabel_payload_step2c(payload, label_seed=1, epsilon_bar=0.5)
    relabeled_b = mod.relabel_payload_step2c(payload, label_seed=2, epsilon_bar=0.5)

    labels_a = mod.edge_labels(relabeled_a)
    labels_b = mod.edge_labels(relabeled_b)

    assert labels_a != labels_b
    assert all(value >= 0.0 for value in labels_a)
    assert all(value >= 0.0 for value in labels_b)


def test_case_c_signature_requires_large_2stage_gap_and_small_spoplus_gap():
    mod = load_module()
    rows = [
        {"method_label": "2stage_val_mse", "solution_rank": 1, "normalized_gap_to_oracle": 0.31},
        {"method_label": "spoplus_val_spoplus_loss", "solution_rank": 1, "normalized_gap_to_oracle": 0.01},
    ]

    assert mod.case_c_signature(rows, two_stage_min_gap=0.05, spoplus_max_gap=0.02)
```

- [ ] **Step 2: Run tests and verify the missing module failure**

Run:

```bash
python -m pytest tests/test_fixed_topology_label_seed.py -q
```

Expected: FAIL because `audit_fixed_topology_label_seed.py` does not exist.

- [ ] **Step 3: Implement the relabel helper functions**

Create `surrogate_experiment_results/decision_analysis/scripts/audit_fixed_topology_label_seed.py` with helper functions:

```python
#!/usr/bin/env python3
"""Audit whether Case C persists when labels vary on a fixed KEP topology."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
STEP2C_SCRIPT = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "Step2"
    / "Step2c_polynomial_degree_multiplicative_noise"
    / "data-processing.py"
)
DEFAULT_DATASET_DIR = (
    PROJECT_ROOT / "dataset" / "processed" / "step2b_poly_d8_main2000_seed20260523"
)
DEFAULT_CASE_INDEX = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "decision_analysis"
    / "results"
    / "case_studies"
    / "case_study_index.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "decision_analysis"
    / "results"
    / "fixed_topology_label_seed"
)


def load_step2c_module():
    spec = importlib.util.spec_from_file_location("step2c_data_processing", STEP2C_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def iter_edge_items(payload: dict[str, Any]):
    for source_id, node in payload.get("data", {}).items():
        for edge_idx, match in enumerate(node.get("matches", []) or []):
            yield str(source_id), int(edge_idx), match


def edge_count(payload: dict[str, Any]) -> int:
    return sum(1 for _ in iter_edge_items(payload))


def topology_edges(payload: dict[str, Any]) -> list[str]:
    edges = []
    for source_id, _, match in iter_edge_items(payload):
        edges.append(f"{source_id}->{match['recipient']}")
    return sorted(edges)


def topology_hash(payload: dict[str, Any]) -> str:
    material = "\n".join(topology_edges(payload)).encode("utf-8")
    return hashlib.sha256(material).hexdigest()[:16]


def edge_labels(payload: dict[str, Any]) -> list[float]:
    return [float(match.get("ground_truth_label", 0.0)) for _, _, match in iter_edge_items(payload)]


def edge_label_hash(payload: dict[str, Any]) -> str:
    material = "\n".join(f"{value:.10g}" for value in edge_labels(payload)).encode("utf-8")
    return hashlib.sha256(material).hexdigest()[:16]


def label_config_from_payload(payload: dict[str, Any], label_seed: int, epsilon_bar: float) -> dict[str, Any]:
    dp = load_step2c_module()
    metadata = payload.get("metadata", {})
    return {
        "label_mode": dp.LABEL_MODE_STEP2C_POLYNOMIAL_DEGREE_MULTIPLICATIVE_NOISE,
        "clean_linear_utility_weight": float(metadata.get("clean_linear_utility_weight", 10.0)),
        "clean_linear_cpra_weight": float(metadata.get("clean_linear_cpra_weight", 5.0)),
        "clean_linear_noise_sigma": float(metadata.get("clean_linear_noise_sigma", 0.08)),
        "step2c_degree": int(metadata.get("step2b_degree", metadata.get("step2c_degree", 8))),
        "step2c_kappa": float(metadata.get("step2b_kappa", metadata.get("step2c_kappa", 3.0))),
        "step2c_delta": float(metadata.get("step2b_delta", metadata.get("step2c_delta", 1e-12))),
        "step2c_epsilon_bar": float(epsilon_bar),
        "label_seed": int(label_seed),
    }


def graph_label_context(payload: dict[str, Any], label_config: dict[str, Any]) -> dict[str, Any]:
    dp = load_step2c_module()
    latent_values = []
    for _, _, match in iter_edge_items(payload):
        latent_values.append(
            dp.clean_linear_utility_cpra_label(
                match.get("utility", 0.0),
                match.get("recipient_cpra", 0.0),
                label_config,
            )
        )
    clean_mean = sum(latent_values) / len(latent_values) if latent_values else 0.0
    scores = [
        dp.step2c_polynomial_score(value, clean_mean, label_config)[1]
        for value in latent_values
    ]
    score_mean = sum(scores) / len(scores) if scores else 1.0
    return {
        "clean_linear_mean": clean_mean,
        "polynomial_score_mean": score_mean,
        "clean_linear_edge_count": len(latent_values),
    }


def stable_source_key(payload: dict[str, Any], source_id: str, edge_idx: int, match: dict[str, Any]) -> str:
    source_file = payload.get("metadata", {}).get("source_file", "processed_graph")
    return f"{source_file}|{source_id}|{edge_idx}|{match.get('recipient')}"


def relabel_payload_step2c(payload: dict[str, Any], label_seed: int, epsilon_bar: float = 0.5) -> dict[str, Any]:
    dp = load_step2c_module()
    output = copy.deepcopy(payload)
    config = label_config_from_payload(output, label_seed=label_seed, epsilon_bar=epsilon_bar)
    context = graph_label_context(output, config)
    for source_id, edge_idx, match in iter_edge_items(output):
        fields = dp.compute_ground_truth_label_fields(
            config,
            expected_transplant_count=match.get("expected_transplant_count", 0.0),
            qaly=match.get("qaly", 0.0),
            priority_multiplier=match.get("priority_multiplier", 1.0),
            source_key=stable_source_key(output, source_id, edge_idx, match),
            utility=match.get("utility", 0.0),
            cpra=match.get("recipient_cpra", 0.0),
            graph_label_context=context,
        )
        match.update(fields)
    output.setdefault("metadata", {})
    output["metadata"].update(
        {
            "ground_truth_label_mode": "fixed_topology_step2c_multiplicative_relabel",
            "fixed_topology_relabel_seed": int(label_seed),
            "fixed_topology_epsilon_bar": float(epsilon_bar),
            "fixed_topology_source_topology_hash": topology_hash(payload),
        }
    )
    return output


def case_c_signature(
    rows: list[dict[str, Any]],
    two_stage_min_gap: float = 0.05,
    spoplus_max_gap: float = 0.02,
    min_gap_reduction: float = 0.05,
) -> bool:
    rank1 = {
        row["method_label"]: float(row["normalized_gap_to_oracle"])
        for row in rows
        if int(float(row["solution_rank"])) == 1
    }
    two_stage_gap = rank1.get("2stage_val_mse")
    spoplus_gap = rank1.get("spoplus_val_spoplus_loss")
    if two_stage_gap is None or spoplus_gap is None:
        return False
    return (
        two_stage_gap >= two_stage_min_gap
        and spoplus_gap <= spoplus_max_gap
        and (two_stage_gap - spoplus_gap) >= min_gap_reduction
    )
```

- [ ] **Step 4: Run tests and verify helper behavior passes**

Run:

```bash
python -m pytest tests/test_fixed_topology_label_seed.py -q
```

Expected: PASS.

### Task 2: Add Gurobi Replay Over Relabeled Graphs

**Files:**
- Modify: `surrogate_experiment_results/decision_analysis/scripts/audit_fixed_topology_label_seed.py`

- [ ] **Step 1: Add replay imports and CSV fields**

Append imports near the top of the script:

```python
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compare_decisions_per_graph import (
    DEFAULT_RUN_ROOT,
    ensure_step1c_imports,
    load_models,
    method_label,
    resolve_run_dir,
)
from compute_second_best_solutions import (
    CSV_FIELDS as SECOND_BEST_FIELDS,
    DEFAULT_METHOD_LABELS,
    rows_for_model_record,
)
```

Add output fields:

```python
FIXED_TOPOLOGY_FIELDS = [
    "base_graph_id",
    "base_subset_seed",
    "label_seed",
    "epsilon_bar",
    "topology_hash",
    "label_hash",
    "topology_edge_count",
    "relabeled_graph_path",
    "case_c_signature_for_label_seed",
]

CSV_FIELDS = FIXED_TOPOLOGY_FIELDS + SECOND_BEST_FIELDS
```

- [ ] **Step 2: Add case-index lookup and relabeled graph writing**

Add functions:

```python
def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def case_seed_by_graph(case_index_path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv_rows(case_index_path)
    return {row["graph_id"]: row for row in rows}


def relabeled_graph_path(output_dir: Path, graph_id: str, label_seed: int) -> Path:
    stem = Path(graph_id).stem
    return output_dir / "graphs" / f"{stem}__fixed_topology_step2c_seed{int(label_seed)}.json"


def write_relabeled_graphs(args) -> list[dict[str, Any]]:
    output_rows = []
    for graph_id in args.graphs:
        base_path = args.dataset_dir / graph_id
        payload = read_json(base_path)
        base_hash = topology_hash(payload)
        for label_seed in args.label_seeds:
            relabeled = relabel_payload_step2c(
                payload,
                label_seed=int(label_seed),
                epsilon_bar=float(args.epsilon_bar),
            )
            if topology_hash(relabeled) != base_hash:
                raise AssertionError(f"Topology changed for {graph_id} label_seed={label_seed}")
            path = relabeled_graph_path(args.output_dir, graph_id, int(label_seed))
            write_json(path, relabeled)
            output_rows.append(
                {
                    "base_graph_id": graph_id,
                    "label_seed": int(label_seed),
                    "epsilon_bar": float(args.epsilon_bar),
                    "topology_hash": topology_hash(relabeled),
                    "label_hash": edge_label_hash(relabeled),
                    "topology_edge_count": edge_count(relabeled),
                    "relabeled_graph_path": path,
                }
            )
    return output_rows
```

- [ ] **Step 3: Add replay function using existing `rows_for_model_record`**

Add:

```python
def replay_relabel_rows(args) -> list[dict[str, Any]]:
    common, load_model_weight, _, _, _ = ensure_step1c_imports()
    import gurobipy as gp

    case_by_graph = case_seed_by_graph(args.case_index)
    relabeled_specs = write_relabeled_graphs(args)
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Seed", int(args.gurobi_seed))
    env.start()

    all_rows: list[dict[str, Any]] = []
    try:
        for spec in relabeled_specs:
            graph_id = spec["base_graph_id"]
            case = case_by_graph[graph_id]
            subset_seed = int(case["subset_seed"])
            run_dir = resolve_run_dir(args.run_root, args.regime, subset_seed)
            models = load_models(run_dir, load_model_weight)
            records = common.load_graph_records(
                [Path(spec["relabeled_graph_path"])],
                env,
                max_cycle=args.max_cycle,
                max_chain=args.max_chain,
            )
            try:
                record = records[0]
                label_seed_rows: list[dict[str, Any]] = []
                for model in models:
                    label = method_label(model)
                    if label not in set(args.method_labels):
                        continue
                    rows = rows_for_model_record(
                        args=args,
                        case=case,
                        model=model,
                        label=label,
                        record=record,
                    )
                    label_seed_rows.extend(rows)

                signature = case_c_signature(label_seed_rows)
                for row in label_seed_rows:
                    row.update(
                        {
                            "base_graph_id": graph_id,
                            "base_subset_seed": subset_seed,
                            "label_seed": int(spec["label_seed"]),
                            "epsilon_bar": float(spec["epsilon_bar"]),
                            "topology_hash": spec["topology_hash"],
                            "label_hash": spec["label_hash"],
                            "topology_edge_count": int(spec["topology_edge_count"]),
                            "relabeled_graph_path": str(spec["relabeled_graph_path"]),
                            "case_c_signature_for_label_seed": signature,
                        }
                    )
                all_rows.extend(label_seed_rows)
            finally:
                common.dispose_graph_records(records)
    finally:
        env.dispose()
    return all_rows
```

- [ ] **Step 4: Add CLI and main function**

Add:

```python
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Replay fixed-topology Step2c label-seed perturbations for selected Case C graphs."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--case-index", type=Path, default=DEFAULT_CASE_INDEX)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--regime", default="step2b_poly_d8")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "fixed_topology_label_seed_rows.csv",
    )
    parser.add_argument("--graphs", nargs="+", default=["G-392.json", "G-1560.json"])
    parser.add_argument("--label-seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--epsilon-bar", type=float, default=0.5)
    parser.add_argument("--max-cycle", type=int, default=3)
    parser.add_argument("--max-chain", type=int, default=4)
    parser.add_argument("--max-solutions", type=int, default=2)
    parser.add_argument("--max-cut-attempts", type=int, default=20)
    parser.add_argument("--method-labels", nargs="+", default=list(DEFAULT_METHOD_LABELS))
    parser.add_argument("--gurobi-seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--no-reset-before-solve", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    rows = replay_relabel_rows(args)
    write_csv(args.output, rows, CSV_FIELDS)
    print(f"Saved {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run the no-Gurobi unit tests again**

Run:

```bash
python -m pytest tests/test_fixed_topology_label_seed.py -q
```

Expected: PASS.

### Task 3: Smoke-Run Fixed-Topology Audit on Garnet

**Files:**
- No code edits.
- Output: `surrogate_experiment_results/decision_analysis/results/fixed_topology_label_seed/fixed_topology_label_seed_rows_smoke.csv`

- [ ] **Step 1: Sync code and tests to garnet**

Run from local repo root:

```bash
rsync -av \
  surrogate_experiment_results/decision_analysis/scripts/audit_fixed_topology_label_seed.py \
  tests/test_fixed_topology_label_seed.py \
  cirrelt:/local1/fuweik/UdeM-Intern/
```

Expected: rsync lists the two files.

- [ ] **Step 2: Run unit tests in the garnet runtime**

Run:

```bash
ssh cirrelt 'cd /local1/fuweik/UdeM-Intern && source configs/runtime/garnet.env && python -m pytest tests/test_fixed_topology_label_seed.py -q'
```

Expected: PASS.

- [ ] **Step 3: Run a two-label-seed smoke replay**

Run:

```bash
ssh cirrelt 'cd /local1/fuweik/UdeM-Intern && source configs/runtime/garnet.env && python surrogate_experiment_results/decision_analysis/scripts/audit_fixed_topology_label_seed.py --graphs G-392.json G-1560.json --label-seeds 0 1 --output surrogate_experiment_results/decision_analysis/results/fixed_topology_label_seed/fixed_topology_label_seed_rows_smoke.csv'
```

Expected:

```text
Saved 16 rows to surrogate_experiment_results/decision_analysis/results/fixed_topology_label_seed/fixed_topology_label_seed_rows_smoke.csv
```

Reason for 16 rows: 2 graphs x 2 label seeds x 2 methods x 2 solution ranks.

- [ ] **Step 4: Inspect smoke row counts**

Run:

```bash
ssh cirrelt 'cd /local1/fuweik/UdeM-Intern && source configs/runtime/garnet.env && python - <<'"'"'PY'"'"'
import csv
from pathlib import Path
p = Path("surrogate_experiment_results/decision_analysis/results/fixed_topology_label_seed/fixed_topology_label_seed_rows_smoke.csv")
rows = list(csv.DictReader(p.open()))
print(len(rows))
print(sorted({r["base_graph_id"] for r in rows}))
print(sorted({r["label_seed"] for r in rows}))
print(sorted({r["method_label"] for r in rows}))
print(sorted({r["solution_rank"] for r in rows}))
print(sorted({r["topology_hash"] for r in rows}))
print(sorted({r["label_hash"] for r in rows})[:5])
PY'
```

Expected:

```text
16
['G-1560.json', 'G-392.json']
['0', '1']
['2stage_val_mse', 'spoplus_val_spoplus_loss']
['1', '2']
```

The topology hash set should contain one hash per graph, while label hashes should vary by label seed.

### Task 4: Full Fixed-Topology Label-Seed Audit

**Files:**
- No code edits.
- Output: `surrogate_experiment_results/decision_analysis/results/fixed_topology_label_seed/fixed_topology_label_seed_rows.csv`

- [ ] **Step 1: Run 50 label seeds on G-392 and G-1560**

Run on garnet:

```bash
ssh cirrelt 'cd /local1/fuweik/UdeM-Intern && source configs/runtime/garnet.env && python surrogate_experiment_results/decision_analysis/scripts/audit_fixed_topology_label_seed.py --graphs G-392.json G-1560.json --label-seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 --output surrogate_experiment_results/decision_analysis/results/fixed_topology_label_seed/fixed_topology_label_seed_rows.csv'
```

Expected:

```text
Saved 400 rows to surrogate_experiment_results/decision_analysis/results/fixed_topology_label_seed/fixed_topology_label_seed_rows.csv
```

Reason for 400 rows: 2 graphs x 50 label seeds x 2 methods x 2 solution ranks.

- [ ] **Step 2: Copy outputs back locally**

Run:

```bash
rsync -av cirrelt:/local1/fuweik/UdeM-Intern/surrogate_experiment_results/decision_analysis/results/fixed_topology_label_seed/ surrogate_experiment_results/decision_analysis/results/fixed_topology_label_seed/
```

Expected: local directory contains `fixed_topology_label_seed_rows.csv` and the generated relabeled graph JSONs.

### Task 5: Summarize Persistence and Failure Modes

**Files:**
- Create: `surrogate_experiment_results/decision_analysis/scripts/summarize_fixed_topology_label_seed.py`
- Output:
  - `surrogate_experiment_results/decision_analysis/results/fixed_topology_label_seed/fixed_topology_label_seed_summary.csv`
  - `surrogate_experiment_results/decision_analysis/results/fixed_topology_label_seed/fixed_topology_label_seed_readout.md`

- [ ] **Step 1: Implement summary script**

Create `surrogate_experiment_results/decision_analysis/scripts/summarize_fixed_topology_label_seed.py` with:

```python
#!/usr/bin/env python3
"""Summarize fixed-topology label-seed replay rows."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DIR = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "decision_analysis"
    / "results"
    / "fixed_topology_label_seed"
)
DEFAULT_INPUT = DEFAULT_DIR / "fixed_topology_label_seed_rows.csv"
DEFAULT_SUMMARY = DEFAULT_DIR / "fixed_topology_label_seed_summary.csv"
DEFAULT_READOUT = DEFAULT_DIR / "fixed_topology_label_seed_readout.md"

SUMMARY_FIELDS = [
    "base_graph_id",
    "label_seed_count",
    "unique_topology_hashes",
    "unique_label_hashes",
    "case_c_preserved_rate",
    "mean_2stage_rank1_gap",
    "mean_spoplus_rank1_gap",
    "mean_rank1_gap_reduction",
    "median_rank1_gap_reduction",
    "spoplus_better_rate",
    "mean_2stage_rank2_gap",
    "mean_spoplus_rank2_gap",
    "oracle_solution_unique_count",
    "mean_oracle_jaccard_to_first_seed",
]


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite_mean(values: list[float]) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(clean) / len(clean)) if clean else float("nan")


def finite_median(values: list[float]) -> float:
    clean = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not clean:
        return float("nan")
    mid = len(clean) // 2
    return clean[mid] if len(clean) % 2 else 0.5 * (clean[mid - 1] + clean[mid])


def edge_set(signature: str) -> set[str]:
    return set(token for token in str(signature).split("|") if token)


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def rows_by_graph_seed(rows: list[dict[str, str]]):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["base_graph_id"], row["label_seed"])].append(row)
    return grouped


def summarize(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped = rows_by_graph_seed(rows)
    by_graph = defaultdict(list)
    for (graph_id, _), seed_rows in grouped.items():
        by_graph[graph_id].append(seed_rows)

    summaries = []
    for graph_id, seed_groups in sorted(by_graph.items()):
        two_stage_rank1 = []
        spoplus_rank1 = []
        two_stage_rank2 = []
        spoplus_rank2 = []
        reductions = []
        case_flags = []
        spoplus_better = []
        topology_hashes = set()
        label_hashes = set()
        oracle_signatures = []

        for seed_rows in seed_groups:
            topology_hashes.update(row["topology_hash"] for row in seed_rows)
            label_hashes.update(row["label_hash"] for row in seed_rows)
            oracle_signatures.append(seed_rows[0].get("oracle_arc_key_signature", ""))
            rank_lookup = {
                (row["method_label"], int(float(row["solution_rank"]))): float(row["normalized_gap_to_oracle"])
                for row in seed_rows
            }
            ts1 = rank_lookup.get(("2stage_val_mse", 1), float("nan"))
            sp1 = rank_lookup.get(("spoplus_val_spoplus_loss", 1), float("nan"))
            ts2 = rank_lookup.get(("2stage_val_mse", 2), float("nan"))
            sp2 = rank_lookup.get(("spoplus_val_spoplus_loss", 2), float("nan"))
            two_stage_rank1.append(ts1)
            spoplus_rank1.append(sp1)
            two_stage_rank2.append(ts2)
            spoplus_rank2.append(sp2)
            reductions.append(ts1 - sp1)
            case_flags.append(str(seed_rows[0].get("case_c_signature_for_label_seed", "")).lower() == "true")
            spoplus_better.append(sp1 < ts1)

        first_oracle = edge_set(oracle_signatures[0]) if oracle_signatures else set()
        oracle_jaccards = [jaccard(edge_set(sig), first_oracle) for sig in oracle_signatures]
        summaries.append(
            {
                "base_graph_id": graph_id,
                "label_seed_count": len(seed_groups),
                "unique_topology_hashes": len(topology_hashes),
                "unique_label_hashes": len(label_hashes),
                "case_c_preserved_rate": finite_mean([float(v) for v in case_flags]),
                "mean_2stage_rank1_gap": finite_mean(two_stage_rank1),
                "mean_spoplus_rank1_gap": finite_mean(spoplus_rank1),
                "mean_rank1_gap_reduction": finite_mean(reductions),
                "median_rank1_gap_reduction": finite_median(reductions),
                "spoplus_better_rate": finite_mean([float(v) for v in spoplus_better]),
                "mean_2stage_rank2_gap": finite_mean(two_stage_rank2),
                "mean_spoplus_rank2_gap": finite_mean(spoplus_rank2),
                "oracle_solution_unique_count": len(set(oracle_signatures)),
                "mean_oracle_jaccard_to_first_seed": finite_mean(oracle_jaccards),
            }
        )
    return summaries


def write_readout(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    lines = ["# Fixed-Topology Label-Seed Robustness", ""]
    lines.append("Interpretation rule:")
    lines.append("- high Case C preservation rate supports a topology-conditioned mechanism;")
    lines.append("- low preservation rate means the original case is sensitive to the label realization or trained theta;")
    lines.append("- unique_topology_hashes must equal 1 for each graph, otherwise the audit is invalid.")
    lines.append("")
    for row in summary_rows:
        lines.append(f"## {row['base_graph_id']}")
        lines.append(f"- label seeds: {row['label_seed_count']}")
        lines.append(f"- unique topology hashes: {row['unique_topology_hashes']}")
        lines.append(f"- unique label hashes: {row['unique_label_hashes']}")
        lines.append(f"- Case C preserved rate: {float(row['case_c_preserved_rate']):.3f}")
        lines.append(f"- SPO+ better rate: {float(row['spoplus_better_rate']):.3f}")
        lines.append(f"- mean rank-1 gap reduction: {float(row['mean_rank1_gap_reduction']):.4f}")
        lines.append(f"- oracle unique solutions: {row['oracle_solution_unique_count']}")
        lines.append(f"- mean oracle Jaccard to first seed: {float(row['mean_oracle_jaccard_to_first_seed']):.3f}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Summarize fixed-topology label-seed audit.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--readout-output", type=Path, default=DEFAULT_READOUT)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    rows = read_csv_rows(args.input)
    summary_rows = summarize(rows)
    write_csv(args.summary_output, summary_rows, SUMMARY_FIELDS)
    write_readout(args.readout_output, summary_rows)
    print(f"Saved {len(summary_rows)} summary rows to {args.summary_output}")
    print(f"Saved readout to {args.readout_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run the summary**

Run:

```bash
python surrogate_experiment_results/decision_analysis/scripts/summarize_fixed_topology_label_seed.py
```

Expected:

```text
Saved 2 summary rows to surrogate_experiment_results/decision_analysis/results/fixed_topology_label_seed/fixed_topology_label_seed_summary.csv
Saved readout to surrogate_experiment_results/decision_analysis/results/fixed_topology_label_seed/fixed_topology_label_seed_readout.md
```

- [ ] **Step 3: Inspect the decision result**

Run:

```bash
column -s, -t surrogate_experiment_results/decision_analysis/results/fixed_topology_label_seed/fixed_topology_label_seed_summary.csv | sed -n '1,20p'
```

Expected:

```text
base_graph_id  label_seed_count  unique_topology_hashes  unique_label_hashes  case_c_preserved_rate
G-1560.json    50                1                       50                   ...
G-392.json     50                1                       50                   ...
```

### Task 6: Plot Fixed-Topology Robustness

**Files:**
- Create: `surrogate_experiment_results/decision_analysis/scripts/plot_fixed_topology_label_seed.py`
- Output:
  - `surrogate_experiment_results/decision_analysis/plots/fixed_topology_label_seed/fixed_topology_label_seed_gaps.png`
  - `surrogate_experiment_results/decision_analysis/plots/fixed_topology_label_seed/fixed_topology_label_seed_summary.png`

- [ ] **Step 1: Implement plotting script**

Create the plotting script with Matplotlib:

```python
#!/usr/bin/env python3
"""Plot fixed-topology label-seed robustness results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DIR = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "decision_analysis"
    / "results"
    / "fixed_topology_label_seed"
)
DEFAULT_PLOT_DIR = (
    PROJECT_ROOT
    / "surrogate_experiment_results"
    / "decision_analysis"
    / "plots"
    / "fixed_topology_label_seed"
)
DEFAULT_INPUT = DEFAULT_DIR / "fixed_topology_label_seed_rows.csv"
DEFAULT_SUMMARY = DEFAULT_DIR / "fixed_topology_label_seed_summary.csv"


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_gaps(rows: list[dict[str, str]], output: Path) -> None:
    rank1 = [row for row in rows if int(float(row["solution_rank"])) == 1]
    graph_ids = sorted({row["base_graph_id"] for row in rank1})
    fig, axes = plt.subplots(len(graph_ids), 1, figsize=(10, 4 * len(graph_ids)), sharex=False)
    if len(graph_ids) == 1:
        axes = [axes]
    colors = {"2stage_val_mse": "#d62728", "spoplus_val_spoplus_loss": "#1f77b4"}
    for ax, graph_id in zip(axes, graph_ids):
        graph_rows = [row for row in rank1 if row["base_graph_id"] == graph_id]
        for method in ("2stage_val_mse", "spoplus_val_spoplus_loss"):
            method_rows = sorted(
                [row for row in graph_rows if row["method_label"] == method],
                key=lambda row: int(row["label_seed"]),
            )
            ax.plot(
                [int(row["label_seed"]) for row in method_rows],
                [float(row["normalized_gap_to_oracle"]) for row in method_rows],
                marker="o",
                linewidth=1.5,
                label=method,
                color=colors[method],
            )
        ax.axhline(0.02, color="#777777", linestyle="--", linewidth=1)
        ax.axhline(0.05, color="#777777", linestyle=":", linewidth=1)
        ax.set_title(graph_id)
        ax.set_ylabel("rank-1 normalized gap")
        ax.legend()
    axes[-1].set_xlabel("label_seed")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_summary(summary_rows: list[dict[str, str]], output: Path) -> None:
    graphs = [row["base_graph_id"] for row in summary_rows]
    case_rates = [float(row["case_c_preserved_rate"]) for row in summary_rows]
    better_rates = [float(row["spoplus_better_rate"]) for row in summary_rows]
    oracle_jaccard = [float(row["mean_oracle_jaccard_to_first_seed"]) for row in summary_rows]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(graphs))
    width = 0.25
    ax.bar([i - width for i in x], case_rates, width=width, label="Case C preserved")
    ax.bar(list(x), better_rates, width=width, label="SPO+ better")
    ax.bar([i + width for i in x], oracle_jaccard, width=width, label="oracle Jaccard")
    ax.set_xticks(list(x), graphs)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rate")
    ax.legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Plot fixed-topology label-seed robustness.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    rows = read_csv_rows(args.input)
    summary_rows = read_csv_rows(args.summary)
    plot_gaps(rows, args.plot_dir / "fixed_topology_label_seed_gaps.png")
    plot_summary(summary_rows, args.plot_dir / "fixed_topology_label_seed_summary.png")
    print(f"Saved plots to {args.plot_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run plotting**

Run:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache python surrogate_experiment_results/decision_analysis/scripts/plot_fixed_topology_label_seed.py
```

Expected:

```text
Saved plots to surrogate_experiment_results/decision_analysis/plots/fixed_topology_label_seed
```

### Task 7: Add README Interpretation

**Files:**
- Modify: `surrogate_experiment_results/decision_analysis/README.md`

- [ ] **Step 1: Add a fixed-topology section**

Append this section:

```markdown
## Fixed-Topology Label-Seed Robustness

Purpose: test whether the Case C behavior for `G-392.json` and `G-1560.json`
is primarily supported by graph topology or by one specific label realization.

Important scope note: Step2b d8 is noiseless after the graph pool is fixed, so
changing `label_seed` in Step2b is a no-op for labels. The robustness audit
therefore uses Step2c-style deterministic multiplicative label perturbations
on the same fixed topology. This isolates label/noise-seed sensitivity without
changing the graph arcs.

Primary command:

```bash
python surrogate_experiment_results/decision_analysis/scripts/audit_fixed_topology_label_seed.py \
  --graphs G-392.json G-1560.json \
  --label-seeds 0 1 2 3 4 5 6 7 8 9 \
  --output surrogate_experiment_results/decision_analysis/results/fixed_topology_label_seed/fixed_topology_label_seed_rows.csv
```

For the full audit, use label seeds `0..49`.

Interpretation:

- If `unique_topology_hashes = 1` and `unique_label_hashes` equals the number
  of label seeds, the audit changed labels without changing topology.
- A high `case_c_preserved_rate` supports a topology-conditioned mechanism:
  this graph tends to expose a region where 2stage makes a decision-relevant
  error and SPO+ repairs it.
- A low `case_c_preserved_rate` means the original single-graph case depends
  strongly on the label realization or on the already-trained Step2b model
  parameters; it should be framed as an illustrative case, not a topology-
  universal property.
- This fixed-model audit is diagnostic. A paper-level robustness claim needs
  the optional consistent relabel+retrain check below.
```

- [ ] **Step 2: Verify README commands are copy-paste valid**

Run:

```bash
rg -n "Fixed-Topology Label-Seed Robustness|case_c_preserved_rate|audit_fixed_topology_label_seed" surrogate_experiment_results/decision_analysis/README.md
```

Expected: all three phrases appear.

### Task 8: Optional Strong Validation With Consistent Relabel + Retrain

**Files:**
- Prefer no new code unless the fixed-model audit preserves the Case C signature.
- Use existing:
  - `surrogate_experiment_results/Step2/run_generate_step2abc_datasets.sh`
  - `surrogate_experiment_results/Step2_resampling/run_phase1_subset_resampling.py`
  - `surrogate_experiment_results/decision_analysis/scripts/compare_decisions_per_graph.py`

- [ ] **Step 1: Choose five label seeds from the audit**

Select:

```text
two seeds where Case C persists strongly,
two seeds where Case C weakens,
one middle seed.
```

Record them in:

```text
surrogate_experiment_results/decision_analysis/results/fixed_topology_label_seed/selected_retrain_label_seeds.txt
```

- [ ] **Step 2: Generate Step2c datasets for those label seeds**

The existing Step2 generation driver does not currently expose block/degree filters. For the retrain extension, use it as-is for each selected label seed; it will generate Step2a, Step2b, and Step2c across degrees `{1,2,4,8}`. That is extra I/O, but it avoids adding driver semantics in the same experiment step.

```bash
DRY_RUN=1 LABEL_SEED=0 STEP2_EPSILON_BAR=0.5 \
  bash surrogate_experiment_results/Step2/run_generate_step2abc_datasets.sh
```

Expected: commands target directories named like:

```text
dataset/processed/step2c_poly_d8_mult_eps050_main2000_seed0
dataset/processed/step2c_poly_d8_mult_eps050_val2000_seed0
dataset/processed/step2c_poly_d8_mult_eps050_unseen10000_seed0
```

Run without `DRY_RUN=1` only after confirming the paths. Repeat the same command with each selected concrete seed, replacing `LABEL_SEED=0` with that seed.

- [ ] **Step 3: Run small retraining on garnet**

For each selected label seed:

```bash
ssh cirrelt 'cd /local1/fuweik/UdeM-Intern && source configs/runtime/garnet.env && ./run_experiment.sh --dry-run 2stg-dfl-lr --train_size 50'
```

Then run the existing Phase 1 resampling driver with:

```bash
ssh cirrelt 'cd /local1/fuweik/UdeM-Intern && source configs/runtime/garnet.env && python surrogate_experiment_results/Step2_resampling/run_phase1_subset_resampling.py --regimes step2c_poly_d8_mult_eps050 --label_seed 0 --subset-seeds 1 30 --train-size 50'
```

Expected: one run for subset seed 1 and one run for subset seed 30 under `LABEL_SEED=0`. Repeat with the other selected concrete label seeds.

- [ ] **Step 4: Replay G-392 and G-1560 only**

Run:

```bash
ssh cirrelt 'cd /local1/fuweik/UdeM-Intern && source configs/runtime/garnet.env && python surrogate_experiment_results/decision_analysis/scripts/compare_decisions_per_graph.py --regime step2c_poly_d8_mult_eps050 --graphs G-392.json G-1560.json --no-require-existing-match --output surrogate_experiment_results/decision_analysis/results/fixed_topology_label_seed/retrain_replay_label_seed_0.csv'
```

Expected: rows for both graphs and both methods under `LABEL_SEED=0`. Use these only after confirming that `G-392.json` and `G-1560.json` in the generated Step2c dataset preserve the same topology hash as the Step2b base graph. Repeat with the other selected concrete label seeds and name each output with the actual seed value, for example `retrain_replay_label_seed_17.csv`.

---

## Decision Rule

Report the result using this matrix:

```text
Fixed-model relabel audit:
  Case C persists across most label seeds
  -> topology-conditioned mechanism is plausible, but trained-model mismatch remains.

Fixed-model relabel audit:
  Case C disappears across most label seeds
  -> original case is label/parameter-realization sensitive; do not claim topology alone.

Consistent relabel+retrain:
  Case C also persists after retraining on relabeled datasets
  -> strongest evidence that this topology repeatedly exposes the 2stage-vs-SPO+ mechanism.

Consistent relabel+retrain:
  Case C does not persist after retraining
  -> frame G-392/G-1560 as illustrative single-graph cases, not topology-universal cases.
```

## Self-Review

- Spec coverage: The plan answers the advisor's question by fixing topology, changing label/noise seed, and separating fixed-model diagnostics from consistent retraining.
- Placeholder scan: No task uses TBD or unspecified paths; all commands and outputs are concrete.
- Type consistency: The row fields used by summary and plotting scripts are produced by the audit script or existing `compute_second_best_solutions.rows_for_model_record`.
