# Experiment Reproduction Guide

This repository now has one unified entrypoint for the main experiment lines:

```bash
./run_experiment.sh <command> [args...]
```

By default the script uses:

```bash
/home/weikang/miniconda3/envs/KEPs/bin/python
```

If your environment moves, override it with:

```bash
KEP_PYTHON=/path/to/python ./run_experiment.sh <command>
```

## 1. Experiment Lines

### A. Data Pipeline

Generate raw instances:

```bash
./run_experiment.sh data-generate --instances 1000 --patients 50
```

Process raw instances into unified pair/NDD graphs:

```bash
./run_experiment.sh data-process --all
```

Inputs:
- `dataset/raw/genjson-*.json`

Outputs:
- `dataset/processed/G-*.json`

Notes:
- `ground_truth_label` is now deterministic per edge, so re-processing the same raw file produces identical labels.

### B. Two-Stage Baselines

#### Two-Stage GNN

```bash
./run_experiment.sh 2stg-gnn
```

What it does:
1. Runs `2-stage1-training-GNN.py`
2. Finds the newest `results/2stg_Gnn_*/best_stage1_model_real.pth`
3. Runs `3-stage2-solver-gurobi.py --model_path ...`

Outputs:
- `results/2stg_Gnn_<timestamp>/`
- `solutions/2stg_Gnn_<timestamp>/`

#### Two-Stage MLP Regression

```bash
./run_experiment.sh 2stg-reg
```

Outputs:
- `results/2stg_Reg_<timestamp>/`
- `solutions/2stg_Reg_<timestamp>/`

Backward-compatible shortcuts still work:

```bash
./2stg_Gnn.sh
./2stg_Reg.sh
```

### C. Decision-Focused Learning (DFL)

#### DFL GNN

```bash
./run_experiment.sh dfl-gnn
```

Optional:

```bash
./run_experiment.sh dfl-gnn --pretrain_PATH results/2stg_Gnn_<timestamp>/best_stage1_model_real.pth
```

Outputs:
- `results/dfl_Gnn_<timestamp>/`
- `solutions/dfl_Gnn_<timestamp>/`

#### DFL MLP

```bash
./run_experiment.sh dfl-reg
```

Optional:

```bash
./run_experiment.sh dfl-reg --pretrain_PATH results/2stg_Reg_<timestamp>/best_stage1_model_real.pth
```

Outputs:
- `results/dfl_Reg_<timestamp>/`
- `solutions/dfl_Reg_<timestamp>/`

### D. Oracle Benchmark

Run oracle mode using ground-truth labels:

```bash
./run_experiment.sh oracle
```

Recommended for fair comparison against a specific experiment:

```bash
./run_experiment.sh oracle results/2stg_Gnn_<timestamp>/best_stage1_model_real.pth
```

Why pass a checkpoint:
- `3-stage2-solver-gurobi.py --gt_mode --model_path ...` copies that experiment's `test_files.txt`
- this lets `4-evaulation.py` compare all methods on the same test split

Outputs:
- `solutions/ground_truth/`
- `results/ground_truth/test_files.txt` if a reference checkpoint is supplied or auto-detected

### E. Evaluation

Compare all solution folders under `solutions/`:

```bash
./run_experiment.sh evaluate
```

Evaluate full datasets instead of test splits:

```bash
./run_experiment.sh evaluate --full_eval
```

Evaluate a custom subset:

```bash
./run_experiment.sh evaluate --test_list results/2stg_Gnn_<timestamp>/test_files.txt
```

### F. Visualization

Launch the Flask app:

```bash
./run_experiment.sh app
```

Pages:
- `/` explores processed graphs
- `/solutions` explores all `solutions/*/*_sol.json` files

## 2. Recommended Reproduction Order

For a clean comparison, run the lines in this order:

1. Data processing

```bash
./run_experiment.sh data-process --all
```

2. Two-stage baselines

```bash
./run_experiment.sh 2stg-gnn
./run_experiment.sh 2stg-reg
```

3. Decision-focused models

```bash
./run_experiment.sh dfl-gnn
./run_experiment.sh dfl-reg
```

4. Oracle benchmark

```bash
./run_experiment.sh oracle results/2stg_Gnn_<timestamp>/best_stage1_model_real.pth
```

5. Unified comparison

```bash
./run_experiment.sh evaluate
```

6. Manual inspection

```bash
./run_experiment.sh app
```

## 3. Output Map

### Processed Data
- `dataset/processed/G-*.json`

### Training Artifacts
- `results/2stg_Gnn_<timestamp>/best_stage1_model_real.pth`
- `results/2stg_Reg_<timestamp>/best_stage1_model_real.pth`
- `results/dfl_Gnn_<timestamp>/best_dfl_model.pth`
- `results/dfl_Reg_<timestamp>/best_dfl_reg_model.pth`

### Split Files
- `results/<experiment>/test_files.txt`

### Solver Outputs
- `solutions/<experiment>/G-*_sol.json`

### Comparison Entry
- `4-evaulation.py`

## 4. Sanity Checks

Preview commands without running them:

```bash
./run_experiment.sh --dry-run 2stg-gnn
./run_experiment.sh --dry-run oracle
```

If you want to reproduce a single exact experiment later, record:
- the command you ran
- the timestamped result directory name
- the corresponding solution directory
- the `test_files.txt` used for evaluation
