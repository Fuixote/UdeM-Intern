# 实验复现说明

本仓库现在统一通过一个入口脚本管理主要实验流程：

```bash
./run_experiment.sh <command> [args...]
```

不再推荐使用旧的 `2stg_Gnn.sh`、`2stg_Reg.sh`。两者已经删除，后续请统一使用 `run_experiment.sh`。

## 1. 运行环境

默认 Python 解释器为：

```bash
/home/weikang/miniconda3/envs/KEPs/bin/python
```

如果你的环境路径不同，可以临时覆盖：

```bash
KEP_PYTHON=/path/to/python ./run_experiment.sh <command>
```

## 2. 统一配置入口

实验中涉及的数据目录、结果目录、解目录，已经统一收敛到 [`experiment_config.py`](/home/weikang/projects/UdeM-Intern/Exps/experiment_config.py)。

默认目录可以通过环境变量覆盖，而不需要改源码：

```bash
export KEP_DATA_DIR=/path/to/processed-data
export KEP_RESULTS_DIR=/path/to/results
export KEP_SOLUTIONS_DIR=/path/to/solutions
```

大多数训练、求解、评估脚本也支持显式命令行参数：

```bash
--data_dir
--results_root
--solutions_root
```

## 3. 实验线路

### 3.1 数据生成

生成原始实例：

```bash
./run_experiment.sh data-generate --instances 1000 --patients 50 --seed 42
```

输入：
- 生成器配置参数

输出：
- 默认输出到独立目录：`dataset/raw/<YYYY-MM-DD_HHMMSS>/`
- 如果传 `--run_name my_run`，目录名会变成：`dataset/raw/<YYYY-MM-DD_HHMMSS>__my_run/`
- 目录内包含：
  - `genjson-*.json`
  - `config.json`
  - `effective_config.json`
  - `run_info.json`
  - `batch_summary.json`
  - `batch_report.md`

可选参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--instances` | `1000` | 生成的图实例数量 |
| `--patients` | `50` | 每个实例中的 patient / pair 数量 |
| `--prob_ndd` | `0.05` | NDD（Non-Directed Donor / altruistic donor）比例 |
| `--prob_o` | `0.4` | patient 血型 O 的概率 |
| `--prob_a` | `0.4` | patient 血型 A 的概率 |
| `--prob_b` | `0.1` | patient 血型 B 的概率 |
| `--donor_prob_o` | `0.4` | donor 血型 O 的概率 |
| `--donor_prob_a` | `0.4` | donor 血型 A 的概率 |
| `--donor_prob_b` | `0.1` | donor 血型 B 的概率 |
| `--donors1` | `1.0` | 只有 1 个 donor 的 patient 比例 |
| `--donors2` | `0.0` | 有 2 个 donors 的 patient 比例 |
| `--donors3` | `0.0` | 有 3 个 donors 的 patient 比例 |
| `--prob_spousal` | `0.0` | donor 为配偶的概率 |
| `--prob_female` | `0.0` | donor 为女性的概率 |
| `--prob_spousal_pra_compat` | `0.0` | 配偶 donor 的 PRA compatibility 概率 |
| `--seed` | `42` | 随机种子；相同参数和相同种子会生成一致的数据 |
| `--output_root` | `dataset/raw` | 输出根目录；会在其下创建带时间戳的批次目录 |
| `--run_name` | 无 | 追加在时间戳目录后面的标签，例如生成 `2026-03-30_124157__my_run` |
| `--output_dir` | 无 | 指定批次目录的父目录；若路径本身已是时间戳批次目录，则直接使用 |
| `--force` | 关闭 | 若目标输出目录已存在且非空，允许覆盖 |
| `--no_tune` | 关闭 | 关闭 tuning；默认启用 tuning |
| `--split_donor_blood` | 关闭 | 根据 recipient 血型使用不同 donor 血型分布 |

常见示例：

```bash
./run_experiment.sh data-generate --seed 42 --run_name my_run
./run_experiment.sh data-generate --output_dir /data/raw_batches --force
./run_experiment.sh data-generate --instances 2000 --patients 100 --prob_ndd 0.1
```

说明：
- 同一组参数加同一个 `--seed`，会生成一致的数据。
- 每次生成都会落到一个带日期时间的批次目录里，避免不同实验的数据混在一起。
- 如果你想覆盖一个已有输出目录，需要显式加 `--force`。
- `config.json` 保存请求参数；`effective_config.json` 保存 `tuning` 后真正用于采样的配置。
- `batch_summary.json` 和 `batch_report.md` 会汇总该批 raw 数据的结构统计、分布偏差、参数快照和告警，避免盲用数据。

### 3.2 数据预处理

将原始 donor-based 图转换成统一的 pair / NDD 图：

```bash
./run_experiment.sh data-process
./run_experiment.sh data-process dataset/raw/<YYYY-MM-DD_HHMMSS> dataset/processed --all
```

输入：
- 默认无参时：扫描 `dataset/raw/<batch_name>/genjson-*.json`
- 显式模式：处理指定的 `dataset/raw/<batch_name>/genjson-*.json`

输出：
- `dataset/processed/<batch_name>/G-*.json`
- `dataset/processed/<batch_name>/run_info.json`
- `dataset/processed/<batch_name>/batch_summary.json`
- `dataset/processed/<batch_name>/batch_report.md`

说明：
- `ground_truth_label` 现在是“按边确定性生成”的，同一份原始文件重复处理会得到一致结果。
- 不显式指定 raw 批次目录时，`1-data-processing.py` 会扫描 `dataset/raw` 下所有合法批次目录，并只补齐尚未完整处理的批次。
- processed 批次目录默认与 raw 批次目录同名，便于追溯。
- 如果同名 processed 批次已经完整存在，则自动跳过；如果缺少部分 `G-*.json` 或说明文件，则只补齐缺失产物并重写 batch 元数据。
- 如果你想强制重建某个批次，可以显式指定该 raw 批次目录并追加 `--force`。

### 3.3 两阶段基线：GNN

```bash
KEP_DATA_DIR=dataset/processed/<batch_name> ./run_experiment.sh 2stg-gnn
```

执行内容：
1. 运行 `2-stage1-training-GNN.py`
2. 自动寻找最新的 `results/2stg_Gnn_*/best_stage1_model_real.pth`
3. 调用 `3-stage2-solver-gurobi.py` 进行第二阶段求解

输出：
- `results/2stg_Gnn_<timestamp>/`
- `solutions/2stg_Gnn_<timestamp>/`

### 3.4 两阶段基线：MLP 回归

```bash
KEP_DATA_DIR=dataset/processed/<batch_name> ./run_experiment.sh 2stg-reg
```

输出：
- `results/2stg_Reg_<timestamp>/`
- `solutions/2stg_Reg_<timestamp>/`

### 3.5 端到端 DFL：GNN

```bash
KEP_DATA_DIR=dataset/processed/<batch_name> ./run_experiment.sh dfl-gnn
```

如果想显式指定预训练模型：

```bash
KEP_DATA_DIR=dataset/processed/<batch_name> ./run_experiment.sh dfl-gnn --pretrain_PATH results/2stg_Gnn_<timestamp>/best_stage1_model_real.pth
```

输出：
- `results/dfl_Gnn_<timestamp>/`
- `solutions/dfl_Gnn_<timestamp>/`

### 3.6 端到端 DFL：MLP

```bash
KEP_DATA_DIR=dataset/processed/<batch_name> ./run_experiment.sh dfl-reg
```

如果想显式指定预训练模型：

```bash
KEP_DATA_DIR=dataset/processed/<batch_name> ./run_experiment.sh dfl-reg --pretrain_PATH results/2stg_Reg_<timestamp>/best_stage1_model_real.pth
```

输出：
- `results/dfl_Reg_<timestamp>/`
- `solutions/dfl_Reg_<timestamp>/`

### 3.7 Oracle 基线

使用真实 `ground_truth_label` 直接求解：

```bash
KEP_DATA_DIR=dataset/processed/<batch_name> ./run_experiment.sh oracle
```

为了和某个实验使用同一测试集，推荐传入参考 checkpoint：

```bash
KEP_DATA_DIR=dataset/processed/<batch_name> ./run_experiment.sh oracle results/2stg_Gnn_<timestamp>/best_stage1_model_real.pth
```

原因：
- `3-stage2-solver-gurobi.py --gt_mode --model_path ...` 会复制该实验的 `test_files.txt`
- 这样 `4-evaulation.py` 在横向比较时会使用同一个测试集

输出：
- `solutions/ground_truth/`
- `results/ground_truth/test_files.txt`

### 3.8 评估

对 `solutions/` 下所有实验目录做统一比较：

```bash
KEP_DATA_DIR=dataset/processed/<batch_name> ./run_experiment.sh evaluate
```

如果要对全量图评估而不是仅测试集：

```bash
KEP_DATA_DIR=dataset/processed/<batch_name> ./run_experiment.sh evaluate --full_eval
```

如果要指定某个测试集文件：

```bash
KEP_DATA_DIR=dataset/processed/<batch_name> ./run_experiment.sh evaluate --test_list results/2stg_Gnn_<timestamp>/test_files.txt
```

### 3.9 可视化

启动 Flask 可视化服务：

```bash
./run_experiment.sh app
```

页面说明：
- `/`：查看处理后的图结构
- `/solutions`：查看 `solutions/*/*_sol.json` 中的求解结果

## 4. 推荐复现顺序

如果你想完整复现实验，建议按这个顺序执行：

1. 预处理数据

```bash
./run_experiment.sh data-generate --instances 1000 --patients 50 --seed 42
./run_experiment.sh data-process
```

2. 运行两阶段基线

```bash
KEP_DATA_DIR=dataset/processed/<batch_name> ./run_experiment.sh 2stg-gnn
KEP_DATA_DIR=dataset/processed/<batch_name> ./run_experiment.sh 2stg-reg
```

3. 运行端到端 DFL

```bash
KEP_DATA_DIR=dataset/processed/<batch_name> ./run_experiment.sh dfl-gnn
KEP_DATA_DIR=dataset/processed/<batch_name> ./run_experiment.sh dfl-reg
```

4. 运行 Oracle 基线

```bash
KEP_DATA_DIR=dataset/processed/<batch_name> ./run_experiment.sh oracle results/2stg_Gnn_<timestamp>/best_stage1_model_real.pth
```

5. 统一评估

```bash
KEP_DATA_DIR=dataset/processed/<batch_name> ./run_experiment.sh evaluate
```

6. 手工查看图和解

```bash
./run_experiment.sh app
```

## 5. 输出目录约定

处理后数据：
- `dataset/processed/<batch_name>/G-*.json`
- `dataset/processed/<batch_name>/run_info.json`
- `dataset/processed/<batch_name>/batch_summary.json`
- `dataset/processed/<batch_name>/batch_report.md`

训练产物：
- `results/2stg_Gnn_<timestamp>/best_stage1_model_real.pth`
- `results/2stg_Reg_<timestamp>/best_stage1_model_real.pth`
- `results/dfl_Gnn_<timestamp>/best_dfl_model.pth`
- `results/dfl_Reg_<timestamp>/best_dfl_reg_model.pth`

测试集文件：
- `results/<experiment>/test_files.txt`

求解输出：
- `solutions/<experiment>/G-*_sol.json`

评估入口：
- [`4-evaulation.py`](/home/weikang/projects/UdeM-Intern/Exps/4-evaulation.py)

## 6. 建议的自检方式

如果你想先看命令会执行什么，而不真正运行：

```bash
./run_experiment.sh --dry-run 2stg-gnn
./run_experiment.sh --dry-run oracle
```

如果你想将来复现实验结果，建议至少记录：
- 你执行的命令
- 生成的 `results/<experiment>` 目录名
- 对应的 `solutions/<experiment>` 目录名
- 使用的 `test_files.txt`
