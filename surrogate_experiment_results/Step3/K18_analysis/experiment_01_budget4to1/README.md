# 建议方案：K18-E1 4:1 sample-size 学习曲线实验

以下按上一轮讨论中的 K18 定义：**原 K16 加入 G-206 和 G-72**。这个实验建议定义为一个全新的、独立的 exploratory experiment：

```text
K18-E1: 18 topologies × 5 data seeds × 3 sample sizes
       = 270 paired jobs
```

其中一个 paired job 同时训练：

```text
2stage
SPO+
```

并在同一个固定 test set 上评估。因此实际包含：

```text
270 paired jobs
540 model-training runs
270 paired evaluations
```

## Current artifact review status

Reviewed on garnet at `2026-06-26T01:57:40-04:00`.

Materialized output root:

```text
/local1/fuweik/UdeM-Intern/surrogate_experiment_results/Step3/K18_analysis/experiment_01_budget4to1/results/materialized
```

Stage 3 data materialization and artifact audit are complete:

```text
test banks:        18 / 18
fit bundles:       90 / 90
bundle audits:     90 / 90 passed
plan rows:         270 / 270 ready
plan-level audit:  passed=true, failures=[]
materialized size: 539M
```

The fresh plan-level audit reported:

```json
{
  "data_seed_count": 5,
  "failures": [],
  "job_count": 270,
  "passed": true,
  "ready_count": 270,
  "sample_sizes": [50, 100, 500],
  "topology_count": 18
}
```

Full manifest review confirmed:

```text
sample_size 50  -> training_size 40,  validation_size 10,  trainer_train_size_arg 40
sample_size 100 -> training_size 80,  validation_size 20,  trainer_train_size_arg 80
sample_size 500 -> training_size 400, validation_size 100, trainer_train_size_arg 400

18 test manifests each contain 1000 screen_test samples.
90 fit manifests each contain 500 samples.
90 split manifests pass the every-fifth validation partition.
training and validation indices are disjoint for every sample size.
training_40 <= training_80 <= training_400.
validation_10 <= validation_20 <= validation_100.
```

Recorded hashes:

```text
k18_topologies.csv:              897b3f95fce6f840c76136a7fed8113c3de6bf198df806359bfb6a745e2414e2
experiment.yaml:                 b2a02b902d6da2d9305bfecfe3ba5964510900834e14c630da27172055450102
context_generator.locked.yaml:   713f598ee636fc61a314ef513831ce4558e68487f882ef72dec4004464674b4f
sample_size_plan.json:           c9f334cf36c97a076a49ef5f4aede1afe6af6d0e1d789a1640c8a5336189f481
sample_size_jobs.csv:            af9ed09c29c95fca5c6a0db932a82fec15c3859954051ef1c9788186355fc0bc
plan_audit_result.json:          406ce56cf06b5385f8c435485a22ac048700184d7880226d2c0f49063efe6402
```

Two execution caveats remain before starting the formal 270 training jobs:

```text
1. sample_size_jobs.csv currently stores validation_path as the basename from
   the eval manifest. run_one_job.py resolves this relative to the eval manifest
   directory, so execution is not blocked, but external launchers should not
   treat validation_path as an absolute path.

2. run_one_job_command in the plan is intentionally dry-run-only and ends with
   --dry-run. The formal launcher must generate execute commands or strip that
   flag deliberately.
```

Do not start the full 270-job training run until Stage 1 tiny smoke, Stage 2
runtime pilot, and the bounded execution launcher have also been reviewed.

## Current tiny smoke status

Reviewed on garnet at `2026-06-26T02:13:47-04:00`.

Tiny smoke output root:

```text
/local1/fuweik/UdeM-Intern/surrogate_experiment_results/Step3/K18_analysis/experiment_01_budget4to1/results/tiny_smoke_6job_20260626
```

The 6-job tiny smoke used:

```text
topologies:    G-364, G-237
data seed:     101
sample sizes:  50, 100, 500
max_epochs:    2
execution:     sequential tmux runner
```

All six paired jobs completed successfully:

```text
paired manifests:      6 / 6
job_status.json:       6 / 6
job status:            success = 6
2stage status:         success = 6
SPO+ status:           success = 6
evaluation status:     success = 6
runner elapsed time:   187.45 seconds
tiny smoke size:       577M
```

The paired job manifests preserved the sample-size contract:

```text
sample_size 50  -> train_size 40,  training_size 40,  validation_size 10,  trainer_train_size_arg 40
sample_size 100 -> train_size 80,  training_size 80,  validation_size 20,  trainer_train_size_arg 80
sample_size 500 -> train_size 400, training_size 400, validation_size 100, trainer_train_size_arg 400
```

The completion watcher sent a Brevo notification with HTTP 201.

## Current runtime pilot status

Reviewed on garnet at `2026-06-26T08:50:43-04:00`.

Runtime pilot output root:

```text
/local1/fuweik/UdeM-Intern/surrogate_experiment_results/Step3/K18_analysis/experiment_01_budget4to1/results/runtime_pilot_9job_full_epoch_20260626
```

The full-epoch runtime pilot used:

```text
topologies:       G-364, G-784, G-237
data seed:        101
sample sizes:     50, 100, 500
max_epochs:       3000
early stopping:   patience=20, min_delta=0.0001
execution:        9 jobs launched in parallel
thread limits:    OMP/MKL/OPENBLAS/NUMEXPR = 1
```

The pilot completed successfully:

```text
paired jobs:         9 / 9
job_status.json:     9 / 9
failed jobs:         0
wall time:           10252.302 seconds = 2h 50m 52s
completion time:     2026-06-26T05:16:19-04:00
notification:        Brevo HTTP 201
```

The wall time was dominated by the slowest paired job:

```text
G-237 sample_size=500 -> 10252.286s
G-784 sample_size=500 -> 3158.885s
G-237 sample_size=100 -> 2082.606s
G-237 sample_size=50  -> 1031.324s
```

This confirms that `G-237` at `sample_size=500` is substantially slower than the
other pilot jobs. Its SPO+ run selected epoch 3000 and did not trigger early
stopping, so the runtime should be treated as an important bound for formal-run
scheduling. The main runtime log and per-job `job_status.json` are the completion
source of truth; `runtime_pilot_summary.json` did not backfill the final row's
elapsed time even though the run status and per-job status are successful.

## Current formal 270-job run status

Launched on garnet at `2026-06-26T09:17:51-04:00`.

Formal output root:

```text
/local1/fuweik/UdeM-Intern/surrogate_experiment_results/Step3/K18_analysis/experiment_01_budget4to1/results/formal_270_full_epoch_20260626
```

Launcher and watcher sessions:

```text
k18_e1_formal_270
notify_k18_e1_formal_270
```

Launch settings:

```text
normal queue: 225 jobs, concurrency=16
long queue:   45 jobs,  concurrency=4
long topologies: G-237, G-670, G-970
thread limits: OMP/MKL/OPENBLAS/NUMEXPR = 1
monitor interval: 60 seconds
```

The first monitor record reported:

```text
active_jobs=20
active_normal=16
active_long=4
failed_jobs=0
pending_normal=209
pending_long=41
```

## 先明确一个关键点

**现有 Step3 pipeline 不能直接把 `train_size=50` 解释成 40 个训练样本加 10 个验证样本。**

当前代码中的含义是：

* `train_size=50` 表示从 train bank 中取前 50 个样本训练；
* validation set 是一个独立的固定数据集；
* 同一 topology 的不同 training sizes 共用同一个固定 validation/test set。

`build_nested_train_bank.py` 目前生成的是纯 training prefixes，例如 50、100、500，不包含 4:1 的验证划分。
`run_one_job.py` 也会把 `train_size` 原样传给两个训练方法，同时从独立 eval manifest 中读取 validation set。
两个训练 wrapper 都会取 train bank 的前 `train_size` 个样本，并把整个 validation NPZ 用于 early stopping。

所以，**不能只把原配置里的 `[50,100,500]` 换个解释就运行**。需要增加一层“sample size → training/validation 数量”的数据构建与 job planning。

本目录名仍为 `experiment_01_budget4to1`，但协议字段不使用 `budget`。后续 config、plan、manifest 和 summary 中应统一使用：

```text
sample_size
training_size
validation_size
trainer_train_size_arg
```

其中 `trainer_train_size_arg` 是为了兼容现有 CLI `--train-size`；它的值等于 `training_size`，不是 `sample_size`。

---

# 1. 锁定 K18

建议 K18 manifest 为：

```text
G-269
G-398
G-784
G-103
G-927
G-970
G-304
G-364
G-730
G-658
G-79
G-836
G-124
G-396
G-670
G-237
G-206
G-72
```

角色组成大致为：

| 角色                | Topologies                                      |
| ----------------- | ----------------------------------------------- |
| 强正例               | G-269, G-398, G-784, G-103, G-927, G-970, G-304 |
| 正向结构覆盖            | G-364, G-730, G-658, G-237                      |
| 稳定负对照             | G-79, G-836, G-124, G-206, G-72                 |
| neutral / no-room | G-396, G-670                                    |

建议新建：

```text
surrogate_experiment_results/Step3/K18_analysis/
└── experiment_01_budget4to1/
    ├── README.md
    ├── configs/
    │   ├── k18_topologies.csv
    │   ├── experiment.yaml
    │   └── context_generator.locked.yaml
    ├── plans/
    ├── summaries/
    └── validation_logs/
```

大数据仍放在 scratch，例如：

```text
/local1/fuweik/step3_k18_exp1_sample_size4to1_20260626/
```

不要把 train banks、validation NPZ、weights 和 per-job materialized JSON 提交到 Git。

---

# 2. 严格区分 sample size 和 training size

建议明确区分以下字段：

| `sample_size` | `training_size` | `validation_size` | `trainer_train_size_arg` / 现有 `--train-size` |
| ----: | ----: | ----: | ----: |
|    50 |    40 |    10 |    40 |
|   100 |    80 |    20 |    80 |
|   500 |   400 |   100 |   400 |

建议字段命名为：

```text
sample_size
training_size
validation_size
trainer_train_size_arg
```

而不是把 `sample_size=50` 继续写成 `train_size=50`。

在本实验中：

```text
sample_size = training_size + validation_size
test_size   is external evaluation data and is not part of sample_size
```

一个 job 的标识应类似：

```text
G-269|data_seed=000101|sample_size=050|training=040|validation=010
```

输出目录建议：

```text
jobs/
└── step2c_poly_d8_mult_eps050/
    └── G-269/
        └── train_seed=000101/
            ├── sample_size=050/
            ├── sample_size=100/
            └── sample_size=500/
```

---

# 3. 推荐的 nested 4:1 划分方法

## 一个 seed 只生成一个 500-sample fit bank

对于每个 `(topology, data_seed)`，只生成一组 500 个完整 ((X,y)) 样本。

然后使用一个**预先固定、永不变化**的角色分配规则：

```text
在每连续 5 个样本中：
    前 4 个属于 training pool
    第 5 个属于 validation pool
```

用 0-based index 表示：

```python
is_validation = (sample_index + 1) % 5 == 0
is_training = not is_validation
```

于是：

### Sample size 50

```text
使用 fit bank 的前 50 个样本
training_size   = 40
validation_size = 10
```

### Sample size 100

```text
使用 fit bank 的前 100 个样本
training_size   = 80
validation_size = 20
```

### Sample size 500

```text
使用 fit bank 的全部 500 个样本
training_size   = 400
validation_size = 100
```

这个设计同时满足：

```text
sample_50 ⊂ sample_100 ⊂ sample_500

training_40 ⊂ training_80 ⊂ training_400

validation_10 ⊂ validation_20 ⊂ validation_100

training_N ∩ validation_N = ∅
```

并且不会发生以下错误：

```text
sample_size=50 时作为 validation 的样本
在 sample_size=100 时变成 training 样本
```

这比简单地使用：

```text
training = bank[0:40]
val   = bank[40:50]

training = bank[0:80]
val   = bank[80:100]
```

更好，因为后一种方案会使较小 sample size 的 validation 样本在更大 sample size 下进入 training，造成 cross-sample-size role leakage。

---

# 4. Test set 如何设置

## Test set 不计入 50/100/500 的预算

定义：

```text
sample_size = training_size + validation_size
test set    = 独立额外数据
```

建议：

```text
test_size = 1000
```

每个 topology 只生成一个新的 fixed test bank：

```text
18 topologies × 1 test bank
```

对应脚本：

```text
scripts/build_k18_test_bank.py
```

该脚本为每个 topology 写入：

```text
test.npz
test_manifest.json
```

`test_hash` 必须来自实际 `test.npz` 的 `dataset_hash`，不能手写或只在 eval manifest 中声明。

同一个 topology 的：

```text
5 seeds × 3 sample sizes = 15 jobs
```

全部使用相同的 test hash。

这可以隔离：

```text
training-data randomness
sample-size effect
method effect
```

## 不要复用原 160×5 screen 的 test bank

因为 K18 就是根据旧 screening 结果选出来的。新实验应生成一个新的 test namespace / experiment version，避免 topology selection 与后续 evaluation 共用同一 test realization。

现有代码只支持 `screen` 和 `confirm` protocol，因此该实验建议暂时使用：

```text
protocol = screen
experiment_family = K18_analysis
experiment_version = k18_exp1_sample_size4to1_fullxy_v1
```

`experiment_version` 会参与 context seed 和 label-noise seed 的派生，因此使用新的 experiment version 能生成新的独立数据。

这项实验仍应标记为：

```text
exploratory K18 learning-curve experiment
not formal confirmation
```

---

# 5. Seeds 设置

建议不要继续用旧 screen 的 1–5，以降低误合并风险。

使用：

```text
data seeds = 101,102,103,104,105
theta_seed = 42
gurobi_seed = 42
```

含义：

* `data seed` 控制每个 topology 的独立 full-((X,y)) fit bank；
* `theta_seed=42` 在所有方法和 jobs 中固定；
* `gurobi_seed=42` 固定；
* 同一个 `(G, data_seed, sample_size)` 下 2stage 和 SPO+ 使用完全相同的 training/validation/test 数据与 theta initialization。

现有 Step3 设计本来就把 `train_seed` 定义为同一 topology 上的独立 training-dataset replication，而不是新 topology。

---

# 6. 建议的核心 experiment config

```yaml
experiment:
  experiment_id: k18_exp1_sample_size4to1_fullxy_v1
  phase: K18_analysis
  protocol: screen
  claim_level: exploratory
  topology_count: 18

topologies:
  manifest: configs/k18_topologies.csv

regime:
  name: step2c_poly_d8_mult_eps050

data:
  sample_sizes: [50, 100, 500]

  split:
    train_fraction: 0.8
    validation_fraction: 0.2
    assignment_rule: every_fifth_sample_is_validation
    nested_across_sample_sizes: true
    roles_never_change_across_sample_sizes: true

  counts:
    50:
      training: 40
      validation: 10
      trainer_train_size_arg: 40
    100:
      training: 80
      validation: 20
      trainer_train_size_arg: 80
    500:
      training: 400
      validation: 100
      trainer_train_size_arg: 400

  max_fit_bank_size: 500
  test_size: 1000
  test_shared_across_seeds: true
  test_shared_across_sample_sizes: true
  test_independent_from_selection_screen: true

seeds:
  data_seeds: [101, 102, 103, 104, 105]
  theta_seed: 42
  gurobi_seed: 42
  master_label_seed: 20260626

training:
  methods:
    - 2stage
    - spoplus
  max_epochs: 3000
  metric_stride: 1
  early_stop_patience: 20
  early_stop_min_delta: 0.0001

execution:
  paired_jobs: 270
  model_training_runs: 540
  paired_evaluations: 270
  scheduler: tmux_bounded_parallel
  normal_workers: 3
  long_workers: 1

evaluation:
  primary_metric: test_mean_decision_gap
  improvement_metric: >
    test_mean_decision_gap_2stage -
    test_mean_decision_gap_spoplus
```

`context_generator.example.yaml` 在仓库中明确标记为 `pilot_not_locked`，不应直接作为本实验的正式 generator config。

应复制出：

```text
configs/context_generator.locked.yaml
```

同一目录还必须存在：

```text
configs/k18_topologies.csv
configs/experiment.yaml
```

如果这三个正式配置文件缺失，或者 `context_generator.locked.yaml` 仍标记为 `pilot_not_locked`，不得进入完整数据物化或 270-job planning。

当前目录已经 materialize 了上述三个配置文件。`context_generator.locked.yaml` 使用新的：

```text
generator_version = fixed_topology_context_v1_k18_exp1_sample_size4to1_locked
status = locked
```

后续正式物化和 planning 必须引用这三个文件，而不是 `surrogate_experiment_results/Step3/configs/context_generator.example.yaml`。

并记录：

```text
generator_config_hash
git commit SHA
topology manifest hash
experiment config hash
```

---

# 7. 最小侵入式代码实现方案

我建议**不要修改原有 confirmation pipeline 的语义**，而是新增 K18 sample-size experiment 的 data-builder 和 planner，然后尽可能复用现有 `run_one_job.py`。

## 7.1 新增数据构建脚本

建议新增：

```text
surrogate_experiment_results/Step3/K18_analysis/experiment_01_budget4to1/scripts/
    build_nested_fit_validation_bank.py
    build_k18_test_bank.py
```

它对每个 `(G, data_seed)`：

1. 生成一个 500-sample fit bank；
2. 按 every-fifth rule 分配 train/validation；
3. 写一个 400-sample training bank；
4. training bank 提供以下 prefix hashes：

```text
40
80
400
```

5. 写三个 validation NPZ：

```text
validation_sample_size050.npz   # 10 samples
validation_sample_size100.npz   # 20 samples
validation_sample_size500.npz   # 100 samples
```

6. 写一个 split manifest：

```json
{
  "sample_sizes": [50, 100, 500],
  "assignment_rule": "every_fifth_sample_is_validation",
  "sample_size_splits": {
    "50": {
      "training_size": 40,
      "validation_size": 10,
      "training_hash": "...",
      "validation_hash": "..."
    },
    "100": {
      "training_size": 80,
      "validation_size": 20,
      "training_hash": "...",
      "validation_hash": "..."
    },
    "500": {
      "training_size": 400,
      "validation_size": 100,
      "training_hash": "...",
      "validation_hash": "..."
    }
  }
}
```

同时写一个完整 `fit_manifest.json`，记录 500 个 fit samples 的 sample rows、role assignment 和 `fit_bank_hash`。Audit 使用它检查：

```text
training_400 ∪ validation_100 = full 500-sample fit set
training_400 ∩ validation_100 = ∅
```

`train_bank.npz` 和 validation NPZ 的顶层 manifest 必须保留 `source_namespace = screen_train`，validation 只通过：

```text
fit_role = validation
validation_scheme = every_fifth_sample
```

表达其在 fit bank 中的角色，不应把顶层 namespace 改成公共工具不认识的新 namespace。

这些 manifests 必须包含非空 provenance：

```text
experiment_version
master_label_seed
generator_version
generator_config_hash
topology_hash
arc_order_hash
feasible_set_hash
```

这样只需生成：

```text
18 × 5 = 90 个 fit banks
```

而不是给 270 个 jobs 分别重新生成数据。

## 7.2 每个 sample size 写一个 eval manifest

对于每个 `(G, data_seed, sample_size)`，写：

```text
eval_manifest_sample_size050.json
eval_manifest_sample_size100.json
eval_manifest_sample_size500.json
```

其中：

```text
validation_path = 该 data_seed / 该 sample_size 的 validation NPZ
test_path       = 该 topology 的共享 test NPZ
```

例如：

```json
{
  "topology_id": "G-269",
  "protocol": "screen",
  "data_seed": 101,
  "sample_size": 50,
  "training_size": 40,
  "validation_size": 10,
  "trainer_train_size_arg": 40,
  "validation_path": ".../validation_sample_size050.npz",
  "validation_hash": "...",
  "test_path": ".../G-269/test/test.npz",
  "test_hash": "..."
}
```

## 7.3 复用现有 `run_one_job.py`

现有 `run_one_job.py` 可以继续执行 2stage、SPO+ 和 evaluation，只需要把**实际 training count** 传进去：

```text
sample_size 50  -> --train-size 40
sample_size 100 -> --train-size 80
sample_size 500 -> --train-size 400
```

在给 `run_one_job.py` 增加 `--sample-size` metadata 参数后，示例命令为：

```bash
python surrogate_experiment_results/Step3/scripts/run_one_job.py \
  --train-bank /local1/fuweik/step3_k18_exp1/data/.../G-269/data_seed=000101/train_bank.npz \
  --eval-manifest /local1/fuweik/step3_k18_exp1/data/.../G-269/data_seed=000101/eval_manifest_sample_size050.json \
  --topology-id G-269 \
  --regime step2c_poly_d8_mult_eps050 \
  --protocol screen \
  --train-seed 101 \
  --train-size 40 \
  --theta-seed 42 \
  --gurobi-seed 42 \
  --max-epochs 3000 \
  --metric-stride 1 \
  --early-stop-patience 20 \
  --early-stop-min-delta 0.0001 \
  --sample-size 50 \
  --output-dir /local1/fuweik/step3_k18_exp1/jobs/.../G-269/data_seed=000101/sample_size=050 \
  --execute
```

建议给 `run_one_job.py` 增加一个**可选 metadata 参数**：

```text
--sample-size 50
```

它不参与训练，只写入 paired manifest，避免以后把：

```text
train_size=40
```

误解为该实验的名义 size。

## 7.4 不要直接使用 `run_confirmation.py`

当前 `run_confirmation.py`：

* 把 `[50,100,500]` 当作实际 training sizes；
* 每个 topology 读取一个固定 eval manifest；
* 所有 seed 和 training size 复用该 validation path。

因此它不适合当前的：

```text
seed-specific validation
sample-size-specific validation size
50 -> 40/10
100 -> 80/20
500 -> 400/100
```

建议新增：

```text
plan_k18_sample_size_jobs.py
audit_k18_sample_size_artifacts.py
```

而不要硬改 formal confirmation planner。

`plan_k18_sample_size_jobs.py` 默认 strict：train bank、eval manifest、validation NPZ、test NPZ 或 hash 缺失时直接失败。只有做纯 planning 草稿时才使用：

```bash
--allow-missing-artifacts
```

strict plan 中的 job row status 必须是：

```text
ready
```

---

# 8. 270-row job plan 应包含的字段

```text
job_id
topology_id
role
data_seed
trainer_train_seed_arg
sample_size
training_size
validation_size
trainer_train_size_arg
fit_bank_path
train_bank_path
validation_path
test_path
eval_manifest_path
training_hash
validation_hash
test_hash
theta_seed
gurobi_seed
max_epochs
early_stop_patience
early_stop_min_delta
output_dir
runtime_class
run_one_job_command
status
```

计数校验：

```text
18 topologies
× 5 seeds
× 3 sample sizes
= 270 rows
```

每个 topology：

```text
5 × 3 = 15 jobs
```

每个 seed：

```text
3 nested sample-size jobs
```

---

# 9. 必须执行的 audit

在启动任何正式 job 前，audit 必须检查：

## Bank-level invariants

```text
fit bank sample count = 500
training-role count   = 400
validation-role count = 100
```

## Sample-size-level invariants

```text
sample_size 50:
    training = 40
    validation = 10

sample_size 100:
    training = 80
    validation = 20

sample_size 500:
    training = 400
    validation = 100
```

## Nesting invariants

```text
training_40 ⊂ training_80 ⊂ training_400
validation_10 ⊂ validation_20 ⊂ validation_100
```

## Leakage invariants

```text
training_40 ∩ val_10 = ∅
training_80 ∩ val_20 = ∅
training_400 ∩ val_100 = ∅
```

## Pairing invariants

对同一个 job：

```text
2stage.train_hash       == SPO+.train_hash
2stage.validation_hash  == SPO+.validation_hash
2stage.test_hash        == SPO+.test_hash
2stage.theta_init       == SPO+.theta_init
```

## Test invariants

同一个 topology 的全部 15 个 jobs：

```text
test_hash 必须完全相同
```

不同 topology 可以有不同 test hash。

Audit 必须实际读取 `test.npz`：

```text
test.npz exists
test sample_count = 1000
computed dataset_hash == eval_manifest.test_hash
split_namespace = screen_test
```

Plan-level audit 还必须检查：

```text
18 unique topologies
5 unique data seeds
3 sample sizes
270 unique jobs
270 / 270 status = ready
one test hash per topology
```

现有 `run_one_job.py` 已经会检查 2stage 和 SPO+ 是否使用相同的 train-prefix、validation、test 和 theta initialization。

在 `--sample-size` 路径下，`run_one_job.py` 还必须 fail-fast 检查：

```text
CLI --train-size == eval_manifest.training_size
CLI --train-size == eval_manifest.trainer_train_size_arg
CLI --train-seed == eval_manifest.data_seed
CLI topology_id/regime/protocol == eval_manifest topology_id/regime/protocol
sample_size == training_size + validation_size
```

---

# 10. 分阶段执行，不要直接启动 270 jobs

## Stage 0：commit lock

运行前记录：

```bash
git status
git rev-parse HEAD
sha256sum configs/k18_topologies.csv
sha256sum configs/experiment.yaml
sha256sum configs/context_generator.locked.yaml
```

要求：

```text
代码、配置和 topology manifest 已提交
worktree clean
```

## Stage 1：tiny smoke

选择：

```text
G-364
G-237
```

运行：

```text
2 topologies × 1 seed × 3 sample sizes = 6 jobs
max_epochs = 2
```

验证完整的数据生成、split、train、early stop、evaluation 和 aggregation 流程。

## Stage 2：full-epoch runtime pilot

建议选择：

| Topology | 用途                   |
| -------- | -------------------- |
| G-364    | fast / sparse        |
| G-784    | rich / high-variance |
| G-237    | worst-case slow      |

运行：

```text
3 topologies × 1 seed × 3 sample sizes = 9 jobs
max_epochs = 3000
```

用这 9 个 jobs 估计：

```text
sample_size 50 -> 100 的 runtime scaling
sample_size 100 -> 500 的 runtime scaling
memory usage
early-stop epoch distribution
```

## Stage 3：完整数据物化和 audit

生成：

```text
90 fit banks
18 fixed test banks
270 job rows
```

要求：

```text
audit passed = 270 / 270
dry-run ready = 270 / 270
```

Bundle audit 示例：

```bash
python scripts/audit_k18_sample_size_artifacts.py \
  --train-bank data/.../train_bank.npz \
  --split-manifest data/.../split_manifest.json \
  --eval-manifest data/.../eval_manifest_sample_size050.json \
  --eval-manifest data/.../eval_manifest_sample_size100.json \
  --eval-manifest data/.../eval_manifest_sample_size500.json \
  --expected-test-size 1000
```

Plan-level audit 示例：

```bash
python scripts/audit_k18_sample_size_artifacts.py \
  --plan-json sample_size_plan.json \
  --expected-topology-count 18 \
  --expected-data-seed-count 5 \
  --expected-sample-sizes 50,100,500 \
  --expected-job-count 270
```

## Stage 4：bounded parallel execution

Garnet 上不依赖 Slurm，建议使用：

```text
tmux + bounded worker launcher
```

显式限制线程：

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

初始队列划分：

### Long queue

```text
G-237
G-670
G-970
```

数量：

```text
3 × 5 × 3 = 45 jobs
```

并发：

```text
1
```

### Normal queue

其余 15 个 topology：

```text
15 × 5 × 3 = 225 jobs
```

并发：

```text
2–4
```

建议初始设为 3。完成 runtime pilot 后再调整。

---

# 11. Training 参数建议

为保持方法间和 sample size 间一致：

```text
max_epochs = 3000
metric_stride = 1
early_stop_patience = 20
early_stop_min_delta = 0.0001
theta_seed = 42
gurobi_seed = 42
```

不要针对 sample size 50、100、500 单独调 patience 或 learning rate，否则 sample-size effect 会混入 hyperparameter effect。

需要承认一个风险：

```text
sample_size=50 时 validation 只有 10 个样本
sample_size=100 时 validation 只有 20 个样本
```

因此 early stopping 会比原来 `validation_size=100` 更不稳定。建议：

* 保存完整 validation trace；
* 报告 best epoch 的分布；
* 不在查看结果后为某个 sample size 临时更改 patience；
* 如需 sensitivity analysis，另开预注册实验，不和主实验混合。

---

# 12. 结果聚合

## Job-level：270 行

至少保存：

```text
topology_id
data_seed
trainer_train_seed_arg
sample_size
training_size
validation_size
trainer_train_size_arg
test_mean_decision_gap_2stage
test_mean_decision_gap_spoplus
spoplus_improvement_gap
2stage_best_epoch
spoplus_best_epoch
runtime_seconds
status
```

## Topology × sample size：54 行

```text
18 topologies × 3 sample sizes = 54 rows
```

每行计算：

```text
mean_2stage_gap
median_2stage_gap
mean_spoplus_gap
median_spoplus_gap
mean_improvement
median_improvement
positive_seed_count
zero_seed_count
negative_seed_count
win_rate
mean_runtime
```

## Topology × seed learning curve：90 行

```text
18 topologies × 5 seeds = 90 rows
```

每行放三个 sample sizes：

```text
improvement_50
improvement_100
improvement_500

2stage_gap_50
2stage_gap_100
2stage_gap_500

spoplus_gap_50
spoplus_gap_100
spoplus_gap_500
```

并计算：

```text
improvement_100 - improvement_50
improvement_500 - improvement_100

2stage_gap_100 - 2stage_gap_50
2stage_gap_500 - 2stage_gap_100

spoplus_gap_100 - spoplus_gap_50
spoplus_gap_500 - spoplus_gap_100
```

这样可以回答：

* 更多样本是否同时改善 2stage 和 SPO+；
* SPO+ 的优势是否主要出现在 low-data regime；
* negative controls 是否随着 sample size 增大而恢复；
* neutral cases 是否一直 neutral；
* computationally hard cases 的 runtime 如何随 sample size 扩大。

---

# 13. Claim boundary

这 270 个新 jobs 与原来的 160×5 screen **不能直接合并成同一个 seed pool**，因为：

```text
原实验：
train_size=50 是 50 个训练样本
validation_size=100 是额外固定验证集

新实验：
sample_size=50 是 40 training + 10 validation
```

因此它应被描述为：

```text
a new K18 4:1 sample-size learning-curve experiment
```

而不是：

```text
an extension of the old 5-seed screen
```

另外：

* K18 是根据旧 screening 结果有目的地选出的 mechanism sample；
* 新 test bank 可以降低 test reuse，但 K18 仍不是随机 population sample；
* 每个 cell 只有 5 seeds；
* 不应声称 final statistical significance；
* 不应把 270 个 jobs 当作 270 个独立科学重复，因为 jobs 在 topology、seed 和 nested sample sizes 上高度相关。

---

# 最终推荐设置

```text
Experiment:
    K18-E1 4:1 sample-size learning curve

Topologies:
    K18 = K16 + G-206 + G-72

Data seeds:
    101,102,103,104,105

Sample sizes:
    50  -> training 40,  validation 10
    100 -> training 80,  validation 20
    500 -> training 400, validation 100

Nesting:
    every fifth fit-bank sample is validation
    smaller train and validation sets are strict subsets of larger ones
    no role switching across sample sizes

Test:
    new independent fixed test bank
    test_size = 1000
    one test hash per topology
    shared across all seeds and sample sizes

Methods:
    2stage
    SPO+

Training:
    max_epochs = 3000
    patience = 20
    min_delta = 0.0001
    theta_seed = 42
    gurobi_seed = 42

Scale:
    90 fit banks
    18 test banks
    270 paired jobs
    540 training runs
    270 evaluations

Protocol:
    screen / exploratory
    not formal confirmation
```

最关键的实现原则是：**job 名义上的 size 使用 `sample_size`，现有 trainer 接收到的 `--train-size` 必须是 40、80、400，而不是 50、100、500。**
