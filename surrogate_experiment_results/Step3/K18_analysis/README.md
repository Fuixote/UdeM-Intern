# Step3 K=18 Extension 拓扑机制分析

> **状态：** K=18 extension selection 已锁定，用于 seed-stability 与 detailed topology-mechanism analysis。  
> **Selection source commit：** `779f55e88bc5578abafdb3c7e6a846946facdeb1`  
> **Selection source run：** `screen160x5_e3000_20260620`  
> **本目录：** `surrogate_experiment_results/Step3/K18_analysis/`（历史上从 K16 机制集合扩展而来；当前目录和协议名均使用 K18）  
> **重要边界：** 本分析仍属于 screening / mechanism-study 阶段，不是 final scientific confirmation。

---

## 1. 研究目的

本目录记录从 Step3 Phase-B′ 160×5 full-\((X,y)\) screening 中锁定的 18 个 topology，并定义后续分析的完整流程。目标不是简单复述“哪些 topology 的 SPO+ mean improvement 最大”，而是回答以下问题：

1. 五个 screening train seeds 上观察到的正向、负向和中性结果，在增加 train seeds 后是否稳定；
2. SPO+ 在哪些 topology 结构和 decision landscape 中明显优于 2stage；
3. SPO+ 在哪些 topology 中无改善空间，或反而伤害 downstream decision；
4. cycle/chain 结构、candidate conflict、oracle landscape、rank reversal 和 feasible-set ambiguity 如何解释方法差异；
5. 哪些 topology 应进入后续独立、锁定的 formal confirmation。

本阶段使用的核心指标为：

```text
spoplus_improvement_gap
    = test_mean_decision_gap_2stage
    - test_mean_decision_gap_spoplus
```

因此：

```text
Δ > 0  -> SPO+ 的 test decision gap 更低
Δ = 0  -> 两种方法在该 job 上相同
Δ < 0  -> SPO+ 比 2stage 更差
```

---

## 2. 来源、运行验证与 claim boundary

### 2.1 Selection source files

K=18 extension 选择只依据以下 commit 中的已跟踪 summary 文件：

```text
surrogate_experiment_results/Step3/pairs20_ndd2/screening/
    screen160x5_e3000_20260620_summary.json
    screen160x5_e3000_20260620_jobs.csv
    screen160x5_e3000_20260620_topology_summary.csv
    phase_b_topologies.csv

surrogate_experiment_results/Step3/validation_logs/
    screen160x5_e3000_20260620.md

surrogate_experiment_results/Step3/
    README.md
```

### 2.2 Source run setting

| 参数 | 值 |
|---|---:|
| protocol | `screen` |
| topologies | 160 |
| train seeds | 1, 2, 3, 4, 5 |
| train size | 50 |
| validation size | 100 |
| test size | 1000 |
| max epochs | 3000 |
| early-stop patience | 20 |
| early-stop min delta | 0.0001 |
| jobs | 800 |
| success / failed / skipped | 800 / 0 / 0 |
| elapsed | 104.22 h |
| scratch output | 78G |

整体 800-job screening 结果为：

| 指标 | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|
| 2stage test decision gap | 3.311511 | 1.324966 | 0.000000 | 33.749763 |
| SPO+ test decision gap | 2.475742 | 1.079440 | 0.000000 | 33.574975 |
| SPO+ improvement Δ | 0.835769 | 0.000000 | -2.198430 | 15.292673 |

Job-level sign counts：

```text
positive = 273
zero     = 427
negative = 100
```

Topology-level：

```text
at least one negative seed = 29 / 160
all five seeds negative    = 11 / 160
```

### 2.3 可允许与不可允许的表述

本阶段可以表述：

```text
low-seed screening evidence
selection-stability evidence
topology-specific mechanism case study
descriptive comparison on a locked K=18 extension set
```

本阶段不可以表述：

```text
final scientific significance
population-average causal effect
the selected K=18 extension represents all 160 topologies
five seeds prove general superiority
screen test set is an independent confirmation set
```

K=18 extension 是按研究角色构造的 mechanism sample，而不是按原始 topology population frequency 抽样。

---

## 3. 锁定的 K=18 extension

### 3.1 Topology IDs

`G-269`, `G-398`, `G-784`, `G-103`, `G-927`, `G-970`, `G-304`, `G-364`, `G-730`, `G-658`, `G-79`, `G-836`, `G-124`, `G-396`, `G-670`, `G-237`, `G-206`, `G-72`

### 3.2 Performance summary

| 顺序 | topology_id | 选择角色 | mean Δ | median Δ | win_rate | negative seeds | mean 2stage gap | mean SPO+ gap | mean runtime (s) |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | G-269 | 强正例 | 15.2832 | 15.2860 | 1.00 | 0 | 15.5248 | 0.2416 | 256.9 |
| 2 | G-398 | 强正例 | 11.6999 | 11.6999 | 1.00 | 0 | 11.7003 | 0.0004 | 691.0 |
| 3 | G-784 | 强正例 | 10.8334 | 10.8296 | 1.00 | 0 | 11.9543 | 1.1209 | 983.7 |
| 4 | G-103 | 强正例 | 10.0232 | 10.0296 | 1.00 | 0 | 12.6283 | 2.6051 | 234.0 |
| 5 | G-927 | 匹配正例 | 8.0005 | 8.0256 | 1.00 | 0 | 9.3017 | 1.3013 | 727.6 |
| 6 | G-970 | 强正例/计算困难 | 7.6602 | 7.8848 | 1.00 | 0 | 17.9502 | 10.2900 | 1781.2 |
| 7 | G-304 | 强正例 | 5.8401 | 5.8887 | 1.00 | 0 | 7.6428 | 1.8027 | 319.2 |
| 8 | G-364 | 稀疏正例锚点 | 3.1866 | 3.3884 | 1.00 | 0 | 6.2040 | 3.0174 | 59.0 |
| 9 | G-730 | 结构覆盖正例 | 2.3368 | 2.3703 | 1.00 | 0 | 4.7911 | 2.4543 | 403.8 |
| 10 | G-658 | neutral-regime 正例锚点 | 0.0872 | 0.1016 | 1.00 | 0 | 1.8431 | 1.7559 | 53.9 |
| 11 | G-79 | 强负对照 | -1.5503 | -1.4970 | 0.00 | 5 | 3.3034 | 4.8537 | 128.3 |
| 12 | G-836 | 匹配负对照 | -0.0830 | -0.0856 | 0.00 | 5 | 9.4322 | 9.5152 | 42.9 |
| 13 | G-124 | 高改善空间负对照 | -0.6136 | -0.6050 | 0.00 | 5 | 11.7861 | 12.3996 | 153.8 |
| 14 | G-396 | 高改善空间中性对照 | 0.0000 | 0.0000 | 0.00 | 0 | 15.4599 | 15.4599 | 197.1 |
| 15 | G-670 | 无改善空间中性/计算困难 | 0.0000 | 0.0000 | 0.00 | 0 | 0.0004 | 0.0004 | 2315.0 |
| 16 | G-237 | 结构/计算困难 | 0.4420 | 0.4405 | 1.00 | 0 | 0.8846 | 0.4426 | 2962.0 |
| 17 | G-206 | 强负例 extension | -1.5274 | -1.5291 | 0.00 | 5 | 4.1291 | 5.6565 | 115.3 |
| 18 | G-72 | 强负例 extension | -0.9737 | -0.8687 | 0.00 | 5 | 5.8322 | 6.8059 | 189.0 |

### 3.3 Structural summary

| topology_id | complexity_bin | structural_type | landscape_regime | candidates | cycles | chains | conflict density | 选择理由 |
|---|---|---|---|---:|---:|---:|---:|---|
| G-269 | medium_rich | chain_only | proxy_aligned | 40 | 0 | 40 | 0.867 | 最大且极稳定的正向案例；表明大幅收益并不局限于 proxy-hard landscape。 |
| G-398 | medium_rich | cycle_chain | easy_control | 40 | 3 | 37 | 0.786 | easy-control、cycle+chain 条件下仍出现巨大且五 seed 完全一致的收益。 |
| G-784 | rich | cycle_chain | high_variance | 58 | 12 | 46 | 0.858 | rich/high-variance/cycle+chain 中的大幅稳定正例，适合研究深度重排序。 |
| G-103 | low_medium | chain_only | proxy_hard | 13 | 0 | 13 | 0.692 | low-medium、chain-only、proxy-hard 的清晰强正例，计算成本适中。 |
| G-927 | medium_rich | chain_only | proxy_hard | 22 | 0 | 22 | 0.775 | 与 G-836 位于同一 structural cell，构成最重要的 matched positive/negative 对照。 |
| G-970 | extreme | cycle_chain | proxy_hard | 86 | 6 | 80 | 0.809 | extreme、slow、proxy-hard 且收益很大；兼具效果与计算难度。 |
| G-304 | low_medium | cycle_chain | proxy_hard | 18 | 2 | 16 | 0.850 | 补足 low-medium/cycle+chain/proxy-hard cell，收益强且稳定。 |
| G-364 | sparse_simple | chain_only | proxy_hard | 7 | 0 | 7 | 0.762 | sparse-simple/chain-only anchor，防止正例只集中在稠密复杂拓扑。 |
| G-730 | rich | chain_only | high_variance | 44 | 0 | 44 | 0.931 | rich/chain-only/high-variance 覆盖点；为结构多样性保留。 |
| G-658 | sparse_simple | cycle_chain | neutral | 8 | 3 | 5 | 0.750 | 唯一的 neutral-landscape 正向 anchor；主要价值是 landscape coverage，而非效应大小。 |
| G-79 | medium_rich | cycle_chain | proxy_hard | 27 | 2 | 25 | 0.946 | 均值最负且五 seed 全负，是首要 harmful control。 |
| G-836 | medium_rich | chain_only | proxy_hard | 39 | 0 | 39 | 0.633 | chain-only 的稳定负对照，可与 G-927 做同 cell 机制比较。 |
| G-124 | rich | cycle_chain | proxy_hard | 59 | 3 | 56 | 0.641 | 高 2stage gap 但 SPO+ 稳定更差，适合研究“有改善空间却受损”的机制。 |
| G-396 | extreme | cycle_chain | proxy_hard | 93 | 16 | 77 | 0.907 | 2stage 与 SPO+ gap 都很高但五 seed 完全相同，是 high-room neutral 对照。 |
| G-670 | medium_rich | cycle_chain | easy_control | 37 | 4 | 33 | 0.887 | 几乎没有 decision room，但运行极慢；用于 no-room 与 computational hardness 分离。 |
| G-237 | rich | cycle_chain | proxy_aligned | 67 | 7 | 60 | 0.969 | 最慢拓扑且 conflict density 极高；用于 feasible-set geometry 与 runtime mechanism。 |
| G-206 | medium_rich | cycle_chain | proxy_hard | 29 | 6 | 23 | 0.933 | 五 seed 全负且效应幅度接近 G-79，是 stronger-negative extension 的核心案例。 |
| G-72 | rich | cycle_chain | proxy_hard | 49 | 7 | 42 | 0.798 | rich/cycle+chain/proxy-hard 的强负例，用于扩展 harmful mechanism 边界。 |

### 3.4 Coverage

```text
seed-sign composition:
    all-positive = 11
    all-negative = 5
    all-zero     = 2

complexity_bin:
    sparse_simple = 2
    low_medium    = 2
    medium_rich   = 7
    rich          = 5
    extreme       = 2

structural_type:
    chain_only  = 6
    cycle_chain = 12

landscape_regime:
    proxy_hard    = 11
    proxy_aligned = 2
    high_variance = 2
    easy_control  = 2
    neutral       = 1
```

### 3.5 预先指定的 matched contrasts

| 对照 | 主要问题 |
|---|---|
| G-927 vs G-836 | 同为 `medium_rich / chain_only / proxy_hard`，为什么一个是强正例、另一个五 seed 全负？ |
| G-398 vs G-670 | 同为 `medium_rich / cycle_chain / easy_control`，为什么一个出现巨大收益，另一个完全无改善空间？ |
| G-124 vs G-396 | 在较高 baseline gap 下，为什么一个稳定受损，另一个 exact tie？ |
| G-269 vs G-237 | 都是 proxy-aligned，但 chain-only 与高冲突 cycle+chain 的机制和 runtime 如何不同？ |
| G-304 vs G-103 | 同为 low-medium/proxy-hard，cycle+chain 与 chain-only 是否产生不同的 rank-reversal 机制？ |
| G-79 vs G-206 vs G-72 vs G-124 | 四个 all-negative harmful cases 的伤害机制是否相同，还是由不同结构原因触发？ |

### 3.6 深度 mechanism analysis 的五个 sentinel

```text
G-269  robust large positive
G-927  matched positive
G-836  matched negative
G-396  high-room neutral
G-206  strong harmful extension
G-237  slow structural-hard
```

全部 18 个 topology 都必须进入 aggregate reporting。上述 sentinel 只是人工深挖优先级，不是排除其他 topology 的依据。

### 3.7 Selection tradeoff：K=18 extension

当前 K=18 extension 是一个 **mechanism-coverage sample plus stronger-negative extension**，不是按 effect magnitude 排序得到的 top-K performance sample。这个取舍需要在后续报告中明确说明。

本版从原 K16 机制覆盖集合中扩展加入：

```text
K = 18 extension:
    add G-206, G-72
```

扩展理由：

```text
G-206:
    5/5 seeds SPO+ worse
    mean Δ = -1.5274
    medium_rich / cycle_chain / proxy_hard

G-72:
    5/5 seeds SPO+ worse
    mean Δ = -0.9737
    rich / cycle_chain / proxy_hard
```

仍被有意排除的强正例包括：

```text
强正例未进主 K=18 extension:
    G-808, G-9
```

这些排除不是因为它们不重要，而是因为主集合已经需要同时覆盖：

```text
strong positive cases
stable matched contrasts
negative controls
neutral / no-room controls
computationally hard cases
complexity_bin / structural_type / landscape_regime diversity
```

若后续目标从“机制覆盖”转向“最强效应边界”，可以采用两个预注册替代方案：

```text
Historical K=16 stronger-negative swap, no longer the current default:
    replace G-658 with G-206
    replace G-730 with G-72
```

当前默认采用 K=18 extension。G-808/G-9 可作为 appendix sensitivity 或 K=20 extension，不应在看过 25-seed 结果后再临时加入主集合。

---

## 4. 建议目录结构

Repository 中只保存 compact、可审计、可复现的分析产物：

```text
K18_analysis/
├── README.md
├── configs/
│   ├── k18_topologies.csv
│   ├── k18_selection_manifest.json
│   ├── seed_extension_config.json
│   └── confirmation_candidate_policy.md
├── manifests/
│   ├── selection_source_commit.txt
│   ├── execution_git_head.txt
│   ├── execution_git_status.txt
│   ├── environment.txt
│   ├── screen_eval_hashes.csv
│   └── run_provenance.json
├── plans/
│   ├── screening_plan.json
│   ├── screening_jobs.csv
│   ├── commands_normal.txt
│   └── commands_long.txt
├── logs/
│   ├── materialization.log
│   ├── execution_summary.log
│   └── aggregation.log
├── summaries/
│   ├── k18_seed_extension_jobs.csv
│   ├── k18_25seed_jobs.csv
│   ├── k18_25seed_topology_summary.csv
│   ├── k18_stability_summary.json
│   └── k18_runtime_summary.csv
├── tables/
├── figures/
├── mechanism/
│   ├── G-269/
│   ├── G-398/
│   ├── ...
│   └── G-237/
└── reports/
    ├── k18_seed_stability.md
    ├── k18_mechanism_synthesis.md
    └── confirmation_recommendation.md
```

以下大文件必须保留在 scratch，不进入 Git：

```text
train banks
validation/test NPZ payloads
model checkpoints and weights
per-epoch metrics
materialized sample directories
per-job full evaluation directories
large candidate enumerations
temporary solver files
```

建议 scratch 结构：

```text
/local1/$USER/step3_k18_seed_extension_<date>/
├── run/
│   ├── data/
│   └── jobs/
└── logs/
```

Repository 仅保存 scratch 路径、hash、summary 和必要的小型表格，不保存 scratch 内容本身。

---

## 5. 术语边界与子实验边界

本 README 的主路线是：

```text
k18_seed_stability_v1:
    extend the locked K=18 set from 5 train seeds to 25 train seeds
    train_size = actual number of training samples sent to the trainer
    validation/test sets are external fixed eval sets
```

因此本 README 中的：

```text
train size = 50
```

表示 **50 个真实 training samples**，不包含 validation samples。

独立子实验：

```text
experiment_01_budget4to1/
```

使用不同语义：

```text
sample_size = training_size + validation_size
sample_size 50  -> training_size 40,  validation_size 10
sample_size 100 -> training_size 80,  validation_size 20
sample_size 500 -> training_size 400, validation_size 100
```

这两个路线不可混用：

```text
k18_seed_stability_v1:
    may reuse original screen eval hashes under Mode A

experiment_01_budget4to1:
    must use its own experiment version and new test bank
```

任何 aggregate、plan、job manifest 或 report 必须明确标记来自哪条路线。不得把 `train_size=50` 的 seed-stability/formal-confirmation rows 与 `sample_size=50, training_size=40` 的 4:1 learning-curve rows 合并为同一个 seed pool。

---

## 6. 全流程概览

```text
锁定 K=18 extension
    ↓
记录 selection-source commit 与 execution commit
    ↓
锁定 context generator、seed namespace 和 eval-set policy
    ↓
复用并验证原 screen validation/test sets
    ↓
物化 train seeds 6..25，或在新 eval bank 上重跑 1..25
    ↓
audit + dry run
    ↓
先执行 normal/long 各一个 smoke job
    ↓
执行全部 paired 2stage/SPO+ jobs
    ↓
完成性与 hash 审计
    ↓
生成 25-seed aggregate
    ↓
seed-stability stress tests
    ↓
18 topology 统一 mechanism diagnostics
    ↓
五个 sentinel 深度分析 + matched contrasts
    ↓
按预注册规则缩小到 8–10 个 topology
    ↓
生成全新的 confirm validation/test sets
    ↓
formal confirmation
```

---

## 7. Stage 0：锁定 selection 与 provenance

### 7.1 环境变量

```bash
cd /home/weikang/projects/UdeM-Intern/Exps

export PROJECT_ROOT="$PWD"
export ANALYSIS_DIR="$PROJECT_ROOT/surrogate_experiment_results/Step3/K18_analysis"
export STEP3_SCRIPTS="$PROJECT_ROOT/surrogate_experiment_results/Step3/scripts"
export REGIME="step2c_poly_d8_mult_eps050"

export EXP_ROOT="/local1/$USER/step3_k18_seed_extension_$(date +%Y%m%d)"
export RUN_ROOT="$EXP_ROOT/run"

mkdir -p "$ANALYSIS_DIR"/{configs,manifests,plans,logs,summaries,tables,figures,mechanism,reports}
mkdir -p "$RUN_ROOT"
```

### 7.2 生成 locked topology CSV

```bash
cat > "$ANALYSIS_DIR/configs/k18_topologies.csv" <<'CSV'
selection_rank,topology_id,selection_bucket
1,G-269,strong_positive
2,G-398,strong_positive
3,G-784,strong_positive
4,G-103,strong_positive
5,G-927,matched_positive
6,G-970,positive_hard
7,G-304,strong_positive
8,G-364,sparse_positive_anchor
9,G-730,coverage_positive
10,G-658,neutral_regime_positive_anchor
11,G-79,strong_negative_control
12,G-836,matched_negative_control
13,G-124,high_room_negative
14,G-396,high_room_neutral
15,G-670,no_room_neutral_hard
16,G-237,slow_structural_hard
17,G-206,strong_negative_extension
18,G-72,strong_negative_extension
CSV
```

selection CSV 一旦写入并审阅，不应根据新增 seed 的结果替换某个 topology。后续即使 topology 的 sign 改变，也应如实报告其 instability。

### 7.3 记录 Git 与环境

```bash
printf '%s\n' '779f55e88bc5578abafdb3c7e6a846946facdeb1'   > "$ANALYSIS_DIR/manifests/selection_source_commit.txt"

git rev-parse HEAD   > "$ANALYSIS_DIR/manifests/execution_git_head.txt"

git status --short   > "$ANALYSIS_DIR/manifests/execution_git_status.txt"

{
  date --iso-8601=seconds
  hostname
  python --version
  python - <<'PY'
import platform
print("platform:", platform.platform())
try:
    import numpy
    print("numpy:", numpy.__version__)
except Exception as exc:
    print("numpy: unavailable", exc)
try:
    import gurobipy
    print("gurobipy:", gurobipy.gurobi.version())
except Exception as exc:
    print("gurobipy: unavailable", exc)
PY
} > "$ANALYSIS_DIR/manifests/environment.txt"
```

`selection_source_commit` 和 `execution_git_head` 是两个不同概念：

```text
selection_source_commit:
    用于证明 K=18 extension 是从哪一版 screening evidence 中选出

execution_git_head:
    用于证明新 seed-extension / mechanism code 实际运行在哪一版
```

如果 execution commit 与 selection source commit 不同，必须在 `run_provenance.json` 中记录差异及原因，不得隐式混用。

---

## 8. Stage 1：锁定 seed-extension 协议

### 8.1 主设置

推荐的 seed-stability extension：

| 参数 | 值 |
|---|---:|
| protocol | `screen` |
| experiment label | `k18_seed_stability_v1` |
| topologies | locked K=18 extension |
| additional train seeds | 6..25 |
| final train seeds if combined | 1..25 |
| train size | 50 |
| validation size | 100 |
| test size | 1000 |
| max epochs | 3000 |
| metric stride | 1 |
| early-stop patience | 20 |
| early-stop min delta | 0.0001 |
| theta seed | 42 |
| Gurobi seed | 42 |
| additional jobs | 18 × 20 = 360 |

基于原 5-seed runtime，K=18 extension 对一个新增 seed 的 serial-equivalent time 约为：

```text
11309.2 seconds = 3.14 hours
```

20 个新增 seeds 约为：

```text
62.8 serial-equivalent hours
```

实际 wall time 由并行度、I/O 与 scheduler 决定。

### 8.2 Evaluation-set policy

Seed-stability 的目标是只改变 training replication。**因此本 README 采用 Mode A 作为主执行路线：复用原 160×5 screen 的 eval bank，只新增 train banks。**

```text
Mode A 主路线：
    reuse original screen_validation / screen_test / eval_manifest hashes
    build only train_seed=6..25 train banks
    run jobs against copied original eval_manifest
    combine old seeds 1..5 and new seeds 6..25 only if eval hashes match exactly

Mode B 备选路线：
    if original eval artifacts cannot be restored or hash-verified
    build a new K18 eval bank
    rerun seeds 1..25 on that new eval bank
    do not merge with old 5-seed job rows
```

这两条路线互斥。任何脚本如果会重新生成 `eval/`，在 Mode A 下都不能作为物化入口使用。

原 run 的 scratch 参考路径为：

```text
/local1/fuweik/step3_screen_160x5_e3000_20260620/run/
```

复制时只复制 selected K=18 extension 的 `eval/`，不要复制完整 78G scratch：

```bash
export ORIGINAL_RUN_ROOT="/local1/fuweik/step3_screen_160x5_e3000_20260620/run"
export REGIME="step2c_poly_d8_mult_eps050"

for g in G-269 G-398 G-784 G-103 G-927 G-970 G-304 G-364 G-730 G-658 G-79 G-836 G-124 G-396 G-670 G-237 G-206 G-72; do
  src="$ORIGINAL_RUN_ROOT/data/$REGIME/$g/eval/"
  dst="$RUN_ROOT/data/$REGIME/$g/eval/"
  test -d "$src"
  mkdir -p "$dst"
  rsync -a --checksum "$src" "$dst"
done
```

复制后记录 manifest hashes：

```bash
python - <<'PY'
import csv, json, os
from pathlib import Path

run_root = Path(os.environ["RUN_ROOT"])
regime = os.environ["REGIME"]
out = Path(os.environ["ANALYSIS_DIR"]) / "manifests" / "screen_eval_hashes.csv"
out.parent.mkdir(parents=True, exist_ok=True)
ids = "G-269 G-398 G-784 G-103 G-927 G-970 G-304 G-364 G-730 G-658 G-79 G-836 G-124 G-396 G-670 G-237 G-206 G-72".split()

rows = []
for topology_id in ids:
    path = run_root / "data" / regime / topology_id / "eval" / "eval_manifest.json"
    payload = json.loads(path.read_text())
    rows.append({
        "topology_id": topology_id,
        "eval_manifest_path": str(path),
        "validation_hash": payload["validation_hash"],
        "test_hash": payload["test_hash"],
        "protocol": payload["protocol"],
        "validation_namespace": payload["validation_namespace"],
        "test_namespace": payload["test_namespace"],
        "generator_config_hash": payload.get("generator_config_hash", ""),
        "topology_hash": payload.get("topology_hash", ""),
        "arc_order_hash": payload.get("arc_order_hash", ""),
        "feasible_set_hash": payload.get("feasible_set_hash", ""),
    })

with out.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(out)
PY
```

#### 原 eval scratch 不可用时

不要静默生成新 eval sets 后直接与原 seeds 1..5 合并。两种合法做法是：

```text
A. 恢复原 eval files，并验证 hash；
B. 生成一个新的 K18 screen eval bank，然后在该 bank 上统一重跑 seeds 1..25。
```

若采用 B：

```text
new eval bank + seeds 1..25 = 450 jobs
```

此时旧 5-seed 结果仅作为 selection provenance，不参与新的 25-seed pooled estimate。任何跨 eval hash 的合并都应被 aggregation 脚本拒绝。

#### Blocking item: original context config

若采用 Mode A 并希望和原 5-seed 结果严格合并，必须恢复原 160×5 run 使用的 context-generator config，并记录它的 hash。不要使用“reviewed config”或“等价 config”替代。

在执行 Stage 2 前，必须填写：

```text
ORIGINAL_CONTEXT_CONFIG = <absolute path to exact config used by original 160x5 screen>
ORIGINAL_GENERATOR_CONFIG_HASH = <hash copied from original train/eval manifests>
ORIGINAL_EXPERIMENT_VERSION = phase_b_prime_full_xy_screening_pilot_v1
```

推荐把原 config 复制成一个 locked artifact：

```bash
mkdir -p "$ANALYSIS_DIR/configs"
cp "$ORIGINAL_CONTEXT_CONFIG" "$ANALYSIS_DIR/configs/original_screen_context_generator.yaml"
export CONTEXT_CONFIG="$ANALYSIS_DIR/configs/original_screen_context_generator.yaml"

# Must match the original manifest hash. Fill this before running.
export ORIGINAL_GENERATOR_CONFIG_HASH="<FILL_FROM_ORIGINAL_MANIFEST>"
export ORIGINAL_EXPERIMENT_VERSION="phase_b_prime_full_xy_screening_pilot_v1"
test "$ORIGINAL_GENERATOR_CONFIG_HASH" != "<FILL_FROM_ORIGINAL_MANIFEST>"
test "$ORIGINAL_EXPERIMENT_VERSION" = "phase_b_prime_full_xy_screening_pilot_v1"
```

如果无法恢复该 exact config/hash，则改用 Mode B，不得合并旧 seeds 1..5。

---

## 9. Stage 2：物化、audit 与 plan

本阶段默认采用 **Mode A：只新增 train banks，复用原 eval bank**。

关键规则：

```text
Do not rebuild eval/ under Mode A.
Do not call wrappers that may regenerate eval sets.
Only build train_seed=6..25 train banks.
All run_one_job calls must point to copied original eval_manifest.json.
```

仓库中 `run_full_xy_screening_pilot.py` 会同时调用 train-bank builder 和 eval-set builder；它适合 tiny pilot 或 Mode B 新 eval bank 统一重跑，但 **不适合作为 Mode A 的 seeds 6..25 物化入口**。Mode A 应直接调用 `build_nested_train_bank.py`。

### 9.1 Context generator hash guard

必须使用 Stage 8.2 中恢复并锁定的 exact original context config：

```bash
export CONTEXT_CONFIG="$ANALYSIS_DIR/configs/original_screen_context_generator.yaml"
test -f "$CONTEXT_CONFIG"
test -n "$ORIGINAL_GENERATOR_CONFIG_HASH"
```

新增 train banks 完成后，每个 train-bank manifest 的 `generator_config_hash` 必须等于 `$ORIGINAL_GENERATOR_CONFIG_HASH`。任何 mismatch 都必须暂停。

### 9.2 只物化 seeds 6..25 的 train banks

```bash
python - <<'PY' 2>&1 | tee "$ANALYSIS_DIR/logs/materialize_train_banks_6_25.log"
import csv
import os
import subprocess
import sys
from pathlib import Path

project_root = Path(os.environ["PROJECT_ROOT"])
analysis_dir = Path(os.environ["ANALYSIS_DIR"])
run_root = Path(os.environ["RUN_ROOT"])
regime = os.environ["REGIME"]
context_config = Path(os.environ["CONTEXT_CONFIG"])
selected_csv = analysis_dir / "configs" / "k18_topologies.csv"
script = project_root / "surrogate_experiment_results" / "Step3" / "scripts" / "build_nested_train_bank.py"
experiment_version = os.environ["ORIGINAL_EXPERIMENT_VERSION"]

topology_root = project_root / "surrogate_experiment_results" / "Step3" / "pairs20_ndd2" / "data" / "topologies"
base_payload_dir = project_root / "dataset" / "processed" / "step3_pairs20_ndd2_step2c_poly_d8_mult_eps050_seed20260619"

rows = list(csv.DictReader(selected_csv.open(newline="", encoding="utf-8")))
rows = sorted(rows, key=lambda r: int(r["selection_rank"]))

for row in rows:
    topology_id = row["topology_id"]
    topology_path = topology_root / topology_id / "template.json"
    if not topology_path.exists():
        topology_path = topology_root / f"{topology_id}.json"
    base_payload = base_payload_dir / f"{topology_id}.json"
    if not base_payload.exists():
        base_payload = base_payload_dir / topology_id / "base_payload.json"
    assert topology_path.exists(), topology_path
    assert base_payload.exists(), base_payload

    for seed in range(6, 26):
        out = run_root / "data" / regime / topology_id / "train_banks" / f"train_seed={seed:06d}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, str(script),
            "--topology", str(topology_path),
            "--base-payload", str(base_payload),
            "--topology-id", topology_id,
            "--regime", regime,
            "--train-seed", str(seed),
            "--protocol", "screen",
            "--max-train-size", "50",
            "--prefix-sizes", "50",
            "--experiment-version", experiment_version,
            "--master-label-seed", "20260619",
            "--config", str(context_config),
            "--output", str(out),
            "--mode", "materialized",
        ]
        print("RUN", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=project_root, check=True)

print("materialized train banks for", len(rows), "topologies × 20 seeds")
PY
```

该步骤只应产生或更新：

```text
$RUN_ROOT/data/$REGIME/<topology>/train_banks/train_seed=000006.npz
...
$RUN_ROOT/data/$REGIME/<topology>/train_banks/train_seed=000025.npz
```

它不应改动：

```text
$RUN_ROOT/data/$REGIME/<topology>/eval/
```

### 9.3 Audit train banks against copied eval manifests

```bash
python - <<'PY' 2>&1 | tee "$ANALYSIS_DIR/logs/audit_train_eval_6_25.log"
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

project_root = Path(os.environ["PROJECT_ROOT"])
analysis_dir = Path(os.environ["ANALYSIS_DIR"])
run_root = Path(os.environ["RUN_ROOT"])
regime = os.environ["REGIME"]
context_config = Path(os.environ["CONTEXT_CONFIG"])
script = project_root / "surrogate_experiment_results" / "Step3" / "scripts" / "audit_fixed_topology_xy.py"
topology_root = project_root / "surrogate_experiment_results" / "Step3" / "pairs20_ndd2" / "data" / "topologies"
base_payload_dir = project_root / "dataset" / "processed" / "step3_pairs20_ndd2_step2c_poly_d8_mult_eps050_seed20260619"
selected_csv = analysis_dir / "configs" / "k18_topologies.csv"
rows = sorted(csv.DictReader(selected_csv.open(newline="", encoding="utf-8")), key=lambda r: int(r["selection_rank"]))
results = []

for row in rows:
    topology_id = row["topology_id"]
    topology_path = topology_root / topology_id / "template.json"
    if not topology_path.exists():
        topology_path = topology_root / f"{topology_id}.json"
    base_payload = base_payload_dir / f"{topology_id}.json"
    if not base_payload.exists():
        base_payload = base_payload_dir / topology_id / "base_payload.json"
    eval_manifest = run_root / "data" / regime / topology_id / "eval" / "eval_manifest.json"
    assert eval_manifest.exists(), eval_manifest

    for seed in range(6, 26):
        train_bank = run_root / "data" / regime / topology_id / "train_banks" / f"train_seed={seed:06d}.npz"
        cmd = [
            sys.executable, str(script),
            "--train-bank", str(train_bank),
            "--eval-manifest", str(eval_manifest),
            "--topology", str(topology_path),
            "--base-payload", str(base_payload),
            "--config", str(context_config),
            "--protocol", "screen",
        ]
        completed = subprocess.run(cmd, cwd=project_root, text=True, capture_output=True)
        status = {
            "topology_id": topology_id,
            "train_seed": seed,
            "returncode": completed.returncode,
            "stdout": completed.stdout[-1000:],
            "stderr": completed.stderr[-1000:],
        }
        print(json.dumps(status, ensure_ascii=False), flush=True)
        results.append(status)
        if completed.returncode != 0:
            raise SystemExit(f"audit failed for {topology_id} seed {seed}")

out = analysis_dir / "manifests" / "audit_train_eval_6_25.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
print(out)
PY
```

Expected：

```text
18 topologies × 20 seeds = 360 audit passes
0 generator_config_hash mismatches
0 validation_hash / test_hash mismatches
0 protocol namespace mismatches
```

### 9.4 生成正式 plan

`plan_full_xy_screening.py` 只生成 plan，不物化数据；在 Mode A 下可以使用它读取已经存在的 copied eval manifest 和 train bank prefix hashes。

```bash
python "$STEP3_SCRIPTS/plan_full_xy_screening.py" \
  --selected-topologies-csv "$ANALYSIS_DIR/configs/k18_topologies.csv" \
  --output-root "$RUN_ROOT" \
  --regime "$REGIME" \
  --train-seed-start 6 \
  --train-seed-count 20 \
  --train-size 50 \
  --validation-size 100 \
  --test-size 1000 \
  --max-topologies 18 \
  --protocol screen \
  --context-generator-config "$CONTEXT_CONFIG" \
  2>&1 | tee "$ANALYSIS_DIR/logs/planning_6_25.log"

cp "$RUN_ROOT/screening_plan.json" "$ANALYSIS_DIR/plans/"
cp "$RUN_ROOT/screening_jobs.csv" "$ANALYSIS_DIR/plans/"
cp "$RUN_ROOT/selected_topology_summary.csv" "$ANALYSIS_DIR/plans/"
```

### 9.5 生成 normal/long command files

长任务 topology：

```text
G-237, G-670, G-970
```

生成命令：

```bash
export KEP_PYTHON="${KEP_PYTHON:-$(command -v python)}"

python - <<'PY'
import csv
import os
import shlex
from pathlib import Path

project_root = Path(os.environ["PROJECT_ROOT"])
analysis_dir = Path(os.environ["ANALYSIS_DIR"])
run_root = Path(os.environ["RUN_ROOT"])
python_bin = os.environ["KEP_PYTHON"]
script = project_root / "surrogate_experiment_results" / "Step3" / "scripts" / "run_one_job.py"
long_ids = {"G-237", "G-670", "G-970"}

normal = []
long = []

with (run_root / "screening_jobs.csv").open(newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        cmd = [
            python_bin, str(script),
            "--train-bank", row["train_bank_path"],
            "--eval-manifest", row["eval_manifest_path"],
            "--topology-id", row["topology_id"],
            "--regime", row["regime"],
            "--protocol", "screen",
            "--train-seed", row["train_seed"],
            "--train-size", row["train_size"],
            "--output-dir", row["output_dir"],
            "--theta-seed", "42",
            "--gurobi-seed", "42",
            "--max-epochs", "3000",
            "--metric-stride", "1",
            "--early-stop-patience", "20",
            "--early-stop-min-delta", "0.0001",
            "--execute",
        ]
        line = shlex.join(cmd)
        (long if row["topology_id"] in long_ids else normal).append(line)

(analysis_dir / "plans" / "commands_normal.txt").write_text("\n".join(normal) + "\n", encoding="utf-8")
(analysis_dir / "plans" / "commands_long.txt").write_text("\n".join(long) + "\n", encoding="utf-8")
print("normal jobs:", len(normal))
print("long jobs:", len(long))
PY
```

预期：

```text
normal jobs = 300
long jobs   = 60
```

### 9.6 Mode B only: new eval bank + rerun seeds 1..25

只有在原 eval artifacts 或 exact context config/hash 无法恢复时，才使用 Mode B。Mode B 可以使用 `run_full_xy_screening_pilot.py` 或等价 wrapper 重新生成 K18 eval bank，但必须统一重跑 seeds 1..25：

```text
Mode B jobs = 18 topologies × 25 seeds = 450 jobs
old 5-seed screen rows = selection provenance only
no pooled estimate across old and new eval hashes
```

---

## 10. Stage 3：smoke 与批量执行

### 10.1 先执行两个 smoke jobs

执行一个 normal job：

```bash
sed -n '1p' "$ANALYSIS_DIR/plans/commands_normal.txt" | bash
```

执行一个 long job：

```bash
sed -n '1p' "$ANALYSIS_DIR/plans/commands_long.txt" | bash
```

检查每个 smoke job：

```text
paired_job_manifest.json exists
job_status.json status = success
2stage status = success
SPO+ status = success
evaluation status = success
test_summary.csv exists
train_prefix_hash matches
validation_hash matches
test_hash matches
theta_init is identical across methods
```

### 10.2 批量执行原则：tmux bounded worker，不使用 Slurm array

当前 Garnet/fuweik 环境下不要假设 `sbatch` 或 Slurm array 可用。推荐执行方式是：

```text
tmux session + bounded parallel launcher
normal concurrency: 2-4 jobs
long concurrency: 1-2 jobs
explicit thread limits
```

不要在 login node 上无限制并行运行 360 个 jobs。先限制 BLAS/OpenMP 线程，避免每个 job 过度占用 CPU：

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
```

建议拆成两个资源组：

```text
normal:
    15 topologies × 20 seeds = 300 jobs
    recommended parallelism = 2 to 4

long:
    3 topologies × 20 seeds = 60 jobs
    G-237, G-670, G-970
    recommended parallelism = 1 to 2
```

创建 bounded launcher：

```bash
cat > "$ANALYSIS_DIR/plans/run_bounded_commands.py" <<'PY'
#!/usr/bin/env python3
import argparse
import concurrent.futures as cf
import os
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--commands", required=True)
parser.add_argument("--parallel", type=int, required=True)
parser.add_argument("--log-dir", required=True)
args = parser.parse_args()

commands = [line.strip() for line in Path(args.commands).read_text().splitlines() if line.strip()]
log_dir = Path(args.log_dir)
log_dir.mkdir(parents=True, exist_ok=True)

env = os.environ.copy()
for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    env.setdefault(key, "1")

def run_one(item):
    idx, cmd = item
    log_path = log_dir / f"job_{idx:04d}.log"
    with log_path.open("w", encoding="utf-8") as log:
        log.write(cmd + "\n\n")
        log.flush()
        completed = subprocess.run(cmd, shell=True, executable="/bin/bash", text=True, stdout=log, stderr=subprocess.STDOUT, env=env)
    return idx, completed.returncode, str(log_path)

failed = []
with cf.ThreadPoolExecutor(max_workers=args.parallel) as pool:
    for idx, rc, log_path in pool.map(run_one, enumerate(commands, start=1)):
        print(f"[{idx}/{len(commands)}] rc={rc} log={log_path}", flush=True)
        if rc != 0:
            failed.append((idx, rc, log_path))

if failed:
    print("FAILED JOBS:")
    for item in failed:
        print(item)
    raise SystemExit(1)
PY
chmod +x "$ANALYSIS_DIR/plans/run_bounded_commands.py"
```

执行建议：

```bash
tmux new -s k18_normal
python "$ANALYSIS_DIR/plans/run_bounded_commands.py" \
  --commands "$ANALYSIS_DIR/plans/commands_normal.txt" \
  --parallel 3 \
  --log-dir "$ANALYSIS_DIR/logs/normal_jobs"
```

另开一个 tmux session 跑 slow group：

```bash
tmux new -s k18_long
python "$ANALYSIS_DIR/plans/run_bounded_commands.py" \
  --commands "$ANALYSIS_DIR/plans/commands_long.txt" \
  --parallel 1 \
  --log-dir "$ANALYSIS_DIR/logs/long_jobs"
```

不要为了让慢 topology 通过而单独改变 epochs、patience、objective、theta seed、Gurobi seed 或 eval set。


### 10.3 Failure handling

同一个 job 失败时：

1. 保留原 `paired_job_manifest.json` 与 `job_status.json`；
2. 记录 failure reason；
3. 修复环境或基础设施问题；
4. 使用完全相同的数据 hash 和参数重跑；
5. 不改变 topology、seed 或 eval set；
6. 不删除失败记录后假装首次成功。

若 failure 来自方法实现或数据不一致，应暂停整个 batch，而不是只重跑失败 topology。

---

## 11. Stage 4：完成性与完整性审计

### 11.1 完成标准

Mode A（追加 seeds 6..25）：

```text
expected additional jobs = 360
success = 360
failed = 0
skipped = 0
```

Mode B（新 eval bank 上重跑 seeds 1..25）：

```text
expected jobs = 450
success = 450
failed = 0
skipped = 0
```

### 11.2 Job integrity

每个 job 必须验证：

```text
audit_status = passed
2stage_status = success
spoplus_status = success
evaluation_status = success
return_code = 0
train_size = 50
protocol = screen
```

每个 topology 内：

```text
all train seeds use the same validation_hash
all train seeds use the same test_hash
2stage and SPO+ use the same train_prefix_hash
2stage and SPO+ use the same theta initialization
```

不同 topology 的 eval hash 可以不同，因为 topology fixed set 不同。

### 11.3 不允许的状态

```text
mixing screen and confirm namespaces
mixing old and new eval hashes in one aggregate
using test data for checkpoint selection
changing practical thresholds after seeing results
dropping a selected topology because its new result is inconvenient
```

---

## 12. Stage 5：结果聚合

### 12.1 Required outputs

Job-level：

```text
summaries/k18_seed_extension_jobs.csv
    360 rows for seeds 6..25

summaries/k18_25seed_jobs.csv
    450 rows for 18 topologies × 25 seeds
    only when eval hashes are identical within each topology
```

Topology-level：

```text
summaries/k18_25seed_topology_summary.csv
    18 rows
```

### 12.2 每个 topology 至少聚合的指标

```text
train_seed_count
success_count
failed_count

positive_seed_count
zero_seed_count
negative_seed_count
win_rate
loss_rate

mean_2stage_gap
median_2stage_gap
std_2stage_gap

mean_spoplus_gap
median_spoplus_gap
std_spoplus_gap

mean_spoplus_improvement_gap
median_spoplus_improvement_gap
std_spoplus_improvement_gap
min_spoplus_improvement_gap
max_spoplus_improvement_gap
IQR_spoplus_improvement_gap

mean_runtime_seconds
median_runtime_seconds
max_runtime_seconds

validation_hash
test_hash
```

必须合并原 structural metadata：

```text
complexity_bin
structural_type
landscape_regime
num_exchange_candidates
num_cycles_total
num_chains_total
candidate_conflict_density
mean_candidates_per_vertex
num_distinct_oracle_solutions
oracle_solution_entropy
dominant_oracle_solution_fraction
fraction_linear_proxy_differs_from_oracle
mean_linear_proxy_normalized_gap_to_oracle
median_top1_top2_margin
mean_pairwise_oracle_jaccard
```

### 12.3 Zero 定义

原 5-seed summary 中存在大量 exact zero。新 aggregation 必须在分析前固定 zero rule，并在 summary 中记录，例如：

```text
zero iff abs(Δ) <= 1e-12
```

不得根据结果调整 tolerance。若要与旧 summary 完全复现，应优先复用旧 aggregation 的相同比较逻辑。

---

## 13. Stage 6：seed-stability 分析

### 13.1 5-seed vs 25-seed

对每个 topology 比较：

```text
mean Δ
median Δ
win_rate
negative_seed_count
effect sign
rank by mean
rank by median
```

总体报告：

```text
Spearman rank correlation: 5-seed mean rank vs 25-seed mean rank
bucket retention rate
number of sign reversals
number of unanimous-5 cases that lose 0.80 win-rate
```

### 13.2 预先锁定的筛选规则

在查看 25-seed aggregate 的分类结果前，必须先在 config 中冻结 practical threshold。这个值不能是 `TBD`，也不能在看过 25-seed outcome 后再修改。

必须新建并填写：

```text
$ANALYSIS_DIR/configs/k18_classification_policy.json
```

模板：

```json
{
  "tau_delta": "FILL_NUMERIC_VALUE_BEFORE_CLASSIFICATION",
  "win_rate_positive": 0.80,
  "win_rate_harmful": 0.20,
  "locked_before_25seed_classification": true,
  "notes": "tau_delta must be numeric before any robust/neutral/harmful labels are generated."
}
```

Blocking rule：

```text
Raw aggregation may compute mean/median/win-rate.
No robust-positive / harmful / neutral classification may be produced while tau_delta is missing or non-numeric.
Any classification script must refuse to run if tau_delta is still FILL/TBD.
```

推荐描述性分类：

```text
robust positive candidate:
    median Δ > tau_delta
    and win_rate >= 0.80

harmful control candidate:
    median Δ < -tau_delta
    or win_rate <= 0.20

neutral candidate:
    abs(median Δ) <= tau_delta
    and uncertainty interval overlaps 0
```

这仍是 screening classification，不是 formal significance claim。


### 13.3 必做 sensitivity checks

```text
mean vs median ranking
20% trimmed mean ranking
leave-one-seed-out
leave-five-seeds-out
bootstrap interval stability
sign-rate interval
practical-threshold sensitivity
with/without three slowest topologies in runtime summaries
```

所有 18 个 topology 都必须在报告中出现，即使其原角色不再成立。

---

## 14. Stage 7：统一 mechanism diagnostics

每个 topology 执行同一组 diagnostics，避免只对“好看的”案例深入分析。

### 14.1 所有 18 个 topology 的统一分析

```text
1. predicted top-M candidate enumeration
2. true oracle top-M landscape
3. predicted-vs-oracle rank reversal
4. top-K promotion and deep-reranking events
5. selected candidate overlap / Jaccard
6. critical-edge signed contributions
7. candidate conflict participation
8. feasible-set ambiguity
9. decision multiplicity / tie diagnostics
10. runtime decomposition
```

### 14.2 Seed policy

低成本 aggregate diagnostics：

```text
all 25 seeds
```

人工深度可视化固定 seeds：

```text
1, 5, 10, 15, 20, 25
```

可额外展示 best/median/worst seed，但这些必须明确标记为 post-hoc exemplars，不能用于主要 claim。

### 14.3 每个 topology 的 compact output

```text
mechanism/<topology_id>/
├── summary.json
├── seed_metrics.csv
├── candidate_landscape_summary.csv
├── rank_reversal_summary.csv
├── overlap_summary.csv
├── critical_edge_summary.csv
├── runtime_summary.csv
└── figures/
```

只提交 compact summaries。完整 top-M candidate payload、per-instance arrays 和 solver traces 保留在 scratch。

---

## 15. Stage 8：重点机制问题

### 15.1 Positive cases

需要区分：

```text
clean top-K promotion
deep reranking
baseline 2stage catastrophic miss
SPO+ improvement from a small number of critical edges
SPO+ improvement from broad coefficient reshaping
```

重点 topology：

```text
G-269, G-398, G-784, G-103, G-927, G-970
```

### 15.2 Negative controls

需要回答：

```text
SPO+ 是否过度推动错误 candidate
是否因接近 decision boundary 而出现 rank reversal
是否牺牲了预测稳定性
是否存在低 margin / 高 ambiguity
是否出现一小部分极差错误驱动 mean
```

重点 topology：

```text
G-79, G-206, G-72, G-836, G-124
```

### 15.3 Neutral controls

区分两类 neutral：

```text
G-670:
    no-room neutral
    baseline gap 接近 0
    两方法无差异并不意外

G-396:
    high-room neutral
    baseline gap 很高
    两方法仍完全相同，需要解释为何训练没有改变 decision
```

### 15.4 Computational hardness

重点分析：

```text
G-237:
    highest selected conflict density
    slowest selected topology

G-670:
    no-room but very slow

G-970:
    extreme + slow + large positive effect
```

Runtime mechanism 至少拆分为：

```text
data materialization
2stage training
SPO+ training
oracle calls
evaluation
candidate enumeration
```

---

## 16. Stage 9：K=18 extension 综合报告

`reports/k18_mechanism_synthesis.md` 至少包含：

1. Run completion and provenance；
2. 5-seed 与 25-seed 稳定性；
3. 18 个 topology 的完整结果表；
4. positive / negative / neutral / hard 四类综合；
5. matched contrasts；
6. sentinel deep dives；
7. runtime 与 scalability；
8. borderline cases；
9. screening-selection bias；
10. formal confirmation recommendation。

禁止只展示平均 improvement 最高的案例。

---

## 17. Stage 10：缩小到 formal confirmation set

完成 25-seed stability 和 mechanism analysis 后，再从锁定 K=18 extension 中缩小到约 8–10 个 topology。

建议 confirmation composition：

```text
3–4 robust helpful cases
2–3 neutral controls
2–3 harmful controls
at least one matched positive/negative contrast
at least one computationally hard but scientifically valuable case
```

在任何 confirmation training 开始前锁定：

```text
configs/confirmation_topologies.json
```

Formal confirmation 必须使用新的 namespaces：

```text
confirm_validation
confirm_test
```

不得复用 screen validation/test data。

建议 formal setting：

```text
train_sizes = 50, 100, 500
confirm_train_seeds = 1..1000
validation_size = 1000
test_size = 5000
methods = 2stage, SPO+
theta_seed = 42
gurobi_seed = 42
nested_train_sets = true
```

`run_confirmation.py` 当前是 plan-only 工具。生成 plan 后仍需单独的 execution launcher。

---

## 18. Reproducibility rules

### 18.1 必须记录

```text
selection source commit
execution commit
git status
context-generator config and hash
topology template hash
arc-order hash
feasible-set hash
train-bank hash
train-prefix hash
validation hash
test hash
theta seed and initialization
Gurobi seed
software versions
host and timestamps
```

### 18.2 Paired comparison

同一个 topology / train seed / train size 下，2stage 与 SPO+ 必须共享：

```text
same train bank
same train prefix
same validation set
same test set
same theta initialization
same topology and candidate ordering
```

只有 training objective 不同。

### 18.3 数据使用边界

```text
training data:
    parameter fitting

validation data:
    checkpoint selection / early stopping

screen test data:
    screening evaluation and K18 stability only

confirm test data:
    final locked confirmation only
```

任何 test set 都不能用于 checkpoint selection 或 hyperparameter tuning。

---

## 19. Deliverables 与完成条件

### 19.1 Seed-extension complete

- [ ] K=18 extension CSV 和 selection manifest 已锁定  
- [ ] execution commit 与环境已记录  
- [ ] eval hash policy 已锁定  
- [ ] 360 个追加 jobs 全部完成，或 450 个统一重跑 jobs 全部完成  
- [ ] failed = 0  
- [ ] 所有 paired hashes 审计通过  
- [ ] 25-seed job-level 和 topology-level summaries 已生成  
- [ ] 所有 18 个 topology 均被报告  

### 19.2 Mechanism analysis complete

- [ ] 18 个 topology 完成统一 diagnostics  
- [ ] 五个 sentinel 完成深度分析  
- [ ] 五组 matched contrasts 完成  
- [ ] negative controls 未被忽略  
- [ ] neutral cases 区分 no-room 与 high-room  
- [ ] runtime-hard cases 完成 runtime decomposition  
- [ ] 大型 scratch artifacts 未提交到 Git  

### 19.3 Confirmation handoff complete

- [ ] confirmation inclusion/exclusion rule 已记录  
- [ ] 8–10 个 topology 已锁定  
- [ ] 新 confirm validation/test sets 已生成  
- [ ] confirmation job plan 已审计  
- [ ] screen 与 confirmation claim 完全分离  

---

## 20. 当前 K=18 extension 的解释状态

在新增 seeds 完成前，下列标签仅是 selection roles：

```text
strong positive
negative control
neutral
hard
```

它们不是最终事实。新增 seeds 后可能发生：

```text
win rate 下降
median 接近 0
sign reversal
runtime rank 改变
原 matched contrast 不再保持
```

这些变化本身就是应报告的结果，不应通过替换 topology 来维持原叙事。

---

## 21. 最终原则

```text
screen broadly
lock K=18 extension
test seed stability
analyze mechanisms uniformly
preserve negative and neutral controls
confirm narrowly on independent data
report every locked topology
```
