我赞同导师的建议，而且我觉得这正好是你现在 Step2 后面最应该补的一组实验。你现在已有的 Step2 结果已经说明：在固定 seed / 固定 graph pool / 固定 subset 的情况下，高 degree misspecification 下 SPO+ 和 FY 能明显改善 2stage 的 synthetic-label decision gap。导师现在问的是另一个层面的稳健性问题：

> 这个结论是不是依赖某一个固定 seed？如果换不同 graph seed / train subset seed / label seed，SPO+ 是否仍然在 MSE 高 regret 的 setting 中改善 decision quality？

这不是对你当前结果的否定，而是一个很自然的下一步：**把 single-seed evidence 变成 distributional evidence**。

你的想法“重做新的 Step2b 和 Step2c，只用 50 个训练图，只比较 2stage MSE 和 SPO+，用尽可能多 seed”方向是对的，但我建议稍微调整成一个更严谨、更能回应导师的实验设计。

---

## 我建议的总体原则

不要只做“挑一个 MSE regret 高的 seed，然后展示 SPO+ 好”。那样容易被理解成 cherry-picking。更好的设计是分成两个层次：

```text
A. Unbiased multi-seed variability experiment
   随机抽很多 seeds，不筛选，报告 MSE / SPO+ / improvement 的分布。

B. Hard-MSE stress-test experiment
   先用 MSE 快速筛出 MSE regret 高的 seeds 或 graphs，
   再检查 SPO+ 是否在这些 hard cases 上改善。
```

这样你既能回答：

> 平均意义上，SPO+ 是否稳定优于 MSE？

也能回答导师提出的：

> 在 MSE regret 高的困难 graph / seed 上，SPO+ 是否能补救？

这两者一定要分开报告。A 是 unbiased robustness evidence；B 是 conditional stress test。

---

## 我会怎么设计这个实验

### 实验目标

新的实验问题可以写成：

> Under repeated resampling of training graphs and/or graph-generation seeds, how variable is 2stage MSE decision quality, and does SPO+ systematically improve over MSE, especially in settings where MSE has high decision gap?

重点不是再证明 Step2b/2c 的所有结论，而是回答 variability 和 hard-case recovery。

---

## 建议只选 Step2b/Step2c 的高难度 regime

你不需要把 Step2a、Step2b d1、d2 全部重跑。导师关心的是 MSE regret 高的情况，而你已有结果显示高 degree 才是 MSE 明显失效的地方。

我建议第一轮只做：

```text
Step2b d8
Step2c d8, epsilon_bar = 0.5
```

如果算力允许，再加一个中等难度：

```text
Step2b d4
Step2c d4
```

这样实验解释会非常清楚：

```text
d4 = moderate misspecification
d8 = high misspecification
Step2b = deterministic nonlinear misspecification
Step2c = nonlinear + multiplicative noise
```

不要一开始把 Step2a / d1 全部放进来，否则计算量变大，结论反而被 near-linear sanity cases 稀释。

---

## train size 固定 50 是合理的

我赞成你固定：

```text
train_size = 50
```

理由是：

1. 小样本下 seed variability 最大，最容易看到 boxplot 分布；
2. 2stage 和 SPO+ 都更快；
3. 导师的建议核心就是 resampling variability，而 n=50 是最敏感的 setting；
4. 你已有 Step2 结果里 n=50 已经足够暴露 high-degree misspecification。

但我建议后面加一个轻量确认：

```text
train_size = 200
```

不一定第一轮跑。等 n=50 的 boxplots 出来后，如果结论明显，再补 n=200 看趋势是否保留。

---

## seed 要分清楚，不要混在一起

你现在要小心：所谓 “seed” 可以指很多东西。

至少有四类随机性：

```text
1. graph_seed
   生成 compatibility graphs / raw graph pool 的 seed

2. subset_seed
   从 main pool 中抽 train_size=50 的 seed

3. label_seed
   Step2c multiplicative noise 的 seed；Step2b noiseless label 可以没有或固定

4. theta_seed / training_seed
   SPO+ initialization / optimizer randomness
```

我建议第一轮不要同时随机化所有东西，否则很难解释 variability 来自哪里。

---

## 第一轮最干净：固定 graph pool，只重采样 train subset

这是最快、最容易先出 boxplots 的版本。

你已有的 Step2 processed datasets 是：

```text
step2b_poly_d8_main2000_seed20260523
step2b_poly_d8_val2000_seed20260523
step2b_poly_d8_unseen10000_seed20260523

step2c_poly_d8_mult_eps050_main2000_seed20260523
step2c_poly_d8_mult_eps050_val2000_seed20260523
step2c_poly_d8_mult_eps050_unseen10000_seed20260523
```

第一轮可以固定这些 dataset，然后只变：

```text
subset_seed ∈ {0, 1, ..., 49}
train_size = 50
theta_seed fixed, e.g. 42
```

每个 subset_seed 下：

```text
train 2stage MSE
train SPO+ from same init protocol
evaluate both on same validation / heldout / unseen10000
```

这个实验回答：

> 在同一个 graph distribution 和 label regime 下，不同 50-graph training subset 会导致多大的 variability？SPO+ 的 paired improvement 是否稳定？

这已经足以做导师要求的 boxplots。

---

## 第二轮：不同 graph generation seed

导师说 “graph is fixed to a random seed; try different seeds” 更像是在说 graph pool seed 也要变。

这可以作为第二层实验：

```text
graph_seed ∈ {s1, ..., sK}
for each graph_seed:
    generate new main/val/unseen processed datasets under Step2b d8 / Step2c d8
    sample 50 training graphs
    train 2stage and SPO+
    evaluate on matching unseen dataset
```

这个更 expensive，因为你要生成新 graph pools。如果你现在的 graph-generation pipeline 很方便，可以做 K=10 或 K=20。否则先用 subset resampling，把初步结论给导师看。

我会把两轮命名成：

```text
Experiment 1: train-subset resampling on fixed Step2 graph pools
Experiment 2: graph-seed resampling stress test
```

---

## 关于“选择 MSE high regret 的 seed”：必须做成两阶段

导师建议：

> choose the seed that produces high regret for MSE, then check if SPO+ improves.

这个思路很好，但统计上有一个风险：如果你用最终 test performance 来挑 seed，再在同一个 test set 上说 SPO+ improves，这就是 post-hoc selection，会有偏。

所以我建议这样做：

### Hard-MSE screening protocol

对很多 seeds 先只跑 MSE：

```text
N_screen = 100 seeds
method = 2stage MSE only
train_size = 50
regime = Step2b d8 / Step2c d8
screen_metric = validation decision gap or heldout400 gap
```

然后按 MSE regret 排序，选：

```text
top 20% high-MSE-regret seeds
middle 20% medium-MSE-regret seeds
bottom 20% low-MSE-regret seeds
```

接着只对这些 seeds 跑 SPO+，并在 **unseen10000** 上评估。

这样你可以画：

```text
SPO+ improvement distribution conditional on MSE difficulty stratum
```

这就非常契合导师的想法，而且更严谨。

不要只选 top 1 seed。最好选 top 10 或 top 20 seeds，画 boxplot。这样不是 anecdote，而是 conditional distribution。

---

## 我建议的最小可行版本

如果你想快速开始，我建议这个最小版本：

```text
Regimes:
  Step2b d8
  Step2c d8 eps050

Train size:
  50

Seeds:
  subset_seed = 0..49
  theta_seed = 42 fixed
  label_seed fixed for first round

Methods:
  2stage selected by validation MSE
  SPO+ selected by validation SPO+ loss
  optional: SPO+ selected by validation decision gap as diagnostic

Evaluation:
  heldout400
  unseen10000

Outputs:
  one summary row per regime/seed/method
  one paired row per regime/seed with:
    gap_2stage
    gap_spoplus
    improvement = gap_2stage - gap_spoplus
    relative_improvement = improvement / gap_2stage
```

这个就是：

```text
2 regimes × 50 seeds × 2 methods = 200 trained/evaluated model rows
```

因为 train_size 只有 50，SPO+ 应该还可接受。

---

## 如果算力更紧张

可以改成导师建议的 screened strategy：

```text
1. Run MSE for 100 subset seeds.
2. Pick:
   top 20 high-MSE seeds
   middle 20 seeds
   bottom 20 seeds
3. Run SPO+ only on those 60 seeds.
```

这样你仍然有 boxplots，而且能直接回答：

> Does SPO+ improve when MSE is bad?

---

## Box plots 应该怎么画

我建议至少画 5 组图。

### Figure 1: MSE regret variability by seed

```text
x-axis: regime
y-axis: MSE mean normalized decision gap
boxplot over seeds
```

这回答：

> 不同 train subset / graph seed 下，MSE 的 regret variability 有多大？

如果 Step2c d8 的 MSE boxplot 更高、更宽，说明它确实更难且更不稳定。

---

### Figure 2: 2stage vs SPO+ gap distribution

```text
x-axis: method
y-axis: mean normalized decision gap
facet: regime
boxplot over seeds
paired lines optionally connecting same seed
```

这回答：

> SPO+ 的 absolute performance distribution 是否低于 MSE？

这张图最直接。

---

### Figure 3: Paired improvement over MSE

```text
x-axis: regime
y-axis: gap_MSE - gap_SPO+
boxplot over seeds
horizontal line at 0
```

这是最重要的一张。

解释：

```text
positive = SPO+ improves over MSE
negative = SPO+ worse than MSE
```

如果 median > 0 且大部分箱体在 0 上方，结论非常强。

---

### Figure 4: Improvement vs MSE baseline gap

```text
x-axis: MSE mean normalized gap
y-axis: SPO+ improvement over MSE
one point per seed
color: regime
```

这正好对应导师的想法。

如果看到：

```text
MSE gap 越高，SPO+ improvement 越大
```

那就是非常漂亮的结果：

> SPO+ is most useful exactly when MSE is bad.

---

### Figure 5: Hard-MSE stratum boxplot

先按 MSE gap 分组：

```text
low / medium / high MSE regret
```

然后画：

```text
x-axis: MSE difficulty stratum
y-axis: SPO+ improvement over MSE
facet: regime
```

这会非常直接地回答导师：

> 在 MSE 高 regret 的 seeds 上，SPO+ 是否改善？

---

## 指标建议

不要只看 raw gap。至少记录：

```text
mean_decision_gap
mean_normalized_gap
median_normalized_gap
paired_improvement_over_2stage
relative_improvement = paired_improvement / mse_gap
fraction_graphs_improved
```

多 seed summary 里最重要的是：

```text
seed-level gap_MSE
seed-level gap_SPO+
seed-level delta = gap_MSE - gap_SPO+
```

然后跨 seed 报告：

```text
mean(delta)
median(delta)
std(delta)
25/75 percentiles
fraction(delta > 0)
bootstrap CI for mean(delta)
Wilcoxon signed-rank test optional
```

---

## “生成 MSE high regret graphs” 还有一个更快的做法

除了重新生成 seeds，你其实可以用现有 per-graph records 做一个 immediate analysis：

```text
在当前 unseen10000 中，按 2stage per-graph normalized gap 排序；
选 top 10% hardest graphs；
比较 SPO+ 在这些 graphs 上的 gap 是否更低。
```

这不需要重新训练，马上就能做。
不过它是 post-hoc test-set stratification，只能作为 diagnostic，不应作为主实验结论。

它可以作为导师建议的补充图：

```text
Improvement on top-decile MSE-hard graphs
```

如果这个结果也支持 SPO+，会很有说服力。

---

## 我不建议一开始“重新做完整 Step2b/Step2c 数据集”

你说“重做新的 Step2b 和 Step2c”，我建议不要理解成重新做完整的：

```text
d1,d2,d4,d8 × main/val/unseen × 4 train sizes × FY/SPO+
```

这会太大，而且偏离导师的核心问题。

更好的新实验命名是：

```text
Step2_seed_sweep
```

或者：

```text
Step2_resampling
```

它不是取代 Step2，而是补充 Step2：

```text
Original Step2:
  broad regime sweep, one seed, 72 training runs, 180 evaluations.

New Step2 resampling:
  narrow hard-regime sweep, many seeds, boxplot variability.
```

这样 narrative 很清楚。

---

## 我建议的目录结构

可以新建：

```text
surrogate_experiment_results/Step2_resampling/
```

里面放：

```text
README.md

run_phase0_hard_graph_diagnostic.py
run_mse_screen.py
run_spoplus_on_selected_seeds.py
run_full_seed_sweep.py

configs/
  step2b_d8_n50.json
  step2c_d8_n50.json

results/
  phase0_hard_graph_diagnostic_summary.csv
  phase0_top_decile_graphs.csv
  mse_screen_summary.csv
  seed_sweep_summary.csv
  hard_mse_selected_summary.csv

plot_results/
  phase0_top_decile_improvement.png
  01_mse_gap_boxplot.png
  02_mse_vs_spoplus_gap_boxplot.png
  03_paired_improvement_boxplot.png
  04_improvement_vs_mse_gap_scatter.png
  05_hard_mse_stratum_boxplot.png
```

README 里写清楚：

```text
This experiment is a resampling robustness and hard-MSE stress test, not a replacement for Step2.
```

---

## 具体分阶段计划

### Phase 0: 不训练，先用现有 per-graph 做 hard-graph diagnostic

如果你本地还有 per-graph CSV，先做：

```text
current Step2b d8, Step2c d8
train_size=1200 or 50
top 10% MSE-hard graphs
compare SPO+ improvement
```

这可以一天内完成，作为 sanity diagnostic。

当前 Phase 0 diagnostic 可以直接运行：

```bash
MPLCONFIGDIR=/tmp/matplotlib \
/home/weikang/miniconda3/envs/KEPs/bin/python \
  surrogate_experiment_results/Step2_resampling/run_phase0_hard_graph_diagnostic.py
```

默认设置：

```text
regimes = {Step2b d8, Step2c d8 eps050}
train_size = {50, 1200}
baseline = 2stage selected by validation MSE
candidate = SPO+ selected by validation SPO+ loss
evaluation = existing unseen10000 per-graph CSV
hard set = top 10% graphs ranked by 2stage normalized gap
```

输出：

```text
results/phase0_hard_graph_diagnostic_summary.csv
results/phase0_top_decile_graphs.csv
plot_results/phase0_top_decile_improvement.png
```

这个结果仍然是 post-hoc test-set stratification，只作为 sanity diagnostic，
不作为无偏主结论。

---

### Phase 1: fixed graph pool + subset resampling

```text
Blocks:
  Step2b
  Step2c eps050

Degrees:
  d = 1, 2, 4, 8

Train:
  train_size = 50
  subset_seed = 0..49

Validation:
  matching val2000 dataset

Evaluation:
  matching unseen10000 dataset

Methods:
  2stage MSE
  DFL SPO+

Fixed seeds:
  theta_seed = 42
  label_seed = 20260523
  gurobi_seed = 42
```

严格说，这一阶段是：

```text
Phase 1 = fixed graph pool + subset resampling
```

也就是固定现有 Step2 processed datasets，只变 `subset_seed`。每个
`subset_seed` 抽一个 `train_size=50` 的训练子集，并使用同一个对应的
`val2000` dataset 做 checkpoint selection。

更清楚的表述是：

> For each regime and degree, I will sample 50 different training subsets using
> `subset_seed=0..49`, each with `train_size=50`.

实验规模为：

```text
2 blocks x 4 degrees x 50 seeds x 2 methods = 800 training runs
```

这是主结果，而且它仍然不是 Phase 2 或 Phase 3：

```text
Not Phase 2:
  It does not first screen high-MSE-regret seeds and then run SPO+ by difficulty
  stratum.

Not Phase 3:
  It does not regenerate graph pools under different graph_seed values.

Expanded Phase 1:
  It keeps graph pools fixed and studies variability from train-subset
  resampling.
```

第一轮先固定 `theta_seed`、`label_seed` 和 `gurobi_seed`，避免把
training initialization、Step2c label-noise realization、Gurobi tie-breaking
等随机性混进 subset-resampling variability。Step2b 是 noiseless polynomial
label regime，因此 `label_seed` 在 Step2b 中主要是 metadata / naming；Step2c
的 `label_seed` 暂时固定，变 label-noise seed 可以作为后续单独实验。

Phase 1 driver:

```text
surrogate_experiment_results/Step2_resampling/run_phase1_subset_resampling.py
```

这个脚本不复制训练逻辑，而是为每个 `(regime, subset_seed)` 调用现有
`surrogate_experiment_results/Step1c/run_step1c.sh`。每个 wrapper job 会完成：

```text
1. train 2stage MSE
2. train DFL SPO+
3. evaluate both methods on the heldout400 split
4. evaluate the three standard checkpoints on matching unseen10000
```

因此脚本层面的 job 数是：

```text
8 regimes x 50 subset seeds = 400 wrapper jobs
```

对应 README 设计中的：

```text
400 wrapper jobs x 2 trained methods = 800 training runs
```

Garnet 上建议先做 dry-run：

```bash
ssh cirrelt
cd /local1/fuweik/UdeM-Intern
source configs/runtime/garnet.env

"$KEP_PYTHON" \
  surrogate_experiment_results/Step2_resampling/run_phase1_subset_resampling.py \
  --dry_run \
  --regimes step2b_poly_d1 \
  --seed_count 1 \
  --workers 1
```

然后做一个最小 real smoke：

```bash
"$KEP_PYTHON" \
  surrogate_experiment_results/Step2_resampling/run_phase1_subset_resampling.py \
  --regimes step2b_poly_d1 \
  --seed_count 1 \
  --workers 1 \
  --epochs_2stage 2 \
  --epochs_spoplus 2 \
  --metric_stride 1 \
  --train_graph_limit 2 \
  --validation_limit 3 \
  --test_limit 3 \
  --unseen_graph_limit 3 \
  --output_root /tmp/step2_phase1_smoke_runs \
  --split_root /tmp/step2_phase1_smoke_splits \
  --log_root /tmp/step2_phase1_smoke_logs \
  --manifest_path /tmp/step2_phase1_smoke_manifest.csv \
  --status_path /tmp/step2_phase1_smoke_status.csv
```

正式跑完整 Phase 1 时建议在 `tmux` 里启动：

```bash
tmux new -s step2_phase1
cd /local1/fuweik/UdeM-Intern
source configs/runtime/garnet.env

"$KEP_PYTHON" \
  surrogate_experiment_results/Step2_resampling/run_phase1_subset_resampling.py \
  --workers 2
```

Garnet 是共享 CPU 机器，而且每个 SPO+ epoch 会触发大量 Gurobi solves。
性能策略不是把所有核心一次性打满，而是用 job-level parallelism：

```text
default workers = 2
default per-worker BLAS threads = 1
```

如果机器负载、内存和 Gurobi license 都稳定，可以逐步尝试：

```bash
--workers 3
--workers 4
```

不要一开始开很高并发，否则多个 Gurobi-heavy jobs 会互相抢 CPU / license /
内存，整体 wall time 可能反而变差。

默认输出：

```text
surrogate_experiment_results/Step2_resampling/phase1_runs/
surrogate_experiment_results/Step2_resampling/logs/phase1/
surrogate_experiment_results/Step2_resampling/results/phase1_job_manifest.csv
surrogate_experiment_results/Step2_resampling/results/phase1_job_status.csv
```

每个 job 的完成标记是三份 standard Step1c weights 加
`metrics/test_summary.csv`。如果没有设置 `--skip_unseen_eval`，完成标记还会
额外要求 `metrics/unseen10000_summary.csv`。脚本默认 `skip_completed=True`，
所以同一条命令可以安全重跑；已完成的 `(regime, subset_seed)` 会被跳过。

2026-05-29 的 overnight Phase 1 run 使用训练优先模式：

```bash
"$KEP_PYTHON" \
  surrogate_experiment_results/Step2_resampling/run_phase1_subset_resampling.py \
  --workers 2 \
  --skip_unseen_eval
```

这个模式保留 Step1c 内部的 heldout400 evaluation，但跳过 matching
`unseen10000` final evaluation。也就是说 overnight 目标是：

```text
400 wrapper jobs:
  train 2stage MSE
  train DFL SPO+
  select checkpoints on matching val2000
  evaluate selected checkpoints on heldout400

deferred:
  evaluate selected checkpoints on matching unseen10000
```

把 `unseen10000` 拆到最后统一做更干净：训练 worker 不会被 final test 占住，
而且后续可以按 regime 批量加载同一个 unseen10000 dataset 后评估 50 个
subset seeds。

如果 Garnet 上还没有 Step2 processed datasets，先从本地同步：

```bash
rsync -av --info=progress2 \
  dataset/processed/step2b_poly_*_seed20260523 \
  dataset/processed/step2c_poly_*_seed20260523 \
  cirrelt:/local1/fuweik/UdeM-Intern/dataset/processed/

rsync -av --info=progress2 \
  surrogate_experiment_results/Step2_resampling/ \
  cirrelt:/local1/fuweik/UdeM-Intern/surrogate_experiment_results/Step2_resampling/
```

---

### Phase 2: MSE screening + hard-seed SPO+

```text
screen seeds = 0..99
run MSE only
rank by validation or heldout MSE gap
choose top/mid/bottom strata
run SPO+ on selected strata
evaluate on unseen10000
```

这是导师建议的直接回应。

---

### Phase 3: graph-generation seed sweep

如果导师仍然强调 graph seed：

```text
graph_seed = 10 or 20 values
generate new graph pools
run n=50 MSE/SPO+
evaluate matching unseen
```

这会更慢，但更接近他说的 “different graph seeds”。

---

## 关键注意：不要把 “chosen high-regret seed” 当作无偏结论

报告里要写清楚：

```text
The random-seed sweep estimates variability under resampling.
The high-MSE subset is a conditional stress test, selected by MSE difficulty.
It is not intended as an unbiased estimate of average performance.
```

这样你既采纳了导师建议，又避免被质疑 cherry-picking。

---

## 我会怎么给导师汇报这个计划

你可以这样说：

> I agree that the current Step2 grid is single-seed. I propose a complementary resampling experiment. First, I will run a multi-seed subset-resampling sweep on the hardest regimes, Step2b d8 and Step2c d8, with train size 50, comparing 2stage MSE and SPO+. I will report boxplots of MSE gap, SPO+ gap, and paired improvement. Second, I will use fast MSE training to screen many seeds, stratify them into low/medium/high MSE-regret groups, and then run SPO+ on those groups to test whether SPO+ improves specifically when MSE is bad. The high-MSE analysis will be reported as a stress test, not as an unbiased average-performance estimate.

这段话就很清楚。

---

## 我的最终建议

你的方案可以，但我会改成：

```text
不要“重做完整 Step2b/Step2c”；
要“做 Step2 hard-regime multi-seed resampling/stress-test”。
```

第一轮：

```text
Step2b d8 + Step2c d8
train_size = 50
50 subset seeds
2stage vs SPO+ only
boxplots + paired improvement + hard-MSE scatter
```

第二轮：

```text
MSE screen 100 seeds
top/mid/bottom strata
run SPO+ on selected seeds
boxplot conditional improvement
```

第三轮可选：

```text
true graph generation seed sweep
```

如果这个结果显示：

```text
MSE regret distribution has high-variance tail,
and SPO+ improvement is positive especially in high-MSE seeds,
```

那导师的建议就被很好地回应了，而且会显著增强你 Step2 结论的可信度。
