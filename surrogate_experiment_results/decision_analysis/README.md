你下一步应该正式进入导师说的第 3 点：

> **为什么 2stage 几乎和 SPO+/FY 一样好？是不是因为一些 edge 预测很差，但这些 edge 不影响最终 KEP solution？**

这已经不是 bug-check，也不是单纯画 boxplot，而是 **decision-level interpretability analysis**。你前面已经有足够理由开始做这一步：shortest-path 和 KEP 上的实现等价性验证都基本完成；KEP Phase 1/2 里 PyEPO reference 与 Step1c 在真实 KEP oracle 上的 loss、gradient、trajectory 已对齐。尤其 Phase 2 里 theta、decision gap、y_adv/y_pred equality rate 都是 0 diff，SPO+ loss diff 也在 `1e-5` 内。

所以现在的研究问题应该换成：

```text
Given that the implementation is reliable, why can 2stage remain competitive?
Which prediction errors are decision-critical, and which are irrelevant?
```

## Toy Example Family

The toy-example generator is:

```text
surrogate_experiment_results/decision_analysis/scripts/build_toy_property_x_examples.py
```

Run it from the repo root:

```bash
MPLCONFIGDIR=/tmp/matplotlib-toy python3 \
  surrogate_experiment_results/decision_analysis/scripts/build_toy_property_x_examples.py
```

It writes a paper-facing toy family under:

```text
surrogate_experiment_results/decision_analysis/results/toy_examples/
surrogate_experiment_results/decision_analysis/plots/toy_examples/
```

The current family includes positive packing-style examples
(`KEP/set-packing`, `stable set`, `weighted matching`,
`cardinality knapsack`, and `partition matroid`), a parametric epsilon
construction where regret can be made arbitrarily small while the selected
solution identity changes, and path-like negative controls (`shortest path` and
`serial path`) where a ranking error redirects the whole connected decision and
creates large regret. Use these as mechanism illustrations, not as a theorem
that every packing instance has small decision-focused-learning gains.

## Randomized Property X Toy Family

The randomized toy-family generator is:

```text
surrogate_experiment_results/decision_analysis/scripts/run_randomized_property_x_toy_experiments.py
```

Run it from the repo root with the KEPs Python environment:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache \
  /home/weikang/miniconda3/envs/KEPs/bin/python \
  surrogate_experiment_results/decision_analysis/scripts/run_randomized_property_x_toy_experiments.py
```

It writes:

```text
surrogate_experiment_results/decision_analysis/results/toy_randomized/
  randomized_packing_summary.csv
  randomized_shortest_path_summary.csv
  randomized_property_x_comparison.csv

surrogate_experiment_results/decision_analysis/plots/toy_randomized/
  identity_mismatch_vs_regret.png
  rank2_gap_distribution.png
  property_x_phase_diagram.png
```

Default protocol: `tau in {0.02, 0.05, 0.10, 0.20, 0.30}`,
`sigma in {0.0, 0.02, 0.05, 0.10, 0.20, 0.30}`, 500 instances per
grid cell. The decomposable packing family uses 12 independent blocks with 4
candidate components per block. The path family uses 4 parallel coupled paths,
each with 12 edges. This first randomized version intentionally covers only the
abstract decomposable packing and shortest-path controls; the KEP-like random
set-packing graph generator remains a natural next extension.

Latest default run:

| family | tau | sigma | identity mismatch | mean regret | median regret | mean regret if mismatch | mean oracle-second gap | within 5% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| decomposable packing | 0.02 | 0.10 | 100.0% | 1.07% | 1.04% | 1.07% | 0.0067% | 100.0% |
| parallel shortest path | 0.02 | 0.10 | 66.8% | 0.47% | 0.31% | 0.71% | 0.4163% | 100.0% |
| decomposable packing | 0.10 | 0.10 | 100.0% | 2.76% | 2.65% | 2.76% | 0.0322% | 95.4% |
| parallel shortest path | 0.10 | 0.10 | 43.0% | 0.83% | 0.00% | 1.93% | 1.9317% | 97.8% |
| decomposable packing | 0.30 | 0.30 | 100.0% | 7.71% | 7.54% | 7.71% | 0.0800% | 21.2% |
| parallel shortest path | 0.30 | 0.30 | 39.8% | 2.30% | 0.00% | 5.78% | 5.7989% | 81.8% |

Interpretation: this is still a toy mechanism study, not a theorem. The useful
claim is conditional: when close substitutes exist, solution identity can flip
without much regret; when the true best-second gap is larger, a ranking flip has
more room to create decision regret and therefore more room for DFL to improve.

---

## 总体路线

我建议你新建一个独立目录，不要再放在 `SPO_validation` 下面，因为这已经不是 validation，而是解释性研究：

```text
surrogate_experiment_results/Step3_decision_analysis/
```

这个目录的目标不是复现 PyEPO，也不是再次证明 SPO+ 公式正确，而是输出：

```text
1. per-graph decision comparison table
2. edge-level prediction-error vs decision-impact analysis
3. solution-overlap / selected-cycle-chain comparison
4. small-graph case studies
5. 2stage almost-good 的解释性结论
```

---

## 先选实验范围：不要一上来全做

第一版只做最能解释现象的 setting：

```text
Regime:
  Step2b d8

Train size:
  50

Evaluation:
  heldout400 first
  optional: unseen10000 later

Methods:
  2stage selected by validation MSE
  SPO+ selected by validation SPO+ loss
  optional later: FY
```

理由是：你当前 resampling boxplot 的随机性主要是 subset resampling，也就是固定 graph pool / fixed label regime，只改变训练子集。README 里也明确说第一轮是固定 graph pool，只变 `subset_seed ∈ {0,...,49}`，`theta_seed` 固定。 这组结果已经足够帮你挑 seed 和 case。

先不要同时做 Step2b + Step2c + FY + all seeds + all graph seeds。先从 Step2b d8 做出一个完整解释链条，再扩展。

---

## Step 1：从 resampling 结果里挑 case seeds

你不要随机挑 graph。先从已有 resampling summary 里挑 3 类 subset seed：

### A. 2stage 和 SPO+ 都好的 seed

```text
2stage gap low
SPO+ gap low
improvement small
```

目的：解释为什么 2stage 已经够好。

### B. 2stage 差、SPO+ 明显好的 seed

```text
2stage gap high
SPO+ gap much lower
improvement high
```

目的：找出 SPO+ 真正修复了什么 decision error。

### C. 2stage 差、SPO+ 没明显改善或也差的 seed

```text
2stage gap high
SPO+ gap similar or worse
```

目的：理解 SPO+ 的失败模式。

你现在的 plot 脚本已经生成 paired reduction，也就是同一个 seed 下：

```text
gap_2stage - gap_spoplus
```

而且图里说明正值代表 SPO+ 有更低的 heldout400 normalized gap。 所以直接用这个 paired reduction 表挑 case。

建议第一版每类选：

```text
3 seeds × 每个 seed 选 5–10 个代表 graph
```

总共 30 个左右 graph，足够做解释。

---

## Step 2：生成 per-graph decision comparison table

写一个脚本：

```text
surrogate_experiment_results/Step3_decision_analysis/compare_decisions_per_graph.py
```

输入：

```text
--regime step2b_poly_d8
--train-size 50
--subset-seeds selected_seed_list
--methods 2stage spoplus
--evaluation heldout400
```

每个 graph、每个 method 输出一行：

```text
regime
subset_seed
graph_id
method
selected_epoch
theta_1
theta_2
optimal_obj
achieved_obj
decision_gap
normalized_gap
num_edges
num_edges_opt
num_edges_pred
same_solution_as_opt
edge_jaccard_with_opt
edge_hamming_with_opt
edge_overlap_count
```

其中：

```text
same_solution_as_opt = all(y_method == y_opt)
edge_jaccard_with_opt = |selected_edges_method ∩ selected_edges_opt| / |union|
edge_hamming_with_opt = mean(y_method != y_opt)
```

这一步回答第一个问题：

```text
2stage 和 SPO+ 是不是经常选同一个 solution？
如果 solution 不同，objective gap 是否仍然很小？
```

你很可能会看到三种情况：

```text
1. 2stage 和 SPO+ 选完全相同的 KEP solution；
2. 它们选不同 solution，但 true objective 很接近；
3. SPO+ 避免了 2stage 选到的明显低质量 solution。
```

这三种就是你后面 case study 的骨架。

---

## Step 3：edge-level prediction error 是否真的影响 decision

导师的 hypothesis 是：

> bad estimations in one edge do not impact the overall performance of a solution.

所以你需要对每个 graph 统计 edge prediction error：

```text
err_e(method) = |w_hat_e(method) - w_true_e|
```

然后按 edge 是否进入不同 solution 分类：

```text
in_opt
in_2stage
in_spoplus
in_any_selected
in_symmetric_difference_2stage = y_2stage xor y_opt
in_symmetric_difference_spoplus = y_spoplus xor y_opt
```

输出第二个表：

```text
edge_id
src
dst
w_true
w_hat_2stage
w_hat_spoplus
abs_err_2stage
abs_err_spoplus
rank_err_2stage
rank_err_spoplus
in_opt
in_2stage
in_spoplus
in_2stage_symdiff
in_spoplus_symdiff
utility
recipient_cPRA
```

然后聚合出这些指标：

```text
top10_error_edges_in_opt_rate
top10_error_edges_in_2stage_rate
top10_error_edges_in_spoplus_rate
top10_error_edges_in_2stage_symdiff_rate
top10_error_edges_in_spoplus_symdiff_rate

mse_all_edges
mse_edges_in_opt
mse_edges_in_pred
mse_edges_in_symdiff
mse_edges_not_selected
```

最关键的图是：

```text
x-axis: edge prediction error rank
y-axis: selected rate / symdiff rate
```

你想验证的模式是：

```text
很多 high-error edges 并不在 y_opt 或 y_pred 中；
或者 high-error edges 虽然存在，但不在 y_pred Δ y_opt 中；
因此它们不改变最终 decision。
```

这就直接支持导师的 hypothesis。

---

## Step 4：solution margin / near-optimal alternatives

2stage 接近 SPO+ 的另一个可能原因是：

```text
不同 solution 其实 objective 很接近。
```

所以你要分析 margin。

对每个 graph，至少计算：

```text
true_opt_obj = w_true^T y_opt
obj_2stage = w_true^T y_2stage
obj_spoplus = w_true^T y_spoplus

gap_2stage = true_opt_obj - obj_2stage
gap_spoplus = true_opt_obj - obj_spoplus
method_gap_diff = gap_2stage - gap_spoplus
```

如果你能枚举一些候选 cycle/chain 或 top-K feasible solutions，更好。第一版不一定要枚举所有 feasible KEP solutions，因为 KEP 组合可能很大。可以先做 **selected-solution margin**：

```text
true_opt_obj - obj_2stage
true_opt_obj - obj_spoplus
abs(obj_2stage - obj_spoplus)
```

再加一个 coarse top-candidate analysis：

```text
top cycles/chains by true weight
top cycles/chains by 2stage prediction
top cycles/chains by SPO+ prediction
overlap of top-K candidate exchanges
```

这能回答：

```text
2stage 是否虽然预测有误，但仍然把关键 high-value cycles/chains 排在前面？
SPO+ 是否主要改变了边缘 cases 的 ranking？
```

---

## Step 5：挑 3 个 case studies 做可解释表

最后你需要能给导师展示几个具体 graph。每个 graph 用一张小表，不要只给统计数字。

### Case 1：2stage 和 SPO+ 都 optimal

展示：

```text
两者选择相同 solution
prediction errors 主要在未选边
```

结论：

```text
Bad edge estimates are non-critical because they do not enter the selected exchange.
```

### Case 2：2stage 和 SPO+ solution 不同，但 gap 都很小

展示：

```text
selected edge overlap maybe not perfect
but true objective difference tiny
```

结论：

```text
The graph has multiple near-optimal exchanges; different decisions are effectively equivalent.
```

### Case 3：SPO+ 明显改善 2stage

展示：

```text
2stage overestimates one or a few edges that form a bad exchange
SPO+ shifts ranking and selects exchange closer to y_opt
```

结论：

```text
SPO+ helps when prediction errors affect decision-critical edges.
```

这三个 case study 就能把导师的 hypothesis 讲清楚。

---

## 推荐输出文件结构

```text
surrogate_experiment_results/Step3_decision_analysis/
  README.md

  scripts/
    select_case_seeds.py
    compare_decisions_per_graph.py
    analyze_edge_error_criticality.py
    make_case_study_tables.py
    plot_decision_analysis.py

  results/
    selected_case_seeds.csv
    per_graph_decision_comparison.csv
    edge_error_criticality.csv
    graph_level_summary.csv
    case_studies/
      case_good_both_graph_*.csv
      case_near_tie_graph_*.csv
      case_spoplus_fix_graph_*.csv

  plots/
    gap_vs_solution_overlap.png
    high_error_edge_selected_rate.png
    paired_gap_vs_mse.png
    case_study_edge_table_*.png
```

---

## 最重要的 4 张图

### Figure 1：prediction MSE vs decision gap

```text
x-axis: method edge MSE per graph
y-axis: normalized decision gap
color: method
shape: case type
```

目的：

```text
证明 high prediction error 不一定 high decision gap。
```

### Figure 2：solution overlap vs decision gap

```text
x-axis: Jaccard(y_method, y_opt)
y-axis: normalized decision gap
```

目的：

```text
看 solution overlap 和 regret 是否强相关。
```

如果 overlap 低但 gap 小，说明存在 near-optimal alternative solutions。

### Figure 3：high-error edge criticality

```text
x-axis: error percentile bin
y-axis: probability edge is selected / in symdiff
```

目的：

```text
证明 bad edges 多数不 critical。
```

### Figure 4：paired improvement vs 2stage difficulty

```text
x-axis: 2stage normalized gap
y-axis: gap_2stage - gap_spoplus
```

目的：

```text
看 SPO+ 是否主要在 2stage hard cases 上改善。
```

---

## 最小可行版本

如果你只想一周内做出能汇报的版本，我建议只做这个：

```text
Regime: Step2b d8
Train size: 50
Seeds: 从 resampling 中选 9 个
  3 easy seeds
  3 SPO+ improves seeds
  3 no-improvement seeds

Evaluation graphs:
  每个 seed 选 heldout400 中 gap 最大的前 5 个 graph
  加上随机 5 个 graph 作对照

Methods:
  2stage
  SPO+
```

输出：

```text
1. per_graph_decision_comparison.csv
2. edge_error_criticality.csv
3. 3 张 case-study tables
4. 4 张 summary plots
```

这已经足够回答导师第 3 点的第一版。

---

## 是否需要 FY？

导师说的是 “SPO+/FY”，但你不一定第一版就加 FY。建议：

```text
第一版：2stage vs SPO+
第二版：加 FY
```

因为 FY 多了 perturbation randomness，解释起来更复杂。你现在的核心问题是 2stage 为什么接近 decision-focused method；先用 SPO+ 足够。

如果第一版结果清楚，再把 FY 作为第三条曲线加入：

```text
2stage
SPO+
FY
```

然后看 FY 的 selected solution 是更接近 SPO+，还是更接近 2stage。

---

## 你下一步的具体行动

我建议你马上做三件事：

### 1. 写 seed selection 脚本

```bash
python3 surrogate_experiment_results/Step3_decision_analysis/scripts/select_case_seeds.py \
  --regime step2b_poly_d8 \
  --train-size 50 \
  --input surrogate_experiment_results/Step2_resampling/results/phase1_heldout400_paired_main.csv
```

输出：

```text
selected_case_seeds.csv
```

包含：

```text
case_type, subset_seed, mse_gap, spoplus_gap, improvement
```

### 2. 写 per-graph decision comparison

对 selected seeds 的 heldout400 graphs 跑：

```text
y_opt
y_2stage
y_spoplus
```

输出：

```text
per_graph_decision_comparison.csv
```

### 3. 写 edge criticality analysis

基于每个 graph 的 edge table，统计 high-error edges 是否进入 selected solutions：

```text
edge_error_criticality.csv
```

---

## 最终你想得到的结论形式

最后你应该能给导师这样的回答：

> In the subset-resampling experiment, SPO+ improves over 2stage more consistently in high-degree regimes, but 2stage remains competitive in many seeds. Decision-level analysis suggests two reasons. First, many large edge-level prediction errors occur on edges that are not selected by either the optimal or predicted KEP solution, so they are non-critical. Second, in several graphs, 2stage and SPO+ select different exchanges whose true objectives are nearly tied. SPO+ helps most when 2stage errors affect edges in the symmetric difference between the predicted and optimal solutions.

中文就是：

> 2stage 接近 SPO+ 的原因可能不是它预测得很好，而是很多预测错的边不影响最终 KEP 解；另外很多小图存在多个 near-optimal exchanges，即使选的边不同，true objective gap 也很小。SPO+ 的优势主要出现在 2stage 的错误落在 decision-critical edges 上时。

这正好回应导师第 3 点。

---

## 2026-06-03 第一版实施记录

本目录按上面的最小可行版本先实现 `step2b_poly_d8` / train size 50 /
heldout400 / `2stage_val_mse` vs `spoplus_val_spoplus_loss`。第一版先做
seed selection、per-graph decision replay、edge criticality、3 张 summary
plots、case-study edge tables、observed-candidate margin analysis 和
decision-critical MSE correlation summary；Top-K candidate / `y_adv_*` / FY
暂未做。

### Scripts

```text
surrogate_experiment_results/decision_analysis/scripts/select_case_seeds.py
surrogate_experiment_results/decision_analysis/scripts/compare_decisions_per_graph.py
surrogate_experiment_results/decision_analysis/scripts/analyze_edge_error_criticality.py
surrogate_experiment_results/decision_analysis/scripts/plot_decision_analysis.py
surrogate_experiment_results/decision_analysis/scripts/make_case_study_tables.py
surrogate_experiment_results/decision_analysis/scripts/analyze_margin_near_ties.py
surrogate_experiment_results/decision_analysis/scripts/summarize_decision_critical_mse.py
surrogate_experiment_results/decision_analysis/scripts/summarize_case_best_second_gaps.py
```

### Commands

Seed selection:

```bash
python3 surrogate_experiment_results/decision_analysis/scripts/select_case_seeds.py
```

Garnet artifact check found all 9 selected seeds already had:

```text
model_weights/2stage_best_by_validation_mse_loss.npz
model_weights/spoplus_best_by_validation_spoplus_loss.npz
metrics/test_per_graph.csv
```

No rerun was needed.

Per-graph decision replay on garnet:

```bash
cd /local1/fuweik/UdeM-Intern
source configs/runtime/garnet.env
python surrogate_experiment_results/decision_analysis/scripts/compare_decisions_per_graph.py \
  --output surrogate_experiment_results/decision_analysis/results/per_graph_decision_comparison.csv
```

Edge criticality on garnet:

```bash
python surrogate_experiment_results/decision_analysis/scripts/analyze_edge_error_criticality.py \
  --edge-output surrogate_experiment_results/decision_analysis/results/edge_error_criticality.csv \
  --summary-output surrogate_experiment_results/decision_analysis/results/graph_level_edge_criticality_summary.csv
```

Plots on garnet:

```bash
python surrogate_experiment_results/decision_analysis/scripts/plot_decision_analysis.py
```

Case-study tables from existing CSV outputs:

```bash
python3 surrogate_experiment_results/decision_analysis/scripts/make_case_study_tables.py
```

Case-level best-vs-second gap summary:

```bash
python3 surrogate_experiment_results/decision_analysis/scripts/summarize_case_best_second_gaps.py
```

Observed-candidate margin / near-tie analysis:

```bash
python3 surrogate_experiment_results/decision_analysis/scripts/analyze_margin_near_ties.py
```

Decision-critical MSE correlations from existing CSV outputs:

```bash
python3 surrogate_experiment_results/decision_analysis/scripts/summarize_decision_critical_mse.py
```

### Selected Seeds

```text
large_improvement: 1, 25, 22
weak_borderline_improvement: 27, 21, 30
easy_low_gap: 41, 16, 33
```

The selected seed table is:

```text
surrogate_experiment_results/decision_analysis/results/selected_case_seeds.csv
```

### Outputs

```text
results/selected_case_seeds.csv                         9 rows
results/per_graph_decision_comparison.csv               7,200 rows
results/edge_error_criticality.csv                      879,642 rows
results/graph_level_edge_criticality_summary.csv         7,200 rows
results/case_studies/case_study_index.csv                    9 rows
results/case_studies/case_*.csv                              9 tables
results/case_studies/case_best_second_gap_summary.csv        18 rows
results/case_studies/case_best_second_gap_by_case_method.csv  6 rows
results/case_studies/case_best_second_gap_summary.tex         LaTeX tabular
results/margin_near_tie_analysis.csv                     3,600 rows
results/observed_candidate_solution_ranking.csv         10,800 rows
results/margin_near_tie_summary.csv                          1 row
results/decision_critical_mse_correlations.csv              12 rows

plots/mse_vs_normalized_gap.png
plots/solution_overlap_vs_gap.png
plots/high_error_edge_selected_symdiff_rate.png
plots/case_best_second_gap_summary.png
```

### Replay Validation

The replayed per-graph gaps were checked against each seed's existing
`metrics/test_per_graph.csv`.

```text
max gap_abs_diff:            5.0663948e-05
max normalized_gap_abs_diff: 1.6472658e-07
```

This is within the intended replay tolerance, so the interpretation analysis is
based on the same decision path as the existing Step2 heldout400 metrics.

Implementation note: to reproduce the existing `test_per_graph.csv` exactly, the
replay script reloads graph records per selected seed and evaluates checkpoint
weights in the original Step1c evaluator order. This avoids cross-seed Gurobi
model warm-state effects and tie-breaking drift.

### First-Pass Observations

Across the 9 selected seeds and 400 heldout graphs:

```text
2stage mean normalized gap:  0.060869886
SPO+ mean normalized gap:    0.049376107

2stage mean Jaccard vs opt:  0.60580236
SPO+ mean Jaccard vs opt:    0.63654963

2stage exact-opt rate:       0.17805556
SPO+ exact-opt rate:         0.23583333
```

The MSE-vs-gap plot shows that SPO+ can have higher edge-level MSE but lower
decision gap, which supports treating raw prediction error as insufficient for
explaining KEP decision quality.

The solution-overlap plot shows the expected trend that lower overlap can
increase gap, but there are also medium-overlap points with small gap. These are
good candidates for later near-optimal alternative case studies.

The high-error criticality plot uses error rank bins: the leftmost bin contains
the highest-error edges. High-error edges are selected or enter the symmetric
difference only at limited rates:

```text
2stage top-10-error selected rate: 0.22869444
2stage top-10-error symdiff rate:  0.093916667

SPO+ top-10-error selected rate:   0.19047222
SPO+ top-10-error symdiff rate:    0.070222222
```

This is consistent with the current hypothesis: many large edge prediction
errors are not decision-critical, and the next useful step is to extract a few
case-study graphs where errors are either irrelevant, near-tied, or actually
fixed by SPO+.

### Case-Study Tables

`make_case_study_tables.py` reads the existing per-graph comparison, graph-level
edge-criticality summary, and edge-criticality CSVs. It does not rerun KEP
solves. The script selects three distinct graph IDs per case label and writes
one full-graph edge table per selected case.

The case-study index is:

```text
surrogate_experiment_results/decision_analysis/results/case_studies/case_study_index.csv
```

Selected cases:

```text
Case A: bad prediction but irrelevant
  seed=41 graph=G-234.json   2stage_gap=0        mse_all_edges=19.6328 top10_symdiff=0.0
  seed=41 graph=G-994.json   2stage_gap=0        mse_all_edges=18.1776 top10_symdiff=0.0
  seed=25 graph=G-1516.json  2stage_gap=0        mse_all_edges=17.7713 top10_symdiff=0.0

Case B: different solution but near-optimal
  seed=1 graph=G-696.json   2stage_gap=0.000643595 jaccard=0.681818 same_opt=False
  seed=1 graph=G-1372.json  2stage_gap=0.001173602 jaccard=0.608696 same_opt=False
  seed=1 graph=G-607.json   2stage_gap=0.001174954 jaccard=0.714286 same_opt=False

Case C: SPO+ fixes 2stage
  seed=1  graph=G-392.json   2stage_gap=0.313849 SPO+_gap=0        2stage_top10_symdiff=0.3 SPO+_top10_symdiff=0.0
  seed=30 graph=G-1560.json  2stage_gap=0.292305 SPO+_gap=0.011964 2stage_top10_symdiff=0.3 SPO+_top10_symdiff=0.0
  seed=1  graph=G-39.json    2stage_gap=0.316226 SPO+_gap=0.080459 2stage_top10_symdiff=0.2 SPO+_top10_symdiff=0.1
```

Each edge table contains:

```text
edge_id, src_dst, w_true, w_hat_2stage, w_hat_spoplus,
abs_err_2stage, abs_err_spoplus,
in_opt, in_2stage, in_spoplus,
in_2stage_symdiff, in_spoplus_symdiff,
utility, recipient_cPRA
```

Interpretation: Case A shows large 2stage edge MSE with zero decision gap and
zero top-error symdiff involvement. Case B shows non-identical 2stage solutions
with medium overlap but near-zero objective gap. Case C shows graphs where SPO+
substantially reduces the decision gap and also reduces high-error symdiff
involvement.

### Case-Level Best-vs-Second Gap Summary

`summarize_case_best_second_gaps.py` joins the selected Case A/B/C index with
`second_best_gap_comparison.csv`. It does not rerun KEP solves. The output
focuses on oracle gap, not predicted margin:

```text
rank1_gap_to_oracle
rank2_gap_to_oracle
rank2_minus_rank1_gap_to_oracle
rank1_normalized_gap
rank2_normalized_gap
rank2_minus_rank1_normalized_gap
rank1_same_oracle
rank2_same_oracle
rank1_jaccard_oracle
rank2_jaccard_oracle
rank2_predicted_margin_from_best
```

Outputs:

```text
results/case_studies/case_best_second_gap_summary.csv
results/case_studies/case_best_second_gap_by_case_method.csv
results/case_studies/case_best_second_gap_summary.tex
plots/case_best_second_gap_summary.png
```

The case/method summary with near threshold 5% is:

```text
Case A 2stage: rank2 mean normalized gap 0.04569; rank2 within 5% in 2/3 graphs
Case A SPO+:   rank2 mean normalized gap 0.03443; rank2 within 5% in 2/3 graphs

Case B 2stage: rank2 mean normalized gap 0.01686; rank2 within 5% in 3/3 graphs
Case B SPO+:   rank2 mean normalized gap 0.02595; rank2 within 5% in 3/3 graphs

Case C 2stage: rank1 mean normalized gap 0.30746; rank2 mean normalized gap 0.21316
Case C SPO+:   rank1 mean normalized gap 0.03081; rank2 mean normalized gap 0.13345
```

Interpretation by case:

```text
Case A:
  Rank-1 is oracle for both methods. Rank-2 is often still near-oracle, but not
  always. This supports the non-critical-error story: the large prediction
  errors do not change the selected optimal solution.

Case B:
  Rank-1 is not oracle but has near-zero gap, and all rank-2 solutions stay
  within 5% oracle gap. This is the strongest selected-case evidence for a
  near-optimal plateau.

Case C:
  2stage rank-1 is bad in all three cases, but rank-2 distinguishes failure
  modes. G-1560 has a near-oracle rank-2, consistent with a local ranking flip;
  G-392 and G-39 keep high rank-2 gaps, suggesting the top predicted region is
  also poor. SPO+ fixes rank-1 on G-392 and greatly improves the other two, but
  its rank-2 can be much worse, showing that in some decision-critical cases
  top-1 ordering matters.
```

### Margin / Near-Optimal Alternative Analysis

`analyze_margin_near_ties.py` uses only existing replay and edge-criticality
outputs. It does not enumerate all feasible KEP solutions and does not yet add
`y_adv_2stage` / `y_adv_spoplus`. The observed candidate set is:

```text
y_opt
y_2stage
y_spoplus
```

The main paired-graph output is:

```text
surrogate_experiment_results/decision_analysis/results/margin_near_tie_analysis.csv
```

It includes:

```text
abs(obj_2stage - obj_spoplus)
gap_2stage
gap_spoplus
normalized_gap_2stage
normalized_gap_spoplus
edge_jaccard(2stage, SPO+)
edge_jaccard(method, opt)
different_solution_near_tie flags
observed_unique_solution_count
```

The observed-candidate ranking table is:

```text
surrogate_experiment_results/decision_analysis/results/observed_candidate_solution_ranking.csv
```

It ranks `y_opt`, `y_2stage`, and `y_spoplus` by true objective for each graph.
With `near_tie_threshold = 0.01`, the first-pass summary is:

```text
paired graph rows:                         3,600
observed candidate rows:                  10,800

mean abs(obj_2stage - obj_spoplus):        4.33655891
median abs(obj_2stage - obj_spoplus):      0

mean Jaccard(2stage, SPO+):                0.82706818
median Jaccard(2stage, SPO+):              1.0

same 2stage/SPO+ solution rate:            0.55083333  (1,983 / 3,600)
2stage different-solution near-tie rate:   0.06083333  (219 / 3,600)
SPO+ different-solution near-tie rate:     0.06583333  (237 / 3,600)
any-method near-tie rate:                  0.09166667  (330 / 3,600)

observed unique solution count:
  1 unique solution:                         532 graphs
  2 unique solutions:                       1,877 graphs
  3 unique solutions:                       1,191 graphs
```

Interpretation: In this selected Step2b d8 subset, 2stage and SPO+ often select
the same solution, and even when a method differs from oracle, a nontrivial
subset has normalized gap below 1%. This supports the near-optimal-alternative
story: solution identity can change while true objective remains very close.
The current analysis is still limited to observed candidates; adding
`y_adv_2stage` / `y_adv_spoplus` would require extending the replay path.

### Decision-Critical MSE Correlations

`summarize_decision_critical_mse.py` joins
`graph_level_edge_criticality_summary.csv` with
`per_graph_decision_comparison.csv` and correlates each prediction-error
summary with normalized decision gap. The output is:

```text
surrogate_experiment_results/decision_analysis/results/decision_critical_mse_correlations.csv
```

The table is long-format, one row per method and predictor, with Pearson,
Spearman, valid-pair count, and within-method absolute Pearson rank.

First-pass Pearson / Spearman correlations with normalized gap:

```text
2stage_val_mse
  mse_all_edges:                         0.094654 / 0.023041  (n=3,600)
  mse_edges_in_pred:                    -0.259619 / -0.236999 (n=3,600)
  mse_edges_in_symdiff:                  0.390390 / 0.528040  (n=2,959)
  top10_error_edges_in_symdiff_rate:     0.569695 / 0.666985  (n=3,600)

spoplus_val_spoplus_loss
  mse_all_edges:                        -0.021118 / -0.075238 (n=3,600)
  mse_edges_in_pred:                    -0.112396 / -0.065935 (n=3,600)
  mse_edges_in_symdiff:                  0.272333 / 0.371808  (n=2,751)
  top10_error_edges_in_symdiff_rate:     0.556733 / 0.639125  (n=3,600)
```

Interpretation: all-edge MSE has weak correlation with decision gap, especially
for SPO+. In contrast, symdiff-focused measures are much more predictive:
`top10_error_edges_in_symdiff_rate` is the strongest predictor for both methods,
followed by `mse_edges_in_symdiff`. This supports the statement that prediction
error matters most when it lands on decision-changing edges, not merely because
raw edge-level error is large.
