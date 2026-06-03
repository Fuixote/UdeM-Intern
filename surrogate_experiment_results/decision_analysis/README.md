你下一步应该正式进入导师说的第 3 点：

> **为什么 2stage 几乎和 SPO+/FY 一样好？是不是因为一些 edge 预测很差，但这些 edge 不影响最终 KEP solution？**

这已经不是 bug-check，也不是单纯画 boxplot，而是 **decision-level interpretability analysis**。你前面已经有足够理由开始做这一步：shortest-path 和 KEP 上的实现等价性验证都基本完成；KEP Phase 1/2 里 PyEPO reference 与 Step1c 在真实 KEP oracle 上的 loss、gradient、trajectory 已对齐。尤其 Phase 2 里 theta、decision gap、y_adv/y_pred equality rate 都是 0 diff，SPO+ loss diff 也在 `1e-5` 内。

所以现在的研究问题应该换成：

```text
Given that the implementation is reliable, why can 2stage remain competitive?
Which prediction errors are decision-critical, and which are irrelevant?
```

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
heldout400 / `2stage_val_mse` vs `spoplus_val_spoplus_loss`。第一版只做
seed selection、per-graph decision replay、edge criticality 和 3 张 summary
plots；Top-K candidate / margin / FY / case-study tables 暂未做。

### Scripts

```text
surrogate_experiment_results/decision_analysis/scripts/select_case_seeds.py
surrogate_experiment_results/decision_analysis/scripts/compare_decisions_per_graph.py
surrogate_experiment_results/decision_analysis/scripts/analyze_edge_error_criticality.py
surrogate_experiment_results/decision_analysis/scripts/plot_decision_analysis.py
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

plots/mse_vs_normalized_gap.png
plots/solution_overlap_vs_gap.png
plots/high_error_edge_selected_symdiff_rate.png
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
