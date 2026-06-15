## 下一步的核心实验：Mechanism Dissection Audit

我建议你把下一步正式命名为：

```text
Step2c Mechanism Dissection Audit:
Topology, feasible-set geometry, and decision-landscape diagnostics.
```

### Implementation layout

This experiment keeps its advisor-facing protocol, commands, and final outputs in:

```text
surrogate_experiment_results/Step2c Mechanism Dissection Audit/
```

The experiment-local entrypoints are:

```text
scripts/compute_predicted_topm_solutions.py
scripts/compute_true_oracle_landscape.py
scripts/summarize_step2c_mechanism_dissection.py
```

They are thin wrappers around shared, tested utilities in:

```text
surrogate_experiment_results/decision_analysis/scripts/
```

This keeps the mechanism-dissection audit self-contained for documentation and
execution, while avoiding a second copy of the solver/replay logic.

### Latest run snapshot: 2026-06-15

The first MVP audit has been run on garnet with:

```text
graphs = 16 selected Step2c graphs
subset_seed = 0,...,49
predicted top-M = 20 for 2stage and SPO+
true-label oracle landscape top-M = 50
max_cycle = 3
max_chain = 4
```

Outputs are in:

```text
results/step2c_selected_graphs_all50_top20_predicted.csv
results/step2c_selected_graphs_true_top50_oracle_landscape.csv
results/step2c_selected_graphs_candidate_basin_diagnostic.csv
results/step2c_selected_graphs_mechanism_atlas.csv
```

Observed row counts:

```text
predicted top20 rows: 32,000 = 16 graphs * 50 seeds * 2 methods * 20 ranks
true oracle top50 rows: 800 = 16 graphs * 50 ranks
candidate-basin rows: 16
mechanism-atlas rows: 16
```

Main finding: the audit turns several apparent mechanisms into more precise
candidate-basin stories.

| Graph | Main result from top20/top50 dissection |
| --- | --- |
| G-392 | Still a clean non-rank2/non-top5 correction case, but not absent from the broader 2stage basin: SPO+ rank1 matches 2stage rank6/7/8, mostly rank8, and 2stage top20 contains a near-oracle solution in 50/50 seeds. |
| G-1285 | Clean exact rank-2 promotion: SPO+ rank1 equals 2stage rank2 in 50/50 seeds and equals the true oracle top1 solution. |
| G-1560 | Large-effect top-K promotion: SPO+ rank1 equals 2stage rank2 in 41/50 seeds and rank3 in 9/50 seeds. |
| G-1169 | Previously "unexplained"; now best described as broad top20 promotion. SPO+ rank1 always appears in 2stage top20, with median matching rank 11, and 2stage top20 contains near-oracle candidates in 50/50 seeds. |
| G-1449 | Previously "unexplained"; now best described as broad top20 promotion outside top5. SPO+ rank1 matches 2stage rank8/9/10 and 2stage top20 contains near-oracle candidates in 50/50 seeds. |
| G-1657 / G-191 | Boundary cases remain useful: SPO+ rank1 equals 2stage rank3 in almost all seeds, while correction-style rates are also high. |
| G-142 / G-946 | Both-poor negative controls: both methods choose the same bad rank1 solution, and 2stage top20 contains no near-oracle solution. |
| G-14 / G-163 / G-1308 | SPO+ worse controls: SPO+ moves to lower-ranked 2stage candidates that are not in the true top50 landscape, often away from an already-good or better 2stage decision. |

Interpretation update: the original correction/promotion labels remain useful,
but the more accurate mechanism axis is now:

```text
near-oracle candidate absent from 2stage top20
vs.
near-oracle candidate present but deeply misranked under 2stage
vs.
SPO+ promotes a non-near-oracle candidate and harms performance
```

Under this refined view, G-392 is not "2stage never found the good basin"; it is
"2stage found it only deep in the candidate list, while SPO+ made it rank1."

固定候选图：

```text
Clean correction:
  G-392

Clean promotion:
  G-1285
  G-1560

Unexplained SPO+ success:
  G-1169
  G-1449

Boundary cases:
  G-1657
  G-191

Negative controls:
  G-142
  G-946
  G-14
  G-163
```

可以把 G-552 / G-1110 / G-178 放 appendix replication，不必都进入主文深挖。主文最重要的是机制清楚、对照完整、叙事不散。

---

## 具体要 dissect 什么

你要把每个图拆成四层：**raw topology → feasible-set geometry → oracle landscape → model-induced decision behavior**。

### 1. Raw topology metrics

这些指标回答：“图本身长什么样？”

每个 selected graph 都算：

```text
num_vertices
num_edges
density
in_degree_mean / out_degree_mean
in_degree_gini / out_degree_gini
reciprocal_edge_count
2-cycle count
3-cycle count
largest_SCC_fraction
number_of_SCCs
degree assortativity, if easy
```

但我要强调：这些只能支持 **topology association**，不能单独支持机制解释。G-392 和 G-1560 之前已经显示 topology 不完全相同，所以不要期待一个简单指标解释全部。

### 2. Feasible-set geometry metrics

这层更重要。它回答：

> “这个图允许哪些 exchange solutions？near-oracle alternatives 是稀少还是很多？oracle solution 是孤立的还是有一簇高质量替代解？”

每个图计算：

```text
num_feasible_2cycles
num_feasible_3cycles
num_feasible_chains_length_1_to_4
num_total_feasible_exchanges
oracle_objective
oracle_to_second_best_gap_pct
num_solutions_within_1pct_of_oracle
num_solutions_within_5pct_of_oracle
num_solutions_within_10pct_of_oracle
near_oracle_solution_entropy
mean_pairwise_jaccard_among_near_oracle_solutions
min_jaccard_to_oracle_among_near_oracle_solutions
edge_criticality_entropy
number_of_oracle_edges_with_high_criticality
```

这里的重点不是“边多不多”，而是：

> **oracle 是否孤立，以及 near-oracle basin 是否容易被预测误差推入或推出。**

这会直接连接你最初的科学问题：为什么 2stage often competitive，以及什么时候 SPO+ helps。

### 3. Oracle / top-M solution landscape

这一步对你的机制分类最关键。

对每个 selected graph，枚举 true-label oracle 下的 top-M solutions，建议：

```text
M = 20 or 50
```

然后对每个 graph 记录：

```text
oracle rank
true objective
normalized gap
solution hash
edge signature
jaccard with oracle
jaccard with 2stage rank1
jaccard with SPO+ rank1
```

同时对每个 method × subset_seed 枚举 predicted top-M：

```text
2stage predicted top-20
SPO+ predicted top-20
```

然后回答四个问题：

```text
Is SPO+ rank1 in 2stage top-5?
Is SPO+ rank1 in 2stage top-20?
Is 2stage top-20 containing any near-oracle solution?
If yes, what predicted rank did 2stage assign to the best near-oracle solution?
If no, is SPO+ discovering a solution outside the 2stage candidate basin?
```

这一步会把 “correction / promotion / unexplained” 变得更精确。

---

## 每类图应该验证什么机制

### G-392：clean correction prototype

你要证明的不是“G-392 上 SPO+ 好”，这个已经证明了。你要证明的是：

> **2stage 的 top candidate region 没有覆盖 near-oracle basin，而 SPO+ 稳定进入了另一个 better decision basin。**

应该输出：

```text
2stage rank1 gap ≈ 25.01%
2stage rank2 gap ≈ 28.63%
2stage top5 是否全部 bad?
2stage top20 是否仍然没有 near-oracle?
SPO+ rank1 gap ≈ 0.83%
SPO+ rank1 是否在 true-label top-M near-oracle set?
SPO+ rank1 与 oracle 的 Jaccard
2stage rank1 与 oracle 的 Jaccard
oracle-vs-2stage symdiff 中 high-error edges 的集中度
oracle-vs-SPO+ symdiff 中 high-error edges 的集中度
```

如果 2stage top20 仍没有 near-oracle，而 SPO+ rank1 在 true top-M 里，那 G-392 的 claim 会很强：

> “This is not promotion of a latent 2stage alternative; SPO+ moves the decision into a different near-oracle basin.”

这就是 AAAI 里最干净的 correction story。

### G-1285：clean exact rank-2 promotion prototype

G-1285 应该成为 promotion 的最干净主例子。你要证明：

> **2stage 已经找到了 near-oracle solution，但 decision-critical ranking error 把它排在第二；SPO+ 修正排序，把同一个 solution 提到 rank1。**

输出：

```text
2stage rank1 gap ≈ 22.36%
2stage rank2 gap ≈ 0.00%
SPO+ rank1 gap ≈ 0.00%
Exact R2 = 1.00
predicted margin between 2stage rank1 and rank2
true-value reversal between 2stage rank1 and rank2
which edges differ between bad rank1 and good rank2
whether SPO+ changes weights specifically on those symdiff edges
```

这里最重要的图可能不是 topology 图，而是一个 **rank reversal diagram**：

```text
2stage predicted ranking:
  rank1 = bad true solution
  rank2 = oracle / near-oracle

SPO+ predicted ranking:
  rank1 = same near-oracle solution
```

这能非常清楚地说明 SPO+ 什么时候帮忙：不是因为全局预测更准，而是因为它修正了决策边界附近的排序。

### G-1560：large-Δ robust top-K promotion

G-1560 不如 G-1285 纯，但它的效应最大。你要把它定位成：

> **largest-effect promotion family example, where exact rank-2 promotion is common but top-K promotion is the more stable mechanism.**

输出：

```text
Exact R2 = 0.82
TopK = 1.00
median Δ = 35.37 pp
which 2stage rank does SPO+ match when it is not rank2? rank3? rank4?
are all matched alternatives near-oracle?
are these alternatives structurally similar?
```

如果 G-1560 的 SPO+ rank1 有时等于 2stage rank3/rank4，但这些 solution 都和 oracle 高 Jaccard、gap ≤ 5%，那么你应该避免只叫它 “rank-2 promotion”。更准确是：

> “near-oracle candidate promotion from the 2stage top-K set.”

### G-1169 / G-1449：unexplained success，最可能有新发现

这两个我非常赞成优先做。它们可能是你论文里最有 novelty 的地方，但现在不能急着讲故事。

对这两个图，先跑三步：

```text
1. Expand K from 5 to 20 or 50.
2. Replace exact hash match by structural similarity:
   Jaccard(SPO+ rank1, 2stage rank k)
3. Compare SPO+ rank1 against true-label top-M oracle landscape.
```

可能出现三种结果：

| 结果                                                            | 解释                                     |
| ------------------------------------------------------------- | -------------------------------------- |
| SPO+ rank1 出现在 2stage top20/top50                             | 其实是 broad promotion，只是 top5 太窄         |
| SPO+ rank1 不同 hash，但和 2stage near-oracle candidates 高 Jaccard | hash 指标太粗，需要 structural promotion      |
| SPO+ rank1 完全不在 2stage candidate basin，但接近 oracle             | 第三类 correction mechanism               |
| SPO+ rank1 near-oracle，但 oracle landscape 有很多等价 alternatives  | 可能是 high-degeneracy solution landscape |

这两个图先不要命名机制。建议暂时叫：

```text
unexplained stable SPO+ success
```

等 top20/top50 + Jaccard diagnostics 出来后再命名。

### G-1657 / G-191：boundary cases

这些图不适合做主例子，但适合做机制边界说明。它们的价值是告诉 reviewer：

> “Our mechanism labels are diagnostic categories, not mutually exclusive laws of nature.”

你要验证：

```text
SPO+ rank1 equals which 2stage rank?
2stage rank2 bad, but rank3/rank4/rank5 good?
topK overlap comes from exact same solution or high structural similarity?
is this really correction within the broader top-K candidate region?
```

这类图可以在 appendix 或 discussion 里讲：

> top-K overlap does not automatically imply rank-2 promotion; it can coexist with correction-like behavior when 2stage’s immediate backup is bad but a broader candidate is good.

### Negative controls：必须做，而且不要放太晚

负对照非常重要。你列的四个很好，但它们其实分成两类：

```text
Both poor:
  G-142
  G-946

SPO+ worse:
  G-14
  G-163
```

你要用它们回答：

> “为什么同样的 SPO+ training 并不总能帮助？”

对 G-142/G-946：

```text
Do both methods choose the same bad basin?
Does either method have a near-oracle solution in predicted top20?
Is the oracle solution isolated?
Are decision-critical edges badly predicted by both?
```

对 G-14/G-163：

```text
Is 2stage rank1 already oracle / near-oracle?
Does SPO+ move away from an already good decision?
Which edge weight changes cause the harmful move?
Is SPO+ overcorrecting toward a solution that looked good under SPO+ predictions but bad under true labels?
```

这会让你的论文更强，因为你不是只说“何时 SPO+ 帮忙”，还说“何时 SPO+ 不帮忙甚至伤害”。

---

## 我建议你现在立刻做的最小版本

不要一开始就做复杂 perturbation。最小但最有科学价值的是：

```text
Selected-graph mechanism dissection, no perturbation yet.
```

范围：

```text
graphs =
  G-392,
  G-1285,
  G-1560,
  G-1169,
  G-1449,
  G-1657,
  G-191,
  G-142,
  G-946,
  G-14,
  G-163

subset_seed = 0..49
top_M_predicted = 20 or 50
top_M_true_oracle = 50
fixed Step2c labels
fixed max_cycle=3, max_chain=4
```

输出三个表。

### Table 1: Mechanism atlas

```text
graph_id
assigned_family
strict_case_c_rate
strong_case_c_rate
correction_rate
exact_r2_rate
top5_rate
top20_rate
median_delta_pp
median_2stage_rank1_gap
median_spoplus_rank1_gap
median_2stage_rank2_gap
```

### Table 2: Feasible-set / oracle landscape

```text
graph_id
num_edges
density_percentile
num_2cycles
num_3cycles
num_feasible_exchanges
oracle_objective
oracle_second_best_gap_pct
num_true_solutions_within_1pct
num_true_solutions_within_5pct
num_true_solutions_within_10pct
near_oracle_jaccard_mean
near_oracle_jaccard_min
oracle_edge_criticality_entropy
```

### Table 3: Candidate-basin diagnostic

```text
graph_id
family
rate_2stage_top5_contains_near_oracle
rate_2stage_top20_contains_near_oracle
rate_spoplus_rank1_in_2stage_top5
rate_spoplus_rank1_in_2stage_top20
median_rank_of_best_near_oracle_under_2stage
median_jaccard_spoplus_rank1_to_oracle
median_jaccard_2stage_rank1_to_oracle
median_jaccard_spoplus_rank1_to_nearest_2stage_top20
```

这三个表出来后，你就能非常严格地说：

```text
Correction = near-oracle basin absent from 2stage top candidates but reached by SPO+.
Promotion = near-oracle basin present in 2stage candidates but misranked by 2stage and promoted by SPO+.
Unexplained = stable SPO+ success not captured by exact/top-K hash diagnostics; requires structural or larger-K analysis.
Negative controls = cases where the needed near-oracle basin is absent, misranked by both, or SPO+ moves away from an already good 2stage decision.
```

---

## 暂时不要做什么

我会暂时避免三件事：

1. **不要马上做 relabel-and-retrain。**
   太贵，而且现在的问题不是 seed robustness，而是 mechanism geometry。

2. **不要先做很多 topology perturbations。**
   你还没有清楚知道要 perturb 哪个结构。如果现在随便 add/remove arcs，解释会很散。

3. **不要把 topology causality 写得太强。**
   现在最多说 feasible-set-conditioned / graph-instance-specific。README 里的表述是对的：all-400 支持 graph-instance specificity 和 feasible-set-conditioned mechanism claim，但仍不能证明 pure topology causality，因为 topology、features、labels、feasible-set geometry 仍然绑定在一起。

---

## AAAI 主线应该怎么写

我建议你的论文核心故事变成：

> 2stage is often competitive in KEP because many graphs contain high-quality alternative solutions and prediction errors are not always decision-critical. SPO+ helps most when its training signal changes the ranking of decision-critical alternatives. In the all-400 Step2c audit, SPO+ success is not uniformly distributed across graphs; instead, it concentrates on a small number of graph instances with stable mechanisms across 50 trained models. We identify at least two mechanisms: correction, where SPO+ reaches a near-oracle decision basin not covered by 2stage’s top candidates, and promotion, where SPO+ promotes a near-oracle alternative that 2stage had already found but misranked. Negative controls show that SPO+ does not help when both methods miss the near-oracle basin or when SPO+ moves away from an already-good 2stage solution.

这个主线比“topology causes SPO+ success”安全得多，也更科学。

---

## 最终建议

你的想法是对的，但我会把下一步定义得更严格：

> **下一步不是继续找例子，而是对固定候选集合做统一的 mechanism dissection audit，重点拆解 topology、feasible-set geometry、oracle landscape 和 model-induced ranking behavior。**

优先级我会写成：

```text
Primary clean mechanisms:
  G-392
  G-1285
  G-1560

Discovery mechanisms:
  G-1169
  G-1449

Controls:
  G-142
  G-946
  G-14
  G-163

Boundary / appendix:
  G-1657
  G-191
```

如果你只能做一个最小实验：**先做 selected graphs 的 top20/top50 predicted-solution + true-oracle-landscape enumeration**。这一步最可能把你的 AAAI 核心故事从“现象观察”推进到“机制解释”。
