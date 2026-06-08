你现在的下一步应该**非常明确地聚焦在导师最新指定的 cycle-length sensitivity**，也就是：

> **当 maximum cycle length 从 3 增加到 4 和 5 时，second-best solution 的 true objective 是否仍然接近 oracle optimal value？**

这一步现在优先级最高。你已经完成了导师原本第 3 点的 first-pass：脚本已经通过 no-good cut 求 best 和 second-best distinct KEP solutions，并用 true weights 评估它们相对 oracle 的 gap。脚本逻辑正是：对每个 selected seed、heldout graph、method，加载 theta，算 `w_hat = X @ theta`，先求 best predicted solution，再加 cut 求 second-best solution，最后用 `w_true` 对 oracle 评估。 cut 的形式也已经实现为排除当前 MIP assignment，并额外检查 distinct edge-selection solution。

所以现在不要再继续扩展第 3 点太多，也不要马上写 AAAI-style report。**下一步就是验证这些 best/second-best 观察是否在 max cycle length = 4, 5 下仍成立。**

---

# Formal Plan: Cycle-Length Sensitivity After Best/Second-Best Analysis

## 0. Current status

你当前已经有一个 baseline result，对应：

```text
Regime: Step2b d8
max_cycle = 3
selected seeds = 9
heldout graphs = 400
total graph cases = 3600 per method/rank
methods = 2stage_val_mse, spoplus_val_spoplus_loss
solution ranks = best / second-best
```

当前整体结果显示，SPO+ 的 best solution 比 2stage 更好，但 second-best solutions 并没有崩掉。2stage rank-2 的 mean normalized oracle gap 是 7.04%，SPO+ rank-2 是 6.32%；并且 2stage rank-2 有 45.19% 在 oracle 5% 以内，SPO+ rank-2 有 47.97% 在 oracle 5% 以内。

这说明在 max cycle length = 3 时，很多 second-best solutions 的 objective value 仍然接近 oracle optimal value。

现在导师想知道：

> 如果 max cycle length 增加到 4 或 5，这个观察还成立吗？

换成她建议的直观说法：

> Does the second-best solution still have a small gap to the oracle optimal value?

---

# 1. Main research question for the next step

下一步的核心问题应该写成：

```text
When the maximum cycle length increases from 3 to 4 and 5,
does the second-best solution still have a small true-objective gap
relative to the oracle optimal solution?
```

或者更具体：

```text
For max_cycle ∈ {3, 4, 5}, compare the true objective value of:
  oracle solution,
  2stage rank-1 solution,
  2stage rank-2 solution,
  SPO+ rank-1 solution,
  SPO+ rank-2 solution.

The key outcome is whether rank-2 remains close to oracle.
```

不要再用 “near-optimal plateau becomes weaker” 作为主表达。可以在自己理解里保留这个概念，但给导师汇报时要说：

```text
second-best gap to oracle becomes larger / smaller
```

---

# 2. Hypothesis to test

导师现在的预期是：

```text
When longer cycles are allowed, we may observe fewer second-best solutions
whose objective value is close to the oracle optimal value.
```

也就是说，实验要验证：

```text
max_cycle = 3:
  many rank-2 solutions are close to oracle

max_cycle = 4 or 5:
  rank-2 gap to oracle may become larger
  within-1% / within-5% rates may decrease
```

如果这个成立，就支持：

> longer cycles may make the oracle solution more distinctive, so the second-best solution is less often close to oracle.

如果不成立，也很有价值，因为那说明：

> even with longer cycles, KEP still has many high-quality alternative solutions.

---

# 3. Scope: what exactly to run

我建议分成两个层次：**MVP case-level run** 和 **full selected heldout run**。

## 3.1 First run: 9 Case A/B/C graphs

先跑你已经选好的 9 个 case-study graphs：

```text
Case A:
  G-234.json
  G-994.json
  G-1516.json

Case B:
  G-696.json
  G-1372.json
  G-607.json

Case C:
  G-392.json
  G-1560.json
  G-39.json
```

这些 case 已经在 `case_study_index.csv` 里明确记录：Case A 是 bad prediction but irrelevant，Case B 是 different solution but near-optimal，Case C 是 SPO+ fixes 2stage。

这一步的目的不是最终统计显著性，而是快速确认：

```text
Case A/B/C 的机制在 max_cycle = 4, 5 下是否还保留？
```

尤其要看：

```text
G-1560 的 promotion mechanism 是否还存在？
```

也就是：

```text
2stage rank2 是否仍然接近 oracle？
SPO+ rank1 是否仍然接近 oracle？
SPO+ 是否仍然把 near-oracle solution promoted to rank1？
```

## 3.2 Main run: 3600 graph cases

MVP 成功后，跑完整 selected set：

```text
9 selected seeds × 400 heldout graphs = 3600 graph cases
```

对每个 max cycle length：

```text
max_cycle = 3, 4, 5
```

每个 graph case 都计算：

```text
oracle solution
2stage rank1
2stage rank2
SPO+ rank1
SPO+ rank2
```

这样最终是：

```text
3 cycle lengths × 3600 graph cases × 2 methods × 2 ranks
```

注意：max_cycle=3 已经有 baseline，可以复用现有 CSV；只需要新跑 max_cycle=4 和 5，然后合并比较。

---

# 4. Important design choice: no retraining

这一步不要重新训练 2stage 或 SPO+。

应该固定已有的：

```text
theta_2stage
theta_spoplus
selected seeds
same graph files
same trained models
same edge weights / labels
```

只改变 downstream KEP feasible set：

```text
max_cycle = 3 → 4 → 5
```

也就是说，这个实验回答的是：

> Given the same learned predictive model, how does changing the KEP optimization structure affect best/second-best gaps to oracle?

这样实验干净，不会把 training variability 混进 cycle length sensitivity。

---

# 5. Implementation plan

## 5.1 Add max-cycle argument

在 `compute_second_best_solutions.py` 里加参数：

```bash
--max-cycle 3
--max-chain 4
```

目前你的 parser / graph loading pipeline 应该默认 max_cycle=3。你需要让 graph loading 能根据参数重新 enumerate candidates。

目标是让命令变成：

```bash
python surrogate_experiment_results/decision_analysis/scripts/compute_second_best_solutions.py \
  --max-cycle 4 \
  --output surrogate_experiment_results/decision_analysis/results/cycle_sensitivity/second_best_gap_maxcycle4.csv \
  --summary-output surrogate_experiment_results/decision_analysis/results/cycle_sensitivity/second_best_summary_maxcycle4.csv
```

以及：

```bash
python surrogate_experiment_results/decision_analysis/scripts/compute_second_best_solutions.py \
  --max-cycle 5 \
  --output surrogate_experiment_results/decision_analysis/results/cycle_sensitivity/second_best_gap_maxcycle5.csv \
  --summary-output surrogate_experiment_results/decision_analysis/results/cycle_sensitivity/second_best_summary_maxcycle5.csv
```

如果你不想改动原脚本太多，可以复制一个新脚本：

```text
scripts/compute_second_best_cycle_sensitivity.py
```

但更好的做法是扩展现有脚本，因为它已经稳定完成 no-good cut 和 summary。

---

## 5.2 Output directory

建议新建：

```text
surrogate_experiment_results/decision_analysis/results/cycle_sensitivity/
```

里面放：

```text
second_best_gap_maxcycle3.csv          # 可以复制现有 baseline
second_best_gap_maxcycle4.csv
second_best_gap_maxcycle5.csv

second_best_summary_maxcycle3.csv
second_best_summary_maxcycle4.csv
second_best_summary_maxcycle5.csv

cycle_length_second_best_summary.csv
cycle_length_case_summary.csv
cycle_length_rank2_gap_by_case.csv
```

plots 放：

```text
surrogate_experiment_results/decision_analysis/plots/cycle_sensitivity/
```

例如：

```text
rank2_gap_by_cycle_length.png
near5_rate_by_cycle_length.png
case_abc_cycle_length_rank1_rank2_gap.png
```

---

# 6. Metrics to compute

导师现在要的是非常直接的 gap to oracle。所以核心指标应是：

## 6.1 Per solution metrics

每一行：

```text
regime
max_cycle
subset_seed
graph_id
method_label
solution_rank        # 1 or 2
oracle_obj
true_obj
gap_to_oracle
normalized_gap_to_oracle
same_solution_as_oracle
edge_jaccard_with_oracle
predicted_margin_from_best
num_edges_selected
num_cycle_candidates
num_chain_candidates
solve_time
```

其中最重要的是：

```text
normalized_gap_to_oracle
```

也就是：

[
\frac{oracle_obj - true_obj}{|oracle_obj|}
]

## 6.2 Main summary metrics

每个：

```text
max_cycle × method × solution_rank
```

统计：

```text
mean normalized gap to oracle
median normalized gap to oracle
exact oracle rate
within 1% oracle rate
within 5% oracle rate
mean rank2-minus-rank1 gap
median rank2-minus-rank1 gap
```

最关键的三列：

```text
rank2 mean normalized gap
rank2 within 1% rate
rank2 within 5% rate
```

汇报时可以只发这三个就够。

---

# 7. What to compare

你最终要形成这张表：

| max cycle | method | rank | mean normalized oracle gap | median normalized oracle gap | within 1% oracle | within 5% oracle |
| --------: | ------ | ---: | -------------------------: | ---------------------------: | ---------------: | ---------------: |
|         3 | 2stage |    1 |                        ... |                          ... |              ... |              ... |
|         3 | 2stage |    2 |                        ... |                          ... |              ... |              ... |
|         3 | SPO+   |    1 |                        ... |                          ... |              ... |              ... |
|         3 | SPO+   |    2 |                        ... |                          ... |              ... |              ... |
|         4 | 2stage |    1 |                        ... |                          ... |              ... |              ... |
|         4 | 2stage |    2 |                        ... |                          ... |              ... |              ... |
|         4 | SPO+   |    1 |                        ... |                          ... |              ... |              ... |
|         4 | SPO+   |    2 |                        ... |                          ... |              ... |              ... |
|         5 | ...    |  ... |                        ... |                          ... |              ... |              ... |

然后重点解读：

```text
Does rank-2 within-5% rate decrease from max_cycle=3 to 4/5?
Does rank-2 mean normalized gap increase?
Does rank2-rank1 gap increase?
```

如果答案是 yes，就支持导师预期。

如果答案是 no，就说明 longer cycles 并没有减少 second-best closeness to oracle。

---

# 8. Case-level analysis to keep

除了 full summary，你还应该保留 Case A/B/C 的图。

当前 Case A/B/C 图已经非常有解释力，尤其是 C2/G-1560：2stage rank2 与 SPO+ rank1 是同一个 near-oracle edge set，说明 SPO+ 把 2stage 的 second-best near-oracle solution 提升到了 best。这个机制很好。

下一步对 max_cycle=4/5 也要检查同样三种机制：

```text
Case A:
  rank1 是否仍然 oracle-quality？
  rank2 是否离 oracle 更远？

Case B:
  rank1/rank2 是否仍然都接近 oracle？
  还是 rank2 gap 变大？

Case C:
  SPO+ 是否仍然把 low-gap solution promoted to rank1？
  2stage rank2 是否仍然 near-oracle？
```

尤其是 Case B 和 C2：

```text
Case B answers:
  Are there still multiple solutions close to oracle?

C2 answers:
  Is SPO+ still doing a useful rank promotion?
```

---

# 9. Interpretation rules

跑完之后，不要只说数字。按下面规则解释。

## 9.1 如果 max_cycle 增加后 rank2 gap 变大

例如：

```text
max_cycle=3: rank2 within 5% = 45%
max_cycle=4: rank2 within 5% = 30%
max_cycle=5: rank2 within 5% = 20%
```

可以说：

> When longer cycles are allowed, the second-best solution is less often close to the oracle optimal value. This suggests that longer cycles can make the optimal solution more distinctive, reducing the number of close second-best alternatives.

这支持导师预期。

## 9.2 如果 max_cycle 增加后 rank2 gap 没变大

例如：

```text
rank2 within 5% remains around 45–50%
```

可以说：

> The observation from max_cycle=3 still holds under longer cycles. Even when cycles of length 4 and 5 are allowed, many second-best solutions remain close to the oracle optimal value.

这说明 KEP 的 redundancy / alternative solution structure 比预期更强。

## 9.3 如果 rank1 改善但 rank2 变差

这也很有意思：

> Longer cycles improve the best solution but make the second-best solution farther from oracle.

这说明：

```text
longer cycles create stronger top solution,
but fewer close alternatives.
```

## 9.4 如果 both rank1 and rank2 improve

说明：

```text
longer cycles increase overall feasible quality,
and alternative solutions also remain strong.
```

这会削弱导师原预期，但仍然是有价值结果。

---

# 10. After cycle length: arc density sensitivity

等 cycle length 完成后，再做导师原本第 2 点的 density sensitivity。

这一步要独立做，不要和 cycle length 混在一起。

选择少量代表 graph：

```text
G-696   # Case B, near-objective tie
G-1560  # Case C2, promotion case
G-392   # Case C1, region correction
```

对每个 graph 生成：

```text
original
+25% arcs
-25% arcs
```

导师原文写了 “increase by 25% the number of arcs” 和 “decrease by 25 the number of arcs”。这里有一点歧义。建议你实现时保留两个选项：

```text
--remove-arc-frac 0.25
--remove-arc-count 25
```

但汇报时先说明你会按 “25%” 做主结果；如果她确实是指 25 arcs，再补一个 small run。

density sensitivity 里保持：

```text
max_cycle = 3
```

先不要同时改 cycle length，否则 confounding。

---

# 11. Paper/report priority after experiments

导师说 priority 是：

```text
3 -> 2 -> 1
```

现在第 3 点已经有结果。下一步是第 2 点：

```text
cycle length sensitivity
then arc density sensitivity
```

等这两个结果完成后，再做第 1 点：

```text
AAAI-style short paper report
```

报告主故事应该是：

```text
When is decision-focused learning useful for graph combinatorial optimization?

Evidence from KEP:
  - SPO+ improves in decision-critical ranking errors.
  - But KEP often has second-best solutions close to oracle.
  - This limits the average upside of SPO+ over 2stage.
  - Cycle length and graph density sensitivity test whether this explanation depends on graph structure.
```

---

# 12. Suggested immediate Teams reply

你可以现在给导师回：

```text
Yes, I understand. I will focus next on the cycle-length sensitivity.

I will keep the same selected seeds/graphs and trained 2stage/SPO+ models, and only change the KEP optimization structure by setting max cycle length to 4 and 5. For each setting, I will recompute the oracle solution and the best/second-best distinct solutions for 2stage and SPO+ using the same no-good-cut procedure.

The main quantity I will report is the normalized gap between the oracle optimal value and the true objective value of the second-best solution. I will compare the mean/median gap and the within-1% / within-5% rates across max cycle lengths 3, 4, and 5.

So the direct question will be: when longer cycles are allowed, does the second-best solution still have a small gap to the oracle optimal value, or does this gap become larger?
```

---

# 13. Final priority list

你现在应该按这个顺序做：

```text
Priority 1:
  Modify / extend second-best script to support max_cycle = 4, 5.

Priority 2:
  Run case-level cycle sensitivity on 9 Case A/B/C graphs.

Priority 3:
  Run full selected 3600 graph cases for max_cycle = 4 and 5.

Priority 4:
  Produce summary table:
    max_cycle × method × rank
    mean/median normalized oracle gap
    within 1% / 5% oracle rates

Priority 5:
  Produce case-level plot:
    Case A/B/C best vs second-best gap under cycle length 3/4/5.

Priority 6:
  Send concise update to导师.

Priority 7:
  Only after that, do arc density sensitivity.

Priority 8:
  Only after sensitivity, convert into AAAI-style report.
```

一句话总结：

> 你下一步不是继续解释现有 max-cycle-3 结果，而是验证这些结果在 max-cycle-4 和 max-cycle-5 下是否还成立；核心指标是 second-best solution 的 objective value 到 oracle optimal value 的 normalized gap。

---

# Completed cycle-length sensitivity result

Run status:

```text
max_cycle = 4: completed on Garnet, exit=0
max_cycle = 5: completed on Garnet, exit=0
```

The formal Garnet CSVs were synced back locally and summarized with:

```bash
python3 surrogate_experiment_results/decision_analysis/scripts/summarize_cycle_sensitivity.py
```

The script explicitly reads:

```text
second_best_gap_maxcycle3.csv
second_best_gap_maxcycle4.csv
second_best_gap_maxcycle5.csv
```

It does not include the smoke-test file:

```text
second_best_gap_maxcycle4_smoke_graphlimit1.csv
```

Generated comparison outputs:

```text
cycle_length_second_best_summary.csv
cycle_length_case_summary.csv
cycle_length_rank2_gap_by_case.csv
cycle_length_rank2_paired_delta_by_case.csv
cycle_length_rank2_paired_delta_summary.csv
```

Generated plots:

```text
surrogate_experiment_results/decision_analysis/plots/cycle_sensitivity/rank2_gap_by_cycle_length.png
surrogate_experiment_results/decision_analysis/plots/cycle_sensitivity/near5_rate_by_cycle_length.png
surrogate_experiment_results/decision_analysis/plots/cycle_sensitivity/case_abc_cycle_length_rank1_rank2_gap.png
```

## Main full-run summary

The central full-run result is:

| max cycle | method | rank | mean normalized gap | median normalized gap | within 1% | within 5% |
| --------: | ------ | ---: | ------------------: | --------------------: | --------: | --------: |
| 3 | 2stage | 1 | 6.09% | 4.40% | 20.69% | 53.14% |
| 3 | 2stage | 2 | 7.04% | 5.75% | 16.03% | 45.19% |
| 3 | SPO+ | 1 | 4.94% | 3.56% | 23.81% | 59.28% |
| 3 | SPO+ | 2 | 6.32% | 5.28% | 17.58% | 47.97% |
| 4 | 2stage | 1 | 6.07% | 4.45% | 21.58% | 53.94% |
| 4 | 2stage | 2 | 6.93% | 5.53% | 16.89% | 47.50% |
| 4 | SPO+ | 1 | 5.12% | 3.60% | 23.50% | 58.67% |
| 4 | SPO+ | 2 | 6.20% | 5.03% | 19.47% | 49.78% |
| 5 | 2stage | 1 | 6.00% | 4.31% | 21.47% | 52.97% |
| 5 | 2stage | 2 | 6.88% | 5.20% | 15.19% | 48.94% |
| 5 | SPO+ | 1 | 4.80% | 3.29% | 25.22% | 61.14% |
| 5 | SPO+ | 2 | 5.61% | 4.39% | 19.19% | 54.83% |

## Interpretation

The full selected-set result does not support the hypothesis that allowing longer cycles makes the second-best solution much farther from oracle.

For 2stage, the rank-2 mean normalized oracle gap is essentially stable:

```text
max_cycle 3: 7.04%
max_cycle 4: 6.93%
max_cycle 5: 6.88%
```

The 2stage rank-2 within-5% rate also increases slightly:

```text
max_cycle 3: 45.19%
max_cycle 4: 47.50%
max_cycle 5: 48.94%
```

For SPO+, the rank-2 result improves more clearly:

```text
mean normalized gap:
  max_cycle 3: 6.32%
  max_cycle 4: 6.20%
  max_cycle 5: 5.61%

within-5% rate:
  max_cycle 3: 47.97%
  max_cycle 4: 49.78%
  max_cycle 5: 54.83%
```

So the main conclusion is:

> The max-cycle-3 observation is robust under longer cycle lengths. Even when cycles of length 4 and 5 are allowed, many second-best solutions remain close to the oracle optimal value. In this selected Step2b d8 setting, longer cycles do not reduce second-best closeness to oracle; if anything, the rank-2 gap becomes slightly smaller, especially for SPO+.

This weakens the specific expectation that longer cycles would make the oracle solution more distinctive and reduce the number of close alternatives. Instead, the evidence suggests that KEP still has substantial high-quality alternative-solution structure under max_cycle 4 and 5.

## Paired graph-level rank-2 delta

The aggregate means above should be checked at the paired graph level. For each fixed:

```text
subset_seed, graph_id, method_label
```

the paired delta is:

```text
delta_rank2_gap_K4 = rank2_normalized_gap_K4 - rank2_normalized_gap_K3
delta_rank2_gap_K5 = rank2_normalized_gap_K5 - rank2_normalized_gap_K3
```

Negative values mean the rank-2 solution became closer to oracle when the cycle length was increased.

The paired summary is:

| comparison | method | paired cases | delta < 0 | delta = 0 | delta > 0 | mean delta | median delta | q25 delta | q75 delta |
| ---------- | ------ | -----------: | --------: | --------: | --------: | ---------: | -----------: | --------: | --------: |
| K4 - K3 | 2stage | 3600 | 42.31% | 19.22% | 38.47% | -0.113 pp | 0.000 pp | -2.886 pp | 2.278 pp |
| K4 - K3 | SPO+ | 3600 | 42.31% | 19.94% | 37.75% | -0.117 pp | 0.000 pp | -2.033 pp | 1.859 pp |
| K5 - K3 | 2stage | 3600 | 46.06% | 11.58% | 42.36% | -0.165 pp | 0.000 pp | -3.134 pp | 2.799 pp |
| K5 - K3 | SPO+ | 3600 | 48.50% | 13.86% | 37.64% | -0.713 pp | 0.000 pp | -2.834 pp | 1.811 pp |

Paired interpretation:

> The average rank-2 gap decreases slightly, but this is not because a clear majority of graph-level pairs improve. Including unchanged pairs, fewer than half of the graph-seed-method pairs have lower rank-2 gap under K=4 or K=5. The median paired delta is exactly zero in all four method/comparison groups.

Among only the non-tied pairs, decreases do slightly outnumber increases:

```text
K4 - K3, 2stage: 52.4% of non-tied pairs decrease
K4 - K3, SPO+:   52.8% of non-tied pairs decrease
K5 - K3, 2stage: 52.1% of non-tied pairs decrease
K5 - K3, SPO+:   56.3% of non-tied pairs decrease
```

So the more careful conclusion is:

> Longer cycles do not systematically worsen rank-2 closeness to oracle. They produce a mixed paired distribution with many unchanged pairs, many improvements, and many degradations. The aggregate mean improvement is mild, and for SPO+ at K=5 it is driven by a somewhat stronger negative tail rather than by a broad majority shift.

This is the right answer to the consistency question: the trend is not a clean majority-of-graphs effect, but it is also not just a single extreme graph. The ten graph IDs with the largest negative mean deltas account for about 18-21% of the total negative-delta magnitude, depending on method and K, while similarly large positive outliers also exist.

## Case A/B/C notes

The selected Case A/B/C examples are useful for illustration, but they are less smooth than the full 3,600-case aggregate.

Case B remains the clearest small-gap alternative-solution example:

```text
Case B rank-2 within-5% rate:
  max_cycle 3: 100.0% for both methods
  max_cycle 4: 66.7% for both methods
  max_cycle 5: 66.7% for both methods
```

Case C becomes less favorable under longer cycles at the selected-example level:

```text
Case C rank-2 within-5% rate:
  max_cycle 3: 33.3%
  max_cycle 4: 0.0%
  max_cycle 5: 0.0%
```

This selected-case behavior should not be overgeneralized, because the full selected-set summary shows rank-2 alternatives remain broadly close to oracle.

## Suggested concise advisor-facing conclusion

```text
I completed the cycle-length sensitivity for max_cycle = 3, 4, and 5 using the same selected Step2b d8 seeds, heldout graphs, and trained 2stage/SPO+ models.

The result does not show that longer cycles make the second-best solution much worse. For 2stage, the rank-2 mean normalized oracle gap stays around 7.0% and the within-5% rate slightly increases from 45.2% to 48.9%. For SPO+, the rank-2 mean gap decreases from 6.3% to 5.6%, and the within-5% rate increases from 48.0% to 54.8%.

So the max-cycle-3 observation appears robust: even with max cycle length 4 or 5, many second-best KEP solutions remain close to the oracle optimal value. This suggests that the alternative-solution structure in these KEP instances persists under longer cycle lengths.
```
