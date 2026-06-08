这份计划**整体很好，可以执行**。它已经抓住了导师原始要求里最关键的点：**independently vary arc density**，并且只在你第 3 点里挑出来的代表 graph 上做，而不是一上来扩展到全量 3600 graphs。这样既快，也容易解释。

我建议你保留这个计划，但做几个重要修正，让它更严谨、更容易和导师沟通。

---

# 总体评价

你的计划现在可以定位成：

> **Arc density sensitivity on selected mechanism-representative KEP graphs.**
> We keep trained 2stage/SPO+ models fixed, keep max_cycle=3 and max_chain=4 fixed, and only perturb the graph arc set. Then we recompute oracle, method rank-1, and method rank-2 solutions to see whether the best/second-best gap patterns survive density changes.

这非常符合导师的第 2 点要求。

尤其这几个设计是正确的：

```text
1. 只做 G-696, G-392, G-1560 三个机制代表图。
2. 使用对应 case seed，而不是乱配模型。
3. 保持 max_cycle=3, max_chain=4，不和 cycle-length sensitivity 混在一起。
4. primary 做 +25% arcs 和 -25 arcs。
5. -25% arcs 作为 robustness，不混入主结论。
6. 复用 compute_second_best_solutions.rows_for_model_record，避免复制 second-best solver 逻辑。
7. original variant 要和已有结果做 integration check。
```

这些都很好。

---

# 需要改进或明确的关键点

## 1. “已有 arc label 保持不变” 是合理的，但要明确定义成 frozen-label structural perturbation

你计划里写：

> 新增 arc 的 label 用 Step2b d8 公式和原图 label scale 计算；已有 arc 的 ground_truth_label 保持不变，避免 original baseline 漂移。

这个选择是合理的，因为你现在要测的是 **arc density / graph structure perturbation**，不是重新生成一个完整 Step2b dataset。如果你重新计算所有 edge labels，原有 edge 的 true weights 也会变，实验就同时改变了：

```text
graph structure
+
edge reward labels
```

那就不好解释。

所以建议你在 README 和 manifest 里明确写：

```text
Label policy:
  Frozen-label structural perturbation.

Existing arcs keep their original ground_truth_label.
New arcs receive synthetic Step2b-d8 labels calibrated to the original graph scale.
This isolates the effect of arc-density changes while keeping the original graph rewards fixed.
```

但同时要承认 limitation：

```text
This is not a clinically valid regenerated compatibility graph.
It is a controlled structural perturbation.
```

你已经有这句话的意思了，建议写得更正式。

---

## 2. 新增 arc 的 label 计算要非常小心：不要无意改变 scale

Step2b d8 原始 label 有 graph-level rescaling。如果你给新增 arc 计算 label，要明确到底用哪个 scaling：

推荐 primary policy：

```text
Use the original graph's Step2b-d8 scaling constants / empirical scale.
Do not recompute the graph-level normalization over E + new arcs.
```

原因是：你想让新增 arc 的 reward scale 和原图兼容，同时不改变旧 arc label。

所以在 manifest 里建议加字段：

```text
label_policy = frozen_existing_edges_original_scale_new_edges
label_scale_reference = original_graph
step2b_degree = 8
step2b_kappa = ...
step2b_delta = ...
```

如果你不方便恢复原始 Step2b constants，也要至少保证新增 arc 的 `ground_truth_label` 分布和原图已有 label 分布在合理范围内。可以加一个 sanity summary：

```text
new_arc_label_mean
existing_arc_label_mean
new_arc_label_min/max
existing_arc_label_min/max
```

否则可能出现新增 arc 标签过大或过小，影响 oracle 结果。

---

## 3. add25pct 只用一个 perturb seed 可能有随机性，建议加一个 optional robustness

你现在计划每个 variant 用固定 seed，这是 first pass 可以接受的。但因为只有 3 张 graph，如果新增/删除的 arc 恰好很特殊，结果可能受 random perturbation pattern 影响。

建议：

### first pass

```text
perturb_seed = fixed, e.g. 202606xx
```

### 如果结果很强或很奇怪，再补小 robustness

```text
perturb_seed ∈ {0,1,2,3,4}
```

不一定一开始就做 5 seeds，但代码结构最好预留：

```bash
--perturb-seeds 0 1 2 3 4
```

或者 manifest 里至少记录：

```text
perturb_seed
```

你已经计划记录 perturb seed，这很好。

---

## 4. 删除 arc 时要记录是否删到了 oracle / rank1 / rank2 的边

这是 density sensitivity 最有解释力的地方。

比如 remove25arcs 后，如果 rank2 gap 突然变大，你需要知道：

```text
是不是删除了 original oracle solution 中的 edge？
是不是删除了 original 2stage/SPO+ rank2 solution 中的 edge？
```

所以建议在 summary 里加这些字段：

```text
removed_edges_overlap_original_oracle
removed_edges_overlap_original_2stage_rank1
removed_edges_overlap_original_2stage_rank2
removed_edges_overlap_original_spoplus_rank1
removed_edges_overlap_original_spoplus_rank2
```

如果只是 manifest 层面不好做，也可以在 summarize 脚本里做。

这会让你能解释：

```text
remove25arcs destroys the close second-best alternative
```

或者：

```text
remove25arcs changes density but does not touch critical edges
```

这比只看 gap 更有机制解释力。

---

## 5. add25pct 时也要记录新增 arc 是否进入 oracle / rank1 / rank2

同理，增加 arc 以后要看：

```text
new arcs used by oracle?
new arcs used by 2stage rank1/rank2?
new arcs used by SPO+ rank1/rank2?
```

建议加字段：

```text
num_added_edges_in_oracle
num_added_edges_in_2stage_rank1
num_added_edges_in_2stage_rank2
num_added_edges_in_spoplus_rank1
num_added_edges_in_spoplus_rank2
```

这样你可以区分两种情况：

### 情况 A：add25pct 让 oracle 变好，但 methods 没用新增 arcs

说明 predictor 没有把新增 arcs 排进 solution，可能 SPO+ / 2stage 对新 arcs generalization 弱。

### 情况 B：oracle 和 method 都使用新增 arcs

说明 density 增加确实产生了新的 high-quality exchange alternatives。

### 情况 C：新增 arcs 进入 rank2 but not rank1

这可能非常有趣，因为它和你之前 best-second story 直接连接。

---

# 对 3 个 case 的选择：很好，但建议明确它们代表什么

你选的 3 个 graph 很合理：

```text
G-696, seed=1:
  Case B, close alternatives.
  用来测试 density 改变后，原来的 small-gap alternatives 是否还存在。

G-392, seed=1:
  Case C1, region-level correction.
  用来测试 density 改变后，SPO+ 是否仍然能把 top predicted region 拉向 oracle。

G-1560, seed=30:
  Case C2, rank2 promotion.
  用来测试 density 改变后，2stage rank2 == SPO+ rank1 这种 promotion mechanism 是否还存在。
```

这三张图组合很好，已经覆盖了你 paper story 里的三种机制：

```text
close alternatives
region correction
rank promotion
```

我建议暂时不要加入 Case A。Case A 的故事是 “no correction needed”，对 density sensitivity 没有 Case B/C 那么关键。你现在三张图足够。

---

# 输出指标建议

你现在的 summary 计划已经不错，但我建议主表聚焦成下面这种形式，方便导师看。

每个 graph variant 一行或每个 method/rank 一行：

```text
base_graph_id
case_label
density_variant
edge_count
oracle_obj
oracle_obj_delta_vs_original
method_label
rank1_normalized_gap
rank2_normalized_gap
rank2_minus_rank1_gap
rank1_near_5pct
rank2_near_5pct
rank1_solution_changed_vs_original
rank2_solution_changed_vs_original
oracle_solution_changed_vs_original
num_added_edges_in_rank1
num_added_edges_in_rank2
num_removed_edges_from_original_rank1
num_removed_edges_from_original_rank2
```

最关键汇报指标仍然是导师喜欢的直观说法：

```text
Does the second-best solution still have a small gap to the oracle optimal value?
```

所以图里和表里都要突出：

```text
rank2_normalized_gap_to_oracle
```

---

# 对 “primary vs robustness” 的处理

你写得对：

```text
primary:
  original
  add25pct
  remove25arcs

robustness:
  remove25pct
```

因为导师原文是：

```text
increase by 25% the number of arcs
decrease by 25 the number of arcs
```

所以不要把 `remove25pct` 放进主结论。可以写：

```text
The main advisor-facing comparison follows the original request:
  +25% arcs and -25 arcs.

The -25% arcs variant is included only as a robustness check because “decrease by 25 arcs” and “decrease by 25% arcs” are easy to confuse.
```

这个很稳妥。

---

# 关于 implementation 的具体建议

## make_arc_density_variants.py

建议 CLI 支持：

```bash
python make_arc_density_variants.py \
  --case-index surrogate_experiment_results/decision_analysis/results/case_studies/case_study_index.csv \
  --graphs G-696.json G-392.json G-1560.json \
  --variants original add25pct remove25arcs remove25pct \
  --perturb-seed 42
```

manifest 建议字段：

```text
case_id
case_label
subset_seed
base_graph_id
variant_id
variant_graph_path
density_variant
arc_delta_type
original_edge_count
variant_edge_count
arc_delta
added_arc_count
removed_arc_count
perturb_seed
generation_policy
label_policy
added_arc_source_policy
added_arc_label_policy
removed_arc_policy
new_arc_label_mean
existing_arc_label_mean
```

## compute_arc_density_sensitivity.py

你说复用 `rows_for_model_record` 是对的。

但要注意：`rows_for_model_record` 当前可能不知道 `case_id`, `density_variant` 等字段。你可以在外面拿到 rows 后 append metadata，不要改核心函数太多。

建议：

```python
rows = rows_for_model_record(...)
for row in rows:
    row.update(variant_metadata)
```

这样最干净。

## summarize_arc_density_sensitivity.py

建议生成四个 summary：

```text
1. arc_density_second_best_summary.csv
   variant × method × rank aggregate

2. arc_density_case_summary.csv
   graph × variant × method summary

3. arc_density_delta_vs_original.csv
   graph × variant × method delta from original

4. arc_density_oracle_change_summary.csv
   graph × variant oracle objective changes and solution changes
```

你计划里已经有这些，很好。

---

# Test plan 评价

你的 test plan 很扎实。我建议加 4 个测试：

## 1. Added arc parser compatibility test

不仅检查字段存在，还要真的跑：

```python
parse_json_to_dfl_data(variant_graph)
```

确保新增 arc 不会导致 parser 崩。

## 2. Original variant exact replay test

你已经写了 integration check：

> original variant 的 3 个 case rank1/rank2 gap 与现有 max_cycle=3 case summary 在容差内一致。

这个非常重要。容差建议：

```text
normalized_gap_abs_diff < 1e-6 or 1e-5
```

如果有 Gurobi tie-breaking drift，就至少 objective gap 要一致，不一定 edge signature 一致。

## 3. Remove arcs count safety test

如果 graph edge count 小于 25，脚本应该报错或自动 min。但你的三张图应该都大于 25。仍然建议写：

```python
if remove_count >= edge_count:
    raise ValueError
```

## 4. Feasibility sanity test

删除 arc 后可能出现 fewer feasible cycles/chains，但 solver 应该仍然能返回 empty solution 或 best solution。测试至少确保：

```text
oracle_obj finite
rank1/rank2 rows generated or clean warning if no second-best exists
```

不过你的 graphs 边数应该够，不太会没有 second-best。

---

# 需要注意的一个风险：synthetic add arcs 不是医学兼容生成

你已经写了这点，非常好。

但建议在 Teams 或 README 中不要过度解释 added arcs 的医学意义。只说：

> This is a controlled structural perturbation to test arc-density sensitivity, not a regenerated clinically realistic compatibility graph.

这样导师不会纠结 added arc 是否 medically plausible。

---

# 建议的执行顺序

你的 plan 里说先本地测试，再 Garnet smoke，再 full tiny run。很好。

我建议正式执行顺序如下：

```text
1. Implement make_arc_density_variants.py
2. Unit tests for graph variants
3. Generate variants locally
4. Validate with parser locally
5. Sync to Garnet
6. Smoke:
     G-696 original + add25pct
7. Original replay check:
     original variants for all 3 graphs match existing max_cycle=3 results
8. Full run:
     3 graphs × 4 variants × 2 methods × 2 ranks
9. Summarize
10. Generate plots
11. Update README
12. Send concise advisor update
```

关于 Brevo watcher：可以用，但不要让它成为 blocker。之前 notification config 好像容易出问题，所以最好同时用：

```text
tmux
log file
tail -f
```

---

# 预期结果怎么解释

这个实验有四种可能结果。你可以提前准备解释框架。

## 1. add25pct 后 rank2 gap 仍小

解释：

> Increasing density creates more feasible alternatives, and the second-best solution remains close to the oracle. This supports the idea that KEP has multiple high-quality alternatives.

## 2. add25pct 后 rank2 gap 变大

解释：

> Added arcs may create a better oracle solution, but methods' second-best alternatives do not track it. This suggests density can make the oracle more distinctive in this graph.

## 3. remove25arcs 后 rank2 gap 变大

解释：

> Removing arcs destroys some alternative exchanges. This supports the idea that graph density contributes to the availability of close second-best solutions.

## 4. remove25arcs 后 rank2 gap 仍小

解释：

> The close-alternative structure is robust even to moderate arc removal.

对三张图分别看：

```text
G-696:
  close alternatives 是否对 density 稳定？

G-392:
  SPO+ region correction 是否依赖某些 arcs？

G-1560:
  2stage rank2 -> SPO+ rank1 promotion 是否被 add/remove arcs 破坏？
```

---

# 我会怎么修改你的 plan 文本

你的计划可以略微改成下面这个更正式版本。

```text
Arc Density Sensitivity Plan

Goal:
  Test whether the best/second-best gap patterns observed under max_cycle=3 are sensitive to graph arc density. This experiment is independent from cycle-length sensitivity: max_cycle=3 and max_chain=4 remain fixed, and only the graph arc set is perturbed.

Scope:
  Use three mechanism-representative case-study graphs:
    G-696.json, seed=1, Case B: close alternatives
    G-392.json, seed=1, Case C1: region-level correction
    G-1560.json, seed=30, Case C2: rank-2 promotion

Density variants:
  original
  add25pct: add round(0.25 * E0) arcs
  remove25arcs: remove 25 arcs, primary decrease condition
  remove25pct: remove round(0.25 * E0) arcs, robustness only

Label policy:
  Use frozen-label structural perturbation.
  Existing arcs keep their original ground_truth_label.
  Added arcs receive synthetic Step2b-d8 labels calibrated to the original graph scale.
  This isolates arc-density changes and avoids drifting the original baseline.
  This is a controlled structural perturbation, not a regenerated medically realistic compatibility graph.

Evaluation:
  For each variant, use the corresponding case seed to load trained 2stage/SPO+ weights.
  Recompute oracle, method rank-1, and method rank-2 solutions with the existing no-good-cut procedure.
  Compare true objective gaps to the variant-specific oracle.

Primary metrics:
  oracle_obj and delta_oracle_obj vs original
  rank1_normalized_gap_to_oracle
  rank2_normalized_gap_to_oracle
  rank2_minus_rank1_gap
  rank1/rank2 solution changed vs original
  oracle solution changed vs original
  overlap of added/removed arcs with oracle/rank1/rank2 solutions

Outputs:
  arc_density_graph_manifest.csv
  arc_density_second_best_gap.csv
  arc_density_second_best_summary.csv
  arc_density_case_summary.csv
  arc_density_delta_vs_original.csv
  arc_density_oracle_change_summary.csv

Plots:
  rank2_gap_by_arc_density.png
  oracle_obj_by_arc_density.png
  case_arc_density_rank1_rank2_gap.png
```

---

# 最终判断

这份计划**可以执行**，而且是你当前最该做的下一步。

我的建议是只做以下小修改后就开始：

```text
1. 把 label policy 明确写成 frozen-label structural perturbation。
2. 增加 added/removed arcs 是否进入 oracle/rank1/rank2 的统计。
3. 原始 variant 必须和现有 max_cycle=3 case results 做严格 replay check。
4. 固定 seed first pass 可以，但代码保留多 perturb seeds 的扩展能力。
5. Brevo watcher 可用但不要作为唯一监控方式。
```

做完这一步后，你就完成了导师第 2 点的两部分：

```text
cycle length sensitivity
+
arc density sensitivity
```

然后就可以进入第 1 点：把所有结果整理成更短的 AAAI-style paper report。
`