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
  randomized_kep_set_packing_summary.csv
  randomized_stable_set_summary.csv
  randomized_shortest_path_summary.csv
  randomized_property_x_comparison.csv

surrogate_experiment_results/decision_analysis/plots/toy_randomized/
  identity_mismatch_vs_regret.png
  rank2_gap_distribution.png
  property_x_phase_diagram.png
```

Default protocol: `tau in {0.02, 0.05, 0.10, 0.20, 0.30}`,
`sigma in {0.0, 0.02, 0.05, 0.10, 0.20, 0.30}`, 500 instances per
grid cell. The abstract decomposable packing control uses 12 independent blocks
with 4 candidate components per block. The KEP-like set-packing family uses 12
patient-donor pair vertices, random directed compatibility arcs with probability
0.35, candidate 2/3-cycles, and a vertex-disjoint cycle-packing feasible set.
The stable-set family uses 12 disjoint conflict cliques with 4 candidate
vertices per clique, so an independent set selects one non-conflicting vertex
from each clique when weights are positive. The path family uses 4 parallel
coupled paths, each with 12 edges.

After changing the randomized toy generator, rerun the script above before
interpreting `Latest default run`, because the committed CSV/plot outputs must
be regenerated from the updated generator.

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

---

## Actual Step2c Replay for G-392 and G-1560

The two original Case C graph/seed pairs were replayed under the actual Step2c
d8 trained models and Step2c labels:

```text
G-392.json  with subset_seed=1
G-1560.json with subset_seed=30
```

Rank-1 replay output:

```text
surrogate_experiment_results/decision_analysis/results/fixed_topology_label_seed/step2c_actual_case_replay_G392_G1560.csv
```

Rank-1/rank-2 replay output:

```text
surrogate_experiment_results/decision_analysis/results/fixed_topology_label_seed/step2c_actual_second_best_G392_G1560.csv
```

Actual Step2c result snapshot:

```text
G-392, seed=1:
  2stage rank1 gap = 25.01%
  2stage rank2 gap = 28.63%
  SPO+   rank1 gap = 0.83%
  SPO+   rank2 gap = 25.01%
  mechanism: SPO+ correction persists.

G-1560, seed=30:
  2stage rank1 gap = 35.89%
  2stage rank2 gap = 0.51%
  SPO+   rank1 gap = 0.51%
  SPO+   rank2 gap = 37.89%
  mechanism: rank-2 promotion persists exactly; 2stage rank2 and SPO+ rank1
  have identical solution signatures.
```

Interpretation: the actual Step2c trained-model replay preserves the two
case-level mechanisms on these selected graph/seed pairs. This is still a
two-graph case study rather than a Step2c-wide frequency claim.

## Main Step2c Fixed-Graph Model-Seed Robustness Result

The next robustness audit holds the graph instance, Step2c labels, arc
features, solver settings, and feasible set fixed, then varies only the trained
model induced by the fixed-pool training subset:

```text
regime: step2c_poly_d8_mult_eps050
graphs: G-392.json, G-1560.json
subset_seed: 0..49
train_size: 50
max_cycle: 3
max_chain: 4
top_k: 5
methods: 2stage selected by validation MSE; SPO+ selected by validation SPO+ loss
```

The replay was run on garnet from `/local1/fuweik/UdeM-Intern`:

```bash
source configs/runtime/garnet.env
python surrogate_experiment_results/decision_analysis/scripts/compute_second_best_solutions.py \
  --regime step2c_poly_d8_mult_eps050 \
  --run-root surrogate_experiment_results/Step2_resampling/phase1_runs \
  --dataset-dir dataset/processed/step2c_poly_d8_mult_eps050_main2000_seed20260523 \
  --split-path surrogate_experiment_results/Step2_resampling/splits/step2c_poly_d8_mult_eps050/master_split_seed=42.json \
  --subset-seed-start 0 \
  --subset-seed-stop 49 \
  --case-type-prefix step2c_fixed_graph_model_seed \
  --graphs G-392.json G-1560.json \
  --max-solutions 5 \
  --max-cut-attempts 60 \
  --progress-every 1 \
  --output surrogate_experiment_results/decision_analysis/results/fixed_graph_model_seed/step2c_g392_g1560_all50_top5_second_best.csv \
  --summary-output surrogate_experiment_results/decision_analysis/results/fixed_graph_model_seed/step2c_g392_g1560_all50_top5_second_best_summary.csv

python surrogate_experiment_results/decision_analysis/scripts/summarize_fixed_graph_model_seed_audit.py \
  --input surrogate_experiment_results/decision_analysis/results/fixed_graph_model_seed/step2c_g392_g1560_all50_top5_second_best.csv \
  --output surrogate_experiment_results/decision_analysis/results/fixed_graph_model_seed/step2c_g392_g1560_all50_seed_audit.csv \
  --summary-output surrogate_experiment_results/decision_analysis/results/fixed_graph_model_seed/step2c_g392_g1560_all50_seed_audit_summary.csv \
  --top-k 5
```

Main outputs:

```text
surrogate_experiment_results/decision_analysis/results/fixed_graph_model_seed/step2c_g392_g1560_all50_top5_second_best.csv
surrogate_experiment_results/decision_analysis/results/fixed_graph_model_seed/step2c_g392_g1560_all50_top5_second_best_summary.csv
surrogate_experiment_results/decision_analysis/results/fixed_graph_model_seed/step2c_g392_g1560_all50_seed_audit.csv
surrogate_experiment_results/decision_analysis/results/fixed_graph_model_seed/step2c_g392_g1560_all50_seed_audit_summary.csv
```

Definitions were fixed before reading the all-50 summary:

```text
Strict Case C:
  2stage rank1 gap >= 10%, SPO+ rank1 gap <= 5%, Delta >= 5 pp

Strong Case C:
  2stage rank1 gap >= 20%, SPO+ rank1 gap <= 5%, Delta >= 10 pp

G-392 correction:
  Strict Case C and 2stage rank2 gap > 10%

G-1560 exact rank-2 promotion:
  Strict Case C, 2stage rank2 gap <= 5%, and
  signature(SPO+ rank1) == signature(2stage rank2)

G-1560 top-5 promotion:
  signature(SPO+ rank1) is one of 2stage ranks 2..5 and
  SPO+ rank1 gap <= 5%
```

Result snapshot:

```text
G-392 all 50 seeds:
  Strict Case C = 50/50, Wilson 95% CI [0.929, 1.000]
  SPO+ better = 50/50
  correction preserved = 50/50, Wilson 95% CI [0.929, 1.000]
  exact rank-2 promotion = 0/50
  median Delta = 24.18 pp
  median gaps: 2stage rank1 25.01%, SPO+ rank1 0.83%, 2stage rank2 28.63%

G-392 excluding discovery seed 1:
  Strict Case C = 49/49, Wilson 95% CI [0.927, 1.000]
  correction preserved = 49/49, Wilson 95% CI [0.927, 1.000]

G-1560 all 50 seeds:
  Strict Case C = 50/50, Wilson 95% CI [0.929, 1.000]
  SPO+ better = 50/50
  exact rank-2 promotion = 41/50, Wilson 95% CI [0.692, 0.902]
  top-5 promotion = 50/50, Wilson 95% CI [0.929, 1.000]
  median Delta = 35.37 pp
  median gaps: 2stage rank1 35.89%, SPO+ rank1 0.51%, 2stage rank2 0.51%

G-1560 excluding discovery seed 30:
  Strict Case C = 49/49, Wilson 95% CI [0.927, 1.000]
  exact rank-2 promotion = 40/49, Wilson 95% CI [0.686, 0.900]
  top-5 promotion = 49/49, Wilson 95% CI [0.927, 1.000]
```

Interpretation: these two selected Step2c graph instances are robust to
trained-model/subset_seed variation under fixed actual labels. `G-392` remains
a correction case across every model seed, and `G-1560` remains a promotion case
across every model seed under the top-5 definition, with exact rank-2 promotion
in most but not all model seeds. This supports selected graph-instance
robustness. It still should not be stated as pure topology causality because
topology, arc features, fixed labels, and feasible-set geometry remain bundled.

## Main Step2c Fixed-Topology Label-Seed Robustness Result

After confirming the two selected graph/seed pairs under actual Step2c trained
models, the fixed-topology audit was rerun on the same Step2c basis:

```text
G-392.json  uses Step2c subset_seed=1  trained weights.
G-1560.json uses Step2c subset_seed=30 trained weights.
```

Only `label_seed` varies. Graph topology, arc features, `max_cycle=3`,
`max_chain=4`, and the trained model weights are fixed. This is still a
fixed-model diagnostic; it is not 1000 independent Step2c retraining runs.

The formal run was launched on garnet in tmux session
`step2c_ft_label_1000`, with completion email handled by
`notify_step2c_ft_label_1000` through `scripts/experiment_notify.py`.

Main outputs:

```text
surrogate_experiment_results/decision_analysis/results/fixed_topology_label_seed/step2c_fixed_topology_label_seed_1000_rows.csv
surrogate_experiment_results/decision_analysis/results/fixed_topology_label_seed/step2c_fixed_topology_label_seed_1000_summary.csv
surrogate_experiment_results/decision_analysis/results/fixed_topology_label_seed/step2c_fixed_topology_label_seed_1000_readout.md
surrogate_experiment_results/decision_analysis/plots/fixed_topology_label_seed/step2c_fixed_topology_1000_seed_summary_v2.png
```

Result snapshot:

```text
rows: 8000

G-392:
  label seeds = 1000
  unique_topology_hashes = 1
  unique_label_hashes = 1000
  strict Case C preserved rate = 0.661 +/- 0.029
  SPO+ better rate = 1.000
  SPO+ correction persistence rate = 0.661 +/- 0.029
  rank-2 promotion persistence rate = 0.000

G-1560:
  label seeds = 1000
  unique_topology_hashes = 1
  unique_label_hashes = 1000
  strict Case C preserved rate = 0.545 +/- 0.031
  SPO+ better rate = 0.998
  SPO+ correction persistence rate = 0.000
  rank-2 promotion persistence rate = 0.545 +/- 0.031
```

Interpretation: with topology fixed and actual Step2c trained weights fixed,
the two case mechanisms remain common across label realizations, but not
universal. `G-392` remains a robust SPO+ correction example in about two thirds
of label seeds, while `G-1560` remains a rank-2 promotion example in slightly
over half of label seeds. This supports a topology-conditioned explanation
conditioned on these trained weights, while preserving the caveat that a
smaller relabel-and-retrain check is still needed before claiming retraining
robustness.
