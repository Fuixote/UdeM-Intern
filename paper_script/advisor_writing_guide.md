把你现在已经完成的工作串起来，已经可以形成一条比较完整的 **research flow**，从现象、机制、实验验证，到 toy example 的理论直觉都已经有了。

我建议你现在把整个项目理解成下面这条主线：

```text
Observation:
  SPO+/DFL 在 KEP 上只比 2stage 有 moderate improvement，
  但在 shortest path 上通常 improvement 更明显。

Question:
  为什么 integrated learning 在 KEP 这种问题上不一定比 2stage 值得？

Hypothesis / Property X:
  KEP 属于 decomposable packing-style problem。
  它的 feasible solutions 是由多个相对独立的 components 组成，例如 disjoint cycles/chains。
  因此可能存在很多 close substitute solutions：
  2stage 即使选了不同于 oracle 的 solution，true objective 也可能很接近 oracle。

Empirical evidence:
  1. best / second-best gap analysis
  2. cycle-length sensitivity
  3. arc-density sensitivity
  4. perturb-seed robustness

Toy intuition:
  KEP / stable set / matching / knapsack / partition matroid 可以出现：
    wrong solution identity, small regret
  shortest path / serial path 可以出现：
    wrong path, large regret
```

---

## 1. 现在的整体流程可以画成这样

### Stage 1：先确认 KEP 里是否真的有 close second-best solutions

你先做了 no-good cut，计算每个 method 的 rank-1 和 rank-2 distinct solution：

```text
For each graph and method:
  rank1 = best predicted solution
  add no-good cut
  rank2 = second-best distinct predicted solution
  evaluate both using true weights against oracle
```

这个实验直接回应导师最早的问题：

> compare the gap to the oracle of the best and second-best solutions given by SPO+ and 2stage.

结果是：rank-2 并没有崩掉。max-cycle=3 时，2stage rank-2 mean normalized gap 是 7.04%，SPO+ rank-2 是 6.32%；rank-2 within 5% oracle 分别是 45.19% 和 47.97%。

所以第一层结论是：

> 在 KEP 中，很多 second-best solutions 的 objective value 仍然接近 oracle。
> 这说明 2stage 选到非 oracle solution 并不一定意味着 decision regret 很大。

---

### Stage 2：检查这个现象是否只是 max cycle length = 3 的 artifact

然后你做了 cycle-length sensitivity：

```text
max_cycle = 3, 4, 5
max_chain = 4 fixed
same trained 2stage/SPO+ models
same selected Step2b d8 heldout graphs
```

核心结果是：增加 cycle length 后，rank-2 gap 没有变大，甚至略微下降。

```text
2stage rank2 mean gap:
  K=3: 7.04%
  K=4: 6.93%
  K=5: 6.88%

SPO+ rank2 mean gap:
  K=3: 6.32%
  K=4: 6.20%
  K=5: 5.61%
```

within-5% rate 也没有下降，反而上升。

你还补了 oracle objective change：K=4 时 oracle objective 在 67.75% 的 cases 里增加，K=5 时在 78.50% 的 cases 里增加，说明 longer cycles 确实改变了 feasible set，而不是没起作用。

所以第二层结论是：

> close second-best solution 不是 max-cycle=3 的偶然结果。
> 即使允许 4-cycles 和 5-cycles，很多 rank-2 solutions 仍然接近 oracle。
> Longer cycles 改变 oracle 和 solution identity，但没有系统性让 second-best 远离 oracle。

---

### Stage 3：检查这个现象对 arc density 是否敏感

接着你做了 density sensitivity：

```text
selected mechanism graphs:
  G-696   close alternatives
  G-392   SPO+ region correction
  G-1560  rank-2 promotion

variants:
  original
  add25pct
  add25arcs
  remove25arcs
  remove25pct
```

你先做 seed=42，后来又补了 perturb seeds 0–4，避免单个随机扰动带来偶然结果。最新 density robustness 里有 3 graphs × 5 perturb seeds × variants × methods × ranks，共 300 solution rows。README 里也明确记录了这个 setup。

多 seed 后，add-arcs 的趋势很清楚：rank-2 gap 平均下降。

```text
original:
  2stage rank2 mean gap = 11.38%
  SPO+ rank2 mean gap   = 11.76%

add25pct:
  2stage rank2 mean gap = 5.83%
  SPO+ rank2 mean gap   = 4.86%

add25arcs:
  2stage rank2 mean gap = 6.76%
  SPO+ rank2 mean gap   = 5.69%
```



Oracle objective 也按预期变化：add arcs 时 100% cases oracle objective 增加，remove arcs 时 100% cases oracle objective 下降。

但删除 arcs 更 seed-sensitive：README 里记录 add-arcs variants 的 rank-2 delta 在 seeds 0–4 下稳定为负，而 removal variants 有一些 seed 接近 0 或略微为正。

所以第三层结论是：

> Arc density 比 cycle length 更强烈地改变具体机制。
> 但多 seed 后可以看出，add arcs 通常不仅提高 oracle，也创造更多 high-quality second-best alternatives。
> remove arcs 的影响更依赖删掉哪些 arcs，但没有显示系统性让 rank-2 变差。

---

### Stage 4：逐图解释 mechanism，而不是只看平均

你也已经意识到，density sensitivity 不能只看 3 graphs × seeds 的平均，因为这三张图是机制图，不是随机样本。

逐图解释应该是：

```text
G-696:
  close alternatives mostly preserved

G-392:
  SPO+ region correction mostly preserved under moderate perturbations,
  but strong percentage removal weakens the distinction

G-1560:
  exact 2stage-rank2 -> SPO+-rank1 promotion is fragile / seed-sensitive
```

这个层次很重要，因为它避免误读平均值。平均值说明 robustness，逐图说明 mechanism。

所以第四层结论是：

> 平均表回答“趋势是否被单个 perturb seed 带偏”；逐图表回答“原来的机制是否被保留或破坏”。

---

### Stage 5：toy examples 抽象出 property X

你最新新增的 toy example 工作是目前最接近 paper story 的部分。最新 commit 是：

```text
9187c27910bba3313699bf6b6e53a7c64e53cfd2
Add toy example solution CSV files and corresponding unit tests
```



你现在不仅有 KEP / set packing、stable set、shortest path，还新增了：

```text
weighted matching
cardinality knapsack
partition matroid
serial path control
parametric epsilon packing family
```

这些结果都汇总进 `toy_policy_summary.csv`。

toy examples 的核心 pattern 很清楚：

#### Packing-style examples：wrong identity, small regret

```text
KEP / set packing:
  2stage selects C13 + C24 instead of C12 + C34
  normalized gap = 1%

Stable set:
  2stage selects C + D instead of A + B
  normalized gap = 1%

Weighted matching:
  2stage selects M13 + M24 instead of M12 + M34
  normalized gap = 2%

Knapsack:
  2stage selects C + D instead of A + B
  normalized gap = 2.5%

Partition matroid:
  2stage selects A2 + B2 instead of A1 + B1
  normalized gap = 1%
```



#### Path-like examples：wrong identity, large regret

```text
Shortest path:
  2stage selects Path B instead of Path A
  normalized regret = 60%

Serial path:
  2stage selects high-cost chain instead of low-cost chain
  normalized regret = 50%
```



#### Parametric epsilon family：regret can be made arbitrarily small

你还加入了 parametric packing family，比如 epsilon=0.05 时 gap=5%，epsilon=0.02 时 gap=2%。这说明：

> 在 packing-style problem 里，只要存在 close substitute solution，2stage 的 regret 可以被这个 substitute gap 控制得任意小，即使它选了不同 solution。

toy explanation 里也已经把 property X 写出来：

```text
Close substitute solutions in a decomposable packing structure.
```

并解释了这类问题的 feasible solutions 由多个相对独立 components 组成，替换 component 往往只轻微改变 true objective。

所以第五层结论是：

> 你的实验现象现在已经从 KEP-specific observation，被抽象成一个更一般的 property X：
> **close substitute solutions in decomposable / packing-like combinatorial problems.**

---

# 现在可以绘出的完整研究流程

可以画成下面这个流程：

```text
Step 0. Initial empirical puzzle
        |
        v
SPO+ improves only moderately over 2stage in KEP,
but much more in shortest path.
        |
        v
Step 1. Hypothesis
        |
        v
KEP is a decomposable packing problem.
Solutions are sets of disjoint cycles/chains.
Prediction errors may change solution identity,
but objective value can remain close to oracle.
        |
        v
Step 2. Best / second-best experiment
        |
        v
Use no-good cut to compute rank-1 and rank-2 solutions
for 2stage and SPO+.
        |
        v
Observation:
rank-2 solutions often remain close to oracle.
        |
        v
Step 3. Cycle-length sensitivity
        |
        v
Increase max cycle length from 3 to 4 and 5.
        |
        v
Observation:
rank-2 gap does not systematically increase.
Close second-best behavior persists.
        |
        v
Step 4. Density sensitivity
        |
        v
Perturb arc density on mechanism graphs.
Run multiple perturb seeds.
        |
        v
Observation:
Density changes mechanisms, but add-arcs often create
additional high-quality alternatives.
Removal is seed-sensitive.
        |
        v
Step 5. Toy examples / property X
        |
        v
KEP, stable set, matching, knapsack, partition matroid:
wrong solution identity but small regret.
Shortest path / serial path:
wrong route/path and large regret.
        |
        v
Preliminary conclusion:
DFL is most valuable when prediction errors change objective value,
not merely solution identity.
For decomposable packing problems with close substitutes,
2stage and SPO+/FY may have similar decision performance.
```

---

# 初步结论可以写成这样

## Main preliminary conclusion

> Decision-focused learning is not uniformly valuable across combinatorial optimization problems. Its value depends on whether prediction errors change the true objective value of the selected solution, not merely whether they change solution identity.

## For KEP

> In KEP, solutions are decomposable packings of disjoint cycles/chains. This structure can create close substitute solutions. Therefore, 2stage may select a different exchange set from the oracle but still obtain a true objective value close to oracle. This limits the average room for SPO+/FY to improve.

## For shortest path

> In shortest path, feasible solutions are connected serial structures. Prediction errors can redirect the entire path, so a wrong selected solution can have much larger regret. This makes decision-focused learning more valuable.

## Generalized property X

> Problems with close substitute solutions in decomposable packing-like feasible sets may show similar 2stage and decision-focused performance. Examples include KEP/set packing, stable set, matching, knapsack, and partition matroid. Path-like problems lack this substitutable component structure and can show larger DFL gains.

---

# 现在你最该做的下一步

你现在不需要继续扩实验。你的下一步应该是：

```text
1. 把 toy examples 写进 paper intuition section
2. 把 empirical KEP results 作为 evidence section
3. 把 cycle/density sensitivity 作为 robustness / sensitivity section
4. 开始形成短 paper-style report
```

论文结构可以是：

```text
1. Introduction
   - Puzzle: SPO+ helps less in KEP than shortest path
   - Research question: when is DFL useful?

2. Hypothesis
   - property X: close substitute solutions in decomposable packing structures

3. Toy examples
   - KEP / stable set / matching / knapsack / partition matroid
   - shortest path / serial path contrast
   - parametric epsilon family

4. Empirical KEP evidence
   - best vs second-best no-good cut
   - rank-2 gap results

5. Sensitivity analysis
   - cycle length 3/4/5
   - density perturbation and per-graph mechanisms

6. Discussion / guidelines
   - DFL useful when errors affect objective value
   - less useful when close substitutes make regret small
```

一句话总结：

> 你现在已经从“KEP 上 SPO+ 提升不大”这个现象，推进到了一个可以写进 paper 的 general story：**对于具有 close substitute solutions 的 decomposable packing-style combinatorial problems，2stage 可能已经能得到接近 oracle 的 decision performance；DFL 的收益受限。而 shortest-path 这类 serial/connected problems 中，prediction error 更容易导致大 regret，因此 DFL 更有价值。**
