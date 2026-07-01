## 1. Executive Summary

* **Step4 的三层机制分析是分层清楚的：** Structural Atlas 只定义固定拓扑、feasible cycle/chain candidates 和 candidate conflict graph；Decision Overlay 把 oracle / 2stage / SPO+ 的 selected edge set 映射回 `candidate_id`；Rank-Reversal Detail 只聚焦 2stage 与 SPO+ 选择不同 candidate set 的 context，并给出 true-objective delta。这个三层设计本身是合理的。

* **K18-E1 formal 270 jobs 完整性通过。** `job_rows=270`，`job_status_files=270`，`status_counts.success=270`，`topology_count=18`，`sample_size_counts` 为 50/100/500 各 90，`failures=[]`，`passed=true`。每个 topology 只有一个 test hash，说明 test set 在该 topology 的 15 个 job 内固定。

* **Step4 当前真正深入分析的是 8 个 sentinel topology：** `G-269, G-398, G-784, G-970, G-364, G-836, G-79, G-670`。Handoff 明确说明这些是 Step4 的 interpretation targets。

* **8 个 topology 中，SPO+ 的改善/伤害/无变化与 candidate-set switching 高度一致。** G-398、G-269、G-784 是 clean / strong beneficial candidate-set correction；G-970、G-364 是 sample-size-responsive beneficial correction；G-79 是 harmful overcorrection；G-836 是 small negative with mostly same decision；G-670 是 no-room decision-equivalent。

* **结构复杂度有解释力，但不能单独解释结果。** 例如 G-398 与 G-670 都是 `medium_rich / cycle_chain / easy_control`，且 conflict density 都较高，但 G-398 是 100% beneficial recovery，G-670 是完全 no-room neutral。结构提供 substrate，真正的 DFL 机制要看 oracle / model candidate ranking 与 conflict resolution。

* **G-398 是最干净的 beneficial case。** 所有 5000 contexts、所有 sample sizes 中，2stage 固定选 `c0010|c0030`，SPO+ 固定选 oracle 的 `c0002|c0009|c0032`，beneficial reversal rate = 1.0，mean delta = 11.704。

* **G-269 是 strong beneficial case。** 2stage 近乎固定选 `c0005|c0033`，SPO+ 固定选 `c0003|c0010`；sample_size=100 时，oracle 对 `c0003,c0010` 的选择率为 0.941，2stage 仅 0.0058，SPO+ 为 1.0。

* **G-784 是 rich replacement。** 主切换是 2stage 的 `c0002|c0011|c0026|c0043` 被 SPO+ 替换为 `c0002|c0011|c0027|c0042`；sample_size=100 时，2stage 对 `c0026,c0043` 的选择率是 0.9862，而 SPO+ 对 `c0027,c0042` 的选择率约 0.989/0.9998。

* **G-970 的 sample-size effect 是真实 candidate-set switching，不只是平均 gap 变化。** 不同 decision rate 从 0.3964 增至 0.5424，beneficial rate 从 0.3308 增至 0.4590；SPO+ 逐步离开 2stage 固定的 `c0000|c0028|c0080`，转向包含 `c0003,c0034,c0058` 等 oracle-heavy candidates 的解。

* **G-79 是 harmful control。** SPO+ 越来越偏向 `c0004|c0020`，而这组 substitution 在 sample_size=500 下 harmful reversal count=1549，rate_total=0.3098，mean_delta=-6.383。

* **G-670 是 no-room neutral。** 2stage、SPO+、oracle 在所有 sample sizes 中都选择同一 candidate `c0036`；different-decision contexts = 0。

---

## 2. Experiment Setup Reconstruction

### Step3 / K18-E1 context

本实验的 outcome convention 是：

```text
Delta = test_mean_decision_gap_2stage - test_mean_decision_gap_spoplus
Delta > 0 means SPO+ has lower downstream decision gap than 2stage.
sample_size 50/100/500 means train/validation splits 40/10, 80/20, 400/100.
```

这是 handoff 明确给出的约定。

K18-E1 formal run 的完整性文件显示：

| item               |        value |
| ------------------ | -----------: |
| job rows           |          270 |
| job status files   |          270 |
| success jobs       |          270 |
| topology count     |           18 |
| data seed count    |            5 |
| sample sizes       | 50, 100, 500 |
| training sizes     |  40, 80, 400 |
| validation sizes   |  10, 20, 100 |
| unique test hashes |           18 |
| failures           |            0 |

这些数字来自 post-run integrity audit。

### Step4 设计

Step4 当前有三层：

1. **Topology Structural Atlas**
   只从 immutable Step3 topology template 出发，构造 compatibility graph、feasible cycle/chain candidate set、candidate conflict graph 和 structural metrics。它不依赖 model weights、`X`、`y`、data seed、sample size 或 test sample。

2. **Decision Overlay**
   对每个 audited context，把 oracle selected edge set、2stage selected edge set、SPO+ selected edge set 映射回 Structural Atlas 的 `candidate_id`，并按 `topology_id × sample_size × candidate_id` 汇总 selection frequency。

3. **Rank-Reversal Detail**
   从 Decision Overlay 的 raw decision rows 出发，只选择 2stage 与 SPO+ 选了不同 structural candidate set 的 contexts，并记录 true objective delta；这个表不是完整 Top-M analysis，而是为后续 Top-M/rank-reversal 计算缩小目标范围。

### Source of truth

当前 GitHub 中 compact source-of-truth 是：

* `formal_post_run_integrity_audit.json`
* `formal_post_run_sample_size_summary.csv`
* `formal_post_run_topology_sample_summary.csv`
* `formal_post_run_vertical_pattern_check.csv`
* `topology_summary.csv`
* `feasible_candidates.csv`
* `candidate_overlay_summary.csv`
* `rank_reversal_case_summary_by_topology_sample.csv`
* `candidate_set_switch_summary.csv`
* `rank_reversal_top20_per_topology.csv`

Handoff 说明 raw `decision_solution_rows.csv` 约 199MB，不在 GitHub 中跟踪；因此本文不假设可以重算所有 context-level raw decisions，只使用 compact summaries 与 tracked target tables。

---

## 3. Data Integrity Check

### 3.1 Formal run integrity

Post-run audit 通过，且没有 failures：

```text
job_rows = 270
job_status_files = 270
status_counts.success = 270
failures = []
passed = true
```

同时每个 topology 的 test hash count 都是 1，说明同一个 topology 的 15 个 jobs 使用同一个 fixed test set。

### 3.2 Sample-size denominator consistency

Step4 Decision Overlay 的 denominator 是：

```text
5 data seeds × 1000 test contexts = 5000 contexts
```

这可以直接从 `candidate_overlay_summary.csv` 中每个 candidate 的 `oracle_denominator=two_stage_denominator=spoplus_denominator=5000` 看出。例如 G-269、G-398、G-784、G-970 的 rows 都使用 5000 分母。

Rank-Reversal summary 也用 `total_contexts=5000`。

### 3.3 K18-E1 sample-size aggregate

全体 18 topology 上，sample_size 变化如下：

| sample_size | jobs | mean 2stage gap | mean SPO+ gap | mean Δ | mean fraction improved | mean runtime |
| ----------: | ---: | --------------: | ------------: | -----: | ---------------------: | -----------: |
|          50 |   90 |          8.4222 |        4.6856 | 3.7366 |                 0.3787 |       227.8s |
|         100 |   90 |          8.4025 |        4.5752 | 3.8273 |                 0.3904 |       432.9s |
|         500 |   90 |          8.3964 |        4.4326 | 3.9638 |                 0.4140 |      2471.2s |

Formula used:

```text
mean Δ = mean_test_gap_2stage - mean_test_gap_spoplus
```

The CSV already reports `mean_test_gap_improvement`; the values match the formula, e.g. for sample_size=500:

```text
8.396370681296455 - 4.432619785939322 = 3.963750895357133
```

Interpretation: increasing sample size barely changes 2stage gap, but improves SPO+ gap more noticeably.

### 3.4 Minor caveat

Runtime source has 4 missing entries in the formal launcher log, but all 270 job statuses are success. This affects exact runtime accounting for those rows, not candidate selection or decision gap interpretation.

---

## 4. Structural Mechanism Analysis

### 4.1 Structural table for the 8 Step4 sentinel topologies

| topology | structural type     | candidates | cycles | chains | conflict density | key structural interpretation                                                 |
| -------- | ------------------- | ---------: | -----: | -----: | ---------------: | ----------------------------------------------------------------------------- |
| G-269    | chain-only          |         40 |      0 |     40 |            0.867 | two NDD-rooted chain families, high conflict, near winner-take-all correction |
| G-398    | cycle-chain         |         40 |      3 |     37 |            0.786 | mixed cycle+chain, clean oracle recovery despite “easy_control”               |
| G-784    | cycle-chain rich    |         58 |     12 |     46 |            0.858 | rich candidate replacement, many cycle/chain alternatives                     |
| G-970    | extreme cycle-chain |         86 |      6 |     80 |            0.809 | many chains, sample-size-responsive correction                                |
| G-364    | sparse chain-only   |          7 |      0 |      7 |            0.762 | sparse bottleneck correction; few candidates but nontrivial switch            |
| G-836    | chain-only          |         39 |      0 |     39 |            0.633 | matched negative, mostly same decision                                        |
| G-79     | cycle-chain         |         27 |      2 |     25 |            0.946 | high-conflict harmful overcorrection                                          |
| G-670    | cycle-chain         |         37 |      4 |     33 |            0.887 | no-room neutral; one candidate dominates all methods                          |

These numbers are from Structural Atlas `topology_summary.csv`.

### 4.2 Chain-only vs cycle-chain

* **Chain-only:** G-269, G-364, G-836. All have `num_cycles_total=0`; feasible candidates are all chains.
* **Cycle-chain:** G-398, G-784, G-970, G-79, G-670. They contain both cycle candidates and chains.

But chain-only does not imply positive or negative: G-269 is strongly positive, G-364 is sample-size-responsive positive, while G-836 is small negative. Therefore topology class alone is insufficient.

### 4.3 High conflict and winner-take-all

All eight topologies have one connected candidate conflict component and high largest-component fraction = 1.0. Conflict density ranges from 0.633 in G-836 to 0.946 in G-79.

However, the high-conflict substrate yields different mechanisms:

* **G-398:** high conflict but clean beneficial recovery: 2stage always chooses wrong set, SPO+ always chooses oracle set.
* **G-670:** high conflict but no-room: all methods choose `c0036`.
* **G-79:** high conflict and harmful overcorrection: SPO+ increasingly chooses harmful `c0004|c0020`.
* **G-269:** high conflict and strong beneficial correction: SPO+ replaces `c0005|c0033` with `c0003|c0010`.

Thus conflict density is a substrate for large decision effects, but sign depends on which candidate set is promoted.

### 4.4 Structural substrates of key candidate substitutions

#### G-269

Target switch:

```text
2stage: c0005 | c0033
SPO+:   c0003 | c0010
```

Candidate signatures:

```text
c0003 = chain:20->12->14->15->5
c0005 = chain:20->12->5->7
c0010 = chain:21->10->1->7
c0033 = chain:21->13->3->14->15
```

This is a two-chain replacement across the two NDD roots. SPO+ selects longer/deeper chain combination `c0003|c0010` that matches oracle in most contexts.

#### G-398

Target switch:

```text
2stage: c0010 | c0030
SPO+:   c0002 | c0009 | c0032
```

Candidate signatures:

```text
c0002 = cycle:9->14->13->9
c0009 = chain:21->16->18->2
c0010 = chain:21->16->18->2->15
c0030 = chain:22->14->13->9->7
c0032 = chain:22->15
```

Mechanistically, 2stage selects two long chains, while SPO+ selects one 3-cycle plus two chains, matching oracle.

#### G-784

Main replacement:

```text
2stage: c0002 | c0011 | c0026 | c0043
SPO+:   c0002 | c0011 | c0027 | c0042
```

Candidate signatures for the replaced pair:

```text
c0026 = chain:20->19->14
c0027 = chain:20->6
c0042 = chain:21->14
c0043 = chain:21->6
```

This is a rich replacement of two one/two-hop chains around nodes 6/14 and two NDD roots; the replacement changes which NDD reaches which recipient.

#### G-970

Repeated 2stage set:

```text
2stage: c0000 | c0028 | c0080
```

Key SPO+ / oracle-heavy candidates:

```text
c0003 = cycle:1->3->19->1
c0034 = chain:21->20
c0058 = chain:22->17->6->13
```

Candidate signatures show c0003 is a cycle, while c0034/c0058 are chains.

2stage’s fixed set contains `c0000` and two chains `c0028,c0080`; SPO+ increasingly moves toward candidate sets containing `c0003,c0034,c0058`.

#### G-364

Sparse switch:

```text
2stage: c0000 | c0003
SPO+:   c0000 | c0006
```

Candidate signatures:

```text
c0000 = chain:20->6
c0003 = chain:21->9->12->18
c0006 = chain:21->9->18
```

Only 7 candidates exist, yet SPO+ correction is strong and sample-size responsive.

#### G-79

Main harmful switch:

```text
2stage: c0001 | c0015
SPO+:   c0004 | c0020
```

Candidate signatures:

```text
c0001 = cycle:2->10->2
c0015 = chain:22->13->19
c0004 = chain:21->10->13->19
c0020 = chain:22->9
```

This is a harmful overcorrection from a short cycle + chain to two chains, one of which (`c0004`) is over-selected by SPO+ despite oracle selection only 0.143 at sample_size=500.

#### G-836

Small negative switch:

```text
c0035 <-> c0037
```

Candidate signatures:

```text
c0035 = chain:23->8->6->14->2
c0037 = chain:23->8->6->18
```

Most contexts do not switch; when they do, the difference is small in aggregate but can be locally harmful.

#### G-670

No-room dominant candidate:

```text
c0036 = chain:23->7->21
```

Candidate overlay shows `c0036` selected by oracle, 2stage, and SPO+ at rate 1.0.

---

## 5. Decision Overlay Analysis

### 5.1 G-398: clean oracle recovery

For sample_size=100:

| candidate | oracle rate | 2stage rate | SPO+ rate |
| --------- | ----------: | ----------: | --------: |
| c0002     |         1.0 |         0.0 |       1.0 |
| c0009     |         1.0 |         0.0 |       1.0 |
| c0032     |         1.0 |         0.0 |       1.0 |
| c0010     |         0.0 |         1.0 |       0.0 |
| c0030     |         0.0 |         1.0 |       0.0 |

Rank-reversal summary confirms all 5000 contexts are beneficial reversals at every sample_size.

**Conclusion:** This hypothesis is fully verified.

### 5.2 G-269: strong beneficial case

For sample_size=100:

| candidate | oracle rate | 2stage rate | SPO+ rate |
| --------- | ----------: | ----------: | --------: |
| c0003     |       0.941 |      0.0058 |       1.0 |
| c0010     |       0.941 |      0.0058 |       1.0 |
| c0005     |       0.059 |      0.9942 |       0.0 |
| c0033     |       0.059 |      0.9942 |       0.0 |

At sample_size=50 and 500 the same pattern holds, with two_stage selection of `c0005,c0033` around 0.99 and SPO+ selection of `c0003,c0010` equal to 1.0.

Rank reversal:

```text
2stage c0005|c0033 -> SPO+ c0003|c0010
beneficial count ≈ 4664–4676 / 5000
harmful count = 295 / 5000
```

**Conclusion:** Verified. It is not only mean-gap positive; it is a systematic candidate-set correction.

### 5.3 G-784: rich candidate replacement

At sample_size=100:

| candidate | oracle rate | 2stage rate | SPO+ rate |
| --------- | ----------: | ----------: | --------: |
| c0026     |       0.010 |      0.9862 |    0.0002 |
| c0043     |       0.006 |      0.9862 |    0.0002 |
| c0027     |       0.636 |      0.0108 |     0.989 |
| c0042     |       0.990 |      0.0138 |    0.9998 |

Dominant switch:

```text
2stage: c0002|c0011|c0026|c0043
SPO+:   c0002|c0011|c0027|c0042
beneficial count = 4862 / 5000 at sample_size=50
beneficial count = 4834 / 5000 at sample_size=100
```

Rank summary confirms beneficial rate around 0.98 at all sample sizes.

**Conclusion:** Verified. This is a rich replacement, not a single-candidate correction.

### 5.4 G-970: sample-size-responsive positive

G-970’s 2stage fixed set includes:

```text
c0000, c0028, c0080
```

At sample_size=50:

| candidate | oracle | 2stage |                                                                   SPO+ |
| --------- | -----: | -----: | ---------------------------------------------------------------------: |
| c0000     |  0.066 |    1.0 |                                                                 0.6036 |
| c0028     |  0.049 |    1.0 |                                                                 0.6036 |
| c0080     |  0.050 |    1.0 | not shown in same excerpt but switch summary confirms fixed 2stage set |
| c0003     |  0.932 |    0.0 |                                                                 0.3964 |
| c0034     |  0.949 |    0.0 |                                                                 0.2604 |
| c0058     |  0.811 |    0.0 |                                                                 0.1222 |

At sample_size=100, SPO+ moves further toward oracle-heavy candidates:

```text
c0003: 0.467
c0034: 0.3398
c0058: 0.1404
c0000/c0028: 0.533
```

At sample_size=500:

```text
c0003: 0.5424
c0034: 0.3972
c0058: 0.1614
c0000/c0028/c0080: 0.4576
```

Rank-reversal summary shows the same pattern:

```text
different_decision_rate: 0.3964 -> 0.4670 -> 0.5424
beneficial_rate_total:   0.3308 -> 0.3958 -> 0.4590
```

**Conclusion:** Verified. The sample-size effect is a true increase in candidate-set switching away from 2stage’s fixed bad set.

### 5.5 G-364: sparse sample-size-responsive positive

Candidate overlay:

| sample_size | c0003 SPO+ rate | c0006 SPO+ rate |
| ----------: | --------------: | --------------: |
|          50 |          0.6598 |          0.3362 |
|         100 |          0.4696 |          0.5290 |
|         500 |          0.1224 |          0.8776 |

2stage is almost fixed on `c0000|c0003`; SPO+ increasingly shifts from `c0003` to `c0006`.

Rank-reversal rates:

```text
different_decision_rate: 0.3398 -> 0.5302 -> 0.8776
beneficial_rate_total:   0.2926 -> 0.4546 -> 0.7392
```

Dominant switch:

```text
2stage c0000|c0003 -> SPO+ c0000|c0006
beneficial count: 1445 -> 2267 -> 3696
```

**Conclusion:** Verified. This is the clearest sparse bottleneck correction.

### 5.6 G-79: harmful control

At sample_size=50, SPO+ already over-selects `c0004,c0020` relative to 2stage:

```text
c0004: 2stage 0.603, SPO+ 0.8612
c0020: 2stage 0.603, SPO+ 0.8612
c0001: 2stage 0.397, SPO+ 0.1388
c0015: 2stage 0.397, SPO+ 0.1388
```

At sample_size=500, this over-selection strengthens:

```text
c0004: 2stage 0.4754, SPO+ 0.9296
c0020: 2stage 0.4754, SPO+ 0.9296
c0001/c0015: 2stage 0.5246, SPO+ 0.0704
```

Switch summary at sample_size=500:

```text
2stage c0001|c0015 -> SPO+ c0004|c0020
harmful count = 1549
rate_total = 0.3098
mean_delta = -6.3827
```

**Conclusion:** Verified. G-79 is not merely “SPO+ fails to improve”; it systematically over-selects a harmful candidate set.

### 5.7 G-836: small negative, mostly same decision

Rank summary:

```text
same decision contexts:
sample_size 50: 4830 / 5000
sample_size 100: 4838 / 5000
sample_size 500: 4810 / 5000
```

Different-decision rate is only 0.0324–0.038.

The main small switch is:

```text
c0037 -> c0035 harmful
c0035 -> c0037 beneficial
```

At sample_size=500:

```text
harmful count = 120
beneficial count = 70
```

Candidate overlay confirms small rate differences: at sample_size=100, `c0035` is 0.554 for 2stage and 0.5612 for SPO+; `c0037` is 0.446 for 2stage and 0.4388 for SPO+.

**Conclusion:** Verified. G-836 is a small negative because models usually choose the same candidate set, not because SPO+ makes frequent large mistakes.

### 5.8 G-670: no-room neutral

Candidate `G-670:c0036` is selected by oracle, 2stage, and SPO+ at rate 1.0 for sample_size=50 in the overlay.

Rank summary confirms:

```text
same_decision_contexts = 5000
different_decision_contexts = 0
for sample_size 50, 100, 500
```

**Conclusion:** Verified. G-670 is decision-equivalent / no-room.

---

## 6. Rank-Reversal Analysis

### 6.1 Different-decision rates

| topology | sample 50 | sample 100 | sample 500 | interpretation                             |
| -------- | --------: | ---------: | ---------: | ------------------------------------------ |
| G-398    |     1.000 |      1.000 |      1.000 | always different, always beneficial        |
| G-269    |    0.9938 |     0.9942 |     0.9918 | nearly always different, mostly beneficial |
| G-784    |    0.9910 |     0.9890 |     0.9920 | nearly always different, mostly beneficial |
| G-970    |    0.3964 |     0.4670 |     0.5424 | sample-size-responsive switching           |
| G-364    |    0.3398 |     0.5302 |     0.8776 | sparse sample-size-responsive switching    |
| G-79     |    0.2678 |     0.4060 |     0.4542 | harmful switching grows with sample size   |
| G-836    |    0.0340 |     0.0324 |     0.0380 | mostly same decision                       |
| G-670    |    0.0000 |     0.0000 |     0.0000 | exact no-room                              |

### 6.2 Beneficial / harmful reversal rates

| topology | sample_size |   beneficial |      harmful | mean delta different |
| -------- | ----------: | -----------: | -----------: | -------------------: |
| G-398    |  50/100/500 |        1.000 |        0.000 |               11.704 |
| G-269    |  50/100/500 | ~0.933–0.935 |        0.059 |         ~15.39–15.41 |
| G-784    |  50/100/500 | ~0.978–0.982 | ~0.010–0.011 |         ~11.04–11.07 |
| G-970    |      50→500 |  0.331→0.459 |  0.066→0.083 |           ~13.2–13.7 |
| G-364    |      50→500 |  0.293→0.739 |  0.047→0.138 |           ~3.52–3.83 |
| G-79     |      50→500 |  0.081→0.144 |  0.186→0.310 |        negative mean |
| G-836    |      50→500 |  0.013→0.014 |  0.021→0.024 |       small negative |
| G-670    |         all |            0 |            0 |                    0 |

All values are from `rank_reversal_case_summary_by_topology_sample.csv`.

### 6.3 Most important recurring substitutions

| priority | topology | substitution | direction      | why important    |                   |                                                 |                                              |                   |                                                        |                                      |
| -------: | -------- | ------------ | -------------- | ---------------- | ----------------- | ----------------------------------------------- | -------------------------------------------- | ----------------- | ------------------------------------------------------ | ------------------------------------ |
|        1 | G-398    | `c0010       | c0030 -> c0002 | c0009            | c0032`            | beneficial                                      | cleanest oracle recovery, 5000/5000 contexts |                   |                                                        |                                      |
|        2 | G-269    | `c0005       | c0033 -> c0003 | c0010`           | mostly beneficial | strong correction with large max delta          |                                              |                   |                                                        |                                      |
|        3 | G-784    | `c0002       | c0011          | c0026            | c0043 -> c0002    | c0011                                           | c0027                                        | c0042`            | beneficial                                             | rich replacement, two-candidate swap |
|        4 | G-970    | `c0000       | c0028          | c0080 -> c0003   | c0005             | c0034                                           | c0036` and variants                          | mostly beneficial | sample-size-responsive departure from fixed 2stage set |                                      |
|        5 | G-364    | `c0000       | c0003 -> c0000 | c0006`           | beneficial        | sparse bottleneck correction                    |                                              |                   |                                                        |                                      |
|        6 | G-79     | `c0001       | c0015 -> c0004 | c0020`           | harmful           | strongest harmful mechanism                     |                                              |                   |                                                        |                                      |
|        7 | G-836    | `c0011       | c0037 -> c0011 | c0035`           | harmful small     | mostly same decision, small systematic negative |                                              |                   |                                                        |                                      |
|        8 | G-670    | no switch    | tie            | no-room baseline |                   |                                                 |                                              |                   |                                                        |                                      |

Switch details and example contexts are in `candidate_set_switch_summary.csv`; examples include G-398 `G-000837.json`, G-269 `G-000929.json`, G-364 `G-000156.json`, G-79 `G-000656.json`, G-836 `G-000744.json`, and G-970 `G-000491.json` / `G-000602.json`.

### 6.4 Sample-size effects as real switch-frequency changes

This is clearest in:

* **G-364:** different-decision rate 0.3398 → 0.5302 → 0.8776; dominant beneficial switch count 1445 → 2267 → 3696.
* **G-970:** different-decision rate 0.3964 → 0.4670 → 0.5424; top beneficial switch count 757 → 982 → 1132.
* **G-79:** harmful rate grows from 0.1864 to 0.3098, with harmful switch count 1549 at sample_size=500.

So sample-size effects are not merely changes in mean gap; they are changes in candidate-set switch frequency.

---

## 7. Mechanism Taxonomy

| category                              | topology | mechanism                                                                              |                                          |                                           |
| ------------------------------------- | -------- | -------------------------------------------------------------------------------------- | ---------------------------------------- | ----------------------------------------- |
| clean oracle recovery                 | G-398    | 2stage always selects wrong two-chain set; SPO+ always recovers oracle cycle+chain set |                                          |                                           |
| strong candidate-set correction       | G-269    | 2stage nearly always selects `c0005                                                    | c0033`; SPO+ selects oracle-heavy `c0003 | c0010`                                    |
| rich replacement                      | G-784    | SPO+ replaces `c0026                                                                   | c0043`with`c0027                         | c0042` inside a rich cycle-chain topology |
| sample-size-responsive positive       | G-970    | SPO+ gradually leaves fixed 2stage set `c0000                                          | c0028                                    | c0080` as sample size grows               |
| sparse bottleneck correction          | G-364    | few candidates; SPO+ increasingly switches from `c0003` to `c0006`                     |                                          |                                           |
| harmful overcorrection                | G-79     | SPO+ increasingly over-selects `c0004                                                  | c0020`, which is harmful on average      |                                           |
| small negative / mostly same decision | G-836    | only 3–4% contexts differ; small imbalance toward harmful `c0035`                      |                                          |                                           |
| no-room decision-equivalent           | G-670    | oracle, 2stage, and SPO+ all select `c0036`; no switch possible                        |                                          |                                           |

---

## 8. What This Does NOT Prove

1. **K18 is not population representative.** It is a mechanism-coverage sample selected from prior Step3 evidence, not a random sample of all 160 topology candidates. The K18 README explicitly frames it as a screening / mechanism-study stage, not final confirmation.

2. **Decision Overlay is selected-solution analysis, not a complete Top-M landscape.** It maps selected edge sets back to structural candidate IDs and summarizes frequencies; it does not enumerate full predicted/true Top-M landscapes for every context.

3. **Rank-Reversal Detail narrows targets; it does not replace full Top-M.** Its README says the target planner “does not replace Top-M analysis” and is a bridge to detailed Top-M/rank-reversal computation.

4. **Raw `decision_solution_rows.csv` is not tracked in GitHub.** The handoff states that the raw file is about 199 MB and intentionally not tracked; therefore conclusions here rely on compact summary and target tables, not arbitrary re-filtering of raw context rows.

5. **Do not confuse sample_size=50 here with old screen train_size=50.** Here sample_size=50 means train/validation = 40/10. The handoff explicitly defines that convention.

---

## 9. Next Analysis Plan

### 9.1 Candidate-set switches to visualize first

I would prioritize these 10:

| priority | topology | switch                              | example context                |                |                                           |                                           |                                    |                                           |                                           |
| -------: | -------- | ----------------------------------- | ------------------------------ | -------------- | ----------------------------------------- | ----------------------------------------- | ---------------------------------- | ----------------------------------------- | ----------------------------------------- |
|        1 | G-398    | `c0010                              | c0030 -> c0002                 | c0009          | c0032`                                    | `data_seed=101, sample=50, G-000837.json` |                                    |                                           |                                           |
|        2 | G-269    | `c0005                              | c0033 -> c0003                 | c0010`         | `data_seed=101, sample=50, G-000929.json` |                                           |                                    |                                           |                                           |
|        3 | G-784    | `c0002                              | c0011                          | c0026          | c0043 -> c0002                            | c0011                                     | c0027                              | c0042`                                    | `data_seed=101, sample=50, G-000472.json` |
|        4 | G-970    | `c0000                              | c0028                          | c0080 -> c0003 | c0005                                     | c0034                                     | c0036`                             | `data_seed=102, sample=50, G-000491.json` |                                           |
|        5 | G-970    | `c0000                              | c0028                          | c0080 -> c0003 | c0034                                     | c0058`                                    | `data_seed=102/104, G-000602.json` |                                           |                                           |
|        6 | G-364    | `c0000                              | c0003 -> c0000                 | c0006`         | `data_seed=101, G-000156.json`            |                                           |                                    |                                           |                                           |
|        7 | G-79     | `c0001                              | c0015 -> c0004                 | c0020` harmful | `data_seed=101, G-000656.json`            |                                           |                                    |                                           |                                           |
|        8 | G-79     | same switch but beneficial minority | `data_seed=101, G-000141.json` |                |                                           |                                           |                                    |                                           |                                           |
|        9 | G-836    | `c0011                              | c0037 -> c0011                 | c0035` harmful | `data_seed=101, G-000744.json`            |                                           |                                    |                                           |                                           |
|       10 | G-670    | no switch, `c0036` all methods      | choose any context             |                |                                           |                                           |                                    |                                           |                                           |

These example contexts and deltas are recorded in `candidate_set_switch_summary.csv` and `rank_reversal_top20_per_topology.csv`.

### 9.2 What to plot for each switch

For each selected switch, produce four figures:

1. **Compatibility graph overlay**
   Nodes = patient-donor pairs / NDDs. Edges = compatibility arcs. Highlight edges used by oracle, 2stage, SPO+.

2. **Candidate conflict graph**
   Candidate nodes colored by selected set: oracle-only, 2stage-only, SPO+-only, shared. Conflict edges show why two candidates cannot co-exist.

3. **Selected solution overlay**
   Candidate table with true objective, 2stage predicted score, SPO+ predicted score, oracle rank, 2stage rank, SPO+ rank.

4. **Top-M landscape**
   For the same context, enumerate true Top-M, 2stage-predicted Top-M, SPO+-predicted Top-M. This is necessary to verify whether SPO+ works by candidate ranking correction rather than global prediction accuracy.

### 9.3 How to test “ranking correction” directly

For each target context:

```text
rank_oracle_solution_under_2stage
rank_oracle_solution_under_spoplus
score_margin_oracle_vs_2stage_selected
score_margin_oracle_vs_spoplus_selected
Jaccard(oracle selected candidates, model selected candidates)
```

Hypothesis to test:

```text
In beneficial cases, SPO+ improves by moving the oracle candidate set, or an oracle-overlapping set, upward in predicted decision ranking.
```

This is strongly suggested by G-398, G-269, G-784, G-970, and G-364, but it still requires Top-M confirmation.

### 9.4 How to explain harmful cases

For G-79 and G-836:

* Plot harmful and beneficial examples separately.
* Compare the same substitution when it is harmful versus beneficial.
* Identify whether the candidate selected by SPO+ is:

  * high-conflict,
  * oracle-frequent but context-sensitive,
  * or a candidate whose predicted score is overcorrected.

For G-79, the crucial question is why SPO+ promotes `c0004|c0020` so aggressively even though that set is harmful in the majority of switching contexts.

---

## Condensed mechanism narrative for report/paper

In the K18-E1 fixed-topology kidney-exchange experiment, the effect of SPO+ is best understood not as a uniform improvement in prediction quality, but as a topology-specific reordering of feasible exchange candidates under conflict constraints. Structural Atlas shows that the eight sentinel topologies range from sparse chain-only graphs to rich cycle-chain graphs with dense candidate conflict structure. Decision Overlay and Rank-Reversal Detail reveal that large positive SPO+ effects correspond to systematic candidate-set correction: in G-398, SPO+ always recovers the oracle set `c0002|c0009|c0032` instead of the 2stage set `c0010|c0030`; in G-269, SPO+ replaces `c0005|c0033` with the oracle-heavy `c0003|c0010`; and in G-784, SPO+ performs a richer replacement from `c0026|c0043` toward `c0027|c0042`. Sample size matters primarily when it changes candidate-set switching frequency, as in G-970 and G-364, where larger sample sizes make SPO+ increasingly leave the 2stage-selected candidate set. Negative controls show the opposite mechanism: in G-79, SPO+ increasingly favors the harmful `c0004|c0020` set, while G-836 shows only rare, small harmful switches and G-670 is decision-equivalent with no switch at all. Thus, the main mechanism hypothesis supported by Step4 is that SPO+ improves or hurts downstream kidney-exchange decisions by changing the ranking and selection of conflict-constrained feasible candidate sets, not by uniformly reducing prediction error across all arcs.
