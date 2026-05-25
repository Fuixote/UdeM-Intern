可以 move on 到 Step2，而且你现在对 Step2 的定位是对的：**Step2 不应该只是“再跑一遍 FY/SPO+”，而应该系统性改变 synthetic label generation，让 FY 和 SPO+ 在不同 misspecification / noise / scale regime 下暴露差异。**

我读完 SPO 原文第 27–30 页后，最值得借鉴的不是某一个具体公式，而是它的实验设计思想：

> 固定 downstream optimization problem 和模型类，然后通过 synthetic data generation 中的 `degree` 和 `noise` 控制“真实条件均值”和“学习模型类”之间的 misspecification 程度。

在 SPO 的 shortest-path 实验里，他们生成特征 (x_i\sim N(0,I_p))，随机生成 (B^\star)，然后用一个 polynomial degree 参数生成 cost vector；虽然 ground truth 是非线性的，但训练时仍然用 linear hypothesis class，因此 `deg` 越大，模型 misspecification 越强。论文也明确说，当 `deg=1` 时条件均值基本是线性的，随着 `deg` 增大，SPO+ 相对 LS/absolute loss 的优势通常更明显。第 29 页 Figure 4 正是 normalized SPO loss vs polynomial degree 的图，展示了这个现象。

## Step2 的总原则

我建议 Step2 里每个 a/b/c/d 都是一个 **label regime**，而不是一个 training method。每个 label regime 下都跑同一套方法：

```text
2stage MSE
FY selected by validation FY loss
FY selected by validation decision gap   # diagnostic
SPO+ selected by validation SPO+ loss
SPO+ selected by validation decision gap # diagnostic
```

并保持这些东西固定：

```text
graph generation / graph structure
train/validation/test split
train sizes
model class: w_hat_e = theta_1 u_e + theta_2 c_e
theta_seed
metric_stride
evaluation metric: synthetic-label decision gap
```

这样 Step2 的核心问题才干净：

> 当真实 reward label 从 well-specified linear-noisy 逐渐变成 nonlinear / heteroskedastic / correlated-noise / misspecified 时，FY 和 SPO+ 的表现如何变化？

## Step2a：你的 additive Gaussian label 很适合作为桥接实验

你现在的想法：

[
w^{syn}_e = \max(0, 10u_e + 5c_e + \epsilon_e)
]

其中：

[
\epsilon_e \sim N(0, 10\bar u + 5\bar c)
]

这个很适合作为 Step2a，因为它和 Step1 的 noisy-linear label 很接近，但把噪声从 multiplicative 改成 additive。它的科学定位应该是：

```text
Step2a = additive-noise linear benchmark
```

也就是：条件均值仍然基本是线性的，所以 2stage MSE 理论上应该很强；FY/SPO+ 如果能显著赢，说明它们真的利用了 KEP decision structure，而不是只是在修复 nonlinear label misspecification。

我建议把你的噪声写得更精确一点。正态分布第二个参数容易有歧义，是 variance 还是 standard deviation。建议定义成：

[
\mu_G = 10\bar u_G + 5\bar c_G
]

[
\epsilon_e \sim N(0, (\rho \mu_G)^2)
]

[
w^{syn}_e = \max(0, 10u_e + 5c_e + \epsilon_e)
]

其中 (\rho) 控制 noise strength。你现在写的 (N(0,10\bar u+5\bar c)) 如果是把第二项当 **standard deviation**，那等价于 (\rho=1)，噪声会比较强，很多低权重 edge 会被截断为 0；如果是 variance，则标准差是 (\sqrt{10\bar u+5\bar c})，噪声温和很多。建议明确写成 `rho` 版本，第一轮可以用：

```text
rho ∈ {0.25, 0.5, 1.0}
main: rho = 0.5 或 1.0
```

不要一开始只固定一个噪声强度，否则如果结果很弱，很难判断是 surrogate 问题还是噪声太小/太大。

## Step2b：直接借鉴 SPO 的 degree，做 polynomial misspecification

SPO 第 27 页最核心的 synthetic design 是：

[
c_{ij}
======

\left[
\left(
\frac{1}{\sqrt p}(B^\star x_i)_j + 3
\right)^{deg}

* 1
  \right]
  \cdot \epsilon^j_i
  ]

其中 `deg` 控制非线性程度；训练时仍然用 linear model，因此 `deg` 是 misspecification knob。

在你的 KEP setting 里，我们没有像 shortest path paper 那样的 (x_i\to c_i) contextual prediction setup；你的 edge features 就是 (u_e, c_e)。所以我建议不要机械复制 (B^\star x)，而是复制 **degree-controlled nonlinear conditional mean** 这个思想。

定义：

[
b_e = 10u_e + 5c_e
]

对每个 graph (G)，令：

[
\mu_G = \frac{1}{|E_G|}\sum_{e\in E_G} b_e
]

[
q_e = \frac{b_e}{\mu_G+\delta}
]

然后用 SPO 风格的 shifted polynomial：

[
r^{(d)}_e = (q_e + \kappa)^d - \kappa^d
]

再做 graph-level rescaling，让不同 degree 的平均 reward scale 可比：

[
m^{(d)}_e
=========

\mu_G
\cdot
\frac{r^{(d)}*e}
{\frac{1}{|E_G|}\sum*{e'\in E_G} r^{(d)}_{e'}+\delta}
]

最后：

[
w^{syn}_e = \max(0, m^{(d)}_e)
]

建议：

```text
kappa = 3
degree d ∈ {1, 2, 4, 8}
```

这个设计有几个优点：

1. 当 (d=1) 时，基本回到 linear label：
   [
   r_e^{(1)} = q_e
   ]
   rescale 后 (m_e^{(1)} \approx b_e)。

2. 当 (d) 变大时，高 (b_e) 的 edge 会被更强地放大，低 (b_e) 的 edge 会被压缩；这会改变 KEP solution 的相对权重结构。

3. 每个 graph 都 rescale 到均值 (\mu_G)，所以 raw decision gap 不会因为 degree 变大而完全由 scale 主导。

这个 Step2b 的科学定位是：

```text
Step2b = noiseless polynomial misspecification
```

它专门回答：

> 如果真实 label 是 deterministic nonlinear function，但模型仍然只能学 (theta_1 u + theta_2 c)，FY 和 SPO+ 谁更会学习“对 KEP 决策更有用”的线性 proxy？

我预期：2stage 在 degree 高时会变弱；SPO+ 可能开始比 MSE 更有优势；FY 是否跟上，要看 perturbation smoothing 是否选到了更 stable 的 decision boundary。

## Step2c：SPO-style degree + multiplicative noise

SPO shortest-path experiment 不是只有 degree，还有 multiplicative noise：

[
\epsilon_i^j \sim Uniform[1-\bar\epsilon, 1+\bar\epsilon]
]

论文第 28–29 页把 noise half-width (\bar\epsilon\in{0,0.5}) 和 degree 一起 sweep，最后用 normalized SPO loss 比较。

所以 Step2c 可以在 Step2b 的 (m_e^{(d)}) 基础上加 multiplicative noise：

[
\eta_e \sim Uniform[1-\bar\epsilon, 1+\bar\epsilon]
]

[
w^{syn}_e = \max(0, m^{(d)}_e \cdot \eta_e)
]

建议：

```text
degree d ∈ {1, 2, 4, 8}
epsilon_bar ∈ {0.25, 0.5}
```

如果 (\bar\epsilon \le 1)，且 (m_e^{(d)}\ge 0)，其实 `max(0, ·)` 不太会生效，但保留也没问题。

这个 Step2c 的科学定位是：

```text
Step2c = nonlinear misspecification + symmetric multiplicative noise
```

它比 Step2b 更接近 SPO paper 的 shortest-path synthetic setup。它也更适合比较 FY 和 SPO+，因为：

* SPO+ 是 structured hinge / upper-bound 风格；
* FY 是 perturbation-smoothed optimizer 风格；
* multiplicative noise 会影响 edge relative weight，可能让 checkpoint selection 更难；
* degree 控制 deterministic misspecification，noise 控制 stochastic uncertainty。

我建议 Step2c 是你最重要的 Step2 实验之一。

## Step2d：从 SPO portfolio experiment 借鉴 correlated factor noise

第 30 页 portfolio experiment 的结果和 shortest-path 类似：degree 越大、misspecification 越强，SPO+ 越可能有优势。Appendix D 里更具体：他们先生成 conditional mean return，然后加入 factor noise：

[
\tilde r_i = \bar r_i + Lf + 0.01\tau\epsilon
]

其中 (f\sim N(0,I_4))，(\epsilon\sim N(0,I_{50}))，(\tau) 控制噪声强度，最后 cost 是 negative return。

这个想法非常值得转到 KEP，因为 KEP edge reward 的误差很可能不是 iid edge noise，而是 graph-level / recipient-level / donor-level latent factor 造成的 correlated noise。

可以这样设计 Step2d：

先用 Step2b 的 nonlinear mean：

[
m^{(d)}_e
]

然后对每个 graph 抽 latent factors：

[
f_G \sim N(0,I_K)
]

为每条 edge 定义 loading：

[
\ell_e =
[
u_e,\ c_e,\ u_e c_e
]
]

标准化后加入 correlated noise：

[
\nu_e
=====

\tau \mu_G
\left(
\ell_e^\top f_G
+
\sigma \xi_e
\right)
]

[
\xi_e\sim N(0,1)
]

最后：

[
w^{syn}_e = \max(0, m^{(d)}_e + \nu_e)
]

建议第一版：

```text
K = 3
degree d ∈ {1, 4}
tau ∈ {0.25, 0.5}
sigma = 0.1
```

这个 Step2d 的科学定位是：

```text
Step2d = nonlinear mean + graph-correlated factor noise
```

它和 Step2c 的区别是：

```text
Step2c: edge-wise independent multiplicative noise
Step2d: graph-level correlated additive/factor noise
```

这个 regime 可能会让 FY 和 SPO+ 出现更明显差异。SPO+ 的 adversarial shifted objective 可能对 margin 更敏感；FY 的 perturbation averaging 可能更像 robustness smoothing。谁更好，不好预判，但这个实验会很有信息量。

## 我建议的 Step2 命名和顺序

你可以这样组织：

```text
Step2a_additive_linear_gaussian
Step2b_polynomial_degree_noiseless
Step2c_polynomial_degree_multiplicative_noise
Step2d_polynomial_degree_factor_noise
```

代码/实验目录按 regime 组织，正式 processed dataset 统一放在 `dataset/processed/`。Dataset name 使用短命名：

```text
dataset/processed/step2<letter>_<short_label_mode>_<params>_<split>_seed<seed>
```

其中：

```text
split in {main2000, val2000, unseen10000}
seed = synthetic label seed, not necessarily graph-generation seed
```

This keeps `surrogate_experiment_results/Step2/...` as the code/protocol area, while large reusable datasets stay under `dataset/processed/`.

推荐先跑：

```text
Step2a: rho = 0.5
Step2b: degree = {1, 2, 4, 8}
Step2c: degree = {1, 2, 4, 8}, epsilon_bar = 0.5
```

等结果稳定后再跑 Step2d。不要一开始就把多个 noise、多个 seed 全部铺开，否则计算量和解释复杂度都会很高。

## 关键实验 protocol

每个 label regime 都应该生成同样结构的 processed dataset。The current naming convention is:

```text
Step2a:
dataset/processed/step2a_additive_rho050_main2000_seed20260523
dataset/processed/step2a_additive_rho050_val2000_seed20260523
dataset/processed/step2a_additive_rho050_unseen10000_seed20260523

Step2b:
dataset/processed/step2b_poly_d1_main2000_seed20260523
dataset/processed/step2b_poly_d1_val2000_seed20260523
dataset/processed/step2b_poly_d1_unseen10000_seed20260523
dataset/processed/step2b_poly_d2_main2000_seed20260523
dataset/processed/step2b_poly_d2_val2000_seed20260523
dataset/processed/step2b_poly_d2_unseen10000_seed20260523
dataset/processed/step2b_poly_d4_main2000_seed20260523
dataset/processed/step2b_poly_d4_val2000_seed20260523
dataset/processed/step2b_poly_d4_unseen10000_seed20260523
dataset/processed/step2b_poly_d8_main2000_seed20260523
dataset/processed/step2b_poly_d8_val2000_seed20260523
dataset/processed/step2b_poly_d8_unseen10000_seed20260523

Step2c:
dataset/processed/step2c_poly_d1_mult_eps050_main2000_seed20260523
dataset/processed/step2c_poly_d1_mult_eps050_val2000_seed20260523
dataset/processed/step2c_poly_d1_mult_eps050_unseen10000_seed20260523
dataset/processed/step2c_poly_d2_mult_eps050_main2000_seed20260523
dataset/processed/step2c_poly_d2_mult_eps050_val2000_seed20260523
dataset/processed/step2c_poly_d2_mult_eps050_unseen10000_seed20260523
dataset/processed/step2c_poly_d4_mult_eps050_main2000_seed20260523
dataset/processed/step2c_poly_d4_mult_eps050_val2000_seed20260523
dataset/processed/step2c_poly_d4_mult_eps050_unseen10000_seed20260523
dataset/processed/step2c_poly_d8_mult_eps050_main2000_seed20260523
dataset/processed/step2c_poly_d8_mult_eps050_val2000_seed20260523
dataset/processed/step2c_poly_d8_mult_eps050_unseen10000_seed20260523

Step2d:
dataset/processed/step2d_poly_d1_factor_tau025_main2000_seed20260523
dataset/processed/step2d_poly_d1_factor_tau025_val2000_seed20260523
dataset/processed/step2d_poly_d1_factor_tau025_unseen10000_seed20260523
dataset/processed/step2d_poly_d4_factor_tau025_main2000_seed20260523
dataset/processed/step2d_poly_d4_factor_tau025_val2000_seed20260523
dataset/processed/step2d_poly_d4_factor_tau025_unseen10000_seed20260523
```

## Current Processing-Script Status

The first three Step2 label-regime processors now exist as Step2-local scripts:

```text
Step2a_additive_linear_gaussian/data-processing.py
  label mode: step2a_additive_linear_gaussian
  status: implemented and smoke-tested

Step2b_polynomial_degree_noiseless/data-processing.py
  label mode: step2b_polynomial_degree_noiseless
  status: implemented and smoke-tested

Step2c_polynomial_degree_multiplicative_noise/data-processing.py
  label mode: step2c_polynomial_degree_multiplicative_noise
  status: implemented and smoke-tested
```

All three are intended to reuse the same raw graph structures and write formal reusable datasets under `dataset/processed/`. Step2c is Step2b plus deterministic uniform multiplicative noise after graph-level polynomial rescaling.

The current production helpers are:

```text
validate_step2_processed_dataset.py
  Reads one processed dataset directory and writes:
    label_diagnostics.json
    label_graph_diagnostics.csv
  The summary includes graph/edge counts, label mean/std/min/max, zero-label fraction,
  clean-linear-vs-label correlation, Step2b polynomial-label correlation, and Step2c multiplier statistics.
  Use --strict with --expected_graph_count and --expected_label_mode for production generation.

run_generate_step2abc_datasets.sh
  Generates the first Step2 ABC dataset grid:
    Step2a rho=0.5
    Step2b degree in {1,2,4,8}
    Step2c degree in {1,2,4,8}, epsilon_bar=0.5
    splits in {main2000,val2000,unseen10000}
  Each processed dataset is immediately passed through the strict validator.
```

Use dry-run mode before generating anything:

```bash
DRY_RUN=1 bash surrogate_experiment_results/Step2/run_generate_step2abc_datasets.sh
```

Run actual generation only after the dry-run paths look correct:

```bash
bash surrogate_experiment_results/Step2/run_generate_step2abc_datasets.sh
```

Set `FORCE=1` only when intentionally rebuilding existing processed datasets:

```bash
FORCE=1 bash surrogate_experiment_results/Step2/run_generate_step2abc_datasets.sh
```

The generation script supports parameterized naming. If `STEP2_RHO=0.25`, Step2a outputs use `rho025`; if `STEP2_EPSILON_BAR=0.25`, Step2c outputs use `eps025`. This prevents parameter/name mismatch:

```bash
DRY_RUN=1 STEP2_RHO=0.25 STEP2_EPSILON_BAR=0.25 \
  bash surrogate_experiment_results/Step2/run_generate_step2abc_datasets.sh
```

For production generation, the validator is called with:

```text
--strict
--expected_graph_count 2000 or 10000 depending on split
--expected_label_mode <label mode>
```

For stochastic Step2 regimes, the deterministic label-noise key includes the raw batch directory name:

```text
raw_batch_name | genjson file name | source node id | target vertex id | utility | label_seed namespace
```

This prevents main/validation/unseen batches from accidentally reusing the same per-edge label noise solely because they contain a same-named `genjson-*.json` file with matching local ids.

## Latest Step2 ABC Dataset Production

Local production run completed on 2026-05-23 with:

```bash
bash surrogate_experiment_results/Step2/run_generate_step2abc_datasets.sh
```

Full log:

```text
logs/step2abc_generation.log
```

The 2026-05-23 run generated and strictly validated the original 21 Step2 ABC processed datasets under the earlier `{1,2,4}` degree grid. On 2026-05-24, the paper-aligned degree `8` extension was generated locally, strictly validated, and synced to garnet. The active Step2 ABC processed dataset grid is now:

```text
Step2a:
  rho=0.5 x {main2000, val2000, unseen10000}

Step2b:
  completed: degree in {1,2,4,8} x {main2000, val2000, unseen10000}

Step2c:
  completed: degree in {1,2,4,8}, epsilon_bar=0.5 x {main2000, val2000, unseen10000}
```

Every dataset has:

```text
G-*.json
run_info.json
batch_summary.json
batch_report.md
label_diagnostics.json
label_graph_diagnostics.csv
```

Strict validation passed with the expected graph counts and label modes:

```text
main2000:      2000 graphs
val2000:       2000 graphs
unseen10000:  10000 graphs
```

QA snapshot for the main2000 training pools:

| dataset | graphs | edges | label mean | label std | zero fraction | corr(clean,label) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| step2a_additive_rho050_main2000_seed20260523 | 2000 | 492684 | 6.1069 | 4.0595 | 0.0839 | 0.7290 |
| step2b_poly_d1_main2000_seed20260523 | 2000 | 492684 | 5.9516 | 3.1484 | 0.0000 | 1.0000 |
| step2b_poly_d2_main2000_seed20260523 | 2000 | 492684 | 5.9549 | 3.5020 | 0.0000 | 0.9971 |
| step2b_poly_d4_main2000_seed20260523 | 2000 | 492684 | 5.9621 | 4.3709 | 0.0000 | 0.9747 |
| step2b_poly_d8_main2000_seed20260523 | 2000 | 492684 | 5.9769 | 6.8146 | 0.0000 | 0.8817 |
| step2c_poly_d1_mult_eps050_main2000_seed20260523 | 2000 | 492684 | 5.9518 | 3.6971 | 0.0000 | 0.8510 |
| step2c_poly_d2_mult_eps050_main2000_seed20260523 | 2000 | 492684 | 5.9550 | 4.0262 | 0.0000 | 0.8667 |
| step2c_poly_d4_mult_eps050_main2000_seed20260523 | 2000 | 492684 | 5.9618 | 4.8580 | 0.0000 | 0.8763 |
| step2c_poly_d8_mult_eps050_main2000_seed20260523 | 2000 | 492684 | 5.9751 | 7.2867 | 0.0000 | 0.8237 |

QA snapshot for the unseen10000 evaluation pools:

| dataset | graphs | edges | label mean | label std | zero fraction | corr(clean,label) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| step2a_additive_rho050_unseen10000_seed20260523 | 10000 | 2338230 | 6.1601 | 4.0871 | 0.0836 | 0.7266 |
| step2b_poly_d1_unseen10000_seed20260523 | 10000 | 2338230 | 6.0137 | 3.1625 | 0.0000 | 1.0000 |
| step2b_poly_d2_unseen10000_seed20260523 | 10000 | 2338230 | 6.0171 | 3.5175 | 0.0000 | 0.9971 |
| step2b_poly_d4_unseen10000_seed20260523 | 10000 | 2338230 | 6.0243 | 4.3893 | 0.0000 | 0.9749 |
| step2b_poly_d8_unseen10000_seed20260523 | 10000 | 2338230 | 6.0391 | 6.8293 | 0.0000 | 0.8834 |
| step2c_poly_d1_mult_eps050_unseen10000_seed20260523 | 10000 | 2338230 | 6.0141 | 3.7209 | 0.0000 | 0.8499 |
| step2c_poly_d2_mult_eps050_unseen10000_seed20260523 | 10000 | 2338230 | 6.0175 | 4.0518 | 0.0000 | 0.8656 |
| step2c_poly_d4_mult_eps050_unseen10000_seed20260523 | 10000 | 2338230 | 6.0247 | 4.8875 | 0.0000 | 0.8755 |
| step2c_poly_d8_mult_eps050_unseen10000_seed20260523 | 10000 | 2338230 | 6.0393 | 7.3170 | 0.0000 | 0.8244 |

Interpretation of the QA:

```text
Step2a introduces additive Gaussian noise and clipping, so about 8.4% of labels are zero.
Step2b keeps graph-level mean scale stable as degree increases; label variance increases with degree.
Step2c adds multiplicative noise on top of Step2b; label variance increases and clean-label correlation drops, as intended.
```

Additional post-generation quality audit:

```text
Graph structure invariance:
  main2000:     9 datasets, 2000 graphs, 492684 edges, no vertex/edge-count mismatches
  val2000:      9 datasets, 2000 graphs, 464520 edges, no vertex/edge-count mismatches
  unseen10000:  9 datasets, 10000 graphs, 2338230 edges, no vertex/edge-count mismatches

Per-graph label-to-clean mean ratio:
  Step2b d1: exactly 1.0000 on every graph, as expected.
  Step2b d2/d4/d8: mean ratio about 1.0005/1.0017/1.0041, with d8 95th percentile below 1.015.
  Step2c: mean ratio about 1.000-1.004, with d8 95th percentile about 1.050 after multiplicative noise.
  Step2a: mean ratio about 1.024 because Gaussian noise is clipped at zero; this is expected.

Stochastic checks:
  Step2a additive noise has near-zero mean and std matching graph-level sigma.
  Step2c multiplier has mean about 1.000, std about 0.2886, and min/max exactly 0.5/1.5.
```

最重要的是：**尽量固定 graph structures，只换 labels。**

也就是说，如果可能，继续用同一批 raw graph / compatibility graph，然后根据不同 label mode 重新写 `w_true`。这样 Step2 比较的是 label generation effect，而不是 graph distribution effect。

每个 regime 下，跑：

```text
train_size ∈ {50, 200, 600, 1200}
methods:
  2stage
  FY
  SPO+
validation:
  validation2000
test:
  heldout400
  unseen10000 或 Step2-specific large unseen
```

不过注意：`realistic2000` 是旧 label regime stress test。如果 Step2 的目标是研究新的 synthetic labels，那更干净的做法是为每个 Step2 label mode 也生成同 regime 的 validation 和 unseen datasets，例如：

```text
step2a_additive_rho050_val2000_seed20260523
step2a_additive_rho050_unseen10000_seed20260523
step2b_poly_d4_val2000_seed20260523
step2b_poly_d4_unseen10000_seed20260523
step2c_poly_d4_mult_eps050_val2000_seed20260523
step2c_poly_d4_mult_eps050_unseen10000_seed20260523
```

这样 unseen test 和 training label regime 一致。

## 对 Step2a/b/c 的预期结果

我会预期：

### Step2a

```text
conditional mean 近似线性
2stage MSE 很强
FY/SPO+ 未必显著超过 2stage
```

如果 SPO+ 或 FY 在 Step2a 大幅赢，要仔细看是不是 validation selection 或 label clipping 造成了非线性。

### Step2b

```text
degree=1: 2stage ≈ FY/SPO+
degree=2: decision-aware methods 可能开始有优势
degree=4+: SPO+ 可能更明显优于 MSE
```

这和 SPO paper 第 29 页 Figure 4 的结论一致：degree 小时方法差异不大，degree 大时 SPO+ 通常更有优势。

### Step2c

```text
degree 高 + noise 高时，checkpoint selection 会更难
validation decision gap 可能更 noisy
validation SPO+ loss / validation FY loss 谁更稳，是重点
```

这里不要只看 final selected checkpoint，还要画 trajectory diagnostics：

```text
epoch vs validation surrogate loss
epoch vs validation decision gap
epoch vs heldout decision gap
epoch vs unseen decision gap
```

你之前 Step1b/Step1c 的经验已经说明：validation decision gap 不一定是最稳定的 selection metric。

## 一个很重要的提醒：scale control

Step2b/2c 的 polynomial degree 会极大改变 reward scale。如果不 rescale，raw decision gap 会变得不可比，甚至 solver 会被少数超大 edge reward 主导。

所以我强烈建议所有 nonlinear label 都做 graph-level rescaling：

[
\frac{1}{|E_G|}\sum_e w^{syn}_e
\approx
\frac{1}{|E_G|}\sum_e (10u_e+5c_e)
]

并且报告：

```text
mean label
std label
min/max label
fraction clipped to zero
oracle objective distribution
```

如果某个 label regime 下 40% edge 被 `max(0, ·)` 截成 0，那这个 regime 的解释就变成了 sparse reward / censored reward，而不只是 noise/misspecification。

## 最推荐的第一轮 Step2

我建议第一轮不要太大，先做这个：

```text
Step2a:
  additive Gaussian
  rho = 0.5

Step2b:
  polynomial noiseless
  degree = 1, 2, 4, 8

Step2c:
  polynomial multiplicative noise
  degree = 1, 2, 4, 8
  epsilon_bar = 0.5
```

每个只先跑：

```text
train_size = 50, 200, 600
theta_seed = 42
subset_seed = 42
```

如果趋势清楚，再补：

```text
train_size = 1200
label_seed repeats
Step2d factor noise
```

## 最短结论

是的，SPO 原论文第 27–30 页的 `degree` 设计非常适合作为 Step2b/2c 的灵感。你的 Step2a 可以作为 additive-linear bridge benchmark；Step2b 用 polynomial degree 做 deterministic misspecification；Step2c 在 polynomial degree 上加 SPO-style multiplicative noise；Step2d 可以借鉴 portfolio experiment 的 factor noise，做 graph-correlated label noise。核心是：**degree 不只是一个公式参数，而是控制“真实 label 生成机制”和“线性 probe model”之间 misspecification 的旋钮。**
