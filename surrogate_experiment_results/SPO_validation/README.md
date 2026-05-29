我觉得导师这个质疑是**合理而且应该认真回应的**。不是说你的 KEP 结果一定有问题，而是因为 SPO+ 很容易在 **min/max 符号、shifted objective、梯度方向、solver tie-breaking** 上出错；做一个外部 benchmark 复现，是最干净的 positive control。

## 当前执行状态

当前目录已经按新的验证主线落地：

```text
01_compare_spoplus_formula_toy_shortest_path.py
02_compare_reward_max_sign_conversion.py
05_validate_kep_spoplus_code_path.py
spoplus_shortest_path.py
spoplus_kep_path_validation.py
compare_with_pyepo_spoplus.py
VALIDATION_STATUS.md
```

其中 `spoplus_shortest_path.py` 是 Gurobi-free 的小型 deterministic shortest-path oracle 和 SPO+ 公式实现；`01_...` 和 `02_...` 是 Level 1 的命令行入口；`05_validate_kep_spoplus_code_path.py` 是 Level 1.5，直接验证当前 Step1c KEP reward-max SPO+ 代码路径；`compare_with_pyepo_spoplus.py` 会在 `KEPs` 环境中用 PyEPO/Gurobi 做 PyEPO 对照。这个 PyEPO 对照需要 Gurobi WLS license 网络访问；如果缺 PyEPO/Gurobi、license 不可用、网络不可用，或者 loss/gradient 不一致，脚本会直接报错并非零退出。

对应测试在：

```text
tests/test_spoplus_shortest_path_validation.py
tests/test_spoplus_kep_path_validation.py
```

当前核心验证主线是：

```text
Level 1: toy shortest-path formula validation
Level 1.5: KEP Step1c code-path validation
Level 2: PyEPO/Warcraft external positive control
Level 3: later synthetic shortest-path degree sweep
```

其中 Level 1.5 是证明 Step2 所依赖的 KEP SPO+ 实现没有明显公式/梯度 bug 的最直接证据。Level 2 可保留为外部 benchmark，但它不是 KEP Step2 代码路径的一比一复刻。

Level 2 的命名一致性文件也已经准备好，但默认运行是 notebook-style full reproduction：

```text
03_run_warcraft_pyepo_reference.py
04_run_warcraft_our_spoplus.py
warcraft_level2_common.py
```

这两个入口默认使用 `k=12`, `batch_size=70`, `epochs=50`, `lr=5e-4`, `seed=135`，数据路径为：

```text
surrogate_experiment_results/SPO_validation/warcraft_shortest_path_oneskin/12x12/
```

当前 `warcraft_maps.tar.gz` 已下载到 `SPO_validation/`，并已解压 12x12 子集。运行 Level 2 时会从上面的本地目录读取 `.npy` 文件；不会自动下载数据，也不会静默跳过。调试时可以显式传 `--train-limit` / `--test-limit` / `--epochs`，但不传这些参数时就是 full reproduction defaults。

数据来源是 PyEPO notebook 中引用的 Warcraft terrains shortest paths dataset：

```text
https://edmond.mpdl.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.YJCQ5S
warcraft_maps.tar.gz
MD5: acea5ea60a47664ff189923a84814e96
```

可运行命令：

```bash
conda run -n KEPs python -m unittest tests.test_spoplus_shortest_path_validation -v
conda run -n KEPs python -m unittest tests.test_spoplus_kep_path_validation -v
conda run -n KEPs python -m unittest tests.test_step1c_spoplus -v
conda run -n KEPs python surrogate_experiment_results/SPO_validation/01_compare_spoplus_formula_toy_shortest_path.py
conda run -n KEPs python surrogate_experiment_results/SPO_validation/02_compare_reward_max_sign_conversion.py
conda run -n KEPs python surrogate_experiment_results/SPO_validation/05_validate_kep_spoplus_code_path.py
conda run -n KEPs python surrogate_experiment_results/SPO_validation/compare_with_pyepo_spoplus.py
```

Level 2 文件检查：

```bash
conda run -n KEPs python -m unittest tests.test_spoplus_warcraft_level2 -v
```

Level 2 full reproduction 入口：

```bash
conda run -n KEPs python surrogate_experiment_results/SPO_validation/03_run_warcraft_pyepo_reference.py
conda run -n KEPs python surrogate_experiment_results/SPO_validation/04_run_warcraft_our_spoplus.py
```

但我会稍微修正一下实验目标：

> 不要把 PyEPO 的 Warcraft notebook 当成“原 SPO+ 论文完全复现”。它更像是 PyEPO 官方风格的 SPO+/shortest-path reference implementation。真正原 SPO+ 论文的 shortest-path 实验是 synthetic/contextual shortest path，用 polynomial degree 控制 misspecification；Warcraft notebook 是图像 terrain maps + ResNet + shortest-path oracle 的现代 demo。

所以现在的优先级是：先做公式/梯度级别的对照，再直接验证 Step1c KEP code path，然后把 Warcraft/PyEPO 当作外部 positive control，最后再补一个更接近原 SPO paper 的 synthetic misspecification sweep。

## 为什么值得做

SPO+ 原论文的核心主张是：SPO loss 直接衡量 predicted costs induced decision 的 error；SPO+ 是一个 convex surrogate，可以用 linear objective optimization oracle 来训练，并且在 shortest path / portfolio 等实验中，尤其在 model misspecification 下，相比普通 prediction loss 更有价值。([arXiv][1])

你的 Step2 其实正是在做类似的问题：degree/noise 增大以后，two-feature linear probe 越来越 misspecified，FY/SPO+ 是否能恢复 decision quality。导师现在问“你的 SPO+ 代码是不是对的”，本质上是在问：

> 你现在看到的 SPO+ 行为，究竟是方法本身在 KEP 上的表现，还是你自己实现的 SPO+ 有符号/梯度 bug？

这个问题必须用 external reference 去消掉。

## 你找到的 PyEPO notebook 应该怎么用

你上传的 notebook 是一个很好的 reference benchmark。它里面做的是 Warcraft shortest path：

* 使用 Warcraft terrain map dataset；
* 输入是 terrain image，label 是 cost / shortest path；
* optimization oracle 是 2D grid shortest path；
* neural network 是 truncated ResNet18；
* SPO+ 部分直接调用：

```python
spoploss = pyepo.func.SPOPlus(optmodel, processes=1)
```

* 然后比较 2stage、SPO+、DBB、DPO、PFYL 等方法。

这对你有两个用途。

第一，它可以作为 **reference implementation**。你可以把 PyEPO 的 `SPOPlus` 当作可信 baseline，检查你的 SPO+ loss 和 gradient 是否在同一个 shortest-path oracle 上完全一致。

第二，它可以作为 **empirical sanity benchmark**。如果 PyEPO SPO+ 在 Warcraft shortest path 上能得到合理的 regret / relative regret / optimality trend，而你的实现接进去以后趋势类似，导师就很难再说“你的 SPO+ 实现明显不对”。

但注意：**不要直接说“我复现了 SPO 原论文”**。更严谨的说法是：

> We validate our SPO+ implementation against a standard PyEPO shortest-path benchmark and, separately, reproduce the qualitative SPO+ behavior under controlled shortest-path misspecification.

## 最关键：先做 loss/gradient 级别对照

经验曲线会受 seed、GPU、CNN、dataset、optimizer、batch order 影响。导师如果很严格，单纯说“曲线差不多”不够。你应该先做一个更硬的 test：

### Cost-minimization SPO+ 公式

PyEPO / shortest path 通常是 **cost minimization**：

[
z^\star(c) \in \arg\min_{z\in Z} c^\top z.
]

SPO+ loss 是：

[
L_{\mathrm{SPO+}}^{\min}(\hat c,c)
==================================

\max_{z\in Z}
(c-2\hat c)^\top z
+
2\hat c^\top z^\star(c)
-----------------------

c^\top z^\star(c).
]

因为

[
\arg\max_{z\in Z}(c-2\hat c)^\top z
===================================

\arg\min_{z\in Z}(2\hat c-c)^\top z,
]

所以 subgradient w.r.t. (\hat c) 是：

[
\nabla_{\hat c} L_{\mathrm{SPO+}}^{\min}
========================================

2\left(z^\star(c)-z^\star(2\hat c-c)\right).
]

### 你的 KEP reward-max SPO+ 公式

你的 KEP 是 **reward maximization**：

[
y^\star(w) \in \arg\max_{y\in Y} w^\top y.
]

令 (c=-w), (\hat c=-\hat w)，可以推出 reward-max 版本：

[
L_{\mathrm{SPO+}}^{\max}(\hat w,w)
==================================

\max_{y\in Y}
(2\hat w-w)^\top y
------------------

2\hat w^\top y^\star(w)
+
w^\top y^\star(w).
]

其梯度是：

[
\nabla_{\hat w}L_{\mathrm{SPO+}}^{\max}
=======================================

2\left(
y^\star(2\hat w-w)-y^\star(w)
\right).
]

如果 (\hat w=X\theta)，则：

[
\nabla_\theta L
===============

2X^\top
\left(
y^\star(2\hat w-w)-y^\star(w)
\right).
]

你当前 KEP 代码如果就是这个方向，那符号上是对的。最容易错的是把 reward-max 当成 cost-min，或者把 (2\hat w-w) 写成 (w-2\hat w)。

## 我建议的验证计划

### Step A：toy shortest-path unit test

先不要跑 ResNet，不要跑 Warcraft。做一个 5×5 或 12×12 grid shortest path toy instance。

对同一批 random true costs (c) 和 predicted costs (\hat c)，比较三件事：

1. PyEPO `SPOPlus` 的 forward loss；
2. 你手写的 minimization-version SPO+ loss；
3. PyTorch autograd 得到的 gradient 与你手写 subgradient：

[
2(z^\star(c)-z^\star(2\hat c-c)).
]

验收标准：

```text
loss difference < 1e-6 或 solver tolerance 级别
gradient cosine similarity ≈ 1
gradient max absolute difference 很小
```

这个 test 是最强的。它能直接回答：

> 我的 SPO+ 数学公式和 PyEPO reference 是否一致？

### Step B：reward-max sign conversion test

然后把同一个 shortest-path toy problem 改成 reward maximization，例如 (w=-c)。

检查：

```text
SPO+_max(w_hat, w)
==
SPO+_min(c_hat=-w_hat, c=-w)
```

梯度也应该满足 chain rule：

```text
grad_w_hat_max == - grad_c_hat_min
```

这个 test 直接验证你 KEP 中最危险的地方：**min-cost SPO+ 到 max-reward SPO+ 的符号转换**。

### Step C：Warcraft notebook reference run

接着再跑你上传的 PyEPO notebook。

先原样跑 PyEPO 的：

```python
spoploss = pyepo.func.SPOPlus(optmodel, processes=1)
```

记录：

```text
2stage regret / relative regret
SPO+ regret / relative regret
path accuracy
optimality ratio
learning curve
```

然后把 PyEPO SPO+ loss 替换成你自己的 shortest-path SPO+ wrapper，保持：

```text
same network
same optimizer
same seeds
same dataloader
same optmodel
same epochs
```

比较两组：

```text
PyEPO SPO+ vs your SPO+
```

不要追求每个 epoch 完全相同，因为并行、Gurobi tie-breaking、GPU nondeterminism 都可能导致细微差异；但趋势和最终 metrics 应该接近。

### Step D：原 SPO paper 风格 synthetic shortest path

如果导师强调“原论文”，那 Warcraft notebook 还不够。你应该再做一个更接近 SPO paper 的 synthetic shortest-path degree experiment：

```text
degree ∈ {1,2,4,8}
models: least squares / two-stage vs SPO+
metric: normalized SPO loss or normalized decision gap
```

目标不是一模一样复刻每个数字，而是复现 qualitative conclusion：

```text
degree=1: two-stage 和 SPO+ 差距小
degree 高: SPO+ 相对 prediction-loss baseline 更有优势
```

这和你 Step2 的思想完全一致，也能形成很强的外部 validation。

## 我建议你怎么组织这个新实验目录

可以新建：

```text
surrogate_experiment_results/SPO_validation/
```

里面放：

```text
01_compare_spoplus_formula_toy_shortest_path.py      # Level 1 / Step A, 已完成
02_compare_reward_max_sign_conversion.py             # Level 1 / Step B, 已完成
spoplus_shortest_path.py                              # toy oracle + explicit SPO+ formulas
compare_with_pyepo_spoplus.py                         # required PyEPO comparison; failures are hard errors
03_run_warcraft_pyepo_reference.py                    # Level 2 / Step C reference entrypoint
04_run_warcraft_our_spoplus.py                        # Level 2 / Step C local SPO+ entrypoint
warcraft_level2_common.py                             # shared Warcraft loader/model/training code
VALIDATION_STATUS.md                                  # current validation status and audit note
README.md
plot_results/
```

后续如果继续做 Level 3，再补：

```text
05_synthetic_shortest_path_degree_reproduction.py
```

README 里明确分三层：

```text
Level 1: formula/gradient agreement with PyEPO
Level 2: empirical agreement on Warcraft shortest path
Level 3: qualitative reproduction of SPO paper degree-misspecification conclusion
```

这样导师看起来会非常清楚。

## 这件事和你当前 KEP SPO+ 的关系

如果验证结果是：

```text
toy loss/gradient 与 PyEPO 一致
reward-max sign conversion 一致
Warcraft trend 与 PyEPO 一致
synthetic shortest-path degree trend 与 SPO paper 一致
```

那么你就可以非常有力地说：

> The SPO+ implementation is validated on canonical shortest-path benchmarks. Therefore, the Step2 KEP results should be interpreted as behavior of SPO+ under our KEP synthetic-label regimes, not as an implementation artifact.

如果验证失败，那也很好，因为你能定位 bug。最常见的 bug 会是：

1. reward-max / cost-min 符号反了；
2. shifted objective 写成 (w-2\hat w) 而不是 (2\hat w-w)；
3. gradient 差了一个负号；
4. 差了 factor 2；
5. solver 不允许 shifted objective 里出现 negative weights；
6. tie-breaking 不稳定；
7. loss normalization 改变了 gradient scale，但学习率没有调。

## 我对导师建议的判断

我支持导师的建议，但我会把它说得更精确：

> 应该复现 SPO+ shortest-path benchmark，但不要只跑 Warcraft notebook；要先做 PyEPO-vs-ours 的 formula/gradient equivalence test，再做 empirical benchmark。

Warcraft notebook 是非常好的起点，因为它已经包含 PyEPO 的 `SPOPlus`、shortest-path optimization model、regret evaluation、learning curve 和 method comparison。你应该用它，但要把它改造成 **对照实验**：

```text
PyEPO SPO+ reference
vs
your SPO+ implementation on the same shortest-path problem
```

最后再补一个 paper-style synthetic degree sweep，才能真正回应“原论文结论是否能复现”。

一句话：**要做，而且这会显著增强你 Step2 结果的可信度；但最重要的不是让 Warcraft 数字完全一模一样，而是证明你的 SPO+ loss/gradient 与 PyEPO/reference 实现数学一致，并且在 shortest-path misspecification benchmark 上复现 SPO+ 的 qualitative advantage。**

[1]: https://arxiv.org/abs/1710.08005?utm_source=chatgpt.com "Smart \"Predict, then Optimize\""
