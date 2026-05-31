## 推荐验证方案

我建议在：

```text
surrogate_experiment_results/SPO_validation/
```

下面新增一个独立验证目录，例如：

```text
surrogate_experiment_results/SPO_validation/step1c_vs_pyepo/
```

不要混进 KEP Step1c 主实验。这个目录只做一件事：**用 PyEPO shortest-path benchmark 验证 Step1c 的 2stage / SPO+ algebra 是否和 PyEPO 一致。**

当前 PyEPO submodule 已经在 `SPO_validation/PyEPO`，README 里也规定先生成 fixed synthetic shortest-path datasets，再跑 PyEPO experiments；这些 fixed datasets 保存 `x_train`, `c_train`, `x_test`, `c_test`，目的是让所有方法比较同一批 split。 这个正好可以作为 benchmark 数据源。

## 验证层级

### Level 0：oracle 对齐

先只验证 shortest-path oracle。

同一个 cost vector `c`，PyEPO oracle 和 Step1c adapter oracle 必须给出相同：

```text
optimal solution y*
optimal objective z*
```

如果有多条最短路径，solution 可能不同，但 objective 相同。这会影响 SPO+ gradient，因为 gradient 依赖 solution vector。因此为了做严格 equality test，要么：

```text
直接调用 PyEPO 的同一个 optmodel.solve()
```

要么：

```text
给 cost 加极小 deterministic tie-breaker，避免多最优解
```

否则“objective 一致但 path 不一致”会导致 gradient 不一致，这不是 Step1c 错，而是 oracle tie-breaking 不同。

### Level 1：2stage LR 对齐

这里不要用当前 Step1c 的 KEP `train_2stage.py`。当前它是 2 维 Adam trajectory，不是 PyEPO 的 `2-stage LR`。

你需要写一个 PyEPO-compatible Step1c LR：

```text
input:
  X_train: shape (n, 5)
  C_train: shape (n, 40)

model:
  C_hat = X @ W + b   # 或者严格按 PyEPO 的 linear model 设定
```

然后用和 PyEPO 完全一样的 regression routine。最稳是直接调用同一个 `sklearn.linear_model.LinearRegression`，这样比较：

```text
W / b
C_hat_test
test MSE
test normalized regret
```

如果这些不一致，才说明 Step1c adapter 的数据读取、model convention、cost direction 或 evaluation metric 有问题。

### Level 2：SPO+ forward loss 对齐

先不要训练，只固定一个参数 `W0`，比较 forward loss。

PyEPO shortest path 是 **cost minimization**。论文里 SPO+ minimization form 是：

```text
L_SPO+(c_hat, c)
= - min_y (2 c_hat - c)^T y
  + 2 c_hat^T y*(c)
  - z*(c)
```

对应 subgradient 是：

```text
2 y*(c) - 2 y*(2 c_hat - c)
```

论文第 3.4.1 节给的就是这个 SPO+ loss 和 subgradient。

你当前 Step1c core 里已经有 cost-min adapter：

```python
cost_min_spoplus_loss(...)
cost_min_prediction_gradient(...)
```

它通过 reward-max sign adapter 实现 cost-min SPO+。 这个应该直接拿来和 PyEPO 的 `SPOPlus` 对。

验收标准：

```text
per-instance loss: allclose(atol=1e-7, rtol=1e-7)
batch mean loss:   allclose(atol=1e-7, rtol=1e-7)
```

如果用 float32，放宽到 `1e-6`。

### Level 3：SPO+ gradient 对齐

这一步最重要。比较两种 gradient：

```text
PyEPO autograd 得到的 dL/dW
Step1c 手算得到的 X^T @ grad_pred_cost
```

对于 cost-min shortest path，Step1c 应该用：

```python
grad_pred_cost = 2 * (y_optimal - y_adversarial)
grad_W = X.T @ grad_pred_cost / batch_size
```

注意你当前 Step1c KEP 主代码用的是 reward-max 形式：

```python
shifted_w = 2.0 * w_hat - w_true
loss = reward_max_spoplus_loss(...)
grad = X.T @ reward_max_prediction_gradient(...)
```



这个 reward-max 形式不能直接拿去和 PyEPO shortest path cost-min 比，必须切到 `cost_min_spoplus_loss` / `cost_min_prediction_gradient`。

验收标准：

```text
dL/dC_hat: allclose
dL/dW:     allclose
```

如果 forward loss 一致但 gradient 不一致，通常是这几个问题之一：

```text
1. reward-max / cost-min sign 错了
2. y_adv 用错了，应该解 shifted cost = 2*c_hat - c
3. batch mean / batch sum 缩放不一致
4. PyTorch loss.mean() 和 Step1c 手算 mean 不一致
5. oracle tie-breaking 不一致
```

### Level 4：one-step update 对齐

再固定：

```text
same W0
same optimizer
same lr
same batch
same loss scaling
```

跑一步 optimizer update。

如果用 Adam，PyTorch Adam 和你手写 Adam 要完全一致并不总是保险，因为 epsilon、bias correction、dtype、weight decay 细节都可能不同。最干净的 one-step check 用 SGD：

```text
W1 = W0 - lr * grad
```

通过后再测 Adam。

验收标准：

```text
W1_pyepo == W1_step1c
```

### Level 5：full trajectory 对齐

最后才跑完整训练 trajectory。建议先用：

```text
full-batch
no shuffle
deterministic oracle
SGD
float64 or controlled float32
small setting: n=100, noise=0.0, degree=1, seed=0
```

通过后再逐步恢复 PyEPO 的真实训练配置。

验收标准：

```text
epoch-wise train SPO+ loss 一致
epoch-wise validation regret 一致
final test normalized regret 一致
```

## 你要比较的输出表

最终验证脚本应该输出类似：

```text
2stage LR
---------
coef_max_abs_diff:          0.0 / <1e-8
test_pred_cost_max_diff:    0.0 / <1e-8
test_mse_diff:              0.0 / <1e-8
test_norm_regret_diff:      0.0 / <1e-8

SPO+
----
forward_loss_max_abs_diff:  <1e-7
grad_pred_max_abs_diff:     <1e-7
grad_W_max_abs_diff:        <1e-7
one_step_W_max_abs_diff:    <1e-7
trajectory_max_abs_diff:    <1e-6
final_norm_regret_diff:     <1e-6
```

## 我的建议判据

你的判断可以写成：

```text
如果 Level 0-4 不一致，Step1c 基础实现大概率有问题；
如果 Level 0-4 一致但 full training 不一致，优先检查 optimizer、batch order、dtype、epoch schedule、regularization 和 tie-breaking；
如果 only final boxplot 不一致，不能直接判定 Step1c 有 bug。
```

## 最短可行路径

第一版不要跑完整 Figure 7。先只跑一个 setting：

```text
grid = 5x5
train_size = 100
noise = 0.0
degree = 1
seed = 0
method = LR, SPO+
```

先验证：

```text
2stage LR exact match
SPO+ fixed-theta forward exact match
SPO+ fixed-theta gradient exact match
SPO+ one-step update exact match
```

这四个过了以后，再扩展到全部：

```text
train_size in {100, 1000, 5000}
noise in {0.0, 0.5}
degree in {1, 2, 4, 6}
seed in {0, ..., 9}
```

## 最终结论

你的目标是合理的，但要把它定义成：

> **Step1c 的 PyEPO-compatible shortest-path adapter 必须在 fixed data / fixed oracle / fixed model / fixed optimizer 下，与 PyEPO 的 2stage LR 和 SPO+ 得到完全一致的 loss、gradient、update 和 regret。**

而不是直接拿当前 KEP Step1c 主代码去和 PyEPO shortest path pipeline 比。当前 Step1c 是 2-feature reward-max KEP probe；PyEPO Figure 7 是 5-feature、40-edge、cost-min shortest path benchmark。这中间必须加 adapter。

## 当前实现

本目录现在实现了一个只读 Step1c adapter harness：

```text
validation_core.py
validate_small_setting.py
latest_small_setting.json
```

实现边界：

```text
不修改 surrogate_experiment_results/Step1c/ 下的成熟实验文件。
只动态加载 Step1c/spoplus_core.py 里的 SPO+ 代数函数。
所有 PyEPO-compatible data/oracle/model/metric adapter 都放在本目录。
```

最小验证命令：

```bash
python3 surrogate_experiment_results/SPO_validation/step1c_vs_pyepo/validate_small_setting.py \
  --json surrogate_experiment_results/SPO_validation/step1c_vs_pyepo/latest_small_setting.json
```

如果在 Codex 沙箱里运行，Gurobi WLS token 访问可能需要提权，因为 PyEPO
shortest-path oracle 会访问 `token.gurobi.com`。

当前默认 setting：

```text
grid = 5x5
train_size = 100
test_size = 1000
noise = 0.0
degree = 1
seed = 0
SPO+ batch_size = 8
```

已通过的检查：

```text
Level 0 oracle solution/objective alignment
Level 1 2stage LR weight, bias, prediction, MSE, normalized regret alignment
Level 2 SPO+ forward loss alignment
Level 3 SPO+ dL/dC_hat, dL/dW, dL/db alignment
Level 4 one-step SGD W/b update alignment
```

最近一次本地结果：

```text
level0_solution_max_abs_diff              0.0000e+00
level0_objective_max_abs_diff             0.0000e+00
level1_lr_weight_max_abs_diff             0.0000e+00
level1_lr_bias_max_abs_diff               0.0000e+00
level1_lr_test_pred_cost_max_abs_diff     2.2204e-16
level1_lr_test_mse_diff                   4.9920e-35
level1_lr_test_norm_regret_diff           0.0000e+00
level2_forward_loss_max_abs_diff          5.8115e-07
level3_grad_pred_max_abs_diff             0.0000e+00
level3_grad_weight_max_abs_diff           9.6858e-08
level3_grad_bias_max_abs_diff             0.0000e+00
level4_sgd_weight_update_max_abs_diff     9.3132e-09
level4_sgd_bias_update_max_abs_diff       1.4901e-09
```

`level2_forward_loss_max_abs_diff` 是 float32 PyEPO autograd path 和
NumPy Step1c algebra path 的差异，因此验收阈值使用 `1e-5`。梯度和
one-step update 均在 `1e-6` 内对齐。

## 新增 my methods runner

本目录现在额外提供两个 Step1c-compatible shortest-path runner：

```text
run_my_2stage_lr.py
run_my_spoplus.py
compare_my_vs_pyepo_csv.py
plot_with_my_methods.py
```

它们不修改 PyEPO 的 `experiments.py` / `pipeline.py` / `run/`，也不修改
Step1c 成熟实验文件。结果默认写到独立目录：

```text
surrogate_experiment_results/SPO_validation/res_step1c_vs_pyepo/sp/h5w5/gurobi/
```

CSV schema 和 PyEPO 保持一致：

```text
True SPO, Unamb SPO, MSE, Elapsed, Epochs
```

文件名示例：

```text
n100p5-d1-e0.0_my-2stage-lr.csv
n100p5-d1-e0.0_my-spoplus.csv
```

smoke 命令：

```bash
python3 surrogate_experiment_results/SPO_validation/step1c_vs_pyepo/run_my_2stage_lr.py \
  --smoke --overwrite

python3 surrogate_experiment_results/SPO_validation/step1c_vs_pyepo/run_my_spoplus.py \
  --smoke --overwrite
```

全量命令：

```bash
python3 surrogate_experiment_results/SPO_validation/step1c_vs_pyepo/run_my_2stage_lr.py \
  --expnum 10

python3 surrogate_experiment_results/SPO_validation/step1c_vs_pyepo/run_my_spoplus.py \
  --expnum 10
```

`my_spoplus` 的全量配置对齐 PyEPO SPO+ 的 linear model 设置：

```text
batch = 32
optimizer = adam
lr = 1e-2
epochs: n=100 -> 200, n=1000 -> 20, n=5000 -> 4
```

`--smoke` 只跑 `n=100, degree=1, noise=0.0, seed=0`，并默认只训练
`my_spoplus` 1 个 epoch，用来确认 pipeline 能跑通。正式结果请使用
`--expnum 10`。

独立 overlay plot：

```bash
MPLCONFIGDIR=/tmp/matplotlib-pyepo \
python3 surrogate_experiment_results/SPO_validation/step1c_vs_pyepo/plot_with_my_methods.py \
  --prob sp
```

如果只想检查 smoke 产生的一张图，可以限制 setting：

```bash
MPLCONFIGDIR=/tmp/matplotlib-pyepo \
python3 surrogate_experiment_results/SPO_validation/step1c_vs_pyepo/plot_with_my_methods.py \
  --prob sp --train-sizes 100 --degs 1 --noises 0.0 --allow-missing
```

最近一次本地 smoke 结果：

```text
my_2stage_lr n=100,d=1,e=0.0,seed=0:
  True SPO diff vs PyEPO LR = 0.0
  Unamb SPO diff vs PyEPO LR = 0.0
  MSE diff vs PyEPO LR = 0.0

my_spoplus n=100,d=1,e=0.0,seed=0,epochs=1:
  True SPO  = 0.0740341
  Unamb SPO = 0.0740341
  MSE       = 1.70881
```

## CSV row-by-row comparison

`compare_my_vs_pyepo_csv.py` 是数值验收工具，不画图。它逐 CSV、逐 seed
比较 PyEPO 原结果和 my methods 结果：

```bash
python3 surrogate_experiment_results/SPO_validation/step1c_vs_pyepo/compare_my_vs_pyepo_csv.py \
  --pair lr

python3 surrogate_experiment_results/SPO_validation/step1c_vs_pyepo/compare_my_vs_pyepo_csv.py \
  --pair spo
```

`--pair lr` 默认比较：

```text
PyEPO 2-stage LR: n{n}p5-d{d}-e{e}_2s-lr.csv
my 2stage LR:    n{n}p5-d{d}-e{e}_my-2stage-lr.csv
```

`--pair spo` 默认比较：

```text
PyEPO SPO+: n{n}p5-d{d}-e{e}_spo_lr_adam0.01_bs32_l10.0l20.0_c1.csv
my SPO+:    n{n}p5-d{d}-e{e}_my-spoplus.csv
```

正式比较要求两边 CSV 行数一致。若只检查 smoke 的前 1 行，可以显式使用：

```bash
python3 surrogate_experiment_results/SPO_validation/step1c_vs_pyepo/compare_my_vs_pyepo_csv.py \
  --pair lr --train-sizes 100 --degs 1 --noises 0.0 --limit-rows 1
```

当前 `my_2stage_lr` 全量结果：

```text
result files: 24 / 24
result rows:  240 / 240

PyEPO LR vs my 2stage LR row-by-row comparison:
  True SPO   global max_abs_diff = 3.11892e-19
  Unamb SPO  global max_abs_diff = 2.06866e-19
  MSE        global max_abs_diff = 1.09476e-47
  Epochs     global max_abs_diff = 0
```

这说明在 fixed-data shortest-path protocol 下，my 2stage LR 的数据读取、
模型 convention、metric convention 和 CSV layout 已经与 PyEPO LR 对齐。
