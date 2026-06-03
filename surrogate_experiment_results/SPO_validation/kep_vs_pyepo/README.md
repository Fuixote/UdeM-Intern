可以做，而且**很有价值**；但它不是“比 shortest path 更好的替代 benchmark”，而是 **shortest path 之后的第二层、更贴近 Step1c/KEP 的 integration benchmark**。

我建议你把它定义成：

```text
Shortest path:
  外部标准 benchmark，用来验证 Step1c SPO+ 公式、sign、gradient、metric 是否能复现 PyEPO。

KEP:
  目标应用 benchmark，用来验证同一套 SPO+ algebra 接到真实 Step1c KEP oracle 后，
  是否还能和 PyEPO SPOPlus reference 完全一致。
```

也就是说：**shortest path 证明“公式没错”；KEP 证明“接到你的真实问题上也没错”。**

---

## 先看当前结果已经说明了什么

你现在 `step1c_vs_pyepo` 的 shortest-path 验证已经很强。小 setting 下已经通过：

```text
Level 0 oracle solution/objective alignment
Level 1 2stage LR weight, bias, prediction, MSE, normalized regret alignment
Level 2 SPO+ forward loss alignment
Level 3 SPO+ dL/dC_hat, dL/dW, dL/db alignment
Level 4 one-step SGD W/b update alignment
```



最近一次记录里，SPO+ forward loss 差异是 `5.8115e-07`，gradient 和 one-step update 都在 `1e-6` 内。 这基本排除了 cost-min SPO+ 公式、gradient scaling、batch mean、W/b shape convention 这类核心 bug。

另外，`my_2stage_lr` 的全量 shortest-path 结果已经和 PyEPO LR 几乎完全一致：

```text
result files: 24 / 24
result rows:  240 / 240

True SPO   global max_abs_diff = 3.11892e-19
Unamb SPO  global max_abs_diff = 2.06866e-19
MSE        global max_abs_diff = 1.09476e-47
Epochs     global max_abs_diff = 0
```



所以当前最准确的状态是：

```text
shortest-path LR: benchmark-equivalent，已通过。
shortest-path SPO+ algebra: 已通过。
shortest-path my SPO+ full CSV equivalence: 需要继续跑 --pair spo 全量比较。
```

现在你问能不能把这一套迁移到 KEP：**能，而且非常适合作为下一步。**

---

## 技术可行性：可行，但不能直接照搬 PyEPO shortest-path pipeline

### 1. KEP 和 shortest path 最大差别：每个 instance 的优化模型不同

Shortest path benchmark 里，所有样本都是同一个 `5×5` grid，cost dimension 固定为 40。因此 PyEPO 可以用一个固定 `shortestPathModel`，批量训练。

KEP 不一样。Step1a/Step1c 的 KEP graph 是从 `G-*.json` 读进来的，每个 graph 有自己的 edge set、cycle candidates、NDD/pair node 结构和 MILP。`load_graph` 会读出 `w_true`、`edge_index`、`cycle_candidates`、`num_edges` 等 graph-specific 数据。

`CachedHybridKepModel` 也是每个 graph 单独构建一个 Gurobi model；它缓存 edge lists、cycle candidates、chain keys、pair nodes、NDD nodes，并在 solve 时更新 objective。

所以 KEP 版不能简单写成：

```python
one PyEPO optmodel + one optDataset + one DataLoader
```

更现实的是：

```text
for each KEP graph:
    build / reuse one PyEPO-compatible optModel wrapper
    compute PyEPO SPOPlus reference loss/gradient
    compute Step1c SPO+ loss/gradient
    compare per graph or aggregate gradient
```

这会慢一些，但作为 validation benchmark 完全可行。

---

### 2. KEP 是 reward maximization，不是 shortest path 的 cost minimization

Step1c 的 KEP solver 是 maximize reward。`CachedHybridKepModel` 里 Gurobi objective sense 是 `GRB.MAXIMIZE`。 Step1c 当前的 SPO+ 主实现也用 reward-max 形式：

```python
w_hat = X @ theta
shifted_w = 2.0 * w_hat - w_true
y_adv = solve_once(shifted_w, ...)
loss = reward_max_spoplus_loss(...)
grad = X.T @ reward_max_prediction_gradient(...)
```



而 shortest-path 验证用的是 cost-min adapter。`spoplus_core.py` 已经同时有 reward-max 和 cost-min 两套接口：reward-max loss/gradient 是 `reward_max_spoplus_loss` 和 `reward_max_prediction_gradient`，cost-min 是通过 sign adapter 实现。 

所以 KEP 验证有两种安全写法：

```text
方案 A：用 PyEPO 的 MAXIMIZE modelSense，对 reward-max SPO+ 直接比较。
方案 B：把 KEP reward max 转成 cost min，即 cost = -reward，用 PyEPO cost-min SPO+ 比较。
```

我更建议 **方案 B**，因为你刚刚 shortest-path 已经验证了 cost-min path，复用成本低，而且能避免 PyEPO maximize path 是否完全等价的额外不确定性。具体做法：

```text
true_cost = -w_true
pred_cost = -w_hat
solver for cost c returns argmin c^T y = argmax reward^T y
```

然后 Step1c 侧也用：

```python
cost_min_spoplus_loss(-w_hat, -w_true, y_opt, y_adv)
```

或直接验证它和 reward-max 形式一致。

---

### 3. KEP 有 tie-breaking 风险，必须用同一个 solver wrapper

这点比 shortest path 更重要。`linear_probe_landscape.py` 里已经明确提醒：cached solver 和 original backend 在有 ties 时可能选出不同 optimal edge selection，因此只能比较 objective equality；但 SPO+ gradient 依赖具体 solution vector，所以如果两个 optimal solution 不同，gradient 就可能不同。

因此 KEP 版验证时不要搞两个不同 solver：

```text
不要：
  PyEPO 用一个 KEP solver
  Step1c 用另一个 KEP solver
```

应该：

```text
同一个 CachedHybridKepModel / same Gurobi model
同时服务 PyEPO SPOPlus reference 和 Step1c manual SPO+
```

否则你看到的 gradient diff 可能只是 tie-breaking，不是 SPO+ bug。

---

## 这是不是比 shortest path 更好的 benchmark？

答案分两种。

### 如果目标是“验证 SPO+ 公式是否正确”：shortest path 更好

shortest path 是更好的 **external correctness benchmark**。原因是：

```text
1. 它是 PyEPO 论文 Figure 7 的标准问题；
2. 维度固定，所有 instance 共用一个 optmodel；
3. PyEPO 原生支持；
4. 你已经复现了 Figure 7 的主要趋势；
5. 你已经做出了 LR 全量 row-by-row equivalence。
```

所以 shortest path 很适合作为“我写的 SPO+ 和 PyEPO SPO+ 是否在标准 benchmark 上一致”的证据。

KEP 不是 PyEPO 论文的标准 benchmark，且需要你自己写 PyEPO-compatible KEP wrapper。这个 wrapper 本身也可能引入 bug。因此从“外部 benchmark”角度，KEP 不如 shortest path 干净。

### 如果目标是“验证 Step1c 真实 KEP 实验是否没问题”：KEP 更好

KEP 是更好的 **domain-specific integration benchmark**。因为 Step1c 的真实问题不是 shortest path，而是：

```text
w_hat_e = theta_1 * utility_e + theta_2 * recipient_cPRA_e
```

Step1c README 也强调它要固定 Step1b 的 data、split、model class、initialization、training budget、checkpoint protocol 和 evaluation metrics，只换 surrogate。 现有 Step1c 训练脚本的模型公式也是这个 2-feature KEP probe。

所以 KEP 版 PyEPO-vs-my comparison 能检查 shortest path 检查不到的问题：

```text
1. reward-max sign 是否和 KEP objective 一致；
2. KEP oracle 返回的 edge-selection vector 是否和 SPO+ gradient 约定一致；
3. variable-size graph records 是否被正确处理；
4. Step1c 的 X @ theta、w_true、y_optimal、y_adv 是否对齐；
5. normalized gap / synthetic-label decision gap 是否和现有 Step1c evaluation 一致；
6. Gurobi cached model reuse 是否影响 solution/tie-breaking。
```

所以我的判断是：

```text
shortest path = 必要的标准单元 benchmark；
KEP = 更贴近论文项目目标的集成 benchmark；
二者不是替代关系，而是递进关系。
```

---

## 我建议的 KEP 版实验设计

不要一上来画大图。先完全复刻你在 shortest path 里做的 Level 0–4。

### Phase 0：KEP preflight，只验证迁移边界

Phase 0 已经有最小脚本：

```text
kep_validation_core.py
validate_kep_phase0.py
```

它不是 PyEPO training，也不是 full trajectory。它只检查迁移到真实 KEP
graph 之前最容易出错的边界：

```text
1. 能从 dataset/processed/step1_noisy_linear_sigma010 选择真实 G-*.json；
2. 能通过 Step1c 的 load_graph_records 构建 CachedHybridKepModel；
3. 能在真实 KEP graph 上计算 w_hat、shifted_w、y_optimal、y_adv；
4. reward-max SPO+ 和 cost-min sign adapter 在同一批 KEP solution 上一致；
5. Step1c 的 spo_plus_loss_and_grad 和本地 algebra check 一致。
```

默认 smoke 使用 2 个 graph：

```bash
python3 surrogate_experiment_results/SPO_validation/kep_vs_pyepo/validate_kep_phase0.py
```

本地 WSL 如果 Gurobi token DNS 被 sandbox 限制，会在
`token.gurobi.com` 处失败；这种情况下应在 garnet runtime 下跑，或允许本地
Gurobi token 网络访问。当前本地用 2 个真实 graph 的 Phase 0 preflight 已通过：

```text
phase0_true_objective_max_abs_diff        0
phase0_reward_cost_loss_max_abs_diff      0
phase0_reward_cost_grad_pred_max_abs_diff 0
phase0_reward_cost_grad_theta_max_abs_diff 0
phase0_step1c_loss_max_abs_diff           0
phase0_step1c_grad_theta_max_abs_diff     0
```

### Phase 1：KEP small validation，只做 correctness

Phase 1 已经有最小脚本：

```text
kep_validation_core.py
validate_kep_small_setting.py
```

新增目录可以叫：

```text
surrogate_experiment_results/SPO_validation/kep_vs_pyepo/
```

或者继续放在：

```text
surrogate_experiment_results/SPO_validation/step1c_vs_pyepo/
```

但建议分文件名，例如：

```text
kep_validation_core.py
validate_kep_small_setting.py
compare_kep_my_vs_pyepo_csv.py
plot_kep_with_my_methods.py
```

最小 setting：

```text
dataset: dataset/processed/step1_noisy_linear_sigma010
train graphs: 5 或 10
validation graphs: 20
test graphs: 50 或 heldout400 的前 50
probe: utility + recipient_cPRA
theta_seed: 42
gurobi_seed: 42
```

验证内容：

```text
Level 0:
  same KEP oracle solution/objective alignment

Level 1:
  LR / OLS theta alignment
  prediction w_hat alignment
  decision gap / normalized gap alignment

Level 2:
  PyEPO SPOPlus forward loss vs Step1c reward-max/cost-min SPO+ loss

Level 3:
  PyEPO dL/dw_hat vs Step1c manual dL/dw_hat
  PyEPO dL/dtheta vs Step1c X.T @ dL/dw_hat

Level 4:
  one-step SGD/Adam theta update alignment
```

这里最关键的是 Level 3：Step1c 真实 gradient 是：

```text
grad_SPO+ = 2 X^T (y_adv - y_oracle)     # reward-max convention
```

Step1c README 里也写了这个 scale。

如果 KEP 版 Level 0–4 也过了，你就可以说：

```text
Step1c 的 reward-max SPO+ 在真实 KEP oracle 上，也和 PyEPO SPOPlus reference 对齐。
```

这比 shortest path 更直接地支持 Step1c。

当前本地默认 small setting 已通过：

```bash
python3 surrogate_experiment_results/SPO_validation/kep_vs_pyepo/validate_kep_small_setting.py
```

默认 setting：

```text
dataset: dataset/processed/step1_noisy_linear_sigma010
graphs: G-0.json ... G-4.json
theta_seed: 42
gurobi_seed: 42
sgd_lr: 0.05
```

当前检查结果：

```text
Level 0 oracle solution/objective alignment: PASS
Level 1 OLS theta, prediction, MSE, decision gap alignment: PASS
Level 2 SPO+ forward loss max_abs_diff = 2.9998e-06 <= 1e-5: PASS
Level 3 dL/dw_hat and dL/dtheta max_abs_diff = 0: PASS
Level 4 one-step SGD theta update max_abs_diff = 0: PASS
```

这仍然只是 correctness small validation，不是 Phase 2 full trajectory，也不是
KEP performance plot。

---

### Phase 2：KEP full trajectory equivalence

等 Level 0–4 通过后，再做 full trajectory。

这里不要先画 performance 图，先比较 trajectory：

```text
same train subset
same validation set
same theta_init
same optimizer
same lr
same n_epochs
same metric_stride
same Gurobi seed
```

比较：

```text
epoch
theta_1
theta_2
train_spoplus_loss
validation_spoplus_loss
train_decision_gap
validation_decision_gap
```

Step1c 现在已经会保存 SPO+ loss curve，包含 `theta_1/theta_2`、train/validation SPO+ loss、normalized SPO+ loss、decision gap、y_adv/y_pred equality rates。

所以 KEP PyEPO reference runner 只要输出同样 schema，就可以逐 epoch diff。

当前 Phase 2 runner：

```bash
python3 surrogate_experiment_results/SPO_validation/kep_vs_pyepo/validate_kep_full_trajectory.py
```

默认 setting：

```text
split_path: results/step1b_splits/master_split_seed=42.json
train_size: 5
validation_size: 5
subset_seed: 42
theta_seed: 42
theta_init: [1.6236203565, 3.3521429192]
optimizer: adam
lr: 0.05
n_epochs: 3
metric_stride: 1
gurobi_seed: 42
```

默认输出：

```text
surrogate_experiment_results/SPO_validation/kep_vs_pyepo/latest_full_trajectory.json
surrogate_experiment_results/SPO_validation/kep_vs_pyepo/phase2_results/train_size=5/
  pyepo_spoplus_loss_curve.csv
  step1c_spoplus_loss_curve.csv
  trajectory_diff_summary.csv
  trajectory_diff_summary.json
  trajectory_pyepo_spoplus.npy
  trajectory_step1c_spoplus.npy
  train_subset.json
  validation_set.json
  run_config.json
```

当前本地真实 KEP smoke 已通过：

```text
train graphs:
  G-1027.json, G-520.json, G-1592.json, G-1164.json, G-1862.json
validation graphs:
  G-1357.json, G-258.json, G-1572.json, G-1278.json, G-41.json

phase2_epoch_max_abs_diff                           = 0
phase2_theta_1_max_abs_diff                         = 0
phase2_theta_2_max_abs_diff                         = 0
phase2_theta_norm_max_abs_diff                      = 0
phase2_train_spoplus_loss_max_abs_diff              = 3.2975e-06 <= 1e-5
phase2_validation_spoplus_loss_max_abs_diff         = 2.2324e-06 <= 1e-5
phase2_train_normalized_spoplus_loss_max_abs_diff   = 2.7551e-08 <= 1e-5
phase2_validation_normalized_spoplus_loss_max_abs_diff = 1.5990e-08 <= 1e-5
phase2_train_decision_gap_max_abs_diff              = 0
phase2_validation_decision_gap_max_abs_diff         = 0
phase2_y_adv/y_pred equality rate diffs             = 0
```

这说明在真实 KEP oracle 上，PyEPO SPOPlus reference trajectory 和
Step1c reward-max SPO+ trajectory 在 Phase 2 小型 full-trajectory setting
中已经对齐。这个结果仍然是 trajectory equivalence，不是 Phase 3
performance plot。

---

### Step2b bridge：degree label regime 上的 PyEPO-vs-Step1c 对齐

Step2b 的 `degree` 不是模型特征 expansion，而是真实 synthetic label 的
非线性程度。模型仍然固定为：

```text
w_hat_e = theta_1 * utility_e + theta_2 * recipient_cPRA_e
```

在进入 Phase 3 degree plot 之前，先做 Step2b small bridge：

```bash
python3 surrogate_experiment_results/SPO_validation/kep_vs_pyepo/validate_step2b_degree_trajectory.py
```

默认 setting：

```text
degrees: 1, 2, 4, 8
source_train_size: 50
train_size: 5
validation_size: 5
theta_seed: 42
gurobi_seed: 42
optimizer: adam
lr: 0.1
n_epochs: 3
metric_stride: 1
```

它复用现有 Step2b formal SPO+ run 保存的 split artifact：

```text
surrogate_experiment_results/Step2/Step2b_polynomial_degree_noiseless/remote_results/
  step2b_poly_d{degree}/step1c_spoplus/formal_2stage500_spoplus500_s10/train_size=50/
    train_subset.json
    validation_set.json
    run_config.json
```

默认输出：

```text
surrogate_experiment_results/SPO_validation/kep_vs_pyepo/step2b_bridge_results/
  latest_step2b_bridge.json
  step2b_poly_d1/
  step2b_poly_d2/
  step2b_poly_d4/
  step2b_poly_d8/
```

每个 degree 子目录保存：

```text
pyepo_lr_summary.csv
step1c_lr_summary.csv
lr_diff_summary.csv
lr_diff_summary.json
theta_pyepo_lr.npy
theta_step1c_lr.npy
pyepo_spoplus_loss_curve.csv
step1c_spoplus_loss_curve.csv
trajectory_diff_summary.csv
trajectory_diff_summary.json
trajectory_pyepo_spoplus.npy
trajectory_step1c_spoplus.npy
train_subset.json
validation_set.json
run_config.json
```

LR bridge 使用 Phase 1 的 no-bias edge-level OLS 口径。原因是 KEP
图的 edge count 不是固定维度，不能直接复用 shortest-path 的
`pyepo.twostage.sklearnPred(LinearRegression())` whole-cost-vector interface；
等价 reference 是在所有 train edges 上拟合：

```text
w_true_e ~ theta_1 * utility_e + theta_2 * recipient_cPRA_e
```

LR 通过标准：

```text
theta max_abs_diff <= 1e-12
train/validation prediction max_abs_diff <= 1e-12
train/validation MSE diff <= 1e-12
train/validation decision gap diff <= 1e-9
train/validation normalized gap diff <= 1e-12
```

SPO+ 通过标准沿用 Phase 2：

```text
theta_1/theta_2/theta_norm max_abs_diff <= 1e-6
SPO+ loss max_abs_diff <= 1e-5
decision gap max_abs_diff <= 1e-9
normalized decision gap max_abs_diff <= 1e-12
y_adv/y_pred equality rate max_abs_diff <= 1e-12
```

这个 bridge 的目的不是证明 Step2b performance，而是证明 Step2b
degree label regime 上，PyEPO-style LR reference 和 Step1c LR 对齐，
同时 PyEPO SPOPlus reference 和 Step1c reward-max SPO+ 仍然逐 epoch
对齐。

当前本地 Step2b bridge smoke 已通过：

```text
step2b_poly_d1:
  LR theta/prediction/MSE/gap/normalized-gap diff = 0
  theta diff = 0
  train_spoplus_loss max_abs_diff = 2.9071e-06 <= 1e-5
  validation_spoplus_loss max_abs_diff = 4.9231e-06 <= 1e-5
  train/validation decision gap diff = 0

step2b_poly_d2:
  LR theta/prediction/MSE/gap/normalized-gap diff = 0
  theta diff = 0
  train_spoplus_loss max_abs_diff = 1.6912e-06 <= 1e-5
  validation_spoplus_loss max_abs_diff = 3.1811e-06 <= 1e-5
  train/validation decision gap diff = 0

step2b_poly_d4:
  LR theta/prediction/MSE/gap/normalized-gap diff = 0
  theta diff = 0
  train_spoplus_loss max_abs_diff = 4.0700e-06 <= 1e-5
  validation_spoplus_loss max_abs_diff = 4.3170e-06 <= 1e-5
  train/validation decision gap diff = 0

step2b_poly_d8:
  LR theta/prediction/MSE/gap/normalized-gap diff = 0
  theta diff = 0
  train_spoplus_loss max_abs_diff = 3.7549e-06 <= 1e-5
  validation_spoplus_loss max_abs_diff = 6.9684e-06 <= 1e-5
  train/validation decision gap diff = 0
```

---

### Phase 3：画 Step2b degree overlay plot

到这一步才画图。对于 Step2b，横坐标就是真实 label regime 的
`degree`。当前先做 `train_size=50` 的 formal test-summary smoke plot：

```bash
python3 surrogate_experiment_results/SPO_validation/kep_vs_pyepo/plot_step2b_degree_overlay.py
```

默认 source 是 `formal-test-summary`，读取每个 degree 的现有 formal
Step2b run：

```text
surrogate_experiment_results/Step2/Step2b_polynomial_degree_noiseless/remote_results/
  step2b_poly_d{degree}/step1c_spoplus/formal_2stage500_spoplus500_s10/train_size=50/
    metrics/test_summary.csv
```

默认选择：

```text
LR:      2stage row selected by validation_mse_loss
SPO+:    spoplus row selected by validation_spoplus_loss
y-axis:  test_mean_normalized_gap
```

输出：

```text
surrogate_experiment_results/SPO_validation/kep_vs_pyepo/step2b_bridge_results/plots/
  step2b_degree_overlay_train_size=50_test_mean_normalized_gap.png
  step2b_degree_overlay_train_size=50_test_mean_normalized_gap.csv
```

当前 `train_size=50` formal plot 已生成。CSV 里的关键值：

```text
degree 1: LR 2.2578e-05, SPO+ 1.2606e-07
degree 2: LR 1.0501e-03, SPO+ 8.3358e-04
degree 4: LR 1.0403e-02, SPO+ 7.9814e-03
degree 8: LR 6.1335e-02, SPO+ 4.9068e-02
```

`PyEPO LR` 与 `my LR`、`PyEPO SPO+` 与 `my SPO+` 在这个 plot
中是 mirror rows：数值来自 formal Step1c run，PyEPO-vs-my 等价性由
前面的 Step2b bridge 证明。这个 plot 的作用是展示 degree 趋势，而不是重新训练
PyEPO 大规模 formal run。

#### Phase 3b：Step2b train_size=50 的 10-seed boxplot

如果要做更接近展示图的 seed-distribution boxplot，用现有
`Step2_resampling` 的 50-seed 汇总，不重新训练：

```bash
python3 surrogate_experiment_results/SPO_validation/kep_vs_pyepo/plot_step2b_seed_boxplots.py \
  --seed-start 0 \
  --seed-count 10 \
  --train-size 50 \
  --formats png pdf
```

默认读取：

```text
surrogate_experiment_results/Step2_resampling/results/
  phase1_heldout400_per_seed.csv
```

默认只取：

```text
block:          step2b
degrees:        1, 2, 4, 8
subset_seed:    0, 1, ..., 9
LR:             2stage_val_mse
SPO+:           spoplus_val_spoplus_loss
y-axis:         test_mean_normalized_gap
```

输出：

```text
surrogate_experiment_results/SPO_validation/kep_vs_pyepo/step2b_bridge_results/plots/
  step2b_degree_boxplot_train_size=50_seeds=0-9_test_mean_normalized_gap.png
  step2b_degree_boxplot_train_size=50_seeds=0-9_test_mean_normalized_gap.pdf
  step2b_degree_boxplot_train_size=50_seeds=0-9_test_mean_normalized_gap.csv
```

当前 CSV 是 `4 degrees x 10 seeds x 4 displayed methods = 160` rows。
四个 displayed methods 的含义仍然和 Phase 3 一样：

```text
PyEPO LR      = mirrored 2stage_val_mse
my LR         = mirrored 2stage_val_mse
PyEPO SPO+    = mirrored spoplus_val_spoplus_loss
my SPO+       = mirrored spoplus_val_spoplus_loss
```

这个 boxplot 的作用是展示 Step2b degree regime 下 train_size=50 的
10 个 subset seeds 分布；PyEPO-vs-my 的数值对齐由前面的 Step2b bridge
负责证明。

如果只想画 small correctness bridge subset，可以显式切换：

```bash
python3 surrogate_experiment_results/SPO_validation/kep_vs_pyepo/plot_step2b_degree_overlay.py \
  --source bridge \
  --metric validation_normalized_gap
```

---

## 这会不会成为“更好的研究对比”？

要看你要证明什么。

### 作为“代码正确性证明”：它是更强的应用内验证

如果你的问题是：

> 我的 Step1c SPO+ 有没有细微问题？

那么 KEP 版 PyEPO-vs-my 是更强的证据。因为它直接验证你的真实 KEP oracle、真实 feature、真实 reward-max convention、真实 Step1c metrics。

### 作为“论文 benchmark 复现”：它不是更好的 benchmark

如果你的问题是：

> 我是否复现了 PyEPO 论文的 benchmark？

那 shortest path 更好。KEP 是你自己的应用，不是 PyEPO 论文 Figure 7 的标准 benchmark。它没有外部 published figure 可以直接对照。

### 作为“科研贡献”：KEP 更有意义，但要换表述

你不能把它表述成：

```text
我们在 KEP 上复现了 PyEPO benchmark。
```

更准确是：

```text
We use PyEPO's SPOPlus implementation as an independent reference implementation
to validate our Step1c SPO+ implementation on the KEP oracle.
```

也就是：**PyEPO 是 reference implementation，不是 benchmark dataset owner。**

---

## 我建议你的最终路线

### 先完成 shortest-path SPO+ full CSV comparison

你现在 LR 已经全量通过，但 SPO+ full benchmark 还没有记录全量 row-by-row comparison。当前 `compare_my_vs_pyepo_csv.py` 已经能逐 CSV、逐 seed 比较 PyEPO 和 my methods，并为 SPO+ 设置了 `True SPO/Unamb SPO <= 1e-5`、`MSE <= 1e-4` 的阈值。 它也会在行数不一致或缺文件时直接报错。 

先跑：

```bash
python3 surrogate_experiment_results/SPO_validation/step1c_vs_pyepo/run_my_spoplus.py \
  --expnum 10

python3 surrogate_experiment_results/SPO_validation/step1c_vs_pyepo/compare_my_vs_pyepo_csv.py \
  --pair spo
```

如果通过，你就有完整的外部标准 benchmark equivalence。

### 再做 KEP small validation

先做 Phase 0 preflight：

```bash
python3 surrogate_experiment_results/SPO_validation/kep_vs_pyepo/validate_kep_phase0.py
```

Phase 0 通过后，再做：

```text
validate_kep_small_setting.py
```

目标不是 performance，而是 exactness：

```text
PyEPO SPOPlus dL/dtheta == Step1c grad_spoplus(theta)
```

这一步比画图更重要。

### 最后画 KEP overlay

最后再画：

```text
PyEPO LR vs my LR
PyEPO SPO+ vs my SPO+
```

但我建议标题写成：

```text
KEP Step1c implementation equivalence check
```

而不是：

```text
KEP benchmark performance comparison
```

因为图的目的不是证明 PyEPO 更好或 my 更好，而是证明二者重合。

---

## 最终判断

**可行，而且值得做。**

但我的建议排序是：

```text
1. 先把 shortest-path my SPO+ full CSV comparison 跑完；
2. 再做 KEP small Level 0–4 validation；
3. 再做 KEP full trajectory equivalence；
4. 最后画 PyEPO LR/SPO+ vs my LR/SPO+ 的 KEP overlay plot。
```

它不是“比 shortest path 更好的标准 benchmark”，但它是**比 shortest path 更贴近 Step1c 的应用内 benchmark**。如果目标是确认你自己写的 Step1c SPO+ 在 KEP 上没有细微问题，KEP 版 PyEPO reference comparison 是非常合理的下一步。
