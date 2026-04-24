## Benchmark framework for variable-selection methods

### 1. Goal

The benchmark should compare methods along three axes:

1. **Selection quality**
   - precision
   - recall
   - F1
   - support size
   - TP / FP / FN

2. **Predictive performance**
   - train / validation / test MSE
   - RMSE
   - MAE
   - \(R^2\)
   - heldout log-likelihood / NLL if available

3. **Computational behavior**
   - runtime
   - convergence stability
   - sensitivity to initialization / seed
   - support stability across repeated runs

The key principle is that all methods must be evaluated under the **same data-generation scheme, same train/validation/test split, and same support-extraction protocol whenever possible**.

---

### 2. Unified experiment protocol

For each simulation setting:

- generate one dataset \((X, y, \beta_{\text{true}})\)
- create one fixed train/validation/test split
- fit each competing method on the same training set
- tune or checkpoint each method using the same validation split whenever the method requires model selection
- evaluate all final outputs on the same test split
- if truth is available, compute support-recovery metrics against \(\beta_{\text{true}}\)

This avoids comparing methods under different randomness, different splits, or different post-processing rules.

---

### 3. Unified output object

Each method should be wrapped so that it returns a common result dictionary:

```python
result = {
    "method": ...,
    "seed": ...,
    "sim_info": ...,
    "splits": ...,
    "runtime_sec": ...,

    "selected_support": ...,
    "beta_est": ...,

    "train_metrics": ...,
    "val_metrics": ...,
    "test_metrics": ...,
    "selection_metrics": ...,

    "pred_table": ...,
    "var_table": ...,
}

This makes downstream comparison, aggregation, plotting, and export much easier.

4. Benchmark code structure

The benchmark layer should contain:

run_flow_method(...)
run_baseline_method(...)
benchmark_row_from_result(...)
run_one_setting_one_seed(...)
run_setting_grid(...)
run_full_benchmark(...)

Recommended logic:

run_flow_method(...)

Runs the full flow-based annealing framework and converts the output to the unified result format.

run_baseline_method(...)

Fits a comparison method and converts its output to the same result format.

benchmark_row_from_result(...)

Compresses one fitted result into a single benchmark row, including:

method name
runtime
support size
predictive metrics
selection metrics
run_one_setting_one_seed(...)

For one simulation configuration and one random seed:

generate data
split data
run one method
return one unified result
run_setting_grid(...)

For one simulation setting and many seeds:

loop over methods
loop over seeds
collect all rows and all raw results
run_full_benchmark(...)

For the full simulation grid:

loop over all settings
call run_setting_grid(...)
concatenate final benchmark tables
5. Simulation settings

A useful benchmark grid should vary:

sample size n
ambient dimension p
signal sparsity
signal-to-noise ratio
optional correlation structure of X

For example:

n∈{120,180,300}
p∈{100,300,500}
true support proportion ∈{0.05,0.1}
SNR ∈{1,2.5,5}

Each setting should be repeated over multiple seeds.

6. Metrics
Selection metrics

When truth is available:

precision
recall
F1
support size
TP / FP / FN / TN
Predictive metrics
train MSE / RMSE / MAE / R
2
validation MSE / RMSE / MAE / R
2
test MSE / RMSE / MAE / R
2
heldout log-likelihood / NLL if supported
Stability metrics
Jaccard similarity of supports across repeated runs
support frequency across checkpoints or seeds
instability score for ambiguous variables
Runtime metrics
total runtime
convergence failures
numerical instability count
7. Support extraction must be standardized

A benchmark is only meaningful if support is extracted in a consistent and transparent way.

For each method, define clearly how the final support is obtained.

Examples:

Flow annealing
posterior draw support from u
j
	​

>t
aggregate by vote rate
final support is chosen using a threshold on inclusion frequency
Spike-and-slab methods
posterior inclusion probabilities
threshold at a fixed cutoff such as 0.5 or 0.9
Continuous shrinkage methods
threshold posterior mean coefficients
or threshold local shrinkage summaries
or use a calibrated support rule based on simulation tuning
Frequentist sparse estimators
support is the set of nonzero fitted coefficients

The support rule should either be shared across methods when possible, or documented explicitly when not.

8. Main benchmark tables

At minimum, produce a main summary table with columns such as:

method
n
p
SNR
true support proportion
precision
recall
F1
selected support size
test MSE
test R
2
runtime

Each entry can be reported as:

mean ± sd
or
median [IQR]

across seeds.

9. Main benchmark plots

Recommended figures:

Runtime vs selection quality
x-axis: runtime
y-axis: F1 or recall
Support size vs predictive fit
x-axis: selected support size
y-axis: test MSE or validation MSE
Selection stability
support-overlap heatmap
Jaccard similarity across seeds
Method-specific diagnostics
for the flow method, retain pathwise diagnostics such as:
training overview
support-vs-predictive path
boundary density
uncertainty vs boundary distance
10. Recommended workflow
First benchmark the flow method alone over multiple seeds to verify that:
outputs are stable
tables are correctly generated
plots are correctly saved
Then add one baseline method at a time.
Only after wrappers are stable, expand to a full simulation grid.

This avoids mixing framework bugs with method-comparison bugs.

11. Benchmark interpretation

The benchmark should answer:

Does the flow-based method improve support recovery?
Does it remain competitive in prediction?
Is the runtime acceptable relative to alternatives?
Does pathwise checkpoint selection improve over naive “take the last iterate” rules?
How stable is the selected support across seeds and settings?



下面说你问的第二件事：**MVB 到底是什么**。

## MVB 是什么

这个缩写**不够标准**，不能默认大家都指同一个方法。

在你这个语境里，可能有几种常见理解：

### 1. Mean-field Variational Bayes
这是最常见、最标准的一类，通常更常写成 **MFVB**。  
意思是把后验近似成各部分独立或分块独立：

$$
q(\theta) = \prod_k q_k(\theta_k)
$$

优点是：
- 快
- 易实现
- 很适合做 baseline

缺点是：
- 常低估后验不确定性
- 对强相关变量或复杂几何可能不够好

如果你要拿一个最经典、最容易对比的 VB baseline，**MFVB spike-and-slab** 是首选。


## 你现在最适合掉包测试的变量选择方法

我按“实现难度”和“比较价值”分层给你。

---

### A. 最容易接进来的 baseline

#### 1. Lasso
最简单的 baseline。

优点：
- 非常容易跑
- 变量选择直观
- 一定要有，作为最低门槛

缺点：
- 不是贝叶斯
- 对相关变量可能不稳定

适合做：
- 最基础 benchmark

---

#### 2. Elastic Net
比 Lasso 稍强，尤其变量相关时更稳。

适合做：
- frequentist baseline
- 检查你的方法是否真的优于简单正则化方法

---

#### 3. Adaptive Lasso
如果你重点强调 support recovery，adaptive lasso 很值得放进来。

优点：
- 比普通 lasso 更接近变量选择目标
- 常被当成 support recovery baseline

---

### B. 最值得的贝叶斯 baseline

#### 4. Mean-field VB spike-and-slab
这是我最建议你优先放进去的。

原因：
- 和你的方法同属 VB
- 但几何表达能力弱很多
- 正好能体现你 flow + annealing 的增益

这是最自然的 “drop-in VB baseline”。

---

#### 5. Spike-and-slab Gibbs / MCMC
这是“金标准式”贝叶斯基准。

优点：
- 后验更完整
- 变量选择解释性强

缺点：
- 慢
- 高维时代价大

适合做：
- 小中等规模 benchmark
- 验证你的 flow 是否接近 MCMC 结果

---

#### 6. Horseshoe prior
可以用 MCMC 版，也可以用近似 VB 版。

优点：
- 连续 shrinkage 很经典
- 高维贝叶斯变量选择常见

缺点：
- support 需要额外后处理
- 不像 spike-and-slab 那么天然离散

如果你放 horseshoe，必须提前写清楚**如何从连续 shrinkage 转成 support**。

---

### C. 很值得考虑的现代 sparse baseline

#### 7. SuSiE
这个非常值得考虑，特别是在高维线性变量选择里。

优点：
- 快
- 输出 posterior inclusion probabilities
- 非常适合和你的 PIP / support 比较

缺点：
- 更偏特定 additive single-effect 框架
- 不是通用 VB flow 方法

但从 benchmark 角度非常有价值。

---

#### 8. ARD / Sparse Bayesian Learning
这类方法也适合做贝叶斯连续 shrinkage baseline。

优点：
- 经典
- 计算上通常不算太重

缺点：
- support 后处理要说明

---

### D. 可选但不一定第一批就上

#### 9. SCAD / MCP
如果你想把非凸罚函数方法也纳入 benchmark，可以放。

#### 10. Stability selection
如果你很强调稳定性，也可以考虑作为补充基准。

---

## 我对你当前 benchmark 组合的建议

如果你现在只想先做一批**最有信息量、最现实**的对比，我建议：

### 第一批
- 你的 Flow annealing method
- Lasso
- Elastic Net
- Mean-field VB spike-and-slab

### 第二批
- Spike-and-slab MCMC
- SuSiE
- Horseshoe

这样推进最合理。

---

## 如果只从“掉包难度”出发

### 最容易掉包
- Lasso
- Elastic Net
- Adaptive Lasso

### 中等难度
- SuSiE
- ARD / SBL

### 最自然的贝叶斯比较对象
- MFVB spike-and-slab
- Spike-and-slab MCMC

### 最难但也最能说明问题
- 你自己的 Flow VB vs MFVB vs MCMC

---


### 关于你现在最适合测试的替代方法
优先顺序我建议是：

1. Lasso  
2. Elastic Net  
3. Mean-field VB spike-and-slab  
4. Spike-and-slab MCMC  
5. SuSiE  
6. Horseshoe

