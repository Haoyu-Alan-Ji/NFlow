import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from meanfield_benchmark_module import (
    SplitConfig,
    SaveConfig,
    MFSpikeSlabConfig,
    MFARDConfig,
    MFBayesLassoConfig,
    run_one_setting_one_seed,
    run_setting_grid,
    run_full_benchmark,
    plot_runtime_vs_f1,
    plot_test_mse_vs_support_size,
    plot_precision_recall,
    plot_support_score_rank,
    save_result_artifacts,
    save_benchmark_table,
)


# =========================================================
# 1. 一个最小的 simfun：接口要和 benchmark 模块一致
#    必须能接收 seed, n, p, snr, true_prop
# =========================================================
def simfun_demo(seed: int, n: int, p: int, snr: float, true_prop: float):
    rng = np.random.default_rng(seed)

    # 真支持集大小
    s = max(1, int(round(p * true_prop)))
    active_idx = np.sort(rng.choice(p, size=s, replace=False))

    # 设计矩阵
    X = rng.normal(size=(n, p))

    # 真 beta
    beta_true = np.zeros(p)
    beta_true[active_idx] = rng.normal(loc=0.0, scale=1.0, size=s)

    # 用 snr 反推噪声方差
    signal = X @ beta_true
    var_signal = np.var(signal, ddof=0)
    sigma2 = var_signal / snr if snr > 0 else 1.0
    sigma = np.sqrt(max(sigma2, 1e-8))

    # 响应
    y = signal + rng.normal(scale=sigma, size=n)

    return {
        "X": X,
        "y": y,
        "beta_true": beta_true,
        "active_idx": active_idx,
        "sigma2": sigma2,
        "snr": snr,
        "n_active": s,
    }


# =========================================================
# 2. split config
# =========================================================
split_cfg = SplitConfig(
    train_frac=0.6,
    val_frac=0.2,
    test_frac=0.2,
    seed=123,
)


# =========================================================
# 3. 三个 mean-field baseline 的配置
# =========================================================
method_cfgs = {
    "mf_spike_slab": MFSpikeSlabConfig(
        pi=0.10,
        slab_var=1.0,
        a_sigma=1.0,
        b_sigma=1.0,
        update_sigma2=True,
        min_sigma2=1e-8,
        support_threshold=0.5,
        beta_eps=0.10,
        standardize_x=True,
        center_y=True,
        max_iter=500,
        tol=1e-5,
        verbose=False,
    ),
    "mf_ard": MFARDConfig(
        a0=1e-2,
        b0=1e-2,
        c0=1e-2,
        d0=1e-2,
        min_sigma2=1e-8,
        support_threshold=0.5,
        beta_eps=0.10,
        standardize_x=True,
        center_y=True,
        max_iter=500,
        tol=1e-5,
        verbose=False,
    ),
    "mf_bayes_lasso": MFBayesLassoConfig(
        lasso_lambda=1.0,
        c0=1e-2,
        d0=1e-2,
        min_sigma2=1e-8,
        support_threshold=0.5,
        beta_eps=0.10,
        standardize_x=True,
        center_y=True,
        max_iter=500,
        tol=1e-5,
        verbose=False,
    ),
}


# =========================================================
# 4. 先跑单个 setting、单个 seed
# =========================================================
methods = [
    "mf_spike_slab",
    "mf_ard",
    "mf_bayes_lasso",
]

results, table = run_one_setting_one_seed(
    simfun=simfun_demo,
    seed=123,
    n=180,
    p=100,
    snr=3.0,
    true_prop=0.10,
    methods=methods,
    split_cfg=split_cfg,
    method_cfgs=method_cfgs,
)

print("\n===== benchmark row table =====")
print(table)

print("\n===== one result keys =====")
print(results[0].keys())

print("\n===== first result: selection metrics =====")
print(results[0]["selection_metrics"])

print("\n===== first result: predictive metrics =====")
print(results[0]["predictive_metrics"])

print("\n===== first result: selected support =====")
print(results[0]["selected_support"])

print("\n===== first result: var_table head =====")
print(results[0]["var_table"].head(10))


# =========================================================
# 5. 画 benchmark 层的图
# =========================================================
fig1 = plot_runtime_vs_f1(table)
fig2 = plot_test_mse_vs_support_size(table)
fig3 = plot_precision_recall(table)
fig4 = plot_support_score_rank(results[0])

plt.show()


# =========================================================
# 6. 可选：保存结果
# =========================================================
save_cfg = SaveConfig(
    output_dir="./benchmark_demo_out",
    save_history_csv=True,
    save_final_json=True,
    save_predictions_csv=True,
    save_var_table_csv=True,
    save_benchmark_csv=True,
    save_plots=True,
)

for res in results:
    save_result_artifacts(res, save_cfg)

save_benchmark_table(table, save_cfg)