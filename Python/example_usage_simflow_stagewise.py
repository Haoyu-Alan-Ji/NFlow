from annealed_flow_framework import (
    SaveConfig,
    SplitConfig,
    default_balanced_profile,
    simflow_stagewise,
    show_run_summary,
    benchmark_row_from_run,
)

# You already have these functions in your project:
# - simfun1
# - build_flow_vi
# import them from your own module before running.

schedule_cfg = default_balanced_profile(base_lr=5e-5, tau_end=0.40)
split_cfg = SplitConfig(train_frac=0.60, val_frac=0.20, test_frac=0.20, seed=123)
save_cfg = SaveConfig(output_dir="./anneal_run_seed123")

out = simflow_stagewise(
    build_flow_vi=build_flow_vi,
    simfun1=simfun1,
    seed=123,
    n=180,
    p=100,
    snr=3.0,
    true_prop=0.1,
    hidden_units=64,
    num_hidden_layers=2,
    K_q=8,
    K_g=8,
    schedule_cfg=schedule_cfg,
    split_cfg=split_cfg,
    save_cfg=save_cfg,
)

show_run_summary(out, top_k=20)
bench_df = benchmark_row_from_run(out, method_name="flow_anneal")
print(bench_df.to_string(index=False))
