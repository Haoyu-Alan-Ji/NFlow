from pathlib import Path
import pandas as pd


root = Path("data/n160p100_last_output/recovery32_grid")

paths = sorted(root.glob("E*/recovery/*/seed_*/last_summary.csv"))
print("n files:", len(paths))

rows = []

for p in paths:
    df = pd.read_csv(p)

    config_id = p.parts[p.parts.index("recovery32_grid") + 1]
    mode = p.parts[p.parts.index("recovery32_grid") + 2]
    setting = p.parts[p.parts.index("recovery32_grid") + 3]
    seed_dir = p.parts[p.parts.index("recovery32_grid") + 4]

    df.insert(0, "config_id", config_id)
    df.insert(1, "run_mode", mode)
    df.insert(2, "run_setting", setting)
    df.insert(3, "seed_dir", seed_dir)

    rows.append(df)

all_df = pd.concat(rows, ignore_index=True)

out_dir = root / "summary"
out_dir.mkdir(parents=True, exist_ok=True)

all_df.to_csv(out_dir / "recovery32_all_summary.csv", index=False)


summary = (
    all_df
    .groupby("config_id")
    .agg(
        n_runs=("seed", "count"),

        tau_start=("tau_start", "first"),
        tau_end=("tau_end", "first"),
        q_entropy_weight=("q_entropy_weight", "first"),
        base_lr=("base_lr", "first"),
        stage_lr_decay=("stage_lr_decay", "first"),
        min_lr_scale=("min_lr_scale", "first"),
        warmup_epochs=("warmup_epochs", "first"),
        n_anneal_stages=("n_anneal_stages", "first"),
        min_stage_epochs=("min_stage_epochs", "first"),
        max_stage_epochs=("max_stage_epochs", "first"),

        moment_recovery_score_median=("moment_recovery_score", "median"),
        moment_recovery_score_mean=("moment_recovery_score", "mean"),

        active_mean_zerr_median=("active_mean_zerr_median", "median"),
        active_mean_zerr_mean=("active_mean_zerr_median", "mean"),

        active_sd_logerr_median=("active_sd_logerr_median", "median"),
        active_sd_logerr_mean=("active_sd_logerr_median", "mean"),

        active_sd_ratio_median=("active_sd_ratio_median", "median"),
        active_sd_ratio_mean=("active_sd_ratio_median", "mean"),

        active_marg_skl_median=("active_marg_skl_median", "median"),
        active_marg_skl_mean=("active_marg_skl_median", "mean"),

        active_joint_skl_median=("active_joint_skl_median", "median"),
        active_joint_skl_mean=("active_joint_skl_median", "mean"),

        softgate_absdiff_median=("softgate_absdiff_median", "median"),
        zero_soft_leakage_median=("zero_soft_leakage_median", "median"),

        selected_epoch_median=("selected_epoch", "median"),
        selected_tau_median=("selected_tau", "median"),

        precision_median=("precision", "median"),
        recall_median=("recall", "median"),
        f1_median=("f1", "median"),
        selected_size_median=("selected_size", "median"),
    )
    .reset_index()
)

summary = summary.sort_values(
    [
        "moment_recovery_score_median",
        "active_mean_zerr_median",
        "active_sd_logerr_median",
        "active_marg_skl_median",
        "active_joint_skl_median",
    ],
    ascending=True,
)

summary.to_csv(out_dir / "recovery32_config_summary.csv", index=False)


rank_cols = [
    "config_id",
    "n_runs",
    "tau_start",
    "tau_end",
    "q_entropy_weight",
    "base_lr",
    "stage_lr_decay",
    "min_lr_scale",
    "warmup_epochs",
    "n_anneal_stages",
    "min_stage_epochs",
    "max_stage_epochs",
    "moment_recovery_score_median",
    "active_mean_zerr_median",
    "active_sd_logerr_median",
    "active_sd_ratio_median",
    "active_marg_skl_median",
    "active_joint_skl_median",
    "selected_epoch_median",
    "selected_tau_median",
    "f1_median",
    "selected_size_median",
]

rank = summary[rank_cols].copy()
rank.to_csv(out_dir / "recovery32_config_rank.csv", index=False)

print()
print("===== config rank =====")
print(rank.to_string(index=False))

print()
print("wrote:", out_dir)