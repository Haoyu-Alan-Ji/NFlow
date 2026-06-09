from pathlib import Path

import pandas as pd


ROOT = Path("data/n160p100_last_output/layer_fixed08")
OUT_DIR = ROOT / "summary"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_MAP = {
    "L01": (16, 4),  "L02": (16, 8),  "L03": (16, 12), "L04": (16, 16),
    "L05": (24, 4),  "L06": (24, 8),  "L07": (24, 12), "L08": (24, 16),
    "L09": (32, 4),  "L10": (32, 8),  "L11": (32, 12), "L12": (32, 16),
    "L13": (48, 4),  "L14": (48, 8),  "L15": (48, 12), "L16": (48, 16),
    "L17": (64, 4),  "L18": (64, 8),  "L19": (64, 12), "L20": (64, 16),
    "L21": (96, 4),  "L22": (96, 8),  "L23": (96, 12), "L24": (96, 16),
    "L25": (64, 24), "L26": (96, 24),
}


def add_agg(agg_dict, out_col, in_col, func, all_df):
    if in_col in all_df.columns:
        agg_dict[out_col] = (in_col, func)


def main():
    paths = sorted(ROOT.glob("L*/recovery/*/seed_*/last_summary.csv"))
    print("n files:", len(paths))

    if len(paths) == 0:
        raise FileNotFoundError(f"No last_summary.csv found under {ROOT}")

    rows = []

    for p in paths:
        df = pd.read_csv(p)

        config_id = p.parts[p.parts.index("layer_fixed08") + 1]
        mode = p.parts[p.parts.index("layer_fixed08") + 2]
        setting = p.parts[p.parts.index("layer_fixed08") + 3]
        seed_dir = p.parts[p.parts.index("layer_fixed08") + 4]

        if config_id not in CONFIG_MAP:
            raise ValueError(f"Unknown config_id: {config_id}")

        k_q, k_g = CONFIG_MAP[config_id]

        df.insert(0, "config_id", config_id)
        df.insert(1, "run_mode", mode)
        df.insert(2, "run_setting", setting)
        df.insert(3, "seed_dir", seed_dir)

        # Force correct layer settings from the experiment design.
        # Do not rely on last_summary.csv, because old runs may store K_q/K_g as -1.
        df["K_q"] = int(k_q)
        df["K_g"] = int(k_g)

        rows.append(df)

    all_df = pd.concat(rows, ignore_index=True)
    all_df.to_csv(OUT_DIR / "layer_fixed08_all_summary.csv", index=False)

    agg = {}

    for col in [
        "K_q",
        "K_g",
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
    ]:
        add_agg(agg, col, col, "first", all_df)

    add_agg(agg, "n_runs", "seed", "count", all_df)

    metric_cols = [
        "moment_recovery_score",
        "recovery_score",
        "active_mean_zerr_median",
        "active_sd_logerr_median",
        "active_sd_ratio_median",
        "sd_ratio_median",
        "active_marg_skl_median",
        "active_joint_skl_median",
        "softgate_absdiff_median",
        "zero_soft_leakage_median",
        "selected_epoch",
        "selected_tau",
        "precision",
        "recall",
        "f1",
        "selected_size",
        "runtime_sec",
    ]

    for col in metric_cols:
        if col in all_df.columns:
            add_agg(agg, f"{col}_median", col, "median", all_df)
            add_agg(agg, f"{col}_mean", col, "mean", all_df)

    summary = (
        all_df
        .groupby("config_id")
        .agg(**agg)
        .reset_index()
    )

    sort_cols = [
        "moment_recovery_score_median",
        "active_mean_zerr_median_median",
        "active_sd_logerr_median_median",
        "active_marg_skl_median_median",
        "active_joint_skl_median_median",
    ]
    sort_cols = [c for c in sort_cols if c in summary.columns]

    if len(sort_cols) > 0:
        summary = summary.sort_values(sort_cols, ascending=True)

    summary.to_csv(OUT_DIR / "layer_fixed08_config_summary.csv", index=False)

    rank_cols = [
        "config_id",
        "n_runs",
        "K_q",
        "K_g",
        "moment_recovery_score_median",
        "active_mean_zerr_median_median",
        "active_sd_logerr_median_median",
        "active_sd_ratio_median_median",
        "sd_ratio_median_median",
        "active_marg_skl_median_median",
        "active_joint_skl_median_median",
        "selected_epoch_median",
        "selected_tau_median",
        "f1_median",
        "selected_size_median",
        "runtime_sec_median",
    ]

    rank_cols = [c for c in rank_cols if c in summary.columns]
    rank = summary[rank_cols].copy()
    rank.to_csv(OUT_DIR / "layer_fixed08_config_rank.csv", index=False)

    print()
    print("===== config rank =====")
    print(rank.to_string(index=False))

    if {"K_q", "K_g", "moment_recovery_score_median"}.issubset(summary.columns):
        print()
        print("===== pivot: moment_recovery_score_median =====")
        print(
            summary
            .pivot_table(
                index="K_q",
                columns="K_g",
                values="moment_recovery_score_median",
                aggfunc="first",
            )
            .sort_index()
            .to_string()
        )

    if {"K_q", "K_g", "active_sd_ratio_median_median"}.issubset(summary.columns):
        print()
        print("===== pivot: active_sd_ratio_median =====")
        print(
            summary
            .pivot_table(
                index="K_q",
                columns="K_g",
                values="active_sd_ratio_median_median",
                aggfunc="first",
            )
            .sort_index()
            .to_string()
        )

    if {"K_q", "K_g", "active_joint_skl_median_median"}.issubset(summary.columns):
        print()
        print("===== pivot: active_joint_skl_median =====")
        print(
            summary
            .pivot_table(
                index="K_q",
                columns="K_g",
                values="active_joint_skl_median_median",
                aggfunc="first",
            )
            .sort_index()
            .to_string()
        )

    print()
    print("wrote:", OUT_DIR)


if __name__ == "__main__":
    main()