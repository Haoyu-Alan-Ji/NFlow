from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "postprocess_tables"
OUT.mkdir(parents=True, exist_ok=True)

RUNS = [
    # group, environment, configuration, method, directory

    # Baseline
    ("sensitivity", "Baseline", "Baseline MLP 2/64", "LaST-Flow",
     ROOT / "data/n160p100/n160p100_last_output/last_default/simple"),
    ("sensitivity", "Baseline", "ResCond", "LaST-Flow",
     ROOT / "data/n160p100/n160p100_last_output/rescond/simple"),
    ("sensitivity", "Baseline", "Deep MLP 4/256", "LaST-Flow",
     ROOT / "data/n160p100/n160p100_last_output/deep_mlp/simple"),
    ("sensitivity", "Baseline", "Mean-field", "Mean-field",
     ROOT / "data/n160p100/n160p100_last_output/meanfield/simple"),

    # Low SNR
    ("sensitivity", "Low SNR", "Baseline MLP 2/64", "LaST-Flow",
     ROOT / "data/n160p100/n160p100_last_output/last_default/low_snr"),
    ("sensitivity", "Low SNR", "ResCond", "LaST-Flow",
     ROOT / "data/n160p100/n160p100_last_output/rescond/low_snr"),
    ("sensitivity", "Low SNR", "Deep MLP 4/256", "LaST-Flow",
     ROOT / "data/n160p100/n160p100_last_output/deep_mlp/low_snr"),
    ("sensitivity", "Low SNR", "Mean-field", "Mean-field",
     ROOT / "data/n160p100/n160p100_last_output/meanfield/low_snr"),

    # n > p
    ("sensitivity", "n>p", "Baseline MLP 2/64", "LaST-Flow",
     ROOT / "data/n1000p100/n1000p100_last_output/last_default/simple"),
    ("sensitivity", "n>p", "ResCond", "LaST-Flow",
     ROOT / "data/n1000p100/n1000p100_last_output/rescond/simple"),
    ("sensitivity", "n>p", "Deep MLP 4/256", "LaST-Flow",
     ROOT / "data/n1000p100/n1000p100_last_output/deep_mlp/simple"),
    ("sensitivity", "n>p", "Mean-field", "Mean-field",
     ROOT / "data/n1000p100/n1000p100_last_output/meanfield/simple"),

    # p >> n
    ("sensitivity", "p>>n", "Baseline MLP 2/64", "LaST-Flow",
     ROOT / "data/n100p500/n100p500_last_output/last_default/simple"),
    ("sensitivity", "p>>n", "ResCond", "LaST-Flow",
     ROOT / "data/n100p500/n100p500_last_output/rescond/simple"),
    ("sensitivity", "p>>n", "Deep MLP 4/256", "LaST-Flow",
     ROOT / "data/n100p500/n100p500_last_output/deep_mlp/simple"),
    ("sensitivity", "p>>n", "Mean-field", "Mean-field",
     ROOT / "data/n100p500/n100p500_last_output/meanfield/simple"),

    # Weak signals
    ("sensitivity", "Weak signals", "Baseline MLP 2/64", "LaST-Flow",
     ROOT / "data/n160p100/n160p100_last_output/last_default/weak_signal"),
    ("sensitivity", "Weak signals", "ResCond", "LaST-Flow",
     ROOT / "data/n160p100/n160p100_last_output/rescond/weak_signal"),
    ("sensitivity", "Weak signals", "Deep MLP 4/256", "LaST-Flow",
     ROOT / "data/n160p100/n160p100_last_output/deep_mlp/weak_signal"),
    ("sensitivity", "Weak signals", "Mean-field", "Mean-field",
     ROOT / "data/n160p100/n160p100_last_output/meanfield/weak_signal"),

    # Ablation
    ("ablation", "Baseline", "Affine coupling", "Affine coupling",
     ROOT / "data/n160p100/n160p100_last_output/affine/simple"),

    # Partition control
    ("partition", "Baseline", "Semantic-Affine", "Semantic-Affine",
     ROOT / "data/n160p100/n160p100_last_output/semantic_affine_control/simple"),
]

# ============================================================
# MH reference directories
# ============================================================

MH_DIRS = {
    "Baseline":
        ROOT / "data/n160p100/n160p100_mh_output/simple",

    "Low SNR":
        ROOT / "data/n160p100/n160p100_mh_output/low_snr",

    "Weak signals":
        ROOT / "data/n160p100/n160p100_mh_output/weak_signal",

    "n>p":
        ROOT / "data/n1000p100/n1000p100_mh_output/simple",

    "p>>n":
        ROOT / "data/n100p500/n100p500_mh_output/simple",
}

N_BOOT = 5000
BOOT_SEED = 2027
ALPHA = 0.05


def read_vi_runtime(summary, seed_dir):
    if "runtime_sec" in summary.index and pd.notna(summary["runtime_sec"]):
        return float(summary["runtime_sec"])

    if (
        "total_runtime_min" in summary.index
        and pd.notna(summary["total_runtime_min"])
    ):
        return 60.0 * float(summary["total_runtime_min"])

    final_file = seed_dir / "final_summary.json"
    if final_file.exists():
        final = json.loads(final_file.read_text())
        if pd.notna(final.get("runtime_sec", np.nan)):
            return float(final["runtime_sec"])

    raise KeyError(f"No VI runtime found in {seed_dir}")


def read_mh_runtime(environment, seed):
    summary_file = (
        MH_DIRS[environment]
        / f"seed_{seed}"
        / "mcmc_summary.csv"
    )

    if not summary_file.exists():
        raise FileNotFoundError(
            f"Missing MH summary: {summary_file}"
        )

    summary = pd.read_csv(summary_file).iloc[0]

    if "runtime_s" not in summary.index:
        raise KeyError(
            f"'runtime_s' not found in {summary_file}"
        )

    return float(summary["runtime_s"])


# ============================================================
# Seed-level metrics
# ============================================================

rows = []

for group, environment, configuration, method, run_dir in RUNS:
    for summary_file in sorted(run_dir.glob("seed_*/summary_row.csv")):
        seed_dir = summary_file.parent

        summary = pd.read_csv(summary_file).iloc[0]
        variable = pd.read_csv(seed_dir / "variable_table.csv")

        recovery_file = seed_dir / "recovery_summary.json"
        recovery = (
            json.loads(recovery_file.read_text())
            if recovery_file.exists()
            else {}
        )

        seed_value = summary.get("seed", np.nan)
        seed = (
            int(seed_value)
            if pd.notna(seed_value)
            else int(seed_dir.name.replace("seed_", ""))
        )

        pip = variable["pip"].to_numpy(float)
        mcmc_pip = variable["mcmc_pip"].to_numpy(float)

        if "truth" in variable.columns:
            truth = variable["truth"].to_numpy(int)
        else:
            truth = (
                np.abs(
                    variable["beta_true"].to_numpy(float)
                ) > 1e-12
            ).astype(int)

        pip_diff = pip - mcmc_pip

        br = np.mean((truth - pip) ** 2)
        mcmc_br = np.mean((truth - mcmc_pip) ** 2)

        rows.append({
            "group": group,
            "environment": environment,
            "configuration": configuration,
            "method": method,
            "seed": seed,

            "D_SKL_A": float(recovery.get(
                "active_marg_skl_median",
                summary.get("active_marg_skl_median", np.nan),
            )),

            "D_JS_0": float(recovery.get(
                "zero_js_median",
                summary.get("zero_js_median", np.nan),
            )),

            "Normalized_L1_PIP": float(
                np.mean(np.abs(pip_diff))
            ),

            "RMSE_PIP": float(
                np.sqrt(np.mean(pip_diff ** 2))
            ),

            "AUROC": float(
                roc_auc_score(truth, pip)
            ),

            "AUPRC": float(
                average_precision_score(truth, pip)
            ),

            "MCMC_AUROC": float(
                roc_auc_score(truth, mcmc_pip)
            ),

            "MCMC_AUPRC": float(
                average_precision_score(truth, mcmc_pip)
            ),

            "MCMCBR": float(mcmc_br),
            "BR": float(br),

            "Time_s": read_vi_runtime(summary, seed_dir),

            "MH_Time_s": read_mh_runtime(
                environment,
                seed,
            ),
        })

all_runs = pd.DataFrame(rows)

if all_runs.empty:
    raise RuntimeError(
        "No completed runs were found. "
        "Check the RUNS paths and downloaded outputs."
    )

runtime_check = (
    all_runs
    .groupby(
        ["environment", "configuration"],
        sort=False,
    )
    .agg(
        N=("seed", "size"),
        N_Time=("Time_s", "count"),
        Mean_Time_s=("Time_s", "mean"),
        N_MH_Time=("MH_Time_s", "count"),
        Mean_MH_Time_s=("MH_Time_s", "mean"),
    )
    .reset_index()
)

print("\nRuntime completeness")
print(runtime_check.round(2).to_string(index=False))

all_runs.to_csv(
    OUT / "all_seed_metrics.csv",
    index=False,
)


# ============================================================
# Metrics and bootstrap confidence intervals
# ============================================================

BASE_METRICS = [
    "D_SKL_A",
    "D_JS_0",
    "Normalized_L1_PIP",
    "RMSE_PIP",
    "AUROC",
    "AUPRC",
    "MCMC_AUROC",
    "MCMC_AUPRC",
    "MCMCBR",
    "BR",
    "Time_s",
    "MH_Time_s",
]

FINAL_METRICS = [
    "D_SKL_A",
    "D_JS_0",
    "Normalized_L1_PIP",
    "RMSE_PIP",
    "AUROC",
    "AUPRC",
    "MCMC_AUROC",
    "MCMC_AUPRC",
    "MCMCBR",
    "BR",
    "BRREL",
    "Time_s",
    "MH_Time_s",
]


def bootstrap_mean_ci(values, rng):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.nan, np.nan, np.nan

    estimate = float(values.mean())

    if values.size == 1:
        return estimate, np.nan, np.nan

    boot = rng.choice(
        values,
        size=(N_BOOT, values.size),
        replace=True,
    ).mean(axis=1)

    low, high = np.quantile(
        boot,
        [ALPHA / 2, 1 - ALPHA / 2],
    )

    return estimate, float(low), float(high)


def bootstrap_brrel_ci(frame, rng):
    pair = (
        frame[["BR", "MCMCBR"]]
        .dropna()
        .to_numpy(float)
    )

    if pair.shape[0] == 0:
        return np.nan, np.nan, np.nan

    br_mean = pair[:, 0].mean()
    mcmc_br_mean = pair[:, 1].mean()

    estimate = (
        br_mean / mcmc_br_mean
        if mcmc_br_mean > 0
        else np.nan
    )

    if pair.shape[0] == 1:
        return estimate, np.nan, np.nan

    index = rng.integers(
        0,
        pair.shape[0],
        size=(N_BOOT, pair.shape[0]),
    )

    boot_pair = pair[index]

    boot_br = boot_pair[:, :, 0].mean(axis=1)
    boot_mcmc_br = boot_pair[:, :, 1].mean(axis=1)

    boot_ratio = np.divide(
        boot_br,
        boot_mcmc_br,
        out=np.full(N_BOOT, np.nan),
        where=boot_mcmc_br > 0,
    )

    boot_ratio = boot_ratio[np.isfinite(boot_ratio)]

    if boot_ratio.size == 0:
        return estimate, np.nan, np.nan

    low, high = np.quantile(
        boot_ratio,
        [ALPHA / 2, 1 - ALPHA / 2],
    )

    return estimate, float(low), float(high)


def summarize(frame, index_cols):
    rng = np.random.default_rng(BOOT_SEED)
    rows = []

    grouped = frame.groupby(
        index_cols,
        sort=False,
        dropna=False,
    )

    for keys, group_frame in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = dict(zip(index_cols, keys))
        row["N"] = len(group_frame)

        for metric in BASE_METRICS:
            estimate, low, high = bootstrap_mean_ci(
                group_frame[metric],
                rng,
            )

            row[metric] = estimate
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high

        estimate, low, high = bootstrap_brrel_ci(
            group_frame,
            rng,
        )

        row["BRREL"] = estimate
        row["BRREL_ci_low"] = low
        row["BRREL_ci_high"] = high

        rows.append(row)

    return pd.DataFrame(rows)


def format_ci(estimate, low, high, digits=4):
    if pd.isna(estimate):
        return ""

    if pd.isna(low) or pd.isna(high):
        return f"{estimate:.{digits}f}"

    return (
        f"{estimate:.{digits}f} "
        f"[{low:.{digits}f}, {high:.{digits}f}]"
    )


def display_table(frame, index_cols):
    out = frame[index_cols + ["N"]].copy()

    for metric in FINAL_METRICS:
        digits = 1 if metric in {"Time_s", "MH_Time_s"} else 4

        out[metric] = [
            format_ci(estimate, low, high, digits)
            for estimate, low, high in zip(
                frame[metric],
                frame[f"{metric}_ci_low"],
                frame[f"{metric}_ci_high"],
            )
        ]

    return out


# ============================================================
# Ablation
# ============================================================

baseline_last = all_runs[
    (all_runs["group"] == "sensitivity")
    & (all_runs["environment"] == "Baseline")
    & (
        all_runs["configuration"]
        == "Baseline MLP 2/64"
    )
].assign(method="RAT-Flow")

baseline_meanfield = all_runs[
    (all_runs["group"] == "sensitivity")
    & (all_runs["environment"] == "Baseline")
    & (all_runs["configuration"] == "Mean-field")
].assign(method="Mean-field")

ablation_seed = pd.concat([
    baseline_meanfield,
    all_runs[all_runs["group"] == "ablation"],
    baseline_last,
], ignore_index=True)

ablation = summarize(
    ablation_seed,
    ["method"],
)

ablation_display = display_table(
    ablation,
    ["method"],
)

ablation.to_csv(
    OUT / "ablation_table.csv",
    index=False,
)

ablation_display.to_csv(
    OUT / "ablation_table_display.csv",
    index=False,
)


# ============================================================
# Partition control
# ============================================================

partition_seed = pd.concat([
    all_runs[all_runs["group"] == "partition"],
    baseline_last,
], ignore_index=True)

partition = summarize(
    partition_seed,
    ["method"],
)

partition_display = display_table(
    partition,
    ["method"],
)

partition.to_csv(
    OUT / "partition_control_table.csv",
    index=False,
)

partition_display.to_csv(
    OUT / "partition_control_table_display.csv",
    index=False,
)


# ============================================================
# Sensitivity
# ============================================================

sensitivity_seed = all_runs[
    all_runs["group"] == "sensitivity"
]

sensitivity = summarize(
    sensitivity_seed,
    ["environment", "configuration"],
)

sensitivity_display = display_table(
    sensitivity,
    ["environment", "configuration"],
)

sensitivity.to_csv(
    OUT / "sensitivity_table.csv",
    index=False,
)

sensitivity_display.to_csv(
    OUT / "sensitivity_table_display.csv",
    index=False,
)


# ============================================================
# Print
# ============================================================

print("\nAblation")
print(
    ablation_display.to_string(index=False)
)

print("\nPartition control")
print(
    partition_display.to_string(index=False)
)

print("\nSensitivity")
print(
    sensitivity_display.to_string(index=False)
)

print(f"\nSaved to: {OUT}")