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
    ("sensitivity", "Baseline", "Baseline MLP 2/64", "LaST-Flow",
     ROOT / "data/n160p100/n160p100_last_output/last_default/simple"),
    ("sensitivity", "Baseline", "ResCond", "LaST-Flow",
     ROOT / "data/n160p100/n160p100_last_output/rescond/simple"),
    ("sensitivity", "Baseline", "Deep MLP 4/256", "LaST-Flow",
     ROOT / "data/n160p100/n160p100_last_output/deep_mlp/simple"),

    ("sensitivity", "Low SNR", "Baseline MLP 2/64", "LaST-Flow",
     ROOT / "data/n160p100/n160p100_last_output/last_default/low_snr"),
    ("sensitivity", "Low SNR", "ResCond", "LaST-Flow",
     ROOT / "data/n160p100/n160p100_last_output/rescond/low_snr"),
    ("sensitivity", "Low SNR", "Deep MLP 4/256", "LaST-Flow",
     ROOT / "data/n160p100/n160p100_last_output/deep_mlp/low_snr"),

    ("sensitivity", "n>p", "Baseline MLP 2/64", "LaST-Flow",
     ROOT / "data/n1000p100/n1000p100_last_output/last_default/simple"),
    ("sensitivity", "n>p", "ResCond", "LaST-Flow",
     ROOT / "data/n1000p100/n1000p100_last_output/rescond/simple"),
    ("sensitivity", "n>p", "Deep MLP 4/256", "LaST-Flow",
     ROOT / "data/n1000p100/n1000p100_last_output/deep_mlp/simple"),

    ("sensitivity", "p>>n", "Baseline MLP 2/64", "LaST-Flow",
     ROOT / "data/n100p500/n100p500_last_output/last_default/simple"),
    ("sensitivity", "p>>n", "ResCond", "LaST-Flow",
     ROOT / "data/n100p500/n100p500_last_output/rescond/simple"),
    ("sensitivity", "p>>n", "Deep MLP 4/256", "LaST-Flow",
     ROOT / "data/n100p500/n100p500_last_output/deep_mlp/simple"),

    ("sensitivity", "Weak signals", "Baseline MLP 2/64", "LaST-Flow",
     ROOT / "data/n160p100/n160p100_last_output/last_default/weak_signal"),
    ("sensitivity", "Weak signals", "ResCond", "LaST-Flow",
     ROOT / "data/n160p100/n160p100_last_output/rescond/weak_signal"),
    ("sensitivity", "Weak signals", "Deep MLP 4/256", "LaST-Flow",
     ROOT / "data/n160p100/n160p100_last_output/deep_mlp/weak_signal"),

    ("ablation", "Baseline", "Mean-field", "Mean-field",
     ROOT / "data/n160p100/n160p100_last_output/meanfield/simple"),
    ("ablation", "Baseline", "Affine coupling", "Affine coupling",
     ROOT / "data/n160p100/n160p100_last_output/affine/simple"),

    ("partition", "Baseline", "Semantic-Affine", "Semantic-Affine",
     ROOT / "data/n160p100/n160p100_last_output/semantic_affine_control/simple"),
]

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

        pip = variable["pip"].to_numpy(float)
        mcmc_pip = variable["mcmc_pip"].to_numpy(float)

        if "truth" in variable.columns:
            truth = variable["truth"].to_numpy(int)
        else:
            truth = (
                np.abs(variable["beta_true"].to_numpy(float)) > 1e-12
            ).astype(int)

        pip_diff = pip - mcmc_pip

        br = np.mean((truth - pip) ** 2)
        mcmc_br = np.mean((truth - mcmc_pip) ** 2)

        time_sec = np.nan
        for name in ("total_runtime_sec", "runtime_sec", "train_runtime_sec"):
            if name in summary.index and pd.notna(summary[name]):
                time_sec = float(summary[name])
                break

        rows.append({
            "group": group,
            "environment": environment,
            "configuration": configuration,
            "method": method,
            "seed": int(summary.get(
                "seed",
                seed_dir.name.replace("seed_", ""),
            )),

            "D_SKL_A": float(recovery.get(
                "active_marg_skl_median",
                summary.get("active_marg_skl_median", np.nan),
            )),
            "D_JS_0": float(recovery.get(
                "zero_js_median",
                summary.get("zero_js_median", np.nan),
            )),
            "Normalized_L1_PIP": float(np.mean(np.abs(pip_diff))),
            "RMSE_PIP": float(np.sqrt(np.mean(pip_diff ** 2))),
            "AUROC": float(roc_auc_score(truth, pip)),
            "AUPRC": float(average_precision_score(truth, pip)),
            "MCMCBR": float(mcmc_br),
            "BR": float(br),
            "Time_s": time_sec,
        })

all_runs = pd.DataFrame(rows)

if all_runs.empty:
    raise RuntimeError(
        "No completed runs were found. Check the RUNS paths and downloaded outputs."
    )

all_runs.to_csv(OUT / "all_seed_metrics.csv", index=False)

BASE_METRICS = [
    "D_SKL_A",
    "D_JS_0",
    "Normalized_L1_PIP",
    "RMSE_PIP",
    "AUROC",
    "AUPRC",
    "MCMCBR",
    "BR",
    "Time_s",
]

FINAL_COLUMNS = [
    "D_SKL_A",
    "D_JS_0",
    "Normalized_L1_PIP",
    "RMSE_PIP",
    "AUROC",
    "AUPRC",
    "MCMCBR",
    "BR",
    "BRREL",
    "Time_s",
]


def summarize(frame, index_cols):
    grouped = frame.groupby(index_cols, sort=False)

    means = grouped[BASE_METRICS].mean()
    sds = grouped[BASE_METRICS].std()

    out = means.copy()
    out.insert(0, "N", grouped.size())

    # Ratio of aggregated means, not the mean of unstable seed-level ratios.
    out["BRREL"] = np.where(
        out["MCMCBR"] > 0,
        out["BR"] / out["MCMCBR"],
        np.nan,
    )

    for metric in BASE_METRICS:
        out[f"{metric}_sd"] = sds[metric]

    return out.reset_index()


baseline_last = all_runs[
    (all_runs["group"] == "sensitivity")
    & (all_runs["environment"] == "Baseline")
    & (all_runs["configuration"] == "Baseline MLP 2/64")
].assign(method="LaST-Flow")


ablation = pd.concat([
    all_runs[all_runs["group"] == "ablation"],
    baseline_last,
], ignore_index=True)

ablation = summarize(ablation, ["method"])
ablation.to_csv(OUT / "ablation_table.csv", index=False)


partition = pd.concat([
    all_runs[all_runs["group"] == "partition"],
    baseline_last,
], ignore_index=True)

partition = summarize(partition, ["method"])
partition.to_csv(OUT / "partition_control_table.csv", index=False)


sensitivity = summarize(
    all_runs[all_runs["group"] == "sensitivity"],
    ["environment", "configuration"],
)
sensitivity.to_csv(OUT / "sensitivity_table.csv", index=False)


print("\nAblation")
print(
    ablation[["method", "N"] + FINAL_COLUMNS]
    .round(4)
    .to_string(index=False)
)

print("\nPartition control")
print(
    partition[["method", "N"] + FINAL_COLUMNS]
    .round(4)
    .to_string(index=False)
)

print("\nSensitivity")
print(
    sensitivity[
        ["environment", "configuration", "N"] + FINAL_COLUMNS
    ]
    .round(4)
    .to_string(index=False)
)

print(f"\nSaved to: {OUT}")