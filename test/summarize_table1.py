from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "data" / "hpc_benchmark"

SUMMARY_NAME = "summary_all_methods.csv"

out_all_rows = OUT_ROOT / "all_seed_method_rows.csv"
out_summary = OUT_ROOT / "table1_support_recovery.csv"
out_formatted = OUT_ROOT / "table1_support_recovery_formatted.csv"
out_latex = OUT_ROOT / "table1_support_recovery.tex"

SETTING_ORDER = ["simple", "block_corr", "jitter"]

SETTING_LABELS = {
    "simple": "Independent Gaussian",
    "block_corr": "Block-correlated Gaussian",
    "jitter": "Jittered group competition",
}

METHOD_ORDER = [
    "flow_stagewise",
    "mf_spike_slab",
    "mf_ard",
    "mf_bayes_lasso",
]

METHOD_LABELS = {
    "flow_stagewise": "ATTF",
    "nf_flow": "ATTF",
    "mf_spike_slab": "MF-SAS",
    "mf_ard": "ARD",
    "mf_bayes_lasso": "Bayes LASSO",
}


def infer_setting_from_path(path: Path) -> str:
    # Expected path:
    # data/hpc_benchmark/<setting>/seed_XXX/summary_all_methods.csv
    return path.parents[1].name


def infer_seed_from_path(path: Path) -> int:
    # Expected seed folder: seed_001
    seed_name = path.parent.name
    return int(seed_name.replace("seed_", ""))


def mean_se(x: pd.Series):
    vals = pd.to_numeric(x, errors="coerce").dropna()
    n = len(vals)

    if n == 0:
        return np.nan, np.nan, 0

    mean = vals.mean()
    se = vals.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    return mean, se, n


def fmt_mean_se(mean, se, digits=3):
    if pd.isna(mean):
        return "--"
    if pd.isna(se):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ({se:.{digits}f})"


paths = sorted(OUT_ROOT.glob(f"*/seed_*/{SUMMARY_NAME}"))

if len(paths) == 0:
    raise FileNotFoundError(f"No {SUMMARY_NAME} files found under {OUT_ROOT}")

dfs = []

for path in paths:
    df = pd.read_csv(path)

    if "setting" not in df.columns:
        df["setting"] = infer_setting_from_path(path)

    if "seed" not in df.columns:
        df["seed"] = infer_seed_from_path(path)

    df["source_file"] = str(path.relative_to(PROJECT_ROOT))
    dfs.append(df)

all_df = pd.concat(dfs, ignore_index=True)


n_files = len(paths)
n_rows = len(all_df)

print(f"[INFO] Found summary files: {n_files}")
print(f"[INFO] Total method rows: {n_rows}")

expected_files = 300
expected_rows = 300 * 4

if n_files != expected_files:
    print(f"[WARNING] Expected {expected_files} files, found {n_files}")

if n_rows != expected_rows:
    print(f"[WARNING] Expected {expected_rows} method rows, found {n_rows}")

print("\n[INFO] Files by setting:")
print(
    all_df[["setting", "seed", "source_file"]]
    .drop_duplicates()
    .groupby("setting")
    .size()
)

print("\n[INFO] Rows by setting and method:")
print(
    all_df.groupby(["setting", "method"])
    .size()
    .unstack(fill_value=0)
)

if {"fp", "fn"}.issubset(all_df.columns):
    all_df["exact_recovery"] = (
        (pd.to_numeric(all_df["fp"], errors="coerce") == 0)
        & (pd.to_numeric(all_df["fn"], errors="coerce") == 0)
    ).astype(float)
else:
    all_df["exact_recovery"] = np.nan

# Paper-style aliases
all_df["TPR"] = all_df["recall"] if "recall" in all_df.columns else np.nan
all_df["FDP"] = all_df["fdr"] if "fdr" in all_df.columns else np.nan
all_df["F1"] = all_df["f1"] if "f1" in all_df.columns else np.nan
all_df["Precision"] = all_df["precision"] if "precision" in all_df.columns else np.nan
all_df["SupportSize"] = all_df["support_size"] if "support_size" in all_df.columns else np.nan
all_df["RuntimeSec"] = all_df["runtime_sec"] if "runtime_sec" in all_df.columns else np.nan

all_df["setting_label"] = all_df["setting"].map(SETTING_LABELS).fillna(all_df["setting"])
all_df["method_label"] = all_df["method"].map(METHOD_LABELS).fillna(all_df["method"])


metrics = [
    "TPR",
    "FDP",
    "F1",
    "Precision",
    "SupportSize",
    "exact_recovery",
    "RuntimeSec",
]

rows = []

for (setting, method), sub in all_df.groupby(["setting", "method"], sort=False):
    row = {
        "setting": setting,
        "setting_label": SETTING_LABELS.get(setting, setting),
        "method": method,
        "method_label": METHOD_LABELS.get(method, method),
        "n_runs": sub[["setting", "seed"]].drop_duplicates().shape[0],
    }

    for metric in metrics:
        mean, se, n = mean_se(sub[metric])
        row[f"{metric}_mean"] = mean
        row[f"{metric}_se"] = se
        row[f"{metric}_n"] = n

    rows.append(row)

summary = pd.DataFrame(rows)


setting_rank = {v: i for i, v in enumerate(SETTING_ORDER)}
method_rank = {v: i for i, v in enumerate(METHOD_ORDER)}

summary["setting_rank"] = summary["setting"].map(setting_rank).fillna(999).astype(int)
summary["method_rank"] = summary["method"].map(method_rank).fillna(999).astype(int)

summary = summary.sort_values(["setting_rank", "method_rank", "setting", "method"])
summary = summary.drop(columns=["setting_rank", "method_rank"])

formatted = summary[
    [
        "setting_label",
        "method_label",
        "n_runs",
        "TPR_mean",
        "TPR_se",
        "FDP_mean",
        "FDP_se",
        "F1_mean",
        "F1_se",
        "Precision_mean",
        "Precision_se",
        "SupportSize_mean",
        "SupportSize_se",
        "exact_recovery_mean",
        "exact_recovery_se",
        "RuntimeSec_mean",
        "RuntimeSec_se",
    ]
].copy()

formatted_table = pd.DataFrame(
    {
        "Setting": formatted["setting_label"],
        "Method": formatted["method_label"],
        "Runs": formatted["n_runs"],
        "TPR": [
            fmt_mean_se(m, s)
            for m, s in zip(formatted["TPR_mean"], formatted["TPR_se"])
        ],
        "FDP": [
            fmt_mean_se(m, s)
            for m, s in zip(formatted["FDP_mean"], formatted["FDP_se"])
        ],
        "F1": [
            fmt_mean_se(m, s)
            for m, s in zip(formatted["F1_mean"], formatted["F1_se"])
        ],
        "Precision": [
            fmt_mean_se(m, s)
            for m, s in zip(formatted["Precision_mean"], formatted["Precision_se"])
        ],
        "Support size": [
            fmt_mean_se(m, s)
            for m, s in zip(formatted["SupportSize_mean"], formatted["SupportSize_se"])
        ],
        "Exact recovery": [
            fmt_mean_se(m, s)
            for m, s in zip(
                formatted["exact_recovery_mean"],
                formatted["exact_recovery_se"],
            )
        ],
        "Runtime (s)": [
            fmt_mean_se(m, s, digits=1)
            for m, s in zip(formatted["RuntimeSec_mean"], formatted["RuntimeSec_se"])
        ],
    }
)


all_df.to_csv(out_all_rows, index=False)
summary.to_csv(out_summary, index=False)
formatted_table.to_csv(out_formatted, index=False)

latex_table = formatted_table.to_latex(
    index=False,
    escape=False,
    column_format="llrrrrrrr",
    caption=(
        "Support recovery performance over 100 simulation replicates per setting. "
        "Entries are mean with standard error in parentheses."
    ),
    label="tab:support_recovery",
)

with open(out_latex, "w", encoding="utf-8") as f:
    f.write(latex_table)

print("\n[INFO] Wrote:")
print(out_all_rows)
print(out_summary)
print(out_formatted)
print(out_latex)

print("\n[INFO] Formatted Table 1:")
print(formatted_table.to_string(index=False))