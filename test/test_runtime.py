from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path("data2")
N_BOOT = 5000
SEED = 20260728


def environment(path):
    s = path.as_posix().lower()

    if "n100p500" in s:
        return "p >> n"
    if "n1000p100" in s:
        return "n > p"
    if "low_snr" in s:
        return "Low SNR"
    if "weak_signal" in s:
        return "Weak signals"
    return "Baseline"


def method(path, meta=""):
    s = f"{path.as_posix()} {meta}".lower().replace("_", "-")

    if "mh8-output" in s or "mh-runtime-summary" in s:
        return "MCMC"
    if "meanfield" in s or "mean-field" in s:
        return "Mean-field"
    if "semantic-affine" in s:
        return "Semantic-Affine"
    if "deep-mlp" in s or "deepmlp" in s:
        return "Deep MLP"
    if "rescond" in s:
        return "ResCond"
    if "affine" in s and "semantic" not in s:
        return "Affine"
    if "rat-k16" in s or "mlp2" in s:
        return "RAT-Flow"
    return "Unknown"


rows = []

for file in ROOT.rglob("final_summary.json"):
    if "local_smoke" in file.parts:
        continue

    with file.open(encoding="utf-8") as f:
        obj = json.load(f)

    runtime = obj.get("runtime_sec")
    if runtime is None and obj.get("total_runtime_min") is not None:
        runtime = 60 * float(obj["total_runtime_min"])
    if runtime is None:
        continue

    meta = " ".join(
        str(obj.get(k, ""))
        for k in (
            "method",
            "configuration",
            "config_name",
            "coupling_type",
            "conditioner_type",
        )
    )

    rows.append(
        {
            "environment": environment(file),
            "method": method(file, meta),
            "runtime_s": float(runtime),
            "source": str(file),
        }
    )

for file in ROOT.rglob("mh_runtime_summary.csv"):
    if "local_smoke" in file.parts:
        continue

    dat = pd.read_csv(file)

    for _, row in dat.iterrows():
        runtime = None
        for col in ("total_runtime_s", "mh_runtime_s", "runtime_s"):
            if col in dat.columns and pd.notna(row[col]):
                runtime = float(row[col])
                break

        if runtime is None:
            continue

        rows.append(
            {
                "environment": environment(file),
                "method": "MCMC",
                "runtime_s": runtime,
                "source": str(file),
            }
        )

raw = pd.DataFrame(rows)

if raw.empty:
    raise SystemExit("No runtime files found under data2/.")

rng = np.random.default_rng(SEED)
summary = []

for (env, meth), group in raw.groupby(["environment", "method"], sort=True):
    x = group["runtime_s"].to_numpy()
    boot = np.median(
        rng.choice(x, size=(N_BOOT, len(x)), replace=True),
        axis=1,
    )
    lo, hi = np.quantile(boot, [0.025, 0.975])
    med = np.median(x)

    summary.append(
        {
            "environment": env,
            "method": meth,
            "N": len(x),
            "median_s": med,
            "ci_low_s": lo,
            "ci_high_s": hi,
            "mean_s": np.mean(x),
            "sd_s": np.std(x, ddof=1) if len(x) > 1 else np.nan,
            "median_min": med / 60,
            "ci_low_min": lo / 60,
            "ci_high_min": hi / 60,
            "time_95ci_s": f"{med:.1f} ({lo:.1f}, {hi:.1f})",
            "time_95ci_min": f"{med / 60:.2f} ({lo / 60:.2f}, {hi / 60:.2f})",
        }
    )

summary = pd.DataFrame(summary)

env_order = {
    "Baseline": 0,
    "Low SNR": 1,
    "Weak signals": 2,
    "n > p": 3,
    "p >> n": 4,
}

method_order = {
    "Mean-field": 0,
    "Affine": 1,
    "Semantic-Affine": 2,
    "RAT-Flow": 3,
    "ResCond": 4,
    "Deep MLP": 5,
    "MCMC": 6,
    "Unknown": 99,
}

summary["_env"] = summary["environment"].map(env_order).fillna(99)
summary["_method"] = summary["method"].map(method_order).fillna(99)
summary = (
    summary.sort_values(["_env", "_method"])
    .drop(columns=["_env", "_method"])
    .reset_index(drop=True)
)

Path("results").mkdir(exist_ok=True)
raw.to_csv("results/runtime_raw.csv", index=False)
summary.to_csv("results/runtime_summary.csv", index=False)

print(
    summary[
        [
            "environment",
            "method",
            "N",
            "time_95ci_s",
            "time_95ci_min",
        ]
    ].to_string(index=False)
)

unknown = raw.loc[raw["method"] == "Unknown", "source"]
if len(unknown):
    print("\nUnknown method paths:")
    print("\n".join(unknown))