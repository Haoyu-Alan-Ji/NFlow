from pathlib import Path
import sys
import os

import numpy as np
import pandas as pd
import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


# ------------------------------------------------------------
# Project root
# ------------------------------------------------------------

NFLOW_ROOT = Path(r"E:\positron\NFlow")

if str(NFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(NFLOW_ROOT))


from Python import simfun as sim


# ============================================================
# Change this block for each dataset
# ============================================================

SEEDS = list(range(400, 500))

# 1) baseline:
# SETTINGS = ["simple"]
#
# 2) low SNR:
# SETTINGS = ["low_snr"]
#
# 3) weak signal / low beta:
# SETTINGS = ["weak_signal"]
#
# 4) optional:
# SETTINGS = ["block_corr"]
# SETTINGS = ["group_competition"]

SETTINGS = ["simple"]

DEVICE = torch.device("cpu")
DTYPE = torch.float32

N = 1000
P = 100

N_ACTIVE = 10

SIGMA2 = 1.0
LOW_SNR_SIGMA2 = 10.0

BETA_LOW = 0.3
BETA_HIGH = 2.0

WEAK_BETA_LOW = 0.1
WEAK_BETA_HIGH = 0.5

BLOCK_SIZE = 10
RHO = 0.8

GROUP_SIZE = P // N_ACTIVE
NOISE_X = 0.15


# ============================================================
# Output paths
# ============================================================

DATA_ID = f"n{N}p{P}"
DATASET_NAME = f"{DATA_ID}_input"

BASE_ROOT = Path("data") / DATA_ID
OUT_ROOT = BASE_ROOT / DATASET_NAME

if SETTINGS == ["simple"]:
    MANIFEST_PATH = BASE_ROOT / f"manifest_{DATA_ID}.csv"
elif len(SETTINGS) == 1:
    MANIFEST_PATH = BASE_ROOT / f"manifest_{DATA_ID}_{SETTINGS[0]}.csv"
else:
    tag = "_".join(SETTINGS)
    MANIFEST_PATH = BASE_ROOT / f"manifest_{DATA_ID}_{tag}.csv"

MCMC_OUTPUT_ROOT = BASE_ROOT / f"{DATA_ID}_mcmc_output"
LAST_OUTPUT_ROOT = BASE_ROOT / f"{DATA_ID}_last_output"


def make_dataset(setting: str, seed: int, device: torch.device, dtype=torch.float32):
    sim_name = setting

    if setting in ["low_snr", "weak_signal", "low_beta"]:
        sim_name = "simple"

    if setting == "jitter":
        sim_name = "group_competition"

    sigma2 = SIGMA2
    beta_low = BETA_LOW
    beta_high = BETA_HIGH

    if setting == "low_snr":
        sigma2 = LOW_SNR_SIGMA2

    if setting in ["weak_signal", "low_beta"]:
        beta_low = WEAK_BETA_LOW
        beta_high = WEAK_BETA_HIGH

    kwargs = dict(
        sim=sim_name,
        n=N,
        p=P,
        seed=seed,
        n_active=N_ACTIVE,
        sigma2=sigma2,
        beta_low=beta_low,
        beta_high=beta_high,
        device=device,
        dtype=dtype,
    )

    if sim_name == "block_corr":
        kwargs.update(
            rho=RHO,
            block_size=BLOCK_SIZE,
        )

    if sim_name == "group_competition":
        kwargs.update(
            group_size=GROUP_SIZE,
            noise_x=NOISE_X,
            one_active_per_group=True,
        )

    X, y, beta, sim_info = sim.simfun1(**kwargs)

    sim_info = dict(sim_info)
    sim_info["setting"] = setting
    sim_info["sim_name"] = sim_name
    sim_info["seed"] = int(seed)
    sim_info["n"] = int(N)
    sim_info["p"] = int(P)
    sim_info["n_active_target"] = int(N_ACTIVE)
    sim_info["sigma2"] = float(sigma2)
    sim_info["sigma"] = float(np.sqrt(sigma2))
    sim_info["beta_low"] = float(beta_low)
    sim_info["beta_high"] = float(beta_high)

    return X, y, beta, sim_info


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def export_one(setting: str, seed: int):
    X, y, beta, sim_info = make_dataset(
        setting=setting,
        seed=seed,
        device=DEVICE,
        dtype=DTYPE,
    )

    X = to_numpy(X)
    y = to_numpy(y).reshape(-1)
    beta = to_numpy(beta).reshape(-1)

    n, p = X.shape

    active_idx0 = np.where(np.abs(beta) > 1e-12)[0]
    active_idx1 = active_idx0 + 1

    if len(active_idx0) == 0:
        raise ValueError(f"No active variables found for {setting}, seed={seed}")

    setting_dir = OUT_ROOT / setting
    data_dir = setting_dir / "dataframe"
    beta_dir = setting_dir / "beta_true"

    data_dir.mkdir(parents=True, exist_ok=True)
    beta_dir.mkdir(parents=True, exist_ok=True)

    data_path = data_dir / f"seed_{seed}.csv"
    beta_path = beta_dir / f"seed_{seed}_beta_true.csv"

    df = pd.DataFrame(X, columns=[f"x{j + 1}" for j in range(p)])
    df.insert(0, "y", y)
    df.to_csv(data_path, index=False)

    beta_tbl = pd.DataFrame({
        "j0": np.arange(p),
        "j1": np.arange(1, p + 1),
        "variable": [f"x{j + 1}" for j in range(p)],
        "beta_true": beta,
        "active": (np.abs(beta) > 1e-12).astype(int),
    })
    beta_tbl.to_csv(beta_path, index=False)

    nonzero_beta = beta[active_idx0]
    beta_value = float(np.mean(nonzero_beta))

    mcmc_out_dir = MCMC_OUTPUT_ROOT / setting / f"seed_{seed}"
    last_out_dir = LAST_OUTPUT_ROOT / setting / f"seed_{seed}"

    row = {
        "setting": setting,
        "sim_name": sim_info["sim_name"],
        "seed": int(seed),
        "n": int(n),
        "p": int(p),
        "n_active": int(len(active_idx0)),
        "n_active_target": int(N_ACTIVE),

        "sigma2": float(sim_info["sigma2"]),
        "sigma": float(sim_info["sigma"]),
        "beta_low": float(sim_info["beta_low"]),
        "beta_high": float(sim_info["beta_high"]),

        "data_path": data_path.as_posix(),
        "beta_path": beta_path.as_posix(),

        # MCMC runner reads this.
        "out_dir": mcmc_out_dir.as_posix(),

        # LaST-Flow runner reads this if present.
        "last_out_dir": last_out_dir.as_posix(),

        # Fallback fields for R if beta_path is missing.
        # active_idx is 1-based for R.
        "active_idx": ";".join(str(j) for j in active_idx1),
        "active_idx0": ";".join(str(j) for j in active_idx0),
        "beta_value": beta_value,

        "signal_var": float(sim_info.get("signal_var", np.nan)),
        "outcome_var": float(sim_info.get("outcome_var", np.nan)),
        "snr_actual": float(sim_info.get("snr_actual", np.nan)),
    }

    if "rho" in sim_info:
        row["rho"] = float(sim_info["rho"])

    if "block_size" in sim_info:
        row["block_size"] = int(sim_info["block_size"])

    if "group_size" in sim_info:
        row["group_size"] = int(sim_info["group_size"])

    if "noise_x" in sim_info:
        row["noise_x"] = float(sim_info["noise_x"])

    return row


def main():
    rows = []

    for setting in SETTINGS:
        for seed in SEEDS:
            rows.append(export_one(setting, seed))

    manifest = pd.DataFrame(rows)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(MANIFEST_PATH, index=False)

    print(f"[done] wrote {MANIFEST_PATH}")
    print(f"[info] n_rows = {len(manifest)}")
    print(f"[info] input_root = {OUT_ROOT}")
    print(f"[info] mcmc_output_root = {MCMC_OUTPUT_ROOT}")
    print(f"[info] last_output_root = {LAST_OUTPUT_ROOT}")
    print(manifest.head())


if __name__ == "__main__":
    main()