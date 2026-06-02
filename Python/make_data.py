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


SEEDS = list(range(400, 500))

SETTINGS = ["simple"]
# SETTINGS = ["simple", "block_corr", "group_competition"]

DEVICE = torch.device("cpu")
DTYPE = torch.float32

N = 100
P = 500
DATASET_NAME = f"n{N}p{P}_input"
OUT_ROOT = Path("data") / DATASET_NAME
MANIFEST_PATH = Path("data") / f"manifest_n{N}p{P}.csv"

N_ACTIVE = 10
SIGMA2 = 1.0

BETA_LOW = 0.3
BETA_HIGH = 2.0

BLOCK_SIZE = 10
RHO = 0.8

GROUP_SIZE = P // N_ACTIVE
NOISE_X = 0.15

def make_dataset(setting: str, seed: int, device: torch.device, dtype=torch.float32):

    sim_name = setting
    if setting == "jitter":
        sim_name = "group_competition"

    kwargs = dict(
        sim=sim_name,
        n=N,
        p=P,
        seed=seed,
        n_active=N_ACTIVE,
        sigma2=SIGMA2,
        beta_low=BETA_LOW,
        beta_high=BETA_HIGH,
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
    sim_info["seed"] = seed
    sim_info["n"] = N
    sim_info["p"] = P
    sim_info["n_active_target"] = N_ACTIVE

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
    beta_dir = setting_dir / "beta_true"

    setting_dir.mkdir(parents=True, exist_ok=True)
    beta_dir.mkdir(parents=True, exist_ok=True)

    data_path = setting_dir / f"seed_{seed}.csv"
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

    mcmc_out_dir = Path("outputs_mcmc") / DATASET_NAME / setting / f"seed_{seed}"
    last_out_dir = Path("outputs_lastflow") / DATASET_NAME / setting / f"seed_{seed}"

    row = {
        "setting": setting,
        "seed": seed,
        "n": n,
        "p": p,
        "n_active": len(active_idx0),
        "sigma2": SIGMA2,
        "data_path": data_path.as_posix(),
        "beta_path": beta_path.as_posix(),

        # MCMC runner reads this.
        "out_dir": mcmc_out_dir.as_posix(),

        # LaST-Flow runner reads this if present.
        "last_out_dir": last_out_dir.as_posix(),

        # Fallback fields for R if beta_path is missing.
        # active_idx is 1-based for R.
        "active_idx": ";".join(str(j) for j in active_idx1),
        "beta_value": beta_value,
    }

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
    print(manifest.head())


if __name__ == "__main__":
    main()