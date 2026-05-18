from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

# Make sure this points to the repo root where test/HPC_benchmark.py exists.
# If this script is already run from repo root, this is enough.
NFLOW_ROOT = Path(r"E:\positron\NFlow")
if str(NFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(NFLOW_ROOT))

from Python import simfun as sim

OUT_ROOT = Path("data/mcmc_inputs")
MANIFEST_PATH = Path("data/manifest_mcmc.csv")

# Exactly 20 seeds: 407, ..., 426.
# If you want to include 427, use range(407, 428), but then you have 21 seeds.
SEEDS = list(range(408, 428))

# For Table 2, use only these two settings.
SETTINGS = ["block_corr", "jitter"]

DEVICE = torch.device("cpu")
DTYPE = torch.float32

def make_dataset(setting: str, seed: int, device: torch.device, dtype=torch.float32):
    """
    Return X, y, beta_true, sim_info for one setting.
    """

    if setting == "simple":
        X, y, beta, sim_info = sim.simfun1(
            n=1000,
            p=100,
            seed=seed,
            snr=2.5,
            true_prop=0.1,
            device=device,
            dtype=dtype,
        )

    elif setting == "block_corr":
        X, y, beta, sim_info = sim.simfun_block_corr(
            n=1000,
            p=100,
            seed=seed,
            snr=2.5,
            true_prop=0.1,
            rho=0.8,
            block_size=10,
            device=device,
            dtype=dtype,
        )

    elif setting == "jitter":
        X, y, beta, sim_info = sim.simfun_group_competition(
            n=1000,
            p=100,
            seed=seed,
            snr=2.5,
            true_prop=0.1,
            group_size=10,
            noise_x=0.15,
            one_active_per_group=True,
            device=device,
            dtype=dtype,
        )

    else:
        raise ValueError(f"Unknown setting: {setting}")

    sim_info = dict(sim_info)
    sim_info["setting"] = setting
    sim_info["seed"] = seed

    return X, y, beta, sim_info

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def export_one(setting, seed):
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
    active_idx1 = active_idx0 + 1  # R uses 1-based indices.

    setting_dir = OUT_ROOT / setting
    setting_dir.mkdir(parents=True, exist_ok=True)

    data_path = setting_dir / f"seed_{seed}.csv"

    df = pd.DataFrame(X, columns=[f"x{j + 1}" for j in range(p)])
    df.insert(0, "y", y)
    df.to_csv(data_path, index=False)

    nonzero_beta = beta[active_idx0]
    if len(nonzero_beta) == 0:
        raise ValueError(f"No active variables found for {setting}, seed={seed}")

    # Your current R run.R reconstructs b0 using active_idx and beta_value.
    # If nonzero beta values are not identical, this value is only used as
    # an initialization proxy. The MCMC target uses X and y.
    beta_value = float(np.mean(nonzero_beta))

    out_dir = Path("outputs_mcmc") / setting / f"seed_{seed}"

    row = {
        "setting": setting,
        "seed": seed,
        "data_path": data_path.as_posix(),
        "out_dir": out_dir.as_posix(),
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
    manifest.to_csv(MANIFEST_PATH, index=False)

    print(f"[done] wrote {MANIFEST_PATH}")
    print(f"[info] n_rows = {len(manifest)}")
    print(manifest.head())


if __name__ == "__main__":
    main()
