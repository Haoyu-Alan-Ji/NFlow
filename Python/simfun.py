from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np
import torch
from .utils import as_numpy_1d, as_numpy_2d


def simfun1(n=180, p=100, seed=123, snr=3.0, true_prop=0.1, device=None, dtype=torch.float32,):

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    X = rng.standard_normal((n, p)).astype(np.float32)
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)

    n_active = int(p * true_prop)
    active_idx = np.sort(rng.choice(p, size=n_active, replace=False))

    beta_true = np.zeros(p, dtype=np.float32)
    magnitudes = rng.uniform(0.3, 2.0, size=n_active).astype(np.float32)
    signs = rng.choice([-1.0, 1.0], size=n_active).astype(np.float32)
    beta_true[active_idx] = signs * magnitudes

    signal = X @ beta_true
    sigma2 = np.var(signal) / snr
    sigma = np.sqrt(sigma2)

    y = signal + sigma * rng.standard_normal(n).astype(np.float32)
    y = y - y.mean()

    X_t = torch.tensor(X, dtype=dtype, device=device)
    y_t = torch.tensor(y, dtype=dtype, device=device)
    beta_true_t = torch.tensor(beta_true, dtype=dtype, device=device)

    info = {"n": n, "p": p, "n_active": n_active, "sigma2": float(sigma2), "sigma": float(sigma), "active_idx": active_idx, "snr": snr,}

    return X_t, y_t, beta_true_t, info

def extract_sim_arrays(sim: Any) -> Dict[str, Any]:
    """Extract X, y, beta_true, active_idx, and sim_info from a simulation payload."""
    if isinstance(sim, (tuple, list)):
        if len(sim) < 3:
            raise ValueError("Tuple/list simulation payload must contain at least (X, y, beta_true).")
        X, y, beta_true = sim[:3]
        beta_true = as_numpy_1d(beta_true)
        return {
            "X": as_numpy_2d(X),
            "y": as_numpy_1d(y),
            "beta_true": beta_true,
            "active_idx": np.flatnonzero(beta_true != 0.0),
            "sim_info": {},
        }

    if isinstance(sim, Mapping):
        keys = {str(k).lower(): k for k in sim.keys()}
        X_key = keys.get("x")
        y_key = keys.get("y")
        if X_key is None or y_key is None:
            raise ValueError("Simulation payload must contain X and y.")

        X = as_numpy_2d(sim[X_key])
        y = as_numpy_1d(sim[y_key])

        beta_true = None
        for candidate in ["beta_true", "beta", "beta0", "b_true"]:
            if candidate in keys:
                beta_true = as_numpy_1d(sim[keys[candidate]])
                break

        active_idx = None
        for candidate in ["active_idx", "support", "truth_idx", "nonzero_idx"]:
            if candidate in keys:
                active_idx = np.asarray(sim[keys[candidate]], dtype=int)
                break

        if beta_true is None and active_idx is not None:
            beta_true = np.zeros(X.shape[1], dtype=float)
            beta_true[active_idx] = 1.0
        if beta_true is not None and active_idx is None:
            active_idx = np.flatnonzero(beta_true != 0.0)

        sim_info = {k: v for k, v in sim.items() if k not in {X_key, y_key}}
        return {
            "X": X,
            "y": y,
            "beta_true": beta_true,
            "active_idx": active_idx,
            "sim_info": sim_info,
        }

    raise TypeError(f"Unsupported simulation payload type: {type(sim)}")


def simfun_block_corr(
    n=180,
    p=100,
    seed=123,
    snr=2.5,
    true_prop=0.1,
    rho=0.8,
    block_size=10,
    beta_low=0.8,
    beta_high=1.8,
    device=None,
    dtype=None,
):
    """
    Block-correlated sparse linear regression simulation.

    Data-generating model:
        y = X beta + eps

    X has block-wise AR(1) correlation:
        Corr(X_j, X_k) = rho ** |j-k| within each block.

    Parameters
    ----------
    n : int
        Number of observations.

    p : int
        Number of predictors.

    seed : int
        Random seed.

    snr : float
        Signal-to-noise ratio:
            snr = Var(X beta) / sigma^2

    true_prop : float
        Proportion of active variables.

    rho : float
        Within-block AR(1) correlation.

    block_size : int
        Number of variables per block.

    beta_low, beta_high : float
        Active coefficient magnitudes are sampled uniformly from
        [beta_low, beta_high], with random signs.

    device, dtype:
        Torch device and dtype.

    Returns
    -------
    X : torch.Tensor, shape (n, p)
    y : torch.Tensor, shape (n,)
    beta_true : torch.Tensor, shape (p,)
    info : dict
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dtype is None:
        dtype = torch.float32

    rng = np.random.default_rng(seed)

    n_active = int(round(p * true_prop))
    n_active = max(1, min(n_active, p))

    # ----------------------------
    # Build block-correlated X
    # ----------------------------
    blocks = []
    remaining = p

    while remaining > 0:
        b = min(block_size, remaining)

        idx = np.arange(b)
        Sigma = rho ** np.abs(idx[:, None] - idx[None, :])

        X_block = rng.multivariate_normal(
            mean=np.zeros(b),
            cov=Sigma,
            size=n,
        )

        blocks.append(X_block)
        remaining -= b

    X_np = np.concatenate(blocks, axis=1)

    # Standardize columns at the data-generation level.
    X_np = X_np - X_np.mean(axis=0, keepdims=True)
    X_np = X_np / (X_np.std(axis=0, ddof=0, keepdims=True) + 1e-12)

    # ----------------------------
    # Sparse beta
    # ----------------------------
    active_idx = np.sort(rng.choice(p, size=n_active, replace=False))

    beta_np = np.zeros(p, dtype=float)
    signs = rng.choice([-1.0, 1.0], size=n_active)
    mags = rng.uniform(beta_low, beta_high, size=n_active)
    beta_np[active_idx] = signs * mags

    # ----------------------------
    # Generate y with target SNR
    # ----------------------------
    signal = X_np @ beta_np
    signal_var = float(np.var(signal, ddof=0))

    sigma2 = signal_var / float(snr)
    sigma = float(np.sqrt(sigma2))

    eps = rng.normal(loc=0.0, scale=sigma, size=n)
    y_np = signal + eps

    # Center y at generation level only if you want.
    # I leave it uncentered because your workflow already handles center_y.
    # y_np = y_np - y_np.mean()

    X = torch.as_tensor(X_np, dtype=dtype, device=device)
    y = torch.as_tensor(y_np, dtype=dtype, device=device)
    beta_true = torch.as_tensor(beta_np, dtype=dtype, device=device)

    info = {
        "sim": "block_corr",
        "n": n,
        "p": p,
        "n_active": n_active,
        "active_idx": active_idx,
        "snr": float(snr),
        "sigma2": float(sigma2),
        "sigma": float(sigma),
        "rho": float(rho),
        "block_size": int(block_size),
        "beta_low": float(beta_low),
        "beta_high": float(beta_high),
    }

    return X, y, beta_true, info

def simfun_group_competition(
    n=180,
    p=100,
    seed=123,
    snr=2.5,
    true_prop=0.1,
    group_size=10,
    noise_x=0.15,
    beta_low=0.8,
    beta_high=1.8,
    one_active_per_group=True,
    device=None,
    dtype=None,
):
    """
    Group-competition sparse linear regression simulation.

    Data-generating model:
        y = X beta + eps

    Within each group:
        X_{g,j} = z_g + noise_x * eta_{g,j}

    When noise_x is small, variables inside the same group are nearly
    exchangeable. This creates support ambiguity and one-of-K competition.

    Parameters
    ----------
    n : int
        Number of observations.

    p : int
        Number of predictors.

    seed : int
        Random seed.

    snr : float
        Signal-to-noise ratio:
            snr = Var(X beta) / sigma^2

    true_prop : float
        Proportion of active variables.

    group_size : int
        Number of variables per group.

    noise_x : float
        Within-group idiosyncratic noise level.
        Smaller values create stronger competition.

    beta_low, beta_high : float
        Active coefficient magnitudes are sampled uniformly from
        [beta_low, beta_high], with random signs.

    one_active_per_group : bool
        If True, active variables are chosen from distinct groups whenever possible.

    device, dtype:
        Torch device and dtype.

    Returns
    -------
    X : torch.Tensor, shape (n, p)
    y : torch.Tensor, shape (n,)
    beta_true : torch.Tensor, shape (p,)
    info : dict
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dtype is None:
        dtype = torch.float32

    rng = np.random.default_rng(seed)

    n_active = int(round(p * true_prop))
    n_active = max(1, min(n_active, p))

    # ----------------------------
    # Construct group structure
    # ----------------------------
    n_groups = int(np.ceil(p / group_size))

    groups = []
    start = 0
    for g in range(n_groups):
        end = min(start + group_size, p)
        groups.append(np.arange(start, end))
        start = end

    # ----------------------------
    # Build group-competition X
    # ----------------------------
    X_np = np.zeros((n, p), dtype=float)

    for g, cols in enumerate(groups):
        z_g = rng.normal(loc=0.0, scale=1.0, size=(n, 1))
        eta_g = rng.normal(loc=0.0, scale=1.0, size=(n, len(cols)))

        X_np[:, cols] = z_g + noise_x * eta_g

    # Standardize columns at the data-generation level.
    X_np = X_np - X_np.mean(axis=0, keepdims=True)
    X_np = X_np / (X_np.std(axis=0, ddof=0, keepdims=True) + 1e-12)

    # ----------------------------
    # Choose active variables
    # ----------------------------
    if one_active_per_group:
        # Choose active groups first.
        n_active_groups = min(n_active, n_groups)
        active_groups = rng.choice(n_groups, size=n_active_groups, replace=False)

        active_list = []
        for g in active_groups:
            cols = groups[g]
            active_list.append(rng.choice(cols))

        # If n_active > n_groups, fill remaining active variables randomly.
        if n_active > n_active_groups:
            remaining_candidates = np.setdiff1d(
                np.arange(p),
                np.asarray(active_list, dtype=int),
            )
            extra = rng.choice(
                remaining_candidates,
                size=n_active - n_active_groups,
                replace=False,
            )
            active_list.extend(extra.tolist())

        active_idx = np.sort(np.asarray(active_list, dtype=int))

    else:
        active_idx = np.sort(rng.choice(p, size=n_active, replace=False))

    beta_np = np.zeros(p, dtype=float)
    signs = rng.choice([-1.0, 1.0], size=len(active_idx))
    mags = rng.uniform(beta_low, beta_high, size=len(active_idx))
    beta_np[active_idx] = signs * mags

    # ----------------------------
    # Generate y with target SNR
    # ----------------------------
    signal = X_np @ beta_np
    signal_var = float(np.var(signal, ddof=0))

    sigma2 = signal_var / float(snr)
    sigma = float(np.sqrt(sigma2))

    eps = rng.normal(loc=0.0, scale=sigma, size=n)
    y_np = signal + eps

    X = torch.as_tensor(X_np, dtype=dtype, device=device)
    y = torch.as_tensor(y_np, dtype=dtype, device=device)
    beta_true = torch.as_tensor(beta_np, dtype=dtype, device=device)

    # Group id for each variable.
    group_id = np.empty(p, dtype=int)
    for g, cols in enumerate(groups):
        group_id[cols] = g

    active_groups = np.unique(group_id[active_idx])

    info = {
        "sim": "group_competition",
        "n": n,
        "p": p,
        "n_active": len(active_idx),
        "active_idx": active_idx,
        "snr": float(snr),
        "sigma2": float(sigma2),
        "sigma": float(sigma),
        "group_size": int(group_size),
        "n_groups": int(n_groups),
        "noise_x": float(noise_x),
        "one_active_per_group": bool(one_active_per_group),
        "group_id": group_id,
        "active_groups": active_groups,
        "beta_low": float(beta_low),
        "beta_high": float(beta_high),
    }

    return X, y, beta_true, info
