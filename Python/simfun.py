import numpy as np
import torch
from typing import Any, Dict, Mapping

from utils import Array, as_numpy_1d, as_numpy_2d

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
    """
    Extract X, y, beta_true and optional metadata from a simulation payload.

    Supported formats:
      1) dict-like with keys X, y, beta_true
      2) tuple/list (X, y, beta_true)
      3) dict-like with beta or active_idx instead of beta_true
    """
    if isinstance(sim, (tuple, list)):
        if len(sim) < 3:
            raise ValueError(
                "Tuple/list simulation payload must contain at least "
                "(X, y, beta_true)."
            )

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

        sim_info = {
            k: v
            for k, v in sim.items()
            if k not in {X_key, y_key}
        }

        return {
            "X": X,
            "y": y,
            "beta_true": beta_true,
            "active_idx": active_idx,
            "sim_info": sim_info,
        }

    raise TypeError(f"Unsupported simulation payload type: {type(sim)}")