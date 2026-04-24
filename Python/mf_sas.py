from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from .config import MeanFieldBenchmarkConfig
from .meanfield_benchmark_core import _finalize_linear_result
from .utils import Array, center_response, standardize_design


@dataclass
class MFSpikeSlabConfig(MeanFieldBenchmarkConfig):
    pi: float = 0.10
    slab_var: float = 1.0
    a_sigma: float = 1.0
    b_sigma: float = 1.0
    update_sigma2: bool = True
    min_sigma2: float = 1e-8


def _fit_mf_spike_slab(X: Array, y: Array, cfg: MFSpikeSlabConfig) -> Dict[str, Any]:
    n, p = X.shape
    x2 = np.sum(X ** 2, axis=0)
    alpha = np.full(p, min(max(cfg.pi, 1e-6), 1.0 - 1e-6), dtype=float)
    mu = np.zeros(p, dtype=float)
    s2 = np.full(p, cfg.slab_var, dtype=float)
    sigma2 = max(float(np.var(y, ddof=0)), cfg.min_sigma2)
    fitted = X @ (alpha * mu)
    hist: List[Dict[str, float]] = []
    converged = False

    for it in range(1, cfg.max_iter + 1):
        max_delta = 0.0
        for j in range(p):
            old_w = alpha[j] * mu[j]
            r_j = y - (fitted - X[:, j] * old_w)
            s2_j = 1.0 / (x2[j] / sigma2 + 1.0 / cfg.slab_var)
            mu_j = (s2_j / sigma2) * float(X[:, j] @ r_j)
            logit_alpha_j = (
                math.log(cfg.pi / (1.0 - cfg.pi))
                + 0.5 * (math.log(max(s2_j, 1e-30)) - math.log(max(cfg.slab_var, 1e-30)))
                + 0.5 * (mu_j ** 2 / max(s2_j, 1e-30))
            )
            z = float(np.clip(logit_alpha_j, -50.0, 50.0))
            alpha_j = float(1.0 / (1.0 + math.exp(-z)))
            new_w = alpha_j * mu_j
            fitted += X[:, j] * (new_w - old_w)
            max_delta = max(max_delta, abs(alpha_j - alpha[j]), abs(mu_j - mu[j]))
            alpha[j] = alpha_j
            mu[j] = mu_j
            s2[j] = s2_j

        var_beta = alpha * (s2 + mu ** 2) - (alpha * mu) ** 2
        eresid = float(np.sum((y - fitted) ** 2) + np.sum(x2 * var_beta))
        if cfg.update_sigma2:
            sigma2 = max(
                (eresid + 2.0 * cfg.b_sigma) / (n + 2.0 * (cfg.a_sigma + 1.0)),
                cfg.min_sigma2,
            )

        hist.append(
            {
                "iter": float(it),
                "sigma2": float(sigma2),
                "eresid": float(eresid),
                "max_delta": float(max_delta),
                "support_size_0.5": float(np.sum(alpha >= 0.5)),
            }
        )
        if max_delta < cfg.tol:
            converged = True
            break

    beta_mean = alpha * mu
    beta_var = alpha * (s2 + mu ** 2) - beta_mean ** 2
    beta_sd = np.sqrt(np.maximum(beta_var, 1e-12))
    return {
        "beta_mean_std": beta_mean,
        "beta_sd_std": beta_sd,
        "support_score_std": alpha,
        "pip_std": alpha,
        "sigma2": float(sigma2),
        "history": pd.DataFrame(hist),
        "converged": converged,
        "n_iter": len(hist),
        "raw": {"alpha": alpha, "mu": mu, "s2": s2},
    }


def run_mf_spike_slab(
    *,
    X: Array,
    y: Array,
    beta_true: Optional[Array],
    active_idx: Optional[Array],
    seed: int,
    sim_info: Mapping[str, Any],
    splits: Mapping[str, Array],
    cfg: Optional[MFSpikeSlabConfig] = None,
) -> Dict[str, Any]:
    cfg = cfg or MFSpikeSlabConfig()
    X_train, X_val, X_test = X[splits["train"]], X[splits["val"]], X[splits["test"]]
    y_train, y_val, y_test = y[splits["train"]], y[splits["val"]], y[splits["test"]]
    X_train_s, _, _, x_mean, x_scale = standardize_design(X_train, X_val, X_test, standardize_x=cfg.standardize_x)
    y_train_s, _, _, y_mean = center_response(y_train, y_val, y_test, center_y=cfg.center_y)

    t0 = time.perf_counter()
    fit_out = _fit_mf_spike_slab(X_train_s, y_train_s, cfg)
    runtime_sec = time.perf_counter() - t0

    return _finalize_linear_result(
        method="mf_spike_slab",
        seed=seed,
        sim_info=sim_info,
        splits=splits,
        X=X,
        y=y,
        beta_true=beta_true,
        active_idx=active_idx,
        fit_out=fit_out,
        x_mean=x_mean,
        x_scale=x_scale,
        y_mean=y_mean,
        cfg=cfg,
        runtime_sec=runtime_sec,
    )
