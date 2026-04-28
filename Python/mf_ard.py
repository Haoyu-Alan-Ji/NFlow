from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from .config import BenchmarkConfig
from .benchmark_tools import _finalize_linear_result
from .metric import prob_abs_gt_eps
from .utils import Array, center_response, standardize_design


@dataclass
class MFARDConfig(BenchmarkConfig):
    a0: float = 1e-2
    b0: float = 1e-2
    c0: float = 1e-2
    d0: float = 1e-2
    min_sigma2: float = 1e-8


def _fit_mf_ard(X: Array, y: Array, cfg: MFARDConfig) -> Dict[str, Any]:
    n, p = X.shape
    x2 = np.sum(X ** 2, axis=0)
    mu = np.zeros(p, dtype=float)
    s2 = np.ones(p, dtype=float)
    tau_mean = np.ones(p, dtype=float)
    sigma2 = max(float(np.var(y, ddof=0)), cfg.min_sigma2)
    fitted = X @ mu
    hist: List[Dict[str, float]] = []
    converged = False

    for it in range(1, cfg.max_iter + 1):
        # Coordinate-ascent iteration over all beta_j

        max_delta = 0.0
        # max_delta = max_j |mu_j_new - mu_j_old|

        noise_prec = 1.0 / max(sigma2, cfg.min_sigma2)
        # Noise precision:
        # sigma^{-2} = 1 / sigma^2

        for j in range(p):
            # Update one coordinate beta_j at a time

            old_mu = mu[j]
            # Store old mean:
            # mu_j_old

            r_j = y - (fitted - X[:, j] * old_mu)
            # Partial residual excluding predictor j:
            # r_j = y - sum_{k != j} X_k mu_k
            #     = y - (X mu - X_j mu_j_old)

            s2_j = 1.0 / (noise_prec * x2[j] + tau_mean[j])
            # Variational posterior variance:
            # s_j^2 = 1 / (sigma^{-2} X_j^T X_j + E_q[tau_j])

            mu_j = s2_j * noise_prec * float(X[:, j] @ r_j)
            # Variational posterior mean:
            # mu_j = s_j^2 sigma^{-2} X_j^T r_j

            fitted += X[:, j] * (mu_j - old_mu)
            # Incrementally update fitted value:
            # X mu_new = X mu_old + X_j (mu_j_new - mu_j_old)

            s2[j] = s2_j
            # Store updated posterior variance:
            # s_j^2 <- s2_j

            mu[j] = mu_j
            # Store updated posterior mean:
            # mu_j <- mu_j_new

            tau_mean[j] = (cfg.a0 + 0.5) / (cfg.b0 + 0.5 * (mu_j ** 2 + s2_j))
            # Update ARD local precision expectation:
            # E_q[tau_j]
            # = (a0 + 1/2) / (b0 + 1/2 E_q[beta_j^2])
            # where E_q[beta_j^2] = mu_j^2 + s_j^2

            max_delta = max(max_delta, abs(mu_j - old_mu))
            # Track largest coordinate change:
            # max_delta = max(max_delta, |mu_j_new - mu_j_old|)

        eresid = float(np.sum((y - fitted) ** 2) + np.sum(x2 * s2))
        # Expected residual sum of squares:
        # E_q[||y - X beta||^2]
        # = ||y - X mu||^2 + sum_j (X_j^T X_j) s_j^2

        sigma2 = max(
            (eresid + 2.0 * cfg.d0) / (n + 2.0 * (cfg.c0 + 1.0)),
            cfg.min_sigma2,
        )
        # Update noise variance using inverse-gamma posterior mode:
        # sigma^2
        # = (E_q[||y - X beta||^2] + 2 d0)
        #   / (n + 2(c0 + 1))
        # with lower bound min_sigma2
        
        hist.append(
            {
                "iter": float(it),
                "sigma2": float(sigma2),
                "eresid": float(eresid),
                "max_delta": float(max_delta),
                "mean_tau": float(np.mean(tau_mean)),
            }
        )
        if max_delta < cfg.tol:
            converged = True
            break

    beta_mean = mu
    beta_sd = np.sqrt(np.maximum(s2, 1e-12))
    support_score = prob_abs_gt_eps(beta_mean, beta_sd, cfg.beta_eps)
    return {
        "beta_mean_std": beta_mean,
        "beta_sd_std": beta_sd,
        "support_score_std": support_score,
        "pip_std": None,
        "sigma2": float(sigma2),
        "history": pd.DataFrame(hist),
        "converged": converged,
        "n_iter": len(hist),
        "raw": {"tau_mean": tau_mean, "mu": mu, "s2": s2},
    }


def run_mf_ard(
    *,
    X: Array,
    y: Array,
    beta_true: Optional[Array],
    active_idx: Optional[Array],
    seed: int,
    sim_info: Mapping[str, Any],
    splits: Mapping[str, Array],
    cfg: Optional[MFARDConfig] = None,
) -> Dict[str, Any]:
    cfg = cfg or MFARDConfig()
    X_train, X_val, X_test = X[splits["train"]], X[splits["val"]], X[splits["test"]]
    y_train, y_val, y_test = y[splits["train"]], y[splits["val"]], y[splits["test"]]
    X_train_s, _, _, x_mean, x_scale = standardize_design(X_train, X_val, X_test, standardize_x=cfg.standardize_x)
    y_train_s, _, _, y_mean = center_response(y_train, y_val, y_test, center_y=cfg.center_y)

    t0 = time.perf_counter()
    fit_out = _fit_mf_ard(X_train_s, y_train_s, cfg)
    runtime_sec = time.perf_counter() - t0

    return _finalize_linear_result(
        method="mf_ard",
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

