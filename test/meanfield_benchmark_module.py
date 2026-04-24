from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit, logit, ndtr


Array = np.ndarray


# -----------------------------------------------------------------------------
# Configs
# -----------------------------------------------------------------------------


@dataclass
class SplitConfig:
    train_frac: float = 0.6
    val_frac: float = 0.2
    test_frac: float = 0.2
    seed: int = 123


@dataclass
class SaveConfig:
    output_dir: Optional[str] = None
    save_history_csv: bool = True
    save_final_json: bool = True
    save_predictions_csv: bool = True
    save_var_table_csv: bool = True
    save_benchmark_csv: bool = True
    save_plots: bool = True


@dataclass
class BenchmarkConfig:
    support_threshold: float = 0.5
    beta_eps: float = 0.10
    standardize_x: bool = True
    center_y: bool = True
    max_iter: int = 500
    tol: float = 1e-5
    verbose: bool = False


@dataclass
class MFSpikeSlabConfig(BenchmarkConfig):
    pi: float = 0.10
    slab_var: float = 1.0
    a_sigma: float = 1.0
    b_sigma: float = 1.0
    update_sigma2: bool = True
    min_sigma2: float = 1e-8


@dataclass
class MFARDConfig(BenchmarkConfig):
    a0: float = 1e-2
    b0: float = 1e-2
    c0: float = 1e-2
    d0: float = 1e-2
    min_sigma2: float = 1e-8


@dataclass
class MFBayesLassoConfig(BenchmarkConfig):
    lasso_lambda: float = 1.0
    c0: float = 1e-2
    d0: float = 1e-2
    min_sigma2: float = 1e-8


# -----------------------------------------------------------------------------
# Serialization helpers
# -----------------------------------------------------------------------------


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if is_dataclass(obj):
            return asdict(obj)
        return super().default(obj)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def _as_numpy_1d(x: Any) -> Array:
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1D array, got shape {arr.shape}.")
    return arr.astype(float)


def _as_numpy_2d(x: Any) -> Array:
    arr = np.asarray(x)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {arr.shape}.")
    return arr.astype(float)


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
            raise ValueError("Tuple/list simulation payload must contain at least (X, y, beta_true).")
        X, y, beta_true = sim[:3]
        return {
            "X": _as_numpy_2d(X),
            "y": _as_numpy_1d(y),
            "beta_true": _as_numpy_1d(beta_true),
            "active_idx": np.flatnonzero(np.asarray(beta_true) != 0.0),
            "sim_info": {},
        }

    if isinstance(sim, Mapping):
        keys = {k.lower(): k for k in sim.keys()}
        X_key = keys.get("x")
        y_key = keys.get("y")
        if X_key is None or y_key is None:
            raise ValueError("Simulation payload must contain X and y.")
        X = _as_numpy_2d(sim[X_key])
        y = _as_numpy_1d(sim[y_key])

        beta_true = None
        for candidate in ["beta_true", "beta", "beta0", "b_true"]:
            if candidate in keys:
                beta_true = _as_numpy_1d(sim[keys[candidate]])
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

    raise TypeError("Unsupported simulation payload type.")



def make_splits(n: int, split_cfg: SplitConfig) -> Dict[str, Array]:
    if not math.isclose(split_cfg.train_frac + split_cfg.val_frac + split_cfg.test_frac, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("train_frac + val_frac + test_frac must equal 1.")
    rng = np.random.default_rng(split_cfg.seed)
    idx = rng.permutation(n)
    n_train = int(round(n * split_cfg.train_frac))
    n_val = int(round(n * split_cfg.val_frac))
    n_train = min(max(n_train, 1), n - 2)
    n_val = min(max(n_val, 1), n - n_train - 1)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return {"train": train_idx, "val": val_idx, "test": test_idx}



def standardize_design(
    X_train: Array,
    X_val: Array,
    X_test: Array,
    *,
    standardize_x: bool = True,
) -> Tuple[Array, Array, Array, Array, Array]:
    x_mean = X_train.mean(axis=0)
    x_scale = X_train.std(axis=0, ddof=0)
    x_scale = np.where(x_scale < 1e-12, 1.0, x_scale)
    if not standardize_x:
        x_mean = np.zeros_like(x_mean)
        x_scale = np.ones_like(x_scale)
    X_train_s = (X_train - x_mean) / x_scale
    X_val_s = (X_val - x_mean) / x_scale
    X_test_s = (X_test - x_mean) / x_scale
    return X_train_s, X_val_s, X_test_s, x_mean, x_scale



def center_response(y_train: Array, y_val: Array, y_test: Array, *, center_y: bool = True) -> Tuple[Array, Array, Array, float]:
    y_mean = float(y_train.mean()) if center_y else 0.0
    return y_train - y_mean, y_val - y_mean, y_test - y_mean, y_mean



def recover_original_scale(beta_std: Array, beta_std_sd: Array, x_mean: Array, x_scale: Array, y_mean: float) -> Tuple[Array, Array, float]:
    beta = beta_std / x_scale
    beta_sd = beta_std_sd / x_scale
    intercept = y_mean - float(np.dot(x_mean, beta))
    return beta, beta_sd, intercept



def predict_linear(X: Array, beta: Array, intercept: float) -> Array:
    return intercept + X @ beta



def gaussian_predictive_metrics(y_true: Array, y_pred: Array, sigma2: float) -> Dict[str, float]:
    resid = y_true - y_pred
    mse = float(np.mean(resid ** 2))
    rmse = float(np.sqrt(max(mse, 0.0)))
    mae = float(np.mean(np.abs(resid)))
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - np.sum(resid ** 2) / denom) if denom > 0 else float("nan")
    sigma2 = float(max(sigma2, 1e-12))
    loglik = float(np.mean(-0.5 * (np.log(2.0 * np.pi * sigma2) + resid ** 2 / sigma2)))
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "heldout_loglik": loglik,
        "nll": -loglik,
    }



def support_metrics(selected_support: Sequence[int], beta_true: Optional[Array] = None, active_idx: Optional[Array] = None, p: Optional[int] = None) -> Dict[str, float]:
    if beta_true is None and active_idx is None:
        return {}
    if beta_true is not None:
        true_support = set(np.flatnonzero(np.asarray(beta_true) != 0.0).tolist())
        p = int(len(beta_true))
    else:
        assert active_idx is not None
        true_support = set(np.asarray(active_idx, dtype=int).tolist())
        if p is None:
            raise ValueError("p must be supplied if beta_true is not available.")

    selected = set(int(j) for j in selected_support)
    tp = len(selected & true_support)
    fp = len(selected - true_support)
    fn = len(true_support - selected)
    tn = int(p) - tp - fp - fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fdr = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fdr": float(fdr),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "support_size": float(len(selected_support)),
    }



def prob_abs_gt_eps(mu: Array, sd: Array, eps: float) -> Array:
    sd_safe = np.maximum(sd, 1e-12)
    upper = 1.0 - ndtr((eps - mu) / sd_safe)
    lower = ndtr((-eps - mu) / sd_safe)
    return np.clip(upper + lower, 0.0, 1.0)



def top_var_table(
    *,
    beta_mean: Array,
    beta_sd: Array,
    support_score: Array,
    selected_support: Sequence[int],
    beta_true: Optional[Array],
    top_k: int = 20,
    score_name: str = "support_score",
) -> pd.DataFrame:
    selected_mask = np.zeros(len(beta_mean), dtype=int)
    selected_mask[list(selected_support)] = 1 if len(selected_support) > 0 else 0
    df = pd.DataFrame({
        "j": np.arange(len(beta_mean), dtype=int),
        "beta_mean": beta_mean,
        "beta_sd": beta_sd,
        score_name: support_score,
        "selected": selected_mask,
    })
    if beta_true is not None:
        beta_true = np.asarray(beta_true)
        truth = (beta_true != 0.0).astype(int)
        df["beta_true"] = beta_true
        df["truth"] = truth
    df["abs_beta_mean"] = np.abs(df["beta_mean"])
    df = df.sort_values([score_name, "abs_beta_mean"], ascending=[False, False]).reset_index(drop=True)
    return df.head(top_k)



def benchmark_row_from_result(result: Mapping[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "method": result.get("method"),
        "seed": result.get("seed"),
        "runtime_sec": result.get("runtime_sec"),
        "converged": result.get("converged"),
        "n_iter": result.get("n_iter"),
        "support_size": result.get("selection_metrics", {}).get("support_size"),
        "precision": result.get("selection_metrics", {}).get("precision"),
        "recall": result.get("selection_metrics", {}).get("recall"),
        "f1": result.get("selection_metrics", {}).get("f1"),
        "fdr": result.get("selection_metrics", {}).get("fdr"),
        "tp": result.get("selection_metrics", {}).get("tp"),
        "fp": result.get("selection_metrics", {}).get("fp"),
        "fn": result.get("selection_metrics", {}).get("fn"),
        "train_mse": result.get("predictive_metrics", {}).get("train", {}).get("mse"),
        "val_mse": result.get("predictive_metrics", {}).get("val", {}).get("mse"),
        "test_mse": result.get("predictive_metrics", {}).get("test", {}).get("mse"),
        "train_r2": result.get("predictive_metrics", {}).get("train", {}).get("r2"),
        "val_r2": result.get("predictive_metrics", {}).get("val", {}).get("r2"),
        "test_r2": result.get("predictive_metrics", {}).get("test", {}).get("r2"),
        "val_nll": result.get("predictive_metrics", {}).get("val", {}).get("nll"),
        "test_nll": result.get("predictive_metrics", {}).get("test", {}).get("nll"),
    }
    sim_info = result.get("sim_info", {}) or {}
    for key in ["n", "p", "snr", "true_prop", "n_active"]:
        if key in sim_info:
            row[key] = sim_info[key]
    return row


# -----------------------------------------------------------------------------
# Mean-field solvers
# -----------------------------------------------------------------------------



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
                logit(cfg.pi)
                + 0.5 * (math.log(max(s2_j, 1e-30)) - math.log(max(cfg.slab_var, 1e-30)))
                + 0.5 * (mu_j ** 2 / max(s2_j, 1e-30))
            )
            alpha_j = float(expit(np.clip(logit_alpha_j, -50.0, 50.0)))
            new_w = alpha_j * mu_j
            fitted += X[:, j] * (new_w - old_w)
            max_delta = max(max_delta, abs(alpha_j - alpha[j]), abs(mu_j - mu[j]))
            alpha[j] = alpha_j
            mu[j] = mu_j
            s2[j] = s2_j

        var_beta = alpha * (s2 + mu ** 2) - (alpha * mu) ** 2
        eresid = float(np.sum((y - fitted) ** 2) + np.sum(x2 * var_beta))
        if cfg.update_sigma2:
            sigma2 = max((eresid + 2.0 * cfg.b_sigma) / (n + 2.0 * (cfg.a_sigma + 1.0)), cfg.min_sigma2)

        hist.append({
            "iter": float(it),
            "sigma2": float(sigma2),
            "eresid": float(eresid),
            "max_delta": float(max_delta),
            "support_size_0.5": float(np.sum(alpha >= 0.5)),
        })
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
        max_delta = 0.0
        noise_prec = 1.0 / max(sigma2, cfg.min_sigma2)
        for j in range(p):
            old_mu = mu[j]
            r_j = y - (fitted - X[:, j] * old_mu)
            s2_j = 1.0 / (noise_prec * x2[j] + tau_mean[j])
            mu_j = s2_j * noise_prec * float(X[:, j] @ r_j)
            fitted += X[:, j] * (mu_j - old_mu)
            s2[j] = s2_j
            mu[j] = mu_j
            tau_mean[j] = (cfg.a0 + 0.5) / (cfg.b0 + 0.5 * (mu_j ** 2 + s2_j))
            max_delta = max(max_delta, abs(mu_j - old_mu))

        eresid = float(np.sum((y - fitted) ** 2) + np.sum(x2 * s2))
        sigma2 = max((eresid + 2.0 * cfg.d0) / (n + 2.0 * (cfg.c0 + 1.0)), cfg.min_sigma2)
        hist.append({
            "iter": float(it),
            "sigma2": float(sigma2),
            "eresid": float(eresid),
            "max_delta": float(max_delta),
            "mean_tau": float(np.mean(tau_mean)),
        })
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



def _fit_mf_bayes_lasso(X: Array, y: Array, cfg: MFBayesLassoConfig) -> Dict[str, Any]:
    n, p = X.shape
    x2 = np.sum(X ** 2, axis=0)
    mu = np.zeros(p, dtype=float)
    s2 = np.ones(p, dtype=float)
    e_inv_tau = np.ones(p, dtype=float)
    sigma2 = max(float(np.var(y, ddof=0)), cfg.min_sigma2)
    fitted = X @ mu
    hist: List[Dict[str, float]] = []
    converged = False

    for it in range(1, cfg.max_iter + 1):
        max_delta = 0.0
        noise_prec = 1.0 / max(sigma2, cfg.min_sigma2)
        for j in range(p):
            old_mu = mu[j]
            prior_prec = noise_prec * e_inv_tau[j]
            r_j = y - (fitted - X[:, j] * old_mu)
            s2_j = 1.0 / (noise_prec * x2[j] + prior_prec)
            mu_j = s2_j * noise_prec * float(X[:, j] @ r_j)
            fitted += X[:, j] * (mu_j - old_mu)
            mu[j] = mu_j
            s2[j] = s2_j
            e_beta2 = mu_j ** 2 + s2_j
            ratio = max(e_beta2 / max(sigma2, cfg.min_sigma2), 1e-12)
            e_inv_tau[j] = cfg.lasso_lambda / math.sqrt(ratio)
            max_delta = max(max_delta, abs(mu_j - old_mu))

        eresid = float(np.sum((y - fitted) ** 2) + np.sum(x2 * s2))
        prior_quad = float(np.sum((mu ** 2 + s2) * e_inv_tau))
        sigma2 = max((eresid + prior_quad + 2.0 * cfg.d0) / (n + p + 2.0 * (cfg.c0 + 1.0)), cfg.min_sigma2)
        hist.append({
            "iter": float(it),
            "sigma2": float(sigma2),
            "eresid": float(eresid),
            "prior_quad": float(prior_quad),
            "max_delta": float(max_delta),
        })
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
        "raw": {"e_inv_tau": e_inv_tau, "mu": mu, "s2": s2},
    }


# -----------------------------------------------------------------------------
# Unified benchmark wrappers
# -----------------------------------------------------------------------------



def _finalize_linear_result(
    *,
    method: str,
    seed: int,
    sim_info: Mapping[str, Any],
    splits: Mapping[str, Array],
    X: Array,
    y: Array,
    beta_true: Optional[Array],
    active_idx: Optional[Array],
    fit_out: Mapping[str, Any],
    x_mean: Array,
    x_scale: Array,
    y_mean: float,
    cfg: BenchmarkConfig,
    runtime_sec: float,
) -> Dict[str, Any]:
    beta_mean, beta_sd, intercept = recover_original_scale(
        np.asarray(fit_out["beta_mean_std"]),
        np.asarray(fit_out["beta_sd_std"]),
        x_mean,
        x_scale,
        y_mean,
    )
    support_score = np.asarray(fit_out["support_score_std"], dtype=float)
    selected_support = np.flatnonzero(support_score >= cfg.support_threshold).tolist()
    sigma2 = float(fit_out.get("sigma2", max(np.var(y, ddof=0), 1e-12)))
    yhat = predict_linear(X, beta_mean, intercept)

    pred_table = []
    predictive_metrics = {}
    for split_name, idx in splits.items():
        split_metrics = gaussian_predictive_metrics(y[idx], yhat[idx], sigma2)
        predictive_metrics[split_name] = split_metrics
        pred_table.append({"split": split_name, **split_metrics})

    selection = support_metrics(selected_support, beta_true=beta_true, active_idx=active_idx, p=X.shape[1])
    var_df = top_var_table(
        beta_mean=beta_mean,
        beta_sd=beta_sd,
        support_score=support_score,
        selected_support=selected_support,
        beta_true=beta_true,
        top_k=min(20, X.shape[1]),
        score_name="support_score",
    )
    if beta_true is None and active_idx is not None:
        truth = np.zeros(X.shape[1], dtype=int)
        truth[np.asarray(active_idx, dtype=int)] = 1
        var_df["truth"] = truth[var_df["j"].to_numpy()]

    result = {
        "method": method,
        "seed": seed,
        "sim_info": dict(sim_info),
        "splits": {k: np.asarray(v, dtype=int) for k, v in splits.items()},
        "runtime_sec": float(runtime_sec),
        "converged": bool(fit_out.get("converged", False)),
        "n_iter": int(fit_out.get("n_iter", 0)),
        "support_threshold": float(cfg.support_threshold),
        "beta_eps": float(cfg.beta_eps),
        "intercept": float(intercept),
        "sigma2": sigma2,
        "selected_support": selected_support,
        "beta_est": beta_mean,
        "beta_sd": beta_sd,
        "support_score": support_score,
        "pip": fit_out.get("pip_std"),
        "predictive_metrics": predictive_metrics,
        "selection_metrics": selection,
        "pred_table": pd.DataFrame(pred_table),
        "var_table": var_df,
        "history": fit_out.get("history", pd.DataFrame()),
        "yhat": yhat,
        "config": asdict(cfg),
        "raw": fit_out.get("raw", {}),
    }
    return result



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
    X_train_s, X_val_s, X_test_s, x_mean, x_scale = standardize_design(
        X_train, X_val, X_test, standardize_x=cfg.standardize_x
    )
    y_train_s, y_val_s, y_test_s, y_mean = center_response(y_train, y_val, y_test, center_y=cfg.center_y)

    t0 = time.perf_counter()
    fit_out = _fit_mf_spike_slab(X_train_s, y_train_s, cfg)
    runtime_sec = time.perf_counter() - t0

    X_all_s = np.empty_like(X)
    X_all_s[splits["train"]] = X_train_s
    X_all_s[splits["val"]] = X_val_s
    X_all_s[splits["test"]] = X_test_s

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
    X_train_s, X_val_s, X_test_s, x_mean, x_scale = standardize_design(
        X_train, X_val, X_test, standardize_x=cfg.standardize_x
    )
    y_train_s, y_val_s, y_test_s, y_mean = center_response(y_train, y_val, y_test, center_y=cfg.center_y)

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



def run_mf_bayes_lasso(
    *,
    X: Array,
    y: Array,
    beta_true: Optional[Array],
    active_idx: Optional[Array],
    seed: int,
    sim_info: Mapping[str, Any],
    splits: Mapping[str, Array],
    cfg: Optional[MFBayesLassoConfig] = None,
) -> Dict[str, Any]:
    cfg = cfg or MFBayesLassoConfig()
    X_train, X_val, X_test = X[splits["train"]], X[splits["val"]], X[splits["test"]]
    y_train, y_val, y_test = y[splits["train"]], y[splits["val"]], y[splits["test"]]
    X_train_s, X_val_s, X_test_s, x_mean, x_scale = standardize_design(
        X_train, X_val, X_test, standardize_x=cfg.standardize_x
    )
    y_train_s, y_val_s, y_test_s, y_mean = center_response(y_train, y_val, y_test, center_y=cfg.center_y)

    t0 = time.perf_counter()
    fit_out = _fit_mf_bayes_lasso(X_train_s, y_train_s, cfg)
    runtime_sec = time.perf_counter() - t0

    return _finalize_linear_result(
        method="mf_bayes_lasso",
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


# -----------------------------------------------------------------------------
# Adapters for existing spike-and-slab and future MCMC
# -----------------------------------------------------------------------------



def adapt_existing_spike_slab_output(
    flow_out: Mapping[str, Any],
    *,
    method: str = "flow_spike_slab",
    benchmark_support_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Convert an existing spike-and-slab result object into the unified format.

    This adapter is intentionally permissive because existing codebases expose
    slightly different field names. Preferred raw fields to expose from the
    existing method are:
      - beta_est or beta_mean
      - beta_sd (optional)
      - pip or support_score
      - selected_support
      - predictive_metrics / pred_table
      - selection_metrics
      - runtime_sec
      - sim_info / splits
    """
    out = dict(flow_out)
    beta = np.asarray(out.get("beta_est", out.get("beta_mean", out.get("beta_hard_mean", []))), dtype=float)
    beta_sd = np.asarray(out.get("beta_sd", np.zeros_like(beta)), dtype=float)

    score = out.get("support_score", out.get("pip", out.get("inclusion_prob", None)))
    if score is not None:
        score = np.asarray(score, dtype=float)
    selected_support = out.get("selected_support", None)
    if selected_support is None:
        if score is None:
            selected_support = []
        else:
            threshold = 0.5 if benchmark_support_threshold is None else benchmark_support_threshold
            selected_support = np.flatnonzero(score >= threshold).tolist()

    pred_table = out.get("pred_table", pd.DataFrame())
    if isinstance(pred_table, list):
        pred_table = pd.DataFrame(pred_table)
    var_table = out.get("var_table", pd.DataFrame())
    if isinstance(var_table, list):
        var_table = pd.DataFrame(var_table)

    if var_table.empty and beta.size > 0:
        var_table = pd.DataFrame({
            "j": np.arange(beta.size, dtype=int),
            "beta_mean": beta,
            "beta_sd": beta_sd,
            "support_score": score if score is not None else np.zeros(beta.size),
            "selected": np.isin(np.arange(beta.size), np.asarray(selected_support, dtype=int)).astype(int),
        })

    return {
        "method": method,
        "seed": out.get("seed"),
        "sim_info": out.get("sim_info", {}),
        "splits": out.get("splits", {}),
        "runtime_sec": out.get("runtime_sec"),
        "converged": out.get("converged", True),
        "n_iter": out.get("n_iter", None),
        "support_threshold": benchmark_support_threshold,
        "beta_eps": out.get("beta_eps", None),
        "intercept": out.get("intercept", 0.0),
        "sigma2": out.get("sigma2", None),
        "selected_support": list(selected_support),
        "beta_est": beta,
        "beta_sd": beta_sd,
        "support_score": score,
        "pip": out.get("pip", score),
        "predictive_metrics": out.get("predictive_metrics", {}),
        "selection_metrics": out.get("selection_metrics", {}),
        "pred_table": pred_table,
        "var_table": var_table,
        "history": out.get("history", pd.DataFrame()),
        "config": out.get("config", {}),
        "raw": out,
    }



def run_mcmc_placeholder(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    raise NotImplementedError(
        "MCMC slot is intentionally reserved in the benchmark registry. "
        "Plug your collaborator's runner here later, then wrap it with adapt_existing_spike_slab_output."
    )


# -----------------------------------------------------------------------------
# Benchmark orchestration
# -----------------------------------------------------------------------------


METHOD_REGISTRY = {
    "mf_spike_slab": run_mf_spike_slab,
    "mf_ard": run_mf_ard,
    "mf_bayes_lasso": run_mf_bayes_lasso,
    "mcmc_spike_slab": run_mcmc_placeholder,
}



def run_baseline_method(
    *,
    method: str,
    X: Array,
    y: Array,
    beta_true: Optional[Array],
    active_idx: Optional[Array],
    seed: int,
    sim_info: Mapping[str, Any],
    splits: Mapping[str, Array],
    method_cfg: Optional[Any] = None,
) -> Dict[str, Any]:
    if method not in METHOD_REGISTRY:
        raise KeyError(f"Unknown method '{method}'. Available: {sorted(METHOD_REGISTRY.keys())}")
    runner = METHOD_REGISTRY[method]
    return runner(
        X=X,
        y=y,
        beta_true=beta_true,
        active_idx=active_idx,
        seed=seed,
        sim_info=sim_info,
        splits=splits,
        cfg=method_cfg,
    )



def run_one_setting_one_seed(
    *,
    simfun: Callable[..., Any],
    seed: int,
    n: int,
    p: int,
    snr: float,
    true_prop: float,
    methods: Sequence[str],
    split_cfg: SplitConfig,
    method_cfgs: Optional[Mapping[str, Any]] = None,
    sim_kwargs: Optional[Mapping[str, Any]] = None,
    external_runners: Optional[Mapping[str, Callable[..., Mapping[str, Any]]]] = None,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    sim_kwargs = dict(sim_kwargs or {})
    sim_payload = simfun(seed=seed, n=n, p=p, snr=snr, true_prop=true_prop, **sim_kwargs)
    sim = extract_sim_arrays(sim_payload)
    X = sim["X"]
    y = sim["y"]
    beta_true = sim.get("beta_true")
    active_idx = sim.get("active_idx")
    sim_info = {"n": n, "p": p, "snr": snr, "true_prop": true_prop}
    sim_info.update({k: v for k, v in sim.get("sim_info", {}).items() if k not in {"X", "y", "beta_true"}})
    if active_idx is not None:
        sim_info.setdefault("n_active", int(len(active_idx)))

    splits = make_splits(X.shape[0], split_cfg)
    method_cfgs = dict(method_cfgs or {})
    external_runners = dict(external_runners or {})

    results: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    for method in methods:
        if method in external_runners:
            out = external_runners[method](
                X=X,
                y=y,
                beta_true=beta_true,
                active_idx=active_idx,
                seed=seed,
                sim_info=sim_info,
                splits=splits,
                method_cfg=method_cfgs.get(method),
            )
            result = adapt_existing_spike_slab_output(out, method=method)
        else:
            result = run_baseline_method(
                method=method,
                X=X,
                y=y,
                beta_true=beta_true,
                active_idx=active_idx,
                seed=seed,
                sim_info=sim_info,
                splits=splits,
                method_cfg=method_cfgs.get(method),
            )
        results.append(result)
        rows.append(benchmark_row_from_result(result))

    return results, pd.DataFrame(rows)



def run_setting_grid(
    *,
    simfun: Callable[..., Any],
    setting: Mapping[str, Any],
    seeds: Sequence[int],
    methods: Sequence[str],
    split_cfg: SplitConfig,
    method_cfgs: Optional[Mapping[str, Any]] = None,
    sim_kwargs: Optional[Mapping[str, Any]] = None,
    external_runners: Optional[Mapping[str, Callable[..., Mapping[str, Any]]]] = None,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    all_results: List[Dict[str, Any]] = []
    all_rows: List[pd.DataFrame] = []
    for seed in seeds:
        results, rows = run_one_setting_one_seed(
            simfun=simfun,
            seed=seed,
            n=int(setting["n"]),
            p=int(setting["p"]),
            snr=float(setting["snr"]),
            true_prop=float(setting["true_prop"]),
            methods=methods,
            split_cfg=split_cfg,
            method_cfgs=method_cfgs,
            sim_kwargs=sim_kwargs,
            external_runners=external_runners,
        )
        all_results.extend(results)
        all_rows.append(rows)
    return all_results, pd.concat(all_rows, axis=0, ignore_index=True) if all_rows else pd.DataFrame()



def run_full_benchmark(
    *,
    simfun: Callable[..., Any],
    setting_grid: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    methods: Sequence[str],
    split_cfg: SplitConfig,
    method_cfgs: Optional[Mapping[str, Any]] = None,
    sim_kwargs: Optional[Mapping[str, Any]] = None,
    external_runners: Optional[Mapping[str, Callable[..., Mapping[str, Any]]]] = None,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    all_results: List[Dict[str, Any]] = []
    all_rows: List[pd.DataFrame] = []
    for setting in setting_grid:
        results, rows = run_setting_grid(
            simfun=simfun,
            setting=setting,
            seeds=seeds,
            methods=methods,
            split_cfg=split_cfg,
            method_cfgs=method_cfgs,
            sim_kwargs=sim_kwargs,
            external_runners=external_runners,
        )
        all_results.extend(results)
        all_rows.append(rows)
    table = pd.concat(all_rows, axis=0, ignore_index=True) if all_rows else pd.DataFrame()
    return all_results, table


# -----------------------------------------------------------------------------
# Plotting helpers (benchmark-level, not method-internal diagnostics)
# -----------------------------------------------------------------------------



def plot_runtime_vs_f1(table: pd.DataFrame, output_path: Optional[str] = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for method, sub in table.groupby("method"):
        ax.scatter(sub["runtime_sec"], sub["f1"], label=method, alpha=0.75)
    ax.set_xlabel("runtime_sec")
    ax.set_ylabel("F1")
    ax.set_title("Runtime vs F1")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.2)
    if output_path:
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig



def plot_test_mse_vs_support_size(table: pd.DataFrame, output_path: Optional[str] = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for method, sub in table.groupby("method"):
        ax.scatter(sub["support_size"], sub["test_mse"], label=method, alpha=0.75)
    ax.set_xlabel("support_size")
    ax.set_ylabel("test_mse")
    ax.set_title("Support size vs test MSE")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.2)
    if output_path:
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig



def plot_precision_recall(table: pd.DataFrame, output_path: Optional[str] = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for method, sub in table.groupby("method"):
        ax.scatter(sub["recall"], sub["precision"], label=method, alpha=0.75)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Precision vs recall")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.2)
    if output_path:
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig



def plot_support_score_rank(result: Mapping[str, Any], output_path: Optional[str] = None) -> plt.Figure:
    score = np.asarray(result.get("support_score", []), dtype=float)
    beta_true = None
    if isinstance(result.get("var_table"), pd.DataFrame) and "truth" in result["var_table"].columns:
        score_df = result["var_table"].sort_values("support_score", ascending=False)
        truth = score_df["truth"].to_numpy(dtype=int)
        score = score_df["support_score"].to_numpy(dtype=float)
        beta_true = truth
    else:
        order = np.argsort(-score)
        score = score[order]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(np.arange(len(score)), score, linewidth=1.6)
    if beta_true is not None:
        hit_idx = np.flatnonzero(beta_true == 1)
        ax.scatter(hit_idx, score[hit_idx], marker="x", s=35, label="truth")
        ax.legend(frameon=False)
    ax.set_xlabel("rank")
    ax.set_ylabel("support_score / PIP")
    ax.set_title(f"Support score ranking: {result.get('method', '')}")
    ax.grid(True, alpha=0.2)
    if output_path:
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig


# -----------------------------------------------------------------------------
# Persistence helpers
# -----------------------------------------------------------------------------



def save_result_artifacts(result: Mapping[str, Any], save_cfg: SaveConfig) -> None:
    if not save_cfg.output_dir:
        return
    outdir = Path(save_cfg.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    method = str(result.get("method", "unknown"))
    seed = result.get("seed", "na")
    stem = f"{method}_seed{seed}"

    if save_cfg.save_history_csv and isinstance(result.get("history"), pd.DataFrame):
        result["history"].to_csv(outdir / f"{stem}_history.csv", index=False)
    if save_cfg.save_predictions_csv:
        yhat = np.asarray(result.get("yhat", []), dtype=float)
        if yhat.size > 0:
            pd.DataFrame({"yhat": yhat}).to_csv(outdir / f"{stem}_predictions.csv", index=False)
    if save_cfg.save_var_table_csv and isinstance(result.get("var_table"), pd.DataFrame):
        result["var_table"].to_csv(outdir / f"{stem}_var_table.csv", index=False)
    if save_cfg.save_final_json:
        json_payload = {k: v for k, v in result.items() if k not in {"history", "pred_table", "var_table", "raw"}}
        with open(outdir / f"{stem}_summary.json", "w", encoding="utf-8") as f:
            json.dump(json_payload, f, cls=NumpyJSONEncoder, ensure_ascii=False, indent=2)



def save_benchmark_table(table: pd.DataFrame, save_cfg: SaveConfig, filename: str = "benchmark_table.csv") -> None:
    if save_cfg.output_dir and save_cfg.save_benchmark_csv:
        outdir = Path(save_cfg.output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        table.to_csv(outdir / filename, index=False)


# -----------------------------------------------------------------------------
# Example wiring with an existing spike-and-slab runner
# -----------------------------------------------------------------------------


EXAMPLE_WIRING = r'''
# Example adapter: force the existing spike-and-slab method to consume shared data
# instead of regenerating its own data. This is the key refactor for fair benchmarking.

# def flow_runner_adapter(X, y, beta_true, active_idx, seed, sim_info, splits, method_cfg=None):
#     flow_out = fw.fit_flow_stagewise_on_data(
#         X=X,
#         y=y,
#         beta_true=beta_true,
#         train_idx=splits["train"],
#         val_idx=splits["val"],
#         test_idx=splits["test"],
#         seed=seed,
#         build_flow_vi=md.build_flow_vi,
#         family="gaussian",
#         schedule_cfg=schedule_cfg,
#         save_cfg=cfg.SaveConfig(output_dir=None),
#     )
#     return flow_out
#
# methods = ["flow_spike_slab", "mf_spike_slab", "mf_ard", "mf_bayes_lasso", "mcmc_spike_slab"]
# method_cfgs = {
#     "mf_spike_slab": MFSpikeSlabConfig(pi=0.1, slab_var=1.0, support_threshold=0.5, beta_eps=0.10),
#     "mf_ard": MFARDConfig(support_threshold=0.5, beta_eps=0.10),
#     "mf_bayes_lasso": MFBayesLassoConfig(lasso_lambda=1.0, support_threshold=0.5, beta_eps=0.10),
# }
# external_runners = {
#     "flow_spike_slab": flow_runner_adapter,
# }
#
# results, table = run_one_setting_one_seed(
#     simfun=sf.simfun1,
#     seed=123,
#     n=180,
#     p=100,
#     snr=3.0,
#     true_prop=0.1,
#     methods=methods[:-1],  # skip MCMC until collaborator plugs it in
#     split_cfg=SplitConfig(train_frac=0.6, val_frac=0.2, test_frac=0.2, seed=123),
#     method_cfgs=method_cfgs,
#     external_runners=external_runners,
# )
'''


if __name__ == "__main__":
    # Minimal smoke test on synthetic data.
    rng = np.random.default_rng(7)
    n, p, s = 80, 30, 5
    X = rng.normal(size=(n, p))
    beta_true = np.zeros(p)
    beta_true[:s] = np.array([2.0, -1.5, 1.2, 0.8, -0.7])
    y = X @ beta_true + rng.normal(scale=1.0, size=n)

    def simple_simfun(seed: int, n: int, p: int, snr: float, true_prop: float):
        return {
            "X": X,
            "y": y,
            "beta_true": beta_true,
            "active_idx": np.flatnonzero(beta_true != 0),
            "sigma2": 1.0,
        }

    results, table = run_one_setting_one_seed(
        simfun=simple_simfun,
        seed=7,
        n=n,
        p=p,
        snr=2.0,
        true_prop=s / p,
        methods=["mf_spike_slab", "mf_ard", "mf_bayes_lasso"],
        split_cfg=SplitConfig(train_frac=0.6, val_frac=0.2, test_frac=0.2, seed=7),
        method_cfgs={
            "mf_spike_slab": MFSpikeSlabConfig(max_iter=200, tol=1e-4, pi=s/p, support_threshold=0.5),
            "mf_ard": MFARDConfig(max_iter=200, tol=1e-4, support_threshold=0.5),
            "mf_bayes_lasso": MFBayesLassoConfig(max_iter=200, tol=1e-4, lasso_lambda=1.0, support_threshold=0.5),
        },
    )
    print(table)
