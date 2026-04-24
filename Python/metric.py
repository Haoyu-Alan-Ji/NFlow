from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .utils import Array, to_numpy


def _torch():
    import torch
    return torch


def _to_torch_float_cpu(x):
    torch = _torch()
    if torch.is_tensor(x):
        return x.detach().cpu().float()
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


def _to_numpy_1d(x) -> Array:
    arr = np.asarray(to_numpy(x))
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


def sample_posterior_latents(model, R: int = 2000) -> Dict[str, Any]:
    model.eval()
    sample = model.q0.rsample(R)
    if isinstance(sample, tuple):
        _, z0 = sample
    else:
        z0 = sample

    flow_out = model.posterior_flow(z0, return_logdet=True)
    if isinstance(flow_out, tuple):
        eps = flow_out[0]
    else:
        eps = flow_out

    dec = model.generative_model.decode(eps)
    required = ["s", "u", "t", "beta"]
    missing = [k for k in required if k not in dec]
    if missing:
        raise KeyError(f"Decoded posterior is missing keys: {missing}")
    return {k: dec[k].detach().cpu() for k in required}


def hard_support_from_draws(draws: Dict[str, Any], support_threshold: float = 0.5) -> Dict[str, Any]:
    torch = _torch()
    s = _to_torch_float_cpu(draws["s"])
    u = _to_torch_float_cpu(draws["u"])
    t = _to_torch_float_cpu(draws["t"])
    if t.ndim == 1:
        t = t.unsqueeze(1)
    ind = (u > t).float()
    vote_rate = ind.mean(dim=0)
    support_mask = vote_rate > support_threshold
    support_idx = torch.where(support_mask)[0].cpu().numpy().astype(int).tolist()
    beta_hard_samples = s * ind
    beta_hard_mean = beta_hard_samples.mean(dim=0)
    boundary = (u - t).float()
    return {
        "support_idx": support_idx,
        "support_mask": support_mask.cpu(),
        "support_size": len(support_idx),
        "vote_rate": vote_rate.cpu(),
        "support_score": vote_rate.cpu(),
        "beta_hard_samples": beta_hard_samples.cpu(),
        "beta_hard_mean": beta_hard_mean.cpu(),
        "boundary": boundary.cpu(),
    }


def posterior_predictions_from_hard_samples(X, beta_hard_samples, family: str = "gaussian"):
    torch = _torch()
    X = _to_torch_float_cpu(X)
    beta_hard_samples = _to_torch_float_cpu(beta_hard_samples)
    eta = X @ beta_hard_samples.T
    if family == "gaussian":
        return eta
    if family == "poisson":
        return torch.exp(eta)
    raise ValueError(f"Unknown family: {family}")


def selection_metrics_from_support(
    support_idx: Sequence[int],
    beta_true=None,
    active_idx: Optional[Sequence[int]] = None,
    p: Optional[int] = None,
    eps: float = 1e-12,
) -> Dict[str, float]:
    if beta_true is not None:
        beta_true_np = _to_numpy_1d(beta_true)
        truth = np.abs(beta_true_np) > eps
        p = int(beta_true_np.shape[0])
        true_support = set(np.flatnonzero(truth).astype(int).tolist())
    elif active_idx is not None:
        if p is None:
            raise ValueError("p must be provided when using active_idx without beta_true.")
        true_support = set(np.asarray(active_idx, dtype=int).tolist())
    else:
        raise ValueError("Either beta_true or active_idx must be provided.")

    selected = set(int(j) for j in support_idx)
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
        "support_size": float(len(selected)),
    }


def support_metrics(
    selected_support: Sequence[int],
    beta_true=None,
    active_idx: Optional[Sequence[int]] = None,
    p: Optional[int] = None,
    eps: float = 1e-12,
) -> Dict[str, float]:
    if beta_true is None and active_idx is None:
        return {}
    return selection_metrics_from_support(selected_support, beta_true=beta_true, active_idx=active_idx, p=p, eps=eps)


def gaussian_predictive_metrics(y_true, y_pred, sigma2: Optional[float]) -> Dict[str, float]:
    y_true = np.asarray(to_numpy(y_true), dtype=float).reshape(-1)
    y_pred = np.asarray(to_numpy(y_pred), dtype=float).reshape(-1)
    resid = y_true - y_pred
    mse = float(np.mean(resid ** 2))
    rmse = float(np.sqrt(max(mse, 0.0)))
    mae = float(np.mean(np.abs(resid)))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - np.sum(resid ** 2) / denom) if denom > 0 else float("nan")
    out = {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}
    if sigma2 is not None:
        sigma2 = float(max(float(sigma2), 1e-12))
        loglik = float(np.mean(-0.5 * (np.log(2.0 * np.pi * sigma2) + resid ** 2 / sigma2)))
        out["heldout_loglik"] = loglik
        out["nll"] = -loglik
    else:
        out["heldout_loglik"] = float("nan")
        out["nll"] = float("nan")
    return out


def posterior_predictive_metrics_from_hard_samples(X, y, beta_hard_samples, sigma2: Optional[float] = None, family: str = "gaussian") -> Dict[str, float]:
    torch = _torch()
    X = _to_torch_float_cpu(X)
    y = _to_torch_float_cpu(y).view(-1)
    beta_hard_samples = _to_torch_float_cpu(beta_hard_samples)
    eta = X @ beta_hard_samples.T
    if family == "gaussian":
        pred_draws = eta
    elif family == "poisson":
        pred_draws = torch.exp(eta)
    else:
        raise ValueError(f"Unknown family: {family}")
    yhat = pred_draws.mean(dim=1)
    resid = y - yhat
    mse = resid.pow(2).mean().item()
    rmse = mse ** 0.5
    mae = resid.abs().mean().item()
    ss_res = resid.pow(2).sum().item()
    ss_tot = ((y - y.mean()) ** 2).sum().item()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    out = {"mse": float(mse), "rmse": float(rmse), "mae": float(mae), "r2": float(r2)}
    if family == "gaussian":
        if sigma2 is not None:
            sigma2 = float(max(float(sigma2), 1e-12))
            ll = -0.5 * (((y[:, None] - pred_draws) ** 2) / sigma2 + math.log(2.0 * math.pi * sigma2))
            out["heldout_loglik"] = torch.logsumexp(ll, dim=1).sub(math.log(pred_draws.shape[1])).mean().item()
            out["nll"] = -out["heldout_loglik"]
        else:
            out["heldout_loglik"] = float("nan")
            out["nll"] = float("nan")
    else:
        mu = pred_draws.clamp_min(1e-8)
        ll = y[:, None] * torch.log(mu) - mu - torch.lgamma(y[:, None] + 1.0)
        out["heldout_loglik"] = torch.logsumexp(ll, dim=1).sub(math.log(mu.shape[1])).mean().item()
        out["nll"] = -out["heldout_loglik"]
        y_safe = y.clamp_min(1e-8)
        yhat_safe = yhat.clamp_min(1e-8)
        dev = 2.0 * torch.where(y > 0, y * torch.log(y_safe / yhat_safe) - (y - yhat_safe), -(y - yhat_safe))
        out["poisson_deviance"] = dev.mean().item()
    return out


def predictive_metrics(X, y, beta_hard_samples, sigma2: Optional[float] = None, family: str = "gaussian"):
    return posterior_predictive_metrics_from_hard_samples(X, y, beta_hard_samples, sigma2=sigma2, family=family)


def normal_cdf(x):
    x = np.asarray(x, dtype=float)
    erf_vec = np.vectorize(math.erf, otypes=[float])
    return 0.5 * (1.0 + erf_vec(x / math.sqrt(2.0)))


def prob_abs_gt_eps(mu, sd, eps: float):
    mu = np.asarray(mu, dtype=float)
    sd = np.asarray(sd, dtype=float)
    sd_safe = np.maximum(sd, 1e-12)
    upper = 1.0 - normal_cdf((eps - mu) / sd_safe)
    lower = normal_cdf((-eps - mu) / sd_safe)
    return np.clip(upper + lower, 0.0, 1.0)
