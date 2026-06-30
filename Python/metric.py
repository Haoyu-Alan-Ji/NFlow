from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.metrics import average_precision_score, roc_auc_score

from .utils import to_numpy

Array = np.ndarray


def _torch():
    import torch
    return torch


def _tensor(x):
    torch = _torch()
    if torch.is_tensor(x):
        return x.detach().cpu().float()
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


def _vec(x) -> Array:
    return np.asarray(to_numpy(x), dtype=float).reshape(-1)


def sample_posterior_latents(model, R: int = 2000) -> Dict[str, Any]:
    model.eval()
    with _torch().no_grad():
        sample = model.q0.rsample(R)
        z0 = sample[1] if isinstance(sample, tuple) else sample
        out = model.posterior_flow(z0, return_logdet=True)
        eps = out[0] if isinstance(out, tuple) else out
        dec = model.generative_model.decode(eps)
    keep = [
        "eps", "xi", "s", "u", "t", "margin", "gate", "active", "beta",
        "group_margin", "group_gate", "group_active", "group_ids",
    ]
    return {k: v.detach().cpu() for k, v in dec.items() if k in keep and hasattr(v, "detach")}


def hard_support_from_draws(draws: Mapping[str, Any], support_threshold: float = 0.5) -> Dict[str, Any]:
    torch = _torch()
    beta = _tensor(draws["beta"])
    active = _tensor(draws.get("active", (beta.abs() > 1e-12).float()))
    pip = active.mean(dim=0)
    mask = pip > float(support_threshold)
    idx = torch.where(mask)[0].cpu().numpy().astype(int).tolist()
    beta_hard = beta * active
    return {
        "support_idx": idx,
        "support_mask": mask.cpu(),
        "support_size": len(idx),
        "vote_rate": pip.cpu(),
        "support_score": pip.cpu(),
        "beta_hard_samples": beta_hard.cpu(),
        "beta_hard_mean": beta_hard.mean(dim=0).cpu(),
        "boundary": _tensor(draws.get("margin", beta)).cpu(),
    }


def selection_metrics_from_support(
    support_idx: Sequence[int],
    beta_true=None,
    active_idx: Optional[Sequence[int]] = None,
    p: Optional[int] = None,
    eps: float = 1e-12,
) -> Dict[str, float]:
    if beta_true is not None:
        truth = np.abs(_vec(beta_true)) > eps
        p = len(truth)
        true_set = set(np.flatnonzero(truth).astype(int))
    else:
        true_set = set(np.asarray(active_idx, dtype=int).tolist())
        p = int(p)
    selected = set(int(j) for j in support_idx)
    tp = len(selected & true_set)
    fp = len(selected - true_set)
    fn = len(true_set - selected)
    tn = int(p) - tp - fp - fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fdr = fp / (tp + fp) if tp + fp else 0.0
    return {
        "precision": float(precision), "recall": float(recall), "f1": float(f1), "fdr": float(fdr),
        "tp": float(tp), "fp": float(fp), "fn": float(fn), "tn": float(tn),
        "support_size": float(len(selected)),
    }


def ranking_metrics(*, support_score, beta_true=None, active_idx=None, p=None) -> Dict[str, float]:
    score = np.asarray(support_score, dtype=float)
    if beta_true is not None:
        truth = (np.abs(np.asarray(beta_true, dtype=float)) > 1e-12).astype(int)
    else:
        truth = np.zeros(int(p or len(score)), dtype=int)
        if active_idx is not None:
            truth[np.asarray(active_idx, dtype=int)] = 1
    if len(np.unique(truth)) < 2:
        return {"auroc": np.nan, "auprc": np.nan}
    return {
        "auroc": float(roc_auc_score(truth, score)),
        "auprc": float(average_precision_score(truth, score)),
    }


def predictive_metrics(X, y, beta_hard_samples, sigma2: Optional[float] = None, family: str = "gaussian") -> Dict[str, float]:
    torch = _torch()
    X = _tensor(X)
    y = _tensor(y).view(-1)
    B = _tensor(beta_hard_samples)
    eta = X @ B.T
    if family == "gaussian":
        pred = eta
    elif family == "poisson":
        pred = torch.exp(eta)
    elif family in {"bernoulli", "binomial", "logistic"}:
        pred = torch.sigmoid(eta)
    else:
        raise ValueError(f"Unknown family: {family}")
    yhat = pred.mean(dim=1)
    resid = y - yhat
    mse = resid.pow(2).mean().item()
    out = {
        "mse": float(mse),
        "rmse": float(mse ** 0.5),
        "mae": float(resid.abs().mean().item()),
        "r2": float(1.0 - resid.pow(2).sum().item() / ((y - y.mean()).pow(2).sum().item() + 1e-12)),
    }
    if family == "gaussian" and sigma2 is not None:
        s2 = max(float(sigma2), 1e-12)
        ll = -0.5 * (((y[:, None] - eta) ** 2) / s2 + math.log(2.0 * math.pi * s2))
        out["heldout_loglik"] = torch.logsumexp(ll, dim=1).sub(math.log(eta.shape[1])).mean().item()
        out["nll"] = -out["heldout_loglik"]
    return out


def _skl_grid(p, q, eps=1e-12):
    p = np.maximum(np.asarray(p, dtype=float), eps)
    q = np.maximum(np.asarray(q, dtype=float), eps)
    p /= p.sum(); q /= q.sum()
    return float(0.5 * (np.sum(p * np.log(p / q)) + np.sum(q * np.log(q / p))))


def kde_skl_1d(x, y, n_grid=128):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    lo = min(np.quantile(x, 0.001), np.quantile(y, 0.001))
    hi = max(np.quantile(x, 0.999), np.quantile(y, 0.999))
    pad = 0.1 * (hi - lo + 1e-8)
    grid = np.linspace(lo - pad, hi + pad, n_grid)
    return _skl_grid(gaussian_kde(x)(grid), gaussian_kde(y)(grid))


def kde_skl_2d(X, Y, n_grid=35):
    X = np.asarray(X, dtype=float); Y = np.asarray(Y, dtype=float)
    xlo = min(np.quantile(X[:, 0], 0.001), np.quantile(Y[:, 0], 0.001))
    xhi = max(np.quantile(X[:, 0], 0.999), np.quantile(Y[:, 0], 0.999))
    ylo = min(np.quantile(X[:, 1], 0.001), np.quantile(Y[:, 1], 0.001))
    yhi = max(np.quantile(X[:, 1], 0.999), np.quantile(Y[:, 1], 0.999))
    gx = np.linspace(xlo - 0.1 * (xhi - xlo + 1e-8), xhi + 0.1 * (xhi - xlo + 1e-8), n_grid)
    gy = np.linspace(ylo - 0.1 * (yhi - ylo + 1e-8), yhi + 0.1 * (yhi - ylo + 1e-8), n_grid)
    xx, yy = np.meshgrid(gx, gy)
    pts = np.vstack([xx.ravel(), yy.ravel()])
    return _skl_grid(gaussian_kde(X.T)(pts), gaussian_kde(Y.T)(pts))


def bernoulli_js(p, q, eps=1e-12):
    p = float(np.clip(p, eps, 1.0 - eps))
    q = float(np.clip(q, eps, 1.0 - eps))
    P = np.array([1 - p, p]); Q = np.array([1 - q, q]); M = 0.5 * (P + Q)
    return float(0.5 * np.sum(P * np.log(P / M)) + 0.5 * np.sum(Q * np.log(Q / M)))


def _active_pairs(beta_true, max_pairs=10):
    idx = np.flatnonzero(np.abs(beta_true) > 1e-12)
    idx = idx[np.argsort(-np.abs(beta_true[idx]))]
    pairs = []
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            pairs.append((int(idx[a]), int(idx[b])))
    return pairs[:max_pairs]


def recovery_metrics(beta_last, active_last, beta_true, mcmc_ref, max_pairs: int = 10) -> Dict[str, float]:
    beta_last = np.asarray(to_numpy(beta_last), dtype=float)
    active_last = np.asarray(to_numpy(active_last), dtype=float)
    beta_ref = np.asarray(mcmc_ref["beta"], dtype=float)
    pip_ref = np.asarray(mcmc_ref["pip"], dtype=float)
    beta_true = _vec(beta_true)
    active_idx = np.flatnonzero(np.abs(beta_true) > 1e-12)
    zero_idx = np.flatnonzero(np.abs(beta_true) <= 1e-12)
    pip = active_last.mean(axis=0)

    active_skl = [kde_skl_1d(beta_last[:, j], beta_ref[:, j]) for j in active_idx]
    joint_skl = [kde_skl_2d(beta_last[:, [j, k]], beta_ref[:, [j, k]]) for j, k in _active_pairs(beta_true, max_pairs)]
    zero_js = [bernoulli_js(pip[j], pip_ref[j]) for j in zero_idx]
    pip_js = [bernoulli_js(pip[j], pip_ref[j]) for j in active_idx]
    all_pip_js = [bernoulli_js(pip[j], pip_ref[j]) for j in range(len(pip))]

    out = {
        "joint_skl_median": float(np.nanmedian(joint_skl)) if joint_skl else np.nan,
        "joint_skl_mean": float(np.nanmean(joint_skl)) if joint_skl else np.nan,
        "active_marg_skl_median": float(np.nanmedian(active_skl)) if active_skl else np.nan,
        "active_marg_skl_mean": float(np.nanmean(active_skl)) if active_skl else np.nan,
        "zero_js_median": float(np.nanmedian(zero_js)) if zero_js else np.nan,
        "zero_js_mean": float(np.nanmean(zero_js)) if zero_js else np.nan,
        "pip_js_median": float(np.nanmedian(pip_js)) if pip_js else np.nan,
        "pip_js_mean": float(np.nanmean(pip_js)) if pip_js else np.nan,
        "all_pip_js_median": float(np.nanmedian(all_pip_js)),
        "pip_absdiff_median": float(np.nanmedian(np.abs(pip - pip_ref))),
        "pip_absdiff_mean": float(np.nanmean(np.abs(pip - pip_ref))),
        "expected_support": float(pip.sum()),
    }
    out.update({
        "active_joint_skl_median": out["joint_skl_median"],
        "active_joint_skl_mean": out["joint_skl_mean"],
    })
    return out


def flow_row_from_result(out_flow: Mapping[str, Any]) -> Dict[str, Any]:
    final = out_flow.get("final", {}) or {}
    sim_info = out_flow.get("sim_info", {}) or {}
    model_config = out_flow.get("model_config", {}) or {}
    row = {
        "method": out_flow.get("method"),
        "seed": out_flow.get("seed"),
        "runtime_sec": out_flow.get("runtime_sec"),
        "selected_ckpt_id": out_flow.get("selected_ckpt_id"),
        "mcmc_available": out_flow.get("mcmc_info", {}).get("mcmc_available"),
        "coupling_type": model_config.get("coupling_type"),
        "conditioner_type": model_config.get("conditioner_type"),
        "beta_mode": model_config.get("beta_mode"),
        "K_q": model_config.get("K_q"),
        "K_g": model_config.get("K_g"),
        "reported_layers": model_config.get("reported_layers"),
    }
    for k in ["joint_skl_median", "active_marg_skl_median", "zero_js_median", "pip_js_median", "pip_absdiff_mean"]:
        row[k] = final.get("recovery_metrics", {}).get(k, np.nan)
    for k, v in (final.get("selection_metrics", {}) or {}).items():
        row[k] = v
    for split in ["train", "val", "test"]:
        for k, v in (final.get(f"{split}_metrics", {}) or {}).items():
            row[f"{split}_{k}"] = v
    for k in ["setting", "n", "p", "n_active", "sigma2", "sigma", "rho", "beta_low", "beta_high"]:
        if k in sim_info:
            row[k] = sim_info[k]
    return row


def print_result(out: Mapping[str, Any], *, top_k: int = 20) -> None:
    final = out.get("final", {}) or {}
    print(f"===== {out.get('method', 'flow')} result =====")
    print(f"seed          : {out.get('seed')}")
    print(f"selected_ckpt : {out.get('selected_ckpt_id')}")
    print(f"mcmc_available: {out.get('mcmc_info', {}).get('mcmc_available')}")
    rec = final.get("recovery_metrics", {}) or {}
    if rec:
        print("\n===== Posterior recovery =====")
        cols = ["joint_skl_median", "active_marg_skl_median", "zero_js_median", "pip_js_median", "pip_absdiff_mean"]
        print(pd.DataFrame([{k: rec.get(k) for k in cols}]).to_string(index=False))
    vt = final.get("var_table")
    if isinstance(vt, pd.DataFrame) and not vt.empty:
        print(f"\n===== Top {top_k} variables by PIP =====")
        print(vt.sort_values("pip", ascending=False).head(top_k).to_string(index=False))


def summarize_ci(table, metrics=None, group_cols=None, level: float = 0.95):
    """
    Summarize per-seed recovery metrics across repeated runs.

    Input is usually a concatenated summary_row.csv table.  The returned table
    contains mean, sd, n, se, and a two-sided confidence interval for each metric.
    """
    from scipy.stats import t as student_t

    df = pd.DataFrame(table).copy()
    if metrics is None:
        metrics = [
            "joint_skl_median",
            "active_marg_skl_median",
            "zero_js_median",
            "pip_js_median",
        ]
    group_cols = list(group_cols or [])
    rows = []

    grouped = [((), df)] if not group_cols else df.groupby(group_cols, dropna=False)
    for key, g in grouped:
        if group_cols and not isinstance(key, tuple):
            key = (key,)
        prefix = dict(zip(group_cols, key)) if group_cols else {}
        for m in metrics:
            if m not in g.columns:
                continue
            x = pd.to_numeric(g[m], errors="coerce").dropna().to_numpy(dtype=float)
            n = int(len(x))
            mean = float(np.mean(x)) if n else np.nan
            sd = float(np.std(x, ddof=1)) if n > 1 else np.nan
            se = float(sd / np.sqrt(n)) if n > 1 else np.nan
            q = float(student_t.ppf(0.5 + level / 2.0, df=n - 1)) if n > 1 else np.nan
            rows.append({
                **prefix,
                "metric": m,
                "n": n,
                "mean": mean,
                "sd": sd,
                "se": se,
                "ci_level": float(level),
                "ci_lower": mean - q * se if n > 1 else np.nan,
                "ci_upper": mean + q * se if n > 1 else np.nan,
            })
    return pd.DataFrame(rows)

