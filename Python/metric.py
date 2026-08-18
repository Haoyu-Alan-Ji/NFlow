from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from .utils import to_numpy
import matplotlib.pyplot as plt

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
        "eps", "xi", "s", "u", "t", "margin", "gate", "active", "beta", "beta_hard",
        "group_margin", "group_gate", "group_active", "group_ids",
    ]
    return {k: v.detach().cpu() for k, v in dec.items() if k in keep and hasattr(v, "detach")}


def hard_support_from_draws(draws: Mapping[str, Any], support_threshold: float = 0.5) -> Dict[str, Any]:
    torch = _torch()
    beta = _tensor(draws["beta"])
    active = _tensor(draws.get("active", (beta.abs() > 1e-12).float()))
    if "beta_hard" in draws:
        beta_hard = _tensor(draws["beta_hard"])
    elif "s" in draws:
        beta_hard = _tensor(draws["s"]) * active
    else:
        raise KeyError("Hard posterior draws require beta_hard or the latent slab s.")
    pip = active.mean(dim=0)
    mask = pip > float(support_threshold)
    idx = torch.where(mask)[0].cpu().numpy().astype(int).tolist()
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

    prob = np.clip(score, 0.0, 1.0)

    out = {
        "br": float(np.mean((truth - prob) ** 2)),
    }

    if len(np.unique(truth)) < 2:
        out["auroc"] = np.nan
        out["auprc"] = np.nan
    else:
        out["auroc"] = float(roc_auc_score(truth, score))
        out["auprc"] = float(average_precision_score(truth, score))

    return out


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
        "normalized_l1": float(resid.abs().sum().item() / (y.abs().sum().item() + 1e-12)),
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


def recovery_metrics(beta_last, active_last, beta_true, mcmc_ref, max_pairs: int = 10) -> Dict[str, Any]:
    beta_last = np.asarray(to_numpy(beta_last), dtype=float)
    active_last = np.asarray(to_numpy(active_last), dtype=float)
    beta_ref = np.asarray(mcmc_ref["beta"], dtype=float)
    pip_ref = np.asarray(mcmc_ref["pip"], dtype=float)
    beta_true = _vec(beta_true)

    active_idx = np.flatnonzero(np.abs(beta_true) > 1e-12)
    zero_idx = np.flatnonzero(np.abs(beta_true) <= 1e-12)
    truth = (np.abs(beta_true) > 1e-12).astype(int)

    pip = active_last.mean(axis=0)
    pip_diff = pip - pip_ref
    pip_absdiff = np.abs(pip_diff)

    active_skl = []
    for j in active_idx:
        try:
            value = kde_skl_1d(
                beta_last[:, j],
                beta_ref[:, j],
            )
        except (ValueError, np.linalg.LinAlgError):
            value = np.nan
        active_skl.append(value)

    joint_skl = []
    for j, k in _active_pairs(beta_true, max_pairs):
        try:
            value = kde_skl_2d(
                beta_last[:, [j, k]],
                beta_ref[:, [j, k]],
            )
        except (ValueError, np.linalg.LinAlgError):
            value = np.nan
        joint_skl.append(value)

    active_skl_valid = np.asarray(active_skl, dtype=float)
    active_skl_valid = active_skl_valid[np.isfinite(active_skl_valid)]
    joint_skl_valid = np.asarray(joint_skl, dtype=float)
    joint_skl_valid = joint_skl_valid[np.isfinite(joint_skl_valid)]
    active_skl_complete = active_skl_valid.size == len(active_skl)

    zero_js = [bernoulli_js(pip[j], pip_ref[j]) for j in zero_idx]
    pip_js = [bernoulli_js(pip[j], pip_ref[j]) for j in active_idx]
    all_pip_js = [bernoulli_js(pip[j], pip_ref[j]) for j in range(len(pip))]

    br_last = float(np.mean((truth - pip) ** 2))
    br_mcmc = float(np.mean((truth - pip_ref) ** 2))
    br_rel = float(br_last / (br_mcmc + 1e-12))

    if len(np.unique(truth)) < 2:
        auroc_last = np.nan
        auprc_last = np.nan
        auroc_mcmc = np.nan
        auprc_mcmc = np.nan
    else:
        auroc_last = float(roc_auc_score(truth, pip))
        auprc_last = float(average_precision_score(truth, pip))
        auroc_mcmc = float(roc_auc_score(truth, pip_ref))
        auprc_mcmc = float(average_precision_score(truth, pip_ref))

    out = {
        "active_marg_skl_n_total": int(len(active_skl)),
        "active_marg_skl_n_valid": int(active_skl_valid.size),
        "active_marg_skl_complete": bool(active_skl_complete),
        "active_joint_skl_n_total": int(len(joint_skl)),
        "active_joint_skl_n_valid": int(joint_skl_valid.size),
        "active_joint_skl_complete": bool(joint_skl_valid.size == len(joint_skl)),

        "joint_skl_median": float(np.median(joint_skl_valid)) if joint_skl_valid.size else np.nan,
        "joint_skl_mean": float(np.mean(joint_skl_valid)) if joint_skl_valid.size else np.nan,

        "active_marg_skl_median": float(np.median(active_skl_valid)) if active_skl_complete else np.nan,
        "active_marg_skl_mean": float(np.mean(active_skl_valid)) if active_skl_complete else np.nan,
        "active_marg_skl_partial_median": float(np.median(active_skl_valid)) if active_skl_valid.size else np.nan,
        "active_marg_skl_partial_mean": float(np.mean(active_skl_valid)) if active_skl_valid.size else np.nan,

        "zero_js_median": float(np.nanmedian(zero_js)) if zero_js else np.nan,
        "zero_js_mean": float(np.nanmean(zero_js)) if zero_js else np.nan,

        "pip_js_median": float(np.nanmedian(pip_js)) if pip_js else np.nan,
        "pip_js_mean": float(np.nanmean(pip_js)) if pip_js else np.nan,
        "all_pip_js_median": float(np.nanmedian(all_pip_js)),
        "all_pip_js_mean": float(np.nanmean(all_pip_js)),

        "pip_l1_sum": float(np.sum(pip_absdiff)),
        "pip_l1_mean": float(np.mean(pip_absdiff)),
        "pip_rmse": float(np.sqrt(np.mean(pip_diff ** 2))),

        "pip_l1_active_mean": float(np.mean(pip_absdiff[active_idx])) if len(active_idx) else np.nan,
        "pip_rmse_active": float(np.sqrt(np.mean(pip_diff[active_idx] ** 2))) if len(active_idx) else np.nan,

        "pip_l1_zero_mean": float(np.mean(pip_absdiff[zero_idx])) if len(zero_idx) else np.nan,
        "pip_rmse_zero": float(np.sqrt(np.mean(pip_diff[zero_idx] ** 2))) if len(zero_idx) else np.nan,

        "pip_absdiff_median": float(np.nanmedian(pip_absdiff)),
        "pip_absdiff_mean": float(np.nanmean(pip_absdiff)),

        "br_last": br_last,
        "br_mcmc": br_mcmc,
        "br_rel": br_rel,

        "auroc_last": auroc_last,
        "auprc_last": auprc_last,
        "auroc_mcmc": auroc_mcmc,
        "auprc_mcmc": auprc_mcmc,

        "expected_support": float(pip.sum()),
        "mcmc_expected_support": float(pip_ref.sum()),
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
        "runtime_sec": out_flow.get("total_runtime_sec", out_flow.get("runtime_sec")),
        "train_runtime_sec": out_flow.get("train_runtime_sec", out_flow.get("runtime_sec")),
        "total_runtime_sec": out_flow.get("total_runtime_sec", out_flow.get("runtime_sec")),
        "selected_ckpt_id": out_flow.get("selected_ckpt_id"),
        "mcmc_available": out_flow.get("mcmc_info", {}).get("mcmc_available"),
        "coupling_type": model_config.get("coupling_type"),
        "conditioner_type": model_config.get("conditioner_type"),
        "beta_mode": model_config.get("beta_mode"),
        "K_q": model_config.get("K_q"),
        "K_g": model_config.get("K_g"),
        "K_flow": model_config.get("K_flow"),
        "reported_layers": model_config.get("reported_layers"),
        "total_coupling_transforms": model_config.get("total_coupling_transforms"),
    }
    for k in [
        "joint_skl_median",
        "active_marg_skl_median",
        "zero_js_median",
        "pip_js_median",
        "pip_rmse",
        "pip_absdiff_mean",
        "auroc_last",
        "auprc_last",
        "active_marg_skl_n_total",
        "active_marg_skl_n_valid",
        "active_marg_skl_complete",
        "active_marg_skl_partial_median",
        "active_marg_skl_partial_mean",
        "active_joint_skl_n_total",
        "active_joint_skl_n_valid",
        "active_joint_skl_complete",
    ]:
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
        cols = [
            "joint_skl_median",
            "active_marg_skl_median",
            "zero_js_median",
            "pip_rmse",
            "auroc_last",
            "auprc_last",
        ]
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


@torch.no_grad()
def posterior_draws(
    decoder,
    xi,
    sigmoid_active_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Decode edge parameters and gate-based structural indicators."""
    xi = _tensor(xi)
    params = decoder.unpack(xi)

    u = xi[:, decoder.s_dim:decoder.s_dim + decoder.u_dim]
    t = xi[:, decoder.s_dim + decoder.u_dim:]

    theta = {}
    active = {}
    gate = {}

    for item in decoder.param_specs:
        name = item["name"]
        sl = slice(item["start"], item["end"])
        margin = u[:, sl] - t[:, item["t"]:item["t"] + 1]

        theta[name] = params[name].detach().cpu()
        gate[name] = decoder.gate(name, margin).reshape(
            xi.shape[0], *item["shape"]
        ).detach().cpu()
        active[name] = decoder.active(
            name,
            margin,
            sigmoid_threshold=sigmoid_active_threshold,
        ).reshape(
            xi.shape[0], *item["shape"]
        ).detach().cpu()

    theta_flat = torch.cat([
        theta[item["name"]].reshape(xi.shape[0], -1)
        for item in decoder.param_specs
    ], dim=1)

    active_flat = torch.cat([
        active[item["name"]].reshape(xi.shape[0], -1)
        for item in decoder.param_specs
    ], dim=1)

    return {
        "theta": theta,
        "gate": gate,
        "active": active,
        "theta_flat": theta_flat,
        "active_flat": active_flat,
        "pip": active_flat.float().mean(dim=0),
    }


@torch.no_grad()
def predict_draws(decoder, X, xi, batch_size: int = 500):
    """Posterior latent-function draws with shape R x n."""
    if torch.is_tensor(X):
        X = X.detach()
    else:
        X = torch.as_tensor(X, dtype=torch.float32)

    if torch.is_tensor(xi):
        xi = xi.detach().to(device=X.device, dtype=X.dtype)
    else:
        xi = torch.as_tensor(xi, device=X.device, dtype=X.dtype)
    draws = []

    for start in range(0, xi.shape[0], batch_size):
        draws.append(
            decoder(X, xi[start:start + batch_size]).detach().cpu()
        )

    return torch.cat(draws, dim=0)


def _skl(x, y):
    try:
        return kde_skl_1d(
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
        )
    except (ValueError, np.linalg.LinAlgError):
        return np.nan


def posterior_metrics(
    last: Mapping[str, Any],
    mcmc: Mapping[str, Any],
    active_threshold: float = 0.5,
):
    """
    Active SKL uses connections with MCMC PIP > active_threshold.
    Zero JS uses the complementary MCMC-inactive connections.
    """
    pip_last = last["pip"].numpy()
    pip_mcmc = mcmc["pip"].numpy()

    active_idx = np.flatnonzero(pip_mcmc > active_threshold)
    zero_idx = np.flatnonzero(pip_mcmc <= active_threshold)

    theta_last = last["theta_flat"].numpy()
    theta_mcmc = mcmc["theta_flat"].numpy()

    a_skl = [
        _skl(theta_last[:, j], theta_mcmc[:, j])
        for j in active_idx
    ]

    zero_js = [
        bernoulli_js(1.0 - pip_last[j], 1.0 - pip_mcmc[j])
        for j in zero_idx
    ]

    summary = {
        "a_skl": float(np.nanmedian(a_skl)) if a_skl else np.nan,
        "a_skl_mean": float(np.nanmean(a_skl)) if a_skl else np.nan,
        "zero_js": float(np.nanmedian(zero_js)) if zero_js else np.nan,
        "zero_js_mean": float(np.nanmean(zero_js)) if zero_js else np.nan,
        "pip_rmse": float(np.sqrt(np.mean((pip_last - pip_mcmc) ** 2))),
        "n_mcmc_active": int(len(active_idx)),
        "n_mcmc_zero": int(len(zero_idx)),
    }

    rows = []
    offset = 0

    for name, theta_l in last["theta"].items():
        n_param = theta_l[0].numel()
        sl = slice(offset, offset + n_param)

        pip_l = pip_last[sl]
        pip_m = pip_mcmc[sl]

        theta_l = theta_l.reshape(theta_l.shape[0], -1).numpy()
        theta_m = mcmc["theta"][name].reshape(
            mcmc["theta"][name].shape[0], -1
        ).numpy()

        active = np.flatnonzero(pip_m > active_threshold)
        zero = np.flatnonzero(pip_m <= active_threshold)

        skl = [
            _skl(theta_l[:, j], theta_m[:, j])
            for j in active
        ]
        js = [
            bernoulli_js(1.0 - pip_l[j], 1.0 - pip_m[j])
            for j in zero
        ]

        rows.append({
            "parameter": name,
            "n_parameter": n_param,
            "n_mcmc_active": len(active),
            "a_skl": float(np.nanmedian(skl)) if skl else np.nan,
            "zero_js": float(np.nanmedian(js)) if js else np.nan,
            "pip_rmse": float(np.sqrt(np.mean((pip_l - pip_m) ** 2))),
        })

        offset += n_param

    return summary, pd.DataFrame(rows)


def _count_row(method, parameter, kind, block, count, n_total):
    count = count.float()

    return {
        "method": method,
        "parameter": parameter,
        "kind": kind,
        "block": block,
        "n_total": int(n_total),
        "expected_n": float(count.mean()),
        "median_n": float(count.median()),
        "p_zero": float((count == 0).float().mean()),
    }


def connection_counts(
    posterior: Mapping[str, Any],
    decoder,
    method: str,
):
    """E[N|y], median(N|y), and P(N=0|y) for every W/b tensor."""
    rows = []

    for item in decoder.param_specs:
        name = item["name"]
        z = posterior["active"][name].reshape(
            posterior["active"][name].shape[0], -1
        )

        kind = "W" if name == "E" or name.startswith("W") else "b"

        rows.append(
            _count_row(
                method=method,
                parameter=name,
                kind=kind,
                block=item["block"],
                count=z.sum(dim=1),
                n_total=z.shape[1],
            )
        )

    for kind in ["W", "b"]:
        names = [
            item["name"]
            for item in decoder.param_specs
            if (
                kind == "W"
                and (item["name"] == "E" or item["name"].startswith("W"))
            )
            or (
                kind == "b"
                and not (
                    item["name"] == "E"
                    or item["name"].startswith("W")
                )
            )
        ]

        z = torch.cat([
            posterior["active"][name].reshape(
                posterior["active"][name].shape[0], -1
            )
            for name in names
        ], dim=1)

        rows.append(
            _count_row(
                method=method,
                parameter=f"all_{kind}",
                kind=kind,
                block="all",
                count=z.sum(dim=1),
                n_total=z.shape[1],
            )
        )

    return pd.DataFrame(rows)


def hidden_unit_counts(
    posterior: Mapping[str, Any],
    decoder,
    method: str,
):
    """
    Effective FFN hidden units:
      W_only  : incoming W1, no b1, and outgoing W2.
      b_only  : no incoming W1, b1, and outgoing W2.
      W_and_b : incoming W1, b1, and outgoing W2.
    """
    rows = []

    for layer in decoder.layers_spec:
        k = layer["block"]

        W1 = posterior["active"][f"W1_{k}"]
        b1 = posterior["active"][f"b1_{k}"]
        W2 = posterior["active"][f"W2_{k}"]

        has_W = W1.any(dim=2)
        has_b = b1
        has_out = W2.any(dim=1)

        groups = {
            "W_only": has_W & ~has_b & has_out,
            "b_only": ~has_W & has_b & has_out,
            "W_and_b": has_W & has_b & has_out,
            "effective_total": (has_W | has_b) & has_out,
        }

        for source, z in groups.items():
            count = z.float().sum(dim=1)

            rows.append({
                "method": method,
                "block": k,
                "source": source,
                "n_total": int(z.shape[1]),
                "expected_n": float(count.mean()),
                "median_n": float(count.median()),
                "p_zero": float((count == 0).float().mean()),
            })

    return pd.DataFrame(rows)


def _signal_r2(signal, pred):
    signal = np.asarray(signal, dtype=float).reshape(-1)
    pred = np.asarray(pred, dtype=float).reshape(-1)

    sst = np.sum((signal - signal.mean()) ** 2)
    return float(1.0 - np.sum((signal - pred) ** 2) / (sst + 1e-12))


def _corr2(signal, pred):
    signal = np.asarray(signal, dtype=float).reshape(-1)
    pred = np.asarray(pred, dtype=float).reshape(-1)

    if np.std(signal) < 1e-12 or np.std(pred) < 1e-12:
        return np.nan

    return float(np.corrcoef(signal, pred)[0, 1] ** 2)


def function_metrics(
    signal,
    mcmc_pred_draws,
    last_pred_draws,
    zero_tol: float = 1e-6,
    constant_tol: float = 1e-6,
):
    """
    Function recovery and dead-network diagnostics.

    signal_r2 = 1 - SSE/SST.
    corr2 = Cor(signal, posterior mean)^2.
    """
    signal = _tensor(signal).reshape(-1)
    mcmc_pred_draws = _tensor(mcmc_pred_draws)
    last_pred_draws = _tensor(last_pred_draws)

    pred_mcmc = mcmc_pred_draws.mean(dim=0)
    pred_last = last_pred_draws.mean(dim=0)

    sst = (signal - signal.mean()).square().sum()

    mcmc_r2_draws = 1.0 - (
        (mcmc_pred_draws - signal[None, :]).square().sum(dim=1)
        / (sst + 1e-12)
    )

    last_r2_draws = 1.0 - (
        (last_pred_draws - signal[None, :]).square().sum(dim=1)
        / (sst + 1e-12)
    )

    def one(method, draws, pred, r2_draws):
        energy = draws.square().mean(dim=1)
        variation = draws.var(dim=1, unbiased=False)

        return {
            f"{method}_mse": float((pred - signal).square().mean()),
            f"{method}_signal_r2": _signal_r2(signal, pred),
            f"{method}_corr2": _corr2(signal, pred),
            f"{method}_r2_draw_median": float(r2_draws.median()),
            f"{method}_r2_positive_prob": float(
                (r2_draws > 0).float().mean()
            ),
            f"{method}_zero_function_prob": float(
                (energy < zero_tol).float().mean()
            ),
            f"{method}_constant_function_prob": float(
                (variation < constant_tol).float().mean()
            ),
        }

    out = {}
    out.update(one("mcmc", mcmc_pred_draws, pred_mcmc, mcmc_r2_draws))
    out.update(one("last", last_pred_draws, pred_last, last_r2_draws))

    return out


def bnn_metrics(
    mcmc_decoder,
    last_decoder,
    mcmc_xi,
    last_xi,
    X,
    signal,
    active_threshold: float = 0.5,
    sigmoid_active_threshold: float = 0.5,
    batch_size: int = 500,
):
    """Complete BNN metric bundle."""
    mcmc_specs = [
        (x["name"], tuple(x["shape"]))
        for x in mcmc_decoder.param_specs
    ]
    last_specs = [
        (x["name"], tuple(x["shape"]))
        for x in last_decoder.param_specs
    ]

    if mcmc_specs != last_specs:
        raise ValueError("MCMC and LaST decoders must have identical parameter specs.")

    mcmc_post = posterior_draws(
        mcmc_decoder,
        mcmc_xi,
        sigmoid_active_threshold=sigmoid_active_threshold,
    )
    last_post = posterior_draws(
        last_decoder,
        last_xi,
        sigmoid_active_threshold=sigmoid_active_threshold,
    )

    recovery, posterior_by_layer = posterior_metrics(
        last=last_post,
        mcmc=mcmc_post,
        active_threshold=active_threshold,
    )

    connection_table = pd.concat([
        connection_counts(mcmc_post, mcmc_decoder, "mcmc"),
        connection_counts(last_post, last_decoder, "last"),
    ], ignore_index=True)

    hidden_table = pd.concat([
        hidden_unit_counts(mcmc_post, mcmc_decoder, "mcmc"),
        hidden_unit_counts(last_post, last_decoder, "last"),
    ], ignore_index=True)

    mcmc_pred_draws = predict_draws(
        mcmc_decoder, X, mcmc_xi, batch_size=batch_size
    )
    last_pred_draws = predict_draws(
        last_decoder, X, last_xi, batch_size=batch_size
    )

    function = function_metrics(
        signal=signal,
        mcmc_pred_draws=mcmc_pred_draws,
        last_pred_draws=last_pred_draws,
    )

    return {
        "summary": {**recovery, **function},
        "posterior_by_layer": posterior_by_layer,
        "connection_counts": connection_table,
        "hidden_units": hidden_table,
        "mcmc_pred_draws": mcmc_pred_draws,
        "last_pred_draws": last_pred_draws,
    }


def plot_function_1d(
    x,
    signal,
    mcmc_pred_draws,
    last_pred_draws=None,
    rat_pred_draws=None,
    interval: float = 0.95,
    method_label: str = "RaT",
):
    """Truth, zero function, MCMC, and RaT on one 1D plot."""
    if last_pred_draws is None:
        last_pred_draws = rat_pred_draws

    x = np.asarray(x, dtype=float).reshape(-1)
    signal = np.asarray(signal, dtype=float).reshape(-1)
    mcmc = np.asarray(mcmc_pred_draws, dtype=float)
    last = np.asarray(last_pred_draws, dtype=float)

    order = np.argsort(x)
    x = x[order]
    signal = signal[order]
    mcmc = mcmc[:, order]
    last = last[:, order]

    alpha = 0.5 * (1.0 - interval)

    mcmc_mean = mcmc.mean(axis=0)
    last_mean = last.mean(axis=0)

    mcmc_lo, mcmc_hi = np.quantile(
        mcmc, [alpha, 1.0 - alpha], axis=0
    )
    last_lo, last_hi = np.quantile(
        last, [alpha, 1.0 - alpha], axis=0
    )

    fig, ax = plt.subplots()

    ax.plot(x, signal, label="True function")
    ax.axhline(0.0, linestyle="--", label="Zero function")
    ax.plot(x, mcmc_mean, label="MCMC mean")
    ax.plot(x, last_mean, label=f"{method_label} mean")

    ax.fill_between(x, mcmc_lo, mcmc_hi, alpha=0.15)
    ax.fill_between(x, last_lo, last_hi, alpha=0.15)

    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend()
    fig.tight_layout()

    return fig, ax


def plot_function_2d(
    X,
    signal,
    mcmc_pred_draws,
    last_pred_draws,
):
    """Four separate 3D surfaces with x1, x2, and f(x1,x2) axes."""
    X = np.asarray(X, dtype=float)
    signal = np.asarray(signal, dtype=float).reshape(-1)
    mcmc_mean = np.asarray(mcmc_pred_draws, dtype=float).mean(axis=0)
    last_mean = np.asarray(last_pred_draws, dtype=float).mean(axis=0)

    if X.shape[1] != 2:
        raise ValueError("plot_function_2d requires X with exactly two columns.")

    surfaces = {
        "truth": ("True function", signal),
        "zero": ("Zero function", np.zeros_like(signal)),
        "mcmc": ("MCMC mean", mcmc_mean),
        "last": ("LaST mean", last_mean),
    }

    figures = {}

    for key, (title, z) in surfaces.items():
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.plot_trisurf(X[:, 0], X[:, 1], z)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("f(x1, x2)")
        ax.set_title(title)
        fig.tight_layout()

        figures[key] = (fig, ax)

    return figures


@torch.no_grad()
def sorted_unit_draws(decoder, xi, breakpoint_eps=1e-4):
    """Order each direct-model draw by b_j / w_j; inactive w_j go last."""

    params, semantics = decoder.unpack(xi, return_semantics=True)
    valid = params["w"].abs() > float(breakpoint_eps)
    breakpoint = torch.where(
        valid,
        params["b"] / params["w"],
        torch.full_like(params["w"], torch.inf),
    )
    unit_id = torch.arange(
        decoder.H,
        device=xi.device,
        dtype=xi.dtype,
    )[None, :]
    sort_key = torch.where(
        valid,
        breakpoint,
        torch.full_like(breakpoint, 1e12) + unit_id,
    )
    order = torch.argsort(sort_key + unit_id * 1e-10, dim=1)

    out = {
        "beta0": params["beta0"],
        "ell": params["ell"],
        "order": order,
        "breakpoint": torch.gather(breakpoint, 1, order),
        "breakpoint_defined": torch.gather(valid, 1, order),
    }

    for role in decoder.role_names:
        out[role] = {
            key: (
                value
                if key == "t"
                else torch.gather(value, 1, order)
            )
            for key, value in semantics[role].items()
        }

    return out


def function_recovery_metrics(
    signal,
    pred_draws,
    prefix=None,
    zero_tol=1e-6,
    constant_tol=1e-6,
):
    signal = torch.as_tensor(signal).detach().cpu().reshape(-1).float()
    draws = torch.as_tensor(pred_draws).detach().cpu().float()
    mean_function = draws.mean(dim=0)
    sst = (signal - signal.mean()).square().sum()
    draw_r2 = 1.0 - (
        (draws - signal[None, :]).square().sum(dim=1)
        / (sst + 1e-12)
    )
    function_energy = draws.square().mean(dim=1)
    function_variation = draws.var(dim=1, unbiased=False)

    if mean_function.std(unbiased=False) < 1e-12:
        corr2 = np.nan
    else:
        corr2 = float(
            torch.corrcoef(
                torch.stack([signal, mean_function])
            )[0, 1].square()
        )

    out = {
        "mse": float((mean_function - signal).square().mean()),
        "signal_r2": float(
            1.0
            - (mean_function - signal).square().sum()
            / (sst + 1e-12)
        ),
        "draw_r2_median": float(draw_r2.median()),
        "draw_r2_positive_prob": float(
            (draw_r2 > 0.0).float().mean()
        ),
        "zero_function_prob": float(
            (function_energy < zero_tol).float().mean()
        ),
        "constant_function_prob": float(
            (function_variation < constant_tol).float().mean()
        ),
        "corr2": corr2,
        "function_energy_mean": float(function_energy.mean()),
        "function_energy_median": float(function_energy.median()),
    }

    if prefix is not None:
        out = {f"{prefix}_{name}": value for name, value in out.items()}

    return out


@torch.no_grad()
def role_diagnostics(
    decoder,
    xi,
    latent_grad=None,
    base_loc_grad=None,
    epoch=None,
):
    _, semantics = decoder.unpack(xi, return_semantics=True)
    rows = []

    for role in decoder.role_names:
        item = semantics[role]
        margin = item["margin"].reshape(-1)
        gate = item["gate"].reshape(-1)
        active = item["active"]
        theta = item["theta"].abs().reshape(-1)
        active_count = active.float().sum(dim=1)
        active_gate = gate[active.reshape(-1)]
        q = torch.quantile(
            margin,
            torch.tensor(
                [0.025, 0.25, 0.50, 0.75, 0.975],
                device=margin.device,
                dtype=margin.dtype,
            ),
        )

        row = {
            "epoch": epoch,
            "role": role,
            "gate_learned": role in decoder.gate_roles,
            "margin_mean": float(margin.mean()),
            "margin_sd": float(margin.std(unbiased=False)),
            "margin_q025": float(q[0]),
            "margin_q25": float(q[1]),
            "margin_q50": float(q[2]),
            "margin_q75": float(q[3]),
            "margin_q975": float(q[4]),
            "mean_pip": float(active.float().mean()),
            "active_count_mean": float(active_count.mean()),
            "active_count_median": float(active_count.median()),
            "active_count_p_zero": float(
                (active_count == 0).float().mean()
            ),
            "gate_mean": float(gate.mean()),
            "active_gate_mean": (
                float(active_gate.mean()) if active_gate.numel() else np.nan
            ),
            "abs_theta_mean": float(theta.mean()),
            "abs_theta_median": float(theta.median()),
        }

        if latent_grad is not None:
            s_slice = decoder.s_role_slices[role]
            u_slice = decoder.u_role_slices[role]
            t_id = decoder.t_role_index[role]
            grad_s = latent_grad[:, s_slice]
            grad_u = latent_grad[
                :,
                decoder.s_dim + u_slice.start:
                decoder.s_dim + u_slice.stop,
            ]
            grad_t = latent_grad[
                :,
                decoder.s_dim + decoder.u_dim + t_id:
                decoder.s_dim + decoder.u_dim + t_id + 1,
            ]

            row.update({
                "grad_s_norm": float(grad_s.norm(dim=1).mean()),
                "grad_u_norm": float(grad_u.norm(dim=1).mean()),
                "grad_t_norm": float(grad_t.norm(dim=1).mean()),
            })

        if base_loc_grad is not None:
            s_slice = decoder.s_role_slices[role]
            u_slice = decoder.u_role_slices[role]
            t_id = decoder.t_role_index[role]

            row.update({
                "q0_loc_grad_s_norm": float(base_loc_grad[s_slice].norm()),
                "q0_loc_grad_u_norm": float(
                    base_loc_grad[
                        decoder.s_dim + u_slice.start:
                        decoder.s_dim + u_slice.stop
                    ].norm()
                ),
                "q0_loc_grad_t_norm": float(
                    base_loc_grad[
                        decoder.s_dim + decoder.u_dim + t_id
                    ].abs()
                ),
            })

        rows.append(row)

    return pd.DataFrame(rows)


@torch.no_grad()
def unit_path_diagnostics(
    decoder,
    xi,
    epoch=None,
    breakpoint_eps=1e-4,
):
    draws = sorted_unit_draws(
        decoder,
        xi,
        breakpoint_eps=breakpoint_eps,
    )
    z_in = draws["input"]["active"]
    z_b = draws["breakpoint"]["active"]
    z_out = draws["output"]["active"]
    rows = []

    for j in range(decoder.H):
        p_in = float(z_in[:, j].float().mean())
        p_b = float(z_b[:, j].float().mean())
        p_out = float(z_out[:, j].float().mean())
        p_joint = float(
            (z_in[:, j] & z_out[:, j]).float().mean()
        )
        p_product = p_in * p_out

        rows.append({
            "epoch": epoch,
            "unit_rank": j + 1,
            "input_pip": p_in,
            "breakpoint_pip": p_b,
            "output_pip": p_out,
            "path_probability": p_joint,
            "input_output_joint": p_joint,
            "input_output_product": p_product,
            "joint_minus_product": p_joint - p_product,
            "joint_over_product": p_joint / (p_product + 1e-12),
            "breakpoint_defined_prob": float(
                draws["breakpoint_defined"][:, j].float().mean()
            ),
        })

    return pd.DataFrame(rows)


@torch.no_grad()
def contribution_diagnostics(
    decoder,
    X_grid,
    xi,
    epoch=None,
    epsilon_C=1e-6,
    breakpoint_eps=1e-4,
):
    draws = sorted_unit_draws(
        decoder,
        xi,
        breakpoint_eps=breakpoint_eps,
    )
    x = torch.as_tensor(
        X_grid,
        device=xi.device,
        dtype=xi.dtype,
    )[:, 0]
    w = draws["input"]["theta"]
    b = draws["breakpoint"]["theta"]
    a = draws["output"]["theta"]
    units = a[:, None, :] * torch.relu(
        w[:, None, :] * x[None, :, None] - b[:, None, :]
    )
    energy = units.square().mean(dim=1)
    energy_sum = energy.sum(dim=1)
    total_energy = units.sum(dim=2).square().mean(dim=1)
    cancellation = total_energy / (energy_sum + 1e-12)
    rows = []

    for j in range(decoder.H):
        q = torch.quantile(
            energy[:, j],
            torch.tensor(
                [0.025, 0.25, 0.50, 0.75, 0.975],
                device=xi.device,
                dtype=xi.dtype,
            ),
        )

        rows.append({
            "epoch": epoch,
            "unit_rank": j + 1,
            "energy_mean": float(energy[:, j].mean()),
            "energy_median": float(energy[:, j].median()),
            "energy_q025": float(q[0]),
            "energy_q25": float(q[1]),
            "energy_q50": float(q[2]),
            "energy_q75": float(q[3]),
            "energy_q975": float(q[4]),
            "energy_active_prob": float(
                (energy[:, j] > epsilon_C).float().mean()
            ),
        })

    q = torch.quantile(
        cancellation,
        torch.tensor(
            [0.025, 0.25, 0.50, 0.75, 0.975],
            device=xi.device,
            dtype=xi.dtype,
        ),
    )
    summary = {
        "epoch": epoch,
        "unit_energy_sum_mean": float(energy_sum.mean()),
        "unit_energy_sum_median": float(energy_sum.median()),
        "total_unit_energy_mean": float(total_energy.mean()),
        "total_unit_energy_median": float(total_energy.median()),
        "cancellation_ratio_mean": float(cancellation.mean()),
        "cancellation_ratio_median": float(cancellation.median()),
        "cancellation_ratio_q025": float(q[0]),
        "cancellation_ratio_q25": float(q[1]),
        "cancellation_ratio_q50": float(q[2]),
        "cancellation_ratio_q75": float(q[3]),
        "cancellation_ratio_q975": float(q[4]),
    }

    return pd.DataFrame(rows), summary


@torch.no_grad()
def spike_slab_metrics(
    rat_decoder,
    rat_xi,
    mcmc_decoder,
    mcmc_xi,
    reference_threshold=0.5,
    min_active_draws=50,
    breakpoint_eps=1e-4,
):
    """Permutation-aware spike mass and conditional active-slab recovery."""

    rat = sorted_unit_draws(
        rat_decoder,
        rat_xi,
        breakpoint_eps=breakpoint_eps,
    )
    mcmc = sorted_unit_draws(
        mcmc_decoder,
        mcmc_xi,
        breakpoint_eps=breakpoint_eps,
    )
    rows = []

    for role in rat_decoder.role_names:
        for j in range(rat_decoder.H):
            rat_active = rat[role]["active"][:, j]
            mcmc_active = mcmc[role]["active"][:, j]
            rat_pip = float(rat_active.float().mean())
            mcmc_pip = float(mcmc_active.float().mean())
            rat_theta = (
                rat[role]["theta"][:, j][rat_active]
                .detach().cpu().numpy()
            )
            mcmc_theta = (
                mcmc[role]["theta"][:, j][mcmc_active]
                .detach().cpu().numpy()
            )

            if (
                len(rat_theta) >= min_active_draws
                and len(mcmc_theta) >= min_active_draws
            ):
                try:
                    slab_skl = kde_skl_1d(rat_theta, mcmc_theta)
                except (ValueError, np.linalg.LinAlgError):
                    slab_skl = np.nan
            else:
                slab_skl = np.nan

            rat_gate = rat[role]["gate"][:, j][rat_active]
            mcmc_gate = mcmc[role]["gate"][:, j][mcmc_active]
            rat_slab = rat[role]["s"][:, j][rat_active].abs()
            mcmc_slab = mcmc[role]["s"][:, j][mcmc_active].abs()

            rows.append({
                "role": role,
                "unit_rank": j + 1,
                "rat_pip": rat_pip,
                "mcmc_pip": mcmc_pip,
                "zero_mass_js": bernoulli_js(rat_pip, mcmc_pip),
                "reference_group": (
                    "reference-active"
                    if mcmc_pip > reference_threshold
                    else "reference-inactive"
                ),
                "rat_active_draws": int(len(rat_theta)),
                "mcmc_active_draws": int(len(mcmc_theta)),
                "conditional_slab_skl": slab_skl,
                "rat_active_gate_mean": (
                    float(rat_gate.mean()) if rat_gate.numel() else np.nan
                ),
                "mcmc_active_gate_mean": (
                    float(mcmc_gate.mean()) if mcmc_gate.numel() else np.nan
                ),
                "rat_active_slab_abs_mean": (
                    float(rat_slab.mean()) if rat_slab.numel() else np.nan
                ),
                "mcmc_active_slab_abs_mean": (
                    float(mcmc_slab.mean()) if mcmc_slab.numel() else np.nan
                ),
            })

    table = pd.DataFrame(rows)
    ref_active = table["reference_group"] == "reference-active"
    ref_inactive = ~ref_active
    valid_skl = table["conditional_slab_skl"].notna()
    summary = {
        "zero_mass_js_mean": float(table["zero_mass_js"].mean()),
        "zero_mass_js_median": float(table["zero_mass_js"].median()),
        "n_ref_active": int(ref_active.sum()),
        "n_ref_inactive": int(ref_inactive.sum()),
        "ref_active_js_mean": (
            float(table.loc[ref_active, "zero_mass_js"].mean())
            if ref_active.any() else np.nan
        ),
        "ref_active_js_median": (
            float(table.loc[ref_active, "zero_mass_js"].median())
            if ref_active.any() else np.nan
        ),
        "ref_inactive_js_mean": (
            float(table.loc[ref_inactive, "zero_mass_js"].mean())
            if ref_inactive.any() else np.nan
        ),
        "ref_inactive_js_median": (
            float(table.loc[ref_inactive, "zero_mass_js"].median())
            if ref_inactive.any() else np.nan
        ),
        "conditional_slab_skl_mean": (
            float(table.loc[valid_skl, "conditional_slab_skl"].mean())
            if valid_skl.any() else np.nan
        ),
        "conditional_slab_skl_median": (
            float(table.loc[valid_skl, "conditional_slab_skl"].median())
            if valid_skl.any() else np.nan
        ),
        "n_conditional_slab_valid": int(valid_skl.sum()),
        "rat_breakpoint_defined_prob": float(
            rat["breakpoint_defined"].float().mean()
        ),
        "mcmc_breakpoint_defined_prob": float(
            mcmc["breakpoint_defined"].float().mean()
        ),
    }

    return summary, table


def summarize_seeds(table, metrics, group_cols=None):
    """NaN-aware cross-seed means with the effective seed count."""

    table = pd.DataFrame(table)
    group_cols = list(group_cols or [])
    groups = (
        [((), table)]
        if not group_cols
        else table.groupby(group_cols, dropna=False)
    )
    rows = []

    for key, group in groups:
        if group_cols and not isinstance(key, tuple):
            key = (key,)
        prefix = dict(zip(group_cols, key)) if group_cols else {}

        for name in metrics:
            values = pd.to_numeric(
                group[name],
                errors="coerce",
            ).to_numpy(dtype=float)
            valid = np.isfinite(values)
            rows.append({
                **prefix,
                "metric": name,
                "mean": (
                    float(np.nanmean(values)) if valid.any() else np.nan
                ),
                "n_valid_seed": int(valid.sum()),
                "n_seed": int(len(values)),
            })

    return pd.DataFrame(rows)


def print_spike_slab_summary(summary):
    print(
        "reference-active JS:",
        (
            f"{summary['ref_active_js_mean']:.6f}"
            if summary["n_ref_active"] > 0
            else "NA (empty group)"
        ),
    )
    print(
        "reference-inactive JS:",
        (
            f"{summary['ref_inactive_js_mean']:.6f}"
            if summary["n_ref_inactive"] > 0
            else "NA (empty group)"
        ),
    )


@torch.no_grad()
def residual_path_metrics(
    decoder,
    xi,
    x_grid,
    method,
    eps_w=0.05,
    eps_a=0.05,
    eps_l=0.05,
    eps_c=1e-4,
    sigmoid_zero_threshold=0.05,
):
    """Effective paths for a one-block, one-dimensional residual BNN."""

    params = decoder.unpack(xi)
    E = params["E"][..., 0]
    e = params["e"]
    W1 = params["W1_0"]
    b1 = params["b1_0"]
    W2 = params["W2_0"].transpose(1, 2)
    Wout = params["Wout"][:, 0, :]
    ell = (Wout * E).sum(dim=1)
    w_eff = (W1 * E[:, None, :]).sum(dim=2)
    b_eff = -(W1 * e[:, None, :]).sum(dim=2) - b1
    a_eff = (W2 * Wout[:, None, :]).sum(dim=2)

    x = torch.as_tensor(
        x_grid,
        device=xi.device,
        dtype=xi.dtype,
    )[:, 0]
    units = a_eff[:, None, :] * torch.relu(
        w_eff[:, None, :] * x[None, :, None]
        - b_eff[:, None, :]
    )
    energy = units.square().mean(dim=1)
    energy_sum = energy.sum(dim=1)
    total_energy = units.sum(dim=2).square().mean(dim=1)
    cancellation = total_energy / (energy_sum + 1e-12)
    functional = (w_eff.abs() > eps_w) & (a_eff.abs() > eps_a)

    u = xi[:, decoder.s_dim:decoder.s_dim + decoder.u_dim]
    t = xi[:, decoder.s_dim + decoder.u_dim:]
    margin_active = {}
    connected = {}
    gates = {}

    for item in decoder.param_specs:
        name = item["name"]
        sl = slice(item["start"], item["end"])
        margin = u[:, sl] - t[:, item["t"]:item["t"] + 1]
        gate = decoder.gate(name, margin)
        gates[name] = gate.reshape(xi.shape[0], *item["shape"])
        margin_active[name] = decoder.active(
            name,
            margin,
            sigmoid_threshold=0.5,
        ).reshape(xi.shape[0], *item["shape"])
        connected[name] = (
            gate > sigmoid_zero_threshold
            if name in decoder.sigmoid_params
            else margin > 0.0
        ).reshape(xi.shape[0], *item["shape"])

    zE = connected["E"][..., 0]
    zW1 = connected["W1_0"]
    zW2 = connected["W2_0"].transpose(1, 2)
    zWout = connected["Wout"][:, 0, :]
    structural = (
        (zW1 & zE[:, None, :]).any(dim=2)
        & (zW2 & zWout[:, None, :]).any(dim=2)
    )
    E_margin = margin_active["E"][..., 0]
    Wout_margin = margin_active["Wout"][:, 0, :]
    E_gate = gates["E"][..., 0]
    Wout_gate = gates["Wout"][:, 0, :]

    summary = pd.DataFrame([{
        "method": method,
        "sigmoid_zero_threshold": float(sigmoid_zero_threshold),
        "E_margin_all_off_prob": float(
            (~E_margin.any(dim=1)).float().mean()
        ),
        "Wout_margin_all_off_prob": float(
            (~Wout_margin.any(dim=1)).float().mean()
        ),
        "E_gate_all_below_cutoff_prob": float(
            (E_gate.amax(dim=1) <= sigmoid_zero_threshold).float().mean()
        ),
        "Wout_gate_all_below_cutoff_prob": float(
            (Wout_gate.amax(dim=1) <= sigmoid_zero_threshold).float().mean()
        ),
        "E_effectively_zero_prob": float(
            (E.abs().amax(dim=1) <= eps_w).float().mean()
        ),
        "Wout_effectively_zero_prob": float(
            (Wout.abs().amax(dim=1) <= eps_a).float().mean()
        ),
        "E_gate_mean": float(E_gate.mean()),
        "Wout_gate_mean": float(Wout_gate.mean()),
        "mean_abs_skip": float(ell.abs().mean()),
        "skip_path_prob": float((ell.abs() > eps_l).float().mean()),
        "expected_structural_paths": float(
            structural.float().sum(dim=1).mean()
        ),
        "expected_functional_paths": float(
            functional.float().sum(dim=1).mean()
        ),
        "zero_functional_path_prob": float(
            (functional.sum(dim=1) == 0).float().mean()
        ),
        "unit_energy_sum_median": float(energy_sum.median()),
        "total_unit_energy_median": float(total_energy.median()),
        "cancellation_ratio_median": float(cancellation.median()),
    }])

    units_table = pd.DataFrame({
        "method": method,
        "unit": torch.arange(
            1,
            w_eff.shape[1] + 1,
        ).cpu().numpy(),
        "structural_path_prob": (
            structural.float().mean(dim=0).cpu().numpy()
        ),
        "functional_path_prob": (
            functional.float().mean(dim=0).cpu().numpy()
        ),
        "mean_abs_w_eff": w_eff.abs().mean(dim=0).cpu().numpy(),
        "mean_abs_a_eff": a_eff.abs().mean(dim=0).cpu().numpy(),
        "energy_mean": energy.mean(dim=0).cpu().numpy(),
        "energy_median": energy.median(dim=0).values.cpu().numpy(),
        "energy_active_prob": (
            (energy > eps_c).float().mean(dim=0).cpu().numpy()
        ),
    })

    return {"summary": summary, "units": units_table}


def group_posterior_summary(decoder, xi, method="RaT", epoch=None):
    """Posterior PIP, gate, margin, and slab norm for each LVR group."""

    with torch.no_grad():
        semantics = decoder.group_semantics(xi)
        slab_norm = decoder.group_slab_norms(xi)

    rows = []
    for meta in decoder.group_meta:
        group_id = int(meta["group_id"])
        rows.append({
            "method": method,
            "epoch": epoch,
            **meta,
            "pip": float(semantics["active"][:, group_id].float().mean()),
            "gate_mean": float(semantics["gate"][:, group_id].mean()),
            "margin_mean": float(semantics["margin"][:, group_id].mean()),
            "margin_sd": float(semantics["margin"][:, group_id].std()),
            "slab_norm_mean": float(slab_norm[:, group_id].mean()),
            "slab_norm_median": float(slab_norm[:, group_id].median()),
        })

    return pd.DataFrame(rows)


def unit_group_summary(decoder, xi, method="RaT", epoch=None):
    """Single-group/single-gate diagnostics for each shallow hidden unit."""

    if decoder.selection_mode != "unit_group":
        return pd.DataFrame()

    with torch.no_grad():
        units = decoder.unit_semantics(xi)

    rows = []
    for index, meta in enumerate(decoder.unit_groups):
        rows.append({
            "method": method,
            "epoch": epoch,
            "block": meta["block"],
            "unit": meta["unit"],
            "group_id": meta["group_id"],
            "unit_pip": float(units["active"][:, index].float().mean()),
            "gate_mean": float(units["gate"][:, index].mean()),
            "margin_mean": float(units["margin"][:, index].mean()),
            "margin_sd": float(units["margin"][:, index].std()),
            "input_slab_norm_mean": float(
                units["input_slab_norm"][:, index].mean()
            ),
            "output_slab_norm_mean": float(
                units["output_slab_norm"][:, index].mean()
            ),
            "slab_strength_mean": float(
                units["slab_strength"][:, index].mean()
            ),
            "effective_strength_mean": float(
                units["effective_strength"][:, index].mean()
            ),
        })

    table = pd.DataFrame(rows)
    if not table.empty:
        rank = table["unit_pip"].rank(method="first", ascending=False).astype(int)
        table.insert(5, "posterior_rank", rank)
        table = table.sort_values(["block", "posterior_rank"]).reset_index(drop=True)
    return table


def _ranked_unit_selection_draws(decoder, xi):
    units = decoder.unit_semantics(xi)
    order = torch.argsort(units["effective_strength"], dim=1, descending=True)
    return {
        "pip_draws": torch.gather(units["active"], 1, order),
        "slab_strength": torch.gather(units["slab_strength"], 1, order),
        "effective_strength": torch.gather(
            units["effective_strength"], 1, order
        ),
    }


def _feature_selection_draws(decoder, xi):
    semantics = decoder.group_semantics(xi)
    slab_norm = decoder.group_slab_norms(xi)
    return {
        "pip_draws": semantics["active"],
        "slab_strength": slab_norm,
        "effective_strength": slab_norm * semantics["gate"],
    }


def _truth_mask(decoder, truth):
    if decoder.selection_mode == "unit_group":
        n_slots = decoder.H
        n_true = int(truth["n_true_units"])
        if n_true > n_slots:
            raise ValueError("n_true_units exceeds the fitted hidden-unit slots.")
        mask = np.zeros(n_slots, dtype=bool)
        mask[:n_true] = True
        labels = [f"unit_rank_{rank + 1}" for rank in range(n_slots)]
        return mask, labels

    feature_true = np.asarray(truth["feature_true"], dtype=float).reshape(-1)
    if feature_true.size != decoder.input_dim:
        raise ValueError("feature_true length does not match decoder.input_dim.")
    return feature_true > 0.5, [f"feature_{j}" for j in range(feature_true.size)]


def _safe_nan_summary(values, reducer):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(reducer(finite)) if finite.size else np.nan


def grouped_recovery_metrics(
    rat_decoder,
    rat_xi,
    mcmc_decoder,
    mcmc_xi,
    truth,
    min_active_draws=50,
):
    """
    Group-selection recovery against the exactly matched MCMC model.

    ``true_active_skl`` is the VI--MCMC conditional slab-strength SKL on
    teacher-active features or permutation-invariant unit ranks. It is not a
    KL divergence from a continuous posterior to a point-valued truth.
    """

    if rat_decoder.compatibility_signature() != mcmc_decoder.compatibility_signature():
        raise ValueError("MCMC and VI must use exactly the same grouped decoder.")

    if rat_decoder.selection_mode == "unit_group":
        rat = _ranked_unit_selection_draws(rat_decoder, rat_xi)
        mcmc = _ranked_unit_selection_draws(mcmc_decoder, mcmc_xi)
        target_type = "unit_rank"
    else:
        rat = _feature_selection_draws(rat_decoder, rat_xi)
        mcmc = _feature_selection_draws(mcmc_decoder, mcmc_xi)
        target_type = "feature"

    truth_active, labels = _truth_mask(rat_decoder, truth)
    rat_active = rat["pip_draws"].detach().cpu().numpy().astype(bool)
    mcmc_active = mcmc["pip_draws"].detach().cpu().numpy().astype(bool)
    rat_strength = rat["slab_strength"].detach().cpu().numpy()
    mcmc_strength = mcmc["slab_strength"].detach().cpu().numpy()
    rat_pip = rat_active.mean(axis=0)
    mcmc_pip = mcmc_active.mean(axis=0)
    rows = []
    active_skl = []
    zero_js = []

    for target, label in enumerate(labels):
        rat_values = rat_strength[rat_active[:, target], target]
        mcmc_values = mcmc_strength[mcmc_active[:, target], target]
        skl = np.nan
        js = np.nan

        if truth_active[target]:
            if (
                rat_values.size >= int(min_active_draws)
                and mcmc_values.size >= int(min_active_draws)
            ):
                skl = _skl(rat_values, mcmc_values)
            active_skl.append(skl)
        else:
            js = bernoulli_js(rat_pip[target], mcmc_pip[target])
            zero_js.append(js)

        rows.append({
            "target_type": target_type,
            "target": target,
            "label": label,
            "truth_active": bool(truth_active[target]),
            "truth_pip": float(truth_active[target]),
            "rat_pip": float(rat_pip[target]),
            "mcmc_pip": float(mcmc_pip[target]),
            "pip_error_truth": float(
                rat_pip[target] - truth_active[target]
            ),
            "pip_error": float(rat_pip[target] - mcmc_pip[target]),
            "conditional_slab_skl": skl,
            "zero_mass_js": js,
            "n_rat_active_draws": int(rat_values.size),
            "n_mcmc_active_draws": int(mcmc_values.size),
        })

    active_skl = np.asarray(active_skl, dtype=float)
    zero_js = np.asarray(zero_js, dtype=float)
    summary = {
        "pip_rmse_truth": float(
            np.sqrt(np.mean((rat_pip - truth_active.astype(float)) ** 2))
        ),
        "pip_rmse_mcmc": float(np.sqrt(np.mean((rat_pip - mcmc_pip) ** 2))),
        "true_active_skl": _safe_nan_summary(active_skl, np.median),
        "true_active_skl_mean": _safe_nan_summary(active_skl, np.mean),
        "zero_js": _safe_nan_summary(zero_js, np.median),
        "zero_js_mean": _safe_nan_summary(zero_js, np.mean),
        "n_truth_active": int(truth_active.sum()),
        "n_truth_zero": int((~truth_active).sum()),
        "n_valid_true_active_skl": int(np.isfinite(active_skl).sum()),
        "selection_mode": rat_decoder.selection_mode,
    }
    summary["rmse_with_mcmc"] = summary["pip_rmse_mcmc"]
    summary["true_skl"] = summary["true_active_skl"]

    return summary, pd.DataFrame(rows)


@torch.no_grad()
def true_active_joint_draws(decoder, xi, truth):
    """
    Return the two truth-active slab-strength coordinates conditional on both
    targets being active in the same posterior draw.

    Unit selection uses permutation-invariant ranked units. Feature selection
    uses the two truth-active raw predictors.
    """

    if decoder.selection_mode == "unit_group":
        draws = _ranked_unit_selection_draws(decoder, xi)
    else:
        draws = _feature_selection_draws(decoder, xi)

    truth_active, labels = _truth_mask(decoder, truth)
    active_idx = np.flatnonzero(truth_active)
    if len(active_idx) != 2:
        raise ValueError(
            "true_active_joint_draws requires exactly two truth-active targets."
        )

    j1, j2 = active_idx
    active = draws["pip_draws"]
    strength = draws["slab_strength"]
    joint_active = active[:, j1] & active[:, j2]
    values = strength[joint_active][:, [j1, j2]].detach().cpu().numpy()

    return {
        "values": values,
        "labels": (labels[j1], labels[j2]),
        "n_joint_active": int(joint_active.sum().item()),
        "joint_active_prob": float(joint_active.float().mean().item()),
    }


def plot_true_active_joint_density(
    rat_decoder,
    rat_xi,
    mcmc_decoder,
    mcmc_xi,
    truth,
    min_draws=50,
    n_grid=100,
):
    """Overlay RaT and MCMC 2D KDE contours for two truth-active targets."""

    rat = true_active_joint_draws(rat_decoder, rat_xi, truth)
    mcmc = true_active_joint_draws(mcmc_decoder, mcmc_xi, truth)

    print(
        f"joint-active draws: RaT={rat['n_joint_active']}, "
        f"MCMC={mcmc['n_joint_active']}"
    )
    print(
        f"joint-active probability: RaT={rat['joint_active_prob']:.4f}, "
        f"MCMC={mcmc['joint_active_prob']:.4f}"
    )

    if (
        rat["n_joint_active"] < int(min_draws)
        or mcmc["n_joint_active"] < int(min_draws)
    ):
        print("Joint density not plotted: insufficient joint-active posterior draws.")
        return None, None

    Xr = rat["values"]
    Xm = mcmc["values"]

    try:
        joint_skl = kde_skl_2d(Xr, Xm)
        print(f"conditional joint SKL: {joint_skl:.4f}")
    except (ValueError, np.linalg.LinAlgError):
        print("conditional joint SKL: NA")

    x_all = np.concatenate([Xr[:, 0], Xm[:, 0]])
    y_all = np.concatenate([Xr[:, 1], Xm[:, 1]])
    x_lo, x_hi = np.quantile(x_all, [0.005, 0.995])
    y_lo, y_hi = np.quantile(y_all, [0.005, 0.995])
    x_pad = 0.10 * (x_hi - x_lo + 1e-8)
    y_pad = 0.10 * (y_hi - y_lo + 1e-8)

    gx = np.linspace(x_lo - x_pad, x_hi + x_pad, int(n_grid))
    gy = np.linspace(y_lo - y_pad, y_hi + y_pad, int(n_grid))
    xx, yy = np.meshgrid(gx, gy)
    points = np.vstack([xx.ravel(), yy.ravel()])

    try:
        kde_rat = gaussian_kde(Xr.T)
        kde_mcmc = gaussian_kde(Xm.T)
        z_rat = kde_rat(points).reshape(xx.shape)
        z_mcmc = kde_mcmc(points).reshape(xx.shape)
    except np.linalg.LinAlgError:
        print("Joint density not plotted: singular KDE covariance.")
        return None, None

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contour(xx, yy, z_rat, levels=6, linestyles="-")
    ax.contour(xx, yy, z_mcmc, levels=6, linestyles="--")
    ax.plot([], [], linestyle="-", label="RaT")
    ax.plot([], [], linestyle="--", label="MCMC")
    ax.set_xlabel(rat["labels"][0])
    ax.set_ylabel(rat["labels"][1])
    ax.set_title("True-active joint posterior density")
    ax.legend()
    fig.tight_layout()
    return fig, ax