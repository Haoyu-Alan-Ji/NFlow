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


def recovery_metrics(beta_last, active_last, beta_true, mcmc_ref, max_pairs: int = 10) -> Dict[str, float]:
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

    active_skl = [kde_skl_1d(beta_last[:, j], beta_ref[:, j]) for j in active_idx]
    joint_skl = [kde_skl_2d(beta_last[:, [j, k]], beta_ref[:, [j, k]]) for j, k in _active_pairs(beta_true, max_pairs)]

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
        "joint_skl_median": float(np.nanmedian(joint_skl)) if joint_skl else np.nan,
        "joint_skl_mean": float(np.nanmean(joint_skl)) if joint_skl else np.nan,

        "active_marg_skl_median": float(np.nanmedian(active_skl)) if active_skl else np.nan,
        "active_marg_skl_mean": float(np.nanmean(active_skl)) if active_skl else np.nan,

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


@torch.no_grad()
def posterior_draws(decoder, xi) -> Dict[str, Any]:
    """Decode theta and structural indicators Z = 1(u > t)."""
    xi = _tensor(xi)
    params = decoder.unpack(xi)

    u = xi[:, decoder.s_dim:decoder.s_dim + decoder.u_dim]
    t = xi[:, decoder.s_dim + decoder.u_dim:]

    theta = {}
    active = {}

    for item in decoder.param_specs:
        name = item["name"]
        sl = slice(item["start"], item["end"])
        margin = u[:, sl] - t[:, item["t"]:item["t"] + 1]

        theta[name] = params[name].detach().cpu()
        active[name] = (
            margin > 0.0
        ).reshape(xi.shape[0], *item["shape"]).detach().cpu()

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

    xi = _tensor(xi)
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

    mcmc_post = posterior_draws(mcmc_decoder, mcmc_xi)
    last_post = posterior_draws(last_decoder, last_xi)

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
    last_pred_draws,
    interval: float = 0.95,
):
    """Truth, zero function, MCMC, and LaST on one 1D plot."""
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
    ax.plot(x, last_mean, label="LaST mean")

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