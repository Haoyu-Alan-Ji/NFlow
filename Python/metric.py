from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Mapping
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import pandas as pd
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


def ranking_metrics(
    *,
    support_score,
    beta_true=None,
    active_idx=None,
    p=None,
):
    score = np.asarray(support_score, dtype=float)

    if beta_true is not None:
        truth = (np.asarray(beta_true) != 0.0).astype(int)
    else:
        if p is None:
            p = len(score)
        truth = np.zeros(int(p), dtype=int)
        if active_idx is not None:
            truth[np.asarray(active_idx, dtype=int)] = 1

    # AUROC is undefined if only one class is present.
    if len(np.unique(truth)) < 2:
        auroc = np.nan
        auprc = np.nan
    else:
        auroc = float(roc_auc_score(truth, score))
        auprc = float(average_precision_score(truth, score))

    return {
        "auroc": auroc,
        "auprc": auprc,
    }


def flow_row_from_result(out_flow):
    """
    Convert one flow result into one benchmark summary row.
    """
    import numpy as np
    from .metric import ranking_metrics

    final = out_flow["final"]
    sim_info = out_flow.get("sim_info", {})

    selection = dict(final.get("selection_metrics", {}))

    var_table = final.get("var_table", None)

    if var_table is not None and not var_table.empty:
        if "support_score" in var_table.columns:
            score = var_table.sort_values("j")["support_score"].to_numpy(dtype=float)
        elif "hard_freq" in var_table.columns:
            score = var_table.sort_values("j")["hard_freq"].to_numpy(dtype=float)
        elif "selected" in var_table.columns:
            score = var_table.sort_values("j")["selected"].to_numpy(dtype=float)
        else:
            score = None

        if score is not None:
            beta_true = None
            active_idx = sim_info.get("active_idx", None)

            if "beta_true" in var_table.columns:
                beta_true = var_table.sort_values("j")["beta_true"].to_numpy(dtype=float)

            ranking = ranking_metrics(
                support_score=score,
                beta_true=beta_true,
                active_idx=active_idx,
                p=len(score),
            )
            selection.update(ranking)

    train_metrics = final.get("train_metrics", {})
    val_metrics = final.get("val_metrics", {})
    test_metrics = final.get("test_metrics", {})

    row = {
        "method": out_flow.get("method", "flow_stagewise"),
        "seed": out_flow.get("seed"),
        "runtime_sec": out_flow.get("runtime_sec"),
        "converged": True,
        "n_iter": out_flow.get("n_iter", None),
        "selected_ckpt_id": out_flow.get("selected_ckpt_id"),

        "support_size": selection.get("support_size"),
        "precision": selection.get("precision"),
        "recall": selection.get("recall"),
        "f1": selection.get("f1"),
        "auroc": selection.get("auroc"),
        "auprc": selection.get("auprc"),
        "fdr": selection.get("fdr"),
        "tp": selection.get("tp"),
        "fp": selection.get("fp"),
        "fn": selection.get("fn"),
        "tn": selection.get("tn"),

        "train_mse": train_metrics.get("mse"),
        "val_mse": val_metrics.get("mse"),
        "test_mse": test_metrics.get("mse"),

        "train_r2": train_metrics.get("r2"),
        "val_r2": val_metrics.get("r2"),
        "test_r2": test_metrics.get("r2"),

        "train_nll": train_metrics.get("nll"),
        "val_nll": val_metrics.get("nll"),
        "test_nll": test_metrics.get("nll"),
    }

    for key in [
        "n",
        "p",
        "snr",
        "true_prop",
        "n_active",
        "sigma2",
        "rho",
        "noise_x",
        "block_size",
        "group_size",
    ]:
        if key in sim_info:
            row[key] = sim_info[key]

    return row

def benchmark_row_from_result(result: Mapping[str, Any]) -> Dict[str, Any]:
    selection = result.get("selection_metrics", {}) or {}
    pred = result.get("predictive_metrics", {}) or {}
    sim_info = result.get("sim_info", {}) or {}

    row: Dict[str, Any] = {
        "method": result.get("method"),
        "seed": result.get("seed"),
        "runtime_sec": result.get("runtime_sec"),
        "converged": result.get("converged"),
        "n_iter": result.get("n_iter"),
        "support_size": selection.get("support_size"),
        "precision": selection.get("precision"),
        "recall": selection.get("recall"),
        "f1": selection.get("f1"),
        "auroc": selection.get("auroc"),
        "auprc": selection.get("auprc"),
        "fdr": selection.get("fdr"),
        "tp": selection.get("tp"),
        "fp": selection.get("fp"),
        "fn": selection.get("fn"),
        "tn": selection.get("tn"),
        "train_mse": pred.get("train", {}).get("mse"),
        "val_mse": pred.get("val", {}).get("mse"),
        "test_mse": pred.get("test", {}).get("mse"),
        "train_r2": pred.get("train", {}).get("r2"),
        "val_r2": pred.get("val", {}).get("r2"),
        "test_r2": pred.get("test", {}).get("r2"),
        "train_nll": pred.get("train", {}).get("nll"),
        "val_nll": pred.get("val", {}).get("nll"),
        "test_nll": pred.get("test", {}).get("nll"),
    }

    for key in ["n", "p", "snr", "true_prop", "n_active", "sigma2"]:
        if key in sim_info:
            row[key] = sim_info[key]
    return row


def predictive_metrics(X, y, beta_hard_samples, sigma2: Optional[float] = None, family: str = "gaussian"):
    return posterior_predictive_metrics_from_hard_samples(X, y, beta_hard_samples, sigma2=sigma2, family=family)

def print_result(out, *, top_k=20):
    final = out.get("final", {})

    method = out.get("method", "unknown")
    seed = out.get("seed", None)

    selection_metrics = dict(
        out.get("selection_metrics", final.get("selection_metrics", {})) or {}
    )

    summary_row = out.get("summary_row", {})
    if "auroc" not in selection_metrics and "auroc" in summary_row:
        selection_metrics["auroc"] = summary_row["auroc"]
    if "auprc" not in selection_metrics and "auprc" in summary_row:
        selection_metrics["auprc"] = summary_row["auprc"]

    predictive_metrics = out.get("predictive_metrics", final.get("predictive_metrics", None))
    var_table = out.get("var_table", final.get("var_table", None))
    pred_table = out.get("pred_table", final.get("pred_table", None))
    selected_support = out.get("selected_support", final.get("selected_support", []))

    print(f"===== {method} result =====")
    print(f"seed          : {seed}")

    if out.get("runtime_sec", None) is not None:
        print(f"runtime_sec   : {out.get('runtime_sec'):.6f}")

    if out.get("converged", None) is not None:
        print(f"converged     : {out.get('converged')}")

    if out.get("n_iter", None) is not None:
        print(f"n_iter        : {out.get('n_iter')}")

    if out.get("selected_ckpt_id", None) is not None:
        print(f"selected_ckpt : {out.get('selected_ckpt_id')}")

    if out.get("support_threshold", None) is not None:
        print(f"threshold     : {out.get('support_threshold')}")

    if out.get("sigma2", None) is not None:
        print(f"sigma2        : {out.get('sigma2')}")

    print(f"selected_size : {len(selected_support)}")

    print("\n===== Selection metrics =====")
    if selection_metrics:
        print(pd.DataFrame([selection_metrics]).to_string(index=False))
    else:
        print("(empty selection_metrics)")

    print("\n===== Selected support =====")
    print(selected_support)

    print(f"\n===== Top {top_k} variables =====")
    if isinstance(var_table, pd.DataFrame) and not var_table.empty:
        vt = var_table.copy()

        if "support_score" in vt.columns:
            vt = vt.sort_values("support_score", ascending=False)
        elif "pip" in vt.columns:
            vt = vt.sort_values("pip", ascending=False)
        elif "hard_freq" in vt.columns:
            vt = vt.sort_values("hard_freq", ascending=False)
        elif "beta_hard_mean" in vt.columns:
            vt = vt.sort_values("beta_hard_mean", key=lambda x: x.abs(), ascending=False)
        elif "abs_beta_mean" in vt.columns:
            vt = vt.sort_values("abs_beta_mean", ascending=False)
        elif "beta_mean" in vt.columns:
            vt = vt.sort_values("beta_mean", key=lambda x: x.abs(), ascending=False)

        print(vt.head(top_k).to_string(index=False))
    else:
        print("(empty var_table)")

    print("\n===== Predictive metrics =====")

    if isinstance(pred_table, pd.DataFrame) and not pred_table.empty:
        print(pred_table.to_string(index=False))
    elif predictive_metrics:
        rows = []
        for split, metrics in predictive_metrics.items():
            row = {"split": split}
            row.update(metrics)
            rows.append(row)
        print(pd.DataFrame(rows).to_string(index=False))
    else:
        train_metrics = final.get("train_metrics", {})
        val_metrics = final.get("val_metrics", {})
        test_metrics = final.get("test_metrics", {})

        rows = []
        if train_metrics:
            rows.append({"split": "train", **train_metrics})
        if val_metrics:
            rows.append({"split": "val", **val_metrics})
        if test_metrics:
            rows.append({"split": "test", **test_metrics})

        if rows:
            print(pd.DataFrame(rows).to_string(index=False))
        else:
            print("(empty predictive metrics)")


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
