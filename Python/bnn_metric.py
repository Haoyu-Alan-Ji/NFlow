"""Metrics used by the grouped-BNN experiments only.

This module intentionally contains only the metrics retained for the paper:
Active SKL, Zero JS, signal MSE/R2, TPR/FPR, AUROC/AUPRC, expected active-unit
count error, network density, active-path density, and cross-run feature
stability (frequency + Kuncheva index).
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.metrics import average_precision_score, roc_auc_score
import torch


def _np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def bernoulli_js(p, q, eps=1e-12):
    p = float(np.clip(p, eps, 1.0 - eps))
    q = float(np.clip(q, eps, 1.0 - eps))
    P = np.array([1.0 - p, p])
    Q = np.array([1.0 - q, q])
    M = 0.5 * (P + Q)
    return float(
        0.5 * np.sum(P * np.log(P / M))
        + 0.5 * np.sum(Q * np.log(Q / M))
    )


def _skl_grid(p, q, eps=1e-12):
    p = np.maximum(np.asarray(p, dtype=float), eps)
    q = np.maximum(np.asarray(q, dtype=float), eps)
    p /= p.sum()
    q /= q.sum()
    return float(
        0.5 * np.sum(p * np.log(p / q))
        + 0.5 * np.sum(q * np.log(q / p))
    )


def kde_skl_1d(x, y, n_grid=128):
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.size < 2 or y.size < 2:
        return np.nan
    lo = min(np.quantile(x, 0.001), np.quantile(y, 0.001))
    hi = max(np.quantile(x, 0.999), np.quantile(y, 0.999))
    pad = 0.1 * (hi - lo + 1e-8)
    grid = np.linspace(lo - pad, hi + pad, int(n_grid))
    try:
        return _skl_grid(gaussian_kde(x)(grid), gaussian_kde(y)(grid))
    except (ValueError, np.linalg.LinAlgError):
        return np.nan


@torch.no_grad()
def predict_draws(decoder, X, xi, batch_size=500):
    X = torch.as_tensor(X)
    xi = torch.as_tensor(xi, device=X.device, dtype=X.dtype)
    out = []
    for start in range(0, xi.shape[0], int(batch_size)):
        out.append(decoder(X, xi[start:start + int(batch_size)]).detach().cpu())
    return torch.cat(out, dim=0)


def function_metrics(signal, pred_draws, prefix=None):
    signal = torch.as_tensor(signal).detach().cpu().float().reshape(-1)
    draws = torch.as_tensor(pred_draws).detach().cpu().float()
    mean_function = draws.mean(dim=0)
    mse = float((mean_function - signal).square().mean())
    sst = (signal - signal.mean()).square().sum()
    r2 = float(
        1.0 - (mean_function - signal).square().sum() / (sst + 1e-12)
    )
    out = {"mse": mse, "r2": r2}
    if prefix is not None:
        out = {f"{prefix}_{key}": value for key, value in out.items()}
    return out


def _feature_truth(decoder, truth):
    if "feature_true" in truth:
        mask = np.asarray(truth["feature_true"], dtype=float).reshape(-1) > 0.5
    else:
        mask = np.zeros(decoder.input_dim, dtype=bool)
        mask[np.asarray(truth["active_features"], dtype=int)] = True
    if mask.size != decoder.input_dim:
        raise ValueError("Feature truth length does not match decoder input_dim.")
    return mask


@torch.no_grad()
def feature_pips(decoder, xi):
    return decoder.feature_semantics(xi)["active"].float().mean(dim=0).cpu().numpy()


def feature_selection_metrics(decoder, xi, truth, threshold=0.5):
    pip = feature_pips(decoder, xi)
    target = _feature_truth(decoder, truth)
    selected = pip > float(threshold)
    tp = int(np.sum(selected & target))
    fp = int(np.sum(selected & ~target))
    fn = int(np.sum(~selected & target))
    tn = int(np.sum(~selected & ~target))
    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    if np.unique(target.astype(int)).size < 2:
        auroc = np.nan
        auprc = np.nan
    else:
        auroc = float(roc_auc_score(target.astype(int), pip))
        auprc = float(average_precision_score(target.astype(int), pip))
    return {
        "tpr": float(tpr),
        "fpr": float(fpr),
        "auroc": auroc,
        "auprc": auprc,
        "selected_features": int(selected.sum()),
        "feature_pip": pip,
    }


@torch.no_grad()
def ranked_unit_draws(decoder, xi):
    units = decoder.unit_semantics(xi)
    active = units["active"]
    effective = units["effective_strength"]
    offset = effective.amax(dim=1, keepdim=True) + 1.0
    order = torch.argsort(
        effective + active.to(effective.dtype) * offset,
        dim=1,
        descending=True,
    )
    return {
        "active": torch.gather(active, 1, order),
        "slab_strength": torch.gather(units["slab_strength"], 1, order),
        "effective_strength": torch.gather(effective, 1, order),
    }


@torch.no_grad()
def selection_draws(decoder, xi):
    if decoder.selection_mode == "feature_group":
        x = decoder.feature_semantics(xi)
        return {
            "active": x["active"],
            "slab_strength": x["slab_strength"],
            "effective_strength": x["effective_strength"],
        }
    if decoder.selection_mode == "unit_group":
        return ranked_unit_draws(decoder, xi)
    if decoder.selection_mode == "feature_unit_induced_edge":
        # Used only for diagnostics; paper-level selection metrics use the
        # identifiable feature block directly.
        x = decoder.group_semantics(xi)
        return {
            "active": x["active"],
            "slab_strength": decoder.group_slab_norms(xi),
            "effective_strength": x["gate"] * decoder.group_slab_norms(xi),
        }
    raise ValueError("Unsupported grouped selection mode.")


def unit_count_error(decoder, xi, truth):
    units = decoder.unit_semantics(xi)
    expected_k = float(units["active"].float().sum(dim=1).mean())
    k_true = int(truth["n_true_units"])
    return {
        "expected_active_units": expected_k,
        "active_unit_count_error": float(abs(expected_k - k_true)),
    }


def _recovery_truth_mask(decoder, truth):
    if decoder.selection_mode == "feature_group":
        return _feature_truth(decoder, truth)
    if decoder.selection_mode == "unit_group":
        mask = np.zeros(decoder.n_units, dtype=bool)
        mask[:int(truth["n_true_units"])] = True
        return mask
    raise ValueError("Posterior recovery is used only for shallow feature/unit modes.")


def posterior_recovery(
    decoder,
    xi,
    reference_decoder,
    reference_xi,
    truth,
    min_active_draws=50,
):
    if decoder.compatibility_signature() != reference_decoder.compatibility_signature():
        raise ValueError("VI and MCMC decoders are not exactly matched.")

    vi = selection_draws(decoder, xi)
    ref = selection_draws(reference_decoder, reference_xi)
    truth_active = _recovery_truth_mask(decoder, truth)
    vi_active = _np(vi["active"]).astype(bool)
    ref_active = _np(ref["active"]).astype(bool)
    vi_strength = _np(vi["slab_strength"])
    ref_strength = _np(ref["slab_strength"])
    vi_pip = vi_active.mean(axis=0)
    ref_pip = ref_active.mean(axis=0)

    active_skl = []
    zero_js = []
    rows = []
    for j in range(len(truth_active)):
        skl = np.nan
        js = np.nan
        if truth_active[j]:
            a = vi_strength[vi_active[:, j], j]
            b = ref_strength[ref_active[:, j], j]
            if a.size >= int(min_active_draws) and b.size >= int(min_active_draws):
                skl = kde_skl_1d(a, b)
            active_skl.append(skl)
        else:
            js = bernoulli_js(vi_pip[j], ref_pip[j])
            zero_js.append(js)
        rows.append({
            "target": j,
            "truth_active": bool(truth_active[j]),
            "vi_pip": float(vi_pip[j]),
            "mcmc_pip": float(ref_pip[j]),
            "active_skl": skl,
            "zero_js": js,
        })

    active_skl = np.asarray(active_skl, dtype=float)
    zero_js = np.asarray(zero_js, dtype=float)
    return {
        "active_skl": float(np.nanmedian(active_skl)) if active_skl.size else np.nan,
        "zero_js": float(np.nanmedian(zero_js)) if zero_js.size else np.nan,
        "n_valid_active_skl": int(np.isfinite(active_skl).sum()),
        "recovery_table": pd.DataFrame(rows),
    }


@torch.no_grad()
def network_density(decoder, xi):
    if decoder.selection_mode != "feature_unit_induced_edge":
        raise ValueError("Network density is defined for induced connectivity.")
    blocks = decoder.edge_semantics(xi)
    active = torch.cat(
        [item["active"].reshape(xi.shape[0], -1) for item in blocks.values()],
        dim=1,
    )
    return float(active.float().mean())


@torch.no_grad()
def active_path_density(decoder, xi):
    """Posterior expected fraction of complete input-to-output active paths."""

    if decoder.selection_mode != "feature_unit_induced_edge":
        raise ValueError("Active-path density requires feature+unit groups.")
    feature_active = decoder.feature_semantics(xi)["active"].float()
    unit_active = decoder.unit_semantics(xi)["active"].float()

    active_paths = feature_active.sum(dim=1)
    total_paths = float(decoder.input_dim)
    for layer, width in enumerate(decoder.hidden_dims):
        count = unit_active[:, decoder.layer_slices[layer]].sum(dim=1)
        active_paths = active_paths * count
        total_paths *= float(width)
    return float((active_paths / total_paths).mean())


@torch.no_grad()
def evaluate_bnn(
    decoder,
    xi,
    X,
    signal,
    truth,
    reference_decoder=None,
    reference_xi=None,
    support_threshold=0.5,
    min_active_draws=50,
):
    pred = predict_draws(decoder, X, xi)
    summary = function_metrics(signal, pred)

    if decoder.has_feature_gates:
        feature = feature_selection_metrics(
            decoder, xi, truth, threshold=support_threshold
        )
        summary.update({
            key: value for key, value in feature.items()
            if key != "feature_pip"
        })
        summary["feature_pip"] = feature["feature_pip"]

    if decoder.selection_mode == "unit_group":
        summary.update(unit_count_error(decoder, xi, truth))

    if decoder.selection_mode == "feature_unit_induced_edge":
        summary["network_density"] = network_density(decoder, xi)
        summary["active_path_density"] = active_path_density(decoder, xi)

    if reference_decoder is not None and reference_xi is not None:
        recovery = posterior_recovery(
            decoder,
            xi,
            reference_decoder,
            reference_xi,
            truth,
            min_active_draws=min_active_draws,
        )
        summary["active_skl"] = recovery["active_skl"]
        summary["zero_js"] = recovery["zero_js"]
        summary["n_valid_active_skl"] = recovery["n_valid_active_skl"]
        summary["recovery_table"] = recovery["recovery_table"]
        reference_pred = predict_draws(reference_decoder, X, reference_xi)
        summary.update(function_metrics(signal, reference_pred, prefix="mcmc"))

    return summary


def kuncheva_index(set_a, set_b, p, k):
    """Kuncheva index for two fixed-size selected sets."""

    set_a = set(map(int, set_a))
    set_b = set(map(int, set_b))
    p = int(p)
    k = int(k)
    if len(set_a) != k or len(set_b) != k:
        raise ValueError("Kuncheva index requires both supports to have size k.")
    denom = k - (k * k) / p
    if abs(denom) < 1e-12:
        return np.nan
    return float((len(set_a & set_b) - (k * k) / p) / denom)


def stability_metrics(pip_matrix, k_true, threshold=0.5):
    """Cross-run feature frequency and mean pairwise Kuncheva stability.

    Feature frequency uses the median-probability support.  Kuncheva uses each
    run's top-k_true set so that the standard fixed-cardinality definition is
    respected even when thresholded support sizes differ.
    """

    pips = np.asarray(pip_matrix, dtype=float)
    if pips.ndim != 2:
        raise ValueError("pip_matrix must have shape [runs, features].")
    n_runs, p = pips.shape
    k_true = int(k_true)
    if not 0 < k_true < p:
        raise ValueError("k_true must satisfy 0 < k_true < p.")

    threshold_support = pips > float(threshold)
    frequency = threshold_support.mean(axis=0)
    topk = [
        set(np.argsort(-row)[:k_true].astype(int).tolist()) for row in pips
    ]
    values = [
        kuncheva_index(topk[i], topk[j], p=p, k=k_true)
        for i, j in combinations(range(n_runs), 2)
    ]
    return {
        "feature_frequency": frequency,
        "kuncheva_index": float(np.nanmean(values)) if values else np.nan,
        "n_runs": int(n_runs),
    }
