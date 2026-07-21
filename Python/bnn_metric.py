from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.metrics import average_precision_score, roc_auc_score

from .utils import to_numpy

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

    xi = _tensor(xi, device=X.device, dtype=X.dtype)
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