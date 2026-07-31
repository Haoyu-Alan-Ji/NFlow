import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


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


def kde_skl_1d(x, y, n_grid=256):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    lo = min(np.quantile(x, 0.001), np.quantile(y, 0.001))
    hi = max(np.quantile(x, 0.999), np.quantile(y, 0.999))
    pad = 0.1 * (hi - lo + 1e-8)
    grid = np.linspace(lo - pad, hi + pad, n_grid)
    p = np.maximum(gaussian_kde(x)(grid), 1e-12)
    q = np.maximum(gaussian_kde(y)(grid), 1e-12)
    p /= p.sum()
    q /= q.sum()

    return float(
        0.5
        * (
            np.sum(p * np.log(p / q))
            + np.sum(q * np.log(q / p))
        )
    )


@torch.no_grad()
def sorted_unit_draws(decoder, xi, breakpoint_eps=1e-4):
    """
    Sort every draw by r_j = b_j / w_j.  Units with |w_j| <= breakpoint_eps
    are placed last.  All role-level quantities follow the same permutation.
    """

    params, semantics = decoder.unpack(
        xi,
        return_semantics=True,
    )

    valid = params["w"].abs() > float(breakpoint_eps)
    raw_breakpoint = torch.where(
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
        raw_breakpoint,
        torch.full_like(raw_breakpoint, 1e12) + unit_id,
    )
    order = torch.argsort(
        sort_key + unit_id * 1e-10,
        dim=1,
    )

    out = {
        "beta0": params["beta0"],
        "ell": params["ell"],
        "order": order,
        "breakpoint": torch.gather(raw_breakpoint, 1, order),
        "breakpoint_defined": torch.gather(valid, 1, order),
    }

    for role in decoder.role_names:
        out[role] = {
            key: torch.gather(value, 1, order)
            if key != "t"
            else value
            for key, value in semantics[role].items()
        }

    return out


@torch.no_grad()
def predict_draws(decoder, X, xi, batch_size=500):
    xi = torch.as_tensor(xi)
    X = torch.as_tensor(
        X,
        device=xi.device,
        dtype=xi.dtype,
    )
    draws = []

    for start in range(0, xi.shape[0], batch_size):
        draws.append(
            decoder(
                X,
                xi[start:start + batch_size],
            ).detach().cpu()
        )

    return torch.cat(draws, dim=0)


def function_metrics(
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
        out = {
            f"{prefix}_{name}": value
            for name, value in out.items()
        }

    return out


@torch.no_grad()
def role_diagnostics(
    decoder,
    xi,
    latent_grad=None,
    base_loc_grad=None,
    epoch=None,
):
    _, semantics = decoder.unpack(
        xi,
        return_semantics=True,
    )
    rows = []

    for role in decoder.role_names:
        item = semantics[role]
        margin = item["margin"].reshape(-1)
        gate = item["gate"].reshape(-1)
        active = item["active"]
        theta = item["theta"].abs().reshape(-1)
        active_count = active.float().sum(dim=1)
        active_gate = gate[active.reshape(-1)]
        quantiles = torch.quantile(
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
            "margin_q025": float(quantiles[0]),
            "margin_q25": float(quantiles[1]),
            "margin_q50": float(quantiles[2]),
            "margin_q75": float(quantiles[3]),
            "margin_q975": float(quantiles[4]),
            "mean_pip": float(active.float().mean()),
            "active_count_mean": float(active_count.mean()),
            "active_count_median": float(active_count.median()),
            "active_count_p_zero": float(
                (active_count == 0).float().mean()
            ),
            "gate_mean": float(gate.mean()),
            "active_gate_mean": (
                float(active_gate.mean())
                if active_gate.numel()
                else np.nan
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
                "grad_s_norm": float(
                    grad_s.norm(dim=1).mean()
                ),
                "grad_u_norm": float(
                    grad_u.norm(dim=1).mean()
                ),
                "grad_t_norm": float(
                    grad_t.norm(dim=1).mean()
                ),
            })

        if base_loc_grad is not None:
            s_slice = decoder.s_role_slices[role]
            u_slice = decoder.u_role_slices[role]
            t_id = decoder.t_role_index[role]

            row.update({
                "q0_loc_grad_s_norm": float(
                    base_loc_grad[s_slice].norm()
                ),
                "q0_loc_grad_u_norm": float(
                    base_loc_grad[
                        decoder.s_dim + u_slice.start:
                        decoder.s_dim + u_slice.stop
                    ].norm()
                ),
                "q0_loc_grad_t_norm": float(
                    base_loc_grad[
                        decoder.s_dim
                        + decoder.u_dim
                        + t_id
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

    units = (
        a[:, None, :]
        * torch.relu(
            w[:, None, :] * x[None, :, None]
            - b[:, None, :]
        )
    )
    energy = units.square().mean(dim=1)
    c_sum = energy.sum(dim=1)
    c_total = units.sum(dim=2).square().mean(dim=1)
    cancellation_ratio = c_total / (c_sum + 1e-12)
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

    q_ratio = torch.quantile(
        cancellation_ratio,
        torch.tensor(
            [0.025, 0.25, 0.50, 0.75, 0.975],
            device=xi.device,
            dtype=xi.dtype,
        ),
    )
    summary = {
        "epoch": epoch,
        "unit_energy_sum_mean": float(c_sum.mean()),
        "unit_energy_sum_median": float(c_sum.median()),
        "total_unit_energy_mean": float(c_total.mean()),
        "total_unit_energy_median": float(c_total.median()),
        "cancellation_ratio_mean": float(
            cancellation_ratio.mean()
        ),
        "cancellation_ratio_median": float(
            cancellation_ratio.median()
        ),
        "cancellation_ratio_q025": float(q_ratio[0]),
        "cancellation_ratio_q25": float(q_ratio[1]),
        "cancellation_ratio_q50": float(q_ratio[2]),
        "cancellation_ratio_q75": float(q_ratio[3]),
        "cancellation_ratio_q975": float(q_ratio[4]),
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
    """
    Spike mass and conditional-slab recovery after draw-wise breakpoint
    ordering.  Empty reference groups remain NaN and retain a zero group size.
    """

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
                .detach()
                .cpu()
                .numpy()
            )
            mcmc_theta = (
                mcmc[role]["theta"][:, j][mcmc_active]
                .detach()
                .cpu()
                .numpy()
            )

            if (
                len(rat_theta) >= min_active_draws
                and len(mcmc_theta) >= min_active_draws
            ):
                try:
                    slab_skl = kde_skl_1d(
                        rat_theta,
                        mcmc_theta,
                    )
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
                "zero_mass_js": bernoulli_js(
                    rat_pip,
                    mcmc_pip,
                ),
                "reference_group": (
                    "reference-active"
                    if mcmc_pip > reference_threshold
                    else "reference-inactive"
                ),
                "rat_active_draws": int(len(rat_theta)),
                "mcmc_active_draws": int(len(mcmc_theta)),
                "conditional_slab_skl": slab_skl,
                "rat_active_gate_mean": (
                    float(rat_gate.mean())
                    if rat_gate.numel()
                    else np.nan
                ),
                "mcmc_active_gate_mean": (
                    float(mcmc_gate.mean())
                    if mcmc_gate.numel()
                    else np.nan
                ),
                "rat_active_slab_abs_mean": (
                    float(rat_slab.mean())
                    if rat_slab.numel()
                    else np.nan
                ),
                "mcmc_active_slab_abs_mean": (
                    float(mcmc_slab.mean())
                    if mcmc_slab.numel()
                    else np.nan
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
            if ref_active.any()
            else np.nan
        ),
        "ref_active_js_median": (
            float(table.loc[ref_active, "zero_mass_js"].median())
            if ref_active.any()
            else np.nan
        ),
        "ref_inactive_js_mean": (
            float(table.loc[ref_inactive, "zero_mass_js"].mean())
            if ref_inactive.any()
            else np.nan
        ),
        "ref_inactive_js_median": (
            float(table.loc[ref_inactive, "zero_mass_js"].median())
            if ref_inactive.any()
            else np.nan
        ),
        "conditional_slab_skl_mean": (
            float(table.loc[valid_skl, "conditional_slab_skl"].mean())
            if valid_skl.any()
            else np.nan
        ),
        "conditional_slab_skl_median": (
            float(table.loc[valid_skl, "conditional_slab_skl"].median())
            if valid_skl.any()
            else np.nan
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

        for metric in metrics:
            values = pd.to_numeric(
                group[metric],
                errors="coerce",
            ).to_numpy(dtype=float)
            valid = np.isfinite(values)
            rows.append({
                **prefix,
                "metric": metric,
                "mean": (
                    float(np.nanmean(values))
                    if valid.any()
                    else np.nan
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


def plot_function_1d(
    x,
    signal,
    mcmc_pred_draws,
    rat_pred_draws,
    interval=0.95,
):
    x = np.asarray(x, dtype=float).reshape(-1)
    signal = np.asarray(signal, dtype=float).reshape(-1)
    mcmc = np.asarray(mcmc_pred_draws, dtype=float)
    rat = np.asarray(rat_pred_draws, dtype=float)
    order = np.argsort(x)
    x = x[order]
    signal = signal[order]
    mcmc = mcmc[:, order]
    rat = rat[:, order]
    alpha = 0.5 * (1.0 - interval)
    mcmc_lo, mcmc_hi = np.quantile(
        mcmc,
        [alpha, 1.0 - alpha],
        axis=0,
    )
    rat_lo, rat_hi = np.quantile(
        rat,
        [alpha, 1.0 - alpha],
        axis=0,
    )

    fig, ax = plt.subplots()
    ax.plot(x, signal, label="True function")
    ax.axhline(0.0, linestyle="--", label="Zero function")
    ax.plot(x, mcmc.mean(axis=0), label="MCMC mean")
    ax.plot(x, rat.mean(axis=0), label="RaT mean")
    ax.fill_between(x, mcmc_lo, mcmc_hi, alpha=0.15)
    ax.fill_between(x, rat_lo, rat_hi, alpha=0.15)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend()
    fig.tight_layout()

    return fig, ax