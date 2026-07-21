import math
import numpy as np
import torch
import torch.nn.functional as F


def run_bnn_mcmc(
    decoder,
    X,
    y,
    sim_info,
    N=2000,
    S_max=100,
    burnin=500,
    thin=1,
    beta_eps=0.05,
    seed=123,
    print_every=100,
):
    """
    Coordinate-wise ESS MCMC reference for DSSAttentionFFNDecoder.

    Samples semantic coordinates:
        xi = (s, u, t)

    The decoder handles the complete mapping:
        xi -> theta -> prediction

    PIP:
        Pr(u > t | data)

    ePIP:
        Pr(|theta| > beta_eps | data)
    """

    rng = np.random.default_rng(seed)

    device = X.device
    dtype = X.dtype
    d = decoder.dim

    fam = str(sim_info["family"]).lower()
    sigma2 = float(sim_info["sigma2"]) if fam == "gaussian" else None

    # -----------------------------
    # 1. Initial semantic state
    # -----------------------------

    b_c = np.zeros(d, dtype=float)

    # -----------------------------
    # 2. Log likelihood
    # -----------------------------

    @torch.no_grad()
    def LL(b):
        xi = torch.as_tensor(
            b,
            device=device,
            dtype=dtype,
        )[None, :]

        pred = decoder(X=X, xi=xi)[0]

        if fam == "gaussian":
            resid = y - pred

            out = -0.5 * (
                resid.square().sum() / sigma2
                + y.numel() * math.log(2.0 * math.pi * sigma2)
            )

            return float(out.item())

        if fam in {"bernoulli", "binomial", "logistic"}:
            out = -F.binary_cross_entropy_with_logits(
                pred,
                y,
                reduction="sum",
            )

            return float(out.item())

        if fam == "poisson":
            log_rate = pred
            rate = torch.exp(
                torch.clamp(log_rate, min=-20.0, max=20.0)
            )

            out = (
                y * log_rate
                - rate
                - torch.lgamma(y + 1.0)
            ).sum()

            return float(out.item())

        raise ValueError(
            "family must be gaussian, bernoulli, logistic, "
            "binomial, or poisson."
        )

    # -----------------------------
    # 3. Coordinate-wise ESS
    # -----------------------------

    sd_0 = np.ones(d, dtype=float)

    mc_b = np.full((N, d), np.nan, dtype=float)
    n_s = np.full((N, d), np.nan, dtype=float)

    ll_c = LL(b_c)

    for i in range(N):
        for j in range(d):
            level = ll_c + math.log(rng.uniform())

            th = rng.uniform(0.0, 2.0 * math.pi)
            lower = th - 2.0 * math.pi
            upper = th

            nu = rng.normal(0.0, sd_0[j])
            b_p = b_c.copy()

            ns = 0

            while ns < S_max:
                b_p[j] = (
                    b_c[j] * math.cos(th)
                    + nu * math.sin(th)
                )

                ns += 1
                ll_p = LL(b_p)

                if ll_p > level:
                    b_c = b_p.copy()
                    ll_c = ll_p
                    break

                if th < 0.0:
                    lower = th
                else:
                    upper = th

                th = rng.uniform(lower, upper)

            n_s[i, j] = ns

        mc_b[i] = b_c

        if print_every is not None and (
            (i + 1) % print_every == 0 or i == 0
        ):
            print(
                f"mcmc_iter={i + 1:05d} "
                f"loglik={ll_c:.3f}"
            )

    # -----------------------------
    # 4. Keep posterior draws
    # -----------------------------

    keep = np.arange(burnin, N, thin)

    xi_draws_np = mc_b[keep]
    xi_draws = torch.as_tensor(
        xi_draws_np,
        device=device,
        dtype=dtype,
    )

    R = xi_draws.shape[0]

    # -----------------------------
    # 5. Decode semantic draws
    # -----------------------------

    with torch.no_grad():
        params = decoder.unpack(
            xi_draws,
            return_summary=False,
        )

    theta_parts = []
    theta_names = []

    active_parts = []

    u = xi_draws[
        :,
        decoder.s_dim:
        decoder.s_dim + decoder.u_dim,
    ]

    t = xi_draws[
        :,
        decoder.s_dim + decoder.u_dim:,
    ]

    for item in decoder.param_specs:
        key = item["name"]
        sl = slice(item["start"], item["end"])

        value = params[key].reshape(R, -1)
        theta_parts.append(value.detach().cpu())

        margin = (
            u[:, sl]
            - t[:, item["t"]:item["t"] + 1]
        )

        active_parts.append(
            (margin > 0.0).detach().cpu()
        )

        for local_id in range(value.shape[1]):
            theta_names.append(f"{key}_{local_id}")

    theta_draws = torch.cat(
        theta_parts,
        dim=1,
    ).numpy()

    active_draw = torch.cat(
        active_parts,
        dim=1,
    ).numpy()

    # -----------------------------
    # 6. Posterior summaries
    # -----------------------------

    pip = active_draw.mean(axis=0)
    selected = (pip > 0.5).astype(int)

    effect_active_draw = (
        np.abs(theta_draws) > float(beta_eps)
    )

    epip = effect_active_draw.mean(axis=0)
    effect_selected = (epip > 0.5).astype(int)

    effect_mean = theta_draws.mean(axis=0)
    effect_sd = theta_draws.std(axis=0, ddof=1)

    t_draws = xi_draws_np[
        :,
        decoder.s_dim + decoder.u_dim:,
    ]

    threshold_names = [
        f"t_{item['name']}"
        for item in decoder.param_specs
    ]

    out = {
        "xi_draws": xi_draws_np,
        "theta_draws": theta_draws,
        "theta_names": theta_names,

        "active_draw": active_draw,
        "pip": pip,
        "selected": selected,

        "effect_active_draw": effect_active_draw,
        "epip": epip,
        "effect_selected": effect_selected,

        "effect_mean": effect_mean,
        "effect_sd": effect_sd,

        "t_draws": t_draws,
        "threshold_names": threshold_names,
        "threshold_mean": t_draws.mean(axis=0),
        "threshold_sd": t_draws.std(axis=0, ddof=1),

        "n_s": n_s,
        "beta_eps": float(beta_eps),
        "burnin": int(burnin),
        "thin": int(thin),
        "n_kept": int(R),

        "family": fam,
        "sigma2": sigma2,

        "sim_info": sim_info,
        "function": sim_info["function"],
        "snr": float(sim_info["snr"]),
        "active_idx": sim_info["active_idx"],
        "n_active": int(sim_info["n_active"]),
    }

    return out