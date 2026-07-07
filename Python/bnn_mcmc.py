import math
import numpy as np
import torch
import torch.nn.functional as F


def run_bnn_mcmc(
    decoder,
    X,
    y,
    family="gaussian",
    sigma2=1.0,
    truth=None,
    N=2000,
    S_max=100,
    burnin=500,
    thin=1,
    beta_eps=0.05,
    seed=123,
    print_every=100,
):
    """
    ESS-within-Gibbs MCMC reference for DSSAttentionFFNDecoder.

    Samples semantic coordinates xi = (s, u, t), then decodes
    theta = D(xi).

    Works with decoder.param_specs and truth keys:
        E, e,
        W1_0, b1_0, W2_0, b2_0,
        ...
        Wout, bout.
    """

    rng = np.random.default_rng(seed)

    device = X.device
    dtype = X.dtype
    d = decoder.dim

    fam = str(family).lower()

    # -----------------------------
    # 1. Initial xi
    # -----------------------------

    xi0 = torch.zeros(d, device=device, dtype=dtype)

    if truth is not None:
        theta0_parts = []
        lambda_parts = []

        for item in decoder.param_specs:
            key = item["name"]

            if key not in truth:
                raise KeyError(f"truth is missing key: {key}")

            val = truth[key].detach().to(device=device, dtype=dtype).reshape(-1)

            theta0_parts.append(val)
            lambda_parts.append(torch.full_like(val, float(item["lambda"])))

        theta0 = torch.cat(theta0_parts)
        lambda_flat = torch.cat(lambda_parts)

        s0 = xi0[:decoder.s_dim]
        u0 = xi0[decoder.s_dim:decoder.s_dim + decoder.u_dim]
        t0 = xi0[decoder.s_dim + decoder.u_dim:]

        active = theta0.abs() > 1e-12

        t0.zero_()

        # Since theta = lambda * s * relu(u - t),
        # setting t=0, u=1 for active gives theta = lambda * s.
        u0[active] = 1.0
        u0[~active] = -1.0

        s0[active] = theta0[active] / lambda_flat[active]
        s0[~active] = 0.0

    b_c = xi0.detach().cpu().numpy().astype(float)

    # -----------------------------
    # 2. Log likelihood
    # -----------------------------

    @torch.no_grad()
    def LL(b):
        xi = torch.as_tensor(b, device=device, dtype=dtype)[None, :]
        pred = decoder(X=X, xi=xi)[0]

        if fam == "gaussian":
            resid = y - pred

            out = -0.5 * (
                resid.pow(2).sum() / float(sigma2)
                + y.numel() * math.log(2.0 * math.pi * float(sigma2))
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
            rate = torch.exp(torch.clamp(log_rate, min=-20.0, max=20.0))

            out = (
                y * log_rate
                - rate
                - torch.lgamma(y + 1.0)
            ).sum()

            return float(out.item())

        raise ValueError("family must be gaussian, bernoulli, logistic, binomial, or poisson.")

    # -----------------------------
    # 3. ESS-within-Gibbs
    # -----------------------------

    sd_0 = np.ones(d, dtype=float)

    mc_b = np.full((N, d), np.nan, dtype=float)
    n_s = np.full((N, d), np.nan, dtype=float)

    for i in range(N):
        for j in range(d):
            ns = 0

            level = LL(b_c) + math.log(rng.uniform())

            th = rng.uniform(0.0, 2.0 * math.pi)
            a = th - 2.0 * math.pi
            A = th

            nu = rng.normal(0.0, sd_0[j])
            b_p = b_c.copy()

            while ns < S_max:
                b_p[j] = b_c[j] * math.cos(th) + nu * math.sin(th)
                ns += 1

                if LL(b_p) > level:
                    b_c = b_p.copy()
                    break

                if th < 0:
                    a = th
                else:
                    A = th

                th = rng.uniform(a, A)

            n_s[i, j] = ns

        mc_b[i, :] = b_c

        if print_every is not None and ((i + 1) % print_every == 0 or i == 0):
            print(f"mcmc_iter={i + 1:05d} loglik={LL(b_c):.3f}")

    # -----------------------------
    # 4. Keep posterior draws
    # -----------------------------

    keep = np.arange(burnin, N, thin)

    xi_draws_np = mc_b[keep, :]
    xi_draws = torch.as_tensor(xi_draws_np, device=device, dtype=dtype)

    # -----------------------------
    # 5. Decode xi draws to theta draws
    # -----------------------------

    with torch.no_grad():
        params = decoder.unpack(xi_draws, return_summary=False)

        theta_parts = []
        theta_names = []

        for item in decoder.param_specs:
            key = item["name"]

            val = params[key].reshape(xi_draws.shape[0], -1)

            theta_parts.append(val.detach().cpu())

            for local_id in range(val.shape[1]):
                theta_names.append(f"{key}_{local_id}")

        theta_draws = torch.cat(theta_parts, dim=1).numpy()

    # -----------------------------
    # 6. Practical inclusion summaries
    # -----------------------------

    active_draw = np.abs(theta_draws) > float(beta_eps)

    pip = active_draw.mean(axis=0)
    selected = (pip > 0.5).astype(int)

    effect_mean = theta_draws.mean(axis=0)
    effect_sd = theta_draws.std(axis=0, ddof=1)

    out = {
        "xi_draws": xi_draws_np,
        "theta_draws": theta_draws,
        "theta_names": theta_names,
        "pip": pip,
        "selected": selected,
        "effect_mean": effect_mean,
        "effect_sd": effect_sd,
        "active_draw": active_draw,
        "n_s": n_s,
        "beta_eps": float(beta_eps),
        "burnin": int(burnin),
        "thin": int(thin),
        "n_kept": int(theta_draws.shape[0]),
        "family": fam,
        "sigma2": float(sigma2) if fam == "gaussian" else None,
    }

    return out