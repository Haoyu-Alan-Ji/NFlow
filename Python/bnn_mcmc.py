import math
import time

import numpy as np
import torch


def run_bnn_mcmc(
    model,
    N=10000,
    S_max=100,
    burnin=2000,
    thin=1,
    seed=123,
    print_every=500,
    initial_state=None,
    chain_id=None,
):
    """
    Coordinate-wise elliptical slice sampling for N(0, I) continuous LVR
    coordinates. The likelihood and decoder are taken directly from model, so
    MCMC and VI use exactly the same forward path and observation model.
    """

    rng = np.random.default_rng(seed)
    device = model.X.device
    dtype = model.X.dtype
    d = model.decoder.dim

    if isinstance(initial_state, str):
        if initial_state != "prior":
            raise ValueError("initial_state string must be 'prior'.")
        state = rng.normal(size=d)
    elif initial_state is None:
        state = np.zeros(d)
    else:
        state = np.asarray(initial_state, dtype=float).reshape(d).copy()
    draws = np.empty((N, d))
    slice_steps = np.empty((N, d))

    @torch.no_grad()
    def log_likelihood(value):
        xi = torch.as_tensor(
            value,
            device=device,
            dtype=dtype,
        )[None, :]

        return float(model.log_likelihood(xi).item())

    current_ll = log_likelihood(state)
    started = time.perf_counter()

    for i in range(N):
        for j in range(d):
            level = current_ll + math.log(rng.uniform())
            angle = rng.uniform(0.0, 2.0 * math.pi)
            lower = angle - 2.0 * math.pi
            upper = angle
            direction = rng.normal()
            proposal = state.copy()

            for step in range(1, S_max + 1):
                proposal[j] = (
                    state[j] * math.cos(angle)
                    + direction * math.sin(angle)
                )
                proposal_ll = log_likelihood(proposal)

                if proposal_ll > level:
                    state = proposal.copy()
                    current_ll = proposal_ll
                    break

                if angle < 0.0:
                    lower = angle
                else:
                    upper = angle

                angle = rng.uniform(lower, upper)

            slice_steps[i, j] = step

        draws[i] = state

        if print_every is not None and (
            i == 0 or (i + 1) % print_every == 0
        ):
            prefix = "" if chain_id is None else f"chain={int(chain_id)} "
            print(
                f"{prefix}mcmc_iter={i + 1:05d} "
                f"loglik={current_ll:.3f}"
            )

    keep = np.arange(burnin, N, thin)
    elapsed = time.perf_counter() - started

    return {
        "xi_draws": draws[keep],
        "n_s": slice_steps,
        "burnin": int(burnin),
        "thin": int(thin),
        "n_kept": int(len(keep)),
        "sampling_time_sec": float(elapsed),
        "sec_per_iteration": float(elapsed / int(N)),
        "family": model.family,
        "sigma2": float(model.sigma2.item()),
        "decoder": type(model.decoder).__name__,
        "selection_mode": getattr(model.decoder, "selection_mode", "edge"),
        "latent_dim": int(d),
        "decoder_signature": (
            model.decoder.compatibility_signature()
            if hasattr(model.decoder, "compatibility_signature")
            else None
        ),
        "seed": int(seed),
        "chain_id": None if chain_id is None else int(chain_id),
    }


def run_bnn_mcmc_chains(
    model,
    *,
    n_chains=4,
    chain_seeds=None,
    initial_state="prior",
    **mcmc_kwargs,
):
    """Run independent decoder-identical ESS chains and retain chain shape."""

    n_chains = int(n_chains)
    if n_chains < 2:
        raise ValueError("Use at least two chains for R-hat diagnostics.")
    if chain_seeds is None:
        base_seed = int(mcmc_kwargs.pop("seed", 123))
        chain_seeds = [base_seed + 1009 * chain for chain in range(n_chains)]
    else:
        chain_seeds = [int(value) for value in chain_seeds]
        if len(chain_seeds) != n_chains:
            raise ValueError("len(chain_seeds) must equal n_chains.")

    started = time.perf_counter()
    chains = []
    for chain, seed in enumerate(chain_seeds, start=1):
        print(f"--- MCMC chain {chain}/{n_chains}; seed={seed} ---")
        chains.append(run_bnn_mcmc(
            model,
            seed=seed,
            initial_state=initial_state,
            chain_id=chain,
            **mcmc_kwargs,
        ))
    wall_time = time.perf_counter() - started
    chain_xi = np.stack([item["xi_draws"] for item in chains], axis=0)
    slice_steps = np.stack([item["n_s"] for item in chains], axis=0)

    result = {
        key: value
        for key, value in chains[0].items()
        if key not in {
            "xi_draws", "n_s", "sampling_time_sec", "sec_per_iteration",
            "seed", "chain_id",
        }
    }
    result.update({
        "xi_draws": chain_xi.reshape(-1, chain_xi.shape[-1]),
        "chain_xi_draws": chain_xi,
        "n_s": slice_steps,
        "n_chains": n_chains,
        "draws_per_chain": int(chain_xi.shape[1]),
        "chain_seeds": chain_seeds,
        "chain_sampling_time_sec": [
            float(item["sampling_time_sec"]) for item in chains
        ],
        "sampling_time_sec": float(sum(
            item["sampling_time_sec"] for item in chains
        )),
        "wall_time_sec": float(wall_time),
        "sec_per_iteration": float(np.mean([
            item["sec_per_iteration"] for item in chains
        ])),
    })
    return result


def _chain_array(values):
    values = np.asarray(values, dtype=float)
    if values.ndim == 2:
        values = values[..., None]
    if values.ndim != 3:
        raise ValueError("Chain values must have shape [chains, draws, variables].")
    if values.shape[0] < 2 or values.shape[1] < 4:
        raise ValueError("Diagnostics require at least 2 chains and 4 draws.")
    return values


def split_rhat(values):
    """Classical split R-hat, returned once per final coordinate."""

    values = _chain_array(values)
    half = values.shape[1] // 2
    split = np.concatenate([values[:, :half], values[:, -half:]], axis=0)
    n = split.shape[1]
    chain_means = split.mean(axis=1)
    chain_vars = split.var(axis=1, ddof=1)
    within = chain_vars.mean(axis=0)
    between = n * chain_means.var(axis=0, ddof=1)
    variance = ((n - 1.0) / n) * within + between / n
    out = np.full_like(within, np.nan, dtype=float)
    varying = within > np.finfo(float).eps
    out[varying] = np.sqrt(variance[varying] / within[varying])
    constant = (~varying) & (between <= np.finfo(float).eps)
    out[constant] = 1.0
    return out


def effective_sample_size(values):
    """Approximate multi-chain ESS using an initial positive sequence."""

    values = _chain_array(values)
    n_chains, n_draws, n_variables = values.shape
    centered = values - values.mean(axis=1, keepdims=True)
    fft_size = 1 << int(np.ceil(np.log2(2 * n_draws)))
    spectrum = np.fft.rfft(centered, n=fft_size, axis=1)
    autocov = np.fft.irfft(
        spectrum * np.conjugate(spectrum), n=fft_size, axis=1
    )[:, :n_draws]
    autocov = autocov / np.arange(n_draws, 0, -1)[None, :, None]

    within = values.var(axis=1, ddof=1).mean(axis=0)
    between = n_draws * values.mean(axis=1).var(axis=0, ddof=1)
    var_plus = ((n_draws - 1.0) / n_draws) * within + between / n_draws
    mean_autocov = autocov.mean(axis=0)
    rho = np.zeros((n_draws, n_variables), dtype=float)
    rho[0] = 1.0
    valid = var_plus > np.finfo(float).eps
    rho[1:, valid] = 1.0 - (
        within[valid][None, :] - mean_autocov[1:, valid]
    ) / var_plus[valid][None, :]

    total = n_chains * n_draws
    ess = np.full(n_variables, float(total))
    for variable in range(n_variables):
        if not valid[variable]:
            continue
        pair_sum = 0.0
        previous = np.inf
        lag = 0
        while lag + 1 < n_draws:
            pair = rho[lag, variable] + rho[lag + 1, variable]
            if not np.isfinite(pair) or pair < 0.0:
                break
            pair = min(pair, previous)
            pair_sum += pair
            previous = pair
            lag += 2
        tau = max(-1.0 + 2.0 * pair_sum, 1.0)
        ess[variable] = min(float(total), float(total) / tau)
    return ess


def chain_diagnostics(values, names=None):
    """R-hat, ESS and MCSE rows for scalar chain summaries."""

    values = _chain_array(values)
    n_variables = values.shape[-1]
    if names is None:
        names = [f"value_{index}" for index in range(n_variables)]
    if len(names) != n_variables:
        raise ValueError("names length must match the final chain dimension.")
    rhat = split_rhat(values)
    ess = effective_sample_size(values)
    pooled = values.reshape(-1, n_variables)
    sd = pooled.std(axis=0, ddof=1)
    mcse = sd / np.sqrt(np.maximum(ess, 1.0))
    return [{
        "variable": str(name),
        "mean": float(pooled[:, index].mean()),
        "sd": float(sd[index]),
        "rhat": float(rhat[index]),
        "ess": float(ess[index]),
        "mcse": float(mcse[index]),
    } for index, name in enumerate(names)]


run_direct_bnn_mcmc = run_bnn_mcmc
