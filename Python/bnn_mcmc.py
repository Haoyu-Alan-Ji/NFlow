import math

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
):
    """
    Coordinate-wise elliptical slice sampling for N(0, I) semantic
    coordinates. The likelihood and decoder are taken directly from model, so
    MCMC and VI use exactly the same forward path and observation model.
    """

    rng = np.random.default_rng(seed)
    device = model.X.device
    dtype = model.X.dtype
    d = model.decoder.dim

    state = np.zeros(d)
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
            print(
                f"mcmc_iter={i + 1:05d} "
                f"loglik={current_ll:.3f}"
            )

    keep = np.arange(burnin, N, thin)

    return {
        "xi_draws": draws[keep],
        "n_s": slice_steps,
        "burnin": int(burnin),
        "thin": int(thin),
        "n_kept": int(len(keep)),
        "family": model.family,
        "sigma2": float(model.sigma2.item()),
        "decoder": type(model.decoder).__name__,
    }


run_direct_bnn_mcmc = run_bnn_mcmc
