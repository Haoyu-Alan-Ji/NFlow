import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model2 import NBase, SemanticFlow


ROLE_NAMES = ("input", "breakpoint", "output")


class DirectUnitDecoder(nn.Module):
    """
    Direct one-dimensional unit model

        f(x) = beta0 + ell * x
             + sum_j a_j ReLU(w_j * x - b_j).

    beta0 and ell are ordinary continuous parameters.  Each hidden-unit role
    has its own slab coordinates, local activation coordinates, and one shared
    role threshold.  A role omitted from gate_roles is fixed open:

        theta_j = s_j.
    """

    def __init__(
        self,
        H=3,
        gate_roles=ROLE_NAMES,
        gate_power=2.0,
        gate_tau=1.0,
    ):
        super().__init__()

        self.H = int(H)
        self.role_names = ROLE_NAMES
        self.gate_roles = tuple(gate_roles)
        self.gate_power = float(gate_power)
        self.gate_tau = None if gate_tau is None else float(gate_tau)

        self.s_role_slices = {
            role: slice(2 + k * self.H, 2 + (k + 1) * self.H)
            for k, role in enumerate(self.role_names)
        }
        self.u_role_slices = {
            role: slice(k * self.H, (k + 1) * self.H)
            for k, role in enumerate(self.role_names)
        }
        self.t_role_index = {
            role: k
            for k, role in enumerate(self.role_names)
        }

        self.s_dim = 2 + 3 * self.H
        self.u_dim = 3 * self.H
        self.t_dim = 3
        self.dim = self.s_dim + self.u_dim + self.t_dim

    def unpack(self, xi, return_semantics=False):
        s = xi[:, :self.s_dim]
        u = xi[:, self.s_dim:self.s_dim + self.u_dim]
        t = xi[:, self.s_dim + self.u_dim:]

        semantics = {}

        for role in self.role_names:
            slab = s[:, self.s_role_slices[role]]
            local = u[:, self.u_role_slices[role]]
            threshold = t[
                :,
                self.t_role_index[role]:self.t_role_index[role] + 1,
            ]
            margin = local - threshold

            if role in self.gate_roles:
                positive_power = F.relu(margin).pow(self.gate_power)

                if self.gate_tau is None:
                    gate = positive_power
                else:
                    gate = positive_power / (
                        self.gate_tau ** self.gate_power
                        + positive_power
                    )

                active = margin > 0.0
            else:
                gate = torch.ones_like(margin)
                active = torch.ones_like(margin, dtype=torch.bool)

            semantics[role] = {
                "s": slab,
                "u": local,
                "t": threshold,
                "margin": margin,
                "gate": gate,
                "active": active,
                "theta": slab * gate,
            }

        params = {
            "beta0": s[:, 0],
            "ell": s[:, 1],
            "w": semantics["input"]["theta"],
            "b": semantics["breakpoint"]["theta"],
            "a": semantics["output"]["theta"],
        }

        if return_semantics:
            return params, semantics

        return params

    def unit_contributions(self, X, xi):
        params = self.unpack(xi)
        x = X[:, 0]

        hidden = F.relu(
            params["w"][:, None, :] * x[None, :, None]
            - params["b"][:, None, :]
        )

        return params["a"][:, None, :] * hidden

    def forward(self, X, xi):
        params = self.unpack(xi)
        x = X[:, 0]
        hidden = F.relu(
            params["w"][:, None, :] * x[None, :, None]
            - params["b"][:, None, :]
        )
        units = params["a"][:, None, :] * hidden

        return (
            params["beta0"][:, None]
            + params["ell"][:, None] * x[None, :]
            + units.sum(dim=2)
        )


class DirectUnitBNNVI(nn.Module):
    def __init__(
        self,
        X,
        y,
        H=3,
        family="gaussian",
        sigma2=1.0,
        gate_roles=ROLE_NAMES,
        gate_power=2.0,
        gate_tau=1.0,
        init_sd=None,
        K_flow=8,
        flow_hidden_units=64,
        flow_hidden_layers=2,
        scale_clip=1.5,
    ):
        super().__init__()

        self.register_buffer("X", X)
        self.register_buffer("y", y)
        self.register_buffer(
            "sigma2",
            torch.tensor(float(sigma2), dtype=X.dtype),
        )

        self.family = family.lower()

        self.decoder = DirectUnitDecoder(
            H=H,
            gate_roles=gate_roles,
            gate_power=gate_power,
            gate_tau=gate_tau,
        )

        self.q0 = NBase(
            self.decoder.dim,
            init_sd=init_sd,
        )

        self.flow = SemanticFlow(
            self.decoder.s_dim,
            self.decoder.u_dim,
            self.decoder.t_dim,
            K=K_flow,
            hidden_units=flow_hidden_units,
            num_hidden_layers=flow_hidden_layers,
            scale_clip=scale_clip,
        )

    def sample_posterior(self, R):
        z0 = self.q0.sample(R)
        xi, logdet = self.flow(z0, return_logdet=True)
        log_q = self.q0.log_prob(z0) - logdet
        return xi, log_q

    def log_likelihood(self, xi, X=None, y=None):
        X = self.X if X is None else X
        y = self.y if y is None else y
        pred = self.decoder(X, xi)

        if self.family == "gaussian":
            resid = y[None, :] - pred

            return -0.5 * (
                resid.square().sum(dim=1) / self.sigma2
                + y.numel()
                * torch.log(2.0 * torch.pi * self.sigma2)
            )

        if self.family in {"bernoulli", "binomial", "logistic"}:
            return -F.binary_cross_entropy_with_logits(
                pred,
                y[None, :].expand_as(pred),
                reduction="none",
            ).sum(dim=1)

        rate = torch.exp(pred.clamp(-20.0, 20.0))
        return (
            y[None, :] * pred
            - rate
            - torch.lgamma(y[None, :] + 1.0)
        ).sum(dim=1)

    def log_prior(self, xi):
        return -0.5 * (
            xi.square()
            + math.log(2.0 * math.pi)
        ).sum(dim=1)

    def elbo_draws(self, R):
        xi, log_q = self.sample_posterior(R)
        log_likelihood = self.log_likelihood(xi)
        log_prior = self.log_prior(xi)

        return {
            "xi": xi,
            "log_likelihood": log_likelihood,
            "log_prior": log_prior,
            "log_q": log_q,
            "kl": log_q - log_prior,
            "elbo": log_likelihood + log_prior - log_q,
        }

    def neg_elbo(self, R=64):
        return -self.elbo_draws(R)["elbo"].mean()

    @torch.no_grad()
    def predict(self, X_new, R=1000):
        xi, _ = self.sample_posterior(R)
        pred = self.decoder(X_new, xi)

        if self.family == "gaussian":
            return pred.mean(dim=0)

        if self.family in {"bernoulli", "binomial", "logistic"}:
            return torch.sigmoid(pred).mean(dim=0)

        return torch.exp(pred.clamp(-20.0, 20.0)).mean(dim=0)


def run_direct_bnn_mcmc(
    model,
    N=10000,
    S_max=100,
    burnin=2000,
    thin=1,
    seed=123,
    print_every=500,
):
    """
    Coordinate-wise elliptical slice sampler under the same decoder,
    likelihood, gate, and N(0, I) semantic-coordinate prior as DirectUnitBNNVI.
    """

    rng = np.random.default_rng(seed)
    device = model.X.device
    dtype = model.X.dtype
    d = model.decoder.dim

    state = np.zeros(d)
    draws = np.empty((N, d))
    slice_steps = np.empty((N, d))

    @torch.no_grad()
    def log_likelihood(state_np):
        xi = torch.as_tensor(
            state_np,
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
            prior_direction = rng.normal()
            proposal = state.copy()

            for step in range(1, S_max + 1):
                proposal[j] = (
                    state[j] * math.cos(angle)
                    + prior_direction * math.sin(angle)
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
        "sigma2": float(model.sigma2),
        "gate_roles": model.decoder.gate_roles,
        "H": model.decoder.H,
    }