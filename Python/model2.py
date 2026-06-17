import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import normflows as nf


def beta_config(p, beta_mode="sigmoid", group_ids=None, group_sizes=None, device=None):
    """
    Return:
        s_dim, u_dim, dim, normalized_group_ids

    sigmoid:
        xi = [s, u, t]
        s in R^p, u in R^p, t in R
        beta_j = s_j * sigmoid((u_j - t) / tau)
        dim = 2p + 1

    relu:
        xi = [s, u, t]
        s in R^p, u in R^p, t in R
        beta_j = s_j * (u_j - t)_+
        dim = 2p + 1

    group_relu:
        xi = [s, u, t]
        s in R^p, u in R^G, t in R
        beta_j = s_j * (u_{g(j)} - t)_+
        dim = p + G + 1

    In group_relu, grouping is applied to the threshold coordinate u, not to s.
    This preserves feature-specific coefficient signs and magnitudes while enforcing
    group-level activation.
    """
    p = int(p)

    if beta_mode in {"sigmoid", "relu"}:
        return p, p, 2 * p + 1, None

    if beta_mode != "group_relu":
        raise ValueError(f"Unknown beta_mode: {beta_mode}")

    if group_sizes is not None:
        if group_ids is not None:
            raise ValueError("Provide either group_ids or group_sizes, not both.")

        sizes = [int(size) for size in group_sizes]
        if any(size <= 0 for size in sizes):
            raise ValueError("All group_sizes must be positive.")
        if sum(sizes) != p:
            raise ValueError(f"sum(group_sizes)={sum(sizes)} must equal p={p}.")

        gids = torch.cat([
            torch.full((size,), g, dtype=torch.long, device=device)
            for g, size in enumerate(sizes)
        ])
    else:
        if group_ids is None:
            raise ValueError("group_relu requires group_ids or group_sizes.")

        gids = torch.as_tensor(group_ids, dtype=torch.long, device=device)
        if gids.ndim != 1:
            raise ValueError("group_ids must be a one-dimensional vector.")
        if gids.numel() != p:
            raise ValueError(f"len(group_ids)={gids.numel()} must equal p={p}.")

    unique = torch.unique(gids, sorted=True)
    new_gids = torch.empty_like(gids)
    for g, old_g in enumerate(unique):
        new_gids[gids == old_g] = g

    G = int(len(unique))

    # Correct group_relu layout: s is feature-level, u is group-level.
    s_dim = p
    u_dim = G
    dim = p + G + 1

    return s_dim, u_dim, dim, new_gids


class SemanticAffineCoupling(nn.Module):
    """
    Affine coupling layer on semantic blocks xi = [s, u, t].

    Supported layouts:
        ungrouped: s in R^p, u in R^p, t in R
        grouped:   s in R^p, u in R^G, t in R

    Modes:
        - "s": update s conditional on (u, t)
        - "u": update u conditional on (s, t)
        - "t": update t conditional on (s, u)
    """
    def __init__(
        self,
        p=None,
        mode=None,
        s_dim=None,
        u_dim=None,
        hidden_units=128,
        num_hidden_layers=2,
        scale_clip=2.0,
    ):
        super().__init__()
        assert mode in {"s", "u", "t"}

        if s_dim is None or u_dim is None:
            if p is None:
                raise ValueError("Provide either p or both s_dim and u_dim.")
            s_dim = int(p)
            u_dim = int(p)

        self.s_dim = int(s_dim)
        self.u_dim = int(u_dim)
        self.t_dim = 1
        self.dim = self.s_dim + self.u_dim + self.t_dim
        self.mode = mode
        self.scale_clip = float(scale_clip)

        if self.s_dim <= 0 or self.u_dim <= 0:
            raise ValueError("s_dim and u_dim must be positive.")

        if mode == "s":
            cond_dim = self.u_dim + self.t_dim
            trans_dim = self.s_dim
        elif mode == "u":
            cond_dim = self.s_dim + self.t_dim
            trans_dim = self.u_dim
        else:
            cond_dim = self.s_dim + self.u_dim
            trans_dim = self.t_dim

        widths = [cond_dim] + [hidden_units] * num_hidden_layers + [2 * trans_dim]
        self.net = nf.nets.MLP(widths, init_zeros=True)

    def _split(self, x):
        s_end = self.s_dim
        u_end = self.s_dim + self.u_dim
        s = x[:, :s_end]
        u = x[:, s_end:u_end]
        t = x[:, u_end:u_end + 1]
        return s, u, t

    def _merge(self, s, u, t):
        return torch.cat([s, u, t], dim=-1)

    def _affine_params(self, cond):
        h = self.net(cond)
        log_scale, shift = torch.chunk(h, 2, dim=-1)
        log_scale = self.scale_clip * torch.tanh(log_scale / self.scale_clip)
        return log_scale, shift

    def forward(self, x, return_logdet=False):
        s, u, t = self._split(x)

        if self.mode == "s":
            cond = torch.cat([u, t], dim=-1)
            log_scale, shift = self._affine_params(cond)
            s = s * torch.exp(log_scale) + shift
            logdet = log_scale.sum(dim=-1)
        elif self.mode == "u":
            cond = torch.cat([s, t], dim=-1)
            log_scale, shift = self._affine_params(cond)
            u = u * torch.exp(log_scale) + shift
            logdet = log_scale.sum(dim=-1)
        else:
            cond = torch.cat([s, u], dim=-1)
            log_scale, shift = self._affine_params(cond)
            t = t * torch.exp(log_scale) + shift
            logdet = log_scale.sum(dim=-1)

        y = self._merge(s, u, t)
        if return_logdet:
            return y, logdet
        return y

    def inverse(self, y, return_logdet=False):
        s, u, t = self._split(y)

        if self.mode == "s":
            cond = torch.cat([u, t], dim=-1)
            log_scale, shift = self._affine_params(cond)
            s = (s - shift) * torch.exp(-log_scale)
            logdet = (-log_scale).sum(dim=-1)
        elif self.mode == "u":
            cond = torch.cat([s, t], dim=-1)
            log_scale, shift = self._affine_params(cond)
            u = (u - shift) * torch.exp(-log_scale)
            logdet = (-log_scale).sum(dim=-1)
        else:
            cond = torch.cat([s, u], dim=-1)
            log_scale, shift = self._affine_params(cond)
            t = (t - shift) * torch.exp(-log_scale)
            logdet = (-log_scale).sum(dim=-1)

        x = self._merge(s, u, t)
        if return_logdet:
            return x, logdet
        return x


class FlowMap(nn.Module):
    def __init__(
        self,
        p=None,
        dim=None,
        s_dim=None,
        u_dim=None,
        K=4,
        hidden_units=128,
        num_hidden_layers=2,
        scale_clip=2.0,
    ):
        super().__init__()

        if s_dim is None or u_dim is None:
            if p is not None:
                s_dim = int(p)
                u_dim = int(p)
            elif dim is not None:
                dim = int(dim)
                if (dim - 1) % 2 != 0:
                    raise ValueError("If s_dim/u_dim are omitted, dim must equal 2*p + 1.")
                p = (dim - 1) // 2
                s_dim = p
                u_dim = p
            else:
                raise ValueError("Provide either p, dim, or both s_dim and u_dim.")

        self.s_dim = int(s_dim)
        self.u_dim = int(u_dim)
        self.t_dim = 1
        self.dim = self.s_dim + self.u_dim + self.t_dim

        if dim is not None and int(dim) != self.dim:
            raise ValueError(f"dim={dim} is inconsistent with s_dim + u_dim + 1 = {self.dim}.")

        layers = []
        for _ in range(K):
            layers.append(SemanticAffineCoupling(
                s_dim=self.s_dim,
                u_dim=self.u_dim,
                mode="s",
                hidden_units=hidden_units,
                num_hidden_layers=num_hidden_layers,
                scale_clip=scale_clip,
            ))
            layers.append(SemanticAffineCoupling(
                s_dim=self.s_dim,
                u_dim=self.u_dim,
                mode="u",
                hidden_units=hidden_units,
                num_hidden_layers=num_hidden_layers,
                scale_clip=scale_clip,
            ))
            layers.append(SemanticAffineCoupling(
                s_dim=self.s_dim,
                u_dim=self.u_dim,
                mode="t",
                hidden_units=hidden_units,
                num_hidden_layers=num_hidden_layers,
                scale_clip=scale_clip,
            ))

        self.layers = nn.ModuleList(layers)

    def forward(self, x, return_logdet=False):
        z = x
        if return_logdet:
            total_logdet = x.new_zeros(x.shape[0])

        for layer in self.layers:
            if return_logdet:
                z, logdet = layer(z, return_logdet=True)
                total_logdet = total_logdet + logdet
            else:
                z = layer(z)

        if return_logdet:
            return z, total_logdet
        return z

    def inverse(self, z, return_logdet=False):
        x = z
        if return_logdet:
            total_logdet = z.new_zeros(z.shape[0])

        for layer in reversed(self.layers):
            if return_logdet:
                x, logdet = layer.inverse(x, return_logdet=True)
                total_logdet = total_logdet + logdet
            else:
                x = layer.inverse(x)

        if return_logdet:
            return x, total_logdet
        return x


class Relaxedsas(nn.Module):
    def __init__(
        self,
        X,
        y,
        sigma2,
        tau,
        g_theta,
        family="gaussian",
        beta_mode="sigmoid",
        group_ids=None,
    ):
        super().__init__()

        if g_theta is None:
            raise ValueError("g_theta must be provided.")
        if beta_mode not in {"sigmoid", "relu", "group_relu"}:
            raise ValueError(f"Unknown beta_mode: {beta_mode}")

        self.register_buffer("X", X)
        self.register_buffer("y", y)
        self.register_buffer(
            "tau",
            torch.tensor(float(tau), dtype=X.dtype, device=X.device),
        )

        self.family = family
        self.beta_mode = beta_mode
        self.n, self.p = X.shape
        self.g_theta = g_theta
        self.s_dim = int(g_theta.s_dim)
        self.u_dim = int(g_theta.u_dim)
        self.dim = int(g_theta.dim)

        if beta_mode == "group_relu":
            if group_ids is None:
                raise ValueError("group_relu requires group_ids.")
            group_ids = torch.as_tensor(group_ids, dtype=torch.long, device=X.device)
            if group_ids.ndim != 1:
                raise ValueError("group_ids must be one-dimensional.")
            if group_ids.numel() != self.p:
                raise ValueError(f"len(group_ids)={group_ids.numel()} must equal p={self.p}.")
            self.register_buffer("group_ids", group_ids)
            self.G = int(group_ids.max().item()) + 1
            if self.s_dim != self.p:
                raise ValueError(f"group_relu requires s_dim=p={self.p}, got s_dim={self.s_dim}.")
            if self.u_dim != self.G:
                raise ValueError(f"group_relu requires u_dim=G={self.G}, got u_dim={self.u_dim}.")
        else:
            self.group_ids = None
            self.G = None
            if self.s_dim != self.p or self.u_dim != self.p:
                raise ValueError(
                    f"{beta_mode} requires s_dim=u_dim=p={self.p}, "
                    f"got s_dim={self.s_dim}, u_dim={self.u_dim}."
                )

        if family == "gaussian":
            self.register_buffer(
                "sigma2",
                torch.tensor(float(sigma2), dtype=X.dtype, device=X.device),
            )
        else:
            self.sigma2 = None

    def set_tau(self, tau):
        self.tau.fill_(float(tau))

    def _split_xi(self, xi):
        s_end = self.s_dim
        u_end = self.s_dim + self.u_dim
        s = xi[:, :s_end]
        u = xi[:, s_end:u_end]
        t = xi[:, u_end:u_end + 1]
        return s, u, t

    def decode(self, eps):
        xi = self.g_theta(eps)
        s, u, t = self._split_xi(xi)

        if self.beta_mode == "sigmoid":
            margin = u - t
            gate = torch.sigmoid(margin / self.tau)
            active = (margin > 0.0).to(gate.dtype)
            beta = s * gate
            return {
                "eps": eps,
                "xi": xi,
                "s": s,
                "u": u,
                "t": t,
                "margin": margin,
                "gate": gate,
                "active": active,
                "beta": beta,
            }

        if self.beta_mode == "relu":
            margin = u - t
            gate = F.relu(margin)
            active = (margin > 0.0).to(gate.dtype)
            beta = s * gate
            return {
                "eps": eps,
                "xi": xi,
                "s": s,
                "u": u,
                "t": t,
                "margin": margin,
                "gate": gate,
                "active": active,
                "beta": beta,
            }

        # group_relu:
        # s is feature-level: [R, p]
        # u is group-level:   [R, G]
        # gate_j = (u_{g(j)} - t)_+
        group_margin = u - t
        group_gate = F.relu(group_margin)
        group_active = (group_margin > 0.0).to(group_gate.dtype)

        margin = group_margin[:, self.group_ids]
        gate = group_gate[:, self.group_ids]
        active = group_active[:, self.group_ids]
        beta = s * gate

        return {
            "eps": eps,
            "xi": xi,
            "s": s,
            "u": u,
            "t": t,
            "margin": margin,
            "gate": gate,
            "active": active,
            "group_margin": group_margin,
            "group_gate": group_gate,
            "group_active": group_active,
            "group_ids": self.group_ids,
            "beta": beta,
        }

    def log_joint(self, eps):
        dec = self.decode(eps)
        beta = dec["beta"]
        eta = self.X @ beta.T

        if self.family == "gaussian":
            resid = self.y[:, None] - eta
            loglik = -0.5 * (
                resid.pow(2).sum(dim=0) / self.sigma2
                + self.n * torch.log(2.0 * torch.pi * self.sigma2)
            )
        elif self.family == "poisson":
            y = self.y[:, None]
            loglik = (y * eta - torch.exp(eta) - torch.lgamma(y + 1.0)).sum(dim=0)
        elif self.family in {"bernoulli", "binomial", "logistic"}:
            y = self.y[:, None].expand_as(eta)

            loglik = -F.binary_cross_entropy_with_logits(
                eta,
                y,
                reduction="none",
            ).sum(dim=0)
        else:
            raise ValueError(f"Unknown family: {self.family}")

        log_p0_eps = -0.5 * (
            eps.pow(2) + math.log(2.0 * math.pi)
        ).sum(dim=1)

        return loglik + log_p0_eps


class NBase(nn.Module):
    def __init__(self, dim, init_loc=0.0, init_log_scale=-2.5):
        super().__init__()
        self.dim = int(dim)
        self.loc = nn.Parameter(torch.full((self.dim,), float(init_loc)))
        self.raw_log_scale = nn.Parameter(torch.full((self.dim,), float(init_log_scale)))

    def log_scale(self):
        return torch.clamp(self.raw_log_scale, min=-5.0, max=2.0)

    def scale(self):
        return torch.exp(self.log_scale())

    def rsample(self, num_samples):
        eta = torch.randn(
            num_samples,
            self.dim,
            device=self.loc.device,
            dtype=self.loc.dtype,
        )
        z0 = self.loc.unsqueeze(0) + self.scale().unsqueeze(0) * eta
        return eta, z0

    def log_prob(self, z0):
        log_scale = self.log_scale().unsqueeze(0)
        var = torch.exp(2.0 * log_scale)
        return -0.5 * (
            ((z0 - self.loc.unsqueeze(0)) ** 2) / var
            + 2.0 * log_scale
            + math.log(2.0 * math.pi)
        ).sum(dim=1)


class FlowVI(nn.Module):
    def __init__(self, q0, posterior_flow, generative_model):
        super().__init__()
        self.q0 = q0
        self.posterior_flow = posterior_flow
        self.generative_model = generative_model

    def sample_posterior(self, num_samples):
        _, z0 = self.q0.rsample(num_samples)
        eps, logdet = self.posterior_flow(z0, return_logdet=True)
        log_q_eps = self.q0.log_prob(z0) - logdet
        return eps, log_q_eps

    def neg_elbo(self, num_samples=256, elbo_beta=1.0):
        """
        Negative ELBO without an additional q-entropy reweighting term.

        Objective:
            E_q[log q(eps) - elbo_beta * log p(y, eps)]
        """
        eps, log_q_eps = self.sample_posterior(num_samples)
        log_joint = self.generative_model.log_joint(eps)
        return (log_q_eps - float(elbo_beta) * log_joint).mean()

    def elbo_terms(self, num_samples=256):
        eps, log_q_eps = self.sample_posterior(num_samples)
        log_joint = self.generative_model.log_joint(eps)
        return {
            "log_q_eps": log_q_eps.mean(),
            "entropy_q": -log_q_eps.mean(),
            "log_joint": log_joint.mean(),
            "neg_elbo": (log_q_eps - log_joint).mean(),
        }


def build_flow_vi(
    X,
    y,
    sigma2,
    tau,
    family,
    beta_mode="sigmoid",
    group_ids=None,
    group_sizes=None,
    K_q=8,
    K_g=8,
    hidden_units=64,
    num_hidden_layers=2,
):
    p = X.shape[1]

    s_dim, u_dim, dim, gids = beta_config(
        p=p,
        beta_mode=beta_mode,
        group_ids=group_ids,
        group_sizes=group_sizes,
        device=X.device,
    )

    g_theta = FlowMap(
        s_dim=s_dim,
        u_dim=u_dim,
        K=K_g,
        hidden_units=hidden_units,
        num_hidden_layers=num_hidden_layers,
        scale_clip=2.0,
    )

    generative_model = Relaxedsas(
        X=X,
        y=y,
        sigma2=sigma2,
        tau=tau,
        g_theta=g_theta,
        family=family,
        beta_mode=beta_mode,
        group_ids=gids,
    )

    q0 = NBase(
        dim=dim,
        init_loc=0.0,
        init_log_scale=-2.5,
    )

    posterior_flow = FlowMap(
        s_dim=s_dim,
        u_dim=u_dim,
        K=K_q,
        hidden_units=hidden_units,
        num_hidden_layers=num_hidden_layers,
        scale_clip=2.0,
    )

    return FlowVI(
        q0=q0,
        posterior_flow=posterior_flow,
        generative_model=generative_model,
    )
