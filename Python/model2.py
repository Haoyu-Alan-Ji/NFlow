import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import normflows as nf

def beta_config(p, beta_mode="sigmoid", group_ids=None, group_sizes=None, device=None):
    """
    Return:
        s_dim, u_dim, dim, group_ids

    sigmoid / relu:
        xi = [s, u, t]
        s in R^p, u in R^p, t in R
        dim = 2p + 1

    group_relu:
        xi = [s, u, t]
        s in R^G, u in R^p, t in R
        dim = G + p + 1
    """
    p = int(p)

    if beta_mode in {"sigmoid", "relu"}:
        return p, p, 2 * p + 1, None

    if beta_mode == "group_relu":
        if group_sizes is not None:
            gids = torch.cat([
                torch.full((int(size),), g, dtype=torch.long, device=device)
                for g, size in enumerate(group_sizes)
            ])
        else:
            gids = torch.as_tensor(group_ids, dtype=torch.long, device=device)

        # relabel to 0, ..., G-1
        unique = torch.unique(gids, sorted=True)
        new_gids = torch.empty_like(gids)
        for g, old_g in enumerate(unique):
            new_gids[gids == old_g] = g

        G = len(unique)
        return G, p, G + p + 1, new_gids

    raise ValueError(f"Unknown beta_mode: {beta_mode}")


class SemanticAffineCoupling(nn.Module):
    """
    Affine coupling layer on semantic blocks xi = [s, u, t].

    This version supports both the old ungrouped layout
        s in R^p, u in R^p, t in R, dim = 2p + 1,
    and the grouped layout
        s in R^G, u in R^p, t in R, dim = G + p + 1.

    Mode:
        - "s": update s conditioned on (u, t)
        - "u": update u conditioned on (s, t)
        - "t": update t conditioned on (s, u)
    """
    def __init__(self, p=None, mode=None, s_dim=None, u_dim=None,
                 hidden_units=128, num_hidden_layers=2, scale_clip=2.0):
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
        else:  # mode == "t"
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
            s_new = s * torch.exp(log_scale) + shift
            y = self._merge(s_new, u, t)
            logdet = log_scale.sum(dim=-1)

        elif self.mode == "u":
            cond = torch.cat([s, t], dim=-1)
            log_scale, shift = self._affine_params(cond)
            u_new = u * torch.exp(log_scale) + shift
            y = self._merge(s, u_new, t)
            logdet = log_scale.sum(dim=-1)

        else:  # mode == "t"
            cond = torch.cat([s, u], dim=-1)
            log_scale, shift = self._affine_params(cond)
            t_new = t * torch.exp(log_scale) + shift
            y = self._merge(s, u, t_new)
            logdet = log_scale.sum(dim=-1)

        if return_logdet:
            return y, logdet
        return y

    def inverse(self, y, return_logdet=False):
        s, u, t = self._split(y)

        if self.mode == "s":
            cond = torch.cat([u, t], dim=-1)
            log_scale, shift = self._affine_params(cond)
            s_old = (s - shift) * torch.exp(-log_scale)
            x = self._merge(s_old, u, t)
            logdet = (-log_scale).sum(dim=-1)

        elif self.mode == "u":
            cond = torch.cat([s, t], dim=-1)
            log_scale, shift = self._affine_params(cond)
            u_old = (u - shift) * torch.exp(-log_scale)
            x = self._merge(s, u_old, t)
            logdet = (-log_scale).sum(dim=-1)

        else:  # mode == "t"
            cond = torch.cat([s, u], dim=-1)
            log_scale, shift = self._affine_params(cond)
            t_old = (t - shift) * torch.exp(-log_scale)
            x = self._merge(s, u, t_old)
            logdet = (-log_scale).sum(dim=-1)

        if return_logdet:
            return x, logdet
        return x


class FlowMap(nn.Module):
    def __init__(self, p=None, dim=None, s_dim=None, u_dim=None, K=4,
                 hidden_units=128, num_hidden_layers=2, scale_clip=2.0):
        super().__init__()

        if s_dim is None or u_dim is None:
            if p is not None:
                s_dim = int(p)
                u_dim = int(p)
            elif dim is not None:
                if (int(dim) - 1) % 2 != 0:
                    raise ValueError("If s_dim/u_dim are omitted, dim must equal 2*p + 1.")
                p = (int(dim) - 1) // 2
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
            layers.append(
                SemanticAffineCoupling(
                    s_dim=self.s_dim, u_dim=self.u_dim, mode="s",
                    hidden_units=hidden_units,
                    num_hidden_layers=num_hidden_layers,
                    scale_clip=scale_clip,
                )
            )
            layers.append(
                SemanticAffineCoupling(
                    s_dim=self.s_dim, u_dim=self.u_dim, mode="u",
                    hidden_units=hidden_units,
                    num_hidden_layers=num_hidden_layers,
                    scale_clip=scale_clip,
                )
            )
            layers.append(
                SemanticAffineCoupling(
                    s_dim=self.s_dim, u_dim=self.u_dim, mode="t",
                    hidden_units=hidden_units,
                    num_hidden_layers=num_hidden_layers,
                    scale_clip=scale_clip,
                )
            )

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

        self.register_buffer("X", X)
        self.register_buffer("y", y)
        self.register_buffer(
            "tau",
            torch.tensor(float(tau), dtype=X.dtype, device=X.device)
        )

        self.family = family
        self.beta_mode = beta_mode

        if group_ids is not None:
            self.register_buffer("group_ids", group_ids)
        else:
            self.group_ids = None

        if family == "gaussian":
            self.register_buffer(
                "sigma2",
                torch.tensor(float(sigma2), dtype=X.dtype, device=X.device)
            )
        else:
            self.sigma2 = None

        self.n, self.p = X.shape
        self.g_theta = g_theta

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

        if self.beta_mode in {"sigmoid", "relu"}:
            p = self.p

            s = xi[:, :p]
            u = xi[:, p:2 * p]
            t = xi[:, 2 * p:2 * p + 1]

            if self.beta_mode == "sigmoid":
                gate = torch.sigmoid((u - t) / self.tau)
            else:
                gate = F.relu(u - t)

            beta = s * gate

        else:  # group_relu
            G = int(self.group_ids.max().item()) + 1
            p = self.p

            s = xi[:, :G]                  # [R, G]
            u = xi[:, G:G + p]             # [R, p]
            t = xi[:, G + p:G + p + 1]     # [R, 1]

            gate = F.relu(u - t)           # [R, p]
            beta = s[:, self.group_ids] * gate

        return {
            "eps": eps,
            "xi": xi,
            "s": s,
            "u": u,
            "t": t,
            "gate": gate,
            "beta": beta,
        }


    def log_joint(self, eps):
        dec = self.decode(eps)
        beta = dec["beta"]                    # [R, p]

        eta = self.X @ beta.T                 # [n, R]

        if self.family == "gaussian":
            resid = self.y[:, None] - eta
            loglik = -0.5 * (
                resid.pow(2).sum(dim=0) / self.sigma2
                + self.n * torch.log(2.0 * torch.pi * self.sigma2)
            )

        elif self.family == "poisson":
            y = self.y[:, None]
            loglik = (y * eta - torch.exp(eta) - torch.lgamma(y + 1.0)).sum(dim=0)

        else:
            raise ValueError(f"Unknown family: {self.family}")

        log_p0_eps = -0.5 * (
            eps.pow(2) + math.log(2.0 * math.pi)
        ).sum(dim=1)

        return loglik + log_p0_eps


class NBase(nn.Module):
    def __init__(self, dim, init_loc=0.0, init_log_scale=-2.5):
        super().__init__()
        self.dim = dim
        self.loc = nn.Parameter(torch.full((dim,), float(init_loc)))
        self.raw_log_scale = nn.Parameter(torch.full((dim,), float(init_log_scale)))

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
        eps, log_q_eps = self.sample_posterior(num_samples)
        log_joint = self.generative_model.log_joint(eps)
        return (log_q_eps - elbo_beta * log_joint).mean()


def build_flow_vi(
    X, y, sigma2, tau, family,
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
