import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def beta_config(p, beta_mode="sigmoid", group_ids=None, group_sizes=None, device=None):
    p = int(p)

    if beta_mode in {"sigmoid", "relu"}:
        return p, p, 2 * p + 1, None

    if beta_mode != "group_relu":
        raise ValueError(f"Unknown beta_mode: {beta_mode}")

    if group_sizes is not None:
        sizes = [int(x) for x in group_sizes]
        gids = torch.cat([
            torch.full((size,), g, dtype=torch.long, device=device)
            for g, size in enumerate(sizes)
        ])
    else:
        gids = torch.as_tensor(group_ids, dtype=torch.long, device=device)

    unique = torch.unique(gids, sorted=True)
    new_gids = torch.empty_like(gids)
    for g, old_g in enumerate(unique):
        new_gids[gids == old_g] = g

    G = int(len(unique))
    return p, G, p + G + 1, new_gids


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_units=128, num_hidden_layers=2, init_zeros=True):
        super().__init__()
        layers = []
        d = int(in_dim)
        for _ in range(int(num_hidden_layers)):
            layers += [nn.Linear(d, int(hidden_units)), nn.ReLU()]
            d = int(hidden_units)
        layers.append(nn.Linear(d, int(out_dim)))
        self.net = nn.Sequential(*layers)
        if init_zeros:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


class ResCond(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_units=128, num_hidden_layers=2, init_zeros=True):
        super().__init__()
        self.in_proj = nn.Linear(int(in_dim), int(hidden_units))
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(),
                nn.Linear(int(hidden_units), int(hidden_units)),
                nn.ReLU(),
                nn.Linear(int(hidden_units), int(hidden_units)),
            )
            for _ in range(int(num_hidden_layers))
        ])
        self.out = nn.Linear(int(hidden_units), int(out_dim))
        if init_zeros:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def forward(self, x):
        h = self.in_proj(x)
        for block in self.blocks:
            h = h + block(h)
        return self.out(F.relu(h))


def make_net(in_dim, out_dim, hidden_units, num_hidden_layers, conditioner_type):
    cls = ResCond if conditioner_type == "resnet" else MLP
    return cls(in_dim, out_dim, hidden_units, num_hidden_layers, init_zeros=True)


class Identity(nn.Module):
    def __init__(self, s_dim, u_dim):
        super().__init__()
        self.s_dim = int(s_dim)
        self.u_dim = int(u_dim)
        self.dim = self.s_dim + self.u_dim + 1

    def forward(self, x, return_logdet=False):
        if return_logdet:
            return x, x.new_zeros(x.shape[0])
        return x

    def inverse(self, x, return_logdet=False):
        if return_logdet:
            return x, x.new_zeros(x.shape[0])
        return x


class Affine(nn.Module):
    def __init__(
        self,
        dim,
        mask,
        hidden_units=128,
        num_hidden_layers=2,
        scale_clip=2.0,
        conditioner_type="mlp",
    ):
        super().__init__()
        self.dim = int(dim)
        self.scale_clip = float(scale_clip)
        self.register_buffer("mask", torch.as_tensor(mask, dtype=torch.bool))
        self.net = make_net(
            int(self.mask.sum().item()),
            2 * int((~self.mask).sum().item()),
            hidden_units,
            num_hidden_layers,
            conditioner_type,
        )

    def params(self, x):
        h = self.net(x[:, self.mask])
        log_scale, shift = torch.chunk(h, 2, dim=-1)
        log_scale = self.scale_clip * torch.tanh(log_scale / self.scale_clip)
        return log_scale, shift

    def forward(self, x, return_logdet=False):
        log_scale, shift = self.params(x)
        y = x.clone()
        y[:, ~self.mask] = x[:, ~self.mask] * torch.exp(log_scale) + shift
        logdet = log_scale.sum(dim=-1)
        if return_logdet:
            return y, logdet
        return y

    def inverse(self, y, return_logdet=False):
        log_scale, shift = self.params(y)
        x = y.clone()
        x[:, ~self.mask] = (y[:, ~self.mask] - shift) * torch.exp(-log_scale)
        logdet = -log_scale.sum(dim=-1)
        if return_logdet:
            return x, logdet
        return x


class Semantic(nn.Module):
    def __init__(
        self,
        s_dim,
        u_dim,
        mode,
        hidden_units=128,
        num_hidden_layers=2,
        scale_clip=2.0,
        conditioner_type="mlp",
    ):
        super().__init__()
        self.s_dim = int(s_dim)
        self.u_dim = int(u_dim)
        self.dim = self.s_dim + self.u_dim + 1
        self.mode = mode
        self.scale_clip = float(scale_clip)

        if mode == "s":
            cond_dim, trans_dim = self.u_dim + 1, self.s_dim
        elif mode == "u":
            cond_dim, trans_dim = self.s_dim + 1, self.u_dim
        elif mode == "t":
            cond_dim, trans_dim = self.s_dim + self.u_dim, 1
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.net = make_net(
            cond_dim,
            2 * trans_dim,
            hidden_units,
            num_hidden_layers,
            conditioner_type,
        )

    def split(self, x):
        s_end = self.s_dim
        u_end = self.s_dim + self.u_dim
        return x[:, :s_end], x[:, s_end:u_end], x[:, u_end:u_end + 1]

    def merge(self, s, u, t):
        return torch.cat([s, u, t], dim=-1)

    def params(self, cond):
        h = self.net(cond)
        log_scale, shift = torch.chunk(h, 2, dim=-1)
        log_scale = self.scale_clip * torch.tanh(log_scale / self.scale_clip)
        return log_scale, shift

    def forward(self, x, return_logdet=False):
        s, u, t = self.split(x)
        if self.mode == "s":
            log_scale, shift = self.params(torch.cat([u, t], dim=-1))
            s = s * torch.exp(log_scale) + shift
        elif self.mode == "u":
            log_scale, shift = self.params(torch.cat([s, t], dim=-1))
            u = u * torch.exp(log_scale) + shift
        else:
            log_scale, shift = self.params(torch.cat([s, u], dim=-1))
            t = t * torch.exp(log_scale) + shift

        y = self.merge(s, u, t)
        logdet = log_scale.sum(dim=-1)
        if return_logdet:
            return y, logdet
        return y

    def inverse(self, y, return_logdet=False):
        s, u, t = self.split(y)
        if self.mode == "s":
            log_scale, shift = self.params(torch.cat([u, t], dim=-1))
            s = (s - shift) * torch.exp(-log_scale)
        elif self.mode == "u":
            log_scale, shift = self.params(torch.cat([s, t], dim=-1))
            u = (u - shift) * torch.exp(-log_scale)
        else:
            log_scale, shift = self.params(torch.cat([s, u], dim=-1))
            t = (t - shift) * torch.exp(-log_scale)

        x = self.merge(s, u, t)
        logdet = -log_scale.sum(dim=-1)
        if return_logdet:
            return x, logdet
        return x


class Flow(nn.Module):
    def __init__(
        self,
        s_dim,
        u_dim,
        K=4,
        coupling_type="semantic",
        hidden_units=128,
        num_hidden_layers=2,
        scale_clip=2.0,
        conditioner_type="mlp",
        affine_layers_per_step=3,
    ):
        super().__init__()
        self.s_dim = int(s_dim)
        self.u_dim = int(u_dim)
        self.dim = self.s_dim + self.u_dim + 1
        self.coupling_type = coupling_type

        layers = []

        if coupling_type == "semantic":
            for _ in range(int(K)):
                for mode in ("s", "u", "t"):
                    layers.append(Semantic(
                        self.s_dim,
                        self.u_dim,
                        mode,
                        hidden_units,
                        num_hidden_layers,
                        scale_clip,
                        conditioner_type,
                    ))

        elif coupling_type == "semantic_affine_control":
            s_mid = self.s_dim // 2
            u_mid = self.u_dim // 2
            s_start = 0
            s_end = self.s_dim
            u_start = self.s_dim
            u_end = self.s_dim + self.u_dim
            t_start = self.s_dim + self.u_dim
            t_end = self.dim

            blocks = [
                torch.arange(s_start, s_start + s_mid),
                torch.arange(s_start + s_mid, s_end),
                torch.arange(u_start, u_start + u_mid),
                torch.arange(u_start + u_mid, u_end),
                torch.arange(t_start, t_end),
            ]

            for _ in range(int(K)):
                for block in blocks:
                    mask = torch.ones(self.dim, dtype=torch.bool)
                    mask[block] = False
                    layers.append(Affine(
                        self.dim,
                        mask,
                        hidden_units,
                        num_hidden_layers,
                        scale_clip,
                        conditioner_type,
                    ))

        elif coupling_type == "affine":
            for k in range(int(K) * int(affine_layers_per_step)):
                mask = (torch.arange(self.dim) + k) % 2 == 0
                layers.append(Affine(
                    self.dim,
                    mask,
                    hidden_units,
                    num_hidden_layers,
                    scale_clip,
                    conditioner_type,
                ))

        elif coupling_type != "meanfield":
            raise ValueError(f"Unknown coupling_type: {coupling_type}")

        self.layers = nn.ModuleList(layers)

    def forward(self, x, return_logdet=False):
        z = x
        total = x.new_zeros(x.shape[0])
        for layer in self.layers:
            z, logdet = layer(z, return_logdet=True)
            total = total + logdet
        if return_logdet:
            return z, total
        return z

    def inverse(self, z, return_logdet=False):
        x = z
        total = z.new_zeros(z.shape[0])
        for layer in reversed(self.layers):
            x, logdet = layer.inverse(x, return_logdet=True)
            total = total + logdet
        if return_logdet:
            return x, total
        return x


class RelaxedSAS(nn.Module):
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
        self.register_buffer("tau", torch.tensor(float(tau), dtype=X.dtype, device=X.device))

        self.family = family
        self.beta_mode = beta_mode
        self.n, self.p = X.shape
        self.g_theta = g_theta
        self.s_dim = int(g_theta.s_dim)
        self.u_dim = int(g_theta.u_dim)
        self.dim = int(g_theta.dim)

        if beta_mode == "group_relu":
            self.register_buffer("group_ids", torch.as_tensor(group_ids, dtype=torch.long, device=X.device))
        else:
            self.group_ids = None

        if family == "gaussian":
            self.register_buffer("sigma2", torch.tensor(float(sigma2), dtype=X.dtype, device=X.device))
        else:
            self.sigma2 = None

    def set_tau(self, tau):
        self.tau.fill_(float(tau))

    def split(self, xi):
        s_end = self.s_dim
        u_end = self.s_dim + self.u_dim
        return xi[:, :s_end], xi[:, s_end:u_end], xi[:, u_end:u_end + 1]

    def decode(self, z):
        xi = self.g_theta(z)
        s, u, t = self.split(xi)

        if self.beta_mode == "sigmoid":
            margin = u - t
            gate = torch.sigmoid(margin / self.tau)
            active = (margin > 0).to(gate.dtype)
            beta = s * gate
            return {"eps": z, "xi": xi, "s": s, "u": u, "t": t, "margin": margin,
                    "gate": gate, "active": active, "beta": beta}

        if self.beta_mode == "relu":
            margin = u - t
            gate = F.relu(margin)
            active = (margin > 0).to(gate.dtype)
            beta = s * gate
            return {"eps": z, "xi": xi, "s": s, "u": u, "t": t, "margin": margin,
                    "gate": gate, "active": active, "beta": beta}

        group_margin = u - t
        group_gate = F.relu(group_margin)
        group_active = (group_margin > 0).to(group_gate.dtype)
        margin = group_margin[:, self.group_ids]
        gate = group_gate[:, self.group_ids]
        active = group_active[:, self.group_ids]
        beta = s * gate
        return {"eps": z, "xi": xi, "s": s, "u": u, "t": t, "margin": margin,
                "gate": gate, "active": active, "group_margin": group_margin,
                "group_gate": group_gate, "group_active": group_active,
                "group_ids": self.group_ids, "beta": beta}

    def log_joint(self, z):
        beta = self.decode(z)["beta"]
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
            loglik = -F.binary_cross_entropy_with_logits(eta, y, reduction="none").sum(dim=0)
        else:
            raise ValueError(f"Unknown family: {self.family}")

        log_p0_z = -0.5 * (z.pow(2) + math.log(2.0 * math.pi)).sum(dim=1)
        return loglik + log_p0_z


class NBase(nn.Module):
    def __init__(self, dim, init_loc=0.0, init_log_scale=-2.5):
        super().__init__()
        self.dim = int(dim)
        self.loc = nn.Parameter(torch.full((self.dim,), float(init_loc)))
        self.raw_log_scale = nn.Parameter(torch.full((self.dim,), float(init_log_scale)))

    def log_scale(self):
        return torch.clamp(self.raw_log_scale, min=-5.0, max=2.0)

    def rsample(self, num_samples):
        eta = torch.randn(num_samples, self.dim, device=self.loc.device, dtype=self.loc.dtype)
        z0 = self.loc.unsqueeze(0) + torch.exp(self.log_scale()).unsqueeze(0) * eta
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
        z, logdet = self.posterior_flow(z0, return_logdet=True)
        log_q = self.q0.log_prob(z0) - logdet
        return z, log_q

    def neg_elbo(self, num_samples=256, elbo_beta=1.0):
        z, log_q = self.sample_posterior(num_samples)
        log_joint = self.generative_model.log_joint(z)
        return (log_q - float(elbo_beta) * log_joint).mean()

    def elbo_terms(self, num_samples=256):
        z, log_q = self.sample_posterior(num_samples)
        log_joint = self.generative_model.log_joint(z)
        return {
            "log_q_eps": log_q.mean(),
            "log_joint": log_joint.mean(),
            "neg_elbo": (log_q - log_joint).mean(),
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
    K_flow=None,
    coupling_type="semantic",
    conditioner_type="mlp",
    hidden_units=64,
    num_hidden_layers=2,
    scale_clip=2.0,
    affine_layers_per_step=3,
):
    s_dim, u_dim, dim, gids = beta_config(
        p=X.shape[1],
        beta_mode=beta_mode,
        group_ids=group_ids,
        group_sizes=group_sizes,
        device=X.device,
    )

    if K_flow is None:
        K_flow = int(K_q) + int(K_g)

    q0 = NBase(dim=dim, init_loc=0.0, init_log_scale=-2.5)

    posterior_flow = Flow(
        s_dim=s_dim,
        u_dim=u_dim,
        K=K_flow,
        coupling_type=coupling_type,
        hidden_units=hidden_units,
        num_hidden_layers=num_hidden_layers,
        scale_clip=scale_clip,
        conditioner_type=conditioner_type,
        affine_layers_per_step=affine_layers_per_step,
    )

    generative_model = RelaxedSAS(
        X=X,
        y=y,
        sigma2=sigma2,
        tau=tau,
        g_theta=Identity(s_dim=s_dim, u_dim=u_dim),
        family=family,
        beta_mode=beta_mode,
        group_ids=gids,
    )

    return FlowVI(
        q0=q0,
        posterior_flow=posterior_flow,
        generative_model=generative_model,
    )
