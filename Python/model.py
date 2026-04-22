import math
import torch
import torch.nn as nn
import normflows as nf

class SemanticAffineCoupling(nn.Module):
    """
    Affine coupling layer on semantic blocks of xi = [s, u, t].

    Mode:
        - "s": update s conditioned on (u, t)
        - "u": update u conditioned on (s, t)
        - "t": update t conditioned on (s, u)

    xi shape: [batch, 2p + 1]
    ordering : [ s(0:p) | u(p:2p) | t(2p) ]
    """
    def __init__(self, p, mode, hidden_units=128, num_hidden_layers=2, scale_clip=2.0):
        super().__init__()
        assert mode in {"s", "u", "t"}
        self.p = int(p)
        self.dim = 2 * p + 1
        self.mode = mode
        self.scale_clip = float(scale_clip)

        if mode == "s":
            cond_dim = p + 1      # (u, t)
            trans_dim = p         # s
        elif mode == "u":
            cond_dim = p + 1      # (s, t)
            trans_dim = p         # u
        else:  # mode == "t"
            cond_dim = 2 * p      # (s, u)
            trans_dim = 1         # t

        widths = [cond_dim] + [hidden_units] * num_hidden_layers + [2 * trans_dim]
        self.net = nf.nets.MLP(widths, init_zeros=True)

    def _split(self, x):
        s = x[:, :self.p]
        u = x[:, self.p:2 * self.p]
        t = x[:, 2 * self.p:2 * self.p + 1]
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
    def __init__(self, p=None, dim=None, K=4,
                 hidden_units=128, num_hidden_layers=2, scale_clip=2.0):
        super().__init__()

        if p is None:
            if dim is None:
                raise ValueError("Either p or dim must be provided.")
            if (dim - 1) % 2 != 0:
                raise ValueError("dim must equal 2*p + 1.")
            p = (dim - 1) // 2

        self.p = int(p)
        self.dim = 2 * self.p + 1

        layers = []
        for _ in range(K):
            layers.append(
                SemanticAffineCoupling(
                    p=self.p, mode="s",
                    hidden_units=hidden_units,
                    num_hidden_layers=num_hidden_layers,
                    scale_clip=scale_clip,
                )
            )
            layers.append(
                SemanticAffineCoupling(
                    p=self.p, mode="u",
                    hidden_units=hidden_units,
                    num_hidden_layers=num_hidden_layers,
                    scale_clip=scale_clip,
                )
            )
            layers.append(
                SemanticAffineCoupling(
                    p=self.p, mode="t",
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
    def __init__(self, X, y, sigma2, tau, g_theta, family="gaussian"):
        super().__init__()
        if g_theta is None:
            raise ValueError("g_theta must be provided.")

        self.register_buffer("X", X)
        self.register_buffer("y", y)
        self.register_buffer("tau", torch.tensor(float(tau), dtype=X.dtype, device=X.device))

        self.family = family
        if family == "gaussian":
            self.register_buffer(
                "sigma2",
                torch.tensor(float(sigma2), dtype=X.dtype, device=X.device)
            )
        else:
            self.sigma2 = None

        self.n, self.p = X.shape
        self.dim = 2 * self.p + 1
        self.g_theta = g_theta

    def set_tau(self, tau):
        self.tau.fill_(float(tau))

    def decode(self, eps):
        xi = self.g_theta(eps)

        p = self.p
        s = xi[:, :p]
        u = xi[:, p:2 * p]
        t = xi[:, 2 * p:2 * p + 1]

        gate = torch.sigmoid((u - t) / self.tau)
        beta = s * gate

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


def build_flow_vi(X, y, sigma2, tau, family, K_q=8, K_g=8, hidden_units=64, num_hidden_layers=2,):
    p = X.shape[1]
    dim = 2 * p + 1

    g_theta = FlowMap(p=X.shape[1], K=K_g, hidden_units=hidden_units,
                          num_hidden_layers=num_hidden_layers, scale_clip=2.0)

    generative_model = Relaxedsas(X=X, y=y, sigma2=sigma2, tau=tau, g_theta=g_theta, family=family)

    q0 = NBase(dim=dim, init_loc=0.0, init_log_scale=-2.5,)
    
    posterior_flow = FlowMap(
        dim=dim,
        K=K_q,
        hidden_units=hidden_units,
        num_hidden_layers=num_hidden_layers,
        scale_clip=2.0,
    )

    return FlowVI(q0=q0, posterior_flow=posterior_flow, generative_model=generative_model,)