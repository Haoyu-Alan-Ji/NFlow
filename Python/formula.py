import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import normflows as nf

def _to_cpu(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return x

def to_tensor(x, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(dtype)
    return torch.tensor(x, dtype=dtype)

    
def _safe_div(a, b, default=0.0):
    return a / b if b != 0 else default

    
class RelaxedSpikeSlabTarget(nn.Module):
    def __init__(self, X, y,  sigma2, lambda_s=1.0,
        mu_t=0.0, sigma_t=1.0, tau=0.5, slab="laplace",  slab_scale=1.0, ):
        super().__init__()
        self.register_buffer("X", X)
        self.register_buffer("y", y)
        self.register_buffer(
            "sigma2",
            torch.as_tensor(float(sigma2), dtype=X.dtype, device=X.device)
        )

        self.p = X.shape[1]
        self.slab = slab

        self.register_buffer(
            "lambda_s",
            torch.as_tensor(float(lambda_s), dtype=X.dtype, device=X.device)
        )
        self.register_buffer(
            "mu_t",
            torch.as_tensor(float(mu_t), dtype=X.dtype, device=X.device)
        )
        self.register_buffer(
            "sigma_t",
            torch.as_tensor(float(sigma_t), dtype=X.dtype, device=X.device)
        )
        self.register_buffer(
            "tau",
            torch.as_tensor(float(tau), dtype=X.dtype, device=X.device)
        )
        self.register_buffer(
            "slab_scale",
            torch.as_tensor(float(slab_scale), dtype=X.dtype, device=X.device)
        )

    @property
    def latent_dim(self):
        return 2 * self.p + 1

    def set_tau(self, tau):
        self.tau.fill_(float(tau))

    def split_latent(self, z):
        s = z[:, :self.p]
        u = z[:, self.p: 2 * self.p]
        t = z[:, 2 * self.p: 2 * self.p + 1]
        return s, u, t

    def gate(self, u, t):
        return torch.sigmoid((u - t) / self.tau)

    def beta_from_latent(self, z):
        if z.dim() == 1:
            z = z.unsqueeze(0)
        s, u, t = self.split_latent(z)
        g = self.gate(u, t)
        beta = s * g
        return beta

    def gate_from_latent(self, z):
        if z.dim() == 1:
            z = z.unsqueeze(0)
        _, u, t = self.split_latent(z)
        return self.gate(u, t)

    def hard_gate_from_latent(self, z, threshold=0.5):
        g = self.gate_from_latent(z)
        return (g > threshold).float()

    def hard_beta_from_latent(self, z, threshold=0.5):
        if z.dim() == 1:
            z = z.unsqueeze(0)
        s, _, _ = self.split_latent(z)
        hard_g = self.hard_gate_from_latent(z, threshold=threshold)
        return s * hard_g

    def log_prob(self, z):
        if z.dim() == 1:
            z = z.unsqueeze(0)

        s, u, t = self.split_latent(z)
        beta = self.beta_from_latent(z)

        mean = beta @ self.X.T
        resid = self.y.unsqueeze(0) - mean
        log_lik = -0.5 / self.sigma2 * (resid ** 2).sum(dim=1)

        
        if self.slab == "laplace":
            slab_dist = torch.distributions.Laplace(
                loc=torch.zeros((), device=z.device, dtype=z.dtype),
                scale=1.0 / self.lambda_s
            )
            log_prior_s = slab_dist.log_prob(s).sum(dim=1)

        elif self.slab == "gaussian":
            slab_dist = torch.distributions.Normal(
                loc=torch.zeros((), device=z.device, dtype=z.dtype),
                scale=self.slab_scale
            )
            log_prior_s = slab_dist.log_prob(s).sum(dim=1)

        else:
            raise ValueError("slab must be 'laplace' or 'gaussian'")

        
        u_dist = torch.distributions.Normal(
            loc=torch.zeros((), device=z.device, dtype=z.dtype),
            scale=torch.ones((), device=z.device, dtype=z.dtype)
        )
        log_prior_u = u_dist.log_prob(u).sum(dim=1)

        
        t_dist = torch.distributions.Normal(loc=self.mu_t, scale=self.sigma_t)
        log_prior_t = t_dist.log_prob(t.squeeze(-1))

        return log_lik + log_prior_s + log_prior_u + log_prior_t


def build_nf(latent_dim, target_dist, flow_type="planar",
    K=8, hidden_units=128, num_hidden_layers=2, device="cpu",):
    
    q0 = nf.distributions.base.DiagGaussian(latent_dim)

    flow_type = flow_type.lower()
    flows = []

    if flow_type == "planar":
        flows = [
            nf.flows.Planar((latent_dim,), act="leaky_relu")
            for _ in range(K)
        ]

    elif flow_type == "radial":
        flows = [
            nf.flows.Radial((latent_dim,))
            for _ in range(K)
        ]

    elif flow_type in {"affine_coupling", "realnvp", "affine"}:
        base_mask = torch.tensor(
            [1 if i % 2 == 0 else 0 for i in range(latent_dim)],
            dtype=torch.float32,
        )

        hidden = [hidden_units] * num_hidden_layers

        for k in range(K):
            mask = base_mask if (k % 2 == 0) else (1.0 - base_mask)

            # s(z_masked), t(z_masked)
            # MLP maps R^(latent_dim) -> R^(latent_dim)
            s_net = nf.nets.MLP(
                [latent_dim, *hidden, latent_dim],
                init_zeros=True,
            )
            t_net = nf.nets.MLP(
                [latent_dim, *hidden, latent_dim],
                init_zeros=True,
            )

            flows.append(nf.flows.MaskedAffineFlow(mask, t_net, s_net))

    
    elif flow_type == "hybrid":
        base_mask = torch.tensor(
            [1 if i % 2 == 0 else 0 for i in range(latent_dim)],
            dtype=torch.float32,
        )
        hidden = [hidden_units] * num_hidden_layers

        for k in range(K):
            if k % 2 == 0:
                flows.append(nf.flows.Planar((latent_dim,), act="leaky_relu"))
            else:
                mask = base_mask if ((k // 2) % 2 == 0) else (1.0 - base_mask)

                s_net = nf.nets.MLP(
                    [latent_dim, *hidden, latent_dim],
                    init_zeros=True,
                )
                t_net = nf.nets.MLP(
                    [latent_dim, *hidden, latent_dim],
                    init_zeros=True,
                )

                flows.append(nf.flows.MaskedAffineFlow(mask, t_net, s_net))

    else:
        raise ValueError(
            "flow_type must be one of: "
            "'planar', 'radial', 'affine_coupling', 'hybrid'"
        )

    model = nf.NormalizingFlow(q0=q0, flows=flows, p=target_dist)
    return model.to(device)


def geometric_anneal(start, end, frac):
    
    if start <= 0 or end <= 0:
        raise ValueError("start and end must be positive for geometric annealing")
    return start * (end / start) ** frac


def train_flow(model, target_dist, epochs=6000, num_samples=1024,
    lr=1e-3, tau_start=1.0, tau_end=0.1, kl_anneal=True, grad_clip=5.0, print_every=500,):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    tau_hist = []

    for it in range(epochs):
        optimizer.zero_grad()

        frac = it / max(epochs - 1, 1)

        tau_now = geometric_anneal(tau_start, tau_end, frac)
        target_dist.set_tau(tau_now)
        tau_hist.append(float(tau_now))

        if kl_anneal:
            anneal_beta = min(1.0, 0.01 + 0.99 * frac)
        else:
            anneal_beta = 1.0

        loss = model.reverse_kld(num_samples=num_samples, beta=anneal_beta)

        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # Parameter update
        optimizer.step()

        losses.append(float(loss.item()))

        # Progress log
        if (it + 1) % print_every == 0 or it == 0:
            print(
                f"epoch {it+1:5d} | "
                f"loss = {loss.item():.4f} | "
                f"tau = {tau_now:.4f} | "
                f"anneal_beta = {anneal_beta:.4f}"
            )

    return losses, tau_hist


