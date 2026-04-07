import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import normflows as nf

torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_regression_data(n=200, p=100, sigma2=0.5, beta_signal=None, device="cpu"):
    if beta_signal is None:
        beta_signal = [1.5, -0.8, 0.7, -1.2, 0.5]

    beta_signal = torch.as_tensor(beta_signal, dtype=torch.float32, device=device)
    k = len(beta_signal)

    if k > p:
        raise ValueError("len(beta_signal) cannot exceed p")

    X = torch.randn(n, p, device=device)
    beta_true = torch.zeros(p, device=device)
    beta_true[:k] = beta_signal

    eps = math.sqrt(sigma2) * torch.randn(n, device=device)
    y = X @ beta_true + eps

    return X, y, beta_true


def summarize_beta_samples(beta_samples):
    
    beta_mean_hat = beta_samples.mean(dim=0)

    beta_centered = beta_samples - beta_mean_hat

    beta_cov_hat = beta_centered.T @ beta_centered / (beta_samples.shape[0] - 1)

    return beta_mean_hat, beta_cov_hat

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


@torch.no_grad()
def posterior_summary(model, target_dist, n_samples=5000,
    gate_threshold=0.5, inclusion_threshold=0.5,):
    
    z, _ = model.sample(num_samples=n_samples)

    beta_soft = target_dist.beta_from_latent(z)      # shape [m, p]
    gate_soft = target_dist.gate_from_latent(z)      # shape [m, p]

    gate_hard = (gate_soft > gate_threshold).float()  # shape [m, p]
    beta_hard = target_dist.hard_beta_from_latent(
        z,
        threshold=gate_threshold
    )

    s, u, t = target_dist.split_latent(z)

    beta_soft_mean = beta_soft.mean(dim=0)
    beta_hard_mean = beta_hard.mean(dim=0)
    gate_soft_mean = gate_soft.mean(dim=0)

    posterior_inclusion_prob = gate_hard.mean(dim=0)

    selected = (posterior_inclusion_prob > inclusion_threshold)
    selected_idx = torch.nonzero(selected, as_tuple=False).squeeze(1)

    out = {
        # raw latent posterior samples
        "z": z,
        "s": s,
        "u": u,
        "t": t.squeeze(-1),

        # soft/hard posterior quantities
        "beta_soft": beta_soft,
        "beta_hard": beta_hard,
        "gate_soft": gate_soft,
        "gate_hard": gate_hard,

        # posterior means
        "beta_soft_mean": beta_soft_mean,
        "beta_hard_mean": beta_hard_mean,
        "gate_soft_mean": gate_soft_mean,

        # variable-selection summaries
        "pip": posterior_inclusion_prob,
        "selected": selected,
        "selected_idx": selected_idx,
    }
    return out

def evaluate_against_truth(beta_true, post_summary):

    beta_true = beta_true.detach().cpu()
    selected = post_summary["selected"].detach().cpu()
    posterior_inclusion_prob = post_summary["pip"].detach().cpu()
    beta_soft_mean = post_summary["beta_soft_mean"].detach().cpu()
    beta_hard_mean = post_summary["beta_hard_mean"].detach().cpu()

    true_support = beta_true.ne(0)
    est_support = selected.bool()

    tp = (true_support & est_support).sum().item()
    fp = ((~true_support) & est_support).sum().item()
    fn = (true_support & (~est_support)).sum().item()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    print("\n===== Variable selection summary =====")
    print("selected indices:", post_summary["selected_idx"].detach().cpu().tolist())
    print("TP:", tp, "FP:", fp, "FN:", fn)
    print("precision:", precision)
    print("recall   :", recall)

    print("\nfirst 20 posterior inclusion probabilities:")
    print(posterior_inclusion_prob[:20])

    print("\nfirst 20 beta soft posterior means:")
    print(beta_soft_mean[:20])

    print("\nfirst 20 beta hard posterior means:")
    print(beta_hard_mean[:20])

    print("\nfirst 20 true beta:")
    print(beta_true[:20])


def plot_training(losses, tau_hist):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].plot(losses)
    ax[0].set_title("Loss history")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("reverse KL estimate")

    ax[1].plot(tau_hist)
    ax[1].set_title("Temperature annealing")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("tau")

    plt.tight_layout()
    plt.show()


def plot_posterior_inclusion_prob(posterior_inclusion_prob, top_k=30, decision_threshold=0.5):
    pip_np = posterior_inclusion_prob.detach().cpu().numpy()
    idx = np.arange(len(pip_np))

    plt.figure(figsize=(10, 4))
    plt.bar(idx[:top_k], pip_np[:top_k])
    plt.axhline(decision_threshold, linestyle="--")
    plt.title(f"Posterior inclusion probabilities (first {top_k})")
    plt.xlabel("j")
    plt.ylabel("P(selected | data)")
    plt.show()