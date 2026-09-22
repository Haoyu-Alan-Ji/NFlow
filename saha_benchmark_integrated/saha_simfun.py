import math
import numpy as np
import torch


def simfun_saha(
    n=2000,
    p=100,
    pi=0.2,
    alpha=2.0,
    sigma2=1.0,
    train_fraction=0.8,
    seed=400,
    device=None,
    dtype=torch.float32,
):
    """Saha, Liu & Liang (2024), Experiment 3 nonlinear DGP.

    X_ij ~ N(0,1), Z_j ~ Bernoulli(pi), beta_j = j/alpha,
    f(x) = exp(|x|) - 2x + sin(2*pi*x),
    y_i = sum_j f(X_ij) beta_j Z_j + eps_i, eps_i ~ N(0, sigma2).

    Returns the same five-object interface as the project's existing simfun():
    X, y, feature_true, signal, info.
    """
    device = torch.device("cpu") if device is None else torch.device(device)
    rng = np.random.default_rng(int(seed))
    n, p = int(n), int(p)

    X_np = rng.normal(size=(n, p)).astype(np.float32)
    z_np = rng.binomial(1, float(pi), size=p).astype(np.float32)
    beta_np = (np.arange(1, p + 1, dtype=np.float32) / float(alpha)).astype(np.float32)

    X = torch.as_tensor(X_np, device=device, dtype=dtype)
    feature_true = torch.as_tensor(z_np, device=device, dtype=dtype)
    beta = torch.as_tensor(beta_np, device=device, dtype=dtype)

    fx = torch.exp(torch.abs(X)) - 2.0 * X + torch.sin(2.0 * math.pi * X)
    signal = (fx * (beta * feature_true).unsqueeze(0)).sum(dim=1)

    # Keep the paper DGP exact: no centering/rescaling of signal.
    noise_np = math.sqrt(float(sigma2)) * rng.normal(size=n).astype(np.float32)
    noise = torch.as_tensor(noise_np, device=device, dtype=dtype)
    y = signal + noise

    perm = rng.permutation(n)
    n_train = int(round(float(train_fraction) * n))
    train_idx = perm[:n_train].astype(np.int64)
    test_idx = perm[n_train:].astype(np.int64)

    active_idx = np.flatnonzero(z_np > 0.5).astype(int)
    info = {
        "sim": "saha_nonlinear",
        "source": "Saha, Liu & Liang (2024), Experiment 3",
        "seed": int(seed),
        "n": n,
        "p": p,
        "pi": float(pi),
        "alpha": float(alpha),
        "sigma2": float(sigma2),
        "train_fraction": float(train_fraction),
        "train_idx": train_idx,
        "test_idx": test_idx,
        "n_active": int(active_idx.size),
        "active_idx": active_idx,
        "feature_true": z_np.copy(),
        "beta": beta_np.copy(),
        "signal_sd": float(signal.std(unbiased=False).detach().cpu()),
        "signal_mean": float(signal.mean().detach().cpu()),
        "dgp": "y=sum_j [exp(|X_j|)-2X_j+sin(2*pi*X_j)]*(j/alpha)*Z_j+eps",
    }
    return X, y, feature_true, signal, info
