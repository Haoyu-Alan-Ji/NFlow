import numpy as np
import torch

def simfun1(n=180, p=100, seed=123, snr=3.0, true_prop=0.1, device=None, dtype=torch.float32,):

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    X = rng.standard_normal((n, p)).astype(np.float32)
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)

    n_active = int(p * true_prop)
    active_idx = np.sort(rng.choice(p, size=n_active, replace=False))

    beta_true = np.zeros(p, dtype=np.float32)
    magnitudes = rng.uniform(0.3, 2.0, size=n_active).astype(np.float32)
    signs = rng.choice([-1.0, 1.0], size=n_active).astype(np.float32)
    beta_true[active_idx] = signs * magnitudes

    signal = X @ beta_true
    sigma2 = np.var(signal) / snr
    sigma = np.sqrt(sigma2)

    y = signal + sigma * rng.standard_normal(n).astype(np.float32)
    y = y - y.mean()

    X_t = torch.tensor(X, dtype=dtype, device=device)
    y_t = torch.tensor(y, dtype=dtype, device=device)
    beta_true_t = torch.tensor(beta_true, dtype=dtype, device=device)

    info = {"n": n, "p": p, "n_active": n_active, "sigma2": float(sigma2), "sigma": float(sigma), "active_idx": active_idx, "snr": snr,}

    return X_t, y_t, beta_true_t, info