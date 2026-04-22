import os
import numpy as np
import torch

def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def safe_float(x):
    if x is None:
        return float("nan")
    if torch.is_tensor(x):
        return x.item()
    return float(x)


class EMA:
    def __init__(self, beta=0.9):
        self.beta = beta
        self.value = None

    def update(self, x):
        if self.value is None:
            self.value = x
        else:
            self.value = self.beta * self.value + (1 - self.beta) * x
        return self.value


def jaccard_distance(a, b):
    a, b = set(a), set(b)
    if len(a | b) == 0:
        return 0.0
    return 1 - len(a & b) / len(a | b)


def jaccard_similarity(a, b):
    return 1 - jaccard_distance(a, b)


def rolling_stage_taus(tau_start, tau_end, n_stages):
    ratio = (tau_end / tau_start) ** (1 / n_stages)
    return [tau_start * ratio**k for k in range(n_stages + 1)]


def set_optimizer_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g["lr"] = lr

def make_optimizer(model, cfg):
    return torch.optim.Adam(
        model.parameters(),
        lr=cfg.base_lr,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )

def save_model_state(model):
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}

def load_model_state(model, state, device=None):
    if device is None:
        model.load_state_dict(state)
    else:
        moved = {k: v.to(device) for k, v in state.items()}
        model.load_state_dict(moved)

def ensure_dir(path):
    if path is not None:
        os.makedirs(path, exist_ok=True)
    return path

def split_data_tensors(X, y, split_cfg):
    n = X.shape[0]
    g = torch.Generator(device=X.device)
    g.manual_seed(split_cfg.seed)
    perm = torch.randperm(n, generator=g, device=X.device)

    n_train = int(round(n * split_cfg.train_frac))
    n_val = int(round(n * split_cfg.val_frac))
    n_train = min(max(n_train, 1), n - 2)
    n_val = min(max(n_val, 1), n - n_train - 1)
    n_test = n - n_train - n_val
    if n_test <= 0:
        n_test = 1
        n_val = max(1, n_val - 1)

    idx_train = perm[:n_train]
    idx_val = perm[n_train:n_train + n_val]
    idx_test = perm[n_train + n_val:]

    return {
        "X_train": X[idx_train],
        "y_train": y[idx_train],
        "X_val": X[idx_val],
        "y_val": y[idx_val],
        "X_test": X[idx_test],
        "y_test": y[idx_test],
        "idx_train": idx_train.detach().cpu(),
        "idx_val": idx_val.detach().cpu(),
        "idx_test": idx_test.detach().cpu(),
    }