import os
import json
import math
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch

Array = np.ndarray

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

def as_numpy_1d(x: Any) -> Array:
    arr = to_numpy(x)
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1D array, got shape {arr.shape}.")
    return arr.astype(float)


def as_numpy_2d(x: Any) -> Array:
    arr = to_numpy(x)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {arr.shape}.")
    return arr.astype(float)


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


def make_split_indices(n: int, split_cfg) -> Dict[str, Array]:
    if not math.isclose(
        split_cfg.train_frac + split_cfg.val_frac + split_cfg.test_frac,
        1.0,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError("train_frac + val_frac + test_frac must equal 1.")

    rng = np.random.default_rng(split_cfg.seed)
    idx = rng.permutation(n)

    n_train = int(round(n * split_cfg.train_frac))
    n_val = int(round(n * split_cfg.val_frac))

    n_train = min(max(n_train, 1), n - 2)
    n_val = min(max(n_val, 1), n - n_train - 1)

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    return {"train": train_idx, "val": val_idx, "test": test_idx}


make_splits = make_split_indices


def standardize_design(
    X_train: Array,
    X_val: Array,
    X_test: Array,
    *,
    standardize_x: bool = True,
) -> Tuple[Array, Array, Array, Array, Array]:
    x_mean = X_train.mean(axis=0)
    x_scale = X_train.std(axis=0, ddof=0)
    x_scale = np.where(x_scale < 1e-12, 1.0, x_scale)

    if not standardize_x:
        x_mean = np.zeros_like(x_mean)
        x_scale = np.ones_like(x_scale)

    X_train_s = (X_train - x_mean) / x_scale
    X_val_s = (X_val - x_mean) / x_scale
    X_test_s = (X_test - x_mean) / x_scale

    return X_train_s, X_val_s, X_test_s, x_mean, x_scale


def center_response(
    y_train: Array,
    y_val: Array,
    y_test: Array,
    *,
    center_y: bool = True,
) -> Tuple[Array, Array, Array, float]:
    y_mean = float(y_train.mean()) if center_y else 0.0
    return y_train - y_mean, y_val - y_mean, y_test - y_mean, y_mean


def recover_original_scale(
    beta_std: Array,
    beta_std_sd: Array,
    x_mean: Array,
    x_scale: Array,
    y_mean: float,
) -> Tuple[Array, Array, float]:
    beta = beta_std / x_scale
    beta_sd = beta_std_sd / x_scale
    intercept = y_mean - float(np.dot(x_mean, beta))
    return beta, beta_sd, intercept


def predict_linear(X: Array, beta: Array, intercept: float) -> Array:
    return intercept + X @ beta


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if torch.is_tensor(obj):
            return obj.detach().cpu().tolist()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if is_dataclass(obj):
            return asdict(obj)
        return super().default(obj)