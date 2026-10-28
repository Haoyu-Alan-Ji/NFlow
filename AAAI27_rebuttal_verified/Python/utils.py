from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

Array = np.ndarray


def is_torch_tensor(x: Any) -> bool:
    return x.__class__.__module__.split(".")[0] == "torch" and hasattr(x, "detach")


def to_numpy(x: Any) -> Array:
    if is_torch_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def safe_float(x: Any) -> float:
    if x is None:
        return float("nan")
    if is_torch_tensor(x):
        return float(x.detach().cpu().item())
    return float(x)


def to_jsonable(obj: Any) -> Any:
    if is_torch_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if is_dataclass(obj):
        return {k: to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    return obj


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        try:
            return to_jsonable(obj)
        except TypeError:
            return super().default(obj)


def write_json(path: str | Path, payload: Any, *, indent: int = 2) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, cls=NumpyJSONEncoder, ensure_ascii=False, indent=indent)


def ensure_dir(path: str | Path | None) -> str | None:
    if path is None:
        return None
    os.makedirs(path, exist_ok=True)
    return str(path)


def as_numpy_1d(x: Any) -> Array:
    arr = np.asarray(to_numpy(x), dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D array, got {arr.shape}.")
    return arr


def as_numpy_2d(x: Any) -> Array:
    arr = np.asarray(to_numpy(x), dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.shape}.")
    return arr


def jaccard_distance(a, b) -> float:
    a, b = set(a), set(b)
    return 0.0 if len(a | b) == 0 else 1.0 - len(a & b) / len(a | b)


def jaccard_similarity(a, b) -> float:
    return 1.0 - jaccard_distance(a, b)


class EMA:
    def __init__(self, beta: float = 0.9):
        self.beta = float(beta)
        self.value = None

    def update(self, x: float) -> float:
        self.value = float(x) if self.value is None else self.beta * self.value + (1.0 - self.beta) * float(x)
        return self.value


def rolling_stage_taus(tau_start: float, tau_end: float, n_stages: int):
    ratio = (float(tau_end) / float(tau_start)) ** (1.0 / int(n_stages))
    return [float(tau_start) * ratio ** k for k in range(int(n_stages) + 1)]


def set_optimizer_lr(optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def make_optimizer(model, cfg):
    import torch
    opt_cls = getattr(cfg, "optimizer_cls", None) or torch.optim.Adam
    return opt_cls(
        model.parameters(),
        lr=cfg.base_lr,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )


def save_model_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_model_state(model, state, device=None) -> None:
    if device is not None:
        state = {k: v.to(device) for k, v in state.items()}
    model.load_state_dict(state)


def set_all_seeds(seed: int, deterministic: bool = False) -> None:
    import torch
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_split(n: int, split_cfg) -> Dict[str, np.ndarray]:
    total = split_cfg.train_frac + split_cfg.val_frac + split_cfg.test_frac
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("train_frac + val_frac + test_frac must equal 1.")
    rng = np.random.default_rng(split_cfg.seed)
    idx = rng.permutation(int(n))
    n_train = min(max(int(round(n * split_cfg.train_frac)), 1), n - 2)
    n_val = min(max(int(round(n * split_cfg.val_frac)), 1), n - n_train - 1)
    return {"train": idx[:n_train], "val": idx[n_train:n_train + n_val], "test": idx[n_train + n_val:]}


def split_data(X, y, splits, *, mode: str = "auto") -> Dict[str, Any]:
    if mode == "auto":
        try:
            import torch
            mode = "tensor" if torch.is_tensor(X) else "numpy"
        except ImportError:
            mode = "numpy"
    if mode not in {"numpy", "tensor"}:
        raise ValueError("mode must be 'auto', 'numpy', or 'tensor'.")

    idx = {k: np.asarray(v, dtype=int) for k, v in splits.items()}
    if mode == "numpy":
        return {
            "X_train": X[idx["train"]], "y_train": y[idx["train"]],
            "X_val": X[idx["val"]], "y_val": y[idx["val"]],
            "X_test": X[idx["test"]], "y_test": y[idx["test"]],
            "idx_train": idx["train"], "idx_val": idx["val"], "idx_test": idx["test"],
            "indices": idx,
        }

    import torch
    device = X.device
    tidx = {k: torch.as_tensor(v, dtype=torch.long, device=device) for k, v in idx.items()}
    return {
        "X_train": X[tidx["train"]], "y_train": y[tidx["train"]],
        "X_val": X[tidx["val"]], "y_val": y[tidx["val"]],
        "X_test": X[tidx["test"]], "y_test": y[tidx["test"]],
        "idx_train": torch.as_tensor(idx["train"], dtype=torch.long),
        "idx_val": torch.as_tensor(idx["val"], dtype=torch.long),
        "idx_test": torch.as_tensor(idx["test"], dtype=torch.long),
        "indices": idx,
    }


def split_hash(indices: Dict[str, Any]) -> str:
    h = hashlib.sha256()
    for key in ["train", "val", "test"]:
        h.update(np.asarray(indices[key], dtype=np.int64).tobytes())
    return h.hexdigest()


def read_mcmc_ref(mcmc_root, mcmc_seed, setting: str = "simple") -> Dict[str, Array]:
    d = Path(mcmc_root) / setting / f"seed_{int(mcmc_seed)}"
    beta = pd.read_csv(d / "mcmc_beta_draws.csv.gz")
    if "draw_id" in beta.columns:
        beta = beta.drop(columns=["draw_id"])
    pip = pd.read_csv(d / "mcmc_pip.csv").sort_values("j0")
    return {"beta": beta.to_numpy(dtype=float), "pip": pip["pip"].to_numpy(dtype=float)}


def standardize_design(X_train: Array, X_val: Array, X_test: Array, *, standardize_x: bool = True) -> Tuple[Array, Array, Array, Array, Array]:
    mean = X_train.mean(axis=0) if standardize_x else np.zeros(X_train.shape[1])
    scale = X_train.std(axis=0, ddof=0) if standardize_x else np.ones(X_train.shape[1])
    scale = np.where(scale < 1e-12, 1.0, scale)
    return (X_train - mean) / scale, (X_val - mean) / scale, (X_test - mean) / scale, mean, scale


def center_response(y_train: Array, y_val: Array, y_test: Array, *, center_y: bool = True) -> Tuple[Array, Array, Array, float]:
    mean = float(np.mean(y_train)) if center_y else 0.0
    return y_train - mean, y_val - mean, y_test - mean, mean


def recover_original_scale(beta_std: Array, beta_std_sd: Array, x_mean: Array, x_scale: Array, y_mean: float) -> Tuple[Array, Array, float]:
    beta = beta_std / x_scale
    beta_sd = beta_std_sd / x_scale
    return beta, beta_sd, float(y_mean - np.dot(x_mean, beta))


def predict_linear(X: Array, beta: Array, intercept: float = 0.0) -> Array:
    return float(intercept) + X @ beta
