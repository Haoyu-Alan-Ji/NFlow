from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import experiment_config as cfg


def _standardize(term: torch.Tensor) -> torch.Tensor:
    return (term - term.mean()) / term.std(unbiased=False).clamp_min(1e-8)


def _teacher_supports(rng, n_active, n_units, features_per_unit):
    m = min(int(features_per_unit), int(n_active))
    if n_units * m < n_active:
        raise ValueError("teacher_units * features_per_unit must cover all active predictors")
    # Preserve the mixed-support construction used in the nonlinear playground:
    # cover every active predictor first, then fill each teacher unit to size m.
    supports = [set() for _ in range(n_units)]
    for pos, feat in enumerate(rng.permutation(n_active)):
        supports[pos % n_units].add(int(feat))
    for unit in range(n_units):
        while len(supports[unit]) < m:
            supports[unit].add(int(rng.integers(n_active)))
    return [tuple(sorted(s)) for s in supports]


def simulate_condition(condition: str, seed: int, device="cpu", dtype=torch.float32):
    if condition not in cfg.TABLE1_CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    spec = cfg.TABLE1_CONDITIONS[condition]
    activation = spec["activation"]
    n_interactions = int(spec["n_interactions"])
    n_quadratic = int(spec["n_quadratic"])

    n, p, s = cfg.N, cfg.P, cfg.N_ACTIVE
    rng = np.random.default_rng(int(seed))
    device = torch.device(device)
    gen = torch.Generator(device=device); gen.manual_seed(int(seed))

    X = cfg.X_LOW + (cfg.X_HIGH - cfg.X_LOW) * torch.rand(
        n, p, generator=gen, device=device, dtype=dtype
    )

    # Fixed labels make pairwise set-stability (Kuncheva) meaningful across replicates.
    active_idx = np.arange(s, dtype=int)
    Xa = X[:, :s]
    supports = _teacher_supports(rng, s, cfg.TEACHER_UNITS, cfg.FEATURES_PER_UNIT)

    W = np.zeros((cfg.TEACHER_UNITS, s), dtype=np.float32)
    for unit, support in enumerate(supports):
        idx = np.asarray(support, dtype=int)
        signs = rng.choice([-1.0, 1.0], size=len(idx))
        magnitude = rng.uniform(0.8, 1.2)
        W[unit, idx] = magnitude * signs / np.sqrt(len(idx))

    bias = rng.uniform(-0.8, 0.8, size=cfg.TEACHER_UNITS).astype(np.float32)
    amp = rng.uniform(0.8, 1.2, size=cfg.TEACHER_UNITS).astype(np.float32)
    Wt = torch.as_tensor(W, device=device, dtype=dtype)
    bt = torch.as_tensor(bias, device=device, dtype=dtype)
    at = torch.as_tensor(amp, device=device, dtype=dtype)
    pre = Xa @ Wt.T - bt

    if activation == "relu":
        hidden = F.relu(pre)
    elif activation == "trig":
        # Trigonometric ridge units. Each unit mixes two active predictors.
        hidden = torch.sin(pre)
    else:
        raise ValueError("activation must be relu or trig")
    signal = hidden @ at

    # Explicit interactions are raw predictor products x_j*x_k, not products of trig bases.
    # Prefer pairs already co-occurring within a two-predictor teacher unit.
    pair_candidates = []
    for support in supports:
        if len(support) == 2:
            pair = tuple(sorted(support))
            if pair not in pair_candidates:
                pair_candidates.append(pair)
    all_pairs = [(j, k) for j in range(s) for k in range(j + 1, s)]
    rng.shuffle(pair_candidates); rng.shuffle(all_pairs)
    pair_candidates += [z for z in all_pairs if z not in pair_candidates]
    interaction_pairs = pair_candidates[:n_interactions]
    for j, k in interaction_pairs:
        signal = signal + cfg.EXTRA_SCALE * _standardize(Xa[:, j] * Xa[:, k])

    quadratic_features = []
    if n_quadratic:
        quadratic_features = rng.choice(s, size=min(n_quadratic, s), replace=False).tolist()
        for j in quadratic_features:
            signal = signal + cfg.EXTRA_SCALE * _standardize(Xa[:, j].square())

    signal = signal - signal.mean()
    signal = signal * cfg.TARGET_SIGNAL_SD / signal.std(unbiased=False).clamp_min(1e-8)
    y = signal + math.sqrt(cfg.SIGMA2) * torch.randn(
        n, generator=gen, device=device, dtype=dtype
    )

    feature_true = torch.zeros(p, device=device, dtype=dtype)
    feature_true[:s] = 1.0

    split_rng = np.random.default_rng(int(seed) + 123456)
    perm = split_rng.permutation(n)
    n_train = int(round(cfg.TRAIN_FRAC * n))
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    info = {
        "condition": condition,
        "activation": activation,
        "n": n, "p": p, "n_active": s,
        "active_idx": active_idx.tolist(),
        "teacher_units": cfg.TEACHER_UNITS,
        "features_per_unit": cfg.FEATURES_PER_UNIT,
        "supports_local": [list(z) for z in supports],
        "interaction_pairs_local": [list(z) for z in interaction_pairs],
        "quadratic_features_local": [int(z) for z in quadratic_features],
        "target_signal_sd": cfg.TARGET_SIGNAL_SD,
        "sigma2": cfg.SIGMA2,
        "seed": int(seed),
    }
    return X, y, feature_true, signal, info, train_idx, test_idx


def save_dataset(path, condition, seed):
    X, y, truth, signal, info, tr, te = simulate_condition(condition, seed)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=X.cpu().numpy(), y=y.cpu().numpy(), signal=signal.cpu().numpy(),
        feature_true=truth.cpu().numpy(), train_idx=np.asarray(tr), test_idx=np.asarray(te),
        condition=np.asarray(condition), seed=np.asarray(int(seed)),
    )
    path.with_suffix(".json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return path
