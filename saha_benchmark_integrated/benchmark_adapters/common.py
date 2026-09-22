from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch


def load_npz(path):
    z = np.load(path, allow_pickle=False)
    tr = z['train_idx'].astype(int); te = z['test_idx'].astype(int)
    return {
        'Xtr': z['X'][tr].astype(np.float32), 'ytr': z['y'][tr].astype(np.float32),
        'Xte': z['X'][te].astype(np.float32), 'yte': z['y'][te].astype(np.float32),
        'signal_te': z['signal'][te].astype(np.float32),
        'truth': (z['feature_true'] > 0.5).astype(bool),
        'p': int(z['X'].shape[1]), 'ntrain': int(len(tr)), 'ntest': int(len(te)),
    }


def device_from_arg(x):
    x = str(x).lower()
    if x == 'auto': return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if x == 'gpu': x = 'cuda'
    return torch.device(x)


def write_json(payload, output):
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    clean = {}
    for k, v in payload.items():
        if isinstance(v, np.ndarray): clean[k] = v.tolist()
        elif isinstance(v, (np.integer,)): clean[k] = int(v)
        elif isinstance(v, (np.floating,)): clean[k] = float(v)
        elif torch.is_tensor(v): clean[k] = v.detach().cpu().numpy().tolist()
        else: clean[k] = v
    Path(output).write_text(json.dumps(clean, indent=2), encoding='utf-8')


def add_common_args(p):
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--repo-root', required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--h1', type=int, default=20)
    p.add_argument('--h2', type=int, default=10)
    p.add_argument('--device', default='auto')
    p.add_argument('--epochs', type=int, default=1200)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--draws', type=int, default=100)
    p.add_argument('--verbose', action='store_true')
    return p


def active_path_masks(weights):
    """Return active-path masks for sequential Linear weight matrices.
    weights: [W1(h1,p), W2(h2,h1), ..., WL(out,hL)].
    A connection is retained only if it lies on a complete input-output path.
    """
    masks = [np.asarray(w) != 0 for w in weights]
    if len(masks) < 1: return [], np.array([], dtype=bool)
    # forward reachability of units from any input
    reaches = []
    prev = np.ones(masks[0].shape[1], dtype=bool)
    for m in masks:
        cur = (m[:, prev].any(axis=1) if prev.any() else np.zeros(m.shape[0], dtype=bool))
        reaches.append(cur); prev = cur
    # backward reachability to output
    backs = [None] * len(masks)
    nxt = np.ones(masks[-1].shape[0], dtype=bool)
    for ell in range(len(masks)-1, -1, -1):
        m = masks[ell]
        prev_back = (m[nxt, :].any(axis=0) if nxt.any() else np.zeros(m.shape[1], dtype=bool))
        backs[ell] = nxt
        nxt = prev_back
    out = []
    prev_reach = np.ones(masks[0].shape[1], dtype=bool)
    for ell, m in enumerate(masks):
        out.append(m & backs[ell][:, None] & prev_reach[None, :])
        prev_reach = reaches[ell]
    selected_features = out[0].any(axis=0) if out else np.zeros(masks[0].shape[1], dtype=bool)
    return out, selected_features
