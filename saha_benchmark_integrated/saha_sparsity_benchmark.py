#!/usr/bin/env python3
"""Unified sparsity benchmark on the Saha-Liu-Liang (2024) nonlinear DGP.

Methods supported by this runner
--------------------------------
Built in:
  - DSS-LVR (uses the local NFlow project: Python/bnn_train.py)
  - IS-ANN-L1 (compact PyTorch implementation; native |w| >= 0.005 pruning)
  - LBBNN-LRT, LBBNN-FLOW, ISLaB-FLOW (calls the CRAN R package LBBNN)

Official-code adapters bundled with this runner:
  - SS-GL / SS-GHS: official Jantre layer and prior implementations
  - Laplace-SpaM: official bundled Laplace marginal-likelihood implementation and OPD score
  - wsBNN: official weight-sharing layer, KL and feature-PIP implementation

The adapters only replace the original hard-coded dataset/driver so all methods
see the same Saha data and 20/10 regression architecture. They do not replace the
methods' native priors or sparsification rules. Clone the official repositories
with benchmark_adapters/setup_external_repos.py before running them.

No training history is saved. The only persistent outputs are:
  <out>/saha_dgp_seed_<seed>.npz
  <out>/saha_sparsity_results_seed_<seed>.csv

Example
-------
python saha_sparsity_benchmark.py --project-root . --methods dss_lvr,is_ann_l1
python saha_sparsity_benchmark.py --methods lbbnn_lrt,lbbnn_flow,islab_flow
python saha_sparsity_benchmark.py --adapter-dir ./benchmark_adapters --methods ss_gl,ss_ghs,laplace_spam,wsbnn
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyTorch is required for this benchmark runner.") from exc


ALL_METHODS = (
    "dss_lvr",
    "is_ann_l1",
    "lbbnn_lrt",
    "lbbnn_flow",
    "islab_flow",
    "ss_gl",
    "ss_ghs",
    "laplace_spam",
    "wsbnn",
)

ADAPTER_METHODS = ("ss_gl", "ss_ghs", "laplace_spam", "wsbnn")

METHOD_LABELS = {
    "dss_lvr": "DSS-LVR",
    "is_ann_l1": "IS-ANN-L1",
    "lbbnn_lrt": "LBBNN-LRT",
    "lbbnn_flow": "LBBNN-FLOW",
    "islab_flow": "ISLaB-FLOW",
    "ss_gl": "SS-GL",
    "ss_ghs": "SS-GHS",
    "laplace_spam": "Laplace-SpaM",
    "wsbnn": "wsBNN",
}

NATIVE_RULES = {
    "dss_lvr": "MPM on hard V>tau states; induce retained edges from selected feature/unit groups",
    "is_ann_l1": "L1-trained input-skip ANN; retain weights with |w| >= 0.005",
    "lbbnn_lrt": "LBBNN median-probability model; PIP > 0.5; remove weights outside active paths",
    "lbbnn_flow": "LBBNN median-probability model; PIP > 0.5; remove weights outside active paths",
    "islab_flow": "ISLaB median-probability model; PIP > 0.5; remove weights outside active paths",
    "ss_gl": "SS-GL posterior node inclusion PIP > 0.5; connection sparsity induced by retained nodes",
    "ss_ghs": "SS-GHS posterior node inclusion PIP > 0.5; connection sparsity induced by retained nodes",
    "laplace_spam": "SpaM OPD posterior-precision x squared-weight score; pre-specified pruning level",
    "wsbnn": "wsBNN shared feature PIP ranking; top-10 feature rule used in its simulations",
}

RESULT_COLUMNS = [
    "method", "status", "seed", "n", "p", "n_active",
    "mse_signal", "r2_signal", "mse_y", "rmse_y",
    "tpr", "fpr", "accuracy", "selected_support",
    "retained_weights", "candidate_weights", "dparam",
    "d_edge", "d_path", "native_density", "runtime_sec",
    "native_rule", "note",
]


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def saha_nonlinear_dgp(
    n: int = 2000,
    p: int = 100,
    pi: float = 0.2,
    alpha: float = 2.0,
    sigma2: float = 1.0,
    seed: int = 400,
    train_fraction: float = 0.8,
) -> dict:
    """Experiment-3/4 style nonlinear DGP from Saha, Liu & Liang (2024).

    X_j ~ N(0,1), Z_j ~ Bernoulli(pi), beta_j=j/alpha,
    f(x)=exp(|x|)-2x+sin(2*pi*x),
    signal_i=sum_j f(X_ij) beta_j Z_j, y=signal+N(0,sigma2).
    """
    rng = np.random.default_rng(int(seed))
    n, p = int(n), int(p)
    X = rng.normal(size=(n, p)).astype(np.float32)
    z = rng.binomial(1, float(pi), size=p).astype(np.int8)
    beta = (np.arange(1, p + 1, dtype=np.float64) / float(alpha)).astype(np.float32)
    fx = np.exp(np.abs(X)).astype(np.float32) - 2.0 * X + np.sin(2.0 * np.pi * X).astype(np.float32)
    signal = (fx * (beta * z)[None, :]).sum(axis=1).astype(np.float32)
    y = (signal + math.sqrt(float(sigma2)) * rng.normal(size=n)).astype(np.float32)
    perm = rng.permutation(n)
    n_train = int(round(float(train_fraction) * n))
    train_idx = perm[:n_train].astype(np.int64)
    test_idx = perm[n_train:].astype(np.int64)
    return {
        "X": X, "y": y, "signal": signal, "feature_true": z.astype(np.float32),
        "beta": beta, "train_idx": train_idx, "test_idx": test_idx,
        "n": n, "p": p, "pi": float(pi), "alpha": float(alpha),
        "sigma2": float(sigma2), "seed": int(seed),
    }


def save_dgp(data: dict, path: Path) -> None:
    np.savez_compressed(
        path,
        X=data["X"], y=data["y"], signal=data["signal"],
        feature_true=data["feature_true"], beta=data["beta"],
        train_idx=data["train_idx"], test_idx=data["test_idx"],
        n=np.array(data["n"]), p=np.array(data["p"]), pi=np.array(data["pi"]),
        alpha=np.array(data["alpha"]), sigma2=np.array(data["sigma2"]), seed=np.array(data["seed"]),
    )


def selection_metrics(selected_features, truth) -> dict:
    if selected_features is None:
        return {"tpr": np.nan, "fpr": np.nan, "accuracy": np.nan, "selected_support": np.nan}
    selected = np.asarray(selected_features).reshape(-1).astype(bool)
    target = np.asarray(truth).reshape(-1) > 0.5
    if selected.size != target.size:
        raise ValueError(f"selected_features has length {selected.size}, expected {target.size}")
    tp = int(np.sum(selected & target))
    fp = int(np.sum(selected & ~target))
    fn = int(np.sum(~selected & target))
    tn = int(np.sum(~selected & ~target))
    return {
        "tpr": tp / max(tp + fn, 1),
        "fpr": fp / max(fp + tn, 1),
        "accuracy": (tp + tn) / max(target.size, 1),
        "selected_support": int(selected.sum()),
    }


def prediction_metrics(y_pred, data: dict) -> dict:
    idx = data["test_idx"]
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    signal = np.asarray(data["signal"][idx], dtype=float)
    y = np.asarray(data["y"][idx], dtype=float)
    if pred.size != idx.size:
        raise ValueError(f"y_pred_test has length {pred.size}, expected {idx.size}")
    mse_signal = float(np.mean((pred - signal) ** 2))
    sst = float(np.sum((signal - signal.mean()) ** 2))
    r2_signal = float(1.0 - np.sum((pred - signal) ** 2) / max(sst, 1e-12))
    mse_y = float(np.mean((pred - y) ** 2))
    return {
        "mse_signal": mse_signal,
        "r2_signal": r2_signal,
        "mse_y": mse_y,
        "rmse_y": float(math.sqrt(mse_y)),
    }


def base_row(method: str, data: dict) -> dict:
    return {
        "method": METHOD_LABELS[method], "status": "OK", "seed": int(data["seed"]),
        "n": int(data["n"]), "p": int(data["p"]),
        "n_active": int(np.asarray(data["feature_true"]).sum()),
        "mse_signal": np.nan, "r2_signal": np.nan, "mse_y": np.nan, "rmse_y": np.nan,
        "tpr": np.nan, "fpr": np.nan, "accuracy": np.nan, "selected_support": np.nan,
        "retained_weights": np.nan, "candidate_weights": np.nan, "dparam": np.nan,
        "d_edge": np.nan, "d_path": np.nan, "native_density": np.nan,
        "runtime_sec": np.nan, "native_rule": NATIVE_RULES[method], "note": "",
    }


def finish_row(method: str, data: dict, payload: dict) -> dict:
    row = base_row(method, data)
    row.update(prediction_metrics(payload["y_pred_test"], data))
    row.update(selection_metrics(payload.get("selected_features"), data["feature_true"]))
    retained = payload.get("retained_weights", np.nan)
    candidate = payload.get("candidate_weights", np.nan)
    dparam = payload.get("dparam", np.nan)
    if (not np.isfinite(float(dparam)) if np.isscalar(dparam) else True):
        if np.isfinite(float(retained)) and np.isfinite(float(candidate)) and float(candidate) > 0:
            dparam = float(retained) / float(candidate)
    row.update({
        "retained_weights": retained,
        "candidate_weights": candidate,
        "dparam": dparam,
        "d_edge": payload.get("d_edge", np.nan),
        "d_path": payload.get("d_path", np.nan),
        "native_density": payload.get("native_density", np.nan),
        "runtime_sec": payload.get("runtime_sec", np.nan),
        "native_rule": payload.get("native_rule", NATIVE_RULES[method]),
        "note": payload.get("note", ""),
    })
    return row


def skipped_row(method: str, data: dict, note: str) -> dict:
    row = base_row(method, data)
    row["status"] = "SKIPPED"
    row["note"] = str(note)
    return row


# -----------------------------------------------------------------------------
# DSS-LVR
# -----------------------------------------------------------------------------

def _import_dss_trainer(project_root: Path):
    root = str(project_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        mod = importlib.import_module("Python.bnn_train")
    except Exception as exc:
        raise RuntimeError(
            "Could not import Python.bnn_train from --project-root. Run this script from the NFlow root "
            "or pass --project-root /path/to/NFlow."
        ) from exc
    fn = getattr(mod, "train_grouped_bnn_fast", None)
    if fn is None:
        raise RuntimeError("Python.bnn_train.train_grouped_bnn_fast was not found.")
    return fn


def run_dss_lvr(data: dict, args) -> dict:
    trainer = _import_dss_trainer(Path(args.project_root))
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    X = torch.as_tensor(data["X"], dtype=torch.float32, device=device)
    y = torch.as_tensor(data["y"], dtype=torch.float32, device=device)
    signal = torch.as_tensor(data["signal"], dtype=torch.float32, device=device)
    tr = torch.as_tensor(data["train_idx"], device=device)
    te = torch.as_tensor(data["test_idx"], device=device)
    Xtr, ytr = X.index_select(0, tr), y.index_select(0, tr)
    Xte, ste = X.index_select(0, te), signal.index_select(0, te)
    diag_n = min(max(128, len(data["train_idx"]) // 10), len(data["train_idx"]))
    Xdiag, sdiag = Xtr[:diag_n], signal.index_select(0, tr[:diag_n])
    truth = {"feature_true": data["feature_true"], "n_true_units": 0, "n_interactions": 0, "n_quadratic": 0}
    kwargs = dict(
        truth=truth, selection_mode="feature_unit_induced_edge", hidden_dims=(args.h1, args.h2),
        sigma2=data["sigma2"], K_flow=args.dss_k, flow_type="iaf",
        iaf_ordering_scheme="cyclic3", iaf_shuffle_within_role=True,
        epochs=args.dss_epochs, warmup_epochs=args.dss_warmup, lr=args.dss_lr,
        R_train=args.dss_r_train, R_eval=args.dss_r_eval, R_final=args.dss_r_final,
        eval_every=max(args.dss_epochs, 1), seed=args.fit_seed,
        support_threshold=0.5, report_structure=True,
    )
    sig = inspect.signature(trainer)
    if "slab_init" in sig.parameters:
        kwargs.update(slab_init="auto", slab_sd_ratio=0.1, slab_bias_sd=0.02)
    started = time.perf_counter()
    sink = contextlib.nullcontext() if args.verbose else contextlib.redirect_stdout(io.StringIO())
    with sink:
        out = trainer(Xtr, ytr, Xdiag, sdiag, Xte, ste, **kwargs)
    runtime = time.perf_counter() - started
    model, xi = out["model"], out["xi"]
    with torch.no_grad():
        pred = model.decoder(Xte, xi).mean(0).detach().cpu().numpy()
        fpip = model.decoder.feature_semantics(xi)["active"].float().mean(0)
        upip = model.decoder.unit_semantics(xi)["active"].float().mean(0)
    selected_f = fpip.cpu().numpy() > 0.5
    selected_u = upip.cpu().numpy() > 0.5
    h1, h2, p = int(args.h1), int(args.h2), int(data["p"])
    u1 = int(selected_u[:h1].sum())
    u2 = int(selected_u[h1:h1 + h2].sum())
    retained = int(selected_f.sum()) * u1 + u1 * u2 + u2
    candidate = p * h1 + h1 * h2 + h2
    result = out.get("result", {})
    return {
        "y_pred_test": pred,
        "selected_features": selected_f,
        "retained_weights": retained,
        "candidate_weights": candidate,
        "dparam": retained / candidate,
        "d_edge": result.get("edge_density", result.get("network_density", np.nan)),
        "d_path": result.get("path_density", np.nan),
        "runtime_sec": runtime,
        "native_rule": NATIVE_RULES["dss_lvr"],
        "note": f"MPM: {int(selected_f.sum())} features, {u1}+{u2} hidden units",
    }


# -----------------------------------------------------------------------------
# IS-ANN-L1: compact native-rule baseline
# -----------------------------------------------------------------------------

class ISANNL1(nn.Module):
    def __init__(self, p: int, h1: int, h2: int):
        super().__init__()
        self.p, self.h1, self.h2 = int(p), int(h1), int(h2)
        self.fc1 = nn.Linear(p, h1)
        self.fc2 = nn.Linear(h1 + p, h2)
        self.out = nn.Linear(h2 + p, 1)
        for layer in (self.fc1, self.fc2, self.out):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)

    def forward(self, x, masks=None):
        if masks is None:
            w1, w2, w3 = self.fc1.weight, self.fc2.weight, self.out.weight
        else:
            w1, w2, w3 = self.fc1.weight * masks[0], self.fc2.weight * masks[1], self.out.weight * masks[2]
        h1 = F.relu(F.linear(x, w1, self.fc1.bias))
        h2 = F.relu(F.linear(torch.cat([h1, x], dim=1), w2, self.fc2.bias))
        return F.linear(torch.cat([h2, x], dim=1), w3, self.out.bias).squeeze(1)

    def connection_weights(self):
        return (self.fc1.weight, self.fc2.weight, self.out.weight)


def _selected_features_from_is_masks(masks, p: int, h1: int, h2: int) -> np.ndarray:
    m1, m2, m3 = [m.detach().cpu().bool().numpy() for m in masks]
    # m1: h1 x p; m2: h2 x (h1+p); m3: 1 x (h2+p)
    selected = np.zeros(p, dtype=bool)
    out_h2 = m3[0, :h2]
    selected |= m3[0, h2:h2 + p]
    for j in range(p):
        if np.any(m2[:, h1 + j] & out_h2):
            selected[j] = True
            continue
        h1_from_j = m1[:, j]
        if np.any((m2[:, :h1] & h1_from_j[None, :]).any(axis=1) & out_h2):
            selected[j] = True
    return selected


def run_is_ann_l1(data: dict, args) -> dict:
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(args.fit_seed)
    tr, te = data["train_idx"], data["test_idx"]
    Xtr = torch.as_tensor(data["X"][tr], dtype=torch.float32)
    ytr = torch.as_tensor(data["y"][tr], dtype=torch.float32)
    Xte = torch.as_tensor(data["X"][te], dtype=torch.float32, device=device)
    ds = TensorDataset(Xtr, ytr)
    gen = torch.Generator().manual_seed(args.fit_seed + 17)
    dl = DataLoader(ds, batch_size=args.is_ann_batch, shuffle=True, generator=gen)
    model = ISANNL1(data["p"], args.h1, args.h2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.is_ann_lr)
    started = time.perf_counter()
    model.train()
    for _ in range(args.is_ann_epochs):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            mse = F.mse_loss(pred, yb)
            l1 = sum(w.abs().sum() for w in model.connection_weights())
            loss = mse + args.is_ann_lambda * l1 / len(tr)
            loss.backward()
            opt.step()
    runtime = time.perf_counter() - started
    threshold = 0.005
    with torch.no_grad():
        masks = tuple((w.abs() >= threshold).to(w.dtype) for w in model.connection_weights())
        pred = model(Xte, masks=masks).cpu().numpy()
    selected = _selected_features_from_is_masks(masks, data["p"], args.h1, args.h2)
    retained = int(sum(m.sum().item() for m in masks))
    candidate = int(sum(m.numel() for m in masks))
    return {
        "y_pred_test": pred, "selected_features": selected,
        "retained_weights": retained, "candidate_weights": candidate,
        "dparam": retained / candidate, "runtime_sec": runtime,
        "native_rule": NATIVE_RULES["is_ann_l1"],
        "note": f"lambda={args.is_ann_lambda:g}; fixed native weight cutoff=0.005",
    }


# -----------------------------------------------------------------------------
# CRAN LBBNN package: LBBNN-LRT / LBBNN-FLOW / ISLaB-FLOW
# -----------------------------------------------------------------------------

LBBNN_R_TEMPLATE = r'''
suppressPackageStartupMessages({library(torch); library(LBBNN); library(jsonlite)})
args <- commandArgs(trailingOnly=TRUE)
method <- args[[1]]; train_x_path <- args[[2]]; train_y_path <- args[[3]]
test_x_path <- args[[4]]; test_y_path <- args[[5]]; out_path <- args[[6]]
seed <- as.integer(args[[7]]); h1 <- as.integer(args[[8]]); h2 <- as.integer(args[[9]])
epochs <- as.integer(args[[10]]); lr <- as.numeric(args[[11]]); draws <- as.integer(args[[12]])
device <- args[[13]]
Xtr <- as.matrix(read.csv(train_x_path, header=FALSE)); ytr <- as.numeric(read.csv(train_y_path, header=FALSE)[[1]])
Xte <- as.matrix(read.csv(test_x_path, header=FALSE)); yte <- as.numeric(read.csv(test_y_path, header=FALSE)[[1]])
xt <- torch_tensor(Xtr, dtype=torch_float()); yt <- torch_tensor(ytr, dtype=torch_float())
xv <- torch_tensor(Xte, dtype=torch_float()); yv <- torch_tensor(yte, dtype=torch_float())
train_loader <- dataloader(tensor_dataset(xt,yt), batch_size=min(256,nrow(Xtr)), shuffle=TRUE)
test_loader <- dataloader(tensor_dataset(xv,yv), batch_size=nrow(Xte), shuffle=FALSE)
p <- ncol(Xtr); sizes <- c(p,h1,h2,1); prior <- rep(0.5,3); stds <- rep(1,3)
input_skip <- method == "islab_flow"; flow <- method %in% c("lbbnn_flow","islab_flow")
torch_manual_seed(seed)
model <- lbbnn_net(problem_type="regression", sizes=sizes, prior=prior, std=stds,
                   inclusion_inits="polarized", input_skip=input_skip, flow=flow,
                   num_transforms=2, dims=c(50,50), device=device,
                   bias_inclusion_prob=FALSE, weight_init="he")
t0 <- proc.time()[[3]]
suppressMessages(train_lbbnn(epochs=epochs, LBBNN=model, lr=lr,
                             train_dl=train_loader, device=device, verbose=FALSE))
runtime <- proc.time()[[3]] - t0
pred_draws <- predict(model, newdata=test_loader, mpm=TRUE, draws=draws, device=device)
a <- as.array(pred_draws); pred <- as.numeric(apply(a,c(2,3),mean)[,1])
val <- validate_lbbnn(model, num_samples=draws, test_dl=test_loader, device=device)
# Native final-model density: PIP>0.5 and weights not belonging to complete active paths.
dparam <- as.numeric(val$density_active_path); native_density <- as.numeric(val$density)
# Feature selection: input-skip has an official active-path input inclusion utility.
selected <- NULL
if (input_skip) {
    inc <- get_input_inclusions(model)
    selected <- as.logical(rowSums(inc) > 0)
} else {
    # Standard LBBNN has no exported input-selection summary. Compute paths, then
    # inspect the first-layer active-path matrix. Rows are targets, columns inputs.
    model$compute_paths()
    ap <- as.array(model$layers[[1]]$alpha_active_path$detach()$cpu())
    selected <- as.logical(colSums(ap > 0) > 0)
}
out <- list(y_pred_test=pred, selected_features=selected, dparam=dparam,
            native_density=native_density, runtime_sec=runtime,
            native_rule="PIP > 0.5 median-probability model; inactive-path weights removed")
write_json(out, out_path, auto_unbox=TRUE, digits=16)
'''


def run_lbbnn(method: str, data: dict, args) -> dict:
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError("Rscript not found. Install R, torch, jsonlite, and CRAN package LBBNN.")
    tr, te = data["train_idx"], data["test_idx"]
    with tempfile.TemporaryDirectory(prefix="dsslvr_lbbnn_") as td:
        td = Path(td)
        files = {
            "xtr": td / "xtr.csv", "ytr": td / "ytr.csv",
            "xte": td / "xte.csv", "yte": td / "yte.csv",
            "r": td / "run.R", "out": td / "out.json",
        }
        np.savetxt(files["xtr"], data["X"][tr], delimiter=",")
        np.savetxt(files["ytr"], data["y"][tr], delimiter=",")
        np.savetxt(files["xte"], data["X"][te], delimiter=",")
        np.savetxt(files["yte"], data["y"][te], delimiter=",")
        files["r"].write_text(LBBNN_R_TEMPLATE, encoding="utf-8")
        r_device = "gpu" if args.device in {"cuda", "gpu"} else "cpu"
        cmd = [rscript, str(files["r"]), method, str(files["xtr"]), str(files["ytr"]),
               str(files["xte"]), str(files["yte"]), str(files["out"]), str(args.fit_seed),
               str(args.h1), str(args.h2), str(args.lbbnn_epochs), str(args.lbbnn_lr),
               str(args.lbbnn_draws), r_device]
        proc = subprocess.run(cmd, capture_output=not args.verbose, text=True)
        if proc.returncode != 0:
            err = proc.stderr[-2500:] if proc.stderr else "R adapter failed without stderr"
            raise RuntimeError(err)
        payload = json.loads(files["out"].read_text(encoding="utf-8"))
    payload["selected_features"] = np.asarray(payload.get("selected_features"), dtype=bool)
    # density_active_path is already a fraction of retained native weights.
    payload["candidate_weights"] = np.nan
    payload["retained_weights"] = np.nan
    payload["note"] = "dparam = LBBNN density_active_path; native_density = raw PIP>0.5 weight density"
    return payload


# -----------------------------------------------------------------------------
# Official-code adapters (SS-GL, SS-GHS, Laplace-SpaM, wsBNN)
# -----------------------------------------------------------------------------

def run_official_adapter(method: str, data_path: Path, data: dict, args) -> dict:
    adapter_dir = Path(args.adapter_dir) if args.adapter_dir else Path(__file__).resolve().parent / "benchmark_adapters"
    script = adapter_dir / f"{method}.py"
    if not script.exists():
        raise RuntimeError(f"Adapter not found: {script}")
    repo_map = {
        "ss_gl": "SS_Group_Shrinkage_New", "ss_ghs": "SS_Group_Shrinkage_New",
        "laplace_spam": "spam-pruning", "wsbnn": "wsBNN",
    }
    repo_root = Path(args.external_root) / repo_map[method]
    if not repo_root.exists():
        raise RuntimeError(
            f"Official repository not found: {repo_root}. Run: "
            f"python {adapter_dir / 'setup_external_repos.py'} --root {args.external_root}"
        )
    with tempfile.TemporaryDirectory(prefix=f"dsslvr_{method}_") as td:
        out = Path(td) / "result.json"
        if method in {"ss_gl", "ss_ghs"}:
            epochs, lr, batch, draws = args.ss_epochs, args.ss_lr, args.ss_batch, args.ss_draws
        elif method == "wsbnn":
            epochs, lr, batch, draws = args.ws_epochs, args.ws_lr, args.ws_batch, args.ws_draws
        else:
            epochs, lr, batch, draws = args.spam_epochs, args.spam_lr, args.spam_batch, 1
        cmd = [sys.executable, str(script), "--data", str(data_path), "--output", str(out),
               "--repo-root", str(repo_root), "--seed", str(args.fit_seed),
               "--h1", str(args.h1), "--h2", str(args.h2), "--device", str(args.device),
               "--epochs", str(epochs), "--lr", str(lr), "--batch", str(batch), "--draws", str(draws)]
        if method == "wsbnn":
            cmd += ["--mc-train", str(args.ws_mc_train), "--topk", str(args.ws_topk)]
        elif method == "laplace_spam":
            cmd += ["--selection", str(args.spam_selection), "--prune", str(args.spam_prune),
                    "--grid", str(args.spam_grid), "--tolerance", str(args.spam_tolerance),
                    "--burnin", str(args.spam_burnin), "--hypersteps", str(args.spam_hypersteps),
                    "--marglik-frequency", str(args.spam_frequency)]
        if args.verbose:
            cmd.append("--verbose")
        proc = subprocess.run(cmd, capture_output=not args.verbose, text=True)
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "adapter failed without output")[-4000:]
            raise RuntimeError(err)
        payload = json.loads(out.read_text(encoding="utf-8"))
    if "y_pred_test" not in payload:
        raise ValueError(f"{method} adapter did not return y_pred_test")
    if "selected_features" in payload and payload["selected_features"] is not None:
        sf = np.asarray(payload["selected_features"])
        if sf.dtype != bool:
            if sf.size != data["p"]:
                mask = np.zeros(data["p"], dtype=bool); mask[sf.astype(int)] = True; sf = mask
            else:
                sf = sf.astype(bool)
        payload["selected_features"] = sf
    payload.setdefault("native_rule", NATIVE_RULES[method])
    return payload

def print_results(df: pd.DataFrame) -> None:
    cols = ["method", "status", "mse_signal", "r2_signal", "tpr", "fpr", "accuracy", "selected_support", "dparam", "runtime_sec"]
    shown = df[cols].copy()
    for c in ("mse_signal", "r2_signal", "tpr", "fpr", "accuracy", "dparam"):
        shown[c] = pd.to_numeric(shown[c], errors="coerce").round(4)
    shown["runtime_sec"] = pd.to_numeric(shown["runtime_sec"], errors="coerce").round(1)
    print("\nFinal benchmark summary")
    print(shown.to_string(index=False))


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--out", default="results_saha_sparsity")
    p.add_argument("--project-root", default=".")
    p.add_argument("--adapter-dir", default=None)
    p.add_argument("--external-root", default="external_methods")
    p.add_argument("--methods", default=",".join(ALL_METHODS))
    p.add_argument("--seed", type=int, default=400)
    p.add_argument("--fit-seed", type=int, default=None)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--p", type=int, default=100)
    p.add_argument("--pi", type=float, default=0.2)
    p.add_argument("--alpha", type=float, default=2.0)
    p.add_argument("--sigma2", type=float, default=1.0)
    p.add_argument("--h1", type=int, default=20)
    p.add_argument("--h2", type=int, default=10)
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "gpu"))
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--fail-on-skip", action="store_true")
    # DSS-LVR
    p.add_argument("--dss-k", type=int, default=6)
    p.add_argument("--dss-epochs", type=int, default=1200)
    p.add_argument("--dss-warmup", type=int, default=300)
    p.add_argument("--dss-lr", type=float, default=3e-4)
    p.add_argument("--dss-r-train", type=int, default=32)
    p.add_argument("--dss-r-eval", type=int, default=128)
    p.add_argument("--dss-r-final", type=int, default=500)
    # IS-ANN-L1
    p.add_argument("--is-ann-epochs", type=int, default=2000)
    p.add_argument("--is-ann-lr", type=float, default=1e-3)
    p.add_argument("--is-ann-lambda", type=float, default=0.01)
    p.add_argument("--is-ann-batch", type=int, default=128)
    # LBBNN family
    p.add_argument("--lbbnn-epochs", type=int, default=2000)
    p.add_argument("--lbbnn-lr", type=float, default=0.01)
    p.add_argument("--lbbnn-draws", type=int, default=500)
    # SS-GL / SS-GHS official adapters
    p.add_argument("--ss-epochs", type=int, default=1200)
    p.add_argument("--ss-lr", type=float, default=1e-3)
    p.add_argument("--ss-batch", type=int, default=128)
    p.add_argument("--ss-draws", type=int, default=100)
    # wsBNN official adapter
    p.add_argument("--ws-epochs", type=int, default=1000)
    p.add_argument("--ws-lr", type=float, default=1e-3)
    p.add_argument("--ws-batch", type=int, default=128)
    p.add_argument("--ws-draws", type=int, default=100)
    p.add_argument("--ws-mc-train", type=int, default=30)
    p.add_argument("--ws-topk", type=int, default=10)
    # Laplace-SpaM official adapter
    p.add_argument("--spam-epochs", type=int, default=300)
    p.add_argument("--spam-lr", type=float, default=1e-3)
    p.add_argument("--spam-batch", type=int, default=128)
    p.add_argument("--spam-selection", choices=("fixed", "val_tolerance"), default="fixed")
    p.add_argument("--spam-prune", type=int, default=50)
    p.add_argument("--spam-grid", default="20,40,50,60,70,80,90")
    p.add_argument("--spam-tolerance", type=float, default=0.01)
    p.add_argument("--spam-burnin", type=int, default=20)
    p.add_argument("--spam-hypersteps", type=int, default=10)
    p.add_argument("--spam-frequency", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    if args.fit_seed is None:
        args.fit_seed = int(args.seed) + 100000
    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in ALL_METHODS]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Allowed: {ALL_METHODS}")
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    data = saha_nonlinear_dgp(args.n, args.p, args.pi, args.alpha, args.sigma2, args.seed)
    data_path = outdir / f"saha_dgp_seed_{args.seed}.npz"
    save_dgp(data, data_path)
    print(
        f"Saha nonlinear DGP | n={data['n']} p={data['p']} pi={data['pi']:.2f} "
        f"active={int(data['feature_true'].sum())} | train/test={len(data['train_idx'])}/{len(data['test_idx'])}"
    )
    print(f"Architecture: {data['p']} -> {args.h1} -> {args.h2} -> 1")
    rows = []
    for method in methods:
        print(f"\n[{METHOD_LABELS[method]}]")
        try:
            if method == "dss_lvr":
                payload = run_dss_lvr(data, args)
            elif method == "is_ann_l1":
                payload = run_is_ann_l1(data, args)
            elif method in {"lbbnn_lrt", "lbbnn_flow", "islab_flow"}:
                payload = run_lbbnn(method, data, args)
            else:
                payload = run_official_adapter(method, data_path, data, args)
            row = finish_row(method, data, payload)
            print(
                f"MSE(signal)={row['mse_signal']:.4g} R2={row['r2_signal']:.4f} "
                f"TPR={row['tpr']:.3f} FPR={row['fpr']:.3f} Acc={row['accuracy']:.3f} "
                f"Dparam={row['dparam']:.4f}"
            )
        except Exception as exc:
            row = skipped_row(method, data, f"{type(exc).__name__}: {exc}")
            print(f"SKIPPED: {row['note']}")
            if args.fail_on_skip:
                raise
        rows.append(row)
    df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    result_path = outdir / f"saha_sparsity_results_seed_{args.seed}.csv"
    df.to_csv(result_path, index=False)
    print_results(df)
    print(f"\nSaved data   : {data_path}")
    print(f"Saved results: {result_path}")


if __name__ == "__main__":
    main()
