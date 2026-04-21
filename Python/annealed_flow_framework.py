from __future__ import annotations

import copy
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


# ==========================================================
# Configuration dataclasses
# ==========================================================


@dataclass
class StagewiseAnnealConfig:
    tau_start: float = 1.0
    tau_end: float = 0.40
    warmup_epochs: int = 200
    n_anneal_stages: int = 5
    min_stage_epochs: int = 80
    max_stage_epochs: int = 220
    base_lr: float = 5e-5
    stage_lr_decay: float = 0.70
    min_lr_scale: float = 0.20
    num_samples: int = 128
    grad_clip: Optional[float] = 3.0
    elbo_beta: float = 1.0
    optimizer_cls: Any = torch.optim.Adam
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    eval_every: int = 25
    checkpoint_every: int = 25
    print_every: int = 25
    ema_beta: float = 0.90
    diag_R_train: int = 256
    diag_R_final: int = 4000
    support_threshold: float = 0.50
    instability_window: int = 2
    plateau_window: int = 2
    plateau_loss_rel: float = 1e-3
    plateau_pred_rel: float = 2e-3
    plateau_grad_slope: float = 0.05
    plateau_churn: float = 0.10
    plateau_abs_dS: int = 1
    alert_grad_slope: float = 0.25
    alert_support_jump_abs: int = 2
    alert_support_jump_rel: float = 0.15
    alert_loss_rel: float = 5e-4
    alert_pred_rel: float = 0.0
    alert_churn: float = 0.25
    retry_shrink_power: float = 0.5
    max_retries_per_drop: int = 2
    history_round_digits: int = 8


@dataclass
class SplitConfig:
    train_frac: float = 0.60
    val_frac: float = 0.20
    test_frac: float = 0.20
    seed: int = 123


@dataclass
class SaveConfig:
    output_dir: Optional[str] = None
    save_plots: bool = True
    save_history_csv: bool = True
    save_checkpoint_manifest: bool = True
    save_final_json: bool = True
    save_predictions_csv: bool = True
    save_support_sets_json: bool = True


# ==========================================================
# Utility functions
# ==========================================================


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

def make_optimizer(model, lr):
    return torch.optim.Adam(model.parameters(), lr=lr)

def save_model_state(model):
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}

def load_model_state(model, state):
    model.load_state_dict(state)

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


# ==========================================================
# Posterior sampling and hard-support diagnostics
# ==========================================================

def sample_posterior_latents(model, R=2000):
    model.eval()
    z0 = model.q0.rsample(R)
    eps, _ = model.posterior_flow(z0, return_logdet=True)
    dec = model.generative_model.decode(eps)
    return {k: dec[k].detach().cpu() for k in ["s", "u", "t", "beta"]}


def hard_support_from_draws(draws, support_threshold=0.5):
    s, u, t = draws["s"], draws["u"], draws["t"]
    if t.ndim == 1:
        t = t[:, None]

    ind = (u > t).float()
    support_mask = ind.mean(dim=0) > support_threshold
    support_idx = torch.where(support_mask)[0].tolist()
    beta_hard_samples = s * ind

    return {
        "support_idx": support_idx,
        "support_size": len(support_idx),
        "beta_hard_samples": beta_hard_samples,
    }


def posterior_predictions_from_hard_samples(X, beta_hard_samples):
    return X.float().cpu() @ beta_hard_samples.float().cpu().T


def predictive_metrics(X, y, beta_hard_samples, sigma2=None):
    pred_draws = posterior_predictions_from_hard_samples(X, beta_hard_samples)
    yhat = pred_draws.mean(dim=1)
    y = y.cpu().float().view(-1)

    resid = y - yhat
    mse = resid.pow(2).mean().item()
    rmse = mse**0.5
    mae = resid.abs().mean().item()

    ss_res = resid.pow(2).sum().item()
    ss_tot = ((y - y.mean())**2).sum().item()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    out = {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

    if sigma2 is not None:
        ll = -0.5 * (((y[:, None] - pred_draws)**2) / sigma2 + math.log(2 * math.pi * sigma2))
        out["heldout_loglik"] = torch.logsumexp(ll, dim=1).sub(math.log(pred_draws.shape[1])).mean().item()
        out["nll"] = -out["heldout_loglik"]

    return out


def selection_metrics_from_support(support_idx, beta_true, eps=1e-12):
    truth = (beta_true.cpu().abs() > eps).numpy().astype(int)
    pred = np.zeros_like(truth)
    pred[support_idx] = 1

    tp = ((pred == 1) & (truth == 1)).sum()
    fp = ((pred == 1) & (truth == 0)).sum()
    fn = ((pred == 0) & (truth == 1)).sum()

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


# ==========================================================
# Checkpoint diagnostics and selection
# ==========================================================


def evaluate_checkpoint(model, X_val, y_val, sigma2, R, support_threshold, prev_record, loss_ema, log_grad_ema,):
    draws = sample_posterior_latents(model, R=R)
    hard = hard_support_from_draws(draws, support_threshold=support_threshold)
    pred = predictive_metrics(X_val, y_val, hard["beta_hard_samples"], sigma2=sigma2)

    rec = {
        "support_idx": hard["support_idx"],
        "support_size": hard["support_size"],
        "loss_ema": float(loss_ema) if loss_ema is not None else float("nan"),
        "log_grad_ema": float(log_grad_ema) if log_grad_ema is not None else float("nan"),
        **{f"val_{k}": v for k, v in pred.items()},
    }

    if prev_record is None:
        rec["dS"] = 0
        rec["churn"] = 0.0
        rec["pred_improve_rel"] = 0.0
        rec["loss_improve_rel"] = 0.0
        rec["grad_slope"] = 0.0
    else:
        rec["dS"] = int(rec["support_size"] - prev_record["support_size"])
        rec["churn"] = float(jaccard_distance(rec["support_idx"], prev_record["support_idx"]))
        prev_mse = max(float(prev_record.get("val_mse", float("nan"))), 1e-12)
        cur_mse = float(rec.get("val_mse", float("nan")))
        rec["pred_improve_rel"] = float(max(prev_mse - cur_mse, 0.0) / prev_mse)
        prev_loss_ema = float(prev_record.get("loss_ema", float("nan")))
        cur_loss_ema = float(rec.get("loss_ema", float("nan")))
        denom = max(abs(prev_loss_ema), 1e-12)
        rec["loss_improve_rel"] = float(max(prev_loss_ema - cur_loss_ema, 0.0) / denom)
        rec["grad_slope"] = float(rec.get("log_grad_ema", float("nan")) - prev_record.get("log_grad_ema", float("nan")))
    return rec


def is_alert_record(rec, cfg):
    dS = rec["dS"]
    prev_S = max(rec["support_size"] - dS, 1)
    rel_jump = dS / prev_S if dS > 0 else 0.0

    return (
        rec["grad_slope"] >= cfg.alert_grad_slope
        and dS >= cfg.alert_support_jump_abs
        and rel_jump >= cfg.alert_support_jump_rel
        and rec["loss_improve_rel"] <= cfg.alert_loss_rel
        and rec["pred_improve_rel"] <= cfg.alert_pred_rel
        and rec["churn"] >= cfg.alert_churn
    )


def stage_can_advance(stage_records, epoch_in_stage, cfg):
    if epoch_in_stage < cfg.min_stage_epochs:
        return False
    if len(stage_records) < cfg.plateau_window:
        return False

    tail = stage_records[-cfg.plateau_window:]
    return all(
        rec["loss_improve_rel"] <= cfg.plateau_loss_rel
        and rec["pred_improve_rel"] <= cfg.plateau_pred_rel
        and abs(rec["dS"]) <= cfg.plateau_abs_dS
        and rec["churn"] <= cfg.plateau_churn
        and rec["grad_slope"] <= cfg.plateau_grad_slope
        for rec in tail
    )


def best_record_key(rec):
    return (rec["val_mse"], rec["support_size"])


def select_checkpoint_from_history(history_df, pred_eps_rel, loss_eps_rel, grad_quantile, ds_quantile, churn_quantile,):
    if history_df.empty:
        raise ValueError("history_df is empty")

    df = history_df.copy().reset_index(drop=True)
    best_pred = float(df["val_mse"].min())
    best_loss = float(df["loss_ema"].min())

    positive_dS = df["dS"].clip(lower=0)
    grad_thr = float(df["grad_slope"].quantile(grad_quantile))
    ds_thr = float(positive_dS.quantile(ds_quantile))
    churn_thr = float(df["churn"].quantile(churn_quantile))

    cand = df[
        (df["val_mse"] <= best_pred * (1.0 + pred_eps_rel))
        & (df["loss_ema"] <= best_loss * (1.0 + loss_eps_rel))
        & (df["grad_slope"] <= grad_thr)
        & (positive_dS <= ds_thr)
        & (df["churn"] <= churn_thr)
    ]

    if cand.empty:
        cand = df[
            (df["val_mse"] <= best_pred * (1.0 + 2.0 * pred_eps_rel))
            & (df["loss_ema"] <= best_loss * (1.0 + 2.0 * loss_eps_rel))
        ]

    if cand.empty:
        cand = df

    cand = cand.sort_values(
        by=["support_size", "val_mse", "loss_ema", "churn", "grad_slope", "epoch"],
        ascending=[True, True, True, True, True, True],
    )
    return int(cand.iloc[0]["ckpt_id"])


# ==========================================================
# Stagewise annealing training loop
# ==========================================================


def train_flow_stagewise(model, X_val, y_val, sigma2, cfg, device,):
    if device is None:
        device = next(model.parameters()).device

    optimizer = make_optimizer(model, cfg)
    loss_ema = EMA(beta=cfg.ema_beta)
    log_grad_ema = EMA(beta=cfg.ema_beta)

    planned_taus = rolling_stage_taus(cfg.tau_start, cfg.tau_end, cfg.n_anneal_stages)
    # planned_taus[0] is warm-up temperature; warm-up handled with cfg.warmup_epochs
    stage_taus = [planned_taus[0]] + planned_taus[1:]
    stage_retry_count: Dict[Tuple[float, float], int] = {}

    t_start = time.time()
    global_epoch = 0
    ckpt_id = 0
    history: List[Dict[str, Any]] = []
    checkpoints: Dict[int, Dict[str, Any]] = {}
    stage_summaries: List[Dict[str, Any]] = []
    prev_record: Optional[Dict[str, Any]] = None
    prev_stage_best_state: Optional[Dict[str, torch.Tensor]] = None
    prev_stage_best_record: Optional[Dict[str, Any]] = None

    stage_index = 0
    while stage_index < len(stage_taus):
        stage_repeated = False
        tau_now = float(stage_taus[stage_index])
        is_warmup = (stage_index == 0)
        lr_scale = max(cfg.min_lr_scale, cfg.stage_lr_decay ** max(stage_index - 1, 0))
        stage_lr = cfg.base_lr * lr_scale
        set_optimizer_lr(optimizer, stage_lr)
        model.generative_model.set_tau(tau_now)

        if is_warmup:
            stage_epoch_cap = cfg.warmup_epochs
        else:
            stage_epoch_cap = cfg.max_stage_epochs

        stage_records: List[Dict[str, Any]] = []
        stage_alerts = 0
        stage_best_state: Optional[Dict[str, torch.Tensor]] = None
        stage_best_record: Optional[Dict[str, Any]] = None

        for epoch_in_stage in range(1, stage_epoch_cap + 1):
            global_epoch += 1
            model.train()
            optimizer.zero_grad(set_to_none=True)
            model.generative_model.set_tau(tau_now)

            loss = model.neg_elbo(num_samples=cfg.num_samples, elbo_beta=cfg.elbo_beta)
            loss.backward()

            if cfg.grad_clip is not None:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
            else:
                grad_norm = 0.0
            optimizer.step()

            loss_val = float(loss.detach().cpu().item())
            loss_ema_val = loss_ema.update(loss_val)
            log_grad = math.log(max(grad_norm, 1e-12))
            log_grad_ema_val = log_grad_ema.update(log_grad)

            needs_eval = (
                epoch_in_stage == 1
                or epoch_in_stage % cfg.eval_every == 0
                or epoch_in_stage == stage_epoch_cap
            )

            if needs_eval:
                rec = evaluate_checkpoint(
                    model=model,
                    X_val=X_val,
                    y_val=y_val,
                    sigma2=sigma2,
                    R=cfg.diag_R_train,
                    support_threshold=cfg.support_threshold,
                    prev_record=prev_record,
                    loss_ema=loss_ema_val,
                    log_grad_ema=log_grad_ema_val,
                )
                rec.update({
                    "ckpt_id": ckpt_id,
                    "stage": stage_index,
                    "is_warmup": bool(is_warmup),
                    "epoch": global_epoch,
                    "epoch_in_stage": epoch_in_stage,
                    "tau": tau_now,
                    "lr": stage_lr,
                    "loss": loss_val,
                    "loss_ema": loss_ema_val,
                    "grad_norm": grad_norm,
                    "log_grad": log_grad,
                    "log_grad_ema": log_grad_ema_val,
                })
                rec["alert"] = bool(is_alert_record(rec, cfg))
                rec["best_so_far"] = False

                if stage_best_record is None or best_record_key(rec) < best_record_key(stage_best_record):
                    stage_best_record = copy.deepcopy(rec)
                    stage_best_state = save_model_state(model)
                    rec["best_so_far"] = True

                checkpoints[ckpt_id] = {
                    "state": save_model_state(model),
                    "meta": copy.deepcopy(rec),
                }
                ckpt_id += 1

                history.append(copy.deepcopy(rec))
                stage_records.append(copy.deepcopy(rec))
                prev_record = copy.deepcopy(rec)

                if epoch_in_stage % cfg.print_every == 0 or epoch_in_stage == 1 or rec["alert"]:
                    flag = "ALERT" if rec["alert"] else "ok"
                    star = "*" if rec["best_so_far"] else ""
                    print(
                        f"[stage {stage_index:02d} | epoch {global_epoch:04d}] "
                        f"tau={tau_now:.4f} lr={stage_lr:.2e} "
                        f"loss_ema={loss_ema_val:.6f} val_mse={rec['val_mse']:.6f} "
                        f"logg={log_grad_ema_val:.4f} dlogg={rec['grad_slope']:+.4f} "
                        f"S={rec['support_size']:3d} dS={rec['dS']:+3d} churn={rec['churn']:.3f} {flag}{star}"
                    )

                if rec["alert"] and not is_warmup:
                    stage_alerts += 1
                else:
                    stage_alerts = 0

                if (
                    not is_warmup
                    and epoch_in_stage <= cfg.eval_every * cfg.instability_window
                    and stage_alerts >= cfg.instability_window
                ):
                    # rollback and retry with a smaller temperature drop
                    if prev_stage_best_state is None or prev_stage_best_record is None:
                        break
                    load_model_state(model, prev_stage_best_state, device=device)
                    prev_tau = float(prev_stage_best_record["tau"])
                    key = (round(prev_tau, 8), round(tau_now, 8))
                    stage_retry_count[key] = stage_retry_count.get(key, 0) + 1
                    if stage_retry_count[key] > cfg.max_retries_per_drop:
                        print("[rollback] retry limit reached; stop annealing and keep previous stage best.")
                        stage_taus = stage_taus[:stage_index]
                        stage_index = len(stage_taus)
                        break
                    tau_retry = prev_tau * ((tau_now / prev_tau) ** cfg.retry_shrink_power)
                    if not (tau_retry < prev_tau - 1e-8 and tau_retry > tau_now + 1e-8):
                        print("[rollback] no valid intermediate tau; stop annealing and keep previous stage best.")
                        stage_taus = stage_taus[:stage_index]
                        stage_index = len(stage_taus)
                        break
                    stage_taus.insert(stage_index, tau_retry)
                    stage_repeated = True
                    print(
                        f"[rollback] stage {stage_index} unstable. restore prev best and retry with tau={tau_retry:.4f}"
                    )
                    break

                if (not is_warmup) and stage_can_advance(stage_records, epoch_in_stage, cfg):
                    print(
                        f"[advance] stage {stage_index} finished at epoch {global_epoch} with tau={tau_now:.4f}"
                    )
                    break

        # stage end bookkeeping
        if stage_best_record is not None:
            stage_summaries.append({
                "stage": stage_index,
                "tau": tau_now,
                "best_ckpt_id": int(stage_best_record["ckpt_id"]),
                "best_epoch": int(stage_best_record["epoch"]),
                "best_val_mse": float(stage_best_record["val_mse"]),
                "best_support_size": int(stage_best_record["support_size"]),
                "n_evals": len(stage_records),
            })
            prev_stage_best_record = copy.deepcopy(stage_best_record)
            prev_stage_best_state = copy.deepcopy(stage_best_state)

        if not stage_repeated:
            stage_index += 1

    runtime_sec = time.time() - t_start
    history_df = pd.DataFrame(history)
    if not history_df.empty:
        num_cols = history_df.select_dtypes(include=[np.number]).columns
        history_df[num_cols] = history_df[num_cols].round(cfg.history_round_digits)
    selected_ckpt_id = select_checkpoint_from_history(history_df)

    return {
        "history": history,
        "history_df": history_df,
        "checkpoints": checkpoints,
        "selected_ckpt_id": selected_ckpt_id,
        "stage_summaries": stage_summaries,
        "runtime_sec": runtime_sec,
        "config": asdict(cfg),
    }


# ==========================================================
# Final evaluation, tables, plots, saving
# ==========================================================


def build_variable_table(beta_hard_mean, selected_support, beta_true=None):
    beta_est = beta_hard_mean.cpu().numpy()
    p = len(beta_est)
    selected = np.zeros(p, dtype=int)
    selected[list(selected_support)] = 1

    df = pd.DataFrame({
        "j": np.arange(p),
        "beta_hard_mean": beta_est,
        "selected": selected,
    })

    if beta_true is not None:
        beta_true = beta_true.cpu().numpy()
        df["beta_true"] = beta_true
        df["truth"] = (np.abs(beta_true) > 1e-12).astype(int)

    return df


def compute_selection_frequency(history_df, p, ckpt_ids=None):
    if ckpt_ids is not None:
        history_df = history_df[history_df["ckpt_id"].isin(ckpt_ids)]
    freq = np.zeros(p)
    if len(history_df) == 0:
        return freq
    for support_idx in history_df["support_idx"]:
        freq[list(support_idx)] += 1
    return freq / len(history_df)


def plot_training_overview(history_df, savepath=None):
    if len(history_df) == 0:
        return
    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)

    axes[0].plot(history_df["epoch"], history_df["loss_ema"])
    axes[1].plot(history_df["epoch"], history_df["val_mse"])
    axes[2].plot(history_df["epoch"], history_df["log_grad_ema"])
    axes[3].step(history_df["epoch"], history_df["support_size"], where="post")

    axes[0].set_ylabel("loss")
    axes[1].set_ylabel("val mse")
    axes[2].set_ylabel("log grad")
    axes[3].set_ylabel("support")
    axes[3].set_xlabel("epoch")

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath)
    plt.close(fig)


def plot_support_vs_predictive(history_df, savepath=None):
    if len(history_df) == 0:
        return
    plt.figure(figsize=(6, 4))
    plt.scatter(history_df["support_size"], history_df["val_mse"], c=history_df["tau"])
    plt.xlabel("support size")
    plt.ylabel("val mse")
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)
    plt.close()


def plot_boundary_density(boundary, final_support, never_selected_idx, savepath=None):
    if boundary.numel() == 0:
        return

    plt.figure(figsize=(7, 4))
    if len(final_support) > 0:
        vals = boundary[:, final_support].reshape(-1).numpy()
        plt.hist(vals, bins=50, density=True, alpha=0.5, label="selected")
    if len(never_selected_idx) > 0:
        vals = boundary[:, never_selected_idx].reshape(-1).numpy()
        plt.hist(vals, bins=50, density=True, alpha=0.5, label="never")

    plt.xlabel("d = u - t")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)
    plt.close()


def plot_uncertainty_vs_abs_boundary(boundary, hard_freq, savepath=None):
    if boundary.numel() == 0:
        return

    x = boundary.mean(dim=0).abs().numpy()
    y = 4 * hard_freq * (1 - hard_freq)

    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, s=15)
    plt.xlabel("|E[d]|")
    plt.ylabel("instability")
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)
    plt.close()


def plot_support_overlap_heatmap(history_df, max_ckpts=20, savepath=None):
    if len(history_df) == 0:
        return

    df = history_df.sort_values("epoch").copy()
    if len(df) > max_ckpts:
        idx = np.linspace(0, len(df) - 1, max_ckpts).round().astype(int)
        df = df.iloc[idx]

    supports = df["support_idx"].tolist()
    n = len(supports)
    mat = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            mat[i, j] = jaccard_similarity(supports[i], supports[j])

    plt.figure(figsize=(6, 5))
    plt.imshow(mat, vmin=0, vmax=1)
    plt.colorbar(label="Jaccard")
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)
    plt.close()


def save_run_artifacts(out, save_cfg,):
    if save_cfg.output_dir is None:
        return
    outdir = ensure_dir(save_cfg.output_dir)
    assert outdir is not None

    history_df = out["history_df"]
    final = out["final"]

    if save_cfg.save_history_csv and isinstance(history_df, pd.DataFrame):
        hist_to_save = history_df.copy()
        if "support_idx" in hist_to_save.columns:
            hist_to_save["support_idx"] = hist_to_save["support_idx"].apply(lambda x: json.dumps(list(map(int, x))))
        hist_to_save.to_csv(os.path.join(outdir, "history.csv"), index=False)

    if save_cfg.save_checkpoint_manifest:
        manifest = []
        for ckpt_id, payload in out["checkpoints"].items():
            meta = payload["meta"].copy()
            meta["support_idx"] = json.dumps(list(map(int, meta["support_idx"])))
            manifest.append(meta)
        pd.DataFrame(manifest).to_csv(os.path.join(outdir, "checkpoint_manifest.csv"), index=False)

    if save_cfg.save_predictions_csv:
        final["var_table"].to_csv(os.path.join(outdir, "variable_table.csv"), index=False)
        final["pred_table"].to_csv(os.path.join(outdir, "prediction_table.csv"), index=False)

    if save_cfg.save_support_sets_json:
        support_sets = {
            "selected_ckpt_id": int(out["selected_ckpt_id"]),
            "selected_support": list(map(int, final["selected_support"])),
            "unstable_idx": list(map(int, final["unstable_idx"])),
            "never_selected_idx": list(map(int, final["never_selected_idx"])),
        }
        with open(os.path.join(outdir, "support_sets.json"), "w", encoding="utf-8") as f:
            json.dump(support_sets, f, indent=2)

    if save_cfg.save_final_json:
        final_json = {
            "selected_ckpt_id": int(out["selected_ckpt_id"]),
            "runtime_sec": float(out["runtime_sec"]),
            "stage_summaries": out["stage_summaries"],
            "checkpoint_meta": out["checkpoints"][out["selected_ckpt_id"]]["meta"],
            "selection_metrics": final["selection_metrics"],
            "train_metrics": final["train_metrics"],
            "val_metrics": final["val_metrics"],
            "test_metrics": final["test_metrics"],
        }
        final_json["checkpoint_meta"]["support_idx"] = list(map(int, final_json["checkpoint_meta"]["support_idx"]))
        with open(os.path.join(outdir, "final_summary.json"), "w", encoding="utf-8") as f:
            json.dump(final_json, f, indent=2)

    if save_cfg.save_plots:
        plot_training_overview(history_df, os.path.join(outdir, "overview_4panel.png"))
        plot_support_vs_predictive(history_df, os.path.join(outdir, "support_vs_predictive.png"))
        plot_boundary_density(
            boundary=final["boundary"],
            final_support=final["selected_support"],
            unstable_idx=final["unstable_idx"],
            never_selected_idx=final["never_selected_idx"],
            savepath=os.path.join(outdir, "boundary_density.png"),
        )
        plot_uncertainty_vs_abs_boundary(
            boundary=final["boundary"],
            hard_freq=final["hard_freq"],
            savepath=os.path.join(outdir, "uncertainty_vs_abs_boundary.png"),
        )
        plot_support_overlap_heatmap(history_df, savepath=os.path.join(outdir, "support_overlap_heatmap.png"))


# ==========================================================
# Main experiment wrapper compatible with the user's API
# ==========================================================


def finalize_selected_checkpoint(
    model,
    selected_ckpt_id,
    checkpoints,
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    sigma2,
    beta_true,
    cfg,
    device=None,
):
    if device is None:
        device = next(model.parameters()).device

    load_model_state(model, checkpoints[selected_ckpt_id]["state"], device=device)
    tau = checkpoints[selected_ckpt_id]["meta"]["tau"]
    model.generative_model.set_tau(float(tau))

    draws = sample_posterior_latents(model, R=cfg.diag_R_final)
    hard = hard_support_from_draws(draws, support_threshold=cfg.support_threshold)

    train_metrics = predictive_metrics(X_train, y_train, hard["beta_hard_samples"], sigma2=sigma2)
    val_metrics = predictive_metrics(X_val, y_val, hard["beta_hard_samples"], sigma2=sigma2)
    test_metrics = predictive_metrics(X_test, y_test, hard["beta_hard_samples"], sigma2=sigma2)

    selection_metrics = selection_metrics_from_support(hard["support_idx"], beta_true=beta_true)

    return {
        "selected_support": hard["support_idx"],
        "beta_hard_mean": hard["beta_hard_mean"],
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "selection_metrics": selection_metrics,
    }


def simflow_stagewise(build_flow_vi, simfun1, seed=123, device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32, hidden_units=64, num_hidden_layers=2, 
    n = 180, p = 100, snr = 3.0, true_prop = 0.1, tau_end = 0.40, K_q = 8, K_g = 8,
    schedule_cfg: Optional[StagewiseAnnealConfig] = None, split_cfg: Optional[SplitConfig] = None, save_cfg: Optional[SaveConfig] = None,):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if schedule_cfg is None:
        schedule_cfg = StagewiseAnnealConfig(tau_end=tau_end)
    if split_cfg is None:
        split_cfg = SplitConfig(seed=seed)
    if save_cfg is None:
        save_cfg = SaveConfig(output_dir=None)

    X, y, beta_true, sim_info = simfun1(n=n, p=p, seed=seed, snr=snr, true_prop=true_prop, device=device, dtype=dtype,)

    splits = split_data_tensors(X, y, split_cfg)

    model = build_flow_vi(
        X=splits["X_train"],
        y=splits["y_train"],
        sigma2=sim_info["sigma2"],
        tau=schedule_cfg.tau_start,
        K_q=K_q,
        K_g=K_g,
        hidden_units=hidden_units,
        num_hidden_layers=num_hidden_layers,
    ).to(device)

    train_out = train_flow_stagewise(
        model=model,
        X_val=splits["X_val"],
        y_val=splits["y_val"],
        sigma2=sim_info["sigma2"],
        cfg=schedule_cfg,
        device=device,
    )

    final = finalize_selected_checkpoint(
        model=model,
        selected_ckpt_id=train_out["selected_ckpt_id"],
        checkpoints=train_out["checkpoints"],
        X_train=splits["X_train"],
        y_train=splits["y_train"],
        X_val=splits["X_val"],
        y_val=splits["y_val"],
        X_test=splits["X_test"],
        y_test=splits["y_test"],
        sigma2=sim_info["sigma2"],
        beta_true=beta_true,
        history_df=train_out["history_df"],
        cfg=schedule_cfg,
        device=device,
    )

    out = {
        "seed": seed,
        "sim_info": sim_info,
        "splits": splits,
        "beta_true": beta_true,
        "model": model,
        "train_out": train_out,
        "final": final,
    }

    save_run_artifacts(out, save_cfg)
    return out


# ==========================================================
# Benchmark utilities and textual summaries
# ==========================================================


def benchmark_row_from_run(out: Dict[str, Any], method_name: str = "flow_anneal") -> pd.DataFrame:
    row = {
        "method": method_name,
        "runtime_sec": float(out["runtime_sec"]),
        "selected_ckpt_id": int(out["selected_ckpt_id"]),
        "selected_epoch": int(out["checkpoints"][out["selected_ckpt_id"]]["meta"]["epoch"]),
        "selected_tau": float(out["checkpoints"][out["selected_ckpt_id"]]["meta"]["tau"]),
        "selected_support_size": int(out["final"]["var_table"]["selected"].sum()),
    }
    for prefix, metrics in [
        ("train", out["final"]["train_metrics"]),
        ("val", out["final"]["val_metrics"]),
        ("test", out["final"]["test_metrics"]),
    ]:
        for k, v in metrics.items():
            row[f"{prefix}_{k}"] = v
    for k, v in out["final"]["selection_metrics"].items():
        row[f"sel_{k}"] = v
    return pd.DataFrame([row])


def combine_benchmark_rows(rows: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if len(rows) == 0:
        return pd.DataFrame()
    return pd.concat(rows, axis=0, ignore_index=True)


def show_run_summary(out: Dict[str, Any], top_k: int = 20) -> None:
    meta = out["checkpoints"][out["selected_ckpt_id"]]["meta"]
    final = out["final"]

    print("===== Selected checkpoint =====")
    print({
        "ckpt_id": int(out["selected_ckpt_id"]),
        "epoch": int(meta["epoch"]),
        "stage": int(meta["stage"]),
        "tau": float(meta["tau"]),
        "support_size": int(meta["support_size"]),
        "val_mse": float(meta["val_mse"]),
        "loss_ema": float(meta["loss_ema"]),
        "runtime_sec": float(out["runtime_sec"]),
    })

    print("\n===== Predictive metrics =====")
    print(final["pred_table"].to_string(index=False))

    if final["selection_metrics"]:
        print("\n===== Selection metrics =====")
        print(final["selection_metrics"])

    print("\n===== Top variables by |beta_hard_mean| =====")
    top_df = final["var_table"].copy()
    top_df["abs_beta_hard_mean"] = top_df["beta_hard_mean"].abs()
    view_cols = [c for c in ["j", "beta_hard_mean", "selected", "beta_true", "truth"] if c in top_df.columns]
    print(top_df.sort_values("abs_beta_hard_mean", ascending=False).head(top_k)[view_cols].to_string(index=False))


# ==========================================================
# Suggested default profiles
# ==========================================================


def default_fast_profile(base_lr: float = 5e-5, tau_end: float = 0.45) -> StagewiseAnnealConfig:
    return StagewiseAnnealConfig(
        tau_start=1.0,
        tau_end=tau_end,
        warmup_epochs=150,
        n_anneal_stages=4,
        min_stage_epochs=60,
        max_stage_epochs=140,
        base_lr=base_lr,
        stage_lr_decay=0.75,
        eval_every=25,
        checkpoint_every=25,
        print_every=25,
        diag_R_train=192,
        diag_R_final=2000,
    )



def default_balanced_profile(base_lr: float = 5e-5, tau_end: float = 0.40) -> StagewiseAnnealConfig:
    return StagewiseAnnealConfig(
        tau_start=1.0,
        tau_end=tau_end,
        warmup_epochs=200,
        n_anneal_stages=5,
        min_stage_epochs=80,
        max_stage_epochs=220,
        base_lr=base_lr,
        stage_lr_decay=0.70,
        eval_every=25,
        checkpoint_every=25,
        print_every=25,
        diag_R_train=256,
        diag_R_final=4000,
    )



def default_slow_profile(base_lr: float = 5e-5, tau_end: float = 0.35) -> StagewiseAnnealConfig:
    return StagewiseAnnealConfig(
        tau_start=1.0,
        tau_end=tau_end,
        warmup_epochs=250,
        n_anneal_stages=6,
        min_stage_epochs=100,
        max_stage_epochs=260,
        base_lr=base_lr,
        stage_lr_decay=0.65,
        eval_every=25,
        checkpoint_every=25,
        print_every=25,
        diag_R_train=320,
        diag_R_final=5000,
    )


__all__ = [
    "StagewiseAnnealConfig",
    "SplitConfig",
    "SaveConfig",
    "default_fast_profile",
    "default_balanced_profile",
    "default_slow_profile",
    "sample_posterior_latents",
    "hard_support_from_draws",
    "predictive_metrics",
    "train_flow_stagewise",
    "simflow_stagewise",
    "benchmark_row_from_run",
    "combine_benchmark_rows",
    "show_run_summary",
    "plot_training_overview",
    "plot_support_vs_predictive",
    "plot_boundary_density",
    "plot_uncertainty_vs_abs_boundary",
    "plot_support_overlap_heatmap",
]
