import os
import json
import time
import math
import copy
import random
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import normflows as nf

from .utils import (
    EMA, 
    jaccard_distance, 
    rolling_stage_taus, 
    make_optimizer, 
    save_model_state, 
    load_model_state, 
    make_split, 
    split_data, 
    set_optimizer_lr
)

from .config import StagewiseAnnealConfig, SplitConfig, SaveConfig

from .model import build_flow_vi

from .metric import (
    sample_posterior_latents,
    hard_support_from_draws,
    predictive_metrics,
    selection_metrics_from_support,
    flow_row_from_result,
    print_result
)

from .artifact import save_run_artifacts


def evaluate_checkpoint(
    model,
    X_val,
    y_val,
    sigma2,
    family,
    R,
    support_threshold,
    prev_record,
    loss_value,
    loss_ema,
    log_grad_ema,
):
    draws = sample_posterior_latents(model, R=R)
    hard = hard_support_from_draws(draws, support_threshold=support_threshold)
    pred = predictive_metrics(X_val, y_val, hard["beta_hard_samples"], sigma2=sigma2, family=family)

    rec = {
        "support_idx": hard["support_idx"],
        "support_size": hard["support_size"],
        "loss": float(loss_value) if loss_value is not None else float("nan"),
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

        prev_loss = float(prev_record.get("loss", float("nan")))
        cur_loss = float(rec.get("loss", float("nan")))
        denom = max(abs(prev_loss), 1e-12)
        rec["loss_improve_rel"] = float(max(prev_loss - cur_loss, 0.0) / denom)

        rec["grad_slope"] = float(
            rec.get("log_grad_ema", float("nan")) - prev_record.get("log_grad_ema", float("nan"))
        )

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


def select_checkpoint_from_history(
    history_df: pd.DataFrame,
    loss_eps_rel: float = 0.01,
    grad_quantile: float = 0.80,
    ds_quantile: float = 0.80,
    churn_quantile: float = 0.80,
    exclude_warmup: bool = True,
    relaxed_loss_mult: float = 2.0,
    relaxed_stability_mult: float = 1.5,
) -> int:
    if history_df.empty:
        raise ValueError("history_df is empty")

    df = history_df.copy().reset_index(drop=True)

    if exclude_warmup and "is_warmup" in df.columns:
        df_nonwarm = df[~df["is_warmup"]].copy()
        if not df_nonwarm.empty:
            df = df_nonwarm

    df["dS_abs"] = df["dS"].abs()
    df["dlogg_abs"] = df["grad_slope"].abs()

    grad_thr = float(df["dlogg_abs"].quantile(grad_quantile))
    ds_thr = float(df["dS_abs"].quantile(ds_quantile))
    churn_thr = float(df["churn"].quantile(churn_quantile))

    stable = df[
        (df["churn"] <= churn_thr)
        & (df["dS_abs"] <= ds_thr)
        & (df["dlogg_abs"] <= grad_thr)
    ].copy()

    if stable.empty:
        stable = df[
            (df["churn"] <= relaxed_stability_mult * churn_thr)
            & (df["dS_abs"] <= relaxed_stability_mult * ds_thr)
            & (df["dlogg_abs"] <= relaxed_stability_mult * grad_thr)
        ].copy()

    if stable.empty:
        stable = df.copy()

    best_loss = float(stable["loss"].min())

    cand = stable[
        stable["loss"] <= best_loss * (1.0 + loss_eps_rel)
    ].copy()

    if cand.empty:
        cand = stable[
            stable["loss"] <= best_loss * (1.0 + relaxed_loss_mult * loss_eps_rel)
        ].copy()

    if cand.empty:
        raise ValueError("No checkpoint found even in the relaxed stable/loss region.")

    cand = cand.sort_values(
        by=["churn", "dS_abs", "dlogg_abs", "loss", "support_size", "epoch"],
        ascending=[True, True, True, True, True, True],
    )

    return int(cand.iloc[0]["ckpt_id"])


# ==========================================================
# Stagewise annealing training loop
# ==========================================================


def train_flow_stagewise(model, X_val, y_val, sigma2, family, cfg, device,):
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
                        family=family,
                        R=cfg.diag_R_train,
                        support_threshold=cfg.support_threshold,
                        prev_record=prev_record,
                        loss_value=loss_val,
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
                        f"loss={loss_val:.6f} "
                        f"logg={log_grad_ema_val:.4f} dlogg={rec['grad_slope']:+.4f} "
                        f"S={rec['support_size']:3d} churn={rec['churn']:.3f} {flag}{star}"
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


def finalize_selected_checkpoint(
    model: torch.nn.Module,
    selected_ckpt_id: int,
    checkpoints: Dict[int, Dict[str, Any]],
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    sigma2: Optional[float],
    beta_true: Optional[torch.Tensor],
    history_df: pd.DataFrame,
    cfg: StagewiseAnnealConfig,
    device: Optional[torch.device] = None,
    family = "gaussian",
) -> Dict[str, Any]:
    if device is None:
        device = next(model.parameters()).device

    load_model_state(model, checkpoints[selected_ckpt_id]["state"], device=device)
    meta = checkpoints[selected_ckpt_id]["meta"]
    model.generative_model.set_tau(float(meta["tau"]))

    draws = sample_posterior_latents(model, R=cfg.diag_R_final)
    hard = hard_support_from_draws(draws, support_threshold=cfg.support_threshold)

    train_metrics = predictive_metrics(X_train, y_train, hard["beta_hard_samples"], sigma2=sigma2, family=family)
    val_metrics = predictive_metrics(X_val, y_val, hard["beta_hard_samples"], sigma2=sigma2, family=family)
    test_metrics = predictive_metrics(X_test, y_test, hard["beta_hard_samples"], sigma2=sigma2, family=family)

    selection_metrics = selection_metrics_from_support(
        hard["support_idx"],
        beta_true=beta_true,
    )

    beta_est = hard["beta_hard_mean"].detach().cpu().numpy()
    p = beta_est.shape[0]
    var_table = pd.DataFrame({
        "j": np.arange(p),
        "beta_hard_mean": beta_est,
        "selected": 0,
    })
    var_table.loc[list(hard["support_idx"]), "selected"] = 1

    if beta_true is not None:
        beta_true_np = beta_true.detach().cpu().numpy()
        var_table["beta_true"] = beta_true_np
        var_table["truth"] = (np.abs(beta_true_np) > 1e-12).astype(int)

    pred_table = pd.DataFrame([
        {"split": "train", **train_metrics},
        {"split": "val", **val_metrics},
        {"split": "test", **test_metrics},
    ])

    freq = np.zeros(p, dtype=float)
    if history_df is not None and not history_df.empty and "support_idx" in history_df.columns:
        for support_idx in history_df["support_idx"]:
            arr = np.zeros(p, dtype=float)
            arr[list(support_idx)] = 1.0
            freq += arr
        freq /= len(history_df)

    unstable_idx = np.where((freq > 0.0) & (freq < 1.0))[0].astype(int).tolist()
    never_selected_idx = np.where(freq == 0.0)[0].astype(int).tolist()

    return {
        "selected_support": hard["support_idx"],
        "support_size": int(len(hard["support_idx"])),
        "beta_hard_mean": hard["beta_hard_mean"],
        "boundary": hard["boundary"],
        "hard_freq": freq,
        "unstable_idx": unstable_idx,
        "never_selected_idx": never_selected_idx,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "selection_metrics": selection_metrics,
        "var_table": var_table,
        "pred_table": pred_table,
    }

def set_all_seeds(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def simflow_stagewise(X, y, beta_true=None, sim_info=None, build_flow_vi=build_flow_vi, seed=123, device: Optional[torch.device] = None, 
    family="gaussian", hidden_units=64, num_hidden_layers=2, show_start=True, show_final=True, tau_end = 0.40, K_q = 8, K_g = 8,
    schedule_cfg: Optional[StagewiseAnnealConfig] = None, split_cfg: Optional[SplitConfig] = None, save_cfg: Optional[SaveConfig] = None,):
    
    set_all_seeds(seed, deterministic=False)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if schedule_cfg is None:
        schedule_cfg = StagewiseAnnealConfig(tau_end=tau_end)
    if split_cfg is None:
        split_cfg = SplitConfig(seed=seed)
    if save_cfg is None:
        save_cfg = SaveConfig(output_dir=None)

    if show_start:
        print("===== Simulation info =====")
        print(sim_info)

    split_indices = make_split(X.shape[0], split_cfg)
    splits = split_data(X, y, split_indices, mode="tensor")

    model = build_flow_vi(
        X=splits["X_train"],
        y=splits["y_train"],
        sigma2=sim_info["sigma2"],
        tau=schedule_cfg.tau_start,
        family=family,
        K_q=K_q,
        K_g=K_g,
        hidden_units=hidden_units,
        num_hidden_layers=num_hidden_layers,
    ).to(device)

    train_out = train_flow_stagewise(
        model=model,
        X_val=splits["X_val"],
        y_val=splits["y_val"],
        family=family,
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
        "method": "flow_stagewise",
        "seed": seed,
        "sim_info": sim_info,
        "splits": splits,
        "beta_true": beta_true,
        "model": model,
        **train_out,
        "final": final,
    }

    out["summary_row"] = flow_row_from_result(out)
    save_run_artifacts(out, save_cfg)

    if show_final:
        print_result(out, top_k=20)

    return out



