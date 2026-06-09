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
from pathlib import Path
from scipy.stats import gaussian_kde

from .utils import (
    EMA, 
    to_numpy,
    read_mcmc_ref,
    jaccard_distance, 
    softgate_from_draws,
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
    print_result,
    recovery_metrics
)

from .checkpoint_rule import predict_lasso_recovery_score_from_record

from .artifact import save_run_artifacts

from .simfun import print_siminfo

def checkpoint_nonoracle_features(draws, gate_soft, beta_hard=None):
    beta = draws["beta"]
    g = gate_soft
    eps = 1e-8

    ent = -(g * torch.log(g + eps) + (1.0 - g) * torch.log(1.0 - g + eps))

    g_mean_j = g.mean(dim=0)
    g_sd_j = g.std(dim=0)

    beta_abs = beta.abs()
    beta_sd_j = beta.std(dim=0)
    beta_mean_abs_j = beta.mean(dim=0).abs()
    beta_l2 = torch.linalg.vector_norm(beta, dim=1)

    out = {
        "softgate_mean": float(g.mean().detach().cpu()),
        "softgate_sd": float(g.std().detach().cpu()),
        "softgate_q05": float(torch.quantile(g, 0.05).detach().cpu()),
        "softgate_q25": float(torch.quantile(g, 0.25).detach().cpu()),
        "softgate_q50": float(torch.quantile(g, 0.50).detach().cpu()),
        "softgate_q75": float(torch.quantile(g, 0.75).detach().cpu()),
        "softgate_q95": float(torch.quantile(g, 0.95).detach().cpu()),

        "softgate_mean_j_q05": float(torch.quantile(g_mean_j, 0.05).detach().cpu()),
        "softgate_mean_j_q25": float(torch.quantile(g_mean_j, 0.25).detach().cpu()),
        "softgate_mean_j_q50": float(torch.quantile(g_mean_j, 0.50).detach().cpu()),
        "softgate_mean_j_q75": float(torch.quantile(g_mean_j, 0.75).detach().cpu()),
        "softgate_mean_j_q95": float(torch.quantile(g_mean_j, 0.95).detach().cpu()),
        "softgate_mean_j_sum": float(g_mean_j.sum().detach().cpu()),

        "softgate_sd_j_median": float(g_sd_j.median().detach().cpu()),
        "softgate_sd_j_q90": float(torch.quantile(g_sd_j, 0.90).detach().cpu()),

        "gate_entropy_mean": float(ent.mean().detach().cpu()),
        "gate_entropy_median": float(ent.median().detach().cpu()),
        "gate_entropy_q90": float(torch.quantile(ent, 0.90).detach().cpu()),

        "beta_abs_mean": float(beta_abs.mean().detach().cpu()),
        "beta_abs_median": float(beta_abs.median().detach().cpu()),
        "beta_abs_q75": float(torch.quantile(beta_abs, 0.75).detach().cpu()),
        "beta_abs_q90": float(torch.quantile(beta_abs, 0.90).detach().cpu()),
        "beta_abs_q95": float(torch.quantile(beta_abs, 0.95).detach().cpu()),

        "beta_mean_abs_j_median": float(beta_mean_abs_j.median().detach().cpu()),
        "beta_mean_abs_j_q90": float(torch.quantile(beta_mean_abs_j, 0.90).detach().cpu()),
        "beta_mean_abs_j_q95": float(torch.quantile(beta_mean_abs_j, 0.95).detach().cpu()),

        "beta_soft_sd_j_mean": float(beta_sd_j.mean().detach().cpu()),
        "beta_soft_sd_j_median": float(beta_sd_j.median().detach().cpu()),
        "beta_soft_sd_j_q75": float(torch.quantile(beta_sd_j, 0.75).detach().cpu()),
        "beta_soft_sd_j_q90": float(torch.quantile(beta_sd_j, 0.90).detach().cpu()),
        "beta_soft_sd_j_q95": float(torch.quantile(beta_sd_j, 0.95).detach().cpu()),

        "beta_l2_draw_mean": float(beta_l2.mean().detach().cpu()),
        "beta_l2_draw_sd": float(beta_l2.std().detach().cpu()),
    }

    if beta_hard is not None:
        pip_hard = (beta_hard.abs() > 1e-12).float().mean(dim=0)
        out.update({
            "pip_hard_mean": float(pip_hard.mean().detach().cpu()),
            "pip_hard_q50": float(torch.quantile(pip_hard, 0.50).detach().cpu()),
            "pip_hard_q75": float(torch.quantile(pip_hard, 0.75).detach().cpu()),
            "pip_hard_q90": float(torch.quantile(pip_hard, 0.90).detach().cpu()),
            "pip_hard_q95": float(torch.quantile(pip_hard, 0.95).detach().cpu()),
            "expected_support_softgate": float(g_mean_j.sum().detach().cpu()),
            "expected_support_hard": float(pip_hard.sum().detach().cpu()),
        })

    return out


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
    tau_now,
    mode="selection",
    beta_true=None,
    mcmc_ref=None,
):
    draws = sample_posterior_latents(model, R=R)
    hard = hard_support_from_draws(draws, support_threshold=support_threshold)

    pred = predictive_metrics(
        X_val,
        y_val,
        hard["beta_hard_samples"],
        sigma2=sigma2,
        family=family,
    )

    rec = {
        "support_idx": hard["support_idx"],
        "support_size": hard["support_size"],
        "loss": float(loss_value),
        "loss_ema": float(loss_ema),
        "log_grad_ema": float(log_grad_ema),
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

        prev_mse = max(float(prev_record["val_mse"]), 1e-12)
        cur_mse = float(rec["val_mse"])
        rec["pred_improve_rel"] = float(max(prev_mse - cur_mse, 0.0) / prev_mse)

        prev_loss = float(prev_record["loss"])
        cur_loss = float(rec["loss"])
        rec["loss_improve_rel"] = float(max(prev_loss - cur_loss, 0.0) / max(abs(prev_loss), 1e-12))

        rec["grad_slope"] = float(rec["log_grad_ema"] - prev_record["log_grad_ema"])

    if mode in {"recovery", "lasso_recovery"}:
        gate = softgate_from_draws(draws, tau_now)

        beta_np = to_numpy(draws["beta"])
        gate_np = to_numpy(gate)
        gate_mean = gate_np.mean(axis=0)

        rec["softgate_mean_vec"] = gate_mean.tolist()
        rec["expected_support_soft"] = float(gate_mean.sum())

        ent = (
            -gate_np * np.log(gate_np + 1e-8)
            - (1.0 - gate_np) * np.log(1.0 - gate_np + 1e-8)
        )
        rec["softgate_entropy"] = float(ent.mean())
        rec["beta_soft_sd_mean"] = float(beta_np.std(axis=0, ddof=1).mean())
        rec["beta_soft_abs_mean"] = float(np.abs(beta_np).mean())

        if prev_record is None or "softgate_mean_vec" not in prev_record:
            rec["softgate_churn"] = 0.0
        else:
            prev_gate = np.asarray(prev_record["softgate_mean_vec"], dtype=float)
            rec["softgate_churn"] = float(np.mean(np.abs(gate_mean - prev_gate)))

        rec.update(
            checkpoint_nonoracle_features(
                draws=draws,
                gate_soft=gate,
                beta_hard=hard["beta_hard_samples"],
            )
        )

        if mode == "recovery":
            rec.update(
                recovery_metrics(
                    beta_last=beta_np,
                    gate_last=gate_np,
                    beta_true=beta_true,
                    mcmc_ref=mcmc_ref,
                )
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


def stage_can_advance(stage_records, epoch_in_stage, cfg, mode="selection"):
    if mode == "lasso_recovery":
        return False

    if epoch_in_stage < cfg.min_stage_epochs:
        return False

    if len(stage_records) < cfg.plateau_window:
        return False

    tail = stage_records[-cfg.plateau_window:]

    if mode == "selection":
        return all(
            rec["loss_improve_rel"] <= cfg.plateau_loss_rel
            and rec["pred_improve_rel"] <= cfg.plateau_pred_rel
            and abs(rec["dS"]) <= cfg.plateau_abs_dS
            and rec["churn"] <= cfg.plateau_churn
            and rec["grad_slope"] <= cfg.plateau_grad_slope
            for rec in tail
        )

    return all(
        rec["loss_improve_rel"] <= cfg.plateau_loss_rel
        and rec["softgate_churn"] <= 0.01
        and abs(rec["grad_slope"]) <= cfg.plateau_grad_slope
        for rec in tail
    )


def best_record_key(rec, mode="selection"):
    if mode == "lasso_recovery":
        return (rec["lasso_recovery_score"], rec["loss"])

    if mode == "recovery":
        return (rec["moment_recovery_score"], rec["loss"])

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
    mode: str = "selection",
    recovery_min_epoch: int = 100,
    recovery_score_col: str = "moment_recovery_score",
) -> int:
    df = history_df.copy().reset_index(drop=True)

    if mode == "lasso_recovery":
        df = df[df["epoch"] >= recovery_min_epoch].copy()
        df = df.sort_values(
            by=["lasso_recovery_score", "epoch"],
            ascending=[True, True],
        )
        return int(df.iloc[0]["ckpt_id"])

    if mode == "recovery":
        df = df[df["epoch"] >= recovery_min_epoch].copy()
        df = df.sort_values(
            by=[
                recovery_score_col,
                "active_mean_zerr_median",
                "active_sd_logerr_median",
                "active_marg_skl_median",
                "active_joint_skl_median",
                "epoch",
            ],
            ascending=[True, True, True, True, True, True],
        )
        return int(df.iloc[0]["ckpt_id"])

    if exclude_warmup and "is_warmup" in df.columns:
        df2 = df[~df["is_warmup"]].copy()
        if len(df2) > 0:
            df = df2

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

    if len(stable) == 0:
        stable = df.copy()

    best_loss = float(stable["loss"].min())
    cand = stable[stable["loss"] <= best_loss * (1.0 + loss_eps_rel)].copy()

    if len(cand) == 0:
        cand = stable.copy()

    cand = cand.sort_values(
        by=["churn", "dS_abs", "dlogg_abs", "loss", "support_size", "epoch"],
        ascending=[True, True, True, True, True, True],
    )

    return int(cand.iloc[0]["ckpt_id"])


# ==========================================================
# Stagewise annealing training loop
# ==========================================================


def train_flow_stagewise(
    model,
    X_val,
    y_val,
    sigma2,
    family,
    cfg,
    device,
    mode="selection",
    beta_true=None,
    mcmc_ref=None,
    checkpoint_rule=None,
):
    if device is None:
        device = next(model.parameters()).device

    optimizer = make_optimizer(model, cfg)
    loss_ema = EMA(beta=cfg.ema_beta)
    log_grad_ema = EMA(beta=cfg.ema_beta)

    planned_taus = rolling_stage_taus(cfg.tau_start, cfg.tau_end, cfg.n_anneal_stages)
    stage_taus = [planned_taus[0]] + planned_taus[1:]

    t_start = time.time()
    global_epoch = 0
    ckpt_id = 0

    history = []
    checkpoints = {}
    stage_summaries = []

    prev_record = None

    print(f"[info] train mode = {mode}")

    for stage_index, tau_now in enumerate(stage_taus):
        tau_now = float(tau_now)
        is_warmup = stage_index == 0

        lr_scale = max(cfg.min_lr_scale, cfg.stage_lr_decay ** max(stage_index - 1, 0))
        stage_lr = cfg.base_lr * lr_scale

        set_optimizer_lr(optimizer, stage_lr)
        model.generative_model.set_tau(tau_now)

        stage_epoch_cap = cfg.warmup_epochs if is_warmup else cfg.max_stage_epochs

        stage_records = []
        stage_best_record = None
        stage_best_state = None

        for epoch_in_stage in range(1, stage_epoch_cap + 1):
            global_epoch += 1

            model.train()
            optimizer.zero_grad(set_to_none=True)
            model.generative_model.set_tau(tau_now)

            loss = model.neg_elbo(
                num_samples=cfg.num_samples,
                elbo_beta=cfg.elbo_beta,
                q_entropy_weight=cfg.q_entropy_weight,
            )

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

            if not needs_eval:
                continue

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
                tau_now=tau_now,
                mode=mode,
                beta_true=beta_true,
                mcmc_ref=mcmc_ref,
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

            if mode == "lasso_recovery":
                rec["lasso_recovery_score"] = predict_lasso_recovery_score_from_record(rec)

            rec["alert"] = bool(is_alert_record(rec, cfg))
            rec["best_so_far"] = False

            if stage_best_record is None or best_record_key(rec, mode=mode) < best_record_key(stage_best_record, mode=mode):
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

                msg = (
                    f"[stage {stage_index:02d} | epoch {global_epoch:04d}] "
                    f"tau={tau_now:.4f} lr={stage_lr:.2e} "
                    f"loss={loss_val:.6f} "
                    f"logg={log_grad_ema_val:.4f} "
                    f"dlogg={rec['grad_slope']:+.4f} "
                    f"S={rec['support_size']:3d} "
                    f"churn={rec['churn']:.3f}"
                )

                if mode == "recovery":
                    msg += (
                        f" rec={rec['moment_recovery_score']:.4f}"
                        f" meanz={rec['active_mean_zerr_median']:.4f}"
                        f" sdlog={rec['active_sd_logerr_median']:.4f}"
                        f" sdr={rec['active_sd_ratio_median']:.3f}"
                        f" marg={rec['active_marg_skl_median']:.4f}"
                        f" joint={rec['active_joint_skl_median']:.4f}"
                    )

                if mode == "lasso_recovery":
                    msg += f" lasso={rec['lasso_recovery_score']:.4f}"

                print(f"{msg} {flag}{star}")

            if (not is_warmup) and stage_can_advance(stage_records, epoch_in_stage, cfg, mode=mode):
                print(
                    f"[advance] stage {stage_index} finished at epoch {global_epoch} "
                    f"with tau={tau_now:.4f}"
                )
                break

        if stage_best_record is not None:
            stage_summary = {
                "stage": stage_index,
                "tau": tau_now,
                "best_ckpt_id": int(stage_best_record["ckpt_id"]),
                "best_epoch": int(stage_best_record["epoch"]),
                "best_val_mse": float(stage_best_record["val_mse"]),
                "best_support_size": int(stage_best_record["support_size"]),
                "n_evals": len(stage_records),
            }

            if mode == "recovery":
                stage_summary.update({
                    "best_moment_recovery_score": float(stage_best_record["moment_recovery_score"]),
                    "best_active_mean_zerr_median": float(stage_best_record["active_mean_zerr_median"]),
                    "best_active_sd_logerr_median": float(stage_best_record["active_sd_logerr_median"]),
                    "best_active_sd_ratio_median": float(stage_best_record["active_sd_ratio_median"]),
                    "best_active_marg_skl_median": float(stage_best_record["active_marg_skl_median"]),
                    "best_active_joint_skl_median": float(stage_best_record["active_joint_skl_median"]),
                })

            if mode == "lasso_recovery":
                stage_summary["best_lasso_recovery_score"] = float(stage_best_record["lasso_recovery_score"])

            stage_summaries.append(stage_summary)

    runtime_sec = time.time() - t_start
    history_df = pd.DataFrame(history)

    if not history_df.empty:
        num_cols = history_df.select_dtypes(include=[np.number]).columns
        history_df[num_cols] = history_df[num_cols].round(cfg.history_round_digits)

    selected_ckpt_id = select_checkpoint_from_history(
        history_df,
        mode=mode,
        recovery_min_epoch=cfg.recovery_min_epoch,
        recovery_score_col=cfg.recovery_score_col,
    )

    selected_row = history_df.loc[history_df["ckpt_id"] == selected_ckpt_id].iloc[0]

    if mode == "recovery":
        print(
            "[selected recovery checkpoint] "
            f"ckpt={int(selected_ckpt_id)} "
            f"epoch={int(selected_row['epoch'])} "
            f"tau={float(selected_row['tau']):.4f} "
            f"rec={float(selected_row['moment_recovery_score']):.4f}"
        )
    elif mode == "lasso_recovery":
        print(
            "[selected lasso recovery checkpoint] "
            f"ckpt={int(selected_ckpt_id)} "
            f"epoch={int(selected_row['epoch'])} "
            f"tau={float(selected_row['tau']):.4f} "
            f"lasso={float(selected_row['lasso_recovery_score']):.4f}"
        )
    else:
        print(
            "[selected selection checkpoint] "
            f"ckpt={int(selected_ckpt_id)} "
            f"epoch={int(selected_row['epoch'])} "
            f"tau={float(selected_row['tau']):.4f} "
            f"loss={float(selected_row['loss']):.4f} "
            f"S={int(selected_row['support_size'])}"
        )

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
    family="gaussian",
    mode="selection",
) -> Dict[str, Any]:

    if device is None:
        device = next(model.parameters()).device

    load_model_state(model, checkpoints[selected_ckpt_id]["state"], device=device)

    meta = checkpoints[selected_ckpt_id]["meta"]
    tau_selected = float(meta["tau"])
    model.generative_model.set_tau(tau_selected)

    draws = sample_posterior_latents(model, R=cfg.diag_R_final)
    hard = hard_support_from_draws(draws, support_threshold=cfg.support_threshold)

    train_metrics = predictive_metrics(
        X_train, y_train, hard["beta_hard_samples"], sigma2=sigma2, family=family
    )
    val_metrics = predictive_metrics(
        X_val, y_val, hard["beta_hard_samples"], sigma2=sigma2, family=family
    )
    test_metrics = predictive_metrics(
        X_test, y_test, hard["beta_hard_samples"], sigma2=sigma2, family=family
    )

    selection_metrics = selection_metrics_from_support(
        hard["support_idx"],
        beta_true=beta_true,
    )

    beta_soft = draws["beta"]
    beta_hard = hard["beta_hard_samples"]
    gate_soft = softgate_from_draws(draws, tau_selected)

    beta_soft_np = to_numpy(beta_soft)
    beta_hard_np = to_numpy(beta_hard)
    gate_np = to_numpy(gate_soft)

    p = beta_soft_np.shape[1]

    var_table = pd.DataFrame({
        "j": np.arange(p),
        "beta_soft_mean": beta_soft_np.mean(axis=0),
        "beta_soft_sd": beta_soft_np.std(axis=0, ddof=1),
        "softgate_mean": gate_np.mean(axis=0),
        "softgate_sd": gate_np.std(axis=0, ddof=1),
        "pip_hard": (np.abs(beta_hard_np) > 1e-12).mean(axis=0),
        "beta_hard_mean": beta_hard_np.mean(axis=0),
        "beta_hard_sd": beta_hard_np.std(axis=0, ddof=1),
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

    out = {
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

    if mode in {"recovery", "lasso_recovery"}:
        out["tau_selected"] = tau_selected
        out["selected_ckpt_meta"] = meta
        out["beta_soft_samples"] = beta_soft
        out["beta_hard_samples"] = beta_hard
        out["softgate_samples"] = gate_soft

    return out

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


def simflow_stagewise(
    X,
    y,
    beta_true=None,
    sim_info=None,
    build_flow_vi=build_flow_vi,
    seed=123,
    device: Optional[torch.device] = None,
    family="gaussian",
    hidden_units=64,
    num_hidden_layers=2,
    show_start=True,
    show_final=True,
    tau_end=0.40,
    K_q=8,
    K_g=8,
    schedule_cfg: Optional[StagewiseAnnealConfig] = None,
    split_cfg: Optional[SplitConfig] = None,
    save_cfg: Optional[SaveConfig] = None,
    mode="selection",
    mcmc_root=None,
    mcmc_setting="simple",
    mcmc_seed=None,
    checkpoint_rule_path=None,
):
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
        print(print_siminfo(sim_info))

    split_indices = make_split(X.shape[0], split_cfg)
    splits = split_data(X, y, split_indices, mode="tensor")

    mcmc_ref = None
    checkpoint_rule = None

    if mode == "recovery":
        if mcmc_seed is None:
            mcmc_seed = sim_info["seed"]

        mcmc_ref = read_mcmc_ref(
            mcmc_root=mcmc_root,
            setting=mcmc_setting,
            mcmc_seed=mcmc_seed,
        )

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
        mode=mode,
        beta_true=beta_true,
        mcmc_ref=mcmc_ref,
        checkpoint_rule=checkpoint_rule,
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
        mode=mode,
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



