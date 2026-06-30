import json
import time
import math
import copy
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch

from .utils import (
    EMA,
    to_numpy,
    jaccard_distance,
    rolling_stage_taus,
    make_optimizer,
    save_model_state,
    load_model_state,
    make_split,
    split_data,
    set_optimizer_lr,
)
from .config import StagewiseAnnealConfig, SplitConfig, SaveConfig
from .model2 import build_flow_vi
from .metric import (
    sample_posterior_latents,
    hard_support_from_draws,
    predictive_metrics,
    selection_metrics_from_support,
    flow_row_from_result,
    print_result,
    recovery_metrics,
)
from .checkpoint_rule import predict_lasso_recovery_score_from_record
from .artifact import save_run_artifacts
from .simfun import print_siminfo


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


def stagewise_config_dict(cfg):
    return {
        "tau_start": cfg.tau_start,
        "tau_end": cfg.tau_end,
        "n_anneal_stages": cfg.n_anneal_stages,
        "warmup_epochs": cfg.warmup_epochs,
        "max_stage_epochs": cfg.max_stage_epochs,
        "min_stage_epochs": cfg.min_stage_epochs,
        "base_lr": cfg.base_lr,
        "stage_lr_decay": cfg.stage_lr_decay,
        "min_lr_scale": cfg.min_lr_scale,
        "num_samples": cfg.num_samples,
        "elbo_beta": cfg.elbo_beta,
        "grad_clip": cfg.grad_clip,
        "eval_every": cfg.eval_every,
        "print_every": cfg.print_every,
        "diag_R_train": cfg.diag_R_train,
        "diag_R_final": cfg.diag_R_final,
        "support_threshold": cfg.support_threshold,
        "ema_beta": cfg.ema_beta,
        "plateau_window": cfg.plateau_window,
        "plateau_loss_rel": cfg.plateau_loss_rel,
        "plateau_pred_rel": cfg.plateau_pred_rel,
        "plateau_abs_dS": cfg.plateau_abs_dS,
        "plateau_churn": cfg.plateau_churn,
        "plateau_grad_slope": cfg.plateau_grad_slope,
        "recovery_min_epoch": cfg.recovery_min_epoch,
        "history_round_digits": cfg.history_round_digits,
    }


def resolve_mcmc_dir(mcmc_root, sim_info, mcmc_setting=None, mcmc_seed=None):
    root = Path(mcmc_root)
    n = int(sim_info["n"])
    p = int(sim_info["p"])
    setting = str(mcmc_setting or sim_info.get("setting") or sim_info.get("sim"))
    seed = int(mcmc_seed if mcmc_seed is not None else sim_info["seed"])
    candidates = [
        root / f"n{n}p{p}_mcmc_output" / setting / f"seed_{seed}",
        root / setting / f"seed_{seed}",
    ]
    for d in candidates:
        if (d / "mcmc_beta_draws.csv.gz").exists() and (d / "mcmc_pip.csv").exists():
            return d
    return None


def read_mcmc_ref_auto(mcmc_root, sim_info, mcmc_setting=None, mcmc_seed=None):
    info = {
        "mcmc_available": False,
        "mcmc_root": None if mcmc_root is None else str(mcmc_root),
        "mcmc_setting": str(mcmc_setting or sim_info.get("setting") or sim_info.get("sim")),
        "mcmc_seed": int(mcmc_seed if mcmc_seed is not None else sim_info.get("seed", -1)),
        "mcmc_layout": "data/n{n}p{p}_mcmc_output/{setting}/seed_{seed}",
    }
    if mcmc_root is None:
        info["mcmc_reason"] = "mcmc_root is None"
        return None, info
    d = resolve_mcmc_dir(mcmc_root, sim_info, mcmc_setting, mcmc_seed)
    if d is None:
        info["mcmc_reason"] = "mcmc files not found"
        return None, info
    beta = pd.read_csv(d / "mcmc_beta_draws.csv.gz")
    if "draw_id" in beta.columns:
        beta = beta.drop(columns=["draw_id"])
    pip = pd.read_csv(d / "mcmc_pip.csv").sort_values("j0")
    info.update({
        "mcmc_available": True,
        "mcmc_dir": str(d),
        "mcmc_beta_file": str(d / "mcmc_beta_draws.csv.gz"),
        "mcmc_pip_file": str(d / "mcmc_pip.csv"),
    })
    return {"beta": beta.to_numpy(dtype=float), "pip": pip["pip"].to_numpy(dtype=float)}, info


def checkpoint_features(draws, beta_hard=None):
    beta = draws["beta"]
    active = draws.get("active", (beta.abs() > 1e-12).float())
    gate = draws.get("gate", active)

    pip = active.float().mean(dim=0)
    gate_mean = gate.float().mean(dim=0)
    beta_abs = beta.abs()
    beta_sd_j = beta.std(dim=0)
    beta_mean_abs_j = beta.mean(dim=0).abs()
    beta_l2 = torch.linalg.vector_norm(beta, dim=1)

    out = {
        "pip_mean": float(pip.mean().detach().cpu()),
        "pip_q50": float(torch.quantile(pip, 0.50).detach().cpu()),
        "pip_q90": float(torch.quantile(pip, 0.90).detach().cpu()),
        "expected_support": float(pip.sum().detach().cpu()),
        "gate_mean": float(gate_mean.mean().detach().cpu()),
        "expected_gate_mass": float(gate_mean.sum().detach().cpu()),
        "beta_abs_mean": float(beta_abs.mean().detach().cpu()),
        "beta_abs_q90": float(torch.quantile(beta_abs, 0.90).detach().cpu()),
        "beta_mean_abs_j_median": float(beta_mean_abs_j.median().detach().cpu()),
        "beta_mean_abs_j_q90": float(torch.quantile(beta_mean_abs_j, 0.90).detach().cpu()),
        "beta_soft_sd_j_mean": float(beta_sd_j.mean().detach().cpu()),
        "beta_soft_sd_j_median": float(beta_sd_j.median().detach().cpu()),
        "beta_l2_draw_mean": float(beta_l2.mean().detach().cpu()),
        "beta_l2_draw_sd": float(beta_l2.std().detach().cpu()),
        "pip_mean_vec": pip.detach().cpu().numpy().tolist(),
        "gate_mean_vec": gate_mean.detach().cpu().numpy().tolist(),
        # legacy aliases used by the learned checkpoint rule
        "softgate_mean_vec": pip.detach().cpu().numpy().tolist(),
        "expected_support_soft": float(pip.sum().detach().cpu()),
        "softgate_mean": float(pip.mean().detach().cpu()),
        "softgate_sd": float(pip.std().detach().cpu()),
        "softgate_mean_j_sum": float(pip.sum().detach().cpu()),
    }
    if beta_hard is not None:
        hard_pip = (beta_hard.abs() > 1e-12).float().mean(dim=0)
        out.update({
            "pip_hard_mean": float(hard_pip.mean().detach().cpu()),
            "expected_support_hard": float(hard_pip.sum().detach().cpu()),
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
    beta_true=None,
    mcmc_ref=None,
    mcmc_during_training=False,
):
    draws = sample_posterior_latents(model, R=R)
    hard = hard_support_from_draws(draws, support_threshold=support_threshold)
    pred = predictive_metrics(X_val, y_val, hard["beta_hard_samples"], sigma2=sigma2, family=family)

    rec = {
        "support_idx": hard["support_idx"],
        "support_size": hard["support_size"],
        "loss": float(loss_value),
        "loss_ema": float(loss_ema),
        "log_grad_ema": float(log_grad_ema),
        **{f"val_{k}": v for k, v in pred.items()},
    }

    if prev_record is None:
        rec.update({"dS": 0, "churn": 0.0, "pred_improve_rel": 0.0, "loss_improve_rel": 0.0, "grad_slope": 0.0, "pip_churn": 0.0})
    else:
        rec["dS"] = int(rec["support_size"] - prev_record["support_size"])
        rec["churn"] = float(jaccard_distance(rec["support_idx"], prev_record["support_idx"]))
        rec["pred_improve_rel"] = float(max(float(prev_record["val_mse"]) - float(rec["val_mse"]), 0.0) / max(float(prev_record["val_mse"]), 1e-12))
        rec["loss_improve_rel"] = float(max(float(prev_record["loss"]) - float(rec["loss"]), 0.0) / max(abs(float(prev_record["loss"])), 1e-12))
        rec["grad_slope"] = float(rec["log_grad_ema"] - prev_record["log_grad_ema"])

    rec.update(checkpoint_features(draws, beta_hard=hard["beta_hard_samples"]))
    if prev_record is not None and "pip_mean_vec" in prev_record:
        rec["pip_churn"] = float(np.mean(np.abs(np.asarray(rec["pip_mean_vec"]) - np.asarray(prev_record["pip_mean_vec"]))))
        rec["softgate_churn"] = rec["pip_churn"]
    else:
        rec["softgate_churn"] = 0.0

    rec["checkpoint_score"] = float(predict_lasso_recovery_score_from_record(rec))

    if mcmc_during_training and mcmc_ref is not None and beta_true is not None:
        rec.update(recovery_metrics(
            beta_last=to_numpy(draws["beta"]),
            active_last=to_numpy(draws["active"]),
            beta_true=beta_true,
            mcmc_ref=mcmc_ref,
        ))
    return rec


def is_alert_record(rec, cfg):
    dS = rec["dS"]
    prev_S = max(rec["support_size"] - dS, 1)
    return (
        rec["grad_slope"] >= cfg.alert_grad_slope
        and dS >= cfg.alert_support_jump_abs
        and (dS / prev_S if dS > 0 else 0.0) >= cfg.alert_support_jump_rel
        and rec["loss_improve_rel"] <= cfg.alert_loss_rel
        and rec["pred_improve_rel"] <= cfg.alert_pred_rel
        and rec["churn"] >= cfg.alert_churn
    )


def stage_can_advance(stage_records, epoch_in_stage, cfg):
    if epoch_in_stage < cfg.min_stage_epochs or len(stage_records) < cfg.plateau_window:
        return False
    tail = stage_records[-cfg.plateau_window:]
    return all(
        rec["loss_improve_rel"] <= cfg.plateau_loss_rel
        and rec["pred_improve_rel"] <= cfg.plateau_pred_rel
        and abs(rec["dS"]) <= cfg.plateau_abs_dS
        and rec["churn"] <= cfg.plateau_churn
        and rec["pip_churn"] <= cfg.plateau_churn
        and abs(rec["grad_slope"]) <= cfg.plateau_grad_slope
        for rec in tail
    )


def select_checkpoint_from_history(history_df: pd.DataFrame, recovery_min_epoch: int = 100) -> int:
    df = history_df[history_df["epoch"] >= recovery_min_epoch].copy()
    if df.empty:
        df = history_df.copy()
    if "checkpoint_score" in df.columns:
        df = df.sort_values(["checkpoint_score", "loss", "epoch"], ascending=[True, True, True])
    else:
        df = df.sort_values(["loss", "churn", "epoch"], ascending=[True, True, True])
    return int(df.iloc[0]["ckpt_id"])


def train_flow_stagewise(
    model,
    X_val,
    y_val,
    sigma2,
    family,
    cfg,
    device,
    beta_true=None,
    mcmc_ref=None,
    mcmc_during_training=False,
):
    if device is None:
        device = next(model.parameters()).device

    optimizer = make_optimizer(model, cfg)
    loss_ema = EMA(beta=cfg.ema_beta)
    log_grad_ema = EMA(beta=cfg.ema_beta)
    stage_taus = rolling_stage_taus(cfg.tau_start, cfg.tau_end, cfg.n_anneal_stages)

    t_start = time.time()
    global_epoch = 0
    ckpt_id = 0
    history = []
    checkpoints = {}
    stage_summaries = []
    prev_record = None

    for stage_index, tau_now in enumerate(stage_taus):
        tau_now = float(tau_now)
        is_warmup = stage_index == 0
        stage_lr = cfg.base_lr * max(cfg.min_lr_scale, cfg.stage_lr_decay ** max(stage_index - 1, 0))
        set_optimizer_lr(optimizer, stage_lr)
        model.generative_model.set_tau(tau_now)
        stage_epoch_cap = cfg.warmup_epochs if is_warmup else cfg.max_stage_epochs
        stage_records = []
        stage_best_record = None

        for epoch_in_stage in range(1, stage_epoch_cap + 1):
            global_epoch += 1
            model.train()
            optimizer.zero_grad(set_to_none=True)
            model.generative_model.set_tau(tau_now)
            loss = model.neg_elbo(num_samples=cfg.num_samples, elbo_beta=cfg.elbo_beta)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)) if cfg.grad_clip is not None else 0.0
            optimizer.step()

            loss_val = float(loss.detach().cpu().item())
            loss_ema_val = loss_ema.update(loss_val)
            log_grad = math.log(max(grad_norm, 1e-12))
            log_grad_ema_val = log_grad_ema.update(log_grad)

            needs_eval = epoch_in_stage == 1 or epoch_in_stage % cfg.eval_every == 0 or epoch_in_stage == stage_epoch_cap
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
                beta_true=beta_true,
                mcmc_ref=mcmc_ref,
                mcmc_during_training=mcmc_during_training,
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
                "alert": False,
                "best_so_far": False,
            })
            rec["alert"] = bool(is_alert_record(rec, cfg))
            if stage_best_record is None or rec["checkpoint_score"] < stage_best_record["checkpoint_score"]:
                stage_best_record = copy.deepcopy(rec)
                rec["best_so_far"] = True

            checkpoints[ckpt_id] = {"state": save_model_state(model), "meta": copy.deepcopy(rec)}
            ckpt_id += 1
            history.append(copy.deepcopy(rec))
            stage_records.append(copy.deepcopy(rec))
            prev_record = copy.deepcopy(rec)

            if epoch_in_stage % cfg.print_every == 0 or epoch_in_stage == 1 or rec["alert"]:
                star = "*" if rec["best_so_far"] else ""
                flag = "ALERT" if rec["alert"] else "ok"
                print(
                    f"[stage {stage_index:02d} | epoch {global_epoch:04d}] "
                    f"tau={tau_now:.4f} lr={stage_lr:.2e} loss={loss_val:.6f} "
                    f"score={rec['checkpoint_score']:.4f} S={rec['support_size']:3d} "
                    f"pip={rec['expected_support']:.2f} churn={rec['churn']:.3f} {flag}{star}"
                )

            if not is_warmup and stage_can_advance(stage_records, epoch_in_stage, cfg):
                print(f"[advance] stage {stage_index} finished at epoch {global_epoch} with tau={tau_now:.4f}")
                break

        if stage_best_record is not None:
            stage_summaries.append({
                "stage": stage_index,
                "tau": tau_now,
                "best_ckpt_id": int(stage_best_record["ckpt_id"]),
                "best_epoch": int(stage_best_record["epoch"]),
                "best_val_mse": float(stage_best_record["val_mse"]),
                "best_support_size": int(stage_best_record["support_size"]),
                "best_checkpoint_score": float(stage_best_record["checkpoint_score"]),
                "n_evals": len(stage_records),
            })

    history_df = pd.DataFrame(history)
    if not history_df.empty:
        num_cols = history_df.select_dtypes(include=[np.number]).columns
        history_df[num_cols] = history_df[num_cols].round(cfg.history_round_digits)
    selected_ckpt_id = select_checkpoint_from_history(history_df, recovery_min_epoch=cfg.recovery_min_epoch)
    row = history_df.loc[history_df["ckpt_id"] == selected_ckpt_id].iloc[0]
    print(
        "[selected checkpoint] "
        f"ckpt={int(selected_ckpt_id)} epoch={int(row['epoch'])} "
        f"tau={float(row['tau']):.4f} score={float(row['checkpoint_score']):.4f}"
    )
    return {
        "history": history,
        "history_df": history_df,
        "checkpoints": checkpoints,
        "selected_ckpt_id": selected_ckpt_id,
        "stage_summaries": stage_summaries,
        "runtime_sec": time.time() - t_start,
        "config": stagewise_config_dict(cfg),
    }


def finalize_selected_checkpoint(
    model,
    selected_ckpt_id,
    checkpoints,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    sigma2,
    beta_true,
    history_df,
    cfg,
    device=None,
    family="gaussian",
    mcmc_ref=None,
):
    if device is None:
        device = next(model.parameters()).device
    load_model_state(model, checkpoints[selected_ckpt_id]["state"], device=device)
    meta = checkpoints[selected_ckpt_id]["meta"]
    tau_selected = float(meta["tau"])
    model.generative_model.set_tau(tau_selected)

    draws = sample_posterior_latents(model, R=cfg.diag_R_final)
    hard = hard_support_from_draws(draws, support_threshold=cfg.support_threshold)

    train_metrics = predictive_metrics(X_train, y_train, hard["beta_hard_samples"], sigma2=sigma2, family=family)
    val_metrics = predictive_metrics(X_val, y_val, hard["beta_hard_samples"], sigma2=sigma2, family=family)
    test_metrics = predictive_metrics(X_test, y_test, hard["beta_hard_samples"], sigma2=sigma2, family=family)
    selection_metrics = selection_metrics_from_support(hard["support_idx"], beta_true=beta_true)

    beta = draws["beta"]
    gate = draws.get("gate", draws.get("active"))
    active = draws.get("active", (beta.abs() > 1e-12).float())
    beta_np = to_numpy(beta)
    gate_np = to_numpy(gate)
    active_np = to_numpy(active)
    p = beta_np.shape[1]

    var_table = pd.DataFrame({
        "j": np.arange(p),
        "beta_mean": beta_np.mean(axis=0),
        "beta_sd": beta_np.std(axis=0, ddof=1),
        "gate_mean": gate_np.mean(axis=0),
        "gate_sd": gate_np.std(axis=0, ddof=1),
        "pip": active_np.mean(axis=0),
        "pip_sd": active_np.std(axis=0, ddof=1),
        "selected": 0,
    })
    var_table.loc[list(hard["support_idx"]), "selected"] = 1
    if beta_true is not None:
        beta_true_np = to_numpy(beta_true)
        var_table["beta_true"] = beta_true_np
        var_table["truth"] = (np.abs(beta_true_np) > 1e-12).astype(int)
    if mcmc_ref is not None:
        ref_beta = np.asarray(mcmc_ref["beta"], dtype=float)
        ref_pip = np.asarray(mcmc_ref["pip"], dtype=float)
        var_table["mcmc_beta_mean"] = ref_beta.mean(axis=0)
        var_table["mcmc_beta_sd"] = ref_beta.std(axis=0, ddof=1)
        var_table["mcmc_pip"] = ref_pip
        var_table["beta_mean_abs_error"] = np.abs(var_table["beta_mean"] - var_table["mcmc_beta_mean"])
        var_table["beta_sd_abs_error"] = np.abs(var_table["beta_sd"] - var_table["mcmc_beta_sd"])
        var_table["pip_abs_error"] = np.abs(var_table["pip"] - var_table["mcmc_pip"])

    pred_table = pd.DataFrame([
        {"split": "train", **train_metrics},
        {"split": "val", **val_metrics},
        {"split": "test", **test_metrics},
    ])

    recovery = {}
    if mcmc_ref is not None and beta_true is not None:
        recovery = recovery_metrics(beta_last=beta_np, active_last=active_np, beta_true=beta_true, mcmc_ref=mcmc_ref)

    freq = np.zeros(p, dtype=float)
    if history_df is not None and not history_df.empty and "support_idx" in history_df.columns:
        for support_idx in history_df["support_idx"]:
            arr = np.zeros(p, dtype=float)
            arr[list(support_idx)] = 1.0
            freq += arr
        freq /= len(history_df)

    return {
        "selected_support": hard["support_idx"],
        "support_size": int(len(hard["support_idx"])),
        "beta_hard_mean": hard["beta_hard_mean"],
        "hard_freq": freq,
        "unstable_idx": np.where((freq > 0.0) & (freq < 1.0))[0].astype(int).tolist(),
        "never_selected_idx": np.where(freq == 0.0)[0].astype(int).tolist(),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "selection_metrics": selection_metrics,
        "recovery_metrics": recovery,
        "var_table": var_table,
        "pred_table": pred_table,
        "tau_selected": tau_selected,
        "selected_ckpt_meta": meta,
        "beta_samples": beta,
        "active_samples": active,
        "gate_samples": gate,
    }


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
    beta_mode="sigmoid",
    group_ids=None,
    group_sizes=None,
    coupling_type="semantic",
    conditioner_type="mlp",
    scale_clip=2.0,
    affine_layers_per_step=3,
    experiment_group="main",
    experiment_name=None,
    environment_name="baseline",
    method_name=None,
    show_start=True,
    show_final=True,
    tau_end=0.40,
    K_q=8,
    K_g=8,
    schedule_cfg: Optional[StagewiseAnnealConfig] = None,
    split_cfg: Optional[SplitConfig] = None,
    save_cfg: Optional[SaveConfig] = None,
    mcmc_root=None,
    mcmc_setting=None,
    mcmc_seed=None,
    compare_mcmc=True,
    mcmc_during_training=False,
):
    set_all_seeds(seed, deterministic=False)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sim_info = sim_info or {}
    schedule_cfg = schedule_cfg or StagewiseAnnealConfig(tau_end=tau_end)
    split_cfg = split_cfg or SplitConfig(seed=seed)
    save_cfg = save_cfg or SaveConfig(output_dir=None)

    if show_start:
        print("===== Simulation info =====")
        print(print_siminfo(sim_info))

    split_indices = make_split(X.shape[0], split_cfg)
    splits = split_data(X, y, split_indices, mode="tensor")

    mcmc_ref, mcmc_info = (None, {"mcmc_available": False, "mcmc_reason": "not requested"})
    if compare_mcmc:
        mcmc_ref, mcmc_info = read_mcmc_ref_auto(mcmc_root, sim_info, mcmc_setting, mcmc_seed)

    K_total = int(K_q) + int(K_g)
    method_name = method_name or f"{coupling_type}_{conditioner_type}"
    experiment_name = experiment_name or method_name
    model_config = {
        "beta_mode": beta_mode,
        "group_ids": None if group_ids is None else [int(x) for x in group_ids],
        "group_sizes": None if group_sizes is None else [int(x) for x in group_sizes],
        "coupling_type": coupling_type,
        "conditioner_type": conditioner_type,
        "hidden_units": int(hidden_units),
        "num_hidden_layers": int(num_hidden_layers),
        "scale_clip": float(scale_clip),
        "affine_layers_per_step": int(affine_layers_per_step),
        "K_q": int(K_q),
        "K_g": int(K_g),
        "reported_layers": K_total,
        "reported_structure": "single_flow",
        "implementation_structure": "double_flow",
        "structure_note": "Training keeps posterior_flow and g_theta; reporting uses one flow depth K_q + K_g.",
    }
    run_manifest = {
        "experiment_group": experiment_group,
        "experiment_name": experiment_name,
        "environment_name": environment_name,
        "method_name": method_name,
        "seed": int(seed),
        "device": str(device),
        "family": family,
        "model_config": model_config,
        "schedule_config": stagewise_config_dict(schedule_cfg),
        "split_config": asdict(split_cfg),
        "save_config": asdict(save_cfg),
        "mcmc_info": mcmc_info,
    }

    model = build_flow_vi(
        X=splits["X_train"],
        y=splits["y_train"],
        sigma2=sim_info["sigma2"],
        tau=schedule_cfg.tau_start,
        family=family,
        beta_mode=beta_mode,
        group_ids=group_ids,
        group_sizes=group_sizes,
        K_q=K_q,
        K_g=K_g,
        coupling_type=coupling_type,
        conditioner_type=conditioner_type,
        hidden_units=hidden_units,
        num_hidden_layers=num_hidden_layers,
        scale_clip=scale_clip,
        affine_layers_per_step=affine_layers_per_step,
    ).to(device)

    train_out = train_flow_stagewise(
        model=model,
        X_val=splits["X_val"],
        y_val=splits["y_val"],
        family=family,
        sigma2=sim_info["sigma2"],
        cfg=schedule_cfg,
        device=device,
        beta_true=beta_true,
        mcmc_ref=mcmc_ref,
        mcmc_during_training=mcmc_during_training,
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
        family=family,
        mcmc_ref=mcmc_ref,
    )

    out = {
        "method": method_name,
        "seed": seed,
        "sim_info": sim_info,
        "run_manifest": run_manifest,
        "model_config": model_config,
        "mcmc_info": mcmc_info,
        "splits": splits,
        "beta_true": beta_true,
        "model": model,
        **train_out,
        "final": final,
    }
    out["summary_row"] = flow_row_from_result(out)
    out["summary_row"].update({
        "method_name": method_name,
        "experiment_group": experiment_group,
        "experiment_name": experiment_name,
        "environment_name": environment_name,
    })
    save_run_artifacts(out, save_cfg)
    if show_final:
        print_result(out, top_k=20)
    return out
