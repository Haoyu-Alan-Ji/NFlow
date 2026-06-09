from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import Python.framework as fw
import Python.config as cfg

from Python.utils import to_numpy, softgate_from_draws
from Python.metric import (
    sample_posterior_latents,
    hard_support_from_draws,
    ranking_metrics,
)


def rpath(x):
    x = Path(x)
    return x if x.is_absolute() else ROOT / x


def write_draws_csv(draws_np, out_dir, filename):
    p = draws_np.shape[1]
    df = pd.DataFrame(draws_np, columns=[f"b{j}" for j in range(p)])
    df.insert(0, "draw_id", np.arange(draws_np.shape[0]))
    df.to_csv(out_dir / filename, index=False, compression="gzip")


def read_job(manifest_path, row_id):
    manifest = pd.read_csv(rpath(manifest_path))
    return manifest.iloc[row_id - 1].copy()


def read_data(job, device):
    dat = pd.read_csv(rpath(job["data_path"]))
    beta_tbl = pd.read_csv(rpath(job["beta_path"]))

    X_np = dat.drop(columns=["y"]).to_numpy(dtype=np.float32)
    y_np = dat["y"].to_numpy(dtype=np.float32)
    beta_np = beta_tbl["beta_true"].to_numpy(dtype=np.float32)

    n, p = X_np.shape
    sigma2 = float(job["sigma2"])

    active_idx = np.where(np.abs(beta_np) > 1e-12)[0]
    active_order = active_idx[np.argsort(-np.abs(beta_np[active_idx]))]

    sim_info = {
        "sim": str(job["setting"]),
        "setting": str(job["setting"]),
        "seed": int(job["seed"]),
        "n": int(n),
        "p": int(p),
        "n_active": int(len(active_idx)),
        "active_idx": active_idx,
        "active_idx_by_abs": active_order,
        "active_table": [
            {
                "rank": int(k + 1),
                "j": int(j),
                "j1": int(j + 1),
                "beta_true": float(beta_np[j]),
                "abs_beta_true": float(abs(beta_np[j])),
            }
            for k, j in enumerate(active_order)
        ],
        "sigma2": sigma2,
        "sigma": float(np.sqrt(sigma2)),
        "center_y": True,
    }

    X = torch.as_tensor(X_np, dtype=torch.float32, device=device)
    y = torch.as_tensor(y_np, dtype=torch.float32, device=device)
    beta_true = torch.as_tensor(beta_np, dtype=torch.float32, device=device)

    return X, y, beta_true, sim_info


def make_schedule(args):
    return cfg.StagewiseAnnealConfig(
        tau_start=args.tau_start,
        tau_end=args.tau_end,
        warmup_epochs=args.warmup_epochs,
        n_anneal_stages=args.n_anneal_stages,
        min_stage_epochs=args.min_stage_epochs,
        max_stage_epochs=args.max_stage_epochs,
        base_lr=args.base_lr,
        stage_lr_decay=args.stage_lr_decay,
        min_lr_scale=args.min_lr_scale,
        eval_every=args.eval_every,
        print_every=args.print_every,
        diag_R_train=args.diag_R_train,
        diag_R_final=args.diag_R_final,
        support_threshold=args.support_threshold,
        q_entropy_weight=args.q_entropy_weight,
        recovery_min_epoch=args.recovery_min_epoch,
        recovery_score_col=args.recovery_score_col,
    )


def make_split(split_seed):
    return cfg.SplitConfig(
        train_frac=0.6,
        val_frac=0.2,
        test_frac=0.2,
        seed=split_seed,
    )


def out_dir_from_job(job, args):
    return (
        rpath(args.output_root)
        / args.mode
        / str(job["setting"])
        / f"seed_{int(job['seed'])}"
    )


def selection_metrics(selected, truth, score):
    selected = np.asarray(selected, dtype=int)
    truth = np.asarray(truth, dtype=int)

    tp = int(((selected == 1) & (truth == 1)).sum())
    fp = int(((selected == 1) & (truth == 0)).sum())
    fn = int(((selected == 0) & (truth == 1)).sum())
    tn = int(((selected == 0) & (truth == 0)).sum())

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    fdr = fp / (tp + fp) if tp + fp > 0 else 0.0

    rank = ranking_metrics(
        support_score=score,
        beta_true=truth,
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fdr": fdr,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "support_size": int(selected.sum()),
        "auroc": rank["auroc"],
        "auprc": rank["auprc"],
    }


def posterior_draws_for_output(flow_out, schedule):
    model = flow_out["model"]
    ckpt_id = int(flow_out["selected_ckpt_id"])

    ckpt = flow_out["checkpoints"][ckpt_id]
    meta = ckpt["meta"]
    state = ckpt["state"]

    tau_selected = float(meta["tau"])
    device = next(model.parameters()).device

    model.load_state_dict({k: v.to(device) for k, v in state.items()})
    model.generative_model.set_tau(tau_selected)
    model.eval()

    with torch.no_grad():
        draws = sample_posterior_latents(model, R=schedule.diag_R_final)
        hard = hard_support_from_draws(
            draws,
            support_threshold=schedule.support_threshold,
        )
        gate_soft = softgate_from_draws(draws, tau_selected)

    beta_soft = to_numpy(draws["beta"])
    beta_hard = to_numpy(hard["beta_hard_samples"])
    gate_soft = to_numpy(gate_soft)

    pip_hard = (np.abs(beta_hard) > 1e-12).mean(axis=0)
    softgate = gate_soft.mean(axis=0)

    return beta_soft, beta_hard, gate_soft, pip_hard, softgate, tau_selected


def write_tables(flow_out, job, schedule, out_dir, mode, K_q=None, K_g=None):
    vt = flow_out["final"]["var_table"].copy()
    vt = vt.rename(columns={"j": "j0"})
    vt["j1"] = vt["j0"] + 1

    beta_soft, beta_hard, gate_soft, pip_hard, softgate, tau_selected = posterior_draws_for_output(
        flow_out,
        schedule,
    )

    vt["pip"] = pip_hard
    vt["pip_hard"] = pip_hard
    vt["softgate"] = softgate
    vt["selected"] = (vt["pip_hard"] > schedule.support_threshold).astype(int)

    vt["effect_mean"] = beta_soft.mean(axis=0)
    vt["effect_sd"] = beta_soft.std(axis=0, ddof=1)
    vt["effect_hard_mean"] = beta_hard.mean(axis=0)
    vt["effect_hard_sd"] = beta_hard.std(axis=0, ddof=1)

    pip_tbl = vt[
        [
            "j0",
            "j1",
            "beta_true",
            "truth",
            "pip",
            "pip_hard",
            "softgate",
            "selected",
            "effect_mean",
            "effect_sd",
            "effect_hard_mean",
            "effect_hard_sd",
        ]
    ].copy()

    pip_tbl.insert(0, "replicate_id", int(job["seed"]))
    pip_tbl.insert(0, "seed", int(job["seed"]))
    pip_tbl.insert(0, "setting", str(job["setting"]))
    pip_tbl.insert(0, "mode", mode)
    pip_tbl.insert(0, "method", "last_flow")

    truth = pip_tbl["truth"].to_numpy(dtype=int)
    selected = pip_tbl["selected"].to_numpy(dtype=int)
    score = pip_tbl["pip_hard"].to_numpy(dtype=float)

    active_idx = np.where(truth == 1)[0]
    selected_idx = np.where(selected == 1)[0]

    metrics = selection_metrics(selected, truth, score)

    ckpt_id = int(flow_out["selected_ckpt_id"])
    meta = flow_out["checkpoints"][ckpt_id]["meta"]

    summary = {
        "method": "last_flow",
        "mode": mode,
        "setting": str(job["setting"]),
        "seed": int(job["seed"]),
        "replicate_id": int(job["seed"]),
        "n": int(flow_out["sim_info"]["n"]),
        "p": int(flow_out["sim_info"]["p"]),
        "n_active": int(truth.sum()),
        "sigma2": float(flow_out["sim_info"]["sigma2"]),
        "support_threshold": float(schedule.support_threshold),
        "n_draws": int(beta_soft.shape[0]),
        "selected_size": int(selected.sum()),
        "active_idx0": ";".join(str(j) for j in active_idx),
        "selected_idx0": ";".join(str(j) for j in selected_idx),
        "selected_ckpt_id": ckpt_id,
        "selected_epoch": int(meta["epoch"]),
        "selected_stage": int(meta["stage"]),
        "selected_tau": float(tau_selected),
        **metrics,
        "K_q": int(K_q) if K_q is not None else -1,
        "K_g": int(K_g) if K_g is not None else -1,
        "tau_start": float(schedule.tau_start),
        "tau_end": float(schedule.tau_end),
        "base_lr": float(schedule.base_lr),
        "stage_lr_decay": float(schedule.stage_lr_decay),
        "min_lr_scale": float(schedule.min_lr_scale),
        "warmup_epochs": int(schedule.warmup_epochs),
        "n_anneal_stages": int(schedule.n_anneal_stages),
        "min_stage_epochs": int(schedule.min_stage_epochs),
        "max_stage_epochs": int(schedule.max_stage_epochs),
        "q_entropy_weight": float(schedule.q_entropy_weight),
    }

    if mode == "recovery":
        summary.update({
            "moment_recovery_score": float(meta["moment_recovery_score"]),
            "recovery_score": float(meta["recovery_score"]),

            "active_mean_zerr_median": float(meta["active_mean_zerr_median"]),
            "active_mean_zerr_mean": float(meta["active_mean_zerr_mean"]),

            "active_sd_logerr_median": float(meta["active_sd_logerr_median"]),
            "active_sd_logerr_mean": float(meta["active_sd_logerr_mean"]),

            "active_sd_ratio_median": float(meta["active_sd_ratio_median"]),
            "active_sd_ratio_mean": float(meta["active_sd_ratio_mean"]),

            "sd_ratio_median": float(meta["sd_ratio_median"]),
            "sd_ratio_mean": float(meta["sd_ratio_mean"]),

            "active_marg_skl_median": float(meta["active_marg_skl_median"]),
            "active_marg_skl_mean": float(meta["active_marg_skl_mean"]),

            "active_joint_skl_median": float(meta["active_joint_skl_median"]),
            "active_joint_skl_mean": float(meta["active_joint_skl_mean"]),

            "softgate_absdiff_median": float(meta["softgate_absdiff_median"]),
            "softgate_absdiff_mean": float(meta["softgate_absdiff_mean"]),

            "zero_soft_leakage_median": float(meta["zero_soft_leakage_median"]),
            "zero_soft_leakage_mean": float(meta["zero_soft_leakage_mean"]),
        })
    if mode == "lasso_recovery":
        summary.update({
            "lasso_recovery_score": float(meta["lasso_recovery_score"]),
        })
    summary_tbl = pd.DataFrame([summary])

    out_dir.mkdir(parents=True, exist_ok=True)

    pip_tbl.to_csv(out_dir / "last_pip.csv", index=False)
    summary_tbl.to_csv(out_dir / "last_summary.csv", index=False)

    hist = flow_out["history_df"].copy()
    if "softgate_mean_vec" in hist.columns:
        hist = hist.drop(columns=["softgate_mean_vec"])
    hist.to_csv(out_dir / "last_history.csv", index=False)

    pd.DataFrame(flow_out["stage_summaries"]).to_csv(
        out_dir / "last_stage_summary.csv",
        index=False,
    )

    write_draws_csv(beta_soft, out_dir, "last_beta_soft_draws.csv.gz")
    write_draws_csv(beta_hard, out_dir, "last_beta_hard_draws.csv.gz")
    write_draws_csv(gate_soft, out_dir, "last_softgate_draws.csv.gz")

    print("===== LaST-Flow result =====")
    print(f"mode          : {mode}")
    print(f"seed          : {int(job['seed'])}")
    print(f"n_draws       : {int(beta_soft.shape[0])}")
    print(f"selected_size : {int(selected.sum())}")
    print(f"selected_ckpt : {ckpt_id}")
    print(f"selected_tau  : {tau_selected:.4f}")
    print(f"support_thr   : {schedule.support_threshold}")
    print()

    print("===== Selection metrics =====")
    print(
        summary_tbl[
            [
                "precision",
                "recall",
                "f1",
                "fdr",
                "tp",
                "fp",
                "fn",
                "tn",
                "support_size",
                "auroc",
                "auprc",
            ]
        ].to_string(index=False)
    )
    print()

    if mode == "recovery":
        print("===== Recovery metrics =====")
        print(
            summary_tbl[
                [
                    "moment_recovery_score",
                    "active_mean_zerr_median",
                    "active_sd_logerr_median",
                    "active_sd_ratio_median",
                    "active_marg_skl_median",
                    "active_joint_skl_median",
                    "softgate_absdiff_median",
                    "zero_soft_leakage_median",
                ]
            ].to_string(index=False)
        )
        print()
    if mode == "lasso_recovery":
        print("===== Lasso recovery checkpoint =====")
        print(
            summary_tbl[
                [
                    "lasso_recovery_score",
                    "selected_epoch",
                    "selected_tau",
                    "selected_size",
                ]
            ].to_string(index=False)
        )
        print()

    print("===== Selected support =====")
    print(selected_idx.tolist())


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("manifest")
    p.add_argument("row_id", type=int)

    p.add_argument(
        "--mode",
        default="selection",
        choices=["selection", "recovery", "lasso_recovery"],
    )
    p.add_argument("--output-root", default="data/n160p100_last_output")
    p.add_argument("--mcmc-root", default="data/n160p100_mcmc_output")
    p.add_argument("--checkpoint-rule-path", default=None)
    p.add_argument("--recovery-min-epoch", type=int, default=100)
    p.add_argument("--recovery-score-col", default="moment_recovery_score")
    p.add_argument("--train-seed", type=int, default=None)

    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    p.add_argument("--hidden-units", type=int, default=64)
    p.add_argument("--num-hidden-layers", type=int, default=2)
    p.add_argument("--K-q", type=int, default=32)
    p.add_argument("--K-g", type=int, default=8)
    p.add_argument("--q-entropy-weight", type=float, default=0.0)

    p.add_argument("--tau-start", type=float, default=0.5)
    p.add_argument("--tau-end", type=float, default=0.1)
    p.add_argument("--warmup-epochs", type=int, default=300)
    p.add_argument("--n-anneal-stages", type=int, default=20)
    p.add_argument("--min-stage-epochs", type=int, default=100)
    p.add_argument("--max-stage-epochs", type=int, default=400)

    p.add_argument("--base-lr", type=float, default=2e-5)
    p.add_argument("--stage-lr-decay", type=float, default=0.9)
    p.add_argument("--min-lr-scale", type=float, default=0.5)

    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--print-every", type=int, default=100)
    p.add_argument("--diag-R-train", type=int, default=256)
    p.add_argument("--diag-R-final", type=int, default=4000)
    p.add_argument("--support-threshold", type=float, default=0.5)

    p.add_argument("--split-seed", type=int, default=12345)

    return p.parse_args()


def main():
    args = parse_args()

    if args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    job = read_job(args.manifest, args.row_id)
    out_dir = out_dir_from_job(job, args)

    X, y, beta_true, sim_info = read_data(job, device)

    schedule = make_schedule(args)
    split = make_split(args.split_seed)
    save = cfg.SaveConfig(output_dir=None)

    print("[info] method: LaST-Flow")
    print("[info] mode:", args.mode)
    print("[info] row_id:", args.row_id)
    print("[info] setting:", job["setting"])
    print("[info] seed:", int(job["seed"]))
    print("[info] split_seed:", args.split_seed)
    print("[info] device:", device)
    print("[info] out_dir:", out_dir)

    extra = {}

    if args.mode == "recovery":
        extra = {
            "mcmc_root": str(rpath(args.mcmc_root)),
            "mcmc_setting": str(job["setting"]),
            "mcmc_seed": int(job["seed"]),
        }

    if args.mode == "lasso_recovery" and args.checkpoint_rule_path is not None:
        extra = {
            "checkpoint_rule_path": str(rpath(args.checkpoint_rule_path)),
        }

    train_seed = int(job["seed"]) if args.train_seed is None else int(args.train_seed)

    flow_out = fw.simflow_stagewise(
        X=X,
        y=y,
        beta_true=beta_true,
        sim_info=sim_info,
        seed=train_seed,
        device=device,
        family="gaussian",
        hidden_units=args.hidden_units,
        num_hidden_layers=args.num_hidden_layers,
        show_start=False,
        show_final=False,
        tau_end=args.tau_end,
        K_q=args.K_q,
        K_g=args.K_g,
        schedule_cfg=schedule,
        split_cfg=split,
        save_cfg=save,
        mode=args.mode,
        **extra,
    )

    write_tables(
        flow_out,
        job,
        schedule,
        out_dir,
        args.mode,
        K_q=args.K_q,
        K_g=args.K_g,
    )

    print("[done] completed")


if __name__ == "__main__":
    main()