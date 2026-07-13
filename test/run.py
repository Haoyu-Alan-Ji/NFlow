from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import Python.framework as fw
import Python.config as cfg


def rpath(x):
    x = Path(x)
    return x if x.is_absolute() else PROJECT_ROOT / x


def read_job(manifest_path, row_id):
    manifest = pd.read_csv(rpath(manifest_path))
    return manifest.iloc[row_id - 1].copy()


def read_data(job, device):
    dat = pd.read_csv(rpath(job["data_path"]))
    beta_tbl = pd.read_csv(rpath(job["beta_path"]))

    X_np = dat.drop(columns=["y"]).to_numpy(dtype=np.float32)
    y_np = dat["y"].to_numpy(dtype=np.float32)
    beta_np = beta_tbl["beta_true"].to_numpy(dtype=np.float32)

    active_idx = np.flatnonzero(np.abs(beta_np) > 1e-12)
    active_order = active_idx[np.argsort(-np.abs(beta_np[active_idx]))]

    sim_info = {
        "sim": str(job["setting"]),
        "setting": str(job["setting"]),
        "seed": int(job["seed"]),
        "n": int(X_np.shape[0]),
        "p": int(X_np.shape[1]),
        "n_active": int(len(active_idx)),
        "active_idx": active_idx,
        "active_idx_by_abs": active_order,
        "sigma2": float(job["sigma2"]),
        "sigma": float(np.sqrt(float(job["sigma2"]))),
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
        recovery_min_epoch=args.recovery_min_epoch,
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("manifest")
    p.add_argument("row_id", type=int)
    p.add_argument("--config-name", default="last_default")
    p.add_argument("--output-root", default="data/n160p100_last_output")
    p.add_argument("--mcmc-root", default="data")
    p.add_argument("--train-seed", type=int, default=None)
    p.add_argument("--split-seed", type=int, default=12345)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    p.add_argument("--beta-mode", default="sigmoid", choices=["sigmoid", "relu", "group_relu"])
    p.add_argument("--coupling-type", default="semantic", choices=["meanfield", "affine", "semantic", "semantic_affine_control"])
    p.add_argument("--conditioner-type", default="mlp", choices=["mlp", "resnet"])
    p.add_argument("--hidden-units", type=int, default=64)
    p.add_argument("--num-hidden-layers", type=int, default=2)
    p.add_argument("--K-q", type=int, default=32)
    p.add_argument("--K-g", type=int, default=8)
    p.add_argument("--affine-layers-per-step", type=int, default=3)

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
    p.add_argument("--recovery-min-epoch", type=int, default=100)
    p.add_argument("--mcmc-during-training", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    job = read_job(args.manifest, args.row_id)
    X, y, beta_true, sim_info = read_data(job, device)
    sim_info["config_name"] = args.config_name

    out_dir = rpath(args.output_root) / str(job["setting"]) / f"seed_{int(job['seed'])}"
    schedule = make_schedule(args)

    try:
        split = cfg.SplitConfig(train_frac=0.6, val_frac=0.2, test_frac=0.2, seed=args.split_seed)
    except TypeError:
        split = cfg.SplitConfig(train_frac=0.6, val_frac=0.2, test_frac=0.2, seed=args.split_seed)

    save = cfg.SaveConfig(output_dir=str(out_dir))
    train_seed = int(job["seed"]) if args.train_seed is None else int(args.train_seed)

    print("[info] project_root:", PROJECT_ROOT)
    print("[info] method: LaST-Flow")
    print("[info] config_name:", args.config_name)
    print("[info] row_id:", args.row_id)
    print("[info] setting:", job["setting"])
    print("[info] seed:", int(job["seed"]))
    print("[info] split_seed:", args.split_seed)
    print("[info] device:", device)
    print("[info] data_path:", rpath(job["data_path"]))
    print("[info] beta_path:", rpath(job["beta_path"]))
    print("[info] out_dir:", out_dir)
    print("[info] mcmc_root:", rpath(args.mcmc_root))

    fw.simflow_stagewise(
        X=X,
        y=y,
        beta_true=beta_true,
        sim_info=sim_info,
        seed=train_seed,
        device=device,
        family="gaussian",
        hidden_units=args.hidden_units,
        num_hidden_layers=args.num_hidden_layers,
        beta_mode=args.beta_mode,
        coupling_type=args.coupling_type,
        conditioner_type=args.conditioner_type,
        K_q=args.K_q,
        K_g=args.K_g,
        schedule_cfg=schedule,
        split_cfg=split,
        save_cfg=save,
        show_start=False,
        show_final=True,
        mcmc_root=str(rpath(args.mcmc_root)),
        mcmc_setting=str(job["setting"]),
        mcmc_seed=int(job["seed"]),
        compare_mcmc=True,
        mcmc_during_training=args.mcmc_during_training,
    )

    print("[done] completed")


if __name__ == "__main__":
    main()