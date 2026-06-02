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
from Python.metric import sample_posterior_latents, hard_support_from_draws


def rpath(x):
    x = Path(x)
    return x if x.is_absolute() else ROOT / x


def read_job(manifest_path, row_id):
    manifest = pd.read_csv(rpath(manifest_path))
    if row_id < 1 or row_id > len(manifest):
        raise ValueError(f"row_id={row_id} is invalid; manifest has {len(manifest)} rows.")
    return manifest.iloc[row_id - 1].copy()


def read_data(job, device):
    dat = pd.read_csv(rpath(job["data_path"]))
    beta_tbl = pd.read_csv(rpath(job["beta_path"]))

    X_np = dat.drop(columns=["y"]).to_numpy(dtype=np.float32)
    y_np = dat["y"].to_numpy(dtype=np.float32)
    beta_np = beta_tbl["beta_true"].to_numpy(dtype=np.float32)

    n, p = X_np.shape
    if len(beta_np) != p:
        raise ValueError(f"beta_true length {len(beta_np)} does not match p={p}.")

    sigma2 = float(job["sigma2"]) if "sigma2" in job.index and not pd.isna(job["sigma2"]) else 1.0

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
    )


def make_split(split_seed):
    return cfg.SplitConfig(
        train_frac=0.6,
        val_frac=0.2,
        test_frac=0.2,
        seed=split_seed,
    )


def out_dir_from_job(job):
    if "last_out_dir" in job.index and isinstance(job["last_out_dir"], str) and len(job["last_out_dir"]) > 0:
        return rpath(job["last_out_dir"])

    if "out_dir" in job.index and isinstance(job["out_dir"], str) and len(job["out_dir"]) > 0:
        return rpath(job["out_dir"].replace("outputs_mcmc", "outputs_lastflow"))

    return ROOT / "outputs_lastflow" / str(job["setting"]) / f"seed_{int(job['seed'])}"


def auroc(score, truth):
    n_pos = int((truth == 1).sum())
    n_neg = int((truth == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan

    ranks = pd.Series(score).rank(method="average").to_numpy()
    return float((ranks[truth == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auprc(score, truth):
    n_pos = int((truth == 1).sum())
    if n_pos == 0:
        return np.nan

    order = np.argsort(-score)
    y = truth[order]

    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = tp / np.maximum(tp + fp, 1)

    return float(precision[y == 1].sum() / n_pos)


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
        "auroc": auroc(score, truth),
        "auprc": auprc(score, truth),
    }


def posterior_pip(flow_out, schedule):
    model = flow_out["model"]

    with torch.no_grad():
        draws = sample_posterior_latents(model, R=schedule.diag_R_final)
        hard = hard_support_from_draws(draws, support_threshold=schedule.support_threshold)

    beta_samples = hard.get("beta_hard_samples", None)

    if beta_samples is None:
        vt = flow_out["final"]["var_table"]
        return vt["selected"].to_numpy(dtype=float)

    if isinstance(beta_samples, torch.Tensor):
        beta_samples = beta_samples.detach().cpu().numpy()

    return (np.abs(beta_samples) > 1e-12).mean(axis=0)


def write_tables(flow_out, job, schedule, out_dir):
    vt = flow_out["final"]["var_table"].copy()
    vt = vt.rename(columns={"j": "j0"})
    vt["j1"] = vt["j0"] + 1

    if "truth" not in vt.columns:
        vt["truth"] = (np.abs(vt["beta_true"]) > 1e-12).astype(int)

    vt["pip"] = posterior_pip(flow_out, schedule)
    vt["selected"] = (vt["pip"] > schedule.support_threshold).astype(int)
    vt["effect_mean"] = vt["beta_hard_mean"]

    pip_tbl = vt[
        ["j0", "j1", "beta_true", "truth", "pip", "selected", "effect_mean"]
    ].copy()

    pip_tbl.insert(0, "replicate_id", int(job["seed"]))
    pip_tbl.insert(0, "seed", int(job["seed"]))
    pip_tbl.insert(0, "setting", str(job["setting"]))
    pip_tbl.insert(0, "method", "last_flow")

    truth = pip_tbl["truth"].to_numpy(dtype=int)
    selected = pip_tbl["selected"].to_numpy(dtype=int)
    score = pip_tbl["pip"].to_numpy(dtype=float)

    active_idx = np.where(truth == 1)[0]
    selected_idx = np.where(selected == 1)[0]

    metrics = selection_metrics(selected, truth, score)

    summary_tbl = pd.DataFrame([{
        "method": "last_flow",
        "setting": str(job["setting"]),
        "seed": int(job["seed"]),
        "replicate_id": int(job["seed"]),
        "n": int(flow_out["sim_info"]["n"]),
        "p": int(flow_out["sim_info"]["p"]),
        "n_active": int(truth.sum()),
        "sigma2": float(flow_out["sim_info"]["sigma2"]),
        "support_threshold": float(schedule.support_threshold),
        "selected_size": int(selected.sum()),
        "active_idx0": ";".join(str(j) for j in active_idx),
        "selected_idx0": ";".join(str(j) for j in selected_idx),
        "selected_ckpt_id": int(flow_out["selected_ckpt_id"]),
        **metrics,
    }])

    out_dir.mkdir(parents=True, exist_ok=True)

    pip_tbl.to_csv(out_dir / "last_pip.csv", index=False)
    summary_tbl.to_csv(out_dir / "last_summary.csv", index=False)

    print("===== LaST-Flow result =====")
    print(f"seed          : {int(job['seed'])}")
    print(f"selected_size : {int(selected.sum())}")
    print(f"support_thr   : {schedule.support_threshold}")
    print()

    print("===== Selection metrics =====")
    print(
        summary_tbl[
            ["precision", "recall", "f1", "fdr", "tp", "fp", "fn", "tn", "support_size", "auroc", "auprc"]
        ].to_string(index=False)
    )
    print()

    print("===== Selected support =====")
    print(selected_idx.tolist())


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("manifest")
    p.add_argument("row_id", type=int)

    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    p.add_argument("--hidden-units", type=int, default=64)
    p.add_argument("--num-hidden-layers", type=int, default=2)
    p.add_argument("--K-q", type=int, default=32)
    p.add_argument("--K-g", type=int, default=8)

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
    p.add_argument("--support-threshold", type=float, default=0.05)

    # Fixed split seed. This is intentionally not read from manifest.
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
    out_dir = out_dir_from_job(job)

    X, y, beta_true, sim_info = read_data(job, device)

    schedule = make_schedule(args)
    split = make_split(args.split_seed)

    # No full artifact saving. The runner writes only last_summary.csv and last_pip.csv.
    save = cfg.SaveConfig(output_dir=None)

    print("[info] method: LaST-Flow")
    print("[info] row_id:", args.row_id)
    print("[info] setting:", job["setting"])
    print("[info] seed:", int(job["seed"]))
    print("[info] split_seed:", args.split_seed)
    print("[info] device:", device)
    print("[info] out_dir:", out_dir)

    flow_out = fw.simflow_stagewise(
        X=X,
        y=y,
        beta_true=beta_true,
        sim_info=sim_info,
        seed=int(job["seed"]),
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
    )

    write_tables(flow_out, job, schedule, out_dir)

    print("[done] completed")


if __name__ == "__main__":
    main()