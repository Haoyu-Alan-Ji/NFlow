from __future__ import annotations

import sys
from pathlib import Path
import argparse
import json
import pickle
import numpy as np
import pandas as pd
import torch


# ------------------------------------------------------------
# Project path
# ------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()

# If this file is test/HPC_benchmark.py,
# then PROJECT_ROOT is the parent directory of test/
PROJECT_ROOT = THIS_FILE.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ------------------------------------------------------------
# Project imports
# ------------------------------------------------------------

import Python.simfun as sim
import Python.benchmark as bm
import Python.config as cfg
import Python.framework as fw
import Python.model as md
import Python.artifact as art

# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------
# Data generators
# ------------------------------------------------------------

def make_dataset(setting: str, seed: int, device: torch.device, dtype=torch.float32):
    """
    Return X, y, beta_true, sim_info for one setting.
    """

    if setting == "simple":
        X, y, beta, sim_info = sim.simfun1(
            n=1000,
            p=100,
            seed=seed,
            snr=2.5,
            true_prop=0.1,
            device=device,
            dtype=dtype,
        )

    elif setting == "block_corr":
        X, y, beta, sim_info = sim.simfun_block_corr(
            n=1000,
            p=100,
            seed=seed,
            snr=2.5,
            true_prop=0.1,
            rho=0.8,
            block_size=10,
            device=device,
            dtype=dtype,
        )

    elif setting == "jitter":
        X, y, beta, sim_info = sim.simfun_group_competition(
            n=1000,
            p=100,
            seed=seed,
            snr=2.5,
            true_prop=0.1,
            group_size=10,
            noise_x=0.15,
            one_active_per_group=True,
            device=device,
            dtype=dtype,
        )

    else:
        raise ValueError(f"Unknown setting: {setting}")

    sim_info = dict(sim_info)
    sim_info["setting"] = setting
    sim_info["seed"] = seed

    return X, y, beta, sim_info


# ------------------------------------------------------------
# Configs
# ------------------------------------------------------------
def make_flow_depths(setting: str):
    """
    Choose flow depths by data-generating setting.

    simple / block_corr:
        K_q = 32, K_g = 16

    jitter:
        K_q = 32, K_g = 8
    """
    if setting in {"simple", "block_corr"}:
        return 32, 16

    if setting == "jitter":
        return 32, 8

    raise ValueError(f"Unknown setting: {setting}")


def make_schedule_cfg():
    return cfg.StagewiseAnnealConfig(
        tau_start=0.5,
        tau_end=0.1,
        warmup_epochs=300,
        n_anneal_stages=20,
        min_stage_epochs=100,
        max_stage_epochs=400,
        base_lr=5e-5,
        stage_lr_decay=0.7,
        eval_every=25,
        print_every=100,
        diag_R_train=256,
        diag_R_final=4000,
        support_threshold=0.5,
    )


def make_split_cfg(seed: int):
    return cfg.SplitConfig(
        train_frac=0.6,
        val_frac=0.2,
        test_frac=0.2,
        seed=seed,
    )


def make_meanfield_cfgs():
    mf_sas_cfg = bm.MFSpikeSlabConfig(
        pi=0.2,
        slab_var=20.0,
        max_iter=1000,
        tol=1e-6,
        update_sigma2=True,
        support_threshold=0.5,
        standardize_x=True,
        center_y=True,
    )

    mf_ard_cfg = bm.MFARDConfig(
        a0=1e-2,
        b0=1e-2,
        c0=1e-2,
        d0=1e-2,
        min_sigma2=1e-8,
        max_iter=1000,
        tol=1e-6,
        beta_eps=0.1,
        support_threshold=0.5,
        standardize_x=True,
        center_y=True,
    )

    lasso_cfg = bm.MFBayesLassoConfig(
        max_iter=1000,
        tol=1e-6,
        beta_eps=0.1,
        support_threshold=0.5,
        standardize_x=True,
        center_y=True,
    )

    return {
        "mf_spike_slab": mf_sas_cfg,
        "mf_ard": mf_ard_cfg,
        "mf_bayes_lasso": lasso_cfg,
    }


def make_save_cfg(output_dir: Path, flow: bool):
    """
    Light save policy for Table 1 plus minimal debugging.
    """
    return cfg.SaveConfig(
        output_dir=str(output_dir),

        save_summary_csv=not flow,
        save_results_pickle=False,
        save_metadata_json=False,
        save_manifest_json=False,
        save_benchmark_csv=False,

        save_history_csv=False,
        save_predictions_csv=False,
        save_var_table_csv=False,
        save_support_sets_json=False,

        save_checkpoint_manifest=False,

        # Keep only small final json for flow debugging.
        save_final_json=bool(flow),

        save_yhat_csv=False,
    )

# ------------------------------------------------------------
# Save helpers
# ------------------------------------------------------------

def safe_json_dump(obj, path: Path):
    def convert(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().tolist()
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.integer, np.floating)):
            return x.item()
        if isinstance(x, set):
            return sorted(list(x))
        return x

    with open(path, "w") as f:
        json.dump(obj, f, default=convert, indent=2)


def resolve_out_root(out_root_arg: str) -> Path:
    out_root = Path(out_root_arg)
    if not out_root.is_absolute():
        out_root = PROJECT_ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)
    return out_root


def save_table(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# ------------------------------------------------------------
# Main worker
# ------------------------------------------------------------

def run_one_seed(args):
    seed = int(args.seed)
    setting = args.setting

    if args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_root = resolve_out_root(args.out_root)
    seed_dir = out_root / setting / f"seed_{seed:03d}"
    mf_dir = seed_dir / "meanfield"
    flow_dir = seed_dir / "flow"

    seed_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] setting={setting}")
    print(f"[INFO] seed={seed}")
    print(f"[INFO] device={device}")
    print(f"[INFO] seed_dir={seed_dir}")

    # ----------------------------
    # Data
    # ----------------------------
    X, y, beta_true, sim_info = make_dataset(
        setting=setting,
        seed=seed,
        device=device,
        dtype=torch.float32,
    )

    safe_json_dump(sim_info, seed_dir / "sim_info.json")

    # ----------------------------
    # Shared configs
    # ----------------------------
    split_cfg = make_split_cfg(seed)
    schedule_cfg = make_schedule_cfg()
    method_cfgs = make_meanfield_cfgs()
    K_q, K_g = make_flow_depths(setting)

    # ----------------------------
    # Mean-field baselines
    # ----------------------------
    all_rows = []

    if not args.skip_meanfield:
        print("[INFO] Running mean-field baselines")

        mfresults, mftable = bm.run_benchmark(
            X=X,
            y=y,
            beta_true=beta_true,
            active_idx=sim_info["active_idx"],
            seed=seed,
            sim_info=sim_info,
            methods=[
                "mf_spike_slab",
                "mf_ard",
                "mf_bayes_lasso",
            ],
            split_cfg=split_cfg,
            method_cfgs=method_cfgs,
        )

        mftable = mftable.copy()
        mftable["setting"] = setting
        mftable["seed"] = seed

        save_mf = make_save_cfg(mf_dir, flow=False)

        # Allow debug pickle only when explicitly requested.
        save_mf.save_results_pickle = bool(args.save_debug_pickle)

        art.save_result_data(
            results=mfresults,
            table=mftable,
            prefix="meanfield",
            save_cfg=save_mf,
        )

        all_rows.append(mftable)

    # ----------------------------
    # Proposed flow method
    # ----------------------------
    if not args.skip_flow:
        print("[INFO] Running proposed flow method")

        save_flow = make_save_cfg(flow_dir, flow=True)

        flow_out = fw.simflow_stagewise(
            X=X,
            y=y,
            beta_true=beta_true,
            sim_info=sim_info,
            seed=seed,
            device=device,
            family="gaussian",
            hidden_units=64,
            num_hidden_layers=2,
            show_start=False,
            show_final=False,
            tau_end=0.1,
            K_q=K_q,
            K_g=K_g,
            schedule_cfg=schedule_cfg,
            split_cfg=split_cfg,
            save_cfg=save_flow,
        )

        flow_row = dict(flow_out["summary_row"])
        flow_row["setting"] = setting
        flow_row["seed"] = seed
        flow_row["method"] = flow_row.get("method", "nf_flow")

        flow_table = pd.DataFrame([flow_row])
        save_table(flow_table, flow_dir / "summary.csv")

        all_rows.append(flow_table)

    # ----------------------------
    # Combined seed-level summary
    # ----------------------------
    if len(all_rows) > 0:
        combined = pd.concat(all_rows, ignore_index=True)
        save_table(combined, seed_dir / "summary_all_methods.csv")
        print("[INFO] Wrote", seed_dir / "summary_all_methods.csv")

    print("[INFO] Done")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--setting",
        type=str,
        required=True,
        choices=["simple", "block_corr", "jitter"],
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="data/hpc_benchmark",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )

    parser.add_argument("--skip-meanfield", action="store_true")
    parser.add_argument("--skip-flow", action="store_true")
    parser.add_argument("--save-debug-pickle", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    run_one_seed(parse_args())