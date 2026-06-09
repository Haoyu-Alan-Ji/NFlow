from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


def seed_from_path(path: Path) -> int:
    return int(re.search(r"seed_(\d+)", str(path)).group(1))


def read_draws(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    if "draw_id" in df.columns:
        df = df.drop(columns=["draw_id"])
    return df.to_numpy(dtype=float)


def find_seed_dirs(root: Path, pip_name: str, draws_name: str):
    out = {}
    for pip_path in root.rglob(pip_name):
        d = pip_path.parent
        if (d / draws_name).exists():
            out[seed_from_path(d)] = d
    return out


def add_jitter(x):
    x = np.asarray(x, dtype=float)
    if np.std(x) < 1e-10 or len(np.unique(x)) < 5:
        rng = np.random.default_rng(123)
        scale = max(np.max(np.abs(x)), 1.0) * 1e-5
        x = x + rng.normal(0.0, scale, size=len(x))
    return x

def infer_config_id(root: Path, seed_dir: Path) -> str:
    rel = seed_dir.relative_to(root)
    parts = rel.parts

    if len(parts) >= 4 and parts[1] in {"recovery", "selection", "lasso_recovery"}:
        return parts[0]

    return "default"


def find_last_runs(root: Path, pip_name: str, draws_name: str):
    rows = []

    for pip_path in root.rglob(pip_name):
        d = pip_path.parent

        if not (d / draws_name).exists():
            continue

        rows.append({
            "config_id": infer_config_id(root, d),
            "seed": seed_from_path(d),
            "dir": d,
        })

    return rows


def find_mcmc_dirs(root: Path):
    out = {}

    for pip_path in root.rglob("mcmc_pip.csv"):
        d = pip_path.parent

        if (d / "mcmc_beta_draws.csv.gz").exists():
            out[seed_from_path(d)] = d

    return out

def symmetric_kl_grid(p, q, eps=1e-12):
    p = np.maximum(p, eps)
    q = np.maximum(q, eps)

    p = p / p.sum()
    q = q / q.sum()

    kl_pq = np.sum(p * np.log(p / q))
    kl_qp = np.sum(q * np.log(q / p))

    return float(0.5 * (kl_pq + kl_qp))


def kde_skl_1d(x, y, n_grid=256):
    x = add_jitter(x)
    y = add_jitter(y)

    lo = min(np.quantile(x, 0.001), np.quantile(y, 0.001))
    hi = max(np.quantile(x, 0.999), np.quantile(y, 0.999))
    pad = 0.1 * (hi - lo + 1e-8)

    grid = np.linspace(lo - pad, hi + pad, n_grid)

    p = gaussian_kde(x)(grid)
    q = gaussian_kde(y)(grid)

    return symmetric_kl_grid(p, q)


def kde_skl_2d(X, Y, n_grid=45):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    X = X + np.random.default_rng(123).normal(0.0, 1e-8, size=X.shape)
    Y = Y + np.random.default_rng(456).normal(0.0, 1e-8, size=Y.shape)

    xlo = min(np.quantile(X[:, 0], 0.001), np.quantile(Y[:, 0], 0.001))
    xhi = max(np.quantile(X[:, 0], 0.999), np.quantile(Y[:, 0], 0.999))
    ylo = min(np.quantile(X[:, 1], 0.001), np.quantile(Y[:, 1], 0.001))
    yhi = max(np.quantile(X[:, 1], 0.999), np.quantile(Y[:, 1], 0.999))

    xpad = 0.1 * (xhi - xlo + 1e-8)
    ypad = 0.1 * (yhi - ylo + 1e-8)

    gx = np.linspace(xlo - xpad, xhi + xpad, n_grid)
    gy = np.linspace(ylo - ypad, yhi + ypad, n_grid)

    xx, yy = np.meshgrid(gx, gy)
    pts = np.vstack([xx.ravel(), yy.ravel()])

    p = gaussian_kde(X.T)(pts)
    q = gaussian_kde(Y.T)(pts)

    return symmetric_kl_grid(p, q)


def bern_js(p, q, eps=1e-8):
    p = np.clip(float(p), eps, 1.0 - eps)
    q = np.clip(float(q), eps, 1.0 - eps)
    m = 0.5 * (p + q)

    kl_pm = p * np.log(p / m) + (1.0 - p) * np.log((1.0 - p) / (1.0 - m))
    kl_qm = q * np.log(q / m) + (1.0 - q) * np.log((1.0 - q) / (1.0 - m))

    return float(0.5 * kl_pm + 0.5 * kl_qm)


def read_last_table(last_dir: Path, p: int):
    df = pd.read_csv(last_dir / "last_pip.csv")

    out = pd.DataFrame({
        "j0": df["j0"].astype(int),
        "j1": df["j1"].astype(int) if "j1" in df.columns else df["j0"].astype(int) + 1,
        "beta_true": df["beta_true"].astype(float),
        "truth": df["truth"].astype(int),
    })

    if "pip" in df.columns:
        out["pip_last"] = df["pip"].astype(float)
    elif "pip_hard" in df.columns:
        out["pip_last"] = df["pip_hard"].astype(float)
    else:
        out["pip_last"] = np.nan

    if "selected" in df.columns:
        out["selected_last"] = df["selected"].astype(int)
    else:
        out["selected_last"] = (out["pip_last"] >= 0.5).astype(int)

    softgate_path = last_dir / "last_softgate_draws.csv.gz"
    if softgate_path.exists():
        G = read_draws(softgate_path)
        out["softgate_last"] = G.mean(axis=0)
    elif "softgate" in df.columns:
        out["softgate_last"] = df["softgate"].astype(float)
    elif "softgate_mean" in df.columns:
        out["softgate_last"] = df["softgate_mean"].astype(float)
    else:
        out["softgate_last"] = out["pip_last"]

    return out


def read_mcmc_table(mcmc_dir: Path):
    df = pd.read_csv(mcmc_dir / "mcmc_pip.csv")

    out = pd.DataFrame({
        "j0": df["j0"].astype(int),
        "pip_mcmc": df["pip"].astype(float),
    })

    if "selected" in df.columns:
        out["selected_mcmc"] = df["selected"].astype(int)
    else:
        out["selected_mcmc"] = (out["pip_mcmc"] >= 0.5).astype(int)

    return out


def variable_metrics(config_id, seed, last_dir, mcmc_dir, kde_grid_1d):
    L = read_draws(last_dir / "last_beta_soft_draws.csv.gz")
    M = read_draws(mcmc_dir / "mcmc_beta_draws.csv.gz")

    p = L.shape[1]

    last_tbl = read_last_table(last_dir, p=p)
    mcmc_tbl = read_mcmc_table(mcmc_dir)

    tab = last_tbl.merge(mcmc_tbl, on="j0", how="inner")

    rows = []
    eps = 1e-8

    for _, r in tab.iterrows():
        j = int(r["j0"])

        x_last = L[:, j]
        x_mcmc = M[:, j]

        mean_last = float(np.mean(x_last))
        mean_mcmc = float(np.mean(x_mcmc))

        sd_last = float(np.std(x_last, ddof=1))
        sd_mcmc = float(np.std(x_mcmc, ddof=1))

        mean_zerr = abs(mean_last - mean_mcmc) / (sd_mcmc + eps)
        sd_logerr = abs(np.log((sd_last + eps) / (sd_mcmc + eps)))
        sd_ratio = sd_last / (sd_mcmc + eps)
        moment_score = mean_zerr + sd_logerr

        truth = int(r["truth"])
        pip_last = float(r["pip_last"])
        pip_mcmc = float(r["pip_mcmc"])
        softgate_last = float(r["softgate_last"])

        if truth == 1:
            beta_skl = kde_skl_1d(x_last, x_mcmc, n_grid=kde_grid_1d)
        else:
            beta_skl = np.nan

        zero_beta_score = (
            abs(mean_last - mean_mcmc)
            + abs(sd_last - sd_mcmc)
            + abs(softgate_last - pip_mcmc)
        )

        rows.append({
            "config_id": config_id,
            "seed": seed,
            "j0": j,
            "j1": int(r["j1"]),
            "beta_true": float(r["beta_true"]),
            "truth": truth,

            "selected_last": int(r["selected_last"]),
            "selected_mcmc": int(r["selected_mcmc"]),

            "pip_last": pip_last,
            "pip_mcmc": pip_mcmc,
            "pip_absdiff": abs(pip_last - pip_mcmc),
            "pip_js": bern_js(pip_last, pip_mcmc),

            "softgate_last": softgate_last,
            "softgate_absdiff": abs(softgate_last - pip_mcmc),
            "softgate_js": bern_js(softgate_last, pip_mcmc),

            "mean_last": mean_last,
            "mean_mcmc": mean_mcmc,
            "sd_last": sd_last,
            "sd_mcmc": sd_mcmc,

            "mean_zerr": mean_zerr,
            "sd_logerr": sd_logerr,
            "sd_ratio": sd_ratio,
            "moment_score": moment_score,

            "beta_skl": beta_skl,
            "zero_beta_score": zero_beta_score,
        })

    return pd.DataFrame(rows), L, M


def active_pair_metrics(config_id, seed, vm, L, M, top_active_vars, kde_grid_2d, min_mcmc_active_pip):
    active = (
        vm[(vm["truth"] == 1) & (vm["pip_mcmc"] >= min_mcmc_active_pip)]
        .sort_values(["moment_score", "beta_skl"])
        .head(top_active_vars)
        .copy()
    )

    active_js = active["j0"].astype(int).tolist()
    rows = []

    for a in range(len(active_js)):
        for b in range(a + 1, len(active_js)):
            j = active_js[a]
            k = active_js[b]

            jr = vm.loc[vm["j0"] == j].iloc[0]
            kr = vm.loc[vm["j0"] == k].iloc[0]

            corr_last = float(np.corrcoef(L[:, j], L[:, k])[0, 1])
            corr_mcmc = float(np.corrcoef(M[:, j], M[:, k])[0, 1])
            corr_absdiff = abs(corr_last - corr_mcmc)

            joint_skl = kde_skl_2d(
                L[:, [j, k]],
                M[:, [j, k]],
                n_grid=kde_grid_2d,
            )

            pair_moment_score = float(jr["moment_score"] + kr["moment_score"])

            rows.append({
                "config_id": config_id,
                "seed": seed,
                "pair_type": "true_active_true_active",
                "j0": j,
                "k0": k,
                "j1": j + 1,
                "k1": k + 1,

                "beta_true_j": float(jr["beta_true"]),
                "beta_true_k": float(kr["beta_true"]),

                "moment_score_j": float(jr["moment_score"]),
                "moment_score_k": float(kr["moment_score"]),
                "pair_moment_score": pair_moment_score,

                "mean_zerr_j": float(jr["mean_zerr"]),
                "mean_zerr_k": float(kr["mean_zerr"]),
                "sd_logerr_j": float(jr["sd_logerr"]),
                "sd_logerr_k": float(kr["sd_logerr"]),

                "beta_skl_j": float(jr["beta_skl"]),
                "beta_skl_k": float(kr["beta_skl"]),

                "corr_last": corr_last,
                "corr_mcmc": corr_mcmc,
                "corr_absdiff": corr_absdiff,

                "joint_skl": joint_skl,

                "pair_score": pair_moment_score + 0.25 * corr_absdiff + 0.10 * joint_skl,
            })

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--mode",
        choices=["recovery", "selection", "lasso_recovery"],
        default="recovery",
    )

    ap.add_argument("--last-root", default=None)
    ap.add_argument("--mcmc-root", default="outputs_mcmc/n160p100_input")
    ap.add_argument("--out-dir", default=None)

    ap.add_argument("--top-active-vars", type=int, default=8)
    ap.add_argument("--top-zero-vars", type=int, default=30)
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--fig4-n-per-group", type=int, default=5)

    ap.add_argument("--kde-grid-1d", type=int, default=256)
    ap.add_argument("--kde-grid-2d", type=int, default=45)
    ap.add_argument("--min-mcmc-active-pip", type=float, default=0.5)

    args = ap.parse_args()

    if args.last_root is None:
        last_root = Path(f"data/n160p100_last_output/{args.mode}")
    else:
        last_root = Path(args.last_root)

    if args.out_dir is None:
        out_dir = Path(f"data/posterior_figure_candidates/n160p100_{args.mode}")
    else:
        out_dir = Path(args.out_dir)

    mcmc_root = Path(args.mcmc_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] mode: {args.mode}")
    print(f"[info] last_root: {last_root}")
    print(f"[info] mcmc_root: {mcmc_root}")
    print(f"[info] out_dir: {out_dir}")

    last_runs = find_last_runs(
        last_root,
        "last_pip.csv",
        "last_beta_soft_draws.csv.gz",
    )

    mcmc_dirs = find_mcmc_dirs(mcmc_root)

    last_runs = [
        r for r in last_runs
        if int(r["seed"]) in mcmc_dirs
    ]

    print(f"[info] last runs with draws: {len(last_runs)}")
    print(f"[info] mcmc seeds with draws: {len(mcmc_dirs)}")
    print(f"[info] matched last runs: {len(last_runs)}")

    all_vm = []
    all_aa = []

    for r in last_runs:
        config_id = str(r["config_id"])
        seed = int(r["seed"])
        last_dir = r["dir"]
        mcmc_dir = mcmc_dirs[seed]

        print(f"[config {config_id} | seed {seed}]")

        vm, L, M = variable_metrics(
            config_id=config_id,
            seed=seed,
            last_dir=last_dir,
            mcmc_dir=mcmc_dir,
            kde_grid_1d=args.kde_grid_1d,
        )

        aa = active_pair_metrics(
            config_id=config_id,
            seed=seed,
            vm=vm,
            L=L,
            M=M,
            top_active_vars=args.top_active_vars,
            kde_grid_2d=args.kde_grid_2d,
            min_mcmc_active_pip=args.min_mcmc_active_pip,
        )

        all_vm.append(vm)
        if len(aa) > 0:
            all_aa.append(aa)

    vm_all = pd.concat(all_vm, ignore_index=True)
    aa_all = pd.concat(all_aa, ignore_index=True) if len(all_aa) > 0 else pd.DataFrame()

    active_pool = vm_all[
        (vm_all["truth"] == 1)
        & (vm_all["pip_mcmc"] >= args.min_mcmc_active_pip)
    ].copy()

    zero_pool = vm_all[
        (vm_all["truth"] == 0)
        & (vm_all["selected_mcmc"] == 0)
    ].copy()

    fig1 = (
        aa_all
        .sort_values(["pair_score", "joint_skl"])
        .head(args.top_n)
        .copy()
    )

    fig2 = (
        active_pool
        .sort_values(["moment_score", "beta_skl"])
        .head(args.top_n)
        .copy()
    )

    fig3 = (
        zero_pool
        .sort_values(["zero_beta_score", "softgate_absdiff"])
        .head(args.top_n)
        .copy()
    )

    # Figure 4: softgate / PIP recovery candidates.
    # Keep both true nonzero and true zero variables, because the figure is meant
    # to show whether LaST softgate tracks the MCMC inclusion probability.
    fig4_active = (
        active_pool
        .sort_values(["softgate_absdiff", "moment_score"])
        .head(args.fig4_n_per_group)
        .copy()
    )
    fig4_active["group"] = "true_nonzero"

    fig4_zero = (
        zero_pool
        .sort_values(["softgate_absdiff", "zero_beta_score"])
        .head(args.fig4_n_per_group)
        .copy()
    )
    fig4_zero["group"] = "true_zero"

    fig4 = pd.concat([fig4_active, fig4_zero], ignore_index=True)

    vm_all.to_csv(out_dir / "variable_metrics_all.csv", index=False)
    aa_all.to_csv(out_dir / "pair_metrics_true_active_true_active_all.csv", index=False)

    fig1.to_csv(out_dir / "fig1_true_active_pair_candidates.csv", index=False)
    fig2.to_csv(out_dir / "fig2_true_active_variable_candidates.csv", index=False)
    fig3.to_csv(out_dir / "fig3_true_zero_variable_candidates.csv", index=False)
    fig4.to_csv(out_dir / "fig4_softgate_recovery_candidates.csv", index=False)

    # Backward-compatible alias, in case older plotting code expects this name.
    fig4.to_csv(out_dir / "fig4_pip_recovery_candidates.csv", index=False)

    selected = []

    if len(fig1) > 0:
        r = fig1.iloc[0]
        selected.append({
            "figure": "fig1_true_active_joint_density",
            "config_id": str(r["config_id"]),
            "seed": int(r["seed"]),
            "j0": int(r["j0"]),
            "k0": int(r["k0"]),
            "j1": int(r["j1"]),
            "k1": int(r["k1"]),
            "score_name": "pair_score",
            "score": float(r["pair_score"]),
            "joint_skl": float(r["joint_skl"]),
            "corr_absdiff": float(r["corr_absdiff"]),
        })

    if len(fig2) > 0:
        r = fig2.iloc[0]
        selected.append({
            "figure": "fig2_true_active_marginal_density",
            "config_id": str(r["config_id"]),
            "seed": int(r["seed"]),
            "j0": int(r["j0"]),
            "k0": "",
            "j1": int(r["j1"]),
            "k1": "",
            "score_name": "moment_score",
            "score": float(r["moment_score"]),
            "beta_skl": float(r["beta_skl"]),
            "mean_zerr": float(r["mean_zerr"]),
            "sd_logerr": float(r["sd_logerr"]),
        })

    if len(fig3) > 0:
        r = fig3.iloc[0]
        selected.append({
            "figure": "fig3_true_zero_beta_density",
            "config_id": str(r["config_id"]),
            "seed": int(r["seed"]),
            "j0": int(r["j0"]),
            "k0": "",
            "j1": int(r["j1"]),
            "k1": "",
            "score_name": "zero_beta_score",
            "score": float(r["zero_beta_score"]),
            "softgate_absdiff": float(r["softgate_absdiff"]),
        })

    for _, r in fig4.iterrows():
        selected.append({
            "figure": f"fig4_softgate_recovery_{r['group']}",
            "config_id": str(r["config_id"]),
            "seed": int(r["seed"]),
            "j0": int(r["j0"]),
            "k0": "",
            "j1": int(r["j1"]),
            "k1": "",
            "score_name": "softgate_absdiff",
            "score": float(r["softgate_absdiff"]),
            "softgate_last": float(r["softgate_last"]),
            "pip_mcmc": float(r["pip_mcmc"]),
            "pip_last": float(r["pip_last"]),
            "truth": int(r["truth"]),
            "group": str(r["group"]),
        })

    pd.DataFrame(selected).to_csv(out_dir / "selected_figure_specs.csv", index=False)

    print("\n[done] wrote candidate tables to:")
    print(out_dir)


if __name__ == "__main__":
    main()