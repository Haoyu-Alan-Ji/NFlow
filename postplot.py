from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


ROOT = Path(__file__).resolve().parent

CAND_DIR = ROOT / "data/posterior_figure_candidates/n160p100_last"
SPEC_PATH = CAND_DIR / "selected_figure_specs.csv"
FIG4_CAND_PATH = CAND_DIR / "fig4_pip_recovery_candidates.csv"

LAST_ROOT = ROOT / "data/n160p100_last_output"
MCMC_ROOT = ROOT / "data/n160p100_mcmc_output"

OUT_DIR = ROOT / "figures/last_recovery/n160p100"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_seed_file(root, seed, filename):
    return list(root.rglob(f"seed_{seed}/{filename}"))[0]


def read_draws(method, seed):
    if method == "last":
        path = find_seed_file(LAST_ROOT, seed, "last_beta_soft_draws.csv.gz")
    elif method == "mcmc":
        path = find_seed_file(MCMC_ROOT, seed, "mcmc_beta_draws.csv.gz")
    else:
        raise ValueError(method)

    df = pd.read_csv(path)
    if "draw_id" in df.columns:
        df = df.drop(columns=["draw_id"])
    return df


def read_pip(method, seed):
    if method == "last":
        path = find_seed_file(LAST_ROOT, seed, "last_pip.csv")
    elif method == "mcmc":
        path = find_seed_file(MCMC_ROOT, seed, "mcmc_pip.csv")
    else:
        raise ValueError(method)

    return pd.read_csv(path)


def get_pip_value(method, seed, j0):
    tbl = read_pip(method, seed)
    row = tbl.loc[tbl["j0"] == j0].iloc[0]

    return {
        "pip": float(row["pip"]),
        "truth": int(row["truth"]),
        "beta_true": float(row["beta_true"]),
    }


def kde_1d(x, grid):
    return gaussian_kde(x)(grid)


def plot_joint_density(seed, j0, k0, title, filename):
    last = read_draws("last", seed)
    mcmc = read_draws("mcmc", seed)

    x_last = last[f"b{j0}"].to_numpy(float)
    y_last = last[f"b{k0}"].to_numpy(float)

    x_mcmc = mcmc[f"b{j0}"].to_numpy(float)
    y_mcmc = mcmc[f"b{k0}"].to_numpy(float)

    x_all = np.concatenate([x_last, x_mcmc])
    y_all = np.concatenate([y_last, y_mcmc])

    x_pad = 0.08 * (x_all.max() - x_all.min() + 1e-8)
    y_pad = 0.08 * (y_all.max() - y_all.min() + 1e-8)

    x_grid = np.linspace(x_all.min() - x_pad, x_all.max() + x_pad, 140)
    y_grid = np.linspace(y_all.min() - y_pad, y_all.max() + y_pad, 140)

    Xg, Yg = np.meshgrid(x_grid, y_grid)
    grid_points = np.vstack([Xg.ravel(), Yg.ravel()])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)

    for ax, x, y, label in [
        (axes[0], x_last, y_last, "last"),
        (axes[1], x_mcmc, y_mcmc, "mcmc"),
    ]:
        vals = np.vstack([x, y])
        Z = gaussian_kde(vals)(grid_points).reshape(Xg.shape)

        ax.contour(Xg, Yg, Z, levels=7)
        ax.scatter(x, y, s=4, alpha=0.12)
        ax.axhline(0, linewidth=0.8)
        ax.axvline(0, linewidth=0.8)

        ax.set_title(label)
        ax.set_xlabel(rf"$\beta_{{{j0}}}$")

    axes[0].set_ylabel(rf"$\beta_{{{k0}}}$")

    fig.suptitle(title)
    fig.tight_layout()

    fig.savefig(OUT_DIR / f"{filename}.png", dpi=300)
    fig.savefig(OUT_DIR / f"{filename}.pdf")
    plt.close(fig)


def plot_marginal_density(seed, j0, title, filename):
    last = read_draws("last", seed)
    mcmc = read_draws("mcmc", seed)

    x_last = last[f"b{j0}"].to_numpy(float)
    x_mcmc = mcmc[f"b{j0}"].to_numpy(float)

    x_all = np.concatenate([x_last, x_mcmc])
    pad = 0.10 * (x_all.max() - x_all.min() + 1e-8)
    grid = np.linspace(x_all.min() - pad, x_all.max() + pad, 600)

    d_last = kde_1d(x_last, grid)
    d_mcmc = kde_1d(x_mcmc, grid)

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(grid, d_last, label="last")
    ax.plot(grid, d_mcmc, label="mcmc")

    ax.axvline(0, linewidth=0.8)
    ax.set_xlabel(rf"$\beta_{{{j0}}}$")
    ax.set_ylabel("Posterior density")
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()

    fig.savefig(OUT_DIR / f"{filename}.png", dpi=300)
    fig.savefig(OUT_DIR / f"{filename}.pdf")
    plt.close(fig)


def plot_zero_density(seed, j0, title, filename):
    last = read_draws("last", seed)
    mcmc = read_draws("mcmc", seed)

    x_last = last[f"b{j0}"].to_numpy(float)
    x_mcmc = mcmc[f"b{j0}"].to_numpy(float)

    x_all = np.concatenate([x_last, x_mcmc])
    pad = 0.10 * (x_all.max() - x_all.min() + 1e-8)

    grid = np.linspace(
        x_all.min() - pad,
        x_all.max() + pad,
        600,
    )

    d_last = gaussian_kde(x_last)(grid)
    d_mcmc = gaussian_kde(x_mcmc)(grid)

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(grid, d_last, label="last")
    ax.plot(grid, d_mcmc, label="mcmc")

    ax.axvline(0, linewidth=0.8)
    ax.set_xlabel(rf"$\beta_{{{j0}}}$")
    ax.set_ylabel("Posterior density")
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()

    fig.savefig(OUT_DIR / f"{filename}.png", dpi=300)
    fig.savefig(OUT_DIR / f"{filename}.pdf")
    plt.close(fig)


def plot_pip_recovery(filename):
    df = pd.read_csv(FIG4_CAND_PATH).copy()

    labels = [
        f"{row.group}\nseed {row.seed}\nj={row.j0}"
        for row in df.itertuples(index=False)
    ]

    x = np.arange(len(df), dtype=float)

    fig, ax = plt.subplots(figsize=(11, 4.8))

    last_col = "softgate_last" if "softgate_last" in df.columns else "pip_last"

    ax.scatter(
        x - 0.08,
        df[last_col],
        marker="o",
        s=45,
        label="LaST softgate",
    )

    ax.scatter(
        x + 0.08,
        df["pip_mcmc"],
        marker="x",
        s=55,
        label="MCMC PIP",
    )

    for i, row in df.iterrows():
        ax.plot(
            [i - 0.08, i + 0.08],
            [row[last_col], row["pip_mcmc"]],
            linewidth=0.8,
            alpha=0.6,
        )

    if "group" in df.columns:
        n_active = int((df["group"] == "true_nonzero").sum())
        if n_active > 0 and n_active < len(df):
            ax.axvline(n_active - 0.5, linewidth=0.8)
            ax.text((n_active - 1) / 2, 1.03, "True nonzero", ha="center", fontsize=9)
            ax.text(n_active + (len(df) - n_active - 1) / 2, 1.03, "True zero", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=0)

    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel(r"$P(\beta_j \neq 0 \mid y)$")
    ax.set_title("Softgate / PIP recovery")
    ax.legend()

    fig.tight_layout()

    fig.savefig(OUT_DIR / f"{filename}.png", dpi=300)
    fig.savefig(OUT_DIR / f"{filename}.pdf")
    df.to_csv(OUT_DIR / f"{filename}_values.csv", index=False)

    plt.close(fig)


def main():
    spec = pd.read_csv(SPEC_PATH)

    fig1 = spec.loc[spec["figure"] == "fig1_true_active_joint_density"].iloc[0]
    plot_joint_density(
        seed=int(fig1["seed"]),
        j0=int(fig1["j0"]),
        k0=int(fig1["k0"]),
        title=(
            "True-active joint posterior recovery\n"
            f"seed {int(fig1['seed'])}, "
            rf"$\beta_{{{int(fig1['j0'])}}}$ vs $\beta_{{{int(fig1['k0'])}}}$"
        ),
        filename="fig1_true_active_joint_density",
    )

    fig2 = spec.loc[spec["figure"] == "fig2_true_active_marginal_density"].iloc[0]
    plot_marginal_density(
        seed=int(fig2["seed"]),
        j0=int(fig2["j0"]),
        title=(
            "True-active marginal posterior recovery\n"
            f"seed {int(fig2['seed'])}, "
            rf"$\beta_{{{int(fig2['j0'])}}}$"
        ),
        filename="fig2_true_active_marginal_density",
    )

    fig3 = spec.loc[spec["figure"] == "fig3_true_zero_beta_density"].iloc[0]
    plot_zero_density(
        seed=int(fig3["seed"]),
        j0=int(fig3["j0"]),
        title=(
            "True-zero marginal posterior recovery\n"
            f"seed {int(fig3['seed'])}, "
            rf"$\beta_{{{int(fig3['j0'])}}}$"
        ),
        filename="fig3_true_zero_marginal_density",
    )

    plot_pip_recovery(
        filename="fig4_pip_recovery",
    )

    print("[done] wrote figures to:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()