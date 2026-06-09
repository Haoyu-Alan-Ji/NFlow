from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


ROOT = Path(__file__).resolve().parent

CAND_DIR = ROOT / "data/posterior_figure_candidates/n160p100_recovery"
SPEC_PATH = CAND_DIR / "selected_figure_specs.csv"

LAST_ROOT = ROOT / "data/n160p100_last_output/recovery32_grid"
MCMC_ROOT = ROOT / "data/n160p100_mcmc_output"

OUT_DIR = ROOT / "figures/posterior_recovery/n160p100_recovery"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_last_file(config_id, seed, filename):
    path = (
        LAST_ROOT
        / str(config_id)
        / "recovery"
        / "simple"
        / f"seed_{int(seed)}"
        / filename
    )
    return path


def find_mcmc_file(seed, filename):
    path = MCMC_ROOT / "simple" / f"seed_{int(seed)}" / filename
    return path


def read_draws(method, seed, config_id=None):
    if method == "last":
        path = find_last_file(config_id, seed, "last_beta_soft_draws.csv.gz")
    elif method == "mcmc":
        path = find_mcmc_file(seed, "mcmc_beta_draws.csv.gz")
    else:
        raise ValueError(method)

    df = pd.read_csv(path)
    if "draw_id" in df.columns:
        df = df.drop(columns=["draw_id"])
    return df


def add_small_jitter(x):
    x = np.asarray(x, dtype=float)
    if np.std(x) < 1e-10 or len(np.unique(x)) < 5:
        rng = np.random.default_rng(123)
        scale = max(np.max(np.abs(x)), 1.0) * 1e-5
        x = x + rng.normal(0.0, scale, size=len(x))
    return x


def kde_1d(x, grid):
    x = add_small_jitter(x)
    return gaussian_kde(x)(grid)


def plot_joint_density(config_id, seed, j0, k0, title, filename):
    last = read_draws("last", seed, config_id=config_id)
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
        (axes[0], x_last, y_last, "LaST"),
        (axes[1], x_mcmc, y_mcmc, "mcmc"),
    ]:
        xj = add_small_jitter(x)
        yj = add_small_jitter(y)
        vals = np.vstack([xj, yj])

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


def plot_marginal_density(config_id, seed, j0, title, filename):
    last = read_draws("last", seed, config_id=config_id)
    mcmc = read_draws("mcmc", seed)

    x_last = last[f"b{j0}"].to_numpy(float)
    x_mcmc = mcmc[f"b{j0}"].to_numpy(float)

    x_all = np.concatenate([x_last, x_mcmc])
    pad = 0.10 * (x_all.max() - x_all.min() + 1e-8)
    grid = np.linspace(x_all.min() - pad, x_all.max() + pad, 600)

    d_last = kde_1d(x_last, grid)
    d_mcmc = kde_1d(x_mcmc, grid)

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(grid, d_last, label="LaST")
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


def main():
    spec = pd.read_csv(SPEC_PATH)

    fig1 = spec.loc[spec["figure"] == "fig1_true_active_joint_density"].iloc[0]
    plot_joint_density(
        config_id=fig1["config_id"],
        seed=int(fig1["seed"]),
        j0=int(fig1["j0"]),
        k0=int(fig1["k0"]),
        title=(
            "True-active joint posterior recovery\n"
            f"config {fig1['config_id']}, seed {int(fig1['seed'])}, "
            rf"$\beta_{{{int(fig1['j0'])}}}$ vs $\beta_{{{int(fig1['k0'])}}}$"
        ),
        filename="fig1_true_active_joint_density",
    )

    fig2 = spec.loc[spec["figure"] == "fig2_true_active_marginal_density"].iloc[0]
    plot_marginal_density(
        config_id=fig2["config_id"],
        seed=int(fig2["seed"]),
        j0=int(fig2["j0"]),
        title=(
            "True-active marginal posterior recovery\n"
            f"config {fig2['config_id']}, seed {int(fig2['seed'])}, "
            rf"$\beta_{{{int(fig2['j0'])}}}$"
        ),
        filename="fig2_true_active_marginal_density",
    )

    fig3 = spec.loc[spec["figure"] == "fig3_true_zero_beta_density"].iloc[0]
    plot_marginal_density(
        config_id=fig3["config_id"],
        seed=int(fig3["seed"]),
        j0=int(fig3["j0"]),
        title=(
            "True-zero marginal posterior recovery\n"
            f"config {fig3['config_id']}, seed {int(fig3['seed'])}, "
            rf"$\beta_{{{int(fig3['j0'])}}}$"
        ),
        filename="fig3_true_zero_marginal_density",
    )

    print("[done] wrote figures to:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()