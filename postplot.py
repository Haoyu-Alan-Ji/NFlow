from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


ROOT = Path(__file__).resolve().parent

CAND_DIR = (
    ROOT
    / "results"
    / "posterior_figure_candidates"
    / "n160p100_last"
)

SPEC_PATH = CAND_DIR / "selected_figure_specs.csv"
FIG4_CAND_PATH = CAND_DIR / "fig4_pip_recovery_candidates.csv"

RAT_ROOT = (
    ROOT
    / "data"
    / "n160p100"
    / "n160p100_last_output"
    / "simple"
)

MCMC_ROOT = (
    ROOT
    / "data"
    / "n160p100"
    / "n160p100_mcmc_output"
    / "simple"
)

OUT_DIR = (
    ROOT
    / "figures"
    / "last_recovery"
    / "n160p100"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_draws(method, seed):
    if method == "ratflow":
        path = (
            RAT_ROOT
            / f"seed_{seed}"
            / "last_beta_soft_draws.csv.gz"
        )
    elif method == "mcmc":
        path = (
            MCMC_ROOT
            / f"seed_{seed}"
            / "mcmc_beta_draws.csv.gz"
        )
    else:
        raise ValueError(method)

    if not path.exists():
        raise FileNotFoundError(path)

    print(f"[read {method}] {path}")

    df = pd.read_csv(path)

    if "draw_id" in df.columns:
        df = df.drop(columns=["draw_id"])

    return df


def read_pip(method, seed):
    if method == "ratflow":
        path = (
            RAT_ROOT
            / f"seed_{seed}"
            / "last_pip.csv"
        )
    elif method == "mcmc":
        path = (
            MCMC_ROOT
            / f"seed_{seed}"
            / "mcmc_pip.csv"
        )
    else:
        raise ValueError(method)

    if not path.exists():
        raise FileNotFoundError(path)

    print(f"[read {method}] {path}")

    return pd.read_csv(path)


def kde_1d(x, grid):
    return gaussian_kde(x)(grid)


def plot_joint_density(seed, j0, k0, filename):
    rat = read_draws("ratflow", seed)
    mcmc = read_draws("mcmc", seed)

    x_rat = rat[f"b{j0}"].to_numpy(float)
    y_rat = rat[f"b{k0}"].to_numpy(float)

    x_mcmc = mcmc[f"b{j0}"].to_numpy(float)
    y_mcmc = mcmc[f"b{k0}"].to_numpy(float)

    x_all = np.concatenate([x_rat, x_mcmc])
    y_all = np.concatenate([y_rat, y_mcmc])

    x_pad = 0.08 * (x_all.max() - x_all.min() + 1e-8)
    y_pad = 0.08 * (y_all.max() - y_all.min() + 1e-8)

    x_grid = np.linspace(x_all.min() - x_pad, x_all.max() + x_pad, 140)
    y_grid = np.linspace(y_all.min() - y_pad, y_all.max() + y_pad, 140)

    Xg, Yg = np.meshgrid(x_grid, y_grid)
    grid_points = np.vstack([Xg.ravel(), Yg.ravel()])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)

    for ax, x, y, panel_title in [
        (axes[0], x_rat, y_rat, "RAT-Flow"),
        (axes[1], x_mcmc, y_mcmc, "MCMC"),
    ]:
        vals = np.vstack([x, y])
        Z = gaussian_kde(vals)(grid_points).reshape(Xg.shape)

        ax.contour(Xg, Yg, Z, levels=7)
        ax.scatter(x, y, s=4, alpha=0.12)
        ax.axhline(0, linewidth=0.8)
        ax.axvline(0, linewidth=0.8)

        ax.set_title(panel_title)
        ax.set_xlabel(r"$\beta$")

    axes[0].set_ylabel(r"$\beta$")

    fig.suptitle("Joint posterior recovery")
    fig.tight_layout()

    fig.savefig(OUT_DIR / f"{filename}.png", dpi=300)
    fig.savefig(OUT_DIR / f"{filename}.pdf")
    plt.close(fig)


def plot_marginal_density(seed, j0, filename, title):
    rat = read_draws("ratflow", seed)
    mcmc = read_draws("mcmc", seed)

    x_rat = rat[f"b{j0}"].to_numpy(float)
    x_mcmc = mcmc[f"b{j0}"].to_numpy(float)

    x_all = np.concatenate([x_rat, x_mcmc])
    pad = 0.10 * (x_all.max() - x_all.min() + 1e-8)
    grid = np.linspace(x_all.min() - pad, x_all.max() + pad, 600)

    d_rat = kde_1d(x_rat, grid)
    d_mcmc = kde_1d(x_mcmc, grid)

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(grid, d_rat, label="RAT-Flow")
    ax.plot(grid, d_mcmc, label="MCMC")

    ax.axvline(0, linewidth=0.8)
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("Posterior density")
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()

    fig.savefig(OUT_DIR / f"{filename}.png", dpi=300)
    fig.savefig(OUT_DIR / f"{filename}.pdf")
    plt.close(fig)


def plot_pip_result(filename):
    df = pd.read_csv(FIG4_CAND_PATH).copy()

    last_col = "softgate_last" if "softgate_last" in df.columns else "pip_last"

    df["group"] = df["group"].replace(
        {
            "true_nonzero": "true_active",
            "true_zero": "true_zero",
        }
    )

    active = df.loc[df["group"] == "true_active"].copy().reset_index(drop=True)
    zero = df.loc[df["group"] == "true_zero"].copy().reset_index(drop=True)

    df_plot = pd.concat([active, zero], ignore_index=True)

    x = np.arange(len(df_plot), dtype=float)
    x_rat = x - 0.08
    x_mcmc = x + 0.08

    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    ax.scatter(
        x_rat,
        df_plot[last_col],
        marker="o",
        s=45,
        label="RAT-Flow",
    )

    ax.scatter(
        x_mcmc,
        df_plot["pip_mcmc"],
        marker="x",
        s=55,
        label="MCMC",
    )

    for i in range(len(df_plot)):
        ax.plot(
            [x_rat[i], x_mcmc[i]],
            [df_plot.iloc[i][last_col], df_plot.iloc[i]["pip_mcmc"]],
            linewidth=0.8,
            alpha=0.7,
        )

    n_active = len(active)
    n_zero = len(zero)

    if 0 < n_active < len(df_plot):
        ax.axvline(n_active - 0.5, linewidth=0.8)

    active_center = (n_active - 1) / 2
    zero_center = n_active + (n_zero - 1) / 2

    ax.set_xticks([active_center, zero_center])
    ax.set_xticklabels(["true_active", "true_zero"])

    ax.text(
        active_center,
        1.02,
        "True active",
        ha="center",
        va="bottom",
        fontsize=10,
    )
    ax.text(
        zero_center,
        1.02,
        "True zero",
        ha="center",
        va="bottom",
        fontsize=10,
    )

    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("Posterior inclusion probability (PIP)")
    ax.set_title("PIP result")
    ax.legend()

    fig.tight_layout()

    fig.savefig(OUT_DIR / f"{filename}.png", dpi=300)
    fig.savefig(OUT_DIR / f"{filename}.pdf")
    df_plot.to_csv(OUT_DIR / f"{filename}_values.csv", index=False)

    plt.close(fig)


def main():
    spec = pd.read_csv(SPEC_PATH)

    joint = spec.loc[
        spec["figure"] == "fig1_true_active_joint_density"
    ].iloc[0]

    active = spec.loc[
        spec["figure"] == "fig2_true_active_marginal_density"
    ].iloc[0]

    zero = spec.loc[
        spec["figure"].isin([
            "fig3_true_zero_beta_density",
            "fig3_true_zero_marginal_density",
        ])
    ].iloc[0]

    print(
        "[joint]",
        "seed =", int(joint["seed"]),
        "j0 =", int(joint["j0"]),
        "k0 =", int(joint["k0"]),
    )

    print(
        "[active]",
        "seed =", int(active["seed"]),
        "j0 =", int(active["j0"]),
    )

    print(
        "[inactive]",
        "seed =", int(zero["seed"]),
        "j0 =", int(zero["j0"]),
    )

    plot_joint_density(
        seed=int(joint["seed"]),
        j0=int(joint["j0"]),
        k0=int(joint["k0"]),
        filename="fig1_joint_density",
    )

    plot_marginal_density(
        seed=int(active["seed"]),
        j0=int(active["j0"]),
        filename="fig2_active_marginal_density",
        title="Active posterior recovery",
    )

    plot_marginal_density(
        seed=int(zero["seed"]),
        j0=int(zero["j0"]),
        filename="fig3_zero_marginal_density",
        title="Inactive posterior recovery",
    )

    plot_pip_result(
        filename="fig4_pip_result",
    )

    print("[done] wrote figures to:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()