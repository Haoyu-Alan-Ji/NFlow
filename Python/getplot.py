import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import normflows as nf
import formula as fo

def plot_training(losses, tau_hist):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].plot(losses)
    ax[0].set_title("Loss history")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("reverse KL estimate")

    ax[1].plot(tau_hist)
    ax[1].set_title("Temperature annealing")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("tau")

    plt.tight_layout()
    plt.show()


def plot_posterior_inclusion_prob(
    posterior_inclusion_prob,
    beta_true=None,
    top_k=30,
    decision_threshold=0.5,
):
    pip_np = fo._to_cpu(posterior_inclusion_prob).numpy()
    idx = np.arange(len(pip_np))

    plt.figure(figsize=(10, 4))
    plt.bar(idx[:top_k], pip_np[:top_k])
    plt.axhline(decision_threshold, linestyle="--")

    if beta_true is not None:
        beta_true = fo._to_cpu(beta_true).numpy()
        true_support = (beta_true != 0).astype(float)
        plt.scatter(
            idx[:top_k],
            true_support[:top_k],
            marker="x",
            s=60,
            label="true support",
        )
        plt.legend()

    plt.title(f"Posterior inclusion probabilities (first {top_k})")
    plt.xlabel("j")
    plt.ylabel("P(selected | data)")
    plt.show()


def plot_beta_marginals(
    post_summary,
    beta_true=None,
    variable_indices=None,
    use_hard=False,
    bins=40,
):
    """
    Plot marginal posterior histograms for selected coordinates.
    """
    beta_samples = post_summary["beta_hard"] if use_hard else post_summary["beta_soft"]
    beta_samples = fo._to_cpu(beta_samples)

    p = beta_samples.shape[1]
    if variable_indices is None:
        variable_indices = list(range(min(6, p)))

    n_plots = len(variable_indices)
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 3), squeeze=False)

    beta_true_cpu = fo._to_cpu(beta_true) if beta_true is not None else None

    for ax, j in zip(axes[0], variable_indices):
        vals = beta_samples[:, j].numpy()
        ax.hist(vals, bins=bins, density=True, alpha=0.7)

        if beta_true_cpu is not None:
            bt = float(beta_true_cpu[j].item())
            ax.axvline(bt, linestyle="--", label="true beta")

            label = "true non-zero" if bt != 0 else "true zero"
            ax.set_title(f"beta[{j}] ({label})")
            ax.legend()
        else:
            ax.set_title(f"beta[{j}]")

        ax.set_xlabel("value")
        ax.set_ylabel("density")

    plt.tight_layout()
    plt.show()


def plot_predictive_fit(y_true, y_pred, title="Predictive fit"):
    y_true = fo._to_cpu(y_true).numpy()
    y_pred = fo._to_cpu(y_pred).numpy()

    plt.figure(figsize=(4.5, 4.5))
    plt.scatter(y_true, y_pred, alpha=0.6)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("true y")
    plt.ylabel("predicted y")
    plt.title(title)
    plt.tight_layout()
    plt.show()
