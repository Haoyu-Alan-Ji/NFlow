import numpy as np
import matplotlib.pyplot as plt

from .utils import jaccard_similarity

def plot_training_overview(history_df, savepath=None):
    if len(history_df) == 0:
        return
    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)

    axes[0].plot(history_df["epoch"], history_df["loss_ema"])
    axes[1].plot(history_df["epoch"], history_df["val_mse"])
    axes[2].plot(history_df["epoch"], history_df["log_grad_ema"])
    axes[3].step(history_df["epoch"], history_df["support_size"], where="post")

    axes[0].set_ylabel("loss")
    axes[1].set_ylabel("val mse")
    axes[2].set_ylabel("log grad")
    axes[3].set_ylabel("support")
    axes[3].set_xlabel("epoch")

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath)
    plt.close(fig)


def plot_support_vs_predictive(history_df, savepath=None):
    if len(history_df) == 0:
        return
    plt.figure(figsize=(6, 4))
    plt.scatter(history_df["support_size"], history_df["val_mse"], c=history_df["tau"])
    plt.xlabel("support size")
    plt.ylabel("val mse")
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)
    plt.close()


def plot_boundary_density(boundary, final_support, never_selected_idx, savepath=None):
    if boundary.numel() == 0:
        return

    plt.figure(figsize=(7, 4))
    if len(final_support) > 0:
        vals = boundary[:, final_support].reshape(-1).numpy()
        plt.hist(vals, bins=50, density=True, alpha=0.5, label="selected")
    if len(never_selected_idx) > 0:
        vals = boundary[:, never_selected_idx].reshape(-1).numpy()
        plt.hist(vals, bins=50, density=True, alpha=0.5, label="never")

    plt.xlabel("d = u - t")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)
    plt.close()


def plot_uncertainty_vs_abs_boundary(boundary, hard_freq, savepath=None):
    if boundary.numel() == 0:
        return

    x = boundary.mean(dim=0).abs().numpy()
    y = 4 * hard_freq * (1 - hard_freq)

    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, s=15)
    plt.xlabel("|E[d]|")
    plt.ylabel("instability")
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)
    plt.close()


def plot_support_overlap_heatmap(history_df, max_ckpts=20, savepath=None):
    if len(history_df) == 0:
        return

    df = history_df.sort_values("epoch").copy()
    if len(df) > max_ckpts:
        idx = np.linspace(0, len(df) - 1, max_ckpts).round().astype(int)
        df = df.iloc[idx]

    supports = df["support_idx"].tolist()
    n = len(supports)
    mat = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            mat[i, j] = jaccard_similarity(supports[i], supports[j])

    plt.figure(figsize=(6, 5))
    plt.imshow(mat, vmin=0, vmax=1)
    plt.colorbar(label="Jaccard")
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)
    plt.close()



