import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Mapping, Optional
import pandas as pd
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


# -----------------------------------------------------------------------------
# Plotting helpers, kept in core for now
# -----------------------------------------------------------------------------


def plot_runtime_vs_f1(table: pd.DataFrame, output_path: Optional[str] = None) -> Any:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for method, sub in table.groupby("method"):
        ax.scatter(sub["runtime_sec"], sub["f1"], label=method, alpha=0.75)
    ax.set_xlabel("runtime_sec")
    ax.set_ylabel("F1")
    ax.set_title("Runtime vs F1")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.2)
    if output_path:
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig


def plot_test_mse_vs_support_size(table: pd.DataFrame, output_path: Optional[str] = None) -> Any:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for method, sub in table.groupby("method"):
        ax.scatter(sub["support_size"], sub["test_mse"], label=method, alpha=0.75)
    ax.set_xlabel("support_size")
    ax.set_ylabel("test_mse")
    ax.set_title("Support size vs test MSE")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.2)
    if output_path:
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig


def plot_precision_recall(table: pd.DataFrame, output_path: Optional[str] = None) -> Any:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for method, sub in table.groupby("method"):
        ax.scatter(sub["recall"], sub["precision"], label=method, alpha=0.75)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Precision vs recall")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.2)
    if output_path:
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig


def plot_support_score_rank(result: Mapping[str, Any], output_path: Optional[str] = None) -> Any:
    import matplotlib.pyplot as plt
    score = np.asarray(result.get("support_score", []), dtype=float)
    beta_true = None
    if isinstance(result.get("var_table"), pd.DataFrame) and "truth" in result["var_table"].columns:
        score_df = result["var_table"].sort_values("support_score", ascending=False)
        truth = score_df["truth"].to_numpy(dtype=int)
        score = score_df["support_score"].to_numpy(dtype=float)
        beta_true = truth
    else:
        order = np.argsort(-score)
        score = score[order]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(np.arange(len(score)), score, linewidth=1.6)
    if beta_true is not None:
        hit_idx = np.flatnonzero(beta_true == 1)
        ax.scatter(hit_idx, score[hit_idx], marker="x", s=35, label="truth")
        ax.legend(frameon=False)
    ax.set_xlabel("rank")
    ax.set_ylabel("support_score / PIP")
    ax.set_title(f"Support score ranking: {result.get('method', '')}")
    ax.grid(True, alpha=0.2)
    if output_path:
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig
