import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import normflows as nf
import formula as fo


# =========================================================
# core comparision
def summarize_beta_samples(beta_samples):
    
    beta_mean_hat = beta_samples.mean(dim=0)

    beta_centered = beta_samples - beta_mean_hat

    beta_cov_hat = beta_centered.T @ beta_centered / (beta_samples.shape[0] - 1)

    return beta_mean_hat, beta_cov_hat


@torch.no_grad()
def postsum(model, target_dist, n_samples=5000, gate_threshold=0.5, inclusion_threshold=0.5,):
    z, _ = model.sample(num_samples=n_samples)

    # Soft quantities
    beta_soft = target_dist.beta_from_latent(z) # [m, p]
    gate_soft = target_dist.gate_from_latent(z) # [m, p]

    # Hard thresholding
    gate_hard = (gate_soft > gate_threshold).float()     # [m, p]
    beta_hard = target_dist.hard_beta_from_latent(z, threshold=gate_threshold) # [m, p]

    # Split latent blocks
    s, u, t = target_dist.split_latent(z) # s:[m,p], u:[m,p], t:[m,1] or [m]

    # Means
    beta_soft_mean = beta_soft.mean(dim=0)
    beta_hard_mean = beta_hard.mean(dim=0)
    gate_soft_mean = gate_soft.mean(dim=0)

    # Variances
    beta_soft_var = beta_soft.var(dim=0, unbiased=True)
    beta_hard_var = beta_hard.var(dim=0, unbiased=True)
    gate_soft_var = gate_soft.var(dim=0, unbiased=True)

    # Quantiles
    beta_soft_q025 = beta_soft.quantile(0.025, dim=0)
    beta_soft_q50  = beta_soft.quantile(0.50, dim=0)
    beta_soft_q975 = beta_soft.quantile(0.975, dim=0)

    beta_hard_q025 = beta_hard.quantile(0.025, dim=0)
    beta_hard_q50  = beta_hard.quantile(0.50, dim=0)
    beta_hard_q975 = beta_hard.quantile(0.975, dim=0)

    # PIP
    pip = gate_hard.mean(dim=0)

    # Final selected set
    selected = (pip > inclusion_threshold)
    selected_idx = torch.nonzero(selected, as_tuple=False).squeeze(1)

    # Posterior covariance summaries
    beta_soft_cov = summarize_beta_samples(beta_soft)[1]
    beta_hard_cov = summarize_beta_samples(beta_hard)[1]

    out = {# raw posterior samples
        "z": z, "s": s, "u": u, "t": t.squeeze(-1) if t.ndim > 1 else t,
        
        # soft/hard posterior samples
        "beta_soft": beta_soft, "beta_hard": beta_hard, "gate_soft": gate_soft, "gate_hard": gate_hard,
        
        # posterior means
        "beta_soft_mean": beta_soft_mean, "beta_hard_mean": beta_hard_mean, "gate_soft_mean": gate_soft_mean,
        
        # posterior variances
        "beta_soft_var": beta_soft_var, "beta_hard_var": beta_hard_var, "gate_soft_var": gate_soft_var,
        
        # posterior covariances
        "beta_soft_cov": beta_soft_cov, "beta_hard_cov": beta_hard_cov,
        
        # posterior quantiles
        "beta_soft_q025": beta_soft_q025, "beta_soft_q50": beta_soft_q50, "beta_soft_q975": beta_soft_q975,
        "beta_hard_q025": beta_hard_q025, "beta_hard_q50": beta_hard_q50, "beta_hard_q975": beta_hard_q975,
        
        # variable-selection summaries
        "pip": pip, "selected": selected, "selected_idx": selected_idx,
        
        # thresholds used
        "gate_threshold": gate_threshold, "inclusion_threshold": inclusion_threshold,
    }
    return out

# =========================================================




# =========================================================
# diagnosis
def y_diagnosis(y_true, y_pred):
    y_true = y_true.detach().cpu()
    y_pred = y_pred.detach().cpu()

    err = y_true - y_pred
    mse = torch.mean(err ** 2).item()
    rmse = math.sqrt(mse)
    mae = torch.mean(torch.abs(err)).item()

    sst = torch.sum((y_true - y_true.mean()) ** 2).item()
    sse = torch.sum(err ** 2).item()
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2,}


def beta_diagnosis(beta_true, beta_hat):

    beta_true = beta_true.detach().cpu()
    beta_hat = beta_hat.detach().cpu()

    diff = beta_hat - beta_true

    l1 = torch.norm(diff, p=1).item()
    l2 = torch.norm(diff, p=2).item()
    linf = torch.norm(diff, p=float("inf")).item()

    denom_l2 = torch.norm(beta_true, p=2).item()
    rel_l2 = l2 / denom_l2 if denom_l2 > 0 else np.nan

    return {"l1_error": l1, "l2_error": l2, "linf_error": linf, "rel_l2_error": rel_l2,}


def pip_diagnosis(pip_hat, pip_true):
    """
    If true PIP is available (e.g. from exact posterior or high-quality MCMC),
    compare estimated PIP against true PIP.
    """
    pip_hat = pip_hat.detach().cpu()
    pip_true = pip_true.detach().cpu()

    diff = pip_hat - pip_true

    mae = torch.mean(torch.abs(diff)).item()
    rmse = torch.sqrt(torch.mean(diff ** 2)).item()

    # rank correlation if scipy available
    try:
        from scipy.stats import spearmanr
        rho = float(spearmanr(pip_hat.numpy(), pip_true.numpy()).statistic)
    except Exception:
        rho = np.nan

    return {"pip_mae": mae, "pip_rmse": rmse, "pip_spearman": rho,}


def normres_compare(beta_samples, mu_post, Sigma_post):
    """
    Sanity check only.
    Compare posterior samples against an exact Gaussian posterior.
    """
    beta_mean_hat, beta_cov_hat = summarize_beta_samples(beta_samples)

    mean_abs_err = torch.norm(beta_mean_hat - mu_post).item()
    cov_abs_err = torch.norm(beta_cov_hat - Sigma_post).item()

    mean_rel_err = (
        torch.norm(beta_mean_hat - mu_post) / torch.norm(mu_post)
    ).item() if torch.norm(mu_post) > 0 else np.nan

    cov_rel_err = (
        torch.norm(beta_cov_hat - Sigma_post) / torch.norm(Sigma_post)
    ).item() if torch.norm(Sigma_post) > 0 else np.nan

    return {"mean_abs_err": mean_abs_err, "mean_rel_err": mean_rel_err, "cov_abs_err": cov_abs_err, "cov_rel_err": cov_rel_err,}

# =========================================================


# =========================================================
# AUROC / AUPRC
def ranking_metrics(labels, scores):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)

    if len(np.unique(labels)) < 2:
        return {"auroc": np.nan, "auprc": np.nan}

    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        return {
            "auroc": float(roc_auc_score(labels, scores)),
            "auprc": float(average_precision_score(labels, scores)),
        }
    except Exception:
        return {"auroc": np.nan, "auprc": np.nan}

# =========================================================




# =========================================================
# Truth-based evaluation (for simulation studies)

def eval_selection(beta_true, pip, selected):
    
    beta_true = beta_true.detach().cpu()
    pip = pip.detach().cpu()
    selected = selected.detach().cpu().bool()

    true_support = beta_true.ne(0)
    est_support = selected

    tp = int((true_support & est_support).sum().item())
    fp = int(((~true_support) & est_support).sum().item())
    fn = int((true_support & (~est_support)).sum().item())
    tn = int(((~true_support) & (~est_support)).sum().item())

    precision = fo._safe_div(tp, tp + fp)
    recall = fo._safe_div(tp, tp + fn)
    f1 = fo._safe_div(2 * precision * recall, precision + recall)
    specificity = fo._safe_div(tn, tn + fp)
    fpr = fo._safe_div(fp, fp + tn)

    denom = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1e-12))
    mcc = ((tp * tn - fp * fn) / denom) if denom > 0 else 0.0

    rank_metrics = ranking_metrics(
        labels=true_support.numpy().astype(int),
        scores=pip.numpy(),
    )

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1, "specificity": specificity,
        "fpr": fpr, "mcc": mcc, "auroc_pip": rank_metrics["auroc"], "auprc_pip": rank_metrics["auprc"],
    }

def evaluate_against_truth(beta_true, post_summary):

    beta_true = beta_true.detach().cpu()
    selected = post_summary["selected"].detach().cpu()
    posterior_inclusion_prob = post_summary["pip"].detach().cpu()
    beta_soft_mean = post_summary["beta_soft_mean"].detach().cpu()
    beta_hard_mean = post_summary["beta_hard_mean"].detach().cpu()

    true_support = beta_true.ne(0)
    est_support = selected.bool()

    tp = (true_support & est_support).sum().item()
    fp = ((~true_support) & est_support).sum().item()
    fn = (true_support & (~est_support)).sum().item()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    print("\n===== Variable selection summary =====")
    print("selected indices:", post_summary["selected_idx"].detach().cpu().tolist())
    print("TP:", tp, "FP:", fp, "FN:", fn)
    print("precision:", precision)
    print("recall   :", recall)

    print("\nfirst 20 posterior inclusion probabilities:")
    print(posterior_inclusion_prob[:20])

    print("\nfirst 20 beta soft posterior means:")
    print(beta_soft_mean[:20])

    print("\nfirst 20 beta hard posterior means:")
    print(beta_hard_mean[:20])

    print("\nfirst 20 true beta:")
    print(beta_true[:20])

# =========================================================



# =========================================================
# prediction
def posterior_predictive_summary(X, beta_samples):
    X = X.detach()
    beta_samples = beta_samples.detach()
    yhat_samples = beta_samples @ X.T   # [m, n]

    out = {
        "yhat_samples": yhat_samples,
        "yhat_mean": yhat_samples.mean(dim=0),
        "yhat_q025": yhat_samples.quantile(0.025, dim=0),
        "yhat_q5": yhat_samples.quantile(0.50, dim=0),
        "yhat_q975": yhat_samples.quantile(0.975, dim=0),
    }
    return out


def pred_output(X_train, y_train, X_test, y_test, beta_samples, beta_point,):
    out = {}

    if X_train is not None and y_train is not None:
        X_train = X_train.detach()
        y_train = y_train.detach()

        if beta_samples is not None:
            pp_train = posterior_predictive_summary(X_train, beta_samples)
            out["train_posterior_predictive"] = y_diagnosis(
                y_train, pp_train["yhat_mean"]
            )

        if beta_point is not None:
            yhat_train = X_train @ beta_point
            out["train_point_prediction"] = y_diagnosis(y_train, yhat_train)

    if X_test is not None and y_test is not None:
        X_test = X_test.detach()
        y_test = y_test.detach()

        if beta_samples is not None:
            pp_test = posterior_predictive_summary(X_test, beta_samples)
            out["test_posterior_predictive"] = y_diagnosis(
                y_test, pp_test["yhat_mean"]
            )

        if beta_point is not None:
            yhat_test = X_test @ beta_point
            out["test_point_prediction"] = y_diagnosis(y_test, yhat_test)

    return out
# =========================================================

# =========================================================
# wrapper
def res_wrapper(post_summary, beta_true, X_train, y_train,
    X_test, y_test, pip_true, exact_gaussian_mean, exact_gaussian_cov,):

    results = {}

    if beta_true is not None:
        results["support_soft"] = eval_selection(beta_true, post_summary["pip"], post_summary["selected"],)

        results["beta_soft_mean"] = beta_diagnosis(beta_true, post_summary["beta_soft_mean"],)

        results["beta_hard_mean"] = beta_diagnosis(beta_true, post_summary["beta_hard_mean"],)

    if pip_true is not None:
        results["pip"] = pip_diagnosis(post_summary["pip"], pip_true,)

    results["prediction_soft"] = pred_output(X_train, y_train, X_test, y_test,
        post_summary["beta_soft"], post_summary["beta_soft_mean"],)

    results["prediction_hard"] = pred_output(X_train, y_train, X_test, y_test,
        post_summary["beta_hard"], post_summary["beta_hard_mean"],)

    # -----------------------------------------------------
    # Optional Gaussian sanity check
    # -----------------------------------------------------
    if exact_gaussian_mean is not None and exact_gaussian_cov is not None:
        results["gaussian_sanity_soft"] = normres_compare(
            beta_samples=post_summary["beta_soft"],
            mu_post=exact_gaussian_mean,
            Sigma_post=exact_gaussian_cov,
        )
        results["gaussian_sanity_hard"] = normres_compare(
            beta_samples=post_summary["beta_hard"],
            mu_post=exact_gaussian_mean,
            Sigma_post=exact_gaussian_cov,
        )

    return results


def print_evaluation(results, post_summary=None, top_k=20):
    """
    Nicely print the evaluation dict.
    """
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    for block_name, block in results.items():
        print(f"\n[{block_name}]")
        if isinstance(block, dict):
            for k, v in block.items():
                if isinstance(v, dict):
                    print(f"  - {k}:")
                    for kk, vv in v.items():
                        print(f"      {kk}: {vv}")
                else:
                    print(f"  - {k}: {v}")
        else:
            print(block)

    if post_summary is not None:
        print("\n[posterior quick look]")
        print("selected_idx:", fo._to_cpu(post_summary["selected_idx"]).tolist())
        print("first PIPs:", fo._to_cpu(post_summary["pip"][:top_k]))
        print("first beta_soft_mean:", fo._to_cpu(post_summary["beta_soft_mean"][:top_k]))
        print("first beta_hard_mean:", fo._to_cpu(post_summary["beta_hard_mean"][:top_k]))

# =========================================================