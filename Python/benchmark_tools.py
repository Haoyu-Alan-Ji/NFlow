from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import BenchmarkConfig, SplitConfig
from .metric import gaussian_predictive_metrics, support_metrics, ranking_metrics, benchmark_row_from_result
from .utils import Array, make_split, predict_linear, recover_original_scale


def top_var_table(
    *,
    beta_mean: Array,
    beta_sd: Array,
    support_score: Array,
    selected_support: Sequence[int],
    beta_true: Optional[Array],
    top_k: Optional[int] = None,
    score_name: str = "support_score",
) -> pd.DataFrame:
    selected_mask = np.zeros(len(beta_mean), dtype=int)
    if len(selected_support) > 0:
        selected_mask[list(selected_support)] = 1

    df = pd.DataFrame(
        {
            "j": np.arange(len(beta_mean), dtype=int),
            "beta_mean": beta_mean,
            "beta_sd": beta_sd,
            score_name: support_score,
            "selected": selected_mask,
        }
    )

    if beta_true is not None:
        beta_true = np.asarray(beta_true)
        truth = (beta_true != 0.0).astype(int)
        df["beta_true"] = beta_true
        df["truth"] = truth

    df["abs_beta_mean"] = np.abs(df["beta_mean"])
    df = df.sort_values([score_name, "abs_beta_mean"], ascending=[False, False]).reset_index(drop=True)

    if top_k is None:
        return df
    return df.head(top_k)

def _finalize_linear_result(
    *,
    method: str,
    seed: int,
    sim_info: Mapping[str, Any],
    splits: Mapping[str, Array],
    X: Array,
    y: Array,
    beta_true: Optional[Array],
    active_idx: Optional[Array],
    fit_out: Mapping[str, Any],
    x_mean: Array,
    x_scale: Array,
    y_mean: float,
    cfg: BenchmarkConfig,
    runtime_sec: float,
) -> Dict[str, Any]:
    beta_mean, beta_sd, intercept = recover_original_scale(
        np.asarray(fit_out["beta_mean_std"]),
        np.asarray(fit_out["beta_sd_std"]),
        x_mean,
        x_scale,
        y_mean,
    )
    support_score = np.asarray(fit_out["support_score_std"], dtype=float)
    selected_support = np.flatnonzero(support_score >= cfg.support_threshold).astype(int).tolist()
    sigma2 = float(fit_out.get("sigma2", max(np.var(y, ddof=0), 1e-12)))
    yhat = predict_linear(X, beta_mean, intercept)

    pred_rows: List[Dict[str, Any]] = []
    predictive_metrics: Dict[str, Dict[str, float]] = {}
    for split_name, idx in splits.items():
        split_metrics = gaussian_predictive_metrics(y[idx], yhat[idx], sigma2)
        predictive_metrics[split_name] = split_metrics
        pred_rows.append({"split": split_name, **split_metrics})

    selection = support_metrics(
        selected_support,
        beta_true=beta_true,
        active_idx=active_idx,
        p=X.shape[1],
    )

    ranking = ranking_metrics(
        support_score=support_score,
        beta_true=beta_true,
        active_idx=active_idx,
        p=X.shape[1],
    )

    selection.update(ranking)

    var_df = top_var_table(
        beta_mean=beta_mean,
        beta_sd=beta_sd,
        support_score=support_score,
        selected_support=selected_support,
        beta_true=beta_true,
        top_k=None,
        score_name="support_score",
    )

    if beta_true is None and active_idx is not None:
        truth = np.zeros(X.shape[1], dtype=int)
        truth[np.asarray(active_idx, dtype=int)] = 1
        var_df["truth"] = truth[var_df["j"].to_numpy()]

    result = {
        "method": method,
        "seed": seed,
        "sim_info": dict(sim_info),
        "splits": {k: np.asarray(v, dtype=int) for k, v in splits.items()},
        "runtime_sec": float(runtime_sec),
        "converged": bool(fit_out.get("converged", False)),
        "n_iter": int(fit_out.get("n_iter", 0)),
        "support_threshold": float(cfg.support_threshold),
        "beta_eps": float(cfg.beta_eps),
        "intercept": float(intercept),
        "sigma2": sigma2,
        "selected_support": selected_support,
        "beta_est": beta_mean,
        "beta_sd": beta_sd,
        "support_score": support_score,
        "pip": fit_out.get("pip_std"),
        "predictive_metrics": predictive_metrics,
        "selection_metrics": selection,
        "pred_table": pd.DataFrame(pred_rows),
        "var_table": var_df,
        "history": fit_out.get("history", pd.DataFrame()),
        "yhat": yhat,
        "config": asdict(cfg),
        "raw": fit_out.get("raw", {}),
    }
    return result


def get_method_registry() -> Dict[str, Callable[..., Dict[str, Any]]]:
    """Lazy registry keeps model files independent of core import order."""
    from .mf_sas import run_mf_spike_slab
    from .mf_ard import run_mf_ard
    from .mf_lasso import run_mf_bayes_lasso

    return {
        "mf_spike_slab": run_mf_spike_slab,
        "mf_ard": run_mf_ard,
        "mf_bayes_lasso": run_mf_bayes_lasso
    }


def run_baseline_method(
    *,
    method: str,
    X: Array,
    y: Array,
    beta_true: Optional[Array],
    active_idx: Optional[Array],
    seed: int,
    sim_info: Mapping[str, Any],
    splits: Mapping[str, Array],
    method_cfg: Optional[Any] = None,
) -> Dict[str, Any]:
    registry = get_method_registry()
    if method not in registry:
        raise KeyError(f"Unknown method '{method}'. Available: {sorted(registry.keys())}")
    runner = registry[method]
    return runner(
        X=X,
        y=y,
        beta_true=beta_true,
        active_idx=active_idx,
        seed=seed,
        sim_info=sim_info,
        splits=splits,
        cfg=method_cfg,
    )

def run_benchmark(
    *,
    X,
    y,
    methods: Sequence[str],
    split_cfg: SplitConfig,
    beta_true: Optional[Any] = None,
    active_idx: Optional[Any] = None,
    seed: int = 123,
    sim_info: Optional[Mapping[str, Any]] = None,
    method_cfgs: Optional[Mapping[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:

    X_np = X.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()

    beta_true_np = None
    if beta_true is not None:
        beta_true_np = beta_true.detach().cpu().numpy()

    if sim_info is None:
        sim_info = {}

    sim_info = dict(sim_info)

    if active_idx is None:
        active_idx = sim_info.get("active_idx", None)

    if active_idx is not None and hasattr(active_idx, "detach"):
        active_idx = active_idx.detach().cpu().numpy()

    if active_idx is not None:
        sim_info.setdefault("n_active", int(len(active_idx)))

    sim_info.setdefault("n", int(X_np.shape[0]))
    sim_info.setdefault("p", int(X_np.shape[1]))

    method_cfgs = dict(method_cfgs or {})

    splits = make_split(X_np.shape[0], split_cfg)

    all_results: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []

    for method in methods:
        result = run_baseline_method(
            method=method,
            X=X_np,
            y=y_np,
            beta_true=beta_true_np,
            active_idx=active_idx,
            seed=seed,
            sim_info=sim_info,
            splits=splits,
            method_cfg=method_cfgs.get(method),
        )

        all_results.append(result)
        all_rows.append(benchmark_row_from_result(result))

    table = pd.DataFrame(all_rows)
    return all_results, table
