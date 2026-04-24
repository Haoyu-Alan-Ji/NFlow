from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


Array = np.ndarray

def top_var_table(
    *,
    beta_mean: Array,
    beta_sd: Array,
    support_score: Array,
    selected_support: Sequence[int],
    beta_true: Optional[Array],
    top_k: int = 20,
    score_name: str = "support_score",
) -> pd.DataFrame:
    selected_mask = np.zeros(len(beta_mean), dtype=int)
    if len(selected_support) > 0:
        selected_mask[list(selected_support)] = 1
    df = pd.DataFrame({
        "j": np.arange(len(beta_mean), dtype=int),
        "beta_mean": beta_mean,
        "beta_sd": beta_sd,
        score_name: support_score,
        "selected": selected_mask,
    })
    if beta_true is not None:
        beta_true = np.asarray(beta_true)
        truth = (beta_true != 0.0).astype(int)
        df["beta_true"] = beta_true
        df["truth"] = truth
    df["abs_beta_mean"] = np.abs(df["beta_mean"])
    df = df.sort_values([score_name, "abs_beta_mean"], ascending=[False, False]).reset_index(drop=True)
    return df.head(top_k)


def benchmark_row_from_result(result: Mapping[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "method": result.get("method"),
        "seed": result.get("seed"),
        "runtime_sec": result.get("runtime_sec"),
        "converged": result.get("converged"),
        "n_iter": result.get("n_iter"),
        "support_size": result.get("selection_metrics", {}).get("support_size"),
        "precision": result.get("selection_metrics", {}).get("precision"),
        "recall": result.get("selection_metrics", {}).get("recall"),
        "f1": result.get("selection_metrics", {}).get("f1"),
        "fdr": result.get("selection_metrics", {}).get("fdr"),
        "tp": result.get("selection_metrics", {}).get("tp"),
        "fp": result.get("selection_metrics", {}).get("fp"),
        "fn": result.get("selection_metrics", {}).get("fn"),
        "train_mse": result.get("predictive_metrics", {}).get("train", {}).get("mse"),
        "val_mse": result.get("predictive_metrics", {}).get("val", {}).get("mse"),
        "test_mse": result.get("predictive_metrics", {}).get("test", {}).get("mse"),
        "train_r2": result.get("predictive_metrics", {}).get("train", {}).get("r2"),
        "val_r2": result.get("predictive_metrics", {}).get("val", {}).get("r2"),
        "test_r2": result.get("predictive_metrics", {}).get("test", {}).get("r2"),
        "val_nll": result.get("predictive_metrics", {}).get("val", {}).get("nll"),
        "test_nll": result.get("predictive_metrics", {}).get("test", {}).get("nll"),
    }
    sim_info = result.get("sim_info", {}) or {}
    for key in ["n", "p", "snr", "true_prop", "n_active"]:
        if key in sim_info:
            row[key] = sim_info[key]
    return row


# -----------------------------------------------------------------------------
# Shared result finalization for linear Gaussian baselines
# -----------------------------------------------------------------------------


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
    selected_support = np.flatnonzero(support_score >= cfg.support_threshold).tolist()
    sigma2 = float(fit_out.get("sigma2", max(np.var(y, ddof=0), 1e-12)))
    yhat = predict_linear(X, beta_mean, intercept)

    pred_table = []
    predictive_metrics = {}
    for split_name, idx in splits.items():
        split_metrics = gaussian_predictive_metrics(y[idx], yhat[idx], sigma2)
        predictive_metrics[split_name] = split_metrics
        pred_table.append({"split": split_name, **split_metrics})

    selection = support_metrics(selected_support, beta_true=beta_true, active_idx=active_idx, p=X.shape[1])
    var_df = top_var_table(
        beta_mean=beta_mean,
        beta_sd=beta_sd,
        support_score=support_score,
        selected_support=selected_support,
        beta_true=beta_true,
        top_k=min(20, X.shape[1]),
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
        "pred_table": pd.DataFrame(pred_table),
        "var_table": var_df,
        "history": fit_out.get("history", pd.DataFrame()),
        "yhat": yhat,
        "config": asdict(cfg),
        "raw": fit_out.get("raw", {}),
    }
    return result


# -----------------------------------------------------------------------------
# Adapters for existing spike-and-slab and future MCMC
# -----------------------------------------------------------------------------


def adapt_existing_spike_slab_output(
    flow_out: Mapping[str, Any],
    *,
    method: str = "flow_spike_slab",
    benchmark_support_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Convert an existing spike-and-slab result object into the unified format.

    Preferred raw fields to expose from the existing method are:
      - beta_est or beta_mean
      - beta_sd optional
      - pip or support_score
      - selected_support
      - predictive_metrics / pred_table
      - selection_metrics
      - runtime_sec
      - sim_info / splits
    """
    out = dict(flow_out)
    beta = np.asarray(out.get("beta_est", out.get("beta_mean", out.get("beta_hard_mean", []))), dtype=float)
    beta_sd = np.asarray(out.get("beta_sd", np.zeros_like(beta)), dtype=float)

    score = out.get("support_score", out.get("pip", out.get("inclusion_prob", None)))
    if score is not None:
        score = np.asarray(score, dtype=float)
    selected_support = out.get("selected_support", None)
    if selected_support is None:
        if score is None:
            selected_support = []
        else:
            threshold = 0.5 if benchmark_support_threshold is None else benchmark_support_threshold
            selected_support = np.flatnonzero(score >= threshold).tolist()

    pred_table = out.get("pred_table", pd.DataFrame())
    if isinstance(pred_table, list):
        pred_table = pd.DataFrame(pred_table)
    var_table = out.get("var_table", pd.DataFrame())
    if isinstance(var_table, list):
        var_table = pd.DataFrame(var_table)

    if var_table.empty and beta.size > 0:
        var_table = pd.DataFrame({
            "j": np.arange(beta.size, dtype=int),
            "beta_mean": beta,
            "beta_sd": beta_sd,
            "support_score": score if score is not None else np.zeros(beta.size),
            "selected": np.isin(np.arange(beta.size), np.asarray(selected_support, dtype=int)).astype(int),
        })

    return {
        "method": method,
        "seed": out.get("seed"),
        "sim_info": out.get("sim_info", {}),
        "splits": out.get("splits", {}),
        "runtime_sec": out.get("runtime_sec"),
        "converged": out.get("converged", True),
        "n_iter": out.get("n_iter", None),
        "support_threshold": benchmark_support_threshold,
        "beta_eps": out.get("beta_eps", None),
        "intercept": out.get("intercept", 0.0),
        "sigma2": out.get("sigma2", None),
        "selected_support": list(selected_support),
        "beta_est": beta,
        "beta_sd": beta_sd,
        "support_score": score,
        "pip": out.get("pip", score),
        "predictive_metrics": out.get("predictive_metrics", {}),
        "selection_metrics": out.get("selection_metrics", {}),
        "pred_table": pred_table,
        "var_table": var_table,
        "history": out.get("history", pd.DataFrame()),
        "config": out.get("config", {}),
        "raw": out,
    }


def run_mcmc_placeholder(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    raise NotImplementedError(
        "MCMC slot is intentionally reserved in the benchmark registry. "
        "Plug your collaborator's runner here later, then wrap it with adapt_existing_spike_slab_output."
    )


def get_method_registry() -> Dict[str, Callable[..., Dict[str, Any]]]:
    """Lazy registry to keep model files independent of core import order."""
    from mf_spike_slab_model import run_mf_spike_slab
    from mf_ard_model import run_mf_ard
    from mf_bayes_lasso_model import run_mf_bayes_lasso

    return {
        "mf_spike_slab": run_mf_spike_slab,
        "mf_ard": run_mf_ard,
        "mf_bayes_lasso": run_mf_bayes_lasso,
        "mcmc_spike_slab": run_mcmc_placeholder,
    }


# -----------------------------------------------------------------------------
# Benchmark orchestration
# -----------------------------------------------------------------------------


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


def run_one_setting_one_seed(
    *,
    simfun: Callable[..., Any],
    seed: int,
    n: int,
    p: int,
    snr: float,
    true_prop: float,
    methods: Sequence[str],
    split_cfg: SplitConfig,
    method_cfgs: Optional[Mapping[str, Any]] = None,
    sim_kwargs: Optional[Mapping[str, Any]] = None,
    external_runners: Optional[Mapping[str, Callable[..., Mapping[str, Any]]]] = None,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    sim_kwargs = dict(sim_kwargs or {})
    sim_payload = simfun(seed=seed, n=n, p=p, snr=snr, true_prop=true_prop, **sim_kwargs)
    sim = extract_sim_arrays(sim_payload)
    X = sim["X"]
    y = sim["y"]
    beta_true = sim.get("beta_true")
    active_idx = sim.get("active_idx")
    sim_info = {"n": n, "p": p, "snr": snr, "true_prop": true_prop}
    sim_info.update({k: v for k, v in sim.get("sim_info", {}).items() if k not in {"X", "y", "beta_true"}})
    if active_idx is not None:
        sim_info.setdefault("n_active", int(len(active_idx)))

    splits = make_splits(X.shape[0], split_cfg)
    method_cfgs = dict(method_cfgs or {})
    external_runners = dict(external_runners or {})

    results: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    for method in methods:
        if method in external_runners:
            out = external_runners[method](
                X=X,
                y=y,
                beta_true=beta_true,
                active_idx=active_idx,
                seed=seed,
                sim_info=sim_info,
                splits=splits,
                method_cfg=method_cfgs.get(method),
            )
            result = adapt_existing_spike_slab_output(out, method=method)
        else:
            result = run_baseline_method(
                method=method,
                X=X,
                y=y,
                beta_true=beta_true,
                active_idx=active_idx,
                seed=seed,
                sim_info=sim_info,
                splits=splits,
                method_cfg=method_cfgs.get(method),
            )
        results.append(result)
        rows.append(benchmark_row_from_result(result))

    return results, pd.DataFrame(rows)


def run_setting_grid(
    *,
    simfun: Callable[..., Any],
    setting: Mapping[str, Any],
    seeds: Sequence[int],
    methods: Sequence[str],
    split_cfg: SplitConfig,
    method_cfgs: Optional[Mapping[str, Any]] = None,
    sim_kwargs: Optional[Mapping[str, Any]] = None,
    external_runners: Optional[Mapping[str, Callable[..., Mapping[str, Any]]]] = None,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    all_results: List[Dict[str, Any]] = []
    all_rows: List[pd.DataFrame] = []
    for seed in seeds:
        results, rows = run_one_setting_one_seed(
            simfun=simfun,
            seed=seed,
            n=int(setting["n"]),
            p=int(setting["p"]),
            snr=float(setting["snr"]),
            true_prop=float(setting["true_prop"]),
            methods=methods,
            split_cfg=split_cfg,
            method_cfgs=method_cfgs,
            sim_kwargs=sim_kwargs,
            external_runners=external_runners,
        )
        all_results.extend(results)
        all_rows.append(rows)
    return all_results, pd.concat(all_rows, axis=0, ignore_index=True) if all_rows else pd.DataFrame()


def run_full_benchmark(
    *,
    simfun: Callable[..., Any],
    setting_grid: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    methods: Sequence[str],
    split_cfg: SplitConfig,
    method_cfgs: Optional[Mapping[str, Any]] = None,
    sim_kwargs: Optional[Mapping[str, Any]] = None,
    external_runners: Optional[Mapping[str, Callable[..., Mapping[str, Any]]]] = None,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    all_results: List[Dict[str, Any]] = []
    all_rows: List[pd.DataFrame] = []
    for setting in setting_grid:
        results, rows = run_setting_grid(
            simfun=simfun,
            setting=setting,
            seeds=seeds,
            methods=methods,
            split_cfg=split_cfg,
            method_cfgs=method_cfgs,
            sim_kwargs=sim_kwargs,
            external_runners=external_runners,
        )
        all_results.extend(results)
        all_rows.append(rows)
    table = pd.concat(all_rows, axis=0, ignore_index=True) if all_rows else pd.DataFrame()
    return all_results, table




