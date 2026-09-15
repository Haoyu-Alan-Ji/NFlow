#!/usr/bin/env python3
"""p=6 bounded-gate x affine/spline x unit/feature comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Python import metric
from Python.bnn_train import train_grouped_bnn
from Python.experiment_utils import (
    Tee,
    choose_device,
    common_joint_axis,
    common_joint_draw_count,
    comparison_metrics,
    make_p6_bounded_data,
    run_reference,
    save_attention_diagnostics,
    save_dataset,
    save_frame,
    save_joint_plot,
    save_json,
    save_mcmc_diagnostics,
    save_reference,
    save_training,
)
from Python.model2 import GroupedBNNVI, run_grouped_acceptance_tests


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument("--run-name", default="p6_bounded_transport")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n", type=int, default=240)
    parser.add_argument("--fit-units", type=int, default=5)
    parser.add_argument("--sigma2", type=float, default=1.0)
    parser.add_argument("--gate-tau", type=float, default=1.0)
    parser.add_argument("--gate-delta", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--warmup-epochs", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--r-train", type=int, default=100)
    parser.add_argument("--r-eval", type=int, default=1000)
    parser.add_argument("--r-final", type=int, default=5000)
    parser.add_argument("--flow-depth", type=int, default=4)
    parser.add_argument("--token-dim", type=int, default=32)
    parser.add_argument("--attention-heads", type=int, default=2)
    parser.add_argument("--scale-clip", type=float, default=1.5)
    parser.add_argument("--spline-bins", type=int, default=8)
    parser.add_argument("--spline-tail-bound", type=float, default=3.0)
    parser.add_argument("--min-bin-width", type=float, default=1e-3)
    parser.add_argument("--min-bin-height", type=float, default=1e-3)
    parser.add_argument("--min-derivative", type=float, default=1e-3)
    parser.add_argument("--inverse-tolerance", type=float, default=1e-4)
    parser.add_argument("--mcmc-chains", type=int, default=4)
    parser.add_argument("--mcmc-n", type=int, default=6000)
    parser.add_argument("--mcmc-burnin", type=int, default=1000)
    parser.add_argument("--mcmc-thin", type=int, default=1)
    parser.add_argument("--min-draws", type=int, default=50)
    parser.add_argument("--density-max-draws", type=int, default=2000)
    parser.add_argument("--attention-draws", type=int, default=1000)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse complete MCMC/VI artifacts from an identical run config.",
    )
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def quicken(args):
    if not args.quick:
        return
    args.epochs = 12
    args.warmup_epochs = 4
    args.eval_every = 4
    args.r_train = 8
    args.r_eval = 32
    args.r_final = 64
    args.mcmc_n = 40
    args.mcmc_burnin = 10
    args.mcmc_thin = 1
    args.min_draws = 3
    args.density_max_draws = 32
    args.attention_draws = 32
    if not args.run_name.endswith("_quick"):
        args.run_name += "_quick"


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def gates(args):
    return {
        "normalized_requ": {
            "gate_type": "normalized_requ",
            "gate_power": 2.0,
            "gate_tau": float(args.gate_tau),
            "gate_delta": float(args.gate_delta),
        },
        "smooth_step": {
            "gate_type": "smooth_step",
            "gate_power": 1.0,
            "gate_tau": None,
            "gate_delta": float(args.gate_delta),
        },
    }


def hard_gate(args):
    return {
        "gate_type": "hard",
        "gate_power": 1.0,
        "gate_tau": None,
        "gate_delta": float(args.gate_delta),
    }


def run_config(args):
    config = dict(vars(args))
    config.update({
        "implementation_version": "p6_bounded_transport_v1",
        "p": 6,
        "H_fit": int(args.fit_units),
        "H_true": 2,
        "p_true": 2,
        "active_features": [0, 3],
        "posterior_draws": int(args.r_final),
        "total_epochs": int(args.epochs),
        "R_train": int(args.r_train),
        "conditioner_type": "improved_separate_attention",
        "selection_modes": ["unit_group", "feature_group"],
        "gate_types": ["normalized_requ", "smooth_step"],
        "coupling_types": ["affine", "spline"],
        "hidden_activation": "relu",
        "architecture_mode": "stacked",
        "hidden_dims": [int(args.fit_units)],
        "target_signal_sd": 1.5,
        "x_range": [-2.5, 2.5],
        "split_rule": "seed+1000 permutation; 60/20/20",
        "learning_rate": 3e-4,
        "prior": "standard_normal_all_latents",
        "init_loc_jitter": 0.05,
        "init_sd": 0.5,
        "grad_clip": 5.0,
    })
    return config


def validate_settings(args):
    fixed = {
        "n": (args.n, 240),
        "fit_units": (args.fit_units, 5),
        "flow_depth": (args.flow_depth, 4),
        "warmup_epochs": (args.warmup_epochs, 500),
        "epochs": (args.epochs, 2000),
        "r_train": (args.r_train, 100),
        "attention_heads": (args.attention_heads, 2),
        "gate_tau": (args.gate_tau, 1.0),
        "gate_delta": (args.gate_delta, 1.0),
    }
    if args.quick:
        fixed = {
            key: value for key, value in fixed.items()
            if key in {"n", "fit_units", "attention_heads", "gate_tau", "gate_delta"}
        }
    failures = [
        f"{name}={actual} (required {expected})"
        for name, (actual, expected) in fixed.items()
        if actual != expected
    ]
    if failures:
        raise ValueError(
            "Formal benchmark settings are fixed by the experiment prompt: "
            + "; ".join(failures)
        )
    if args.mcmc_chains < 2:
        raise ValueError("mcmc_chains must be at least 2 for R-hat.")


def print_settings(config, device, output_dir):
    keys = (
        "n", "p", "H_fit", "H_true", "p_true", "active_features",
        "flow_depth", "warmup_epochs", "total_epochs", "R_train",
        "posterior_draws", "seed", "selection_modes", "gate_types",
        "coupling_types", "conditioner_type", "attention_heads",
        "spline_bins", "mcmc_chains", "mcmc_n", "mcmc_burnin",
        "sigma2", "target_signal_sd", "split_rule", "learning_rate",
    )
    print("=== fixed experiment settings ===")
    print(f"device={device}")
    print(f"output_dir={output_dir}")
    for key in keys:
        print(f"{key}={config[key]}")
    if config["quick"]:
        print("quick_mode=True (smoke test only; not a formal comparison)")


def ensure_config(output_dir, config, resume):
    path = output_dir / "run_config.json"
    if resume and path.exists():
        previous = read_json(path)
        ignored = {"results_root", "resume"}
        left = {key: value for key, value in previous.items() if key not in ignored}
        right = {key: value for key, value in config.items() if key not in ignored}
        if left != right:
            raise RuntimeError(
                "Existing results use a different configuration. Choose a new "
                "--run-name or use the original arguments; artifacts will not be mixed."
            )
    save_json(path, config)


def reference_model(data, args, selection_mode, gate):
    model = GroupedBNNVI(
        X=data["X_train"],
        y=data["y_train"],
        input_dim=6,
        hidden_dims=(args.fit_units,),
        out_dim=1,
        selection_mode=selection_mode,
        architecture_mode="stacked",
        family="gaussian",
        sigma2=args.sigma2,
        init_sd=0.5,
        K_flow=0,
        flow_type="meanfield",
        repu_power=None,
        linear_skip=False,
        **gate,
    ).to(data["X_train"].device)
    return model


def reference_ready(path):
    return all((path / name).exists() for name in (
        "posterior_draws.npz", "summary.json"
    ))


def load_reference(path, data, args, selection_mode, gate):
    model = reference_model(data, args, selection_mode, gate)
    with np.load(path / "posterior_draws.npz") as archive:
        xi_numpy = archive["xi"]
        chain_xi = archive["chain_xi"]
    xi = torch.as_tensor(
        xi_numpy,
        device=data["X_train"].device,
        dtype=data["X_train"].dtype,
    )
    summary = read_json(path / "summary.json")
    return model.decoder, xi, chain_xi, summary


def get_reference(
    path,
    data,
    args,
    selection_mode,
    gate,
    seed,
):
    if args.resume and reference_ready(path):
        print(f"resume MCMC: {path.relative_to(path.parents[3])}")
        decoder, xi, chain_xi, summary = load_reference(
            path, data, args, selection_mode, gate
        )
        if not (path / "diagnostics.json").exists():
            diagnostics = save_mcmc_diagnostics(
                path, decoder, chain_xi, data["X_test"]
            )
            summary["diagnostics"] = diagnostics["summary"]
            save_json(path / "summary.json", summary)
        return decoder, xi, chain_xi, summary

    decoder, xi, result = run_reference(
        data,
        fit_units=args.fit_units,
        sigma2=args.sigma2,
        gate=gate,
        mcmc_n=args.mcmc_n,
        mcmc_burnin=args.mcmc_burnin,
        mcmc_thin=args.mcmc_thin,
        seed=seed,
        print_every=max(1, args.mcmc_n // 10),
        selection_mode=selection_mode,
        input_dim=6,
        n_chains=args.mcmc_chains,
        initial_state="prior",
    )
    summary = save_reference(
        path,
        result,
        decoder=decoder,
        X_predict=data["X_test"],
    )
    return decoder, xi, result["chain_xi_draws"], summary


def vi_model(data, args, selection_mode, gate, coupling_type, seed):
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    model = GroupedBNNVI(
        X=data["X_train"],
        y=data["y_train"],
        input_dim=6,
        hidden_dims=(args.fit_units,),
        out_dim=1,
        selection_mode=selection_mode,
        architecture_mode="stacked",
        family="gaussian",
        sigma2=args.sigma2,
        init_sd=0.5,
        K_flow=args.flow_depth,
        flow_type=f"improved_separate_attention_{coupling_type}",
        conditioner_type="improved_separate_attention",
        coupling_type=coupling_type,
        flow_hidden_units=64,
        flow_hidden_layers=2,
        scale_clip=args.scale_clip,
        flow_token_dim=args.token_dim,
        flow_num_heads=args.attention_heads,
        flow_mask_seed=args.seed + 50 + int(selection_mode == "feature_group"),
        spline_num_bins=args.spline_bins,
        spline_tail_bound=args.spline_tail_bound,
        spline_min_bin_width=args.min_bin_width,
        spline_min_bin_height=args.min_bin_height,
        spline_min_derivative=args.min_derivative,
        repu_power=None,
        linear_skip=False,
        **gate,
    ).to(data["X_train"].device)
    return model


def vi_ready(path):
    return all((path / name).exists() for name in (
        "model_state.pt", "posterior_draws.npz", "summary_matched.json",
        "config.json", "recovery_matched.csv",
    ))


def load_vi(path, data, args, selection_mode, gate, coupling_type, seed):
    model = vi_model(
        data, args, selection_mode, gate, coupling_type, seed
    )
    try:
        state = torch.load(
            path / "model_state.pt",
            map_location=data["X_train"].device,
            weights_only=True,
        )
    except TypeError:
        state = torch.load(
            path / "model_state.pt",
            map_location=data["X_train"].device,
        )
    model.load_state_dict(state)
    model.eval()
    with np.load(path / "posterior_draws.npz") as archive:
        xi = torch.as_tensor(
            archive["xi"],
            device=data["X_train"].device,
            dtype=data["X_train"].dtype,
        )
    return {
        "model": model,
        "final": {
            "xi": xi,
            "summary": read_json(path / "summary_matched.json"),
            "recovery_by_target": pd.read_csv(path / "recovery_matched.csv"),
        },
        "config": read_json(path / "config.json"),
    }


def get_vi(
    path,
    data,
    truth,
    args,
    selection_mode,
    gate,
    coupling_type,
    seed,
    matched_decoder,
    matched_xi,
):
    if args.resume and vi_ready(path):
        print(f"resume VI: {path.relative_to(path.parents[4])}")
        return load_vi(
            path, data, args, selection_mode, gate, coupling_type, seed
        )

    output = train_grouped_bnn(
        data["X_train"],
        data["y_train"],
        data["X_eval"],
        data["signal_eval"],
        X_final=data["X_test"],
        signal_final=data["signal_test"],
        mcmc_decoder=matched_decoder,
        mcmc_xi=matched_xi,
        truth=truth,
        selection_mode=selection_mode,
        input_dim=6,
        H=args.fit_units,
        hidden_dims=(args.fit_units,),
        out_dim=1,
        architecture_mode="stacked",
        family="gaussian",
        sigma2=args.sigma2,
        init_sd=0.5,
        K_flow=args.flow_depth,
        flow_type=f"improved_separate_attention_{coupling_type}",
        conditioner_type="improved_separate_attention",
        coupling_type=coupling_type,
        flow_hidden_units=64,
        flow_hidden_layers=2,
        scale_clip=args.scale_clip,
        flow_token_dim=args.token_dim,
        flow_num_heads=args.attention_heads,
        flow_mask_seed=args.seed + 50 + int(selection_mode == "feature_group"),
        spline_num_bins=args.spline_bins,
        spline_tail_bound=args.spline_tail_bound,
        spline_min_bin_width=args.min_bin_width,
        spline_min_bin_height=args.min_bin_height,
        spline_min_derivative=args.min_derivative,
        spline_inverse_tolerance=args.inverse_tolerance,
        repu_power=None,
        linear_skip=False,
        epochs=args.epochs,
        lr=3e-4,
        R_train=args.r_train,
        R_eval=args.r_eval,
        R_final=args.r_final,
        sampling_timing_repeats=3,
        eval_every=args.eval_every,
        selection_warmup_epochs=args.warmup_epochs,
        init_loc_jitter=0.05,
        endpoint="last",
        grad_clip=5.0,
        min_active_draws=args.min_draws,
        seed=seed,
        **gate,
    )
    save_training(path, output)
    return output


def pip_payload(decoder, xi):
    draws = metric.grouped_selection_draws(decoder, xi)
    pip = draws["pip_draws"].float().mean(dim=0).detach().cpu().numpy()
    expected_count = float(
        decoder.group_semantics(xi)["active"].float().sum(dim=1).mean()
    )
    return pip, expected_count


def save_pip_comparison(path, selection_mode, entries):
    rows = []
    for method, decoder, xi in entries:
        pip, expected_count = pip_payload(decoder, xi)
        labels = (
            [f"unit_rank_{index + 1}" for index in range(len(pip))]
            if selection_mode == "unit_group"
            else [f"x{index}" for index in range(len(pip))]
        )
        rows.extend({
            "method": method,
            "target": label,
            "pip": float(value),
            "expected_active_count": expected_count,
        } for label, value in zip(labels, pip))
    save_frame(path / "pip_comparison.csv", rows)


def summary_row(
    *,
    args,
    selection_mode,
    activation,
    coupling_type,
    output,
    comparison,
    matched_decoder,
    matched_xi,
    hard_decoder,
    hard_xi,
    mcmc_summary,
    attention_summary,
    vi_seed,
):
    base = output["final"]["summary"]
    vi_decoder = output["model"].decoder
    vi_xi = output["final"]["xi"]
    vi_pip, vi_count = pip_payload(vi_decoder, vi_xi)
    mcmc_pip, mcmc_count = pip_payload(matched_decoder, matched_xi)
    hard_pip, hard_count = pip_payload(hard_decoder, hard_xi)
    mcmc_diagnostics = mcmc_summary.get("diagnostics", {})
    threshold_diagnostics = mcmc_diagnostics.get("threshold", {})
    count_diagnostics = mcmc_diagnostics.get("active_count", {})
    row = {
        "selection_mode": selection_mode,
        "activation": activation,
        "gate_parameter_name": (
            "tau_g" if activation == "normalized_requ" else "delta"
        ),
        "gate_parameter": (
            args.gate_tau if activation == "normalized_requ"
            else args.gate_delta
        ),
        "coupling_type": coupling_type,
        "conditioner_type": "improved_separate_attention",
        "n": args.n,
        "p": 6,
        "H_fit": args.fit_units,
        "H_true": 2,
        "p_true": 2,
        "active_features": "x0,x3",
        "flow_depth": args.flow_depth,
        "warmup_epochs": args.warmup_epochs,
        "total_epochs": args.epochs,
        "R_train": args.r_train,
        "posterior_draws": args.r_final,
        "seed": args.seed,
        "vi_seed": vi_seed,
        "true_active_skl": comparison["true_active_skl"],
        "true_active_skl_mean": comparison["true_active_skl_mean"],
        "zero_js": comparison["zero_js"],
        "zero_js_mean": comparison["zero_js_mean"],
        "conditional_joint_skl": comparison["conditional_joint_skl"],
        "n_valid_true_active_skl": comparison["n_valid_true_active_skl"],
        "rat_test_signal_r2": base["rat_signal_r2"],
        "rat_test_signal_mse": base["rat_mse"],
        "mcmc_reference_signal_r2": base["mcmc_signal_r2"],
        "mcmc_reference_mse": base["mcmc_mse"],
        "elbo": base["elbo"],
        "train_time_sec": base["train_time_sec"],
        "sec_per_epoch": base["sec_per_epoch"],
        "posterior_sampling_time_sec": base["posterior_sampling_time_sec"],
        "posterior_sampling_time_iqr_sec": base[
            "posterior_sampling_time_iqr_sec"
        ],
        "trainable_params": base["trainable_params"],
        "flow_trainable_params": base["flow_trainable_params"],
        "mcmc_sampling_time_sec": mcmc_summary["sampling_time_sec"],
        "mcmc_wall_time_sec": mcmc_summary.get(
            "wall_time_sec", mcmc_summary["sampling_time_sec"]
        ),
        "mcmc_threshold_rhat": threshold_diagnostics.get("rhat", np.nan),
        "mcmc_threshold_ess": threshold_diagnostics.get("ess", np.nan),
        "mcmc_threshold_mcse": threshold_diagnostics.get("mcse", np.nan),
        "mcmc_active_count_rhat": count_diagnostics.get("rhat", np.nan),
        "mcmc_active_count_ess": count_diagnostics.get("ess", np.nan),
        "mcmc_active_count_mcse": count_diagnostics.get("mcse", np.nan),
        "mcmc_max_predictive_rhat": mcmc_diagnostics.get(
            "max_predictive_rhat", np.nan
        ),
        "mcmc_min_predictive_ess": mcmc_diagnostics.get(
            "min_predictive_ess", np.nan
        ),
        "reference_type": "matched",
        "reference_compatibility": "exact",
        "spline_num_bins": (
            args.spline_bins if coupling_type == "spline" else np.nan
        ),
        "max_inverse_error": base.get("max_inverse_error"),
        "max_logdet_consistency_error": base.get(
            "max_logdet_consistency_error"
        ),
        "n_nonfinite_forward": base.get("n_nonfinite_forward"),
        "n_nonfinite_inverse": base.get("n_nonfinite_inverse"),
        "n_nonfinite_logdet": base.get("n_nonfinite_logdet"),
        "n_nonfinite_spline_parameters": base.get(
            "n_nonfinite_spline_parameters"
        ),
        "n_nonfinite_gradients": base.get("n_nonfinite_gradients"),
        "attention_shift_head_difference": attention_summary.get(
            "shift_head_mean_abs_difference", np.nan
        ),
        "attention_scale_head_difference": attention_summary.get(
            "scale_head_mean_abs_difference", np.nan
        ),
        "vi_expected_active_count": vi_count,
        "mcmc_expected_active_count": mcmc_count,
        "hard_mcmc_expected_active_count": hard_count,
    }
    for rank in range(5):
        row[f"vi_pip_rank{rank + 1}"] = (
            float(vi_pip[rank]) if selection_mode == "unit_group" else np.nan
        )
        row[f"mcmc_pip_rank{rank + 1}"] = (
            float(mcmc_pip[rank]) if selection_mode == "unit_group" else np.nan
        )
        row[f"hard_mcmc_pip_rank{rank + 1}"] = (
            float(hard_pip[rank]) if selection_mode == "unit_group" else np.nan
        )
    for feature in range(6):
        row[f"vi_pip_x{feature}"] = (
            float(vi_pip[feature]) if selection_mode == "feature_group" else np.nan
        )
        row[f"mcmc_pip_x{feature}"] = (
            float(mcmc_pip[feature]) if selection_mode == "feature_group" else np.nan
        )
        row[f"hard_mcmc_pip_x{feature}"] = (
            float(hard_pip[feature]) if selection_mode == "feature_group" else np.nan
        )
    row["pip_rmse_truth"] = (
        comparison["pip_rmse_truth"]
        if selection_mode == "feature_group" else np.nan
    )
    row["pip_rmse_mcmc"] = (
        comparison["pip_rmse_mcmc"]
        if selection_mode == "feature_group" else np.nan
    )
    return row


def compact_summary(summary):
    columns = {
        "selection_mode": "Selection",
        "activation": "Gate",
        "coupling_type": "Transport",
        "true_active_skl": "Active SKL",
        "conditional_joint_skl": "Joint SKL",
        "zero_js": "Zero JS",
        "rat_test_signal_mse": "VI MSE",
        "mcmc_reference_mse": "MCMC MSE",
        "rat_test_signal_r2": "VI R2",
        "train_time_sec": "Train s",
        "posterior_sampling_time_sec": "Sample s",
        "trainable_params": "Params",
    }
    return summary[list(columns)].rename(columns=columns)


def lower_metric_winner(summary, group, metric_name):
    values = summary.groupby(group)[metric_name].median().dropna()
    if values.empty:
        return "NA", {}
    winner = values.idxmin()
    return winner, values.to_dict()


def automatic_summary(summary):
    lines = ["p=6 bounded transport 自动总结", ""]
    labels = {
        "rat_test_signal_mse": "function MSE",
        "true_active_skl": "active marginal SKL",
        "conditional_joint_skl": "joint SKL",
    }
    for metric_name, label in labels.items():
        winner, values = lower_metric_winner(summary, "activation", metric_name)
        detail = ", ".join(f"{key}={value:.6g}" for key, value in values.items())
        if values:
            lines.append(
                f"- {label}: {winner} 的跨模式/transport中位数更低（{detail}）。"
            )
        else:
            lines.append(f"- {label}: 当前有效draw不足，不能比较。")

    pivot_keys = ["selection_mode", "activation"]
    paired = summary.pivot_table(
        index=pivot_keys,
        columns="coupling_type",
        values=[
            "true_active_skl", "conditional_joint_skl",
            "rat_test_signal_mse", "train_time_sec",
        ],
        aggfunc="first",
    )
    wins = {}
    for metric_name in (
        "true_active_skl", "conditional_joint_skl", "rat_test_signal_mse"
    ):
        delta = paired[(metric_name, "spline")] - paired[(metric_name, "affine")]
        wins[metric_name] = int((delta < 0).sum())
    runtime_ratio = np.median(
        paired[("train_time_sec", "spline")]
        / paired[("train_time_sec", "affine")]
    )
    stable = (
        wins["true_active_skl"] >= 3
        and wins["conditional_joint_skl"] >= 3
    )
    lines.append(
        "- spline稳定性: "
        f"active SKL改善{wins['true_active_skl']}/4，"
        f"joint SKL改善{wins['conditional_joint_skl']}/4，"
        f"function MSE改善{wins['rat_test_signal_mse']}/4；"
        f"训练时间中位比 spline/affine={runtime_ratio:.3f}。"
    )
    lines.append(
        "- spline成本判断: "
        + (
            "posterior density 的改善较一致，可结合上述时间倍率认为值得保留。"
            if stable else
            "posterior density 的改善不够一致，当前额外成本尚不能稳定换来收益。"
        )
    )

    unit = summary[summary["selection_mode"] == "unit_group"]
    lines.append("- Unit ranked PIP（VI | matched MCMC）:")
    for _, row in unit.iterrows():
        vi = ",".join(f"{row[f'vi_pip_rank{i}']:.3f}" for i in range(1, 6))
        mc = ",".join(f"{row[f'mcmc_pip_rank{i}']:.3f}" for i in range(1, 6))
        lines.append(
            f"  {row['activation']}+{row['coupling_type']}: ({vi}) | ({mc})"
        )

    feature = summary[summary["selection_mode"] == "feature_group"]
    active_values = feature[["vi_pip_x0", "vi_pip_x3"]].to_numpy(float)
    inactive_columns = [f"vi_pip_x{i}" for i in (1, 2, 4, 5)]
    inactive_values = feature[inactive_columns].to_numpy(float)
    feature_stable = (
        np.nanmin(active_values) >= 0.5
        and np.nanmax(inactive_values) < 0.5
    )
    lines.append(
        "- Feature selection: "
        f"x0/x3 PIP范围={np.nanmin(active_values):.3f}–"
        f"{np.nanmax(active_values):.3f}；inactive PIP最大值="
        f"{np.nanmax(inactive_values):.3f}；"
        + ("结构选择稳定。" if feature_stable else "尚未达到稳定分离。")
    )

    shift_diff = float(summary["attention_shift_head_difference"].median())
    scale_diff = float(summary["attention_scale_head_difference"].median())
    head_distinct = max(shift_diff, scale_diff) >= 0.01
    lines.append(
        "- Attention heads: 两个head的平均绝对权重差为 "
        f"shift={shift_diff:.6g}, scale/shape={scale_diff:.6g}；"
        + (
            "存在可见的不同interaction pattern。"
            if head_distinct else
            "当前两个head仍较接近，尚无充分分工证据。"
        )
        + "不预设head语义。"
    )

    spline = summary[summary["coupling_type"] == "spline"]
    nonfinite = spline[[
        "n_nonfinite_forward", "n_nonfinite_inverse",
        "n_nonfinite_logdet", "n_nonfinite_gradients",
        "n_nonfinite_spline_parameters",
    ]].fillna(0).to_numpy().sum()
    max_error = float(spline["max_inverse_error"].max())
    lines.append(
        "- Spline numerical check: "
        f"nonfinite总数={int(nonfinite)}，最大inverse error={max_error:.6g}。"
    )
    lines.append(
        "- MCMC diagnostics: "
        f"threshold最大R-hat={summary['mcmc_threshold_rhat'].max():.4f}，"
        f"active-count最大R-hat={summary['mcmc_active_count_rhat'].max():.4f}，"
        f"predictive最大R-hat={summary['mcmc_max_predictive_rhat'].max():.4f}，"
        f"predictive最小ESS={summary['mcmc_min_predictive_ess'].min():.1f}。"
    )

    aggregate = summary.groupby(["activation", "coupling_type"], as_index=False)[[
        "conditional_joint_skl", "true_active_skl",
        "rat_test_signal_mse", "train_time_sec",
    ]].median()
    aggregate = aggregate.dropna(subset=[
        "conditional_joint_skl", "true_active_skl", "rat_test_signal_mse"
    ])
    aggregate = aggregate.sort_values([
        "conditional_joint_skl", "true_active_skl",
        "rat_test_signal_mse", "train_time_sec",
    ])
    if aggregate.empty:
        lines.append(
            "- 进入multi-layer阶段的暂定组合: 当前有效joint density指标不足，"
            "不做选择。"
        )
    else:
        best = aggregate.iloc[0]
        lines.append(
            "- 进入multi-layer阶段的暂定组合: "
            f"{best['activation']} + {best['coupling_type']}。"
            "选择顺序是joint density、active marginal、function recovery、cost，"
            "没有构造composite score，也没有使用unit PIP RMSE选冠军。"
        )
    return "\n".join(lines) + "\n"


def run(args, output_dir):
    validate_settings(args)
    device = choose_device(args.device)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = run_config(args)
    ensure_config(output_dir, config, args.resume)
    print_settings(config, device, output_dir)

    acceptance = run_grouped_acceptance_tests(device=device, dtype=torch.float32)
    save_json(output_dir / "acceptance_tests.json", acceptance)

    data, truth, split, full_data = make_p6_bounded_data(
        n=args.n,
        fit_units=args.fit_units,
        sigma2=args.sigma2,
        seed=args.seed,
        device=device,
    )
    save_dataset(output_dir / "dataset.npz", full_data, split, truth)
    print("teacher_units=", truth["teacher_units"])
    print(f"snr={truth['snr']:.6f}; split_sizes="
          f"{len(split['train'])}/{len(split['eval'])}/{len(split['test'])}")

    gate_specs = gates(args)
    references = {}
    for mode_index, selection_mode in enumerate(("unit_group", "feature_group")):
        short_mode = "unit" if selection_mode == "unit_group" else "feature"
        reference_gates = {"hard": hard_gate(args), **gate_specs}
        for gate_index, (name, gate) in enumerate(reference_gates.items()):
            print(f"\n=== MCMC {short_mode} / {name} ===")
            path = output_dir / "mcmc" / short_mode / name
            references[(selection_mode, name)] = get_reference(
                path,
                data,
                args,
                selection_mode,
                gate,
                seed=args.seed + 1000 + 100 * mode_index + 10 * gate_index,
            )

    outputs = {}
    comparisons = {}
    attention_summaries = {}
    summary_rows = []
    for mode_index, selection_mode in enumerate(("unit_group", "feature_group")):
        short_mode = "unit" if selection_mode == "unit_group" else "feature"
        hard_decoder, hard_xi, _, _ = references[(selection_mode, "hard")]
        for gate_index, (activation, gate) in enumerate(gate_specs.items()):
            matched_decoder, matched_xi, _, mcmc_summary = references[
                (selection_mode, activation)
            ]
            for coupling_type in ("affine", "spline"):
                key = (selection_mode, activation, coupling_type)
                vi_seed = args.seed + 2000 + 100 * mode_index + 10 * gate_index
                print(
                    f"\n=== VI {short_mode} / {activation} / {coupling_type} ==="
                )
                vi_dir = (
                    output_dir / "vi" / short_mode / activation / coupling_type
                )
                output = get_vi(
                    vi_dir,
                    data,
                    truth,
                    args,
                    selection_mode,
                    gate,
                    coupling_type,
                    vi_seed,
                    matched_decoder,
                    matched_xi,
                )
                outputs[key] = output
                vi_decoder = output["model"].decoder
                vi_xi = output["final"]["xi"]
                if vi_decoder.compatibility_signature() != (
                    matched_decoder.compatibility_signature()
                ):
                    raise RuntimeError("VI and matched MCMC decoder mismatch.")

                comparison, table = comparison_metrics(
                    vi_decoder,
                    vi_xi,
                    matched_decoder,
                    matched_xi,
                    truth,
                    min_draws=args.min_draws,
                    compatibility="exact",
                )
                comparisons[key] = comparison
                save_json(vi_dir / "comparison_matched.json", comparison)
                save_frame(vi_dir / "recovery_matched.csv", table)
                hard_comparison, hard_table = comparison_metrics(
                    vi_decoder,
                    vi_xi,
                    hard_decoder,
                    hard_xi,
                    truth,
                    min_draws=args.min_draws,
                    compatibility="structural",
                )
                save_json(vi_dir / "comparison_hard.json", hard_comparison)
                save_frame(vi_dir / "recovery_hard.csv", hard_table)
                save_pip_comparison(vi_dir, selection_mode, (
                    ("VI", vi_decoder, vi_xi),
                    ("matched_MCMC", matched_decoder, matched_xi),
                    ("hard_MCMC", hard_decoder, hard_xi),
                ))
                attention = save_attention_diagnostics(
                    vi_dir,
                    output["model"],
                    vi_xi,
                    max_draws=args.attention_draws,
                )
                attention_summaries[key] = attention
                summary_rows.append(summary_row(
                    args=args,
                    selection_mode=selection_mode,
                    activation=activation,
                    coupling_type=coupling_type,
                    output=output,
                    comparison=comparison,
                    matched_decoder=matched_decoder,
                    matched_xi=matched_xi,
                    hard_decoder=hard_decoder,
                    hard_xi=hard_xi,
                    mcmc_summary=mcmc_summary,
                    attention_summary=attention,
                    vi_seed=vi_seed,
                ))

    summary = pd.DataFrame(summary_rows).sort_values([
        "selection_mode", "activation", "coupling_type"
    ]).reset_index(drop=True)
    save_frame(output_dir / "summary.csv", summary)
    compact = compact_summary(summary)
    save_frame(output_dir / "summary_compact.csv", compact)

    plot_manifest = {}
    for selection_mode in ("unit_group", "feature_group"):
        short_mode = "unit" if selection_mode == "unit_group" else "feature"
        draw_sets = []
        for activation in gate_specs:
            matched_decoder, matched_xi, _, _ = references[
                (selection_mode, activation)
            ]
            draw_sets.append((matched_decoder, matched_xi))
            for coupling_type in ("affine", "spline"):
                output = outputs[(selection_mode, activation, coupling_type)]
                draw_sets.append((output["model"].decoder, output["final"]["xi"]))
        axis_limits = common_joint_axis(draw_sets, truth)
        density_draws = common_joint_draw_count(
            draw_sets,
            truth,
            min_draws=args.min_draws,
            max_draws=args.density_max_draws,
        )
        for activation in gate_specs:
            matched_decoder, matched_xi, _, _ = references[
                (selection_mode, activation)
            ]
            for coupling_type in ("affine", "spline"):
                key = (selection_mode, activation, coupling_type)
                output = outputs[key]
                name = f"{short_mode}_{activation}_{coupling_type}_vs_matched"
                plot_manifest[name] = save_joint_plot(
                    output_dir / "plots" / name,
                    vi_decoder=output["model"].decoder,
                    vi_xi=output["final"]["xi"],
                    reference_decoder=matched_decoder,
                    reference_xi=matched_xi,
                    truth=truth,
                    min_draws=args.min_draws,
                    axis_limits=axis_limits,
                    vi_label=f"VI {activation} {coupling_type}",
                    reference_label=f"MCMC {activation}",
                    title=name.replace("_", " "),
                    max_draws=density_draws,
                    random_seed=args.seed,
                )
        plot_manifest[f"{short_mode}_shared_axis_limits"] = axis_limits
        plot_manifest[f"{short_mode}_shared_density_draws"] = density_draws
    save_json(output_dir / "plot_manifest.json", plot_manifest)

    narrative = automatic_summary(summary)
    with (output_dir / "automatic_summary.txt").open("w", encoding="utf-8") as stream:
        stream.write(narrative)
    print("\n=== compact summary ===")
    print(compact.to_string(index=False))
    print("\n=== automatic summary ===")
    print(narrative)


def main():
    args = parse_args()
    quicken(args)
    output_dir = args.results_root / args.run_name
    with Tee(output_dir / "run.log"):
        run(args, output_dir)


if __name__ == "__main__":
    main()
