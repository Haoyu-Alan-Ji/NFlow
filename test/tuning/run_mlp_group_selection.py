#!/usr/bin/env python3
"""Matched-MCMC study of sparse stacked and embed-output MLPs."""

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
    save_dataset,
    save_frame,
    save_joint_plot,
    save_json,
    save_mcmc_diagnostics,
    save_mlp_structure_outputs,
    save_reference,
    save_training,
)
from Python.bnn_mcmc import run_bnn_mcmc_chains
from Python.model2 import GroupedBNNVI, run_grouped_acceptance_tests


SELECTION_MODES = (
    "feature_group",
    "feature_unit_induced_edge",
    "edge_group",
)
ARCHITECTURES = ("stacked", "embed_output")


def parse_dims(value):
    dims = tuple(int(part.strip()) for part in str(value).split(","))
    if not dims or any(width < 1 for width in dims):
        raise argparse.ArgumentTypeError("hidden dimensions must be positive")
    return dims


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument("--run-name", default="mlp_group_selection")
    parser.add_argument("--stage", choices=("1", "2", "3", "all"), default="1")
    parser.add_argument(
        "--include-embed-smooth-step",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also run the optional three embed-output Stage-2 ablations.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n", type=int, default=240)
    parser.add_argument("--hidden-dims", type=parse_dims, default=(5, 5))
    parser.add_argument("--deep-hidden-dims", type=parse_dims, default=(6, 5, 4))
    parser.add_argument("--embedding-dim", type=int, default=6)
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
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse complete MCMC and VI artifacts from identical configurations.",
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
    args.mcmc_chains = 2
    args.mcmc_n = 40
    args.mcmc_burnin = 10
    args.min_draws = 3
    args.density_max_draws = 32
    if not args.run_name.endswith("_quick"):
        args.run_name += "_quick"


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def gate_config(args, gate_type):
    if gate_type == "normalized_requ":
        return {
            "gate_type": "normalized_requ",
            "gate_power": 2.0,
            "gate_tau": float(args.gate_tau),
            "gate_delta": float(args.gate_delta),
        }
    if gate_type == "smooth_step":
        return {
            "gate_type": "smooth_step",
            "gate_power": 1.0,
            "gate_tau": None,
            "gate_delta": float(args.gate_delta),
        }
    raise ValueError(f"Unsupported gate: {gate_type}")


def experiment_matrix(args):
    requested = {1, 2, 3} if args.stage == "all" else {int(args.stage)}
    rows = []

    def add(stage, architectures, modes, gate, hidden_dims):
        for architecture in architectures:
            for selection_mode in modes:
                config_id = (
                    f"stage{stage}__{architecture}__{selection_mode}__{gate}"
                    f"__h{'x'.join(map(str, hidden_dims))}"
                )
                rows.append({
                    "config_id": config_id,
                    "stage": stage,
                    "architecture": architecture,
                    "selection_mode": selection_mode,
                    "gate_type": gate,
                    "hidden_dims": tuple(hidden_dims),
                    "embedding_dim": (
                        int(args.embedding_dim)
                        if architecture == "embed_output" else None
                    ),
                })

    if 1 in requested:
        add(1, ARCHITECTURES, SELECTION_MODES, "normalized_requ", args.hidden_dims)
    if 2 in requested:
        architectures = list(ARCHITECTURES) if args.include_embed_smooth_step else ["stacked"]
        add(2, architectures, SELECTION_MODES, "smooth_step", args.hidden_dims)
    if 3 in requested:
        add(
            3,
            ARCHITECTURES,
            ("feature_unit_induced_edge", "edge_group"),
            "normalized_requ",
            args.deep_hidden_dims,
        )

    offsets = {
        "stacked": 1000,
        "embed_output": 2000,
        "feature_group": 100,
        "feature_unit_induced_edge": 200,
        "edge_group": 300,
        "normalized_requ": 10,
        "smooth_step": 20,
    }
    for row in rows:
        row["seed"] = int(
            args.seed
            + row["stage"] * 10000
            + offsets[row["architecture"]]
            + offsets[row["selection_mode"]]
            + offsets[row["gate_type"]]
        )
    return rows


def root_config(args):
    return {
        "implementation_version": "mlp_group_selection_v1",
        "requested_stage": args.stage,
        "include_embed_smooth_step": bool(args.include_embed_smooth_step),
        "results_root": str(args.results_root),
        "run_name": args.run_name,
        "device_request": args.device,
        "resume": bool(args.resume),
        "quick": bool(args.quick),
        "n": int(args.n),
        "p": 6,
        "active_features": [0, 3],
        "sigma2": float(args.sigma2),
        "target_signal_sd": 1.5,
        "seed": int(args.seed),
        "split_rule": "seed+1000 permutation; 60/20/20",
        "hidden_dims": list(args.hidden_dims),
        "deep_hidden_dims": list(args.deep_hidden_dims),
        "embedding_dim": int(args.embedding_dim),
        "gate_tau": float(args.gate_tau),
        "gate_delta": float(args.gate_delta),
        "hidden_activation": "relu",
        "flow_depth": int(args.flow_depth),
        "coupling_type": "spline",
        "conditioner_type": "improved_separate_attention",
        "attention_heads": int(args.attention_heads),
        "spline_bins": int(args.spline_bins),
        "epochs": int(args.epochs),
        "warmup_epochs": int(args.warmup_epochs),
        "R_train": int(args.r_train),
        "R_eval": int(args.r_eval),
        "posterior_draws": int(args.r_final),
        "mcmc_chains": int(args.mcmc_chains),
        "mcmc_n": int(args.mcmc_n),
        "mcmc_burnin": int(args.mcmc_burnin),
        "mcmc_thin": int(args.mcmc_thin),
        "learning_rate": 3e-4,
        "prior": "standard_normal_all_latents",
    }


def validate_settings(args):
    if args.quick:
        return
    required = {
        "n": (args.n, 240),
        "hidden_dims": (tuple(args.hidden_dims), (5, 5)),
        "deep_hidden_dims": (tuple(args.deep_hidden_dims), (6, 5, 4)),
        "embedding_dim": (args.embedding_dim, 6),
        "flow_depth": (args.flow_depth, 4),
        "attention_heads": (args.attention_heads, 2),
        "spline_bins": (args.spline_bins, 8),
        "gate_tau": (args.gate_tau, 1.0),
        "gate_delta": (args.gate_delta, 1.0),
        "warmup_epochs": (args.warmup_epochs, 500),
        "epochs": (args.epochs, 2000),
        "r_train": (args.r_train, 100),
        "mcmc_chains": (args.mcmc_chains, 4),
        "mcmc_n": (args.mcmc_n, 6000),
        "mcmc_burnin": (args.mcmc_burnin, 1000),
    }
    failures = [
        f"{name}={actual} (required {expected})"
        for name, (actual, expected) in required.items()
        if actual != expected
    ]
    if failures:
        raise ValueError(
            "Formal benchmark settings are fixed: " + "; ".join(failures)
        )


def ensure_root_config(path, config):
    path = Path(path)
    ignored = {
        "requested_stage",
        "include_embed_smooth_step",
        "results_root",
        "resume",
    }
    if path.exists():
        previous = read_json(path)
        left = {key: value for key, value in previous.items() if key not in ignored}
        right = {key: value for key, value in config.items() if key not in ignored}
        if left != right:
            raise RuntimeError(
                "Existing results use different fixed settings. Choose a new "
                "--run-name or restore the original settings."
            )
    save_json(path, config)


def ensure_configuration(path, config):
    path = Path(path)
    normalized = json.loads(json.dumps(config))
    if path.exists() and read_json(path) != normalized:
        raise RuntimeError(f"Configuration mismatch in {path.parent}")
    save_json(path, normalized)


def build_model(data, args, config, *, variational):
    gate = gate_config(args, config["gate_type"])
    common = dict(
        X=data["X_train"],
        y=data["y_train"],
        input_dim=6,
        hidden_dims=config["hidden_dims"],
        out_dim=1,
        selection_mode=config["selection_mode"],
        architecture_mode=config["architecture"],
        embedding_dim=config["embedding_dim"],
        family="gaussian",
        sigma2=args.sigma2,
        init_sd=0.5,
        repu_power=None,
        linear_skip=False,
        **gate,
    )
    if not variational:
        return GroupedBNNVI(
            **common, K_flow=0, flow_type="meanfield"
        ).to(data["X_train"].device)
    return GroupedBNNVI(
        **common,
        K_flow=args.flow_depth,
        flow_type="improved_separate_attention_spline",
        conditioner_type="improved_separate_attention",
        coupling_type="spline",
        scale_clip=args.scale_clip,
        flow_token_dim=args.token_dim,
        flow_num_heads=args.attention_heads,
        flow_mask_seed=config["seed"] + 37,
        spline_num_bins=args.spline_bins,
        spline_tail_bound=args.spline_tail_bound,
        spline_min_bin_width=args.min_bin_width,
        spline_min_bin_height=args.min_bin_height,
        spline_min_derivative=args.min_derivative,
    ).to(data["X_train"].device)


def reference_ready(path):
    return all((path / name).exists() for name in (
        "posterior_draws.npz", "summary.json"
    ))


def get_reference(path, data, args, config):
    model = build_model(data, args, config, variational=False)
    if args.resume and reference_ready(path):
        print(f"resume MCMC: {config['config_id']}")
        with np.load(path / "posterior_draws.npz") as archive:
            xi_np = archive["xi"]
            chain_xi = archive["chain_xi"]
        xi = torch.as_tensor(
            xi_np,
            device=data["X_train"].device,
            dtype=data["X_train"].dtype,
        )
        summary = read_json(path / "summary.json")
        if not (path / "diagnostics.json").exists():
            diagnostics = save_mcmc_diagnostics(
                path, model.decoder, chain_xi, data["X_test"]
            )
            summary["diagnostics"] = diagnostics["summary"]
            save_json(path / "summary.json", summary)
        return model.decoder, xi, chain_xi, summary

    result = run_bnn_mcmc_chains(
        model,
        n_chains=args.mcmc_chains,
        N=args.mcmc_n,
        burnin=args.mcmc_burnin,
        thin=args.mcmc_thin,
        seed=config["seed"],
        print_every=max(1, args.mcmc_n // 10),
        initial_state="prior",
    )
    summary = save_reference(
        path,
        result,
        decoder=model.decoder,
        X_predict=data["X_test"],
    )
    xi = torch.as_tensor(
        result["xi_draws"],
        device=data["X_train"].device,
        dtype=data["X_train"].dtype,
    )
    return model.decoder, xi, result["chain_xi_draws"], summary


def vi_ready(path):
    return all((path / name).exists() for name in (
        "model_state.pt",
        "posterior_draws.npz",
        "summary_matched.json",
        "config.json",
    ))


def load_vi(path, data, args, config):
    model = build_model(data, args, config, variational=True)
    try:
        state = torch.load(
            path / "model_state.pt",
            map_location=data["X_train"].device,
            weights_only=True,
        )
    except TypeError:
        state = torch.load(
            path / "model_state.pt", map_location=data["X_train"].device
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
        },
        "config": read_json(path / "config.json"),
    }


def get_vi(path, data, truth, args, config, mcmc_decoder, mcmc_xi):
    if args.resume and vi_ready(path):
        print(f"resume VI: {config['config_id']}")
        return load_vi(path, data, args, config)

    gate = gate_config(args, config["gate_type"])
    output = train_grouped_bnn(
        data["X_train"],
        data["y_train"],
        data["X_eval"],
        data["signal_eval"],
        X_final=data["X_test"],
        signal_final=data["signal_test"],
        mcmc_decoder=mcmc_decoder,
        mcmc_xi=mcmc_xi,
        truth=truth,
        selection_mode=config["selection_mode"],
        input_dim=6,
        H=None,
        hidden_dims=config["hidden_dims"],
        out_dim=1,
        architecture_mode=config["architecture"],
        embedding_dim=config["embedding_dim"],
        family="gaussian",
        sigma2=args.sigma2,
        init_sd=0.5,
        K_flow=args.flow_depth,
        flow_type="improved_separate_attention_spline",
        conditioner_type="improved_separate_attention",
        coupling_type="spline",
        scale_clip=args.scale_clip,
        flow_token_dim=args.token_dim,
        flow_num_heads=args.attention_heads,
        flow_mask_seed=config["seed"] + 37,
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
        recovery_eval_max_mcmc_draws=min(2000, int(mcmc_xi.shape[0])),
        seed=config["seed"] + 1,
        **gate,
    )
    save_training(path, output)
    return output


def reference_reliability(summary):
    diagnostics = summary.get("diagnostics", {})
    if not diagnostics:
        return False
    values = (
        diagnostics.get("max_threshold_rhat"),
        diagnostics.get("min_threshold_ess"),
        diagnostics.get("max_predictive_rhat"),
        diagnostics.get("min_predictive_ess"),
    )
    if any(value is None or not np.isfinite(float(value)) for value in values):
        return False
    return bool(
        float(values[0]) <= 1.05
        and float(values[1]) >= 100.0
        and float(values[2]) <= 1.05
        and float(values[3]) >= 100.0
    )


def curve_plot_name(config):
    mode = {
        "feature_unit_induced_edge": "induced_edge",
        "edge_group": "edge_group",
    }[config["selection_mode"]]
    if config["stage"] == 1 and config["gate_type"] == "normalized_requ":
        return f"{config['architecture']}_{mode}_mse_vs_density"
    return f"{config['config_id']}__mse_vs_density"


def save_root_curve_plot(plot_dir, name, curve, title):
    vi_curve = curve[curve["method"] == "VI"]
    mcmc_curve = curve[curve["method"] == "MCMC"]
    fig, _ = metric.plot_mse_edge_density_curve(
        vi_curve, mcmc_curve, title=title
    )
    fig.savefig(plot_dir / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(plot_dir / f"{name}.pdf", bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def summary_row(config, args, output, comparison, structure, mcmc_summary):
    base = output["final"]["summary"]
    gate_parameter = (
        args.gate_tau
        if config["gate_type"] == "normalized_requ" else args.gate_delta
    )
    reliable = reference_reliability(mcmc_summary)
    return {
        "config_id": config["config_id"],
        "stage": int(config["stage"]),
        "architecture": config["architecture"],
        "hidden_dims": "x".join(map(str, config["hidden_dims"])),
        "embedding_dim": config["embedding_dim"],
        "selection_mode": config["selection_mode"],
        "gate_type": config["gate_type"],
        "gate_parameter": float(gate_parameter),
        "threshold_roles": ";".join(output["model"].decoder.threshold_roles),
        "coupling_type": "spline",
        "conditioner_type": "improved_separate_attention",
        "flow_depth": int(args.flow_depth),
        "attention_heads": int(args.attention_heads),
        "spline_bins": int(args.spline_bins),
        "n": int(args.n),
        "p": 6,
        "active_features": "0;3",
        "seed": int(config["seed"]),
        "vi_signal_mse": float(base["rat_mse"]),
        "mcmc_signal_mse": float(base["mcmc_mse"]),
        "active_skl": comparison["active_skl"],
        "zero_js": comparison["zero_js"],
        "joint_skl": comparison["conditional_joint_skl"],
        "structure_density_definition": {
            "feature_group": "feature_density",
            "feature_unit_induced_edge": "induced_edge_density",
            "edge_group": "independent_edge_density",
        }[config["selection_mode"]],
        "vi_expected_structure_density": structure[
            "vi_expected_structure_density"
        ],
        "mcmc_expected_structure_density": structure[
            "mcmc_expected_structure_density"
        ],
        "n_candidate_edges": int(output["model"].decoder.n_candidate_edges),
        "train_time_sec": float(base["train_time_sec"]),
        "mcmc_time_sec": float(mcmc_summary["sampling_time_sec"]),
        "mcmc_reference_reliable": reliable,
        "scientific_interpretation_allowed": reliable,
    }


def run(args, output_dir):
    validate_settings(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    config_root = output_dir / "configurations"
    config_root.mkdir(parents=True, exist_ok=True)

    config = root_config(args)
    ensure_root_config(output_dir / "run_config.json", config)
    matrix = experiment_matrix(args)
    print("=== MLP grouped-selection experiment ===")
    print(f"device={choose_device(args.device)}")
    print(f"output_dir={output_dir}")
    print(f"requested_stage={args.stage}")
    print(f"n_configurations={len(matrix)}")
    print(f"hidden_dims={args.hidden_dims}")
    print(f"deep_hidden_dims={args.deep_hidden_dims}")
    print(f"selection_modes={SELECTION_MODES}")
    print("coupling=spline; conditioner=improved_separate_attention")
    if args.quick:
        print("quick_mode=True (smoke test only; not a scientific comparison)")

    device = choose_device(args.device)
    acceptance = run_grouped_acceptance_tests(
        device=device, dtype=torch.float32
    )
    save_json(output_dir / "acceptance_tests.json", acceptance)

    data, truth, split, full_data = make_p6_bounded_data(
        n=args.n,
        fit_units=max(args.hidden_dims),
        sigma2=args.sigma2,
        seed=args.seed,
        device=device,
    )
    save_dataset(output_dir / "dataset.npz", full_data, split, truth)

    summary_rows = []
    manifest_path = output_dir / "plot_manifest.json"
    plot_manifest = (
        read_json(manifest_path)
        if args.resume and manifest_path.exists() else {}
    )
    for index, item in enumerate(matrix, start=1):
        item = dict(item)
        item.update({
            "n": int(args.n),
            "p": 6,
            "active_features": [0, 3],
            "sigma2": float(args.sigma2),
            "coupling_type": "spline",
            "conditioner_type": "improved_separate_attention",
            "flow_depth": int(args.flow_depth),
            "attention_heads": int(args.attention_heads),
            "spline_bins": int(args.spline_bins),
            "gate_parameter": (
                float(args.gate_tau)
                if item["gate_type"] == "normalized_requ"
                else float(args.gate_delta)
            ),
        })
        config_dir = config_root / item["config_id"]
        ensure_configuration(config_dir / "configuration.json", item)
        print(f"\n=== [{index}/{len(matrix)}] {item['config_id']} ===")

        mcmc_decoder, mcmc_xi, _, mcmc_summary = get_reference(
            config_dir / "mcmc", data, args, item
        )
        output = get_vi(
            config_dir / "vi",
            data,
            truth,
            args,
            item,
            mcmc_decoder,
            mcmc_xi,
        )
        vi_decoder = output["model"].decoder
        vi_xi = output["final"]["xi"]

        density_draws = common_joint_draw_count(
            [(vi_decoder, vi_xi), (mcmc_decoder, mcmc_xi)],
            truth,
            min_draws=args.min_draws,
            max_draws=args.density_max_draws,
        )
        comparison, recovery_table = comparison_metrics(
            vi_decoder,
            vi_xi,
            mcmc_decoder,
            mcmc_xi,
            truth,
            min_draws=args.min_draws,
            compatibility="exact",
            max_joint_draws=density_draws,
            joint_random_seed=item["seed"] + 71,
        )
        save_frame(config_dir / "posterior_discrepancy.csv", recovery_table)
        save_json(config_dir / "posterior_discrepancy.json", comparison)

        axis = common_joint_axis(
            [(vi_decoder, vi_xi), (mcmc_decoder, mcmc_xi)], truth
        )
        density_name = f"{item['config_id']}__matched_density"
        density_ok = save_joint_plot(
            plot_dir / density_name,
            vi_decoder=vi_decoder,
            vi_xi=vi_xi,
            reference_decoder=mcmc_decoder,
            reference_xi=mcmc_xi,
            truth=truth,
            min_draws=args.min_draws,
            axis_limits=axis,
            vi_label="VI",
            reference_label="MCMC",
            title=f"{item['architecture']} | {item['selection_mode']}",
            max_draws=density_draws,
            random_seed=item["seed"] + 71,
        )
        plot_manifest[density_name] = {
            "created": bool(density_ok),
            "joint_skl": comparison["conditional_joint_skl"],
            "draws_per_method": density_draws,
        }

        structure = save_mlp_structure_outputs(
            config_dir / "structure",
            vi_decoder=vi_decoder,
            vi_xi=vi_xi,
            mcmc_decoder=mcmc_decoder,
            mcmc_xi=mcmc_xi,
            X=data["X_test"],
            signal=data["signal_test"],
        )
        if not structure["curve"].empty:
            curve_name = curve_plot_name(item)
            save_root_curve_plot(
                plot_dir,
                curve_name,
                structure["curve"],
                f"{item['architecture']} | {item['selection_mode']}",
            )
            plot_manifest[curve_name] = {
                "created": True,
                "median_probability_marked": True,
            }

        row = summary_row(
            item, args, output, comparison, structure, mcmc_summary
        )
        summary_rows.append(row)
        save_json(config_dir / "scientific_summary.json", row)
        print(pd.DataFrame([row]).to_string(index=False))

    new_summary = pd.DataFrame(summary_rows)
    summary_path = output_dir / "summary.csv"
    if args.resume and summary_path.exists():
        previous = pd.read_csv(summary_path)
        previous = previous[
            ~previous["config_id"].isin(new_summary["config_id"])
        ]
        new_summary = pd.concat([previous, new_summary], ignore_index=True)
    new_summary = new_summary.sort_values(
        ["stage", "architecture", "selection_mode", "gate_type"]
    ).reset_index(drop=True)
    save_frame(summary_path, new_summary)
    save_json(manifest_path, plot_manifest)
    print("\n=== current scientific summary ===")
    print(new_summary.to_string(index=False))


def main():
    args = parse_args()
    quicken(args)
    output_dir = args.results_root / args.run_name
    with Tee(output_dir / "run.log"):
        run(args, output_dir)


if __name__ == "__main__":
    main()
