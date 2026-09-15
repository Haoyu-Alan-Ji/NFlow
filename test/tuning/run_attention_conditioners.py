#!/usr/bin/env python3
"""Task 2: full, separate, and shared attention conditioner comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Python.bnn_train import train_grouped_bnn
from Python.experiment_utils import (
    Tee,
    choose_device,
    common_joint_axis,
    comparison_metrics,
    make_one_layer_unit_data,
    run_reference,
    save_dataset,
    save_frame,
    save_joint_plot,
    save_json,
    save_reference,
    save_training,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument("--run-name", default="attention_conditioners")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n", type=int, default=160)
    parser.add_argument("--fit-units", type=int, default=5)
    parser.add_argument("--sigma2", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--warmup-epochs", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--r-train", type=int, default=64)
    parser.add_argument("--r-eval", type=int, default=1000)
    parser.add_argument("--r-final", type=int, default=5000)
    parser.add_argument("--mcmc-n", type=int, default=6000)
    parser.add_argument("--mcmc-burnin", type=int, default=1000)
    parser.add_argument("--mcmc-thin", type=int, default=1)
    parser.add_argument("--flow-depth", type=int, default=8)
    parser.add_argument("--token-dim", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--min-draws", type=int, default=50)
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
    args.flow_depth = 2
    args.token_dim = 8
    args.num_heads = 2
    args.min_draws = 3


def run(args, output_dir):
    device = choose_device(args.device)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device} output={output_dir}")
    save_json(output_dir / "run_config.json", vars(args))

    data, truth, split, full_data = make_one_layer_unit_data(
        n=args.n,
        fit_units=args.fit_units,
        sigma2=args.sigma2,
        seed=args.seed,
        device=device,
    )
    save_dataset(output_dir / "dataset.npz", full_data, split, truth)

    # The decoder is fixed to the consolidated ReLU-indicator-squared
    # relaxation: G(m)=(m_+)^2, i.e. an unnormalized ReQU gate.
    gate = {
        "gate_type": "requ",
        "gate_power": 2.0,
        "gate_tau": None,
        "gate_delta": 1.0,
    }
    print("\n=== shared matched ReQU MCMC ===")
    mcmc_decoder, mcmc_xi, mcmc_result = run_reference(
        data,
        fit_units=args.fit_units,
        sigma2=args.sigma2,
        gate=gate,
        mcmc_n=args.mcmc_n,
        mcmc_burnin=args.mcmc_burnin,
        mcmc_thin=args.mcmc_thin,
        seed=args.seed + 300,
        print_every=max(1, args.mcmc_n // 10),
    )
    mcmc_summary = save_reference(output_dir / "mcmc" / "requ", mcmc_result)

    conditioners = {
        "full_attention": "full_attention_affine",
        "separate_attention": "separate_attention_affine",
        "shared_attention": "shared_attention_affine",
    }
    outputs = {}
    summary_rows = []
    for name, flow_type in conditioners.items():
        print(f"\n=== VI conditioner: {name} ===")
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
            selection_mode="unit_group",
            input_dim=1,
            H=args.fit_units,
            hidden_dims=(args.fit_units,),
            architecture_mode="stacked",
            family="gaussian",
            sigma2=args.sigma2,
            K_flow=args.flow_depth,
            flow_type=flow_type,
            scale_clip=1.5,
            flow_token_dim=args.token_dim,
            flow_num_heads=args.num_heads,
            flow_mask_seed=args.seed + 50,
            **gate,
            epochs=args.epochs,
            selection_warmup_epochs=args.warmup_epochs,
            eval_every=args.eval_every,
            R_train=args.r_train,
            R_eval=args.r_eval,
            R_final=args.r_final,
            endpoint="last",
            min_active_draws=args.min_draws,
            seed=args.seed,
        )
        outputs[name] = output
        vi_dir = output_dir / "vi" / name
        save_training(vi_dir, output)
        comparison, table = comparison_metrics(
            output["model"].decoder,
            output["final"]["xi"],
            mcmc_decoder,
            mcmc_xi,
            truth,
            min_draws=args.min_draws,
            compatibility="exact",
        )
        save_json(vi_dir / "comparison_mcmc.json", comparison)
        save_frame(vi_dir / "recovery_mcmc.csv", table)
        base = output["final"]["summary"]
        summary_rows.append({
            "conditioner": name,
            "flow_type": flow_type,
            **comparison,
            "rat_test_signal_r2": base["rat_signal_r2"],
            "rat_test_signal_mse": base["rat_mse"],
            "mcmc_reference_signal_r2": base["mcmc_signal_r2"],
            "mcmc_reference_mse": base["mcmc_mse"],
            "elbo": base["elbo"],
            "train_time_sec": base["train_time_sec"],
            "sec_per_epoch": base["sec_per_epoch"],
            "posterior_sampling_time_sec": base[
                "posterior_sampling_time_sec"
            ],
            "posterior_sampling_time_iqr_sec": base[
                "posterior_sampling_time_iqr_sec"
            ],
            "trainable_params": base["trainable_params"],
            "flow_trainable_params": base["flow_trainable_params"],
            "gpu_peak_memory_bytes": base["gpu_peak_memory_bytes"],
            "mcmc_sampling_time_sec": mcmc_summary["sampling_time_sec"],
        })

    summary = pd.DataFrame(summary_rows)
    save_frame(output_dir / "summary.csv", summary)
    draw_sets = [(mcmc_decoder, mcmc_xi)] + [
        (output["model"].decoder, output["final"]["xi"])
        for output in outputs.values()
    ]
    axis_limits = common_joint_axis(draw_sets, truth)
    plot_manifest = {}
    for name, output in outputs.items():
        plot_manifest[name] = save_joint_plot(
            output_dir / "plots" / f"{name}_vs_mcmc",
            vi_decoder=output["model"].decoder,
            vi_xi=output["final"]["xi"],
            reference_decoder=mcmc_decoder,
            reference_xi=mcmc_xi,
            truth=truth,
            min_draws=args.min_draws,
            axis_limits=axis_limits,
            vi_label=f"VI {name}",
            reference_label="MCMC ReQU",
            title=f"{name.replace('_', ' ')} vs matched MCMC",
        )
    save_json(output_dir / "plot_manifest.json", {
        "shared_axis_limits": axis_limits,
        "plots_created": plot_manifest,
    })
    print("\n=== completed ===")
    print(summary.to_string(index=False))


def main():
    args = parse_args()
    quicken(args)
    output_dir = args.results_root / args.run_name
    with Tee(output_dir / "run.log"):
        run(args, output_dir)


if __name__ == "__main__":
    main()
