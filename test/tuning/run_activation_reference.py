#!/usr/bin/env python3
"""Task 1: three relaxed gates against hard and matched MCMC references."""

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
from Python import metric
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
    parser.add_argument("--run-name", default="activation_reference")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n", type=int, default=160)
    parser.add_argument("--fit-units", type=int, default=5)
    parser.add_argument("--sigma2", type=float, default=1.0)
    parser.add_argument("--gate-tau", type=float, default=1.0)
    parser.add_argument("--gate-delta", type=float, default=1.0)
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

    gates = {
        "relu": {
            "gate_type": "relu",
            "gate_power": 1.0,
            "gate_tau": None,
            "gate_delta": args.gate_delta,
        },
        "normalized_requ": {
            "gate_type": "normalized_requ",
            "gate_power": 2.0,
            "gate_tau": args.gate_tau,
            "gate_delta": args.gate_delta,
        },
        "smooth_step": {
            "gate_type": "smooth_step",
            "gate_power": 1.0,
            "gate_tau": None,
            "gate_delta": args.gate_delta,
        },
    }
    hard_gate = {
        "gate_type": "hard",
        "gate_power": 1.0,
        "gate_tau": None,
        "gate_delta": args.gate_delta,
    }
    print_every = max(1, args.mcmc_n // 10)

    print("\n=== shared hard-gate MCMC ===")
    hard_decoder, hard_xi, hard_result = run_reference(
        data,
        fit_units=args.fit_units,
        sigma2=args.sigma2,
        gate=hard_gate,
        mcmc_n=args.mcmc_n,
        mcmc_burnin=args.mcmc_burnin,
        mcmc_thin=args.mcmc_thin,
        seed=args.seed + 200,
        print_every=print_every,
    )
    reference_summaries = {
        "hard": save_reference(output_dir / "mcmc" / "hard", hard_result)
    }
    hard_function = metric.function_recovery_metrics(
        data["signal_test"],
        metric.predict_draws(hard_decoder, data["X_test"], hard_xi),
        prefix="mcmc_reference",
    )
    matched = {}
    for index, (name, gate) in enumerate(gates.items()):
        print(f"\n=== matched {name} MCMC ===")
        decoder, xi, result = run_reference(
            data,
            fit_units=args.fit_units,
            sigma2=args.sigma2,
            gate=gate,
            mcmc_n=args.mcmc_n,
            mcmc_burnin=args.mcmc_burnin,
            mcmc_thin=args.mcmc_thin,
            seed=args.seed + 300 + index,
            print_every=print_every,
        )
        matched[name] = (decoder, xi)
        reference_summaries[name] = save_reference(
            output_dir / "mcmc" / name, result
        )

    outputs = {}
    comparisons = {}
    summary_rows = []
    for index, (name, gate) in enumerate(gates.items()):
        print(f"\n=== VI activation: {name} ===")
        matched_decoder, matched_xi = matched[name]
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
            selection_mode="unit_group",
            input_dim=1,
            H=args.fit_units,
            hidden_dims=(args.fit_units,),
            architecture_mode="stacked",
            family="gaussian",
            sigma2=args.sigma2,
            K_flow=args.flow_depth,
            flow_type="full_attention_affine",
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
        vi_decoder = output["model"].decoder
        vi_xi = output["final"]["xi"]

        for reference_name, reference_decoder, reference_xi, compatibility in (
            ("matched", matched_decoder, matched_xi, "exact"),
            ("hard", hard_decoder, hard_xi, "structural"),
        ):
            comparison, table = comparison_metrics(
                vi_decoder,
                vi_xi,
                reference_decoder,
                reference_xi,
                truth,
                min_draws=args.min_draws,
                compatibility=compatibility,
            )
            comparisons[(name, reference_name)] = comparison
            save_json(vi_dir / f"comparison_{reference_name}.json", comparison)
            save_frame(vi_dir / f"recovery_{reference_name}.csv", table)
            base = output["final"]["summary"]
            reference_function = (
                {
                    "mcmc_reference_signal_r2": base["mcmc_signal_r2"],
                    "mcmc_reference_mse": base["mcmc_mse"],
                }
                if reference_name == "matched" else hard_function
            )
            summary_rows.append({
                "activation": name,
                "reference": reference_name,
                **comparison,
                "rat_test_signal_r2": base["rat_signal_r2"],
                "rat_test_signal_mse": base["rat_mse"],
                **reference_function,
                "elbo": base["elbo"],
                "train_time_sec": base["train_time_sec"],
                "posterior_sampling_time_sec": base[
                    "posterior_sampling_time_sec"
                ],
                "posterior_sampling_time_iqr_sec": base[
                    "posterior_sampling_time_iqr_sec"
                ],
                "trainable_params": base["trainable_params"],
                "flow_trainable_params": base["flow_trainable_params"],
                "mcmc_sampling_time_sec": reference_summaries[
                    name if reference_name == "matched" else "hard"
                ]["sampling_time_sec"],
            })

    summary = pd.DataFrame(summary_rows)
    save_frame(output_dir / "summary.csv", summary)

    all_draw_sets = [(hard_decoder, hard_xi)]
    all_draw_sets.extend(matched.values())
    all_draw_sets.extend(
        (output["model"].decoder, output["final"]["xi"])
        for output in outputs.values()
    )
    axis_limits = common_joint_axis(all_draw_sets, truth)
    plot_manifest = {}
    for name, output in outputs.items():
        vi_decoder = output["model"].decoder
        vi_xi = output["final"]["xi"]
        for reference_name, (reference_decoder, reference_xi) in {
            "matched": matched[name],
            "hard": (hard_decoder, hard_xi),
        }.items():
            key = f"{name}_vs_{reference_name}"
            plot_manifest[key] = save_joint_plot(
                output_dir / "plots" / key,
                vi_decoder=vi_decoder,
                vi_xi=vi_xi,
                reference_decoder=reference_decoder,
                reference_xi=reference_xi,
                truth=truth,
                min_draws=args.min_draws,
                axis_limits=axis_limits,
                vi_label=f"VI {name}",
                reference_label=(
                    f"MCMC {name}" if reference_name == "matched"
                    else "MCMC hard"
                ),
                title=key.replace("_", " "),
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
