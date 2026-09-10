"""Shared, intentionally small helpers for the two grouped-BNN experiments."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import metric
from .bnn_mcmc import (
    chain_diagnostics,
    run_bnn_mcmc,
    run_bnn_mcmc_chains,
)
from .model2 import GroupedBNNVI
from .simfun import simfun_grouped_bnn


class Tee:
    """Mirror stdout/stderr to one experiment log."""

    def __init__(self, path):
        self.path = Path(path)
        self.stream = None
        self.old_stdout = None
        self.old_stderr = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("w", encoding="utf-8")
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = self
        return self

    def write(self, text):
        self.old_stdout.write(text)
        self.stream.write(text)

    def flush(self):
        self.old_stdout.flush()
        self.stream.flush()

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout, sys.stderr = self.old_stdout, self.old_stderr
        self.stream.close()


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        scalar = float(value)
        return scalar if np.isfinite(scalar) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(jsonable(payload), stream, indent=2, ensure_ascii=False)


def save_frame(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(frame).to_csv(path, index=False)


def choose_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda was requested but CUDA is unavailable.")
    return torch.device(name)


def make_one_layer_unit_data(
    *,
    n,
    fit_units,
    sigma2,
    seed,
    device,
    dtype=torch.float32,
):
    """One shared dataset with exactly two truth-active unit ranks."""

    X, y, _, _, signal, truth = simfun_grouped_bnn(
        n=n,
        p=1,
        n_active_features=1,
        n_true_units=2,
        fit_units=fit_units,
        sigma2=sigma2,
        seed=seed,
        device=device,
        dtype=dtype,
    )
    rng = np.random.default_rng(seed + 1000)
    order = rng.permutation(int(n))
    n_train = int(round(0.60 * n))
    n_eval = int(round(0.20 * n))
    split = {
        "train": order[:n_train],
        "eval": order[n_train:n_train + n_eval],
        "test": order[n_train + n_eval:],
    }
    index = {
        name: torch.as_tensor(values, device=X.device, dtype=torch.long)
        for name, values in split.items()
    }
    data = {
        "X_train": X[index["train"]],
        "y_train": y[index["train"]],
        "X_eval": X[index["eval"]],
        "signal_eval": signal[index["eval"]],
        "X_test": X[index["test"]],
        "signal_test": signal[index["test"]],
    }
    return data, truth, split, (X, y, signal)


def make_p6_bounded_data(
    *,
    n=240,
    fit_units=5,
    sigma2=1.0,
    seed=123,
    device=None,
    dtype=torch.float32,
):
    """Restore the fixed p=6, x0/x3, two-unit benchmark."""

    # These are exactly the unscaled teacher specifications produced by the
    # previously used seed-123 canonical generator. Supplying them explicitly
    # prevents future simulator defaults from silently changing the teacher.
    unit_specs = (
        {
            "feature": 0,
            "slope": 0.8991619427476751,
            "breakpoint": 0.02917629811970296,
            "amplitude": 0.8106230864192018,
        },
        {
            "feature": 3,
            "slope": -0.8791576554882636,
            "breakpoint": -0.07138863699164437,
            "amplitude": 1.1872567039934643,
        },
    )
    X, y, _, _, signal, truth = simfun_grouped_bnn(
        n=int(n),
        p=6,
        active_features=(0, 3),
        n_true_units=2,
        fit_units=int(fit_units),
        unit_specs=unit_specs,
        repu_power=None,
        sigma2=float(sigma2),
        x_low=-2.5,
        x_high=2.5,
        target_signal_sd=1.5,
        seed=int(seed),
        device=device,
        dtype=dtype,
    )
    rng = np.random.default_rng(int(seed) + 1000)
    order = rng.permutation(int(n))
    n_train = int(round(0.60 * int(n)))
    n_eval = int(round(0.20 * int(n)))
    split = {
        "train": order[:n_train],
        "eval": order[n_train:n_train + n_eval],
        "test": order[n_train + n_eval:],
    }
    index = {
        name: torch.as_tensor(values, device=X.device, dtype=torch.long)
        for name, values in split.items()
    }
    data = {
        "X_train": X[index["train"]],
        "y_train": y[index["train"]],
        "X_eval": X[index["eval"]],
        "signal_eval": signal[index["eval"]],
        "X_test": X[index["test"]],
        "signal_test": signal[index["test"]],
    }
    truth.update({
        "H_true": 2,
        "p_true": 2,
        "active_features": [0, 3],
        "teacher_source": "fixed_seed123_p6_teacher",
        "split_rule": "seed+1000 permutation; 60/20/20",
    })
    return data, truth, split, (X, y, signal)


def save_dataset(path, full_data, split, truth):
    X, y, signal = full_data
    np.savez_compressed(
        path,
        X=X.detach().cpu().numpy(),
        y=y.detach().cpu().numpy(),
        signal=signal.detach().cpu().numpy(),
        train_index=split["train"],
        eval_index=split["eval"],
        test_index=split["test"],
    )
    save_json(Path(path).with_suffix(".truth.json"), truth)


def run_reference(
    data,
    *,
    fit_units=None,
    hidden_dims=None,
    architecture_mode="stacked",
    embedding_dim=None,
    sigma2,
    gate,
    mcmc_n,
    mcmc_burnin,
    mcmc_thin,
    seed,
    print_every,
    selection_mode="unit_group",
    input_dim=None,
    n_chains=1,
    initial_state="prior",
):
    """Build one decoder and run its exactly matched ESS reference."""

    if input_dim is None:
        input_dim = int(data["X_train"].shape[1])
    if hidden_dims is None:
        if fit_units is None:
            raise ValueError("Provide hidden_dims or fit_units.")
        hidden_dims = (int(fit_units),)

    model = GroupedBNNVI(
        X=data["X_train"],
        y=data["y_train"],
        input_dim=int(input_dim),
        hidden_dims=tuple(int(width) for width in hidden_dims),
        out_dim=1,
        selection_mode=selection_mode,
        architecture_mode=architecture_mode,
        embedding_dim=embedding_dim,
        family="gaussian",
        sigma2=sigma2,
        K_flow=0,
        flow_type="meanfield",
        **gate,
    ).to(data["X_train"].device)
    mcmc_kwargs = {
        "N": mcmc_n,
        "burnin": mcmc_burnin,
        "thin": mcmc_thin,
        "seed": seed,
        "print_every": print_every,
    }
    if int(n_chains) == 1:
        result = run_bnn_mcmc(
            model,
            initial_state=initial_state,
            **mcmc_kwargs,
        )
    else:
        result = run_bnn_mcmc_chains(
            model,
            n_chains=n_chains,
            initial_state=initial_state,
            **mcmc_kwargs,
        )
    xi = torch.as_tensor(
        result["xi_draws"],
        device=data["X_train"].device,
        dtype=data["X_train"].dtype,
    )
    return model.decoder, xi, result


@torch.no_grad()
def mcmc_reference_diagnostics(decoder, chain_xi, X_predict):
    """Derived multi-chain diagnostics in label-invariant selection coordinates."""

    chain_xi = np.asarray(chain_xi, dtype=float)
    n_chains, n_draws, latent_dim = chain_xi.shape
    X_predict = torch.as_tensor(X_predict)
    chain_tensor = torch.as_tensor(
        chain_xi,
        device=X_predict.device,
        dtype=X_predict.dtype,
    )
    flat = chain_tensor.reshape(-1, latent_dim)
    selection = metric.grouped_selection_draws(decoder, flat)
    active = selection["pip_draws"].reshape(
        n_chains, n_draws, -1
    ).detach().cpu().numpy().astype(float)
    slab = selection["slab_strength"].reshape(
        n_chains, n_draws, -1
    ).detach().cpu().numpy()
    effective = selection["effective_strength"].reshape(
        n_chains, n_draws, -1
    ).detach().cpu().numpy()
    n_targets = active.shape[-1]
    if decoder.selection_mode == "unit_group":
        labels = [f"unit_rank_{index + 1}" for index in range(n_targets)]
    elif decoder.selection_mode == "feature_group":
        labels = [f"x{index}" for index in range(n_targets)]
    else:
        labels = []
        for meta in decoder.group_meta:
            if meta["selection_type"] == "feature":
                labels.append(f"feature_x{meta['feature']}")
            elif meta["selection_type"] == "unit":
                labels.append(
                    f"unit_L{int(meta['layer']) + 1}_{int(meta['unit'])}"
                )
            else:
                labels.append(
                    f"edge_{meta['parameter']}_{meta['target']}_{meta['source']}"
                )

    semantics = decoder.group_semantics(flat)
    threshold = semantics["t"].reshape(
        n_chains, n_draws, -1
    ).detach().cpu().numpy()
    active_count = active.sum(axis=-1, keepdims=True)
    derived = np.concatenate([threshold, active_count], axis=-1)
    derived_names = [
        f"threshold_{index}" for index in range(threshold.shape[-1])
    ] + ["active_count"]

    latent_diagnostics = pd.DataFrame(chain_diagnostics(
        chain_xi,
        names=[f"xi_{index}" for index in range(latent_dim)],
    ))
    selection_rows = []
    for quantity, values in (
        ("active_indicator", active),
        ("slab_strength", slab),
        ("effective_strength", effective),
        ("derived", derived),
    ):
        names = labels if quantity != "derived" else derived_names
        rows = chain_diagnostics(values, names=names)
        for row in rows:
            row["quantity"] = quantity
        selection_rows.extend(rows)
    selection_diagnostics = pd.DataFrame(selection_rows)

    pip_rows = []
    for target, label in enumerate(labels):
        for chain in range(n_chains):
            pip_rows.append({
                "target": target,
                "label": label,
                "chain": chain + 1,
                "pip": float(active[chain, :, target].mean()),
            })
        pip_rows.append({
            "target": target,
            "label": label,
            "chain": "pooled",
            "pip": float(active[:, :, target].mean()),
        })
    selection_pip = pd.DataFrame(pip_rows)

    predictive = []
    for chain in range(n_chains):
        prediction = metric.predict_draws(
            decoder, X_predict, chain_tensor[chain]
        )
        predictive.append(prediction.detach().cpu().numpy())
    predictive = np.stack(predictive, axis=0)
    predictive_diagnostics = pd.DataFrame(chain_diagnostics(
        predictive,
        names=[f"prediction_{index}" for index in range(predictive.shape[-1])],
    ))

    threshold_rows = selection_diagnostics[
        (selection_diagnostics["quantity"] == "derived")
        & selection_diagnostics["variable"].str.startswith("threshold_")
    ]
    threshold_row = threshold_rows.iloc[0]
    active_count_row = selection_diagnostics[
        (selection_diagnostics["quantity"] == "derived")
        & (selection_diagnostics["variable"] == "active_count")
    ].iloc[0]
    summary = {
        "n_chains": int(n_chains),
        "draws_per_chain": int(n_draws),
        "threshold": threshold_row.to_dict(),
        "thresholds": threshold_rows.to_dict("records"),
        "threshold_roles": list(getattr(decoder, "threshold_roles", ("shared",))),
        "max_threshold_rhat": float(np.nanmax(threshold_rows["rhat"])),
        "min_threshold_ess": float(np.nanmin(threshold_rows["ess"])),
        "active_count": active_count_row.to_dict(),
        "pooled_pip": {
            label: float(active[:, :, index].mean())
            for index, label in enumerate(labels)
        },
        "chain_pip": {
            f"chain_{chain + 1}": {
                label: float(active[chain, :, index].mean())
                for index, label in enumerate(labels)
            }
            for chain in range(n_chains)
        },
        "max_latent_rhat": float(np.nanmax(latent_diagnostics["rhat"])),
        "min_latent_ess": float(np.nanmin(latent_diagnostics["ess"])),
        "max_predictive_rhat": float(
            np.nanmax(predictive_diagnostics["rhat"])
        ),
        "min_predictive_ess": float(
            np.nanmin(predictive_diagnostics["ess"])
        ),
    }
    return {
        "latent_diagnostics": latent_diagnostics,
        "selection_diagnostics": selection_diagnostics,
        "selection_pip": selection_pip,
        "predictive_diagnostics": predictive_diagnostics,
        "predictive_draws": predictive,
        "summary": summary,
    }


def save_reference(path, result, decoder=None, X_predict=None):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    arrays = {
        "xi": result["xi_draws"],
        "slice_steps": result["n_s"],
    }
    if "chain_xi_draws" in result:
        arrays["chain_xi"] = result["chain_xi_draws"]
        arrays["chain_seeds"] = np.asarray(result["chain_seeds"], dtype=int)
    np.savez_compressed(path / "posterior_draws.npz", **arrays)
    summary = {
        key: value
        for key, value in result.items()
        if key not in {"xi_draws", "chain_xi_draws", "n_s"}
    }
    # Persist timing/config immediately so a later diagnostics failure never
    # forces the expensive chains to run again.
    save_json(path / "summary.json", summary)
    if decoder is not None and X_predict is not None and "chain_xi_draws" in result:
        diagnostics = save_mcmc_diagnostics(
            path,
            decoder,
            result["chain_xi_draws"],
            X_predict,
        )
        summary["diagnostics"] = diagnostics["summary"]
    save_json(path / "summary.json", summary)
    return summary


def save_mcmc_diagnostics(path, decoder, chain_xi, X_predict):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    diagnostics = mcmc_reference_diagnostics(
        decoder,
        chain_xi,
        X_predict,
    )
    for name in (
        "latent_diagnostics",
        "selection_diagnostics",
        "selection_pip",
        "predictive_diagnostics",
    ):
        save_frame(path / f"{name}.csv", diagnostics[name])
    np.savez_compressed(
        path / "predictive_draws.npz",
        prediction=diagnostics["predictive_draws"],
    )
    save_json(path / "diagnostics.json", diagnostics["summary"])
    return diagnostics


def save_training(path, output):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    save_frame(path / "history.csv", output["history"])
    save_frame(path / "group_history.csv", output["group_history"])
    save_frame(path / "unit_history.csv", output["unit_history"])
    save_frame(
        path / "recovery_matched.csv",
        output["final"]["recovery_by_target"],
    )
    save_frame(path / "group_metrics.csv", output["final"]["group_metrics"])
    save_frame(path / "unit_metrics.csv", output["final"]["unit_metrics"])
    if output["config"]["selection_mode"] == "feature_group":
        save_frame(
            path / "feature_metrics.csv",
            output["final"]["group_metrics"],
        )
    def array(value):
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    np.savez_compressed(
        path / "posterior_draws.npz",
        xi=array(output["final"]["xi"]),
        rat_prediction_draws=array(output["final"]["rat_prediction_draws"]),
        mcmc_prediction_draws=array(output["final"]["mcmc_prediction_draws"]),
    )
    save_json(path / "config.json", output["config"])
    save_json(path / "summary_matched.json", output["final"]["summary"])
    if output["final"].get("flow_sanity") is not None:
        save_json(
            path / "flow_sanity.json",
            output["final"]["flow_sanity"],
        )
        if output["config"].get("coupling_type") == "spline":
            save_json(
                path / "spline_sanity.json",
                output["final"]["flow_sanity"],
            )
    torch.save(output["model"].state_dict(), path / "model_state.pt")


def save_attention_diagnostics(path, model, xi, max_draws=1000):
    """Save mean state-dependent attention maps and entropy by head."""

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if not hasattr(model.flow, "attention_diagnostics"):
        return {}
    payload = model.flow.attention_diagnostics(
        xi, max_draws=int(max_draws)
    )
    if not payload:
        return {}

    metadata = {
        name: value.detach().cpu().numpy()
        for name, value in model.decoder.flow_metadata().items()
    }
    files = {}
    entropy_rows = []
    branch_head_values = {}
    reserved = {"layer", "fixed_idx", "target_idx"}
    branches = tuple(key for key in payload[0] if key not in reserved)
    branch_output_name = {
        "shift": "shift",
        "shape": "scale",
        "width_height": "width_height",
        "derivative": "derivative",
    }
    for branch in branches:
        output_branch = branch_output_name.get(branch, branch)
        n_heads = int(payload[0][branch].shape[1])
        for head in range(n_heads):
            rows = []
            head_values = []
            for item in payload:
                weights = item[branch][:, head].numpy()
                mean_weights = weights.mean(axis=0)
                entropy = -np.sum(
                    weights * np.log(np.clip(weights, 1e-12, 1.0)),
                    axis=-1,
                )
                fixed_idx = item["fixed_idx"].numpy()
                target_idx = item["target_idx"].numpy()
                for target_local, target_coord in enumerate(target_idx):
                    entropy_rows.append({
                        "branch": output_branch,
                        "head": head + 1,
                        "coupling_layer": item["layer"],
                        "target_coord": int(target_coord),
                        "mean_entropy": float(entropy[:, target_local].mean()),
                    })
                    for fixed_local, fixed_coord in enumerate(fixed_idx):
                        row = {
                            "coupling_layer": item["layer"],
                            "target_coord": int(target_coord),
                            "fixed_coord": int(fixed_coord),
                            "mean_attention": float(
                                mean_weights[target_local, fixed_local]
                            ),
                        }
                        for name, values in metadata.items():
                            row[f"target_{name}"] = int(values[target_coord])
                            row[f"fixed_{name}"] = int(values[fixed_coord])
                        rows.append(row)
                head_values.append(weights)
            filename = f"attention_{output_branch}_head{head + 1}.csv"
            save_frame(path / filename, rows)
            files[f"{output_branch}_head{head + 1}"] = filename
            branch_head_values[(branch, head)] = head_values

    entropy = pd.DataFrame(entropy_rows)
    save_frame(path / "attention_entropy.csv", entropy)
    summary = {
        "n_draws": int(min(int(max_draws), xi.shape[0])),
        "n_coupling_layers": int(len(payload)),
        "files": files,
        "attention_branches": list(branches),
        "mean_entropy": {
            f"{branch_output_name.get(branch, branch)}_head{head + 1}": float(
                entropy[
                    (entropy["branch"] == branch_output_name.get(branch, branch))
                    & (entropy["head"] == head + 1)
                ]["mean_entropy"].mean()
            )
            for branch in branches
            for head in range(int(payload[0][branch].shape[1]))
        },
    }
    for branch in branches:
        if int(payload[0][branch].shape[1]) >= 2:
            differences = []
            for first, second in zip(
                branch_head_values[(branch, 0)],
                branch_head_values[(branch, 1)],
            ):
                differences.append(np.abs(first - second).mean())
            output_branch = branch_output_name.get(branch, branch)
            summary[f"{output_branch}_head_mean_abs_difference"] = float(
                np.mean(differences)
            )
    save_json(path / "attention_summary.json", summary)
    return summary


def comparison_metrics(
    vi_decoder,
    vi_xi,
    reference_decoder,
    reference_xi,
    truth,
    *,
    min_draws,
    compatibility,
    max_joint_draws=None,
    joint_random_seed=123,
):
    summary, table = metric.grouped_recovery_metrics(
        rat_decoder=vi_decoder,
        rat_xi=vi_xi,
        mcmc_decoder=reference_decoder,
        mcmc_xi=reference_xi,
        truth=truth,
        min_active_draws=min_draws,
        compatibility=compatibility,
    )
    joint = metric.true_active_joint_skl(
        rat_decoder=vi_decoder,
        rat_xi=vi_xi,
        mcmc_decoder=reference_decoder,
        mcmc_xi=reference_xi,
        truth=truth,
        min_draws=min_draws,
        compatibility=compatibility,
        max_draws=max_joint_draws,
        random_seed=joint_random_seed,
    )
    return {**summary, **joint}, table


def common_joint_axis(comparisons, truth):
    """One robust axis range shared by all requested density panels."""

    arrays = []
    for decoder, xi in comparisons:
        payload = metric.true_active_joint_draws(decoder, xi, truth)
        if payload["values"].size:
            arrays.append(payload["values"])
    if not arrays:
        return None
    values = np.concatenate(arrays, axis=0)
    x_lo, x_hi = np.quantile(values[:, 0], [0.005, 0.995])
    y_lo, y_hi = np.quantile(values[:, 1], [0.005, 0.995])
    x_pad = 0.10 * (x_hi - x_lo + 1e-8)
    y_pad = 0.10 * (y_hi - y_lo + 1e-8)
    return (
        float(x_lo - x_pad),
        float(x_hi + x_pad),
        float(y_lo - y_pad),
        float(y_hi + y_pad),
    )


def common_joint_draw_count(
    comparisons,
    truth,
    *,
    min_draws,
    max_draws=2000,
):
    """Shared conditional sample count for directly comparable HDR panels."""

    counts = [
        metric.true_active_joint_draws(decoder, xi, truth)["n_joint_active"]
        for decoder, xi in comparisons
    ]
    if any(count < int(min_draws) for count in counts):
        return None
    return int(min(min(counts), int(max_draws)))


def save_joint_plot(
    path,
    *,
    vi_decoder,
    vi_xi,
    reference_decoder,
    reference_xi,
    truth,
    min_draws,
    axis_limits,
    vi_label,
    reference_label,
    title,
    max_draws=None,
    random_seed=123,
):
    fig, _ = metric.plot_true_active_joint_density(
        rat_decoder=vi_decoder,
        rat_xi=vi_xi,
        mcmc_decoder=reference_decoder,
        mcmc_xi=reference_xi,
        truth=truth,
        min_draws=min_draws,
        axis_limits=axis_limits,
        rat_label=vi_label,
        mcmc_label=reference_label,
        title=title,
        max_draws=max_draws,
        random_seed=random_seed,
    )
    if fig is None:
        return False
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return True


def _pip_matrix(table):
    table = pd.DataFrame(table)
    matrix = table.pivot(index="target", columns="source", values="pip")
    matrix.index.name = "target"
    matrix.columns = [f"source_{int(value)}" for value in matrix.columns]
    return matrix.reset_index()


def save_mlp_structure_outputs(
    path,
    *,
    vi_decoder,
    vi_xi,
    mcmc_decoder,
    mcmc_xi,
    X,
    signal,
    thresholds=tuple(np.arange(0.1, 1.0, 0.1)),
):
    """Save role-specific PIPs and optional VI/MCMC MSE--density curves."""

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    vi = metric.mlp_structure_summary(vi_decoder, vi_xi, method="VI")
    mcmc = metric.mlp_structure_summary(
        mcmc_decoder, mcmc_xi, method="MCMC"
    )

    for role in ("feature", "unit"):
        if not vi[role].empty:
            save_frame(path / f"{role}_pip_vi.csv", vi[role])
            save_frame(path / f"{role}_pip_mcmc.csv", mcmc[role])

    manifest = {
        "selection_mode": vi_decoder.selection_mode,
        "n_candidate_edges": int(vi_decoder.n_candidate_edges),
        "layers": [],
    }
    if vi["edges"]:
        prefix = (
            "induced_edge"
            if vi_decoder.selection_mode == "feature_unit_induced_edge"
            else "edge"
        )
        for layer_index, item in enumerate(
            vi_decoder.layout.linear_weight_specs, start=1
        ):
            name = item["name"]
            vi_name = f"{prefix}_pip_vi_layer{layer_index}.csv"
            mcmc_name = f"{prefix}_pip_mcmc_layer{layer_index}.csv"
            save_frame(path / vi_name, _pip_matrix(vi["edges"][name]))
            save_frame(path / mcmc_name, _pip_matrix(mcmc["edges"][name]))
            manifest["layers"].append({
                "layer_index": layer_index,
                "parameter": name,
                "role": item["role"],
                "shape": list(item["shape"]),
                "vi_file": vi_name,
                "mcmc_file": mcmc_name,
            })

        vi_curve = metric.mse_edge_density_curve(
            vi_decoder, vi_xi, X, signal,
            thresholds=thresholds, method="VI",
        )
        mcmc_curve = metric.mse_edge_density_curve(
            mcmc_decoder, mcmc_xi, X, signal,
            thresholds=thresholds, method="MCMC",
        )
        curve = pd.concat([vi_curve, mcmc_curve], ignore_index=True)
        save_frame(path / "mse_vs_edge_density.csv", curve)
        fig, _ = metric.plot_mse_edge_density_curve(
            vi_curve,
            mcmc_curve,
            title=(
                f"{vi_decoder.architecture_mode}: "
                f"{vi_decoder.selection_mode}"
            ),
        )
        fig.savefig(
            path / "mse_vs_edge_density.png", dpi=180, bbox_inches="tight"
        )
        fig.savefig(path / "mse_vs_edge_density.pdf", bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)
    else:
        curve = pd.DataFrame()

    save_json(path / "structure_manifest.json", manifest)
    return {
        "vi_expected_structure_density": vi["expected_structure_density"],
        "mcmc_expected_structure_density": mcmc[
            "expected_structure_density"
        ],
        "vi": vi,
        "mcmc": mcmc,
        "curve": curve,
    }
