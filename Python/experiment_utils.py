"""Small utilities shared by the cleaned grouped-BNN experiments."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import bnn_metric
from .bnn_mcmc import chain_diagnostics, run_bnn_mcmc, run_bnn_mcmc_chains
from .model2 import GroupedBNNVI
from .simfun import simfun_grouped_bnn


class Tee:
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
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
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


def choose_device(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(name)


def make_p6_bounded_data(
    *,
    n=240,
    fit_units=5,
    sigma2=1.0,
    seed=123,
    device=None,
    dtype=torch.float32,
):
    """Fixed p=6 teacher with active features x0/x3 and two true units."""

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
    idx = {
        k: torch.as_tensor(v, device=X.device, dtype=torch.long)
        for k, v in split.items()
    }
    data = {
        "X_train": X[idx["train"]],
        "y_train": y[idx["train"]],
        "X_eval": X[idx["eval"]],
        "signal_eval": signal[idx["eval"]],
        "X_test": X[idx["test"]],
        "signal_test": signal[idx["test"]],
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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=X.detach().cpu().numpy(),
        y=y.detach().cpu().numpy(),
        signal=signal.detach().cpu().numpy(),
        train_index=split["train"],
        eval_index=split["eval"],
        test_index=split["test"],
    )
    save_json(path.with_suffix(".truth.json"), truth)


def run_reference(
    data,
    *,
    hidden_dims,
    selection_mode,
    sigma2=1.0,
    gate_scale=1.0,
    mcmc_n=6000,
    mcmc_burnin=1000,
    mcmc_thin=1,
    n_chains=4,
    seed=123,
    print_every=600,
    initial_state="prior",
):
    """Run decoder-identical ESS for the shallow feature/unit experiments."""

    if selection_mode not in {"feature_group", "unit_group"}:
        raise ValueError("MCMC reference is retained only for shallow feature/unit modes.")
    model = GroupedBNNVI(
        X=data["X_train"],
        y=data["y_train"],
        input_dim=int(data["X_train"].shape[1]),
        hidden_dims=hidden_dims,
        selection_mode=selection_mode,
        family="gaussian",
        sigma2=sigma2,
        K_flow=0,
        flow_type="meanfield",
        gate_scale=gate_scale,
    ).to(data["X_train"].device)

    kwargs = {
        "N": int(mcmc_n),
        "burnin": int(mcmc_burnin),
        "thin": int(mcmc_thin),
        "seed": int(seed),
        "print_every": int(print_every),
    }
    if int(n_chains) == 1:
        result = run_bnn_mcmc(model, initial_state=initial_state, **kwargs)
    else:
        result = run_bnn_mcmc_chains(
            model,
            n_chains=int(n_chains),
            initial_state=initial_state,
            **kwargs,
        )
    xi = torch.as_tensor(
        result["xi_draws"],
        device=data["X_train"].device,
        dtype=data["X_train"].dtype,
    )
    return model.decoder, xi, result


@torch.no_grad()
def mcmc_reference_diagnostics(decoder, chain_xi, X_predict):
    """Compact convergence diagnostics in selection and prediction coordinates."""

    chain_xi = np.asarray(chain_xi, dtype=float)
    n_chains, n_draws, latent_dim = chain_xi.shape
    X_predict = torch.as_tensor(X_predict)
    chain_tensor = torch.as_tensor(
        chain_xi, device=X_predict.device, dtype=X_predict.dtype
    )
    flat = chain_tensor.reshape(-1, latent_dim)
    selection = bnn_metric.selection_draws(decoder, flat)
    active = selection["active"].reshape(
        n_chains, n_draws, -1
    ).float().cpu().numpy()

    semantics = decoder.group_semantics(flat)
    threshold = semantics["t"].reshape(
        n_chains, n_draws, -1
    ).cpu().numpy()
    active_count = active.sum(axis=-1, keepdims=True)
    threshold_diag = chain_diagnostics(
        threshold,
        names=[f"threshold_{i}" for i in range(threshold.shape[-1])],
    )
    count_diag = chain_diagnostics(active_count, names=["active_count"])[0]

    prediction = []
    for chain in range(n_chains):
        draw = bnn_metric.predict_draws(
            decoder, X_predict, chain_tensor[chain]
        )
        prediction.append(draw.numpy())
    prediction = np.stack(prediction, axis=0)
    predictive_diag = chain_diagnostics(
        prediction,
        names=[f"prediction_{i}" for i in range(prediction.shape[-1])],
    )

    return {
        "n_chains": int(n_chains),
        "draws_per_chain": int(n_draws),
        "threshold": threshold_diag,
        "active_count": count_diag,
        "max_threshold_rhat": float(max(row["rhat"] for row in threshold_diag)),
        "min_threshold_ess": float(min(row["ess"] for row in threshold_diag)),
        "max_predictive_rhat": float(max(row["rhat"] for row in predictive_diag)),
        "min_predictive_ess": float(min(row["ess"] for row in predictive_diag)),
    }


def save_reference(path, result, decoder=None, X_predict=None):
    """Persist only the reference draws and compact convergence summary."""

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    arrays = {"xi": result["xi_draws"]}
    if "chain_xi_draws" in result:
        arrays["chain_xi"] = result["chain_xi_draws"]
        arrays["chain_seeds"] = np.asarray(result["chain_seeds"], dtype=int)
    np.savez_compressed(path / "posterior_draws.npz", **arrays)

    summary = {
        key: value for key, value in result.items()
        if key not in {"xi_draws", "chain_xi_draws", "n_s"}
    }
    if decoder is not None and X_predict is not None and "chain_xi_draws" in result:
        summary["diagnostics"] = mcmc_reference_diagnostics(
            decoder, result["chain_xi_draws"], X_predict
        )
    save_json(path / "summary.json", summary)
    return summary


def save_training(path, output, *, save_draws=True, save_model=False):
    """Minimal per-run persistence: summary/config and optional draws/state."""

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    save_json(path / "summary.json", output["final"]["summary"])
    save_json(path / "config.json", output["config"])

    feature_pip = output["final"].get("feature_pip")
    if feature_pip is not None:
        save_frame(
            path / "feature_pip.csv",
            pd.DataFrame({
                "feature": np.arange(len(feature_pip)),
                "pip": np.asarray(feature_pip, dtype=float),
            }),
        )

    if save_draws:
        np.savez_compressed(
            path / "posterior_draws.npz",
            xi=output["final"]["xi"].detach().cpu().numpy(),
        )
    if save_model:
        torch.save(output["model"].state_dict(), path / "model_state.pt")


def stability_from_runs(summary_or_pips, k_true, threshold=0.5):
    """Post-process repeated VI runs into feature frequency + Kuncheva index."""

    if isinstance(summary_or_pips, pd.DataFrame):
        pip_columns = sorted(
            [col for col in summary_or_pips.columns if col.startswith("pip_x")],
            key=lambda x: int(x.replace("pip_x", "")),
        )
        pips = summary_or_pips[pip_columns].to_numpy(dtype=float)
    else:
        pips = np.asarray(summary_or_pips, dtype=float)
    result = bnn_metric.stability_metrics(
        pips, k_true=int(k_true), threshold=float(threshold)
    )
    frequency = pd.DataFrame({
        "feature": np.arange(pips.shape[1]),
        "frequency": result["feature_frequency"],
    })
    return {
        "kuncheva_index": result["kuncheva_index"],
        "n_runs": result["n_runs"],
        "feature_frequency": frequency,
    }
