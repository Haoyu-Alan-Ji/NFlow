"""Training loop for the cleaned grouped DSS-LVR BNN."""

from __future__ import annotations

import random
import time

import numpy as np
import pandas as pd
import torch

from . import bnn_metric
from .model2 import GroupedBNNVI


def train_grouped_bnn(
    X_train,
    y_train,
    X_eval,
    signal_eval,
    *,
    truth,
    X_final=None,
    signal_final=None,
    reference_decoder=None,
    reference_xi=None,
    selection_mode="feature_group",
    input_dim=None,
    hidden_dims=(5,),
    out_dim=1,
    family="gaussian",
    sigma2=1.0,
    init_sd=0.5,
    K_flow=6,
    flow_type="iaf",
    flow_hidden_units=128,
    flow_hidden_layers=2,
    scale_clip=2.0,
    flow_seed=None,
    iaf_ordering_scheme="cyclic3",
    iaf_shuffle_within_role=True,
    gate_type="normalized_requ",
    gate_scale=1.0,
    epochs=2000,
    warmup_epochs=500,
    lr=3e-4,
    R_train=100,
    R_eval=1000,
    R_final=5000,
    eval_every=250,
    sampling_timing_repeats=3,
    init_loc_jitter=0.05,
    grad_clip=5.0,
    support_threshold=0.5,
    min_active_draws=50,
    seed=123,
):
    """Train one grouped BNN and compute only final paper-level metrics.

    MCMC is optional.  When a reference is supplied, Active SKL and Zero JS
    are added to the final summary; otherwise the same function serves the MLP
    experiments without any MCMC dependency.
    """

    if selection_mode not in {
        "feature_group", "unit_group", "feature_unit_induced_edge"
    }:
        raise ValueError("Unsupported selection_mode.")
    if not 0 <= int(warmup_epochs) < int(epochs):
        raise ValueError("warmup_epochs must be in [0, epochs).")
    if int(sampling_timing_repeats) < 1:
        raise ValueError("sampling_timing_repeats must be positive.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if flow_seed is None:
        flow_seed = int(seed)

    device = X_train.device
    dtype = X_train.dtype
    X_eval = torch.as_tensor(X_eval, device=device, dtype=dtype)
    signal_eval = torch.as_tensor(signal_eval, device=device, dtype=dtype)
    X_final = X_eval if X_final is None else torch.as_tensor(
        X_final, device=device, dtype=dtype
    )
    signal_final = signal_eval if signal_final is None else torch.as_tensor(
        signal_final, device=device, dtype=dtype
    )

    if reference_xi is not None:
        reference_xi = torch.as_tensor(reference_xi, device=device, dtype=dtype)
    if (reference_decoder is None) != (reference_xi is None):
        raise ValueError("reference_decoder and reference_xi must be supplied together.")

    model = GroupedBNNVI(
        X=X_train,
        y=y_train,
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        out_dim=out_dim,
        selection_mode=selection_mode,
        family=family,
        sigma2=sigma2,
        init_sd=init_sd,
        K_flow=K_flow,
        flow_type=flow_type,
        flow_hidden_units=flow_hidden_units,
        flow_hidden_layers=flow_hidden_layers,
        scale_clip=scale_clip,
        flow_seed=flow_seed,
        iaf_ordering_scheme=iaf_ordering_scheme,
        iaf_shuffle_within_role=iaf_shuffle_within_role,
        gate_type=gate_type,
        gate_scale=gate_scale,
    ).to(device)

    if reference_decoder is not None and (
        model.decoder.compatibility_signature()
        != reference_decoder.compatibility_signature()
    ):
        raise ValueError("MCMC and VI decoders must be exactly matched.")

    if float(init_loc_jitter) > 0:
        with torch.no_grad():
            model.q0.loc.add_(
                float(init_loc_jitter) * torch.randn_like(model.q0.loc)
            )

    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    trainable_params = int(sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    ))
    flow_trainable_params = int(sum(
        parameter.numel()
        for parameter in model.flow.parameters()
        if parameter.requires_grad
    ))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    history = []

    for epoch in range(1, int(epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        warmup = epoch <= int(warmup_epochs)

        if warmup:
            xi_train, log_q = model.sample_posterior(R_train)
            log_likelihood = model.log_likelihood(xi_train, force_all_on=True)
            log_prior = model.log_prior(xi_train)
            elbo = log_likelihood + log_prior - log_q
        else:
            terms = model.elbo_draws(R_train)
            elbo = terms["elbo"]
        loss = -elbo.mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch}.")

        loss.backward()
        if not all(
            p.grad is None or bool(torch.isfinite(p.grad).all())
            for p in model.parameters()
        ):
            raise FloatingPointError(f"Non-finite gradient at epoch {epoch}.")
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        optimizer.step()

        if (
            epoch == 1
            or epoch % int(eval_every) == 0
            or epoch == int(epochs)
        ):
            model.eval()
            with torch.no_grad():
                xi_eval, _ = model.sample_posterior(R_eval)
                pred_eval = model.decoder(
                    X_eval, xi_eval, force_all_on=warmup
                )
            function = bnn_metric.function_metrics(signal_eval, pred_eval)
            row = {
                "epoch": int(epoch),
                "phase": "repr" if warmup else "select",
                "loss": float(loss.detach()),
                "elbo": float(elbo.mean().detach()),
                "val_mse": function["mse"],
                "val_r2": function["r2"],
            }
            history.append(row)
            print(
                f"epoch={epoch:04d} phase={row['phase']:6s} "
                f"valMSE={row['val_mse']:.5f} valR2={row['val_r2']:.4f}"
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_time_sec = time.perf_counter() - started

    model.eval()
    sampling_times = []
    xi_final = None
    log_q_final = None
    for repeat in range(int(sampling_timing_repeats)):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        sample_started = time.perf_counter()
        with torch.no_grad():
            xi_sample, log_q_sample = model.sample_posterior(R_final)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        sampling_times.append(time.perf_counter() - sample_started)
        if repeat == 0:
            xi_final = xi_sample
            log_q_final = log_q_sample

    with torch.no_grad():
        final_ll = model.log_likelihood(xi_final)
        final_prior = model.log_prior(xi_final)

    metrics = bnn_metric.evaluate_bnn(
        decoder=model.decoder,
        xi=xi_final,
        X=X_final,
        signal=signal_final,
        truth=truth,
        reference_decoder=reference_decoder,
        reference_xi=reference_xi,
        support_threshold=support_threshold,
        min_active_draws=min_active_draws,
    )

    recovery_table = metrics.pop("recovery_table", None)
    feature_pip = metrics.pop("feature_pip", None)
    summary = {
        "flow_type": model.flow_type,
        "selection_mode": selection_mode,
        "train_time_sec": float(train_time_sec),
        "sec_per_epoch": float(train_time_sec / int(epochs)),
        "posterior_sampling_time_sec": float(np.median(sampling_times)),
        "posterior_sampling_time_iqr_sec": float(
            np.quantile(sampling_times, 0.75)
            - np.quantile(sampling_times, 0.25)
        ),
        "trainable_params": trainable_params,
        "flow_trainable_params": flow_trainable_params,
        "elbo": float((final_ll + final_prior - log_q_final).mean()),
        "expected_log_likelihood": float(final_ll.mean()),
        "expected_log_prior": float(final_prior.mean()),
        "expected_log_q": float(log_q_final.mean()),
        "gpu_peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda" else None
        ),
        **metrics,
    }

    ordering_summary = (
        model.flow.ordering_summary()
        if hasattr(model.flow, "ordering_summary") else None
    )
    role_position_counts = (
        model.flow.role_position_counts()
        if hasattr(model.flow, "role_position_counts") else None
    )
    if ordering_summary is not None:
        role_orders = [
            "generic" if row["role_order"] is None
            else "<".join(row["role_order"])
            for row in ordering_summary
        ]
        summary["iaf_ordering_scheme"] = getattr(
            model.flow, "ordering_scheme", str(iaf_ordering_scheme)
        )
        summary["iaf_role_orders"] = " | ".join(role_orders)

    flow_sanity = None
    if hasattr(model.flow, "numerical_sanity_check"):
        with torch.no_grad():
            base = model.q0.sample(min(256, int(R_final)))
            flow_sanity = model.flow.numerical_sanity_check(base)
        summary.update({
            f"flow_{key}": value for key, value in flow_sanity.items()
        })

    return {
        "model": model,
        "history": pd.DataFrame(history),
        "final": {
            "summary": summary,
            "xi": xi_final.detach(),
            "feature_pip": feature_pip,
            "recovery_table": recovery_table,
            "flow_sanity": flow_sanity,
            "ordering_summary": ordering_summary,
            "role_position_counts": role_position_counts,
        },
        "config": {
            "selection_mode": selection_mode,
            "input_dim": int(model.decoder.input_dim),
            "hidden_dims": tuple(model.decoder.hidden_dims),
            "out_dim": int(out_dim),
            "family": family,
            "sigma2": float(sigma2),
            "gate_type": str(gate_type),
            "gate_scale": float(gate_scale),
            "flow_type": model.flow_type,
            "K_flow": int(K_flow),
            "flow_hidden_units": int(flow_hidden_units),
            "flow_hidden_layers": int(flow_hidden_layers),
            "scale_clip": float(scale_clip),
            "flow_seed": int(flow_seed),
            "iaf_ordering_scheme": str(iaf_ordering_scheme),
            "iaf_shuffle_within_role": bool(iaf_shuffle_within_role),
            "epochs": int(epochs),
            "warmup_epochs": int(warmup_epochs),
            "R_train": int(R_train),
            "R_eval": int(R_eval),
            "R_final": int(R_final),
            "lr": float(lr),
            "seed": int(seed),
        },
    }
