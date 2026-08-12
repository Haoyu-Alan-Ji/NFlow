import copy
import random

import numpy as np
import pandas as pd
import torch

from . import metric
from .model2 import DirectUnitBNNVI, GroupedBNNVI, LaSTBNNVI, ROLE_NAMES


def train_direct_bnn(
    X_train,
    y_train,
    X_eval,
    signal_eval,
    *,
    X_final=None,
    signal_final=None,
    mcmc_decoder,
    mcmc_xi,
    H=3,
    family="gaussian",
    sigma2=1.0,
    gate_roles=ROLE_NAMES,
    gate_power=2.0,
    gate_tau=1.0,
    init_sd=None,
    K_flow=8,
    flow_hidden_units=64,
    flow_hidden_layers=2,
    scale_clip=1.5,
    epochs=6000,
    lr=3e-4,
    R_train=64,
    R_eval=1000,
    R_final=5000,
    eval_every=250,
    grad_clip=5.0,
    epsilon_C=1e-6,
    breakpoint_eps=1e-4,
    zero_tol=1e-6,
    constant_tol=1e-6,
    reference_threshold=0.5,
    min_active_draws=50,
    seed=123,
):
    """Train direct-unit RaT and select by validation-grid signal R2."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = X_train.device
    dtype = X_train.dtype
    X_eval = torch.as_tensor(X_eval, device=device, dtype=dtype)
    signal_eval = torch.as_tensor(
        signal_eval,
        device=device,
        dtype=dtype,
    )
    X_final = X_eval if X_final is None else torch.as_tensor(
        X_final,
        device=device,
        dtype=dtype,
    )
    signal_final = (
        signal_eval
        if signal_final is None
        else torch.as_tensor(
            signal_final,
            device=device,
            dtype=dtype,
        )
    )

    model = DirectUnitBNNVI(
        X=X_train,
        y=y_train,
        H=H,
        family=family,
        sigma2=sigma2,
        gate_roles=gate_roles,
        gate_power=gate_power,
        gate_tau=gate_tau,
        init_sd=init_sd,
        K_flow=K_flow,
        flow_hidden_units=flow_hidden_units,
        flow_hidden_layers=flow_hidden_layers,
        scale_clip=scale_clip,
    ).to(device)

    if mcmc_decoder is not None and (
        mcmc_decoder.H != model.decoder.H
        or mcmc_decoder.gate_roles != model.decoder.gate_roles
        or mcmc_decoder.gate_power != model.decoder.gate_power
        or mcmc_decoder.gate_tau != model.decoder.gate_tau
    ):
        raise ValueError("MCMC and RaT must use the same direct decoder.")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    role_history = []
    unit_history = []
    contribution_history = []
    best_r2 = -np.inf
    best_epoch = None
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_terms = model.elbo_draws(R_train)
        train_terms["xi"].retain_grad()
        loss = -train_terms["elbo"].mean()
        loss.backward()

        checkpoint = epoch == 1 or epoch % eval_every == 0
        latent_grad = (
            train_terms["xi"].grad.detach().clone() * R_train
            if checkpoint else None
        )
        base_loc_grad = (
            model.q0.loc.grad.detach().clone()
            if checkpoint else None
        )

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if not checkpoint:
            continue

        model.eval()

        with torch.no_grad():
            xi_eval, log_q_eval = model.sample_posterior(R_eval)
            log_likelihood = model.log_likelihood(xi_eval)
            log_prior = model.log_prior(xi_eval)
            pred_eval = model.decoder(X_eval, xi_eval)
            continuous = model.decoder.unpack(xi_eval)

        function = metric.function_recovery_metrics(
            signal=signal_eval,
            pred_draws=pred_eval,
            prefix="val",
            zero_tol=zero_tol,
            constant_tol=constant_tol,
        )
        roles = metric.role_diagnostics(
            decoder=model.decoder,
            xi=xi_eval,
            latent_grad=latent_grad,
            base_loc_grad=base_loc_grad,
            epoch=epoch,
        )
        units = metric.unit_path_diagnostics(
            decoder=model.decoder,
            xi=xi_eval,
            epoch=epoch,
            breakpoint_eps=breakpoint_eps,
        )
        contributions, cancellation = metric.contribution_diagnostics(
            decoder=model.decoder,
            X_grid=X_eval,
            xi=xi_eval,
            epoch=epoch,
            epsilon_C=epsilon_C,
            breakpoint_eps=breakpoint_eps,
        )

        row = {
            "epoch": epoch,
            "loss": float(loss.detach()),
            "expected_log_likelihood": float(log_likelihood.mean()),
            "expected_log_prior": float(log_prior.mean()),
            "expected_log_q": float(log_q_eval.mean()),
            "kl_q_prior": float((log_q_eval - log_prior).mean()),
            "elbo": float(
                (log_likelihood + log_prior - log_q_eval).mean()
            ),
            "expected_path_count": float(
                units["path_probability"].sum()
            ),
            "beta0_abs_mean": float(
                continuous["beta0"].abs().mean()
            ),
            "beta0_abs_median": float(
                continuous["beta0"].abs().median()
            ),
            "ell_abs_mean": float(continuous["ell"].abs().mean()),
            "ell_abs_median": float(
                continuous["ell"].abs().median()
            ),
            "grad_beta0_norm": float(
                latent_grad[:, 0].abs().mean()
            ),
            "grad_ell_norm": float(
                latent_grad[:, 1].abs().mean()
            ),
            **function,
            **{
                name: value
                for name, value in cancellation.items()
                if name != "epoch"
            },
        }

        history.append(row)
        role_history.extend(roles.to_dict("records"))
        unit_history.extend(units.to_dict("records"))
        contribution_history.extend(contributions.to_dict("records"))

        if row["val_signal_r2"] > best_r2:
            best_r2 = row["val_signal_r2"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"epoch={epoch:04d} "
            f"elbo={row['elbo']:.3f} "
            f"valR2={row['val_signal_r2']:.4f} "
            f"pathN={row['expected_path_count']:.3f} "
            f"cancel={row['cancellation_ratio_median']:.3f}"
        )

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        xi_final, log_q_final = model.sample_posterior(R_final)
        final_log_likelihood = model.log_likelihood(xi_final)
        final_log_prior = model.log_prior(xi_final)
        final_pred = model.decoder(X_final, xi_final)
        final_continuous = model.decoder.unpack(xi_final)

    final_function = metric.function_recovery_metrics(
        signal=signal_final,
        pred_draws=final_pred,
        prefix="rat",
        zero_tol=zero_tol,
        constant_tol=constant_tol,
    )
    final_role = metric.role_diagnostics(
        decoder=model.decoder,
        xi=xi_final,
        epoch=best_epoch,
    )
    final_unit = metric.unit_path_diagnostics(
        decoder=model.decoder,
        xi=xi_final,
        epoch=best_epoch,
        breakpoint_eps=breakpoint_eps,
    )
    final_contribution, final_cancellation = (
        metric.contribution_diagnostics(
            decoder=model.decoder,
            X_grid=X_eval,
            xi=xi_final,
            epoch=best_epoch,
            epsilon_C=epsilon_C,
            breakpoint_eps=breakpoint_eps,
        )
    )

    final_summary = {
        "best_epoch": best_epoch,
        "best_val_signal_r2": best_r2,
        "expected_log_likelihood": float(final_log_likelihood.mean()),
        "expected_log_prior": float(final_log_prior.mean()),
        "expected_log_q": float(log_q_final.mean()),
        "kl_q_prior": float((log_q_final - final_log_prior).mean()),
        "elbo": float(
            (
                final_log_likelihood
                + final_log_prior
                - log_q_final
            ).mean()
        ),
        "beta0_abs_mean": float(
            final_continuous["beta0"].abs().mean()
        ),
        "beta0_abs_median": float(
            final_continuous["beta0"].abs().median()
        ),
        "ell_abs_mean": float(final_continuous["ell"].abs().mean()),
        "ell_abs_median": float(
            final_continuous["ell"].abs().median()
        ),
        **final_function,
        **{
            name: value
            for name, value in final_cancellation.items()
            if name != "epoch"
        },
    }

    final = {
        "summary": final_summary,
        "xi": xi_final.detach(),
        "prediction_draws": final_pred.detach().cpu(),
        "role_metrics": final_role,
        "unit_metrics": final_unit,
        "contribution_metrics": final_contribution,
        "cancellation_metrics": final_cancellation,
    }

    if mcmc_decoder is not None and mcmc_xi is not None:
        mcmc_xi = torch.as_tensor(
            mcmc_xi,
            device=device,
            dtype=dtype,
        )
        mcmc_pred = metric.predict_draws(
            mcmc_decoder,
            X_final,
            mcmc_xi,
        )
        mcmc_function = metric.function_recovery_metrics(
            signal=signal_final,
            pred_draws=mcmc_pred,
            prefix="mcmc",
            zero_tol=zero_tol,
            constant_tol=constant_tol,
        )
        spike_summary, spike_table = metric.spike_slab_metrics(
            rat_decoder=model.decoder,
            rat_xi=xi_final,
            mcmc_decoder=mcmc_decoder,
            mcmc_xi=mcmc_xi,
            reference_threshold=reference_threshold,
            min_active_draws=min_active_draws,
            breakpoint_eps=breakpoint_eps,
        )
        mcmc_contribution, mcmc_cancellation = (
            metric.contribution_diagnostics(
                decoder=mcmc_decoder,
                X_grid=X_eval,
                xi=mcmc_xi,
                epsilon_C=epsilon_C,
                breakpoint_eps=breakpoint_eps,
            )
        )

        final_summary.update(mcmc_function)
        final_summary.update(spike_summary)
        final_summary.update({
            f"mcmc_{name}": value
            for name, value in mcmc_cancellation.items()
            if name != "epoch"
        })
        final.update({
            "mcmc_prediction_draws": mcmc_pred,
            "mcmc_role_metrics": metric.role_diagnostics(
                mcmc_decoder,
                mcmc_xi,
            ),
            "mcmc_unit_metrics": metric.unit_path_diagnostics(
                mcmc_decoder,
                mcmc_xi,
                breakpoint_eps=breakpoint_eps,
            ),
            "mcmc_contribution_metrics": mcmc_contribution,
            "mcmc_cancellation_metrics": mcmc_cancellation,
            "spike_slab_metrics": spike_table,
        })

    return {
        "model": model,
        "history": pd.DataFrame(history),
        "role_history": pd.DataFrame(role_history),
        "unit_history": pd.DataFrame(unit_history),
        "contribution_history": pd.DataFrame(contribution_history),
        "final": final,
        "config": {
            "H": int(H),
            "family": family,
            "sigma2": float(sigma2),
            "gate_roles": tuple(gate_roles),
            "init_sd": model.q0.init_sd,
            "gate_power": float(gate_power),
            "gate_tau": gate_tau,
            "K_flow": int(K_flow),
            "R_train": int(R_train),
            "R_eval": int(R_eval),
            "R_final": int(R_final),
            "eval_every": int(eval_every),
            "epochs": int(epochs),
            "seed": int(seed),
        },
    }


def train_edge_bnn(
    X_train,
    y_train,
    X_eval,
    signal_eval,
    *,
    X_final=None,
    signal_final=None,
    mcmc_decoder,
    mcmc_xi,
    input_dim=None,
    d_model=2,
    n_blocks=1,
    ffn_dims=3,
    out_dim=1,
    family="gaussian",
    sigma2=1.0,
    init_sd=None,
    K_flow=8,
    flow_type="semantic",
    flow_hidden_units=64,
    flow_hidden_layers=2,
    scale_clip=1.5,
    flow_token_dim=32,
    flow_num_heads=4,
    bounded=None,
    gate_power=2.0,
    gate_tau=1.0,
    sigmoid_params=("E", "Wout"),
    sigmoid_tau=1.0,
    attention_type="none",
    ffn_activation="relu",
    epochs=6000,
    lr=3e-4,
    R_train=64,
    R_eval=1000,
    R_final=5000,
    eval_every=250,
    grad_clip=5.0,
    active_threshold=0.5,
    sigmoid_active_threshold=0.5,
    sigmoid_zero_threshold=0.05,
    eps_w=0.05,
    eps_a=0.05,
    eps_l=0.05,
    eps_c=1e-4,
    seed=123,
):
    """Train the one-block edge model and select by validation signal R2."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = X_train.device
    dtype = X_train.dtype
    X_eval = torch.as_tensor(X_eval, device=device, dtype=dtype)
    signal_eval = torch.as_tensor(
        signal_eval,
        device=device,
        dtype=dtype,
    )
    X_final = X_eval if X_final is None else torch.as_tensor(
        X_final,
        device=device,
        dtype=dtype,
    )
    signal_final = (
        signal_eval
        if signal_final is None
        else torch.as_tensor(
            signal_final,
            device=device,
            dtype=dtype,
        )
    )

    model = LaSTBNNVI(
        X=X_train,
        y=y_train,
        input_dim=input_dim,
        d_model=d_model,
        n_blocks=n_blocks,
        ffn_dims=ffn_dims,
        out_dim=out_dim,
        family=family,
        sigma2=sigma2,
        init_sd=init_sd,
        K_flow=K_flow,
        flow_type=flow_type,
        flow_hidden_units=flow_hidden_units,
        flow_hidden_layers=flow_hidden_layers,
        scale_clip=scale_clip,
        flow_token_dim=flow_token_dim,
        flow_num_heads=flow_num_heads,
        bounded=bounded,
        gate_power=gate_power,
        gate_tau=gate_tau,
        sigmoid_params=sigmoid_params,
        sigmoid_tau=sigmoid_tau,
        attention_type=attention_type,
        ffn_activation=ffn_activation,
    ).to(device)

    if mcmc_decoder is not None:
        mcmc_specs = [
            (item["name"], tuple(item["shape"]))
            for item in mcmc_decoder.param_specs
        ]
        rat_specs = [
            (item["name"], tuple(item["shape"]))
            for item in model.decoder.param_specs
        ]

        if (
            mcmc_specs != rat_specs
            or mcmc_decoder.sigmoid_params
            != model.decoder.sigmoid_params
            or mcmc_decoder.sigmoid_tau != model.decoder.sigmoid_tau
            or mcmc_decoder.gate_power != model.decoder.gate_power
            or mcmc_decoder.gate_tau != model.decoder.gate_tau
        ):
            raise ValueError("MCMC and RaT must use the same edge decoder.")

    mcmc_xi = torch.as_tensor(
        mcmc_xi,
        device=device,
        dtype=dtype,
    )
    mcmc_post = metric.posterior_draws(
        mcmc_decoder,
        mcmc_xi,
        sigmoid_active_threshold=sigmoid_active_threshold,
    )
    mcmc_pred_eval = metric.predict_draws(
        mcmc_decoder,
        X_eval,
        mcmc_xi,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    layer_history = []
    path_history = []
    best_r2 = -np.inf
    best_epoch = None
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_terms = model.elbo_draws(R_train)
        loss = -train_terms["elbo"].mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if epoch != 1 and epoch % eval_every != 0:
            continue

        model.eval()

        with torch.no_grad():
            xi_eval, log_q_eval = model.sample_posterior(R_eval)
            log_likelihood = model.log_likelihood(xi_eval)
            log_prior = model.log_prior(xi_eval)

        rat_post = metric.posterior_draws(
            model.decoder,
            xi_eval,
            sigmoid_active_threshold=sigmoid_active_threshold,
        )
        recovery, layers = metric.posterior_metrics(
            last=rat_post,
            mcmc=mcmc_post,
            active_threshold=active_threshold,
        )
        rat_pred_eval = metric.predict_draws(
            model.decoder,
            X_eval,
            xi_eval,
        )
        function = metric.function_metrics(
            signal=signal_eval,
            mcmc_pred_draws=mcmc_pred_eval,
            last_pred_draws=rat_pred_eval,
        )
        paths = metric.residual_path_metrics(
            decoder=model.decoder,
            xi=xi_eval,
            x_grid=X_eval,
            method="RaT",
            eps_w=eps_w,
            eps_a=eps_a,
            eps_l=eps_l,
            eps_c=eps_c,
            sigmoid_zero_threshold=sigmoid_zero_threshold,
        )
        path_summary = paths["summary"].iloc[0].to_dict()
        hidden = layers["parameter"].str.match(r"W1_|b1_|W2_|b2_")

        row = {
            "epoch": epoch,
            "loss": float(loss.detach()),
            "expected_log_likelihood": float(log_likelihood.mean()),
            "expected_log_prior": float(log_prior.mean()),
            "expected_log_q": float(log_q_eval.mean()),
            "kl_q_prior": float((log_q_eval - log_prior).mean()),
            "elbo": float(
                (log_likelihood + log_prior - log_q_eval).mean()
            ),
            "hidden_a_skl": float(
                np.nanmedian(layers.loc[hidden, "a_skl"])
            ),
            "hidden_pip_rmse": float(
                np.sqrt(np.nanmean(layers.loc[hidden, "pip_rmse"] ** 2))
            ),
            **recovery,
            **function,
            **{
                name: value
                for name, value in path_summary.items()
                if name != "method"
            },
        }

        history.append(row)
        layer_history.extend(
            layers.assign(epoch=epoch).to_dict("records")
        )
        path_history.extend(
            paths["units"].assign(epoch=epoch).to_dict("records")
        )

        if row["last_signal_r2"] > best_r2:
            best_r2 = row["last_signal_r2"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"epoch={epoch:04d} "
            f"valR2={row['last_signal_r2']:.4f} "
            f"hiddenSKL={row['hidden_a_skl']:.4f} "
            f"pathN={row['expected_functional_paths']:.3f} "
            f"zeroPath={row['zero_functional_path_prob']:.3f}"
        )

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        xi_final, _ = model.sample_posterior(R_final)

    bundle = metric.bnn_metrics(
        mcmc_decoder=mcmc_decoder,
        last_decoder=model.decoder,
        mcmc_xi=mcmc_xi,
        last_xi=xi_final,
        X=X_final,
        signal=signal_final,
        active_threshold=active_threshold,
        sigmoid_active_threshold=sigmoid_active_threshold,
    )
    rat_paths = metric.residual_path_metrics(
        decoder=model.decoder,
        xi=xi_final,
        x_grid=X_eval,
        method="RaT",
        eps_w=eps_w,
        eps_a=eps_a,
        eps_l=eps_l,
        eps_c=eps_c,
        sigmoid_zero_threshold=sigmoid_zero_threshold,
    )
    mcmc_paths = metric.residual_path_metrics(
        decoder=mcmc_decoder,
        xi=mcmc_xi,
        x_grid=X_eval,
        method="MCMC",
        eps_w=eps_w,
        eps_a=eps_a,
        eps_l=eps_l,
        eps_c=eps_c,
        sigmoid_zero_threshold=sigmoid_zero_threshold,
    )
    rat_path_summary = rat_paths["summary"].iloc[0].to_dict()
    mcmc_path_summary = mcmc_paths["summary"].iloc[0].to_dict()
    summary = {
        "best_epoch": best_epoch,
        "best_val_signal_r2": best_r2,
        **bundle["summary"],
        **{
            name: value
            for name, value in rat_path_summary.items()
            if name != "method"
        },
        **{
            f"mcmc_{name}": value
            for name, value in mcmc_path_summary.items()
            if name != "method"
        },
    }

    return {
        "model": model,
        "history": pd.DataFrame(history),
        "layer_history": pd.DataFrame(layer_history),
        "path_history": pd.DataFrame(path_history),
        "final": {
            "summary": summary,
            "xi": xi_final.detach(),
            "posterior_by_layer": bundle["posterior_by_layer"],
            "connection_counts": bundle["connection_counts"],
            "hidden_units": bundle["hidden_units"],
            "mcmc_prediction_draws": bundle["mcmc_pred_draws"],
            "rat_prediction_draws": bundle["last_pred_draws"],
            "path_summary": pd.concat(
                [rat_paths["summary"], mcmc_paths["summary"]],
                ignore_index=True,
            ),
            "path_units": pd.concat(
                [rat_paths["units"], mcmc_paths["units"]],
                ignore_index=True,
            ),
        },
        "config": {
            "input_dim": model.decoder.input_dim,
            "d_model": int(d_model),
            "n_blocks": int(n_blocks),
            "ffn_dims": ffn_dims,
            "sigmoid_params": tuple(sigmoid_params),
            "sigmoid_tau": float(sigmoid_tau),
            "sigmoid_zero_threshold": float(sigmoid_zero_threshold),
            "init_sd": model.q0.init_sd,
            "K_flow": int(K_flow),
            "flow_type": flow_type,
            "R_train": int(R_train),
            "R_eval": int(R_eval),
            "R_final": int(R_final),
            "eval_every": int(eval_every),
            "epochs": int(epochs),
            "seed": int(seed),
        },
    }


def train_grouped_bnn(
    X_train,
    y_train,
    X_eval,
    signal_eval,
    *,
    mcmc_decoder,
    mcmc_xi,
    truth,
    X_final=None,
    signal_final=None,
    selection_mode="unit_group",
    input_dim=None,
    H=5,
    out_dim=1,
    family="gaussian",
    sigma2=1.0,
    init_sd=None,
    K_flow=8,
    flow_type="attention_affine",
    flow_hidden_units=64,
    flow_hidden_layers=2,
    scale_clip=1.5,
    flow_token_dim=32,
    flow_num_heads=4,
    flow_mask_seed=None,
    gate_power=1.0,
    gate_tau=None,
    repu_power=None,
    linear_skip=False,
    epochs=6000,
    lr=3e-4,
    R_train=64,
    R_eval=1000,
    R_final=5000,
    eval_every=250,
    grad_clip=5.0,
    min_active_draws=50,
    zero_tol=1e-6,
    constant_tol=1e-6,
    seed=123,
):
    """Train the single-group/single-ReLU-gate shallow BNN."""

    if selection_mode not in {"unit_group", "feature_group"}:
        raise ValueError("Use train_edge_bnn for selection_mode='edge'.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if flow_mask_seed is None:
        flow_mask_seed = int(seed)

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
    mcmc_xi = torch.as_tensor(mcmc_xi, device=device, dtype=dtype)

    model = GroupedBNNVI(
        X=X_train,
        y=y_train,
        input_dim=input_dim,
        H=H,
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
        flow_token_dim=flow_token_dim,
        flow_num_heads=flow_num_heads,
        flow_mask_seed=flow_mask_seed,
        gate_power=gate_power,
        gate_tau=gate_tau,
        repu_power=repu_power,
        linear_skip=linear_skip,
    ).to(device)

    if model.decoder.compatibility_signature() != mcmc_decoder.compatibility_signature():
        raise ValueError("MCMC and VI must use exactly the same grouped decoder.")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    group_history = []
    unit_history = []
    best_r2 = -np.inf
    best_epoch = None
    best_state = None

    for epoch in range(1, int(epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_terms = model.elbo_draws(R_train)
        loss = -train_terms["elbo"].mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if epoch != 1 and epoch % int(eval_every) != 0:
            continue

        model.eval()
        with torch.no_grad():
            xi_eval, log_q_eval = model.sample_posterior(R_eval)
            log_likelihood = model.log_likelihood(xi_eval)
            log_prior = model.log_prior(xi_eval)
            pred_eval = model.decoder(X_eval, xi_eval)

        function = metric.function_recovery_metrics(
            signal=signal_eval,
            pred_draws=pred_eval,
            prefix="val",
            zero_tol=zero_tol,
            constant_tol=constant_tol,
        )
        recovery, _ = metric.grouped_recovery_metrics(
            rat_decoder=model.decoder,
            rat_xi=xi_eval,
            mcmc_decoder=mcmc_decoder,
            mcmc_xi=mcmc_xi,
            truth=truth,
            min_active_draws=min_active_draws,
        )
        groups = metric.group_posterior_summary(
            model.decoder, xi_eval, method="RaT", epoch=epoch
        )
        units = metric.unit_group_summary(
            model.decoder, xi_eval, method="RaT", epoch=epoch
        )
        row = {
            "epoch": epoch,
            "loss": float(loss.detach()),
            "expected_log_likelihood": float(log_likelihood.mean()),
            "expected_log_prior": float(log_prior.mean()),
            "expected_log_q": float(log_q_eval.mean()),
            "kl_q_prior": float((log_q_eval - log_prior).mean()),
            "elbo": float((log_likelihood + log_prior - log_q_eval).mean()),
            **function,
            **recovery,
        }
        history.append(row)
        group_history.extend(groups.to_dict("records"))
        unit_history.extend(units.to_dict("records"))

        if row["val_signal_r2"] > best_r2:
            best_r2 = row["val_signal_r2"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"epoch={epoch:04d} "
            f"valR2={row['val_signal_r2']:.4f} "
            f"pipRMSE={row['pip_rmse_mcmc']:.4f} "
            f"trueSKL={row['true_active_skl']:.4f} "
            f"zeroJS={row['zero_js']:.4f}"
        )

    if best_state is None:
        raise RuntimeError("No evaluation checkpoint was produced.")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        xi_final, log_q_final = model.sample_posterior(R_final)
        final_log_likelihood = model.log_likelihood(xi_final)
        final_log_prior = model.log_prior(xi_final)
        rat_pred = metric.predict_draws(model.decoder, X_final, xi_final)
        mcmc_pred = metric.predict_draws(mcmc_decoder, X_final, mcmc_xi)

    recovery, recovery_table = metric.grouped_recovery_metrics(
        rat_decoder=model.decoder,
        rat_xi=xi_final,
        mcmc_decoder=mcmc_decoder,
        mcmc_xi=mcmc_xi,
        truth=truth,
        min_active_draws=min_active_draws,
    )
    rat_function = metric.function_recovery_metrics(
        signal=signal_final,
        pred_draws=rat_pred,
        prefix="rat",
        zero_tol=zero_tol,
        constant_tol=constant_tol,
    )
    mcmc_function = metric.function_recovery_metrics(
        signal=signal_final,
        pred_draws=mcmc_pred,
        prefix="mcmc",
        zero_tol=zero_tol,
        constant_tol=constant_tol,
    )
    summary = {
        "best_epoch": best_epoch,
        "best_val_signal_r2": best_r2,
        "expected_log_likelihood": float(final_log_likelihood.mean()),
        "expected_log_prior": float(final_log_prior.mean()),
        "expected_log_q": float(log_q_final.mean()),
        "kl_q_prior": float((log_q_final - final_log_prior).mean()),
        "elbo": float(
            (final_log_likelihood + final_log_prior - log_q_final).mean()
        ),
        **recovery,
        **rat_function,
        **mcmc_function,
    }

    rat_groups = metric.group_posterior_summary(
        model.decoder, xi_final, method="RaT", epoch=best_epoch
    )
    mcmc_groups = metric.group_posterior_summary(
        mcmc_decoder, mcmc_xi, method="MCMC"
    )
    rat_units = metric.unit_group_summary(
        model.decoder, xi_final, method="RaT", epoch=best_epoch
    )
    mcmc_units = metric.unit_group_summary(
        mcmc_decoder, mcmc_xi, method="MCMC"
    )

    return {
        "model": model,
        "history": pd.DataFrame(history),
        "group_history": pd.DataFrame(group_history),
        "unit_history": pd.DataFrame(unit_history),
        "final": {
            "summary": summary,
            "xi": xi_final.detach(),
            "rat_prediction_draws": rat_pred,
            "mcmc_prediction_draws": mcmc_pred,
            "recovery_by_target": recovery_table,
            "group_metrics": pd.concat(
                [rat_groups, mcmc_groups], ignore_index=True
            ),
            "unit_metrics": pd.concat(
                [rat_units, mcmc_units], ignore_index=True
            ) if not rat_units.empty else pd.DataFrame(),
        },
        "config": {
            "selection_mode": selection_mode,
            "input_dim": model.decoder.input_dim,
            "H": int(model.decoder.H),
            "out_dim": int(out_dim),
            "linear_skip": bool(linear_skip),
            "flow_type": flow_type,
            "K_flow": int(K_flow),
            "flow_token_dim": int(flow_token_dim),
            "flow_num_heads": int(flow_num_heads),
            "flow_mask_seed": int(flow_mask_seed),
            "repu_power": repu_power,
            "gate_power": float(gate_power),
            "gate_tau": gate_tau,
            "init_sd": model.q0.init_sd,
            "R_train": int(R_train),
            "R_eval": int(R_eval),
            "R_final": int(R_final),
            "epochs": int(epochs),
            "seed": int(seed),
        },
    }


def train_bnn(*args, selection_mode="unit_group", **kwargs):
    """Unified training dispatcher while retaining the legacy edge baseline."""

    if selection_mode == "edge":
        return train_edge_bnn(*args, **kwargs)
    return train_grouped_bnn(
        *args,
        selection_mode=selection_mode,
        **kwargs,
    )