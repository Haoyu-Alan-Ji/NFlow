import copy
import random
import numpy as np
import pandas as pd
import torch

from .direct_bnn import DirectUnitBNNVI, ROLE_NAMES
from . import direct_metric as metric


def train_direct_bnn(
    X_train,
    y_train,
    X_eval,
    signal_eval,
    *,
    X_final=None,
    signal_final=None,
    mcmc_decoder=None,
    mcmc_xi=None,
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
    """
    Train one direct unit-level RaT model and select the checkpoint that
    maximizes validation-grid signal R2.

    gate_roles controls the isolation experiment:
        ()                                  no-gate direct BNN
        ("input", "breakpoint", "output")   all roles learned
        ("breakpoint", "output")            input gate fixed open
        ("input", "breakpoint")             output gate fixed open
        ("input", "output")                 breakpoint gate fixed open
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = X_train.device
    dtype = X_train.dtype
    X_eval = torch.as_tensor(
        X_eval,
        device=device,
        dtype=dtype,
    )
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
        raise ValueError(
            "MCMC and RaT must use the same direct decoder and gate."
        )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

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
            if checkpoint
            else None
        )
        base_loc_grad = (
            model.q0.loc.grad.detach().clone()
            if checkpoint
            else None
        )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            grad_clip,
        )
        optimizer.step()

        if checkpoint:
            model.eval()

            with torch.no_grad():
                xi_eval, log_q_eval = model.sample_posterior(R_eval)
                log_likelihood = model.log_likelihood(xi_eval)
                log_prior = model.log_prior(xi_eval)
                kl = log_q_eval - log_prior
                pred_eval = model.decoder(X_eval, xi_eval)
                continuous = model.decoder.unpack(xi_eval)

            function = metric.function_metrics(
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
            contributions, cancellation = (
                metric.contribution_diagnostics(
                    decoder=model.decoder,
                    X_grid=X_eval,
                    xi=xi_eval,
                    epoch=epoch,
                    epsilon_C=epsilon_C,
                    breakpoint_eps=breakpoint_eps,
                )
            )

            row = {
                "epoch": epoch,
                "loss": float(loss.detach()),
                "expected_log_likelihood": float(
                    log_likelihood.mean()
                ),
                "expected_log_prior": float(log_prior.mean()),
                "expected_log_q": float(log_q_eval.mean()),
                "kl_q_prior": float(kl.mean()),
                "elbo": float(
                    (
                        log_likelihood
                        + log_prior
                        - log_q_eval
                    ).mean()
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
                "ell_abs_mean": float(
                    continuous["ell"].abs().mean()
                ),
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
            contribution_history.extend(
                contributions.to_dict("records")
            )

            if row["val_signal_r2"] > best_r2:
                best_r2 = row["val_signal_r2"]
                best_epoch = epoch
                best_state = copy.deepcopy(
                    model.state_dict()
                )

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

    final_function = metric.function_metrics(
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
        "expected_log_likelihood": float(
            final_log_likelihood.mean()
        ),
        "expected_log_prior": float(final_log_prior.mean()),
        "expected_log_q": float(log_q_final.mean()),
        "kl_q_prior": float(
            (log_q_final - final_log_prior).mean()
        ),
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
        "ell_abs_mean": float(
            final_continuous["ell"].abs().mean()
        ),
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
        mcmc_function = metric.function_metrics(
            signal=signal_final,
            pred_draws=mcmc_pred,
            prefix="mcmc",
            zero_tol=zero_tol,
            constant_tol=constant_tol,
        )
        spike_slab_summary, spike_slab_table = (
            metric.spike_slab_metrics(
                rat_decoder=model.decoder,
                rat_xi=xi_final,
                mcmc_decoder=mcmc_decoder,
                mcmc_xi=mcmc_xi,
                reference_threshold=reference_threshold,
                min_active_draws=min_active_draws,
                breakpoint_eps=breakpoint_eps,
            )
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
        final_summary.update(spike_slab_summary)
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
            "spike_slab_metrics": spike_slab_table,
        })

    return {
        "model": model,
        "history": pd.DataFrame(history),
        "role_history": pd.DataFrame(role_history),
        "unit_history": pd.DataFrame(unit_history),
        "contribution_history": pd.DataFrame(
            contribution_history
        ),
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