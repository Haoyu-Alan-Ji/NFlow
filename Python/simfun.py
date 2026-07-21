from __future__ import annotations

from typing import Any, Dict, Mapping
import math
import numpy as np
import torch
import torch.nn.functional as F

def simfun1(
    sim="simple",
    n=180,
    p=100,
    seed=123,
    n_active=10,
    sigma2=1.0,
    beta_low=0.3,
    beta_high=2.0,
    rho=0.8,
    block_size=10,
    group_size=10,
    noise_x=0.15,
    one_active_per_group=True,
    center_y=True,
    device=None,
    dtype=torch.float32,
):
    """
    Unified sparse linear regression simulation.

    Data-generating model:
        y = X beta + eps,
        eps ~ N(0, sigma2).

    Available simulation settings:
        sim = "simple"
            Independent standardized Gaussian predictors.

        sim = "block_corr"
            Block-wise AR(1) correlated predictors.

        sim = "group_competition"
            Within-group near-duplicate predictors, creating
            one-of-K variable-selection ambiguity.

    Parameters
    ----------
    n : int
        Number of observations.

    p : int
        Number of predictors.

    seed : int
        Random seed.

    n_active : int
        Number of truly active variables.

    sigma2 : float
        Noise variance.

    beta_low, beta_high : float
        Active coefficient magnitudes are sampled uniformly from
        [beta_low, beta_high], with random signs.

    rho : float
        Within-block AR(1) correlation for sim="block_corr".

    block_size : int
        Block size for sim="block_corr".

    group_size : int
        Group size for sim="group_competition".

    noise_x : float
        Within-group idiosyncratic noise level for sim="group_competition".
        Smaller values create stronger within-group competition.

    one_active_per_group : bool
        If True, active variables are chosen from distinct groups whenever possible
        for sim="group_competition".

    center_y : bool
        If True, center y by subtracting its sample mean.

    device, dtype :
        Torch device and dtype.

    Returns
    -------
    X : torch.Tensor, shape (n, p)
    y : torch.Tensor, shape (n,)
    beta_true : torch.Tensor, shape (p,)
    info : dict
    """

    if device is None:
        device = torch.device("cpu")

    rng = np.random.default_rng(seed)

    if n_active > p:
        raise ValueError("n_active cannot be larger than p.")

    if sigma2 <= 0:
        raise ValueError("sigma2 must be positive.")

    if beta_low <= 0 or beta_high <= 0 or beta_low > beta_high:
        raise ValueError("Require 0 < beta_low <= beta_high.")

    # --------------------------------------------------
    # 1. Generate design matrix X
    # --------------------------------------------------
    if sim == "simple":
        X_np = rng.standard_normal((n, p)).astype(np.float32)

    elif sim == "block_corr":
        blocks = []
        remaining = p

        while remaining > 0:
            b = min(block_size, remaining)

            idx = np.arange(b)
            Sigma = rho ** np.abs(idx[:, None] - idx[None, :])

            X_block = rng.multivariate_normal(
                mean=np.zeros(b),
                cov=Sigma,
                size=n,
            )

            blocks.append(X_block)
            remaining -= b

        X_np = np.concatenate(blocks, axis=1).astype(np.float32)

    elif sim == "group_competition":
        n_groups = int(np.ceil(p / group_size))

        groups = []
        start = 0
        for g in range(n_groups):
            end = min(start + group_size, p)
            groups.append(np.arange(start, end))
            start = end

        X_np = np.zeros((n, p), dtype=np.float32)

        for g, cols in enumerate(groups):
            z_g = rng.normal(loc=0.0, scale=1.0, size=(n, 1))
            eta_g = rng.normal(loc=0.0, scale=1.0, size=(n, len(cols)))

            X_np[:, cols] = z_g + noise_x * eta_g

    else:
        raise ValueError(
            "Unknown sim. Use one of: 'simple', 'block_corr', 'group_competition'."
        )

    # Standardize columns.
    X_np = X_np - X_np.mean(axis=0, keepdims=True)
    X_np = X_np / (X_np.std(axis=0, ddof=0, keepdims=True) + 1e-12)

    # --------------------------------------------------
    # 2. Choose active variables
    # --------------------------------------------------
    if sim == "group_competition" and one_active_per_group:
        n_groups = int(np.ceil(p / group_size))

        groups = []
        start = 0
        for g in range(n_groups):
            end = min(start + group_size, p)
            groups.append(np.arange(start, end))
            start = end

        n_active_groups = min(n_active, n_groups)
        active_groups = rng.choice(n_groups, size=n_active_groups, replace=False)

        active_list = []
        for g in active_groups:
            cols = groups[g]
            active_list.append(rng.choice(cols))

        if n_active > n_active_groups:
            remaining_candidates = np.setdiff1d(
                np.arange(p),
                np.asarray(active_list, dtype=int),
            )
            extra = rng.choice(
                remaining_candidates,
                size=n_active - n_active_groups,
                replace=False,
            )
            active_list.extend(extra.tolist())

        active_idx = np.sort(np.asarray(active_list, dtype=int))

    else:
        active_idx = np.sort(rng.choice(p, size=n_active, replace=False))

    # --------------------------------------------------
    # 3. Generate sparse beta
    # --------------------------------------------------
    beta_np = np.zeros(p, dtype=np.float32)

    signs = rng.choice([-1.0, 1.0], size=len(active_idx)).astype(np.float32)
    mags = rng.uniform(beta_low, beta_high, size=len(active_idx)).astype(np.float32)

    beta_np[active_idx] = signs * mags

    # Active variables sorted by decreasing absolute true effect size.
    active_order = active_idx[np.argsort(-np.abs(beta_np[active_idx]))]

    active_table = [
        {
            "rank": int(r + 1),
            "j": int(j),                 # zero-based index
            "j1": int(j + 1),             # one-based index, useful for R/output
            "beta_true": float(beta_np[j]),
            "abs_beta_true": float(abs(beta_np[j])),
        }
        for r, j in enumerate(active_order)
    ]

    # --------------------------------------------------
    # 4. Generate y
    # --------------------------------------------------
    signal = X_np @ beta_np

    sigma = float(np.sqrt(sigma2))
    eps = sigma * rng.standard_normal(n).astype(np.float32)

    y_np = signal + eps

    if center_y:
        y_np = y_np - y_np.mean()

    # --------------------------------------------------
    # 5. Diagnostics
    # --------------------------------------------------
    signal_var = float(np.var(signal, ddof=0))
    outcome_var = float(np.var(y_np, ddof=0))
    snr_actual = signal_var / float(sigma2)

    # --------------------------------------------------
    # 6. Convert to torch tensors
    # --------------------------------------------------
    X = torch.as_tensor(X_np, dtype=dtype, device=device)
    y = torch.as_tensor(y_np, dtype=dtype, device=device)
    beta_true = torch.as_tensor(beta_np, dtype=dtype, device=device)

    # --------------------------------------------------
    # 7. Info dictionary
    # --------------------------------------------------
    info = {
        "sim": sim,
        "n": int(n),
        "p": int(p),
        "n_active": int(len(active_idx)),

        # Original active set, zero-based.
        "active_idx": active_idx,

        # Active variables sorted by |beta_true|, zero-based.
        "active_idx_by_abs": active_order,

        # A cleaner table with rank, index, and true coefficient.
        "active_table": active_table,

        "sigma2": float(sigma2),
        "sigma": float(sigma),
        "signal_var": signal_var,
        "outcome_var": outcome_var,
        "snr_actual": float(snr_actual),
        "beta_low": float(beta_low),
        "beta_high": float(beta_high),
        "center_y": bool(center_y),
    }

    if sim == "block_corr":
        info.update({
            "rho": float(rho),
            "block_size": int(block_size),
        })

    if sim == "group_competition":
        group_id = np.empty(p, dtype=int)
        for g, cols in enumerate(groups):
            group_id[cols] = g

        active_groups = np.unique(group_id[active_idx])

        info.update({
            "group_size": int(group_size),
            "n_groups": int(len(groups)),
            "noise_x": float(noise_x),
            "one_active_per_group": bool(one_active_per_group),
            "group_id": group_id,
            "active_groups": active_groups,
        })

    return X, y, beta_true, info


def print_siminfo(sim_info, digits=4):
    """
    Format simulation information in a readable multi-line style.
    """
    if sim_info is None:
        return "sim_info = None"

    lines = []

    # Basic scalar fields.
    scalar_keys = [
        "sim",
        "setting",
        "seed",
        "n",
        "p",
        "n_active",
        "sigma2",
        "sigma",
        "signal_var",
        "outcome_var",
        "snr_actual",
        "beta_low",
        "beta_high",
        "center_y",
        "rho",
        "block_size",
        "group_size",
        "n_groups",
        "noise_x",
        "one_active_per_group",
    ]

    lines.append("Simulation summary:")
    for key in scalar_keys:
        if key in sim_info:
            val = sim_info[key]
            if isinstance(val, float):
                lines.append(f"  {key:<22}: {val:.{digits}f}")
            else:
                lines.append(f"  {key:<22}: {val}")

    # Original active indices.
    if "active_idx" in sim_info:
        active_idx = sim_info["active_idx"]
        if hasattr(active_idx, "tolist"):
            active_idx = active_idx.tolist()
        lines.append("")
        lines.append("Active indices, zero-based:")
        lines.append(f"  {active_idx}")

    # Sorted active table.
    if "active_table" in sim_info:
        lines.append("")
        lines.append("Active variables sorted by |beta_true|:")
        lines.append(f"  {'rank':>4} {'j':>5} {'j1':>5} {'beta_true':>12} {'|beta|':>12}")

        for row in sim_info["active_table"]:
            lines.append(
                f"  {row['rank']:>4d} "
                f"{row['j']:>5d} "
                f"{row['j1']:>5d} "
                f"{row['beta_true']:>12.{digits}f} "
                f"{row['abs_beta_true']:>12.{digits}f}"
            )

    return "\n".join(lines)


def simfun_nonlinear(
    n=160,
    p=1,
    n_active=1,
    interaction=False,
    seed=123,
    device=None,
    dtype=torch.float32,
):
    """
    Nonlinear Gaussian regression simulation.

    X_ij ~ Uniform(-pi, pi)

    y_i = f(X_i) + eps_i
    eps_i ~ N(0, 1)

    The first n_active features are active.

    Main-effect functions:
        feature 1: cos(x)
        feature 2: sin(x)
        feature 3: sin(x)^2
        feature 4: cos(x)^2

    For more than four active features, function types are sampled
    from {cos, sin, sin2, cos2}.

    When interaction=True, pairwise products of active terms are
    randomly included, with at least one interaction.
    """

    if device is None:
        device = torch.device("cpu")

    if n_active < 1 or n_active > p:
        raise ValueError("n_active must be between 1 and p.")

    if interaction and n_active < 2:
        raise ValueError("interaction=True requires n_active >= 2.")

    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    X = (
        2.0 * math.pi
        * torch.rand(
            n,
            p,
            generator=gen,
            device=device,
            dtype=dtype,
        )
        - math.pi
    )

    choices = ["cos", "sin", "sin2", "cos2"]
    term_types = choices[:min(n_active, 4)]

    if n_active > 4:
        term_types += rng.choice(
            choices,
            size=n_active - 4,
            replace=True,
        ).tolist()

    terms = []
    term_names = []

    for j, kind in enumerate(term_types):
        x = X[:, j]

        if kind == "cos":
            term = torch.cos(x)
            name = f"cos(x{j + 1})"

        elif kind == "sin":
            term = torch.sin(x)
            name = f"sin(x{j + 1})"

        elif kind == "sin2":
            term = torch.sin(x).square()
            name = f"sin(x{j + 1})^2"

        else:
            term = torch.cos(x).square()
            name = f"cos(x{j + 1})^2"

        terms.append(term)
        term_names.append(name)

    signal = torch.stack(terms).sum(dim=0)

    interaction_pairs = []

    if interaction:
        all_pairs = [
            (j, k)
            for j in range(n_active)
            for k in range(j + 1, n_active)
        ]

        keep = rng.integers(
            0,
            2,
            size=len(all_pairs),
        ).astype(bool)

        if not keep.any():
            keep[rng.integers(len(all_pairs))] = True

        interaction_pairs = [
            pair
            for pair, selected in zip(all_pairs, keep)
            if selected
        ]

        for j, k in interaction_pairs:
            signal = signal + terms[j] * terms[k]

    noise = torch.randn(
        n,
        generator=gen,
        device=device,
        dtype=dtype,
    )

    y = signal + noise

    feature_true = torch.zeros(
        p,
        device=device,
        dtype=dtype,
    )
    feature_true[:n_active] = 1.0

    function_terms = term_names.copy()

    for j, k in interaction_pairs:
        function_terms.append(
            f"({term_names[j]})({term_names[k]})"
        )

    function_text = " + ".join(function_terms)

    signal_var = float(
        signal.var(unbiased=False).item()
    )

    outcome_var = float(
        y.var(unbiased=False).item()
    )

    sim_info = {
        "sim": "nonlinear_trigonometric",
        "seed": int(seed),
        "family": "gaussian",
        "n": int(n),
        "p": int(p),
        "n_active": int(n_active),
        "active_idx": np.arange(n_active),
        "term_types": term_types,
        "interaction": bool(interaction),
        "interaction_pairs": [
            (j + 1, k + 1)
            for j, k in interaction_pairs
        ],
        "n_interactions": len(interaction_pairs),
        "sigma2": 1.0,
        "sigma": 1.0,
        "signal_var": signal_var,
        "outcome_var": outcome_var,
        "snr": signal_var,
        "x_distribution": "Uniform(-pi, pi)",
        "function": function_text,
    }

    return X, y, feature_true, signal, sim_info


def siminfo(sim_info, digits=4):
    lines = ["Nonlinear simulation summary:"]

    keys = [
        "sim",
        "seed",
        "family",
        "n",
        "p",
        "n_active",
        "sigma2",
        "sigma",
        "signal_var",
        "outcome_var",
        "snr",
        "interaction",
        "n_interactions",
        "x_distribution",
    ]

    for key in keys:
        value = sim_info[key]

        if isinstance(value, float):
            lines.append(
                f"  {key:<20}: {value:.{digits}f}"
            )
        else:
            lines.append(
                f"  {key:<20}: {value}"
            )

    lines.append("")
    lines.append("Active features, zero-based:")
    lines.append(
        f"  {sim_info['active_idx'].tolist()}"
    )

    lines.append("")
    lines.append("Active feature functions:")

    for j, kind in enumerate(sim_info["term_types"]):
        lines.append(
            f"  x{j + 1:<3}: {kind}"
        )

    lines.append("")
    lines.append("Interaction pairs, one-based:")
    lines.append(
        f"  {sim_info['interaction_pairs']}"
    )

    lines.append("")
    lines.append("Generating function:")
    lines.append(
        f"  f(x) = {sim_info['function']}"
    )

    return "\n".join(lines)