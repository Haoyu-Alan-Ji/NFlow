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



def bnn_attention(x, attention_type="self"):
    """
    Parameter-free attention used by both simulator and decoder.

    x shape:
        n x d

    returns:
        n x d
    """

    if attention_type == "self":
        scale = math.sqrt(float(x.shape[-1]))
        score = x @ x.T / scale
        weight = torch.softmax(score, dim=-1)
        return weight @ x

    if attention_type == "feature":
        weight = torch.softmax(x, dim=-1)
        return weight * x

    if attention_type == "identity":
        return x

    raise ValueError("attention_type must be self, feature, or identity.")


def bnn_activate(x, ffn_activation="relu"):
    if ffn_activation == "gelu":
        return F.gelu(x)

    if ffn_activation == "tanh":
        return torch.tanh(x)

    return F.relu(x)


def bnn_truth_forward(
    X,
    truth,
    n_blocks,
    attention_type="self",
    ffn_activation="relu",
):
    """
    Forward pass for the sparse attention-FFN oracle.

    Structure:
        z0 = X E^T + e

        z_{k+1}
        =
        z_k
        +
        activation(Attn_k(z_k) W1_k^T + b1_k) W2_k^T + b2_k

        eta = z_K Wout^T + bout
    """

    E = truth["E"]
    e = truth["e"]

    z = X @ E.T + e

    for k in range(int(n_blocks)):
        W1 = truth[f"W1_{k}"]
        b1 = truth[f"b1_{k}"]
        W2 = truth[f"W2_{k}"]
        b2 = truth[f"b2_{k}"]

        att = bnn_attention(z, attention_type=attention_type)

        hidden = att @ W1.T + b1
        hidden = bnn_activate(hidden, ffn_activation=ffn_activation)

        delta = hidden @ W2.T + b2

        z = z + delta

    signal = z @ truth["Wout"].T + truth["bout"]

    if signal.shape[-1] == 1:
        return signal[:, 0]

    return signal


def simfun_bnn(
    n=240,
    p=20,
    seed=123,
    family="gaussian",
    sigma2=0.25,
    layer_dims=None,
    d_model=8,
    n_blocks=2,
    ffn_dims=None,
    out_dim=1,
    n_active=4,
    active_idx=None,
    n_active_state=None,
    n_active_ff=None,
    weight_low=0.35,
    weight_high=0.80,
    bias_sd=0.20,
    bounded=None,
    center_y=True,
    attention_type="self",
    ffn_activation="relu",
    device=None,
    dtype=torch.float32,
):
    """
    Sparse attention-FFN BNN oracle simulation.

    Data-generating model:

        X_i ~ N(0, I_p)

        z0 = X E^T + e

        z_{k+1}
        =
        z_k
        +
        activation(Attn_k(z_k) W1_k^T + b1_k) W2_k^T + b2_k

        eta = z_K Wout^T + bout

    Families:

        gaussian:
            y = eta + eps, eps ~ N(0, sigma2)

        bernoulli / logistic / binomial:
            y ~ Bernoulli(sigmoid(eta))

        poisson:
            y ~ Poisson(exp(eta))

    Sparse truth:

        E, e,
        W1_k, b1_k, W2_k, b2_k,
        Wout, bout

    All true parameters are ordinary tensors.
    DSS variables s, u, t are not generated here.
    """

    if device is None:
        device = torch.device("cpu")

    rng = np.random.default_rng(seed)

    if layer_dims is not None:
        layer_dims = [int(x) for x in layer_dims]

        if len(layer_dims) < 3:
            raise ValueError("layer_dims should look like [p, d_model, out_dim].")

        if int(layer_dims[0]) != int(p):
            raise ValueError("layer_dims[0] must equal p.")

        d_model = int(layer_dims[1])
        out_dim = int(layer_dims[-1])

    d_model = int(d_model)
    n_blocks = int(n_blocks)
    out_dim = int(out_dim)

    if out_dim != 1:
        raise ValueError("This simulator currently assumes scalar output.")

    if ffn_dims is None:
        ffn_dims = [d_model for _ in range(n_blocks)]
    elif isinstance(ffn_dims, int):
        ffn_dims = [int(ffn_dims) for _ in range(n_blocks)]
    else:
        ffn_dims = [int(x) for x in ffn_dims]

    if len(ffn_dims) != n_blocks:
        raise ValueError("ffn_dims must have length n_blocks.")

    if n_active > p:
        raise ValueError("n_active cannot exceed p.")

    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))

    # -----------------------------
    # 1. Gaussian design
    # -----------------------------

    X = torch.randn(n, p, generator=gen, device=device, dtype=dtype)

    X = X - X.mean(dim=0, keepdim=True)
    X = X / (X.std(dim=0, unbiased=False, keepdim=True) + 1e-12)

    # -----------------------------
    # 2. Sparse supports
    # -----------------------------

    if active_idx is None:
        active_idx = np.sort(rng.choice(p, size=n_active, replace=False))
    else:
        active_idx = np.sort(np.asarray(active_idx, dtype=int))
        n_active = len(active_idx)

    if n_active_state is None:
        n_active_state = min(max(n_active, 2), d_model)
    else:
        n_active_state = int(n_active_state)

    active_state = np.sort(
        rng.choice(d_model, size=min(n_active_state, d_model), replace=False)
    )

    active_ff = []

    for dff in ffn_dims:
        if n_active_ff is None:
            n_ff = min(max(len(active_state), 2), dff)
        else:
            n_ff = min(int(n_active_ff), dff)

        active_ff.append(
            np.sort(rng.choice(dff, size=n_ff, replace=False))
        )

    # -----------------------------
    # 3. Sparse true parameters
    # -----------------------------

    truth = {}

    truth["E"] = torch.zeros(d_model, p, device=device, dtype=dtype)
    truth["e"] = torch.zeros(d_model, device=device, dtype=dtype)

    for k, dff in enumerate(ffn_dims):
        truth[f"W1_{k}"] = torch.zeros(dff, d_model, device=device, dtype=dtype)
        truth[f"b1_{k}"] = torch.zeros(dff, device=device, dtype=dtype)

        truth[f"W2_{k}"] = torch.zeros(d_model, dff, device=device, dtype=dtype)
        truth[f"b2_{k}"] = torch.zeros(d_model, device=device, dtype=dtype)

    truth["Wout"] = torch.zeros(out_dim, d_model, device=device, dtype=dtype)
    truth["bout"] = torch.zeros(out_dim, device=device, dtype=dtype)

    def put(M, rows, cols, low, high, density=1.0):
        for r in rows:
            for c in cols:
                if rng.random() <= density:
                    val = low + (high - low) * rng.random()
                    val = -val if rng.random() < 0.5 else val
                    M[int(r), int(c)] = float(val)

    # Input embedding:
    # active input features -> active state coordinates
    put(
        truth["E"],
        rows=active_state,
        cols=active_idx,
        low=weight_low,
        high=weight_high,
    )

    truth["e"][torch.as_tensor(active_state, device=device)] = (
        bias_sd
        * torch.randn(
            len(active_state),
            generator=gen,
            device=device,
            dtype=dtype,
        )
    )

    # Residual attention-FFN blocks:
    # active state -> active FFN -> active state
    for k, dff in enumerate(ffn_dims):
        aff = active_ff[k]

        put(
            truth[f"W1_{k}"],
            rows=aff,
            cols=active_state,
            low=weight_low,
            high=weight_high,
        )

        truth[f"b1_{k}"][torch.as_tensor(aff, device=device)] = (
            bias_sd
            * torch.randn(
                len(aff),
                generator=gen,
                device=device,
                dtype=dtype,
            )
        )

        put(
            truth[f"W2_{k}"],
            rows=active_state,
            cols=aff,
            low=weight_low,
            high=weight_high,
        )

        truth[f"b2_{k}"][torch.as_tensor(active_state, device=device)] = (
            bias_sd
            * torch.randn(
                len(active_state),
                generator=gen,
                device=device,
                dtype=dtype,
            )
        )

    # Output head:
    # active state -> scalar output
    put(
        truth["Wout"],
        rows=[0],
        cols=active_state,
        low=weight_low,
        high=weight_high,
    )

    truth["bout"][0] = bias_sd * torch.randn(
        (),
        generator=gen,
        device=device,
        dtype=dtype,
    )

    if bounded is not None:
        lo, hi = float(bounded[0]), float(bounded[1])
        for key in truth:
            truth[key] = truth[key].clamp(lo, hi)

    # -----------------------------
    # 4. Generate outcome
    # -----------------------------

    signal = bnn_truth_forward(
        X=X,
        truth=truth,
        n_blocks=n_blocks,
        attention_type=attention_type,
        ffn_activation=ffn_activation,
    )

    fam = str(family).lower()

    if fam == "gaussian":
        sigma = float(np.sqrt(sigma2))

        eps = sigma * torch.randn(
            n,
            generator=gen,
            device=device,
            dtype=dtype,
        )

        y = signal + eps

        if center_y:
            y = y - y.mean()

    elif fam in {"bernoulli", "binomial", "logistic"}:
        prob = torch.sigmoid(signal)
        y = torch.bernoulli(prob).to(dtype)

        sigma = np.nan

    elif fam == "poisson":
        rate = torch.exp(torch.clamp(signal, min=-20.0, max=20.0))
        y = torch.poisson(rate).to(dtype)

        sigma = np.nan

    else:
        raise ValueError("family must be gaussian, bernoulli, logistic, binomial, or poisson.")

    # -----------------------------
    # 5. Truth objects and info
    # -----------------------------

    feature_true = torch.zeros(p, device=device, dtype=dtype)
    feature_true[torch.as_tensor(active_idx, device=device)] = 1.0

    support_counts = {
        key: int((val.abs() > 1e-12).sum().item())
        for key, val in truth.items()
    }

    n_param = int(sum(v.numel() for v in truth.values()))
    n_nonzero = int(sum(support_counts.values()))
    truth_density = float(n_nonzero / max(n_param, 1))

    active_table = [
        {"rank": int(i + 1), "j": int(j), "j1": int(j + 1)}
        for i, j in enumerate(active_idx)
    ]

    signal_var = float(signal.var(unbiased=False).item())
    outcome_var = float(y.var(unbiased=False).item())

    sim_info = {
        "sim": "attention_ffn_sparse_oracle",
        "seed": int(seed),
        "family": fam,
        "n": int(n),
        "p": int(p),
        "n_active": int(n_active),
        "sigma2": float(sigma2) if fam == "gaussian" else None,
        "sigma": float(sigma) if fam == "gaussian" else None,
        "signal_var": signal_var,
        "outcome_var": outcome_var,
        "input_dim": int(p),
        "d_model": int(d_model),
        "n_blocks": int(n_blocks),
        "ffn_dims": [int(x) for x in ffn_dims],
        "out_dim": int(out_dim),
        "attention_type": attention_type,
        "ffn_activation": ffn_activation,
        "active_idx": active_idx,
        "active_state": active_state,
        "active_ff": active_ff,
        "active_table": active_table,
        "support_counts": support_counts,
        "n_param": n_param,
        "n_nonzero": n_nonzero,
        "truth_density": truth_density,
        "weight_low": float(weight_low),
        "weight_high": float(weight_high),
        "bias_sd": float(bias_sd),
        "bounded": bounded,
        "center_y": bool(center_y) if fam == "gaussian" else False,
    }

    if fam in {"bernoulli", "binomial", "logistic"}:
        sim_info["prob_mean"] = float(prob.mean().item())
        sim_info["positive_rate"] = float(y.mean().item())

    if fam == "poisson":
        sim_info["rate_mean"] = float(rate.mean().item())
        sim_info["count_mean"] = float(y.mean().item())

    return X, y, feature_true, truth, sim_info


def print_bnn_siminfo(sim_info: Dict[str, Any], digits=4):
    """
    Compact simulation summary for attention-FFN sparse BNN.
    """

    if sim_info is None:
        return "sim_info = None"

    lines = ["Sparse attention-FFN BNN simulation summary:"]

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
        "prob_mean",
        "positive_rate",
        "rate_mean",
        "count_mean",
        "input_dim",
        "d_model",
        "n_blocks",
        "ffn_dims",
        "out_dim",
        "attention_type",
        "ffn_activation",
        "bounded",
        "center_y",
        "n_param",
        "n_nonzero",
        "truth_density",
    ]

    for key in keys:
        if key in sim_info and sim_info[key] is not None:
            val = sim_info[key]

            if isinstance(val, float):
                lines.append(f"  {key:<22}: {val:.{digits}f}")
            else:
                lines.append(f"  {key:<22}: {val}")

    lines.append("")
    lines.append("Active input features, zero-based:")
    lines.append(f"  {np.asarray(sim_info['active_idx']).tolist()}")

    lines.append("")
    lines.append("Active state coordinates:")
    lines.append(f"  active_state : {np.asarray(sim_info['active_state']).tolist()}")

    lines.append("")
    lines.append("Active FFN units by residual block:")

    for k, aff in enumerate(sim_info["active_ff"]):
        lines.append(f"  block {k:<3}: {np.asarray(aff).tolist()}")

    lines.append("")
    lines.append("True nonzero counts:")

    for key, val in sim_info["support_counts"].items():
        lines.append(f"  {key:<8}: {int(val)}")

    return "\n".join(lines)