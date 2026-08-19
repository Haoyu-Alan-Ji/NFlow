


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_units=128, num_hidden_layers=2):
        super().__init__()

        layers = []
        d = int(in_dim)

        for _ in range(int(num_hidden_layers)):
            layers += [nn.Linear(d, hidden_units), nn.ReLU()]
            d = hidden_units

        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


class SemanticLayer(nn.Module):
    def __init__(
        self,
        s_dim,
        u_dim,
        t_dim,
        mode,
        hidden_units=128,
        num_hidden_layers=2,
        scale_clip=2.0,
    ):
        super().__init__()

        self.s_dim = int(s_dim)
        self.u_dim = int(u_dim)
        self.t_dim = int(t_dim)
        self.mode = mode
        self.scale_clip = float(scale_clip)

        if mode == "s":
            cond_dim = self.u_dim + self.t_dim
            trans_dim = self.s_dim
        elif mode == "u":
            cond_dim = self.s_dim + self.t_dim
            trans_dim = self.u_dim
        else:
            cond_dim = self.s_dim + self.u_dim
            trans_dim = self.t_dim

        self.net = MLP(
            cond_dim,
            2 * trans_dim,
            hidden_units,
            num_hidden_layers,
        )

    def forward(self, x, return_logdet=False):
        s = x[:, :self.s_dim]
        u = x[:, self.s_dim:self.s_dim + self.u_dim]
        t = x[:, self.s_dim + self.u_dim:]

        if self.mode == "s":
            cond = torch.cat([u, t], dim=1)
            log_scale, shift = self.net(cond).chunk(2, dim=1)
            log_scale = self.scale_clip * torch.tanh(
                log_scale / self.scale_clip
            )
            s = s * torch.exp(log_scale) + shift

        elif self.mode == "u":
            cond = torch.cat([s, t], dim=1)
            log_scale, shift = self.net(cond).chunk(2, dim=1)
            log_scale = self.scale_clip * torch.tanh(
                log_scale / self.scale_clip
            )
            u = u * torch.exp(log_scale) + shift

        else:
            cond = torch.cat([s, u], dim=1)
            log_scale, shift = self.net(cond).chunk(2, dim=1)
            log_scale = self.scale_clip * torch.tanh(
                log_scale / self.scale_clip
            )
            t = t * torch.exp(log_scale) + shift

        y = torch.cat([s, u, t], dim=1)
        logdet = log_scale.sum(dim=1)

        if return_logdet:
            return y, logdet

        return y

    def inverse(self, y, return_logdet=False):
        s = y[:, :self.s_dim]
        u = y[:, self.s_dim:self.s_dim + self.u_dim]
        t = y[:, self.s_dim + self.u_dim:]

        if self.mode == "s":
            cond = torch.cat([u, t], dim=1)
            log_scale, shift = self.net(cond).chunk(2, dim=1)
            log_scale = self.scale_clip * torch.tanh(
                log_scale / self.scale_clip
            )
            s = (s - shift) * torch.exp(-log_scale)

        elif self.mode == "u":
            cond = torch.cat([s, t], dim=1)
            log_scale, shift = self.net(cond).chunk(2, dim=1)
            log_scale = self.scale_clip * torch.tanh(
                log_scale / self.scale_clip
            )
            u = (u - shift) * torch.exp(-log_scale)

        else:
            cond = torch.cat([s, u], dim=1)
            log_scale, shift = self.net(cond).chunk(2, dim=1)
            log_scale = self.scale_clip * torch.tanh(
                log_scale / self.scale_clip
            )
            t = (t - shift) * torch.exp(-log_scale)

        x = torch.cat([s, u, t], dim=1)
        logdet = -log_scale.sum(dim=1)

        if return_logdet:
            return x, logdet

        return x


class SemanticFlow(nn.Module):
    def __init__(
        self,
        s_dim,
        u_dim,
        t_dim,
        K=4,
        hidden_units=128,
        num_hidden_layers=2,
        scale_clip=2.0,
    ):
        super().__init__()

        self.s_dim = int(s_dim)
        self.u_dim = int(u_dim)
        self.t_dim = int(t_dim)
        self.dim = self.s_dim + self.u_dim + self.t_dim

        self.layers = nn.ModuleList()

        for _ in range(int(K)):
            for mode in ["s", "u", "t"]:
                self.layers.append(
                    SemanticLayer(
                        self.s_dim,
                        self.u_dim,
                        self.t_dim,
                        mode,
                        hidden_units,
                        num_hidden_layers,
                        scale_clip,
                    )
                )

    def forward(self, x, return_logdet=False):
        z = x
        total_logdet = x.new_zeros(x.shape[0])

        for layer in self.layers:
            z, logdet = layer(z, return_logdet=True)
            total_logdet += logdet

        if return_logdet:
            return z, total_logdet

        return z

    def inverse(self, z, return_logdet=False):
        x = z
        total_logdet = z.new_zeros(z.shape[0])

        for layer in reversed(self.layers):
            x, logdet = layer.inverse(x, return_logdet=True)
            total_logdet += logdet

        if return_logdet:
            return x, total_logdet

        return x


class NBase(nn.Module):
    def __init__(self, dim, init_sd=None):
        super().__init__()

        self.dim = int(dim)
        self.init_sd = 0.5 if init_sd is None else float(init_sd)
        self.loc = nn.Parameter(torch.zeros(self.dim))
        self.raw_log_scale = nn.Parameter(
            torch.full(
                (self.dim,),
                math.log(self.init_sd),
            )
        )

    def sample(self, R):
        eps = torch.randn(
            int(R),
            self.dim,
            device=self.loc.device,
            dtype=self.loc.dtype,
        )

        log_scale = self.raw_log_scale.clamp(-5.0, 2.0)

        return self.loc[None, :] + torch.exp(log_scale)[None, :] * eps

    def log_prob(self, z):
        log_scale = self.raw_log_scale.clamp(-5.0, 2.0)[None, :]
        var = torch.exp(2.0 * log_scale)

        return -0.5 * (
            (z - self.loc[None, :]).pow(2) / var
            + 2.0 * log_scale
            + math.log(2.0 * math.pi)
        ).sum(dim=1)


class IdentityFlow(nn.Module):
    """Identity posterior map used by the mean-field and MCMC controls."""

    def forward(self, x, return_logdet=False):
        if return_logdet:
            return x, x.new_zeros(x.shape[0])
        return x

    def inverse(self, x, return_logdet=False):
        if return_logdet:
            return x, x.new_zeros(x.shape[0])
        return x


class MLPAffineConditioner(nn.Module):
    """Generic Real-NVP conditioner using only the fixed coordinates."""

    def __init__(
        self,
        fixed_dim,
        target_dim,
        hidden_units=128,
        num_hidden_layers=2,
    ):
        super().__init__()
        self.net = MLP(
            fixed_dim,
            2 * target_dim,
            hidden_units,
            num_hidden_layers,
        )

    def forward(self, fixed_values):
        raw = self.net(fixed_values)
        return raw.chunk(2, dim=-1)


class LatentAttentionConditioner(nn.Module):
    """
    Attention conditioner for one triangular affine-coupling direction.

    Current random values are provided only for fixed coordinates. Target
    coordinates enter through fixed metadata/identity queries, so the target
    values cannot leak into their own scale/shift parameters.
    """

    def __init__(
        self,
        dim,
        fixed_idx,
        target_idx,
        latent_metadata,
        token_dim=32,
        num_heads=4,
    ):
        super().__init__()
        self.dim = int(dim)
        self.token_dim = int(token_dim)
        self.num_heads = int(num_heads)

        if self.token_dim % self.num_heads != 0:
            raise ValueError("flow_token_dim must be divisible by flow_num_heads.")

        self.register_buffer(
            "fixed_idx", torch.as_tensor(fixed_idx, dtype=torch.long)
        )
        self.register_buffer(
            "target_idx", torch.as_tensor(target_idx, dtype=torch.long)
        )

        required = ("latent_type", "parameter_type", "group", "unit", "side")
        for name in required:
            if name not in latent_metadata:
                raise ValueError(f"Missing latent metadata field: {name}")
            value = torch.as_tensor(latent_metadata[name], dtype=torch.long)
            if value.numel() != self.dim:
                raise ValueError(f"latent metadata '{name}' must have length dim.")
            self.register_buffer(f"meta_{name}", value)

        self.value_projection = nn.Linear(1, self.token_dim)
        self.coordinate_embedding = nn.Embedding(self.dim, self.token_dim)
        self.latent_type_embedding = nn.Embedding(
            int(self.meta_latent_type.max().item()) + 1,
            self.token_dim,
        )
        self.parameter_type_embedding = nn.Embedding(
            int(self.meta_parameter_type.max().item()) + 1,
            self.token_dim,
        )
        self.group_embedding = nn.Embedding(
            int(self.meta_group.max().item()) + 1,
            self.token_dim,
        )
        self.unit_embedding = nn.Embedding(
            int(self.meta_unit.max().item()) + 1,
            self.token_dim,
        )
        self.side_embedding = nn.Embedding(
            int(self.meta_side.max().item()) + 1,
            self.token_dim,
        )

        self.self_attention = nn.MultiheadAttention(
            self.token_dim,
            self.num_heads,
            batch_first=True,
        )
        self.cross_attention = nn.MultiheadAttention(
            self.token_dim,
            self.num_heads,
            batch_first=True,
        )
        self.fixed_norm = nn.LayerNorm(self.token_dim)
        self.target_norm = nn.LayerNorm(self.token_dim)
        self.readout = nn.Linear(self.token_dim, 2)

        # Start every coupling layer close to identity.
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def _identity_embedding(self, indices):
        return (
            self.coordinate_embedding(indices)
            + self.latent_type_embedding(
                self.meta_latent_type.index_select(0, indices)
            )
            + self.parameter_type_embedding(
                self.meta_parameter_type.index_select(0, indices)
            )
            + self.group_embedding(self.meta_group.index_select(0, indices))
            + self.unit_embedding(self.meta_unit.index_select(0, indices))
            + self.side_embedding(self.meta_side.index_select(0, indices))
        )

    def forward(self, x):
        fixed_values = x.index_select(1, self.fixed_idx).unsqueeze(-1)
        fixed_tokens = (
            self.value_projection(fixed_values)
            + self._identity_embedding(self.fixed_idx)[None, :, :]
        )
        fixed_context, _ = self.self_attention(
            fixed_tokens,
            fixed_tokens,
            fixed_tokens,
            need_weights=False,
        )
        fixed_context = self.fixed_norm(fixed_tokens + fixed_context)

        # Target queries contain identity/structural metadata only, never the
        # current random target values.
        target_query = self._identity_embedding(self.target_idx)[None, :, :]
        target_query = target_query.expand(x.shape[0], -1, -1)
        target_context, _ = self.cross_attention(
            target_query,
            fixed_context,
            fixed_context,
            need_weights=False,
        )
        target_context = self.target_norm(target_query + target_context)
        raw = self.readout(target_context)
        return raw[..., 0], raw[..., 1]


class AffineCoupling(nn.Module):
    """Triangular affine coupling with an interchangeable conditioner."""

    def __init__(
        self,
        dim,
        mask,
        latent_metadata,
        conditioner_type="mlp",
        hidden_units=128,
        num_hidden_layers=2,
        scale_clip=2.0,
        token_dim=32,
        num_heads=4,
    ):
        super().__init__()
        self.dim = int(dim)
        self.scale_clip = float(scale_clip)
        self.conditioner_type = conditioner_type
        self.register_buffer("mask", torch.as_tensor(mask, dtype=torch.bool))

        fixed_idx = torch.nonzero(self.mask, as_tuple=False).flatten()
        target_idx = torch.nonzero(~self.mask, as_tuple=False).flatten()
        self.register_buffer("fixed_idx", fixed_idx)
        self.register_buffer("target_idx", target_idx)

        if fixed_idx.numel() == 0 or target_idx.numel() == 0:
            raise ValueError("Each affine mask needs non-empty fixed and target sides.")

        if conditioner_type == "attention":
            self.conditioner = LatentAttentionConditioner(
                dim=self.dim,
                fixed_idx=fixed_idx,
                target_idx=target_idx,
                latent_metadata=latent_metadata,
                token_dim=token_dim,
                num_heads=num_heads,
            )
        elif conditioner_type == "mlp":
            self.conditioner = MLPAffineConditioner(
                fixed_dim=int(fixed_idx.numel()),
                target_dim=int(target_idx.numel()),
                hidden_units=hidden_units,
                num_hidden_layers=num_hidden_layers,
            )
        else:
            raise ValueError(f"Unknown affine conditioner: {conditioner_type}")

    def params(self, x):
        if self.conditioner_type == "attention":
            raw_log_scale, shift = self.conditioner(x)
        else:
            fixed = x.index_select(1, self.fixed_idx)
            raw_log_scale, shift = self.conditioner(fixed)

        log_scale = self.scale_clip * torch.tanh(
            raw_log_scale / self.scale_clip
        )
        return log_scale, shift

    def forward(self, x, return_logdet=False):
        log_scale, shift = self.params(x)
        y = x.clone()
        target = x.index_select(1, self.target_idx)
        y[:, self.target_idx] = target * torch.exp(log_scale) + shift
        logdet = log_scale.sum(dim=1)
        if return_logdet:
            return y, logdet
        return y

    def inverse(self, y, return_logdet=False):
        # y_fixed == x_fixed, so conditioner parameters are exactly recoverable.
        log_scale, shift = self.params(y)
        x = y.clone()
        target = y.index_select(1, self.target_idx)
        x[:, self.target_idx] = (target - shift) * torch.exp(-log_scale)
        logdet = -log_scale.sum(dim=1)
        if return_logdet:
            return x, logdet
        return x


def latent_type_ids(s_dim, u_dim, t_dim):
    return torch.cat([
        torch.zeros(int(s_dim), dtype=torch.long),
        torch.ones(int(u_dim), dtype=torch.long),
        torch.full((int(t_dim),), 2, dtype=torch.long),
    ])


def default_latent_metadata(s_dim, u_dim, t_dim):
    """Basic metadata fallback for legacy/edge models."""
    latent_type = latent_type_ids(s_dim, u_dim, t_dim)
    zeros = torch.zeros_like(latent_type)
    return {
        "latent_type": latent_type,
        "parameter_type": zeros,
        "group": zeros,
        "unit": zeros,
        "side": zeros,
    }


def _normalize_dependency_pairs(dependency_pairs, dim):
    pairs = []
    seen = set()
    for pair in dependency_pairs or ():
        a, b = (int(pair[0]), int(pair[1]))
        if a == b:
            continue
        if not (0 <= a < dim and 0 <= b < dim):
            raise ValueError("A flow dependency pair contains an invalid coordinate.")
        key = tuple(sorted((a, b)))
        if key not in seen:
            seen.add(key)
            pairs.append(key)
    return pairs


def _pair_coverage(masks, pairs):
    return [
        any(bool(mask[a] != mask[b]) for mask in masks)
        for a, b in pairs
    ]


class AlternatingAffineFlow(nn.Module):
    """
    K random balanced Real-NVP cycles.

    Each cycle draws one fixed random partition M_k at model construction and
    adds the complementary reverse direction immediately afterwards. Masks are
    never redrawn during forward/inverse. Optional dependency pairs are used to
    reject mask sets that never separate a structurally important pair.
    """

    def __init__(
        self,
        dim,
        latent_metadata,
        K=4,
        conditioner_type="mlp",
        hidden_units=128,
        num_hidden_layers=2,
        scale_clip=2.0,
        token_dim=32,
        num_heads=4,
        mask_seed=123,
        dependency_pairs=None,
        max_mask_tries=1000,
    ):
        super().__init__()
        self.dim = int(dim)
        self.K = int(K)
        self.conditioner_type = conditioner_type
        self.mask_seed = int(mask_seed)
        self.dependency_pairs = _normalize_dependency_pairs(
            dependency_pairs, self.dim
        )

        if self.dim < 2:
            raise ValueError("Alternating affine flow requires latent dim >= 2.")
        if self.K < 1:
            raise ValueError("K must be positive for an affine flow.")

        n_fixed = self.dim // 2
        accepted = None
        coverage = None
        for attempt in range(int(max_mask_tries)):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.mask_seed + attempt)
            masks = []
            for _ in range(self.K):
                perm = torch.randperm(self.dim, generator=generator)
                mask = torch.zeros(self.dim, dtype=torch.bool)
                mask[perm[:n_fixed]] = True
                masks.append(mask)

            coverage = _pair_coverage(masks, self.dependency_pairs)
            if all(coverage):
                accepted = masks
                break

        if accepted is None:
            missing = [
                pair for pair, ok in zip(self.dependency_pairs, coverage) if not ok
            ]
            raise RuntimeError(
                "Could not construct random affine masks with full dependency "
                f"pair coverage; uncovered pairs include {missing[:10]}."
            )

        self.register_buffer("cycle_masks", torch.stack(accepted, dim=0))
        layers = []
        for mask in accepted:
            for direction_mask in (mask, ~mask):
                layers.append(AffineCoupling(
                    dim=self.dim,
                    mask=direction_mask,
                    latent_metadata=latent_metadata,
                    conditioner_type=conditioner_type,
                    hidden_units=hidden_units,
                    num_hidden_layers=num_hidden_layers,
                    scale_clip=scale_clip,
                    token_dim=token_dim,
                    num_heads=num_heads,
                ))
        self.layers = nn.ModuleList(layers)

    def transformed_coverage(self):
        coverage = torch.zeros(self.dim, dtype=torch.bool)
        for layer in self.layers:
            coverage[layer.target_idx.cpu()] = True
        return coverage

    def dependency_pair_coverage(self):
        if not self.dependency_pairs:
            return torch.ones(0, dtype=torch.bool)
        masks = [mask.cpu() for mask in self.cycle_masks]
        return torch.as_tensor(
            _pair_coverage(masks, self.dependency_pairs), dtype=torch.bool
        )

    def forward(self, x, return_logdet=False):
        z = x
        total = x.new_zeros(x.shape[0])
        for layer in self.layers:
            z, logdet = layer(z, return_logdet=True)
            total = total + logdet
        if return_logdet:
            return z, total
        return z

    def inverse(self, z, return_logdet=False):
        x = z
        total = z.new_zeros(z.shape[0])
        for layer in reversed(self.layers):
            x, logdet = layer.inverse(x, return_logdet=True)
            total = total + logdet
        if return_logdet:
            return x, total
        return x


def build_posterior_flow(
    s_dim,
    u_dim,
    t_dim,
    K_flow,
    flow_type="semantic",
    hidden_units=128,
    num_hidden_layers=2,
    scale_clip=2.0,
    token_dim=32,
    num_heads=4,
    latent_metadata=None,
    dependency_pairs=None,
    mask_seed=123,
):
    """Build the requested posterior transport."""

    flow_type = flow_type.lower()

    if int(K_flow) == 0 or flow_type == "meanfield":
        return IdentityFlow()

    # Retained only for legacy DirectUnit/edge baselines. The grouped model
    # below uses attention_affine by default and does not require this branch.
    if flow_type == "semantic":
        return SemanticFlow(
            s_dim,
            u_dim,
            t_dim,
            K=K_flow,
            hidden_units=hidden_units,
            num_hidden_layers=num_hidden_layers,
            scale_clip=scale_clip,
        )

    conditioner = {
        "affine": "mlp",
        "attention_affine": "attention",
    }.get(flow_type)

    if conditioner is None:
        raise ValueError(
            "flow_type must be semantic, affine, attention_affine, or meanfield."
        )

    if latent_metadata is None:
        latent_metadata = default_latent_metadata(s_dim, u_dim, t_dim)

    return AlternatingAffineFlow(
        dim=int(s_dim) + int(u_dim) + int(t_dim),
        latent_metadata=latent_metadata,
        K=K_flow,
        conditioner_type=conditioner,
        hidden_units=hidden_units,
        num_hidden_layers=num_hidden_layers,
        scale_clip=scale_clip,
        token_dim=token_dim,
        num_heads=num_heads,
        mask_seed=mask_seed,
        dependency_pairs=dependency_pairs,
    )


class DSSAttentionFFNDecoder(nn.Module):
    """
    Normalized-RePU gates are used by default. Parameters named in
    sigmoid_params use G(m) = sigmoid(m / sigmoid_tau).

        R_alpha(m) = (m_+)^alpha

    gate_tau=None:
        G(m) = (m_+)^alpha

    gate_tau>0:
        G(m) = (m_+)^alpha /
               (tau^alpha + (m_+)^alpha)

    Unbounded:
        theta = s G(u - t)

    Bounded:
        effect = midpoint + half_range * tanh(s)
        theta  = effect G(u - t)
    """

    def __init__(
        self,
        input_dim,
        d_model,
        n_blocks,
        ffn_dims=None,
        out_dim=1,
        bounded=None,
        gate_power=2.0,
        gate_tau=1.0,
        sigmoid_params=(),
        sigmoid_tau=1.0,
        attention_type="self",
        ffn_activation="relu",
    ):
        super().__init__()

        self.input_dim = int(input_dim)
        self.d_model = int(d_model)
        self.n_blocks = int(n_blocks)
        self.out_dim = int(out_dim)

        self.bounded = bounded
        self.gate_power = float(gate_power)
        self.gate_tau = (
            None if gate_tau is None else float(gate_tau)
        )
        self.sigmoid_params = tuple(sigmoid_params)
        self.sigmoid_tau = float(sigmoid_tau)

        self.attention_type = attention_type
        self.ffn_activation = ffn_activation.lower()

        if bounded is not None:
            lower, upper = bounded
            self.bound_mid = 0.5 * (lower + upper)
            self.bound_half = 0.5 * (upper - lower)

        if ffn_dims is None:
            self.ffn_dims = [4 * self.d_model] * self.n_blocks
        elif isinstance(ffn_dims, int):
            self.ffn_dims = [ffn_dims] * self.n_blocks
        else:
            self.ffn_dims = list(ffn_dims)

        raw_specs = [
            ("E", (self.d_model, self.input_dim), "input"),
            ("e", (self.d_model,), "input"),
        ]

        for k, dff in enumerate(self.ffn_dims):
            raw_specs += [
                (f"W1_{k}", (dff, self.d_model), k),
                (f"b1_{k}", (dff,), k),
                (f"W2_{k}", (self.d_model, dff), k),
                (f"b2_{k}", (self.d_model,), k),
            ]

        raw_specs += [
            ("Wout", (self.out_dim, self.d_model), "output"),
            ("bout", (self.out_dim,), "output"),
        ]

        self.param_specs = []
        m = 0

        for g, (name, shape, block) in enumerate(raw_specs):
            n_elem = math.prod(shape)

            item = {
                "name": name,
                "shape": shape,
                "block": block,
                "start": m,
                "end": m + n_elem,
                "t": g,
                "lambda": 1.0,
            }

            self.param_specs.append(item)
            setattr(self, name, item)

            m += n_elem

        self.layers_spec = []

        for k, dff in enumerate(self.ffn_dims):
            self.layers_spec.append({
                "block": k,
                "d_model": self.d_model,
                "dff": dff,
                "W1": getattr(self, f"W1_{k}"),
                "b1": getattr(self, f"b1_{k}"),
                "W2": getattr(self, f"W2_{k}"),
                "b2": getattr(self, f"b2_{k}"),
            })

        self.s_dim = m
        self.u_dim = m
        self.t_dim = len(self.param_specs)
        self.dim = 2 * m + self.t_dim

    def attention(self, z):
        if self.attention_type == "self":
            scores = torch.bmm(
                z,
                z.transpose(1, 2),
            ) / math.sqrt(z.shape[-1])

            weights = torch.softmax(scores, dim=-1)

            return torch.bmm(weights, z)

        if self.attention_type == "feature":
            return torch.softmax(z, dim=-1) * z

        return z

    def activate(self, x):
        if self.ffn_activation == "gelu":
            return F.gelu(x)

        if self.ffn_activation == "silu":
            return F.silu(x)

        return F.relu(x)

    def gate(self, name, margin):
        if name in self.sigmoid_params:
            return torch.sigmoid(margin / self.sigmoid_tau)

        positive_power = F.relu(margin).pow(self.gate_power)

        if self.gate_tau is None:
            return positive_power

        return positive_power / (
            self.gate_tau ** self.gate_power + positive_power
        )

    def active(self, name, margin, sigmoid_threshold=0.5):
        if name in self.sigmoid_params:
            return self.gate(name, margin) > float(sigmoid_threshold)

        return margin > 0.0

    def unpack(
        self,
        xi,
        return_summary=False,
        beta_eps=0.05,
        sigmoid_active_threshold=0.5,
    ):
        R = xi.shape[0]

        s = xi[:, :self.s_dim]

        u = xi[
            :,
            self.s_dim:
            self.s_dim + self.u_dim,
        ]

        t = xi[
            :,
            self.s_dim + self.u_dim:,
        ]

        params = {}
        summary = {}

        for item in self.param_specs:
            name = item["name"]
            sl = slice(item["start"], item["end"])

            margin = (
                u[:, sl]
                - t[:, item["t"]:item["t"] + 1]
            )

            gate = self.gate(name, margin)

            if self.bounded is None:
                val = s[:, sl] * gate
            else:
                effect = (
                    self.bound_mid
                    + self.bound_half
                    * torch.tanh(s[:, sl])
                )

                val = effect * gate

            val = val.reshape(
                R,
                *item["shape"],
            )

            params[name] = val

            if return_summary:
                active = self.active(
                    name,
                    margin,
                    sigmoid_threshold=sigmoid_active_threshold,
                ).to(xi.dtype).reshape(
                    R,
                    *item["shape"],
                )

                summary[f"{name}_pip"] = (
                    active.mean(dim=0)
                )

                summary[f"{name}_epip"] = (
                    val.abs() > beta_eps
                ).to(xi.dtype).mean(dim=0)

                summary[f"{name}_gate_mean"] = (
                    gate.mean(dim=0).reshape(
                        item["shape"]
                    )
                )

                summary[f"{name}_mean"] = (
                    val.mean(dim=0)
                )

                summary[f"{name}_sd"] = (
                    val.std(dim=0)
                )

        if return_summary:
            summary["t_mean"] = t.mean(dim=0)
            summary["t_sd"] = t.std(dim=0)

            summary["gate_type"] = (
                "mixed"
                if self.sigmoid_params
                else (
                    "repu"
                    if self.gate_tau is None
                    else "normalized_repu"
                )
            )

            summary["gate_type_by_parameter"] = {
                item["name"]: (
                    "sigmoid"
                    if item["name"] in self.sigmoid_params
                    else (
                        "repu"
                        if self.gate_tau is None
                        else "normalized_repu"
                    )
                )
                for item in self.param_specs
            }

            summary["gate_power"] = self.gate_power
            summary["gate_tau"] = self.gate_tau
            summary["sigmoid_tau"] = self.sigmoid_tau
            summary["sigmoid_active_threshold"] = float(
                sigmoid_active_threshold
            )
            summary["beta_eps"] = beta_eps

            return params, summary

        return params

    def forward(self, X, xi):
        params = self.unpack(xi)

        R = xi.shape[0]
        n = X.shape[0]

        Xr = X[None, :, :].expand(
            R,
            n,
            self.input_dim,
        )

        z = (
            torch.bmm(
                Xr,
                params["E"].transpose(1, 2),
            )
            + params["e"][:, None, :]
        )

        for k in range(self.n_blocks):
            att = self.attention(z)

            hidden = (
                torch.bmm(
                    att,
                    params[f"W1_{k}"].transpose(
                        1,
                        2,
                    ),
                )
                + params[f"b1_{k}"][:, None, :]
            )

            hidden = self.activate(hidden)

            delta = (
                torch.bmm(
                    hidden,
                    params[f"W2_{k}"].transpose(
                        1,
                        2,
                    ),
                )
                + params[f"b2_{k}"][:, None, :]
            )

            z = z + delta

        out = (
            torch.bmm(
                z,
                params["Wout"].transpose(1, 2),
            )
            + params["bout"][:, None, :]
        )

        if self.out_dim == 1:
            return out[..., 0]

        return out

class LaSTBNNVI(nn.Module):
    def __init__(
        self,
        X,
        y,
        input_dim=None,
        d_model=8,
        n_blocks=2,
        ffn_dims=None,
        out_dim=1,
        family="gaussian",
        sigma2=1.0,
        init_sd=None,
        K_flow=4,
        flow_hidden_units=128,
        flow_hidden_layers=2,
        scale_clip=2.0,
        bounded=None,
        gate_power=2.0,
        gate_tau=1.0,
        sigmoid_params=(),
        sigmoid_tau=1.0,
        attention_type="self",
        ffn_activation="relu",
        flow_type="semantic",
        flow_token_dim=32,
        flow_num_heads=4,
    ):
        super().__init__()

        self.register_buffer("X", X)
        self.register_buffer("y", y)

        if input_dim is None:
            input_dim = X.shape[1]

        self.family = family.lower()

        self.register_buffer(
            "sigma2",
            torch.tensor(
                float(sigma2),
                dtype=X.dtype,
            ),
        )

        self.decoder = DSSAttentionFFNDecoder(
            input_dim=input_dim,
            d_model=d_model,
            n_blocks=n_blocks,
            ffn_dims=ffn_dims,
            out_dim=out_dim,
            bounded=bounded,
            gate_power=gate_power,
            gate_tau=gate_tau,
            sigmoid_params=sigmoid_params,
            sigmoid_tau=sigmoid_tau,
            attention_type=attention_type,
            ffn_activation=ffn_activation,
        )

        self.q0 = NBase(
            self.decoder.dim,
            init_sd=init_sd,
        )
        self.init_sd = self.q0.init_sd

        self.flow_type = flow_type.lower()
        self.flow = build_posterior_flow(
            s_dim=self.decoder.s_dim,
            u_dim=self.decoder.u_dim,
            t_dim=self.decoder.t_dim,
            K_flow=K_flow,
            flow_type=self.flow_type,
            hidden_units=flow_hidden_units,
            num_hidden_layers=flow_hidden_layers,
            scale_clip=scale_clip,
            token_dim=flow_token_dim,
            num_heads=flow_num_heads,
        )

    def sample_posterior(self, R):
        z0 = self.q0.sample(R)

        xi, logdet = self.flow(
            z0,
            return_logdet=True,
        )

        log_q = (
            self.q0.log_prob(z0)
            - logdet
        )

        return xi, log_q

    def log_likelihood(self, xi, X=None, y=None, **decoder_kwargs):
        X = self.X if X is None else X
        y = self.y if y is None else y

        pred = self.decoder(
            X,
            xi,
            **decoder_kwargs,
        )

        if self.family == "gaussian":
            resid = (
                y[None, :]
                - pred
            )

            return -0.5 * (
                resid.square().sum(dim=1)
                / self.sigma2
                + y.numel()
                * torch.log(
                    2.0
                    * torch.pi
                    * self.sigma2
                )
            )

        if self.family in {
            "bernoulli",
            "binomial",
            "logistic",
        }:
            y = y[None, :].expand_as(
                pred
            )

            return (
                -F.binary_cross_entropy_with_logits(
                    pred,
                    y,
                    reduction="none",
                ).sum(dim=1)
            )

        if self.family == "poisson":
            y = y[None, :].expand_as(
                pred
            )

            rate = torch.exp(
                pred.clamp(-20.0, 20.0)
            )

            return (
                y * pred
                - rate
                - torch.lgamma(y + 1.0)
            ).sum(dim=1)

        logp = F.log_softmax(
            pred,
            dim=-1,
        )

        idx = torch.arange(
            y.numel(),
            device=y.device,
        )

        return logp[
            :,
            idx,
            y.long(),
        ].sum(dim=1)

    def log_prior(self, xi):
        return -0.5 * (
            xi.square()
            + math.log(2.0 * math.pi)
        ).sum(dim=1)

    def log_joint(self, xi):
        return (
            self.log_likelihood(xi)
            + self.log_prior(xi)
        )

    def elbo_draws(self, R):
        xi, log_q = self.sample_posterior(R)
        log_likelihood = self.log_likelihood(xi)
        log_prior = self.log_prior(xi)

        return {
            "xi": xi,
            "log_likelihood": log_likelihood,
            "log_prior": log_prior,
            "log_q": log_q,
            "kl": log_q - log_prior,
            "elbo": log_likelihood + log_prior - log_q,
        }

    def neg_elbo(
        self,
        R=64,
        elbo_beta=1.0,
    ):
        draws = self.elbo_draws(R)

        return -(
            float(elbo_beta) * draws["log_likelihood"]
            + draws["log_prior"]
            - draws["log_q"]
        ).mean()

    @torch.no_grad()
    def predict(
        self,
        X_new,
        R=200,
    ):
        xi, _ = self.sample_posterior(R)

        pred = self.decoder(
            X_new,
            xi,
        )

        if self.family == "gaussian":
            return pred.mean(dim=0)

        if self.family in {
            "bernoulli",
            "binomial",
            "logistic",
        }:
            return torch.sigmoid(
                pred
            ).mean(dim=0)

        if self.family == "poisson":
            return torch.exp(
                pred.clamp(-20.0, 20.0)
            ).mean(dim=0)

        return F.softmax(
            pred,
            dim=-1,
        ).mean(dim=0)

    @torch.no_grad()
    def posterior_summary(
        self,
        R=500,
        beta_eps=0.05,
    ):
        xi, _ = self.sample_posterior(R)

        _, summary = self.decoder.unpack(
            xi,
            return_summary=True,
            beta_eps=beta_eps,
        )

        return summary


PARAMETER_TYPE_IDS = {
    "none": 0,
    "beta0": 1,
    "ell": 2,
    "W1": 3,
    "b1": 4,
    "W2": 5,
    "group_activation": 6,
    "threshold": 7,
}

SIDE_IDS = {
    "none": 0,
    "input": 1,
    "output": 2,
    "linear": 3,
    "global": 4,
    "group": 5,
}


class GroupLayout:
    """
    Scalar-slab layout for the clean shallow grouped BNN.

    Unit selection uses one selectable object per hidden unit:

        G_j^U = {W1[j,:], b1[j], W2[:,j]}

    so one unit has one activation latent u_j and one hard indicator I_j.
    Input/output are retained only as attention-token metadata on slab
    coordinates; they do not create separate selection groups.

    Feature selection keeps one group per raw predictor. With linear_skip=True,
    the corresponding ell coefficient joins the same feature group.
    """

    def __init__(
        self,
        input_dim,
        H,
        out_dim,
        selection_mode,
        linear_skip=False,
    ):
        self.input_dim = int(input_dim)
        self.H = int(H)
        self.out_dim = int(out_dim)
        self.selection_mode = selection_mode
        self.linear_skip = bool(linear_skip)
        self.group_meta = []
        self.unit_groups = []

        if self.H < 1:
            raise ValueError("H must be positive.")

        if selection_mode == "unit_group":
            group_lookup = {}
            for unit in range(self.H):
                group_id = len(self.group_meta)
                meta = {
                    "group_id": group_id,
                    "selection_type": "unit",
                    "block": 0,
                    "unit": unit,
                    "side": "group",
                }
                self.group_meta.append(meta)
                self.unit_groups.append(dict(meta))
                group_lookup[unit] = group_id

        elif selection_mode == "feature_group":
            group_lookup = {}
            for feature in range(self.input_dim):
                self.group_meta.append({
                    "group_id": feature,
                    "selection_type": "feature",
                    "block": "input",
                    "feature": feature,
                    "side": "input",
                })
        else:
            raise ValueError(
                "Grouped layout selection_mode must be unit_group or feature_group."
            )

        raw_specs = [("beta0", (self.out_dim,), "beta0")]
        if self.linear_skip:
            raw_specs.append(("ell", (self.out_dim, self.input_dim), "ell"))
        raw_specs.extend([
            ("W1", (self.H, self.input_dim), "W1"),
            ("b1", (self.H,), "b1"),
            ("W2", (self.out_dim, self.H), "W2"),
        ])

        self.param_specs = []
        start = 0

        for name, shape, param_type in raw_specs:
            n_elem = math.prod(shape)
            group_ids = [-1] * n_elem
            selection_type = "open"

            if selection_mode == "feature_group" and name == "W1":
                group_ids = list(range(self.input_dim)) * self.H
                selection_type = "feature"

            if (
                selection_mode == "feature_group"
                and self.linear_skip
                and name == "ell"
            ):
                group_ids = list(range(self.input_dim)) * self.out_dim
                selection_type = "feature"

            if selection_mode == "unit_group" and name == "W1":
                group_ids = [
                    group_lookup[unit]
                    for unit in range(self.H)
                    for _ in range(self.input_dim)
                ]
                selection_type = "unit"

            if selection_mode == "unit_group" and name == "b1":
                group_ids = [group_lookup[unit] for unit in range(self.H)]
                selection_type = "unit"

            if selection_mode == "unit_group" and name == "W2":
                group_ids = [
                    group_lookup[unit]
                    for _ in range(self.out_dim)
                    for unit in range(self.H)
                ]
                selection_type = "unit"

            self.param_specs.append({
                "name": name,
                "shape": tuple(shape),
                "start": start,
                "end": start + n_elem,
                "parameter_type": param_type,
                "selection_type": selection_type,
                "group_ids": tuple(group_ids),
            })
            start += n_elem

        self.s_dim = start
        self.u_dim = len(self.group_meta)
        self.t_dim = 1
        self.dim = self.s_dim + self.u_dim + self.t_dim

        members = [[] for _ in range(self.u_dim)]
        for item in self.param_specs:
            for local_index, group_id in enumerate(item["group_ids"]):
                if group_id >= 0:
                    members[group_id].append(item["start"] + local_index)
        self.group_scalar_indices = [tuple(x) for x in members]

        self.latent_metadata = self._build_latent_metadata()
        self.dependency_pairs = self._build_dependency_pairs()

    def _scalar_unit(self, item, local_index):
        name = item["name"]
        if name == "W1":
            return local_index // self.input_dim
        if name == "b1":
            return local_index
        if name == "W2":
            return local_index % self.H
        return -1

    def _scalar_side(self, item):
        return {
            "W1": "input",
            "b1": "input",
            "W2": "output",
            "ell": "linear",
            "beta0": "global",
        }.get(item["name"], "none")

    def _build_latent_metadata(self):
        latent_type = []
        parameter_type = []
        group = []
        unit = []
        side = []

        # Slab coordinates. A unit's W1/b1/W2 share one group id, while side
        # remains input/output metadata for the attention conditioner.
        for item in self.param_specs:
            for local_index, group_id in enumerate(item["group_ids"]):
                latent_type.append(0)
                parameter_type.append(PARAMETER_TYPE_IDS[item["parameter_type"]])
                group.append(group_id + 1 if group_id >= 0 else 0)
                unit_id = self._scalar_unit(item, local_index)
                unit.append(unit_id + 1 if unit_id >= 0 else 0)
                side.append(SIDE_IDS[self._scalar_side(item)])

        # One activation coordinate per selectable group.
        for meta in self.group_meta:
            latent_type.append(1)
            parameter_type.append(PARAMETER_TYPE_IDS["group_activation"])
            group.append(int(meta["group_id"]) + 1)
            unit_id = int(meta.get("unit", -1))
            unit.append(unit_id + 1 if unit_id >= 0 else 0)
            side.append(SIDE_IDS.get(meta.get("side", "none"), 0))

        # One shared threshold coordinate.
        latent_type.append(2)
        parameter_type.append(PARAMETER_TYPE_IDS["threshold"])
        group.append(0)
        unit.append(0)
        side.append(SIDE_IDS["global"])

        return {
            "latent_type": torch.as_tensor(latent_type, dtype=torch.long),
            "parameter_type": torch.as_tensor(parameter_type, dtype=torch.long),
            "group": torch.as_tensor(group, dtype=torch.long),
            "unit": torch.as_tensor(unit, dtype=torch.long),
            "side": torch.as_tensor(side, dtype=torch.long),
        }

    def _build_dependency_pairs(self):
        pairs = []
        threshold_index = self.s_dim + self.u_dim

        # Each selected slab can condition/receive information from its one
        # group activation in at least one random partition; each activation can
        # likewise interact with the shared threshold.
        for group_id, scalar_indices in enumerate(self.group_scalar_indices):
            u_index = self.s_dim + group_id
            pairs.extend((s_index, u_index) for s_index in scalar_indices)
            pairs.append((u_index, threshold_index))

        return tuple(pairs)


class GroupGateDecoder(nn.Module):
    """
    Clean shallow grouped BNN with one LVR gate per selectable group.

    Unit mode uses one gate per hidden unit and applies it exactly once to the
    unit contribution:

        f(x) = beta0 + sum_j g_j W2[:,j] ReLU(W1[j,:] x + b1[j])
        g_j  = G(u_j - t)

    The hard model contains one idempotent unit indicator I_j. After using
    I_j^2 = I_j, that consolidated indicator is relaxed once via G. Supported
    maps are RePU, G(m)=(m_+)^power, and normalized RePU,
    G(m)=(m_+)^power/(tau^power+(m_+)^power).

    Feature mode is unchanged conceptually: one gate per raw predictor is
    applied to its W1 column (and ell column when linear_skip=True).
    """

    def __init__(
        self,
        input_dim,
        H=5,
        out_dim=1,
        selection_mode="unit_group",
        gate_power=1.0,
        gate_tau=None,
        repu_power=None,
        linear_skip=False,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.H = int(H)
        self.out_dim = int(out_dim)
        self.selection_mode = selection_mode
        self.gate_power = float(gate_power)
        self.gate_tau = None if gate_tau is None else float(gate_tau)
        self.repu_power = repu_power
        self.linear_skip = bool(linear_skip)

        if self.gate_power <= 0.0:
            raise ValueError("gate_power must be positive.")
        if self.gate_tau is not None and self.gate_tau <= 0.0:
            raise ValueError("gate_tau must be positive or None.")
        if self.repu_power is not None:
            raise ValueError(
                "GroupedBNNVI currently uses ordinary ReLU hidden activation; "
                "repu_power must be None."
            )

        self.layout = GroupLayout(
            input_dim=self.input_dim,
            H=self.H,
            out_dim=self.out_dim,
            selection_mode=self.selection_mode,
            linear_skip=self.linear_skip,
        )
        self.param_specs = self.layout.param_specs
        self.group_meta = self.layout.group_meta
        self.unit_groups = self.layout.unit_groups
        self.s_dim = self.layout.s_dim
        self.u_dim = self.layout.u_dim
        self.t_dim = self.layout.t_dim
        self.dim = self.layout.dim

        for item in self.param_specs:
            self.register_buffer(
                f"_group_ids_{item['name']}",
                torch.as_tensor(item["group_ids"], dtype=torch.long),
            )

        for group_id, indices in enumerate(self.layout.group_scalar_indices):
            self.register_buffer(
                f"_group_scalar_{group_id}",
                torch.as_tensor(indices, dtype=torch.long),
            )

    @property
    def activation_degree(self):
        return 1.0

    def activate(self, x):
        return F.relu(x)

    def group_gate(self, margin):
        positive = F.relu(margin).pow(self.gate_power)
        if self.gate_tau is None:
            return positive
        return positive / (
            self.gate_tau ** self.gate_power + positive
        )

    def split_latent(self, xi):
        s = xi[:, :self.s_dim]
        u = xi[:, self.s_dim:self.s_dim + self.u_dim]
        t = xi[:, self.s_dim + self.u_dim:]
        return s, u, t

    def group_semantics(self, xi):
        s, u, t = self.split_latent(xi)
        margin = u - t
        return {
            "s": s,
            "u": u,
            "t": t,
            "margin": margin,
            "gate": self.group_gate(margin),
            "active": margin > 0.0,
        }

    def group_slab_norms(self, xi):
        s = xi[:, :self.s_dim]
        norms = []
        for group_id in range(self.u_dim):
            idx = getattr(self, f"_group_scalar_{group_id}")
            norms.append(s.index_select(1, idx).square().sum(dim=1).sqrt())
        return torch.stack(norms, dim=1)

    def unpack_slabs(self, xi):
        s = xi[:, :self.s_dim]
        R = xi.shape[0]
        return {
            item["name"]: s[:, item["start"]:item["end"]].reshape(
                R, *item["shape"]
            )
            for item in self.param_specs
        }

    def _unit_strength_components(self, slabs):
        input_norm = torch.cat(
            [slabs["W1"], slabs["b1"].unsqueeze(2)], dim=2
        ).norm(dim=2)
        output_norm = slabs["W2"].norm(dim=1)
        slab_strength = input_norm * output_norm
        return input_norm, output_norm, slab_strength

    def unit_semantics(self, xi):
        if self.selection_mode != "unit_group":
            raise ValueError("unit_semantics requires selection_mode='unit_group'.")

        semantics = self.group_semantics(xi)
        slabs = self.unpack_slabs(xi)
        input_norm, output_norm, slab_strength = self._unit_strength_components(slabs)
        gate = semantics["gate"]
        active = semantics["active"]

        return {
            "active": active,
            "gate": gate,
            "margin": semantics["margin"],
            "input_slab_norm": input_norm,
            "output_slab_norm": output_norm,
            "slab_strength": slab_strength,
            "effective_strength": gate * slab_strength,
            "strength_definition": "single_gate_direct_shallow_parameter",
        }

    def unpack(
        self,
        xi,
        return_summary=False,
        beta_eps=0.05,
        sigmoid_active_threshold=0.5,
        force_all_on=False,
    ):
        del beta_eps, sigmoid_active_threshold
        R = xi.shape[0]
        s, _, _ = self.split_latent(xi)
        semantics = self.group_semantics(xi)
        likelihood_gate = (
            torch.ones_like(semantics["gate"])
            if force_all_on
            else semantics["gate"]
        )
        params = {}

        for item in self.param_specs:
            values = s[:, item["start"]:item["end"]]

            # In unit mode W1/b1/W2 are one statistical group, but the single
            # relaxed unit gate is applied once in forward(), not once per
            # parameter occurrence. In feature mode the feature gate continues
            # to act directly on its W1/ell coordinates.
            if self.selection_mode == "feature_group":
                group_ids = getattr(self, f"_group_ids_{item['name']}")
                selected = group_ids >= 0
                if selected.any():
                    safe_ids = group_ids.clamp_min(0)
                    selected_gates = likelihood_gate.index_select(1, safe_ids)
                    gates = torch.where(
                        selected[None, :],
                        selected_gates,
                        torch.ones_like(selected_gates),
                    )
                    values = values * gates

            params[item["name"]] = values.reshape(R, *item["shape"])

        if return_summary:
            summary = {
                "group_pip": semantics["active"].float().mean(dim=0),
                "group_gate_mean": semantics["gate"].mean(dim=0),
                "group_margin_mean": semantics["margin"].mean(dim=0),
                "t_mean": semantics["t"].mean(dim=0),
                "t_sd": semantics["t"].std(dim=0),
                "selection_mode": self.selection_mode,
                "gate_type": (
                    "repu" if self.gate_tau is None else "normalized_repu"
                ),
                "gate_power": self.gate_power,
                "gate_tau": self.gate_tau,
                "repu_power": self.repu_power,
                "linear_skip": self.linear_skip,
            }
            if self.selection_mode == "unit_group":
                units = self.unit_semantics(xi)
                summary["unit_pip"] = units["active"].float().mean(dim=0)
                summary["unit_gate_mean"] = units["gate"].mean(dim=0)
            return params, summary

        return params

    def compatibility_signature(self):
        specs = tuple(
            (
                item["name"],
                item["shape"],
                item["selection_type"],
                item["group_ids"],
            )
            for item in self.param_specs
        )
        return (
            "single_group_single_gate_v1",
            self.selection_mode,
            specs,
            self.gate_power,
            self.gate_tau,
            self.repu_power,
            self.linear_skip,
        )

    def flow_metadata(self):
        return self.layout.latent_metadata

    def flow_dependency_pairs(self):
        return self.layout.dependency_pairs

    def forward(self, X, xi, force_all_on=False):
        params = self.unpack(xi, force_all_on=force_all_on)
        semantics = self.group_semantics(xi)
        R = xi.shape[0]
        n = X.shape[0]
        Xr = X[None, :, :].expand(R, n, self.input_dim)

        hidden = torch.bmm(
            Xr, params["W1"].transpose(1, 2)
        ) + params["b1"][:, None, :]
        hidden = self.activate(hidden)

        if self.selection_mode == "unit_group":
            # One consolidated gate per unit, applied exactly once.
            gate = (
                torch.ones_like(semantics["gate"])
                if force_all_on
                else semantics["gate"]
            )
            hidden = hidden * gate[:, None, :]

        out = torch.bmm(hidden, params["W2"].transpose(1, 2))

        if self.linear_skip:
            out = out + torch.bmm(Xr, params["ell"].transpose(1, 2))

        out = out + params["beta0"][:, None, :]
        if self.out_dim == 1:
            return out[..., 0]
        return out


class GroupedBNNVI(LaSTBNNVI):
    """VI model for the clean shallow grouped BNN."""

    def __init__(
        self,
        X,
        y,
        input_dim=None,
        H=5,
        out_dim=1,
        selection_mode="unit_group",
        family="gaussian",
        sigma2=1.0,
        init_sd=None,
        K_flow=4,
        flow_type="attention_affine",
        flow_hidden_units=128,
        flow_hidden_layers=2,
        scale_clip=2.0,
        flow_token_dim=32,
        flow_num_heads=4,
        flow_mask_seed=123,
        gate_power=1.0,
        gate_tau=None,
        repu_power=None,
        linear_skip=False,
    ):
        nn.Module.__init__(self)
        self.register_buffer("X", X)
        self.register_buffer("y", y)

        if input_dim is None:
            input_dim = X.shape[1]

        self.family = family.lower()
        self.register_buffer(
            "sigma2",
            torch.tensor(float(sigma2), dtype=X.dtype),
        )
        self.decoder = GroupGateDecoder(
            input_dim=input_dim,
            H=H,
            out_dim=out_dim,
            selection_mode=selection_mode,
            gate_power=gate_power,
            gate_tau=gate_tau,
            repu_power=repu_power,
            linear_skip=linear_skip,
        )
        self.q0 = NBase(self.decoder.dim, init_sd=init_sd)
        self.init_sd = self.q0.init_sd
        self.flow_type = flow_type.lower()
        self.flow = build_posterior_flow(
            s_dim=self.decoder.s_dim,
            u_dim=self.decoder.u_dim,
            t_dim=self.decoder.t_dim,
            K_flow=K_flow,
            flow_type=self.flow_type,
            hidden_units=flow_hidden_units,
            num_hidden_layers=flow_hidden_layers,
            scale_clip=scale_clip,
            token_dim=flow_token_dim,
            num_heads=flow_num_heads,
            latent_metadata=self.decoder.flow_metadata(),
            dependency_pairs=self.decoder.flow_dependency_pairs(),
            mask_seed=flow_mask_seed,
        )


@torch.no_grad()
def run_grouped_acceptance_tests(device=None, dtype=torch.float64):
    """Deterministic checks for the single-group/single-gate grouped model."""

    device = torch.device("cpu") if device is None else torch.device(device)
    decoder = GroupGateDecoder(
        input_dim=2,
        H=3,
        out_dim=1,
        selection_mode="unit_group",
        gate_power=1.0,
        gate_tau=None,
        repu_power=None,
        linear_skip=False,
    ).to(device=device, dtype=dtype)
    specs = {item["name"]: item for item in decoder.param_specs}

    assert decoder.H == 3
    assert specs["W1"]["shape"] == (3, 2)
    assert specs["b1"]["shape"] == (3,)
    assert specs["W2"]["shape"] == (1, 3)
    assert specs["beta0"]["shape"] == (1,)
    assert "ell" not in specs
    assert not any(x in specs for x in ("E", "e", "Wout", "bout", "b2"))

    # One activation latent per unit, not two input/output activations.
    assert decoder.u_dim == decoder.H == 3
    assert len(decoder.unit_groups) == decoder.H

    unit = 1
    group_id = decoder.unit_groups[unit]["group_id"]
    # For p=2, out_dim=1 the group contains W1 row (2), b1 (1), W2 column (1).
    assert len(decoder.layout.group_scalar_indices[group_id]) == 4

    w1_ids = specs["W1"]["group_ids"]
    b1_ids = specs["b1"]["group_ids"]
    w2_ids = specs["W2"]["group_ids"]
    assert w1_ids[unit * 2:(unit + 1) * 2] == (group_id, group_id)
    assert b1_ids[unit] == group_id
    assert w2_ids[unit] == group_id

    # Attention metadata keeps input/output role even though group id is shared.
    meta = decoder.flow_metadata()
    w1_coord = specs["W1"]["start"] + unit * decoder.input_dim
    b1_coord = specs["b1"]["start"] + unit
    w2_coord = specs["W2"]["start"] + unit
    u_coord = decoder.s_dim + group_id
    assert int(meta["group"][w1_coord]) == group_id + 1
    assert int(meta["group"][w2_coord]) == group_id + 1
    assert int(meta["side"][w1_coord]) == SIDE_IDS["input"]
    assert int(meta["side"][b1_coord]) == SIDE_IDS["input"]
    assert int(meta["side"][w2_coord]) == SIDE_IDS["output"]
    assert int(meta["side"][u_coord]) == SIDE_IDS["group"]

    xi = torch.zeros(1, decoder.dim, device=device, dtype=dtype)
    w1_raw = torch.tensor([1.2, -0.7], device=device, dtype=dtype)
    b1_raw = torch.tensor(0.3, device=device, dtype=dtype)
    w2_raw = torch.tensor(0.8, device=device, dtype=dtype)
    q = torch.tensor([[0.4, -0.2]], device=device, dtype=dtype)

    w1_start = specs["W1"]["start"] + unit * decoder.input_dim
    xi[0, w1_start:w1_start + decoder.input_dim] = w1_raw
    xi[0, specs["b1"]["start"] + unit] = b1_raw
    xi[0, specs["W2"]["start"] + unit] = w2_raw

    # A positive margin of 0.4 gives g=0.4 exactly (plain ReLU).
    xi_open = xi.clone()
    xi_open[0, decoder.s_dim + group_id] = 0.4
    semantics = decoder.unit_semantics(xi_open)
    assert torch.allclose(
        semantics["gate"][0, unit],
        torch.tensor(0.4, device=device, dtype=dtype),
    )

    # Unit-mode unpack returns slabs; it must not silently multiply g into W1,
    # b1 and W2 separately.
    open_params = decoder.unpack(xi_open)
    assert torch.allclose(open_params["W1"][0, unit], w1_raw)
    assert torch.allclose(open_params["b1"][0, unit], b1_raw)
    assert torch.allclose(open_params["W2"][0, 0, unit], w2_raw)

    pred = decoder(q, xi_open)[0, 0]
    raw_unit = w2_raw * F.relu((w1_raw * q[0]).sum() + b1_raw)
    expected = 0.4 * raw_unit
    assert torch.allclose(pred, expected, atol=1e-10, rtol=1e-10)

    # Closing the one unit gate zeros its functional contribution even though
    # the slabs themselves remain available to the posterior flow.
    xi_closed = xi.clone()
    xi_closed[0, decoder.s_dim + group_id] = -0.2
    closed_pred = decoder(q, xi_closed)[0, 0]
    assert torch.allclose(
        closed_pred,
        torch.tensor(0.0, device=device, dtype=dtype),
        atol=1e-12,
        rtol=0.0,
    )

    # Feature grouping still gives one gate per raw predictor and includes ell
    # when a linear skip is explicitly enabled.
    feature_decoder = GroupGateDecoder(
        input_dim=3,
        H=2,
        out_dim=1,
        selection_mode="feature_group",
        linear_skip=True,
    ).to(device=device, dtype=dtype)
    assert all(
        len(indices) == feature_decoder.H + feature_decoder.out_dim
        for indices in feature_decoder.layout.group_scalar_indices
    )

    flow = AlternatingAffineFlow(
        dim=decoder.dim,
        latent_metadata=decoder.flow_metadata(),
        K=4,
        conditioner_type="attention",
        token_dim=8,
        num_heads=2,
        mask_seed=321,
        dependency_pairs=decoder.flow_dependency_pairs(),
    ).to(device=device, dtype=dtype)

    for layer_index, layer_item in enumerate(flow.layers):
        weight = layer_item.conditioner.readout.weight
        values = torch.linspace(
            -0.04, 0.04, weight.numel(), device=device, dtype=dtype
        ).reshape_as(weight)
        layer_item.conditioner.readout.weight.copy_(values)
        layer_item.conditioner.readout.bias.fill_(0.01 * (layer_index + 1))

    x = torch.randn(6, decoder.dim, device=device, dtype=dtype)
    y, logdet = flow(x, return_logdet=True)
    x_rec, inv_logdet = flow.inverse(y, return_logdet=True)
    inverse_error = float((x - x_rec).abs().max())
    logdet_error = float((logdet + inv_logdet).abs().max())
    assert inverse_error < 1e-9
    assert logdet_error < 1e-9

    layer = flow.layers[0]
    changed_target = x.clone()
    changed_target[:, layer.target_idx] += 3.0
    params_a = layer.params(x)
    params_b = layer.params(changed_target)
    conditioner_error = max(
        float((a - b).abs().max()) for a, b in zip(params_a, params_b)
    )
    assert conditioner_error < 1e-12
    assert bool(flow.transformed_coverage().all())
    assert bool(flow.dependency_pair_coverage().all())

    return {
        "hidden_units": decoder.H,
        "scalar_slabs": decoder.s_dim,
        "unit_groups": decoder.u_dim,
        "group_activations": decoder.u_dim,
        "one_gate_per_unit": True,
        "unit_gate_applied_once": True,
        "plain_relu_gate": True,
        "relu_hidden_activation": True,
        "attention_input_output_metadata": True,
        "embedding_removed": True,
        "extra_output_projection_removed": True,
        "b2_removed": True,
        "linear_skip_default": False,
        "flow_inverse_error": inverse_error,
        "flow_logdet_error": logdet_error,
        "conditioner_independence_error": conditioner_error,
        "random_mask_dependency_coverage": bool(
            flow.dependency_pair_coverage().all()
        ),
    }

ROLE_NAMES = ("input", "breakpoint", "output")


class DirectUnitDecoder(nn.Module):
    """
    One-dimensional direct unit model

        f(x) = beta0 + ell*x + sum_j a_j ReLU(w_j*x - b_j).

    beta0 and ell are continuous. Each role has H slab/local coordinates and
    one shared threshold. Roles omitted from gate_roles are fixed open.
    """

    def __init__(
        self,
        H=3,
        gate_roles=ROLE_NAMES,
        gate_power=2.0,
        gate_tau=1.0,
    ):
        super().__init__()

        self.H = int(H)
        self.role_names = ROLE_NAMES
        self.gate_roles = tuple(gate_roles)
        self.gate_power = float(gate_power)
        self.gate_tau = None if gate_tau is None else float(gate_tau)

        self.s_role_slices = {
            role: slice(2 + k * self.H, 2 + (k + 1) * self.H)
            for k, role in enumerate(self.role_names)
        }
        self.u_role_slices = {
            role: slice(k * self.H, (k + 1) * self.H)
            for k, role in enumerate(self.role_names)
        }
        self.t_role_index = {
            role: k for k, role in enumerate(self.role_names)
        }

        self.s_dim = 2 + 3 * self.H
        self.u_dim = 3 * self.H
        self.t_dim = 3
        self.dim = self.s_dim + self.u_dim + self.t_dim

    def gate(self, role, margin):
        if role not in self.gate_roles:
            return torch.ones_like(margin)

        positive_power = F.relu(margin).pow(self.gate_power)

        if self.gate_tau is None:
            return positive_power

        return positive_power / (
            self.gate_tau ** self.gate_power + positive_power
        )

    def unpack(self, xi, return_semantics=False):
        s = xi[:, :self.s_dim]
        u = xi[:, self.s_dim:self.s_dim + self.u_dim]
        t = xi[:, self.s_dim + self.u_dim:]
        semantics = {}

        for role in self.role_names:
            slab = s[:, self.s_role_slices[role]]
            local = u[:, self.u_role_slices[role]]
            threshold = t[
                :,
                self.t_role_index[role]:self.t_role_index[role] + 1,
            ]
            margin = local - threshold
            gate = self.gate(role, margin)
            active = (
                margin > 0.0
                if role in self.gate_roles
                else torch.ones_like(margin, dtype=torch.bool)
            )

            semantics[role] = {
                "s": slab,
                "u": local,
                "t": threshold,
                "margin": margin,
                "gate": gate,
                "active": active,
                "theta": slab * gate,
            }

        params = {
            "beta0": s[:, 0],
            "ell": s[:, 1],
            "w": semantics["input"]["theta"],
            "b": semantics["breakpoint"]["theta"],
            "a": semantics["output"]["theta"],
        }

        if return_semantics:
            return params, semantics

        return params

    def unit_contributions(self, X, xi):
        params = self.unpack(xi)
        x = X[:, 0]
        hidden = F.relu(
            params["w"][:, None, :] * x[None, :, None]
            - params["b"][:, None, :]
        )

        return params["a"][:, None, :] * hidden

    def forward(self, X, xi):
        params = self.unpack(xi)
        x = X[:, 0]
        hidden = F.relu(
            params["w"][:, None, :] * x[None, :, None]
            - params["b"][:, None, :]
        )
        units = params["a"][:, None, :] * hidden

        return (
            params["beta0"][:, None]
            + params["ell"][:, None] * x[None, :]
            + units.sum(dim=2)
        )


class DirectUnitBNNVI(nn.Module):
    def __init__(
        self,
        X,
        y,
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
    ):
        super().__init__()

        self.register_buffer("X", X)
        self.register_buffer("y", y)
        self.register_buffer(
            "sigma2",
            torch.tensor(float(sigma2), dtype=X.dtype),
        )

        self.family = family.lower()
        self.decoder = DirectUnitDecoder(
            H=H,
            gate_roles=gate_roles,
            gate_power=gate_power,
            gate_tau=gate_tau,
        )
        self.q0 = NBase(self.decoder.dim, init_sd=init_sd)
        self.init_sd = self.q0.init_sd
        self.flow = SemanticFlow(
            self.decoder.s_dim,
            self.decoder.u_dim,
            self.decoder.t_dim,
            K=K_flow,
            hidden_units=flow_hidden_units,
            num_hidden_layers=flow_hidden_layers,
            scale_clip=scale_clip,
        )

    def sample_posterior(self, R):
        z0 = self.q0.sample(R)
        xi, logdet = self.flow(z0, return_logdet=True)
        log_q = self.q0.log_prob(z0) - logdet

        return xi, log_q

    def log_likelihood(self, xi, X=None, y=None):
        X = self.X if X is None else X
        y = self.y if y is None else y
        pred = self.decoder(X, xi)

        if self.family == "gaussian":
            resid = y[None, :] - pred

            return -0.5 * (
                resid.square().sum(dim=1) / self.sigma2
                + y.numel() * torch.log(2.0 * torch.pi * self.sigma2)
            )

        if self.family in {"bernoulli", "binomial", "logistic"}:
            return -F.binary_cross_entropy_with_logits(
                pred,
                y[None, :].expand_as(pred),
                reduction="none",
            ).sum(dim=1)

        rate = torch.exp(pred.clamp(-20.0, 20.0))

        return (
            y[None, :] * pred
            - rate
            - torch.lgamma(y[None, :] + 1.0)
        ).sum(dim=1)

    def log_prior(self, xi):
        return -0.5 * (
            xi.square() + math.log(2.0 * math.pi)
        ).sum(dim=1)

    def elbo_draws(self, R):
        xi, log_q = self.sample_posterior(R)
        log_likelihood = self.log_likelihood(xi)
        log_prior = self.log_prior(xi)

        return {
            "xi": xi,
            "log_likelihood": log_likelihood,
            "log_prior": log_prior,
            "log_q": log_q,
            "kl": log_q - log_prior,
            "elbo": log_likelihood + log_prior - log_q,
        }

    def neg_elbo(self, R=64):
        return -self.elbo_draws(R)["elbo"].mean()

    @torch.no_grad()
    def predict(self, X_new, R=1000):
        xi, _ = self.sample_posterior(R)
        pred = self.decoder(X_new, xi)

        if self.family == "gaussian":
            return pred.mean(dim=0)

        if self.family in {"bernoulli", "binomial", "logistic"}:
            return torch.sigmoid(pred).mean(dim=0)

        return torch.exp(pred.clamp(-20.0, 20.0)).mean(dim=0)