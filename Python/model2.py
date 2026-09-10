


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


class LightweightAttentionConditioner(nn.Module):
    """
    Metadata-only attention with raw scalar fixed-coordinate values.

    ``shared_attention`` uses one Q/K attention map for both affine outputs.
    ``separate_attention`` uses independent Q/K maps for shift and scale.
    There is deliberately no value projection, self-attention, MLP, or target
    value input.  Each target receives a weighted scalar context followed by a
    target-specific scalar affine readout.
    """

    def __init__(
        self,
        dim,
        fixed_idx,
        target_idx,
        latent_metadata,
        token_dim=32,
        mode="shared_attention",
        scale_clip=2.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.token_dim = int(token_dim)
        self.mode = str(mode)
        self.scale_clip = float(scale_clip)
        if self.mode not in {"shared_attention", "separate_attention"}:
            raise ValueError(
                "mode must be shared_attention or separate_attention."
            )

        self.register_buffer(
            "fixed_idx", torch.as_tensor(fixed_idx, dtype=torch.long)
        )
        self.register_buffer(
            "target_idx", torch.as_tensor(target_idx, dtype=torch.long)
        )
        required = (
            "latent_type",
            "parameter_type",
            "group",
            "unit",
            "layer",
            "side",
        )
        for name in required:
            if name not in latent_metadata:
                raise ValueError(f"Missing latent metadata field: {name}")
            value = torch.as_tensor(latent_metadata[name], dtype=torch.long)
            if value.numel() != self.dim:
                raise ValueError(f"latent metadata '{name}' must have length dim.")
            self.register_buffer(f"meta_{name}", value)

        self.coordinate_embedding = nn.Embedding(self.dim, self.token_dim)
        for name in required:
            if name == "layer":
                metadata = self.meta_layer
            else:
                metadata = getattr(self, f"meta_{name}")
            setattr(
                self,
                f"{name}_embedding",
                nn.Embedding(int(metadata.max().item()) + 1, self.token_dim),
            )

        if self.mode == "shared_attention":
            self.query = nn.Linear(self.token_dim, self.token_dim, bias=False)
            self.key = nn.Linear(self.token_dim, self.token_dim, bias=False)
        else:
            self.shift_query = nn.Linear(
                self.token_dim, self.token_dim, bias=False
            )
            self.shift_key = nn.Linear(
                self.token_dim, self.token_dim, bias=False
            )
            self.scale_query = nn.Linear(
                self.token_dim, self.token_dim, bias=False
            )
            self.scale_key = nn.Linear(
                self.token_dim, self.token_dim, bias=False
            )

        target_dim = int(self.target_idx.numel())
        self.shift_slope = nn.Parameter(torch.zeros(target_dim))
        self.shift_bias = nn.Parameter(torch.zeros(target_dim))
        self.scale_slope = nn.Parameter(torch.zeros(target_dim))
        self.scale_bias = nn.Parameter(torch.zeros(target_dim))

    def _metadata_embedding(self, indices):
        out = self.coordinate_embedding(indices)
        for name in (
            "latent_type",
            "parameter_type",
            "group",
            "unit",
            "layer",
            "side",
        ):
            values = getattr(self, f"meta_{name}").index_select(0, indices)
            out = out + getattr(self, f"{name}_embedding")(values)
        return out

    def _context(self, values, query_layer, key_layer):
        target_meta = self._metadata_embedding(self.target_idx)
        fixed_meta = self._metadata_embedding(self.fixed_idx)
        scores = query_layer(target_meta) @ key_layer(fixed_meta).T
        scores = scores / math.sqrt(float(self.token_dim))
        weights = torch.softmax(scores, dim=-1)
        return torch.einsum("tf,rf->rt", weights, values)

    def forward(self, x):
        # Values are the raw fixed scalar coordinates, with no V projection.
        fixed_values = x.index_select(1, self.fixed_idx)
        if self.mode == "shared_attention":
            context = self._context(fixed_values, self.query, self.key)
            shift_context = context
            scale_context = context
        else:
            shift_context = self._context(
                fixed_values, self.shift_query, self.shift_key
            )
            scale_context = self._context(
                fixed_values, self.scale_query, self.scale_key
            )

        shift = self.shift_bias + self.shift_slope * shift_context
        # AffineCoupling applies c*tanh(raw/c), yielding exactly
        # c*tanh(b + a*context) for this lightweight readout.
        raw_log_scale = self.scale_clip * (
            self.scale_bias + self.scale_slope * scale_context
        )
        return raw_log_scale, shift


class ImprovedSeparateAttention(nn.Module):
    """
    Value-dependent, low-head, separate attention for triangular couplings.

    Target queries use structural metadata only. Source keys additionally use
    the current fixed-coordinate value, so the attention map changes with the
    posterior state without leaking target values into their own parameters.
    The two named branches have independent Q/K projections. Affine coupling
    uses ``shift/shape``; spline coupling uses ``width_height/derivative``.
    Values are the deterministic low-dimensional feature map

        [xi_i, xi_i**2, P_v e_i].

    No MLP, self-attention, target-value input, or Transformer FFN is used.
    """

    def __init__(
        self,
        dim,
        fixed_idx,
        target_idx,
        latent_metadata,
        token_dim=32,
        num_heads=2,
        branch_names=("shift", "shape"),
    ):
        super().__init__()
        self.dim = int(dim)
        self.token_dim = int(token_dim)
        self.num_heads = int(num_heads)
        self.branch_names = tuple(str(name) for name in branch_names)
        if len(self.branch_names) != 2 or len(set(self.branch_names)) != 2:
            raise ValueError("Improved attention requires two distinct branches.")
        if self.num_heads < 1:
            raise ValueError("num_heads must be positive.")
        if self.token_dim % self.num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads.")
        self.head_dim = self.token_dim // self.num_heads
        self.value_meta_dim = self.head_dim
        self.value_dim = 2 + self.value_meta_dim
        self.context_dim = self.num_heads * self.value_dim

        self.register_buffer(
            "fixed_idx", torch.as_tensor(fixed_idx, dtype=torch.long)
        )
        self.register_buffer(
            "target_idx", torch.as_tensor(target_idx, dtype=torch.long)
        )
        required = (
            "latent_type",
            "parameter_type",
            "group",
            "unit",
            "layer",
            "side",
        )
        self.metadata_names = required
        for name in required:
            if name not in latent_metadata:
                raise ValueError(f"Missing latent metadata field: {name}")
            value = torch.as_tensor(latent_metadata[name], dtype=torch.long)
            if value.numel() != self.dim:
                raise ValueError(f"latent metadata '{name}' must have length dim.")
            if bool((value < 0).any()):
                raise ValueError(f"latent metadata '{name}' must be nonnegative.")
            self.register_buffer(f"meta_{name}", value)

        self.coordinate_embedding = nn.Embedding(self.dim, self.token_dim)
        for name in required:
            metadata = getattr(self, f"meta_{name}")
            setattr(
                self,
                f"{name}_embedding",
                nn.Embedding(int(metadata.max().item()) + 1, self.token_dim),
            )

        self.query = nn.ModuleDict()
        self.key = nn.ModuleDict()
        for branch in self.branch_names:
            self.query[branch] = nn.ModuleList([
                nn.Linear(self.token_dim, self.head_dim, bias=False)
                for _ in range(self.num_heads)
            ])
            self.key[branch] = nn.ModuleList([
                nn.Linear(self.token_dim + 1, self.head_dim, bias=False)
                for _ in range(self.num_heads)
            ])

        self.value_metadata_projection = nn.Linear(
            self.token_dim,
            self.value_meta_dim,
            bias=False,
        )

    def metadata_embedding(self, indices):
        out = self.coordinate_embedding(indices)
        for name in self.metadata_names:
            values = getattr(self, f"meta_{name}").index_select(0, indices)
            out = out + getattr(self, f"{name}_embedding")(values)
        return out

    def _value_features(self, fixed_values, fixed_metadata):
        projected = self.value_metadata_projection(fixed_metadata)
        projected = projected[None, :, :].expand(fixed_values.shape[0], -1, -1)
        return torch.cat([
            fixed_values.unsqueeze(-1),
            fixed_values.square().unsqueeze(-1),
            projected,
        ], dim=-1)

    def _branch_context(self, x, branch):
        fixed_values = x.index_select(1, self.fixed_idx)
        fixed_metadata = self.metadata_embedding(self.fixed_idx)
        target_metadata = self.metadata_embedding(self.target_idx)
        key_input = torch.cat([
            fixed_metadata[None, :, :].expand(x.shape[0], -1, -1),
            fixed_values.unsqueeze(-1),
        ], dim=-1)

        weights = []
        for query_layer, key_layer in zip(
            self.query[branch], self.key[branch]
        ):
            query = query_layer(target_metadata)
            key = key_layer(key_input)
            score = torch.einsum("th,rfh->rtf", query, key)
            score = score / math.sqrt(float(self.head_dim))
            weights.append(torch.softmax(score, dim=-1))
        weights = torch.stack(weights, dim=1)

        values = self._value_features(fixed_values, fixed_metadata)
        context = torch.einsum("rhtf,rfv->rhtv", weights, values)
        context = context.permute(0, 2, 1, 3).reshape(
            x.shape[0], int(self.target_idx.numel()), self.context_dim
        )
        return context, weights

    def forward(self, x, return_attention=False):
        contexts = {}
        weights = {}
        for branch in self.branch_names:
            contexts[branch], weights[branch] = self._branch_context(x, branch)
        if not return_attention:
            return contexts
        return contexts, weights

    def attention_weights(self, x):
        _, weights = self.forward(x, return_attention=True)
        return weights


class AffineParameterHead(nn.Module):
    """Linear readouts from separate attention contexts to affine parameters."""

    def __init__(self, context_dim, scale_clip=2.0):
        super().__init__()
        self.scale_clip = float(scale_clip)
        # Head construction must not advance the outer RNG: affine and spline
        # should initialize their shared conditioners identically under one seed.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            self.shift = nn.Linear(int(context_dim), 1)
            self.log_scale = nn.Linear(int(context_dim), 1)
        nn.init.zeros_(self.shift.weight)
        nn.init.zeros_(self.shift.bias)
        nn.init.zeros_(self.log_scale.weight)
        nn.init.zeros_(self.log_scale.bias)

    def forward(self, contexts):
        shift = self.shift(contexts["shift"]).squeeze(-1)
        raw_log_scale = self.log_scale(contexts["shape"]).squeeze(-1)
        log_scale = self.scale_clip * torch.tanh(raw_log_scale)
        return log_scale, shift


class SplineParameterHead(nn.Module):
    """Linear attention-context readouts for a monotone RQ spline."""

    def __init__(
        self,
        context_dim,
        num_bins=8,
        tail_bound=3.0,
        min_bin_width=1e-3,
        min_bin_height=1e-3,
        min_derivative=1e-3,
    ):
        super().__init__()
        self.num_bins = int(num_bins)
        self.tail_bound = float(tail_bound)
        self.min_bin_width = float(min_bin_width)
        self.min_bin_height = float(min_bin_height)
        self.min_derivative = float(min_derivative)
        if self.num_bins < 2:
            raise ValueError("num_bins must be at least 2.")
        total = 2.0 * self.tail_bound
        if self.num_bins * self.min_bin_width >= total:
            raise ValueError("num_bins * min_bin_width must be below 2*tail_bound.")
        if self.num_bins * self.min_bin_height >= total:
            raise ValueError("num_bins * min_bin_height must be below 2*tail_bound.")
        if not 0.0 < self.min_derivative < 1.0:
            raise ValueError("min_derivative must lie in (0, 1).")

        context_dim = int(context_dim)
        # Preserve the outer RNG for conditioner-matched affine/spline starts.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            self.widths = nn.Linear(context_dim, self.num_bins)
            self.heights = nn.Linear(context_dim, self.num_bins)
            self.internal_derivatives = nn.Linear(
                context_dim, self.num_bins - 1
            )
        nn.init.zeros_(self.widths.weight)
        nn.init.zeros_(self.widths.bias)
        nn.init.zeros_(self.heights.weight)
        nn.init.zeros_(self.heights.bias)
        nn.init.zeros_(self.internal_derivatives.weight)
        derivative_bias = math.log(
            math.expm1(1.0 - self.min_derivative)
        )
        nn.init.constant_(self.internal_derivatives.bias, derivative_bias)

    def forward(self, contexts):
        total = 2.0 * self.tail_bound
        width_height_context = contexts["width_height"]
        derivative_context = contexts["derivative"]
        width_logits = self.widths(width_height_context)
        height_logits = self.heights(width_height_context)
        internal_logits = self.internal_derivatives(derivative_context)

        widths = self.min_bin_width + (
            total - self.min_bin_width * self.num_bins
        ) * torch.softmax(width_logits, dim=-1)
        heights = self.min_bin_height + (
            total - self.min_bin_height * self.num_bins
        ) * torch.softmax(height_logits, dim=-1)
        internal = self.min_derivative + F.softplus(internal_logits)
        boundary = torch.ones_like(internal[..., :1])
        derivatives = torch.cat([boundary, internal, boundary], dim=-1)
        return {
            "widths": widths,
            "heights": heights,
            "derivatives": derivatives,
        }


def _select_bin(values, cumulative, n_bins):
    index = torch.searchsorted(
        cumulative.contiguous(),
        values.unsqueeze(-1).contiguous(),
        right=True,
    ).squeeze(-1) - 1
    return index.clamp(min=0, max=int(n_bins) - 1)


def _gather_bin(values, index):
    return torch.gather(values, -1, index.unsqueeze(-1)).squeeze(-1)


def rational_quadratic_spline(
    inputs,
    widths,
    heights,
    derivatives,
    *,
    inverse=False,
    tail_bound=3.0,
    eps=1e-10,
):
    """Elementwise monotone rational-quadratic spline with identity tails."""

    n_bins = int(widths.shape[-1])
    if heights.shape[-1] != n_bins or derivatives.shape[-1] != n_bins + 1:
        raise ValueError("Invalid rational-quadratic spline parameter shapes.")
    bound = float(tail_bound)
    inside = (inputs >= -bound) & (inputs <= bound)
    safe_inputs = inputs.clamp(min=-bound, max=bound)

    cumulative_width = torch.cumsum(widths, dim=-1)
    cumulative_height = torch.cumsum(heights, dim=-1)
    left_width = torch.full_like(cumulative_width[..., :1], -bound)
    right_width = torch.full_like(cumulative_width[..., :1], bound)
    left_height = torch.full_like(cumulative_height[..., :1], -bound)
    right_height = torch.full_like(cumulative_height[..., :1], bound)
    cumwidths = torch.cat([
        left_width,
        cumulative_width[..., :-1] - bound,
        right_width,
    ], dim=-1)
    cumheights = torch.cat([
        left_height,
        cumulative_height[..., :-1] - bound,
        right_height,
    ], dim=-1)

    cumulative = cumheights if inverse else cumwidths
    index = _select_bin(safe_inputs, cumulative, n_bins)
    input_cumwidth = _gather_bin(cumwidths[..., :-1], index)
    input_bin_width = _gather_bin(widths, index)
    input_cumheight = _gather_bin(cumheights[..., :-1], index)
    input_bin_height = _gather_bin(heights, index)
    delta = input_bin_height / input_bin_width
    derivative_left = _gather_bin(derivatives[..., :-1], index)
    derivative_right = _gather_bin(derivatives[..., 1:], index)

    if inverse:
        y_delta = safe_inputs - input_cumheight
        derivative_term = derivative_left + derivative_right - 2.0 * delta
        a = y_delta * derivative_term + input_bin_height * (
            delta - derivative_left
        )
        b = input_bin_height * derivative_left - y_delta * derivative_term
        c = -delta * y_delta
        discriminant = (b.square() - 4.0 * a * c).clamp_min(0.0)
        root_denominator = -b - torch.sqrt(discriminant)
        linear_root = -c / torch.where(
            b.abs() > eps, b, torch.full_like(b, eps)
        )
        quadratic_root = 2.0 * c / torch.where(
            root_denominator.abs() > eps,
            root_denominator,
            torch.full_like(root_denominator, -eps),
        )
        theta = torch.where(a.abs() < eps, linear_root, quadratic_root)
        theta = theta.clamp(0.0, 1.0)
        outputs_inside = input_cumwidth + theta * input_bin_width
    else:
        theta = (safe_inputs - input_cumwidth) / input_bin_width
        theta = theta.clamp(0.0, 1.0)
        theta_one_minus = theta * (1.0 - theta)
        numerator = input_bin_height * (
            delta * theta.square()
            + derivative_left * theta_one_minus
        )
        denominator = delta + (
            derivative_left + derivative_right - 2.0 * delta
        ) * theta_one_minus
        outputs_inside = input_cumheight + numerator / denominator.clamp_min(eps)

    theta_one_minus = theta * (1.0 - theta)
    denominator = delta + (
        derivative_left + derivative_right - 2.0 * delta
    ) * theta_one_minus
    derivative_numerator = delta.square() * (
        derivative_right * theta.square()
        + 2.0 * delta * theta_one_minus
        + derivative_left * (1.0 - theta).square()
    )
    derivative = derivative_numerator / denominator.square().clamp_min(eps)
    logabsdet_inside = torch.log(derivative.clamp_min(eps))
    if inverse:
        logabsdet_inside = -logabsdet_inside

    outputs = torch.where(inside, outputs_inside, inputs)
    logabsdet = torch.where(
        inside, logabsdet_inside, torch.zeros_like(logabsdet_inside)
    )
    return outputs, logabsdet


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

        conditioner_type = {
            "attention": "full_attention",
        }.get(conditioner_type, conditioner_type)
        self.conditioner_type = conditioner_type

        if conditioner_type == "full_attention":
            self.conditioner = LatentAttentionConditioner(
                dim=self.dim,
                fixed_idx=fixed_idx,
                target_idx=target_idx,
                latent_metadata=latent_metadata,
                token_dim=token_dim,
                num_heads=num_heads,
            )
        elif conditioner_type in {
            "separate_attention",
            "shared_attention",
        }:
            self.conditioner = LightweightAttentionConditioner(
                dim=self.dim,
                fixed_idx=fixed_idx,
                target_idx=target_idx,
                latent_metadata=latent_metadata,
                token_dim=token_dim,
                mode=conditioner_type,
                scale_clip=self.scale_clip,
            )
        elif conditioner_type == "improved_separate_attention":
            self.conditioner = ImprovedSeparateAttention(
                dim=self.dim,
                fixed_idx=fixed_idx,
                target_idx=target_idx,
                latent_metadata=latent_metadata,
                token_dim=token_dim,
                num_heads=num_heads,
            )
            self.parameter_head = AffineParameterHead(
                context_dim=self.conditioner.context_dim,
                scale_clip=self.scale_clip,
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
        if self.conditioner_type == "improved_separate_attention":
            return self.parameter_head(self.conditioner(x))
        if self.conditioner_type in {
            "full_attention",
            "separate_attention",
            "shared_attention",
        }:
            raw_log_scale, shift = self.conditioner(x)
        else:
            fixed = x.index_select(1, self.fixed_idx)
            raw_log_scale, shift = self.conditioner(fixed)

        log_scale = self.scale_clip * torch.tanh(
            raw_log_scale / self.scale_clip
        )
        return log_scale, shift

    def attention_weights(self, x):
        if not hasattr(self.conditioner, "attention_weights"):
            return None
        return self.conditioner.attention_weights(x)

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


class SplineCoupling(nn.Module):
    """Triangular monotone rational-quadratic spline coupling."""

    def __init__(
        self,
        dim,
        mask,
        latent_metadata,
        conditioner_type="improved_separate_attention",
        token_dim=32,
        num_heads=2,
        num_bins=8,
        tail_bound=3.0,
        min_bin_width=1e-3,
        min_bin_height=1e-3,
        min_derivative=1e-3,
    ):
        super().__init__()
        self.dim = int(dim)
        self.conditioner_type = str(conditioner_type)
        self.num_bins = int(num_bins)
        self.tail_bound = float(tail_bound)
        self.register_buffer("mask", torch.as_tensor(mask, dtype=torch.bool))
        fixed_idx = torch.nonzero(self.mask, as_tuple=False).flatten()
        target_idx = torch.nonzero(~self.mask, as_tuple=False).flatten()
        self.register_buffer("fixed_idx", fixed_idx)
        self.register_buffer("target_idx", target_idx)
        if fixed_idx.numel() == 0 or target_idx.numel() == 0:
            raise ValueError("Each spline mask needs non-empty fixed and target sides.")
        if self.conditioner_type != "improved_separate_attention":
            raise ValueError(
                "SplineCoupling currently requires improved_separate_attention."
            )

        self.conditioner = ImprovedSeparateAttention(
            dim=self.dim,
            fixed_idx=fixed_idx,
            target_idx=target_idx,
            latent_metadata=latent_metadata,
            token_dim=token_dim,
            num_heads=num_heads,
            branch_names=("width_height", "derivative"),
        )
        self.parameter_head = SplineParameterHead(
            context_dim=self.conditioner.context_dim,
            num_bins=self.num_bins,
            tail_bound=self.tail_bound,
            min_bin_width=min_bin_width,
            min_bin_height=min_bin_height,
            min_derivative=min_derivative,
        )

    def params(self, x):
        return self.parameter_head(self.conditioner(x))

    def attention_weights(self, x):
        return self.conditioner.attention_weights(x)

    def forward(self, x, return_logdet=False):
        params = self.params(x)
        target = x.index_select(1, self.target_idx)
        transformed, element_logdet = rational_quadratic_spline(
            target,
            **params,
            inverse=False,
            tail_bound=self.tail_bound,
        )
        y = x.clone()
        y[:, self.target_idx] = transformed
        logdet = element_logdet.sum(dim=1)
        if return_logdet:
            return y, logdet
        return y

    def inverse(self, y, return_logdet=False):
        # Fixed coordinates are unchanged, so the same parameters are recovered.
        params = self.params(y)
        target = y.index_select(1, self.target_idx)
        transformed, element_logdet = rational_quadratic_spline(
            target,
            **params,
            inverse=True,
            tail_bound=self.tail_bound,
        )
        x = y.clone()
        x[:, self.target_idx] = transformed
        logdet = element_logdet.sum(dim=1)
        if return_logdet:
            return x, logdet
        return x

    def parameter_minima(self, x):
        params = self.params(x)
        return {
            "min_width": params["widths"].min(),
            "min_height": params["heights"].min(),
            "min_derivative": params["derivatives"].min(),
            "n_nonfinite_spline_parameters": sum(
                int((~torch.isfinite(value)).sum())
                for value in params.values()
            ),
        }


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
        "layer": zeros,
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
    K random balanced coupling cycles with affine or spline targets.

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
        coupling_type="affine",
        spline_num_bins=8,
        spline_tail_bound=3.0,
        spline_min_bin_width=1e-3,
        spline_min_bin_height=1e-3,
        spline_min_derivative=1e-3,
    ):
        super().__init__()
        self.dim = int(dim)
        self.K = int(K)
        self.conditioner_type = str(conditioner_type)
        self.coupling_type = str(coupling_type).lower()
        self.spline_num_bins = int(spline_num_bins)
        self.spline_tail_bound = float(spline_tail_bound)
        self.mask_seed = int(mask_seed)
        self.dependency_pairs = _normalize_dependency_pairs(
            dependency_pairs, self.dim
        )

        if self.dim < 2:
            raise ValueError("Alternating coupling flow requires latent dim >= 2.")
        if self.K < 1:
            raise ValueError("K must be positive for a coupling flow.")
        if self.coupling_type not in {"affine", "spline"}:
            raise ValueError("coupling_type must be affine or spline.")

        n_fixed = self.dim // 2
        accepted = None
        coverage = None
        semantic_mask = None
        if self.dependency_pairs and latent_metadata is not None:
            latent_type = torch.as_tensor(
                latent_metadata["latent_type"], dtype=torch.long
            )
            candidate = latent_type == 1
            if (
                bool(candidate.any())
                and bool((~candidate).any())
                and all(_pair_coverage([candidate], self.dependency_pairs))
            ):
                semantic_mask = candidate

        if semantic_mask is not None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.mask_seed)
            masks = [semantic_mask]
            for _ in range(self.K - 1):
                perm = torch.randperm(self.dim, generator=generator)
                mask = torch.zeros(self.dim, dtype=torch.bool)
                mask[perm[:n_fixed]] = True
                masks.append(mask)
            accepted = masks
            coverage = _pair_coverage(masks, self.dependency_pairs)
            self.mask_strategy = "dependency_aware_plus_random"
        else:
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
                    self.mask_strategy = "random_balanced"
                    break

        if accepted is None:
            missing = [
                pair for pair, ok in zip(self.dependency_pairs, coverage) if not ok
            ]
            raise RuntimeError(
                "Could not construct random coupling masks with full dependency "
                f"pair coverage; uncovered pairs include {missing[:10]}."
            )

        self.register_buffer("cycle_masks", torch.stack(accepted, dim=0))
        layers = []
        for mask in accepted:
            for direction_mask in (mask, ~mask):
                if self.coupling_type == "affine":
                    layer = AffineCoupling(
                        dim=self.dim,
                        mask=direction_mask,
                        latent_metadata=latent_metadata,
                        conditioner_type=conditioner_type,
                        hidden_units=hidden_units,
                        num_hidden_layers=num_hidden_layers,
                        scale_clip=scale_clip,
                        token_dim=token_dim,
                        num_heads=num_heads,
                    )
                else:
                    layer = SplineCoupling(
                        dim=self.dim,
                        mask=direction_mask,
                        latent_metadata=latent_metadata,
                        conditioner_type=conditioner_type,
                        token_dim=token_dim,
                        num_heads=num_heads,
                        num_bins=spline_num_bins,
                        tail_bound=spline_tail_bound,
                        min_bin_width=spline_min_bin_width,
                        min_bin_height=spline_min_bin_height,
                        min_derivative=spline_min_derivative,
                    )
                layers.append(layer)
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

    @torch.no_grad()
    def attention_diagnostics(self, posterior_x, max_draws=1000):
        """Return state-dependent attention weights along the forward path."""

        posterior_x = posterior_x[:int(max_draws)]
        base_x = self.inverse(posterior_x)
        z = base_x
        payload = []
        for layer_index, layer in enumerate(self.layers):
            weights = layer.attention_weights(z)
            if weights is not None:
                item = {
                    "layer": int(layer_index),
                    "fixed_idx": layer.fixed_idx.detach().cpu(),
                    "target_idx": layer.target_idx.detach().cpu(),
                }
                item.update({
                    branch: value.detach().cpu()
                    for branch, value in weights.items()
                })
                payload.append(item)
            z = layer(z)
        return payload

    @torch.no_grad()
    def numerical_sanity_check(self, base_x):
        """Check inverse/logdet consistency and spline parameter constraints."""

        minima = {
            "min_width": float("nan"),
            "min_height": float("nan"),
            "min_derivative": float("nan"),
        }
        n_nonfinite_spline_parameters = 0
        if self.coupling_type == "spline":
            values = {key: [] for key in minima}
            z = base_x
            for layer in self.layers:
                layer_minima = layer.parameter_minima(z)
                n_nonfinite_spline_parameters += int(
                    layer_minima["n_nonfinite_spline_parameters"]
                )
                for key in minima:
                    value = layer_minima[key]
                    values[key].append(float(value.detach().cpu()))
                z = layer(z)
            minima = {key: min(value) for key, value in values.items()}

        transformed, forward_logdet = self.forward(base_x, return_logdet=True)
        reconstructed, inverse_logdet = self.inverse(
            transformed, return_logdet=True
        )
        finite_forward = torch.isfinite(transformed)
        finite_inverse = torch.isfinite(reconstructed)
        finite_logdet = torch.isfinite(forward_logdet) & torch.isfinite(
            inverse_logdet
        )
        finite_pair = finite_forward & finite_inverse
        inverse_error = (
            float((base_x[finite_pair] - reconstructed[finite_pair]).abs().max())
            if bool(finite_pair.any()) else float("inf")
        )
        logdet_pair = torch.isfinite(forward_logdet + inverse_logdet)
        logdet_error = (
            float((forward_logdet[logdet_pair] + inverse_logdet[logdet_pair]).abs().max())
            if bool(logdet_pair.any()) else float("inf")
        )
        return {
            "max_inverse_error": inverse_error,
            "max_logdet_consistency_error": logdet_error,
            "n_nonfinite_forward": int((~finite_forward).sum()),
            "n_nonfinite_inverse": int((~finite_inverse).sum()),
            "n_nonfinite_logdet": int((~finite_logdet).sum()),
            "n_nonfinite_spline_parameters": int(
                n_nonfinite_spline_parameters
            ),
            **minima,
        }


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
    conditioner_type=None,
    coupling_type=None,
    spline_num_bins=8,
    spline_tail_bound=3.0,
    spline_min_bin_width=1e-3,
    spline_min_bin_height=1e-3,
    spline_min_derivative=1e-3,
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

    flow_spec = {
        "affine": ("mlp", "affine"),
        "attention_affine": ("full_attention", "affine"),
        "full_attention_affine": ("full_attention", "affine"),
        "separate_attention_affine": ("separate_attention", "affine"),
        "shared_attention_affine": ("shared_attention", "affine"),
        "improved_separate_attention_affine": (
            "improved_separate_attention", "affine"
        ),
        "improved_separate_attention_spline": (
            "improved_separate_attention", "spline"
        ),
    }.get(flow_type)

    if conditioner_type is None and coupling_type is None:
        if flow_spec is None:
            raise ValueError(
                "Unknown posterior flow_type. Use semantic, meanfield, a legacy "
                "affine type, improved_separate_attention_affine, or "
                "improved_separate_attention_spline."
            )
        conditioner_type, coupling_type = flow_spec
    else:
        if conditioner_type is None or coupling_type is None:
            raise ValueError(
                "conditioner_type and coupling_type must be supplied together."
            )
        conditioner_type = str(conditioner_type).lower()
        coupling_type = str(coupling_type).lower()

    if coupling_type == "spline" and (
        conditioner_type != "improved_separate_attention"
    ):
        raise ValueError(
            "Spline comparison requires improved_separate_attention."
        )

    if latent_metadata is None:
        latent_metadata = default_latent_metadata(s_dim, u_dim, t_dim)

    return AlternatingAffineFlow(
        dim=int(s_dim) + int(u_dim) + int(t_dim),
        latent_metadata=latent_metadata,
        K=K_flow,
        conditioner_type=conditioner_type,
        hidden_units=hidden_units,
        num_hidden_layers=num_hidden_layers,
        scale_clip=scale_clip,
        token_dim=token_dim,
        num_heads=num_heads,
        mask_seed=mask_seed,
        dependency_pairs=dependency_pairs,
        coupling_type=coupling_type,
        spline_num_bins=spline_num_bins,
        spline_tail_bound=spline_tail_bound,
        spline_min_bin_width=spline_min_bin_width,
        spline_min_bin_height=spline_min_bin_height,
        spline_min_derivative=spline_min_derivative,
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
    "E": 8,
    "e": 9,
    "W_hidden": 10,
    "b_hidden": 11,
    "Wout": 12,
    "bout": 13,
}

SIDE_IDS = {
    "none": 0,
    "input": 1,
    "output": 2,
    "linear": 3,
    "global": 4,
    "group": 5,
}


class MultiLayerGroupLayout:
    """Centralized scalar layout for stacked and embed-output grouped BNNs."""

    def __init__(
        self,
        input_dim,
        hidden_dims,
        out_dim,
        selection_mode,
        architecture_mode="stacked",
        embedding_dim=None,
        linear_skip=False,
    ):
        self.input_dim = int(input_dim)
        self.hidden_dims = tuple(int(width) for width in hidden_dims)
        self.out_dim = int(out_dim)
        self.selection_mode = str(selection_mode)
        self.architecture_mode = str(architecture_mode)
        self.embedding_dim = (
            None if embedding_dim is None else int(embedding_dim)
        )
        self.linear_skip = bool(linear_skip)

        if not self.hidden_dims or any(width < 1 for width in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive widths.")
        if self.architecture_mode not in {"stacked", "embed_output"}:
            raise ValueError(
                "architecture_mode must be stacked or embed_output."
            )
        if self.selection_mode not in {
            "unit_group",
            "feature_group",
            "feature_unit_induced_edge",
            "edge_group",
        }:
            raise ValueError(
                "selection_mode must be unit_group, feature_group, "
                "feature_unit_induced_edge, or edge_group."
            )
        if self.architecture_mode == "embed_output":
            if self.embedding_dim is None or self.embedding_dim < 1:
                raise ValueError(
                    "embed_output requires a positive embedding_dim."
                )
            if self.linear_skip:
                raise ValueError(
                    "linear_skip is only supported by architecture_mode='stacked'."
                )

        offsets = [0]
        for width in self.hidden_dims:
            offsets.append(offsets[-1] + width)
        self.unit_offsets = tuple(offsets)
        self.layer_slices = tuple(
            slice(offsets[layer], offsets[layer + 1])
            for layer in range(len(self.hidden_dims))
        )
        self.n_units = offsets[-1]
        self.unit_to_layer = tuple(
            layer
            for layer, width in enumerate(self.hidden_dims)
            for _ in range(width)
        )
        self.unit_to_local = tuple(
            local
            for width in self.hidden_dims
            for local in range(width)
        )
        self.unit_to_local_id = self.unit_to_local

        self.has_feature_gates = self.selection_mode in {
            "feature_group", "feature_unit_induced_edge"
        }
        self.has_unit_gates = self.selection_mode in {
            "unit_group", "feature_unit_induced_edge"
        }
        self.has_edge_gates = self.selection_mode == "edge_group"

        self.group_meta = []
        self.unit_groups = []
        self.feature_group_ids = []
        self.unit_group_ids = []
        self.edge_group_ids = {}

        if self.has_feature_gates:
            for feature in range(self.input_dim):
                group_id = len(self.group_meta)
                self.feature_group_ids.append(group_id)
                self.group_meta.append({
                    "group_id": group_id,
                    "selection_type": "feature",
                    "block": "input",
                    "layer": -1,
                    "feature": feature,
                    "side": "input",
                })

        if self.has_unit_gates:
            for global_unit in range(self.n_units):
                layer = self.unit_to_layer[global_unit]
                local = self.unit_to_local[global_unit]
                group_id = len(self.group_meta)
                self.unit_group_ids.append(group_id)
                meta = {
                    "group_id": group_id,
                    "selection_type": "unit",
                    "block": layer,
                    "layer": layer,
                    "unit": local,
                    "global_unit": global_unit,
                    "side": "group",
                }
                self.group_meta.append(meta)
                self.unit_groups.append(dict(meta))

        self.hidden_weight_names = tuple(
            f"W{layer + 1}" for layer in range(len(self.hidden_dims))
        )
        self.hidden_bias_names = tuple(
            f"b{layer + 1}" for layer in range(len(self.hidden_dims))
        )

        raw_specs = []
        if self.architecture_mode == "stacked":
            # For hidden_dims=(H,), this is the exact legacy ordering:
            # beta0, optional ell, W1, b1, W2.
            raw_specs.append(("beta0", (self.out_dim,), "beta0", "output", -1))
            if self.linear_skip:
                raw_specs.append((
                    "ell",
                    (self.out_dim, self.input_dim),
                    "ell",
                    "linear",
                    -1,
                ))
            previous = self.input_dim
            for layer, width in enumerate(self.hidden_dims):
                raw_specs.extend([
                    (
                        self.hidden_weight_names[layer],
                        (width, previous),
                        "W1" if layer == 0 else "W_hidden",
                        "hidden_weight",
                        layer,
                    ),
                    (
                        self.hidden_bias_names[layer],
                        (width,),
                        "b1" if layer == 0 else "b_hidden",
                        "hidden_bias",
                        layer,
                    ),
                ])
                previous = width
            self.output_weight_name = f"W{len(self.hidden_dims) + 1}"
            self.output_bias_name = "beta0"
            raw_specs.append((
                self.output_weight_name,
                (self.out_dim, self.hidden_dims[-1]),
                "W2" if len(self.hidden_dims) == 1 else "Wout",
                "output_weight",
                len(self.hidden_dims),
            ))
        else:
            raw_specs.extend([
                (
                    "E",
                    (self.embedding_dim, self.input_dim),
                    "E",
                    "embedding_weight",
                    -1,
                ),
                ("e", (self.embedding_dim,), "e", "embedding_bias", -1),
            ])
            previous = self.embedding_dim
            for layer, width in enumerate(self.hidden_dims):
                raw_specs.extend([
                    (
                        self.hidden_weight_names[layer],
                        (width, previous),
                        "W1" if layer == 0 else "W_hidden",
                        "hidden_weight",
                        layer,
                    ),
                    (
                        self.hidden_bias_names[layer],
                        (width,),
                        "b1" if layer == 0 else "b_hidden",
                        "hidden_bias",
                        layer,
                    ),
                ])
                previous = width
            self.output_weight_name = "Wout"
            self.output_bias_name = "bout"
            raw_specs.extend([
                (
                    self.output_weight_name,
                    (self.out_dim, self.hidden_dims[-1]),
                    "Wout",
                    "output_weight",
                    len(self.hidden_dims),
                ),
                (
                    self.output_bias_name,
                    (self.out_dim,),
                    "bout",
                    "output_bias",
                    len(self.hidden_dims),
                ),
            ])

        self.param_specs = []
        start = 0
        for name, shape, parameter_type, role, layer in raw_specs:
            n_elem = math.prod(shape)
            group_ids = [-1] * n_elem
            metadata_group_ids = [-1] * n_elem
            selection_type = "open"

            if role in {"hidden_weight", "hidden_bias"}:
                width = self.hidden_dims[layer]
                fan_in = shape[1] if role == "hidden_weight" else 1
                global_units = [
                    self.unit_offsets[layer] + unit
                    for unit in range(width)
                    for _ in range(fan_in)
                ]
                if self.has_unit_gates:
                    unit_ids = [
                        self.unit_group_ids[global_unit]
                        for global_unit in global_units
                    ]
                    metadata_group_ids = unit_ids
                    # Incoming row + bias is a diagnostic strength component;
                    # the unit gate itself is applied only to the activation.
                    group_ids = unit_ids
                    selection_type = "unit_context"

            if role == "output_weight" and self.has_unit_gates:
                last_offset = self.unit_offsets[-2]
                metadata_group_ids = [
                    self.unit_group_ids[last_offset + unit]
                    for _ in range(self.out_dim)
                    for unit in range(self.hidden_dims[-1])
                ]

            if self.has_feature_gates and not self.has_unit_gates:
                if role == "embedding_weight":
                    group_ids = (
                        list(self.feature_group_ids) * self.embedding_dim
                    )
                    metadata_group_ids = list(group_ids)
                    selection_type = "feature_context"
                elif role == "hidden_weight" and layer == 0 and (
                    self.architecture_mode == "stacked"
                ):
                    group_ids = (
                        list(self.feature_group_ids) * self.hidden_dims[0]
                    )
                    metadata_group_ids = list(group_ids)
                    selection_type = "feature_context"
                elif role == "linear":
                    group_ids = list(self.feature_group_ids) * self.out_dim
                    metadata_group_ids = list(group_ids)
                    selection_type = "feature_context"

            if self.has_feature_gates and self.has_unit_gates and (
                role == "embedding_weight"
            ):
                # E[:, k] retains original-variable identity. Hidden incoming
                # weights use target-unit metadata; dependency pairs below add
                # both feature and unit relationships without duplicating gates.
                metadata_group_ids = (
                    list(self.feature_group_ids) * self.embedding_dim
                )

            if self.has_edge_gates and role in {
                "embedding_weight", "hidden_weight", "output_weight", "linear"
            }:
                edge_ids = []
                fan_in = int(shape[1])
                for local_index in range(n_elem):
                    target = local_index // fan_in
                    source = local_index % fan_in
                    group_id = len(self.group_meta)
                    edge_ids.append(group_id)
                    if role == "embedding_weight":
                        edge_layer = "embedding"
                        edge_layer_index = -1
                    elif role == "hidden_weight":
                        edge_layer = f"hidden_layer_{layer + 1}"
                        edge_layer_index = int(layer)
                    elif role == "output_weight":
                        edge_layer = "output"
                        edge_layer_index = len(self.hidden_dims)
                    else:
                        edge_layer = "linear_skip"
                        edge_layer_index = len(self.hidden_dims)
                    self.group_meta.append({
                        "group_id": group_id,
                        "selection_type": "edge",
                        "block": edge_layer,
                        "layer": edge_layer_index,
                        "edge_layer": edge_layer,
                        "parameter": name,
                        "target": int(target),
                        "source": int(source),
                        "side": self._side({"role": role}),
                    })
                self.edge_group_ids[name] = tuple(edge_ids)
                group_ids = edge_ids
                metadata_group_ids = edge_ids
                selection_type = "edge"

            self.param_specs.append({
                "name": name,
                "shape": tuple(shape),
                "start": start,
                "end": start + n_elem,
                "parameter_type": parameter_type,
                "role": role,
                "layer": layer,
                "selection_type": selection_type,
                "group_ids": tuple(group_ids),
                "metadata_group_ids": tuple(metadata_group_ids),
            })
            start += n_elem

        self.s_dim = start
        self.u_dim = len(self.group_meta)
        if self.selection_mode == "feature_unit_induced_edge":
            self.threshold_roles = ("feature", "unit")
            self.group_threshold_ids = tuple(
                [0] * len(self.feature_group_ids)
                + [1] * len(self.unit_group_ids)
            )
        else:
            threshold_role = {
                "feature_group": "feature",
                "unit_group": "unit",
                "edge_group": "edge",
            }[self.selection_mode]
            self.threshold_roles = (threshold_role,)
            self.group_threshold_ids = tuple(0 for _ in range(self.u_dim))
        self.t_dim = len(self.threshold_roles)
        self.dim = self.s_dim + self.u_dim + self.t_dim

        self.feature_group_slice = slice(
            0, len(self.feature_group_ids)
        )
        unit_start = len(self.feature_group_ids)
        self.unit_group_slice = slice(
            unit_start, unit_start + len(self.unit_group_ids)
        )
        self.linear_weight_specs = tuple(
            item for item in self.param_specs
            if item["role"] in {
                "embedding_weight", "hidden_weight", "output_weight", "linear"
            }
        )
        self.n_candidate_edges = int(sum(
            item["end"] - item["start"] for item in self.linear_weight_specs
        ))
        self.primary_edge_parameter = self.linear_weight_specs[0]["name"]

        members = [[] for _ in range(self.u_dim)]
        for item in self.param_specs:
            role = item["role"]
            for local_index in range(item["end"] - item["start"]):
                scalar_index = item["start"] + local_index
                if self.has_feature_gates:
                    feature = None
                    if role == "embedding_weight":
                        feature = local_index % self.input_dim
                    elif (
                        role == "hidden_weight"
                        and item["layer"] == 0
                        and self.architecture_mode == "stacked"
                    ):
                        feature = local_index % self.input_dim
                    elif role == "linear":
                        feature = local_index % self.input_dim
                    if feature is not None:
                        members[self.feature_group_ids[feature]].append(
                            scalar_index
                        )
                if self.has_unit_gates and role in {
                    "hidden_weight", "hidden_bias"
                }:
                    fan_in = item["shape"][1] if role == "hidden_weight" else 1
                    local_unit = local_index // fan_in
                    global_unit = self.unit_offsets[item["layer"]] + local_unit
                    members[self.unit_group_ids[global_unit]].append(scalar_index)
                if self.has_edge_gates and role in {
                    "embedding_weight", "hidden_weight", "output_weight", "linear"
                }:
                    members[item["group_ids"][local_index]].append(scalar_index)
        self.group_scalar_indices = [tuple(indices) for indices in members]
        if any(not indices for indices in self.group_scalar_indices):
            raise RuntimeError("Every selectable group needs diagnostic slabs.")

        self.latent_metadata = self._build_latent_metadata()
        self.dependency_pairs = self._build_dependency_pairs()

    def _scalar_unit(self, item, local_index):
        role = item["role"]
        layer = item["layer"]
        if role == "hidden_weight":
            local = local_index // item["shape"][1]
            return self.unit_offsets[layer] + local, layer
        if role == "hidden_bias":
            return self.unit_offsets[layer] + local_index, layer
        if role == "output_weight":
            local = local_index % self.hidden_dims[-1]
            return self.unit_offsets[-2] + local, len(self.hidden_dims) - 1
        return -1, -1

    def _metadata_layer(self, item):
        role = item["role"]
        if self.architecture_mode == "embed_output":
            if role.startswith("embedding"):
                return 1
            if role.startswith("hidden"):
                return item["layer"] + 2
            if role.startswith("output"):
                return len(self.hidden_dims) + 2
        else:
            if role.startswith("hidden"):
                return item["layer"] + 1
            if role in {"output", "output_weight"}:
                return len(self.hidden_dims) + 1
        return 0

    @staticmethod
    def _side(item):
        return {
            "hidden_weight": "input",
            "hidden_bias": "input",
            "output_weight": "output",
            "embedding_weight": "input",
            "embedding_bias": "input",
            "linear": "linear",
            "output": "global",
            "output_bias": "global",
        }.get(item["role"], "none")

    def _build_latent_metadata(self):
        fields = {
            name: []
            for name in (
                "latent_type",
                "parameter_type",
                "group",
                "unit",
                "layer",
                "side",
            )
        }
        for item in self.param_specs:
            for local_index, group_id in enumerate(item["metadata_group_ids"]):
                global_unit, _ = self._scalar_unit(item, local_index)
                fields["latent_type"].append(0)
                fields["parameter_type"].append(
                    PARAMETER_TYPE_IDS[item["parameter_type"]]
                )
                fields["group"].append(group_id + 1 if group_id >= 0 else 0)
                fields["unit"].append(global_unit + 1 if global_unit >= 0 else 0)
                fields["layer"].append(self._metadata_layer(item))
                fields["side"].append(SIDE_IDS[self._side(item)])

        for meta in self.group_meta:
            fields["latent_type"].append(1)
            fields["parameter_type"].append(
                PARAMETER_TYPE_IDS["group_activation"]
            )
            fields["group"].append(int(meta["group_id"]) + 1)
            global_unit = int(meta.get("global_unit", -1))
            fields["unit"].append(global_unit + 1 if global_unit >= 0 else 0)
            layer = int(meta.get("layer", -1))
            fields["layer"].append(
                layer + 1 + int(self.architecture_mode == "embed_output")
                if layer >= 0 else 0
            )
            fields["side"].append(SIDE_IDS.get(meta.get("side", "none"), 0))

        threshold_sides = {
            "feature": "input",
            "unit": "group",
            "edge": "linear",
        }
        for threshold_role in self.threshold_roles:
            fields["latent_type"].append(2)
            fields["parameter_type"].append(PARAMETER_TYPE_IDS["threshold"])
            fields["group"].append(0)
            fields["unit"].append(0)
            fields["layer"].append(0)
            fields["side"].append(SIDE_IDS[threshold_sides[threshold_role]])
        return {
            name: torch.as_tensor(values, dtype=torch.long)
            for name, values in fields.items()
        }

    def _build_dependency_pairs(self):
        pairs = []
        threshold_start = self.s_dim + self.u_dim
        for group_id, scalar_indices in enumerate(self.group_scalar_indices):
            activation_index = self.s_dim + group_id
            pairs.extend(
                (scalar_index, activation_index)
                for scalar_index in scalar_indices
            )
            threshold_index = (
                threshold_start + self.group_threshold_ids[group_id]
            )
            pairs.append((activation_index, threshold_index))
        return tuple(pairs)


class GroupGateDecoder(nn.Module):
    """
    Multi-layer grouped BNN with one activation latent per selectable object.

    Unit selection always gates a hidden activation exactly once.  Feature
    selection gates each raw input before the first learned map (and therefore
    also before ``E`` in ``embed_output`` mode).  ``hidden_dims=(H,)`` with
    ``architecture_mode='stacked'`` retains the legacy shallow parameter order
    and forward equation.
    """

    def __init__(
        self,
        input_dim,
        H=None,
        hidden_dims=None,
        out_dim=1,
        selection_mode="unit_group",
        architecture_mode="stacked",
        embedding_dim=None,
        gate_type=None,
        gate_power=1.0,
        gate_tau=None,
        gate_delta=1.0,
        repu_power=None,
        linear_skip=False,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = (5 if H is None else int(H),)
        else:
            hidden_dims = tuple(int(width) for width in hidden_dims)
            if H is not None and len(hidden_dims) == 1 and int(H) != hidden_dims[0]:
                raise ValueError("H and hidden_dims specify different widths.")

        self.input_dim = int(input_dim)
        self.hidden_dims = tuple(hidden_dims)
        self.num_hidden_layers = len(self.hidden_dims)
        self.n_units = int(sum(self.hidden_dims))
        self.H = self.hidden_dims[0] if self.num_hidden_layers == 1 else self.n_units
        self.H_total = self.n_units
        self.out_dim = int(out_dim)
        self.selection_mode = str(selection_mode)
        self.architecture_mode = str(architecture_mode)
        self.embedding_dim = (
            None if embedding_dim is None else int(embedding_dim)
        )
        self.gate_power = float(gate_power)
        self.gate_tau = None if gate_tau is None else float(gate_tau)
        self.gate_delta = float(gate_delta)
        self.repu_power = None if repu_power is None else float(repu_power)
        self.linear_skip = bool(linear_skip)

        if self.gate_power <= 0.0:
            raise ValueError("gate_power must be positive.")
        if self.gate_tau is not None and self.gate_tau <= 0.0:
            raise ValueError("gate_tau must be positive or None.")
        if self.gate_delta <= 0.0:
            raise ValueError("gate_delta must be positive.")
        if self.repu_power is not None and self.repu_power <= 0.0:
            raise ValueError("repu_power must be positive or None.")

        self.gate_type = self._resolve_gate_type(gate_type)
        if self.gate_type in {"normalized_requ", "normalized_repu"} and (
            self.gate_tau is None
        ):
            self.gate_tau = 1.0

        self.layout = MultiLayerGroupLayout(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            out_dim=self.out_dim,
            selection_mode=self.selection_mode,
            architecture_mode=self.architecture_mode,
            embedding_dim=self.embedding_dim,
            linear_skip=self.linear_skip,
        )
        self.param_specs = self.layout.param_specs
        self.group_meta = self.layout.group_meta
        self.unit_groups = self.layout.unit_groups
        self.feature_group_ids = tuple(self.layout.feature_group_ids)
        self.unit_group_ids = tuple(self.layout.unit_group_ids)
        self.edge_group_ids = dict(self.layout.edge_group_ids)
        self.feature_group_slice = self.layout.feature_group_slice
        self.unit_group_slice = self.layout.unit_group_slice
        self.threshold_roles = tuple(self.layout.threshold_roles)
        self.has_feature_gates = self.layout.has_feature_gates
        self.has_unit_gates = self.layout.has_unit_gates
        self.has_edge_gates = self.layout.has_edge_gates
        self.n_candidate_edges = self.layout.n_candidate_edges
        self.primary_edge_parameter = self.layout.primary_edge_parameter
        self.unit_offsets = self.layout.unit_offsets
        self.layer_slices = self.layout.layer_slices
        self.unit_to_layer = self.layout.unit_to_layer
        self.unit_to_local = self.layout.unit_to_local
        self.unit_to_local_id = self.layout.unit_to_local_id
        self.s_dim = self.layout.s_dim
        self.u_dim = self.layout.u_dim
        self.t_dim = self.layout.t_dim
        self.dim = self.layout.dim

        self.register_buffer(
            "_group_threshold_ids",
            torch.as_tensor(
                self.layout.group_threshold_ids, dtype=torch.long
            ),
        )

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

    def _resolve_gate_type(self, gate_type):
        if gate_type is None:
            if self.gate_tau is None:
                if self.gate_power == 1.0:
                    return "relu"
                if self.gate_power == 2.0:
                    return "requ"
                return "repu"
            if self.gate_power == 2.0:
                return "normalized_requ"
            return "normalized_repu"
        aliases = {
            "normalized_requ": "normalized_requ",
            "normalized_repu": "normalized_repu",
            "smooth": "smooth_step",
            "step": "hard",
        }
        resolved = aliases.get(str(gate_type).lower(), str(gate_type).lower())
        allowed = {
            "hard",
            "relu",
            "requ",
            "repu",
            "normalized_requ",
            "normalized_repu",
            "smooth_step",
        }
        if resolved not in allowed:
            raise ValueError(f"Unknown grouped gate_type: {gate_type}")
        return resolved

    @property
    def activation_degree(self):
        return 1.0 if self.repu_power is None else self.repu_power

    def activate(self, x):
        positive = F.relu(x)
        if self.repu_power is None or self.repu_power == 1.0:
            return positive
        return positive.pow(self.repu_power)

    def group_gate(self, margin):
        if self.gate_type == "hard":
            return (margin > 0.0).to(margin.dtype)
        if self.gate_type == "smooth_step":
            # Masked evaluation avoids divisions outside the transition band.
            out = torch.zeros_like(margin)
            middle = (margin > 0.0) & (margin < self.gate_delta)
            high = margin >= self.gate_delta
            out[high] = 1.0
            if middle.any():
                m = margin[middle]
                logit = (
                    self.gate_delta / (self.gate_delta - m)
                    - self.gate_delta / m
                )
                out[middle] = torch.sigmoid(logit)
            return out

        power = {
            "relu": 1.0,
            "requ": 2.0,
            "normalized_requ": 2.0,
        }.get(self.gate_type, self.gate_power)
        positive = F.relu(margin).pow(power)
        if self.gate_type in {"normalized_requ", "normalized_repu"}:
            return positive / (self.gate_tau ** power + positive)
        return positive

    def split_latent(self, xi):
        s = xi[:, :self.s_dim]
        u = xi[:, self.s_dim:self.s_dim + self.u_dim]
        t = xi[:, self.s_dim + self.u_dim:]
        return s, u, t

    def group_semantics(self, xi):
        s, u, t = self.split_latent(xi)
        group_threshold = t.index_select(1, self._group_threshold_ids)
        margin = u - group_threshold
        return {
            "s": s,
            "u": u,
            "t": t,
            "group_threshold": group_threshold,
            "margin": margin,
            "gate": self.group_gate(margin),
            # PIP is always P(u > t), independent of the relaxed gate map.
            "active": margin > 0.0,
        }

    def group_slab_norms(self, xi):
        s = xi[:, :self.s_dim]
        norms = []
        for group_id in range(self.u_dim):
            idx = getattr(self, f"_group_scalar_{group_id}")
            norms.append(s.index_select(1, idx).square().sum(dim=1).sqrt())
        return torch.stack(norms, dim=1)

    def feature_semantics(self, xi):
        """Feature structural draws in original predictor order."""

        if not self.has_feature_gates:
            raise ValueError(
                "feature_semantics requires feature_group or "
                "feature_unit_induced_edge."
            )
        semantics = self.group_semantics(xi)
        group_norm = self.group_slab_norms(xi)
        block = self.feature_group_slice
        slab_strength = group_norm[:, block]
        gate = semantics["gate"][:, block]
        return {
            "active": semantics["active"][:, block],
            "gate": gate,
            "margin": semantics["margin"][:, block],
            "slab_strength": slab_strength,
            "effective_strength": gate * slab_strength,
        }

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
        input_parts = []
        output_parts = []
        strength_parts = []
        for layer, (weight_name, bias_name) in enumerate(zip(
            self.layout.hidden_weight_names,
            self.layout.hidden_bias_names,
        )):
            incoming = torch.cat(
                [slabs[weight_name], slabs[bias_name].unsqueeze(2)], dim=2
            ).norm(dim=2)
            if layer + 1 < self.num_hidden_layers:
                next_weight = slabs[self.layout.hidden_weight_names[layer + 1]]
                outgoing = next_weight.norm(dim=1)
            else:
                outgoing = slabs[self.layout.output_weight_name].norm(dim=1)
            input_parts.append(incoming)
            output_parts.append(outgoing)
            strength_parts.append(incoming * outgoing)
        return (
            torch.cat(input_parts, dim=1),
            torch.cat(output_parts, dim=1),
            torch.cat(strength_parts, dim=1),
        )

    def unit_semantics(self, xi):
        if not self.has_unit_gates:
            raise ValueError(
                "unit_semantics requires unit_group or "
                "feature_unit_induced_edge."
            )
        semantics = self.group_semantics(xi)
        slabs = self.unpack_slabs(xi)
        input_norm, output_norm, slab_strength = self._unit_strength_components(slabs)
        block = self.unit_group_slice
        gate = semantics["gate"][:, block]
        return {
            "active": semantics["active"][:, block],
            "gate": gate,
            "margin": semantics["margin"][:, block],
            "input_slab_norm": input_norm,
            "output_slab_norm": output_norm,
            "slab_strength": slab_strength,
            "effective_strength": gate * slab_strength,
            "strength_definition": "incoming_norm_times_outgoing_norm",
        }

    def edge_semantics(self, xi):
        """Literal or induced connectivity for every linear-map weight."""

        if self.selection_mode not in {
            "feature_unit_induced_edge", "edge_group"
        }:
            raise ValueError(
                "edge_semantics requires induced-edge or independent-edge mode."
            )
        params = self.unpack_slabs(xi)
        semantics = self.group_semantics(xi)
        feature = self.feature_semantics(xi) if self.has_feature_gates else None
        units = self.unit_semantics(xi) if self.has_unit_gates else None
        out = {}

        for item in self.layout.linear_weight_specs:
            name = item["name"]
            role = item["role"]
            shape = item["shape"]
            weight = params[name]

            if self.has_edge_gates:
                ids = getattr(self, f"_group_ids_{name}")
                active = semantics["active"].index_select(1, ids).reshape(
                    xi.shape[0], *shape
                )
                gate = semantics["gate"].index_select(1, ids).reshape(
                    xi.shape[0], *shape
                )
            elif role == "embedding_weight":
                active = feature["active"][:, None, :].expand_as(weight)
                gate = feature["gate"][:, None, :].expand_as(weight)
            elif role == "hidden_weight":
                layer = int(item["layer"])
                current = units["active"][:, self.layer_slices[layer]]
                current_gate = units["gate"][:, self.layer_slices[layer]]
                if layer == 0 and self.architecture_mode == "stacked":
                    active = current[:, :, None] & feature["active"][:, None, :]
                    gate = current_gate[:, :, None] * feature["gate"][:, None, :]
                elif layer == 0:
                    active = current[:, :, None].expand_as(weight)
                    gate = current_gate[:, :, None].expand_as(weight)
                else:
                    previous = units["active"][:, self.layer_slices[layer - 1]]
                    previous_gate = units["gate"][:, self.layer_slices[layer - 1]]
                    active = current[:, :, None] & previous[:, None, :]
                    gate = current_gate[:, :, None] * previous_gate[:, None, :]
            elif role == "output_weight":
                last = units["active"][:, self.layer_slices[-1]]
                last_gate = units["gate"][:, self.layer_slices[-1]]
                active = last[:, None, :].expand_as(weight)
                gate = last_gate[:, None, :].expand_as(weight)
            elif role == "linear":
                active = feature["active"][:, None, :].expand_as(weight)
                gate = feature["gate"][:, None, :].expand_as(weight)
            else:
                raise RuntimeError(f"Unsupported linear-map role: {role}")

            out[name] = {
                "parameter": name,
                "role": role,
                "layer": int(item["layer"]),
                "active": active,
                "gate": gate,
                "weight": weight,
                "effective_strength": gate * weight.abs(),
            }
        return out

    def unpack(
        self,
        xi,
        return_summary=False,
        beta_eps=0.05,
        sigmoid_active_threshold=0.5,
        force_all_on=False,
    ):
        del beta_eps, sigmoid_active_threshold, force_all_on
        params = self.unpack_slabs(xi)
        if not return_summary:
            return params
        semantics = self.group_semantics(xi)
        summary = {
            "group_pip": semantics["active"].float().mean(dim=0),
            "group_gate_mean": semantics["gate"].mean(dim=0),
            "group_margin_mean": semantics["margin"].mean(dim=0),
            "t_mean": semantics["t"].mean(dim=0),
            "t_sd": semantics["t"].std(dim=0),
            "threshold_roles": self.threshold_roles,
            "selection_mode": self.selection_mode,
            "architecture_mode": self.architecture_mode,
            "hidden_dims": self.hidden_dims,
            "gate_type": self.gate_type,
            "gate_power": self.gate_power,
            "gate_tau": self.gate_tau,
            "gate_delta": self.gate_delta,
            "repu_power": self.repu_power,
            "linear_skip": self.linear_skip,
        }
        if self.has_feature_gates:
            summary["feature_pip"] = summary["group_pip"][
                self.feature_group_slice
            ]
            summary["feature_gate_mean"] = summary["group_gate_mean"][
                self.feature_group_slice
            ]
        if self.has_unit_gates:
            summary["unit_pip"] = summary["group_pip"][self.unit_group_slice]
            summary["unit_gate_mean"] = summary["group_gate_mean"][
                self.unit_group_slice
            ]
        if self.has_edge_gates:
            summary["edge_pip"] = summary["group_pip"]
            summary["edge_gate_mean"] = summary["group_gate_mean"]
        return params, summary

    def structural_signature(self):
        specs = tuple(
            (item["name"], item["shape"], item["role"])
            for item in self.param_specs
        )
        return (
            "grouped_mlp_structure_v3",
            self.selection_mode,
            self.architecture_mode,
            self.hidden_dims,
            self.embedding_dim,
            specs,
            self.threshold_roles,
            self.n_candidate_edges,
            self.repu_power,
            self.linear_skip,
        )

    def compatibility_signature(self):
        return self.structural_signature() + (
            self.gate_type,
            self.gate_power,
            self.gate_tau,
            self.gate_delta,
        )

    def flow_metadata(self):
        return self.layout.latent_metadata

    def flow_dependency_pairs(self):
        return self.layout.dependency_pairs

    def forward(self, X, xi, force_all_on=False, structural_mask=None):
        params = self.unpack_slabs(xi)
        semantics = self.group_semantics(xi)
        R = xi.shape[0]
        n = X.shape[0]
        Xr = X[None, :, :].expand(R, n, self.input_dim)
        gate = (
            torch.ones_like(semantics["gate"])
            if force_all_on else semantics["gate"]
        )
        structural_mask = {} if structural_mask is None else structural_mask

        def fixed_mask(name, shape):
            value = structural_mask.get(name)
            if value is None:
                return None
            return torch.as_tensor(
                value, device=xi.device, dtype=xi.dtype
            ).reshape(shape)

        def effective_weight(name):
            weight = params[name]
            if not self.has_edge_gates:
                return weight
            ids = getattr(self, f"_group_ids_{name}")
            edge_gate = gate.index_select(1, ids).reshape_as(weight)
            edge_masks = structural_mask.get("edge", {})
            if name in edge_masks:
                edge_gate = edge_gate * torch.as_tensor(
                    edge_masks[name], device=xi.device, dtype=xi.dtype
                ).reshape((1,) + tuple(weight.shape[1:]))
            return weight * edge_gate

        # Feature selection is a gate on the original variable, including
        # before the optional embedding map. No duplicate parameter gate exists.
        if self.has_feature_gates:
            feature_gate = gate[:, self.feature_group_slice]
            feature_mask = fixed_mask("feature", (1, self.input_dim))
            if feature_mask is not None:
                feature_gate = feature_gate * feature_mask
            Xr = Xr * feature_gate[:, None, :]

        if self.architecture_mode == "embed_output":
            hidden = torch.bmm(Xr, effective_weight("E").transpose(1, 2))
            hidden = hidden + params["e"][:, None, :]
        else:
            hidden = Xr

        unit_gate = None
        if self.has_unit_gates:
            unit_gate = gate[:, self.unit_group_slice]
            unit_mask = fixed_mask("unit", (1, self.n_units))
            if unit_mask is not None:
                unit_gate = unit_gate * unit_mask

        for layer, (weight_name, bias_name) in enumerate(zip(
            self.layout.hidden_weight_names,
            self.layout.hidden_bias_names,
        )):
            hidden = torch.bmm(
                hidden, effective_weight(weight_name).transpose(1, 2)
            ) + params[bias_name][:, None, :]
            hidden = self.activate(hidden)
            if self.has_unit_gates:
                hidden = hidden * unit_gate[
                    :, None, self.layer_slices[layer]
                ]

        out = torch.bmm(
            hidden,
            effective_weight(self.layout.output_weight_name).transpose(1, 2),
        )
        out = out + params[self.layout.output_bias_name][:, None, :]
        if self.linear_skip:
            out = out + torch.bmm(
                Xr, effective_weight("ell").transpose(1, 2)
            )
        if self.out_dim == 1:
            return out[..., 0]
        return out


GroupedMLPDecoder = GroupGateDecoder


class GroupedBNNVI(LaSTBNNVI):
    """VI wrapper for stacked or embed-output grouped BNNs."""

    def __init__(
        self,
        X,
        y,
        input_dim=None,
        H=None,
        hidden_dims=None,
        out_dim=1,
        selection_mode="unit_group",
        architecture_mode="stacked",
        embedding_dim=None,
        family="gaussian",
        sigma2=1.0,
        init_sd=None,
        K_flow=4,
        flow_type="attention_affine",
        flow_hidden_units=128,
        flow_hidden_layers=2,
        scale_clip=2.0,
        flow_token_dim=32,
        flow_num_heads=2,
        flow_mask_seed=123,
        conditioner_type=None,
        coupling_type=None,
        spline_num_bins=8,
        spline_tail_bound=3.0,
        spline_min_bin_width=1e-3,
        spline_min_bin_height=1e-3,
        spline_min_derivative=1e-3,
        gate_type=None,
        gate_power=1.0,
        gate_tau=None,
        gate_delta=1.0,
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
        self.decoder = GroupedMLPDecoder(
            input_dim=input_dim,
            H=H,
            hidden_dims=hidden_dims,
            out_dim=out_dim,
            selection_mode=selection_mode,
            architecture_mode=architecture_mode,
            embedding_dim=embedding_dim,
            gate_type=gate_type,
            gate_power=gate_power,
            gate_tau=gate_tau,
            gate_delta=gate_delta,
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
            conditioner_type=conditioner_type,
            coupling_type=coupling_type,
            spline_num_bins=spline_num_bins,
            spline_tail_bound=spline_tail_bound,
            spline_min_bin_width=spline_min_bin_width,
            spline_min_bin_height=spline_min_bin_height,
            spline_min_derivative=spline_min_derivative,
        )
        self.conditioner_type = getattr(
            self.flow, "conditioner_type", conditioner_type
        )
        self.coupling_type = getattr(self.flow, "coupling_type", coupling_type)
        self.spline_num_bins = int(spline_num_bins)
        self.spline_tail_bound = float(spline_tail_bound)


@torch.no_grad()
def run_grouped_acceptance_tests(device=None, dtype=torch.float64):
    """Deterministic checks for the single-group/single-gate grouped model."""

    device = torch.device("cpu") if device is None else torch.device(device)
    # These checks are also run by the experiment scripts in float32.  A fixed
    # 1e-9 round-trip threshold is below float32 resolution and can reject a
    # numerically correct inverse.  Keep the original strictness for float64,
    # while scaling tolerances with the arithmetic precision actually used.
    dtype_epsilon = float(torch.finfo(dtype).eps)
    functional_tolerance = max(1e-10, 20.0 * dtype_epsilon)
    roundtrip_tolerance = max(1e-9, 100.0 * dtype_epsilon)
    improved_roundtrip_tolerance = max(1e-7, 800.0 * dtype_epsilon)
    independence_tolerance = max(1e-12, 10.0 * dtype_epsilon)
    dependence_floor = max(1e-12, dtype_epsilon)

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
    # The diagnostic context is the incoming W1 row plus b1. W2 is associated
    # metadata, not another location at which the unit gate is applied.
    assert len(decoder.layout.group_scalar_indices[group_id]) == 3

    w1_ids = specs["W1"]["group_ids"]
    b1_ids = specs["b1"]["group_ids"]
    w2_ids = specs["W2"]["group_ids"]
    w2_meta_ids = specs["W2"]["metadata_group_ids"]
    assert w1_ids[unit * 2:(unit + 1) * 2] == (group_id, group_id)
    assert b1_ids[unit] == group_id
    assert w2_ids[unit] == -1
    assert w2_meta_ids[unit] == group_id

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
    assert int(meta["layer"][w1_coord]) == 1

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
    assert torch.allclose(
        pred,
        expected,
        atol=functional_tolerance,
        rtol=functional_tolerance,
    )

    # Closing the one unit gate zeros its functional contribution even though
    # the slabs themselves remain available to the posterior flow.
    xi_closed = xi.clone()
    xi_closed[0, decoder.s_dim + group_id] = -0.2
    closed_pred = decoder(q, xi_closed)[0, 0]
    assert torch.allclose(
        closed_pred,
        torch.tensor(0.0, device=device, dtype=dtype),
        atol=functional_tolerance,
        rtol=0.0,
    )
    forced_pred = decoder(q, xi_closed, force_all_on=True)[0, 0]
    assert torch.allclose(
        forced_pred,
        raw_unit,
        atol=functional_tolerance,
        rtol=functional_tolerance,
    )

    smooth = GroupGateDecoder(
        input_dim=2,
        hidden_dims=(3,),
        gate_type="smooth_step",
        gate_delta=0.8,
    ).to(device=device, dtype=dtype)
    smooth_values = smooth.group_gate(torch.tensor(
        [-1.0, 0.0, 0.4, 0.8, 2.0], device=device, dtype=dtype
    ))
    assert torch.equal(smooth_values[[0, 1]], torch.zeros(2, device=device, dtype=dtype))
    assert torch.equal(smooth_values[[3, 4]], torch.ones(2, device=device, dtype=dtype))
    assert torch.allclose(smooth_values[2], torch.tensor(0.5, device=device, dtype=dtype))

    deep = GroupGateDecoder(
        input_dim=2,
        hidden_dims=(3, 2),
        selection_mode="unit_group",
    ).to(device=device, dtype=dtype)
    assert deep.u_dim == 5
    assert deep.unit_offsets == (0, 3, 5)
    assert deep.layer_slices == (slice(0, 3), slice(3, 5))
    assert tuple(deep.flow_metadata()["layer"].shape) == (deep.dim,)

    embedded = GroupGateDecoder(
        input_dim=3,
        hidden_dims=(2,),
        architecture_mode="embed_output",
        embedding_dim=4,
        selection_mode="feature_group",
    ).to(device=device, dtype=dtype)
    assert embedded.u_dim == 3
    assert embedded.layout.param_specs[0]["name"] == "E"

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

    # Joint feature+unit mode has separate role thresholds. Its induced edge
    # product is diagnostic only: the forward path applies the feature and
    # target-unit gates exactly once each.
    induced = GroupGateDecoder(
        input_dim=2,
        hidden_dims=(1,),
        selection_mode="feature_unit_induced_edge",
        gate_type="relu",
    ).to(device=device, dtype=dtype)
    assert induced.u_dim == 3
    assert induced.t_dim == 2
    assert induced.threshold_roles == ("feature", "unit")
    induced_specs = {item["name"]: item for item in induced.param_specs}
    induced_xi = torch.zeros(1, induced.dim, device=device, dtype=dtype)
    induced_xi[0, induced_specs["W1"]["start"]] = 1.0
    induced_xi[0, induced_specs["W2"]["start"]] = 1.0
    induced_xi[0, induced.s_dim + induced.feature_group_ids[0]] = 0.4
    induced_xi[0, induced.s_dim + induced.feature_group_ids[1]] = -0.2
    induced_xi[0, induced.s_dim + induced.unit_group_ids[0]] = 0.5
    induced_pred = induced(
        torch.tensor([[1.0, 0.0]], device=device, dtype=dtype), induced_xi
    )[0, 0]
    assert torch.allclose(
        induced_pred,
        torch.tensor(0.2, device=device, dtype=dtype),
        atol=functional_tolerance,
        rtol=functional_tolerance,
    )
    induced_edges = induced.edge_semantics(induced_xi)
    assert bool(induced_edges["W1"]["active"][0, 0, 0])
    assert not bool(induced_edges["W1"]["active"][0, 0, 1])

    # A second-layer bias belongs to its target unit because the gate closes
    # the whole post-activation output, not just the incoming weight row.
    deep_induced = GroupGateDecoder(
        input_dim=2,
        hidden_dims=(1, 1),
        selection_mode="feature_unit_induced_edge",
        gate_type="relu",
    ).to(device=device, dtype=dtype)
    deep_specs = {item["name"]: item for item in deep_induced.param_specs}
    deep_xi = torch.zeros(1, deep_induced.dim, device=device, dtype=dtype)
    deep_xi[0, deep_specs["b2"]["start"]] = 1.0
    deep_xi[0, deep_specs["W3"]["start"]] = 1.0
    for group_id in deep_induced.feature_group_ids:
        deep_xi[0, deep_induced.s_dim + group_id] = 1.0
    deep_xi[0, deep_induced.s_dim + deep_induced.unit_group_ids[0]] = 1.0
    deep_xi[0, deep_induced.s_dim + deep_induced.unit_group_ids[1]] = 0.6
    zero_input = torch.zeros(1, 2, device=device, dtype=dtype)
    bias_open = deep_induced(zero_input, deep_xi)[0, 0]
    deep_xi[0, deep_induced.s_dim + deep_induced.unit_group_ids[1]] = -0.1
    bias_closed = deep_induced(zero_input, deep_xi)[0, 0]
    assert torch.allclose(
        bias_open,
        torch.tensor(0.6, device=device, dtype=dtype),
        atol=functional_tolerance,
        rtol=functional_tolerance,
    )
    assert torch.allclose(
        bias_closed,
        torch.tensor(0.0, device=device, dtype=dtype),
        atol=functional_tolerance,
        rtol=0.0,
    )

    # Edge-only mode gates every linear weight independently and never treats
    # a bias as an edge.
    edge_only = GroupGateDecoder(
        input_dim=2,
        hidden_dims=(1,),
        selection_mode="edge_group",
        gate_type="relu",
    ).to(device=device, dtype=dtype)
    edge_specs = {item["name"]: item for item in edge_only.param_specs}
    assert edge_only.u_dim == edge_only.n_candidate_edges == 3
    assert all(
        all(group_id < 0 for group_id in edge_specs[name]["group_ids"])
        for name in ("beta0", "b1")
    )
    edge_xi = torch.zeros(1, edge_only.dim, device=device, dtype=dtype)
    edge_xi[0, edge_specs["W1"]["start"]] = 2.0
    edge_xi[0, edge_specs["b1"]["start"]] = 1.0
    edge_xi[0, edge_specs["W2"]["start"]] = 1.0
    edge_xi[0, edge_only.s_dim + edge_only.edge_group_ids["W1"][0]] = 0.5
    edge_xi[0, edge_only.s_dim + edge_only.edge_group_ids["W1"][1]] = -0.2
    edge_xi[0, edge_only.s_dim + edge_only.edge_group_ids["W2"][0]] = 0.4
    edge_pred = edge_only(
        torch.tensor([[1.0, 0.0]], device=device, dtype=dtype), edge_xi
    )[0, 0]
    assert torch.allclose(
        edge_pred,
        torch.tensor(0.8, device=device, dtype=dtype),
        atol=functional_tolerance,
        rtol=functional_tolerance,
    )

    embed_edge = GroupGateDecoder(
        input_dim=3,
        hidden_dims=(2, 2),
        architecture_mode="embed_output",
        embedding_dim=4,
        selection_mode="edge_group",
    ).to(device=device, dtype=dtype)
    assert embed_edge.n_candidate_edges == 26

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
    assert inverse_error < roundtrip_tolerance, (
        f"affine inverse error {inverse_error:.3e} exceeds "
        f"{roundtrip_tolerance:.3e} for {dtype}"
    )
    assert logdet_error < roundtrip_tolerance, (
        f"affine logdet consistency error {logdet_error:.3e} exceeds "
        f"{roundtrip_tolerance:.3e} for {dtype}"
    )

    layer = flow.layers[0]
    changed_target = x.clone()
    changed_target[:, layer.target_idx] += 3.0
    params_a = layer.params(x)
    params_b = layer.params(changed_target)
    conditioner_error = max(
        float((a - b).abs().max()) for a, b in zip(params_a, params_b)
    )
    assert conditioner_error < independence_tolerance, (
        f"conditioner target leakage {conditioner_error:.3e} exceeds "
        f"{independence_tolerance:.3e} for {dtype}"
    )
    assert bool(flow.transformed_coverage().all())
    assert bool(flow.dependency_pair_coverage().all())

    lightweight_errors = {}
    for conditioner_type in ("separate_attention", "shared_attention"):
        light_flow = AlternatingAffineFlow(
            dim=decoder.dim,
            latent_metadata=decoder.flow_metadata(),
            K=2,
            conditioner_type=conditioner_type,
            token_dim=8,
            num_heads=2,
            mask_seed=654,
            dependency_pairs=decoder.flow_dependency_pairs(),
        ).to(device=device, dtype=dtype)
        for light_layer in light_flow.layers:
            light_layer.conditioner.shift_slope.fill_(0.1)
            light_layer.conditioner.scale_slope.fill_(0.05)
        light_y, light_logdet = light_flow(x, return_logdet=True)
        light_x, light_inverse_logdet = light_flow.inverse(
            light_y, return_logdet=True
        )
        error = float((x - light_x).abs().max())
        det_error = float((light_logdet + light_inverse_logdet).abs().max())
        assert error < roundtrip_tolerance, (
            f"{conditioner_type} inverse error {error:.3e} exceeds "
            f"{roundtrip_tolerance:.3e} for {dtype}"
        )
        assert det_error < roundtrip_tolerance, (
            f"{conditioner_type} logdet error {det_error:.3e} exceeds "
            f"{roundtrip_tolerance:.3e} for {dtype}"
        )
        lightweight_errors[conditioner_type] = error

    improved_errors = {}
    improved_value_dependence = {}
    for coupling_type in ("affine", "spline"):
        improved_flow = AlternatingAffineFlow(
            dim=decoder.dim,
            latent_metadata=decoder.flow_metadata(),
            K=2,
            conditioner_type="improved_separate_attention",
            coupling_type=coupling_type,
            token_dim=8,
            num_heads=2,
            mask_seed=777,
            dependency_pairs=decoder.flow_dependency_pairs(),
            spline_num_bins=6,
        ).to(device=device, dtype=dtype)
        for improved_layer in improved_flow.layers:
            if coupling_type == "affine":
                improved_layer.parameter_head.shift.bias.fill_(0.03)
                improved_layer.parameter_head.log_scale.bias.fill_(0.02)
            else:
                bins = improved_layer.parameter_head.num_bins
                improved_layer.parameter_head.widths.bias.copy_(torch.linspace(
                    -0.15, 0.15, bins, device=device, dtype=dtype
                ))
                improved_layer.parameter_head.heights.bias.copy_(torch.linspace(
                    0.10, -0.10, bins, device=device, dtype=dtype
                ))

        improved_y, improved_logdet = improved_flow(
            x, return_logdet=True
        )
        improved_x, improved_inverse_logdet = improved_flow.inverse(
            improved_y, return_logdet=True
        )
        error = float((x - improved_x).abs().max())
        det_error = float(
            (improved_logdet + improved_inverse_logdet).abs().max()
        )
        assert error < improved_roundtrip_tolerance, (
            f"improved {coupling_type} inverse error {error:.3e} exceeds "
            f"{improved_roundtrip_tolerance:.3e} for {dtype}"
        )
        assert det_error < improved_roundtrip_tolerance, (
            f"improved {coupling_type} logdet error {det_error:.3e} exceeds "
            f"{improved_roundtrip_tolerance:.3e} for {dtype}"
        )
        assert bool(torch.isfinite(improved_y).all())
        assert bool(torch.isfinite(improved_logdet).all())

        first_layer = improved_flow.layers[0]
        weights = first_layer.attention_weights(x)
        branches = tuple(first_layer.conditioner.branch_names)
        expected_branches = (
            ("shift", "shape")
            if coupling_type == "affine"
            else ("width_height", "derivative")
        )
        assert branches == expected_branches
        target_changed = x.clone()
        target_changed[:, first_layer.target_idx] += 2.0
        target_weights = first_layer.attention_weights(target_changed)
        target_leakage = max(
            float((weights[branch] - target_weights[branch]).abs().max())
            for branch in branches
        )
        assert target_leakage < independence_tolerance, (
            f"improved {coupling_type} target leakage "
            f"{target_leakage:.3e} exceeds {independence_tolerance:.3e} "
            f"for {dtype}"
        )
        fixed_changed = x.clone()
        fixed_changed[:, first_layer.fixed_idx] += torch.linspace(
            -0.7,
            0.7,
            int(first_layer.fixed_idx.numel()),
            device=device,
            dtype=dtype,
        )[None, :]
        fixed_weights = first_layer.attention_weights(fixed_changed)
        value_change = max(
            float((weights[branch] - fixed_weights[branch]).abs().max())
            for branch in branches
        )
        assert value_change > dependence_floor, (
            f"improved {coupling_type} conditioner value response "
            f"{value_change:.3e} does not exceed {dependence_floor:.3e} "
            f"for {dtype}"
        )

        sanity = improved_flow.numerical_sanity_check(x)
        assert sanity["n_nonfinite_forward"] == 0
        assert sanity["n_nonfinite_inverse"] == 0
        assert sanity["n_nonfinite_logdet"] == 0
        assert sanity["n_nonfinite_spline_parameters"] == 0
        assert sanity["max_inverse_error"] < improved_roundtrip_tolerance, (
            f"improved {coupling_type} sanity inverse error "
            f"{sanity['max_inverse_error']:.3e} exceeds "
            f"{improved_roundtrip_tolerance:.3e} for {dtype}"
        )
        improved_errors[coupling_type] = {
            "inverse": error,
            "logdet": det_error,
            "target_value_leakage": target_leakage,
        }
        improved_value_dependence[coupling_type] = value_change

    return {
        "hidden_units": decoder.H,
        "scalar_slabs": decoder.s_dim,
        "unit_groups": decoder.u_dim,
        "group_activations": decoder.u_dim,
        "one_gate_per_unit": True,
        "unit_gate_applied_once": True,
        "force_all_gates_on_without_latent_mutation": True,
        "smooth_exact_zero_and_one": True,
        "multilayer_unit_mappings": True,
        "embed_output_feature_gate": True,
        "feature_unit_induced_edge": True,
        "induced_edge_not_double_gated": True,
        "second_layer_bias_unit_gated": True,
        "independent_edge_selection": True,
        "edge_biases_ungated": True,
        "embed_output_all_linear_edges": True,
        "plain_relu_gate": True,
        "relu_hidden_activation": True,
        "attention_input_output_metadata": True,
        "embedding_removed": True,
        "extra_output_projection_removed": True,
        "b2_removed": True,
        "linear_skip_default": False,
        "dtype": str(dtype),
        "functional_tolerance": functional_tolerance,
        "roundtrip_tolerance": roundtrip_tolerance,
        "improved_roundtrip_tolerance": improved_roundtrip_tolerance,
        "independence_tolerance": independence_tolerance,
        "flow_inverse_error": inverse_error,
        "flow_logdet_error": logdet_error,
        "conditioner_independence_error": conditioner_error,
        "lightweight_inverse_errors": lightweight_errors,
        "improved_transport_errors": improved_errors,
        "improved_attention_value_dependence": improved_value_dependence,
        "improved_attention_heads": 2,
        "spline_attention_branches": ("width_height", "derivative"),
        "spline_sanity_passed": True,
        "random_mask_dependency_coverage": bool(
            flow.dependency_pair_coverage().all()
        ),
        "flow_mask_strategy": flow.mask_strategy,
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
