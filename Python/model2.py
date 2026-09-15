"""Grouped DSS-LVR Bayesian neural network.

This cleaned implementation keeps only the structures used by the paper:

- stacked feed-forward networks;
- feature groups, unit groups, or feature+unit induced connectivity;
- bounded normalized-ReQU structural gates;
- full-attention affine coupling or affine IAF posterior transport;
- a diagonal Gaussian mean-field control.

Legacy spline, lightweight-attention, embed-output, and independent-edge paths
are intentionally removed.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .autoregressive import StackedIAF
except ImportError:
    from autoregressive import StackedIAF


PARAMETER_TYPE_IDS = {
    "none": 0,
    "beta0": 1,
    "W1": 2,
    "b1": 3,
    "W_hidden": 4,
    "b_hidden": 5,
    "Wout": 6,
    "group_activation": 7,
    "threshold": 8,
}

SIDE_IDS = {
    "none": 0,
    "input": 1,
    "output": 2,
    "global": 3,
    "group": 4,
}


class NBase(nn.Module):
    """Diagonal Gaussian base distribution with trainable location/scale."""

    def __init__(self, dim, init_sd=0.5):
        super().__init__()
        self.dim = int(dim)
        self.init_sd = float(init_sd)
        self.loc = nn.Parameter(torch.zeros(self.dim))
        self.raw_log_scale = nn.Parameter(
            torch.full((self.dim,), math.log(self.init_sd))
        )

    def sample(self, R):
        eps = torch.randn(
            int(R), self.dim, device=self.loc.device, dtype=self.loc.dtype
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
    """Mean-field transport."""

    flow_type = "meanfield"

    def forward(self, x, return_logdet=False):
        if return_logdet:
            return x, x.new_zeros(x.shape[0])
        return x

    def inverse(self, x, return_logdet=False):
        if return_logdet:
            return x, x.new_zeros(x.shape[0])
        return x


class FullAttentionConditioner(nn.Module):
    """Full self-attention + target cross-attention affine conditioner."""

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
            raise ValueError("token_dim must be divisible by num_heads.")

        self.register_buffer(
            "fixed_idx", torch.as_tensor(fixed_idx, dtype=torch.long)
        )
        self.register_buffer(
            "target_idx", torch.as_tensor(target_idx, dtype=torch.long)
        )

        required = ("latent_type", "parameter_type", "group", "unit", "side")
        for name in required:
            value = torch.as_tensor(latent_metadata[name], dtype=torch.long)
            if value.numel() != self.dim:
                raise ValueError(f"metadata {name!r} must have length dim.")
            self.register_buffer(f"meta_{name}", value)

        self.value_projection = nn.Linear(1, self.token_dim)
        self.coordinate_embedding = nn.Embedding(self.dim, self.token_dim)
        self.latent_type_embedding = nn.Embedding(
            int(self.meta_latent_type.max()) + 1, self.token_dim
        )
        self.parameter_type_embedding = nn.Embedding(
            int(self.meta_parameter_type.max()) + 1, self.token_dim
        )
        self.group_embedding = nn.Embedding(
            int(self.meta_group.max()) + 1, self.token_dim
        )
        self.unit_embedding = nn.Embedding(
            int(self.meta_unit.max()) + 1, self.token_dim
        )
        self.side_embedding = nn.Embedding(
            int(self.meta_side.max()) + 1, self.token_dim
        )

        self.self_attention = nn.MultiheadAttention(
            self.token_dim, self.num_heads, batch_first=True
        )
        self.cross_attention = nn.MultiheadAttention(
            self.token_dim, self.num_heads, batch_first=True
        )
        self.fixed_norm = nn.LayerNorm(self.token_dim)
        self.target_norm = nn.LayerNorm(self.token_dim)
        self.readout = nn.Linear(self.token_dim, 2)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def _metadata_embedding(self, indices):
        return (
            self.coordinate_embedding(indices)
            + self.latent_type_embedding(self.meta_latent_type.index_select(0, indices))
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
            + self._metadata_embedding(self.fixed_idx)[None, :, :]
        )
        fixed_context, _ = self.self_attention(
            fixed_tokens, fixed_tokens, fixed_tokens, need_weights=False
        )
        fixed_context = self.fixed_norm(fixed_tokens + fixed_context)

        target_query = self._metadata_embedding(self.target_idx)[None, :, :]
        target_query = target_query.expand(x.shape[0], -1, -1)
        target_context, _ = self.cross_attention(
            target_query, fixed_context, fixed_context, need_weights=False
        )
        target_context = self.target_norm(target_query + target_context)
        raw = self.readout(target_context)
        return raw[..., 0], raw[..., 1]


class AffineCoupling(nn.Module):
    def __init__(
        self,
        dim,
        mask,
        latent_metadata,
        scale_clip=2.0,
        token_dim=32,
        num_heads=4,
    ):
        super().__init__()
        self.dim = int(dim)
        self.scale_clip = float(scale_clip)
        self.register_buffer("mask", torch.as_tensor(mask, dtype=torch.bool))
        fixed_idx = torch.nonzero(self.mask, as_tuple=False).flatten()
        target_idx = torch.nonzero(~self.mask, as_tuple=False).flatten()
        if fixed_idx.numel() == 0 or target_idx.numel() == 0:
            raise ValueError("Each coupling mask needs fixed and target coordinates.")
        self.register_buffer("fixed_idx", fixed_idx)
        self.register_buffer("target_idx", target_idx)
        self.conditioner = FullAttentionConditioner(
            dim=self.dim,
            fixed_idx=fixed_idx,
            target_idx=target_idx,
            latent_metadata=latent_metadata,
            token_dim=token_dim,
            num_heads=num_heads,
        )

    def params(self, x):
        raw_log_scale, shift = self.conditioner(x)
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
        log_scale, shift = self.params(y)
        x = y.clone()
        target = y.index_select(1, self.target_idx)
        x[:, self.target_idx] = (target - shift) * torch.exp(-log_scale)
        logdet = -log_scale.sum(dim=1)
        if return_logdet:
            return x, logdet
        return x


def _normalize_pairs(pairs, dim):
    out = []
    seen = set()
    for a, b in pairs or ():
        a, b = int(a), int(b)
        if a == b:
            continue
        if not (0 <= a < dim and 0 <= b < dim):
            raise ValueError("dependency pair contains an invalid coordinate.")
        key = tuple(sorted((a, b)))
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _pair_coverage(masks, pairs):
    return [any(bool(mask[a] != mask[b]) for mask in masks) for a, b in pairs]


class FullAttentionAffineFlow(nn.Module):
    """Complementary affine coupling cycles with the full attention conditioner."""

    def __init__(
        self,
        dim,
        latent_metadata,
        K=4,
        scale_clip=2.0,
        token_dim=32,
        num_heads=4,
        seed=123,
        dependency_pairs=None,
        max_mask_tries=1000,
    ):
        super().__init__()
        self.dim = int(dim)
        self.K = int(K)
        self.seed = int(seed)
        self.flow_type = "attention"
        self.dependency_pairs = _normalize_pairs(dependency_pairs, self.dim)
        if self.dim < 2 or self.K < 1:
            raise ValueError("Attention flow requires dim >= 2 and K >= 1.")

        n_fixed = self.dim // 2
        semantic_mask = torch.as_tensor(
            latent_metadata["latent_type"], dtype=torch.long
        ) == 1
        masks = None

        if (
            bool(semantic_mask.any())
            and bool((~semantic_mask).any())
            and all(_pair_coverage([semantic_mask], self.dependency_pairs))
        ):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed)
            masks = [semantic_mask]
            for _ in range(self.K - 1):
                perm = torch.randperm(self.dim, generator=generator)
                mask = torch.zeros(self.dim, dtype=torch.bool)
                mask[perm[:n_fixed]] = True
                masks.append(mask)
            self.mask_strategy = "semantic_plus_random"
        else:
            for attempt in range(int(max_mask_tries)):
                generator = torch.Generator(device="cpu")
                generator.manual_seed(self.seed + attempt)
                candidate = []
                for _ in range(self.K):
                    perm = torch.randperm(self.dim, generator=generator)
                    mask = torch.zeros(self.dim, dtype=torch.bool)
                    mask[perm[:n_fixed]] = True
                    candidate.append(mask)
                if all(_pair_coverage(candidate, self.dependency_pairs)):
                    masks = candidate
                    self.mask_strategy = "random_balanced"
                    break
        if masks is None:
            raise RuntimeError("Could not construct dependency-covering masks.")

        self.register_buffer("cycle_masks", torch.stack(masks, dim=0))
        self.layers = nn.ModuleList()
        for mask in masks:
            for direction in (mask, ~mask):
                self.layers.append(
                    AffineCoupling(
                        dim=self.dim,
                        mask=direction,
                        latent_metadata=latent_metadata,
                        scale_clip=scale_clip,
                        token_dim=token_dim,
                        num_heads=num_heads,
                    )
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
    def numerical_sanity_check(self, base_x):
        transformed, forward_logdet = self.forward(base_x, return_logdet=True)
        reconstructed, inverse_logdet = self.inverse(
            transformed, return_logdet=True
        )
        return {
            "max_inverse_error": float((base_x - reconstructed).abs().max()),
            "max_logdet_consistency_error": float(
                (forward_logdet + inverse_logdet).abs().max()
            ),
            "n_nonfinite_forward": int((~torch.isfinite(transformed)).sum()),
            "n_nonfinite_inverse": int((~torch.isfinite(reconstructed)).sum()),
            "n_nonfinite_logdet": int(
                (~torch.isfinite(forward_logdet)).sum()
                + (~torch.isfinite(inverse_logdet)).sum()
            ),
        }


class MultiLayerGroupLayout:
    """Scalar layout for the stacked grouped BNN."""

    def __init__(self, input_dim, hidden_dims, out_dim, selection_mode):
        self.input_dim = int(input_dim)
        self.hidden_dims = tuple(int(width) for width in hidden_dims)
        self.out_dim = int(out_dim)
        self.selection_mode = str(selection_mode)
        if not self.hidden_dims or any(width < 1 for width in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive widths.")
        if self.selection_mode not in {
            "feature_group", "unit_group", "feature_unit_induced_edge"
        }:
            raise ValueError(
                "selection_mode must be feature_group, unit_group, or "
                "feature_unit_induced_edge."
            )

        offsets = [0]
        for width in self.hidden_dims:
            offsets.append(offsets[-1] + width)
        self.unit_offsets = tuple(offsets)
        self.layer_slices = tuple(
            slice(offsets[i], offsets[i + 1])
            for i in range(len(self.hidden_dims))
        )
        self.n_units = offsets[-1]
        self.unit_to_layer = tuple(
            layer
            for layer, width in enumerate(self.hidden_dims)
            for _ in range(width)
        )
        self.unit_to_local = tuple(
            local for width in self.hidden_dims for local in range(width)
        )

        self.has_feature_gates = self.selection_mode in {
            "feature_group", "feature_unit_induced_edge"
        }
        self.has_unit_gates = self.selection_mode in {
            "unit_group", "feature_unit_induced_edge"
        }

        self.group_meta = []
        self.feature_group_ids = []
        self.unit_group_ids = []
        self.unit_groups = []
        if self.has_feature_gates:
            for feature in range(self.input_dim):
                gid = len(self.group_meta)
                self.feature_group_ids.append(gid)
                self.group_meta.append({
                    "group_id": gid,
                    "selection_type": "feature",
                    "layer": -1,
                    "feature": feature,
                    "side": "input",
                })
        if self.has_unit_gates:
            for global_unit in range(self.n_units):
                layer = self.unit_to_layer[global_unit]
                local = self.unit_to_local[global_unit]
                gid = len(self.group_meta)
                self.unit_group_ids.append(gid)
                meta = {
                    "group_id": gid,
                    "selection_type": "unit",
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
        self.output_weight_name = f"W{len(self.hidden_dims) + 1}"
        self.output_bias_name = "beta0"

        raw_specs = [("beta0", (self.out_dim,), "beta0", "output", -1)]
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
        raw_specs.append((
            self.output_weight_name,
            (self.out_dim, self.hidden_dims[-1]),
            "Wout",
            "output_weight",
            len(self.hidden_dims),
        ))

        self.param_specs = []
        start = 0
        for name, shape, parameter_type, role, layer in raw_specs:
            n_elem = math.prod(shape)
            metadata_group_ids = [-1] * n_elem
            if role == "hidden_weight" and self.has_unit_gates:
                fan_in = shape[1]
                metadata_group_ids = [
                    self.unit_group_ids[self.unit_offsets[layer] + target]
                    for target in range(shape[0])
                    for _ in range(fan_in)
                ]
            elif role == "hidden_bias" and self.has_unit_gates:
                metadata_group_ids = [
                    self.unit_group_ids[self.unit_offsets[layer] + target]
                    for target in range(shape[0])
                ]
            elif role == "output_weight" and self.has_unit_gates:
                last_offset = self.unit_offsets[-2]
                metadata_group_ids = [
                    self.unit_group_ids[last_offset + source]
                    for _ in range(self.out_dim)
                    for source in range(self.hidden_dims[-1])
                ]
            elif (
                role == "hidden_weight"
                and layer == 0
                and self.has_feature_gates
                and not self.has_unit_gates
            ):
                metadata_group_ids = [
                    self.feature_group_ids[source]
                    for _ in range(shape[0])
                    for source in range(self.input_dim)
                ]

            self.param_specs.append({
                "name": name,
                "shape": tuple(shape),
                "start": start,
                "end": start + n_elem,
                "parameter_type": parameter_type,
                "role": role,
                "layer": layer,
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
        elif self.selection_mode == "feature_group":
            self.threshold_roles = ("feature",)
            self.group_threshold_ids = tuple(0 for _ in range(self.u_dim))
        else:
            self.threshold_roles = ("unit",)
            self.group_threshold_ids = tuple(0 for _ in range(self.u_dim))
        self.t_dim = len(self.threshold_roles)
        self.dim = self.s_dim + self.u_dim + self.t_dim

        self.feature_group_slice = slice(0, len(self.feature_group_ids))
        unit_start = len(self.feature_group_ids)
        self.unit_group_slice = slice(
            unit_start, unit_start + len(self.unit_group_ids)
        )
        self.linear_weight_specs = tuple(
            item for item in self.param_specs
            if item["role"] in {"hidden_weight", "output_weight"}
        )
        self.n_candidate_edges = int(sum(
            item["end"] - item["start"] for item in self.linear_weight_specs
        ))

        self.group_scalar_indices = self._build_group_members()
        self.latent_metadata = self._build_latent_metadata()
        self.dependency_pairs = self._build_dependency_pairs()

    def _spec(self, name):
        return next(item for item in self.param_specs if item["name"] == name)

    def _build_group_members(self):
        members = [set() for _ in range(self.u_dim)]

        if self.has_feature_gates:
            first = self._spec(self.hidden_weight_names[0])
            fan_in = first["shape"][1]
            for target in range(first["shape"][0]):
                for source in range(fan_in):
                    scalar = first["start"] + target * fan_in + source
                    members[self.feature_group_ids[source]].add(scalar)

        if self.has_unit_gates:
            for layer, (w_name, b_name) in enumerate(zip(
                self.hidden_weight_names, self.hidden_bias_names
            )):
                w = self._spec(w_name)
                b = self._spec(b_name)
                fan_in = w["shape"][1]
                width = w["shape"][0]
                for local in range(width):
                    global_unit = self.unit_offsets[layer] + local
                    gid = self.unit_group_ids[global_unit]
                    for source in range(fan_in):
                        members[gid].add(w["start"] + local * fan_in + source)
                    members[gid].add(b["start"] + local)

                    if layer + 1 < len(self.hidden_dims):
                        outgoing = self._spec(self.hidden_weight_names[layer + 1])
                        next_width = outgoing["shape"][0]
                        out_fan_in = outgoing["shape"][1]
                        for target in range(next_width):
                            members[gid].add(
                                outgoing["start"] + target * out_fan_in + local
                            )
                    else:
                        outgoing = self._spec(self.output_weight_name)
                        fan = outgoing["shape"][1]
                        for target in range(self.out_dim):
                            members[gid].add(
                                outgoing["start"] + target * fan + local
                            )

        result = [tuple(sorted(indices)) for indices in members]
        if any(len(indices) == 0 for indices in result):
            raise RuntimeError("Every structural group must contain slab coordinates.")
        return result

    def _scalar_unit(self, item, local_index):
        role = item["role"]
        if role == "hidden_weight":
            local = local_index // item["shape"][1]
            return self.unit_offsets[item["layer"]] + local
        if role == "hidden_bias":
            return self.unit_offsets[item["layer"]] + local_index
        if role == "output_weight":
            local = local_index % self.hidden_dims[-1]
            return self.unit_offsets[-2] + local
        return -1

    @staticmethod
    def _side(role):
        if role in {"hidden_weight", "hidden_bias"}:
            return "input"
        if role == "output_weight":
            return "output"
        if role == "output":
            return "global"
        return "none"

    def _build_latent_metadata(self):
        fields = {name: [] for name in (
            "latent_type", "parameter_type", "group", "unit", "side"
        )}
        for item in self.param_specs:
            for local_index, gid in enumerate(item["metadata_group_ids"]):
                global_unit = self._scalar_unit(item, local_index)
                fields["latent_type"].append(0)
                fields["parameter_type"].append(
                    PARAMETER_TYPE_IDS[item["parameter_type"]]
                )
                fields["group"].append(gid + 1 if gid >= 0 else 0)
                fields["unit"].append(global_unit + 1 if global_unit >= 0 else 0)
                fields["side"].append(SIDE_IDS[self._side(item["role"])])

        for meta in self.group_meta:
            fields["latent_type"].append(1)
            fields["parameter_type"].append(
                PARAMETER_TYPE_IDS["group_activation"]
            )
            fields["group"].append(int(meta["group_id"]) + 1)
            global_unit = int(meta.get("global_unit", -1))
            fields["unit"].append(global_unit + 1 if global_unit >= 0 else 0)
            fields["side"].append(SIDE_IDS[meta["side"]])

        threshold_side = {"feature": "input", "unit": "group"}
        for role in self.threshold_roles:
            fields["latent_type"].append(2)
            fields["parameter_type"].append(PARAMETER_TYPE_IDS["threshold"])
            fields["group"].append(0)
            fields["unit"].append(0)
            fields["side"].append(SIDE_IDS[threshold_side[role]])

        return {
            name: torch.as_tensor(values, dtype=torch.long)
            for name, values in fields.items()
        }

    def _build_dependency_pairs(self):
        pairs = []
        threshold_start = self.s_dim + self.u_dim
        for gid, scalar_indices in enumerate(self.group_scalar_indices):
            activation_index = self.s_dim + gid
            pairs.extend((s, activation_index) for s in scalar_indices)
            pairs.append((
                activation_index,
                threshold_start + self.group_threshold_ids[gid],
            ))
        return tuple(pairs)


class GroupedMLPDecoder(nn.Module):
    """Stacked MLP decoder with normalized-ReQU group gates."""

    def __init__(
        self,
        input_dim,
        hidden_dims=(5,),
        out_dim=1,
        selection_mode="feature_group",
        gate_scale=1.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dims = tuple(int(x) for x in hidden_dims)
        self.num_hidden_layers = len(self.hidden_dims)
        self.n_units = int(sum(self.hidden_dims))
        self.H = self.hidden_dims[0] if self.num_hidden_layers == 1 else self.n_units
        self.out_dim = int(out_dim)
        self.selection_mode = str(selection_mode)
        self.gate_scale = float(gate_scale)
        if self.gate_scale <= 0:
            raise ValueError("gate_scale must be positive.")

        self.layout = MultiLayerGroupLayout(
            self.input_dim, self.hidden_dims, self.out_dim, self.selection_mode
        )
        self.param_specs = self.layout.param_specs
        self.group_meta = self.layout.group_meta
        self.unit_groups = self.layout.unit_groups
        self.feature_group_ids = tuple(self.layout.feature_group_ids)
        self.unit_group_ids = tuple(self.layout.unit_group_ids)
        self.feature_group_slice = self.layout.feature_group_slice
        self.unit_group_slice = self.layout.unit_group_slice
        self.threshold_roles = tuple(self.layout.threshold_roles)
        self.has_feature_gates = self.layout.has_feature_gates
        self.has_unit_gates = self.layout.has_unit_gates
        self.n_candidate_edges = self.layout.n_candidate_edges
        self.unit_offsets = self.layout.unit_offsets
        self.layer_slices = self.layout.layer_slices
        self.s_dim = self.layout.s_dim
        self.u_dim = self.layout.u_dim
        self.t_dim = self.layout.t_dim
        self.dim = self.layout.dim

        self.register_buffer(
            "_group_threshold_ids",
            torch.as_tensor(self.layout.group_threshold_ids, dtype=torch.long),
        )
        for gid, indices in enumerate(self.layout.group_scalar_indices):
            self.register_buffer(
                f"_group_scalar_{gid}", torch.as_tensor(indices, dtype=torch.long)
            )

    @staticmethod
    def activate(x):
        return F.relu(x)

    def group_gate(self, margin):
        positive = F.relu(margin).square()
        return positive / (self.gate_scale ** 2 + positive)

    def split_latent(self, xi):
        s = xi[:, :self.s_dim]
        u = xi[:, self.s_dim:self.s_dim + self.u_dim]
        t = xi[:, self.s_dim + self.u_dim:]
        return s, u, t

    def group_semantics(self, xi):
        s, u, t = self.split_latent(xi)
        threshold = t.index_select(1, self._group_threshold_ids)
        margin = u - threshold
        return {
            "s": s,
            "u": u,
            "t": t,
            "group_threshold": threshold,
            "margin": margin,
            "gate": self.group_gate(margin),
            "active": margin > 0.0,
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

    def group_slab_norms(self, xi):
        s = xi[:, :self.s_dim]
        norms = []
        for gid in range(self.u_dim):
            idx = getattr(self, f"_group_scalar_{gid}")
            norms.append(s.index_select(1, idx).square().sum(dim=1).sqrt())
        return torch.stack(norms, dim=1)

    def feature_semantics(self, xi):
        if not self.has_feature_gates:
            raise ValueError("This model has no feature groups.")
        semantics = self.group_semantics(xi)
        strength = self.group_slab_norms(xi)[:, self.feature_group_slice]
        gate = semantics["gate"][:, self.feature_group_slice]
        return {
            "active": semantics["active"][:, self.feature_group_slice],
            "gate": gate,
            "margin": semantics["margin"][:, self.feature_group_slice],
            "slab_strength": strength,
            "effective_strength": gate * strength,
        }

    def _unit_strength_components(self, slabs):
        incoming_parts = []
        outgoing_parts = []
        strength_parts = []
        for layer, (w_name, b_name) in enumerate(zip(
            self.layout.hidden_weight_names, self.layout.hidden_bias_names
        )):
            incoming = torch.cat(
                [slabs[w_name], slabs[b_name].unsqueeze(2)], dim=2
            ).norm(dim=2)
            if layer + 1 < self.num_hidden_layers:
                outgoing = slabs[self.layout.hidden_weight_names[layer + 1]].norm(
                    dim=1
                )
            else:
                outgoing = slabs[self.layout.output_weight_name].norm(dim=1)
            incoming_parts.append(incoming)
            outgoing_parts.append(outgoing)
            strength_parts.append(incoming * outgoing)
        return (
            torch.cat(incoming_parts, dim=1),
            torch.cat(outgoing_parts, dim=1),
            torch.cat(strength_parts, dim=1),
        )

    def unit_semantics(self, xi):
        if not self.has_unit_gates:
            raise ValueError("This model has no unit groups.")
        semantics = self.group_semantics(xi)
        slabs = self.unpack_slabs(xi)
        incoming, outgoing, strength = self._unit_strength_components(slabs)
        gate = semantics["gate"][:, self.unit_group_slice]
        return {
            "active": semantics["active"][:, self.unit_group_slice],
            "gate": gate,
            "margin": semantics["margin"][:, self.unit_group_slice],
            "input_slab_norm": incoming,
            "output_slab_norm": outgoing,
            "slab_strength": strength,
            "effective_strength": gate * strength,
        }

    def edge_semantics(self, xi):
        if self.selection_mode != "feature_unit_induced_edge":
            raise ValueError("Induced edge semantics require feature+unit groups.")
        slabs = self.unpack_slabs(xi)
        feature = self.feature_semantics(xi)
        units = self.unit_semantics(xi)
        out = {}
        for item in self.layout.linear_weight_specs:
            name = item["name"]
            weight = slabs[name]
            if item["role"] == "hidden_weight":
                layer = int(item["layer"])
                current = units["active"][:, self.layer_slices[layer]]
                current_gate = units["gate"][:, self.layer_slices[layer]]
                if layer == 0:
                    active = current[:, :, None] & feature["active"][:, None, :]
                    gate = current_gate[:, :, None] * feature["gate"][:, None, :]
                else:
                    previous = units["active"][:, self.layer_slices[layer - 1]]
                    previous_gate = units["gate"][:, self.layer_slices[layer - 1]]
                    active = current[:, :, None] & previous[:, None, :]
                    gate = current_gate[:, :, None] * previous_gate[:, None, :]
            else:
                last = units["active"][:, self.layer_slices[-1]]
                last_gate = units["gate"][:, self.layer_slices[-1]]
                active = last[:, None, :].expand_as(weight)
                gate = last_gate[:, None, :].expand_as(weight)
            out[name] = {
                "parameter": name,
                "role": item["role"],
                "layer": int(item["layer"]),
                "active": active,
                "gate": gate,
                "weight": weight,
                "effective_strength": gate * weight.abs(),
            }
        return out

    def structural_signature(self):
        specs = tuple(
            (item["name"], item["shape"], item["role"])
            for item in self.param_specs
        )
        return (
            "grouped_stacked_bnn_v1",
            self.selection_mode,
            self.hidden_dims,
            specs,
            self.threshold_roles,
            self.n_candidate_edges,
        )

    def compatibility_signature(self):
        return self.structural_signature() + (self.gate_scale,)

    def flow_metadata(self):
        return self.layout.latent_metadata

    def flow_dependency_pairs(self):
        return self.layout.dependency_pairs

    def forward(self, X, xi, force_all_on=False, structural_mask=None):
        slabs = self.unpack_slabs(xi)
        semantics = self.group_semantics(xi)
        R = xi.shape[0]
        n = X.shape[0]
        hidden = X[None, :, :].expand(R, n, self.input_dim)
        gate = (
            torch.ones_like(semantics["gate"])
            if force_all_on else semantics["gate"]
        )
        structural_mask = {} if structural_mask is None else structural_mask

        if self.has_feature_gates:
            feature_gate = gate[:, self.feature_group_slice]
            if "feature" in structural_mask:
                mask = torch.as_tensor(
                    structural_mask["feature"], device=xi.device, dtype=xi.dtype
                ).reshape(1, self.input_dim)
                feature_gate = feature_gate * mask
            hidden = hidden * feature_gate[:, None, :]

        unit_gate = None
        if self.has_unit_gates:
            unit_gate = gate[:, self.unit_group_slice]
            if "unit" in structural_mask:
                mask = torch.as_tensor(
                    structural_mask["unit"], device=xi.device, dtype=xi.dtype
                ).reshape(1, self.n_units)
                unit_gate = unit_gate * mask

        for layer, (w_name, b_name) in enumerate(zip(
            self.layout.hidden_weight_names, self.layout.hidden_bias_names
        )):
            hidden = torch.bmm(
                hidden, slabs[w_name].transpose(1, 2)
            ) + slabs[b_name][:, None, :]
            hidden = self.activate(hidden)
            if unit_gate is not None:
                hidden = hidden * unit_gate[:, None, self.layer_slices[layer]]

        out = torch.bmm(
            hidden, slabs[self.layout.output_weight_name].transpose(1, 2)
        ) + slabs["beta0"][:, None, :]
        if self.out_dim == 1:
            return out[..., 0]
        return out


class GroupedBNNVI(nn.Module):
    """Variational wrapper for the grouped stacked BNN."""

    def __init__(
        self,
        X,
        y,
        input_dim=None,
        hidden_dims=(5,),
        out_dim=1,
        selection_mode="feature_group",
        family="gaussian",
        sigma2=1.0,
        init_sd=0.5,
        K_flow=4,
        flow_type="attention",
        flow_hidden_units=128,
        flow_hidden_layers=2,
        scale_clip=2.0,
        flow_token_dim=32,
        flow_num_heads=4,
        flow_seed=123,
        gate_scale=1.0,
    ):
        super().__init__()
        self.register_buffer("X", X)
        self.register_buffer("y", y)
        if input_dim is None:
            input_dim = X.shape[1]
        self.family = str(family).lower()
        self.register_buffer(
            "sigma2", torch.tensor(float(sigma2), dtype=X.dtype)
        )
        self.decoder = GroupedMLPDecoder(
            input_dim=int(input_dim),
            hidden_dims=hidden_dims,
            out_dim=out_dim,
            selection_mode=selection_mode,
            gate_scale=gate_scale,
        )
        self.q0 = NBase(self.decoder.dim, init_sd=init_sd)
        self.init_sd = self.q0.init_sd
        self.flow_type = str(flow_type).lower()

        if int(K_flow) == 0 or self.flow_type == "meanfield":
            self.flow = IdentityFlow()
            self.flow_type = "meanfield"
        elif self.flow_type in {"attention", "attention_affine", "full_attention"}:
            self.flow = FullAttentionAffineFlow(
                dim=self.decoder.dim,
                latent_metadata=self.decoder.flow_metadata(),
                K=K_flow,
                scale_clip=scale_clip,
                token_dim=flow_token_dim,
                num_heads=flow_num_heads,
                seed=flow_seed,
                dependency_pairs=self.decoder.flow_dependency_pairs(),
            )
            self.flow_type = "attention"
        elif self.flow_type == "iaf":
            self.flow = StackedIAF(
                dim=self.decoder.dim,
                K=K_flow,
                hidden_units=flow_hidden_units,
                num_hidden_layers=flow_hidden_layers,
                scale_clip=scale_clip,
                seed=flow_seed,
            )
        else:
            raise ValueError("flow_type must be meanfield, attention, or iaf.")

    def sample_posterior(self, R):
        z0 = self.q0.sample(R)
        xi, logdet = self.flow(z0, return_logdet=True)
        return xi, self.q0.log_prob(z0) - logdet

    def log_likelihood(self, xi, X=None, y=None, **decoder_kwargs):
        X = self.X if X is None else X
        y = self.y if y is None else y
        pred = self.decoder(X, xi, **decoder_kwargs)
        if self.family == "gaussian":
            resid = y[None, :] - pred
            return -0.5 * (
                resid.square().sum(dim=1) / self.sigma2
                + y.numel() * torch.log(2.0 * torch.pi * self.sigma2)
            )
        if self.family in {"bernoulli", "binomial", "logistic"}:
            target = y[None, :].expand_as(pred)
            return -F.binary_cross_entropy_with_logits(
                pred, target, reduction="none"
            ).sum(dim=1)
        if self.family == "poisson":
            target = y[None, :].expand_as(pred)
            rate = torch.exp(pred.clamp(-20.0, 20.0))
            return (target * pred - rate - torch.lgamma(target + 1.0)).sum(dim=1)
        logp = F.log_softmax(pred, dim=-1)
        idx = torch.arange(y.numel(), device=y.device)
        return logp[:, idx, y.long()].sum(dim=1)

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
            "elbo": log_likelihood + log_prior - log_q,
        }

    @torch.no_grad()
    def predict(self, X_new, R=200):
        xi, _ = self.sample_posterior(R)
        pred = self.decoder(X_new, xi)
        if self.family == "gaussian":
            return pred.mean(dim=0)
        if self.family in {"bernoulli", "binomial", "logistic"}:
            return torch.sigmoid(pred).mean(dim=0)
        if self.family == "poisson":
            return torch.exp(pred.clamp(-20.0, 20.0)).mean(dim=0)
        return F.softmax(pred, dim=-1).mean(dim=0)


@torch.no_grad()
def run_grouped_acceptance_tests(device=None, dtype=torch.float32):
    """Small structural/flow smoke tests for the retained implementation."""

    device = torch.device("cpu") if device is None else torch.device(device)
    results = {}
    for mode in ("feature_group", "unit_group", "feature_unit_induced_edge"):
        decoder = GroupedMLPDecoder(
            input_dim=3,
            hidden_dims=(3, 2),
            selection_mode=mode,
            gate_scale=1.0,
        ).to(device=device, dtype=dtype)
        xi = torch.randn(8, decoder.dim, device=device, dtype=dtype)
        X = torch.randn(7, 3, device=device, dtype=dtype)
        pred = decoder(X, xi)
        results[f"{mode}_finite"] = bool(torch.isfinite(pred).all())

    decoder = GroupedMLPDecoder(
        input_dim=3,
        hidden_dims=(3, 2),
        selection_mode="feature_unit_induced_edge",
    ).to(device=device, dtype=dtype)
    xi = torch.randn(16, decoder.dim, device=device, dtype=dtype)
    edges = decoder.edge_semantics(xi)
    results["induced_edges_finite"] = all(
        bool(torch.isfinite(item["gate"]).all()) for item in edges.values()
    )

    X = torch.randn(12, 3, device=device, dtype=dtype)
    y = torch.randn(12, device=device, dtype=dtype)
    for flow_type in ("attention", "iaf"):
        model = GroupedBNNVI(
            X, y,
            hidden_dims=(3,),
            selection_mode="feature_group",
            K_flow=2,
            flow_type=flow_type,
            flow_hidden_units=16,
            flow_hidden_layers=1,
            flow_token_dim=16,
            flow_num_heads=2,
        ).to(device)
        base = model.q0.sample(8)
        check = model.flow.numerical_sanity_check(base)
        results[f"{flow_type}_inverse_error"] = check["max_inverse_error"]
        results[f"{flow_type}_logdet_error"] = check[
            "max_logdet_consistency_error"
        ]

    return results