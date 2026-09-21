"""Grouped DSS-LVR Bayesian neural network with role-aware affine IAF.

Retained implementation:
- stacked feed-forward networks;
- feature groups, unit groups, or feature+unit induced connectivity;
- normalized-ReQU or smooth exact-zero structural maps;
- role-aware affine inverse autoregressive posterior transport;
- diagonal Gaussian mean-field control.

Attention/coupling, spline, embed-output, and independent-edge flow paths are
removed from the main model.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(int(in_features), int(out_features), bias=bias)
        self.register_buffer("mask", torch.ones(self.out_features, self.in_features))

    def set_mask(self, mask):
        mask = torch.as_tensor(mask, dtype=self.weight.dtype, device=self.mask.device)
        if mask.shape != self.weight.shape:
            raise ValueError("MADE mask shape does not match the linear layer.")
        self.mask.copy_(mask)

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)


class MADE(nn.Module):
    """Masked MLP returning one affine shift and log-scale per coordinate."""

    def __init__(self, dim, hidden_units=128, num_hidden_layers=2):
        super().__init__()
        self.dim = int(dim)
        self.hidden_units = int(hidden_units)
        self.num_hidden_layers = int(num_hidden_layers)
        if self.dim < 1 or self.hidden_units < 1 or self.num_hidden_layers < 1:
            raise ValueError("MADE dimensions must be positive.")

        layers = []
        previous = self.dim
        for _ in range(self.num_hidden_layers):
            layers.append(MaskedLinear(previous, self.hidden_units))
            previous = self.hidden_units
        self.hidden = nn.ModuleList(layers)
        self.output = MaskedLinear(previous, 2 * self.dim)
        self._set_masks()
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _hidden_degrees(self):
        if self.dim == 1:
            return torch.ones(self.hidden_units, dtype=torch.long)
        positions = torch.linspace(1, self.dim - 1, steps=self.hidden_units)
        return positions.round().long().clamp_(1, self.dim - 1)

    def _set_masks(self):
        previous_degree = torch.arange(1, self.dim + 1, dtype=torch.long)
        for layer in self.hidden:
            hidden_degree = self._hidden_degrees()
            layer.set_mask((previous_degree[None, :] <= hidden_degree[:, None]).float())
            previous_degree = hidden_degree

        output_degree = torch.arange(1, self.dim + 1, dtype=torch.long).repeat(2)
        self.output.set_mask((previous_degree[None, :] < output_degree[:, None]).float())

    def forward(self, x):
        h = x
        for layer in self.hidden:
            h = F.relu(layer(h))
        return self.output(h).chunk(2, dim=-1)


class AffineIAFLayer(nn.Module):
    """One affine IAF transformation under one fixed coordinate ordering."""

    def __init__(self, dim, permutation, hidden_units=128, num_hidden_layers=2, scale_clip=2.0):
        super().__init__()
        self.dim = int(dim)
        self.scale_clip = float(scale_clip)
        if self.scale_clip <= 0:
            raise ValueError("scale_clip must be positive.")
        permutation = torch.as_tensor(permutation, dtype=torch.long)
        if permutation.numel() != self.dim or sorted(permutation.tolist()) != list(range(self.dim)):
            raise ValueError("permutation must contain every coordinate exactly once.")
        self.register_buffer("permutation", permutation)
        self.register_buffer("inverse_permutation", torch.argsort(permutation))
        self.conditioner = MADE(self.dim, hidden_units=hidden_units, num_hidden_layers=num_hidden_layers)

    def _params(self, x_permuted):
        raw_log_scale, shift = self.conditioner(x_permuted)
        log_scale = self.scale_clip * torch.tanh(raw_log_scale / self.scale_clip)
        return log_scale, shift

    def forward(self, x, return_logdet=False):
        xp = x.index_select(1, self.permutation)
        log_scale, shift = self._params(xp)
        yp = xp * torch.exp(log_scale) + shift
        y = yp.index_select(1, self.inverse_permutation)
        logdet = log_scale.sum(dim=1)
        return (y, logdet) if return_logdet else y

    def inverse(self, y, return_logdet=False):
        yp = y.index_select(1, self.permutation)
        xp = torch.zeros_like(yp)
        inverse_logdet = y.new_zeros(y.shape[0])
        for j in range(self.dim):
            log_scale, shift = self._params(xp)
            xp[:, j] = (yp[:, j] - shift[:, j]) * torch.exp(-log_scale[:, j])
            inverse_logdet = inverse_logdet - log_scale[:, j]
        x = xp.index_select(1, self.inverse_permutation)
        return (x, inverse_logdet) if return_logdet else x


class StackedIAF(nn.Module):
    """Stacked affine IAF with configurable role-aware ordering schedules.

    ``K`` is the actual number of IAF transformations.  ``cyclic3`` repeats
    U<V<tau, V<tau<U, tau<U<V.  ``six_permutations`` cycles through all six
    role permutations.  ``generic`` is an order-agnostic control.
    """

    ROLE_CYCLE_3 = (
        ("U", "V", "tau"),
        ("V", "tau", "U"),
        ("tau", "U", "V"),
    )
    ROLE_CYCLE_6 = (
        ("U", "V", "tau"),
        ("V", "tau", "U"),
        ("tau", "U", "V"),
        ("V", "U", "tau"),
        ("U", "tau", "V"),
        ("tau", "V", "U"),
    )

    def __init__(
        self,
        dim,
        role_indices,
        K=6,
        hidden_units=128,
        num_hidden_layers=2,
        scale_clip=2.0,
        seed=123,
        ordering_scheme="cyclic3",
        shuffle_within_role=True,
        custom_role_cycle=None,
    ):
        super().__init__()
        self.dim = int(dim)
        self.K = int(K)
        self.scale_clip = float(scale_clip)
        self.seed = int(seed)
        self.ordering_scheme = str(ordering_scheme).lower()
        self.shuffle_within_role = bool(shuffle_within_role)
        self.flow_type = "iaf"
        if self.dim < 1 or self.K < 1:
            raise ValueError("StackedIAF requires dim >= 1 and K >= 1.")

        self.role_indices = {
            str(role): tuple(int(i) for i in indices)
            for role, indices in role_indices.items()
        }
        self._validate_roles()
        self.role_cycle = self._resolve_role_cycle(custom_role_cycle)
        orderings = self._make_orderings()
        self.register_buffer("orderings", torch.stack(orderings, dim=0))
        self.layers = nn.ModuleList([
            AffineIAFLayer(
                self.dim,
                permutation=order,
                hidden_units=hidden_units,
                num_hidden_layers=num_hidden_layers,
                scale_clip=self.scale_clip,
            )
            for order in orderings
        ])

    def _validate_roles(self):
        required = {"U", "V", "tau"}
        if set(self.role_indices) != required:
            raise ValueError("role_indices must contain exactly U, V, and tau.")
        all_indices = [i for role in ("U", "V", "tau") for i in self.role_indices[role]]
        if any(len(self.role_indices[role]) == 0 for role in required):
            raise ValueError("Every role must contain at least one coordinate.")
        if sorted(all_indices) != list(range(self.dim)):
            raise ValueError("role_indices must cover 0,...,dim-1 exactly once.")

    def _resolve_role_cycle(self, custom_role_cycle):
        if custom_role_cycle is not None:
            cycle = tuple(tuple(str(role) for role in order) for order in custom_role_cycle)
        elif self.ordering_scheme in {"cyclic3", "role_cycle", "three_cycle"}:
            cycle = self.ROLE_CYCLE_3
            self.ordering_scheme = "cyclic3"
        elif self.ordering_scheme in {"six_permutations", "all6", "balanced6"}:
            cycle = self.ROLE_CYCLE_6
            self.ordering_scheme = "six_permutations"
        elif self.ordering_scheme in {"generic", "coordinate"}:
            return None
        else:
            raise ValueError("ordering_scheme must be cyclic3, six_permutations, or generic.")

        roles = {"U", "V", "tau"}
        if not cycle or any(len(order) != 3 or set(order) != roles for order in cycle):
            raise ValueError("Each role ordering must contain U, V, and tau exactly once.")
        return cycle

    @staticmethod
    def _stable_role_code(role):
        return sum((i + 1) * ord(ch) for i, ch in enumerate(role))

    def _role_block(self, role, layer_id):
        indices = torch.as_tensor(self.role_indices[role], dtype=torch.long)
        if not self.shuffle_within_role or indices.numel() <= 1:
            return indices
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + 100003 * (layer_id + 1) + self._stable_role_code(role))
        return indices.index_select(0, torch.randperm(indices.numel(), generator=generator))

    def _make_generic_orderings(self):
        identity = torch.arange(self.dim, dtype=torch.long)
        orderings = [identity]
        if self.K >= 2:
            orderings.append(torch.flip(identity, dims=[0]))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        while len(orderings) < self.K:
            candidate = torch.randperm(self.dim, generator=generator)
            if self.dim == 1 or not torch.equal(candidate, orderings[-1]):
                orderings.append(candidate)
        return orderings

    def _make_orderings(self):
        if self.role_cycle is None:
            return self._make_generic_orderings()
        orderings = []
        for layer_id in range(self.K):
            role_order = self.role_cycle[layer_id % len(self.role_cycle)]
            orderings.append(torch.cat([self._role_block(role, layer_id) for role in role_order]))
        return orderings

    def forward(self, x, return_logdet=False):
        z = x
        total = x.new_zeros(x.shape[0])
        for layer in self.layers:
            z, logdet = layer(z, return_logdet=True)
            total = total + logdet
        return (z, total) if return_logdet else z

    def inverse(self, z, return_logdet=False):
        x = z
        total = z.new_zeros(z.shape[0])
        for layer in reversed(self.layers):
            x, logdet = layer.inverse(x, return_logdet=True)
            total = total + logdet
        return (x, total) if return_logdet else x

    def layer_role_order(self, layer_id):
        if self.role_cycle is None:
            return None
        return self.role_cycle[int(layer_id) % len(self.role_cycle)]

    def ordering_summary(self):
        rows = []
        for layer_id in range(self.K):
            rows.append({
                "layer": layer_id + 1,
                "role_order": self.layer_role_order(layer_id),
                "coordinate_order": tuple(int(i) for i in self.orderings[layer_id].tolist()),
            })
        return rows

    def role_position_counts(self):
        if self.role_cycle is None:
            return None
        out = {role: {"first": 0, "middle": 0, "last": 0} for role in ("U", "V", "tau")}
        labels = ("first", "middle", "last")
        for layer_id in range(self.K):
            for pos, role in enumerate(self.layer_role_order(layer_id)):
                out[role][labels[pos]] += 1
        return out

    @torch.no_grad()
    def numerical_sanity_check(self, base_x):
        transformed, forward_logdet = self.forward(base_x, return_logdet=True)
        reconstructed, inverse_logdet = self.inverse(transformed, return_logdet=True)
        return {
            "max_inverse_error": float((base_x - reconstructed).abs().max()),
            "max_logdet_consistency_error": float((forward_logdet + inverse_logdet).abs().max()),
            "n_nonfinite_forward": int((~torch.isfinite(transformed)).sum()),
            "n_nonfinite_inverse": int((~torch.isfinite(reconstructed)).sum()),
            "n_nonfinite_logdet": int((~torch.isfinite(forward_logdet)).sum() + (~torch.isfinite(inverse_logdet)).sum()),
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



class GroupedMLPDecoder(nn.Module):
    """Stacked MLP decoder with normalized-ReQU group gates."""

    def __init__(
        self,
        input_dim,
        hidden_dims=(5,),
        out_dim=1,
        selection_mode="feature_group",
        gate_type="normalized_requ",
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
        self.gate_type = str(gate_type).lower()
        aliases = {"nr": "normalized_requ", "smooth": "smooth_step"}
        self.gate_type = aliases.get(self.gate_type, self.gate_type)
        if self.gate_type not in {"normalized_requ", "smooth_step"}:
            raise ValueError("gate_type must be normalized_requ or smooth_step.")
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
        if self.gate_type == "normalized_requ":
            positive = F.relu(margin).square()
            return positive / (self.gate_scale ** 2 + positive)

        out = torch.zeros_like(margin)
        middle = (margin > 0.0) & (margin < self.gate_scale)
        out[margin >= self.gate_scale] = 1.0
        if middle.any():
            m = margin[middle]
            logit = self.gate_scale / (self.gate_scale - m) - self.gate_scale / m
            out[middle] = torch.sigmoid(logit)
        return out

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
        return self.structural_signature() + (self.gate_type, self.gate_scale)

    def flow_role_indices(self):
        return {
            "U": tuple(range(0, self.s_dim)),
            "V": tuple(range(self.s_dim, self.s_dim + self.u_dim)),
            "tau": tuple(range(self.s_dim + self.u_dim, self.dim)),
        }

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
        K_flow=6,
        flow_type="iaf",
        flow_hidden_units=128,
        flow_hidden_layers=2,
        scale_clip=2.0,
        flow_seed=123,
        iaf_ordering_scheme="cyclic3",
        iaf_shuffle_within_role=True,
        gate_type="normalized_requ",
        gate_scale=1.0,
        slab_init="auto",
        slab_sd_ratio=0.1,
        slab_bias_sd=0.02,
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
            gate_type=gate_type,
            gate_scale=gate_scale,
        )
        self.q0 = NBase(self.decoder.dim, init_sd=init_sd)
        self.init_sd = self.q0.init_sd
        self.flow_type = str(flow_type).lower()

        if int(K_flow) == 0 or self.flow_type == "meanfield":
            self.flow = IdentityFlow()
            self.flow_type = "meanfield"
        elif self.flow_type in {"iaf", "role_aware_iaf", "autoregressive"}:
            self.flow = StackedIAF(
                dim=self.decoder.dim,
                role_indices=self.decoder.flow_role_indices(),
                K=K_flow,
                hidden_units=flow_hidden_units,
                num_hidden_layers=flow_hidden_layers,
                scale_clip=scale_clip,
                seed=flow_seed,
                ordering_scheme=iaf_ordering_scheme,
                shuffle_within_role=iaf_shuffle_within_role,
            )
            self.flow_type = "iaf"
        else:
            raise ValueError("flow_type must be iaf or meanfield.")

        self.initialize_slab_scale(slab_init, slab_sd_ratio, slab_bias_sd)

    @torch.no_grad()
    def initialize_slab_scale(self, mode="auto", sd_ratio=0.1, bias_sd=0.02):
        """Set only the initial q0 slab SD; leave the prior and means unchanged.

        auto preserves legacy initialization for a single hidden layer and
        uses fan_in for deeper networks. Call only before optimization: an
        already trained flow need not be the identity. init_sd still controls
        the initial V/tau SD. The base distribution's log-SD floor is respected.
        """
        mode = str(mode).lower()
        if mode not in {"auto", "legacy", "fan_in"}:
            raise ValueError("slab_init must be auto, legacy, or fan_in.")
        if not math.isfinite(float(sd_ratio)) or float(sd_ratio) <= 0:
            raise ValueError("slab_sd_ratio must be finite and positive.")
        if not math.isfinite(float(bias_sd)) or float(bias_sd) <= 0:
            raise ValueError("slab_bias_sd must be finite and positive.")
        self.slab_init = (
            "legacy" if self.decoder.num_hidden_layers == 1 else "fan_in"
        ) if mode == "auto" else mode
        self.slab_sd_ratio = float(sd_ratio)
        self.slab_bias_sd = float(bias_sd)
        if self.slab_init == "legacy":
            return
        for spec in self.decoder.param_specs:
            if len(spec["shape"]) == 2:
                gain2 = 1.0 if spec["role"] == "output_weight" else 2.0
                sd = self.slab_sd_ratio * math.sqrt(gain2 / spec["shape"][1])
            else:
                sd = self.slab_bias_sd
            self.q0.raw_log_scale[spec["start"]:spec["end"]].fill_(
                max(-5.0, min(2.0, math.log(sd)))
            )

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
    device = torch.device("cpu") if device is None else torch.device(device)
    results = {}
    for mode in ("feature_group", "unit_group", "feature_unit_induced_edge"):
        decoder = GroupedMLPDecoder(
            input_dim=3,
            hidden_dims=(3, 2),
            selection_mode=mode,
        ).to(device=device, dtype=dtype)
        xi = torch.randn(8, decoder.dim, device=device, dtype=dtype)
        X = torch.randn(7, 3, device=device, dtype=dtype)
        results[f"{mode}_finite"] = bool(torch.isfinite(decoder(X, xi)).all())

    X = torch.randn(12, 3, device=device, dtype=dtype)
    y = torch.randn(12, device=device, dtype=dtype)
    for scheme, K in (("cyclic3", 4), ("six_permutations", 6)):
        model = GroupedBNNVI(
            X, y,
            hidden_dims=(3,),
            selection_mode="feature_group",
            K_flow=K,
            flow_type="iaf",
            flow_hidden_units=16,
            flow_hidden_layers=1,
            iaf_ordering_scheme=scheme,
        ).to(device)
        base = model.q0.sample(8)
        check = model.flow.numerical_sanity_check(base)
        results[f"{scheme}_inverse_error"] = check["max_inverse_error"]
        results[f"{scheme}_logdet_error"] = check["max_logdet_consistency_error"]

    smooth = GroupedMLPDecoder(
        input_dim=2,
        hidden_dims=(2,),
        selection_mode="feature_group",
        gate_type="smooth_step",
        gate_scale=1.0,
    ).to(device=device, dtype=dtype)
    gate = smooth.group_gate(torch.tensor([-1.0, 0.5, 2.0], device=device, dtype=dtype))
    results["smooth_gate_finite"] = bool(torch.isfinite(gate).all())
    results["smooth_gate_exact_zero_one"] = bool(gate[0] == 0 and gate[-1] == 1)
    return results
