"""Affine inverse autoregressive flows for grouped DSS-LVR models.

Two variants are provided:

1. ``StackedIAF``: generic coordinate-mixing IAF used as an ablation/control.
2. ``RoleAwareStackedIAF``: IAF whose layerwise autoregressive order cycles over
   semantic DSS-LVR roles.  For the default U/V/tau roles, one complete cycle is

       U < V < tau
       V < tau < U
       tau < U < V

   Each IAF layer still updates every latent coordinate in parallel after one
   MADE evaluation.  Role awareness enters only through the MADE ordering, not
   through sequential role-specific coupling updates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(int(in_features), int(out_features), bias=bias)
        self.register_buffer(
            "mask", torch.ones(self.out_features, self.in_features)
        )

    def set_mask(self, mask):
        mask = torch.as_tensor(mask, dtype=self.weight.dtype)
        if mask.shape != self.weight.shape:
            raise ValueError("MADE mask shape does not match the linear layer.")
        self.mask.copy_(mask.to(device=self.mask.device))

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)


class MADE(nn.Module):
    """Masked MLP returning one affine shift and log-scale per coordinate.

    The input supplied to MADE is already arranged in the desired autoregressive
    order.  Output coordinate j therefore depends only on input coordinates
    0, ..., j-1.
    """

    def __init__(self, dim, hidden_units=128, num_hidden_layers=2):
        super().__init__()
        self.dim = int(dim)
        self.hidden_units = int(hidden_units)
        self.num_hidden_layers = int(num_hidden_layers)
        if self.dim < 1:
            raise ValueError("dim must be positive.")
        if self.hidden_units < 1 or self.num_hidden_layers < 1:
            raise ValueError("MADE requires positive hidden size and depth.")

        layers = []
        previous = self.dim
        for _ in range(self.num_hidden_layers):
            layers.append(MaskedLinear(previous, self.hidden_units))
            previous = self.hidden_units
        self.hidden = nn.ModuleList(layers)
        self.output = MaskedLinear(previous, 2 * self.dim)
        self._set_masks()

        # The affine IAF starts exactly at the identity map.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _hidden_degrees(self):
        if self.dim == 1:
            return torch.ones(self.hidden_units, dtype=torch.long)

        # Spread hidden degrees over the full autoregressive range.  This is
        # preferable to using only the first hidden_units degrees when dim is
        # larger than the conditioner width.
        positions = torch.linspace(
            1, self.dim - 1, steps=self.hidden_units, dtype=torch.float32
        )
        return positions.round().to(torch.long).clamp_(1, self.dim - 1)

    def _set_masks(self):
        input_degree = torch.arange(1, self.dim + 1, dtype=torch.long)
        previous_degree = input_degree

        for layer in self.hidden:
            hidden_degree = self._hidden_degrees()
            mask = (
                previous_degree[None, :] <= hidden_degree[:, None]
            ).to(torch.float32)
            layer.set_mask(mask)
            previous_degree = hidden_degree

        output_degree = torch.arange(1, self.dim + 1, dtype=torch.long)
        output_degree = torch.cat([output_degree, output_degree], dim=0)
        mask = (
            previous_degree[None, :] < output_degree[:, None]
        ).to(torch.float32)
        self.output.set_mask(mask)

    def forward(self, x):
        h = x
        for layer in self.hidden:
            h = F.relu(layer(h))
        raw_log_scale, shift = self.output(h).chunk(2, dim=-1)
        return raw_log_scale, shift


class AffineIAFLayer(nn.Module):
    """One affine IAF layer under one fixed coordinate ordering."""

    def __init__(
        self,
        dim,
        permutation,
        hidden_units=128,
        num_hidden_layers=2,
        scale_clip=2.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.scale_clip = float(scale_clip)
        if self.scale_clip <= 0:
            raise ValueError("scale_clip must be positive.")

        permutation = torch.as_tensor(permutation, dtype=torch.long)
        if permutation.numel() != self.dim:
            raise ValueError("permutation length must equal dim.")
        if sorted(permutation.tolist()) != list(range(self.dim)):
            raise ValueError("permutation must contain each coordinate once.")

        self.register_buffer("permutation", permutation)
        self.register_buffer("inverse_permutation", torch.argsort(permutation))
        self.conditioner = MADE(
            self.dim,
            hidden_units=hidden_units,
            num_hidden_layers=num_hidden_layers,
        )

    def _params(self, x_permuted):
        raw_log_scale, shift = self.conditioner(x_permuted)
        log_scale = self.scale_clip * torch.tanh(
            raw_log_scale / self.scale_clip
        )
        return log_scale, shift

    def forward(self, x, return_logdet=False):
        # Reorder coordinates only inside this IAF layer.  The output is put
        # back into the canonical DSS-LVR coordinate layout afterwards.
        xp = x.index_select(1, self.permutation)
        log_scale, shift = self._params(xp)
        yp = xp * torch.exp(log_scale) + shift
        y = yp.index_select(1, self.inverse_permutation)
        logdet = log_scale.sum(dim=1)
        if return_logdet:
            return y, logdet
        return y

    def inverse(self, y, return_logdet=False):
        # Inversion is sequential because MADE parameter j depends on x_<j.
        # This is not the direction used repeatedly during VI sampling.
        yp = y.index_select(1, self.permutation)
        xp = torch.zeros_like(yp)
        inverse_logdet = y.new_zeros(y.shape[0])

        for j in range(self.dim):
            log_scale, shift = self._params(xp)
            xp[:, j] = (
                yp[:, j] - shift[:, j]
            ) * torch.exp(-log_scale[:, j])
            inverse_logdet = inverse_logdet - log_scale[:, j]

        x = xp.index_select(1, self.inverse_permutation)
        if return_logdet:
            return x, inverse_logdet
        return x


class _BaseStackedIAF(nn.Module):
    def _build_layers(
        self,
        orderings,
        hidden_units,
        num_hidden_layers,
    ):
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


class StackedIAF(_BaseStackedIAF):
    """Generic affine IAF with identity/reverse/random coordinate orderings.

    Retained as a useful control.  It has no knowledge of U/V/tau roles.
    """

    def __init__(
        self,
        dim,
        K=4,
        hidden_units=128,
        num_hidden_layers=2,
        scale_clip=2.0,
        seed=123,
    ):
        super().__init__()
        self.dim = int(dim)
        self.K = int(K)
        self.scale_clip = float(scale_clip)
        self.seed = int(seed)
        self.flow_type = "generic_iaf"
        if self.dim < 1 or self.K < 1:
            raise ValueError("StackedIAF requires dim >= 1 and K >= 1.")

        orderings = self._make_orderings()
        self._build_layers(orderings, hidden_units, num_hidden_layers)

    def _make_orderings(self):
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


class RoleAwareStackedIAF(_BaseStackedIAF):
    """Affine IAF with a repeating role-level autoregressive cycle.

    Parameters
    ----------
    dim : int
        Total continuous latent dimension.
    role_indices : mapping
        Mapping from role names to canonical coordinate indices.  For DSS-LVR,
        a typical mapping is ``{"U": slab_indices, "V": activation_indices,
        "tau": threshold_indices}``.
    K : int
        Number of actual IAF layers, not the number of cycles.  With the default
        three-role cycle, K=3 is one complete role-aware cycle and K=6 is two.
    role_cycle : sequence of sequences
        Layerwise role orders.  The default cycle is
        U<V<tau, V<tau<U, tau<U<V, then repeats.
    shuffle_within_role : bool
        If True, coordinates are deterministically shuffled inside each role in
        each layer while the role-level order itself remains fixed.
    """

    DEFAULT_ROLE_CYCLE = (
        ("U", "V", "tau"),
        ("V", "tau", "U"),
        ("tau", "U", "V"),
    )

    def __init__(
        self,
        dim,
        role_indices: Mapping[str, Sequence[int]],
        K=3,
        hidden_units=128,
        num_hidden_layers=2,
        scale_clip=2.0,
        seed=123,
        role_cycle=None,
        shuffle_within_role=True,
    ):
        super().__init__()
        self.dim = int(dim)
        self.K = int(K)
        self.scale_clip = float(scale_clip)
        self.seed = int(seed)
        self.flow_type = "role_aware_iaf"
        self.shuffle_within_role = bool(shuffle_within_role)
        if self.dim < 1 or self.K < 1:
            raise ValueError("RoleAwareStackedIAF requires dim >= 1 and K >= 1.")

        self.role_indices = {
            str(role): tuple(int(i) for i in indices)
            for role, indices in role_indices.items()
        }
        self.role_cycle = tuple(
            tuple(str(role) for role in order)
            for order in (
                self.DEFAULT_ROLE_CYCLE if role_cycle is None else role_cycle
            )
        )
        self._validate_roles()

        orderings = self._make_role_orderings()
        self._build_layers(orderings, hidden_units, num_hidden_layers)

    def _validate_roles(self):
        if not self.role_indices:
            raise ValueError("role_indices cannot be empty.")
        role_names = tuple(self.role_indices.keys())
        role_set = set(role_names)

        all_indices = []
        for role, indices in self.role_indices.items():
            if len(indices) == 0:
                raise ValueError(f"role {role!r} has no coordinates.")
            all_indices.extend(indices)

        if sorted(all_indices) != list(range(self.dim)):
            raise ValueError(
                "role_indices must be disjoint and cover every coordinate "
                "from 0 to dim-1 exactly once."
            )

        if not self.role_cycle:
            raise ValueError("role_cycle cannot be empty.")
        for order in self.role_cycle:
            if len(order) != len(role_names) or set(order) != role_set:
                raise ValueError(
                    "Each role_cycle entry must contain every role exactly once."
                )

    def _shuffle_role(self, indices, generator):
        indices = torch.as_tensor(indices, dtype=torch.long)
        if not self.shuffle_within_role or indices.numel() <= 1:
            return indices
        local_perm = torch.randperm(indices.numel(), generator=generator)
        return indices.index_select(0, local_perm)

    def _make_role_orderings(self):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        orderings = []

        for layer_id in range(self.K):
            role_order = self.role_cycle[layer_id % len(self.role_cycle)]
            blocks = [
                self._shuffle_role(self.role_indices[role], generator)
                for role in role_order
            ]
            orderings.append(torch.cat(blocks, dim=0))
        return orderings

    def layer_role_order(self, layer_id):
        """Return the semantic role order used by one IAF layer."""
        layer_id = int(layer_id)
        if not 0 <= layer_id < self.K:
            raise IndexError("layer_id out of range.")
        return self.role_cycle[layer_id % len(self.role_cycle)]

    def ordering_summary(self):
        """Human-readable description useful for experiment manifests/tests."""
        rows = []
        for layer_id in range(self.K):
            rows.append({
                "layer": layer_id + 1,
                "role_order": self.layer_role_order(layer_id),
                "coordinate_order": tuple(
                    int(i) for i in self.orderings[layer_id].tolist()
                ),
            })
        return rows
