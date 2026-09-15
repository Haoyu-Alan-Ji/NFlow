"""Affine inverse autoregressive flow used by the grouped BNN.

The implementation is deliberately standard: MADE conditioners, bounded affine
log-scales, and a different fixed coordinate ordering in each layer.  It does
not encode DSS-LVR roles; those remain properties of the decoder.
"""

from __future__ import annotations

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

        # Identity initialization for the affine map.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _hidden_degrees(self):
        if self.dim == 1:
            return torch.ones(self.hidden_units, dtype=torch.long)
        return (torch.arange(self.hidden_units) % (self.dim - 1)) + 1

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
    """One affine IAF layer under a fixed coordinate ordering."""

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
        xp = x.index_select(1, self.permutation)
        log_scale, shift = self._params(xp)
        yp = xp * torch.exp(log_scale) + shift
        y = yp.index_select(1, self.inverse_permutation)
        logdet = log_scale.sum(dim=1)
        if return_logdet:
            return y, logdet
        return y

    def inverse(self, y, return_logdet=False):
        yp = y.index_select(1, self.permutation)
        xp = torch.zeros_like(yp)
        inverse_logdet = y.new_zeros(y.shape[0])

        # MADE output j depends only on x_{<j}, so inversion is sequential.
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


class StackedIAF(nn.Module):
    """Stacked affine IAF with aggressive layerwise order mixing."""

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
        self.flow_type = "iaf"
        if self.dim < 1 or self.K < 1:
            raise ValueError("StackedIAF requires dim >= 1 and K >= 1.")

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
