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
    def __init__(self, dim, init_log_scale=-2.5):
        super().__init__()

        self.dim = int(dim)
        self.loc = nn.Parameter(torch.zeros(self.dim))
        self.raw_log_scale = nn.Parameter(
            torch.full((self.dim,), float(init_log_scale))
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


class DSSAttentionFFNDecoder(nn.Module):
    """
    RePU gate:

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

    def unpack(
        self,
        xi,
        return_summary=False,
        beta_eps=0.05,
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

            positive = torch.where(
                margin > 0.0,
                margin,
                torch.zeros_like(margin),
            )

            positive_power = positive.pow(
                self.gate_power
            )

            if self.gate_tau is None:
                gate = positive_power
            else:
                gate = positive_power / (
                    self.gate_tau ** self.gate_power
                    + positive_power
                )

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
                active = (
                    margin > 0.0
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
                "repu"
                if self.gate_tau is None
                else "normalized_repu"
            )

            summary["gate_power"] = self.gate_power
            summary["gate_tau"] = self.gate_tau
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
        K_flow=4,
        flow_hidden_units=128,
        flow_hidden_layers=2,
        scale_clip=2.0,
        bounded=None,
        gate_power=2.0,
        gate_tau=1.0,
        attention_type="self",
        ffn_activation="relu",
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
            attention_type=attention_type,
            ffn_activation=ffn_activation,
        )

        self.q0 = NBase(
            self.decoder.dim,
        )

        self.flow = SemanticFlow(
            self.decoder.s_dim,
            self.decoder.u_dim,
            self.decoder.t_dim,
            K_flow,
            flow_hidden_units,
            flow_hidden_layers,
            scale_clip,
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

    def log_likelihood(self, xi):
        pred = self.decoder(
            self.X,
            xi,
        )

        if self.family == "gaussian":
            resid = (
                self.y[None, :]
                - pred
            )

            return -0.5 * (
                resid.square().sum(dim=1)
                / self.sigma2
                + self.y.numel()
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
            y = self.y[None, :].expand_as(
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
            y = self.y[None, :].expand_as(
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
            self.y.numel(),
            device=self.y.device,
        )

        return logp[
            :,
            idx,
            self.y.long(),
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

    def neg_elbo(
        self,
        R=64,
        elbo_beta=1.0,
    ):
        xi, log_q = self.sample_posterior(R)

        return (
            log_q
            - float(elbo_beta)
            * self.log_likelihood(xi)
            - self.log_prior(xi)
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