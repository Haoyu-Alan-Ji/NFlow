import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Conditioner network used inside the semantic normalizing flow.

    This MLP is not the Bayesian neural network itself.
    It only produces affine-coupling parameters for the variational flow.
    """

    def __init__(self, in_dim, out_dim, hidden_units=128, num_hidden_layers=2):
        super().__init__()

        layers = []
        d = int(in_dim)

        for _ in range(int(num_hidden_layers)):
            layers += [nn.Linear(d, int(hidden_units)), nn.ReLU()]
            d = int(hidden_units)

        layers.append(nn.Linear(d, int(out_dim)))
        self.net = nn.Sequential(*layers)

        # Start each coupling transformation close to identity.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


class SemanticLayer(nn.Module):
    """
    One semantic affine-coupling layer over xi=(s,u,t).

    s: active-side magnitude coordinates.
    u: local evidence coordinates.
    t: global threshold coordinates.

    The layer updates only one block among s, u, and t, while conditioning
    on the other two blocks. This keeps the flow role-aware.
    """

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
            in_dim=cond_dim,
            out_dim=2 * trans_dim,
            hidden_units=hidden_units,
            num_hidden_layers=num_hidden_layers,
        )

    def forward(self, x, return_logdet=False):
        s = x[:, :self.s_dim]
        u = x[:, self.s_dim:self.s_dim + self.u_dim]
        t = x[:, self.s_dim + self.u_dim:]

        if self.mode == "s":
            cond = torch.cat([u, t], dim=1)
            log_scale, shift = torch.chunk(self.net(cond), 2, dim=1)
            log_scale = self.scale_clip * torch.tanh(log_scale / self.scale_clip)
            s = s * torch.exp(log_scale) + shift

        elif self.mode == "u":
            cond = torch.cat([s, t], dim=1)
            log_scale, shift = torch.chunk(self.net(cond), 2, dim=1)
            log_scale = self.scale_clip * torch.tanh(log_scale / self.scale_clip)
            u = u * torch.exp(log_scale) + shift

        else:
            cond = torch.cat([s, u], dim=1)
            log_scale, shift = torch.chunk(self.net(cond), 2, dim=1)
            log_scale = self.scale_clip * torch.tanh(log_scale / self.scale_clip)
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
            log_scale, shift = torch.chunk(self.net(cond), 2, dim=1)
            log_scale = self.scale_clip * torch.tanh(log_scale / self.scale_clip)
            s = (s - shift) * torch.exp(-log_scale)

        elif self.mode == "u":
            cond = torch.cat([s, t], dim=1)
            log_scale, shift = torch.chunk(self.net(cond), 2, dim=1)
            log_scale = self.scale_clip * torch.tanh(log_scale / self.scale_clip)
            u = (u - shift) * torch.exp(-log_scale)

        else:
            cond = torch.cat([s, u], dim=1)
            log_scale, shift = torch.chunk(self.net(cond), 2, dim=1)
            log_scale = self.scale_clip * torch.tanh(log_scale / self.scale_clip)
            t = (t - shift) * torch.exp(-log_scale)

        x = torch.cat([s, u, t], dim=1)
        logdet = -log_scale.sum(dim=1)

        if return_logdet:
            return x, logdet
        return x


class SemanticFlow(nn.Module):
    """
    Single posterior flow T_psi: z0 -> xi.

    This is the only flow in the model.
    There is no second generative flow g_theta.
    """

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
            self.layers.append(
                SemanticLayer(
                    self.s_dim,
                    self.u_dim,
                    self.t_dim,
                    mode="s",
                    hidden_units=hidden_units,
                    num_hidden_layers=num_hidden_layers,
                    scale_clip=scale_clip,
                )
            )

            self.layers.append(
                SemanticLayer(
                    self.s_dim,
                    self.u_dim,
                    self.t_dim,
                    mode="u",
                    hidden_units=hidden_units,
                    num_hidden_layers=num_hidden_layers,
                    scale_clip=scale_clip,
                )
            )

            self.layers.append(
                SemanticLayer(
                    self.s_dim,
                    self.u_dim,
                    self.t_dim,
                    mode="t",
                    hidden_units=hidden_units,
                    num_hidden_layers=num_hidden_layers,
                    scale_clip=scale_clip,
                )
            )

    def forward(self, x, return_logdet=False):
        z = x
        total_logdet = x.new_zeros(x.shape[0])

        for layer in self.layers:
            z, logdet = layer(z, return_logdet=True)
            total_logdet = total_logdet + logdet

        if return_logdet:
            return z, total_logdet
        return z

    def inverse(self, z, return_logdet=False):
        x = z
        total_logdet = z.new_zeros(z.shape[0])

        for layer in reversed(self.layers):
            x, logdet = layer.inverse(x, return_logdet=True)
            total_logdet = total_logdet + logdet

        if return_logdet:
            return x, total_logdet
        return x


class NBase(nn.Module):
    """
    Trainable diagonal Gaussian base distribution q0(z0).

    The variational posterior q_psi(xi) is induced by:
        z0 ~ q0
        xi = T_psi(z0)
    """

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

        log_scale = torch.clamp(self.raw_log_scale, min=-5.0, max=2.0)
        z0 = self.loc[None, :] + torch.exp(log_scale)[None, :] * eps

        return z0

    def log_prob(self, z0):
        log_scale = torch.clamp(self.raw_log_scale, min=-5.0, max=2.0)[None, :]
        var = torch.exp(2.0 * log_scale)

        return -0.5 * (
            ((z0 - self.loc[None, :]) ** 2) / var
            + 2.0 * log_scale
            + math.log(2.0 * math.pi)
        ).sum(dim=1)


class DSSBNNDecoder(nn.Module):
    """
    Deterministic DSS decoder D(xi) -> neural-network parameters.

    Each residual FFN block uses:
        hidden = activation(h W1^T + b1)
        branch = hidden W2^T + b2
        h_next = branch + projection(h)

    Every W, b, and projection P is generated by a DSS semantic rule.

    Unbounded mode:
        theta = lambda * s * relu(u - t)

    Bounded mode with bounded=(-1,1):
        theta = B * tanh(s) * relu(u - t)/(1 + relu(u - t))

    Bounded mode still has exact zeros because relu(u-t)=0 when u<=t.
    There is no sigmoid relaxation in this model.
    """

    def __init__(
        self,
        layer_dims,
        ffn_dims=None,
        lambda_w=1.0,
        lambda_b=1.0,
        lambda_p=1.0,
        bounded=None,
        projection="identity_or_sparse",
        ffn_activation="relu",
    ):
        super().__init__()

        self.layer_dims = [int(x) for x in layer_dims]
        self.ffn_dims = ffn_dims

        self.lambda_w = float(lambda_w)
        self.lambda_b = float(lambda_b)
        self.lambda_p = float(lambda_p)

        self.bounded = bounded
        self.projection = projection
        self.ffn_activation = ffn_activation

        # bounded=None means unbounded parameters.
        # bounded=(-1,1) means final decoded parameters are inside [-1,1].
        if bounded is None:
            self.bound = None
        else:
            self.bound = float(max(abs(float(bounded[0])), abs(float(bounded[1]))))

        self.layers_spec = []

        # m indexes the flattened s/u coordinates.
        # g indexes the threshold coordinates.
        m = 0
        g = 0

        for k, (din, dout) in enumerate(zip(self.layer_dims[:-1], self.layer_dims[1:])):
            if ffn_dims is None:
                dff = max(din, dout)
            else:
                dff = int(ffn_dims[k])

            # First FFN matrix: W1 in R^{dff x din}.
            W1 = {
                "start": m,
                "end": m + dff * din,
                "t": g,
                "shape": (dff, din),
                "lambda": self.lambda_w,
            }
            m += dff * din
            g += 1

            # First FFN bias: b1 in R^{dff}.
            b1 = {
                "start": m,
                "end": m + dff,
                "t": g,
                "shape": (dff,),
                "lambda": self.lambda_b,
            }
            m += dff
            g += 1

            # Second FFN matrix: W2 in R^{dout x dff}.
            W2 = {
                "start": m,
                "end": m + dout * dff,
                "t": g,
                "shape": (dout, dff),
                "lambda": self.lambda_w,
            }
            m += dout * dff
            g += 1

            # Second FFN bias: b2 in R^{dout}.
            b2 = {
                "start": m,
                "end": m + dout,
                "t": g,
                "shape": (dout,),
                "lambda": self.lambda_b,
            }
            m += dout
            g += 1

            # Projection P in R^{dout x din}.
            # If dimensions match and projection="identity_or_sparse",
            # no sparse P is created; identity residual is used.
            P = None
            if projection == "sparse" or (
                projection == "identity_or_sparse" and din != dout
            ):
                P = {
                    "start": m,
                    "end": m + dout * din,
                    "t": g,
                    "shape": (dout, din),
                    "lambda": self.lambda_p,
                }
                m += dout * din
                g += 1

            self.layers_spec.append(
                {
                    "din": din,
                    "dff": dff,
                    "dout": dout,
                    "W1": W1,
                    "b1": b1,
                    "W2": W2,
                    "b2": b2,
                    "P": P,
                }
            )

        self.s_dim = m
        self.u_dim = m
        self.t_dim = g
        self.dim = 2 * self.s_dim + self.t_dim

    def unpack(self, xi, return_summary=False):
        """
        Convert semantic coordinates xi=(s,u,t) into actual neural parameters.

        xi shape:
            R x dim

        R is the number of posterior Monte Carlo draws.
        """

        R = xi.shape[0]

        s = xi[:, :self.s_dim]
        u = xi[:, self.s_dim:self.s_dim + self.u_dim]
        t = xi[:, self.s_dim + self.u_dim:]

        layers = []
        summary = {}

        for k, spec in enumerate(self.layers_spec):
            layer_out = {}

            for name in ("W1", "b1", "W2", "b2", "P"):
                item = spec[name]

                if item is None:
                    layer_out[name] = None
                    continue

                sl = slice(item["start"], item["end"])

                # margin = local evidence - global threshold
                margin = u[:, sl] - t[:, item["t"]:item["t"] + 1]

                # ReLU DSS gate.
                # This gives exact zero when margin <= 0.
                raw_gate = F.relu(margin)

                if self.bound is None:
                    # General unbounded model:
                    # theta = lambda * s * (u - t)_+
                    gate = raw_gate
                    val = item["lambda"] * s[:, sl] * gate
                else:
                    # Bounded model:
                    # theta = B * tanh(s) * r/(1+r), r=(u-t)_+
                    #
                    # This keeps exact zero and avoids sigmoid relaxation.
                    gate = raw_gate / (1.0 + raw_gate)
                    val = self.bound * torch.tanh(s[:, sl]) * gate

                layer_out[name] = val.reshape(R, *item["shape"])

                if return_summary:
                    active = (margin > 0).to(xi.dtype)

                    summary[f"{name}{k}_pip"] = active.mean(dim=0).reshape(
                        item["shape"]
                    )

                    summary[f"{name}{k}_gate_mean"] = gate.mean(dim=0).reshape(
                        item["shape"]
                    )

                    summary[f"{name}{k}_mean"] = layer_out[name].mean(dim=0)

                    summary[f"{name}{k}_sd"] = layer_out[name].std(dim=0).reshape(
                        item["shape"]
                    )

            layers.append(layer_out)

        if return_summary:
            summary["t_mean"] = t.mean(dim=0)
            summary["t_sd"] = t.std(dim=0)
            return layers, summary

        return layers

    def forward(self, X, xi):
        """
        Forward pass through posterior-sampled BNN.

        X shape:
            n x p

        xi shape:
            R x dim

        output shape:
            R x n          for scalar output
            R x n x C      for multi-class output
        """

        layers = self.unpack(xi, return_summary=False)

        # h shape: R x n x p
        h = X[None, :, :].expand(
            xi.shape[0],
            X.shape[0],
            X.shape[1],
        )

        for k, spec in enumerate(self.layers_spec):
            W1 = layers[k]["W1"]
            b1 = layers[k]["b1"]
            W2 = layers[k]["W2"]
            b2 = layers[k]["b2"]
            P = layers[k]["P"]

            # First FFN affine map:
            # hidden = h W1^T + b1
            hidden = torch.bmm(h, W1.transpose(1, 2)) + b1[:, None, :]

            # FFN activation.
            # Default is ReLU.
            if self.ffn_activation == "gelu":
                hidden = F.gelu(hidden)
            elif self.ffn_activation == "tanh":
                hidden = torch.tanh(hidden)
            else:
                hidden = F.relu(hidden)

            # Second FFN affine map:
            # branch = activation(h W1^T + b1) W2^T + b2
            branch = torch.bmm(hidden, W2.transpose(1, 2)) + b2[:, None, :]

            # Residual / projection update.
            if self.projection == "none":
                h = branch
            elif P is not None:
                h = branch + torch.bmm(h, P.transpose(1, 2))
            elif spec["din"] == spec["dout"]:
                h = branch + h
            else:
                h = branch

        if h.shape[-1] == 1:
            return h[..., 0]

        return h


class LaSTBNNVI(nn.Module):
    """
    Single-flow LaST-BNN variational model.

    Sampling path:
        z0 ~ q0
        xi = T_psi(z0)
        Theta = D(xi)
        y ~ p(y | X, Theta)

    Negative ELBO:
        E_q[log q(xi) - log p(y | X, D(xi)) - log p0(xi)]
    """

    def __init__(
        self,
        X,
        y,
        layer_dims,
        ffn_dims=None,
        family="gaussian",
        sigma2=1.0,
        K_flow=4,
        flow_hidden_units=128,
        flow_hidden_layers=2,
        scale_clip=2.0,
        lambda_w=1.0,
        lambda_b=1.0,
        lambda_p=1.0,
        bounded=None,
        projection="identity_or_sparse",
        ffn_activation="relu",
    ):
        super().__init__()

        self.register_buffer("X", X)
        self.register_buffer("y", y)

        self.register_buffer(
            "sigma2",
            torch.tensor(float(sigma2), dtype=X.dtype, device=X.device),
        )

        self.family = family

        self.decoder = DSSBNNDecoder(
            layer_dims=layer_dims,
            ffn_dims=ffn_dims,
            lambda_w=lambda_w,
            lambda_b=lambda_b,
            lambda_p=lambda_p,
            bounded=bounded,
            projection=projection,
            ffn_activation=ffn_activation,
        )

        self.q0 = NBase(dim=self.decoder.dim)

        self.flow = SemanticFlow(
            s_dim=self.decoder.s_dim,
            u_dim=self.decoder.u_dim,
            t_dim=self.decoder.t_dim,
            K=K_flow,
            hidden_units=flow_hidden_units,
            num_hidden_layers=flow_hidden_layers,
            scale_clip=scale_clip,
        )

    def sample_posterior(self, R):
        """
        Draw posterior samples from q_psi(xi).

        Returns:
            xi:     R x dim semantic coordinate draws
            log_q:  R-dimensional log q_psi(xi)
        """

        z0 = self.q0.sample(R)
        xi, logdet = self.flow(z0, return_logdet=True)

        log_q = self.q0.log_prob(z0) - logdet

        return xi, log_q

    def log_joint(self, xi):
        """
        Compute log p(y | X, D(xi)) + log p0(xi).

        The prior p0(xi) is standard Gaussian over semantic coordinates.
        """

        pred = self.decoder(X=self.X, xi=xi)

        if self.family == "gaussian":
            resid = self.y[None, :] - pred

            loglik = -0.5 * (
                resid.pow(2).sum(dim=1) / self.sigma2
                + self.y.numel() * torch.log(2.0 * torch.pi * self.sigma2)
            )

        elif self.family in {"bernoulli", "binomial", "logistic"}:
            y = self.y[None, :].expand_as(pred)

            loglik = -F.binary_cross_entropy_with_logits(
                pred,
                y,
                reduction="none",
            ).sum(dim=1)

        elif self.family == "multiclass":
            logp = F.log_softmax(pred, dim=-1)
            idx = torch.arange(self.y.numel(), device=self.y.device)

            loglik = logp[:, idx, self.y.long()].sum(dim=1)

        else:
            raise ValueError(
                "family must be gaussian, bernoulli, logistic, binomial, or multiclass"
            )

        logprior = -0.5 * (
            xi.pow(2) + math.log(2.0 * math.pi)
        ).sum(dim=1)

        return loglik + logprior

    def neg_elbo(self, R=64, elbo_beta=1.0):
        """
        Monte Carlo estimate of negative ELBO.

        elbo_beta can be used for likelihood tempering.
        """

        xi, log_q = self.sample_posterior(R)
        log_joint = self.log_joint(xi)

        return (log_q - float(elbo_beta) * log_joint).mean()

    @torch.no_grad()
    def predict(self, X_new, R=200):
        """
        Posterior predictive mean.

        For Gaussian:
            returns mean prediction.

        For Bernoulli:
            returns posterior mean probability.

        For multiclass:
            returns posterior mean class-probability vector.
        """

        xi, _ = self.sample_posterior(R)
        pred = self.decoder(X=X_new, xi=xi)

        if self.family == "gaussian":
            return pred.mean(dim=0)

        if self.family in {"bernoulli", "binomial", "logistic"}:
            return torch.sigmoid(pred).mean(dim=0)

        return F.softmax(pred, dim=-1).mean(dim=0)

    @torch.no_grad()
    def posterior_summary(self, R=500):
        """
        Posterior summaries of DSS neural parameters.

        Includes:
            PIP = Pr(u > t | data)
            gate_mean = E[ReLU-gate | data]
            parameter posterior mean
            parameter posterior sd
        """

        xi, _ = self.sample_posterior(R)

        _, summary = self.decoder.unpack(
            xi=xi,
            return_summary=True,
        )

        return summary