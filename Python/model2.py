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


class DSSAttentionFFNDecoder(nn.Module):
    """
    DSS decoder for sparse attention-FFN residual BNN.

    Model structure:

        z0 = X E^T + e

        for k = 0,...,K-1:

            a_k = Attn_k(z_k)

            delta_k =
                activation(a_k W1_k^T + b1_k) W2_k^T + b2_k

            z_{k+1} = z_k + delta_k

        eta = z_K Wout^T + bout

    DSS semantic rule for every W and b:

        theta = lambda * s * relu(u - t)

    where t is shared by the whole matrix/vector block.
    """

    def __init__(
        self,
        input_dim,
        d_model,
        n_blocks,
        ffn_dims=None,
        out_dim=1,
        lambda_w=1.0,
        lambda_b=1.0,
        bounded=None,
        attention_type="self",
        ffn_activation="relu",
    ):
        super().__init__()

        self.input_dim = int(input_dim)
        self.d_model = int(d_model)
        self.n_blocks = int(n_blocks)
        self.out_dim = int(out_dim)

        self.lambda_w = float(lambda_w)
        self.lambda_b = float(lambda_b)

        self.bounded = bounded
        self.attention_type = attention_type
        self.ffn_activation = ffn_activation

        if bounded is None:
            self.bound = None
        else:
            self.bound = float(max(abs(float(bounded[0])), abs(float(bounded[1]))))

        if ffn_dims is None:
            self.ffn_dims = [4 * self.d_model for _ in range(self.n_blocks)]
        elif isinstance(ffn_dims, int):
            self.ffn_dims = [int(ffn_dims) for _ in range(self.n_blocks)]
        else:
            self.ffn_dims = [int(x) for x in ffn_dims]

        if len(self.ffn_dims) != self.n_blocks:
            raise ValueError("ffn_dims must have length n_blocks.")

        self.param_specs = []
        self.layers_spec = []

        # m indexes flattened s/u coordinates.
        # g indexes global threshold coordinates.
        m = 0
        g = 0

        def add_param(name, shape, lam, block=None):
            nonlocal m, g

            n_elem = int(math.prod(shape))

            item = {
                "name": name,
                "block": block,
                "start": m,
                "end": m + n_elem,
                "t": g,
                "shape": tuple(shape),
                "lambda": float(lam),
            }

            m += n_elem
            g += 1

            self.param_specs.append(item)

            return item

        # Input embedding: X -> z0
        self.E = add_param(
            name="E",
            shape=(self.d_model, self.input_dim),
            lam=self.lambda_w,
            block="input",
        )

        self.e = add_param(
            name="e",
            shape=(self.d_model,),
            lam=self.lambda_b,
            block="input",
        )

        # Residual attention-FFN blocks.
        for k in range(self.n_blocks):
            dff = self.ffn_dims[k]

            W1 = add_param(
                name=f"W1_{k}",
                shape=(dff, self.d_model),
                lam=self.lambda_w,
                block=k,
            )

            b1 = add_param(
                name=f"b1_{k}",
                shape=(dff,),
                lam=self.lambda_b,
                block=k,
            )

            W2 = add_param(
                name=f"W2_{k}",
                shape=(self.d_model, dff),
                lam=self.lambda_w,
                block=k,
            )

            b2 = add_param(
                name=f"b2_{k}",
                shape=(self.d_model,),
                lam=self.lambda_b,
                block=k,
            )

            self.layers_spec.append(
                {
                    "block": k,
                    "d_model": self.d_model,
                    "dff": dff,
                    "W1": W1,
                    "b1": b1,
                    "W2": W2,
                    "b2": b2,
                }
            )

        # Output head: z_K -> eta
        self.Wout = add_param(
            name="Wout",
            shape=(self.out_dim, self.d_model),
            lam=self.lambda_w,
            block="output",
        )

        self.bout = add_param(
            name="bout",
            shape=(self.out_dim,),
            lam=self.lambda_b,
            block="output",
        )

        self.s_dim = m
        self.u_dim = m
        self.t_dim = g
        self.dim = 2 * self.s_dim + self.t_dim

    def attention(self, z):
        """
        Attention operator.

        z shape:
            R x n x d_model

        returns:
            R x n x d_model

        Default:
            parameter-free self-attention across observations.

        This keeps Attn(z) explicitly inside the block:
            FFN_k(Attn_k(z)).
        """

        if self.attention_type == "self":
            scale = math.sqrt(float(z.shape[-1]))

            scores = torch.bmm(z, z.transpose(1, 2)) / scale
            weights = torch.softmax(scores, dim=-1)

            return torch.bmm(weights, z)

        if self.attention_type == "feature":
            weights = torch.softmax(z, dim=-1)
            return weights * z

        if self.attention_type == "identity":
            return z

        raise ValueError("attention_type must be self, feature, or identity.")

    def activate(self, x):
        if self.ffn_activation == "gelu":
            return F.gelu(x)

        if self.ffn_activation == "tanh":
            return torch.tanh(x)

        return F.relu(x)

    def unpack(self, xi, return_summary=False, beta_eps=0.05):
        """
        Convert semantic coordinates xi=(s,u,t) into neural parameters.

        xi shape:
            R x dim

        PIP:
            Pr(u > t | data)

        ePIP:
            Pr(|theta| > beta_eps | data)
        """

        R = xi.shape[0]

        s = xi[:, :self.s_dim]
        u = xi[:, self.s_dim:self.s_dim + self.u_dim]
        t = xi[:, self.s_dim + self.u_dim:]

        params = {}
        summary = {}

        for item in self.param_specs:
            name = item["name"]
            sl = slice(item["start"], item["end"])

            margin = u[:, sl] - t[:, item["t"]:item["t"] + 1]
            raw_gate = F.relu(margin)

            if self.bound is None:
                gate = raw_gate
                val = item["lambda"] * s[:, sl] * gate
            else:
                gate = raw_gate / (1.0 + raw_gate)
                val = self.bound * torch.tanh(s[:, sl]) * gate

            val = val.reshape(R, *item["shape"])
            params[name] = val

            if return_summary:
                active = (margin > 0).to(xi.dtype).reshape(R, *item["shape"])
                epip = (val.abs() > float(beta_eps)).to(xi.dtype)

                summary[f"{name}_pip"] = active.mean(dim=0)
                summary[f"{name}_epip"] = epip.mean(dim=0)
                summary[f"{name}_gate_mean"] = gate.mean(dim=0).reshape(item["shape"])
                summary[f"{name}_mean"] = val.mean(dim=0)
                summary[f"{name}_sd"] = val.std(dim=0)

        if return_summary:
            summary["t_mean"] = t.mean(dim=0)
            summary["t_sd"] = t.std(dim=0)
            summary["beta_eps"] = float(beta_eps)

            return params, summary

        return params

    def forward(self, X, xi):
        """
        Forward pass through posterior-sampled sparse attention-FFN BNN.

        X shape:
            n x input_dim

        xi shape:
            R x dim

        output shape:
            R x n          if out_dim == 1
            R x n x out_dim otherwise
        """

        params = self.unpack(xi, return_summary=False)

        R = xi.shape[0]
        n = X.shape[0]

        # Input embedding:
        # z0 = X E^T + e
        E = params["E"]
        e = params["e"]

        X_exp = X[None, :, :].expand(R, n, self.input_dim)

        z = torch.bmm(X_exp, E.transpose(1, 2)) + e[:, None, :]

        # Residual attention-FFN blocks:
        #
        # a_k = Attn_k(z_k)
        # delta_k = activation(a_k W1_k^T + b1_k) W2_k^T + b2_k
        # z_{k+1} = z_k + delta_k
        for k in range(self.n_blocks):
            W1 = params[f"W1_{k}"]
            b1 = params[f"b1_{k}"]
            W2 = params[f"W2_{k}"]
            b2 = params[f"b2_{k}"]

            att = self.attention(z)

            hidden = torch.bmm(att, W1.transpose(1, 2)) + b1[:, None, :]
            hidden = self.activate(hidden)

            delta = torch.bmm(hidden, W2.transpose(1, 2)) + b2[:, None, :]

            z = z + delta

        # Output head:
        # eta = z_K Wout^T + bout
        Wout = params["Wout"]
        bout = params["bout"]

        out = torch.bmm(z, Wout.transpose(1, 2)) + bout[:, None, :]

        if out.shape[-1] == 1:
            return out[..., 0]

        return out

class LaSTBNNVI(nn.Module):
    """
    Single-flow LaST attention-FFN BNN variational model.

    Sampling path:
        z0 ~ q0
        xi = T_psi(z0)
        Theta = D(xi)
        y ~ p(y | X, Theta)

    Decoder structure:
        z0 = X E^T + e

        z_{k+1}
        =
        z_k
        +
        FFN_k(Attn_k(z_k))

        FFN_k(x)
        =
        activation(x W1_k^T + b1_k) W2_k^T + b2_k

        eta = z_K Wout^T + bout
    """

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
        lambda_w=1.0,
        lambda_b=1.0,
        bounded=None,
        attention_type="self",
        ffn_activation="relu",
    ):
        super().__init__()

        self.register_buffer("X", X)
        self.register_buffer("y", y)

        if input_dim is None:
            input_dim = X.shape[1]

        self.input_dim = int(input_dim)
        self.d_model = int(d_model)
        self.n_blocks = int(n_blocks)
        self.out_dim = int(out_dim)

        self.family = family

        self.register_buffer(
            "sigma2",
            torch.tensor(float(sigma2), dtype=X.dtype, device=X.device),
        )

        self.decoder = DSSAttentionFFNDecoder(
            input_dim=self.input_dim,
            d_model=self.d_model,
            n_blocks=self.n_blocks,
            ffn_dims=ffn_dims,
            out_dim=self.out_dim,
            lambda_w=lambda_w,
            lambda_b=lambda_b,
            bounded=bounded,
            attention_type=attention_type,
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

    def log_likelihood(self, xi):
        """
        Compute log p(y | X, D(xi)).
        """

        pred = self.decoder(X=self.X, xi=xi)

        if self.family == "gaussian":
            resid = self.y[None, :] - pred

            loglik = -0.5 * (
                resid.pow(2).sum(dim=1) / self.sigma2
                + self.y.numel() * torch.log(2.0 * torch.pi * self.sigma2)
            )

            return loglik

        if self.family in {"bernoulli", "binomial", "logistic"}:
            y = self.y[None, :].expand_as(pred)

            loglik = -F.binary_cross_entropy_with_logits(
                pred,
                y,
                reduction="none",
            ).sum(dim=1)

            return loglik

        if self.family == "poisson":
            y = self.y[None, :].expand_as(pred)

            log_rate = pred
            rate = torch.exp(torch.clamp(log_rate, min=-20.0, max=20.0))

            loglik = (
                y * log_rate
                - rate
                - torch.lgamma(y + 1.0)
            ).sum(dim=1)

            return loglik

        if self.family == "multiclass":
            logp = F.log_softmax(pred, dim=-1)
            idx = torch.arange(self.y.numel(), device=self.y.device)

            loglik = logp[:, idx, self.y.long()].sum(dim=1)

            return loglik

        raise ValueError(
            "family must be gaussian, bernoulli, logistic, binomial, poisson, or multiclass"
        )

    def log_prior(self, xi):
        """
        Standard Gaussian prior over semantic coordinates xi.
        """

        return -0.5 * (
            xi.pow(2) + math.log(2.0 * math.pi)
        ).sum(dim=1)

    def log_joint(self, xi):
        """
        Compute log p(y | X, D(xi)) + log p0(xi).
        """

        return self.log_likelihood(xi) + self.log_prior(xi)

    def neg_elbo(self, R=64, elbo_beta=1.0):
        """
        Monte Carlo estimate of negative ELBO.

        elbo_beta tempers only the likelihood term:
            log p0(xi) + beta * log p(y | xi)
        """

        xi, log_q = self.sample_posterior(R)

        loglik = self.log_likelihood(xi)
        logprior = self.log_prior(xi)

        target = float(elbo_beta) * loglik + logprior

        return (log_q - target).mean()

    @torch.no_grad()
    def predict(self, X_new, R=200):
        """
        Posterior predictive mean.

        Gaussian:
            returns posterior mean of eta.

        Bernoulli:
            returns posterior mean probability.

        Poisson:
            returns posterior mean rate.

        Multiclass:
            returns posterior mean class probabilities.
        """

        xi, _ = self.sample_posterior(R)
        pred = self.decoder(X=X_new, xi=xi)

        if self.family == "gaussian":
            return pred.mean(dim=0)

        if self.family in {"bernoulli", "binomial", "logistic"}:
            return torch.sigmoid(pred).mean(dim=0)

        if self.family == "poisson":
            return torch.exp(torch.clamp(pred, min=-20.0, max=20.0)).mean(dim=0)

        return F.softmax(pred, dim=-1).mean(dim=0)

    @torch.no_grad()
    def posterior_summary(self, R=500, beta_eps=0.05):
        """
        Posterior summaries of DSS neural parameters.

        Includes:
            pip       = Pr(u > t | data)
            epip      = Pr(|theta| > beta_eps | data)
            gate_mean = E[gate | data)
            mean      = E[theta | data]
            sd        = SD(theta | data)
        """

        xi, _ = self.sample_posterior(R)

        _, summary = self.decoder.unpack(
            xi=xi,
            return_summary=True,
            beta_eps=beta_eps,
        )

        return summary