import random
import numpy as np
import torch

from model2 import LaSTBNNVI


seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# 1. Simulate a small regression dataset
# -----------------------------

n = 160
p = 20

X = torch.randn(n, p, device=device)

true_beta = torch.zeros(p, device=device)
true_beta[0] = 1.50
true_beta[3] = -1.20
true_beta[7] = 0.90
true_beta[11] = -0.70

eta = (
    X @ true_beta
    + 0.80 * torch.relu(X[:, 1])
    - 0.60 * X[:, 2] * X[:, 4]
)

sigma = 0.50
y = eta + sigma * torch.randn(n, device=device)


# -----------------------------
# 2. Train-test split
# -----------------------------

idx = torch.randperm(n, device=device)

train_idx = idx[:120]
test_idx = idx[120:]

X_train = X[train_idx]
y_train = y[train_idx]

X_test = X[test_idx]
y_test = y[test_idx]


# -----------------------------
# 3. Build LaST-BNN
# -----------------------------
#
# layer_dims=[p, 24, 12, 1] means:
#   input dimension  p
#   hidden dimension 24
#   hidden dimension 12
#   scalar output    1
#
# ffn_dims=[32, 24, 12] means:
#   each transition block uses a two-layer FFN:
#       din -> dff -> dout
#
# bounded=(-1,1) means:
#   all decoded W, b, P are bounded in [-1,1]
#   while still having exact ReLU-gate zeros.

model = LaSTBNNVI(
    X=X_train,
    y=y_train,
    layer_dims=[p, 8, 1],
    ffn_dims=[12, 6],
    family="gaussian",
    sigma2=sigma ** 2,
    K_flow=3,
    flow_hidden_units=96,
    flow_hidden_layers=2,
    scale_clip=2.0,
    lambda_w=1.0,
    lambda_b=1.0,
    lambda_p=1.0,
    bounded=(-1, 1),
    projection="identity_or_sparse",
    ffn_activation="relu",
).to(device)


# -----------------------------
# 4. Train
# -----------------------------

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1, 3001):
    optimizer.zero_grad(set_to_none=True)

    loss = model.neg_elbo(R=32, elbo_beta=1.0)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()

    if epoch == 1 or epoch % 50 == 0:
        with torch.no_grad():
            pred_train = model.predict(X_train, R=64)
            pred_test = model.predict(X_test, R=64)

            train_mse = ((pred_train - y_train) ** 2).mean().item()
            test_mse = ((pred_test - y_test) ** 2).mean().item()

        print(
            f"epoch={epoch:04d} "
            f"loss={loss.item():.3f} "
            f"train_mse={train_mse:.3f} "
            f"test_mse={test_mse:.3f}"
        )


# -----------------------------
# 5. Posterior summaries
# -----------------------------

summary = model.posterior_summary(R=500)

print("\n===== posterior object dimensions =====")
print("decoder dim:", model.decoder.dim)
print("s_dim:", model.decoder.s_dim)
print("u_dim:", model.decoder.u_dim)
print("t_dim:", model.decoder.t_dim)

print("\n===== threshold posterior means =====")
print(summary["t_mean"].detach().cpu().numpy())

print("\n===== first block W1 PIP shape =====")
print(summary["W10_pip"].shape)

print("\n===== first block W1 top 10 PIPs =====")
pip0 = summary["W10_pip"].detach().cpu().reshape(-1)
top = torch.topk(pip0, k=10)

print("top indices:", top.indices.numpy())
print("top PIPs:", top.values.numpy())

print("\n===== first block W1 posterior mean range =====")
wmean = summary["W10_mean"].detach().cpu()
print(float(wmean.min()), float(wmean.max()))