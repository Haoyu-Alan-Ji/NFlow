import random
import numpy as np
import torch

from model2 import LaSTBNNVI
from simfun import simfun_bnn, print_bnn_siminfo
from utils import make_split, split_data
from config import SplitConfig


# -----------------------------
# 0. Seed and device
# -----------------------------

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# 1. Simulate sparse BNN oracle data
# -----------------------------

n = 240
p = 20
sigma2 = 0.25
sigma = sigma2 ** 0.5

layer_dims = [p, 8, 1]
ffn_dims = [12, 6]

X, y, feature_true, truth, sim_info = simfun_bnn(
    n=600,
    p=20,
    seed=123,
    sigma2=0.25,
    layer_dims=[20, 8, 1],
    ffn_dims=[12, 6],
    n_active=4,
    use_projection=False,
    bounded=None,
    center_y=True,
    device=device,
)

print(print_bnn_siminfo(sim_info))


# -----------------------------
# 2. Split data
# -----------------------------

split_cfg = SplitConfig(
    train_frac=0.60,
    val_frac=0.20,
    test_frac=0.20,
    seed=seed,
)

indices = make_split(X.shape[0], split_cfg)
splits = split_data(X, y, indices, mode="tensor")

X_train = splits["X_train"]
y_train = splits["y_train"]

X_val = splits["X_val"]
y_val = splits["y_val"]

X_test = splits["X_test"]
y_test = splits["y_test"]


# -----------------------------
# 3. Build LaST-BNN
# -----------------------------

model = LaSTBNNVI(
    X=X_train,
    y=y_train,
    layer_dims=[p, 8, 1],
    ffn_dims=[12, 6],
    family="gaussian",
    sigma2=sigma2,
    K_flow=2,
    flow_hidden_units=64,
    flow_hidden_layers=2,
    scale_clip=2.0,
    lambda_w=1.0,
    lambda_b=1.0,
    lambda_p=1.0,
    bounded=None,
    projection="none",
    ffn_activation="relu",
).to(device)


# -----------------------------
# 4. Train
# -----------------------------

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

R_train = 32
R_eval = 300

for epoch in range(1, 3001):
    optimizer.zero_grad(set_to_none=True)

    loss = model.neg_elbo(R=R_train, elbo_beta=1.0)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()

    if epoch == 1 or epoch % 50 == 0:
        with torch.no_grad():
            pred_train = model.predict(X_train, R=R_eval)
            pred_val = model.predict(X_val, R=R_eval)
            pred_test = model.predict(X_test, R=R_eval)

            train_mse = ((pred_train - y_train) ** 2).mean().item()
            val_mse = ((pred_val - y_val) ** 2).mean().item()
            test_mse = ((pred_test - y_test) ** 2).mean().item()

        print(
            f"epoch={epoch:04d} "
            f"loss={loss.item():.3f} "
            f"train_mse={train_mse:.3f} "
            f"val_mse={val_mse:.3f} "
            f"test_mse={test_mse:.3f}"
        )

# -----------------------------
# 5. Recovery metric utilities
# -----------------------------

def support_metrics(score, truth_mask, cut=0.5):
    score = score.detach().cpu().reshape(-1)
    truth_mask = truth_mask.detach().cpu().reshape(-1).bool()

    selected = score >= cut

    tp = int((selected & truth_mask).sum().item())
    fp = int((selected & ~truth_mask).sum().item())
    fn = int((~selected & truth_mask).sum().item())
    tn = int((~selected & ~truth_mask).sum().item())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    fdr = fp / max(tp + fp, 1)
    jaccard = tp / max(tp + fp + fn, 1)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fdr": fdr,
        "jaccard": jaccard,
        "selected_size": int(selected.sum().item()),
        "true_size": int(truth_mask.sum().item()),
        "selected_idx": torch.where(selected)[0].numpy().tolist(),
    }


def topk_metrics(score, truth_mask):
    score = score.detach().cpu().reshape(-1)
    truth_mask = truth_mask.detach().cpu().reshape(-1).bool()

    k = int(truth_mask.sum().item())
    selected_idx = torch.topk(score, k=min(k, score.numel())).indices

    selected = torch.zeros_like(truth_mask)
    selected[selected_idx] = True

    tp = int((selected & truth_mask).sum().item())
    fp = int((selected & ~truth_mask).sum().item())
    fn = int((~selected & truth_mask).sum().item())
    tn = int((~selected & ~truth_mask).sum().item())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    fdr = fp / max(tp + fp, 1)
    jaccard = tp / max(tp + fp + fn, 1)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fdr": fdr,
        "jaccard": jaccard,
        "selected_idx": selected_idx.numpy().tolist(),
    }

# -----------------------------
# 6. Posterior summaries
# -----------------------------

summary = model.posterior_summary(R=1000)

print("\n===== posterior object dimensions =====")
print("decoder dim:", model.decoder.dim)
print("s_dim:", model.decoder.s_dim)
print("u_dim:", model.decoder.u_dim)
print("t_dim:", model.decoder.t_dim)

print("\n===== threshold posterior means =====")
print(summary["t_mean"].detach().cpu().numpy())


# -----------------------------
# 7. Feature-level variable selection
# -----------------------------
#
# Input feature j is considered active if it enters the first block through
# either W10 or P0.
#
# W10_pip has shape dff0 x p.
# P0_pip has shape h x p.
#
# Therefore:
#   input_pip_j = max_l PIP(W10_lj), max_r PIP(P0_rj)

feature_parts = []

if "W10_pip" in summary:
    feature_parts.append(summary["W10_pip"].detach().cpu().max(dim=0).values)

if "P0_pip" in summary:
    feature_parts.append(summary["P0_pip"].detach().cpu().max(dim=0).values)

input_pip = torch.stack(feature_parts, dim=0).max(dim=0).values
feature_mask = feature_true.detach().cpu().bool()

print("\n===== feature-level input PIP ranking =====")
top = torch.topk(input_pip, k=min(10, p))

print("true active:", torch.where(feature_mask)[0].numpy().tolist())
print("top features:", top.indices.numpy().tolist())
print("top input PIPs:", top.values.numpy().tolist())

print("\n===== feature-level selection metrics =====")

for cut in [0.20, 0.25, 0.40, 0.50]:
    m = support_metrics(input_pip, feature_mask, cut=cut)

    print(
        f"cut={cut:.2f} "
        f"S={m['selected_size']:2d}/{m['true_size']:2d} "
        f"TP={m['tp']:2d} FP={m['fp']:2d} FN={m['fn']:2d} TN={m['tn']:2d} "
        f"P={m['precision']:.3f} "
        f"R={m['recall']:.3f} "
        f"F1={m['f1']:.3f} "
        f"FDR={m['fdr']:.3f} "
        f"J={m['jaccard']:.3f} "
        f"selected={m['selected_idx']}"
    )

m_top = topk_metrics(input_pip, feature_mask)

print("\n===== feature-level top-s0 metrics =====")
print(
    f"top_s0 "
    f"TP={m_top['tp']:2d} FP={m_top['fp']:2d} "
    f"FN={m_top['fn']:2d} TN={m_top['tn']:2d} "
    f"P={m_top['precision']:.3f} "
    f"R={m_top['recall']:.3f} "
    f"F1={m_top['f1']:.3f} "
    f"FDR={m_top['fdr']:.3f} "
    f"J={m_top['jaccard']:.3f} "
    f"selected={m_top['selected_idx']}"
)


# -----------------------------
# 8. Element-level W, b, P recovery
# -----------------------------

print("\n===== element-level recovery by parameter block =====")

param_keys = [
    "W10", "b10", "W20", "b20", "P0",
    "W11", "b11", "W21", "b21", "P1",
]

for key in param_keys:
    pip_key = f"{key}_pip"

    if pip_key not in summary:
        continue

    score = summary[pip_key].detach().cpu()
    mask = (truth[key].detach().cpu().abs() > 1e-12)

    m_cut = support_metrics(score, mask, cut=0.4)
    m_top = topk_metrics(score, mask)

    print(
        f"{key:<4} "
        f"true={m_cut['true_size']:3d} "
        f"cut0.5: S={m_cut['selected_size']:3d} "
        f"P={m_cut['precision']:.3f} "
        f"R={m_cut['recall']:.3f} "
        f"F1={m_cut['f1']:.3f} "
        f"J={m_cut['jaccard']:.3f} "
        f"| topS: "
        f"P={m_top['precision']:.3f} "
        f"R={m_top['recall']:.3f} "
        f"F1={m_top['f1']:.3f} "
        f"J={m_top['jaccard']:.3f}"
    )

# -----------------------------
# 9. Local diagnostic: W10 top PIPs
# -----------------------------

print("\n===== first block W10 connection-level top 10 PIPs =====")

pip0 = summary["W10_pip"].detach().cpu().reshape(-1)
top = torch.topk(pip0, k=10)

rows = top.indices // p
cols = top.indices % p

print("flat indices:", top.indices.numpy().tolist())
print("rows:", rows.numpy().tolist())
print("cols:", cols.numpy().tolist())
print("PIPs:", top.values.numpy().tolist())

print("\n===== first block W10 posterior mean range =====")
wmean = summary["W10_mean"].detach().cpu()
print(float(wmean.min()), float(wmean.max()))