#!/usr/bin/env python3
"""Export one saha_dgp_seed_*.npz file to the CSV directory consumed by saha_r_methods.R."""
from pathlib import Path
import argparse, json
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("npz")
p.add_argument("--out", default=None)
a = p.parse_args()
src = Path(a.npz)
out = Path(a.out) if a.out else src.with_suffix("").with_name(src.stem + "_r")
out.mkdir(parents=True, exist_ok=True)
z = np.load(src, allow_pickle=False)
X, y, signal = z["X"], z["y"], z["signal"]
tr, te = z["train_idx"].astype(int), z["test_idx"].astype(int)
np.savetxt(out / "X_train.csv", X[tr], delimiter=",")
np.savetxt(out / "y_train.csv", y[tr], delimiter=",")
np.savetxt(out / "X_test.csv", X[te], delimiter=",")
np.savetxt(out / "y_test.csv", y[te], delimiter=",")
np.savetxt(out / "signal_test.csv", signal[te], delimiter=",")
np.savetxt(out / "feature_true.csv", z["feature_true"], delimiter=",")
meta = {k: np.asarray(z[k]).item() for k in ("n", "p", "pi", "alpha", "sigma2", "seed") if k in z}
(out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(out)
