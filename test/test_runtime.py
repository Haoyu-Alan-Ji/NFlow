from pathlib import Path

import numpy as np
import pandas as pd


root = Path("data2")

paths = {
    r"$n>p$": root / "n1000p100" / "n1000p100_mh8_output",
    r"$p\gg n$": root / "n100p500" / "n100p500_mh8_output",
}

rng = np.random.default_rng(20260728)
rows = []

for environment, path in paths.items():
    files = list(path.rglob("mh_runtime_summary.csv"))

    x = np.array([
        pd.read_csv(file).loc[0, "total_runtime_s"]
        for file in files
    ], dtype=float)

    boot_mean = rng.choice(
        x,
        size=(10000, len(x)),
        replace=True,
    ).mean(axis=1)

    ci_low, ci_high = np.quantile(
        boot_mean,
        [0.025, 0.975],
    )

    rows.append({
        "environment": environment,
        "N": len(x),
        "mean_time_s": x.mean(),
        "ci_low_s": ci_low,
        "ci_high_s": ci_high,
        "time_95ci_s": (
            f"{x.mean():.1f} "
            f"[{ci_low:.1f}, {ci_high:.1f}]"
        ),
    })

result = pd.DataFrame(rows)

print(result.to_string(index=False))

Path("results").mkdir(exist_ok=True)

result.to_csv(
    "results/mh_time_two_envs.csv",
    index=False,
)