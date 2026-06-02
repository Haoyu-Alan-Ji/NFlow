from pathlib import Path
import argparse
import re
import numpy as np
import pandas as pd


def infer_seed_from_path(path: Path):
    m = re.search(r"seed_(\d+)", str(path))
    return int(m.group(1)) if m else None


def read_summary_files(root: Path, filename: str, method_name: str):
    files = sorted(root.rglob(filename))
    if not files:
        raise FileNotFoundError(f"No {filename} found under {root}")

    rows = []
    for f in files:
        df = pd.read_csv(f)

        if "seed" not in df.columns:
            seed = infer_seed_from_path(f)
            if seed is None:
                raise ValueError(f"Cannot infer seed from {f}")
            df["seed"] = seed

        df["method"] = method_name
        df["source_file"] = str(f)

        if "support_size" not in df.columns and "selected_size" in df.columns:
            df["support_size"] = df["selected_size"]

        rows.append(df)

    out = pd.concat(rows, ignore_index=True)

    required = [
        "seed",
        "precision",
        "recall",
        "f1",
        "fdr",
        "support_size",
        "auroc",
        "auprc",
    ]

    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"{method_name} summary missing columns: {missing}")

    if out["seed"].duplicated().any():
        dup = out.loc[out["seed"].duplicated(), "seed"].tolist()
        print(f"[warn] duplicated summary seeds for {method_name}: {dup}")
        out = out.drop_duplicates("seed", keep="last")

    return out


def read_pip_files(root: Path, filename: str, method_name: str, pip_threshold: float):
    files = sorted(root.rglob(filename))
    if not files:
        raise FileNotFoundError(f"No {filename} found under {root}")

    rows = []
    for f in files:
        df = pd.read_csv(f)

        if "seed" not in df.columns:
            seed = infer_seed_from_path(f)
            if seed is None:
                raise ValueError(f"Cannot infer seed from {f}")
            df["seed"] = seed

        df["method"] = method_name
        df["source_file"] = str(f)
        rows.append(df)

    out = pd.concat(rows, ignore_index=True)

    required = ["seed", "j0", "truth", "pip"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"{method_name} pip table missing columns: {missing}")

    out["seed"] = out["seed"].astype(int)
    out["j0"] = out["j0"].astype(int)
    out["truth"] = out["truth"].astype(int)
    out["pip"] = out["pip"].astype(float)

    # 优先使用原始输出里的 selected；如果没有，才用 PIP > 0.5 兜底。
    if "selected" in out.columns:
        out["selected"] = out["selected"].astype(int)
    else:
        out["selected"] = (out["pip"] > pip_threshold).astype(int)

    return out


def mean_se(x):
    x = pd.Series(x).dropna()
    if len(x) == 0:
        return np.nan, np.nan

    mean = x.mean()
    se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
    return mean, se


def fmt_mean_se(x, digits=3):
    m, se = mean_se(x)
    if pd.isna(m):
        return "NA"
    return f"{m:.{digits}f} ({se:.{digits}f})"


def jaccard(a, b):
    a = set(a)
    b = set(b)
    u = a | b
    return len(a & b) / len(u) if len(u) > 0 else 1.0


def make_table1(last_summary, mcmc_summary, common_seeds, regime, n, p, digits):
    summary = pd.concat([last_summary, mcmc_summary], ignore_index=True)
    summary = summary[summary["seed"].isin(common_seeds)].copy()

    metric_cols = [
        "precision",
        "recall",
        "f1",
        "fdr",
        "support_size",
        "auprc",
        "auroc",
    ]

    for c in metric_cols:
        summary[c] = pd.to_numeric(summary[c], errors="coerce")

    rows = []
    for method, g in summary.groupby("method"):
        rows.append({
            "Regime": regime,
            "n": n,
            "p": p,
            "Method": method,
            "Replicates": len(g),
            "Precision": fmt_mean_se(g["precision"], digits),
            "Recall": fmt_mean_se(g["recall"], digits),
            "F1": fmt_mean_se(g["f1"], digits),
            "FDR": fmt_mean_se(g["fdr"], digits),
            "Support size": fmt_mean_se(g["support_size"], digits),
            "AUPRC": fmt_mean_se(g["auprc"], digits),
            "AUROC": fmt_mean_se(g["auroc"], digits),
        })

    table = pd.DataFrame(rows)

    method_order = {"MCMC": 0, "LaST-Flow": 1}
    table["_order"] = table["Method"].map(method_order).fillna(99)
    table = table.sort_values(["Regime", "_order"]).drop(columns="_order")

    return table


def make_table2(last_pip, mcmc_pip, common_seeds, regime, n, p, digits):
    rows = []

    for seed in common_seeds:
        l = last_pip[last_pip["seed"] == seed][["j0", "truth", "pip", "selected"]].copy()
        m = mcmc_pip[mcmc_pip["seed"] == seed][["j0", "pip", "selected"]].copy()

        l = l.rename(columns={
            "pip": "pip_last",
            "selected": "selected_last",
        })

        m = m.rename(columns={
            "pip": "pip_mcmc",
            "selected": "selected_mcmc",
        })

        merged = l.merge(m, on="j0", how="inner")

        if len(merged) != p:
            raise ValueError(
                f"Seed {seed}: merged variable count = {len(merged)}, expected p = {p}"
            )

        pip_last = merged["pip_last"].to_numpy(float)
        pip_mcmc = merged["pip_mcmc"].to_numpy(float)

        sel_last = merged.loc[merged["selected_last"] == 1, "j0"].astype(int)
        sel_mcmc = merged.loc[merged["selected_mcmc"] == 1, "j0"].astype(int)

        rows.append({
            "seed": seed,
            "pip_mae": np.mean(np.abs(pip_last - pip_mcmc)),
            "pip_rmse": np.sqrt(np.mean((pip_last - pip_mcmc) ** 2)),
            "pearson": pd.Series(pip_last).corr(pd.Series(pip_mcmc), method="pearson"),
            "spearman": pd.Series(pip_last).corr(pd.Series(pip_mcmc), method="spearman"),
            "jaccard": jaccard(sel_last, sel_mcmc),
            "last_support_size": len(sel_last),
            "mcmc_support_size": len(sel_mcmc),
        })

    per_seed = pd.DataFrame(rows)

    table = pd.DataFrame([{
        "Regime": regime,
        "n": n,
        "p": p,
        "Replicates": len(per_seed),
        "PIP-MAE": fmt_mean_se(per_seed["pip_mae"], digits),
        "PIP-RMSE": fmt_mean_se(per_seed["pip_rmse"], digits),
        "Spearman": fmt_mean_se(per_seed["spearman"], digits),
        "Jaccard": fmt_mean_se(per_seed["jaccard"], digits),
        "MCMC support size": fmt_mean_se(per_seed["mcmc_support_size"], digits),
        "LaST support size": fmt_mean_se(per_seed["last_support_size"], digits),
    }])

    return table, per_seed


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--last-root",
        type=str,
        default="data/n160p100_output",
        help="Folder containing LaST-Flow output files."
    )

    parser.add_argument(
        "--mcmc-root",
        type=str,
        default="data/n160p100_mcmc_output",
        help="Folder containing MCMC output files."
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/n160p100_tables",
        help="Folder for generated paper tables."
    )

    parser.add_argument("--regime", type=str, default="Near-boundary")
    parser.add_argument("--n", type=int, default=160)
    parser.add_argument("--p", type=int, default=100)
    parser.add_argument("--pip-threshold", type=float, default=0.5)
    parser.add_argument("--digits", type=int, default=3)

    return parser.parse_args()


def main():
    args = parse_args()

    last_root = Path(args.last_root)
    mcmc_root = Path(args.mcmc_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    last_summary = read_summary_files(
        root=last_root,
        filename="last_summary.csv",
        method_name="LaST-Flow",
    )

    mcmc_summary = read_summary_files(
        root=mcmc_root,
        filename="mcmc_summary.csv",
        method_name="MCMC",
    )

    last_pip = read_pip_files(
        root=last_root,
        filename="last_pip.csv",
        method_name="LaST-Flow",
        pip_threshold=args.pip_threshold,
    )

    mcmc_pip = read_pip_files(
        root=mcmc_root,
        filename="mcmc_pip.csv",
        method_name="MCMC",
        pip_threshold=args.pip_threshold,
    )

    last_summary_seeds = set(last_summary["seed"].astype(int))
    mcmc_summary_seeds = set(mcmc_summary["seed"].astype(int))
    last_pip_seeds = set(last_pip["seed"].astype(int))
    mcmc_pip_seeds = set(mcmc_pip["seed"].astype(int))

    common_seeds = sorted(
        last_summary_seeds
        & mcmc_summary_seeds
        & last_pip_seeds
        & mcmc_pip_seeds
    )

    print(f"[info] LaST summary seeds: {len(last_summary_seeds)}")
    print(f"[info] MCMC summary seeds: {len(mcmc_summary_seeds)}")
    print(f"[info] LaST pip seeds: {len(last_pip_seeds)}")
    print(f"[info] MCMC pip seeds: {len(mcmc_pip_seeds)}")
    print(f"[info] common seeds used: {len(common_seeds)}")

    if len(common_seeds) == 0:
        raise RuntimeError("No common seeds found between LaST and MCMC outputs.")

    table1 = make_table1(
        last_summary=last_summary,
        mcmc_summary=mcmc_summary,
        common_seeds=common_seeds,
        regime=args.regime,
        n=args.n,
        p=args.p,
        digits=args.digits,
    )

    table2, table2_per_seed = make_table2(
        last_pip=last_pip,
        mcmc_pip=mcmc_pip,
        common_seeds=common_seeds,
        regime=args.regime,
        n=args.n,
        p=args.p,
        digits=args.digits,
    )

    table1_path = out_dir / "paper_table1_n160p100_support_recovery.csv"
    table2_path = out_dir / "paper_table2_n160p100_pip_agreement.csv"
    table2_per_seed_path = out_dir / "n160p100_pip_agreement_per_seed.csv"

    table1.to_csv(table1_path, index=False)
    table2.to_csv(table2_path, index=False)
    table2_per_seed.to_csv(table2_per_seed_path, index=False)

    # Optional LaTeX outputs for manuscript / qmd.
    table1.to_latex(out_dir / "paper_table1_n160p100_support_recovery.tex", index=False)
    table2.to_latex(out_dir / "paper_table2_n160p100_pip_agreement.tex", index=False)

    print("\n===== Table 1: Support recovery =====")
    print(table1.to_string(index=False))

    print("\n===== Table 2: PIP agreement =====")
    print(table2.to_string(index=False))

    print("\n[done] wrote:")
    print(table1_path)
    print(table2_path)
    print(table2_per_seed_path)


if __name__ == "__main__":
    main()