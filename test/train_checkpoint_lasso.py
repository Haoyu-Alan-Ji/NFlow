from pathlib import Path
import time
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOTS = [
    Path("data/n160p100_last_output/recovery64_4_lasso100/F01"),
]

OUT_DIR = Path("data/checkpoint_rule/lasso64_4")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "moment_recovery_score"

# Set to False if you want to see every convergence warning.
SUPPRESS_CONVERGENCE_WARNINGS = False


ORACLE_COLS = {
    "moment_recovery_score",
    "recovery_score",
    "active_mean_zerr_median",
    "active_mean_zerr_mean",
    "active_sd_logerr_median",
    "active_sd_logerr_mean",
    "active_sd_ratio_median",
    "active_sd_ratio_mean",
    "sd_ratio_median",
    "sd_ratio_mean",
    "active_marg_skl_median",
    "active_marg_skl_mean",
    "active_joint_skl_median",
    "active_joint_skl_mean",
    "softgate_absdiff_median",
    "softgate_absdiff_mean",
    "zero_soft_leakage_median",
    "zero_soft_leakage_mean",
    "precision",
    "recall",
    "f1",
    "fdr",
    "tp",
    "fp",
    "fn",
    "tn",
    "auroc",
    "auprc",
}

META_COLS = {
    "config_id",
    "run_mode",
    "run_setting",
    "seed_dir",
    "run_id",
    "seed",
    "setting",
    "ckpt_id",
    "stage",
    "epoch",
    "epoch_in_stage",
    "is_warmup",
    "best_so_far",
    "alert",
}


def log(msg):
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def find_history_files():
    files = []
    for root in ROOTS:
        files.extend(sorted(root.glob("recovery/*/seed_*/*history*.csv")))
    return files


def load_dataset():
    rows = []
    files = find_history_files()

    log(f"found history files: {len(files)}")

    for p in files:
        df = pd.read_csv(p)

        parts = p.parts
        seed_dir = [x for x in parts if x.startswith("seed_")][-1]
        seed = int(seed_dir.replace("seed_", ""))

        df.insert(0, "run_id", f"F01_seed_{seed}")
        df.insert(1, "seed", seed)
        df.insert(2, "history_path", str(p))

        rows.append(df)

    if len(rows) == 0:
        raise FileNotFoundError("No history files found.")

    x = pd.concat(rows, ignore_index=True)

    if TARGET not in x.columns:
        raise KeyError(f"Target column not found: {TARGET}")

    x = x.replace([np.inf, -np.inf], np.nan)

    before = len(x)
    x = x.loc[x[TARGET].notna()].copy()
    after = len(x)

    if after < before:
        log(f"dropped rows with missing target: {before - after}")

    return x


def numeric_feature_columns(df):
    bad = set(ORACLE_COLS) | set(META_COLS) | {"history_path"}

    cols = []
    for c in df.columns:
        if c in bad:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)

    return cols


def selected_row_by_min(df, col):
    return df.loc[df[col].idxmin()]


def selected_row_final(df):
    return df.sort_values(["epoch", "ckpt_id"]).iloc[-1]


def evaluate_selector(df, pred_col):
    rows = []

    for run_id, g in df.groupby("run_id"):
        oracle = selected_row_by_min(g, TARGET)
        lasso = selected_row_by_min(g, pred_col)
        final = selected_row_final(g)

        if "loss_ema" in g.columns:
            min_loss_ema = selected_row_by_min(g, "loss_ema")
        else:
            min_loss_ema = selected_row_by_min(g, "loss")

        min_loss = selected_row_by_min(g, "loss")

        for name, row in [
            ("oracle", oracle),
            ("lasso", lasso),
            ("final", final),
            ("min_loss", min_loss),
            ("min_loss_ema", min_loss_ema),
        ]:
            score = float(row[TARGET])
            oracle_score = float(oracle[TARGET])

            rows.append({
                "run_id": run_id,
                "seed": int(g["seed"].iloc[0]),
                "selector": name,
                "ckpt_id": int(row["ckpt_id"]),
                "epoch": int(row["epoch"]),
                "tau": float(row["tau"]) if "tau" in row else np.nan,
                "score": score,
                "oracle_score": oracle_score,
                "regret": score - oracle_score,
                "ratio": score / max(oracle_score, 1e-8),
                "final_score": float(final[TARGET]),
            })

    return pd.DataFrame(rows)


def make_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "model",
            Lasso(
                max_iter=100000,
                tol=1e-3,
                random_state=123,
                selection="random",
            ),
        ),
    ])


def print_selector_summary(run_eval, title):
    log(title)

    med = (
        run_eval
        .groupby("selector")[["score", "regret", "ratio", "epoch"]]
        .median()
        .sort_values("score")
    )

    mean = (
        run_eval
        .groupby("selector")[["score", "regret", "ratio", "epoch"]]
        .mean()
        .sort_values("score")
    )

    print("\n===== selector summary, median =====", flush=True)
    print(med.to_string(), flush=True)

    print("\n===== selector summary, mean =====", flush=True)
    print(mean.to_string(), flush=True)


def main():
    if SUPPRESS_CONVERGENCE_WARNINGS:
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

    log("loading checkpoint dataset")
    df = load_dataset()
    feature_cols = numeric_feature_columns(df)

    dataset_path = OUT_DIR / "checkpoint_lasso_dataset.csv"
    feature_path = OUT_DIR / "lasso_feature_columns.csv"

    df.to_csv(dataset_path, index=False)
    pd.DataFrame({"feature": feature_cols}).to_csv(feature_path, index=False)

    log(f"n rows: {len(df)}")
    log(f"n runs: {df['run_id'].nunique()}")
    log(f"n features: {len(feature_cols)}")
    log(f"dataset: {dataset_path}")
    log(f"feature list: {feature_path}")

    if len(feature_cols) == 0:
        raise ValueError("No numeric non-oracle feature columns found.")

    X = df[feature_cols]
    y = df[TARGET].to_numpy(float)
    groups = df["run_id"].to_numpy()

    unique_groups = np.unique(groups)
    n_splits = min(5, len(unique_groups))

    log(f"outer GroupKFold n_splits = {n_splits}")

    cv = GroupKFold(n_splits=n_splits)

    grid = {
        "model__alpha": np.logspace(-3, 0.5, 25)
    }

    log(f"alpha grid: {grid['model__alpha']}")

    oof = np.full(len(df), np.nan)

    fold_summaries = []

    for fold, (tr, te) in enumerate(cv.split(X, y, groups), start=1):
        fold_start = time.time()

        train_groups = np.unique(groups[tr])
        test_groups = np.unique(groups[te])

        inner_splits = min(4, len(train_groups))

        log(
            f"outer fold {fold}/{n_splits} started | "
            f"train rows={len(tr)}, test rows={len(te)}, "
            f"train runs={len(train_groups)}, test runs={len(test_groups)}, "
            f"inner_splits={inner_splits}"
        )

        search = GridSearchCV(
            estimator=make_pipeline(),
            param_grid=grid,
            scoring="neg_mean_squared_error",
            cv=GroupKFold(n_splits=inner_splits),
            n_jobs=1,
            verbose=1,
        )

        log(f"outer fold {fold}: fitting GridSearchCV")
        search.fit(X.iloc[tr], y[tr], groups=groups[tr])

        best_alpha = search.best_params_["model__alpha"]
        best_score = search.best_score_

        log(
            f"outer fold {fold}: finished fit | "
            f"best alpha={best_alpha:.6g}, "
            f"best neg-MSE={best_score:.6g}"
        )

        pred = search.predict(X.iloc[te])
        oof[te] = pred

        fold_time = time.time() - fold_start

        fold_summaries.append({
            "fold": fold,
            "train_rows": len(tr),
            "test_rows": len(te),
            "train_runs": len(train_groups),
            "test_runs": len(test_groups),
            "inner_splits": inner_splits,
            "best_alpha": best_alpha,
            "best_neg_mse": best_score,
            "elapsed_sec": fold_time,
        })

        tmp = df.copy()
        tmp["lasso_pred_score_oof"] = oof

        tmp_eval = evaluate_selector(
            tmp.loc[tmp["lasso_pred_score_oof"].notna()].copy(),
            "lasso_pred_score_oof",
        )

        print_selector_summary(
            tmp_eval,
            title=f"outer fold {fold}: interim OOF selector summary",
        )

        log(f"outer fold {fold}/{n_splits} completed in {fold_time:.1f} sec")

    df["lasso_pred_score_oof"] = oof

    if df["lasso_pred_score_oof"].isna().any():
        n_missing = int(df["lasso_pred_score_oof"].isna().sum())
        raise RuntimeError(f"OOF predictions still contain NaN: {n_missing}")

    pd.DataFrame(fold_summaries).to_csv(
        OUT_DIR / "lasso_fold_summaries.csv",
        index=False,
    )

    run_eval = evaluate_selector(df, "lasso_pred_score_oof")
    run_eval.to_csv(OUT_DIR / "lasso_oof_run_summary.csv", index=False)

    print_selector_summary(run_eval, title="final OOF selector summary")

    df.to_csv(OUT_DIR / "checkpoint_lasso_dataset_with_oof_pred.csv", index=False)

    log("fitting final Lasso model on all checkpoint rows")

    final_search = GridSearchCV(
        estimator=make_pipeline(),
        param_grid=grid,
        scoring="neg_mean_squared_error",
        cv=GroupKFold(n_splits=n_splits),
        n_jobs=1,
        verbose=1,
    )

    final_search.fit(X, y, groups=groups)

    best_model = final_search.best_estimator_
    lasso = best_model.named_steps["model"]

    bundle = {
        "model": best_model,
        "feature_cols": feature_cols,
        "target": TARGET,
        "train_root": [str(x) for x in ROOTS],
        "best_alpha": final_search.best_params_["model__alpha"],
    }

    joblib.dump(bundle, OUT_DIR / "lasso_checkpoint_rule.joblib")

    log(f"saved lasso checkpoint rule: {OUT_DIR / 'lasso_checkpoint_rule.joblib'}")

    coef = pd.DataFrame({
        "feature": feature_cols,
        "coef": lasso.coef_,
        "abs_coef": np.abs(lasso.coef_),
    }).sort_values("abs_coef", ascending=False)

    coef.to_csv(OUT_DIR / "lasso_coefficients.csv", index=False)

    log(f"best alpha final: {final_search.best_params_['model__alpha']}")
    log(f"nonzero coefficients: {int((coef['abs_coef'] > 0).sum())}")

    print("\n===== top coefficients =====", flush=True)
    print(coef.head(40).to_string(index=False), flush=True)

    log(f"wrote: {OUT_DIR}")


if __name__ == "__main__":
    main()