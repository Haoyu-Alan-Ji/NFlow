from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any, Mapping, Optional
import torch
import numpy as np
import pandas as pd

from .config import SaveConfig
from .utils import NumpyJSONEncoder, ensure_dir


def save_run_artifacts(out, save_cfg):

    if getattr(save_cfg, "output_dir", None) is None:
        return

    outdir = ensure_dir(save_cfg.output_dir)
    assert outdir is not None

    history_df = out.get("history_df", None)
    final = out.get("final", {})

    if getattr(save_cfg, "save_history_csv", False) and isinstance(history_df, pd.DataFrame):
        hist_to_save = history_df.copy()
        if "support_idx" in hist_to_save.columns:
            hist_to_save["support_idx"] = hist_to_save["support_idx"].apply(
                lambda x: json.dumps(list(map(int, x)))
            )
        hist_to_save.to_csv(os.path.join(outdir, "history.csv"), index=False)

    if getattr(save_cfg, "save_checkpoint_manifest", False):
        manifest = []
        for ckpt_id, payload in out.get("checkpoints", {}).items():
            meta = payload["meta"].copy()
            if "support_idx" in meta:
                meta["support_idx"] = json.dumps(list(map(int, meta["support_idx"])))
            manifest.append(meta)

        pd.DataFrame(manifest).to_csv(
            os.path.join(outdir, "checkpoint_manifest.csv"),
            index=False,
        )

    if getattr(save_cfg, "save_var_table_csv", False):
        if isinstance(final.get("var_table", None), pd.DataFrame):
            final["var_table"].to_csv(
                os.path.join(outdir, "variable_table.csv"),
                index=False,
            )

    if getattr(save_cfg, "save_predictions_csv", False):
        if isinstance(final.get("pred_table", None), pd.DataFrame):
            final["pred_table"].to_csv(
                os.path.join(outdir, "prediction_table.csv"),
                index=False,
            )

    if getattr(save_cfg, "save_support_sets_json", False):
        support_sets = {
            "selected_ckpt_id": int(out.get("selected_ckpt_id", -1)),
            "selected_support": list(map(int, final.get("selected_support", []))),
            "unstable_idx": list(map(int, final.get("unstable_idx", []))),
            "never_selected_idx": list(map(int, final.get("never_selected_idx", []))),
        }
        with open(os.path.join(outdir, "support_sets.json"), "w", encoding="utf-8") as f:
            json.dump(support_sets, f, indent=2)

    if getattr(save_cfg, "save_final_json", False):
        selected_ckpt_id = out.get("selected_ckpt_id", None)
        checkpoints = out.get("checkpoints", {})

        checkpoint_meta = None
        if selected_ckpt_id is not None and selected_ckpt_id in checkpoints:
            checkpoint_meta = checkpoints[selected_ckpt_id].get("meta", {}).copy()
            if "support_idx" in checkpoint_meta:
                checkpoint_meta["support_idx"] = list(
                    map(int, checkpoint_meta["support_idx"])
                )

        final_json = {
            "selected_ckpt_id": int(selected_ckpt_id) if selected_ckpt_id is not None else None,
            "runtime_sec": float(out.get("runtime_sec", np.nan)),
            "stage_summaries": out.get("stage_summaries", []),
            "checkpoint_meta": checkpoint_meta,
            "selection_metrics": final.get("selection_metrics", {}),
            "train_metrics": final.get("train_metrics", {}),
            "val_metrics": final.get("val_metrics", {}),
            "test_metrics": final.get("test_metrics", {}),
        }

        with open(os.path.join(outdir, "final_summary.json"), "w", encoding="utf-8") as f:
            json.dump(final_json, f, cls=NumpyJSONEncoder, ensure_ascii=False, indent=2)


def save_flow_run_artifacts(out, save_cfg):
    return save_run_artifacts(out, save_cfg)


def save_result_artifacts(result: Mapping[str, Any], save_cfg: SaveConfig) -> None:

    if not getattr(save_cfg, "output_dir", None):
        return

    outdir = Path(save_cfg.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    method = str(result.get("method", "unknown"))
    seed = result.get("seed", "na")
    stem = f"{method}_seed{seed}"

    if getattr(save_cfg, "save_history_csv", False) and isinstance(result.get("history"), pd.DataFrame):
        result["history"].to_csv(outdir / f"{stem}_history.csv", index=False)

    if getattr(save_cfg, "save_predictions_csv", False):
        if isinstance(result.get("pred_table"), pd.DataFrame):
            result["pred_table"].to_csv(outdir / f"{stem}_pred_table.csv", index=False)

        yhat = np.asarray(result.get("yhat", []), dtype=float)
        if yhat.size > 0:
            pd.DataFrame({"yhat": yhat}).to_csv(
                outdir / f"{stem}_predictions.csv",
                index=False,
            )

    if getattr(save_cfg, "save_var_table_csv", False) and isinstance(result.get("var_table"), pd.DataFrame):
        result["var_table"].to_csv(outdir / f"{stem}_var_table.csv", index=False)

    if getattr(save_cfg, "save_support_sets_json", False):
        selected_support = result.get("selected_support", [])
        with open(outdir / f"{stem}_selected_support.json", "w", encoding="utf-8") as f:
            json.dump(list(map(int, selected_support)), f, indent=2)

    if getattr(save_cfg, "save_final_json", False):
        json_payload = {
            "method": result.get("method"),
            "seed": result.get("seed"),
            "runtime_sec": result.get("runtime_sec"),
            "converged": result.get("converged"),
            "n_iter": result.get("n_iter"),
            "support_threshold": result.get("support_threshold"),
            "beta_eps": result.get("beta_eps"),
            "sigma2": result.get("sigma2"),
            "intercept": result.get("intercept"),
            "selected_support": result.get("selected_support"),
            "selection_metrics": result.get("selection_metrics"),
            "predictive_metrics": result.get("predictive_metrics"),
            "sim_info": result.get("sim_info"),
            "config": result.get("config"),
        }

        with open(outdir / f"{stem}_summary.json", "w", encoding="utf-8") as f:
            json.dump(json_payload, f, cls=NumpyJSONEncoder, ensure_ascii=False, indent=2)

def save_meanfield_result_artifacts(result: Mapping[str, Any], save_cfg: SaveConfig) -> None:
    return save_result_artifacts(result, save_cfg)


def save_benchmark_table(
    table: pd.DataFrame,
    save_cfg: SaveConfig,
    filename: str = "benchmark_table.csv",
) -> None:
    if getattr(save_cfg, "output_dir", None) and getattr(save_cfg, "save_benchmark_csv", False):
        outdir = Path(save_cfg.output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        table.to_csv(outdir / filename, index=False)


def save_result_data(
    *,
    results,
    output_dir: Optional[str] = None,
    table: Optional[pd.DataFrame] = None,
    prefix: str = "benchmark",
    save_cfg: Optional[SaveConfig] = None,
) -> None:
    if save_cfg is None:
        save_cfg = SaveConfig(output_dir=output_dir)

    if output_dir is not None:
        save_cfg.output_dir = output_dir

    if not getattr(save_cfg, "output_dir", None):
        return

    outdir = Path(save_cfg.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if isinstance(results, Mapping):
        results = [results]
    else:
        results = list(results)

    if table is not None and getattr(save_cfg, "save_summary_csv", False):
        table.to_csv(outdir / f"{prefix}_summary.csv", index=False)

    if getattr(save_cfg, "save_results_pickle", False):
        with open(outdir / f"{prefix}_results.pkl", "wb") as f:
            pickle.dump(results, f)

    manifest = []

    for i, result in enumerate(results):
        method = str(result.get("method", "flow"))
        seed = result.get("seed", "na")
        stem = f"{prefix}_{method}_seed{seed}_{i}"

        final = result.get("final", {})

        var_table = result.get("var_table", final.get("var_table"))
        pred_table = result.get("pred_table", final.get("pred_table"))
        history = result.get("history", result.get("history_df"))

        if getattr(save_cfg, "save_var_table_csv", False) and isinstance(var_table, pd.DataFrame):
            var_table.to_csv(outdir / f"{stem}_var_table.csv", index=False)

        if getattr(save_cfg, "save_predictions_csv", False) and isinstance(pred_table, pd.DataFrame):
            pred_table.to_csv(outdir / f"{stem}_pred_table.csv", index=False)

        if getattr(save_cfg, "save_history_csv", False) and isinstance(history, pd.DataFrame):
            hist = history.copy()

            if "support_idx" in hist.columns:
                def encode_support_idx(x):
                    if x is None:
                        return json.dumps([])
                    if isinstance(x, str):
                        return x
                    if isinstance(x, torch.Tensor):
                        x = x.detach().cpu().numpy()
                    if isinstance(x, np.ndarray):
                        x = x.tolist()
                    return json.dumps(list(map(int, x)))

                hist["support_idx"] = hist["support_idx"].apply(encode_support_idx)

            hist.to_csv(outdir / f"{stem}_history.csv", index=False)

        selected_support = result.get(
            "selected_support",
            final.get("selected_support", []),
        )

        if getattr(save_cfg, "save_support_sets_json", False):
            if isinstance(selected_support, torch.Tensor):
                selected_support_json = selected_support.detach().cpu().numpy().tolist()
            elif isinstance(selected_support, np.ndarray):
                selected_support_json = selected_support.tolist()
            else:
                selected_support_json = list(selected_support)

            with open(outdir / f"{stem}_selected_support.json", "w", encoding="utf-8") as f:
                json.dump(list(map(int, selected_support_json)), f, indent=2)

        if getattr(save_cfg, "save_yhat_csv", False):
            yhat = result.get("yhat", final.get("yhat"))
            if yhat is not None:
                if isinstance(yhat, torch.Tensor):
                    yhat = yhat.detach().cpu().numpy()
                yhat = np.asarray(yhat, dtype=float)
                pd.DataFrame({"yhat": yhat}).to_csv(
                    outdir / f"{stem}_yhat.csv",
                    index=False,
                )

        if getattr(save_cfg, "save_metadata_json", False):
            metadata = {
                "method": method,
                "seed": seed,
                "runtime_sec": result.get("runtime_sec"),
                "converged": result.get("converged"),
                "n_iter": result.get("n_iter"),
                "support_threshold": result.get("support_threshold"),
                "beta_eps": result.get("beta_eps"),
                "sigma2": result.get("sigma2"),
                "intercept": result.get("intercept"),
                "selection_metrics": result.get(
                    "selection_metrics",
                    final.get("selection_metrics"),
                ),
                "predictive_metrics": result.get(
                    "predictive_metrics",
                    final.get("predictive_metrics"),
                ),
                "train_metrics": final.get("train_metrics"),
                "val_metrics": final.get("val_metrics"),
                "test_metrics": final.get("test_metrics"),
                "sim_info": result.get("sim_info"),
                "config": result.get("config"),
                "selected_support": selected_support,
            }

            with open(outdir / f"{stem}_metadata.json", "w", encoding="utf-8") as f:
                json.dump(
                    metadata,
                    f,
                    cls=NumpyJSONEncoder,
                    ensure_ascii=False,
                    indent=2,
                )

        manifest.append(
            {
                "method": method,
                "seed": seed,
                "stem": stem,
            }
        )

    if getattr(save_cfg, "save_manifest_json", False):
        with open(outdir / f"{prefix}_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)