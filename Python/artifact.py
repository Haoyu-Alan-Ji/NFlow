from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd
import torch

from .config import SaveConfig
from .utils import NumpyJSONEncoder, ensure_dir, to_jsonable, write_json


def _df_to_csv(df, path: Path) -> None:
    if isinstance(df, pd.DataFrame):
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)


def _support_json(x):
    if x is None:
        return []
    if isinstance(x, str):
        return x
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        x = x.tolist()
    return json.dumps(list(map(int, x)))


def _history_for_csv(history) -> Optional[pd.DataFrame]:
    if history is None:
        return None
    df = history if isinstance(history, pd.DataFrame) else pd.DataFrame(history)
    df = df.copy()
    for col in ["support_idx", "selected_support", "unstable_idx", "never_selected_idx"]:
        if col in df.columns:
            df[col] = df[col].apply(_support_json)
    return df


def save_run_artifacts(out: Mapping[str, Any], save_cfg: SaveConfig) -> None:
    if getattr(save_cfg, "output_dir", None) is None:
        return

    outdir = Path(ensure_dir(save_cfg.output_dir))
    final = out.get("final", {}) or {}

    if getattr(save_cfg, "save_run_manifest_json", True):
        write_json(outdir / "run_manifest.json", out.get("run_manifest", {}))
    if getattr(save_cfg, "save_model_config_json", True):
        write_json(outdir / "model_config.json", out.get("model_config", out.get("run_manifest", {}).get("model_config", {})))
    if getattr(save_cfg, "save_training_config_json", True):
        write_json(outdir / "training_config.json", out.get("config", out.get("run_manifest", {}).get("training_config", {})))
    if getattr(save_cfg, "save_split_config_json", True):
        split_payload = out.get("split_config", out.get("run_manifest", {}).get("split_config", {}))
        if "splits" in out and isinstance(out["splits"], Mapping):
            idx = out["splits"].get("indices")
            if idx is not None:
                split_payload = {**to_jsonable(split_payload), "indices": to_jsonable(idx)}
        write_json(outdir / "split_config.json", split_payload)
    if getattr(save_cfg, "save_mcmc_config_json", True):
        write_json(outdir / "mcmc_config.json", out.get("mcmc_config", out.get("mcmc_info", out.get("run_manifest", {}).get("mcmc_info", {}))))

    if getattr(save_cfg, "save_history_csv", True):
        hist = _history_for_csv(out.get("history_df", out.get("history")))
        _df_to_csv(hist, outdir / "history.csv")

    if getattr(save_cfg, "save_stage_summaries_csv", True):
        stage = out.get("stage_summaries")
        if stage is not None:
            _df_to_csv(pd.DataFrame(stage), outdir / "stage_summaries.csv")

    if getattr(save_cfg, "save_checkpoint_manifest", False):
        rows = []
        for ckpt_id, payload in (out.get("checkpoints", {}) or {}).items():
            meta = dict(payload.get("meta", {}))
            meta["ckpt_id"] = int(ckpt_id)
            rows.append(meta)
        _df_to_csv(_history_for_csv(rows), outdir / "checkpoint_manifest.csv")

    if getattr(save_cfg, "save_var_table_csv", True):
        _df_to_csv(final.get("var_table"), outdir / "variable_table.csv")
    if getattr(save_cfg, "save_predictions_csv", True):
        _df_to_csv(final.get("pred_table"), outdir / "prediction_table.csv")

    if getattr(save_cfg, "save_support_sets_json", True):
        write_json(outdir / "support_sets.json", {
            "selected_ckpt_id": out.get("selected_ckpt_id"),
            "selected_support": final.get("selected_support", []),
            "unstable_idx": final.get("unstable_idx", []),
            "never_selected_idx": final.get("never_selected_idx", []),
        })

    summary_row = out.get("summary_row")
    if getattr(save_cfg, "save_summary_csv", True) and summary_row is not None:
        _df_to_csv(pd.DataFrame([summary_row]), outdir / "summary_row.csv")

    recovery_metrics = final.get("recovery_metrics", {}) or {}
    if recovery_metrics:
        write_json(outdir / "recovery_summary.json", recovery_metrics)

    if getattr(save_cfg, "save_final_json", True):
        selected_ckpt_id = out.get("selected_ckpt_id")
        checkpoints = out.get("checkpoints", {}) or {}
        meta = None
        if selected_ckpt_id is not None and selected_ckpt_id in checkpoints:
            meta = checkpoints[selected_ckpt_id].get("meta", {})
        write_json(outdir / "final_summary.json", {
            "selected_ckpt_id": selected_ckpt_id,
            "runtime_sec": out.get("runtime_sec"),
            "stage_summaries": out.get("stage_summaries", []),
            "checkpoint_meta": meta,
            "selection_metrics": final.get("selection_metrics", {}),
            "recovery_metrics": final.get("recovery_metrics", {}),
            "train_metrics": final.get("train_metrics", {}),
            "val_metrics": final.get("val_metrics", {}),
            "test_metrics": final.get("test_metrics", {}),
        })

    if getattr(save_cfg, "save_model_state", False):
        model = out.get("model")
        if model is not None:
            torch.save(model.state_dict(), outdir / "model_state.pt")

    if getattr(save_cfg, "save_results_pickle", False):
        with open(outdir / "result.pkl", "wb") as f:
            pickle.dump(dict(out), f)


def save_flow_run_artifacts(out, save_cfg):
    save_run_artifacts(out, save_cfg)


def save_result_artifacts(result: Mapping[str, Any], save_cfg: SaveConfig) -> None:
    save_run_artifacts(result, save_cfg)


def save_meanfield_result_artifacts(result: Mapping[str, Any], save_cfg: SaveConfig) -> None:
    save_result_artifacts(result, save_cfg)


def save_benchmark_table(table: pd.DataFrame, save_cfg: SaveConfig, filename: str = "benchmark_table.csv") -> None:
    if getattr(save_cfg, "output_dir", None) and getattr(save_cfg, "save_summary_csv", True):
        outdir = Path(save_cfg.output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        table.to_csv(outdir / filename, index=False)


def save_result_data(*, results, output_dir: Optional[str] = None, table: Optional[pd.DataFrame] = None, prefix: str = "benchmark", save_cfg: Optional[SaveConfig] = None) -> None:
    if save_cfg is None:
        save_cfg = SaveConfig(output_dir=output_dir)
    if output_dir is not None:
        save_cfg.output_dir = output_dir
    if not getattr(save_cfg, "output_dir", None):
        return

    outdir = Path(save_cfg.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    results = [results] if isinstance(results, Mapping) else list(results)

    if table is not None and getattr(save_cfg, "save_summary_csv", True):
        table.to_csv(outdir / f"{prefix}_summary.csv", index=False)
    if getattr(save_cfg, "save_results_pickle", False):
        with open(outdir / f"{prefix}_results.pkl", "wb") as f:
            pickle.dump(results, f)

    manifest = []
    for i, result in enumerate(results):
        method = str(result.get("method", "flow"))
        seed = result.get("seed", "na")
        stem = f"{prefix}_{method}_seed{seed}_{i}"
        final = result.get("final", {}) or {}
        _df_to_csv(final.get("var_table", result.get("var_table")), outdir / f"{stem}_var_table.csv")
        _df_to_csv(final.get("pred_table", result.get("pred_table")), outdir / f"{stem}_pred_table.csv")
        _df_to_csv(_history_for_csv(result.get("history_df", result.get("history"))), outdir / f"{stem}_history.csv")
        write_json(outdir / f"{stem}_metadata.json", {
            "method": method,
            "seed": seed,
            "runtime_sec": result.get("runtime_sec"),
            "sim_info": result.get("sim_info"),
            "run_manifest": result.get("run_manifest"),
            "summary_row": result.get("summary_row"),
            "selected_support": final.get("selected_support", result.get("selected_support", [])),
        })
        manifest.append({"method": method, "seed": seed, "stem": stem})

    write_json(outdir / f"{prefix}_manifest.json", manifest)
