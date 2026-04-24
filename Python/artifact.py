# artifact.py

import os
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from config import SaveConfig
from utils import ensure_dir, NumpyJSONEncoder

from diagplot import (
    plot_training_overview,
    plot_support_vs_predictive,
    plot_boundary_density,
    plot_uncertainty_vs_abs_boundary,
    plot_support_overlap_heatmap,
)


def save_run_artifacts(out, save_cfg):
    if save_cfg.output_dir is None:
        return

    outdir = ensure_dir(save_cfg.output_dir)
    assert outdir is not None

    history_df = out["history_df"]
    final = out["final"]

    if save_cfg.save_history_csv and isinstance(history_df, pd.DataFrame):
        hist_to_save = history_df.copy()
        if "support_idx" in hist_to_save.columns:
            hist_to_save["support_idx"] = hist_to_save["support_idx"].apply(
                lambda x: json.dumps(list(map(int, x)))
            )
        hist_to_save.to_csv(os.path.join(outdir, "history.csv"), index=False)

    if save_cfg.save_checkpoint_manifest:
        manifest = []
        for ckpt_id, payload in out["checkpoints"].items():
            meta = payload["meta"].copy()
            meta["support_idx"] = json.dumps(list(map(int, meta["support_idx"])))
            manifest.append(meta)
        pd.DataFrame(manifest).to_csv(
            os.path.join(outdir, "checkpoint_manifest.csv"),
            index=False,
        )

    if save_cfg.save_predictions_csv:
        final["var_table"].to_csv(
            os.path.join(outdir, "variable_table.csv"),
            index=False,
        )
        final["pred_table"].to_csv(
            os.path.join(outdir, "prediction_table.csv"),
            index=False,
        )

    if save_cfg.save_support_sets_json:
        support_sets = {
            "selected_ckpt_id": int(out["selected_ckpt_id"]),
            "selected_support": list(map(int, final["selected_support"])),
            "unstable_idx": list(map(int, final["unstable_idx"])),
            "never_selected_idx": list(map(int, final["never_selected_idx"])),
        }
        with open(os.path.join(outdir, "support_sets.json"), "w", encoding="utf-8") as f:
            json.dump(support_sets, f, indent=2)

    if save_cfg.save_final_json:
        final_json = {
            "selected_ckpt_id": int(out["selected_ckpt_id"]),
            "runtime_sec": float(out["runtime_sec"]),
            "stage_summaries": out["stage_summaries"],
            "checkpoint_meta": out["checkpoints"][out["selected_ckpt_id"]]["meta"],
            "selection_metrics": final["selection_metrics"],
            "train_metrics": final["train_metrics"],
            "val_metrics": final["val_metrics"],
            "test_metrics": final["test_metrics"],
        }
        final_json["checkpoint_meta"]["support_idx"] = list(
            map(int, final_json["checkpoint_meta"]["support_idx"])
        )
        with open(os.path.join(outdir, "final_summary.json"), "w", encoding="utf-8") as f:
            json.dump(final_json, f, indent=2)

    if save_cfg.save_plots:
        plot_training_overview(
            history_df,
            os.path.join(outdir, "overview_4panel.png"),
        )
        plot_support_vs_predictive(
            history_df,
            os.path.join(outdir, "support_vs_predictive.png"),
        )
        plot_boundary_density(
            boundary=final["boundary"],
            final_support=final["selected_support"],
            unstable_idx=final["unstable_idx"],
            never_selected_idx=final["never_selected_idx"],
            savepath=os.path.join(outdir, "boundary_density.png"),
        )
        plot_uncertainty_vs_abs_boundary(
            boundary=final["boundary"],
            hard_freq=final["hard_freq"],
            savepath=os.path.join(outdir, "uncertainty_vs_abs_boundary.png"),
        )
        plot_support_overlap_heatmap(
            history_df,
            savepath=os.path.join(outdir, "support_overlap_heatmap.png"),
        )


# =============================================================================
# Mean-field benchmark artifacts
# Keep this function behavior identical to the old meanfield_benchmark_core.py.
# =============================================================================
def save_result_artifacts(result: Mapping[str, Any], save_cfg: SaveConfig) -> None:
    if not save_cfg.output_dir:
        return

    outdir = Path(save_cfg.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    method = str(result.get("method", "unknown"))
    seed = result.get("seed", "na")
    stem = f"{method}_seed{seed}"

    if save_cfg.save_history_csv and isinstance(result.get("history"), pd.DataFrame):
        result["history"].to_csv(outdir / f"{stem}_history.csv", index=False)

    if save_cfg.save_predictions_csv:
        yhat = np.asarray(result.get("yhat", []), dtype=float)
        if yhat.size > 0:
            pd.DataFrame({"yhat": yhat}).to_csv(
                outdir / f"{stem}_predictions.csv",
                index=False,
            )

    if save_cfg.save_var_table_csv and isinstance(result.get("var_table"), pd.DataFrame):
        result["var_table"].to_csv(outdir / f"{stem}_var_table.csv", index=False)

    if save_cfg.save_final_json:
        json_payload = {
            k: v
            for k, v in result.items()
            if k not in {"history", "pred_table", "var_table", "raw"}
        }
        with open(outdir / f"{stem}_summary.json", "w", encoding="utf-8") as f:
            json.dump(
                json_payload,
                f,
                cls=NumpyJSONEncoder,
                ensure_ascii=False,
                indent=2,
            )


def save_benchmark_table(
    table: pd.DataFrame,
    save_cfg: SaveConfig,
    filename: str = "benchmark_table.csv",
) -> None:
    if save_cfg.output_dir and save_cfg.save_benchmark_csv:
        outdir = Path(save_cfg.output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        table.to_csv(outdir / filename, index=False)