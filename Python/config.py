from dataclasses import dataclass
from typing import Any, Optional
import torch


@dataclass
class StagewiseAnnealConfig:
    tau_start: float = 1.0
    tau_end: float = 0.40
    warmup_epochs: int = 200
    n_anneal_stages: int = 5
    min_stage_epochs: int = 80
    max_stage_epochs: int = 220
    base_lr: float = 5e-5
    stage_lr_decay: float = 0.70
    min_lr_scale: float = 0.20
    num_samples: int = 128
    grad_clip: Optional[float] = 3.0
    elbo_beta: float = 1.0
    optimizer_cls: Any = torch.optim.Adam
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    eval_every: int = 25
    checkpoint_every: int = 25
    print_every: int = 25
    ema_beta: float = 0.90
    diag_R_train: int = 256
    diag_R_final: int = 4000
    support_threshold: float = 0.50
    instability_window: int = 2
    plateau_window: int = 2
    plateau_loss_rel: float = 1e-3
    plateau_pred_rel: float = 2e-3
    plateau_grad_slope: float = 0.05
    plateau_churn: float = 0.10
    plateau_abs_dS: int = 1
    alert_grad_slope: float = 0.25
    alert_support_jump_abs: int = 2
    alert_support_jump_rel: float = 0.15
    alert_loss_rel: float = 5e-4
    alert_pred_rel: float = 0.0
    alert_churn: float = 0.25
    retry_shrink_power: float = 0.5
    max_retries_per_drop: int = 2
    history_round_digits: int = 8


@dataclass
class SplitConfig:
    train_frac: float = 0.60
    val_frac: float = 0.20
    test_frac: float = 0.20
    seed: int = 123


@dataclass
class SaveConfig:
    output_dir: Optional[str] = None
    save_plots: bool = True
    save_history_csv: bool = True
    save_checkpoint_manifest: bool = True
    save_final_json: bool = True
    save_predictions_csv: bool = True
    save_support_sets_json: bool = True