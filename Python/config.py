from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


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
    optimizer_cls: Any = None
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
    history_round_digits: int = 8
    recovery_min_epoch: int = 100
    recovery_score_col: str = "moment_recovery_score"


@dataclass
class SplitConfig:
    train_frac: float = 0.60
    val_frac: float = 0.20
    test_frac: float = 0.20
    seed: int = 123


@dataclass
class ModelConfig:
    beta_mode: str = "sigmoid"
    coupling_type: str = "semantic"
    conditioner_type: str = "mlp"
    hidden_units: int = 64
    num_hidden_layers: int = 2
    K_q: int = 8
    K_g: int = 8
    scale_clip: float = 2.0
    affine_layers_per_step: int = 3
    group_ids: Any = None
    group_sizes: Any = None
    implementation_structure: str = "double_flow"
    reported_structure: str = "single_flow"

    @property
    def reported_layers(self) -> int:
        return int(self.K_q) + int(self.K_g)


@dataclass
class ExperimentConfig:
    experiment_group: str = "main"
    experiment_name: str = "baseline"
    environment_name: str = "baseline"
    method_name: str = "lastflow"
    table_target: Optional[str] = None


@dataclass
class MCMCConfig:
    mcmc_root: Optional[str] = None
    mcmc_setting: str = "simple"
    mcmc_seed: Optional[int] = None
    reference_role: str = "computational_reference"


@dataclass
class SaveConfig:
    output_dir: Optional[str] = None
    save_summary_csv: bool = True
    save_results_pickle: bool = False
    save_run_manifest_json: bool = True
    save_model_config_json: bool = True
    save_training_config_json: bool = True
    save_split_config_json: bool = True
    save_mcmc_config_json: bool = True
    save_history_csv: bool = True
    save_stage_summaries_csv: bool = True
    save_checkpoint_manifest: bool = False
    save_var_table_csv: bool = True
    save_predictions_csv: bool = True
    save_support_sets_json: bool = True
    save_final_json: bool = True
    save_recovery_json: bool = True
    save_model_state: bool = False
    save_large_samples: bool = False


@dataclass
class BenchmarkConfig:
    support_threshold: float = 0.5
    beta_eps: float = 0.10
    standardize_x: bool = True
    center_y: bool = True
    max_iter: int = 500
    tol: float = 1e-5
    verbose: bool = False
