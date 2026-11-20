from dataclasses import dataclass
from typing import Optional


@dataclass
class DVAICholeskyGVBConfig:
    lr: float = 2e-3
    max_iter: int = 10000
    num_mc: int = 20
    check_every: int = 1
    grad_clip: float = 0.0

    window: int = 50
    seed: int = 42

    track_history: bool = True

    patience: Optional[int] = 500
    min_improve: float = 0.0
    restore_best: bool = False
    jitter: float = 1e-6
    beta1: float = 0.9
    beta2: float = 0.9
    step_adaptive: int = 500
    init_scale: float = 0.1
    denominator_floor: float = 1e-12
    min_abs_diag: float = 1e-12
    eigen_diagnostic_every: int = 0
    final_elbo_samples: int = 2000
    diagnostic_seed: int = 920260


dvai_cholesky_gvb_config = DVAICholeskyGVBConfig()


@dataclass
class DiffusionConfig:
    hidden: int = 1024
    lr: float = 1e-4
    batch_size: int = 4096
    epochs: int = 10000

    sigma_min: float = 0.002
    sigma_max: float = 80.0
    edm_P_mean: float = -1.2
    edm_P_std: float = 1.2
    sigma_data: float = 0.5


    sample_steps: int = 50
    L: int = 6
    langevin_sigma: float = 0.4
    langevin_step_scale: float = 0.02

    data_scale: float = 2.0


    train_seed: Optional[int] = None


diff_config = DiffusionConfig()


@dataclass
class DiffPuterBaselineConfig:
    K: int = 10
    num_imputations: int = 20
    sample_steps: int = 50
    refine_steps: int = 20
    hidden: int = 1024
    epochs: int = 10000
    batch_size: int = 4096
    lr: float = 1e-4
    seed: int = 2026


diffputer_baseline_config = DiffPuterBaselineConfig()


@dataclass
class DVAIConfig:
    K: int = 10
    J: int = 20
    posterior_tol: float = 0.1
    min_iter: int = 3
    patience: int = 2

    final_sample_clamp_val: Optional[float] = None


dvai_config = DVAIConfig()


@dataclass
class EvalConfig:
    theta_mc_samples: int = 50
    acc_threshold: float = 0.5


eval_config = EvalConfig()


@dataclass
class MissForestBaselineConfig:
    n_estimators: int = 100
    max_iter: int = 10
    max_depth: Optional[int] = None
    min_samples_leaf: int = 5
    max_features: str = "sqrt"
    random_state: int = 20260426
    transform_iter: int = 5


missforest_baseline_config = MissForestBaselineConfig()


@dataclass
class MIWAEBaselineConfig:
    latent_dim: int = 20
    hidden_dim: int = 128
    epochs: int = 10000
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0

    num_importance_samples: int = 50
    n_imputations_train: int = 30
    n_imputations_test: int = 30

    patience: int = 50
    monitor_every: int = 100

    seed: int = 20260426


miwae_baseline_config = MIWAEBaselineConfig()


@dataclass
class MICEBaselineConfig:
    num_imputations: int = 5
    max_iter: int = 20
    seed: int = 20260320


mice_baseline_config = MICEBaselineConfig()


@dataclass
class RealDataRunConfig:
    prior_var_used: float = 5.0


realdata_run_config = RealDataRunConfig()


def dvai_stop_protocol(tolerance=None, *, k_max=None):
    if tolerance is not None and tolerance != dvai_config.posterior_tol:
        raise ValueError(f'DVAI outer tolerance is fixed at {dvai_config.posterior_tol}; got {tolerance}')
    return dict(distance='DB/p', comparison='strict_less_than',
                posterior_pair='adjacent_iterations_same_run', covariance='full',
                p='regression_parameter_dimension_including_intercept',
                tolerance=dvai_config.posterior_tol, min_iter=dvai_config.min_iter,
                patience=dvai_config.patience, K_max=dvai_config.K if k_max is None else k_max)


