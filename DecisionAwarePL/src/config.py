from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    version: str = "wu2024_original"
    structure_type: str = "peaked"
    centered_penalty_strength: float = 1.8
    additive_term_scale: float = 1.0
    saturated_interaction_width_ratio: float = 0.4
    saturated_interaction_slope_ratio: float = 3.0
    n_train: int = 2000
    n_val: int = 0
    n_test: int = 3000
    dim_x: int = 5
    dim_z: int = 2
    dim_w: int = 2
    support_lower: float | None = None
    support_upper: float | None = None
    support_quantile_low: float = 0.001
    support_quantile_high: float = 0.999
    support_pilot_size: int = 50000
    support_seed: int = 2718
    policy_grid_size: int = 51
    factual_noise_scale: float = 0.35
    outcome_noise_scale: float = 0.0
    proxy_confounding_scale: float = 0.35
    outcome_confounding_scale: float = 1.2
    nonadditive_confounding_gamma: float = 0.7
    nonadditive_confounding_eta: float = 0.0
    nonadditive_confounding_scale_ratio: float = 0.18
    alpha_nonadditivity: float = 0.3
    bias_strength: float = 0.5
    softened_assignment_scale: float = 1.2
    softened_assignment_center_mix: float = 0.25
    x_cov_rho: float = 0.5
    local_eval_eps: float = 0.05
    single_peak_width_ratio: float = 0.1
    single_peak_shift_scale: float = 0.4
    single_peak_shift_slope: float = 2.7
    single_peak_amplitude_scale: float = 0.25
    single_peak_tail_penalty: float = 0.06
    single_peak_skew_scale: float = 0.
    single_peak_asymmetry_scale: float = 0.70
    # Shaping knobs used when structure_type=single_peaked_heterogeneous and
    # version is nonadditive (e.g., wu2024_nonadditive).
    single_peak_nonadd_width_multiplier: float = 0.45
    single_peak_nonadd_shift_scale_multiplier: float = 0.5
    single_peak_nonadd_amplitude_boost: float = 0.1
    single_peak_nonadd_tail_penalty_boost: float = 0.0

    nonadditive_treatment_term: float = 0.25

    # Synthetic assumption-violation controls. Defaults keep the original DGP.
    proxy_independence_violation_strength: float = 0.0
    proxy_independence_violation_mode: str = "shared_latent_nonlinear_substitution"
    # ``shared_latent_nonlinear_substitution`` is the legacy observed-only
    # intervention: it replaces a growing fraction of the latent-U signals by
    # nonlinear functions of a newly drawn shared latent variable, while the
    # structural proxies remain at baseline.
    # ``gaussian_copula_correlated_disturbances_structural`` instead preserves
    # the latent-U signals and Uniform(-1, 1) disturbance marginals, correlates
    # aligned Z/W disturbances through a Gaussian copula, and uses the same
    # resulting proxies both structurally and observationally.
    # Retained for compatibility with saved configurations.
    proxy_independence_rescale_output: bool = True
    completeness_violation_strength: float = 0.0
    completeness_violation_mode: str = "moving_relu_thresholds"
    # For strength > 0, Z and W collapse a shared central band of the latent
    # confounder. The dead-zone threshold grows from 0 to ``max_threshold``.
    completeness_relu_max_threshold_z: float = 2.0
    completeness_relu_max_threshold_w: float = 2.0
    # Optional additive Gaussian noise applied after completeness coarseening.
    completeness_noise_scale_max: float = 0.0

    # Horizontal shift in treatment argument driven by hidden U_z in *_shifted.
    shifted_confounder_treatment_scale: float = 0.3
    shifted_confounder_treatment_slope: float = 1.8

    causal_mc_samples: int = 2048
    causal_mc_batch_size: int = 1024
    causal_mc_seed: int = 31415
    associational_mc_samples: int = 10000
    associational_min_bin_count: int = 10
    associational_kernel: bool = True

    # TCGA semi-synthetic DGP (version="semisynthetic_tcga", structure_type="v1").
    tcga_data_path: str = "datasets/tcga.p"
    tcga_processed_path: str = "datasets/tcga_semisynthetic_v1_features.csv"
    tcga_metadata_path: str = "datasets/tcga_semisynthetic_v1_metadata.json"
    tcga_top_genes: int = 180
    tcga_block_size: int = 50
    tcga_x_pca_dim: int = 5
    tcga_z_pca_dim: int = 3
    tcga_w_pca_dim: int = 3
    tcga_preprocess_seed: int = 2718
    tcga_param_seed: int = 31415
    tcga_sample_with_replacement: bool = False
    tcga_proxy_noise_scale: float = 0.2
    tcga_treatment_phi: float = 20.0
    tcga_outcome_noise_scale: float = 0.1
    tcga_nonadditive_gamma: float = 0.4
    tcga_normalization_pilot_size: int = 50000
    tcga_causal_mc_samples_per_x: int = 256
    tcga_common_random_numbers: bool = True
    tcga_policy_value_mc_samples: int = 256


@dataclass
class FlowConfig:
    bins: int = 8
    transforms: int = 4
    hidden_features: tuple[int, ...] = (64, 64)
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 400
    min_epochs: int = 50
    early_stop_patience: int = 50
    early_stop_min_delta: float = 1e-4
    batch_size: int = 256
    val_fraction: float = 0.2


def default_propensity_flow_config() -> FlowConfig:
    return FlowConfig(
        bins=6,
        transforms=1,
        hidden_features=(32, 32),
        lr=1e-4,
        weight_decay=1e-5,
        epochs= 1000,
        min_epochs=20,
        early_stop_patience=20,
        early_stop_min_delta=1e-4,
    )


@dataclass
class MeanConfig:
    method: str = "pseudo_outcomes"
    treatment_grid_size: int = 51
    use_treatment_grid: bool = True
    evaluation_grid_size: int = 51
    crossfit_folds: int = 3
    crossfit_shuffle: bool = True
    hidden_dims: tuple[int, ...] = (64, 64)
    lr: float = 3e-5
    weight_decay: float = 1e-6
    epochs: int = 400
    min_epochs: int = 50
    early_stop_patience: int = 30
    early_stop_min_delta: float = 1e-4
    early_stopping_mode: str = "standard"
    early_stopping_v2_alpha: float = 0.02
    early_stopping_v2_patience: int = 5
    batch_size: int = 512
    n_mc_samples: int = 64
    val_fraction: float = 0.2


def default_mean_pmmr_config() -> MeanConfig:
    return MeanConfig(
        hidden_dims=(64, 64, 64),
        lr=1e-4,
        weight_decay=1e-7,
        epochs=2000,
        min_epochs=50,
        early_stop_patience=60,
        batch_size=512,
        crossfit_folds=3,
    )


def default_mean_kpv_config() -> MeanConfig:
    return MeanConfig(
        hidden_dims=(64, 64, 64),
        lr=1e-4,
        weight_decay=1e-7,
        epochs=2000,
        min_epochs=50,
        early_stop_patience=60,
        batch_size=512,
        crossfit_folds=3,
    )


def default_mean_dfpv_config() -> MeanConfig:
    return MeanConfig(
        hidden_dims=(64, 64, 64),
        lr=1e-4,
        weight_decay=1e-7,
        epochs=2000,
        min_epochs=50,
        early_stop_patience=60,
        batch_size=512,
        crossfit_folds=3,
    )


def default_mean_nmmr_u_config() -> MeanConfig:
    return MeanConfig(
        hidden_dims=(64, 64, 64),
        lr=1e-4,
        weight_decay=1e-7,
        epochs=300,
        min_epochs=50,
        early_stop_patience=60,
        batch_size=512,
        crossfit_folds=3,
    )


def default_mean_nmmr_v_config() -> MeanConfig:
    return MeanConfig(
        hidden_dims=(64, 64, 64),
        lr=1e-4,
        weight_decay=1e-7,
        epochs=2000,
        min_epochs=50,
        early_stop_patience=60,
        batch_size=512,
        crossfit_folds=3,
    )


@dataclass
class PMMRConfig:
    ridge: float = 0.05
    jitter: float = 1e-9
    awx_scale: float = 3.0
    azx_scale: float = 1.50
    moment_weight_power: float = 0.5


@dataclass
class KPVConfig:
    lambda1: float | None = None
    lambda2: float | None = None
    optimize_lambda1: bool = True
    optimize_lambda2: bool = True
    lambda1_min: float = 1e-4
    lambda1_max: float = 1e-1
    n_lambda1_search: int = 5
    lambda2_min: float = 1e-3
    lambda2_max: float = 1.0
    n_lambda2_search: int = 5
    kernel_variance_scale: float = 2.0
    split_ratio: float = 0.5
    max_train_size: int | None = None
    standardize: bool = True
    jitter: float = 1e-9
    prediction_batch_size: int = 2048


@dataclass
class DFPVConfig:
    lam1: float = 0.1
    lam2: float = 0.1
    stage1_iter: int = 20
    stage2_iter: int = 1
    epochs: int = 2000
    min_epochs: int = 100
    early_stop_patience: int = 80
    early_stop_min_delta: float = 1e-5
    split_ratio: float = 0.5
    max_train_size: int | None = 3000
    hidden_dims: tuple[int, ...] = (32, 16)
    feature_dim: int = 8
    activation: str = "relu"
    final_activation: bool = False
    lr: float = 1e-3
    weight_decay: float = 0.01
    standardize: bool = True
    include_covariates: bool = True
    jitter: float = 1e-6
    grad_clip_norm: float = 5.0
    prediction_batch_size: int = 2048


@dataclass
class DRKPVConfig:
    kpv_lambda1: float = 0.1
    kpv_lambda2: float = 0.1
    kap_lambda: float = 1e-3
    kap_eta: float = 1e-3
    kap_lambda2: float = 1e-3
    kap_zeta: float = 1e-3
    lambda_dr: float = 1e-3
    optimize_internal: bool = False
    regularization_grid_points: int = 25
    lambda_min: float = 1e-5
    lambda_max: float = 1.0
    kernel_variance_scale: float = 1.0
    stage1_perc: float = 0.5
    max_train_size: int | None = 1000
    standardize: bool = True
    jitter: float = 1e-9
    prediction_batch_size: int = 512
    label_variance_in_lambda_opt: float = 1.0
    label_variance_in_eta_opt: float = 1.0


@dataclass
class DRPMMRMarginalConfig:
    pmmr_lambda: float = 1e-3
    kap_lambda: float = 1e-3
    kap_eta: float = 1e-3
    kap_lambda2: float = 1e-3
    kap_zeta: float = 1e-3
    lambda_dr: float = 1e-3
    optimize_internal: bool = False
    regularization_grid_points: int = 25
    lambda_min: float = 1e-5
    lambda_max: float = 1.0
    kernel_variance_scale: float = 1.0
    stage1_perc: float = 0.5
    max_train_size: int | None = 1000
    standardize: bool = True
    jitter: float = 1e-9
    prediction_batch_size: int = 512


@dataclass
class DRPMMRConditionalConfig(DRPMMRMarginalConfig):
    pass


@dataclass
class NMMRConfig:
    hidden_dims: tuple[int, ...] = (64, 64, 64)
    activation: str = "relu"
    bridge_all_data: bool = True
    deactivate_rolloff: bool = False
    product_kernel: bool = False
    a_scale: float = 1.0
    z_scale: float = 1.0
    x_scale: float = 1.0
    quadratic_treatment: bool = False
    lr: float = 1e-4
    weight_decay: float = 1e-7
    epochs: int = 2000
    min_epochs: int = 75
    early_stop_patience: int = 60
    early_stop_min_delta: float = 1e-5
    # Positive integer mini-batch size, or "all" for full-batch bridge updates.
    batch_size: int | str = 1024
    loss_name: str = "V_statistic"
    azx_scale: float = 0.5
    grad_clip_norm: float = 5.0
    val_fraction: float = 0.2
    moment_weight_power: float = 0.5


def default_nmmr_u_config() -> NMMRConfig:
    return NMMRConfig(
        hidden_dims=(64, 64, 64),
        activation="relu",
        deactivate_rolloff=True,
        product_kernel=False,
        a_scale=1.0,
        z_scale=1.0,
        x_scale=1.0,
        quadratic_treatment=False,
        lr=1e-4,
        weight_decay=1e-8, #5e-6,
        epochs=300,
        min_epochs=75,
        early_stop_patience=60,
        early_stop_min_delta=1e-5,
        batch_size=1024,
        loss_name="U_statistic",
        azx_scale=0.5,
        grad_clip_norm=3.0,
        val_fraction=0.2,
    )


def default_nmmr_v_config() -> NMMRConfig:
    return NMMRConfig(
        hidden_dims=(64, 64, 64),
        activation="relu",
        deactivate_rolloff=True,
        product_kernel=False,
        a_scale=1.0,
        z_scale=1.0,
        x_scale=1.0,
        quadratic_treatment=False,
        lr=1e-4,
        weight_decay=1e-6,
        epochs=2000,
        min_epochs=75,
        early_stop_patience=60,
        early_stop_min_delta=1e-5,
        batch_size=1024,
        loss_name="V_statistic",
        azx_scale=0.5,
        grad_clip_norm=5.0,
        val_fraction=0.2,
    )


@dataclass
class SLearnerConfig:
    hidden_dims: tuple[int, ...] = (128, 64, 32)
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 1000
    min_epochs: int = 75
    early_stop_patience: int = 60
    early_stop_min_delta: float = 1e-4
    batch_size: int = 256
    val_fraction: float = 0.2


@dataclass
class DeCaFlowConfig:
    flow_type: str = "nsf"
    prior_type: str = "maf"
    latent_dim: int = 2
    adjacency_override: tuple[tuple[int, ...], ...] | None = None
    encoder_hidden_features: tuple[int, ...] = (128, 128)
    decoder_hidden_features: tuple[int, ...] = (64, 64, 64)
    conditioner_hidden_features: tuple[int, ...] = (16,16)
    activation: str = "relu"
    regularize: bool = True
    warmup: int = 50
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 400
    min_epochs: int = 50
    early_stop_patience: int = 40
    early_stop_min_delta: float = 1e-4
    batch_size: int = 256
    prediction_batch_size: int = 512
    interventional_mc_samples: int = 1024


@dataclass
class WeightingConfig:
    weight_mode: str = "decision_aware"
    lam: float = 0.4
    tau: float = 1.3
    root_weights: bool = False
    weight_mean_regressor: bool = False
    warm_start: bool = True
    truncated_kernel_weights: bool = True
    normalization_mode: str = "paper_normalization"
    normalize_grid_size: int = 1000
    stabilized: bool = True
    n_rounds: int = 2
    early_stop_patience: int = 30
    min_improvement: float = 1e-4
    clip_min: float = 0.05
    clip_max: float = 20.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "_explicit_fields", set())
        object.__setattr__(self, "_track_explicit_fields", True)

    def __setattr__(self, name: str, value: object) -> None:
        object.__setattr__(self, name, value)
        if name.startswith("_"):
            return
        if getattr(self, "_track_explicit_fields", False):
            self._explicit_fields.add(name)


def default_local_weighting_config() -> WeightingConfig:
    cfg = WeightingConfig()
    cfg.weight_mode = "local_only"
    return cfg

@dataclass
class KernelDiagnosticsConfig:
    enabled: bool = False
    max_points: int = 300
    seed: int = 13
    bandwidth_multipliers: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
    neighbor_thresholds: tuple[float, ...] = (0.1, 0.5)
    near_zero_threshold: float = 0.01
    near_one_threshold: float = 0.95
    max_offdiag_values: int = 4000
    save_offdiag_values: bool = False


@dataclass
class ExperimentConfig:
    seed: int = 13
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    verbose: int = 0
    plot_losses: bool = False
    bridge_diagnostics: bool = False
    save_outputs: bool = True
    save_figures: bool = True
    fast_factual_eval: bool = False
    use_mean_slearner: bool = False
    use_mean_oosr: bool = False
    data: DataConfig = field(default_factory=DataConfig)
    propensity_flow: FlowConfig = field(default_factory=default_propensity_flow_config)
    w_flow: FlowConfig = field(default_factory=FlowConfig)
    mean: MeanConfig = field(default_factory=MeanConfig)
    mean_pmmr: MeanConfig = field(default_factory=default_mean_pmmr_config)
    mean_kpv: MeanConfig = field(default_factory=default_mean_kpv_config)
    mean_dfpv: MeanConfig = field(default_factory=default_mean_dfpv_config)
    mean_nmmr_u: MeanConfig = field(default_factory=default_mean_nmmr_u_config)
    mean_nmmr_v: MeanConfig = field(default_factory=default_mean_nmmr_v_config)
    pmmr: PMMRConfig = field(default_factory=PMMRConfig)
    kpv: KPVConfig = field(default_factory=KPVConfig)
    dfpv: DFPVConfig = field(default_factory=DFPVConfig)
    dr_kpv: DRKPVConfig = field(default_factory=DRKPVConfig)
    dr_pmmr: DRPMMRMarginalConfig = field(default_factory=DRPMMRMarginalConfig)
    dr_pmmr_conditional: DRPMMRConditionalConfig = field(default_factory=DRPMMRConditionalConfig)
    nmmr: NMMRConfig = field(default_factory=NMMRConfig)
    nmmr_u: NMMRConfig = field(default_factory=default_nmmr_u_config)
    nmmr_v: NMMRConfig = field(default_factory=default_nmmr_v_config)
    s_learner: SLearnerConfig = field(default_factory=SLearnerConfig)
    decaflow: DeCaFlowConfig = field(default_factory=DeCaFlowConfig)
    weighting: WeightingConfig = field(default_factory=WeightingConfig)
    weighting_kpv: WeightingConfig = field(default_factory=WeightingConfig)
    weighting_dfpv: WeightingConfig = field(default_factory=WeightingConfig)
    weighting_pmmr: WeightingConfig = field(default_factory=WeightingConfig)
    weighting_local_kpv: WeightingConfig = field(default_factory=default_local_weighting_config)
    weighting_local_dfpv: WeightingConfig = field(default_factory=default_local_weighting_config)
    weighting_local_pmmr: WeightingConfig = field(default_factory=default_local_weighting_config)
    weighting_nmmr_u: WeightingConfig = field(default_factory=WeightingConfig)
    weighting_nmmr_v: WeightingConfig = field(default_factory=WeightingConfig)
    weighting_naive_oosr: WeightingConfig = field(default_factory=WeightingConfig)
    weighting_oracle_naive_oosr: WeightingConfig = field(default_factory=WeightingConfig)
    kernel_diagnostics: KernelDiagnosticsConfig = field(default_factory=KernelDiagnosticsConfig)


def config_to_dict(config: ExperimentConfig) -> dict:
    return asdict(config)


def mean_config_for_model(config: ExperimentConfig, model_key: str) -> MeanConfig:
    if model_key in {"pmmr", "weighted_pmmr", "local_pmmr"}:
        return config.mean_pmmr
    if model_key in {"kpv", "weighted_kpv", "local_kpv"}:
        return config.mean_kpv
    if model_key in {"dfpv", "weighted_dfpv", "local_dfpv"}:
        return config.mean_dfpv
    if model_key == "dr_kpv":
        return config.mean_kpv
    if model_key in {"dr_pmmr", "dr_pmmr_conditional"}:
        return config.mean_pmmr
    if model_key in {"nmmr_u", "weighted_nmmr_u"}:
        return config.mean_nmmr_u
    if model_key in {"nmmr_v", "weighted_nmmr_v"}:
        return config.mean_nmmr_v
    if model_key in {"nmmr", "weighted_nmmr"}:
        return config.mean_nmmr_u if config.nmmr.loss_name == "U_statistic" else config.mean_nmmr_v
    return config.mean


def all_mean_configs(config: ExperimentConfig) -> tuple[MeanConfig, ...]:
    return (
        config.mean,
        config.mean_pmmr,
        config.mean_kpv,
        config.mean_dfpv,
        config.mean_nmmr_u,
        config.mean_nmmr_v,
    )


def nmmr_config_for_model(config: ExperimentConfig, model_key: str) -> NMMRConfig:
    if model_key in {"nmmr_u", "weighted_nmmr_u"}:
        return config.nmmr_u
    if model_key in {"nmmr_v", "weighted_nmmr_v"}:
        return config.nmmr_v
    if model_key in {"weighted_nmmr"}:
        return config.nmmr_u if config.nmmr.loss_name == "U_statistic" else config.nmmr_v
    if model_key == "nmmr":
        if config.nmmr.loss_name == "U_statistic":
            return config.nmmr_u
        if config.nmmr.loss_name == "V_statistic":
            return config.nmmr_v
    return config.nmmr


def all_nmmr_configs(config: ExperimentConfig) -> tuple[NMMRConfig, ...]:
    return (
        config.nmmr,
        config.nmmr_u,
        config.nmmr_v,
    )


def all_weighting_configs(config: ExperimentConfig) -> tuple[WeightingConfig, ...]:
    return (
        config.weighting,
        config.weighting_kpv,
        config.weighting_dfpv,
        config.weighting_pmmr,
        config.weighting_local_kpv,
        config.weighting_local_dfpv,
        config.weighting_local_pmmr,
        config.weighting_nmmr_u,
        config.weighting_nmmr_v,
        config.weighting_naive_oosr,
        config.weighting_oracle_naive_oosr,
    )




def _merged_weighting_config(base: WeightingConfig, override: WeightingConfig) -> WeightingConfig:
    merged = asdict(base)
    override_dict = asdict(override)
    explicit_fields = getattr(override, "_explicit_fields", set())
    if explicit_fields:
        for key in explicit_fields:
            merged[key] = override_dict[key]
        return WeightingConfig(**merged)
    default_dict = asdict(WeightingConfig())
    for key, value in override_dict.items():
        if value != default_dict[key]:
            merged[key] = value
    return WeightingConfig(**merged)


def weighting_config_for_model(config: ExperimentConfig, model_key: str) -> WeightingConfig:
    if model_key in {"weighted_kpv"}:
        return _merged_weighting_config(config.weighting, config.weighting_kpv)
    if model_key in {"weighted_dfpv"}:
        return _merged_weighting_config(config.weighting, config.weighting_dfpv)
    if model_key in {"weighted_pmmr"}:
        return _merged_weighting_config(config.weighting, config.weighting_pmmr)
    if model_key in {"local_kpv"}:
        return _merged_weighting_config(config.weighting, config.weighting_local_kpv)
    if model_key in {"local_dfpv"}:
        return _merged_weighting_config(config.weighting, config.weighting_local_dfpv)
    if model_key in {"local_pmmr"}:
        return _merged_weighting_config(config.weighting, config.weighting_local_pmmr)
    if model_key in {"weighted_nmmr_u"}:
        return _merged_weighting_config(config.weighting, config.weighting_nmmr_u)
    if model_key in {"weighted_nmmr_v"}:
        return _merged_weighting_config(config.weighting, config.weighting_nmmr_v)
    if model_key == "weighted_nmmr":
        target = config.weighting_nmmr_u if nmmr_config_for_model(config, model_key).loss_name == "U_statistic" else config.weighting_nmmr_v
        return _merged_weighting_config(config.weighting, target)
    if model_key == "naive_oosr":
        return _merged_weighting_config(config.weighting, config.weighting_naive_oosr)
    if model_key == "oracle_naive_oosr":
        return _merged_weighting_config(config.weighting, config.weighting_oracle_naive_oosr)
    return config.weighting
