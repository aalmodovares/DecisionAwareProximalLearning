from __future__ import annotations

import copy
import json
from pathlib import Path
import csv
import time

import numpy as np

from src.config import (
    ExperimentConfig,
    all_mean_configs,
    config_to_dict,
    mean_config_for_model,
    nmmr_config_for_model,
    weighting_config_for_model,
)
from src.data.factory import make_simulator
from src.data.simulator import FactualSample, Wu2024Simulator
from src.evaluation import (
    build_causal_surface_reference,
    evaluate_direct_surface,
    evaluate_marginal_curve,
    evaluate_mean_surface,
    evaluate_policy_metrics,
    marginal_curve_from_bridge_predictor,
    marginal_curve_from_direct_predictor,
    marginal_curve_from_mean_predictor,
    policy_from_direct_predictor,
    policy_from_mean_predictor,
)
from src.models.flows import ConditionalFlowDensity
from src.models.decaflow import ConditionalDeCaFlow
from src.models.dfpv import BridgeTrainData as DFPVTrainData
from src.models.dfpv import DFPVBridge
from src.models.dr_kernel_proxy import BridgeTrainData as DRKernelTrainData
from src.models.dr_kernel_proxy import DRKPVBridge, DRPMMRConditionalBridge, DRPMMRMarginalBridge
from src.models.kpv import BridgeTrainData as KPVTrainData
from src.models.kpv import KPVBridge
from src.models.nmmr import BridgeTrainData as NMMRTrainData
from src.models.nmmr import NMMRBridge
from src.models.pmmr import BridgeTrainData as PMMRTrainData
from src.models.pmmr import PMMRBridge
from src.reporting import save_bridge_diagnostics, save_loss_figures, save_run_figures
from src.reporting_kernel import save_kernel_diagnostics
from src.models.s_learner import SLearner
from src.results_export import ResultsExportConfig, save_result_curve_tables
from src.utils import effective_sample_size, ensure_dir, save_json, set_seed, train_val_split, verbose_print
from src.utils import kfold_indices
from src.weighting.oosr import build_base_density_weights, build_local_only_weights, finalize_weights, local_oosr_factor


ALL_MODEL_KEYS = (
    "pmmr",
    "local_pmmr",
    "kpv",
    "local_kpv",
    "weighted_kpv",
    "dfpv",
    "local_dfpv",
    "weighted_dfpv",
    "dr_kpv",
    "nmmr",
    "nmmr_u",
    "nmmr_v",
    "weighted_pmmr",
    "weighted_nmmr",
    "weighted_nmmr_u",
    "weighted_nmmr_v",
    "s_learner",
    "oracle_s_learner",
    "naive_oosr",
    "oracle_naive_oosr",
)

EXPERIMENTAL_MODEL_KEYS = (
    "decaflow",
    "dr_pmmr",
    "dr_pmmr_conditional",
)

VALID_MODEL_KEYS = ALL_MODEL_KEYS + EXPERIMENTAL_MODEL_KEYS

MODEL_DISPLAY_NAMES = {
    "pmmr": "PMMR",
    "local_pmmr": "Local PMMR",
    "kpv": "KPV",
    "local_kpv": "Local KPV",
    "weighted_kpv": "Weighted KPV",
    "dfpv": "DFPV",
    "local_dfpv": "Local DFPV",
    "weighted_dfpv": "Weighted DFPV",
    "dr_kpv": "DR-KPV",
    "decaflow": "DeCaFlow",
    "dr_pmmr": "DR-PMMR marginal",
    "dr_pmmr_conditional": "DR-PMMR conditional",
    "nmmr": "NMMR",
    "nmmr_u": "NMMR-U",
    "nmmr_v": "NMMR-V",
    "weighted_pmmr": "Weighted PMMR",
    "weighted_nmmr": "Weighted NMMR",
    "weighted_nmmr_u": "Weighted NMMR-U",
    "weighted_nmmr_v": "Weighted NMMR-V",
    "s_learner": "S-Learner",
    "oracle_s_learner": "Oracle S-Learner",
    "naive_oosr": "Naive OOSR",
    "oracle_naive_oosr": "Oracle Naive OOSR",
}

MODEL_POLICY_KEYS = {
    "pmmr": "PMMR",
    "local_pmmr": "Local_PMMR",
    "kpv": "KPV",
    "local_kpv": "Local_KPV",
    "weighted_kpv": "Weighted_KPV",
    "dfpv": "DFPV",
    "local_dfpv": "Local_DFPV",
    "weighted_dfpv": "Weighted_DFPV",
    "dr_kpv": "DR_KPV",
    "decaflow": "DeCaFlow",
    "dr_pmmr": "DR_PMMR_Marginal",
    "dr_pmmr_conditional": "DR_PMMR_Conditional",
    "nmmr": "NMMR",
    "nmmr_u": "NMMR_U",
    "nmmr_v": "NMMR_V",
    "weighted_pmmr": "Weighted_PMMR",
    "weighted_nmmr": "Weighted_NMMR",
    "weighted_nmmr_u": "Weighted_NMMR_U",
    "weighted_nmmr_v": "Weighted_NMMR_V",
    "s_learner": "S_Learner",
    "oracle_s_learner": "Oracle_S_Learner",
    "naive_oosr": "Naive_OOSR",
    "oracle_naive_oosr": "Oracle_Naive_OOSR",
}

BRIDGE_MEAN_MODELS = {
    "pmmr",
    "local_pmmr",
    "kpv",
    "local_kpv",
    "weighted_kpv",
    "dfpv",
    "local_dfpv",
    "weighted_dfpv",
    "dr_kpv",
    "dr_pmmr",
    "dr_pmmr_conditional",
    "nmmr",
    "nmmr_u",
    "nmmr_v",
    "weighted_pmmr",
    "weighted_nmmr",
    "weighted_nmmr_u",
    "weighted_nmmr_v",
}


def _subset_sample(sample: FactualSample, idx: np.ndarray) -> FactualSample:
    return FactualSample(
        treatment=sample.treatment[idx],
        treatment_proxy=sample.treatment_proxy[idx],
        outcome_proxy=sample.outcome_proxy[idx],
        covariates=sample.covariates[idx],
        outcome=sample.outcome[idx],
        optimal_treatment=sample.optimal_treatment[idx],
        assignment_mean=sample.assignment_mean[idx],
        latent_uz=sample.latent_uz[idx],
        latent_uw=sample.latent_uw[idx],
    )


def _model_hparam_summary(model_key: str, config: ExperimentConfig) -> dict[str, object]:
    if model_key == "decaflow":
        return {
            "model_key": model_key,
            "seed": config.seed,
            "dataset": {
                "version": config.data.version,
                "structure_type": config.data.structure_type,
                "n_train": config.data.n_train,
                "n_val": config.data.n_val,
                "n_test": config.data.n_test,
                "policy_grid_size": config.data.policy_grid_size,
            },
            "surface_model": {
                "type": "DeCaFlow",
                "flow_type": config.decaflow.flow_type,
                "latent_dim": config.decaflow.latent_dim,
                "adjacency_override": config.decaflow.adjacency_override,
                "encoder_hidden_features": config.decaflow.encoder_hidden_features,
                "decoder_hidden_features": config.decaflow.decoder_hidden_features,
                "conditioner_hidden_features": config.decaflow.conditioner_hidden_features,
                "activation": config.decaflow.activation,
                "regularize": config.decaflow.regularize,
                "warmup": config.decaflow.warmup,
                "lr": config.decaflow.lr,
                "weight_decay": config.decaflow.weight_decay,
                "epochs": config.decaflow.epochs,
                "min_epochs": config.decaflow.min_epochs,
                "early_stop_patience": config.decaflow.early_stop_patience,
                "batch_size": config.decaflow.batch_size,
                "prediction_batch_size": config.decaflow.prediction_batch_size,
                "interventional_mc_samples": config.decaflow.interventional_mc_samples,
            },
            "kernel_diagnostics": {
                "enabled": config.kernel_diagnostics.enabled,
                "bandwidth_multipliers": config.kernel_diagnostics.bandwidth_multipliers,
                "max_points": config.kernel_diagnostics.max_points,
                "max_offdiag_values": config.kernel_diagnostics.max_offdiag_values,
            },
        }
    selected_mean = mean_config_for_model(config, model_key)
    base = {
        "model_key": model_key,
        "seed": config.seed,
        "dataset": {
            "version": config.data.version,
            "structure_type": config.data.structure_type,
            "n_train": config.data.n_train,
            "n_val": config.data.n_val,
            "n_test": config.data.n_test,
            "policy_grid_size": config.data.policy_grid_size,
        },
        "mean": {
            "method": selected_mean.method,
            "use_treatment_grid": selected_mean.use_treatment_grid,
            "crossfit_folds": selected_mean.crossfit_folds,
            "crossfit_shuffle": selected_mean.crossfit_shuffle,
            "treatment_grid_size": selected_mean.treatment_grid_size,
            "evaluation_grid_size": selected_mean.evaluation_grid_size,
            "hidden_dims": selected_mean.hidden_dims,
            "lr": selected_mean.lr,
            "weight_decay": selected_mean.weight_decay,
            "epochs": selected_mean.epochs,
            "min_epochs": selected_mean.min_epochs,
            "early_stop_patience": selected_mean.early_stop_patience,
            "batch_size": selected_mean.batch_size,
        },
        "kernel_diagnostics": {
            "enabled": config.kernel_diagnostics.enabled,
            "bandwidth_multipliers": config.kernel_diagnostics.bandwidth_multipliers,
            "max_points": config.kernel_diagnostics.max_points,
            "max_offdiag_values": config.kernel_diagnostics.max_offdiag_values,
        },
    }
    if model_key in {"pmmr", "weighted_pmmr", "local_pmmr"}:
        base["bridge"] = {
            "type": "PMMR",
            "ridge": config.pmmr.ridge,
            "jitter": config.pmmr.jitter,
            "awx_scale": config.pmmr.awx_scale,
            "azx_scale": config.pmmr.azx_scale,
        }
    elif model_key in {"kpv", "weighted_kpv", "local_kpv"}:
        base["bridge"] = {
            "type": "KPV",
            "lambda1": config.kpv.lambda1,
            "lambda2": config.kpv.lambda2,
            "optimize_lambda1": config.kpv.optimize_lambda1,
            "optimize_lambda2": config.kpv.optimize_lambda2,
            "lambda1_min": config.kpv.lambda1_min,
            "lambda1_max": config.kpv.lambda1_max,
            "n_lambda1_search": config.kpv.n_lambda1_search,
            "lambda2_min": config.kpv.lambda2_min,
            "lambda2_max": config.kpv.lambda2_max,
            "n_lambda2_search": config.kpv.n_lambda2_search,
            "kernel_variance_scale": config.kpv.kernel_variance_scale,
            "split_ratio": config.kpv.split_ratio,
            "max_train_size": config.kpv.max_train_size,
            "standardize": config.kpv.standardize,
            "jitter": config.kpv.jitter,
        }
    elif model_key in {"dfpv", "weighted_dfpv", "local_dfpv"}:
        base["bridge"] = {
            "type": "DFPV",
            "lam1": config.dfpv.lam1,
            "lam2": config.dfpv.lam2,
            "stage1_iter": config.dfpv.stage1_iter,
            "stage2_iter": config.dfpv.stage2_iter,
            "epochs": config.dfpv.epochs,
            "min_epochs": config.dfpv.min_epochs,
            "early_stop_patience": config.dfpv.early_stop_patience,
            "split_ratio": config.dfpv.split_ratio,
            "max_train_size": config.dfpv.max_train_size,
            "hidden_dims": config.dfpv.hidden_dims,
            "feature_dim": config.dfpv.feature_dim,
            "activation": config.dfpv.activation,
            "final_activation": config.dfpv.final_activation,
            "lr": config.dfpv.lr,
            "weight_decay": config.dfpv.weight_decay,
            "standardize": config.dfpv.standardize,
            "include_covariates": config.dfpv.include_covariates,
            "jitter": config.dfpv.jitter,
            "grad_clip_norm": config.dfpv.grad_clip_norm,
        }
    elif model_key == "dr_kpv":
        base["bridge"] = {
            "type": "DR-KPV",
            "estimand": "conditional",
            "v_variable": "covariates",
            "kpv_lambda1": config.dr_kpv.kpv_lambda1,
            "kpv_lambda2": config.dr_kpv.kpv_lambda2,
            "kap_lambda": config.dr_kpv.kap_lambda,
            "kap_eta": config.dr_kpv.kap_eta,
            "kap_lambda2": config.dr_kpv.kap_lambda2,
            "kap_zeta": config.dr_kpv.kap_zeta,
            "lambda_dr": config.dr_kpv.lambda_dr,
            "optimize_internal": config.dr_kpv.optimize_internal,
            "regularization_grid_points": config.dr_kpv.regularization_grid_points,
            "lambda_min": config.dr_kpv.lambda_min,
            "lambda_max": config.dr_kpv.lambda_max,
            "kernel_variance_scale": config.dr_kpv.kernel_variance_scale,
            "stage1_perc": config.dr_kpv.stage1_perc,
            "max_train_size": config.dr_kpv.max_train_size,
            "standardize": config.dr_kpv.standardize,
            "jitter": config.dr_kpv.jitter,
            "prediction_batch_size": config.dr_kpv.prediction_batch_size,
        }
    elif model_key == "dr_pmmr":
        base["bridge"] = {
            "type": "DR-PMMR",
            "estimand": "marginal",
            "pmmr_lambda": config.dr_pmmr.pmmr_lambda,
            "kap_lambda": config.dr_pmmr.kap_lambda,
            "kap_eta": config.dr_pmmr.kap_eta,
            "kap_lambda2": config.dr_pmmr.kap_lambda2,
            "kap_zeta": config.dr_pmmr.kap_zeta,
            "lambda_dr": config.dr_pmmr.lambda_dr,
            "optimize_internal": config.dr_pmmr.optimize_internal,
            "regularization_grid_points": config.dr_pmmr.regularization_grid_points,
            "lambda_min": config.dr_pmmr.lambda_min,
            "lambda_max": config.dr_pmmr.lambda_max,
            "kernel_variance_scale": config.dr_pmmr.kernel_variance_scale,
            "stage1_perc": config.dr_pmmr.stage1_perc,
            "max_train_size": config.dr_pmmr.max_train_size,
            "standardize": config.dr_pmmr.standardize,
            "jitter": config.dr_pmmr.jitter,
            "prediction_batch_size": config.dr_pmmr.prediction_batch_size,
        }
    elif model_key == "dr_pmmr_conditional":
        base["bridge"] = {
            "type": "DR-PMMR",
            "estimand": "conditional",
            "v_variable": "covariates",
            "outcome_component": "PMMR mean_regressor",
            "pmmr_lambda": config.dr_pmmr_conditional.pmmr_lambda,
            "kap_lambda": config.dr_pmmr_conditional.kap_lambda,
            "kap_eta": config.dr_pmmr_conditional.kap_eta,
            "kap_lambda2": config.dr_pmmr_conditional.kap_lambda2,
            "kap_zeta": config.dr_pmmr_conditional.kap_zeta,
            "lambda_dr": config.dr_pmmr_conditional.lambda_dr,
            "optimize_internal": config.dr_pmmr_conditional.optimize_internal,
            "regularization_grid_points": config.dr_pmmr_conditional.regularization_grid_points,
            "lambda_min": config.dr_pmmr_conditional.lambda_min,
            "lambda_max": config.dr_pmmr_conditional.lambda_max,
            "kernel_variance_scale": config.dr_pmmr_conditional.kernel_variance_scale,
            "stage1_perc": config.dr_pmmr_conditional.stage1_perc,
            "max_train_size": config.dr_pmmr_conditional.max_train_size,
            "standardize": config.dr_pmmr_conditional.standardize,
            "jitter": config.dr_pmmr_conditional.jitter,
            "prediction_batch_size": config.dr_pmmr_conditional.prediction_batch_size,
        }
    elif model_key in {"nmmr", "nmmr_u", "nmmr_v", "weighted_nmmr", "weighted_nmmr_u", "weighted_nmmr_v"}:
        nmmr_cfg = nmmr_config_for_model(config, model_key)
        base["bridge"] = {
            "type": "NMMR",
            "loss_name": nmmr_cfg.loss_name,
            "hidden_dims": nmmr_cfg.hidden_dims,
            "activation": nmmr_cfg.activation,
            "bridge_all_data": nmmr_cfg.bridge_all_data,
            "deactivate_rolloff": nmmr_cfg.deactivate_rolloff,
            "product_kernel": nmmr_cfg.product_kernel,
            "quadratic_treatment": nmmr_cfg.quadratic_treatment,
            "lr": nmmr_cfg.lr,
            "weight_decay": nmmr_cfg.weight_decay,
            "epochs": nmmr_cfg.epochs,
            "min_epochs": nmmr_cfg.min_epochs,
            "early_stop_patience": nmmr_cfg.early_stop_patience,
            "batch_size": nmmr_cfg.batch_size,
            "azx_scale": nmmr_cfg.azx_scale,
            "a_scale": nmmr_cfg.a_scale,
            "z_scale": nmmr_cfg.z_scale,
            "x_scale": nmmr_cfg.x_scale,
            "grad_clip_norm": nmmr_cfg.grad_clip_norm,
        }
    else:
        base["bridge"] = {"type": "direct_or_other"}

    if model_key in {
        "local_kpv",
        "local_dfpv",
        "local_pmmr",
        "weighted_kpv",
        "weighted_dfpv",
        "weighted_pmmr",
        "weighted_nmmr",
        "weighted_nmmr_u",
        "weighted_nmmr_v",
        "naive_oosr",
        "oracle_naive_oosr",
    }:
        weighting_cfg = weighting_config_for_model(config, model_key)
        base["weighting"] = {
            "weight_mode": weighting_cfg.weight_mode,
            "lam": weighting_cfg.lam,
            "tau": weighting_cfg.tau,
            "root_weights": weighting_cfg.root_weights,
            "weight_mean_regressor": weighting_cfg.weight_mean_regressor,
            "warm_start": weighting_cfg.warm_start,
            "truncated_kernel_weights": weighting_cfg.truncated_kernel_weights,
            "n_rounds": weighting_cfg.n_rounds,
            "normalization_mode": weighting_cfg.normalization_mode,
            "normalize_grid_size": weighting_cfg.normalize_grid_size,
            "stabilized": weighting_cfg.stabilized,
            "clip_min": weighting_cfg.clip_min,
            "clip_max": weighting_cfg.clip_max,
            "early_stop_patience": weighting_cfg.early_stop_patience,
            "min_improvement": weighting_cfg.min_improvement,
        }
        base["propensity_flow"] = {
            "bins": config.propensity_flow.bins,
            "transforms": config.propensity_flow.transforms,
            "hidden_features": config.propensity_flow.hidden_features,
            "lr": config.propensity_flow.lr,
            "weight_decay": config.propensity_flow.weight_decay,
            "epochs": config.propensity_flow.epochs,
            "min_epochs": config.propensity_flow.min_epochs,
            "early_stop_patience": config.propensity_flow.early_stop_patience,
            "batch_size": config.propensity_flow.batch_size,
        }
    if model_key in {"s_learner", "oracle_s_learner"}:
        oracle = model_key.startswith("oracle_")
        base["direct_family"] = {
            "requested_two_stage": bool(config.use_mean_slearner),
            "effective_two_stage": _effective_use_mean_slearner(config, oracle=oracle),
        }
    if model_key in {"naive_oosr", "oracle_naive_oosr"}:
        oracle = model_key.startswith("oracle_")
        base["direct_family"] = {
            "requested_two_stage": bool(config.use_mean_oosr),
            "effective_two_stage": _effective_use_mean_oosr(config, oracle=oracle),
        }
    return base


def _pmmr_data(sample: FactualSample) -> PMMRTrainData:
    return PMMRTrainData(
        treatment=sample.treatment,
        treatment_proxy=sample.treatment_proxy,
        outcome_proxy=sample.outcome_proxy,
        covariates=sample.covariates,
        outcome=sample.outcome,
    )


def _kpv_data(sample: FactualSample) -> KPVTrainData:
    return KPVTrainData(
        treatment=sample.treatment,
        treatment_proxy=sample.treatment_proxy,
        outcome_proxy=sample.outcome_proxy,
        covariates=sample.covariates,
        outcome=sample.outcome,
    )


def _dfpv_data(sample: FactualSample) -> DFPVTrainData:
    return DFPVTrainData(
        treatment=sample.treatment,
        treatment_proxy=sample.treatment_proxy,
        outcome_proxy=sample.outcome_proxy,
        covariates=sample.covariates,
        outcome=sample.outcome,
    )


def _dr_kernel_data(sample: FactualSample) -> DRKernelTrainData:
    return DRKernelTrainData(
        treatment=sample.treatment,
        treatment_proxy=sample.treatment_proxy,
        outcome_proxy=sample.outcome_proxy,
        covariates=sample.covariates,
        outcome=sample.outcome,
    )


def _nmmr_data(sample: FactualSample) -> NMMRTrainData:
    return NMMRTrainData(
        treatment=sample.treatment,
        treatment_proxy=sample.treatment_proxy,
        outcome_proxy=sample.outcome_proxy,
        covariates=sample.covariates,
        outcome=sample.outcome,
    )


def _augment_covariates_with_latents(sample: FactualSample) -> np.ndarray:
    return np.concatenate([sample.covariates, sample.latent_uz, sample.latent_uw], axis=1)


def _direct_covariates(sample: FactualSample, *, oracle: bool) -> np.ndarray:
    return _augment_covariates_with_latents(sample) if oracle else sample.covariates


def _effective_use_mean_slearner(config: ExperimentConfig, *, oracle: bool) -> bool:
    return True if oracle else bool(config.use_mean_slearner)


def _effective_use_mean_oosr(config: ExperimentConfig, *, oracle: bool) -> bool:
    return True if oracle else bool(config.use_mean_oosr)


def _evaluation_grid_size_for_selected(
    config: ExperimentConfig,
    model_configs: dict[str, ExperimentConfig],
    selected: list[str],
) -> int:
    sizes: list[int] = []
    for model_key in selected:
        local_config = model_configs.get(model_key, config)
        if model_key == "decaflow":
            sizes.append(int(local_config.data.policy_grid_size))
            continue
        if model_key in BRIDGE_MEAN_MODELS:
            mean_cfg = mean_config_for_model(local_config, model_key)
            if bool(mean_cfg.use_treatment_grid):
                sizes.append(int(mean_cfg.treatment_grid_size))
            else:
                sizes.append(int(mean_cfg.evaluation_grid_size))
        else:
            sizes.append(int(local_config.data.policy_grid_size))
    fallback = int(config.data.policy_grid_size)
    return max(3, max(sizes) if sizes else fallback)


def _oracle_direct_predict(
    model: SLearner,
    treatment: np.ndarray,
    treatment_proxy: np.ndarray,
    outcome_proxy: np.ndarray,
    covariates: np.ndarray,
    latent_uz: np.ndarray,
    latent_uw: np.ndarray,
) -> np.ndarray:
    augmented_covariates = np.concatenate([covariates, latent_uz, latent_uw], axis=1)
    return model.direct_predict(treatment, treatment_proxy, outcome_proxy, augmented_covariates)


def _oracle_conditioned_marginal_curve(
    simulator: Wu2024Simulator,
    sample: FactualSample,
    treatment_grid: np.ndarray,
) -> np.ndarray:
    grid = np.asarray(treatment_grid, dtype=float).reshape(-1)
    curves = []
    for a_val in grid:
        a = np.full(sample.covariates.shape[0], a_val, dtype=float)
        curves.append(
            float(
                np.mean(
                    simulator.oracle_conditioned_outcome(
                        a,
                        sample.covariates,
                        sample.outcome_proxy,
                        sample.latent_uz,
                        sample.latent_uw,
                    )
                )
            )
        )
    return np.asarray(curves, dtype=float)


def _proxy_conditioned_marginal_curve(
    simulator: Wu2024Simulator,
    sample: FactualSample,
    treatment_grid: np.ndarray,
) -> np.ndarray:
    grid = np.asarray(treatment_grid, dtype=float).reshape(-1)
    curves = []
    for a_val in grid:
        a = np.full(sample.covariates.shape[0], a_val, dtype=float)
        curves.append(
            float(
                np.mean(
                    simulator.proxy_conditioned_outcome(
                        a,
                        sample.covariates,
                        sample.outcome_proxy,
                    )
                )
            )
        )
    return np.asarray(curves, dtype=float)


class _MeanPredictProgress:
    def __init__(self, predictor, *, verbose: int, name: str, total: int):
        self.predictor = predictor
        self.verbose = int(verbose)
        self.name = name
        self.total = max(1, int(total))
        self.done = 0
        self.phase = "all"
        self.phase_total = self.total
        self.phase_done = 0

    def set_phase(self, phase: str, phase_total: int) -> None:
        self.phase = phase
        self.phase_total = max(1, int(phase_total))
        self.phase_done = 0
        verbose_print(
            self.verbose,
            3,
            f"[eval] {self.name} mean_predict phase={self.phase} expected_calls={self.phase_total}",
        )

    def __call__(self, treatment: np.ndarray, covariates: np.ndarray) -> np.ndarray:
        self.done += 1
        self.phase_done += 1
        treatment_arr = np.asarray(treatment)
        n = int(treatment_arr.shape[0]) if treatment_arr.ndim > 0 else 1
        start = time.perf_counter()
        verbose_print(
            self.verbose,
            3,
            (
                f"[eval] {self.name} mean_predict {self.done}/{self.total} "
                f"phase={self.phase} {self.phase_done}/{self.phase_total} n={n} start"
            ),
        )
        out = self.predictor(treatment, covariates)
        verbose_print(
            self.verbose,
            3,
            (
                f"[eval] {self.name} mean_predict {self.done}/{self.total} "
                f"phase={self.phase} {self.phase_done}/{self.phase_total} n={n} "
                f"done elapsed={time.perf_counter() - start:.2f}s"
            ),
        )
        return out


def _wrap_mean_predictors_for_figure_progress(
    mean_predictors: dict[str, object],
    *,
    verbose: int,
    n_patients: int,
) -> dict[str, object]:
    if not mean_predictors:
        return mean_predictors
    expected_per_predictor = max(1, 2 * int(n_patients))
    wrapped = {}
    for method, predictor in mean_predictors.items():
        tracker = _MeanPredictProgress(
            predictor,
            verbose=verbose,
            name=f"figures/{method}",
            total=expected_per_predictor,
        )
        tracker.set_phase("patient_surface_figures", expected_per_predictor)
        wrapped[method] = tracker
    return wrapped


def _evaluate_bridge_model(
    model,
    simulator: Wu2024Simulator,
    test_sample: FactualSample,
    reference_curve: np.ndarray | None = None,
    causal_reference: dict[str, np.ndarray] | None = None,
    *,
    verbose: int = 0,
    name: str = "model",
    fast_factual_only: bool = False,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    total_start = time.perf_counter()
    grid_size = len(simulator.policy_grid)
    fast_factual_only = fast_factual_only or bool(
        getattr(getattr(model, "model_config", None), "fast_factual_eval", False)
    )
    if fast_factual_only:
        verbose_print(
            verbose,
            2,
            (
                f"[eval-fast] {name} factual-only start n={test_sample.covariates.shape[0]} "
                "expected_mean_predict_calls=1"
            ),
        )
        step_start = time.perf_counter()
        pred_y = model.mean_predict(test_sample.treatment, test_sample.covariates).reshape(-1)
        y = test_sample.outcome.reshape(-1)
        residual = y - pred_y
        metrics = {
            "factual_mean_rmse": float(np.sqrt(np.mean(np.square(residual)))),
            "factual_mean_mae": float(np.mean(np.abs(residual))),
        }
        policy = np.full(test_sample.covariates.shape[0], np.nan, dtype=float)
        curve = np.full(grid_size, np.nan, dtype=float)
        verbose_print(
            verbose,
            2,
            f"[eval-fast] {name} factual-only done elapsed={time.perf_counter() - step_start:.2f}s",
        )
        return metrics, policy, curve

    expected_mean_predict_calls = 3 * grid_size + 2
    mean_predict = _MeanPredictProgress(
        model.mean_predict,
        verbose=verbose,
        name=name,
        total=expected_mean_predict_calls,
    )
    verbose_print(
        verbose,
        2,
        (
            f"[eval] {name} start n={test_sample.covariates.shape[0]} grid={grid_size} "
            f"expected_mean_predict_calls={expected_mean_predict_calls}"
        ),
    )
    step_start = time.perf_counter()
    verbose_print(verbose, 2, f"[eval] {name} policy prediction start")
    mean_predict.set_phase("policy_grid", grid_size)
    policy = policy_from_mean_predictor(test_sample.covariates, simulator.policy_grid, mean_predict)
    verbose_print(verbose, 2, f"[eval] {name} policy prediction done elapsed={time.perf_counter() - step_start:.2f}s")
    step_start = time.perf_counter()
    verbose_print(verbose, 2, f"[eval] {name} policy metrics start")
    metrics = evaluate_policy_metrics(simulator, test_sample, policy)
    verbose_print(verbose, 2, f"[eval] {name} policy metrics done elapsed={time.perf_counter() - step_start:.2f}s")
    step_start = time.perf_counter()
    verbose_print(verbose, 2, f"[eval] {name} mean surface start")
    mean_predict.set_phase("mean_surface", grid_size + 1)
    metrics.update(
        evaluate_mean_surface(
            simulator,
            test_sample,
            simulator.policy_grid,
            mean_predict,
            causal_reference=causal_reference,
        )
    )
    verbose_print(verbose, 2, f"[eval] {name} mean surface done elapsed={time.perf_counter() - step_start:.2f}s")
    step_start = time.perf_counter()
    verbose_print(verbose, 2, f"[eval] {name} marginal curve start")
    mean_predict.set_phase("marginal_curve_grid", grid_size)
    curve = marginal_curve_from_mean_predictor(test_sample, simulator.policy_grid, mean_predict)
    verbose_print(verbose, 2, f"[eval] {name} marginal curve done elapsed={time.perf_counter() - step_start:.2f}s")
    metrics.update(evaluate_marginal_curve(simulator, test_sample, simulator.policy_grid, curve, reference_curve=reference_curve))
    step_start = time.perf_counter()
    verbose_print(verbose, 2, f"[eval] {name} factual mean prediction start")
    mean_predict.set_phase("factual", 1)
    pred_y = mean_predict(test_sample.treatment, test_sample.covariates).reshape(-1)
    verbose_print(verbose, 2, f"[eval] {name} factual mean prediction done elapsed={time.perf_counter() - step_start:.2f}s")
    y = test_sample.outcome.reshape(-1)
    residual = y - pred_y
    metrics["factual_mean_rmse"] = float(np.sqrt(np.mean(np.square(residual))))
    metrics["factual_mean_mae"] = float(np.mean(np.abs(residual)))
    verbose_print(
        verbose,
        2,
        (
            f"[eval] {name} done mean_predict_calls={mean_predict.done}/{mean_predict.total} "
            f"elapsed={time.perf_counter() - total_start:.2f}s"
        ),
    )
    return metrics, policy, curve


def _evaluate_s_learner(
    model: SLearner,
    simulator: Wu2024Simulator,
    test_sample: FactualSample,
    *,
    oracle: bool = False,
    reference_curve: np.ndarray | None = None,
    causal_reference: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    if oracle:
        values = []
        for a_val in simulator.policy_grid:
            a = np.full((test_sample.covariates.shape[0], 1), a_val)
            values.append(
                _oracle_direct_predict(
                    model,
                    a,
                    test_sample.treatment_proxy,
                    test_sample.outcome_proxy,
                    test_sample.covariates,
                    test_sample.latent_uz,
                    test_sample.latent_uw,
                ).reshape(-1)
            )
        values_arr = np.stack(values, axis=0)
        policy = simulator.policy_grid[np.argmax(values_arr, axis=0)]
        metrics = evaluate_policy_metrics(simulator, test_sample, policy)

        reference = causal_reference or build_causal_surface_reference(
            simulator,
            test_sample,
            simulator.policy_grid,
        )
        grid = np.asarray(reference["treatment_grid"], dtype=float).reshape(-1)
        x = np.asarray(reference["covariates"], dtype=float)
        n = x.shape[0]
        z = test_sample.treatment_proxy[:n]
        w = test_sample.outcome_proxy[:n]
        uz = test_sample.latent_uz[:n]
        uw = test_sample.latent_uw[:n]
        local_treatment = np.asarray(reference["local_treatment"], dtype=float)
        local_true = np.asarray(reference["local_true"], dtype=float).reshape(-1)
        local_pred = _oracle_direct_predict(model, local_treatment, z, w, x, uz, uw).reshape(-1)
        global_pred = []
        for a_val in grid:
            a = np.full((x.shape[0], 1), a_val)
            global_pred.append(_oracle_direct_predict(model, a, z, w, x, uz, uw).reshape(-1))
        global_pred_arr = np.stack(global_pred, axis=0)
        global_true_arr = np.asarray(reference["true_surface"], dtype=float)
        metrics.update(
            {
                "local_rmse": float(np.sqrt(np.mean(np.square(local_pred - local_true)))),
                "global_rmse": float(np.sqrt(np.mean(np.square(global_pred_arr - global_true_arr)))),
            }
        )
        curve = np.asarray([float(np.mean(row)) for row in values_arr], dtype=float)
        metrics.update(
            evaluate_marginal_curve(
                simulator,
                test_sample,
                simulator.policy_grid,
                curve,
                reference_curve=reference_curve,
            )
        )
        return metrics, policy, curve

    policy = policy_from_direct_predictor(
        test_sample.treatment_proxy,
        test_sample.outcome_proxy,
        test_sample.covariates,
        simulator.policy_grid,
        model.direct_predict,
    )
    metrics = evaluate_policy_metrics(simulator, test_sample, policy)
    metrics.update(
        evaluate_direct_surface(
            simulator,
            test_sample,
            simulator.policy_grid,
            model.direct_predict,
            causal_reference=causal_reference,
        )
    )
    curve = marginal_curve_from_direct_predictor(test_sample, simulator.policy_grid, model.direct_predict)
    metrics.update(
        evaluate_marginal_curve(
            simulator,
            test_sample,
            simulator.policy_grid,
            curve,
            reference_curve=reference_curve,
        )
    )
    return metrics, policy, curve


def _evaluate_s_learner_mean(
    model: SLearner,
    simulator: Wu2024Simulator,
    test_sample: FactualSample,
    *,
    reference_curve: np.ndarray | None = None,
    causal_reference: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    policy = policy_from_mean_predictor(test_sample.covariates, simulator.policy_grid, model.mean_predict)
    metrics = evaluate_policy_metrics(simulator, test_sample, policy)
    metrics.update(
        evaluate_mean_surface(
            simulator,
            test_sample,
            simulator.policy_grid,
            model.mean_predict,
            causal_reference=causal_reference,
        )
    )
    curve = marginal_curve_from_mean_predictor(test_sample, simulator.policy_grid, model.mean_predict)
    metrics.update(
        evaluate_marginal_curve(
            simulator,
            test_sample,
            simulator.policy_grid,
            curve,
            reference_curve=reference_curve,
        )
    )
    pred_y = model.mean_predict(test_sample.treatment, test_sample.covariates).reshape(-1)
    y = test_sample.outcome.reshape(-1)
    residual = y - pred_y
    metrics["factual_mean_rmse"] = float(np.sqrt(np.mean(np.square(residual))))
    metrics["factual_mean_mae"] = float(np.mean(np.abs(residual)))
    return metrics, policy, curve


def _fit_slearner_mean_head(
    model: SLearner,
    simulator: Wu2024Simulator,
    fit_sample: FactualSample,
    local_config: ExperimentConfig,
    *,
    oracle: bool,
    sample_weight: np.ndarray | None = None,
    name: str,
) -> None:
    n = fit_sample.covariates.shape[0]
    n_folds = int(local_config.mean.crossfit_folds)
    if n_folds <= 1 or n <= 1:
        direct_covariates = _augment_covariates_with_latents(fit_sample) if oracle else fit_sample.covariates
        model.fit_mean_regressor(
            treatment_grid=simulator.policy_grid,
            covariates_x=fit_sample.covariates,
            treatment_proxy=fit_sample.treatment_proxy,
            outcome_proxy=fit_sample.outcome_proxy,
            direct_covariates=direct_covariates,
            sample_weight=sample_weight,
            verbose=local_config.verbose,
            name=name,
            seed=local_config.seed,
        )
        return

    folds = kfold_indices(
        n,
        n_folds,
        seed=local_config.seed,
        shuffle=bool(local_config.mean.crossfit_shuffle),
    )
    treatment_grid = np.asarray(simulator.policy_grid, dtype=float).reshape(-1)
    a_rows: list[np.ndarray] = []
    x_rows: list[np.ndarray] = []
    pseudo_rows: list[np.ndarray] = []
    weight_rows: list[np.ndarray] = []
    group_rows: list[np.ndarray] = []
    base_weight = None if sample_weight is None else np.asarray(sample_weight, dtype=float).reshape(-1)

    for fold_id, (train_idx, holdout_idx) in enumerate(folds):
        fold_train = _subset_sample(fit_sample, train_idx)
        fold_holdout = _subset_sample(fit_sample, holdout_idx)
        fold_model = SLearner(local_config.s_learner, local_config.mean, simulator.support)
        fold_covariates = _augment_covariates_with_latents(fold_train) if oracle else fold_train.covariates
        fold_sw = None if base_weight is None else base_weight[train_idx]
        fold_model.fit(
            fold_train.treatment,
            fold_train.treatment_proxy,
            fold_train.outcome_proxy,
            fold_covariates,
            fold_train.outcome,
            sample_weight=fold_sw,
            verbose=local_config.verbose,
            name=f"{name}/mean_crossfit_fold_{fold_id}",
            seed=local_config.seed + fold_id,
        )

        holdout_direct_cov = _augment_covariates_with_latents(fold_holdout) if oracle else fold_holdout.covariates
        for a_val in treatment_grid:
            a = np.full((holdout_idx.shape[0], 1), a_val, dtype=float)
            pseudo = fold_model.direct_predict(
                a,
                fold_holdout.treatment_proxy,
                fold_holdout.outcome_proxy,
                holdout_direct_cov,
            ).reshape(-1, 1)
            a_rows.append(a)
            x_rows.append(fold_holdout.covariates)
            pseudo_rows.append(pseudo)
            if base_weight is None:
                weight_rows.append(np.ones(holdout_idx.shape[0], dtype=float))
            else:
                weight_rows.append(base_weight[holdout_idx])
            group_rows.append(holdout_idx.astype(int, copy=False))

    a_train = np.vstack(a_rows)
    x_train = np.vstack(x_rows)
    pseudo_train = np.vstack(pseudo_rows)
    weight_train = np.concatenate(weight_rows)
    group_ids = np.concatenate(group_rows)
    model.fit_mean_regressor_from_pseudo(
        treatment=a_train,
        covariates=x_train,
        pseudo_outcome=pseudo_train,
        sample_weight=weight_train,
        group_ids=group_ids,
        verbose=local_config.verbose,
        name=f"{name}/mean_crossfit",
        seed=local_config.seed,
    )


def _attach_pseudooutcome_metrics(metrics: dict[str, float], model) -> None:
    diag = getattr(model, "mean_pseudooutcome_diagnostics_", None)
    if not diag:
        return
    try:
        metrics["pseudo_outcome_std"] = float(diag.get("std", np.nan))
        metrics["pseudo_outcome_iqr"] = float(diag.get("iqr", np.nan))
    except (TypeError, ValueError):
        return


def _observed_bridge_rmse(model, sample: FactualSample) -> float:
    try:
        pred = model.bridge_predict(sample.treatment, sample.outcome_proxy, sample.covariates).reshape(-1)
    except RuntimeError:
        return float("nan")
    truth = sample.outcome.reshape(-1)
    return float(np.sqrt(np.mean(np.square(pred - truth))))


def _observed_bridge_rmse_logged(model, sample: FactualSample, *, verbose: int, name: str, split: str) -> float:
    step_start = time.perf_counter()
    verbose_print(verbose, 2, f"[eval] {name} {split} observed bridge RMSE start")
    value = _observed_bridge_rmse(model, sample)
    verbose_print(
        verbose,
        2,
        f"[eval] {name} {split} observed bridge RMSE done elapsed={time.perf_counter() - step_start:.2f}s",
    )
    return value


def _observed_direct_rmse(model: SLearner, sample: FactualSample, covariates: np.ndarray) -> float:
    pred = model.direct_predict(sample.treatment, sample.treatment_proxy, sample.outcome_proxy, covariates).reshape(-1)
    truth = sample.outcome.reshape(-1)
    return float(np.sqrt(np.mean(np.square(pred - truth))))


def _attach_factual_rmse_from_observed(metrics: dict[str, float]) -> None:
    if "factual_mean_rmse" not in metrics and "observed_rmse" in metrics:
        metrics["factual_mean_rmse"] = float(metrics["observed_rmse"])


def _round_summary(round_id: int, val_rmse: float, weights: np.ndarray) -> dict[str, float]:
    return {
        "round": round_id,
        "val_factual_rmse": float(val_rmse),
        "weight_ess": effective_sample_size(weights),
        "weight_min": float(np.min(weights)),
        "weight_max": float(np.max(weights)),
    }


def _validated_weight_mode(model_name: str, weight_mode: str) -> str:
    normalized = str(weight_mode).strip().lower()
    valid_modes = {"decision_aware", "local_only"}
    if normalized not in valid_modes:
        raise ValueError(f"Unsupported weight_mode={weight_mode!r} for {model_name}. Expected one of {sorted(valid_modes)}.")
    return normalized


def _updated_round_weights(
    *,
    treatment: np.ndarray,
    pseudo_optimal_treatment: np.ndarray,
    conditional_density: np.ndarray,
    base_weights: np.ndarray,
    simulator: Wu2024Simulator,
    weighting_cfg,
    weight_mode: str,
) -> np.ndarray:
    if weight_mode == "local_only":
        weights = build_local_only_weights(
            treatment=treatment,
            pseudo_optimal_treatment=pseudo_optimal_treatment,
            conditional_density=conditional_density,
            tau=weighting_cfg.tau,
            truncated_kernel=bool(weighting_cfg.truncated_kernel_weights),
        )
    else:
        local = local_oosr_factor(
            treatment=treatment,
            pseudo_optimal_treatment=pseudo_optimal_treatment,
            tau=weighting_cfg.tau,
            support=simulator.support,
            normalization_mode=weighting_cfg.normalization_mode,
            grid_size=weighting_cfg.normalize_grid_size,
            truncated_kernel=bool(weighting_cfg.truncated_kernel_weights),
        )
        weights = base_weights * (1.0 + weighting_cfg.lam * local)
    return finalize_weights(weights, weighting_cfg.clip_min, weighting_cfg.clip_max)


def _fit_weighted_proximal(
    model_name: str,
    simulator: Wu2024Simulator,
    fit_sample: FactualSample,
    val_sample: FactualSample,
    config: ExperimentConfig,
):
    display_names = {
        "local_kpv": "Local KPV",
        "local_dfpv": "Local DFPV",
        "local_pmmr": "Local PMMR",
        "weighted_kpv": "Weighted KPV",
        "weighted_dfpv": "Weighted DFPV",
        "weighted_pmmr": "Weighted PMMR",
        "weighted_nmmr": "Weighted NMMR",
        "weighted_nmmr_u": "Weighted NMMR-U",
        "weighted_nmmr_v": "Weighted NMMR-V",
    }
    display_name = display_names.get(model_name, str(model_name))
    mean_cfg = mean_config_for_model(config, model_name)
    weighting_cfg = weighting_config_for_model(config, model_name)
    weight_mode = _validated_weight_mode(model_name, weighting_cfg.weight_mode)
    if model_name in {"weighted_kpv", "local_kpv"}:
        model_factory = lambda: KPVBridge(config.kpv, mean_cfg, config.w_flow, simulator.support)
        to_data = _kpv_data
    elif model_name in {"weighted_dfpv", "local_dfpv"}:
        model_factory = lambda: DFPVBridge(config.dfpv, mean_cfg, config.w_flow, simulator.support)
        to_data = _dfpv_data
    elif model_name in {"weighted_pmmr", "local_pmmr"}:
        model_factory = lambda: PMMRBridge(config.pmmr, mean_cfg, config.w_flow, simulator.support)
        to_data = _pmmr_data
    elif model_name in {"weighted_nmmr", "weighted_nmmr_u", "weighted_nmmr_v"}:
        model_factory = lambda: NMMRBridge(nmmr_config_for_model(config, model_name), mean_cfg, config.w_flow, simulator.support)
        to_data = _nmmr_data
    else:
        raise ValueError(model_name)

    training_history: list[dict[str, object]] = []
    kernel_diagnostics_rows: list[dict[str, object]] = []
    kernel_offdiag_rows: list[dict[str, object]] = []
    kernel_spectrum_rows: list[dict[str, object]] = []
    verbose_print(config.verbose, 1, f"[start] {display_name} propensity flow")
    propensity = ConditionalFlowDensity(
        features=1,
        context_dim=fit_sample.covariates.shape[1],
        config=config.propensity_flow,
        support=simulator.support,
    )
    propensity_result = propensity.fit(
        fit_sample.treatment,
        fit_sample.covariates,
        verbose=config.verbose,
        name=f"{display_name}/propensity_flow",
        seed=config.seed,
    )
    training_history.append(
        {
            "stage": f"{display_name}/propensity_flow",
            "train_losses": propensity_result["train_losses"],
            "val_losses": propensity_result["val_losses"],
            "best_epoch": propensity_result["best_epoch"],
            "stopped_epoch": propensity_result["stopped_epoch"],
        }
    )
    density = np.clip(propensity.pdf(fit_sample.treatment, fit_sample.covariates), 1e-8, None)
    base_weights = build_base_density_weights(
        treatment=fit_sample.treatment,
        conditional_density=density,
        support=simulator.support,
        stabilized=weighting_cfg.stabilized,
    )
    base_weights = finalize_weights(base_weights, weighting_cfg.clip_min, weighting_cfg.clip_max)

    def _fit_model(model, *, sample_weight: np.ndarray | None, stage_name: str, stage_seed: int, warm_started: bool) -> None:
        if model_name in {"weighted_dfpv", "local_dfpv"}:
            model.fit(
                to_data(fit_sample),
                sample_weight=sample_weight,
                mean_method=mean_cfg.method,
                verbose=config.verbose,
                name=stage_name,
                seed=stage_seed,
            )
            return
        model.fit(
            to_data(fit_sample),
            sample_weight=sample_weight,
            mean_method=mean_cfg.method,
            weight_mean_regressor=bool(weighting_cfg.weight_mean_regressor),
            warm_start=warm_started,
            verbose=config.verbose,
            name=stage_name,
            seed=stage_seed,
        )

    history = []
    best_model = None
    best_metrics = None
    best_policy = None
    best_diagnostics = None
    best_val_rmse = np.inf
    rounds_without_improvement = 0
    current_weights = base_weights
    previous_model = None
    bootstrap_reference_policy = None

    if weight_mode == "local_only":
        verbose_print(config.verbose, 1, f"[start] {display_name} bootstrap unweighted policy")
        bootstrap_model = model_factory()
        _fit_model(
            bootstrap_model,
            sample_weight=None,
            stage_name=f"{display_name}/bootstrap_unweighted",
            stage_seed=config.seed,
            warm_started=False,
        )
        training_history.extend(bootstrap_model.training_history)
        bootstrap_policy = policy_from_mean_predictor(
            fit_sample.covariates,
            simulator.policy_grid,
            bootstrap_model.mean_predict,
        )
        bootstrap_reference_policy = bootstrap_policy[:, None].copy()
        current_weights = _updated_round_weights(
            treatment=fit_sample.treatment,
            pseudo_optimal_treatment=bootstrap_reference_policy,
            conditional_density=density,
            base_weights=base_weights,
            simulator=simulator,
            weighting_cfg=weighting_cfg,
            weight_mode=weight_mode,
        )
        previous_model = copy.deepcopy(bootstrap_model) if bool(weighting_cfg.warm_start) else None
        verbose_print(config.verbose, 1, f"[done] {display_name} bootstrap unweighted policy")

    effective_rounds = max(1, int(weighting_cfg.n_rounds))

    for round_id in range(effective_rounds):
        verbose_print(config.verbose, 1, f"[start] {display_name} round {round_id + 1}/{effective_rounds}")
        warm_started = bool(weighting_cfg.warm_start) and previous_model is not None
        if warm_started:
            model = copy.deepcopy(previous_model)
            verbose_print(config.verbose, 1, f"[warm-start] {display_name} round {round_id + 1}")
        else:
            model = model_factory()
        fit_weights_for_bridge = np.sqrt(current_weights) if bool(weighting_cfg.root_weights) else current_weights
        _fit_model(
            model,
            sample_weight=fit_weights_for_bridge,
            stage_name=f"{display_name}/round_{round_id}",
            stage_seed=config.seed + round_id,
            warm_started=warm_started,
        )
        if config.kernel_diagnostics.enabled and hasattr(model, "kernel_diagnostics"):
            kernel_result = model.kernel_diagnostics(
                config=config.kernel_diagnostics,
                model_name=display_name,
                stage=f"{display_name}/round_{round_id}/bridge",
                round_id=round_id,
                sample_weight=fit_weights_for_bridge,
            )
            kernel_diagnostics_rows.extend(kernel_result.rows)
            kernel_offdiag_rows.extend(kernel_result.offdiag_rows)
            kernel_spectrum_rows.extend(kernel_result.spectrum_rows)
        training_history.extend(model.training_history)
        fit_policy = policy_from_mean_predictor(fit_sample.covariates, simulator.policy_grid, model.mean_predict)
        val_policy = policy_from_mean_predictor(val_sample.covariates, simulator.policy_grid, model.mean_predict)
        val_predictions = model.mean_predict(val_sample.treatment, val_sample.covariates).reshape(-1)
        val_outcomes = val_sample.outcome.reshape(-1)
        val_rmse = float(np.sqrt(np.mean(np.square(val_predictions - val_outcomes))))
        history.append(_round_summary(round_id, val_rmse, current_weights))
        verbose_print(config.verbose, 1, f"[done] {display_name} round {round_id + 1} val_factual_rmse={val_rmse:.4f}")

        if np.isfinite(val_rmse) and val_rmse < best_val_rmse - weighting_cfg.min_improvement:
            best_val_rmse = val_rmse
            best_model = copy.deepcopy(model)
            best_policy = val_policy.copy()
            best_metrics = history[-1].copy()
            best_diagnostics = {
                "round": np.array([round_id], dtype=int),
                "fit_weights": current_weights.copy(),
                "bridge_fit_weights": fit_weights_for_bridge.copy(),
                "base_weights": base_weights.copy(),
                "conditional_density": density.copy(),
                "reference_policy": fit_policy[:, None].copy(),
                "weight_mode": np.array([weight_mode]),
                "weight_mean_regressor": np.array([bool(weighting_cfg.weight_mean_regressor)]),
                "warm_start": np.array([bool(weighting_cfg.warm_start)]),
                "truncated_kernel_weights": np.array([bool(weighting_cfg.truncated_kernel_weights)]),
                "treatment": fit_sample.treatment.copy(),
                "optimal_treatment": fit_sample.optimal_treatment.copy(),
                "assignment_mean": fit_sample.assignment_mean.copy(),
            }
            if bootstrap_reference_policy is not None:
                best_diagnostics["bootstrap_reference_policy"] = bootstrap_reference_policy.copy()
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += 1
            if rounds_without_improvement >= weighting_cfg.early_stop_patience:
                break

        previous_model = copy.deepcopy(model) if bool(weighting_cfg.warm_start) else None
        current_weights = _updated_round_weights(
            treatment=fit_sample.treatment,
            pseudo_optimal_treatment=fit_policy[:, None],
            conditional_density=density,
            base_weights=base_weights,
            simulator=simulator,
            weighting_cfg=weighting_cfg,
            weight_mode=weight_mode,
        )

    if best_model is None:
        raise RuntimeError(f"{display_name} produced no finite validation factual RMSE.")
    if best_policy is None:
        best_policy = policy_from_mean_predictor(val_sample.covariates, simulator.policy_grid, best_model.mean_predict)
    if best_metrics is None:
        best_metrics = history[-1].copy() if history else _round_summary(0, float("nan"), current_weights)
    if best_diagnostics is None:
        best_diagnostics = {
            "round": np.array([history[-1]["round"] if history else 0], dtype=int),
            "fit_weights": current_weights.copy(),
            "bridge_fit_weights": current_weights.copy(),
            "base_weights": base_weights.copy(),
            "reference_policy": policy_from_mean_predictor(
                fit_sample.covariates,
                simulator.policy_grid,
                best_model.mean_predict,
            )[:, None],
            "weight_mean_regressor": np.array([bool(weighting_cfg.weight_mean_regressor)]),
            "warm_start": np.array([bool(weighting_cfg.warm_start)]),
            "truncated_kernel_weights": np.array([bool(weighting_cfg.truncated_kernel_weights)]),
            "treatment": fit_sample.treatment.copy(),
            "optimal_treatment": fit_sample.optimal_treatment.copy(),
            "assignment_mean": fit_sample.assignment_mean.copy(),
        }

    return (
        best_model,
        history,
        best_metrics,
        best_policy,
        best_diagnostics,
        training_history,
        kernel_diagnostics_rows,
        kernel_offdiag_rows,
        kernel_spectrum_rows,
    )


def _fit_naive_oosr(
    simulator: Wu2024Simulator,
    fit_sample: FactualSample,
    val_sample: FactualSample,
    config: ExperimentConfig,
    *,
    oracle: bool = False,
    include_proxies: bool = True,
):
    model_key = "oracle_naive_oosr" if oracle else "naive_oosr"
    display_name = "Oracle Naive OOSR" if oracle else "Naive OOSR"
    fit_covariates = _direct_covariates(fit_sample, oracle=oracle)
    val_covariates = _direct_covariates(val_sample, oracle=oracle)
    if include_proxies:
        fit_obs_covariates = np.concatenate([fit_covariates, fit_sample.treatment_proxy, fit_sample.outcome_proxy], axis=1)
    else:
        fit_obs_covariates = fit_covariates
    training_history: list[dict[str, object]] = []
    weighting_cfg = weighting_config_for_model(config, model_key)
    propensity = ConditionalFlowDensity(
        features=1,
        context_dim=fit_obs_covariates.shape[1],
        config=config.propensity_flow,
        support=simulator.support,
    )
    propensity_result = propensity.fit(
        fit_sample.treatment,
        fit_obs_covariates,
        verbose=config.verbose,
        name=f"{display_name}/propensity_flow",
        seed=config.seed,
    )
    training_history.append(
        {
            "stage": f"{display_name}/propensity_flow",
            "train_losses": propensity_result["train_losses"],
            "val_losses": propensity_result["val_losses"],
            "best_epoch": propensity_result["best_epoch"],
            "stopped_epoch": propensity_result["stopped_epoch"],
        }
    )
    density = propensity.pdf(fit_sample.treatment, fit_obs_covariates)
    base_weights = build_base_density_weights(
        treatment=fit_sample.treatment,
        conditional_density=density,
        support=simulator.support,
        stabilized=weighting_cfg.stabilized,
    )
    base_weights = finalize_weights(base_weights, weighting_cfg.clip_min, weighting_cfg.clip_max)

    history = []
    best_model = None
    best_diagnostics = None
    best_val_rmse = np.inf
    rounds_without_improvement = 0
    current_weights = base_weights
    previous_model = None
    effective_rounds = max(1, int(weighting_cfg.n_rounds))
    for round_id in range(effective_rounds):
        verbose_print(config.verbose, 1, f"[start] {display_name} round {round_id + 1}/{effective_rounds}")
        warm_started = bool(weighting_cfg.warm_start) and previous_model is not None
        if warm_started:
            model = copy.deepcopy(previous_model)
        else:
            model = SLearner(config.s_learner, config.mean, simulator.support)
        model.fit(
            fit_sample.treatment,
            fit_sample.treatment_proxy,
            fit_sample.outcome_proxy,
            fit_covariates,
            fit_sample.outcome,
            sample_weight=current_weights,
            verbose=config.verbose,
            name=f"{display_name}/round_{round_id}",
            seed=config.seed + round_id,
            include_proxies=include_proxies,
        )
        training_history.extend(model.training_history)
        direct_rho = policy_from_direct_predictor(
            fit_sample.treatment_proxy,
            fit_sample.outcome_proxy,
            fit_covariates,
            simulator.policy_grid,
            model.direct_predict,
        )
        val_rmse = _observed_direct_rmse(model, val_sample, val_covariates)
        history.append(_round_summary(round_id, val_rmse, current_weights))
        verbose_print(config.verbose, 1, f"[done] {display_name} round {round_id + 1} val_factual_rmse={val_rmse:.4f}")
        if np.isfinite(val_rmse) and val_rmse < best_val_rmse - weighting_cfg.min_improvement:
            best_val_rmse = val_rmse
            best_model = copy.deepcopy(model)
            best_diagnostics = {
                "round": np.array([round_id], dtype=int),
                "fit_weights": current_weights.copy(),
                "base_weights": base_weights.copy(),
                "reference_policy": direct_rho[:, None].copy(),
                "warm_start": np.array([bool(weighting_cfg.warm_start)]),
                "truncated_kernel_weights": np.array([bool(weighting_cfg.truncated_kernel_weights)]),
                "treatment": fit_sample.treatment.copy(),
                "optimal_treatment": fit_sample.optimal_treatment.copy(),
                "assignment_mean": fit_sample.assignment_mean.copy(),
            }
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += 1
            if rounds_without_improvement >= weighting_cfg.early_stop_patience:
                break
        previous_model = copy.deepcopy(model) if bool(weighting_cfg.warm_start) else None
        local = local_oosr_factor(
            treatment=fit_sample.treatment,
            pseudo_optimal_treatment=direct_rho[:, None],
            tau=weighting_cfg.tau,
            support=simulator.support,
            normalization_mode=weighting_cfg.normalization_mode,
            grid_size=weighting_cfg.normalize_grid_size,
            truncated_kernel=bool(weighting_cfg.truncated_kernel_weights),
        )
        current_weights = finalize_weights(
            base_weights * (1.0 + weighting_cfg.lam * local),
            weighting_cfg.clip_min,
            weighting_cfg.clip_max,
        )

    if best_model is None:
        raise RuntimeError(f"{display_name} produced no finite validation factual RMSE.")
    if best_diagnostics is None:
        best_diagnostics = {
            "round": np.array([history[-1]["round"] if history else 0], dtype=int),
            "fit_weights": current_weights.copy(),
            "base_weights": base_weights.copy(),
            "warm_start": np.array([bool(weighting_cfg.warm_start)]),
            "truncated_kernel_weights": np.array([bool(weighting_cfg.truncated_kernel_weights)]),
            "treatment": fit_sample.treatment.copy(),
            "optimal_treatment": fit_sample.optimal_treatment.copy(),
            "assignment_mean": fit_sample.assignment_mean.copy(),
        }
    return best_model, history, best_diagnostics, training_history


def run_experiment(
    config: ExperimentConfig,
    *,
    selected_models: list[str] | tuple[str, ...] | None = None,
    model_configs: dict[str, ExperimentConfig] | None = None,
    results_export: ResultsExportConfig | None = None,
    model_progress: bool = False,
) -> dict:
    selected = list(selected_models or ALL_MODEL_KEYS)
    unknown = sorted(set(selected) - set(VALID_MODEL_KEYS))
    if unknown:
        raise ValueError(f"Unknown model keys requested: {unknown}")
    model_configs = model_configs or {}
    config.data.policy_grid_size = _evaluation_grid_size_for_selected(config, model_configs, selected)

    set_seed(config.seed)
    output_dir = ensure_dir(Path(config.output_dir)) if config.save_outputs else Path(config.output_dir)
    verbose_print(config.verbose, 1, f"[start] experiment seed={config.seed} dataset={config.data.version}")
    simulator = make_simulator(config.data)
    explicit_n_val = int(getattr(config.data, "n_val", 0) or 0)
    if explicit_n_val > 0:
        train_full = simulator.sample_factual(config.data.n_train + explicit_n_val, config.seed)
        fit_sample = _subset_sample(train_full, np.arange(config.data.n_train))
        val_sample = _subset_sample(train_full, np.arange(config.data.n_train, config.data.n_train + explicit_n_val))
        train_sample = train_full
        test_sample = simulator.sample_factual(config.data.n_test, config.seed + 1)
    else:
        train_sample, test_sample = simulator.sample_train_test(config.seed)
        fit_idx, val_idx = train_val_split(train_sample.treatment.shape[0], val_fraction=0.2, seed=config.seed)
        fit_sample = _subset_sample(train_sample, fit_idx)
        val_sample = _subset_sample(train_sample, val_idx)

    if config.verbose >= 3:
        for model_key in selected:
            local_config = model_configs.get(model_key, config)
            summary = _model_hparam_summary(model_key, local_config)
            verbose_print(
                config.verbose,
                3,
                "[hparams] " + json.dumps(summary, indent=2, default=str),
            )

    results: list[dict] = []
    validation_results: list[dict] = []
    artifacts = {
        "config": config_to_dict(config),
        "per_model_configs": {key: config_to_dict(value) for key, value in model_configs.items()},
        "selected_models": selected,
        "data_overlap_rate": simulator.overlap_rate(train_sample),
    }
    saved_policies: dict[str, np.ndarray] = {}
    mean_outcome_curves: dict[str, np.ndarray] = {}
    mean_outcome_curves_direct: dict[str, np.ndarray] = {}
    mean_predictors = {}
    direct_predictors = {}
    oracle_direct_predictors = {}
    weight_diagnostics: dict[str, dict[str, np.ndarray]] = {}
    training_histories: dict[str, list[dict[str, object]]] = {}
    weighted_histories: dict[str, list[dict[str, object]]] = {}
    bridge_models: dict[str, object] = {}
    kernel_diagnostic_rows: list[dict[str, object]] = []
    kernel_offdiag_rows: list[dict[str, object]] = []
    kernel_spectrum_rows: list[dict[str, object]] = []
    if config.fast_factual_eval:
        verbose_print(config.verbose, 1, "[eval-fast] enabled: skipping policy/surface/curve metrics for trial evaluation")
        test_oracle_curve = None
        val_oracle_curve = None
    else:
        test_oracle_curve = _oracle_conditioned_marginal_curve(simulator, test_sample, simulator.policy_grid)
        val_oracle_curve = _oracle_conditioned_marginal_curve(simulator, val_sample, simulator.policy_grid)

    total_models = len(selected)
    for model_index, model_key in enumerate(selected, start=1):
        local_config = model_configs.get(model_key, config)
        if local_config.fast_factual_eval:
            for model_cfg in (
                local_config.pmmr,
                local_config.kpv,
                local_config.dfpv,
                local_config.dr_kpv,
                local_config.dr_pmmr,
                local_config.dr_pmmr_conditional,
                local_config.nmmr,
                local_config.nmmr_u,
                local_config.nmmr_v,
            ):
                setattr(model_cfg, "fast_factual_eval", True)
        display_name = MODEL_DISPLAY_NAMES[model_key]
        policy_key = MODEL_POLICY_KEYS[model_key]
        bridge_model_for_direct = None
        model_start = time.perf_counter()
        if model_progress:
            print(
                (
                    f"[model-start] seed={config.seed} {model_index}/{total_models} "
                    f"model={display_name} key={model_key} output_dir={output_dir}"
                ),
                flush=True,
            )

        if model_key == "pmmr":
            model = PMMRBridge(local_config.pmmr, mean_config_for_model(local_config, model_key), local_config.w_flow, simulator.support)
            model.fit(
                _pmmr_data(fit_sample),
                mean_method=mean_config_for_model(local_config, model_key).method,
                verbose=local_config.verbose,
                name=display_name,
                seed=local_config.seed,
            )
            if local_config.kernel_diagnostics.enabled:
                kernel_result = model.kernel_diagnostics(
                    config=local_config.kernel_diagnostics,
                    model_name=display_name,
                    stage=f"{display_name}/bridge",
                )

                kernel_diagnostic_rows.extend(kernel_result.rows)
                kernel_offdiag_rows.extend(kernel_result.offdiag_rows)
                kernel_spectrum_rows.extend(kernel_result.spectrum_rows)
            test_metrics, test_policy, test_curve = _evaluate_bridge_model(
                model,
                simulator,
                test_sample,
                reference_curve=test_oracle_curve,
            )
            val_metrics, _, _ = _evaluate_bridge_model(
                model,
                simulator,
                val_sample,
                reference_curve=val_oracle_curve,
            )
            test_metrics["observed_rmse"] = _observed_bridge_rmse(model, test_sample)
            val_metrics["observed_rmse"] = _observed_bridge_rmse(model, val_sample)
            _attach_pseudooutcome_metrics(test_metrics, model)
            _attach_pseudooutcome_metrics(val_metrics, model)
            training_histories["pmmr_losses"] = model.training_history
            mean_predictors[policy_key] = model.mean_predict
            bridge_models[policy_key] = model
            bridge_model_for_direct = model
        elif model_key == "kpv":
            model = KPVBridge(local_config.kpv, mean_config_for_model(local_config, model_key), local_config.w_flow, simulator.support)
            model.fit(
                _kpv_data(fit_sample),
                mean_method=mean_config_for_model(local_config, model_key).method,
                verbose=local_config.verbose,
                name=display_name,
                seed=local_config.seed,
            )
            if local_config.kernel_diagnostics.enabled:
                kernel_result = model.kernel_diagnostics(
                    config=local_config.kernel_diagnostics,
                    model_name=display_name,
                    stage=f"{display_name}/bridge",
                )

                kernel_diagnostic_rows.extend(kernel_result.rows)
                kernel_offdiag_rows.extend(kernel_result.offdiag_rows)
                kernel_spectrum_rows.extend(kernel_result.spectrum_rows)
            test_metrics, test_policy, test_curve = _evaluate_bridge_model(
                model,
                simulator,
                test_sample,
                reference_curve=test_oracle_curve,
            )
            val_metrics, _, _ = _evaluate_bridge_model(
                model,
                simulator,
                val_sample,
                reference_curve=val_oracle_curve,
            )
            test_metrics["observed_rmse"] = _observed_bridge_rmse(model, test_sample)
            val_metrics["observed_rmse"] = _observed_bridge_rmse(model, val_sample)
            _attach_pseudooutcome_metrics(test_metrics, model)
            _attach_pseudooutcome_metrics(val_metrics, model)
            test_metrics["kpv_lambda1"] = float(model.lambda1_) if model.lambda1_ is not None else float("nan")
            test_metrics["kpv_lambda2"] = float(model.lambda2_) if model.lambda2_ is not None else float("nan")
            val_metrics["kpv_lambda1"] = test_metrics["kpv_lambda1"]
            val_metrics["kpv_lambda2"] = test_metrics["kpv_lambda2"]
            training_histories["kpv_losses"] = model.training_history
            mean_predictors[policy_key] = model.mean_predict
            bridge_models[policy_key] = model
            bridge_model_for_direct = model
        elif model_key == "dfpv":
            model = DFPVBridge(local_config.dfpv, mean_config_for_model(local_config, model_key), local_config.w_flow, simulator.support)
            model.fit(
                _dfpv_data(fit_sample),
                mean_method=mean_config_for_model(local_config, model_key).method,
                verbose=local_config.verbose,
                name=display_name,
                seed=local_config.seed,
            )
            test_metrics, test_policy, test_curve = _evaluate_bridge_model(
                model,
                simulator,
                test_sample,
                reference_curve=test_oracle_curve,
            )
            val_metrics, _, _ = _evaluate_bridge_model(
                model,
                simulator,
                val_sample,
                reference_curve=val_oracle_curve,
            )
            test_metrics["observed_rmse"] = _observed_bridge_rmse(model, test_sample)
            val_metrics["observed_rmse"] = _observed_bridge_rmse(model, val_sample)
            _attach_pseudooutcome_metrics(test_metrics, model)
            _attach_pseudooutcome_metrics(val_metrics, model)
            training_histories["dfpv_losses"] = model.training_history
            mean_predictors[policy_key] = model.mean_predict
            bridge_models[policy_key] = model
            bridge_model_for_direct = model
        elif model_key == "dr_kpv":
            model = DRKPVBridge(local_config.dr_kpv, mean_config_for_model(local_config, model_key), simulator.support)
            model.fit(
                _dr_kernel_data(fit_sample),
                verbose=local_config.verbose,
                name=display_name,
                seed=local_config.seed,
            )
            if local_config.kernel_diagnostics.enabled:
                kernel_result = model.kernel_diagnostics(
                    config=local_config.kernel_diagnostics,
                    model_name=display_name,
                    stage=f"{display_name}/bridge",
                )
                kernel_diagnostic_rows.extend(kernel_result.rows)
                kernel_offdiag_rows.extend(kernel_result.offdiag_rows)
                kernel_spectrum_rows.extend(kernel_result.spectrum_rows)
            test_metrics, test_policy, test_curve = _evaluate_bridge_model(
                model,
                simulator,
                test_sample,
                reference_curve=test_oracle_curve,
                verbose=local_config.verbose,
                name=f"{display_name}/test",
                fast_factual_only=local_config.fast_factual_eval,
            )
            val_metrics, _, val_curve = _evaluate_bridge_model(
                model,
                simulator,
                val_sample,
                reference_curve=val_oracle_curve,
                verbose=local_config.verbose,
                name=f"{display_name}/validation",
                fast_factual_only=local_config.fast_factual_eval,
            )
            if local_config.fast_factual_eval:
                test_metrics["observed_rmse"] = float("nan")
                val_metrics["observed_rmse"] = float("nan")
            else:
                test_metrics["observed_rmse"] = _observed_bridge_rmse_logged(
                    model,
                    test_sample,
                    verbose=local_config.verbose,
                    name=display_name,
                    split="test",
                )
                val_metrics["observed_rmse"] = _observed_bridge_rmse_logged(
                    model,
                    val_sample,
                    verbose=local_config.verbose,
                    name=display_name,
                    split="validation",
                )
            test_metrics["dr_kpv_lambda1"] = float(model.outcome_bridge.lambda1)
            test_metrics["dr_kpv_lambda2"] = float(model.outcome_bridge.lambda2)
            test_metrics["dr_kap_lambda"] = float(model.treatment_bridge.lambda_)
            test_metrics["dr_kap_eta"] = float(model.treatment_bridge.eta)
            test_metrics["dr_kap_lambda2"] = float(model.treatment_bridge.lambda2)
            test_metrics["dr_kap_zeta"] = float(model.treatment_bridge.zeta)
            test_metrics["dr_lambda_dr"] = float(model.lambda_dr_)
            for key in (
                "dr_kpv_lambda1",
                "dr_kpv_lambda2",
                "dr_kap_lambda",
                "dr_kap_eta",
                "dr_kap_lambda2",
                "dr_kap_zeta",
                "dr_lambda_dr",
            ):
                val_metrics[key] = test_metrics[key]
            training_histories["dr_kpv_losses"] = model.training_history
            mean_predictors[policy_key] = model.mean_predict
            bridge_models[policy_key] = model
            bridge_model_for_direct = model
        elif model_key == "dr_pmmr":
            model = DRPMMRMarginalBridge(local_config.dr_pmmr, mean_config_for_model(local_config, model_key), simulator.support)
            model.fit(
                _dr_kernel_data(fit_sample),
                verbose=local_config.verbose,
                name=display_name,
                seed=local_config.seed,
            )
            test_metrics, test_policy, test_curve = _evaluate_bridge_model(
                model,
                simulator,
                test_sample,
                reference_curve=test_oracle_curve,
                verbose=local_config.verbose,
                name=f"{display_name}/test",
                fast_factual_only=local_config.fast_factual_eval,
            )
            val_metrics, _, val_curve = _evaluate_bridge_model(
                model,
                simulator,
                val_sample,
                reference_curve=val_oracle_curve,
                verbose=local_config.verbose,
                name=f"{display_name}/validation",
                fast_factual_only=local_config.fast_factual_eval,
            )
            if local_config.fast_factual_eval:
                test_metrics["observed_rmse"] = float("nan")
                val_metrics["observed_rmse"] = float("nan")
            else:
                test_metrics["observed_rmse"] = _observed_bridge_rmse_logged(
                    model,
                    test_sample,
                    verbose=local_config.verbose,
                    name=display_name,
                    split="test",
                )
                val_metrics["observed_rmse"] = _observed_bridge_rmse_logged(
                    model,
                    val_sample,
                    verbose=local_config.verbose,
                    name=display_name,
                    split="validation",
                )
            test_metrics["dr_pmmr_lambda"] = float(local_config.dr_pmmr.pmmr_lambda)
            test_metrics["dr_kap_lambda"] = float(model.treatment_bridge.lambda_)
            test_metrics["dr_kap_eta"] = float(model.treatment_bridge.eta)
            test_metrics["dr_kap_lambda2"] = float(model.treatment_bridge.lambda2)
            test_metrics["dr_kap_zeta"] = float(model.treatment_bridge.zeta)
            test_metrics["dr_lambda_dr"] = float(model.lambda_dr_)
            for key in (
                "dr_pmmr_lambda",
                "dr_kap_lambda",
                "dr_kap_eta",
                "dr_kap_lambda2",
                "dr_kap_zeta",
                "dr_lambda_dr",
            ):
                val_metrics[key] = test_metrics[key]
            training_histories["dr_pmmr_losses"] = model.training_history
            mean_predictors[policy_key] = model.mean_predict
            bridge_models[policy_key] = model
            bridge_model_for_direct = model
        elif model_key == "dr_pmmr_conditional":
            model = DRPMMRConditionalBridge(
                local_config.dr_pmmr_conditional,
                mean_config_for_model(local_config, model_key),
                simulator.support,
            )
            model.fit(
                _dr_kernel_data(fit_sample),
                mean_method=mean_config_for_model(local_config, model_key).method,
                verbose=local_config.verbose,
                name=display_name,
                seed=local_config.seed,
            )
            test_metrics, test_policy, test_curve = _evaluate_bridge_model(
                model,
                simulator,
                test_sample,
                reference_curve=test_oracle_curve,
                verbose=local_config.verbose,
                name=f"{display_name}/test",
                fast_factual_only=local_config.fast_factual_eval,
            )
            val_metrics, _, val_curve = _evaluate_bridge_model(
                model,
                simulator,
                val_sample,
                reference_curve=val_oracle_curve,
                verbose=local_config.verbose,
                name=f"{display_name}/validation",
                fast_factual_only=local_config.fast_factual_eval,
            )
            if local_config.fast_factual_eval:
                test_metrics["observed_rmse"] = float("nan")
                val_metrics["observed_rmse"] = float("nan")
            else:
                test_metrics["observed_rmse"] = _observed_bridge_rmse_logged(
                    model,
                    test_sample,
                    verbose=local_config.verbose,
                    name=display_name,
                    split="test",
                )
                val_metrics["observed_rmse"] = _observed_bridge_rmse_logged(
                    model,
                    val_sample,
                    verbose=local_config.verbose,
                    name=display_name,
                    split="validation",
                )
            test_metrics["dr_pmmr_lambda"] = float(local_config.dr_pmmr_conditional.pmmr_lambda)
            test_metrics["dr_kap_lambda"] = float(model.treatment_bridge.lambda_)
            test_metrics["dr_kap_eta"] = float(model.treatment_bridge.eta)
            test_metrics["dr_kap_lambda2"] = float(model.treatment_bridge.lambda2)
            test_metrics["dr_kap_zeta"] = float(model.treatment_bridge.zeta)
            test_metrics["dr_lambda_dr"] = float(model.lambda_dr_)
            for key in (
                "dr_pmmr_lambda",
                "dr_kap_lambda",
                "dr_kap_eta",
                "dr_kap_lambda2",
                "dr_kap_zeta",
                "dr_lambda_dr",
            ):
                val_metrics[key] = test_metrics[key]
            training_histories["dr_pmmr_conditional_losses"] = model.training_history
            mean_predictors[policy_key] = model.mean_predict
            bridge_models[policy_key] = model
            bridge_model_for_direct = model
        elif model_key == "decaflow":
            model = ConditionalDeCaFlow(local_config.decaflow)
            model.fit(
                fit_sample.treatment,
                fit_sample.treatment_proxy,
                fit_sample.outcome_proxy,
                fit_sample.covariates,
                fit_sample.outcome,
                val_treatment=val_sample.treatment,
                val_treatment_proxy=val_sample.treatment_proxy,
                val_outcome_proxy=val_sample.outcome_proxy,
                val_covariates=val_sample.covariates,
                val_outcome=val_sample.outcome,
                verbose=local_config.verbose,
                name=display_name,
                seed=local_config.seed,
            )
            test_metrics, test_policy, test_curve = _evaluate_bridge_model(
                model,
                simulator,
                test_sample,
                reference_curve=test_oracle_curve,
                verbose=local_config.verbose,
                name=f"{display_name}/test",
                fast_factual_only=local_config.fast_factual_eval,
            )
            val_metrics, _, _ = _evaluate_bridge_model(
                model,
                simulator,
                val_sample,
                reference_curve=val_oracle_curve,
                verbose=local_config.verbose,
                name=f"{display_name}/validation",
                fast_factual_only=local_config.fast_factual_eval,
            )
            test_metrics.update(
                model.observational_metrics(
                    test_sample.treatment,
                    test_sample.treatment_proxy,
                    test_sample.outcome_proxy,
                    test_sample.covariates,
                    test_sample.outcome,
                )
            )
            val_metrics.update(model.validation_observational_metrics())
            test_metrics["observed_rmse"] = float("nan")
            val_metrics["observed_rmse"] = float("nan")
            training_histories["decaflow_losses"] = model.training_history
            mean_predictors[policy_key] = model.mean_predict
        elif model_key in {"nmmr", "nmmr_u", "nmmr_v"}:
            nmmr_cfg = copy.deepcopy(nmmr_config_for_model(local_config, model_key))
            model = NMMRBridge(nmmr_cfg, mean_config_for_model(local_config, model_key), local_config.w_flow, simulator.support)
            model.fit(
                _nmmr_data(fit_sample),
                mean_method=mean_config_for_model(local_config, model_key).method,
                verbose=local_config.verbose,
                name=display_name,
                seed=local_config.seed,
            )
            if local_config.kernel_diagnostics.enabled:
                kernel_result = model.kernel_diagnostics(
                    config=local_config.kernel_diagnostics,
                    model_name=display_name,
                    stage=f"{display_name}/bridge",
                )
                kernel_diagnostic_rows.extend(kernel_result.rows)
                kernel_offdiag_rows.extend(kernel_result.offdiag_rows)
                kernel_spectrum_rows.extend(kernel_result.spectrum_rows)
            test_metrics, test_policy, test_curve = _evaluate_bridge_model(
                model,
                simulator,
                test_sample,
                reference_curve=test_oracle_curve,
            )
            val_metrics, _, _ = _evaluate_bridge_model(
                model,
                simulator,
                val_sample,
                reference_curve=val_oracle_curve,
            )
            test_metrics["observed_rmse"] = _observed_bridge_rmse(model, test_sample)
            val_metrics["observed_rmse"] = _observed_bridge_rmse(model, val_sample)
            _attach_pseudooutcome_metrics(test_metrics, model)
            _attach_pseudooutcome_metrics(val_metrics, model)
            training_histories[f"{model_key}_losses"] = model.training_history
            mean_predictors[policy_key] = model.mean_predict
            bridge_models[policy_key] = model
            bridge_model_for_direct = model
        elif model_key in {"weighted_kpv", "local_kpv"}:
            (
                model,
                history,
                _,
                _,
                diagnostics,
                training,
                kernel_rows,
                offdiag_rows,
                spectrum_rows,
            ) = _fit_weighted_proximal(
                model_key, simulator, fit_sample, val_sample, local_config
            )
            kernel_diagnostic_rows.extend(kernel_rows)
            kernel_offdiag_rows.extend(offdiag_rows)
            kernel_spectrum_rows.extend(spectrum_rows)
            test_metrics, test_policy, test_curve = _evaluate_bridge_model(
                model,
                simulator,
                test_sample,
                reference_curve=test_oracle_curve,
            )
            val_metrics, _, _ = _evaluate_bridge_model(
                model,
                simulator,
                val_sample,
                reference_curve=val_oracle_curve,
            )
            test_metrics["observed_rmse"] = _observed_bridge_rmse(model, test_sample)
            val_metrics["observed_rmse"] = _observed_bridge_rmse(model, val_sample)
            _attach_pseudooutcome_metrics(test_metrics, model)
            _attach_pseudooutcome_metrics(val_metrics, model)
            test_metrics["weight_ess_final"] = history[-1]["weight_ess"]
            val_metrics["weight_ess_final"] = history[-1]["weight_ess"]
            weighted_histories[model_key] = history
            weight_diagnostics[model_key] = diagnostics
            training_histories[f"{model_key}_losses"] = training
            mean_predictors[policy_key] = model.mean_predict
            bridge_models[policy_key] = model
            bridge_model_for_direct = model
        elif model_key in {"weighted_dfpv", "local_dfpv"}:
            (
                model,
                history,
                _,
                _,
                diagnostics,
                training,
                kernel_rows,
                offdiag_rows,
                spectrum_rows,
            ) = _fit_weighted_proximal(
                model_key, simulator, fit_sample, val_sample, local_config
            )
            kernel_diagnostic_rows.extend(kernel_rows)
            kernel_offdiag_rows.extend(offdiag_rows)
            kernel_spectrum_rows.extend(spectrum_rows)
            test_metrics, test_policy, test_curve = _evaluate_bridge_model(
                model,
                simulator,
                test_sample,
                reference_curve=test_oracle_curve,
            )
            val_metrics, _, _ = _evaluate_bridge_model(
                model,
                simulator,
                val_sample,
                reference_curve=val_oracle_curve,
            )
            test_metrics["observed_rmse"] = _observed_bridge_rmse(model, test_sample)
            val_metrics["observed_rmse"] = _observed_bridge_rmse(model, val_sample)
            _attach_pseudooutcome_metrics(test_metrics, model)
            _attach_pseudooutcome_metrics(val_metrics, model)
            test_metrics["weight_ess_final"] = history[-1]["weight_ess"]
            val_metrics["weight_ess_final"] = history[-1]["weight_ess"]
            weighted_histories[model_key] = history
            weight_diagnostics[model_key] = diagnostics
            training_histories[f"{model_key}_losses"] = training
            mean_predictors[policy_key] = model.mean_predict
            bridge_models[policy_key] = model
            bridge_model_for_direct = model
        elif model_key in {"weighted_pmmr", "local_pmmr"}:
            (
                model,
                history,
                _,
                _,
                diagnostics,
                training,
                kernel_rows,
                offdiag_rows,
                spectrum_rows,
            ) = _fit_weighted_proximal(
                model_key, simulator, fit_sample, val_sample, local_config
            )
            kernel_diagnostic_rows.extend(kernel_rows)
            kernel_offdiag_rows.extend(offdiag_rows)
            kernel_spectrum_rows.extend(spectrum_rows)
            test_metrics, test_policy, test_curve = _evaluate_bridge_model(
                model,
                simulator,
                test_sample,
                reference_curve=test_oracle_curve,
            )
            val_metrics, _, _ = _evaluate_bridge_model(
                model,
                simulator,
                val_sample,
                reference_curve=val_oracle_curve,
            )
            test_metrics["observed_rmse"] = _observed_bridge_rmse(model, test_sample)
            val_metrics["observed_rmse"] = _observed_bridge_rmse(model, val_sample)
            _attach_pseudooutcome_metrics(test_metrics, model)
            _attach_pseudooutcome_metrics(val_metrics, model)
            test_metrics["weight_ess_final"] = history[-1]["weight_ess"]
            val_metrics["weight_ess_final"] = history[-1]["weight_ess"]
            weighted_histories[model_key] = history
            weight_diagnostics[model_key] = diagnostics
            training_histories[f"{model_key}_losses"] = training
            mean_predictors[policy_key] = model.mean_predict
            bridge_models[policy_key] = model
            bridge_model_for_direct = model
        elif model_key in {"weighted_nmmr", "weighted_nmmr_u", "weighted_nmmr_v"}:
            (
                model,
                history,
                _,
                _,
                diagnostics,
                training,
                kernel_rows,
                offdiag_rows,
                spectrum_rows,
            ) = _fit_weighted_proximal(
                model_key, simulator, fit_sample, val_sample, local_config
            )
            kernel_diagnostic_rows.extend(kernel_rows)
            kernel_offdiag_rows.extend(offdiag_rows)
            kernel_spectrum_rows.extend(spectrum_rows)
            test_metrics, test_policy, test_curve = _evaluate_bridge_model(
                model,
                simulator,
                test_sample,
                reference_curve=test_oracle_curve,
            )
            val_metrics, _, _ = _evaluate_bridge_model(
                model,
                simulator,
                val_sample,
                reference_curve=val_oracle_curve,
            )
            test_metrics["observed_rmse"] = _observed_bridge_rmse(model, test_sample)
            val_metrics["observed_rmse"] = _observed_bridge_rmse(model, val_sample)
            _attach_pseudooutcome_metrics(test_metrics, model)
            _attach_pseudooutcome_metrics(val_metrics, model)
            test_metrics["weight_ess_final"] = history[-1]["weight_ess"]
            val_metrics["weight_ess_final"] = history[-1]["weight_ess"]
            weighted_histories[model_key] = history
            weight_diagnostics[model_key] = diagnostics
            training_histories[f"{model_key}_losses"] = training
            mean_predictors[policy_key] = model.mean_predict
            bridge_models[policy_key] = model
            bridge_model_for_direct = model
        elif model_key == "s_learner":
            use_two_stage = _effective_use_mean_slearner(local_config, oracle=False)
            model = SLearner(local_config.s_learner, local_config.mean, simulator.support)
            model.fit(
                fit_sample.treatment,
                fit_sample.treatment_proxy,
                fit_sample.outcome_proxy,
                fit_sample.covariates,
                fit_sample.outcome,
                verbose=local_config.verbose,
                name=display_name,
                seed=local_config.seed,
                include_proxies=use_two_stage,
            )
            if use_two_stage:
                _fit_slearner_mean_head(model, simulator, fit_sample, local_config, oracle=False, name=display_name)
                test_metrics, test_policy, test_curve = _evaluate_s_learner_mean(
                    model,
                    simulator,
                    test_sample,
                    reference_curve=None,
                )
                val_metrics, _, _ = _evaluate_s_learner_mean(
                    model,
                    simulator,
                    val_sample,
                    reference_curve=None,
                )
                mean_predictors[policy_key] = model.mean_predict
            else:
                test_metrics, test_policy, test_curve = _evaluate_s_learner(
                    model,
                    simulator,
                    test_sample,
                    reference_curve=test_oracle_curve,
                )
                val_metrics, _, _ = _evaluate_s_learner(
                    model,
                    simulator,
                    val_sample,
                    reference_curve=val_oracle_curve,
                )
                direct_predictors[policy_key] = model.direct_predict
            test_metrics["observed_rmse"] = _observed_direct_rmse(model, test_sample, test_sample.covariates)
            val_metrics["observed_rmse"] = _observed_direct_rmse(model, val_sample, val_sample.covariates)
            _attach_factual_rmse_from_observed(test_metrics)
            _attach_factual_rmse_from_observed(val_metrics)
            training_histories["s_learner_losses"] = model.training_history
        elif model_key == "oracle_s_learner":
            use_two_stage = _effective_use_mean_slearner(local_config, oracle=True)
            model = SLearner(local_config.s_learner, local_config.mean, simulator.support)
            model.fit(
                fit_sample.treatment,
                fit_sample.treatment_proxy,
                fit_sample.outcome_proxy,
                _augment_covariates_with_latents(fit_sample),
                fit_sample.outcome,
                verbose=local_config.verbose,
                name=display_name,
                seed=local_config.seed,
                include_proxies=use_two_stage,
            )
            if use_two_stage:
                _fit_slearner_mean_head(model, simulator, fit_sample, local_config, oracle=True, name=display_name)
                test_metrics, test_policy, test_curve = _evaluate_s_learner_mean(
                    model,
                    simulator,
                    test_sample,
                    reference_curve=None,
                )
                val_metrics, _, _ = _evaluate_s_learner_mean(
                    model,
                    simulator,
                    val_sample,
                    reference_curve=None,
                )
                mean_predictors[policy_key] = model.mean_predict
            else:
                test_metrics, test_policy, test_curve = _evaluate_s_learner(
                    model,
                    simulator,
                    test_sample,
                    oracle=True,
                    reference_curve=test_oracle_curve,
                )
                val_metrics, _, _ = _evaluate_s_learner(
                    model,
                    simulator,
                    val_sample,
                    oracle=True,
                    reference_curve=val_oracle_curve,
                )
                oracle_direct_predictors[policy_key] = (
                    lambda treatment, treatment_proxy, outcome_proxy, covariates, latent_uz, latent_uw, _model=model: _oracle_direct_predict(
                        _model,
                        treatment,
                        treatment_proxy,
                        outcome_proxy,
                        covariates,
                        latent_uz,
                        latent_uw,
                    )
                )
            test_metrics["observed_rmse"] = _observed_direct_rmse(model, test_sample, _augment_covariates_with_latents(test_sample))
            val_metrics["observed_rmse"] = _observed_direct_rmse(model, val_sample, _augment_covariates_with_latents(val_sample))
            _attach_factual_rmse_from_observed(test_metrics)
            _attach_factual_rmse_from_observed(val_metrics)
            training_histories["oracle_s_learner_losses"] = model.training_history
        elif model_key == "naive_oosr":
            use_two_stage = _effective_use_mean_oosr(local_config, oracle=False)
            model, history, diagnostics, training = _fit_naive_oosr(
                simulator,
                fit_sample,
                val_sample,
                local_config,
                include_proxies=use_two_stage,
            )
            if use_two_stage:
                _fit_slearner_mean_head(
                    model,
                    simulator,
                    fit_sample,
                    local_config,
                    oracle=False,
                    sample_weight=diagnostics.get("fit_weights") if diagnostics else None,
                    name=display_name,
                )
                test_metrics, test_policy, test_curve = _evaluate_s_learner_mean(
                    model,
                    simulator,
                    test_sample,
                    reference_curve=None,
                )
                val_metrics, _, _ = _evaluate_s_learner_mean(
                    model,
                    simulator,
                    val_sample,
                    reference_curve=None,
                )
                mean_predictors[policy_key] = model.mean_predict
            else:
                test_metrics, test_policy, test_curve = _evaluate_s_learner(
                    model,
                    simulator,
                    test_sample,
                    reference_curve=test_oracle_curve,
                )
                val_metrics, _, _ = _evaluate_s_learner(
                    model,
                    simulator,
                    val_sample,
                    reference_curve=val_oracle_curve,
                )
                direct_predictors[policy_key] = model.direct_predict
            test_metrics["observed_rmse"] = _observed_direct_rmse(model, test_sample, test_sample.covariates)
            val_metrics["observed_rmse"] = _observed_direct_rmse(model, val_sample, val_sample.covariates)
            test_metrics["weight_ess_final"] = history[-1]["weight_ess"]
            val_metrics["weight_ess_final"] = history[-1]["weight_ess"]
            weighted_histories["naive_oosr"] = history
            weight_diagnostics["naive_oosr"] = diagnostics
            training_histories["naive_oosr_losses"] = training
        elif model_key == "oracle_naive_oosr":
            use_two_stage = _effective_use_mean_oosr(local_config, oracle=True)
            model, history, diagnostics, training = _fit_naive_oosr(
                simulator,
                fit_sample,
                val_sample,
                local_config,
                oracle=True,
                include_proxies=use_two_stage,
            )
            if use_two_stage:
                _fit_slearner_mean_head(
                    model,
                    simulator,
                    fit_sample,
                    local_config,
                    oracle=True,
                    sample_weight=diagnostics.get("fit_weights") if diagnostics else None,
                    name=display_name,
                )
                test_metrics, test_policy, test_curve = _evaluate_s_learner_mean(
                    model,
                    simulator,
                    test_sample,
                    reference_curve=None,
                )
                val_metrics, _, _ = _evaluate_s_learner_mean(
                    model,
                    simulator,
                    val_sample,
                    reference_curve=None,
                )
                mean_predictors[policy_key] = model.mean_predict
            else:
                test_metrics, test_policy, test_curve = _evaluate_s_learner(
                    model,
                    simulator,
                    test_sample,
                    oracle=True,
                    reference_curve=test_oracle_curve,
                )
                val_metrics, _, _ = _evaluate_s_learner(
                    model,
                    simulator,
                    val_sample,
                    oracle=True,
                    reference_curve=val_oracle_curve,
                )
                oracle_direct_predictors[policy_key] = (
                    lambda treatment, treatment_proxy, outcome_proxy, covariates, latent_uz, latent_uw, _model=model: _oracle_direct_predict(
                        _model,
                        treatment,
                        treatment_proxy,
                        outcome_proxy,
                        covariates,
                        latent_uz,
                        latent_uw,
                    )
                )
            test_metrics["observed_rmse"] = _observed_direct_rmse(model, test_sample, _augment_covariates_with_latents(test_sample))
            val_metrics["observed_rmse"] = _observed_direct_rmse(model, val_sample, _augment_covariates_with_latents(val_sample))
            test_metrics["weight_ess_final"] = history[-1]["weight_ess"]
            val_metrics["weight_ess_final"] = history[-1]["weight_ess"]
            weighted_histories["oracle_naive_oosr"] = history
            weight_diagnostics["oracle_naive_oosr"] = diagnostics
            training_histories["oracle_naive_oosr_losses"] = training
        else:
            raise ValueError(f"Unhandled model key in run_experiment: {model_key}")

        if bridge_model_for_direct is not None and not local_config.fast_factual_eval:
            if model_key in {"dr_kpv", "dr_pmmr", "dr_pmmr_conditional"}:
                verbose_print(
                    local_config.verbose,
                    2,
                    f"[eval] {display_name} direct curve reuses mean curve for DR adapter",
                )
                test_direct_curve = test_curve.copy()
                val_direct_curve = val_curve.copy()
            else:
                step_start = time.perf_counter()
                verbose_print(local_config.verbose, 2, f"[eval] {display_name} direct curve start")
                try:
                    test_direct_curve = marginal_curve_from_bridge_predictor(
                        test_sample,
                        simulator.policy_grid,
                        bridge_model_for_direct.bridge_predict,
                    )
                    val_direct_curve = marginal_curve_from_bridge_predictor(
                        val_sample,
                        simulator.policy_grid,
                        bridge_model_for_direct.bridge_predict,
                    )
                    verbose_print(
                        local_config.verbose,
                        2,
                        f"[eval] {display_name} direct curve done elapsed={time.perf_counter() - step_start:.2f}s",
                    )
                except RuntimeError:
                    test_direct_curve = None
                    val_direct_curve = None
                    verbose_print(
                        local_config.verbose,
                        2,
                        f"[eval] {display_name} direct curve skipped elapsed={time.perf_counter() - step_start:.2f}s",
                    )
            if test_direct_curve is not None and val_direct_curve is not None:
                test_metrics.update(
                    evaluate_marginal_curve(
                        simulator,
                        test_sample,
                        simulator.policy_grid,
                        test_direct_curve,
                        prefix="mean_curve_direct",
                        reference_curve=test_oracle_curve,
                    )
                )
                val_metrics.update(
                    evaluate_marginal_curve(
                        simulator,
                        val_sample,
                        simulator.policy_grid,
                        val_direct_curve,
                        prefix="mean_curve_direct",
                        reference_curve=val_oracle_curve,
                    )
                )
                mean_outcome_curves_direct[policy_key] = test_direct_curve

        saved_policies[policy_key] = test_policy
        mean_outcome_curves[policy_key] = test_curve
        results.append({"method": display_name, **test_metrics})
        validation_results.append({"method": display_name, **val_metrics})
        if model_progress:
            test_regret = test_metrics.get("regret", float("nan"))
            val_regret = val_metrics.get("regret", float("nan"))
            print(
                (
                    f"[model-done] seed={config.seed} {model_index}/{total_models} "
                    f"model={display_name} elapsed={time.perf_counter() - model_start:.2f}s "
                    f"test_regret={float(test_regret):.6g} val_regret={float(val_regret):.6g}"
                ),
                flush=True,
            )

    if config.save_outputs:
        output_start = time.perf_counter()
        verbose_print(config.verbose, 2, "[output] saving scalar metrics and samples start")
        save_json(
            {
                "selected_models": selected,
                "global_config": config_to_dict(config),
                "per_model_configs": {key: config_to_dict(value) for key, value in model_configs.items()},
            },
            output_dir / "run_hyperparameters.json",
        )
        with (output_dir / "metrics.csv").open("w", newline="") as handle:
            fieldnames = sorted({key for row in results for key in row.keys()})
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        with (output_dir / "validation_metrics.csv").open("w", newline="") as handle:
            fieldnames = sorted({key for row in validation_results for key in row.keys()})
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(validation_results)

        for name, history in weighted_histories.items():
            save_json(history, output_dir / f"{name}_history.json")
        save_json(training_histories, output_dir / "training_histories.json")
        for name, diag in weight_diagnostics.items():
            np.savez(output_dir / f"{name}_diagnostics.npz", **diag)
        np.savez(
            output_dir / "train_sample.npz",
            treatment=train_sample.treatment,
            treatment_proxy=train_sample.treatment_proxy,
            outcome_proxy=train_sample.outcome_proxy,
            covariates=train_sample.covariates,
            outcome=train_sample.outcome,
            optimal_treatment=train_sample.optimal_treatment,
            assignment_mean=train_sample.assignment_mean,
            latent_uz=train_sample.latent_uz,
            latent_uw=train_sample.latent_uw,
        )
        np.savez(
            output_dir / "test_sample.npz",
            treatment=test_sample.treatment,
            treatment_proxy=test_sample.treatment_proxy,
            outcome_proxy=test_sample.outcome_proxy,
            covariates=test_sample.covariates,
            outcome=test_sample.outcome,
            optimal_treatment=test_sample.optimal_treatment,
            assignment_mean=test_sample.assignment_mean,
            latent_uz=test_sample.latent_uz,
            latent_uw=test_sample.latent_uw,
        )
        np.savez(
            output_dir / "policies.npz",
            optimal_treatment=test_sample.optimal_treatment.reshape(-1),
            observed_treatment=test_sample.treatment.reshape(-1),
            x0=test_sample.covariates[:, 0],
            x1=test_sample.covariates[:, 1],
            **{key: value.reshape(-1) for key, value in saved_policies.items()},
        )
        verbose_print(
            config.verbose,
            2,
            f"[output] saving scalar metrics and samples done elapsed={time.perf_counter() - output_start:.2f}s",
        )
        step_start = time.perf_counter()
        verbose_print(config.verbose, 2, "[output] building saved mean curves start")
        curve_grid, causal_mean_curve, structure_curve = simulator.average_treatment_effect_curve(
            test_sample.covariates,
            grid=simulator.policy_grid,
        )
        oracle_mean_curve = _oracle_conditioned_marginal_curve(simulator, test_sample, curve_grid)
        proxy_mean_curve = _proxy_conditioned_marginal_curve(simulator, test_sample, curve_grid)
        np.savez(
            output_dir / "mean_potential_outcome_curves.npz",
            treatment_grid=curve_grid,
            ground_truth=oracle_mean_curve,
            structural_mean=causal_mean_curve,
            causal_marginal=causal_mean_curve,
            proxy_conditioned=proxy_mean_curve,
            structure_term=structure_curve,
            **{key: value.reshape(-1) for key, value in mean_outcome_curves.items()},
        )
        if mean_outcome_curves_direct:
            direct_arrays = {key: value.reshape(-1) for key, value in mean_outcome_curves_direct.items()}
            np.savez(
                output_dir / "mean_potential_outcome_curves_direct.npz",
                treatment_grid=curve_grid,
                ground_truth=oracle_mean_curve,
                structural_mean=causal_mean_curve,
                causal_marginal=causal_mean_curve,
                proxy_conditioned=proxy_mean_curve,
                structure_term=structure_curve,
                **direct_arrays,
            )
            np.savez(
                output_dir / "mean_curve_direct.npz",
                treatment_grid=curve_grid,
                ground_truth=oracle_mean_curve,
                structural_mean=causal_mean_curve,
                causal_marginal=causal_mean_curve,
                proxy_conditioned=proxy_mean_curve,
                structure_term=structure_curve,
                **direct_arrays,
            )
        verbose_print(
            config.verbose,
            2,
            f"[output] building saved mean curves done elapsed={time.perf_counter() - step_start:.2f}s",
        )
        step_start = time.perf_counter()
        verbose_print(config.verbose, 2, "[output] writing mean curve CSV files start")
        with (output_dir / "mean_potential_outcome_curves.csv").open("w", newline="") as handle:
            fieldnames = [
                "method",
                "treatment",
                "ground_truth",
                "structural_mean",
                "causal_marginal",
                "proxy_conditioned",
                "structure_term",
                "prediction",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for treatment, truth, causal, proxy, structure in zip(
                curve_grid,
                oracle_mean_curve,
                causal_mean_curve,
                proxy_mean_curve,
                structure_curve,
            ):
                writer.writerow(
                    {
                        "method": "Ground truth",
                        "treatment": float(treatment),
                        "ground_truth": float(truth),
                        "structural_mean": float(causal),
                        "causal_marginal": float(causal),
                        "proxy_conditioned": float(proxy),
                        "structure_term": float(structure),
                        "prediction": float(truth),
                    }
                )
            for method, curve in mean_outcome_curves.items():
                for treatment, truth, causal, proxy, structure, pred in zip(
                    curve_grid,
                    oracle_mean_curve,
                    causal_mean_curve,
                    proxy_mean_curve,
                    structure_curve,
                    curve.reshape(-1),
                ):
                    writer.writerow(
                        {
                            "method": method,
                            "treatment": float(treatment),
                            "ground_truth": float(truth),
                            "structural_mean": float(causal),
                            "causal_marginal": float(causal),
                            "proxy_conditioned": float(proxy),
                            "structure_term": float(structure),
                            "prediction": float(pred),
                        }
                    )
        if mean_outcome_curves_direct:
            with (output_dir / "mean_potential_outcome_curves_direct.csv").open("w", newline="") as handle:
                fieldnames = [
                    "method",
                    "treatment",
                    "ground_truth",
                    "structural_mean",
                    "causal_marginal",
                    "proxy_conditioned",
                    "structure_term",
                    "prediction",
                ]
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for treatment, truth, causal, proxy, structure in zip(
                    curve_grid,
                    oracle_mean_curve,
                    causal_mean_curve,
                    proxy_mean_curve,
                    structure_curve,
                ):
                    writer.writerow(
                        {
                            "method": "Ground truth",
                            "treatment": float(treatment),
                            "ground_truth": float(truth),
                            "structural_mean": float(causal),
                            "causal_marginal": float(causal),
                            "proxy_conditioned": float(proxy),
                            "structure_term": float(structure),
                            "prediction": float(truth),
                        }
                    )
                for method, curve in mean_outcome_curves_direct.items():
                    for treatment, truth, causal, proxy, structure, pred in zip(
                        curve_grid,
                        oracle_mean_curve,
                        causal_mean_curve,
                        proxy_mean_curve,
                        structure_curve,
                        curve.reshape(-1),
                    ):
                        writer.writerow(
                            {
                                "method": method,
                                "treatment": float(treatment),
                                "ground_truth": float(truth),
                                "structural_mean": float(causal),
                                "causal_marginal": float(causal),
                                "proxy_conditioned": float(proxy),
                                "structure_term": float(structure),
                                "prediction": float(pred),
                            }
                        )
            with (output_dir / "mean_curve_direct.csv").open("w", newline="") as handle:
                fieldnames = [
                    "method",
                    "treatment",
                    "ground_truth",
                    "structural_mean",
                    "causal_marginal",
                    "proxy_conditioned",
                    "structure_term",
                    "prediction",
                ]
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for treatment, truth, causal, proxy, structure in zip(
                    curve_grid,
                    oracle_mean_curve,
                    causal_mean_curve,
                    proxy_mean_curve,
                    structure_curve,
                ):
                    writer.writerow(
                        {
                            "method": "Ground truth",
                            "treatment": float(treatment),
                            "ground_truth": float(truth),
                            "structural_mean": float(causal),
                            "causal_marginal": float(causal),
                            "proxy_conditioned": float(proxy),
                            "structure_term": float(structure),
                            "prediction": float(truth),
                        }
                    )
                for method, curve in mean_outcome_curves_direct.items():
                    for treatment, truth, causal, proxy, structure, pred in zip(
                        curve_grid,
                        oracle_mean_curve,
                        causal_mean_curve,
                        proxy_mean_curve,
                        structure_curve,
                        curve.reshape(-1),
                    ):
                        writer.writerow(
                            {
                                "method": method,
                                "treatment": float(treatment),
                                "ground_truth": float(truth),
                                "structural_mean": float(causal),
                                "causal_marginal": float(causal),
                                "proxy_conditioned": float(proxy),
                                "structure_term": float(structure),
                                "prediction": float(pred),
                            }
                        )
        verbose_print(
            config.verbose,
            2,
            f"[output] writing mean curve CSV files done elapsed={time.perf_counter() - step_start:.2f}s",
        )
        if results_export is not None and results_export.enabled:
            step_start = time.perf_counter()
            verbose_print(config.verbose, 1, "[output] saving result curve tables start")
            result_curve_files = save_result_curve_tables(
                output_dir=output_dir,
                simulator=simulator,
                sample=test_sample,
                dataset=config.data.version,
                seed=config.seed,
                treatment_grid=curve_grid,
                ground_truth_curve=oracle_mean_curve,
                causal_marginal_curve=causal_mean_curve,
                proxy_conditioned_curve=proxy_mean_curve,
                structure_curve=structure_curve,
                mean_outcome_curves=mean_outcome_curves,
                mean_outcome_curves_direct=mean_outcome_curves_direct,
                mean_predictors=mean_predictors,
                direct_predictors=direct_predictors,
                oracle_direct_predictors=oracle_direct_predictors,
                config=results_export,
                verbose=config.verbose,
            )
            artifacts["result_curve_files"] = result_curve_files
            verbose_print(
                config.verbose,
                1,
                f"[output] saving result curve tables done elapsed={time.perf_counter() - step_start:.2f}s",
            )
        if config.save_figures:
            evaluation_treatment_grid = np.asarray(simulator.policy_grid, dtype=float).reshape(-1)
            step_start = time.perf_counter()
            n_patient_figures = min(10, int(test_sample.covariates.shape[0]))
            figure_mean_predictors = _wrap_mean_predictors_for_figure_progress(
                mean_predictors,
                verbose=config.verbose,
                n_patients=n_patient_figures,
            )
            verbose_print(
                config.verbose,
                2,
                (
                    f"[output] saving figures start expected_mean_predict_calls="
                    f"{2 * n_patient_figures * len(mean_predictors)}"
                ),
            )
            created_figures = save_run_figures(
                output_dir=output_dir,
                simulator=simulator,
                train_sample=train_sample,
                test_sample=test_sample,
                metrics=results,
                policies=saved_policies,
                mean_outcome_curves=mean_outcome_curves,
                mean_outcome_curves_direct=mean_outcome_curves_direct,
                ground_truth_curve=oracle_mean_curve,
                causal_marginal_curve=causal_mean_curve,
                proxy_conditioned_curve=proxy_mean_curve,
                mean_predictors=figure_mean_predictors,
                direct_predictors=direct_predictors,
                oracle_direct_predictors=oracle_direct_predictors,
                weight_diagnostics=weight_diagnostics,
                bridge_models=bridge_models,
                evaluation_treatment_grid=evaluation_treatment_grid,
                training_histories=training_histories,
            )
            artifacts["figure_files"] = created_figures
            verbose_print(
                config.verbose,
                2,
                f"[output] saving figures done count={len(created_figures)} elapsed={time.perf_counter() - step_start:.2f}s",
            )
            if config.plot_losses:
                step_start = time.perf_counter()
                verbose_print(config.verbose, 2, "[output] saving loss figures start")
                loss_figures = save_loss_figures(output_dir, training_histories)
                artifacts["loss_figure_files"] = loss_figures
                verbose_print(
                    config.verbose,
                    2,
                    f"[output] saving loss figures done count={len(loss_figures)} elapsed={time.perf_counter() - step_start:.2f}s",
                )
            if config.bridge_diagnostics and bridge_models:
                step_start = time.perf_counter()
                verbose_print(config.verbose, 2, "[output] saving bridge diagnostics start")
                rows, bridge_figures = save_bridge_diagnostics(
                    output_dir=output_dir,
                    simulator=simulator,
                    samples={"fit": fit_sample, "validation": val_sample, "test": test_sample},
                    bridge_models=bridge_models,
                    treatment_grid=evaluation_treatment_grid,
                )
                artifacts["bridge_diagnostic_rows"] = rows
                artifacts["bridge_diagnostic_figure_files"] = bridge_figures
                verbose_print(
                    config.verbose,
                    2,
                    (
                        f"[output] saving bridge diagnostics done rows={len(rows)} "
                        f"figures={len(bridge_figures)} elapsed={time.perf_counter() - step_start:.2f}s"
                    ),
                )
        if kernel_diagnostic_rows:
            step_start = time.perf_counter()
            verbose_print(config.verbose, 2, "[output] saving kernel diagnostics start")
            kernel_files = save_kernel_diagnostics(
                output_dir=output_dir,
                rows=kernel_diagnostic_rows,
                offdiag_rows=kernel_offdiag_rows,
                spectrum_rows=kernel_spectrum_rows,
                save_offdiag_values=config.kernel_diagnostics.save_offdiag_values,
                save_figures=config.save_figures,
            )
            artifacts["kernel_diagnostic_files"] = kernel_files
            verbose_print(
                config.verbose,
                2,
                f"[output] saving kernel diagnostics done count={len(kernel_files)} elapsed={time.perf_counter() - step_start:.2f}s",
            )
        save_json(artifacts, output_dir / "artifacts.json")
        verbose_print(config.verbose, 1, f"[done] experiment output_dir={output_dir}")
    return {"metrics": results, "validation_metrics": validation_results, "artifacts": artifacts}
