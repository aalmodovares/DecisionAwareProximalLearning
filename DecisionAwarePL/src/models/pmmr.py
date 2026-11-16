from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import FlowConfig, KernelDiagnosticsConfig, MeanConfig, PMMRConfig
from src.models.common import TorchRegressor
from src.models.flows import ConditionalFlowDensity
from src.models.kernel_diagnostics import KernelDiagnosticResult, diagnose_product_rbf_kernel, empty_kernel_diagnostic_result, merge_kernel_diagnostic_results
from src.models.kernel_utils import median_bandwidth_per_dimension, rbf_product_kernel
from src.utils import ArrayScaler, kfold_indices, split_index, verbose_print


@dataclass
class BridgeTrainData:
    treatment: np.ndarray
    treatment_proxy: np.ndarray
    outcome_proxy: np.ndarray
    covariates: np.ndarray
    outcome: np.ndarray


def _subset_data(data: BridgeTrainData, idx: np.ndarray) -> BridgeTrainData:
    return BridgeTrainData(
        treatment=data.treatment[idx],
        treatment_proxy=data.treatment_proxy[idx],
        outcome_proxy=data.outcome_proxy[idx],
        covariates=data.covariates[idx],
        outcome=data.outcome[idx],
    )


class PMMRBridge:
    def __init__(
        self,
        model_config: PMMRConfig,
        mean_config: MeanConfig,
        w_flow_config: FlowConfig,
        support: tuple[float, float],
    ):
        self.model_config = model_config
        self.mean_config = mean_config
        self.w_flow_config = w_flow_config
        self.support = support
        self.awx_scaler = ArrayScaler()
        self.azx_scaler = ArrayScaler()
        self.x_scaler = ArrayScaler()
        self.w_scaler = ArrayScaler()
        self.alpha: np.ndarray | None = None
        self.train_awx: np.ndarray | None = None
        self.train_azx: np.ndarray | None = None
        self.awx_bandwidth: np.ndarray | None = None
        self.azx_bandwidth: np.ndarray | None = None
        self.mean_regressor: TorchRegressor | None = None
        self.w_flow: ConditionalFlowDensity | None = None
        self.mean_method: str | None = None
        self.training_history: list[dict[str, object]] = []
        self.mean_crossfit_folds_: list[dict[str, np.ndarray | float | int]] = []
        self.mean_pseudooutcome_metadata_: dict[str, int | bool] = {}
        self.mean_regressor_group_ids_: np.ndarray | None = None
        self.mean_pseudooutcome_diagnostics_: dict[str, np.ndarray | float] = {}

    def _bridge_objective(
        self,
        treatment: np.ndarray,
        treatment_proxy: np.ndarray,
        outcome_proxy: np.ndarray,
        covariates: np.ndarray,
        outcome: np.ndarray,
        sample_weight: np.ndarray,
    ) -> float:
        awx = np.concatenate([treatment, outcome_proxy, covariates], axis=1)
        azx = np.concatenate([treatment, treatment_proxy, covariates], axis=1)
        awx_scaled = self.awx_scaler.transform(awx)
        azx_scaled = self.azx_scaler.transform(azx)
        k_l = rbf_product_kernel(awx_scaled, self.train_awx, self.awx_bandwidth)
        bandwidth = median_bandwidth_per_dimension(azx_scaled, self.model_config.azx_scale)
        k_m = rbf_product_kernel(azx_scaled, azx_scaled, bandwidth)
        pred = k_l @ self.alpha
        residual = outcome.reshape(-1, 1) - pred
        moment_weight = np.power(np.clip(sample_weight, 1e-12, None), self.model_config.moment_weight_power)
        weighted = residual * moment_weight.reshape(-1, 1)
        numerator = weighted.T @ k_m @ weighted
        norm = max(float(np.square(sample_weight.sum())), 1e-8)
        reg = self.model_config.ridge * float(np.mean(np.square(pred))) / max(norm, 1e-8)
        return float(numerator[0, 0] / norm + reg)

    def _fit_bridge(
        self,
        data: BridgeTrainData,
        sample_weight: np.ndarray | None = None,
        *,
        verbose: int = 0,
        name: str = "PMMR",
        seed: int = 13,
    ) -> list[dict[str, object]]:
        verbose_print(verbose, 1, f"[start] {name} bridge solve")
        a = data.treatment
        z = data.treatment_proxy
        w = data.outcome_proxy
        x = data.covariates
        y = data.outcome.reshape(-1, 1)
        n = a.shape[0]
        if sample_weight is None:
            sample_weight = np.ones(n, dtype=float)
        sample_weight = np.asarray(sample_weight, dtype=float).reshape(-1)

        awx = np.concatenate([a, w, x], axis=1)
        azx = np.concatenate([a, z, x], axis=1)
        self.train_awx = self.awx_scaler.fit_transform(awx)
        self.train_azx = self.azx_scaler.fit_transform(azx)
        self.x_scaler.fit(x)
        self.w_scaler.fit(w)

        self.awx_bandwidth = median_bandwidth_per_dimension(self.train_awx, self.model_config.awx_scale)
        self.azx_bandwidth = median_bandwidth_per_dimension(self.train_azx, self.model_config.azx_scale)

        k_l = rbf_product_kernel(self.train_awx, self.train_awx, self.awx_bandwidth)
        k_m = rbf_product_kernel(self.train_azx, self.train_azx, self.azx_bandwidth)
        moment_weight = np.power(np.clip(sample_weight, 1e-12, None), self.model_config.moment_weight_power)
        Ww = k_m * np.outer(moment_weight, moment_weight)
        system = k_l @ Ww @ k_l + self.model_config.ridge * k_l + self.model_config.jitter * np.eye(n)
        target = k_l @ Ww @ y
        self.alpha = np.linalg.solve(system, target)

        train_idx, val_idx = split_index(n, val_fraction=self.mean_config.val_fraction, seed=seed)
        train_loss = self._bridge_objective(a[train_idx], z[train_idx], w[train_idx], x[train_idx], y[train_idx], sample_weight[train_idx])
        if len(val_idx) > 0:
            val_loss = self._bridge_objective(a[val_idx], z[val_idx], w[val_idx], x[val_idx], y[val_idx], sample_weight[val_idx])
        else:
            val_loss = train_loss
        history = [
            {
                "stage": f"{name}/bridge",
                "train_losses": [train_loss],
                "val_losses": [val_loss],
                "best_epoch": 0,
                "stopped_epoch": 0,
            }
        ]
        verbose_print(verbose, 1, f"[done] {name} bridge solve train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        return history

    def kernel_diagnostics(
        self,
        *,
        config: KernelDiagnosticsConfig,
        model_name: str = "PMMR",
        stage: str = "PMMR/bridge",
        round_id: int | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> KernelDiagnosticResult:
        if self.train_awx is None or self.train_azx is None or self.awx_bandwidth is None or self.azx_bandwidth is None:
            return empty_kernel_diagnostic_result()
        base_metadata = {
            "model": model_name,
            "stage": stage,
            "round": "" if round_id is None else int(round_id),
            "estimator": "PMMR",
        }
        raw_results = [
            diagnose_product_rbf_kernel(
                features=self.train_awx,
                bandwidths=self.awx_bandwidth,
                config=config,
                metadata={**base_metadata, "kernel": "K_AWX", "weighted": False},
            ),
            diagnose_product_rbf_kernel(
                features=self.train_azx,
                bandwidths=self.azx_bandwidth,
                config=config,
                metadata={**base_metadata, "kernel": "K_AZX", "weighted": False},
            ),
        ]
        if sample_weight is not None:
            raw_results.extend(
                [
                    diagnose_product_rbf_kernel(
                        features=self.train_awx,
                        bandwidths=self.awx_bandwidth,
                        config=config,
                        metadata={**base_metadata, "kernel": "K_AWX", "weighted": True},
                        sample_weight=sample_weight,
                    ),
                    diagnose_product_rbf_kernel(
                        features=self.train_azx,
                        bandwidths=self.azx_bandwidth,
                        config=config,
                        metadata={**base_metadata, "kernel": "K_AZX", "weighted": True},
                        sample_weight=sample_weight,
                    ),
                ]
            )
        return merge_kernel_diagnostic_results(raw_results)

    def fit(
        self,
        data: BridgeTrainData,
        sample_weight: np.ndarray | None = None,
        mean_method: str = "pseudo_outcomes",
        *,
        weight_mean_regressor: bool = True,
        warm_start: bool = False,
        verbose: int = 0,
        name: str = "PMMR",
        seed: int = 13,
    ) -> None:
        n = data.treatment.shape[0]
        if sample_weight is None:
            sample_weight = np.ones(n, dtype=float)
        sample_weight = np.asarray(sample_weight, dtype=float).reshape(-1)
        w = data.outcome_proxy
        x = data.covariates

        self.training_history = self._fit_bridge(
            data,
            sample_weight=sample_weight,
            verbose=verbose,
            name=name,
            seed=seed,
        )

        self.mean_method = mean_method
        if mean_method == "pseudo_outcomes":
            mean_sample_weight = sample_weight if bool(weight_mean_regressor) else np.ones_like(sample_weight)
            self._fit_mean_regressor(
                data,
                bridge_sample_weight=sample_weight,
                regressor_sample_weight=mean_sample_weight,
                weight_mean_regressor=bool(weight_mean_regressor),
                warm_start=bool(warm_start),
                verbose=verbose,
                name=f"{name}/mean_regressor",
                seed=seed,
            )
        elif mean_method == "flow_mc":
            verbose_print(verbose, 1, f"[start] {name} W|X flow")
            self.w_flow = ConditionalFlowDensity(
                features=w.shape[1],
                context_dim=x.shape[1],
                config=self.w_flow_config,
                support=None,
            )
            flow_result = self.w_flow.fit(w, x, verbose=verbose, name=f"{name}/w_flow", seed=seed)
            self.training_history.append(
                {
                    "stage": f"{name}/w_flow",
                    "train_losses": flow_result["train_losses"],
                    "val_losses": flow_result["val_losses"],
                    "best_epoch": flow_result["best_epoch"],
                    "stopped_epoch": flow_result["stopped_epoch"],
                }
            )
        else:
            raise ValueError(f"Unknown mean method: {mean_method}")

    def bridge_predict(self, treatment: np.ndarray, outcome_proxy: np.ndarray, covariates: np.ndarray) -> np.ndarray:
        awx = np.concatenate([treatment, outcome_proxy, covariates], axis=1)
        awx_scaled = self.awx_scaler.transform(awx)
        k_test = rbf_product_kernel(awx_scaled, self.train_awx, self.awx_bandwidth)
        return k_test @ self.alpha

    def _append_pseudo_outcomes(
        self,
        *,
        bridge: "PMMRBridge",
        data: BridgeTrainData,
        original_idx: np.ndarray,
        grid: np.ndarray | None,
        regressors_x: list[np.ndarray],
        pseudo_y: list[np.ndarray],
        group_ids: list[np.ndarray],
    ) -> None:
        x = data.covariates
        w = data.outcome_proxy
        if grid is None:
            bridge_values = bridge.bridge_predict(data.treatment, w, x)
            regressors_x.append(np.concatenate([data.treatment, x], axis=1))
            pseudo_y.append(bridge_values)
            group_ids.append(original_idx.reshape(-1))
            return
        for a_val in grid:
            a = np.full((x.shape[0], 1), a_val)
            bridge_values = bridge.bridge_predict(a, w, x)
            regressors_x.append(np.concatenate([a, x], axis=1))
            pseudo_y.append(bridge_values)
            group_ids.append(original_idx.reshape(-1))

    def _fit_mean_regressor(
        self,
        data: BridgeTrainData,
        bridge_sample_weight: np.ndarray,
        regressor_sample_weight: np.ndarray,
        weight_mean_regressor: bool,
        warm_start: bool,
        *,
        verbose: int = 0,
        name: str = "PMMR/mean_regressor",
        seed: int = 13,
    ) -> None:
        use_treatment_grid = bool(self.mean_config.use_treatment_grid)
        grid = (
            np.linspace(self.support[0], self.support[1], self.mean_config.treatment_grid_size)
            if use_treatment_grid
            else None
        )
        regressors_x = []
        pseudo_y = []
        group_ids = []
        n = data.treatment.shape[0]
        n_folds = int(self.mean_config.crossfit_folds)
        self.mean_crossfit_folds_ = []
        if n_folds <= 1:
            self._append_pseudo_outcomes(
                bridge=self,
                data=data,
                original_idx=np.arange(n),
                grid=grid,
                regressors_x=regressors_x,
                pseudo_y=pseudo_y,
                group_ids=group_ids,
            )
            self.mean_crossfit_folds_.append(
                {
                    "fold": 0,
                    "train_idx": np.arange(n),
                    "holdout_idx": np.arange(n),
                    "train_weight_sum": float(np.sum(bridge_sample_weight)),
                    "holdout_weight_sum": float(np.sum(bridge_sample_weight)),
                }
            )
        else:
            folds = kfold_indices(n, n_folds, seed=seed, shuffle=self.mean_config.crossfit_shuffle)
            for fold_id, (train_idx, holdout_idx) in enumerate(folds):
                fold_model = PMMRBridge(self.model_config, self.mean_config, self.w_flow_config, self.support)
                fold_history = fold_model._fit_bridge(
                    _subset_data(data, train_idx),
                    sample_weight=bridge_sample_weight[train_idx],
                    verbose=verbose,
                    name=f"{name}/fold_{fold_id}",
                    seed=seed + fold_id + 1,
                )
                self.training_history.extend(fold_history)
                holdout_data = _subset_data(data, holdout_idx)
                self._append_pseudo_outcomes(
                    bridge=fold_model,
                    data=holdout_data,
                    original_idx=holdout_idx,
                    grid=grid,
                    regressors_x=regressors_x,
                    pseudo_y=pseudo_y,
                    group_ids=group_ids,
                )
                self.mean_crossfit_folds_.append(
                    {
                        "fold": fold_id,
                        "train_idx": train_idx.copy(),
                        "holdout_idx": holdout_idx.copy(),
                        "train_weight_sum": float(np.sum(bridge_sample_weight[train_idx])),
                        "holdout_weight_sum": float(np.sum(bridge_sample_weight[holdout_idx])),
                    }
                )
        reg_x = np.concatenate(regressors_x, axis=0)
        reg_y = np.concatenate(pseudo_y, axis=0)
        reg_group_ids = np.concatenate(group_ids, axis=0)
        reg_row_weights = regressor_sample_weight[reg_group_ids]
        reg_treatment = reg_x[:, 0].reshape(-1)
        unique_treatment = np.unique(reg_treatment) if use_treatment_grid else np.array([], dtype=float)
        std_by_treatment = []
        iqr_by_treatment = []
        for a_val in unique_treatment:
            mask = np.isclose(reg_treatment, a_val)
            values = reg_y[mask].reshape(-1)
            std_by_treatment.append(float(np.std(values)))
            iqr_by_treatment.append(float(np.quantile(values, 0.75) - np.quantile(values, 0.25)))
        self.mean_pseudooutcome_diagnostics_ = {
            "std": float(np.std(reg_y.reshape(-1))),
            "iqr": float(np.quantile(reg_y.reshape(-1), 0.75) - np.quantile(reg_y.reshape(-1), 0.25)),
            "treatment_grid": unique_treatment.astype(float),
            "std_by_treatment": np.asarray(std_by_treatment, dtype=float),
            "iqr_by_treatment": np.asarray(iqr_by_treatment, dtype=float),
        }
        self.mean_regressor_group_ids_ = reg_group_ids.copy()
        self.mean_pseudooutcome_metadata_ = {
            "crossfit": bool(n_folds > 1),
            "target_source": "pseudo_outcomes",
            "uses_treatment_grid": use_treatment_grid,
            "n_folds": min(max(n_folds, 1), n) if n > 0 else 1,
            "n_original": n,
            "treatment_grid_size": int(0 if grid is None else grid.shape[0]),
            "evaluation_grid_size": int(
                self.mean_config.treatment_grid_size if use_treatment_grid else self.mean_config.evaluation_grid_size
            ),
            "n_rows": reg_x.shape[0],
            "pseudo_outcome_std": self.mean_pseudooutcome_diagnostics_["std"],
            "pseudo_outcome_iqr": self.mean_pseudooutcome_diagnostics_["iqr"],
            "weighted_mean_regressor": bool(weight_mean_regressor),
        }
        can_warm_start_mean = (
            bool(warm_start)
            and self.mean_regressor is not None
            and getattr(self.mean_regressor, "input_dim", None) == reg_x.shape[1]
        )
        if can_warm_start_mean:
            verbose_print(verbose, 1, f"[warm-start] {name}")
        else:
            self.mean_regressor = TorchRegressor(
                input_dim=reg_x.shape[1],
                hidden_dims=self.mean_config.hidden_dims,
                lr=self.mean_config.lr,
                weight_decay=self.mean_config.weight_decay,
                epochs=self.mean_config.epochs,
                min_epochs=self.mean_config.min_epochs,
                early_stop_patience=self.mean_config.early_stop_patience,
                early_stop_min_delta=self.mean_config.early_stop_min_delta,
                batch_size=self.mean_config.batch_size,
                early_stopping_mode=self.mean_config.early_stopping_mode,
                early_stopping_v2_alpha=self.mean_config.early_stopping_v2_alpha,
                early_stopping_v2_patience=self.mean_config.early_stopping_v2_patience,
            )
        fit_result = self.mean_regressor.fit(
            reg_x,
            reg_y,
            sample_weight=reg_row_weights,
            verbose=verbose,
            name=name,
            val_fraction=self.mean_config.val_fraction,
            seed=seed,
            group_ids=reg_group_ids,
        )
        self.training_history.append(
            {
                "stage": name,
                "train_losses": fit_result.train_losses,
                "val_losses": fit_result.val_losses,
                "best_epoch": fit_result.best_epoch,
                "stopped_epoch": fit_result.stopped_epoch,
            }
        )

    def mean_predict(self, treatment: np.ndarray, covariates: np.ndarray) -> np.ndarray:
        if self.mean_method == "pseudo_outcomes":
            features = np.concatenate([treatment, covariates], axis=1)
            return self.mean_regressor.predict(features)
        if self.mean_method == "flow_mc":
            samples = self.w_flow.sample(covariates, self.mean_config.n_mc_samples)
            preds = []
            for sample_id in range(samples.shape[0]):
                preds.append(self.bridge_predict(treatment, samples[sample_id], covariates))
            return np.mean(np.stack(preds, axis=0), axis=0)
        raise RuntimeError("PMMRBridge has not been fitted.")
