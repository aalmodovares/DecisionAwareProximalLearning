from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.config import FlowConfig, KernelDiagnosticsConfig, MeanConfig, NMMRConfig
from src.models.common import TorchRegressor
from src.models.flows import ConditionalFlowDensity
from src.models.kernel_diagnostics import (
    KernelDiagnosticResult,
    diagnose_product_rbf_kernel,
    diagnose_rbf_kernel,
    empty_kernel_diagnostic_result,
    merge_kernel_diagnostic_results,
)
from src.models.kernel_utils import median_bandwidth, median_bandwidth_per_dimension, rbf_kernel, rbf_product_kernel
from src.utils import ArrayScaler, EarlyStoppingTracker, kfold_indices, progress_iter, safe_float, split_index, verbose_print


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


def _activation_module(name: str) -> torch.nn.Module:
    activation_map: dict[str, type[torch.nn.Module]] = {
        "elu": torch.nn.ELU,
        "relu": torch.nn.ReLU,
        "gelu": torch.nn.GELU,
        "tanh": torch.nn.Tanh,
        "silu": torch.nn.SiLU,
    }
    activation_cls = activation_map.get(str(name).lower())
    if activation_cls is None:
        raise ValueError(f"Unknown activation {name!r}.")
    return activation_cls()


class _NMMRMLP(torch.nn.Module):
    """Replica del MLP original de NMMR, incluyendo su rolloff final opcional."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        activation: str,
        *,
        deactivate_rolloff: bool,
    ):
        super().__init__()
        widths = [input_dim, *hidden_dims, 1]
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(widths[idx], widths[idx + 1]) for idx in range(len(widths) - 1)]
        )
        self.activation_name = activation
        self.deactivate_rolloff = bool(deactivate_rolloff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activation = _activation_module(self.activation_name)
        last_idx = len(self.layers) - 1
        out = x
    
        for idx, layer in enumerate(self.layers):
            out = layer(out)
    
            # Apply activation only to hidden layers, not the scalar bridge output.
            if idx < last_idx:
                out = activation(out)
    
        return out


class NMMRBridge:
    def __init__(
        self,
        model_config: NMMRConfig,
        mean_config: MeanConfig,
        w_flow_config: FlowConfig,
        support: tuple[float, float],
    ):
        self.model_config = model_config
        self.mean_config = mean_config
        self.w_flow_config = w_flow_config
        self.support = support
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_scaler = ArrayScaler()
        self.moment_scaler = ArrayScaler()
        self.target_scaler = ArrayScaler()
        self.model: torch.nn.Module | None = None
        self.train_moment: np.ndarray | None = None
        self.azx_bandwidth: float | np.ndarray | None = None
        self.mean_regressor: TorchRegressor | None = None
        self.w_flow: ConditionalFlowDensity | None = None
        self.mean_method: str | None = None
        self.training_history: list[dict[str, object]] = []
        self.mean_crossfit_folds_: list[dict[str, np.ndarray | float | int]] = []
        self.mean_pseudooutcome_metadata_: dict[str, int | bool] = {}
        self.mean_regressor_group_ids_: np.ndarray | None = None
        self.mean_pseudooutcome_diagnostics_: dict[str, np.ndarray | float] = {}
        self.moment_group_slices_: dict[str, slice] = {}

    def _resolved_bridge_batch_size(self, train_size: int) -> int:
        configured = self.model_config.batch_size
        if isinstance(configured, str):
            if configured.strip().lower() == "all":
                return max(1, int(train_size))
            raise ValueError(f"Unsupported NMMR batch_size string {configured!r}. Use 'all' or a positive integer.")
        resolved = int(configured)
        if resolved <= 0:
            raise ValueError("NMMR batch_size must be a positive integer or 'all'.")
        return min(resolved, max(1, int(train_size)))

    def _bridge_model_input(
        self,
        treatment: np.ndarray,
        outcome_proxy: np.ndarray,
        covariates: np.ndarray,
    ) -> np.ndarray:
        treatment = np.asarray(treatment, dtype=float)
        features = [treatment]
        if self.model_config.quadratic_treatment:
            features.append(-np.square(treatment))
        features.extend([np.asarray(outcome_proxy, dtype=float), np.asarray(covariates, dtype=float)])
        return np.concatenate(features, axis=1)

    def _kernel_matrix(self, features: np.ndarray) -> np.ndarray:
        if self.azx_bandwidth is None:
            raise RuntimeError("NMMR bandwidth is unavailable. Fit bridge before calling objective.")
        if self.model_config.product_kernel:
            bandwidths = np.asarray(self.azx_bandwidth, dtype=float).reshape(-1)
            return rbf_product_kernel(features, features, bandwidths)
        bandwidth = float(np.asarray(self.azx_bandwidth, dtype=float).reshape(-1)[0])
        return rbf_kernel(features, features, bandwidth)

    def _objective(
        self,
        x_scaled: np.ndarray,
        y_scaled: np.ndarray,
        m_scaled: np.ndarray,
        sample_weight: np.ndarray,
    ) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("NMMR bridge model is unavailable. Fit bridge before calling objective.")
        kernel = torch.tensor(
            self._kernel_matrix(m_scaled),
            dtype=torch.float32,
            device=self.device,
        )
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32, device=self.device)
        moment_weight_np = np.power(
            np.clip(sample_weight, 1e-12, None),
            self.model_config.moment_weight_power,
        )
        
        moment_weight = torch.tensor(
            moment_weight_np.reshape(-1, 1),
            dtype=torch.float32,
            device=self.device,
        )
        
        pred = self.model(x_tensor)
        residual = y_tensor - pred
        weighted_residual = moment_weight * residual
        if self.model_config.loss_name == "U_statistic":
            kernel_u = kernel.clone()
            kernel_u.fill_diagonal_(0.0)
            numerator = weighted_residual.T @ kernel_u @ weighted_residual
        
            denom = max(
                float(np.sum(np.outer(moment_weight_np, moment_weight_np)) - np.sum(np.square(moment_weight_np))),
                1e-8,
            )
        
            loss = numerator[0, 0] / denom
        else:
            numerator = weighted_residual.T @ kernel @ weighted_residual
        
            denom = max(float(np.square(moment_weight_np.sum())), 1e-8)
        
            loss = numerator[0, 0] / denom
        return loss

    def _fit_bridge(
        self,
        data: BridgeTrainData,
        sample_weight: np.ndarray | None = None,
        *,
        warm_start: bool = False,
        verbose: int = 0,
        name: str = "NMMR",
        seed: int = 13,
    ) -> list[dict[str, object]]:
        verbose_print(verbose, 1, f"[start] {name} bridge")
        a = data.treatment
        z = data.treatment_proxy
        w = data.outcome_proxy
        x = data.covariates
        y = data.outcome.reshape(-1, 1)
        n = a.shape[0]
        if sample_weight is None:
            sample_weight = np.ones(n, dtype=float)
        sample_weight = np.asarray(sample_weight, dtype=float).reshape(-1)

        model_input = self._bridge_model_input(a, w, x)
        moment_input = np.concatenate([a, z, x], axis=1)
        x_scaled = self.input_scaler.fit_transform(model_input)
        m_scaled = self.moment_scaler.fit_transform(moment_input)
        self.train_moment = m_scaled
        a_dim = int(a.shape[1])
        z_dim = int(z.shape[1])
        x_dim = int(x.shape[1])
        self.moment_group_slices_ = {
            "A": slice(0, a_dim),
            "Z": slice(a_dim, a_dim + z_dim),
            "X": slice(a_dim + z_dim, a_dim + z_dim + x_dim),
        }
        if self.model_config.product_kernel:
            bandwidths = median_bandwidth_per_dimension(m_scaled, self.model_config.azx_scale)
            group_scales = np.ones_like(bandwidths)
            group_scales[self.moment_group_slices_["A"]] *= float(self.model_config.a_scale)
            group_scales[self.moment_group_slices_["Z"]] *= float(self.model_config.z_scale)
            group_scales[self.moment_group_slices_["X"]] *= float(self.model_config.x_scale)
            self.azx_bandwidth = np.maximum(bandwidths * group_scales, 1e-8)
        else:
            self.azx_bandwidth = median_bandwidth(m_scaled, self.model_config.azx_scale)
        y_scaled = self.target_scaler.fit_transform(y)

        train_idx, val_idx = split_index(n, val_fraction=self.model_config.val_fraction, seed=seed)
        first_layer = None
        if self.model is not None:
            first_layer = next((module for module in self.model.modules() if isinstance(module, torch.nn.Linear)), None)
        can_warm_start_bridge = (
            bool(warm_start)
            and self.model is not None
            and first_layer is not None
            and int(first_layer.in_features) == int(x_scaled.shape[1])
        )
        if can_warm_start_bridge:
            self.model = self.model.to(self.device)
            verbose_print(verbose, 1, f"[warm-start] {name} bridge")
        else:
            self.model = _NMMRMLP(
                x_scaled.shape[1],
                self.model_config.hidden_dims,
                self.model_config.activation,
                deactivate_rolloff=self.model_config.deactivate_rolloff,
            ).to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.model_config.lr,
            weight_decay=self.model_config.weight_decay,
        )

        best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
        best_loss = float("inf")
        best_epoch = 0
        stopped_epoch = self.model_config.epochs - 1
        train_losses: list[float] = []
        val_losses: list[float] = []
        stopper = EarlyStoppingTracker(
            min_epochs=max(1, self.model_config.min_epochs),
            patience=max(1, self.model_config.early_stop_patience),
            min_delta=self.model_config.early_stop_min_delta,
            mode="min",
        )

        epoch_iter = progress_iter(range(self.model_config.epochs), verbose, desc=f"{name}/bridge", leave=False)
        for epoch in epoch_iter:
            self.model.train(True)
            permutation = np.random.default_rng(seed + epoch).permutation(train_idx)
            batch_size = self._resolved_bridge_batch_size(len(permutation))
            epoch_batch_sizes: list[int] = []
            for start in range(0, len(permutation), batch_size):
                batch_idx = permutation[start : start + batch_size]
                if batch_idx.shape[0] <= 1 and self.model_config.loss_name == "U_statistic":
                    continue
                if epoch == 0 and verbose >= 3:
                    epoch_batch_sizes.append(int(batch_idx.shape[0]))
                train_loss = self._objective(
                    x_scaled[batch_idx],
                    y_scaled[batch_idx],
                    m_scaled[batch_idx],
                    sample_weight[batch_idx],
                )
                optimizer.zero_grad()
                train_loss.backward()
                if self.model_config.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.model_config.grad_clip_norm)
                optimizer.step()
            if epoch == 0 and verbose >= 3:
                verbose_print(
                    verbose,
                    3,
                    f"[batch-debug] {name}/bridge epoch=0 batch_sizes={epoch_batch_sizes}",
                )

            self.model.train(False)
            with torch.no_grad():
                train_eval = self._objective(
                    x_scaled[train_idx],
                    y_scaled[train_idx],
                    m_scaled[train_idx],
                    sample_weight[train_idx],
                )
                train_value = safe_float(train_eval)
                if len(val_idx) > 0:
                    val_eval = self._objective(
                        x_scaled[val_idx],
                        y_scaled[val_idx],
                        m_scaled[val_idx],
                        sample_weight[val_idx],
                    )
                    val_value = safe_float(val_eval)
                    monitor = val_value
                else:
                    val_value = train_value
                    monitor = train_value

            train_losses.append(train_value)
            val_losses.append(val_value)
            if verbose >= 2 and hasattr(epoch_iter, "set_postfix"):
                epoch_iter.set_postfix(train=f"{train_value:.4f}", val=f"{val_value:.4f}", refresh=False)

            improved, should_stop = stopper.update(monitor, epoch)
            if improved:
                best_loss = monitor
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
            if should_stop:
                stopped_epoch = epoch
                break
        else:
            stopped_epoch = max(0, self.model_config.epochs - 1)

        self.model.load_state_dict(best_state)
        self.model.train(False)
        self.training_history = [
            {
                "stage": f"{name}/bridge",
                "train_losses": train_losses,
                "val_losses": val_losses,
                "best_epoch": best_epoch,
                "stopped_epoch": stopped_epoch,
                "best_loss": best_loss,
            }
        ]
        verbose_print(
            verbose,
            1,
            f"[done] {name} bridge best_epoch={best_epoch} stopped_epoch={stopped_epoch} "
            f"train_loss={train_losses[best_epoch]:.4f} val_loss={val_losses[best_epoch]:.4f}",
        )
        return [
            {
                "stage": f"{name}/bridge",
                "train_losses": train_losses,
                "val_losses": val_losses,
                "best_epoch": best_epoch,
                "stopped_epoch": stopped_epoch,
                "best_loss": best_loss,
            }
        ]

    def fit(
        self,
        data: BridgeTrainData,
        sample_weight: np.ndarray | None = None,
        mean_method: str = "pseudo_outcomes",
        *,
        weight_mean_regressor: bool = True,
        warm_start: bool = False,
        verbose: int = 0,
        name: str = "NMMR",
        seed: int = 13,
    ) -> None:
        n = data.treatment.shape[0]
        if sample_weight is None:
            sample_weight = np.ones(n, dtype=float)
        sample_weight = np.asarray(sample_weight, dtype=float).reshape(-1)
        w = data.outcome_proxy
        x = data.covariates
        train_bridge_all_data = bool(self.model_config.bridge_all_data)
        if train_bridge_all_data:
            self.training_history = self._fit_bridge(
                data,
                sample_weight=sample_weight,
                warm_start=bool(warm_start),
                verbose=verbose,
                name=name,
                seed=seed,
            )
        else:
            self.training_history = []
            self.model = None
            self.train_moment = None
            self.azx_bandwidth = None
        self.mean_method = mean_method
        if mean_method == "pseudo_outcomes":
            if (not train_bridge_all_data) and int(self.mean_config.crossfit_folds) <= 1:
                raise ValueError(
                    "NMMR requires mean.crossfit_folds > 1 when bridge_all_data=False for pseudo_outcomes."
                )
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
            if not train_bridge_all_data:
                raise ValueError("NMMR flow_mc requires bridge_all_data=True.")
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
        if self.model is None:
            raise RuntimeError("NMMR bridge is unavailable. Enable bridge_all_data or use mean_predict only.")
        model_input = self._bridge_model_input(treatment, outcome_proxy, covariates)
        scaled = self.input_scaler.transform(model_input)
        with torch.no_grad():
            pred = self.model(torch.tensor(scaled, dtype=torch.float32, device=self.device)).cpu().numpy()
        return self.target_scaler.inverse_transform(pred)

    def _append_pseudo_outcomes(
        self,
        *,
        bridge: "NMMRBridge",
        data: BridgeTrainData,
        original_idx: np.ndarray,
        grid: np.ndarray,
        regressors_x: list[np.ndarray],
        pseudo_y: list[np.ndarray],
        group_ids: list[np.ndarray],
    ) -> None:
        x = data.covariates
        w = data.outcome_proxy
        for a_val in grid:
            a = np.full((x.shape[0], 1), a_val)
            regressors_x.append(np.concatenate([a, x], axis=1))
            pseudo_y.append(bridge.bridge_predict(a, w, x))
            group_ids.append(original_idx.reshape(-1))

    def _append_factual_pseudo_outcomes(
        self,
        *,
        bridge: "NMMRBridge",
        data: BridgeTrainData,
        original_idx: np.ndarray,
        regressors_x: list[np.ndarray],
        pseudo_y: list[np.ndarray],
        group_ids: list[np.ndarray],
    ) -> None:
        a = data.treatment
        x = data.covariates
        w = data.outcome_proxy
        regressors_x.append(np.concatenate([a, x], axis=1))
        pseudo_y.append(bridge.bridge_predict(a, w, x))
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
        name: str = "NMMR/mean_regressor",
        seed: int = 13,
    ) -> None:
        use_treatment_grid = bool(self.mean_config.use_treatment_grid)
        grid = (
            np.linspace(self.support[0], self.support[1], self.mean_config.treatment_grid_size)
            if use_treatment_grid
            else None
        )
        reg_x = []
        reg_y = []
        group_ids = []
        n = data.treatment.shape[0]
        n_folds = int(self.mean_config.crossfit_folds)
        self.mean_crossfit_folds_ = []
        if n_folds <= 1:
            if use_treatment_grid:
                self._append_pseudo_outcomes(
                    bridge=self,
                    data=data,
                    original_idx=np.arange(n),
                    grid=grid,
                    regressors_x=reg_x,
                    pseudo_y=reg_y,
                    group_ids=group_ids,
                )
            else:
                self._append_factual_pseudo_outcomes(
                    bridge=self,
                    data=data,
                    original_idx=np.arange(n),
                    regressors_x=reg_x,
                    pseudo_y=reg_y,
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
                if verbose >= 3:
                    verbose_print(
                        verbose,
                        3,
                        f"[crossfit] {name} fold={fold_id} bridge_train_n={len(train_idx)} holdout_n={len(holdout_idx)}",
                    )
                fold_model = NMMRBridge(self.model_config, self.mean_config, self.w_flow_config, self.support)
                fold_history = fold_model._fit_bridge(
                    _subset_data(data, train_idx),
                    sample_weight=bridge_sample_weight[train_idx],
                    verbose=verbose,
                    name=f"{name}/fold_{fold_id}",
                    seed=seed + fold_id + 1,
                )
                self.training_history.extend(fold_history)
                holdout_data = _subset_data(data, holdout_idx)
                if use_treatment_grid:
                    self._append_pseudo_outcomes(
                        bridge=fold_model,
                        data=holdout_data,
                        original_idx=holdout_idx,
                        grid=grid,
                        regressors_x=reg_x,
                        pseudo_y=reg_y,
                        group_ids=group_ids,
                    )
                else:
                    self._append_factual_pseudo_outcomes(
                        bridge=fold_model,
                        data=holdout_data,
                        original_idx=holdout_idx,
                        regressors_x=reg_x,
                        pseudo_y=reg_y,
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
        reg_x = np.concatenate(reg_x, axis=0)
        reg_y = np.concatenate(reg_y, axis=0)
        reg_group_ids = np.concatenate(group_ids, axis=0)
        reg_row_weights = regressor_sample_weight[reg_group_ids]
        reg_treatment = reg_x[:, 0].reshape(-1)
        unique_treatment = np.unique(reg_treatment)
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
            "n_folds": min(max(n_folds, 1), n) if n > 0 else 1,
            "n_original": n,
            "treatment_grid_size": 0 if grid is None else int(grid.shape[0]),
            "uses_treatment_grid": use_treatment_grid,
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
            return self.mean_regressor.predict(np.concatenate([treatment, covariates], axis=1))
        if self.mean_method == "flow_mc":
            samples = self.w_flow.sample(covariates, self.mean_config.n_mc_samples)
            preds = []
            for sample_id in range(samples.shape[0]):
                preds.append(self.bridge_predict(treatment, samples[sample_id], covariates))
            return np.mean(np.stack(preds, axis=0), axis=0)
        raise RuntimeError("NMMRBridge has not been fitted.")

    def kernel_diagnostics(
        self,
        *,
        config: KernelDiagnosticsConfig,
        model_name: str = "NMMR",
        stage: str = "NMMR/bridge",
        round_id: int | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> KernelDiagnosticResult:
        if self.train_moment is None or self.azx_bandwidth is None:
            return empty_kernel_diagnostic_result()
        base_metadata = {
            "model": model_name,
            "stage": stage,
            "round": "" if round_id is None else int(round_id),
            "estimator": "NMMR",
            "global_scale": float(self.model_config.azx_scale),
            "a_scale": float(self.model_config.a_scale),
            "z_scale": float(self.model_config.z_scale),
            "x_scale": float(self.model_config.x_scale),
            "kernel": "K_AZX_product" if self.model_config.product_kernel else "K_AZX",
        }
        if self.model_config.product_kernel:
            combined_bandwidths = np.asarray(self.azx_bandwidth, dtype=float).reshape(-1)

            def diagnose(weighted: bool) -> KernelDiagnosticResult:
                results = [
                    diagnose_product_rbf_kernel(
                        features=self.train_moment,
                        bandwidths=combined_bandwidths,
                        config=config,
                        metadata={**base_metadata, "weighted": weighted, "component": "AZX"},
                        sample_weight=sample_weight if weighted else None,
                    )
                ]
                for group_name, kernel_name, group_scale in (
                    ("A", "K_A", self.model_config.a_scale),
                    ("Z", "K_Z", self.model_config.z_scale),
                    ("X", "K_X", self.model_config.x_scale),
                ):
                    group_slice = self.moment_group_slices_.get(group_name)
                    if group_slice is None:
                        continue
                    group_features = self.train_moment[:, group_slice]
                    if group_features.size == 0:
                        continue
                    group_bandwidths = combined_bandwidths[group_slice]
                    results.append(
                        diagnose_product_rbf_kernel(
                            features=group_features,
                            bandwidths=group_bandwidths,
                            config=config,
                            metadata={
                                **base_metadata,
                                "weighted": weighted,
                                "kernel": kernel_name,
                                "component": group_name,
                                "group_scale": float(group_scale),
                            },
                            sample_weight=sample_weight if weighted else None,
                        )
                    )
                return merge_kernel_diagnostic_results(results)
        else:
            diagnose = lambda weighted: diagnose_rbf_kernel(
                features=self.train_moment,
                bandwidth=float(np.asarray(self.azx_bandwidth, dtype=float).reshape(-1)[0]),
                config=config,
                metadata={**base_metadata, "weighted": weighted},
                sample_weight=sample_weight if weighted else None,
            )
        results = [diagnose(False)]
        if sample_weight is not None:
            results.append(diagnose(True))
        return merge_kernel_diagnostic_results(results)
