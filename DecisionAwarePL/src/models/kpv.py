from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import FlowConfig, KPVConfig, KernelDiagnosticsConfig, MeanConfig
from src.models.common import TorchRegressor
from src.models.kernel_diagnostics import (
    KernelDiagnosticResult,
    diagnose_product_rbf_kernel,
    empty_kernel_diagnostic_result,
    merge_kernel_diagnostic_results,
)
from src.models.kernel_utils import rbf_product_kernel
from src.utils import ArrayScaler, kfold_indices, verbose_print


@dataclass
class BridgeTrainData:
    treatment: np.ndarray
    treatment_proxy: np.ndarray
    outcome_proxy: np.ndarray
    covariates: np.ndarray
    outcome: np.ndarray

def _weighted_krr_dual(kernel, y, sample_weight, lam, jitter):
    n = kernel.shape[0]
    weights = np.asarray(sample_weight, dtype=float).reshape(-1)
    weights = np.clip(weights, 1e-12, None)

    sqrt_weight = np.sqrt(weights).reshape(-1, 1)
    weighted_kernel = kernel * (sqrt_weight @ sqrt_weight.T)

    system = weighted_kernel + n * float(lam) * np.eye(n) + float(jitter) * np.eye(n)
    beta = _solve(system, sqrt_weight * y)

    dual = sqrt_weight * beta
    return dual, weighted_kernel, sqrt_weight


def _subset_data(data: BridgeTrainData, idx: np.ndarray) -> BridgeTrainData:
    return BridgeTrainData(
        treatment=data.treatment[idx],
        treatment_proxy=data.treatment_proxy[idx],
        outcome_proxy=data.outcome_proxy[idx],
        covariates=data.covariates[idx],
        outcome=data.outcome[idx],
    )


def _as_2d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        return arr[:, None]
    return arr


def _solve(system: np.ndarray, target: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(system, target)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(system, target, rcond=None)[0]


class _IdentityScaler:
    def fit(self, x: np.ndarray) -> "_IdentityScaler":
        return self

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float)


def _make_scaler(enabled: bool):
    return ArrayScaler() if enabled else _IdentityScaler()


def _kpv_bandwidth_per_dimension(x: np.ndarray, variance_scale: float) -> np.ndarray:
    x = _as_2d(x)
    if x.shape[0] <= 1:
        return np.ones(x.shape[1], dtype=float)
    subset = x[: min(1000, x.shape[0])]
    scale = max(float(variance_scale), 1e-12)
    bandwidths = []
    for dim in range(subset.shape[1]):
        diff = np.abs(subset[:, None, dim] - subset[None, :, dim])
        median_distance = float(np.median(diff.reshape(-1)))
        if median_distance <= 1e-8:
            median_distance = 1.0
        bandwidths.append(np.sqrt(scale) * median_distance)
    return np.maximum(np.asarray(bandwidths, dtype=float), 1e-8)


def _product_kernel_from_parts(
    left_a: np.ndarray,
    right_a: np.ndarray,
    a_bandwidth: np.ndarray,
    left_x: np.ndarray,
    right_x: np.ndarray,
    x_bandwidth: np.ndarray,
) -> np.ndarray:
    return rbf_product_kernel(left_a, right_a, a_bandwidth) * rbf_product_kernel(left_x, right_x, x_bandwidth)


def _loocv_embedding_loss(kernel: np.ndarray, target_kernel: np.ndarray, lam: float, jitter: float) -> float:
    n = kernel.shape[0]
    identity = np.eye(n)
    system = kernel + n * float(lam) * identity + float(jitter) * identity
    hat = kernel @ _solve(system, identity)
    residual_hat = identity - hat
    diag = np.diag(residual_hat).copy()
    diag[np.abs(diag) < 1e-10] = np.sign(diag[np.abs(diag) < 1e-10] + 1e-12) * 1e-10
    scaled = residual_hat / diag[:, None]
    return float(np.trace(scaled @ target_kernel @ scaled.T))


def _loocv_stage2_loss(sigma: np.ndarray, y: np.ndarray, lam: float, jitter: float) -> float:
    n = sigma.shape[0]
    identity = np.eye(n)
    system = sigma + n * float(lam) * identity + float(jitter) * identity
    hat = sigma @ _solve(system, identity)
    residual_hat = identity - hat
    diag = np.diag(residual_hat).copy()
    diag[np.abs(diag) < 1e-10] = np.sign(diag[np.abs(diag) < 1e-10] + 1e-12) * 1e-10
    residual = residual_hat @ y
    scaled = residual / diag[:, None]
    return float(np.linalg.norm(scaled))


def _select_lambda(
    *,
    loss_fn,
    low: float,
    high: float,
    n_grid: int,
) -> tuple[float, float, list[dict[str, float]]]:
    low = max(float(low), 1e-12)
    high = max(float(high), low)
    n_grid = max(1, int(n_grid))
    candidates = np.logspace(np.log10(low), np.log10(high), n_grid)
    rows = []
    for lam in candidates:
        value = float(loss_fn(float(lam)))
        rows.append({"lambda": float(lam), "loss": value})
    best = min(rows, key=lambda item: item["loss"])
    return float(best["lambda"]), float(best["loss"]), rows


class KPVBridge:
    """Kernel Proxy Variable bridge with the original explicit alpha matrix."""

    def __init__(
        self,
        model_config: KPVConfig,
        mean_config: MeanConfig,
        w_flow_config: FlowConfig,
        support: tuple[float, float],
    ):
        self.model_config = model_config
        self.mean_config = mean_config
        self.w_flow_config = w_flow_config
        self.support = support
        self.a_scaler = _make_scaler(model_config.standardize)
        self.z_scaler = _make_scaler(model_config.standardize)
        self.w_scaler = _make_scaler(model_config.standardize)
        self.x_scaler = _make_scaler(model_config.standardize)
        self.y_scaler = _make_scaler(model_config.standardize)
        self.alpha: np.ndarray | None = None
        self.stage1_w: np.ndarray | None = None
        self.stage2_a: np.ndarray | None = None
        self.stage2_x: np.ndarray | None = None
        self.a_bandwidth: np.ndarray | None = None
        self.z_bandwidth: np.ndarray | None = None
        self.w_bandwidth: np.ndarray | None = None
        self.x_bandwidth: np.ndarray | None = None
        self.mean_regressor: TorchRegressor | None = None
        self.mean_method: str | None = None
        self.training_history: list[dict[str, object]] = []
        self.mean_crossfit_folds_: list[dict[str, np.ndarray | float | int]] = []
        self.mean_pseudooutcome_metadata_: dict[str, int | bool | float | str] = {}
        self.mean_regressor_group_ids_: np.ndarray | None = None
        self.mean_pseudooutcome_diagnostics_: dict[str, np.ndarray | float] = {}
        self.lambda1_: float | None = None
        self.lambda2_: float | None = None
        self.lambda1_search_: list[dict[str, float]] = []
        self.lambda2_search_: list[dict[str, float]] = []
        self.fit_indices_: np.ndarray | None = None
        self.stage1_indices_: np.ndarray | None = None
        self.stage2_indices_: np.ndarray | None = None
        self.train_azx: np.ndarray | None = None
        self.train_ax: np.ndarray | None = None

    def _cap_data(self, data: BridgeTrainData, seed: int) -> BridgeTrainData:
        n = data.treatment.shape[0]
        max_train_size = self.model_config.max_train_size
        if max_train_size is None or int(max_train_size) <= 0 or n <= int(max_train_size):
            self.fit_indices_ = np.arange(n, dtype=int)
            return data
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=int(max_train_size), replace=False)).astype(int, copy=False)
        self.fit_indices_ = idx
        return _subset_data(data, idx)

    def _stage_split(self, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        ratio = float(self.model_config.split_ratio)
        if ratio <= 0.0 or ratio >= 1.0 or n <= 2:
            idx = np.arange(n, dtype=int)
            return idx, idx
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        n_stage1 = min(max(1, int(round(n * ratio))), n - 1)
        return idx[:n_stage1].astype(int, copy=False), idx[n_stage1:].astype(int, copy=False)

    def _fit_bridge(
        self,
        data: BridgeTrainData,
        sample_weight: np.ndarray | None = None,
        *,
        verbose: int = 0,
        name: str = "KPV",
        seed: int = 13,
    ) -> list[dict[str, object]]:
        verbose_print(verbose, 1, f"[start] {name} bridge solve")
        capped = self._cap_data(data, seed)
        a_raw = _as_2d(capped.treatment)
        z_raw = _as_2d(capped.treatment_proxy)
        w_raw = _as_2d(capped.outcome_proxy)
        x_raw = _as_2d(capped.covariates)
        y_raw = _as_2d(capped.outcome)

        a = self.a_scaler.fit_transform(a_raw)
        z = self.z_scaler.fit_transform(z_raw)
        w = self.w_scaler.fit_transform(w_raw)
        x = self.x_scaler.fit_transform(x_raw)
        y = self.y_scaler.fit_transform(y_raw)

        n = a.shape[0]
        if sample_weight is None:
            bridge_weight = np.ones(n, dtype=float)
        else:
            raw_weight = np.asarray(sample_weight, dtype=float).reshape(-1)
            if raw_weight.shape[0] != data.treatment.shape[0]:
                raise ValueError("KPV sample_weight must match the number of input rows.")
            fit_idx = self.fit_indices_ if self.fit_indices_ is not None else np.arange(n, dtype=int)
            bridge_weight = raw_weight[fit_idx]
        bridge_weight = np.clip(bridge_weight, 1e-12, None)

        stage1_idx, stage2_idx = self._stage_split(n, seed)
        self.stage1_indices_ = stage1_idx.copy()
        self.stage2_indices_ = stage2_idx.copy()

        a1, z1, w1, x1 = a[stage1_idx], z[stage1_idx], w[stage1_idx], x[stage1_idx]
        a2, z2, x2, y2 = a[stage2_idx], z[stage2_idx], x[stage2_idx], y[stage2_idx]
        m1 = stage1_idx.shape[0]
        m2 = stage2_idx.shape[0]
        verbose_print(
            verbose,
            3,
            (
                f"[data] {name} bridge n_input={data.treatment.shape[0]} "
                f"n_fit={n} n_stage1={m1} n_stage2={m2} "
                f"split_ratio={self.model_config.split_ratio} "
                f"max_train_size={self.model_config.max_train_size}"
            ),
        )

        self.a_bandwidth = _kpv_bandwidth_per_dimension(a, self.model_config.kernel_variance_scale)
        self.z_bandwidth = _kpv_bandwidth_per_dimension(z, self.model_config.kernel_variance_scale)
        self.w_bandwidth = _kpv_bandwidth_per_dimension(w, self.model_config.kernel_variance_scale)
        self.x_bandwidth = _kpv_bandwidth_per_dimension(x, self.model_config.kernel_variance_scale)

        k_a_11 = rbf_product_kernel(a1, a1, self.a_bandwidth)
        k_a_12 = rbf_product_kernel(a1, a2, self.a_bandwidth)
        k_a_22 = rbf_product_kernel(a2, a2, self.a_bandwidth)
        k_z_11 = rbf_product_kernel(z1, z1, self.z_bandwidth)
        k_z_12 = rbf_product_kernel(z1, z2, self.z_bandwidth)
        k_x_11 = rbf_product_kernel(x1, x1, self.x_bandwidth)
        k_x_12 = rbf_product_kernel(x1, x2, self.x_bandwidth)
        k_x_22 = rbf_product_kernel(x2, x2, self.x_bandwidth)
        k_w_11 = rbf_product_kernel(w1, w1, self.w_bandwidth)

        k_azx_11 = k_a_11 * k_z_11 * k_x_11
        k_azx_12 = k_a_12 * k_z_12 * k_x_12
        k_ax_22 = k_a_22 * k_x_22

        if self.model_config.lambda1 is not None:
            lambda1 = float(self.model_config.lambda1)
            lambda1_loss = _loocv_embedding_loss(k_azx_11, k_w_11, lambda1, self.model_config.jitter)
        elif self.model_config.optimize_lambda1:
            lambda1, lambda1_loss, self.lambda1_search_ = _select_lambda(
                loss_fn=lambda lam: _loocv_embedding_loss(k_azx_11, k_w_11, lam, self.model_config.jitter),
                low=self.model_config.lambda1_min,
                high=self.model_config.lambda1_max,
                n_grid=self.model_config.n_lambda1_search,
            )
        else:
            lambda1 = float(self.model_config.lambda1_min)
            lambda1_loss = _loocv_embedding_loss(k_azx_11, k_w_11, lambda1, self.model_config.jitter)

        core_stage1 = k_azx_11 + m1 * lambda1 * np.eye(m1) + self.model_config.jitter * np.eye(m1)
        gamma_w = _solve(core_stage1, k_azx_12)
        kw1_gamma = k_w_11 @ gamma_w
        sigma = (gamma_w.T @ kw1_gamma) * k_ax_22

        if self.model_config.lambda2 is not None:
            lambda2 = float(self.model_config.lambda2)
            lambda2_loss = _loocv_stage2_loss(sigma, y2, lambda2, self.model_config.jitter)
        elif self.model_config.optimize_lambda2:
            lambda2, lambda2_loss, self.lambda2_search_ = _select_lambda(
                loss_fn=lambda lam: _loocv_stage2_loss(sigma, y2, lam, self.model_config.jitter),
                low=self.model_config.lambda2_min,
                high=self.model_config.lambda2_max,
                n_grid=self.model_config.n_lambda2_search,
            )
        else:
            lambda2 = float(self.model_config.lambda2_min)
            lambda2_loss = _loocv_stage2_loss(sigma, y2, lambda2, self.model_config.jitter)

        stage2_weight = bridge_weight[stage2_idx]
        stage2_vec, weighted_sigma, _ = _weighted_krr_dual(
            sigma,
            y2,
            stage2_weight,
            lambda2,
            self.model_config.jitter,
        )
        self.alpha = gamma_w * stage2_vec.reshape(1, -1)
        self.stage1_w = w1.copy()
        self.stage2_a = a2.copy()
        self.stage2_x = x2.copy()
        self.train_azx = np.concatenate([a1, z1, x1], axis=1)
        self.train_ax = np.concatenate([a2, x2], axis=1)
        self.lambda1_ = lambda1
        self.lambda2_ = lambda2

        fitted = self.bridge_predict(capped.treatment, capped.outcome_proxy, capped.covariates).reshape(-1, 1)
        factual_mse = float(np.mean(np.square(fitted - y_raw)))
        history = [
            {
                "stage": f"{name}/stage1_loocv",
                "train_losses": [float(lambda1_loss)],
                "val_losses": [float(lambda1_loss)],
                "best_epoch": 0,
                "stopped_epoch": 0,
                "metadata": {"lambda1": float(lambda1), "n_stage1": int(m1)},
            },
            {
                "stage": f"{name}/stage2_loocv",
                "train_losses": [float(lambda2_loss)],
                "val_losses": [float(lambda2_loss)],
                "best_epoch": 0,
                "stopped_epoch": 0,
                "metadata": {"lambda2": float(lambda2), "n_stage2": int(m2)},
            },
            {
                "stage": f"{name}/bridge",
                "train_losses": [factual_mse],
                "val_losses": [factual_mse],
                "best_epoch": 0,
                "stopped_epoch": 0,
                "metadata": {
                    "lambda1": float(lambda1),
                    "lambda2": float(lambda2),
                    "n_fit": int(n),
                    "n_stage1": int(m1),
                    "n_stage2": int(m2),
                },
            },
        ]
        verbose_print(
            verbose,
            1,
            f"[done] {name} bridge solve lambda1={lambda1:.4g} lambda2={lambda2:.4g} factual_mse={factual_mse:.4f}",
        )
        return history

    def kernel_diagnostics(
        self,
        *,
        config: KernelDiagnosticsConfig,
        model_name: str = "KPV",
        stage: str = "KPV/bridge",
        round_id: int | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> KernelDiagnosticResult:
        if (
            self.train_azx is None
            or self.train_ax is None
            or self.stage1_w is None
            or self.a_bandwidth is None
            or self.z_bandwidth is None
            or self.x_bandwidth is None
            or self.w_bandwidth is None
        ):
            return empty_kernel_diagnostic_result()
        base_metadata = {
            "model": model_name,
            "stage": stage,
            "round": "" if round_id is None else int(round_id),
            "estimator": "KPV",
        }
        results = [
            diagnose_product_rbf_kernel(
                features=self.train_azx,
                bandwidths=np.concatenate([self.a_bandwidth, self.z_bandwidth, self.x_bandwidth]),
                config=config,
                metadata={**base_metadata, "kernel": "K_AZX_stage1", "weighted": False},
            ),
            diagnose_product_rbf_kernel(
                features=self.stage1_w,
                bandwidths=self.w_bandwidth,
                config=config,
                metadata={**base_metadata, "kernel": "K_W_stage1", "weighted": False},
            ),
            diagnose_product_rbf_kernel(
                features=self.train_ax,
                bandwidths=np.concatenate([self.a_bandwidth, self.x_bandwidth]),
                config=config,
                metadata={**base_metadata, "kernel": "K_AX_stage2", "weighted": False},
            ),
        ]
        if sample_weight is not None:
            results.append(
                diagnose_product_rbf_kernel(
                    features=self.train_azx,
                    bandwidths=np.concatenate([self.a_bandwidth, self.z_bandwidth, self.x_bandwidth]),
                    config=config,
                    metadata={**base_metadata, "kernel": "K_AZX_stage1", "weighted": True},
                    sample_weight=sample_weight,
                )
            )
        return merge_kernel_diagnostic_results(results)

    def fit(
        self,
        data: BridgeTrainData,
        sample_weight: np.ndarray | None = None,
        mean_method: str = "pseudo_outcomes",
        *,
        weight_mean_regressor: bool = True,
        warm_start: bool = False,
        verbose: int = 0,
        name: str = "KPV",
        seed: int = 13,
    ) -> None:
        n = data.treatment.shape[0]
        if sample_weight is None:
            sample_weight = np.ones(n, dtype=float)
        sample_weight = np.asarray(sample_weight, dtype=float).reshape(-1)
        self.training_history = self._fit_bridge(
            data,
            sample_weight=sample_weight,
            verbose=verbose,
            name=name,
            seed=seed,
        )
        self.mean_method = mean_method
        if mean_method != "pseudo_outcomes":
            raise ValueError("KPVBridge currently supports mean_method='pseudo_outcomes'.")
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

    def bridge_predict(self, treatment: np.ndarray, outcome_proxy: np.ndarray, covariates: np.ndarray) -> np.ndarray:
        if (
            self.alpha is None
            or self.stage1_w is None
            or self.stage2_a is None
            or self.stage2_x is None
            or self.a_bandwidth is None
            or self.w_bandwidth is None
            or self.x_bandwidth is None
        ):
            raise RuntimeError("KPVBridge has not been fitted.")
        a = self.a_scaler.transform(_as_2d(treatment))
        w = self.w_scaler.transform(_as_2d(outcome_proxy))
        x = self.x_scaler.transform(_as_2d(covariates))
        n = a.shape[0]
        batch_size = max(1, int(self.model_config.prediction_batch_size))
        preds = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            k_w = rbf_product_kernel(w[start:end], self.stage1_w, self.w_bandwidth)
            k_ax = _product_kernel_from_parts(
                self.stage2_a,
                a[start:end],
                self.a_bandwidth,
                self.stage2_x,
                x[start:end],
                self.x_bandwidth,
            )
            left = k_w @ self.alpha
            pred_scaled = np.sum(left * k_ax.T, axis=1, keepdims=True)
            preds.append(pred_scaled)
        scaled = np.vstack(preds)
        return self.y_scaler.inverse_transform(scaled)

    def _append_pseudo_outcomes(
        self,
        *,
        bridge: "KPVBridge",
        data: BridgeTrainData,
        original_idx: np.ndarray,
        regressors_x: list[np.ndarray],
        pseudo_y: list[np.ndarray],
        group_ids: list[np.ndarray],
        grid: np.ndarray | None,
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
        *,
        weight_mean_regressor: bool = True,
        warm_start: bool = False,
        verbose: int = 0,
        name: str = "KPV/mean_regressor",
        seed: int = 13,
    ) -> None:
        n = data.treatment.shape[0]
        bridge_sample_weight = np.asarray(bridge_sample_weight, dtype=float).reshape(-1)
        regressor_sample_weight = np.asarray(regressor_sample_weight, dtype=float).reshape(-1)
        if bool(self.mean_config.use_treatment_grid):
            grid = np.linspace(self.support[0], self.support[1], self.mean_config.treatment_grid_size)
        else:
            grid = None
        regressors_x: list[np.ndarray] = []
        pseudo_y: list[np.ndarray] = []
        group_ids: list[np.ndarray] = []
        n_folds = int(self.mean_config.crossfit_folds)
        self.mean_crossfit_folds_ = []
        if n_folds <= 1:
            self._append_pseudo_outcomes(
                bridge=self,
                data=data,
                original_idx=np.arange(n),
                regressors_x=regressors_x,
                pseudo_y=pseudo_y,
                group_ids=group_ids,
                grid=grid,
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
                fold_model = KPVBridge(self.model_config, self.mean_config, self.w_flow_config, self.support)
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
                    regressors_x=regressors_x,
                    pseudo_y=pseudo_y,
                    group_ids=group_ids,
                    grid=grid,
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
        unique_treatment = np.unique(reg_treatment) if grid is not None else np.array([], dtype=float)
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
            "weight_mean_regressor": bool(weight_mean_regressor),
            "warm_start": bool(warm_start),
            "uses_treatment_grid": bool(grid is not None),
            "n_folds": min(max(n_folds, 1), n) if n > 0 else 1,
            "n_original": n,
            "treatment_grid_size": int(0 if grid is None else grid.shape[0]),
            "n_rows": int(reg_x.shape[0]),
            "pseudo_outcome_std": float(self.mean_pseudooutcome_diagnostics_["std"]),
            "pseudo_outcome_iqr": float(self.mean_pseudooutcome_diagnostics_["iqr"]),
        }
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
        if self.mean_method != "pseudo_outcomes" or self.mean_regressor is None:
            raise RuntimeError("KPVBridge has not been fitted with a mean regressor.")
        features = np.concatenate([_as_2d(treatment), _as_2d(covariates)], axis=1)
        return self.mean_regressor.predict(features)
