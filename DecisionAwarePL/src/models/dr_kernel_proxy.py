from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from src.config import DRKPVConfig, DRPMMRConditionalConfig, DRPMMRMarginalConfig, KernelDiagnosticsConfig, MeanConfig, PMMRConfig
from src.models.kernel_diagnostics import KernelDiagnosticResult, empty_kernel_diagnostic_result
from src.models.kernel_utils import median_bandwidth, rbf_kernel, rbf_product_kernel
from src.models.pmmr import BridgeTrainData as PMMRTrainData
from src.models.pmmr import PMMRBridge
from src.utils import ArrayScaler, verbose_print


@dataclass
class BridgeTrainData:
    treatment: np.ndarray
    treatment_proxy: np.ndarray
    outcome_proxy: np.ndarray
    covariates: np.ndarray
    outcome: np.ndarray


def _as_2d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        return arr[:, None]
    return arr


def _grid_as_2d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr[:, None]
    return _as_2d(arr)


def _solve(system: np.ndarray, target: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(system, target)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(system, target, rcond=None)[0]


@dataclass
class _SystemFactor:
    system: np.ndarray
    cho: tuple[np.ndarray, bool] | None = None


def _factor_system(system: np.ndarray, jitter: float = 0.0) -> _SystemFactor:
    arr = _make_psd(system, jitter)
    try:
        return _SystemFactor(system=arr, cho=cho_factor(arr, lower=True, check_finite=False))
    except Exception:
        return _SystemFactor(system=arr, cho=None)


def _solve_factorized(factor: _SystemFactor, target: np.ndarray) -> np.ndarray:
    if factor.cho is not None:
        try:
            return cho_solve(factor.cho, target, check_finite=False)
        except Exception:
            pass
    return _solve(factor.system, target)


def _make_psd(matrix: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        arr = 0.5 * (arr + arr.T)
        arr = arr + float(eps) * np.eye(arr.shape[0])
    return arr


def _ix(kernel: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return kernel[np.ix_(left, right)]


def _remove_diagonal_ones(n: int) -> np.ndarray:
    return np.ones((n, n), dtype=float) - np.eye(n, dtype=float)


def _log_grid(low: float, high: float, n_grid: int) -> np.ndarray:
    low = max(float(low), 1e-12)
    high = max(float(high), low)
    n_grid = max(1, int(n_grid))
    return np.logspace(np.log10(low), np.log10(high), n_grid)


class _IdentityScaler:
    def fit(self, x: np.ndarray) -> "_IdentityScaler":
        return self

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float)


def _make_scaler(enabled: bool):
    return ArrayScaler() if enabled else _IdentityScaler()


class _FrozenRBFKernel:
    def __init__(self, *, variance_scale: float = 1.0):
        self.variance_scale = max(float(variance_scale), 1e-12)
        self.bandwidth_: float | None = None
        self.fit_data_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "_FrozenRBFKernel":
        x = _as_2d(x)
        self.fit_data_ = x.copy()
        scale = float(np.sqrt(self.variance_scale))
        self.bandwidth_ = max(float(median_bandwidth(x, scale=scale)), 1e-6)
        return self

    def __call__(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        if self.bandwidth_ is None:
            raise RuntimeError("Kernel bandwidth has not been fitted.")
        return rbf_kernel(_as_2d(x1), _as_2d(x2), self.bandwidth_)


def _loocv_kernel_loss(
    kernel: np.ndarray,
    target_kernel: np.ndarray,
    lam: float,
    jitter: float,
    *,
    label_variance: float = 0.0,
) -> float:
    n = kernel.shape[0]
    identity = np.eye(n)
    system = _make_psd(kernel + n * float(lam) * identity, jitter)
    smoother = _solve(system, kernel).T
    residual_hat = identity - smoother
    diag = np.diag(residual_hat).copy()
    small = np.abs(diag) < 1e-10
    diag[small] = np.sign(diag[small] + 1e-12) * 1e-10
    scaled = residual_hat / diag[:, None]
    objective = float(np.trace(scaled @ target_kernel @ scaled.T) / max(n, 1))
    if label_variance:
        inv_diag = np.diag(1.0 / diag)
        objective += float(label_variance) * float(np.trace(smoother))
        objective += float(label_variance) * float(np.sum((diag - 1.0) / diag) / max(n, 1))
        objective += float(label_variance) * float(np.trace(smoother @ inv_diag @ smoother.T) / max(n, 1))
    return objective


def _loocv_second_stage_loss(kernel: np.ndarray, target_kernel: np.ndarray, lam: float, jitter: float) -> float:
    n = kernel.shape[0]
    identity = np.eye(n)
    system = _make_psd(kernel + n * float(lam) * identity, jitter)
    smoother = _solve(system, kernel).T
    residual_hat = identity - smoother
    diag = np.diag(residual_hat).copy()
    small = np.abs(diag) < 1e-10
    diag[small] = np.sign(diag[small] + 1e-12) * 1e-10
    scaled = residual_hat / diag[:, None]
    return float(np.trace(scaled @ target_kernel @ scaled.T) / max(n, 1))


def _zeta_objective(zeta: float, k_vv: np.ndarray, target_kernel: np.ndarray, jitter: float) -> float:
    n = k_vv.shape[0]
    smoother = _solve(_make_psd(k_vv + n * float(zeta) * np.eye(n), jitter), k_vv).T
    diag = 1.0 - np.diag(smoother)
    small = np.abs(diag) < 1e-10
    diag[small] = np.sign(diag[small] + 1e-12) * 1e-10
    scale = np.diag(1.0 / np.square(diag))
    residual_kernel = target_kernel - 2.0 * target_kernel @ smoother.T + smoother @ target_kernel @ smoother.T
    return float(np.trace(scale @ residual_kernel))


def _select_from_grid(grid: np.ndarray, loss_fn) -> tuple[float, float, list[dict[str, float]]]:
    rows = []
    for value in grid:
        loss = float(loss_fn(float(value)))
        rows.append({"lambda": float(value), "loss": loss})
    best = min(rows, key=lambda row: row["loss"])
    return float(best["lambda"]), float(best["loss"]), rows


def _now() -> float:
    return time.perf_counter()


def _log_elapsed(verbose: int, level: int, name: str, message: str, start: float) -> None:
    verbose_print(verbose, level, f"[dr-kernel] {name} {message} elapsed={_now() - start:.2f}s")


def _nan_loss() -> float:
    return float("nan")


def _prediction_verbose(verbose: int, n: int) -> int:
    return int(verbose) if int(n) >= 2048 else 0


def _subset_data(data: BridgeTrainData, idx: np.ndarray) -> BridgeTrainData:
    return BridgeTrainData(
        treatment=data.treatment[idx],
        treatment_proxy=data.treatment_proxy[idx],
        outcome_proxy=data.outcome_proxy[idx],
        covariates=data.covariates[idx],
        outcome=data.outcome[idx],
    )


class _KernelProxyVariableCATE:
    def __init__(
        self,
        *,
        kernel_a: _FrozenRBFKernel,
        kernel_w: _FrozenRBFKernel,
        kernel_z: _FrozenRBFKernel,
        kernel_v: _FrozenRBFKernel,
        kernel_x: _FrozenRBFKernel | None,
        lambda1: float,
        lambda2: float,
        zeta: float,
        optimize_internal: bool,
        lambda_min: float,
        lambda_max: float,
        regularization_grid_points: int,
        stage1_perc: float,
        jitter: float,
    ):
        self.kernel_a = kernel_a
        self.kernel_w = kernel_w
        self.kernel_z = kernel_z
        self.kernel_v = kernel_v
        self.kernel_x = kernel_x
        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)
        self.zeta = float(zeta)
        self.optimize_internal = bool(optimize_internal)
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.regularization_grid_points = int(regularization_grid_points)
        self.stage1_perc = float(stage1_perc)
        self.jitter = float(jitter)
        self.training_history: list[dict[str, object]] = []
        self.lambda1_search_: list[dict[str, float]] = []
        self.lambda2_search_: list[dict[str, float]] = []
        self.zeta_search_: list[dict[str, float]] = []

    def _stage_split(self, n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        train_indices = rng.permutation(n).astype(int, copy=False)
        if 0.0 < self.stage1_perc < 1.0 and n > 2:
            n_stage1 = min(max(1, int(n * self.stage1_perc)), n - 1)
            return train_indices, train_indices[:n_stage1], train_indices[n_stage1:]
        return train_indices, train_indices, train_indices

    def fit(
        self,
        *,
        a: np.ndarray,
        w: np.ndarray,
        z: np.ndarray,
        v: np.ndarray,
        x_extra: np.ndarray | None,
        y: np.ndarray,
        seed: int,
        name: str,
        verbose: int = 0,
    ) -> None:
        total_start = _now()
        a, w, z, v, y = _as_2d(a), _as_2d(w), _as_2d(z), _as_2d(v), _as_2d(y)
        x_extra = None if x_extra is None else _as_2d(x_extra)
        n = a.shape[0]
        train_idx, stage1_idx, stage2_idx = self._stage_split(n, seed)
        n1, n2 = stage1_idx.shape[0], stage2_idx.shape[0]
        verbose_print(
            verbose,
            3,
            f"[dr-kernel] {name}/outcome_kpv start n={n} n_stage1={n1} n_stage2={n2}",
        )

        step_start = _now()
        k_a = self.kernel_a(a, a)
        k_w = self.kernel_w(w, w)
        k_z = self.kernel_z(z, z)
        k_v = self.kernel_v(v, v)
        k_x = np.ones((n, n), dtype=float) if x_extra is None else self.kernel_x(x_extra, x_extra)
        _log_elapsed(verbose, 3, f"{name}/outcome_kpv", "built full kernels", step_start)

        step_start = _now()
        k_aa = _ix(k_a, stage1_idx, stage1_idx)
        k_a_tilde = _ix(k_a, stage1_idx, stage2_idx)
        k_tilde_tilde_a = _ix(k_a, stage2_idx, stage2_idx)
        k_vv = _ix(k_v, stage1_idx, stage1_idx)
        k_v_tilde = _ix(k_v, stage1_idx, stage2_idx)
        k_tilde_tilde_v = _ix(k_v, stage2_idx, stage2_idx)
        k_ww = _ix(k_w, stage1_idx, stage1_idx)
        k_w_tilde = _ix(k_w, stage1_idx, stage2_idx)
        k_tilde_tilde_w = _ix(k_w, stage2_idx, stage2_idx)
        k_zz = _ix(k_z, stage1_idx, stage1_idx)
        k_z_tilde = _ix(k_z, stage1_idx, stage2_idx)
        k_xx = _ix(k_x, stage1_idx, stage1_idx)
        k_x_tilde = _ix(k_x, stage1_idx, stage2_idx)
        k_tilde_tilde_x = _ix(k_x, stage2_idx, stage2_idx)
        _log_elapsed(verbose, 3, f"{name}/outcome_kpv", "sliced stage kernels", step_start)

        k_zavx = k_zz * k_aa * k_vv * k_xx
        if self.optimize_internal:
            step_start = _now()
            grid = _log_grid(self.lambda_min, self.lambda_max, self.regularization_grid_points)
            self.lambda1, lambda1_loss, self.lambda1_search_ = _select_from_grid(
                grid,
                lambda lam: _loocv_kernel_loss(k_zavx, k_ww, lam, self.jitter),
            )
            _log_elapsed(verbose, 3, f"{name}/outcome_kpv", "selected lambda1", step_start)
        else:
            lambda1_loss = _nan_loss()

        step_start = _now()
        stage1_system = _make_psd(k_zavx + n1 * self.lambda1 * np.eye(n1), self.jitter)
        b = _solve(stage1_system, k_z_tilde * k_a_tilde * k_v_tilde * k_x_tilde)
        stage2_core = k_tilde_tilde_a * (b.T @ k_ww @ b) * k_tilde_tilde_v * k_tilde_tilde_x
        _log_elapsed(verbose, 3, f"{name}/outcome_kpv", f"solved stage1 system shape={stage1_system.shape}", step_start)
        y2 = y[stage2_idx]
        if self.optimize_internal:
            step_start = _now()
            grid = _log_grid(self.lambda_min, self.lambda_max, self.regularization_grid_points)
            self.lambda2, lambda2_loss, self.lambda2_search_ = _select_from_grid(
                grid,
                lambda lam: _loocv_second_stage_loss(stage2_core, y2 @ y2.T, lam, self.jitter),
            )
            _log_elapsed(verbose, 3, f"{name}/outcome_kpv", "selected lambda2", step_start)
        else:
            lambda2_loss = _nan_loss()

        step_start = _now()
        stage2_system = _make_psd(stage2_core + n2 * self.lambda2 * np.eye(n2), self.jitter)
        self.alpha = _solve(stage2_system, y2)
        _log_elapsed(verbose, 3, f"{name}/outcome_kpv", f"solved stage2 system shape={stage2_system.shape}", step_start)

        if self.optimize_internal:
            step_start = _now()
            grid = _log_grid(self.lambda_min, self.lambda_max, self.regularization_grid_points)
            self.zeta, zeta_loss, self.zeta_search_ = _select_from_grid(
                grid,
                lambda lam: _zeta_objective(lam, k_tilde_tilde_v, k_tilde_tilde_x * k_tilde_tilde_w, self.jitter),
            )
            _log_elapsed(verbose, 3, f"{name}/outcome_kpv", "selected zeta", step_start)
        else:
            zeta_loss = _nan_loss()

        self.b = b
        self.a_tilde = a[stage2_idx]
        self.v_tilde = v[stage2_idx]
        self.w_stage1 = w[stage1_idx]
        self.x_tilde = None if x_extra is None else x_extra[stage2_idx]
        self.k_x_tilde_tilde = k_tilde_tilde_x
        self.k_w_tilde = k_w_tilde
        self.k_v_tilde_tilde = k_tilde_tilde_v
        self.predict_core = self.k_x_tilde_tilde * (self.b.T @ self.k_w_tilde)
        self.v_cme_factor = _factor_system(
            self.k_v_tilde_tilde + n2 * self.zeta * np.eye(n2),
            self.jitter,
        )
        self.stage1_idx = stage1_idx
        self.stage2_idx = stage2_idx
        self.train_idx = train_idx
        self.training_history = [
            {
                "stage": f"{name}/outcome_kpv_stage1_loocv",
                "train_losses": [float(lambda1_loss)],
                "val_losses": [float(lambda1_loss)],
                "best_epoch": 0,
                "stopped_epoch": 0,
                "metadata": {"lambda1": float(self.lambda1), "n_stage1": int(n1)},
            },
            {
                "stage": f"{name}/outcome_kpv_stage2_loocv",
                "train_losses": [float(lambda2_loss)],
                "val_losses": [float(lambda2_loss)],
                "best_epoch": 0,
                "stopped_epoch": 0,
                "metadata": {"lambda2": float(self.lambda2), "n_stage2": int(n2)},
            },
            {
                "stage": f"{name}/outcome_kpv_zeta_loocv",
                "train_losses": [float(zeta_loss)],
                "val_losses": [float(zeta_loss)],
                "best_epoch": 0,
                "stopped_epoch": 0,
                "metadata": {"zeta": float(self.zeta), "n_stage2": int(n2)},
            },
            {
                "stage": f"{name}/outcome_kpv",
                "train_losses": [],
                "val_losses": [],
                "best_epoch": 0,
                "stopped_epoch": 0,
                "metadata": {"fit_mse_computed": False},
            },
        ]
        _log_elapsed(verbose, 3, f"{name}/outcome_kpv", "finished", total_start)

    def predict(self, a: np.ndarray, v: np.ndarray) -> np.ndarray:
        a, v = _as_2d(a), _as_2d(v)
        k_a = self.kernel_a(self.a_tilde, a)
        k_v = self.kernel_v(self.v_tilde, v)
        n2 = self.a_tilde.shape[0]
        if hasattr(self, "v_cme_factor"):
            cme_weights = _solve_factorized(self.v_cme_factor, k_v)
        else:
            cme_weights = _solve(
                _make_psd(self.k_v_tilde_tilde + n2 * self.zeta * np.eye(n2), self.jitter),
                k_v,
            )
        core = getattr(self, "predict_core", None)
        if core is None:
            core = self.k_x_tilde_tilde * (self.b.T @ self.k_w_tilde)
        av = k_a * k_v
        pred = self.alpha.T @ (av * (core @ cme_weights))
        return pred.T

    def bridge_func(self, a: np.ndarray, w: np.ndarray, v: np.ndarray, x_extra: np.ndarray | None = None) -> np.ndarray:
        a, w, v = _as_2d(a), _as_2d(w), _as_2d(v)
        x_extra = None if x_extra is None else _as_2d(x_extra)
        k_a = self.kernel_a(self.a_tilde, a)
        k_v = self.kernel_v(self.v_tilde, v)
        k_w = self.kernel_w(self.w_stage1, w)
        if x_extra is not None and self.x_tilde is not None:
            k_x = self.kernel_x(self.x_tilde, x_extra)
        else:
            k_x = np.ones((self.a_tilde.shape[0], w.shape[0]), dtype=float)
        core = (self.b.T @ k_w) * k_x
        weighted_av = np.asarray(self.alpha, dtype=float).reshape(-1, 1) * (k_a * k_v)
        return (core.T @ weighted_av).T


class _KernelAlternativeProxyCATE:
    def __init__(
        self,
        *,
        kernel_a: _FrozenRBFKernel,
        kernel_w: _FrozenRBFKernel,
        kernel_z: _FrozenRBFKernel,
        kernel_v: _FrozenRBFKernel,
        kernel_x: _FrozenRBFKernel | None,
        lambda_: float,
        eta: float,
        lambda2: float,
        zeta: float,
        optimize_internal: bool,
        lambda_min: float,
        lambda_max: float,
        regularization_grid_points: int,
        stage1_perc: float,
        jitter: float,
        label_variance_in_lambda_opt: float,
        label_variance_in_eta_opt: float,
    ):
        self.kernel_a = kernel_a
        self.kernel_w = kernel_w
        self.kernel_z = kernel_z
        self.kernel_v = kernel_v
        self.kernel_x = kernel_x
        self.lambda_ = float(lambda_)
        self.eta = float(eta)
        self.lambda2 = float(lambda2)
        self.zeta = float(zeta)
        self.optimize_internal = bool(optimize_internal)
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.regularization_grid_points = int(regularization_grid_points)
        self.stage1_perc = float(stage1_perc)
        self.jitter = float(jitter)
        self.label_variance_in_lambda_opt = float(label_variance_in_lambda_opt)
        self.label_variance_in_eta_opt = float(label_variance_in_eta_opt)
        self.training_history: list[dict[str, object]] = []
        self.lambda_search_: list[dict[str, float]] = []
        self.eta_search_: list[dict[str, float]] = []
        self.lambda2_search_: list[dict[str, float]] = []
        self.zeta_search_: list[dict[str, float]] = []

    def _stage_split(self, n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        train_indices = rng.permutation(n).astype(int, copy=False)
        if 0.0 < self.stage1_perc < 1.0 and n > 2:
            n_stage1 = min(max(1, int(n * self.stage1_perc)), n - 1)
            return train_indices, train_indices[:n_stage1], train_indices[n_stage1:]
        return train_indices, train_indices, train_indices

    def fit(
        self,
        *,
        a: np.ndarray,
        w: np.ndarray,
        z: np.ndarray,
        v: np.ndarray,
        x_extra: np.ndarray | None,
        y: np.ndarray,
        seed: int,
        name: str,
        verbose: int = 0,
    ) -> None:
        total_start = _now()
        a, w, z, v, y = _as_2d(a), _as_2d(w), _as_2d(z), _as_2d(v), _as_2d(y)
        x_extra = None if x_extra is None else _as_2d(x_extra)
        n = a.shape[0]
        train_idx, stage1_idx, stage2_idx = self._stage_split(n, seed)
        n1, n2 = stage1_idx.shape[0], stage2_idx.shape[0]
        verbose_print(verbose, 3, f"[dr-kernel] {name}/kap start n={n} n_stage1={n1} n_stage2={n2}")

        step_start = _now()
        k_a = self.kernel_a(a, a)
        k_w = self.kernel_w(w, w)
        k_z = self.kernel_z(z, z)
        k_v = self.kernel_v(v, v)
        k_x = np.ones((n, n), dtype=float) if x_extra is None else self.kernel_x(x_extra, x_extra)
        _log_elapsed(verbose, 3, f"{name}/kap", "built full kernels", step_start)

        step_start = _now()
        k_aa = _ix(k_a, stage1_idx, stage1_idx)
        k_a_tilde = _ix(k_a, stage1_idx, stage2_idx)
        k_tilde_a = k_a_tilde.T
        k_tilde_tilde_a = _ix(k_a, stage2_idx, stage2_idx)
        k_ww = _ix(k_w, stage1_idx, stage1_idx)
        k_w_tilde = _ix(k_w, stage1_idx, stage2_idx)
        k_tilde_tilde_w = _ix(k_w, stage2_idx, stage2_idx)
        k_zz = _ix(k_z, stage1_idx, stage1_idx)
        k_vv = _ix(k_v, stage1_idx, stage1_idx)
        k_v_tilde = _ix(k_v, stage1_idx, stage2_idx)
        k_tilde_v = k_v_tilde.T
        k_tilde_tilde_v = _ix(k_v, stage2_idx, stage2_idx)
        k_xx = _ix(k_x, stage1_idx, stage1_idx)
        k_x_tilde = _ix(k_x, stage1_idx, stage2_idx)
        k_tilde_tilde_x = _ix(k_x, stage2_idx, stage2_idx)
        _log_elapsed(verbose, 3, f"{name}/kap", "sliced stage kernels", step_start)

        k_awvx = k_aa * k_ww * k_vv * k_xx
        if self.optimize_internal:
            step_start = _now()
            grid = _log_grid(self.lambda_min, self.lambda_max, self.regularization_grid_points)
            self.lambda_, lambda_loss, self.lambda_search_ = _select_from_grid(
                grid,
                lambda lam: _loocv_kernel_loss(
                    k_awvx,
                    k_zz,
                    lam,
                    self.jitter,
                    label_variance=self.label_variance_in_lambda_opt,
                ),
            )
            self.zeta, zeta_loss, self.zeta_search_ = _select_from_grid(
                grid,
                lambda lam: _zeta_objective(lam, k_tilde_tilde_v, k_tilde_tilde_w * k_tilde_tilde_x, self.jitter),
            )
            _log_elapsed(verbose, 3, f"{name}/kap", "selected lambda and zeta", step_start)
        else:
            lambda_loss = _nan_loss()
            zeta_loss = _nan_loss()

        step_start = _now()
        stage1_system = _make_psd(k_awvx + n1 * self.lambda_ * np.eye(n1), self.jitter)
        theta = _solve(_make_psd(k_tilde_tilde_v + n2 * self.zeta * np.eye(n2), self.jitter), k_tilde_tilde_v)
        offdiag_m = _remove_diagonal_ones(n2)
        b = _solve(stage1_system, k_w_tilde * k_x_tilde * k_a_tilde * k_v_tilde)
        b_bar = _solve(stage1_system, ((k_w_tilde * k_x_tilde) @ (theta * offdiag_m)) * (k_a_tilde * k_v_tilde))
        _log_elapsed(verbose, 3, f"{name}/kap", f"solved first-stage systems shape={stage1_system.shape}", step_start)

        step_start = _now()
        block1 = (b.T @ k_zz @ b) * k_tilde_tilde_a * k_tilde_tilde_v
        block2 = (b.T @ k_zz @ b_bar) * k_tilde_tilde_a * k_tilde_tilde_v
        block4 = (b_bar.T @ k_zz @ b_bar) * k_tilde_tilde_a * k_tilde_tilde_v
        ones_m = np.ones(n2, dtype=float) / max(n2, 1)
        l_sub = np.vstack((block1, ones_m.T @ block2.T))
        l_mat = l_sub @ l_sub.T
        m_vec = np.vstack(
            (
                (block2 @ ones_m).reshape(-1, 1),
                np.asarray([[float(ones_m.T @ block4 @ ones_m)]]),
            )
        )
        p = np.hstack((block1, (block2 @ ones_m).reshape(-1, 1)))
        r = np.hstack(
            (
                (ones_m.T @ block2.T).reshape(1, -1),
                np.asarray([[float(ones_m.T @ block4 @ ones_m)]]),
            )
        )
        n_mat = np.vstack((p, r))
        _log_elapsed(verbose, 3, f"{name}/kap", f"built eta system shape={n_mat.shape}", step_start)

        if self.optimize_internal:
            step_start = _now()
            theta2 = _solve(_make_psd(k_vv + n1 * self.zeta * np.eye(n1), self.jitter), k_vv)
            offdiag_n = _remove_diagonal_ones(n1)
            b2 = _solve(stage1_system, k_awvx)
            b2_bar = _solve(stage1_system, ((k_ww * k_xx) @ (theta2 * offdiag_n)) * (k_aa * k_vv))
            ones_n = np.ones(n1, dtype=float) / max(n1, 1)
            block12 = (b2.T @ k_zz @ b) * k_a_tilde * k_v_tilde
            block22 = (b2.T @ k_zz @ b_bar) * k_a_tilde * k_v_tilde
            block32 = (b.T @ k_zz @ b2_bar) * k_tilde_a * k_tilde_v
            block42 = (b_bar.T @ k_zz @ b2_bar) * k_tilde_a * k_tilde_v
            l2_sub = np.vstack((block12.T, ones_m.T @ block22.T))
            l2 = l2_sub @ l2_sub.T
            m2 = np.vstack(
                (
                    (block32 @ ones_n).reshape(-1, 1),
                    np.asarray([[float(ones_m.T @ block42 @ ones_n)]]),
                )
            )
            grid = _log_grid(self.lambda_min, self.lambda_max, self.regularization_grid_points)
            self.eta, eta_loss, self.eta_search_ = _select_from_grid(
                grid,
                lambda lam: self._eta_objective(lam, l_mat, l_sub, m_vec, n_mat, l2, m2, n1),
            )
            _log_elapsed(verbose, 3, f"{name}/kap", "selected eta", step_start)
        else:
            eta_loss = _nan_loss()

        step_start = _now()
        self.alpha = _solve(_make_psd(l_mat / max(n2, 1) + self.eta * n_mat, self.jitter), m_vec)
        _log_elapsed(verbose, 3, f"{name}/kap", f"solved alpha system shape={l_mat.shape}", step_start)

        if self.optimize_internal:
            step_start = _now()
            grid = _log_grid(self.lambda_min, self.lambda_max, self.regularization_grid_points)
            k_av_full = k_a * k_v
            k_zw_target = k_z * (y @ y.T)
            self.lambda2, lambda2_loss, self.lambda2_search_ = _select_from_grid(
                grid,
                lambda lam: _loocv_kernel_loss(
                    k_av_full,
                    k_zw_target,
                    lam,
                    self.jitter,
                    label_variance=self.label_variance_in_lambda_opt,
                ),
            )
            _log_elapsed(verbose, 3, f"{name}/kap", "selected third-stage lambda2", step_start)
        else:
            lambda2_loss = _nan_loss()

        step_start = _now()
        k_z_stage1_train = _ix(k_z, stage1_idx, train_idx)
        k_av_train = _ix(k_a, train_idx, train_idx) * _ix(k_v, train_idx, train_idx)
        target = k_z_stage1_train.T * y[train_idx]
        self.third_stage_weights = _solve(
            _make_psd(k_av_train + n * self.lambda2 * np.eye(n), self.jitter),
            target,
        ).T
        self.b = b
        self.b_bar = b_bar
        self.predict_stage1_map = self.b.T @ self.third_stage_weights
        self.predict_stage1_bar_map = self.b_bar.T @ self.third_stage_weights
        _log_elapsed(verbose, 3, f"{name}/kap", f"solved third-stage system shape={k_av_train.shape}", step_start)
        self.ones_m = ones_m
        self.a_train = a
        self.v_train = v
        self.z_train = z
        self.x_train = x_extra
        self.a_stage2 = a[stage2_idx]
        self.v_stage2 = v[stage2_idx]
        self.z_stage1 = z[stage1_idx]
        self.stage1_idx = stage1_idx
        self.stage2_idx = stage2_idx
        self.train_idx = train_idx
        self.training_history = [
            {
                "stage": f"{name}/kap_stage1_loocv",
                "train_losses": [float(lambda_loss)],
                "val_losses": [float(lambda_loss)],
                "best_epoch": 0,
                "stopped_epoch": 0,
                "metadata": {"lambda": float(self.lambda_), "n_stage1": int(n1)},
            },
            {
                "stage": f"{name}/kap_eta",
                "train_losses": [float(eta_loss)],
                "val_losses": [float(eta_loss)],
                "best_epoch": 0,
                "stopped_epoch": 0,
                "metadata": {"eta": float(self.eta), "n_stage2": int(n2)},
            },
            {
                "stage": f"{name}/kap_zeta",
                "train_losses": [float(zeta_loss)],
                "val_losses": [float(zeta_loss)],
                "best_epoch": 0,
                "stopped_epoch": 0,
                "metadata": {"zeta": float(self.zeta), "n_stage2": int(n2)},
            },
            {
                "stage": f"{name}/kap_third_stage",
                "train_losses": [float(lambda2_loss)],
                "val_losses": [float(lambda2_loss)],
                "best_epoch": 0,
                "stopped_epoch": 0,
                "metadata": {"lambda2": float(self.lambda2), "n_train": int(n)},
            },
            {
                "stage": f"{name}/kap",
                "train_losses": [],
                "val_losses": [],
                "best_epoch": 0,
                "stopped_epoch": 0,
                "metadata": {"fit_mse_computed": False},
            },
        ]
        _log_elapsed(verbose, 3, f"{name}/kap", "finished", total_start)

    def _eta_objective(
        self,
        eta: float,
        l_mat: np.ndarray,
        l_sub: np.ndarray,
        m_vec: np.ndarray,
        n_mat: np.ndarray,
        l2: np.ndarray,
        m2: np.ndarray,
        n_stage1: int,
    ) -> float:
        n_stage2 = max(l_mat.shape[0] - 1, 1)
        alpha = _solve(_make_psd(l_mat / n_stage2 + float(eta) * n_mat, self.jitter), m_vec)
        cost = float((alpha.T @ _make_psd(l2, self.jitter) @ alpha / max(n_stage1, 1) - 2.0 * alpha.T @ m2).reshape(-1)[0])
        if self.label_variance_in_eta_opt:
            smoother = _solve(_make_psd(l_mat + n_stage2 * float(eta) * n_mat, self.jitter), l_mat)
            cost += float(self.label_variance_in_eta_opt) * 2.0 * float(np.trace(smoother)) / n_stage2
        return cost

    def predict(self, a: np.ndarray, v: np.ndarray) -> np.ndarray:
        a, v = _as_2d(a), _as_2d(v)
        k_a_train = self.kernel_a(self.a_train, a)
        k_v_train = self.kernel_v(self.v_train, v)
        k_a_tilde = self.kernel_a(self.a_stage2, a)
        k_v_tilde = self.kernel_v(self.v_stage2, v)
        train_kernel = k_a_train[self.train_idx, :] * k_v_train[self.train_idx, :]
        stage2_vec = k_a_tilde * k_v_tilde
        first_term = self.predict_stage1_map @ train_kernel
        second_term = self.predict_stage1_bar_map @ train_kernel
        alpha_main = np.asarray(self.alpha[:-1], dtype=float).reshape(-1, 1)
        alpha_bias = float(np.asarray(self.alpha[-1], dtype=float).reshape(-1)[0])
        first = alpha_main.T @ (first_term * stage2_vec)
        second = alpha_bias * (self.ones_m.reshape(1, -1) @ (second_term * stage2_vec))
        return (first + second).T

    def bridge_func(
        self,
        a: np.ndarray,
        z: np.ndarray,
        v: np.ndarray,
        x_extra: np.ndarray | None = None,
    ) -> np.ndarray:
        a, z, v = _as_2d(a), _as_2d(z), _as_2d(v)
        k_z = self.kernel_z(self.z_stage1, z)
        k_a = self.kernel_a(self.a_stage2, a)
        k_v = self.kernel_v(self.v_stage2, v)
        av = k_a * k_v
        term1_core = self.b.T @ k_z
        term2_core = self.b_bar.T @ k_z
        alpha_main = np.asarray(self.alpha[:-1], dtype=float).reshape(-1, 1)
        alpha_bias = float(np.asarray(self.alpha[-1], dtype=float).reshape(-1)[0])
        first = term1_core.T @ (alpha_main * av)
        second = term2_core.T @ ((self.ones_m.reshape(-1, 1) * av) * alpha_bias)
        return (first + second).T


class DRKPVBridge:
    """Density-ratio-free doubly robust KPV baseline with V=X conditioning."""

    def __init__(
        self,
        model_config: DRKPVConfig,
        mean_config: MeanConfig | None,
        support: tuple[float, float],
    ):
        self.model_config = model_config
        self.mean_config = mean_config
        self.support = support
        self.a_scaler = _make_scaler(model_config.standardize)
        self.z_scaler = _make_scaler(model_config.standardize)
        self.w_scaler = _make_scaler(model_config.standardize)
        self.v_scaler = _make_scaler(model_config.standardize)
        self.training_history: list[dict[str, object]] = []
        self.fit_indices_: np.ndarray | None = None
        self.lambda_dr_: float = float(model_config.lambda_dr)

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

    def fit(
        self,
        data: BridgeTrainData,
        sample_weight: np.ndarray | None = None,
        mean_method: str = "dr_cate",
        *,
        verbose: int = 0,
        name: str = "DR-KPV",
        seed: int = 13,
    ) -> None:
        del sample_weight, mean_method
        self.verbose_ = int(verbose)
        self.name_ = name
        verbose_print(verbose, 1, f"[start] {name} fit")
        total_start = _now()
        capped = self._cap_data(data, seed)
        a = self.a_scaler.fit_transform(_as_2d(capped.treatment))
        z = self.z_scaler.fit_transform(_as_2d(capped.treatment_proxy))
        w = self.w_scaler.fit_transform(_as_2d(capped.outcome_proxy))
        v = self.v_scaler.fit_transform(_as_2d(capped.covariates))
        y = _as_2d(capped.outcome)
        n = a.shape[0]
        verbose_print(
            verbose,
            3,
            (
                f"[data] {name} n_input={data.treatment.shape[0]} n_fit={n} "
                f"stage1_perc={self.model_config.stage1_perc} max_train_size={self.model_config.max_train_size}"
            ),
        )

        step_start = _now()
        kernel_a = _FrozenRBFKernel(variance_scale=self.model_config.kernel_variance_scale).fit(a)
        kernel_z = _FrozenRBFKernel(variance_scale=self.model_config.kernel_variance_scale).fit(z)
        kernel_w = _FrozenRBFKernel(variance_scale=self.model_config.kernel_variance_scale).fit(w)
        kernel_v = _FrozenRBFKernel(variance_scale=self.model_config.kernel_variance_scale).fit(v)
        _log_elapsed(verbose, 3, name, "fitted RBF kernels", step_start)

        self.outcome_bridge = _KernelProxyVariableCATE(
            kernel_a=kernel_a,
            kernel_w=kernel_w,
            kernel_z=kernel_z,
            kernel_v=kernel_v,
            kernel_x=None,
            lambda1=self.model_config.kpv_lambda1,
            lambda2=self.model_config.kpv_lambda2,
            zeta=self.model_config.kap_zeta,
            optimize_internal=self.model_config.optimize_internal,
            lambda_min=self.model_config.lambda_min,
            lambda_max=self.model_config.lambda_max,
            regularization_grid_points=self.model_config.regularization_grid_points,
            stage1_perc=self.model_config.stage1_perc,
            jitter=self.model_config.jitter,
        )
        self.treatment_bridge = _KernelAlternativeProxyCATE(
            kernel_a=kernel_a,
            kernel_w=kernel_w,
            kernel_z=kernel_z,
            kernel_v=kernel_v,
            kernel_x=None,
            lambda_=self.model_config.kap_lambda,
            eta=self.model_config.kap_eta,
            lambda2=self.model_config.kap_lambda2,
            zeta=self.model_config.kap_zeta,
            optimize_internal=self.model_config.optimize_internal,
            lambda_min=self.model_config.lambda_min,
            lambda_max=self.model_config.lambda_max,
            regularization_grid_points=self.model_config.regularization_grid_points,
            stage1_perc=self.model_config.stage1_perc,
            jitter=self.model_config.jitter,
            label_variance_in_lambda_opt=self.model_config.label_variance_in_lambda_opt,
            label_variance_in_eta_opt=self.model_config.label_variance_in_eta_opt,
        )
        verbose_print(verbose, 2, f"[dr-kernel] {name} fitting outcome KPV bridge")
        self.outcome_bridge.fit(a=a, w=w, z=z, v=v, x_extra=None, y=y, seed=seed + 101, name=name, verbose=verbose)
        verbose_print(verbose, 2, f"[dr-kernel] {name} fitting treatment KAP bridge")
        self.treatment_bridge.fit(a=a, w=w, z=z, v=v, x_extra=None, y=y, seed=seed + 202, name=name, verbose=verbose)

        step_start = _now()
        self.a_train = a
        self.z_train = z
        self.w_train = w
        self.v_train = v
        self.y_train = y
        self.kernel_a = kernel_a
        self.kernel_z = kernel_z
        self.kernel_w = kernel_w
        self.kernel_v = kernel_v
        self._cache_eval_terms()
        k_a = kernel_a(a, a)
        k_v = kernel_v(v, v)
        self.k_av_train = k_a * k_v
        if self.model_config.optimize_internal:
            k_zw = kernel_z(z, z) * kernel_w(w, w)
            grid = _log_grid(self.model_config.lambda_min, self.model_config.lambda_max, self.model_config.regularization_grid_points)
            self.lambda_dr_, lambda_dr_loss, self.lambda_dr_search_ = _select_from_grid(
                grid,
                lambda lam: _loocv_kernel_loss(k_zw, self.k_av_train, lam, self.model_config.jitter),
            )
        else:
            lambda_dr_loss = _nan_loss()
            self.lambda_dr_search_ = []
        self.dr_system = _make_psd(self.k_av_train + n * self.lambda_dr_ * np.eye(n), self.model_config.jitter)
        self.dr_factor = _factor_system(self.dr_system, 0.0)
        _log_elapsed(verbose, 3, name, f"prepared DR system shape={self.dr_system.shape}", step_start)
        self.training_history = []
        self.training_history.extend(self.outcome_bridge.training_history)
        self.training_history.extend(self.treatment_bridge.training_history)
        self.training_history.append(
            {
                "stage": f"{name}/lambda_dr_loocv",
                "train_losses": [float(lambda_dr_loss)],
                "val_losses": [float(lambda_dr_loss)],
                "best_epoch": 0,
                "stopped_epoch": 0,
                "metadata": {"lambda_dr": float(self.lambda_dr_), "n_fit": int(n)},
            }
        )
        self.training_history.append(
            {
                "stage": f"{name}/dr_cate",
                "train_losses": [],
                "val_losses": [],
                "best_epoch": 0,
                "stopped_epoch": 0,
                "metadata": {"fit_mse_computed": False},
            }
        )
        verbose_print(
            verbose,
            1,
            (
                f"[done] {name} kpv_lambda1={self.outcome_bridge.lambda1:.4g} "
                f"kpv_lambda2={self.outcome_bridge.lambda2:.4g} kap_lambda={self.treatment_bridge.lambda_:.4g} "
                f"kap_eta={self.treatment_bridge.eta:.4g} lambda_dr={self.lambda_dr_:.4g}"
            ),
        )
        _log_elapsed(verbose, 3, name, "finished fit", total_start)

    def _cache_eval_terms(self) -> None:
        k_w_train = self.outcome_bridge.kernel_w(self.outcome_bridge.w_stage1, self.w_train)
        self._outcome_bridge_core_train = self.outcome_bridge.b.T @ k_w_train
        k_z_train = self.treatment_bridge.kernel_z(self.treatment_bridge.z_stage1, self.z_train)
        self._phi_bridge_term1_train = self.treatment_bridge.b.T @ k_z_train
        self._phi_bridge_term2_train = self.treatment_bridge.b_bar.T @ k_z_train

    def _outcome_bridge_on_train_w(self, a: np.ndarray, v: np.ndarray) -> np.ndarray:
        k_a = self.outcome_bridge.kernel_a(self.outcome_bridge.a_tilde, a)
        k_v = self.outcome_bridge.kernel_v(self.outcome_bridge.v_tilde, v)
        weighted_av = np.asarray(self.outcome_bridge.alpha, dtype=float).reshape(-1, 1) * (k_a * k_v)
        return (self._outcome_bridge_core_train.T @ weighted_av).T

    def _treatment_bridge_on_train_z(self, a: np.ndarray, v: np.ndarray) -> np.ndarray:
        k_a = self.treatment_bridge.kernel_a(self.treatment_bridge.a_stage2, a)
        k_v = self.treatment_bridge.kernel_v(self.treatment_bridge.v_stage2, v)
        av = k_a * k_v
        alpha_main = np.asarray(self.treatment_bridge.alpha[:-1], dtype=float).reshape(-1, 1)
        alpha_bias = float(np.asarray(self.treatment_bridge.alpha[-1], dtype=float).reshape(-1)[0])
        first = self._phi_bridge_term1_train.T @ (alpha_main * av)
        second = self._phi_bridge_term2_train.T @ ((self.treatment_bridge.ones_m.reshape(-1, 1) * av) * alpha_bias)
        return (first + second).T

    def _mean_predict_batch(self, a: np.ndarray, v: np.ndarray) -> np.ndarray:
        treatment_part = self.treatment_bridge.predict(a, v).reshape(-1)
        outcome_part = self.outcome_bridge.predict(a, v).reshape(-1)
        phi_bridge = self._treatment_bridge_on_train_z(a, v)
        outcome_bridge = self._outcome_bridge_on_train_w(a, v)
        k_av = self.kernel_a(self.a_train, a) * self.kernel_v(self.v_train, v)
        weights = _solve_factorized(self.dr_factor, k_av)
        slack = np.sum(phi_bridge * outcome_bridge * weights.T, axis=1)
        return (treatment_part + outcome_part - slack).reshape(-1, 1)

    def mean_predict(self, treatment: np.ndarray, covariates: np.ndarray) -> np.ndarray:
        if not hasattr(self, "dr_system"):
            raise RuntimeError("DRKPVBridge has not been fitted.")
        verbose = int(getattr(self, "verbose_", 0))
        name = str(getattr(self, "name_", "DR-KPV"))
        total_start = _now()
        a = self.a_scaler.transform(_as_2d(treatment))
        v = self.v_scaler.transform(_as_2d(covariates))
        n = a.shape[0]
        batch_size = max(1, int(self.model_config.prediction_batch_size))
        n_batches = max(1, int(np.ceil(n / batch_size)))
        predict_verbose = _prediction_verbose(verbose, n)
        verbose_print(predict_verbose, 3, f"[dr-kernel] {name} mean_predict start n={n} batch_size={batch_size} batches={n_batches}")
        preds = []
        for batch_id, start in enumerate(range(0, n, batch_size), start=1):
            end = min(start + batch_size, n)
            batch_start = _now()
            verbose_print(
                predict_verbose,
                3,
                f"[dr-kernel] {name} mean_predict batch {batch_id}/{n_batches} start size={end - start}",
            )
            preds.append(self._mean_predict_batch(a[start:end], v[start:end]))
            _log_elapsed(predict_verbose, 3, name, f"mean_predict batch {batch_id}/{n_batches} done", batch_start)
        out = np.vstack(preds)
        _log_elapsed(predict_verbose, 3, name, f"mean_predict done n={n} batches={n_batches}", total_start)
        return out

    def mean_predict_grid(self, treatment_grid: np.ndarray, covariates: np.ndarray) -> np.ndarray:
        if not hasattr(self, "dr_system"):
            raise RuntimeError("DRKPVBridge has not been fitted.")
        grid_raw = _grid_as_2d(treatment_grid)
        cov_arr = _as_2d(covariates)
        grid_scaled = self.a_scaler.transform(grid_raw)
        cov_scaled = self.v_scaler.transform(cov_arr)
        n_grid = grid_scaled.shape[0]
        n_cov = cov_scaled.shape[0]
        out = np.empty((n_grid, n_cov), dtype=float)
        pair_budget = max(1, int(self.model_config.prediction_batch_size))
        cov_batch_size = max(1, pair_budget // max(n_grid, 1))
        for start in range(0, n_cov, cov_batch_size):
            end = min(start + cov_batch_size, n_cov)
            cov_batch = cov_scaled[start:end]
            a_pairs = np.repeat(grid_scaled, end - start, axis=0)
            v_pairs = np.tile(cov_batch, (n_grid, 1))
            out[:, start:end] = self._mean_predict_batch(a_pairs, v_pairs).reshape(n_grid, end - start)
        return out

    def bridge_predict(self, treatment: np.ndarray, outcome_proxy: np.ndarray, covariates: np.ndarray) -> np.ndarray:
        del outcome_proxy
        return self.mean_predict(treatment, covariates)

    def kernel_diagnostics(
        self,
        *,
        config: KernelDiagnosticsConfig,
        model_name: str = "DR-KPV",
        stage: str = "DR-KPV/bridge",
        round_id: int | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> KernelDiagnosticResult:
        del config, model_name, stage, round_id, sample_weight
        return empty_kernel_diagnostic_result()


class DRPMMRMarginalBridge:
    """Marginal DR-PMMR adapter. It is exposed for manual runs, not default comparisons."""

    def __init__(
        self,
        model_config: DRPMMRMarginalConfig,
        mean_config: MeanConfig | None,
        support: tuple[float, float],
    ):
        self.model_config = model_config
        self.mean_config = mean_config or MeanConfig()
        self.support = support
        self.a_scaler = _make_scaler(model_config.standardize)
        self.z_scaler = _make_scaler(model_config.standardize)
        self.w_scaler = _make_scaler(model_config.standardize)
        self.x_scaler = _make_scaler(model_config.standardize)
        self.training_history: list[dict[str, object]] = []
        self.lambda_dr_: float = float(model_config.lambda_dr)
        self._pmmr_marginal_h_cache: dict[tuple[float, ...], np.ndarray] = {}

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

    def fit(
        self,
        data: BridgeTrainData,
        sample_weight: np.ndarray | None = None,
        mean_method: str = "dr_ate",
        *,
        verbose: int = 0,
        name: str = "DR-PMMR marginal",
        seed: int = 13,
    ) -> None:
        del sample_weight, mean_method
        self.verbose_ = int(verbose)
        self.name_ = name
        self._pmmr_marginal_h_cache = {}
        verbose_print(verbose, 1, f"[start] {name} fit")
        total_start = _now()
        capped = self._cap_data(data, seed)
        a = self.a_scaler.fit_transform(_as_2d(capped.treatment))
        z = self.z_scaler.fit_transform(_as_2d(capped.treatment_proxy))
        w = self.w_scaler.fit_transform(_as_2d(capped.outcome_proxy))
        x = self.x_scaler.fit_transform(_as_2d(capped.covariates))
        v = np.ones((a.shape[0], 1), dtype=float)
        y = _as_2d(capped.outcome)
        n = a.shape[0]
        verbose_print(
            verbose,
            3,
            (
                f"[data] {name} n_input={data.treatment.shape[0]} n_fit={n} "
                f"stage1_perc={self.model_config.stage1_perc} max_train_size={self.model_config.max_train_size}"
            ),
        )

        step_start = _now()
        kernel_a = _FrozenRBFKernel(variance_scale=self.model_config.kernel_variance_scale).fit(a)
        kernel_z = _FrozenRBFKernel(variance_scale=self.model_config.kernel_variance_scale).fit(z)
        kernel_w = _FrozenRBFKernel(variance_scale=self.model_config.kernel_variance_scale).fit(w)
        kernel_v = _FrozenRBFKernel(variance_scale=1.0).fit(v)
        kernel_x = _FrozenRBFKernel(variance_scale=self.model_config.kernel_variance_scale).fit(x)
        _log_elapsed(verbose, 3, name, "fitted RBF kernels", step_start)
        self.treatment_bridge = _KernelAlternativeProxyCATE(
            kernel_a=kernel_a,
            kernel_w=kernel_w,
            kernel_z=kernel_z,
            kernel_v=kernel_v,
            kernel_x=kernel_x,
            lambda_=self.model_config.kap_lambda,
            eta=self.model_config.kap_eta,
            lambda2=self.model_config.kap_lambda2,
            zeta=self.model_config.kap_zeta,
            optimize_internal=self.model_config.optimize_internal,
            lambda_min=self.model_config.lambda_min,
            lambda_max=self.model_config.lambda_max,
            regularization_grid_points=self.model_config.regularization_grid_points,
            stage1_perc=self.model_config.stage1_perc,
            jitter=self.model_config.jitter,
            label_variance_in_lambda_opt=0.0,
            label_variance_in_eta_opt=0.0,
        )
        verbose_print(verbose, 2, f"[dr-kernel] {name} fitting treatment KAP bridge")
        self.treatment_bridge.fit(a=a, w=w, z=z, v=v, x_extra=x, y=y, seed=seed + 303, name=name, verbose=verbose)
        pmmr_config = PMMRConfig(
            ridge=float(self.model_config.pmmr_lambda),
            jitter=float(self.model_config.jitter),
            awx_scale=float(self.model_config.kernel_variance_scale),
            azx_scale=float(self.model_config.kernel_variance_scale),
        )
        self.outcome_bridge = PMMRBridge(pmmr_config, self.mean_config, FlowlessConfig(), self.support)
        pmmr_data = PMMRTrainData(
            treatment=capped.treatment,
            treatment_proxy=capped.treatment_proxy,
            outcome_proxy=capped.outcome_proxy,
            covariates=capped.covariates,
            outcome=capped.outcome,
        )
        verbose_print(verbose, 2, f"[dr-kernel] {name} fitting PMMR outcome bridge")
        self.training_history = self.outcome_bridge._fit_bridge(
            pmmr_data,
            verbose=verbose,
            name=f"{name}/outcome_pmmr",
            seed=seed + 404,
        )
        self.a_train = a
        self.z_train = z
        self.w_train_raw = capped.outcome_proxy
        self.x_train_raw = capped.covariates
        self.v_train = v
        self.x_scaled = x
        self.kernel_a = kernel_a
        self.kernel_v = kernel_v
        self._cache_eval_terms()
        step_start = _now()
        self.k_a_train = kernel_a(a, a)
        if self.model_config.optimize_internal:
            k_zw = kernel_z(z, z) * kernel_w(w, w)
            grid = _log_grid(self.model_config.lambda_min, self.model_config.lambda_max, self.model_config.regularization_grid_points)
            self.lambda_dr_, lambda_dr_loss, self.lambda_dr_search_ = _select_from_grid(
                grid,
                lambda lam: _loocv_kernel_loss(k_zw, self.k_a_train, lam, self.model_config.jitter),
            )
        else:
            lambda_dr_loss = _nan_loss()
            self.lambda_dr_search_ = []
        self.dr_system = _make_psd(self.k_a_train + n * self.lambda_dr_ * np.eye(n), self.model_config.jitter)
        self.dr_factor = _factor_system(self.dr_system, 0.0)
        _log_elapsed(verbose, 3, name, f"prepared DR system shape={self.dr_system.shape}", step_start)
        self.training_history.extend(self.treatment_bridge.training_history)
        self.training_history.append(
            {
                "stage": f"{name}/lambda_dr_loocv",
                "train_losses": [float(lambda_dr_loss)],
                "val_losses": [float(lambda_dr_loss)],
                "best_epoch": 0,
                "stopped_epoch": 0,
                "metadata": {"lambda_dr": float(self.lambda_dr_), "n_fit": int(n)},
            }
        )
        verbose_print(verbose, 1, f"[done] {name} pmmr_lambda={self.model_config.pmmr_lambda:.4g} lambda_dr={self.lambda_dr_:.4g}")
        _log_elapsed(verbose, 3, name, "finished fit", total_start)

    def _cache_eval_terms(self) -> None:
        k_z_train = self.treatment_bridge.kernel_z(self.treatment_bridge.z_stage1, self.z_train)
        self._phi_bridge_term1_train = self.treatment_bridge.b.T @ k_z_train
        self._phi_bridge_term2_train = self.treatment_bridge.b_bar.T @ k_z_train

    @staticmethod
    def _marginal_h_cache_key(treatment_row: np.ndarray) -> tuple[float, ...]:
        return tuple(np.round(np.asarray(treatment_row, dtype=float).reshape(-1), 12).tolist())

    def _pmmr_marginal_h_on_train_w(self, treatment: np.ndarray) -> np.ndarray:
        treatment_arr = _as_2d(treatment)
        cache = self._pmmr_marginal_h_cache
        rows: list[np.ndarray | None] = [None] * treatment_arr.shape[0]
        missing_positions: list[int] = []
        missing_rows: list[np.ndarray] = []
        for idx, row in enumerate(treatment_arr):
            key = self._marginal_h_cache_key(row)
            values = cache.get(key)
            if values is None:
                missing_positions.append(idx)
                missing_rows.append(row.reshape(1, -1))
            else:
                rows[idx] = values
        if missing_rows:
            unique_rows: dict[tuple[float, ...], np.ndarray] = {}
            for row in missing_rows:
                unique_rows.setdefault(self._marginal_h_cache_key(row), row)
            unique_matrix = np.vstack(list(unique_rows.values()))
            repeated_a = np.repeat(unique_matrix, self.w_train_raw.shape[0], axis=0)
            repeated_w = np.tile(self.w_train_raw, (unique_matrix.shape[0], 1))
            repeated_x = np.tile(self.x_train_raw, (unique_matrix.shape[0], 1))
            predicted = self.outcome_bridge.bridge_predict(
                repeated_a,
                repeated_w,
                repeated_x,
            ).reshape(unique_matrix.shape[0], self.w_train_raw.shape[0])
            for row, values in zip(unique_matrix, predicted, strict=False):
                cache[self._marginal_h_cache_key(row)] = values
            for idx, row in zip(missing_positions, missing_rows, strict=False):
                rows[idx] = cache[self._marginal_h_cache_key(row)]
        return np.asarray(rows, dtype=float)

    def _treatment_bridge_on_train_z(self, a: np.ndarray, v: np.ndarray) -> np.ndarray:
        k_a = self.treatment_bridge.kernel_a(self.treatment_bridge.a_stage2, a)
        k_v = self.treatment_bridge.kernel_v(self.treatment_bridge.v_stage2, v)
        av = k_a * k_v
        alpha_main = np.asarray(self.treatment_bridge.alpha[:-1], dtype=float).reshape(-1, 1)
        alpha_bias = float(np.asarray(self.treatment_bridge.alpha[-1], dtype=float).reshape(-1)[0])
        first = self._phi_bridge_term1_train.T @ (alpha_main * av)
        second = self._phi_bridge_term2_train.T @ ((self.treatment_bridge.ones_m.reshape(-1, 1) * av) * alpha_bias)
        return (first + second).T

    def _mean_predict_batch(self, a_raw: np.ndarray, a: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, int]:
        treatment_part = self.treatment_bridge.predict(a, v).reshape(-1)
        cache_before = len(self._pmmr_marginal_h_cache)
        h_bridge_arr = self._pmmr_marginal_h_on_train_w(a_raw)
        cache_misses = len(self._pmmr_marginal_h_cache) - cache_before
        outcome_part = np.mean(h_bridge_arr, axis=1)
        phi_bridge = self._treatment_bridge_on_train_z(a, v)
        k_a = self.kernel_a(self.a_train, a)
        weights = _solve_factorized(self.dr_factor, k_a)
        slack = np.sum(phi_bridge * h_bridge_arr * weights.T, axis=1)
        pred = (treatment_part + outcome_part - slack).reshape(-1, 1)
        return pred, cache_misses

    def mean_predict(self, treatment: np.ndarray, covariates: np.ndarray) -> np.ndarray:
        if not hasattr(self, "dr_system"):
            raise RuntimeError("DRPMMRMarginalBridge has not been fitted.")
        del covariates
        verbose = int(getattr(self, "verbose_", 0))
        name = str(getattr(self, "name_", "DR-PMMR marginal"))
        total_start = _now()
        a_raw = _as_2d(treatment)
        a = self.a_scaler.transform(a_raw)
        v = np.ones((a.shape[0], 1), dtype=float)
        n = a.shape[0]
        batch_size = max(1, int(self.model_config.prediction_batch_size))
        n_batches = max(1, int(np.ceil(n / batch_size)))
        predict_verbose = _prediction_verbose(verbose, n)
        verbose_print(predict_verbose, 3, f"[dr-kernel] {name} mean_predict start n={n} batch_size={batch_size} batches={n_batches}")
        preds = []
        for batch_id, start in enumerate(range(0, n, batch_size), start=1):
            end = min(start + batch_size, n)
            batch_start = _now()
            verbose_print(
                predict_verbose,
                3,
                f"[dr-kernel] {name} mean_predict batch {batch_id}/{n_batches} start size={end - start}",
            )
            pred_batch, cache_misses = self._mean_predict_batch(a_raw[start:end], a[start:end], v[start:end])
            preds.append(pred_batch)
            _log_elapsed(
                predict_verbose,
                3,
                name,
                (
                    f"mean_predict batch {batch_id}/{n_batches} done "
                    f"pmmr_cache_misses={cache_misses} pmmr_cache_size={len(self._pmmr_marginal_h_cache)}"
                ),
                batch_start,
            )
        out = np.vstack(preds)
        _log_elapsed(predict_verbose, 3, name, f"mean_predict done n={n} batches={n_batches}", total_start)
        return out

    def mean_predict_grid(self, treatment_grid: np.ndarray, covariates: np.ndarray) -> np.ndarray:
        if not hasattr(self, "dr_system"):
            raise RuntimeError("DRPMMRMarginalBridge has not been fitted.")
        grid_raw = _grid_as_2d(treatment_grid)
        curve, _ = self._mean_predict_batch(
            grid_raw,
            self.a_scaler.transform(grid_raw),
            np.ones((grid_raw.shape[0], 1), dtype=float),
        )
        n_cov = _as_2d(covariates).shape[0]
        return np.repeat(curve.reshape(-1, 1), n_cov, axis=1)

    def bridge_predict(self, treatment: np.ndarray, outcome_proxy: np.ndarray, covariates: np.ndarray) -> np.ndarray:
        del outcome_proxy
        return self.mean_predict(treatment, covariates)

    def kernel_diagnostics(
        self,
        *,
        config: KernelDiagnosticsConfig,
        model_name: str = "DR-PMMR marginal",
        stage: str = "DR-PMMR marginal/bridge",
        round_id: int | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> KernelDiagnosticResult:
        del config, model_name, stage, round_id, sample_weight
        return empty_kernel_diagnostic_result()


class DRPMMRConditionalBridge(DRPMMRMarginalBridge):
    """Conditional DR-PMMR adapter using V=X and the PMMR mean regressor."""

    def __init__(
        self,
        model_config: DRPMMRConditionalConfig,
        mean_config: MeanConfig | None,
        support: tuple[float, float],
    ):
        super().__init__(model_config, mean_config, support)

    def _cache_pmmr_bridge_components(self) -> None:
        a_dim = int(self.treatment_dim_)
        w_dim = int(self.outcome_proxy_dim_)
        train_awx = np.asarray(self.outcome_bridge.train_awx, dtype=float)
        bandwidth = np.asarray(self.outcome_bridge.awx_bandwidth, dtype=float).reshape(-1)
        self.pmmr_train_a_scaled = train_awx[:, :a_dim]
        self.pmmr_train_w_scaled = train_awx[:, a_dim : a_dim + w_dim]
        self.pmmr_train_x_scaled = train_awx[:, a_dim + w_dim :]
        self.pmmr_bandwidth_a = bandwidth[:a_dim]
        self.pmmr_bandwidth_w = bandwidth[a_dim : a_dim + w_dim]
        self.pmmr_bandwidth_x = bandwidth[a_dim + w_dim :]
        self.pmmr_k_w_train = rbf_product_kernel(
            self.pmmr_train_w_scaled,
            self.pmmr_train_w_scaled,
            self.pmmr_bandwidth_w,
        )
        alpha = np.asarray(self.outcome_bridge.alpha, dtype=float).reshape(1, -1)
        self.pmmr_k_w_alpha = self.pmmr_k_w_train * alpha

    def _pmmr_bridge_on_train_w(self, treatment: np.ndarray, covariates: np.ndarray) -> np.ndarray:
        a_raw = _as_2d(treatment)
        x_raw = _as_2d(covariates)
        dummy_w = np.repeat(self.w_train_raw[:1], a_raw.shape[0], axis=0)
        awx = np.concatenate([a_raw, dummy_w, x_raw], axis=1)
        awx_scaled = self.outcome_bridge.awx_scaler.transform(awx)
        a_dim = int(self.treatment_dim_)
        w_dim = int(self.outcome_proxy_dim_)
        a_scaled = awx_scaled[:, :a_dim]
        x_scaled = awx_scaled[:, a_dim + w_dim :]
        k_a = rbf_product_kernel(a_scaled, self.pmmr_train_a_scaled, self.pmmr_bandwidth_a).T
        k_x = rbf_product_kernel(x_scaled, self.pmmr_train_x_scaled, self.pmmr_bandwidth_x).T
        return (self.pmmr_k_w_alpha @ (k_a * k_x)).T

    def fit(
        self,
        data: BridgeTrainData,
        sample_weight: np.ndarray | None = None,
        mean_method: str = "pseudo_outcomes",
        *,
        verbose: int = 0,
        name: str = "DR-PMMR conditional",
        seed: int = 13,
    ) -> None:
        del sample_weight
        self.verbose_ = int(verbose)
        self.name_ = name
        verbose_print(verbose, 1, f"[start] {name} fit")
        total_start = _now()
        capped = self._cap_data(data, seed)
        a_raw = _as_2d(capped.treatment)
        z_raw = _as_2d(capped.treatment_proxy)
        w_raw = _as_2d(capped.outcome_proxy)
        x_raw = _as_2d(capped.covariates)
        a = self.a_scaler.fit_transform(a_raw)
        z = self.z_scaler.fit_transform(z_raw)
        w = self.w_scaler.fit_transform(w_raw)
        x = self.x_scaler.fit_transform(x_raw)
        v = x
        y = _as_2d(capped.outcome)
        n = a.shape[0]
        self.treatment_dim_ = a_raw.shape[1]
        self.outcome_proxy_dim_ = w_raw.shape[1]
        self.covariate_dim_ = x_raw.shape[1]
        verbose_print(
            verbose,
            3,
            (
                f"[data] {name} n_input={data.treatment.shape[0]} n_fit={n} "
                f"stage1_perc={self.model_config.stage1_perc} max_train_size={self.model_config.max_train_size} "
                f"conditioning_dim={self.covariate_dim_}"
            ),
        )

        step_start = _now()
        kernel_a = _FrozenRBFKernel(variance_scale=self.model_config.kernel_variance_scale).fit(a)
        kernel_z = _FrozenRBFKernel(variance_scale=self.model_config.kernel_variance_scale).fit(z)
        kernel_w = _FrozenRBFKernel(variance_scale=self.model_config.kernel_variance_scale).fit(w)
        kernel_v = _FrozenRBFKernel(variance_scale=self.model_config.kernel_variance_scale).fit(v)
        _log_elapsed(verbose, 3, name, "fitted RBF kernels", step_start)

        self.treatment_bridge = _KernelAlternativeProxyCATE(
            kernel_a=kernel_a,
            kernel_w=kernel_w,
            kernel_z=kernel_z,
            kernel_v=kernel_v,
            kernel_x=None,
            lambda_=self.model_config.kap_lambda,
            eta=self.model_config.kap_eta,
            lambda2=self.model_config.kap_lambda2,
            zeta=self.model_config.kap_zeta,
            optimize_internal=self.model_config.optimize_internal,
            lambda_min=self.model_config.lambda_min,
            lambda_max=self.model_config.lambda_max,
            regularization_grid_points=self.model_config.regularization_grid_points,
            stage1_perc=self.model_config.stage1_perc,
            jitter=self.model_config.jitter,
            label_variance_in_lambda_opt=0.0,
            label_variance_in_eta_opt=0.0,
        )
        verbose_print(verbose, 2, f"[dr-kernel] {name} fitting treatment KAP bridge")
        self.treatment_bridge.fit(a=a, w=w, z=z, v=v, x_extra=None, y=y, seed=seed + 303, name=name, verbose=verbose)

        pmmr_config = PMMRConfig(
            ridge=float(self.model_config.pmmr_lambda),
            jitter=float(self.model_config.jitter),
            awx_scale=float(self.model_config.kernel_variance_scale),
            azx_scale=float(self.model_config.kernel_variance_scale),
        )
        self.outcome_bridge = PMMRBridge(pmmr_config, self.mean_config, FlowlessConfig(), self.support)
        pmmr_data = PMMRTrainData(
            treatment=a_raw,
            treatment_proxy=z_raw,
            outcome_proxy=w_raw,
            covariates=x_raw,
            outcome=y,
        )
        verbose_print(verbose, 2, f"[dr-kernel] {name} fitting PMMR outcome bridge")
        self.outcome_bridge.fit(
            pmmr_data,
            mean_method=mean_method,
            verbose=verbose,
            name=f"{name}/outcome_pmmr",
            seed=seed + 404,
        )
        self._cache_pmmr_bridge_components()
        for attr in (
            "mean_crossfit_folds_",
            "mean_pseudooutcome_metadata_",
            "mean_regressor_group_ids_",
            "mean_pseudooutcome_diagnostics_",
        ):
            setattr(self, attr, getattr(self.outcome_bridge, attr, None))

        self.a_train = a
        self.z_train = z
        self.w_train_raw = w_raw
        self.x_train_raw = x_raw
        self.v_train = v
        self.kernel_a = kernel_a
        self.kernel_z = kernel_z
        self.kernel_w = kernel_w
        self.kernel_v = kernel_v
        self._cache_eval_terms()
        step_start = _now()
        k_a = kernel_a(a, a)
        k_v = kernel_v(v, v)
        self.k_av_train = k_a * k_v
        if self.model_config.optimize_internal:
            k_zw = kernel_z(z, z) * kernel_w(w, w)
            grid = _log_grid(
                self.model_config.lambda_min,
                self.model_config.lambda_max,
                self.model_config.regularization_grid_points,
            )
            self.lambda_dr_, lambda_dr_loss, self.lambda_dr_search_ = _select_from_grid(
                grid,
                lambda lam: _loocv_kernel_loss(k_zw, self.k_av_train, lam, self.model_config.jitter),
            )
        else:
            lambda_dr_loss = _nan_loss()
            self.lambda_dr_search_ = []
        self.dr_system = _make_psd(self.k_av_train + n * self.lambda_dr_ * np.eye(n), self.model_config.jitter)
        self.dr_factor = _factor_system(self.dr_system, 0.0)
        _log_elapsed(verbose, 3, name, f"prepared DR system shape={self.dr_system.shape}", step_start)
        self.training_history = []
        self.training_history.extend(self.outcome_bridge.training_history)
        self.training_history.extend(self.treatment_bridge.training_history)
        self.training_history.append(
            {
                "stage": f"{name}/lambda_dr_loocv",
                "train_losses": [float(lambda_dr_loss)],
                "val_losses": [float(lambda_dr_loss)],
                "best_epoch": 0,
                "stopped_epoch": 0,
                "metadata": {"lambda_dr": float(self.lambda_dr_), "n_fit": int(n), "conditioning": "X"},
            }
        )
        self.training_history.append(
            {
                "stage": f"{name}/dr_pmmr_conditional",
                "train_losses": [],
                "val_losses": [],
                "best_epoch": 0,
                "stopped_epoch": 0,
                "metadata": {"fit_mse_computed": False},
            }
        )
        verbose_print(
            verbose,
            1,
            (
                f"[done] {name} pmmr_lambda={self.model_config.pmmr_lambda:.4g} "
                f"kap_lambda={self.treatment_bridge.lambda_:.4g} kap_eta={self.treatment_bridge.eta:.4g} "
                f"lambda_dr={self.lambda_dr_:.4g}"
            ),
        )
        _log_elapsed(verbose, 3, name, "finished fit", total_start)

    def _mean_predict_batch(self, a_raw: np.ndarray, x_raw: np.ndarray, a: np.ndarray, v: np.ndarray) -> np.ndarray:
        treatment_part = self.treatment_bridge.predict(a, v).reshape(-1)
        outcome_part = self.outcome_bridge.mean_predict(a_raw, x_raw).reshape(-1)
        phi_bridge = self._treatment_bridge_on_train_z(a, v)
        h_bridge = self._pmmr_bridge_on_train_w(a_raw, x_raw)
        k_av = self.kernel_a(self.a_train, a) * self.kernel_v(self.v_train, v)
        weights = _solve_factorized(self.dr_factor, k_av)
        slack = np.sum(phi_bridge * h_bridge * weights.T, axis=1)
        return (treatment_part + outcome_part - slack).reshape(-1, 1)

    def mean_predict(self, treatment: np.ndarray, covariates: np.ndarray) -> np.ndarray:
        if not hasattr(self, "dr_system"):
            raise RuntimeError("DRPMMRConditionalBridge has not been fitted.")
        verbose = int(getattr(self, "verbose_", 0))
        name = str(getattr(self, "name_", "DR-PMMR conditional"))
        total_start = _now()
        a_raw = _as_2d(treatment)
        x_raw = _as_2d(covariates)
        a = self.a_scaler.transform(a_raw)
        v = self.x_scaler.transform(x_raw)
        n = a.shape[0]
        batch_size = max(1, int(self.model_config.prediction_batch_size))
        n_batches = max(1, int(np.ceil(n / batch_size)))
        predict_verbose = _prediction_verbose(verbose, n)
        verbose_print(predict_verbose, 3, f"[dr-kernel] {name} mean_predict start n={n} batch_size={batch_size} batches={n_batches}")
        preds = []
        for batch_id, start in enumerate(range(0, n, batch_size), start=1):
            end = min(start + batch_size, n)
            batch_start = _now()
            verbose_print(
                predict_verbose,
                3,
                f"[dr-kernel] {name} mean_predict batch {batch_id}/{n_batches} start size={end - start}",
            )
            preds.append(self._mean_predict_batch(a_raw[start:end], x_raw[start:end], a[start:end], v[start:end]))
            _log_elapsed(predict_verbose, 3, name, f"mean_predict batch {batch_id}/{n_batches} done", batch_start)
        out = np.vstack(preds)
        _log_elapsed(predict_verbose, 3, name, f"mean_predict done n={n} batches={n_batches}", total_start)
        return out

    def mean_predict_grid(self, treatment_grid: np.ndarray, covariates: np.ndarray) -> np.ndarray:
        if not hasattr(self, "dr_system"):
            raise RuntimeError("DRPMMRConditionalBridge has not been fitted.")
        grid_raw = _grid_as_2d(treatment_grid)
        cov_arr = _as_2d(covariates)
        grid_scaled = self.a_scaler.transform(grid_raw)
        cov_scaled = self.x_scaler.transform(cov_arr)
        n_grid = grid_scaled.shape[0]
        n_cov = cov_scaled.shape[0]
        out = np.empty((n_grid, n_cov), dtype=float)
        pair_budget = max(1, int(self.model_config.prediction_batch_size))
        cov_batch_size = max(1, pair_budget // max(n_grid, 1))
        for start in range(0, n_cov, cov_batch_size):
            end = min(start + cov_batch_size, n_cov)
            x_batch_raw = cov_arr[start:end]
            x_batch_scaled = cov_scaled[start:end]
            a_pairs_raw = np.repeat(grid_raw, end - start, axis=0)
            x_pairs_raw = np.tile(x_batch_raw, (n_grid, 1))
            a_pairs = np.repeat(grid_scaled, end - start, axis=0)
            v_pairs = np.tile(x_batch_scaled, (n_grid, 1))
            out[:, start:end] = self._mean_predict_batch(
                a_pairs_raw,
                x_pairs_raw,
                a_pairs,
                v_pairs,
            ).reshape(n_grid, end - start)
        return out

    def bridge_predict(self, treatment: np.ndarray, outcome_proxy: np.ndarray, covariates: np.ndarray) -> np.ndarray:
        del outcome_proxy
        return self.mean_predict(treatment, covariates)

    def kernel_diagnostics(
        self,
        *,
        config: KernelDiagnosticsConfig,
        model_name: str = "DR-PMMR conditional",
        stage: str = "DR-PMMR conditional/bridge",
        round_id: int | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> KernelDiagnosticResult:
        del config, model_name, stage, round_id, sample_weight
        return empty_kernel_diagnostic_result()


class FlowlessConfig:
    bins: int = 1
    transforms: int = 1
    hidden_features: tuple[int, ...] = (8,)
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 1
    min_epochs: int = 1
    early_stop_patience: int = 1
    early_stop_min_delta: float = 0.0
    batch_size: int = 32
    val_fraction: float = 0.0
