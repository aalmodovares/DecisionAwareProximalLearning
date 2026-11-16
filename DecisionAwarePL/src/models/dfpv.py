from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from src.config import DFPVConfig, FlowConfig, MeanConfig
from src.models.common import TorchRegressor
from src.utils import ArrayScaler, EarlyStoppingTracker, kfold_indices, progress_iter, safe_float, verbose_print


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


def _as_2d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        return arr[:, None]
    return arr


def _activation(name: str) -> type[nn.Module]:
    activations: dict[str, type[nn.Module]] = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
    }
    selected = activations.get(str(name).lower())
    if selected is None:
        raise ValueError(f"Unknown activation {name!r}.")
    return selected


def _build_feature_net(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
    activation: str,
    *,
    final_activation: bool,
) -> nn.Sequential:
    activation_cls = _activation(activation)
    layers: list[nn.Module] = []
    prev = int(input_dim)
    for width in hidden_dims:
        layers.extend([nn.Linear(prev, int(width)), activation_cls()])
        prev = int(width)
    layers.append(nn.Linear(prev, int(output_dim)))
    if final_activation:
        layers.append(activation_cls())
    return nn.Sequential(*layers)


def _add_const_col(mat: torch.Tensor) -> torch.Tensor:
    return torch.cat([mat, torch.ones((mat.shape[0], 1), dtype=mat.dtype, device=mat.device)], dim=1)


def _outer_prod(mat1: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    mat1_shape = tuple(mat1.shape)
    mat2_shape = tuple(mat2.shape)
    if mat1_shape[0] != mat2_shape[0]:
        raise ValueError("Outer-product inputs must have the same first dimension.")
    n_data = mat1_shape[0]
    aug_mat1 = mat1.reshape(mat1_shape + (1,) * (len(mat2_shape) - 1))
    aug_mat2 = mat2.reshape((n_data,) + (1,) * (len(mat1_shape) - 1) + mat2_shape[1:])
    return aug_mat1 * aug_mat2


def _fit_linear(
    target: torch.Tensor,
    feature: torch.Tensor,
    reg: float,
    jitter: float,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if feature.dim() != 2 or target.dim() != 2:
        raise ValueError("DFPV ridge solve expects 2D feature and target tensors.")

    n_data, n_dim = feature.shape
    ridge = float(reg) * float(n_data) + float(jitter)

    if sample_weight is None:
        weighted_feature = feature
        weighted_target = target
    else:
        weight = torch.as_tensor(sample_weight, dtype=feature.dtype, device=feature.device).reshape(-1, 1)
        if weight.shape[0] != n_data:
            raise ValueError("DFPV sample_weight must match the number of rows.")
        sqrt_weight = torch.sqrt(torch.clamp(weight, min=1e-12))
        weighted_feature = feature * sqrt_weight
        weighted_target = target * sqrt_weight

    if n_dim > n_data:
        eye = torch.eye(n_data, dtype=feature.dtype, device=feature.device)
        system = weighted_feature @ weighted_feature.T + ridge * eye
        dual_weight = torch.linalg.solve(system, weighted_target)
        return weighted_feature.T @ dual_weight

    eye = torch.eye(n_dim, dtype=feature.dtype, device=feature.device)
    system = weighted_feature.T @ weighted_feature + ridge * eye
    rhs = weighted_feature.T @ weighted_target
    return torch.linalg.solve(system, rhs)


def _linear_reg_pred(feature: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return feature @ weight


def _linear_reg_loss(target: torch.Tensor, feature: torch.Tensor, reg: float, jitter: float) -> torch.Tensor:
    weight = _fit_linear(target, feature, reg, jitter)
    pred = _linear_reg_pred(feature, weight)
    n_data = feature.shape[0]
    return torch.norm(target - pred).square() + float(reg) * torch.norm(weight).square() * float(n_data)


def _weighted_squared_loss(residual: torch.Tensor, sample_weight: torch.Tensor | None) -> torch.Tensor:
    if sample_weight is None:
        return torch.norm(residual).square()

    weight = torch.as_tensor(sample_weight, dtype=residual.dtype, device=residual.device).reshape(-1, 1)
    if weight.shape[0] != residual.shape[0]:
        raise ValueError("DFPV sample_weight must match the residual rows.")

    return torch.sum(torch.clamp(weight, min=1e-12) * residual.square())


class DFPVBridge:
    """Deep Feature Proxy Variable bridge with the repository mean-regressor head."""

    def __init__(
        self,
        model_config: DFPVConfig,
        mean_config: MeanConfig,
        w_flow_config: FlowConfig,
        support: tuple[float, float],
    ):
        self.model_config = model_config
        self.mean_config = mean_config
        self.w_flow_config = w_flow_config
        self.support = support
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.a_scaler = _make_scaler(model_config.standardize)
        self.z_scaler = _make_scaler(model_config.standardize)
        self.w_scaler = _make_scaler(model_config.standardize)
        self.x_scaler = _make_scaler(model_config.standardize)
        self.y_scaler = _make_scaler(model_config.standardize)
        self.treatment_1st_net: nn.Module | None = None
        self.treatment_2nd_net: nn.Module | None = None
        self.treatment_proxy_net: nn.Module | None = None
        self.outcome_proxy_net: nn.Module | None = None
        self.covariate_1st_net: nn.Module | None = None
        self.covariate_2nd_net: nn.Module | None = None
        self.stage1_weight: torch.Tensor | None = None
        self.stage2_weight: torch.Tensor | None = None
        self.mean_regressor: TorchRegressor | None = None
        self.mean_method: str | None = None
        self.training_history: list[dict[str, object]] = []
        self.mean_crossfit_folds_: list[dict[str, np.ndarray | float | int]] = []
        self.mean_pseudooutcome_metadata_: dict[str, int | bool | float] = {}
        self.mean_regressor_group_ids_: np.ndarray | None = None
        self.mean_pseudooutcome_diagnostics_: dict[str, np.ndarray | float] = {}
        self.fit_indices_: np.ndarray | None = None
        self.stage1_indices_: np.ndarray | None = None
        self.stage2_indices_: np.ndarray | None = None

    @staticmethod
    def _augment_stage1_feature(
        treatment_feature: torch.Tensor,
        treatment_proxy_feature: torch.Tensor,
        covariate_feature: torch.Tensor | None,
    ) -> torch.Tensor:
        feature = _outer_prod(_add_const_col(treatment_feature), _add_const_col(treatment_proxy_feature))
        feature = torch.flatten(feature, start_dim=1)
        if covariate_feature is not None:
            feature = _outer_prod(feature, _add_const_col(covariate_feature))
            feature = torch.flatten(feature, start_dim=1)
        return feature

    @staticmethod
    def _augment_stage2_feature(
        outcome_proxy_feature: torch.Tensor,
        treatment_feature: torch.Tensor,
        covariate_feature: torch.Tensor | None,
    ) -> torch.Tensor:
        feature = _outer_prod(_add_const_col(treatment_feature), _add_const_col(outcome_proxy_feature))
        feature = torch.flatten(feature, start_dim=1)
        if covariate_feature is not None:
            feature = _outer_prod(feature, _add_const_col(covariate_feature))
            feature = torch.flatten(feature, start_dim=1)
        return feature

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

    def _build_networks(self, a_dim: int, z_dim: int, w_dim: int, x_dim: int) -> None:
        cfg = self.model_config
        kwargs = dict(
            hidden_dims=cfg.hidden_dims,
            output_dim=int(cfg.feature_dim),
            activation=cfg.activation,
            final_activation=bool(cfg.final_activation),
        )
        self.treatment_1st_net = _build_feature_net(a_dim, **kwargs).to(self.device)
        self.treatment_2nd_net = _build_feature_net(a_dim, **kwargs).to(self.device)
        self.treatment_proxy_net = _build_feature_net(z_dim, **kwargs).to(self.device)
        self.outcome_proxy_net = _build_feature_net(w_dim, **kwargs).to(self.device)
        if cfg.include_covariates:
            self.covariate_1st_net = _build_feature_net(x_dim, **kwargs).to(self.device)
            self.covariate_2nd_net = _build_feature_net(x_dim, **kwargs).to(self.device)
        else:
            self.covariate_1st_net = None
            self.covariate_2nd_net = None

    def _stage1_params(self) -> list[nn.Parameter]:
        assert self.treatment_1st_net is not None and self.treatment_proxy_net is not None
        params = list(self.treatment_1st_net.parameters()) + list(self.treatment_proxy_net.parameters())
        if self.covariate_1st_net is not None:
            params.extend(self.covariate_1st_net.parameters())
        return params

    def _stage2_params(self) -> list[nn.Parameter]:
        assert self.treatment_2nd_net is not None and self.outcome_proxy_net is not None
        params = list(self.treatment_2nd_net.parameters()) + list(self.outcome_proxy_net.parameters())
        if self.covariate_2nd_net is not None:
            params.extend(self.covariate_2nd_net.parameters())
        return params

    def _fit_2sls(
        self,
        treatment_1st_feature_1st: torch.Tensor,
        treatment_1st_feature_2nd: torch.Tensor,
        treatment_2nd_feature_2nd: torch.Tensor,
        treatment_proxy_feature_1st: torch.Tensor,
        treatment_proxy_feature_2nd: torch.Tensor,
        outcome_proxy_feature_1st: torch.Tensor,
        covariate_1st_feature_1st: torch.Tensor | None,
        covariate_1st_feature_2nd: torch.Tensor | None,
        covariate_2nd_feature_2nd: torch.Tensor | None,
        outcome_2nd: torch.Tensor,
        sample_weight_2nd: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        cfg = self.model_config
        stage1_feature = self._augment_stage1_feature(
            treatment_1st_feature_1st,
            treatment_proxy_feature_1st,
            covariate_1st_feature_1st,
        )
        stage1_weight = _fit_linear(outcome_proxy_feature_1st, stage1_feature, cfg.lam1, cfg.jitter)
        stage1_feature_2nd = self._augment_stage1_feature(
            treatment_1st_feature_2nd,
            treatment_proxy_feature_2nd,
            covariate_1st_feature_2nd,
        )
        predicted_outcome_proxy_feature = _linear_reg_pred(stage1_feature_2nd, stage1_weight)
        stage2_feature = self._augment_stage2_feature(
            predicted_outcome_proxy_feature,
            treatment_2nd_feature_2nd,
            covariate_2nd_feature_2nd,
        )
        stage2_weight = _fit_linear(
            outcome_2nd,
            stage2_feature,
            cfg.lam2,
            cfg.jitter,
            sample_weight=sample_weight_2nd,
        )

        pred = _linear_reg_pred(stage2_feature, stage2_weight)

        stage2_loss = (
            _weighted_squared_loss(outcome_2nd - pred, sample_weight_2nd)
            + float(cfg.lam2) * torch.norm(stage2_weight).square()
        )
        return {
            "stage1_weight": stage1_weight,
            "stage2_weight": stage2_weight,
            "stage2_loss": stage2_loss,
        }

    def _stage1_update(
        self,
        a1: torch.Tensor,
        z1: torch.Tensor,
        w1: torch.Tensor,
        x1: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        assert self.treatment_1st_net is not None
        assert self.treatment_2nd_net is not None
        assert self.treatment_proxy_net is not None
        assert self.outcome_proxy_net is not None
        self.treatment_1st_net.train(True)
        self.treatment_proxy_net.train(True)
        self.outcome_proxy_net.train(False)
        if self.covariate_1st_net is not None:
            self.covariate_1st_net.train(True)
        with torch.no_grad():
            outcome_proxy_feature = self.outcome_proxy_net(w1)
        last_loss = float("nan")
        for _ in range(int(self.model_config.stage1_iter)):
            optimizer.zero_grad()
            treatment_feature = self.treatment_1st_net(a1)
            treatment_proxy_feature = self.treatment_proxy_net(z1)
            covariate_feature = self.covariate_1st_net(x1) if self.covariate_1st_net is not None else None
            feature = self._augment_stage1_feature(treatment_feature, treatment_proxy_feature, covariate_feature)
            loss = _linear_reg_loss(outcome_proxy_feature, feature, self.model_config.lam1, self.model_config.jitter)
            loss.backward()
            if self.model_config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self._stage1_params(), self.model_config.grad_clip_norm)
            optimizer.step()
            last_loss = safe_float(loss)
        return last_loss

    def _stage2_update(
        self,
        a1: torch.Tensor,
        z1: torch.Tensor,
        w1: torch.Tensor,
        x1: torch.Tensor,
        a2: torch.Tensor,
        z2: torch.Tensor,
        x2: torch.Tensor,
        y2: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        sample_weight_2nd: torch.Tensor | None = None,
    ) -> float:
        assert self.treatment_1st_net is not None
        assert self.treatment_2nd_net is not None
        assert self.treatment_proxy_net is not None
        assert self.outcome_proxy_net is not None
        self.treatment_1st_net.train(False)
        self.treatment_proxy_net.train(False)
        self.treatment_2nd_net.train(True)
        self.outcome_proxy_net.train(True)
        if self.covariate_1st_net is not None:
            self.covariate_1st_net.train(False)
        if self.covariate_2nd_net is not None:
            self.covariate_2nd_net.train(True)
        with torch.no_grad():
            treatment_1st_feature_1st = self.treatment_1st_net(a1)
            treatment_1st_feature_2nd = self.treatment_1st_net(a2)
            treatment_proxy_feature_1st = self.treatment_proxy_net(z1)
            treatment_proxy_feature_2nd = self.treatment_proxy_net(z2)
            covariate_1st_feature_1st = self.covariate_1st_net(x1) if self.covariate_1st_net is not None else None
            covariate_1st_feature_2nd = self.covariate_1st_net(x2) if self.covariate_1st_net is not None else None
        last_loss = float("nan")
        for _ in range(int(self.model_config.stage2_iter)):
            optimizer.zero_grad()
            outcome_proxy_feature_1st = self.outcome_proxy_net(w1)
            treatment_2nd_feature_2nd = self.treatment_2nd_net(a2)
            covariate_2nd_feature_2nd = self.covariate_2nd_net(x2) if self.covariate_2nd_net is not None else None
            res = self._fit_2sls(
                treatment_1st_feature_1st,
                treatment_1st_feature_2nd,
                treatment_2nd_feature_2nd,
                treatment_proxy_feature_1st,
                treatment_proxy_feature_2nd,
                outcome_proxy_feature_1st,
                covariate_1st_feature_1st,
                covariate_1st_feature_2nd,
                covariate_2nd_feature_2nd,
                y2,
                sample_weight_2nd=sample_weight_2nd,
            )
            loss = res["stage2_loss"]
            loss.backward()
            if self.model_config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self._stage2_params(), self.model_config.grad_clip_norm)
            optimizer.step()
            last_loss = safe_float(loss)
        return last_loss

    def _finalize_bridge_weights(
        self,
        a1: torch.Tensor,
        z1: torch.Tensor,
        w1: torch.Tensor,
        x1: torch.Tensor,
        a2: torch.Tensor,
        z2: torch.Tensor,
        x2: torch.Tensor,
        y2: torch.Tensor,
        sample_weight_2nd: torch.Tensor | None = None,
    ) -> None:
        assert self.treatment_1st_net is not None
        assert self.treatment_2nd_net is not None
        assert self.treatment_proxy_net is not None
        assert self.outcome_proxy_net is not None
        self.treatment_1st_net.train(False)
        self.treatment_2nd_net.train(False)
        self.treatment_proxy_net.train(False)
        self.outcome_proxy_net.train(False)
        if self.covariate_1st_net is not None:
            self.covariate_1st_net.train(False)
        if self.covariate_2nd_net is not None:
            self.covariate_2nd_net.train(False)
        with torch.no_grad():
            treatment_1st_feature_1st = self.treatment_1st_net(a1)
            treatment_1st_feature_2nd = self.treatment_1st_net(a2)
            treatment_2nd_feature_2nd = self.treatment_2nd_net(a2)
            treatment_proxy_feature_1st = self.treatment_proxy_net(z1)
            treatment_proxy_feature_2nd = self.treatment_proxy_net(z2)
            outcome_proxy_feature_1st = self.outcome_proxy_net(w1)
            covariate_1st_feature_1st = self.covariate_1st_net(x1) if self.covariate_1st_net is not None else None
            covariate_1st_feature_2nd = self.covariate_1st_net(x2) if self.covariate_1st_net is not None else None
            covariate_2nd_feature_2nd = self.covariate_2nd_net(x2) if self.covariate_2nd_net is not None else None
            res = self._fit_2sls(
                treatment_1st_feature_1st,
                treatment_1st_feature_2nd,
                treatment_2nd_feature_2nd,
                treatment_proxy_feature_1st,
                treatment_proxy_feature_2nd,
                outcome_proxy_feature_1st,
                covariate_1st_feature_1st,
                covariate_1st_feature_2nd,
                covariate_2nd_feature_2nd,
                y2,
                sample_weight_2nd=sample_weight_2nd,
            )
            self.stage1_weight = res["stage1_weight"].detach().clone()
            self.stage2_weight = res["stage2_weight"].detach().clone()

    def _fit_bridge(
        self,
        data: BridgeTrainData,
        sample_weight: np.ndarray | None = None,
        *,
        verbose: int = 0,
        name: str = "DFPV",
        seed: int = 13,
    ) -> list[dict[str, object]]:
        verbose_print(verbose, 1, f"[start] {name} bridge")
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
                raise ValueError("DFPV sample_weight must match the number of input rows.")
            fit_idx = self.fit_indices_ if self.fit_indices_ is not None else np.arange(n, dtype=int)
            bridge_weight = raw_weight[fit_idx]

        bridge_weight = np.clip(bridge_weight, 1e-12, None)
        bridge_weight = bridge_weight / np.mean(bridge_weight)

        stage1_idx, stage2_idx = self._stage_split(n, seed)
        self.stage1_indices_ = stage1_idx.copy()
        self.stage2_indices_ = stage2_idx.copy()
        if verbose >= 3:
            verbose_print(
                verbose,
                3,
                (
                    f"[data] {name} bridge n_input={data.treatment.shape[0]} n_fit={n} "
                    f"n_stage1={stage1_idx.shape[0]} n_stage2={stage2_idx.shape[0]} "
                    f"include_covariates={self.model_config.include_covariates} "
                    f"feature_dim={self.model_config.feature_dim}"
                ),
            )
        self._build_networks(a.shape[1], z.shape[1], w.shape[1], x.shape[1])
        a_t = torch.tensor(a, dtype=torch.float32, device=self.device)
        z_t = torch.tensor(z, dtype=torch.float32, device=self.device)
        w_t = torch.tensor(w, dtype=torch.float32, device=self.device)
        x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.float32, device=self.device)
        bridge_weight_t = torch.tensor(bridge_weight, dtype=torch.float32, device=self.device).reshape(-1, 1)
        s1 = torch.tensor(stage1_idx, dtype=torch.long, device=self.device)
        s2 = torch.tensor(stage2_idx, dtype=torch.long, device=self.device)
        stage1_optimizer = torch.optim.Adam(
            self._stage1_params(),
            lr=self.model_config.lr,
            weight_decay=self.model_config.weight_decay,
        )
        stage2_optimizer = torch.optim.Adam(
            self._stage2_params(),
            lr=self.model_config.lr,
            weight_decay=self.model_config.weight_decay,
        )
        best_state = {
            "treatment_1st_net": copy.deepcopy(self.treatment_1st_net.state_dict()),
            "treatment_2nd_net": copy.deepcopy(self.treatment_2nd_net.state_dict()),
            "treatment_proxy_net": copy.deepcopy(self.treatment_proxy_net.state_dict()),
            "outcome_proxy_net": copy.deepcopy(self.outcome_proxy_net.state_dict()),
            "covariate_1st_net": copy.deepcopy(self.covariate_1st_net.state_dict()) if self.covariate_1st_net is not None else None,
            "covariate_2nd_net": copy.deepcopy(self.covariate_2nd_net.state_dict()) if self.covariate_2nd_net is not None else None,
        }
        best_loss = float("inf")
        best_epoch = 0
        stopped_epoch = int(self.model_config.epochs) - 1
        train_losses: list[float] = []
        val_losses: list[float] = []
        stopper = EarlyStoppingTracker(
            min_epochs=max(1, int(self.model_config.min_epochs)),
            patience=max(1, int(self.model_config.early_stop_patience)),
            min_delta=float(self.model_config.early_stop_min_delta),
            mode="min",
        )
        epoch_iter = progress_iter(range(int(self.model_config.epochs)), verbose, desc=f"{name}/bridge", leave=False)
        for epoch in epoch_iter:
            stage1_loss = self._stage1_update(a_t[s1], z_t[s1], w_t[s1], x_t[s1], stage1_optimizer)
            stage2_loss = self._stage2_update(
                a_t[s1],
                z_t[s1],
                w_t[s1],
                x_t[s1],
                a_t[s2],
                z_t[s2],
                x_t[s2],
                y_t[s2],
                stage2_optimizer,
                sample_weight_2nd=bridge_weight_t[s2],
            )
            train_losses.append(float(stage2_loss))
            val_losses.append(float(stage2_loss))
            if verbose >= 2 and hasattr(epoch_iter, "set_postfix"):
                epoch_iter.set_postfix(stage1=f"{stage1_loss:.4f}", stage2=f"{stage2_loss:.4f}", refresh=False)
            improved, should_stop = stopper.update(float(stage2_loss), epoch)
            if improved:
                best_loss = float(stage2_loss)
                best_epoch = epoch
                best_state = {
                    "treatment_1st_net": copy.deepcopy(self.treatment_1st_net.state_dict()),
                    "treatment_2nd_net": copy.deepcopy(self.treatment_2nd_net.state_dict()),
                    "treatment_proxy_net": copy.deepcopy(self.treatment_proxy_net.state_dict()),
                    "outcome_proxy_net": copy.deepcopy(self.outcome_proxy_net.state_dict()),
                    "covariate_1st_net": copy.deepcopy(self.covariate_1st_net.state_dict()) if self.covariate_1st_net is not None else None,
                    "covariate_2nd_net": copy.deepcopy(self.covariate_2nd_net.state_dict()) if self.covariate_2nd_net is not None else None,
                }
            if should_stop:
                stopped_epoch = epoch
                break
        else:
            stopped_epoch = max(0, int(self.model_config.epochs) - 1)

        self.treatment_1st_net.load_state_dict(best_state["treatment_1st_net"])
        self.treatment_2nd_net.load_state_dict(best_state["treatment_2nd_net"])
        self.treatment_proxy_net.load_state_dict(best_state["treatment_proxy_net"])
        self.outcome_proxy_net.load_state_dict(best_state["outcome_proxy_net"])
        if self.covariate_1st_net is not None and best_state["covariate_1st_net"] is not None:
            self.covariate_1st_net.load_state_dict(best_state["covariate_1st_net"])
        if self.covariate_2nd_net is not None and best_state["covariate_2nd_net"] is not None:
            self.covariate_2nd_net.load_state_dict(best_state["covariate_2nd_net"])
        self._finalize_bridge_weights(
            a_t[s1],
            z_t[s1],
            w_t[s1],
            x_t[s1],
            a_t[s2],
            z_t[s2],
            x_t[s2],
            y_t[s2],
            sample_weight_2nd=bridge_weight_t[s2],
        )
        history = [
            {
                "stage": f"{name}/bridge",
                "train_losses": train_losses,
                "val_losses": val_losses,
                "best_epoch": best_epoch,
                "stopped_epoch": stopped_epoch,
                "best_loss": best_loss,
                "metadata": {
                    "n_fit": int(n),
                    "n_stage1": int(stage1_idx.shape[0]),
                    "n_stage2": int(stage2_idx.shape[0]),
                    "include_covariates": bool(self.model_config.include_covariates),
                    "weighted_second_stage": True,
                    "weight_mean": float(np.mean(bridge_weight)),
                    "weight_min": float(np.min(bridge_weight)),
                    "weight_max": float(np.max(bridge_weight)),
                },
            }
        ]
        verbose_print(
            verbose,
            1,
            f"[done] {name} bridge best_epoch={best_epoch} stopped_epoch={stopped_epoch} loss={best_loss:.4f}",
        )
        return history

    def fit(
        self,
        data: BridgeTrainData,
        sample_weight: np.ndarray | None = None,
        mean_method: str = "pseudo_outcomes",
        *,
        verbose: int = 0,
        name: str = "DFPV",
        seed: int = 13,
    ) -> None:
        n = data.treatment.shape[0]
        if sample_weight is None:
            sample_weight = np.ones(n, dtype=float)
        sample_weight = np.asarray(sample_weight, dtype=float).reshape(-1)
        if sample_weight.shape[0] != n:
            raise ValueError("DFPV sample_weight must match the number of input rows.")

        self.training_history = self._fit_bridge(
            data,
            sample_weight=sample_weight,
            verbose=verbose,
            name=name,
            seed=seed,
        )
        self.mean_method = mean_method
        if mean_method != "pseudo_outcomes":
            raise ValueError("DFPVBridge currently supports mean_method='pseudo_outcomes'.")
        self._fit_mean_regressor(
            data,
            sample_weight=sample_weight,
            verbose=verbose,
            name=f"{name}/mean_regressor",
            seed=seed,
        )

    def bridge_predict(self, treatment: np.ndarray, outcome_proxy: np.ndarray, covariates: np.ndarray) -> np.ndarray:
        if self.stage2_weight is None:
            raise RuntimeError("DFPVBridge has not been fitted.")
        assert self.treatment_2nd_net is not None
        assert self.outcome_proxy_net is not None
        a = self.a_scaler.transform(_as_2d(treatment))
        w = self.w_scaler.transform(_as_2d(outcome_proxy))
        x = self.x_scaler.transform(_as_2d(covariates))
        n = a.shape[0]
        batch_size = max(1, int(self.model_config.prediction_batch_size))
        preds = []
        self.treatment_2nd_net.train(False)
        self.outcome_proxy_net.train(False)
        if self.covariate_2nd_net is not None:
            self.covariate_2nd_net.train(False)
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                a_t = torch.tensor(a[start:end], dtype=torch.float32, device=self.device)
                w_t = torch.tensor(w[start:end], dtype=torch.float32, device=self.device)
                x_t = torch.tensor(x[start:end], dtype=torch.float32, device=self.device)
                treatment_feature = self.treatment_2nd_net(a_t)
                outcome_proxy_feature = self.outcome_proxy_net(w_t)
                covariate_feature = self.covariate_2nd_net(x_t) if self.covariate_2nd_net is not None else None
                feature = self._augment_stage2_feature(outcome_proxy_feature, treatment_feature, covariate_feature)
                pred = _linear_reg_pred(feature, self.stage2_weight)
                preds.append(pred.cpu().numpy())
        return self.y_scaler.inverse_transform(np.vstack(preds))

    def _append_pseudo_outcomes(
        self,
        *,
        bridge: "DFPVBridge",
        data: BridgeTrainData,
        original_idx: np.ndarray,
        regressors_x: list[np.ndarray],
        pseudo_y: list[np.ndarray],
        group_ids: list[np.ndarray],
        grid: np.ndarray | None,
    ) -> None:
        x = _as_2d(data.covariates)
        w = _as_2d(data.outcome_proxy)
        if grid is None:
            pseudo_y.append(bridge.bridge_predict(data.treatment, w, x))
            regressors_x.append(np.concatenate([_as_2d(data.treatment), x], axis=1))
            group_ids.append(original_idx.reshape(-1))
            return
        for a_val in grid:
            a = np.full((x.shape[0], 1), float(a_val))
            pseudo_y.append(bridge.bridge_predict(a, w, x))
            regressors_x.append(np.concatenate([a, x], axis=1))
            group_ids.append(original_idx.reshape(-1))

    def _fit_mean_regressor(
        self,
        data: BridgeTrainData,
        sample_weight: np.ndarray,
        *,
        verbose: int = 0,
        name: str = "DFPV/mean_regressor",
        seed: int = 13,
    ) -> None:
        n = data.treatment.shape[0]
        use_treatment_grid = bool(self.mean_config.use_treatment_grid)
        grid = (
            np.linspace(self.support[0], self.support[1], int(self.mean_config.treatment_grid_size))
            if use_treatment_grid
            else None
        )
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
                    "train_weight_sum": float(np.sum(sample_weight)),
                    "holdout_weight_sum": float(np.sum(sample_weight)),
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
                fold_model = DFPVBridge(self.model_config, self.mean_config, self.w_flow_config, self.support)
                fold_history = fold_model._fit_bridge(
                    _subset_data(data, train_idx),
                    sample_weight=sample_weight[train_idx],
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
                        "train_weight_sum": float(np.sum(sample_weight[train_idx])),
                        "holdout_weight_sum": float(np.sum(sample_weight[holdout_idx])),
                    }
                )
        reg_x = np.concatenate(regressors_x, axis=0)
        reg_y = np.concatenate(pseudo_y, axis=0)
        reg_group_ids = np.concatenate(group_ids, axis=0)
        reg_row_weights = sample_weight[reg_group_ids]
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
            "n_rows": int(reg_x.shape[0]),
            "pseudo_outcome_std": self.mean_pseudooutcome_diagnostics_["std"],
            "pseudo_outcome_iqr": self.mean_pseudooutcome_diagnostics_["iqr"],
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
            raise RuntimeError("DFPVBridge has not been fitted with a mean regressor.")
        features = np.concatenate([_as_2d(treatment), _as_2d(covariates)], axis=1)
        return self.mean_regressor.predict(features)

