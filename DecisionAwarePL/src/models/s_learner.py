from __future__ import annotations

import numpy as np

from src.config import MeanConfig, SLearnerConfig
from src.models.common import TorchRegressor


class SLearner:
    def __init__(self, model_config: SLearnerConfig, mean_config: MeanConfig | None = None, *unused_args):
        self.model_config = model_config
        self.mean_config = mean_config
        self.outcome_regressor: TorchRegressor | None = None
        self.mean_regressor: TorchRegressor | None = None
        self.direct_use_proxies: bool = True
        self.training_history: list[dict[str, object]] = []

    def _direct_features(
        self,
        treatment: np.ndarray,
        covariates: np.ndarray,
        treatment_proxy: np.ndarray,
        outcome_proxy: np.ndarray,
        *,
        include_proxies: bool,
    ) -> np.ndarray:
        if include_proxies:
            return np.concatenate([treatment, covariates, treatment_proxy, outcome_proxy], axis=1)
        return np.concatenate([treatment, covariates], axis=1)

    def fit(
        self,
        treatment: np.ndarray,
        treatment_proxy: np.ndarray,
        outcome_proxy: np.ndarray,
        covariates: np.ndarray,
        outcome: np.ndarray,
        sample_weight: np.ndarray | None = None,
        *,
        verbose: int = 0,
        name: str = "S-Learner",
        seed: int = 13,
        include_proxies: bool = True,
    ) -> None:
        self.direct_use_proxies = bool(include_proxies)
        features = self._direct_features(
            treatment,
            covariates,
            treatment_proxy,
            outcome_proxy,
            include_proxies=self.direct_use_proxies,
        )
        self.outcome_regressor = TorchRegressor(
            input_dim=features.shape[1],
            hidden_dims=self.model_config.hidden_dims,
            lr=self.model_config.lr,
            weight_decay=self.model_config.weight_decay,
            epochs=self.model_config.epochs,
            min_epochs=self.model_config.min_epochs,
            early_stop_patience=self.model_config.early_stop_patience,
            early_stop_min_delta=self.model_config.early_stop_min_delta,
            batch_size=self.model_config.batch_size,
        )
        outcome_fit = self.outcome_regressor.fit(
            features,
            outcome,
            sample_weight=sample_weight,
            verbose=verbose,
            name=f"{name}/outcome_regressor",
            val_fraction=self.model_config.val_fraction,
            seed=seed,
        )
        self.training_history = [
            {
                "stage": f"{name}/outcome_regressor",
                "train_losses": outcome_fit.train_losses,
                "val_losses": outcome_fit.val_losses,
                "best_epoch": outcome_fit.best_epoch,
                "stopped_epoch": outcome_fit.stopped_epoch,
            }
        ]

    def fit_mean_regressor(
        self,
        treatment_grid: np.ndarray,
        covariates_x: np.ndarray,
        treatment_proxy: np.ndarray,
        outcome_proxy: np.ndarray,
        direct_covariates: np.ndarray,
        sample_weight: np.ndarray | None = None,
        *,
        verbose: int = 0,
        name: str = "S-Learner",
        seed: int = 13,
    ) -> None:
        if self.mean_config is None:
            raise RuntimeError("SLearner mean_config is required to fit the mean regressor.")
        if self.outcome_regressor is None:
            raise RuntimeError("SLearner must be fitted before calling fit_mean_regressor.")

        grid = np.asarray(treatment_grid, dtype=float).reshape(-1)
        x_policy = np.asarray(covariates_x, dtype=float)
        z = np.asarray(treatment_proxy, dtype=float)
        w = np.asarray(outcome_proxy, dtype=float)
        x_direct = np.asarray(direct_covariates, dtype=float)
        n = x_policy.shape[0]

        a_rep = np.repeat(grid, n).reshape(-1, 1)
        x_policy_rep = np.tile(x_policy, (grid.size, 1))
        z_rep = np.tile(z, (grid.size, 1))
        w_rep = np.tile(w, (grid.size, 1))
        x_direct_rep = np.tile(x_direct, (grid.size, 1))

        pseudo = self.direct_predict(a_rep, z_rep, w_rep, x_direct_rep)
        if sample_weight is None:
            expanded_weight = None
        else:
            base_weight = np.asarray(sample_weight, dtype=float).reshape(-1)
            expanded_weight = np.tile(base_weight, grid.size)
        group_ids = np.tile(np.arange(n, dtype=int), grid.size)

        self.fit_mean_regressor_from_pseudo(
            treatment=a_rep,
            covariates=x_policy_rep,
            pseudo_outcome=pseudo,
            sample_weight=expanded_weight,
            group_ids=group_ids,
            verbose=verbose,
            name=name,
            seed=seed,
        )

    def fit_mean_regressor_from_pseudo(
        self,
        treatment: np.ndarray,
        covariates: np.ndarray,
        pseudo_outcome: np.ndarray,
        sample_weight: np.ndarray | None = None,
        group_ids: np.ndarray | None = None,
        *,
        verbose: int = 0,
        name: str = "S-Learner",
        seed: int = 13,
    ) -> None:
        if self.mean_config is None:
            raise RuntimeError("SLearner mean_config is required to fit the mean regressor.")
        mean_features = np.concatenate([treatment, covariates], axis=1)

        self.mean_regressor = TorchRegressor(
            input_dim=mean_features.shape[1],
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
        mean_fit = self.mean_regressor.fit(
            mean_features,
            pseudo_outcome,
            sample_weight=sample_weight,
            verbose=verbose,
            name=f"{name}/mean_regressor",
            val_fraction=self.mean_config.val_fraction,
            seed=seed,
            group_ids=group_ids,
        )
        self.training_history.append(
            {
                "stage": f"{name}/mean_regressor",
                "train_losses": mean_fit.train_losses,
                "val_losses": mean_fit.val_losses,
                "best_epoch": mean_fit.best_epoch,
                "stopped_epoch": mean_fit.stopped_epoch,
            }
        )

    def direct_predict(self, treatment: np.ndarray, treatment_proxy: np.ndarray, outcome_proxy: np.ndarray, covariates: np.ndarray) -> np.ndarray:
        if self.outcome_regressor is None:
            raise RuntimeError("SLearner must be fitted before calling direct_predict.")
        features = self._direct_features(
            treatment,
            covariates,
            treatment_proxy,
            outcome_proxy,
            include_proxies=self.direct_use_proxies,
        )
        return self.outcome_regressor.predict(features)

    def mean_predict(self, treatment: np.ndarray, covariates: np.ndarray) -> np.ndarray:
        if self.mean_regressor is None:
            raise RuntimeError("SLearner mean_regressor is not fitted. Call fit_mean_regressor first.")
        features = np.concatenate([treatment, covariates], axis=1)
        return self.mean_regressor.predict(features)

