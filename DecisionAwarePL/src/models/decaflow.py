from __future__ import annotations

import copy
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import DeCaFlowConfig


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DECAFLOW_ROOT = _PROJECT_ROOT / "vendors" / "DeCaFlow_modified"
if _DECAFLOW_ROOT.exists() and str(_DECAFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(_DECAFLOW_ROOT))


def _import_decaflow_components() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    try:
        import lightning as L
        from lightning.pytorch.callbacks import Callback, EarlyStopping
        from decaflow.models import DeCaFlow, Decoder, Encoder, PriorConditioner, has_conditional_prior
    except ImportError as exc:
        raise ImportError(
            "Conditional DeCaFlow dependencies are unavailable. "
            f"Expected to import from {_DECAFLOW_ROOT} and the installed lightning/causalflows packages."
        ) from exc
    return L, Callback, EarlyStopping, DeCaFlow, Decoder, Encoder, PriorConditioner, has_conditional_prior


def _activation_cls(name: str) -> type[torch.nn.Module]:
    activation_map: dict[str, type[torch.nn.Module]] = {
        "elu": torch.nn.ELU,
        "relu": torch.nn.ReLU,
        "gelu": torch.nn.GELU,
        "tanh": torch.nn.Tanh,
        "silu": torch.nn.SiLU,
    }
    activation_cls = activation_map.get(str(name).lower())
    if activation_cls is None:
        raise ValueError(f"Unknown DeCaFlow activation {name!r}.")
    return activation_cls


def _safe_float(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return float("nan")
        value = value.detach().cpu().reshape(-1)[0].item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


@dataclass
class _ObservedSpec:
    observed_dim: int
    z_start: int
    z_end: int
    w_start: int
    w_end: int
    treatment_index: int
    outcome_index: int
    dim_z: int
    dim_w: int
    dim_x: int


class ConditionalDeCaFlow:
    def __init__(self, model_config: DeCaFlowConfig):
        self.model_config = model_config
        self.model = None
        self.training_history: list[dict[str, object]] = []
        self.best_val_metrics_: dict[str, float] = {}
        self._spec: _ObservedSpec | None = None
        self._device = torch.device("cpu")
        self.full_adjacency_: torch.BoolTensor | None = None
        self.encoder_adjacency_: torch.BoolTensor | None = None
        self.decoder_adjacency_: torch.BoolTensor | None = None
        self.prior_adjacency_: torch.BoolTensor | None = None

    @staticmethod
    def _build_observed_matrix(
        treatment: np.ndarray,
        treatment_proxy: np.ndarray,
        outcome_proxy: np.ndarray,
        outcome: np.ndarray,
    ) -> np.ndarray:
        arrays = [
            np.asarray(treatment_proxy, dtype=np.float32),
            np.asarray(outcome_proxy, dtype=np.float32),
            np.asarray(treatment, dtype=np.float32).reshape(-1, 1),
            np.asarray(outcome, dtype=np.float32).reshape(-1, 1),
        ]
        return np.concatenate(arrays, axis=1).astype(np.float32, copy=False)

    def _observed_spec(
        self,
        treatment_proxy: np.ndarray,
        outcome_proxy: np.ndarray,
        covariates: np.ndarray,
    ) -> _ObservedSpec:
        if self._spec is None:
            dim_z = int(np.asarray(treatment_proxy).shape[1])
            dim_w = int(np.asarray(outcome_proxy).shape[1])
            dim_x = int(np.asarray(covariates).shape[1])
            z_start = 0
            z_end = dim_z
            w_start = z_end
            w_end = w_start + dim_w
            treatment_index = w_end
            observed_dim = treatment_index + 2
            self._spec = _ObservedSpec(
                observed_dim=observed_dim,
                z_start=z_start,
                z_end=z_end,
                w_start=w_start,
                w_end=w_end,
                treatment_index=treatment_index,
                outcome_index=observed_dim - 1,
                dim_z=dim_z,
                dim_w=dim_w,
                dim_x=dim_x,
            )
        return self._spec

    def _build_adjacency(self, spec: _ObservedSpec) -> torch.BoolTensor:
        latent_dim = int(self.model_config.latent_dim)
        total_dim = latent_dim + spec.observed_dim + spec.dim_x
        if self.model_config.adjacency_override is not None:
            override = torch.as_tensor(self.model_config.adjacency_override, dtype=torch.bool)
            expected_shape = (total_dim, total_dim)
            if tuple(override.shape) != expected_shape:
                raise ValueError(
                    "decaflow.adjacency_override has incompatible shape. "
                    f"Expected {expected_shape}, got {tuple(override.shape)}."
                )
            return override.clone()
        adj = torch.eye(total_dim, dtype=torch.bool)

        latent_idx = list(range(latent_dim))
        z_idx = list(range(latent_dim + spec.z_start, latent_dim + spec.z_end))
        w_idx = list(range(latent_dim + spec.w_start, latent_dim + spec.w_end))
        treatment_idx = latent_dim + spec.treatment_index
        outcome_idx = latent_dim + spec.outcome_index
        x_idx = list(range(latent_dim + spec.observed_dim, total_dim))

        for obs_idx in [treatment_idx, *z_idx, *w_idx, outcome_idx]:
            for hidden_idx in latent_idx:
                adj[obs_idx, hidden_idx] = True

        for zi in z_idx:
            adj[treatment_idx, zi] = True

        for wi in w_idx:
            adj[outcome_idx, wi] = True

        adj[outcome_idx, treatment_idx] = True

        for cond_idx in x_idx:
            for node_idx in [*latent_idx, treatment_idx, *z_idx, *w_idx, outcome_idx]:
                adj[node_idx, cond_idx] = True

        return adj

    def _build_model(self, spec: _ObservedSpec):
        L, Callback, EarlyStopping, DeCaFlow, Decoder, Encoder, PriorConditioner, has_conditional_prior = _import_decaflow_components()
        activation_cls = _activation_cls(self.model_config.activation)
        adjacency = self._build_adjacency(spec)
        self.full_adjacency_ = adjacency.clone()

        encoder = Encoder(
            flow_type=self.model_config.flow_type,
            num_hidden=int(self.model_config.latent_dim),
            adjacency=adjacency.clone(),
            conditional_vars=spec.dim_x,
            features=int(self.model_config.latent_dim),
            context=spec.observed_dim + spec.dim_x,
            hidden_features=list(self.model_config.encoder_hidden_features),
            activation=activation_cls,
        )
        self.encoder_adjacency_ = encoder.adjacency.clone()
        decoder = Decoder(
            flow_type=self.model_config.flow_type,
            num_hidden=int(self.model_config.latent_dim),
            adjacency=adjacency.clone(),
            conditional_vars=spec.dim_x,
            features=spec.observed_dim,
            context=int(self.model_config.latent_dim) + spec.dim_x,
            hidden_features=list(self.model_config.decoder_hidden_features),
            activation=activation_cls,
        )
        self.decoder_adjacency_ = decoder.adjacency.clone()
        use_conditional_prior = spec.dim_x > 0 and has_conditional_prior(
            adjacency,
            int(self.model_config.latent_dim),
            spec.dim_x,
        )
        prior_conditioner = None
        self.prior_adjacency_ = None
        if use_conditional_prior:
            prior_conditioner = PriorConditioner(
                flow_type=self.model_config.prior_type,
                num_hidden=int(self.model_config.latent_dim),
                adjacency=adjacency,
                conditional_vars=spec.dim_x,
                features=int(self.model_config.latent_dim),
                context=spec.dim_x,
                hidden_features=list(self.model_config.conditioner_hidden_features),
                activation=activation_cls,
            )
            self.prior_adjacency_ = prior_conditioner.adjacency.clone()
        model = DeCaFlow(
            encoder=encoder,
            flow=decoder,
            regularize=bool(self.model_config.regularize),
            warmup=int(self.model_config.warmup),
            lr=float(self.model_config.lr),
            optimizer_cls=torch.optim.Adam,
            optimizer_kwargs={"weight_decay": float(self.model_config.weight_decay)},
            conditional_vars=spec.dim_x,
            prior_conditioner=prior_conditioner,
        )
        return L, Callback, EarlyStopping, model

    def fit(
        self,
        treatment: np.ndarray,
        treatment_proxy: np.ndarray,
        outcome_proxy: np.ndarray,
        covariates: np.ndarray,
        outcome: np.ndarray,
        *,
        val_treatment: np.ndarray,
        val_treatment_proxy: np.ndarray,
        val_outcome_proxy: np.ndarray,
        val_covariates: np.ndarray,
        val_outcome: np.ndarray,
        verbose: int = 0,
        name: str = "DeCaFlow",
        seed: int = 13,
    ) -> None:
        spec = self._observed_spec(treatment_proxy, outcome_proxy, covariates)
        L, Callback, EarlyStopping, model = self._build_model(spec)

        train_x = torch.tensor(
            self._build_observed_matrix(treatment, treatment_proxy, outcome_proxy, outcome),
            dtype=torch.float32,
        )
        train_c = torch.tensor(np.asarray(covariates, dtype=np.float32), dtype=torch.float32)
        val_x = torch.tensor(
            self._build_observed_matrix(
                val_treatment,
                val_treatment_proxy,
                val_outcome_proxy,
                val_outcome,
            ),
            dtype=torch.float32,
        )
        val_c = torch.tensor(np.asarray(val_covariates, dtype=np.float32), dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(train_x, train_c),
            batch_size=min(int(self.model_config.batch_size), int(train_x.shape[0])),
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(val_x, val_c),
            batch_size=min(int(self.model_config.batch_size), int(val_x.shape[0])),
            shuffle=False,
        )

        logger = None
        try:
            from decaflow.utils.logger import MyLogger

            logger = MyLogger()
        except Exception:
            logger = True

        class _BestStateCallback(Callback):
            def __init__(self):
                super().__init__()
                self.best_score = float("inf")
                self.best_state_dict: dict[str, torch.Tensor] | None = None
                self.best_epoch = -1
                self.best_metrics: dict[str, float] = {}

            def on_validation_epoch_end(self, trainer, pl_module) -> None:
                if getattr(trainer, "sanity_checking", False):
                    return
                score = _safe_float(trainer.callback_metrics.get("val_loss"))
                if not math.isfinite(score):
                    return
                if score < self.best_score - float(self_model_config.early_stop_min_delta):
                    self.best_score = score
                    self.best_epoch = int(trainer.current_epoch)
                    self.best_state_dict = copy.deepcopy(pl_module.state_dict())
                    self.best_metrics = {
                        "observational_loss": score,
                        "observational_log_prob": _safe_float(trainer.callback_metrics.get("val_log_prob")),
                        "observational_kl": _safe_float(trainer.callback_metrics.get("val_kl_z")),
                    }

        class _EpochMetricsCallback(Callback):
            def __init__(self):
                super().__init__()
                self.train_losses: list[float] = []
                self.val_losses: list[float] = []

            def on_validation_epoch_end(self, trainer, pl_module) -> None:
                if getattr(trainer, "sanity_checking", False):
                    return
                self.train_losses.append(_safe_float(trainer.callback_metrics.get("train_loss")))
                self.val_losses.append(_safe_float(trainer.callback_metrics.get("val_loss")))

        self_model_config = self.model_config
        best_state_callback = _BestStateCallback()
        epoch_metrics_callback = _EpochMetricsCallback()
        early_stopping = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=max(1, int(self.model_config.early_stop_patience)),
            min_delta=float(self.model_config.early_stop_min_delta),
        )

        torch.manual_seed(int(seed))
        trainer = L.Trainer(
            max_epochs=int(self.model_config.epochs),
            min_epochs=max(1, int(self.model_config.min_epochs)),
            logger=logger,
            callbacks=[epoch_metrics_callback, best_state_callback, early_stopping],
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=bool(verbose >= 2),
            log_every_n_steps=1,
            num_sanity_val_steps=0,
            deterministic=False,
        )
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        if best_state_callback.best_state_dict is not None:
            model.load_state_dict(best_state_callback.best_state_dict)
        model.eval()
        self.model = model
        self._device = next(model.parameters()).device

        train_losses = list(epoch_metrics_callback.train_losses)
        val_losses = list(epoch_metrics_callback.val_losses)

        if best_state_callback.best_metrics:
            self.best_val_metrics_ = dict(best_state_callback.best_metrics)
        else:
            self.best_val_metrics_ = self.observational_metrics(
                val_treatment,
                val_treatment_proxy,
                val_outcome_proxy,
                val_covariates,
                val_outcome,
            )

        best_epoch = int(best_state_callback.best_epoch if best_state_callback.best_epoch >= 0 else max(0, len(val_losses) - 1))
        stopped_epoch = max(0, len(val_losses) - 1) if val_losses else max(0, int(trainer.current_epoch))
        self.training_history = [
            {
                "stage": name,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "best_epoch": best_epoch,
                "stopped_epoch": stopped_epoch,
            }
        ]

    def _check_fitted(self) -> tuple[Any, _ObservedSpec]:
        if self.model is None or self._spec is None:
            raise RuntimeError("ConditionalDeCaFlow must be fitted before prediction.")
        return self.model, self._spec

    def _interventional_outcome_samples_scalar(
        self,
        treatment_value: float,
        covariates: np.ndarray,
    ) -> np.ndarray:
        model, spec = self._check_fitted()
        covariates_arr = np.asarray(covariates, dtype=np.float32)
        mc = max(1, int(self.model_config.interventional_mc_samples))
        repeated_covariates = np.repeat(covariates_arr, mc, axis=0)
        conditional_tensor = torch.tensor(repeated_covariates, dtype=torch.float32, device=self._device)

        model.eval()
        with torch.no_grad():
            sampled_x, _ = model.sample_interventional(
                index=spec.treatment_index,
                value=float(treatment_value),
                sample_shape=(repeated_covariates.shape[0],),
                conditional_samples=conditional_tensor,
            )
        outcome = sampled_x[:, spec.outcome_index].detach().cpu().numpy().reshape(-1)
        return outcome.reshape(covariates_arr.shape[0], mc)

    def mean_predict(self, treatment: np.ndarray, covariates: np.ndarray) -> np.ndarray:
        treatment_arr = np.asarray(treatment, dtype=np.float32).reshape(-1, 1)
        covariates_arr = np.asarray(covariates, dtype=np.float32)
        if treatment_arr.shape[0] != covariates_arr.shape[0]:
            raise ValueError("treatment and covariates must have the same number of rows.")
        batch_size = max(1, int(self.model_config.prediction_batch_size))
        preds: list[np.ndarray] = []
        same_treatment = bool(treatment_arr.shape[0] > 0 and np.allclose(treatment_arr, treatment_arr[0]))
        if same_treatment:
            scalar_treatment = float(treatment_arr[0, 0])
            for start in range(0, treatment_arr.shape[0], batch_size):
                end = min(treatment_arr.shape[0], start + batch_size)
                batch_samples = self._interventional_outcome_samples_scalar(
                    scalar_treatment,
                    covariates_arr[start:end],
                )
                preds.append(batch_samples.mean(axis=1))
        else:
            for row_id in range(treatment_arr.shape[0]):
                batch_samples = self._interventional_outcome_samples_scalar(
                    float(treatment_arr[row_id, 0]),
                    covariates_arr[row_id : row_id + 1],
                )
                preds.append(batch_samples.mean(axis=1))
        return np.concatenate(preds, axis=0).reshape(-1, 1)

    def mean_predict_grid(self, treatment_grid: np.ndarray, covariates: np.ndarray) -> np.ndarray:
        grid = np.asarray(treatment_grid, dtype=np.float32).reshape(-1)
        covariates_arr = np.asarray(covariates, dtype=np.float32)
        rows: list[np.ndarray] = []
        for a_val in grid:
            batch_treatment = np.full((covariates_arr.shape[0], 1), a_val, dtype=np.float32)
            rows.append(self.mean_predict(batch_treatment, covariates_arr).reshape(-1))
        return np.vstack(rows)

    def predict_surface(self, treatment_grid: np.ndarray, covariates: np.ndarray) -> np.ndarray:
        return self.mean_predict_grid(treatment_grid, covariates)

    def observational_metrics(
        self,
        treatment: np.ndarray,
        treatment_proxy: np.ndarray,
        outcome_proxy: np.ndarray,
        covariates: np.ndarray,
        outcome: np.ndarray,
    ) -> dict[str, float]:
        model, spec = self._check_fitted()
        x = torch.tensor(
            self._build_observed_matrix(treatment, treatment_proxy, outcome_proxy, outcome),
            dtype=torch.float32,
            device=self._device,
        )
        c = torch.tensor(np.asarray(covariates, dtype=np.float32), dtype=torch.float32, device=self._device)
        model.eval()
        with torch.no_grad():
            if model.encoder is not None:
                encoder_input = torch.cat([x, c], dim=-1)
                q_z = model.encoder(encoder_input)
                z = q_z.rsample()
                if model.prior_conditioner is None:
                    p_z = torch.distributions.Independent(
                        torch.distributions.Normal(loc=torch.zeros_like(z), scale=torch.ones_like(z)),
                        reinterpreted_batch_ndims=1,
                    )
                else:
                    p_z = model.prior_conditioner(c)
                kl_z = q_z.log_prob(z) - p_z.log_prob(z)
            else:
                z = None
                kl_z = None

            decoder_input = c if z is None else torch.cat([z, c], dim=-1)
            log_prob_x = model.flow(decoder_input).log_prob(x)
            if kl_z is not None:
                loss = -(log_prob_x - kl_z).mean()
                kl_value = kl_z.mean()
            else:
                loss = -log_prob_x.mean()
                kl_value = torch.tensor(float("nan"), device=self._device)
        return {
            "observational_loss": _safe_float(loss),
            "observational_log_prob": _safe_float(log_prob_x.mean()),
            "observational_kl": _safe_float(kl_value),
        }

    def validation_observational_metrics(self) -> dict[str, float]:
        if not self.best_val_metrics_:
            return {
                "observational_loss": float("nan"),
                "observational_log_prob": float("nan"),
                "observational_kl": float("nan"),
            }
        return dict(self.best_val_metrics_)
