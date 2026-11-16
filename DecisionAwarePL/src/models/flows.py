from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from zuko.flows import NSF

from src.config import FlowConfig
from src.utils import ArrayScaler, EarlyStoppingTracker, progress_iter, safe_float, split_index, verbose_print


@dataclass
class SupportTransform:
    lower: float
    upper: float
    eps: float = 1e-4

    def forward(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u = np.clip((y - self.lower) / (self.upper - self.lower), self.eps, 1.0 - self.eps)
        z = np.log(u) - np.log1p(-u)
        log_abs_det = -np.log(self.upper - self.lower) - np.log(u) - np.log1p(-u)
        return z, log_abs_det

    def inverse(self, z: np.ndarray) -> np.ndarray:
        u = 1.0 / (1.0 + np.exp(-z))
        return self.lower + (self.upper - self.lower) * u


class ConditionalFlowDensity:
    def __init__(
        self,
        features: int,
        context_dim: int,
        config: FlowConfig,
        support: tuple[float, float] | None = None,
        device: str | None = None,
    ):
        self.features = features
        self.context_dim = context_dim
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.context_scaler = ArrayScaler()
        self.target_scaler = ArrayScaler()
        self.support_transform = SupportTransform(*support) if support is not None else None
        self.flow = NSF(
            features=features,
            context=context_dim,
            bins=config.bins,
            transforms=config.transforms,
            hidden_features=config.hidden_features,
        ).to(self.device)
        self.fit_result: dict[str, list[float] | float | int] | None = None

    def _forward_target(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        transformed = np.asarray(y, dtype=float)
        log_det = np.zeros_like(transformed)
        if self.support_transform is not None:
            transformed, log_det = self.support_transform.forward(transformed)
        transformed = self.target_scaler.fit_transform(transformed)
        log_det = log_det - np.log(self.target_scaler.scale_)
        return transformed, log_det

    def _forward_target_eval(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        transformed = np.asarray(y, dtype=float)
        log_det = np.zeros_like(transformed)
        if self.support_transform is not None:
            transformed, log_det = self.support_transform.forward(transformed)
        transformed = self.target_scaler.transform(transformed)
        log_det = log_det - np.log(self.target_scaler.scale_)
        return transformed, log_det

    def _inverse_target(self, transformed: np.ndarray) -> np.ndarray:
        out = self.target_scaler.inverse_transform(transformed)
        if self.support_transform is not None:
            out = self.support_transform.inverse(out)
        return out

    @staticmethod
    def _neg_loglik(flow: NSF, y_arr: np.ndarray, c_arr: np.ndarray, device: torch.device) -> torch.Tensor:
        y_tensor = torch.tensor(y_arr, dtype=torch.float32, device=device)
        c_tensor = torch.tensor(c_arr, dtype=torch.float32, device=device)
        return -flow(c_tensor).log_prob(y_tensor).mean()

    def fit(
        self,
        y: np.ndarray,
        context: np.ndarray,
        *,
        verbose: int = 0,
        name: str = "ConditionalFlowDensity",
        seed: int = 13,
    ) -> dict[str, list[float] | float | int]:
        verbose_print(verbose, 1, f"[start] {name}")
        context_scaled = self.context_scaler.fit_transform(context)
        y_scaled, _ = self._forward_target(y)

        train_idx, val_idx = split_index(y.shape[0], val_fraction=self.config.val_fraction, seed=seed)
        dataset = TensorDataset(
            torch.tensor(y_scaled[train_idx], dtype=torch.float32),
            torch.tensor(context_scaled[train_idx], dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=min(self.config.batch_size, max(1, len(train_idx))), shuffle=True)
        optimizer = torch.optim.Adam(
            self.flow.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        best_state = {key: value.detach().cpu().clone() for key, value in self.flow.state_dict().items()}
        best_loss = float("inf")
        best_epoch = 0
        stopped_epoch = self.config.epochs - 1
        train_losses: list[float] = []
        val_losses: list[float] = []
        stopper = EarlyStoppingTracker(
            min_epochs=max(1, self.config.min_epochs),
            patience=max(1, self.config.early_stop_patience),
            min_delta=self.config.early_stop_min_delta,
            mode="min",
        )

        epoch_iter = progress_iter(range(self.config.epochs), verbose, desc=name, leave=False)
        for epoch in epoch_iter:
            self.flow.train(True)
            for yb, cb in loader:
                yb = yb.to(self.device)
                cb = cb.to(self.device)
                loss = -self.flow(cb).log_prob(yb).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.flow.train(False)
            with torch.no_grad():
                train_loss = self._neg_loglik(self.flow, y_scaled[train_idx], context_scaled[train_idx], self.device)
                train_value = safe_float(train_loss)
                if len(val_idx) > 0:
                    val_loss = self._neg_loglik(self.flow, y_scaled[val_idx], context_scaled[val_idx], self.device)
                    val_value = safe_float(val_loss)
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
                best_state = {key: value.detach().cpu().clone() for key, value in self.flow.state_dict().items()}
            if should_stop:
                stopped_epoch = epoch
                break
        else:
            stopped_epoch = max(0, self.config.epochs - 1)

        self.flow.load_state_dict(best_state)
        self.flow.train(False)
        self.fit_result = {
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
            "best_loss": best_loss,
            "train_losses": train_losses,
            "val_losses": val_losses,
        }
        verbose_print(
            verbose,
            1,
            f"[done] {name} best_epoch={best_epoch} stopped_epoch={stopped_epoch} "
            f"train_loss={train_losses[best_epoch]:.4f} val_loss={val_losses[best_epoch]:.4f}",
        )
        return self.fit_result

    def log_prob(self, y: np.ndarray, context: np.ndarray) -> np.ndarray:
        context_scaled = self.context_scaler.transform(context)
        y_scaled, log_det = self._forward_target_eval(y)
        with torch.no_grad():
            base_log_prob = (
                self.flow(torch.tensor(context_scaled, dtype=torch.float32, device=self.device))
                .log_prob(torch.tensor(y_scaled, dtype=torch.float32, device=self.device))
                .cpu()
                .numpy()
            )
        return base_log_prob + np.sum(log_det, axis=1)

    def pdf(self, y: np.ndarray, context: np.ndarray) -> np.ndarray:
        return np.exp(self.log_prob(y, context))

    def sample(self, context: np.ndarray, n_samples: int) -> np.ndarray:
        context_scaled = self.context_scaler.transform(context)
        with torch.no_grad():
            dist = self.flow(torch.tensor(context_scaled, dtype=torch.float32, device=self.device))
            samples = dist.sample((n_samples,)).cpu().numpy()
        flat = samples.reshape(-1, self.features)
        original = self._inverse_target(flat)
        return original.reshape(n_samples, context.shape[0], self.features)
