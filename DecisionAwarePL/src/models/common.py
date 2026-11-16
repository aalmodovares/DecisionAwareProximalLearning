from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils import ArrayScaler, EarlyStoppingTracker, progress_iter, safe_float, split_index, verbose_print


def build_mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int = 1,
    activation: str = "elu",
) -> nn.Sequential:
    activation_map: dict[str, type[nn.Module]] = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
    }
    activation_cls = activation_map.get(str(activation).lower())
    if activation_cls is None:
        raise ValueError(f"Unknown activation {activation!r}.")
    layers: list[nn.Module] = []
    prev = input_dim
    for width in hidden_dims:
        layers.extend([nn.Linear(prev, width), activation_cls()])
        prev = width
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)




@dataclass
class FitResult:
    best_epoch: int
    stopped_epoch: int
    best_loss: float
    train_losses: list[float]
    val_losses: list[float]


class TorchRegressor:
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        lr: float,
        weight_decay: float,
        epochs: int,
        min_epochs: int,
        early_stop_patience: int,
        early_stop_min_delta: float,
        batch_size: int,
        early_stopping_mode: str = "standard",
        early_stopping_v2_alpha: float = 0.05,
        early_stopping_v2_patience: int = 5,
        device: str | None = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.input_dim = int(input_dim)
        self.model = build_mlp(input_dim, hidden_dims).to(self.device)
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.min_epochs = min_epochs
        self.early_stop_patience = early_stop_patience
        self.early_stop_min_delta = early_stop_min_delta
        self.early_stopping_mode = str(early_stopping_mode).lower()
        self.early_stopping_v2_alpha = float(early_stopping_v2_alpha)
        self.early_stopping_v2_patience = int(early_stopping_v2_patience)
        self.batch_size = batch_size
        self.x_scaler = ArrayScaler()
        self.y_scaler = ArrayScaler()
        self.fit_result: FitResult | None = None

    @staticmethod
    def _weighted_mse(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        denom = torch.clamp(weight.sum(), min=1e-8)
        return ((pred - target).square() * weight).sum() / denom

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        *,
        verbose: int = 0,
        name: str = "TorchRegressor",
        val_fraction: float = 0.2,
        seed: int = 13,
        group_ids: np.ndarray | None = None,
    ) -> FitResult:
        verbose_print(verbose, 1, f"[start] {name}")
        x_scaled = self.x_scaler.fit_transform(x)
        y_scaled = self.y_scaler.fit_transform(y)
        if sample_weight is None:
            sample_weight = np.ones(x.shape[0], dtype=float)
        sample_weight = np.asarray(sample_weight, dtype=float).reshape(-1, 1)

        if group_ids is None:
            train_idx, val_idx = split_index(x.shape[0], val_fraction=val_fraction, seed=seed)
        else:
            group_ids = np.asarray(group_ids).reshape(-1)
            if group_ids.shape[0] != x.shape[0]:
                raise ValueError("group_ids must have one entry per training row.")
            groups = np.unique(group_ids)
            group_train_idx, group_val_idx = split_index(groups.shape[0], val_fraction=val_fraction, seed=seed)
            train_groups = groups[group_train_idx]
            val_groups = groups[group_val_idx]
            train_idx = np.flatnonzero(np.isin(group_ids, train_groups))
            val_idx = np.flatnonzero(np.isin(group_ids, val_groups))
        x_train = x_scaled[train_idx]
        y_train = y_scaled[train_idx]
        w_train = sample_weight[train_idx]

        dataset = TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
            torch.tensor(w_train, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=min(self.batch_size, max(1, len(train_idx))), shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        train_eval_x = torch.tensor(x_train, dtype=torch.float32, device=self.device)
        train_eval_y = torch.tensor(y_train, dtype=torch.float32, device=self.device)
        train_eval_w = torch.tensor(w_train, dtype=torch.float32, device=self.device)
        val_eval_x = val_eval_y = val_eval_w = None
        if len(val_idx) > 0:
            val_eval_x = torch.tensor(x_scaled[val_idx], dtype=torch.float32, device=self.device)
            val_eval_y = torch.tensor(y_scaled[val_idx], dtype=torch.float32, device=self.device)
            val_eval_w = torch.tensor(sample_weight[val_idx], dtype=torch.float32, device=self.device)

        best_state = copy.deepcopy(self.model.state_dict())
        best_loss = float("inf")
        best_epoch = 0
        stopped_epoch = self.epochs - 1
        train_losses: list[float] = []
        val_losses: list[float] = []
        if self.early_stopping_mode not in {"standard", "v2"}:
            raise ValueError(
                f"Unknown early_stopping_mode {self.early_stopping_mode!r}. Expected 'standard' or 'v2'."
            )
        stopper = None
        if self.early_stopping_mode == "standard":
            stopper = EarlyStoppingTracker(
                min_epochs=max(1, self.min_epochs),
                patience=max(1, self.early_stop_patience),
                min_delta=self.early_stop_min_delta,
                mode="min",
            )
        else:
            v2_patience = max(1, self.early_stopping_v2_patience)
            v2_alpha = max(0.0, self.early_stopping_v2_alpha)
            last_within_epoch = 0
            consecutive_outside = 0

        epoch_iter = progress_iter(range(self.epochs), verbose, desc=name, leave=False)
        for epoch in epoch_iter:
            self.model.train(True)
            for xb, yb, wb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                wb = wb.to(self.device)
                pred = self.model(xb)
                loss = self._weighted_mse(pred, yb, wb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.model.train(False)
            with torch.no_grad():
                train_pred = self.model(train_eval_x)
                train_loss = self._weighted_mse(train_pred, train_eval_y, train_eval_w)
                train_value = safe_float(train_loss)
                if val_eval_x is not None:
                    val_pred = self.model(val_eval_x)
                    val_loss = self._weighted_mse(val_pred, val_eval_y, val_eval_w)
                    val_value = safe_float(val_loss)
                    monitor = val_value
                else:
                    val_value = train_value
                    monitor = train_value

            train_losses.append(train_value)
            val_losses.append(val_value)
            if verbose >= 2 and hasattr(epoch_iter, "set_postfix"):
                epoch_iter.set_postfix(train=f"{train_value:.4f}", val=f"{val_value:.4f}", refresh=False)

            if self.early_stopping_mode == "standard":
                improved, should_stop = stopper.update(monitor, epoch)
                if improved:
                    best_loss = monitor
                    best_epoch = epoch
                    best_state = copy.deepcopy(self.model.state_dict())
                if should_stop:
                    stopped_epoch = epoch
                    break
            else:
                if monitor < best_loss:
                    best_loss = monitor
                threshold = best_loss + v2_alpha
                if monitor <= threshold:
                    last_within_epoch = epoch
                    best_epoch = epoch
                    best_state = copy.deepcopy(self.model.state_dict())
                    consecutive_outside = 0
                else:
                    consecutive_outside += 1
                if (epoch + 1) >= max(1, self.min_epochs) and consecutive_outside >= v2_patience:
                    stopped_epoch = epoch
                    best_epoch = last_within_epoch
                    break
        else:
            stopped_epoch = max(0, self.epochs - 1)

        self.model.load_state_dict(best_state)
        self.model.train(False)
        self.fit_result = FitResult(
            best_epoch=best_epoch,
            stopped_epoch=stopped_epoch,
            best_loss=best_loss,
            train_losses=train_losses,
            val_losses=val_losses,
        )
        verbose_print(
            verbose,
            1,
            f"[done] {name} best_epoch={best_epoch} stopped_epoch={stopped_epoch} "
            f"train_loss={train_losses[best_epoch]:.4f} val_loss={val_losses[best_epoch]:.4f}",
        )
        return self.fit_result

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_scaled = self.x_scaler.transform(x)
        with torch.no_grad():
            pred = self.model(torch.tensor(x_scaled, dtype=torch.float32, device=self.device)).cpu().numpy()
        return self.y_scaler.inverse_transform(pred)
