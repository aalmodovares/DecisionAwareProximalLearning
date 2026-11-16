from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm


class ArrayScaler:
    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "ArrayScaler":
        arr = np.asarray(x, dtype=float)
        self.mean_ = arr.mean(axis=0, keepdims=True)
        self.scale_ = arr.std(axis=0, keepdims=True)
        self.scale_[self.scale_ < 1e-8] = 1.0
        return self

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.transform(x)

    def transform(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        return (arr - self.mean_) / self.scale_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        return arr * self.scale_ + self.mean_


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Any, path: Path) -> None:
    if is_dataclass(data):
        data = asdict(data)
    path.write_text(json.dumps(data, indent=2, default=_json_default))


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable.")


@dataclass
class EarlyStoppingTracker:
    min_epochs: int
    patience: int
    min_delta: float
    mode: str = "min"
    best_score: float | None = None
    best_epoch: int = 0
    bad_epochs: int = 0

    def update(self, score: float, epoch: int) -> tuple[bool, bool]:
        if self.mode not in {"min", "max"}:
            raise ValueError(f"Unknown early stopping mode {self.mode!r}.")
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.bad_epochs = 0
            return True, False

        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        should_stop = epoch + 1 >= self.min_epochs and self.bad_epochs >= self.patience
        return improved, should_stop


def train_val_split(n: int, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(round(n * val_fraction)))
    return idx[n_val:], idx[:n_val]


def kfold_indices(
    n: int,
    n_folds: int,
    seed: int = 13,
    *,
    shuffle: bool = True,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if n <= 1:
        idx = np.arange(n)
        return [(idx, idx)]
    if n_folds <= 1:
        idx = np.arange(n)
        return [(idx, idx)]
    n_folds = min(int(n_folds), n)
    idx = np.arange(n)
    if shuffle:
        idx = np.random.default_rng(seed).permutation(idx)
    folds = [fold.astype(int, copy=False) for fold in np.array_split(idx, n_folds) if fold.size > 0]
    out = []
    for fold_id, holdout_idx in enumerate(folds):
        train_parts = [folds[j] for j in range(len(folds)) if j != fold_id]
        train_idx = np.concatenate(train_parts).astype(int, copy=False) if train_parts else holdout_idx.copy()
        out.append((train_idx, holdout_idx.astype(int, copy=False)))
    return out


def effective_sample_size(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float).reshape(-1)
    denom = np.square(w).sum()
    if denom <= 0.0:
        return 0.0
    return float(np.square(w.sum()) / denom)




def verbose_print(verbose: int, threshold: int, message: str) -> None:
    if verbose >= threshold:
        tqdm.write(str(message), file=_progress_stream())


def progress_iter(iterable, verbose: int, desc: str, leave: bool = False):
    if verbose >= 2:
        stream = _progress_stream()
        return tqdm(
            iterable,
            desc=desc,
            leave=leave,
            dynamic_ncols=True,
            file=stream,
            position=0,
            mininterval=0.2,
            maxinterval=1.0,
            disable=not stream.isatty(),
        )
    return iterable


def _progress_stream():
    if sys.stderr is not None and sys.stderr.isatty():
        return sys.stderr
    return sys.stdout


def safe_float(value: float | np.ndarray | torch.Tensor | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    if isinstance(value, np.ndarray):
        return float(np.asarray(value).reshape(-1)[0])
    return float(value)


def split_index(
    n: int,
    val_fraction: float,
    seed: int = 13,
) -> tuple[np.ndarray, np.ndarray]:
    if n <= 2 or val_fraction <= 0.0:
        idx = np.arange(n)
        return idx, np.array([], dtype=int)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = min(max(1, int(round(n * val_fraction))), n - 1)
    return idx[n_val:], idx[:n_val]
