from __future__ import annotations

import numpy as np


def median_bandwidth(x: np.ndarray, scale: float = 1.0) -> float:
    if x.shape[0] <= 1:
        return 1.0
    subset = x[: min(1000, x.shape[0])]
    diff = subset[:, None, :] - subset[None, :, :]
    dist = np.sqrt(np.square(diff).sum(axis=-1))
    positive = dist[dist > 0]
    if positive.size == 0:
        return 1.0
    return float(np.median(positive) * scale)


def median_bandwidth_per_dimension(x: np.ndarray, scale: float = 1.0) -> np.ndarray:
    if x.shape[0] <= 1:
        return np.ones(x.shape[1], dtype=float)
    subset = x[: min(1000, x.shape[0])]
    bandwidths = []
    for dim in range(subset.shape[1]):
        diff = np.abs(subset[:, None, dim] - subset[None, :, dim])
        positive = diff[diff > 0]
        if positive.size == 0:
            bandwidths.append(1.0)
        else:
            bandwidths.append(float(np.median(positive) * scale))
    return np.maximum(np.asarray(bandwidths, dtype=float), 1e-6)


def rbf_kernel(x1: np.ndarray, x2: np.ndarray, bandwidth: float) -> np.ndarray:
    bandwidth = max(float(bandwidth), 1e-6)
    diff = x1[:, None, :] - x2[None, :, :]
    sqdist = np.square(diff).sum(axis=-1)
    return np.exp(-sqdist / (2.0 * bandwidth * bandwidth))


def rbf_product_kernel(x1: np.ndarray, x2: np.ndarray, bandwidths: np.ndarray) -> np.ndarray:
    bandwidths = np.maximum(np.asarray(bandwidths, dtype=float).reshape(1, 1, -1), 1e-6)
    diff = (x1[:, None, :] - x2[None, :, :]) / bandwidths
    sqdist = np.square(diff).sum(axis=-1)
    return np.exp(-0.5 * sqdist)
