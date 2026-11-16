from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.config import KernelDiagnosticsConfig
from src.models.kernel_utils import median_bandwidth, rbf_kernel, rbf_product_kernel


@dataclass
class KernelDiagnosticResult:
    rows: list[dict[str, Any]]
    offdiag_rows: list[dict[str, Any]]
    spectrum_rows: list[dict[str, Any]]


def empty_kernel_diagnostic_result() -> KernelDiagnosticResult:
    return KernelDiagnosticResult(rows=[], offdiag_rows=[], spectrum_rows=[])


def merge_kernel_diagnostic_results(results: list[KernelDiagnosticResult]) -> KernelDiagnosticResult:
    merged = empty_kernel_diagnostic_result()
    for result in results:
        merged.rows.extend(result.rows)
        merged.offdiag_rows.extend(result.offdiag_rows)
        merged.spectrum_rows.extend(result.spectrum_rows)
    return merged


def subsample_indices(n: int, max_points: int, seed: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_points, replace=False))


def _offdiag_mask(n: int) -> np.ndarray:
    return ~np.eye(n, dtype=bool)


def _weighted_quantile(values: np.ndarray, quantiles: list[float], weights: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    quantiles_arr = np.asarray(quantiles, dtype=float)
    if values.size == 0:
        return np.full(quantiles_arr.shape, np.nan, dtype=float)
    if weights is None:
        return np.quantile(values, quantiles_arr)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    weights = np.maximum(weights, 0.0)
    if weights.size != values.size or float(np.sum(weights)) <= 0.0:
        return np.quantile(values, quantiles_arr)
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    cumulative = cumulative / max(float(cumulative[-1]), 1e-12)
    return np.interp(quantiles_arr, cumulative, sorted_values)


def _weighted_mean(values: np.ndarray, weights: np.ndarray | None = None) -> float:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return float("nan")
    if weights is None:
        return float(np.mean(values))
    weights = np.asarray(weights, dtype=float).reshape(-1)
    weights = np.maximum(weights, 0.0)
    total = float(np.sum(weights))
    if weights.size != values.size or total <= 0.0:
        return float(np.mean(values))
    return float(np.sum(values * weights) / total)


def _weighted_std(values: np.ndarray, weights: np.ndarray | None = None) -> float:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return float("nan")
    mean = _weighted_mean(values, weights)
    if weights is None:
        return float(np.std(values))
    weights = np.asarray(weights, dtype=float).reshape(-1)
    weights = np.maximum(weights, 0.0)
    total = float(np.sum(weights))
    if weights.size != values.size or total <= 0.0:
        return float(np.std(values))
    return float(np.sqrt(np.sum(weights * np.square(values - mean)) / total))


def _effective_sample_size(weights: np.ndarray | None) -> float:
    if weights is None:
        return float("nan")
    weights = np.asarray(weights, dtype=float).reshape(-1)
    denom = float(np.sum(np.square(weights)))
    if denom <= 0.0:
        return 0.0
    return float(np.square(np.sum(weights)) / denom)


def _spectral_summary(kernel: np.ndarray, sample_weight: np.ndarray | None = None) -> tuple[dict[str, float], np.ndarray]:
    k = np.asarray(kernel, dtype=float)
    k = 0.5 * (k + k.T)
    if sample_weight is not None:
        weights = np.asarray(sample_weight, dtype=float).reshape(-1)
        weights = np.maximum(weights, 0.0)
        if weights.size == k.shape[0] and float(np.mean(weights)) > 0.0:
            scaled = np.sqrt(weights / max(float(np.mean(weights)), 1e-12))
            k = scaled[:, None] * k * scaled[None, :]
    eigenvalues = np.linalg.eigvalsh(k)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = float(np.sum(eigenvalues))
    if total <= 1e-12:
        return {
            "effective_rank": 0.0,
            "effective_rank_ratio": 0.0,
            "top_eigen_share": 0.0,
            "condition_number": float("inf"),
        }, np.zeros_like(eigenvalues)
    shares = eigenvalues / total
    positive_shares = shares[shares > 0.0]
    effective_rank = float(np.exp(-np.sum(positive_shares * np.log(positive_shares))))
    positive_eigenvalues = eigenvalues[eigenvalues > 1e-12]
    condition_number = (
        float(np.max(positive_eigenvalues) / np.min(positive_eigenvalues))
        if positive_eigenvalues.size
        else float("inf")
    )
    return {
        "effective_rank": effective_rank,
        "effective_rank_ratio": effective_rank / max(k.shape[0], 1),
        "top_eigen_share": float(np.max(shares)),
        "condition_number": condition_number,
    }, shares[::-1]


def _classify_kernel(row: dict[str, Any], config: KernelDiagnosticsConfig) -> str:
    if row["near_zero_rate"] >= 0.90 or row["offdiag_q50"] <= config.near_zero_threshold:
        return "TOO_DIAGONAL"
    if row.get("avg_neighbors_0p1", float("inf")) <= 1.0:
        return "TOO_DIAGONAL"
    if row["near_one_rate"] >= 0.80:
        return "TOO_CONSTANT"
    if row["offdiag_std"] <= 0.02 and row["offdiag_mean"] >= 0.85:
        return "TOO_CONSTANT"
    if row["effective_rank_ratio"] <= 0.05 and row["top_eigen_share"] >= 0.80:
        return "TOO_CONSTANT"
    if row["effective_rank_ratio"] <= 0.05:
        return "LOW_RANK"
    return "OK"


def summarize_kernel(
    kernel: np.ndarray,
    *,
    metadata: dict[str, Any],
    config: KernelDiagnosticsConfig,
    sample_weight: np.ndarray | None = None,
) -> KernelDiagnosticResult:
    k = np.asarray(kernel, dtype=float)
    if k.ndim != 2 or k.shape[0] != k.shape[1]:
        raise ValueError("Kernel diagnostics require a square kernel matrix.")
    n = k.shape[0]
    mask = _offdiag_mask(n)
    offdiag = k[mask]

    pair_weights = None
    diag_weights = None
    if sample_weight is not None:
        weights = np.asarray(sample_weight, dtype=float).reshape(-1)
        if weights.size != n:
            raise ValueError("sample_weight length must match kernel shape.")
        weights = np.maximum(weights, 0.0)
        pair_weights = np.outer(weights, weights)[mask]
        diag_weights = weights

    quantiles = _weighted_quantile(offdiag, [0.01, 0.05, 0.50, 0.95, 0.99], pair_weights)
    spectral, eigen_shares = _spectral_summary(k, sample_weight)
    row: dict[str, Any] = {
        **metadata,
        "n": int(n),
        "diag_mean": _weighted_mean(np.diag(k), diag_weights),
        "offdiag_mean": _weighted_mean(offdiag, pair_weights),
        "offdiag_std": _weighted_std(offdiag, pair_weights),
        "offdiag_q01": float(quantiles[0]),
        "offdiag_q05": float(quantiles[1]),
        "offdiag_q50": float(quantiles[2]),
        "offdiag_q95": float(quantiles[3]),
        "offdiag_q99": float(quantiles[4]),
        "near_zero_rate": _weighted_mean((offdiag < config.near_zero_threshold).astype(float), pair_weights),
        "near_one_rate": _weighted_mean((offdiag > config.near_one_threshold).astype(float), pair_weights),
        "identity_distance": float(np.linalg.norm(k - np.eye(n), ord="fro") / max(np.sqrt(n), 1e-12)),
        "constant_distance": float(np.linalg.norm(k - np.ones((n, n)), ord="fro") / max(float(n), 1e-12)),
        "weight_ess": _effective_sample_size(sample_weight),
    }
    for threshold in config.neighbor_thresholds:
        key = f"avg_neighbors_{str(threshold).replace('.', 'p')}"
        row[key] = float(np.mean(np.sum((k > threshold) & mask, axis=1)))
    row.update(spectral)
    row["status"] = _classify_kernel(row, config)

    offdiag_rows: list[dict[str, Any]] = []
    if np.isclose(float(metadata.get("bandwidth_multiplier", 1.0)), 1.0):
        max_values = min(config.max_offdiag_values, offdiag.shape[0])
        if max_values > 0:
            rng = np.random.default_rng(config.seed)
            value_idx = (
                np.arange(offdiag.shape[0])
                if offdiag.shape[0] <= max_values
                else rng.choice(offdiag.shape[0], size=max_values, replace=False)
            )
            offdiag_rows = [{**metadata, "offdiag_value": float(value)} for value in offdiag[value_idx]]

    spectrum_rows = []
    if np.isclose(float(metadata.get("bandwidth_multiplier", 1.0)), 1.0):
        for eigen_index, share in enumerate(eigen_shares[: min(50, eigen_shares.shape[0])]):
            spectrum_rows.append({**metadata, "eigen_index": int(eigen_index), "eigen_share": float(share)})

    return KernelDiagnosticResult(rows=[row], offdiag_rows=offdiag_rows, spectrum_rows=spectrum_rows)


def _append_result(target: KernelDiagnosticResult, result: KernelDiagnosticResult) -> None:
    target.rows.extend(result.rows)
    target.offdiag_rows.extend(result.offdiag_rows)
    target.spectrum_rows.extend(result.spectrum_rows)


def diagnose_product_rbf_kernel(
    *,
    features: np.ndarray,
    bandwidths: np.ndarray,
    config: KernelDiagnosticsConfig,
    metadata: dict[str, Any],
    sample_weight: np.ndarray | None = None,
) -> KernelDiagnosticResult:
    features = np.asarray(features, dtype=float)
    idx = subsample_indices(features.shape[0], config.max_points, config.seed)
    features_sub = features[idx]
    weight_sub = None if sample_weight is None else np.asarray(sample_weight, dtype=float).reshape(-1)[idx]
    result = empty_kernel_diagnostic_result()
    base_bandwidths = np.asarray(bandwidths, dtype=float).reshape(-1)
    for multiplier in config.bandwidth_multipliers:
        scaled_bandwidths = np.maximum(base_bandwidths * float(multiplier), 1e-8)
        kernel = rbf_product_kernel(features_sub, features_sub, scaled_bandwidths)
        row_meta = {
            **metadata,
            "bandwidth_multiplier": float(multiplier),
            "bandwidth_min": float(np.min(scaled_bandwidths)),
            "bandwidth_mean": float(np.mean(scaled_bandwidths)),
            "bandwidth_max": float(np.max(scaled_bandwidths)),
        }
        _append_result(
            result,
            summarize_kernel(kernel, metadata=row_meta, config=config, sample_weight=weight_sub),
        )
    return result


def diagnose_rbf_kernel(
    *,
    features: np.ndarray,
    bandwidth: float | None,
    config: KernelDiagnosticsConfig,
    metadata: dict[str, Any],
    sample_weight: np.ndarray | None = None,
) -> KernelDiagnosticResult:
    features = np.asarray(features, dtype=float)
    idx = subsample_indices(features.shape[0], config.max_points, config.seed)
    features_sub = features[idx]
    weight_sub = None if sample_weight is None else np.asarray(sample_weight, dtype=float).reshape(-1)[idx]
    result = empty_kernel_diagnostic_result()
    base_bandwidth = float(bandwidth if bandwidth is not None else median_bandwidth(features_sub))
    for multiplier in config.bandwidth_multipliers:
        scaled_bandwidth = max(base_bandwidth * float(multiplier), 1e-8)
        kernel = rbf_kernel(features_sub, features_sub, scaled_bandwidth)
        row_meta = {
            **metadata,
            "bandwidth_multiplier": float(multiplier),
            "bandwidth_min": scaled_bandwidth,
            "bandwidth_mean": scaled_bandwidth,
            "bandwidth_max": scaled_bandwidth,
        }
        _append_result(
            result,
            summarize_kernel(kernel, metadata=row_meta, config=config, sample_weight=weight_sub),
        )
    return result
