from __future__ import annotations

import numpy as np
from scipy.stats import gaussian_kde, norm


GAUSSIAN_TRUNCATION_MASS = float(norm.cdf(1.0) - norm.cdf(-1.0))


def gaussian_oosr_kernel(scaled_distance: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * np.square(scaled_distance)) / np.sqrt(2.0 * np.pi)


def truncated_gaussian_oosr_kernel(scaled_distance: np.ndarray) -> np.ndarray:
    scaled = np.asarray(scaled_distance, dtype=float)
    values = gaussian_oosr_kernel(scaled) / GAUSSIAN_TRUNCATION_MASS
    return np.where(np.abs(scaled) <= 1.0, values, 0.0)


def _paper_grid_normalizer(
    target_treatment: np.ndarray,
    tau: float,
    support: tuple[float, float],
    grid_size: int,
    *,
    truncated_kernel: bool,
) -> np.ndarray:
    grid = np.linspace(support[0], support[1], grid_size + 1)
    scaled = (target_treatment[:, None] - grid[None, :]) / tau
    kernel = truncated_gaussian_oosr_kernel if truncated_kernel else gaussian_oosr_kernel
    return kernel(scaled).mean(axis=1)


def _exact_kernel_normalizer(
    target_treatment: np.ndarray,
    tau: float,
    support: tuple[float, float],
    *,
    truncated_kernel: bool,
) -> np.ndarray:
    a, b = support
    rho = np.asarray(target_treatment, dtype=float).reshape(-1)
    if truncated_kernel:
        lower_u = np.maximum((rho - b) / tau, -1.0)
        upper_u = np.minimum((rho - a) / tau, 1.0)
        mass = np.maximum(norm.cdf(upper_u) - norm.cdf(lower_u), 0.0)
        return tau * mass / (GAUSSIAN_TRUNCATION_MASS * (b - a))

    upper = norm.cdf((b - rho) / tau)
    lower = norm.cdf((a - rho) / tau)
    return tau * (upper - lower) / (b - a)


def local_oosr_factor(
    treatment: np.ndarray,
    pseudo_optimal_treatment: np.ndarray,
    tau: float,
    support: tuple[float, float],
    normalization_mode: str,
    grid_size: int,
    truncated_kernel: bool = True,
) -> np.ndarray:
    """Compute the OOSR local factor.

    Modes:
    - `none`: matches the paper expression K((rho(x) - t) / tau) in Eq. (10).
    - `paper_normalization`: matches the released OOSR code, which divides by a
      grid-based normalizer over the treatment range.
    - `exact_normalization`: exact continuous-support replacement of the code's
      grid normalizer for the kernel used in OOSR.
    """

    t = np.asarray(treatment, dtype=float).reshape(-1)
    rho = np.asarray(pseudo_optimal_treatment, dtype=float).reshape(-1)
    kernel = truncated_gaussian_oosr_kernel if truncated_kernel else gaussian_oosr_kernel
    base = kernel((rho - t) / tau)

    if normalization_mode == "none":
        return base
    if normalization_mode == "paper_normalization":
        denom = _paper_grid_normalizer(rho, tau, support, grid_size, truncated_kernel=truncated_kernel)
        return base / np.clip(denom, 1e-8, None)
    if normalization_mode == "exact_normalization":
        denom = _exact_kernel_normalizer(rho, tau, support, truncated_kernel=truncated_kernel)
        return base / np.clip(denom, 1e-8, None)
    raise ValueError(f"Unknown normalization mode: {normalization_mode}")


def marginal_density_kde(treatment: np.ndarray) -> gaussian_kde:
    return gaussian_kde(np.asarray(treatment, dtype=float).reshape(-1))


def build_base_density_weights(
    treatment: np.ndarray,
    conditional_density: np.ndarray,
    support: tuple[float, float],
    stabilized: bool = False,
) -> np.ndarray:
    t = np.asarray(treatment, dtype=float).reshape(-1)
    cond = np.asarray(conditional_density, dtype=float).reshape(-1)
    cond = np.clip(cond, 1e-8, None)
    if stabilized:
        marginal = np.clip(marginal_density_kde(t)(t), 1e-8, None)
        return marginal / cond
    return 1.0 / ((support[1] - support[0]) * cond)


def build_local_only_weights(
    treatment: np.ndarray,
    pseudo_optimal_treatment: np.ndarray,
    conditional_density: np.ndarray,
    tau: float,
    *,
    truncated_kernel: bool = True,
) -> np.ndarray:
    t = np.asarray(treatment, dtype=float).reshape(-1)
    rho = np.asarray(pseudo_optimal_treatment, dtype=float).reshape(-1)
    cond = np.clip(np.asarray(conditional_density, dtype=float).reshape(-1), 1e-8, None)
    kernel = truncated_gaussian_oosr_kernel if truncated_kernel else gaussian_oosr_kernel
    local = kernel((rho - t) / tau)
    return local / (float(tau) * cond)


def finalize_weights(weights: np.ndarray, clip_min: float, clip_max: float) -> np.ndarray:
    w = np.asarray(weights, dtype=float).reshape(-1)
    w /= np.mean(w)
    w = np.clip(w, clip_min, clip_max)
    w /= np.mean(w)
    return w
