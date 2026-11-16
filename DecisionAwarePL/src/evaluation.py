from __future__ import annotations

from typing import Callable

import numpy as np

from src.data.simulator import FactualSample, Wu2024Simulator


_CAUSAL_SURFACE_CACHE: dict[tuple[int, int, tuple[float, ...], int], dict[str, np.ndarray]] = {}


def build_causal_surface_reference(
    simulator: Wu2024Simulator,
    sample: FactualSample,
    treatment_grid: np.ndarray,
    *,
    n_global_x: int | None = 1000,
) -> dict[str, np.ndarray]:
    """Builds a cached Monte Carlo reference for E[Y(a) | X=x] on a grid.

    When n_global_x is None, all patients in the provided sample are used.
    """
    grid = np.asarray(treatment_grid, dtype=float).reshape(-1)
    n_available = int(sample.covariates.shape[0])
    if n_global_x is None:
        n = n_available
    else:
        n = min(n_available, max(0, int(n_global_x)))
    key = (id(simulator), id(sample), tuple(float(v) for v in grid.tolist()), int(n))
    cached = _CAUSAL_SURFACE_CACHE.get(key)
    if cached is not None:
        return cached

    x = sample.covariates[:n]
    if n == 0:
        empty = {
            "treatment_grid": grid,
            "covariates": x,
            "true_surface": np.zeros((grid.shape[0], 0), dtype=float),
            "local_treatment": np.zeros((0, 1), dtype=float),
            "local_true": np.zeros((0,), dtype=float),
        }
        _CAUSAL_SURFACE_CACHE[key] = empty
        return empty

    true_surface = np.empty((grid.shape[0], n), dtype=float)
    base_seed = int(simulator.config.causal_mc_seed)
    for idx in range(n):
        _, curve = simulator.causal_curve_given_x_mc(
            x[idx],
            grid=grid,
            seed=base_seed + idx,
        )
        true_surface[:, idx] = np.asarray(curve, dtype=float).reshape(-1)

    local_idx = np.argmax(true_surface, axis=0)
    local_treatment = grid[local_idx][:, None]
    local_true = true_surface[local_idx, np.arange(n)]
    out = {
        "treatment_grid": grid,
        "covariates": x,
        "true_surface": true_surface,
        "local_treatment": local_treatment,
        "local_true": local_true,
    }
    _CAUSAL_SURFACE_CACHE[key] = out
    return out


def policy_from_mean_predictor(
    covariates: np.ndarray,
    treatment_grid: np.ndarray,
    mean_predictor: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    values = []
    for a_val in treatment_grid:
        a = np.full((covariates.shape[0], 1), a_val)
        values.append(mean_predictor(a, covariates).reshape(-1))
    values = np.stack(values, axis=0)
    return treatment_grid[np.argmax(values, axis=0)]


def policy_from_direct_predictor(
    treatment_proxy: np.ndarray,
    outcome_proxy: np.ndarray,
    covariates: np.ndarray,
    treatment_grid: np.ndarray,
    direct_predictor: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    values = []
    for a_val in treatment_grid:
        a = np.full((covariates.shape[0], 1), a_val)
        values.append(direct_predictor(a, treatment_proxy, outcome_proxy, covariates).reshape(-1))
    values = np.stack(values, axis=0)
    return treatment_grid[np.argmax(values, axis=0)]


def evaluate_policy_metrics(
    simulator: Wu2024Simulator,
    sample: FactualSample,
    predicted_policy: np.ndarray,
    causal_reference: dict[str, np.ndarray] | None = None,
    n_global_x: int | None = None,
) -> dict[str, float]:
    predicted_policy = np.asarray(predicted_policy, dtype=float).reshape(-1)
    x = sample.covariates
    w = sample.outcome_proxy
    uz = sample.latent_uz
    uw = sample.latent_uw
    structural_optimal = sample.optimal_treatment.reshape(-1)

    # Structural objective used by simulator.policy_value / optimal_treatment.
    structural_value = simulator.policy_value(predicted_policy, x)
    structural_optimal_value = simulator.policy_value(structural_optimal, x)
    structural_regret = structural_optimal_value - structural_value
    structural_policy_mse = float(np.mean(np.square(predicted_policy - structural_optimal)))

    reference = causal_reference or build_causal_surface_reference(
        simulator,
        sample,
        simulator.policy_grid,
        n_global_x=n_global_x,
    )
    grid = np.asarray(reference["treatment_grid"], dtype=float).reshape(-1)
    local_treatment = np.asarray(reference["local_treatment"], dtype=float).reshape(-1)
    local_true = np.asarray(reference["local_true"], dtype=float).reshape(-1)
    true_surface = np.asarray(reference["true_surface"], dtype=float)
    n_local = local_treatment.shape[0]
    if n_local > 0:
        pred_local = predicted_policy[:n_local]
        # Policies are evaluated on the MC grid by nearest grid value per unit.
        pred_idx = np.argmin(np.abs(grid[:, None] - pred_local[None, :]), axis=0)
        pred_causal_values = true_surface[pred_idx, np.arange(n_local)]
        causal_marginal_value = float(np.mean(pred_causal_values))
        causal_marginal_optimal_value = float(np.mean(local_true))
        causal_marginal_regret = causal_marginal_optimal_value - causal_marginal_value
        causal_marginal_policy_mse = float(np.mean(np.square(pred_local - local_treatment)))
    else:
        causal_marginal_value = float("nan")
        causal_marginal_optimal_value = float("nan")
        causal_marginal_regret = float("nan")
        causal_marginal_policy_mse = float("nan")

    # Legacy policy value/regret now follow the causal-MC objective for default reporting.

    # Oracle objective conditions on latent confounders and observed outcome proxies.
    oracle_pred_value = float(
        np.mean(simulator.oracle_conditioned_outcome(predicted_policy, x, w, uz, uw))
    )
    oracle_surface = np.stack(
        [
            simulator.oracle_conditioned_outcome(
                np.full(x.shape[0], a_val, dtype=float),
                x,
                w,
                uz,
                uw,
            )
            for a_val in simulator.policy_grid
        ],
        axis=0,
    )
    oracle_optimal = simulator.policy_grid[np.argmax(oracle_surface, axis=0)]
    oracle_optimal_value = float(np.mean(np.max(oracle_surface, axis=0)))
    oracle_regret = oracle_optimal_value - oracle_pred_value
    oracle_policy_mse = float(np.mean(np.square(predicted_policy - oracle_optimal)))

    return {
        # Legacy keys (kept for compatibility with existing summaries/HPO objectives).
        "policy_value": causal_marginal_value,
        "optimal_policy_value": causal_marginal_optimal_value,
        "regret": causal_marginal_regret,
        "policy_mse": causal_marginal_policy_mse,
        # Explicit objectives.
        "policy_value_causal_marginal": causal_marginal_value,
        "optimal_policy_value_causal_marginal": causal_marginal_optimal_value,
        "regret_causal_marginal": causal_marginal_regret,
        "policy_mse_causal_marginal": causal_marginal_policy_mse,
        "policy_value_structural": structural_value,
        "optimal_policy_value_structural": structural_optimal_value,
        "regret_structural": structural_regret,
        "policy_mse_structural": structural_policy_mse,
        "policy_value_oracle": oracle_pred_value,
        "optimal_policy_value_oracle": oracle_optimal_value,
        "regret_oracle": oracle_regret,
        "policy_mse_oracle": oracle_policy_mse,
    }


def evaluate_mean_surface(
    simulator: Wu2024Simulator,
    sample: FactualSample,
    treatment_grid: np.ndarray,
    mean_predictor: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_global_x: int | None = None,
    causal_reference: dict[str, np.ndarray] | None = None,
) -> dict[str, float]:
    reference = causal_reference or build_causal_surface_reference(
        simulator,
        sample,
        treatment_grid,
        n_global_x=n_global_x,
    )
    grid = np.asarray(reference["treatment_grid"], dtype=float).reshape(-1)
    x = np.asarray(reference["covariates"], dtype=float)
    local_treatment = np.asarray(reference["local_treatment"], dtype=float)
    local_true = np.asarray(reference["local_true"], dtype=float).reshape(-1)
    global_true_arr = np.asarray(reference["true_surface"], dtype=float)

    local_pred = mean_predictor(local_treatment, x).reshape(-1)
    global_pred = []
    for a_val in grid:
        a = np.full((x.shape[0], 1), a_val)
        global_pred.append(mean_predictor(a, x).reshape(-1))
    global_pred_arr = np.stack(global_pred, axis=0)
    return {
        "local_rmse": float(np.sqrt(np.mean(np.square(local_pred - local_true)))),
        "global_rmse": float(np.sqrt(np.mean(np.square(global_pred_arr - global_true_arr)))),
    }


def evaluate_direct_surface(
    simulator: Wu2024Simulator,
    sample: FactualSample,
    treatment_grid: np.ndarray,
    direct_predictor: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    n_global_x: int | None = None,
    causal_reference: dict[str, np.ndarray] | None = None,
) -> dict[str, float]:
    reference = causal_reference or build_causal_surface_reference(
        simulator,
        sample,
        treatment_grid,
        n_global_x=n_global_x,
    )
    grid = np.asarray(reference["treatment_grid"], dtype=float).reshape(-1)
    x = np.asarray(reference["covariates"], dtype=float)
    n = x.shape[0]
    z = sample.treatment_proxy[:n]
    w = sample.outcome_proxy[:n]
    local_treatment = np.asarray(reference["local_treatment"], dtype=float)
    local_true = np.asarray(reference["local_true"], dtype=float).reshape(-1)
    global_true_arr = np.asarray(reference["true_surface"], dtype=float)

    local_pred = direct_predictor(local_treatment, z, w, x).reshape(-1)
    global_pred = []
    for a_val in grid:
        a = np.full((x.shape[0], 1), a_val)
        global_pred.append(direct_predictor(a, z, w, x).reshape(-1))
    global_pred_arr = np.stack(global_pred, axis=0)
    return {
        "local_rmse": float(np.sqrt(np.mean(np.square(local_pred - local_true)))),
        "global_rmse": float(np.sqrt(np.mean(np.square(global_pred_arr - global_true_arr)))),
    }


def marginal_curve_from_mean_predictor(
    sample: FactualSample,
    treatment_grid: np.ndarray,
    mean_predictor: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    """Estimate the population curve a -> E_X[mu_hat(a, X)]."""
    curves = []
    grid = np.asarray(treatment_grid, dtype=float).reshape(-1)
    for a_val in grid:
        a = np.full((sample.covariates.shape[0], 1), a_val)
        curves.append(float(np.mean(mean_predictor(a, sample.covariates))))
    return np.asarray(curves, dtype=float)


def marginal_curve_from_bridge_predictor(
    sample: FactualSample,
    treatment_grid: np.ndarray,
    bridge_predictor: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    """Estimate a -> E[h_hat(a, W, X)] over the empirical test distribution."""
    curves = []
    grid = np.asarray(treatment_grid, dtype=float).reshape(-1)
    for a_val in grid:
        a = np.full((sample.covariates.shape[0], 1), a_val)
        pred = bridge_predictor(a, sample.outcome_proxy, sample.covariates)
        curves.append(float(np.mean(pred)))
    return np.asarray(curves, dtype=float)


def marginal_curve_from_direct_predictor(
    sample: FactualSample,
    treatment_grid: np.ndarray,
    direct_predictor: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    """Estimate the direct-regression plug-in curve averaged over observed X,Z,W.

    This is the marginal curve implied by baselines that treat proxies as ordinary
    covariates. It is not a proximal bridge estimand; it averages their
    E[Y(a)|X,Z,W] predictions over the empirical test distribution.
    """
    curves = []
    grid = np.asarray(treatment_grid, dtype=float).reshape(-1)
    for a_val in grid:
        a = np.full((sample.covariates.shape[0], 1), a_val)
        pred = direct_predictor(a, sample.treatment_proxy, sample.outcome_proxy, sample.covariates)
        curves.append(float(np.mean(pred)))
    return np.asarray(curves, dtype=float)


def evaluate_marginal_curve(
    simulator: Wu2024Simulator,
    sample: FactualSample,
    treatment_grid: np.ndarray,
    predicted_curve: np.ndarray,
    prefix: str = "mean_curve",
    reference_curve: np.ndarray | None = None,
) -> dict[str, float]:
    grid, truth, _ = simulator.average_treatment_effect_curve(sample.covariates, grid=treatment_grid)
    if reference_curve is not None:
        truth = np.asarray(reference_curve, dtype=float).reshape(-1)
    predicted = np.asarray(predicted_curve, dtype=float).reshape(-1)
    truth = np.asarray(truth, dtype=float).reshape(-1)
    residual = predicted - truth
    pred_argmax = float(grid[int(np.argmax(predicted))])
    true_argmax = float(grid[int(np.argmax(truth))])
    return {
        f"{prefix}_rmse": float(np.sqrt(np.mean(np.square(residual)))),
        f"{prefix}_mae": float(np.mean(np.abs(residual))),
        f"{prefix}_bias": float(np.mean(residual)),
        f"{prefix}_pred_argmax": pred_argmax,
        f"{prefix}_true_argmax": true_argmax,
        f"{prefix}_argmax_abs_error": float(abs(pred_argmax - true_argmax)),
    }
