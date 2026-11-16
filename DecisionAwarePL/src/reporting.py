from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.simulator import FactualSample, Wu2024Simulator
from src.utils import ensure_dir


def _sample_index(n: int, max_points: int, seed: int = 13) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if n <= max_points:
        return np.arange(n)
    return rng.choice(n, size=max_points, replace=False)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _safe_corrcoef(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if x_arr.size == 0 or y_arr.size == 0 or x_arr.size != y_arr.size:
        return float("nan")
    if float(np.std(x_arr)) <= eps or float(np.std(y_arr)) <= eps:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _mark_curve_optimum_star(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: str) -> None:
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if x_arr.size == 0 or y_arr.size == 0 or x_arr.size != y_arr.size:
        return
    idx = int(np.argmax(y_arr))
    ax.scatter(
        [float(x_arr[idx])],
        [float(y_arr[idx])],
        marker="*",
        s=60,
        color=color,
        alpha=0.3,
        label="_nolegend_",
        zorder=7,
    )


def _latent_outcome_confounding_term(simulator: Wu2024Simulator, sample: FactualSample) -> np.ndarray:
    try:
        return simulator.latent_outcome_confounding_term(
            sample.treatment,
            sample.covariates,
            sample.latent_uz,
            latent_uw=sample.latent_uw,
        )
    except TypeError:
        return simulator.latent_outcome_confounding_term(
            sample.treatment,
            sample.covariates,
            sample.latent_uz,
        )


def plot_dataset_diagnostics(
    simulator: Wu2024Simulator,
    sample: FactualSample,
    max_points: int = 600,
) -> plt.Figure:
    idx = _sample_index(sample.treatment.shape[0], max_points)
    t = sample.treatment[idx].reshape(-1)
    opt = sample.optimal_treatment[idx].reshape(-1)
    assign = sample.assignment_mean[idx].reshape(-1)
    uw = sample.latent_uw[idx].reshape(-1)
    y = sample.outcome[idx].reshape(-1)
    latent_outcome_term = _latent_outcome_confounding_term(simulator, sample)
    latent_outcome_term_subset = latent_outcome_term[idx].reshape(-1)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes[0, 0].scatter(opt, t, s=10, alpha=0.6)
    lo = min(np.min(opt), np.min(t))
    hi = max(np.max(opt), np.max(t))
    axes[0, 0].plot([lo, hi], [lo, hi], "k--", linewidth=1)
    axes[0, 0].set_title("Observed vs optimal treatment")
    axes[0, 0].set_xlabel("Optimal treatment")
    axes[0, 0].set_ylabel("Observed treatment")

    axes[0, 1].scatter(opt, assign, s=10, alpha=0.6, color="tab:orange")
    axes[0, 1].plot([lo, hi], [lo, hi], "k--", linewidth=1)
    axes[0, 1].set_title("Assignment mean vs optimal")
    axes[0, 1].set_xlabel("Optimal treatment")
    axes[0, 1].set_ylabel("Assignment mean")

    axes[0, 2].hist(t - opt, bins=30, alpha=0.75, color="tab:green")
    axes[0, 2].axvline(0.0, color="k", linestyle="--", linewidth=1)
    axes[0, 2].set_title("Treatment minus optimal")
    axes[0, 2].set_xlabel("Observed - optimal")

    axes[1, 0].scatter(uw, t, s=10, alpha=0.6, color="tab:red")
    axes[1, 0].set_title("Treatment confounder effect")
    axes[1, 0].set_xlabel("Latent Uw")
    axes[1, 0].set_ylabel("Observed treatment")

    axes[1, 1].scatter(latent_outcome_term_subset, y, s=10, alpha=0.6, color="tab:purple")
    axes[1, 1].set_title("Outcome confounder effect")
    axes[1, 1].set_xlabel("Latent outcome term")
    axes[1, 1].set_ylabel("Outcome")

    strengths = np.array(
        [
            _safe_corrcoef(sample.latent_uw.reshape(-1), sample.treatment.reshape(-1)),
            _safe_corrcoef(latent_outcome_term.reshape(-1), sample.outcome.reshape(-1)),
            _safe_corrcoef(sample.assignment_mean.reshape(-1), sample.treatment.reshape(-1)),
        ]
    )
    axes[1, 2].bar(
        ["corr(Uw,A)", "corr(Cy,Y)", "corr(mu_A,A)"],
        strengths,
        color=["tab:red", "tab:purple", "tab:orange"],
    )
    axes[1, 2].set_ylim(-1, 1)
    axes[1, 2].set_title("Confounding strength diagnostics")
    plt.setp(axes[1, 2].get_xticklabels(), rotation=20)
    fig.suptitle(
        f"Dataset diagnostics (support=[{simulator.support[0]:.3f}, {simulator.support[1]:.3f}])",
        y=1.02,
    )
    return fig


def plot_population_ate_curve(simulator: Wu2024Simulator, sample: FactualSample) -> plt.Figure:
    grid, ate_curve, structure_curve = simulator.average_treatment_effect_curve(sample.covariates)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(grid, ate_curve, linewidth=2.2, label="Population dose-response E[Y(a)]")
    ax.plot(grid, structure_curve, linewidth=1.8, linestyle="--", label="Structure term")
    ax.axvline(grid[np.argmax(ate_curve)], color="k", linestyle=":", linewidth=1.2, label="ATE optimum")
    ax.set_title("ATE depending on treatment")
    ax.set_xlabel("Treatment")
    ax.set_ylabel("Average outcome")
    ax.legend()
    ax.grid(alpha=0.2)
    return fig


def plot_conditional_distributions(sample: FactualSample) -> plt.Figure:
    x0 = sample.covariates[:, 0]
    treatment = sample.treatment.reshape(-1)
    outcome = sample.outcome.reshape(-1)
    q_low, q_high = np.quantile(x0, [0.25, 0.75])
    low_mask = x0 <= q_low
    high_mask = x0 >= q_high

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(treatment[low_mask], bins=30, alpha=0.6, density=True, label="X0 low")
    axes[0].hist(treatment[high_mask], bins=30, alpha=0.6, density=True, label="X0 high")
    axes[0].set_title("Conditional treatment distributions")
    axes[0].set_xlabel("Treatment")
    axes[0].legend()

    axes[1].hist(outcome[low_mask], bins=30, alpha=0.6, density=True, label="X0 low")
    axes[1].hist(outcome[high_mask], bins=30, alpha=0.6, density=True, label="X0 high")
    axes[1].set_title("Conditional outcome distributions")
    axes[1].set_xlabel("Outcome")
    axes[1].legend()
    return fig


def plot_pairwise_scatters(sample: FactualSample, max_points: int = 500) -> plt.Figure:
    idx = _sample_index(sample.treatment.shape[0], max_points, seed=17)
    x0 = sample.covariates[idx, 0]
    z0 = sample.treatment_proxy[idx, 0]
    w0 = sample.outcome_proxy[idx, 0]
    treatment = sample.treatment[idx].reshape(-1)
    outcome = sample.outcome[idx].reshape(-1)
    optimal = sample.optimal_treatment[idx].reshape(-1)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    pairs = [
        (x0, treatment, "X0", "A"),
        (z0, treatment, "Z0", "A"),
        (w0, outcome, "W0", "Y"),
        (x0, optimal, "X0", "a*(X)"),
        (x0, outcome, "X0", "Y"),
        (treatment, outcome, "A", "Y"),
    ]
    for ax, (xv, yv, xl, yl) in zip(axes.ravel(), pairs):
        ax.scatter(xv, yv, s=10, alpha=0.6)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
    fig.suptitle("Pairwise scatter diagnostics", y=1.02)
    return fig


def _method_to_history_key(method_name: str) -> str:
    key = str(method_name).strip().lower()
    key = key.replace("-", "_").replace(" ", "_")
    while "__" in key:
        key = key.replace("__", "_")
    return key


def _history_train_loss_at_best(history: dict[str, object]) -> float:
    train_losses = np.asarray(history.get("train_losses", []), dtype=float).reshape(-1)
    if train_losses.size == 0:
        return float("nan")
    best_epoch = int(history.get("best_epoch", train_losses.size - 1))
    if best_epoch < 0 or best_epoch >= train_losses.size:
        best_epoch = train_losses.size - 1
    value = float(train_losses[best_epoch])
    return value if np.isfinite(value) else float("nan")


def _mean_outcome_train_loss(
    method_name: str,
    training_histories: dict[str, list[dict[str, object]]] | None,
) -> float:
    if not training_histories:
        return float("nan")
    history_key = f"{_method_to_history_key(method_name)}_losses"
    stage_histories = training_histories.get(history_key, [])
    if not stage_histories:
        return float("nan")
    mean_stages = [h for h in stage_histories if "mean_regressor" in str(h.get("stage", "")).lower()]
    if mean_stages:
        return _history_train_loss_at_best(mean_stages[-1])
    # Fallback for direct-only S-learner/OOSR runs where no mean head is trained.
    outcome_stages = [h for h in stage_histories if "outcome_regressor" in str(h.get("stage", "")).lower()]
    if outcome_stages:
        return _history_train_loss_at_best(outcome_stages[-1])
    return float("nan")


def plot_method_metrics(
    metrics: list[dict[str, float]],
    training_histories: dict[str, list[dict[str, object]]] | None = None,
) -> plt.Figure:
    methods = [row["method"] for row in metrics]
    regret_causal = [float(row.get("regret_causal_marginal", row.get("regret", np.nan))) for row in metrics]
    factual_rmse = [
        float(row.get("factual_mean_rmse", row.get("factual_rmse", row.get("observed_rmse", np.nan))))
        for row in metrics
    ]
    mean_outcome_train_loss = [_mean_outcome_train_loss(method, training_histories) for method in methods]
    policy_mse = [float(row.get("policy_mse_causal_marginal", row.get("policy_mse", np.nan))) for row in metrics]
    local_rmse = [float(row.get("local_rmse", np.nan)) for row in metrics]
    global_rmse = [float(row.get("global_rmse", np.nan)) for row in metrics]

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    series = [
        (regret_causal, "Regret (causal marginal)"),
        (factual_rmse, "Factual RMSE"),
        (mean_outcome_train_loss, "Mean/Outcome train loss"),
        (policy_mse, "Policy MSE (causal marginal)"),
        (local_rmse, "Local RMSE"),
        (global_rmse, "Global RMSE"),
    ]
    for ax, (values, title) in zip(axes.ravel(), series):
        ax.bar(methods, values, color="tab:blue", alpha=0.8)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
    return fig


def plot_policy_scatter(policies: dict[str, np.ndarray], optimal: np.ndarray) -> plt.Figure:
    methods = list(policies.keys())
    ncols = 2
    nrows = int(np.ceil(len(methods) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for ax, method in zip(axes.ravel(), methods):
        pred = policies[method].reshape(-1)
        ax.scatter(optimal, pred, s=10, alpha=0.5)
        lo = min(optimal.min(), pred.min())
        hi = max(optimal.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set_title(method.replace("_", " "))
        ax.set_xlabel("True optimal policy")
        ax.set_ylabel("Estimated policy")

    for ax in axes.ravel()[len(methods):]:
        ax.axis("off")
    return fig


def plot_policy_vs_x0(
    policies: dict[str, np.ndarray],
    optimal: np.ndarray,
    x0: np.ndarray,
) -> plt.Figure:
    order = np.argsort(x0)
    methods = list(policies.keys())
    ncols = 2
    nrows = int(np.ceil(len(methods) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for ax, method in zip(axes.ravel(), methods):
        pred = policies[method].reshape(-1)
        ax.plot(x0[order], optimal[order], label="true", linewidth=2)
        ax.plot(x0[order], pred[order], label="pred", linewidth=1.5)
        ax.set_title(f"{method.replace('_', ' ')} vs X0")
        ax.set_xlabel("X0 (sorted)")
        ax.set_ylabel("Policy")

    for ax in axes.ravel()[len(methods):]:
        ax.axis("off")
    if methods:
        axes.ravel()[0].legend()
    return fig


def plot_policy_error_by_region(
    simulator: Wu2024Simulator,
    policies: dict[str, np.ndarray],
    optimal: np.ndarray,
) -> plt.Figure:
    bins = np.linspace(simulator.support[0], simulator.support[1], 7)
    centers = 0.5 * (bins[:-1] + bins[1:])
    region_ids = np.digitize(optimal, bins[1:-1], right=False)
    counts = np.array([(region_ids == idx).sum() for idx in range(len(centers))], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for method, pred in policies.items():
        errors = []
        pred = pred.reshape(-1)
        for idx in range(len(centers)):
            mask = region_ids == idx
            if not np.any(mask):
                errors.append(np.nan)
            else:
                errors.append(float(np.mean(np.abs(pred[mask] - optimal[mask]))))
        axes[0].plot(centers, errors, marker="o", label=method.replace("_", " "))
    axes[0].set_ylabel("Mean absolute policy error")
    axes[0].set_title("Policy error by true optimal-treatment region")
    axes[0].legend()

    axes[1].bar(centers, counts, width=(bins[1] - bins[0]) * 0.9, alpha=0.7, color="tab:gray")
    axes[1].set_xlabel("True optimal-treatment region")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Region sample counts")
    return fig


def plot_mean_potential_outcome_curves(
    simulator: Wu2024Simulator,
    sample: FactualSample,
    mean_outcome_curves: dict[str, np.ndarray],
    treatment_grid: np.ndarray | None = None,
    *,
    title: str = "Population mean potential-outcome curves",
    ground_truth_curve: np.ndarray | None = None,
    causal_marginal_curve: np.ndarray | None = None,
    proxy_conditioned_curve: np.ndarray | None = None,
) -> plt.Figure:
    grid, default_true_curve, structure_curve = simulator.average_treatment_effect_curve(sample.covariates, grid=treatment_grid)
    true_curve = np.asarray(default_true_curve if ground_truth_curve is None else ground_truth_curve, dtype=float).reshape(-1)
    causal_curve = np.asarray(causal_marginal_curve, dtype=float).reshape(-1) if causal_marginal_curve is not None else None
    proxy_curve = np.asarray(proxy_conditioned_curve, dtype=float).reshape(-1) if proxy_conditioned_curve is not None else None
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(grid, true_curve, color="k", linewidth=2.6, label="Ground truth oracle-conditioned")
    if proxy_curve is not None:
        ax.plot(
            grid,
            proxy_curve,
            color="tab:gray",
            linewidth=2.0,
            linestyle="--",
            label="Proxy-conditioned E[Y|A,X,W] (reference)",
        )
    if causal_curve is not None:
        ax.plot(
            grid,
            causal_curve,
            color="tab:blue",
            linewidth=1.8,
            linestyle=":",
            label="Promedio estructural mean_outcome (no oracle ground truth)",
        )
    ax.plot(grid, structure_curve, color="tab:green", linewidth=1.4, linestyle="-.", label="Structure term")
    for method, curve in mean_outcome_curves.items():
        curve_arr = np.asarray(curve, dtype=float).reshape(-1)
        (line,) = ax.plot(grid, curve_arr, linewidth=1.7, label=method.replace("_", " "))
        _mark_curve_optimum_star(ax, grid, curve_arr, color=str(line.get_color()))
    ax.axvline(grid[np.argmax(true_curve)], color="k", linestyle=":", linewidth=1.2, label="Oracle-curve optimum")
    ax.set_title(title)
    ax.set_xlabel("Treatment")
    ax.set_ylabel("Mean potential outcome")
    ax.set_xlim(simulator.support)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.2)
    return fig


def plot_weight_region_effect(
    method_name: str,
    diagnostics: dict[str, np.ndarray],
) -> plt.Figure:
    treatment = diagnostics["treatment"].reshape(-1)
    reference_policy = diagnostics["reference_policy"].reshape(-1)
    optimal = diagnostics["optimal_treatment"].reshape(-1)
    fit_weights = diagnostics["fit_weights"].reshape(-1)
    base_weights = diagnostics["base_weights"].reshape(-1)
    ratio = fit_weights / np.maximum(base_weights, 1e-8)
    distance = np.abs(treatment - reference_policy)
    top_cut = np.quantile(fit_weights, 0.9)
    top_mask = fit_weights >= top_cut

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    scatter = axes[0, 0].scatter(reference_policy, treatment, c=fit_weights, s=14, alpha=0.7, cmap="viridis")
    lo = min(reference_policy.min(), treatment.min())
    hi = max(reference_policy.max(), treatment.max())
    axes[0, 0].plot([lo, hi], [lo, hi], "k--", linewidth=1)
    axes[0, 0].set_title(f"{method_name}: treatment vs weighted region")
    axes[0, 0].set_xlabel("Pseudo-optimal treatment")
    axes[0, 0].set_ylabel("Observed treatment")
    fig.colorbar(scatter, ax=axes[0, 0], label="Final sample weight")

    axes[0, 1].scatter(distance, fit_weights, s=12, alpha=0.6, color="tab:orange")
    axes[0, 1].set_title("Weight vs distance to pseudo-optimal region")
    axes[0, 1].set_xlabel("|A - rho_hat(X)|")
    axes[0, 1].set_ylabel("Final sample weight")

    bins = np.linspace(distance.min(), distance.max() + 1e-12, 12)
    bin_ids = np.digitize(distance, bins[1:-1], right=False)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    mean_ratio = []
    for idx in range(len(bin_centers)):
        mask = bin_ids == idx
        if not np.any(mask):
            mean_ratio.append(np.nan)
        else:
            mean_ratio.append(float(np.mean(ratio[mask])))
    axes[1, 0].plot(bin_centers, mean_ratio, marker="o", color="tab:green")
    axes[1, 0].axhline(1.0, color="k", linestyle="--", linewidth=1)
    axes[1, 0].set_title("Region emphasis relative to base density weights")
    axes[1, 0].set_xlabel("|A - rho_hat(X)|")
    axes[1, 0].set_ylabel("mean(final/base)")

    axes[1, 1].hist(distance, bins=25, alpha=0.55, density=True, label="all samples")
    axes[1, 1].hist(distance[top_mask], bins=25, alpha=0.55, density=True, label="top 10% weights")
    axes[1, 1].axvline(np.mean(np.abs(treatment - optimal)), color="tab:red", linestyle="--", linewidth=1.2, label="mean |A-a*|")
    axes[1, 1].set_title("Distance distribution after weighting")
    axes[1, 1].set_xlabel("|A - rho_hat(X)|")
    axes[1, 1].legend()
    return fig


def plot_mean_pseudooutcome_diagnostics(bridge_models: dict[str, object]) -> tuple[plt.Figure, plt.Figure] | None:
    rows = []
    for model_name, model in bridge_models.items():
        diag = getattr(model, "mean_pseudooutcome_diagnostics_", None)
        if not diag:
            continue
        std = float(diag.get("std", np.nan))
        iqr = float(diag.get("iqr", np.nan))
        treatment_grid = np.asarray(diag.get("treatment_grid", []), dtype=float).reshape(-1)
        std_by_treatment = np.asarray(diag.get("std_by_treatment", []), dtype=float).reshape(-1)
        iqr_by_treatment = np.asarray(diag.get("iqr_by_treatment", []), dtype=float).reshape(-1)
        rows.append((model_name, std, iqr, treatment_grid, std_by_treatment, iqr_by_treatment))
    if not rows:
        return None

    labels = [name.replace("_", " ") for name, *_ in rows]
    std_values = [std for _, std, _, *_ in rows]
    iqr_values = [iqr for _, _, iqr, *_ in rows]

    fig_summary, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(rows), dtype=float)
    width = 0.38
    ax.bar(x - 0.5 * width, std_values, width=width, label="std", color="tab:blue", alpha=0.8)
    ax.bar(x + 0.5 * width, iqr_values, width=width, label="IQR", color="tab:orange", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("Pseudo-outcome scale")
    ax.set_title("Pseudo-outcome diagnostics by bridge model")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    fig_curve, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for model_name, _, _, treatment_grid, std_by_treatment, iqr_by_treatment in rows:
        if treatment_grid.size == 0:
            continue
        label = model_name.replace("_", " ")
        axes[0].plot(treatment_grid, std_by_treatment, linewidth=1.8, label=label)
        axes[1].plot(treatment_grid, iqr_by_treatment, linewidth=1.8, label=label)
    axes[0].set_ylabel("std")
    axes[0].set_title("Pseudo-outcome std across treatment grid")
    axes[0].grid(alpha=0.2)
    axes[1].set_ylabel("IQR")
    axes[1].set_xlabel("Treatment")
    axes[1].set_title("Pseudo-outcome IQR across treatment grid")
    axes[1].grid(alpha=0.2)
    axes[0].legend(fontsize=8, ncol=2)
    return fig_summary, fig_curve


def _patient_indices(sample: FactualSample, n_patients: int) -> np.ndarray:
    optimal = sample.optimal_treatment.reshape(-1)
    x0 = sample.covariates[:, 0]
    order = np.lexsort((x0, optimal))
    if len(order) <= n_patients:
        return order
    positions = np.linspace(0, len(order) - 1, n_patients)
    return order[np.round(positions).astype(int)]


def _patient_curve(
    simulator: Wu2024Simulator,
    x_row: np.ndarray,
    treatment_grid: np.ndarray,
) -> np.ndarray:
    _, curve = simulator.causal_curve_given_x_mc(x_row, grid=treatment_grid)
    return curve.reshape(-1)


def _patient_associational_curve(
    simulator: Wu2024Simulator,
    x_row: np.ndarray,
    treatment_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    return simulator.associational_curve_given_x(x_row, grid=treatment_grid)


def build_patient_curve_table(
    simulator: Wu2024Simulator,
    sample: FactualSample,
    mean_predictors: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]],
    direct_predictors: dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]] | None,
    oracle_direct_predictors: dict[
        str,
        Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    ]
    | None,
    treatment_grid: np.ndarray,
    *,
    patient_indices: np.ndarray | None = None,
    n_patients: int = 50,
    dataset: str | None = None,
    seed: int | None = None,
    split: str = "test",
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Build the patient-specific dose-response table used by surface plots.

    The returned curve rows mirror ``plot_patient_surface_gallery``: causal,
    associational, proxy-conditioned, oracle-conditioned, and every available
    model curve are evaluated on the same patient/treatment grid.
    """
    direct_predictors = direct_predictors or {}
    oracle_direct_predictors = oracle_direct_predictors or {}
    grid = np.asarray(treatment_grid, dtype=float).reshape(-1)
    if patient_indices is None:
        patient_indices = _patient_indices(sample, n_patients=n_patients)
    patient_indices = np.asarray(patient_indices, dtype=int).reshape(-1)

    curve_rows: list[dict[str, object]] = []
    metadata_rows: list[dict[str, object]] = []

    def add_curve(
        *,
        patient_rank: int,
        patient_idx: int,
        method: str,
        predictor_type: str,
        curve_type: str,
        curve_grid: np.ndarray,
        values: np.ndarray,
        associational_method: str | None = None,
        associational_count: np.ndarray | None = None,
        associational_low_count: np.ndarray | None = None,
        associational_empty: np.ndarray | None = None,
        associational_fallback_treatment: np.ndarray | None = None,
        associational_fallback_outcome: np.ndarray | None = None,
    ) -> None:
        value_arr = np.asarray(values, dtype=float).reshape(-1)
        grid_arr = np.asarray(curve_grid, dtype=float).reshape(-1)
        if value_arr.shape[0] != grid_arr.shape[0]:
            raise ValueError(
                f"Curve length mismatch for patient {patient_idx}, method {method!r}: "
                f"{value_arr.shape[0]} values for {grid_arr.shape[0]} treatments."
            )
        count_arr = (
            np.asarray(associational_count).reshape(-1)
            if associational_count is not None
            else np.full(grid_arr.shape[0], np.nan)
        )
        low_arr = (
            np.asarray(associational_low_count).reshape(-1)
            if associational_low_count is not None
            else np.full(grid_arr.shape[0], False)
        )
        empty_arr = (
            np.asarray(associational_empty).reshape(-1)
            if associational_empty is not None
            else np.full(grid_arr.shape[0], False)
        )
        fallback_t_arr = (
            np.asarray(associational_fallback_treatment, dtype=float).reshape(-1)
            if associational_fallback_treatment is not None
            else np.full(grid_arr.shape[0], np.nan)
        )
        fallback_y_arr = (
            np.asarray(associational_fallback_outcome, dtype=float).reshape(-1)
            if associational_fallback_outcome is not None
            else np.full(grid_arr.shape[0], np.nan)
        )
        for point_idx, (a_val, y_val) in enumerate(zip(grid_arr, value_arr)):
            curve_rows.append(
                {
                    "dataset": dataset or "",
                    "seed": "" if seed is None else int(seed),
                    "split": split,
                    "patient_rank": int(patient_rank),
                    "patient_id": int(patient_idx),
                    "method": method,
                    "predictor_type": predictor_type,
                    "curve_type": curve_type,
                    "treatment": float(a_val),
                    "value": float(y_val),
                    "associational_method": associational_method or "",
                    "associational_count": (
                        float(count_arr[point_idx]) if np.isfinite(count_arr[point_idx]) else ""
                    ),
                    "associational_low_count": bool(low_arr[point_idx]),
                    "associational_empty": bool(empty_arr[point_idx]),
                    "associational_fallback_treatment": (
                        float(fallback_t_arr[point_idx]) if np.isfinite(fallback_t_arr[point_idx]) else ""
                    ),
                    "associational_fallback_outcome": (
                        float(fallback_y_arr[point_idx]) if np.isfinite(fallback_y_arr[point_idx]) else ""
                    ),
                }
            )

    for patient_rank, patient_idx_raw in enumerate(patient_indices):
        patient_idx = int(patient_idx_raw)
        x_row = sample.covariates[patient_idx]
        z_row = sample.treatment_proxy[patient_idx]
        w_row = sample.outcome_proxy[patient_idx]
        uz_row = sample.latent_uz[patient_idx]
        uw_row = sample.latent_uw[patient_idx]
        x_rep = np.repeat(x_row[None, :], grid.shape[0], axis=0)
        z_rep = np.repeat(z_row[None, :], grid.shape[0], axis=0)
        w_rep = np.repeat(w_row[None, :], grid.shape[0], axis=0)
        uz_rep = np.repeat(uz_row[None, :], grid.shape[0], axis=0)
        uw_rep = np.repeat(uw_row[None, :], grid.shape[0], axis=0)
        a_grid = grid[:, None]

        causal_curve = _patient_curve(simulator, x_row, grid)
        (
            assoc_grid,
            assoc_curve,
            assoc_counts,
            assoc_low_count,
            assoc_empty,
            assoc_fallback_t,
            assoc_fallback_y,
            assoc_method,
        ) = _patient_associational_curve(simulator, x_row, grid)
        proxy_curve = simulator.proxy_conditioned_outcome(grid, x_rep, w_rep).reshape(-1)
        oracle_curve = simulator.oracle_conditioned_outcome(grid, x_rep, w_rep, uz_rep, uw_rep).reshape(-1)

        metadata: dict[str, object] = {
            "dataset": dataset or "",
            "seed": "" if seed is None else int(seed),
            "split": split,
            "patient_rank": int(patient_rank),
            "patient_id": int(patient_idx),
            "observed_treatment": float(sample.treatment[patient_idx, 0]),
            "observed_outcome": float(sample.outcome[patient_idx, 0]),
            "assignment_mean": float(sample.assignment_mean[patient_idx, 0]),
            "structural_optimal_treatment": float(sample.optimal_treatment[patient_idx, 0]),
            "surface_optimal_treatment": float(grid[int(np.argmax(causal_curve))]),
        }
        for col_idx, value in enumerate(np.asarray(x_row, dtype=float).reshape(-1)):
            metadata[f"x_{col_idx}"] = float(value)
        for col_idx, value in enumerate(np.asarray(z_row, dtype=float).reshape(-1)):
            metadata[f"z_{col_idx}"] = float(value)
        for col_idx, value in enumerate(np.asarray(w_row, dtype=float).reshape(-1)):
            metadata[f"w_{col_idx}"] = float(value)
        for col_idx, value in enumerate(np.asarray(uz_row, dtype=float).reshape(-1)):
            metadata[f"latent_uz_{col_idx}"] = float(value)
        for col_idx, value in enumerate(np.asarray(uw_row, dtype=float).reshape(-1)):
            metadata[f"latent_uw_{col_idx}"] = float(value)
        metadata_rows.append(metadata)

        add_curve(
            patient_rank=patient_rank,
            patient_idx=patient_idx,
            method="reference",
            predictor_type="reference",
            curve_type="ground_truth_causal",
            curve_grid=grid,
            values=causal_curve,
        )
        add_curve(
            patient_rank=patient_rank,
            patient_idx=patient_idx,
            method="reference",
            predictor_type="reference",
            curve_type="associational",
            curve_grid=assoc_grid,
            values=assoc_curve,
            associational_method=assoc_method,
            associational_count=assoc_counts,
            associational_low_count=assoc_low_count,
            associational_empty=assoc_empty,
            associational_fallback_treatment=assoc_fallback_t,
            associational_fallback_outcome=assoc_fallback_y,
        )
        add_curve(
            patient_rank=patient_rank,
            patient_idx=patient_idx,
            method="reference",
            predictor_type="reference",
            curve_type="proxy_conditioned",
            curve_grid=grid,
            values=proxy_curve,
        )
        add_curve(
            patient_rank=patient_rank,
            patient_idx=patient_idx,
            method="reference",
            predictor_type="reference",
            curve_type="oracle_conditioned",
            curve_grid=grid,
            values=oracle_curve,
        )

        for method, mean_predict in mean_predictors.items():
            add_curve(
                patient_rank=patient_rank,
                patient_idx=patient_idx,
                method=method,
                predictor_type="mean",
                curve_type="prediction",
                curve_grid=grid,
                values=mean_predict(a_grid, x_rep).reshape(-1),
            )
        for method, direct_predict in direct_predictors.items():
            add_curve(
                patient_rank=patient_rank,
                patient_idx=patient_idx,
                method=method,
                predictor_type="direct",
                curve_type="prediction",
                curve_grid=grid,
                values=direct_predict(a_grid, z_rep, w_rep, x_rep).reshape(-1),
            )
        for method, direct_predict in oracle_direct_predictors.items():
            add_curve(
                patient_rank=patient_rank,
                patient_idx=patient_idx,
                method=method,
                predictor_type="oracle_direct",
                curve_type="prediction",
                curve_grid=grid,
                values=direct_predict(a_grid, z_rep, w_rep, x_rep, uz_rep, uw_rep).reshape(-1),
            )

    return curve_rows, metadata_rows


def plot_patient_surface(
    patient_idx: int,
    simulator: Wu2024Simulator,
    sample: FactualSample,
    mean_predictors: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]],
    direct_predictors: dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]],
    oracle_direct_predictors: dict[
        str,
        Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    ],
    treatment_grid: np.ndarray,
) -> plt.Figure:
    x_row = sample.covariates[patient_idx]
    z_row = sample.treatment_proxy[patient_idx]
    w_row = sample.outcome_proxy[patient_idx]
    uz_row = sample.latent_uz[patient_idx]
    uw_row = sample.latent_uw[patient_idx]
    x_rep = np.repeat(x_row[None, :], treatment_grid.shape[0], axis=0)
    z_rep = np.repeat(z_row[None, :], treatment_grid.shape[0], axis=0)
    w_rep = np.repeat(w_row[None, :], treatment_grid.shape[0], axis=0)
    uz_rep = np.repeat(uz_row[None, :], treatment_grid.shape[0], axis=0)
    uw_rep = np.repeat(uw_row[None, :], treatment_grid.shape[0], axis=0)
    a_grid = treatment_grid[:, None]
    causal_curve = _patient_curve(simulator, x_row, treatment_grid)
    (
        assoc_grid,
        assoc_curve,
        assoc_counts,
        assoc_low_count,
        assoc_empty,
        assoc_fallback_t,
        assoc_fallback_y,
        assoc_method,
    ) = _patient_associational_curve(simulator, x_row, treatment_grid)
    proxy_curve = simulator.proxy_conditioned_outcome(treatment_grid, x_rep, w_rep).reshape(-1)
    oracle_curve = simulator.oracle_conditioned_outcome(treatment_grid, x_rep, w_rep, uz_rep, uw_rep).reshape(-1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(treatment_grid, causal_curve, color="k", linewidth=2.5, label="Ground truth causal E[Y(a)|X=x_i]")
    ax.plot(
        assoc_grid,
        assoc_curve,
        color="0.35",
        linewidth=1.8,
        linestyle="--",
        alpha=0.75,
        label=f"E[Y|X,A] (reference - {assoc_method})",
    )
    if np.any(assoc_low_count):
        ax.scatter(
            assoc_grid[assoc_low_count],
            assoc_curve[assoc_low_count],
            marker="x",
            s=34,
            color="0.20",
            label=(
                f"Low-ESS points (kernel, n_eff<{simulator.config.associational_min_bin_count})"
                if assoc_method == "kernel"
                else f"Low-count bins (n<{simulator.config.associational_min_bin_count})"
            ),
            zorder=6,
        )
    if assoc_method == "binned" and np.any(assoc_empty):
        ax.scatter(
            assoc_fallback_t[assoc_empty],
            assoc_fallback_y[assoc_empty],
            marker="^",
            s=42,
            facecolors="none",
            edgecolors="tab:red",
            linewidths=1.2,
            label="Empty-bin fallback (nearest observed point)",
            zorder=7,
        )
    ax.plot(
        treatment_grid,
        proxy_curve,
        color="tab:gray",
        linestyle="--",
        linewidth=2.0,
        label="Proxy-conditioned E[Y|A,X,W] (reference)",
    )
    ax.plot(
        treatment_grid,
        oracle_curve,
        color="tab:purple",
        linestyle="-.",
        linewidth=1.8,
        label="Oracle-conditioned E[Y|do(a),X,W_i,U_i] (reference)",
    )

    for method, mean_predict in mean_predictors.items():
        pred_curve = mean_predict(a_grid, x_rep).reshape(-1)
        ax.plot(treatment_grid, pred_curve, linewidth=1.6, label=method.replace("_", " "))
    for method, direct_predict in direct_predictors.items():
        pred_curve = direct_predict(a_grid, z_rep, w_rep, x_rep).reshape(-1)
        ax.plot(treatment_grid, pred_curve, linewidth=1.6, label=method.replace("_", " "))
    for method, direct_predict in oracle_direct_predictors.items():
        pred_curve = direct_predict(a_grid, z_rep, w_rep, x_rep, uz_rep, uw_rep).reshape(-1)
        ax.plot(treatment_grid, pred_curve, linewidth=1.6, label=method.replace("_", " "))

    true_opt = float(treatment_grid[int(np.argmax(causal_curve))])
    factual_t = float(sample.treatment[patient_idx, 0])
    factual_y = float(sample.outcome[patient_idx, 0])
    assign_mean = float(sample.assignment_mean[patient_idx, 0])

    ax.axvline(true_opt, color="k", linestyle="--", linewidth=1.2, label="Oracle optimal")
    ax.axvline(factual_t, color="tab:gray", linestyle=":", linewidth=1.2, label="Observed treatment")
    ax.axvline(assign_mean, color="tab:brown", linestyle="-.", linewidth=1.2, label="Assignment mean")
    ax.scatter([factual_t], [factual_y], color="tab:red", s=36, zorder=5, label="Observed outcome")

    ax.set_title(
        f"Patient {patient_idx}: x0={x_row[0]:.3f}, a*={true_opt:.3f}, A={factual_t:.3f}, mu_A={assign_mean:.3f}"
    )
    ax.set_xlabel("Treatment")
    ax.set_ylabel("Outcome")
    ax.set_xlim(simulator.support)
    ax.legend(fontsize=8, ncol=2)
    return fig


def plot_patient_surface_gallery(
    simulator: Wu2024Simulator,
    sample: FactualSample,
    mean_predictors: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]],
    direct_predictors: dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]],
    oracle_direct_predictors: dict[
        str,
        Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    ],
    patient_indices: np.ndarray,
    treatment_grid: np.ndarray,
) -> plt.Figure:
    ncols = 2
    nrows = int(np.ceil(len(patient_indices) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.2 * nrows))
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for panel_id, (ax, patient_idx) in enumerate(zip(axes.ravel(), patient_indices)):
        x_row = sample.covariates[patient_idx]
        z_row = sample.treatment_proxy[patient_idx]
        w_row = sample.outcome_proxy[patient_idx]
        uz_row = sample.latent_uz[patient_idx]
        uw_row = sample.latent_uw[patient_idx]
        x_rep = np.repeat(x_row[None, :], treatment_grid.shape[0], axis=0)
        z_rep = np.repeat(z_row[None, :], treatment_grid.shape[0], axis=0)
        w_rep = np.repeat(w_row[None, :], treatment_grid.shape[0], axis=0)
        uz_rep = np.repeat(uz_row[None, :], treatment_grid.shape[0], axis=0)
        uw_rep = np.repeat(uw_row[None, :], treatment_grid.shape[0], axis=0)
        a_grid = treatment_grid[:, None]
        causal_curve = _patient_curve(simulator, x_row, treatment_grid)
        (
            assoc_grid,
            assoc_curve,
            _assoc_counts,
            assoc_low_count,
            assoc_empty,
            assoc_fallback_t,
            assoc_fallback_y,
            assoc_method,
        ) = _patient_associational_curve(simulator, x_row, treatment_grid)
        proxy_curve = simulator.proxy_conditioned_outcome(treatment_grid, x_rep, w_rep).reshape(-1)
        oracle_curve = simulator.oracle_conditioned_outcome(treatment_grid, x_rep, w_rep, uz_rep, uw_rep).reshape(-1)
        ax.plot(treatment_grid, causal_curve, color="k", linewidth=2.0, label="Ground truth causal E[Y(a)|X=x_i]" if panel_id == 0 else None)
        ax.plot(
            assoc_grid,
            assoc_curve,
            color="brown",
            linestyle=":",
            linewidth=1.4,
            alpha=0.35,
            label=(f"E[Y|X,A] (reference - {assoc_method})" if panel_id == 0 else None),
        )
        if np.any(assoc_low_count):
            ax.scatter(
                assoc_grid[assoc_low_count],
                assoc_curve[assoc_low_count],
                marker="x",
                s=20,
                color="0.20",
                zorder=6,
                label=(
                    (
                        f"Low-ESS points (kernel, n_eff<{simulator.config.associational_min_bin_count})"
                        if assoc_method == "kernel"
                        else f"Low-count bins (n<{simulator.config.associational_min_bin_count})"
                    )
                    if panel_id == 0
                    else None
                ),
            )
        if assoc_method == "binned" and np.any(assoc_empty):
            ax.scatter(
                assoc_fallback_t[assoc_empty],
                assoc_fallback_y[assoc_empty],
                marker="^",
                s=24,
                facecolors="none",
                edgecolors="tab:red",
                linewidths=1.0,
                zorder=7,
                label="Empty-bin fallback (nearest observed point)" if panel_id == 0 else None,
            )
        ax.plot(
            treatment_grid,
            proxy_curve,
            color="tab:gray",
            linestyle="--",
            linewidth=1.6,
            label="Proxy-conditioned E[Y|A,X,W] (reference)" if panel_id == 0 else None,
        )
        ax.plot(
            treatment_grid,
            oracle_curve,
            color="tab:purple",
            linestyle="-.",
            linewidth=1.5,
            label="Oracle-conditioned E[Y|do(a),X,W_i,U_i] (reference)" if panel_id == 0 else None,
        )
        for method, mean_predict in mean_predictors.items():
            pred_curve = mean_predict(a_grid, x_rep).reshape(-1)
            label = method.replace("_", " ") if panel_id == 0 else None
            (line,) = ax.plot(treatment_grid, pred_curve, linewidth=1.1, alpha=0.95, label=label)
            _mark_curve_optimum_star(ax, treatment_grid, pred_curve, color=str(line.get_color()))
        for method, direct_predict in direct_predictors.items():
            pred_curve = direct_predict(a_grid, z_rep, w_rep, x_rep).reshape(-1)
            label = method.replace("_", " ") if panel_id == 0 else None
            (line,) = ax.plot(treatment_grid, pred_curve, linewidth=1.1, alpha=0.95, label=label)
            _mark_curve_optimum_star(ax, treatment_grid, pred_curve, color=str(line.get_color()))
        for method, direct_predict in oracle_direct_predictors.items():
            pred_curve = direct_predict(a_grid, z_rep, w_rep, x_rep, uz_rep, uw_rep).reshape(-1)
            label = method.replace("_", " ") if panel_id == 0 else None
            (line,) = ax.plot(treatment_grid, pred_curve, linewidth=1.1, alpha=0.95, label=label)
            _mark_curve_optimum_star(ax, treatment_grid, pred_curve, color=str(line.get_color()))
        ax.axvline(
            float(treatment_grid[int(np.argmax(causal_curve))]),
            color="k",
            linestyle="--",
            linewidth=0.9,
            label="Oracle optimal" if panel_id == 0 else None,
        )
        ax.axvline(
            float(sample.treatment[patient_idx, 0]),
            color="tab:gray",
            linestyle=":",
            linewidth=0.9,
            label="Observed treatment" if panel_id == 0 else None,
        )
        ax.set_xlim(simulator.support)
        ax.set_title(f"Patient {patient_idx}")
        ax.set_xlabel("Treatment")
        ax.set_ylabel("Outcome")

    for ax in axes.ravel()[len(patient_indices):]:
        ax.axis("off")

    handles, labels = [], []
    for ax in axes.ravel():
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if label and label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=9)
    fig.suptitle("Patient-specific dose-response surfaces", y=0.998)
    return fig


def save_run_figures(
    output_dir: str | Path,
    simulator: Wu2024Simulator,
    train_sample: FactualSample,
    test_sample: FactualSample,
    metrics: list[dict[str, float]],
    policies: dict[str, np.ndarray],
    mean_outcome_curves: dict[str, np.ndarray] | None,
    mean_outcome_curves_direct: dict[str, np.ndarray] | None,
    ground_truth_curve: np.ndarray | None,
    causal_marginal_curve: np.ndarray | None,
    proxy_conditioned_curve: np.ndarray | None,
    mean_predictors: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]],
    direct_predictors: dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]] | None,
    oracle_direct_predictors: dict[
        str,
        Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    ] | None,
    weight_diagnostics: dict[str, dict[str, np.ndarray]],
    bridge_models: dict[str, object] | None = None,
    evaluation_treatment_grid: np.ndarray | None = None,
    training_histories: dict[str, list[dict[str, object]]] | None = None,
) -> list[str]:
    output_dir = Path(output_dir)
    figures_dir = ensure_dir(output_dir / "figures")
    patient_dir = ensure_dir(figures_dir / "patient_surfaces")
    created: list[str] = []
    direct_predictors = direct_predictors or {}
    oracle_direct_predictors = oracle_direct_predictors or {}
    mean_outcome_curves = mean_outcome_curves or {}
    mean_outcome_curves_direct = mean_outcome_curves_direct or {}
    bridge_models = bridge_models or {}

    figure_builders = [
        ("train_dataset_diagnostics.png", plot_dataset_diagnostics(simulator, train_sample)),
        ("train_population_ate_curve.png", plot_population_ate_curve(simulator, train_sample)),
        ("train_conditional_distributions.png", plot_conditional_distributions(train_sample)),
        ("train_pairwise_scatters.png", plot_pairwise_scatters(train_sample)),
        ("method_metrics.png", plot_method_metrics(metrics, training_histories=training_histories)),
        (
            "policy_scatter.png",
            plot_policy_scatter(policies, test_sample.optimal_treatment.reshape(-1)),
        ),
        (
            "policy_vs_x0.png",
            plot_policy_vs_x0(policies, test_sample.optimal_treatment.reshape(-1), test_sample.covariates[:, 0]),
        ),
        (
            "policy_error_by_region.png",
            plot_policy_error_by_region(simulator, policies, test_sample.optimal_treatment.reshape(-1)),
        ),
    ]
    if mean_outcome_curves:
        figure_builders.append(
            (
                "mean_potential_outcome_curves.png",
                plot_mean_potential_outcome_curves(
                    simulator,
                    test_sample,
                    mean_outcome_curves,
                    simulator.policy_grid,
                    ground_truth_curve=ground_truth_curve,
                    causal_marginal_curve=causal_marginal_curve,
                    proxy_conditioned_curve=proxy_conditioned_curve,
                ),
            )
        )
    if mean_outcome_curves_direct:
        direct_fig = plot_mean_potential_outcome_curves(
            simulator,
            test_sample,
            mean_outcome_curves_direct,
            simulator.policy_grid,
            title="Population curves from empirical bridge averaging",
            ground_truth_curve=ground_truth_curve,
            causal_marginal_curve=causal_marginal_curve,
            proxy_conditioned_curve=proxy_conditioned_curve,
        )
        figure_builders.append(("mean_curve_direct.png", direct_fig))
        figure_builders.append(
            (
                "mean_potential_outcome_curves_direct.png",
                plot_mean_potential_outcome_curves(
                    simulator,
                    test_sample,
                    mean_outcome_curves_direct,
                    simulator.policy_grid,
                    title="Population curves from empirical bridge averaging",
                    ground_truth_curve=ground_truth_curve,
                    causal_marginal_curve=causal_marginal_curve,
                    proxy_conditioned_curve=proxy_conditioned_curve,
                ),
            )
        )

    for name, fig in figure_builders:
        _save_figure(fig, figures_dir / name)
        created.append(str(figures_dir / name))

    for name, diag in weight_diagnostics.items():
        if not diag:
            continue
        fig = plot_weight_region_effect(name.replace("_", " "), diag)
        path = figures_dir / f"{name}_region_effect.png"
        _save_figure(fig, path)
        created.append(str(path))

    pseudo_diag_figures = plot_mean_pseudooutcome_diagnostics(bridge_models)
    if pseudo_diag_figures is not None:
        summary_path = figures_dir / "mean_pseudooutcome_diagnostics.png"
        curve_path = figures_dir / "mean_pseudooutcome_diagnostics_by_treatment.png"
        _save_figure(pseudo_diag_figures[0], summary_path)
        _save_figure(pseudo_diag_figures[1], curve_path)
        created.extend([str(summary_path), str(curve_path)])

    patient_indices = _patient_indices(test_sample, n_patients=10)
    plot_grid = (
        np.asarray(evaluation_treatment_grid, dtype=float).reshape(-1)
        if evaluation_treatment_grid is not None
        else np.asarray(simulator.policy_grid, dtype=float).reshape(-1)
    )
    for rank, patient_idx in enumerate(patient_indices):
        fig = plot_patient_surface(
            patient_idx=int(patient_idx),
            simulator=simulator,
            sample=test_sample,
            mean_predictors=mean_predictors,
            direct_predictors=direct_predictors,
            oracle_direct_predictors=oracle_direct_predictors,
            treatment_grid=plot_grid,
        )
        path = patient_dir / f"patient_{rank:02d}_idx_{int(patient_idx):04d}.png"
        _save_figure(fig, path)
        created.append(str(path))

    gallery = plot_patient_surface_gallery(
        simulator=simulator,
        sample=test_sample,
        mean_predictors=mean_predictors,
        direct_predictors=direct_predictors,
        oracle_direct_predictors=oracle_direct_predictors,
        patient_indices=patient_indices,
        treatment_grid=plot_grid,
    )
    gallery_path = patient_dir / "patient_surface_gallery.png"
    _save_figure(gallery, gallery_path)
    created.append(str(gallery_path))

    return created


def save_loss_figures(
    output_dir: str | Path,
    training_histories: dict[str, list[dict[str, object]]],
) -> list[str]:
    output_dir = Path(output_dir)
    loss_dir = ensure_dir(output_dir / "figures" / "losses")
    created: list[str] = []

    for model_name, stage_histories in training_histories.items():
        if not stage_histories:
            continue
        nrows = len(stage_histories)
        fig, axes = plt.subplots(nrows, 1, figsize=(10, 3.5 * nrows), squeeze=False)
        for ax, history in zip(axes.ravel(), stage_histories):
            train_losses = np.asarray(history.get("train_losses", []), dtype=float)
            val_losses = np.asarray(history.get("val_losses", []), dtype=float)
            epochs = np.arange(1, len(train_losses) + 1)
            if len(train_losses) > 0:
                ax.plot(epochs, train_losses, label="train", linewidth=1.8)
            if len(val_losses) > 0:
                ax.plot(epochs, val_losses, label="validation", linewidth=1.5)
            ax.set_title(str(history.get("stage", model_name)))
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
            ax.grid(alpha=0.2)
        path = loss_dir / f"{model_name}.png"
        _save_figure(fig, path)
        created.append(str(path))

    return created


def _bridge_diagnostic_row(
    model_name: str,
    split_name: str,
    model,
    simulator: Wu2024Simulator,
    sample: FactualSample,
) -> dict[str, float | str]:
    pred = model.bridge_predict(sample.treatment, sample.outcome_proxy, sample.covariates).reshape(-1)
    y = sample.outcome.reshape(-1)
    residual = y - pred
    row: dict[str, float | str] = {
        "model": model_name,
        "split": split_name,
        "factual_bridge_rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "factual_bridge_mae": float(np.mean(np.abs(residual))),
        "bridge_pred_std": float(np.std(pred)),
        "outcome_std": float(np.std(y)),
        "residual_std": float(np.std(residual)),
        "corr_residual_treatment": _safe_corrcoef(residual, sample.treatment.reshape(-1)),
        "corr_residual_x0": _safe_corrcoef(residual, sample.covariates[:, 0]),
        "corr_residual_z0": _safe_corrcoef(residual, sample.treatment_proxy[:, 0]),
        "corr_residual_w0": _safe_corrcoef(residual, sample.outcome_proxy[:, 0]),
    }
    truth = simulator.true_bridge(sample.treatment, sample.outcome_proxy, sample.covariates)
    if truth is not None:
        truth = truth.reshape(-1)
        row["oracle_bridge_rmse"] = float(np.sqrt(np.mean(np.square(pred - truth))))
        row["oracle_bridge_mae"] = float(np.mean(np.abs(pred - truth)))
    if hasattr(model, "alpha") and getattr(model, "alpha") is not None:
        row["alpha_norm"] = float(np.linalg.norm(model.alpha))
    if hasattr(model, "awx_bandwidth") and getattr(model, "awx_bandwidth") is not None:
        bw = np.asarray(model.awx_bandwidth, dtype=float).reshape(-1)
        row["awx_bandwidth_min"] = float(np.min(bw))
        row["awx_bandwidth_mean"] = float(np.mean(bw))
        row["awx_bandwidth_max"] = float(np.max(bw))
    return row


def plot_bridge_residuals(
    model_name: str,
    model,
    simulator: Wu2024Simulator,
    sample: FactualSample,
    max_points: int = 600,
) -> plt.Figure:
    idx = _sample_index(sample.treatment.shape[0], max_points, seed=23)
    pred = model.bridge_predict(sample.treatment[idx], sample.outcome_proxy[idx], sample.covariates[idx]).reshape(-1)
    y = sample.outcome[idx].reshape(-1)
    residual = y - pred
    truth = simulator.true_bridge(sample.treatment[idx], sample.outcome_proxy[idx], sample.covariates[idx])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].scatter(y, pred, s=12, alpha=0.55)
    lo = min(np.min(y), np.min(pred))
    hi = max(np.max(y), np.max(pred))
    axes[0, 0].plot([lo, hi], [lo, hi], "k--", linewidth=1)
    axes[0, 0].set_title(f"{model_name}: factual Y vs h(A,W,X)")
    axes[0, 0].set_xlabel("Observed Y")
    axes[0, 0].set_ylabel("Bridge prediction")

    axes[0, 1].scatter(sample.treatment[idx].reshape(-1), residual, s=12, alpha=0.55, color="tab:orange")
    axes[0, 1].axhline(0.0, color="k", linestyle="--", linewidth=1)
    axes[0, 1].set_title("Residual vs treatment")
    axes[0, 1].set_xlabel("A")
    axes[0, 1].set_ylabel("Y - h(A,W,X)")

    axes[1, 0].scatter(sample.treatment_proxy[idx, 0], residual, s=12, alpha=0.55, color="tab:green")
    axes[1, 0].axhline(0.0, color="k", linestyle="--", linewidth=1)
    axes[1, 0].set_title("Residual vs Z0")
    axes[1, 0].set_xlabel("Z0")
    axes[1, 0].set_ylabel("Y - h(A,W,X)")

    if truth is not None:
        truth = truth.reshape(-1)
        axes[1, 1].scatter(truth, pred, s=12, alpha=0.55, color="tab:purple")
        lo = min(np.min(truth), np.min(pred))
        hi = max(np.max(truth), np.max(pred))
        axes[1, 1].plot([lo, hi], [lo, hi], "k--", linewidth=1)
        axes[1, 1].set_title("Oracle bridge vs estimated bridge")
        axes[1, 1].set_xlabel("True bridge")
        axes[1, 1].set_ylabel("Estimated bridge")
    else:
        axes[1, 1].scatter(sample.outcome_proxy[idx, 0], residual, s=12, alpha=0.55, color="tab:purple")
        axes[1, 1].axhline(0.0, color="k", linestyle="--", linewidth=1)
        axes[1, 1].set_title("Residual vs W0")
        axes[1, 1].set_xlabel("W0")
        axes[1, 1].set_ylabel("Y - h(A,W,X)")
    return fig


def plot_bridge_patient_curves(
    model_name: str,
    model,
    simulator: Wu2024Simulator,
    sample: FactualSample,
    treatment_grid: np.ndarray,
    n_patients: int = 6,
) -> plt.Figure:
    patient_indices = _patient_indices(sample, n_patients=n_patients)
    ncols = 2
    nrows = int(np.ceil(len(patient_indices) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.0 * nrows))
    axes = np.atleast_1d(axes).reshape(nrows, ncols)
    for ax, patient_idx in zip(axes.ravel(), patient_indices):
        x_row = sample.covariates[patient_idx]
        w_row = sample.outcome_proxy[patient_idx]
        x_rep = np.repeat(x_row[None, :], treatment_grid.shape[0], axis=0)
        w_rep = np.repeat(w_row[None, :], treatment_grid.shape[0], axis=0)
        a_grid = treatment_grid[:, None]
        bridge_curve = model.bridge_predict(a_grid, w_rep, x_rep).reshape(-1)
        ax.plot(treatment_grid, bridge_curve, label="Estimated bridge", linewidth=2)
        truth = simulator.true_bridge(a_grid, w_rep, x_rep)
        if truth is not None:
            ax.plot(treatment_grid, truth.reshape(-1), label="True bridge", color="k", linestyle="--", linewidth=2)
        ax.axvline(float(sample.optimal_treatment[patient_idx, 0]), color="tab:red", linestyle=":", linewidth=1.2, label="a*(X)")
        ax.axvline(float(sample.treatment[patient_idx, 0]), color="tab:gray", linestyle="-.", linewidth=1.2, label="A observed")
        ax.set_xlim(simulator.support)
        ax.set_title(f"{model_name}: patient {int(patient_idx)}")
        ax.set_xlabel("Treatment")
        ax.set_ylabel("Bridge h(a,W_i,X_i)")
    for ax in axes.ravel()[len(patient_indices):]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=9)
    fig.suptitle(f"{model_name}: bridge curves at fixed observed W", y=0.998)
    return fig


def save_bridge_diagnostics(
    output_dir: str | Path,
    simulator: Wu2024Simulator,
    samples: dict[str, FactualSample],
    bridge_models: dict[str, object],
    treatment_grid: np.ndarray | None = None,
) -> tuple[list[dict[str, float | str]], list[str]]:
    output_dir = Path(output_dir)
    diagnostics_dir = ensure_dir(output_dir / "figures" / "bridge_diagnostics")
    rows: list[dict[str, float | str]] = []
    created: list[str] = []
    plot_grid = (
        np.asarray(treatment_grid, dtype=float).reshape(-1)
        if treatment_grid is not None
        else np.asarray(simulator.policy_grid, dtype=float).reshape(-1)
    )

    for model_name, model in bridge_models.items():
        for split_name, sample in samples.items():
            rows.append(_bridge_diagnostic_row(model_name, split_name, model, simulator, sample))

        fig = plot_bridge_residuals(model_name, model, simulator, samples["test"])
        path = diagnostics_dir / f"{model_name}_residuals.png"
        _save_figure(fig, path)
        created.append(str(path))

        fig = plot_bridge_patient_curves(model_name, model, simulator, samples["test"], plot_grid)
        path = diagnostics_dir / f"{model_name}_patient_bridge_curves.png"
        _save_figure(fig, path)
        created.append(str(path))

    if rows:
        with (output_dir / "bridge_diagnostics.csv").open("w", newline="") as handle:
            fieldnames = sorted({key for row in rows for key in row.keys()})
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return rows, created
