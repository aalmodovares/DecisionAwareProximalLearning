from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config import ExperimentConfig
from src.data.factory import make_simulator as make_data_simulator
from src.data.simulator import FactualSample, Wu2024Simulator


def make_simulator(
    version: str,
    seed: int = 13,
    n_train: int = 1200,
    n_val: int = 0,
    n_test: int = 3000,
    structure_type: str = "peaked",
    nonadditive_gamma: float | None = None,
    nonadditive_eta: float | None = None,
    nonadditive_scale_ratio: float | None = None,
    centered_penalty_strength: float | None = None,
    additive_term_scale: float | None = None,
    outcome_noise_scale: float | None = None,
) -> tuple[Wu2024Simulator, FactualSample, FactualSample]:
    cfg = ExperimentConfig()
    cfg.seed = seed
    cfg.data.version = version
    cfg.data.n_train = n_train
    cfg.data.n_val = n_val
    cfg.data.n_test = n_test
    cfg.data.structure_type = structure_type
    if nonadditive_gamma is not None:
        cfg.data.nonadditive_confounding_gamma = nonadditive_gamma
    if nonadditive_eta is not None:
        cfg.data.nonadditive_confounding_eta = nonadditive_eta
    if nonadditive_scale_ratio is not None:
        cfg.data.nonadditive_confounding_scale_ratio = nonadditive_scale_ratio
    if centered_penalty_strength is not None:
        cfg.data.centered_penalty_strength = centered_penalty_strength
    if additive_term_scale is not None:
        cfg.data.additive_term_scale = additive_term_scale
    if outcome_noise_scale is not None:
        cfg.data.outcome_noise_scale = outcome_noise_scale
    simulator = make_data_simulator(cfg.data)
    train_sample, test_sample = simulator.sample_train_test(seed)
    return simulator, train_sample, test_sample


def figure_stem(version: str, structure_type: str, figure_kind: str) -> str:
    raw = f"{version}__{structure_type}__{figure_kind}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


def label_and_save_figure(
    fig: plt.Figure,
    output_dir: str | Path,
    version: str,
    structure_type: str,
    figure_kind: str,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    context = f"{version} | {structure_type}"
    title = figure_kind.replace("_", " ").title()
    if len(fig.axes) == 1:
        current = fig.axes[0].get_title().strip()
        fig.axes[0].set_title(f"{context} | {current or title}")
    else:
        fig.suptitle(f"{context} | {title}", y=1.02)
    fig.tight_layout()
    stem = figure_stem(version, structure_type, figure_kind)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    return png_path, pdf_path


def _sample_index(n: int, max_points: int, seed: int = 13) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if n <= max_points:
        return np.arange(n)
    return rng.choice(n, size=max_points, replace=False)


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


def plot_dataset_diagnostics(simulator: Wu2024Simulator, sample: FactualSample, max_points: int = 600) -> plt.Figure:
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
            np.corrcoef(sample.latent_uw.reshape(-1), sample.treatment.reshape(-1))[0, 1],
            np.corrcoef(latent_outcome_term.reshape(-1), sample.outcome.reshape(-1))[0, 1],
            np.corrcoef(sample.assignment_mean.reshape(-1), sample.treatment.reshape(-1))[0, 1],
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
    fig.tight_layout()
    return fig


def plot_optimal_vs_observed(sample: FactualSample, max_points: int = 600) -> plt.Figure:
    idx = _sample_index(sample.treatment.shape[0], max_points)
    treatment = sample.treatment[idx].reshape(-1)
    optimal = sample.optimal_treatment[idx].reshape(-1)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(optimal, treatment, s=12, alpha=0.6)
    lo = min(np.min(optimal), np.min(treatment))
    hi = max(np.max(optimal), np.max(treatment))
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    ax.set_title("Observed treatment vs optimal treatment")
    ax.set_xlabel("Optimal treatment")
    ax.set_ylabel("Observed treatment")
    fig.tight_layout()
    return fig


def plot_optimal_vs_x0(sample: FactualSample, max_points: int = 800) -> plt.Figure:
    idx = _sample_index(sample.treatment.shape[0], max_points, seed=19)
    x0 = sample.covariates[idx, 0]
    optimal = sample.optimal_treatment[idx].reshape(-1)
    order = np.argsort(x0)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x0, optimal, s=12, alpha=0.55)
    ax.plot(x0[order], optimal[order], linewidth=1.0, alpha=0.8, color="tab:orange")
    ax.set_title("Optimal treatment vs X0")
    ax.set_xlabel("X0")
    ax.set_ylabel("Optimal treatment")
    fig.tight_layout()
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
    fig.tight_layout()
    return fig


def plot_conditional_distributions(sample: FactualSample) -> plt.Figure:
    x0 = sample.covariates[:, 0]
    treatment = sample.treatment.reshape(-1)
    q_low, q_high = np.quantile(x0, [0.25, 0.75])
    low_mask = x0 <= q_low
    high_mask = x0 >= q_high

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(treatment[low_mask], bins=30, alpha=0.6, density=True, label="X0 low")
    axes[0].hist(treatment[high_mask], bins=30, alpha=0.6, density=True, label="X0 high")
    axes[0].set_title("Conditional treatment distributions")
    axes[0].set_xlabel("Treatment")
    axes[0].legend()

    axes[1].hist(sample.outcome.reshape(-1)[low_mask], bins=30, alpha=0.6, density=True, label="X0 low")
    axes[1].hist(sample.outcome.reshape(-1)[high_mask], bins=30, alpha=0.6, density=True, label="X0 high")
    axes[1].set_title("Conditional outcome distributions")
    axes[1].set_xlabel("Outcome")
    axes[1].legend()
    fig.tight_layout()
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
    fig.tight_layout()
    return fig






