from __future__ import annotations

import numpy as np
from scipy.special import ndtri

from src.config import DataConfig
from src.data.simulator import Wu2024Simulator


COPULA_MODE = "gaussian_copula_correlated_disturbances_structural"
LEGACY_PROXY_MODE = "shared_latent_nonlinear_substitution"


def _make_simulator(version: str, structure_type: str = "peaked", **overrides) -> Wu2024Simulator:
    cfg = DataConfig(
        version=version,
        n_train=200,
        n_test=250,
        policy_grid_size=51,
        structure_type=structure_type,
        **overrides,
    )
    return Wu2024Simulator(cfg)


def test_simulator_shapes_and_keys():
    simulator = _make_simulator("wu2024_original")
    train, test = simulator.sample_train_test(seed=13)
    assert simulator.support[0] < simulator.support[1]
    assert train.treatment.shape == (200, 1)
    assert train.treatment_proxy.shape == (200, simulator.config.dim_z)
    assert train.outcome_proxy.shape == (200, simulator.config.dim_w)
    assert train.covariates.shape == (200, simulator.config.dim_x)
    assert train.outcome.shape == (200, 1)
    assert train.optimal_treatment.shape == (200, 1)
    assert train.assignment_mean.shape == (200, 1)
    assert train.latent_uz.shape == (200, 1)
    assert train.latent_uw.shape == (200, 1)
    assert test.treatment.shape == (250, 1)
    assert np.all(train.treatment >= simulator.support[0] - 1e-12)
    assert np.all(train.treatment <= simulator.support[1] + 1e-12)
    assert np.all(train.assignment_mean >= simulator.support[0] - 1e-12)
    assert np.all(train.assignment_mean <= simulator.support[1] + 1e-12)


def test_biased_assignment_moves_treatment_farther_from_optimal():
    original = _make_simulator("wu2024_original")
    biased = _make_simulator("wu2024_biased_assignment")
    train_original, _ = original.sample_train_test(seed=13)
    train_biased, _ = biased.sample_train_test(seed=13)

    original_gap = np.mean(np.abs(train_original.treatment - train_original.optimal_treatment))
    biased_gap = np.mean(np.abs(train_biased.treatment - train_biased.optimal_treatment))

    assert biased_gap > original_gap


def test_nonadditive_version_keeps_assignment_but_changes_latent_outcome_term():
    original = _make_simulator("wu2024_original")
    nonadditive = _make_simulator("wu2024_nonadditive")
    original_sample, _ = original.sample_train_test(seed=13)
    nonadditive_sample, _ = nonadditive.sample_train_test(seed=13)

    assert np.allclose(original_sample.treatment, nonadditive_sample.treatment)
    assert np.allclose(original_sample.assignment_mean, nonadditive_sample.assignment_mean)
    assert not np.allclose(original_sample.outcome, nonadditive_sample.outcome)

    term = nonadditive.latent_outcome_confounding_term(
        nonadditive_sample.treatment,
        nonadditive_sample.covariates,
        nonadditive_sample.latent_uz,
    )
    additive = 0.25 * nonadditive_sample.latent_uz.reshape(-1)
    assert term.shape == additive.shape
    assert not np.allclose(term, additive)


def test_assumption_violations_strength_zero_keeps_synthetic_sample_exactly_unchanged():
    baseline = _make_simulator("wu2024_nonadditive", structure_type="single_peaked_heterogeneous")
    explicit_zero = _make_simulator(
        "wu2024_nonadditive",
        structure_type="single_peaked_heterogeneous",
        proxy_independence_violation_strength=0.0,
        proxy_independence_violation_mode=COPULA_MODE,
        completeness_violation_strength=0.0,
    )
    baseline_sample, _ = baseline.sample_train_test(seed=13)
    zero_sample, _ = explicit_zero.sample_train_test(seed=13)

    for field in (
        "treatment",
        "treatment_proxy",
        "outcome_proxy",
        "outcome",
        "assignment_mean",
        "latent_uz",
        "latent_uw",
        "optimal_treatment",
    ):
        assert np.array_equal(getattr(baseline_sample, field), getattr(zero_sample, field))


def test_proxy_dependence_violation_changes_only_observed_proxies():
    baseline = _make_simulator("wu2024_nonadditive", structure_type="single_peaked_heterogeneous")
    violated = _make_simulator(
        "wu2024_nonadditive",
        structure_type="single_peaked_heterogeneous",
        proxy_independence_violation_strength=0.6,
        proxy_independence_violation_mode=LEGACY_PROXY_MODE,
    )
    baseline_sample, _ = baseline.sample_train_test(seed=21)
    violated_sample, _ = violated.sample_train_test(seed=21)

    assert np.allclose(baseline_sample.treatment, violated_sample.treatment)
    assert np.allclose(baseline_sample.outcome, violated_sample.outcome)
    assert np.allclose(baseline_sample.assignment_mean, violated_sample.assignment_mean)
    assert not np.allclose(baseline_sample.treatment_proxy, violated_sample.treatment_proxy)
    assert not np.allclose(baseline_sample.outcome_proxy, violated_sample.outcome_proxy)
    assert np.allclose(baseline_sample.latent_uz, violated_sample.latent_uz)
    assert np.allclose(baseline_sample.latent_uw, violated_sample.latent_uw)


def test_proxy_dependence_violation_substitutes_latent_signal_with_shared_spurious_signal():
    simulator = _make_simulator(
        "wu2024_nonadditive",
        structure_type="single_peaked_heterogeneous",
        proxy_independence_violation_strength=1.0,
        proxy_independence_violation_mode=LEGACY_PROXY_MODE,
    )
    uz = np.array([0.5, -1.0])
    uw = np.array([-0.25, 0.75])
    vz = np.array([[0.1, -0.2], [0.3, 0.4]])
    vw = np.array([[0.2, -0.1], [-0.3, 0.5]])
    rng = np.random.default_rng(123)
    structural_z, structural_w, observed_z, observed_w = simulator._generate_proxies(uz, uw, vz, vw, rng)

    expected_structural_z = vz + simulator._proxy_latent_signal(uz, simulator.config.dim_z)
    expected_structural_w = vw + simulator._proxy_latent_signal(uw, simulator.config.dim_w)

    replay_rng = np.random.default_rng(123)
    shared_latent = replay_rng.normal(0.0, 1.0, size=uz.shape[0])
    expected_observed_z = vz + simulator._shared_proxy_spurious_signal(
        shared_latent,
        simulator.config.dim_z,
        kind="z",
    )
    expected_observed_w = vw + simulator._shared_proxy_spurious_signal(
        shared_latent,
        simulator.config.dim_w,
        kind="w",
    )

    assert np.allclose(structural_z, expected_structural_z)
    assert np.allclose(structural_w, expected_structural_w)
    assert np.allclose(observed_z, expected_observed_z)
    assert np.allclose(observed_w, expected_observed_w)


def test_copula_mode_structural_and_observed_proxies_are_identical():
    simulator = _make_simulator(
        "wu2024_nonadditive",
        structure_type="single_peaked_heterogeneous",
        proxy_independence_violation_strength=0.7,
        proxy_independence_violation_mode=COPULA_MODE,
        dim_z=2,
        dim_w=5,
    )
    rng = np.random.default_rng(123)
    uz, uw, vz, vw, _x = simulator._sample_latents(1000, rng)
    structural_z, structural_w, observed_z, observed_w = simulator._generate_proxies(
        uz,
        uw,
        vz,
        vw,
        np.random.default_rng(456),
    )

    assert np.array_equal(structural_z, observed_z)
    assert np.array_equal(structural_w, observed_w)
    assert structural_z.shape == (1000, 2)
    assert structural_w.shape == (1000, 5)


def test_copula_mode_uses_transformed_w_structurally_with_paired_draws():
    baseline = _make_simulator(
        "wu2024_nonadditive",
        structure_type="single_peaked_heterogeneous",
        proxy_independence_violation_strength=0.0,
        proxy_independence_violation_mode=COPULA_MODE,
    )
    violated = _make_simulator(
        "wu2024_nonadditive",
        structure_type="single_peaked_heterogeneous",
        proxy_independence_violation_strength=0.7,
        proxy_independence_violation_mode=COPULA_MODE,
    )
    baseline_sample = baseline.sample_factual(1000, seed=21)
    violated_sample = violated.sample_factual(1000, seed=21)

    assert np.array_equal(baseline_sample.treatment_proxy, violated_sample.treatment_proxy)
    assert np.array_equal(baseline_sample.assignment_mean, violated_sample.assignment_mean)
    assert np.array_equal(baseline_sample.treatment, violated_sample.treatment)
    assert np.all(baseline_sample.outcome_proxy != violated_sample.outcome_proxy)
    assert np.all(baseline_sample.outcome != violated_sample.outcome)


def test_copula_uniform_noise_preserves_marginals_and_calibrates_latent_dependence():
    simulator = Wu2024Simulator(
        DataConfig(
            support_lower=-1.0,
            support_upper=2.0,
            dim_z=2,
            dim_w=5,
            proxy_independence_violation_mode=COPULA_MODE,
        )
    )
    rng = np.random.default_rng(987)
    vz = rng.uniform(-1.0, 1.0, size=(120_000, 2))
    vw = rng.uniform(-1.0, 1.0, size=(120_000, 5))
    aligned_vz = simulator._align_columns(vz, vw.shape[1])
    gaussian_z = ndtri(np.clip((aligned_vz + 1.0) / 2.0, 1e-7, 1.0 - 1e-7))
    target_std = np.sqrt(1.0 / 3.0)
    correlations = []

    zero_noise = simulator._gaussian_copula_correlated_uniform_noise(vz, vw, 0.0)
    assert zero_noise is vw

    for rho in (0.0, 0.3, 0.7, 1.0):
        transformed = simulator._gaussian_copula_correlated_uniform_noise(vz, vw, rho)
        gaussian_w = ndtri(np.clip((transformed + 1.0) / 2.0, 1e-7, 1.0 - 1e-7))
        correlations.append(
            float(
                np.mean(
                    [
                        np.corrcoef(gaussian_z[:, j], gaussian_w[:, j])[0, 1]
                        for j in range(vw.shape[1])
                    ]
                )
            )
        )
        assert np.all(np.isfinite(transformed))
        assert np.min(transformed) >= -1.0
        assert np.max(transformed) <= 1.0
        assert abs(float(np.mean(transformed))) < 0.01
        assert abs(float(np.std(transformed)) - target_std) < 0.01
        assert abs(float(np.std(transformed)) - float(np.std(vw))) < 0.01

    assert np.allclose(correlations, (0.0, 0.3, 0.7, 1.0), atol=0.015)
    assert np.all(np.diff(correlations) >= -0.005)


def test_copula_mode_preserves_proxy_relevance_across_rho():
    correlations = []
    for rho in (0.0, 0.3, 0.7, 1.0):
        simulator = Wu2024Simulator(
            DataConfig(
                version="wu2024_nonadditive",
                structure_type="single_peaked_heterogeneous",
                support_lower=-1.0,
                support_upper=2.0,
                proxy_independence_violation_strength=rho,
                proxy_independence_violation_mode=COPULA_MODE,
            )
        )
        sample = simulator.sample_factual(40_000, seed=2468)
        z_relevance = float(
            np.mean(
                [
                    np.corrcoef(sample.treatment_proxy[:, j], sample.latent_uz[:, 0])[0, 1]
                    for j in range(sample.treatment_proxy.shape[1])
                ]
            )
        )
        w_relevance = float(
            np.mean(
                [
                    np.corrcoef(sample.outcome_proxy[:, j], sample.latent_uw[:, 0])[0, 1]
                    for j in range(sample.outcome_proxy.shape[1])
                ]
            )
        )
        correlations.append((z_relevance, w_relevance))

    correlations_array = np.asarray(correlations)
    assert np.max(np.abs(correlations_array - correlations_array[0]), axis=0)[0] == 0.0
    assert np.max(np.abs(correlations_array - correlations_array[0]), axis=0)[1] < 0.02


def test_completeness_violation_changes_proxies_but_keeps_finite_outputs():
    baseline = _make_simulator("wu2024_nonadditive", structure_type="single_peaked_heterogeneous")
    violated = _make_simulator(
        "wu2024_nonadditive",
        structure_type="single_peaked_heterogeneous",
        completeness_violation_strength=1.0,
        completeness_relu_max_threshold_z=2.0,
        completeness_relu_max_threshold_w=2.0,
    )
    baseline_sample, _ = baseline.sample_train_test(seed=8)
    violated_sample, _ = violated.sample_train_test(seed=8)

    assert not np.allclose(baseline_sample.treatment_proxy, violated_sample.treatment_proxy)
    assert not np.allclose(baseline_sample.outcome_proxy, violated_sample.outcome_proxy)
    assert np.all(np.isfinite(violated_sample.treatment_proxy))
    assert np.all(np.isfinite(violated_sample.outcome_proxy))
    assert np.all(np.isfinite(violated_sample.outcome))
    assert np.all(violated_sample.treatment >= violated.support[0] - 1e-12)
    assert np.all(violated_sample.treatment <= violated.support[1] + 1e-12)


def test_completeness_violation_collapses_shared_central_band_without_rescaling():
    violated = _make_simulator(
        "wu2024_nonadditive",
        structure_type="single_peaked_heterogeneous",
        completeness_violation_strength=1.0,
        completeness_relu_max_threshold_z=2.0,
        completeness_relu_max_threshold_w=2.0,
    )
    latent = np.array([[-3.0], [-1.0], [0.0], [1.0], [3.0]])

    z_signal = violated._apply_completeness_violation_to_signal(
        latent,
        violated.config.dim_z,
        keep_negative_side=False,
        max_threshold=violated.config.completeness_relu_max_threshold_z,
    )
    w_signal = violated._apply_completeness_violation_to_signal(
        latent,
        violated.config.dim_w,
        keep_negative_side=True,
        max_threshold=violated.config.completeness_relu_max_threshold_w,
    )

    assert np.allclose(z_signal[1:4], 0.0)
    assert np.allclose(w_signal[1:4], 0.0)
    assert np.all(z_signal[4] > 0.0)
    assert np.all(w_signal[0] < 0.0)
    assert np.max(np.abs(z_signal)) < violated.config.proxy_confounding_scale * 3.0
    assert np.max(np.abs(w_signal)) < violated.config.proxy_confounding_scale * 3.0


def test_completeness_central_band_dropout_noise_collapses_shared_region():
    violated = _make_simulator(
        "wu2024_nonadditive",
        structure_type="single_peaked_heterogeneous",
        completeness_violation_strength=1.0,
        completeness_violation_mode="central_band_dropout_noise",
        completeness_relu_max_threshold_z=2.0,
        completeness_relu_max_threshold_w=2.0,
        completeness_noise_scale_max=0.0,
    )
    latent = np.array([[-3.0], [-1.0], [0.0], [1.0], [3.0]])

    z_signal = violated._apply_completeness_violation_to_signal(
        latent,
        violated.config.dim_z,
        keep_negative_side=False,
        max_threshold=violated.config.completeness_relu_max_threshold_z,
    )
    w_signal = violated._apply_completeness_violation_to_signal(
        latent,
        violated.config.dim_w,
        keep_negative_side=True,
        max_threshold=violated.config.completeness_relu_max_threshold_w,
    )

    assert np.allclose(z_signal[1:4], 0.0)
    assert np.allclose(w_signal[1:4], 0.0)
    assert np.all(z_signal[0] < 0.0)
    assert np.all(z_signal[4] > 0.0)
    assert np.all(w_signal[0] < 0.0)
    assert np.all(w_signal[4] > 0.0)


def test_completeness_central_band_dropout_noise_adds_post_coarsening_noise():
    simulator = _make_simulator(
        "wu2024_nonadditive",
        structure_type="single_peaked_heterogeneous",
        completeness_violation_strength=0.5,
        completeness_violation_mode="central_band_dropout_noise",
        completeness_noise_scale_max=0.2,
    )
    signal = np.zeros((4, simulator.config.dim_z), dtype=float)
    rng = np.random.default_rng(7)
    noisy = simulator._apply_completeness_noise(signal, rng=rng, strength=0.5)

    assert noisy.shape == signal.shape
    assert not np.allclose(noisy, signal)
    assert np.all(np.isfinite(noisy))


def test_optimal_treatment_lives_on_policy_grid():
    simulator = _make_simulator("wu2024_original")
    train, _ = simulator.sample_train_test(seed=21)
    values = np.unique(train.optimal_treatment.reshape(-1))
    assert np.all(np.isin(values, simulator.policy_grid))


def test_average_treatment_effect_curve_matches_grid():
    simulator = _make_simulator("wu2024_original")
    train, _ = simulator.sample_train_test(seed=21)
    grid, ate_curve, structure_curve = simulator.average_treatment_effect_curve(train.covariates)
    assert grid.shape == (simulator.config.policy_grid_size,)
    assert ate_curve.shape == grid.shape
    assert structure_curve.shape == grid.shape
    assert np.all(np.isfinite(ate_curve))


def test_causal_curve_given_x_mc_has_expected_shape_and_finite_values():
    cfg = DataConfig(
        version="wu2024_original",
        n_train=80,
        n_test=80,
        policy_grid_size=13,
        causal_mc_samples=192,
        causal_mc_batch_size=64,
    )
    simulator = Wu2024Simulator(cfg)
    sample, _ = simulator.sample_train_test(seed=13)
    x_row = sample.covariates[0]
    grid, curve = simulator.causal_curve_given_x_mc(x_row)
    assert grid.shape == (cfg.policy_grid_size,)
    assert curve.shape == grid.shape
    assert np.all(np.isfinite(curve))


def test_causal_curve_given_x_mc_is_reproducible_for_fixed_seed():
    cfg = DataConfig(
        version="wu2024_original",
        n_train=60,
        n_test=60,
        policy_grid_size=11,
        causal_mc_samples=160,
        causal_mc_batch_size=40,
    )
    simulator = Wu2024Simulator(cfg)
    sample, _ = simulator.sample_train_test(seed=19)
    x_row = sample.covariates[3]
    grid_a, curve_a = simulator.causal_curve_given_x_mc(x_row, seed=123)
    grid_b, curve_b = simulator.causal_curve_given_x_mc(x_row, seed=123)
    assert np.allclose(grid_a, grid_b)
    assert np.allclose(curve_a, curve_b)


def test_causal_curve_given_x_mc_matches_mean_outcome_in_additive_case():
    cfg = DataConfig(
        version="wu2024_original",
        n_train=60,
        n_test=60,
        policy_grid_size=9,
        outcome_noise_scale=0.0,
        causal_mc_samples=2000,
        causal_mc_batch_size=250,
    )
    simulator = Wu2024Simulator(cfg)
    sample, _ = simulator.sample_train_test(seed=31)
    x_row = sample.covariates[5]
    grid, curve_mc = simulator.causal_curve_given_x_mc(x_row, seed=2718)
    x_rep = np.repeat(x_row[None, :], grid.shape[0], axis=0)
    curve_structural = simulator.mean_outcome(grid, x_rep).reshape(-1)
    assert np.allclose(curve_mc, curve_structural, atol=0.12)


def test_associational_curve_given_x_binned_has_expected_shapes_and_masks():
    cfg = DataConfig(
        version="wu2024_original",
        n_train=80,
        n_test=80,
        policy_grid_size=17,
        associational_mc_samples=400,
        associational_min_bin_count=10,
    )
    simulator = Wu2024Simulator(cfg)
    sample, _ = simulator.sample_train_test(seed=11)
    x_row = sample.covariates[0]
    (
        grid,
        curve,
        counts,
        low_count,
        empty,
        fallback_t,
        fallback_y,
    ) = simulator.associational_curve_given_x_binned(x_row, seed=123)

    assert grid.shape == (cfg.policy_grid_size,)
    assert curve.shape == grid.shape
    assert counts.shape == grid.shape
    assert low_count.shape == grid.shape
    assert empty.shape == grid.shape
    assert fallback_t.shape == grid.shape
    assert fallback_y.shape == grid.shape
    assert np.all(np.isfinite(curve))
    assert np.all(counts >= 0)
    assert np.all(low_count <= (counts > 0))


def test_associational_curve_given_x_binned_empty_bins_use_nearest_fallback():
    cfg = DataConfig(
        version="wu2024_original",
        n_train=50,
        n_test=50,
        policy_grid_size=51,
        associational_mc_samples=120,
        associational_min_bin_count=10,
    )
    simulator = Wu2024Simulator(cfg)
    sample, _ = simulator.sample_train_test(seed=12)
    x_row = sample.covariates[3]
    (
        _grid,
        _curve,
        counts,
        _low_count,
        empty,
        fallback_t,
        fallback_y,
    ) = simulator.associational_curve_given_x_binned(x_row, seed=77)

    if np.any(empty):
        assert np.all(np.isfinite(fallback_t[empty]))
        assert np.all(np.isfinite(fallback_y[empty]))
        assert np.all(counts[empty] == 0)


def test_associational_curve_selector_defaults_to_kernel_and_supports_binned():
    cfg = DataConfig(
        version="wu2024_original",
        n_train=60,
        n_test=60,
        policy_grid_size=21,
        associational_kernel=True,
        associational_mc_samples=300,
    )
    simulator = Wu2024Simulator(cfg)
    sample, _ = simulator.sample_train_test(seed=14)
    x_row = sample.covariates[2]

    grid_k, curve_k, *_rest_k, method_k = simulator.associational_curve_given_x(x_row, seed=17)
    grid_b, curve_b, *_rest_b, method_b = simulator.associational_curve_given_x(x_row, method="binned", seed=17)

    assert method_k == "kernel"
    assert method_b == "binned"
    assert grid_k.shape == (cfg.policy_grid_size,)
    assert curve_k.shape == grid_k.shape
    assert np.all(np.isfinite(curve_k))
    assert np.allclose(grid_k, grid_b)
    assert curve_b.shape == grid_b.shape


def test_peaked_centered_reduces_upper_boundary_optima():
    peaked_cfg = DataConfig(
        version="wu2024_original",
        n_train=1500,
        n_test=1000,
        policy_grid_size=81,
        structure_type="peaked",
    )
    centered_cfg = DataConfig(
        version="wu2024_original",
        n_train=1500,
        n_test=1000,
        policy_grid_size=81,
        structure_type="peaked_centered",
    )
    peaked = Wu2024Simulator(peaked_cfg)
    centered = Wu2024Simulator(centered_cfg)
    peaked_sample, _ = peaked.sample_train_test(seed=13)
    centered_sample, _ = centered.sample_train_test(seed=13)
    peaked_upper = np.mean(np.isclose(peaked_sample.optimal_treatment.reshape(-1), peaked.support[1]))
    centered_upper = np.mean(np.isclose(centered_sample.optimal_treatment.reshape(-1), centered.support[1]))
    assert centered_upper < peaked_upper


def test_saturated_interaction_reduces_x0_treatment_separation():
    original = _make_simulator("wu2024_original")
    saturated = _make_simulator("wu2024_saturated_interaction")
    original_sample, _ = original.sample_train_test(seed=13)
    saturated_sample, _ = saturated.sample_train_test(seed=13)

    def treatment_gap(sample) -> float:
        x0 = sample.covariates[:, 0]
        treatment = sample.treatment.reshape(-1)
        q_low, q_high = np.quantile(x0, [0.25, 0.75])
        low_mean = float(np.mean(treatment[x0 <= q_low]))
        high_mean = float(np.mean(treatment[x0 >= q_high]))
        return abs(high_mean - low_mean)

    assert treatment_gap(saturated_sample) < treatment_gap(original_sample)


def test_saturated_interaction_reduces_boundary_optima():
    original = _make_simulator("wu2024_original", structure_type="peaked")
    saturated = _make_simulator("wu2024_saturated_interaction", structure_type="peaked")
    original_sample, _ = original.sample_train_test(seed=13)
    saturated_sample, _ = saturated.sample_train_test(seed=13)

    def boundary_fraction(simulator: Wu2024Simulator, sample) -> float:
        optimal = sample.optimal_treatment.reshape(-1)
        return float(
            np.mean(
                np.isclose(optimal, simulator.support[0]) | np.isclose(optimal, simulator.support[1])
            )
        )

    assert boundary_fraction(saturated, saturated_sample) < boundary_fraction(original, original_sample)
