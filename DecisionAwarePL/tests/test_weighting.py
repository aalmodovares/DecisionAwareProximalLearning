from __future__ import annotations


import numpy as np

from src.config import ExperimentConfig, weighting_config_for_model
from src.weighting.oosr import (
    GAUSSIAN_TRUNCATION_MASS,
    build_base_density_weights,
    build_local_only_weights,
    finalize_weights,
    gaussian_oosr_kernel,
    local_oosr_factor,
    truncated_gaussian_oosr_kernel,
)


def test_base_density_weights_are_finite():
    rng = np.random.default_rng(13)
    treatment = rng.normal(size=(100, 1))
    conditional_density = rng.uniform(0.05, 0.5, size=100)
    support = (0.0, 1.0)

    unstabilized = build_base_density_weights(treatment, conditional_density, support, stabilized=False)
    stabilized = build_base_density_weights(treatment, conditional_density, support, stabilized=True)

    assert np.all(np.isfinite(unstabilized))
    assert np.all(np.isfinite(stabilized))
    assert np.all(unstabilized > 0)
    assert np.all(stabilized > 0)


def test_local_oosr_factor_all_modes_are_nonnegative_and_finite():
    rng = np.random.default_rng(7)
    treatment = rng.uniform(0.0, 1.0, size=(120, 1))
    policy = rng.uniform(0.0, 1.0, size=(120, 1))
    support = (0.0, 1.0)

    for mode in ["none", "paper_normalization", "exact_normalization"]:
        factor = local_oosr_factor(
            treatment=treatment,
            pseudo_optimal_treatment=policy,
            tau=0.1,
            support=support,
            normalization_mode=mode,
            grid_size=200,
        )
        assert factor.shape == (120,)
        assert np.all(np.isfinite(factor))
        assert np.all(factor >= 0)


def test_truncated_kernel_weights_are_zero_outside_unit_band_and_legacy_can_be_requested():
    scaled = np.array([-1.2, -1.0, 0.0, 1.0, 1.2])
    truncated = truncated_gaussian_oosr_kernel(scaled)

    assert np.isclose(GAUSSIAN_TRUNCATION_MASS, 0.682689492137, atol=1e-12)
    assert truncated[0] == 0.0
    assert truncated[-1] == 0.0
    assert np.isclose(truncated[2], gaussian_oosr_kernel(np.array([0.0]))[0] / GAUSSIAN_TRUNCATION_MASS)

    truncated_factor = local_oosr_factor(
        treatment=np.array([[2.0]]),
        pseudo_optimal_treatment=np.array([[0.0]]),
        tau=1.0,
        support=(-5.0, 5.0),
        normalization_mode="none",
        grid_size=200,
        truncated_kernel=True,
    )
    legacy_factor = local_oosr_factor(
        treatment=np.array([[2.0]]),
        pseudo_optimal_treatment=np.array([[0.0]]),
        tau=1.0,
        support=(-5.0, 5.0),
        normalization_mode="none",
        grid_size=200,
        truncated_kernel=False,
    )
    assert truncated_factor[0] == 0.0
    assert legacy_factor[0] > 0.0


def test_finalize_weights_normalizes_to_mean_one():
    raw = np.array([0.01, 0.1, 1.0, 50.0], dtype=float)
    final = finalize_weights(raw, clip_min=0.05, clip_max=10.0)
    assert np.isclose(final.mean(), 1.0)
    assert np.min(final) >= 0.05 / np.mean(np.clip(raw / raw.mean(), 0.05, 10.0)) - 1e-8
    assert np.all(np.isfinite(final))


def test_local_only_weights_match_closed_form_formula():
    treatment = np.array([[0.10], [0.45], [0.80]], dtype=float)
    policy = np.array([[0.15], [0.40], [0.75]], dtype=float)
    density = np.array([0.20, 0.35, 0.50], dtype=float)
    tau = 0.25

    weights = build_local_only_weights(
        treatment=treatment,
        pseudo_optimal_treatment=policy,
        conditional_density=density,
        tau=tau,
        truncated_kernel=False,
    )
    expected = gaussian_oosr_kernel((policy.reshape(-1) - treatment.reshape(-1)) / tau) / (tau * density)
    assert np.allclose(weights, expected)


def test_weighting_config_for_model_falls_back_to_global_and_supports_specific_overrides():
    cfg = ExperimentConfig()
    cfg.weighting.tau = 0.37
    cfg.weighting.lam = 4.2
    cfg.weighting.n_rounds = 6

    pmmr_cfg = weighting_config_for_model(cfg, "weighted_pmmr")
    local_pmmr_cfg = weighting_config_for_model(cfg, "local_pmmr")
    kpv_cfg = weighting_config_for_model(cfg, "weighted_kpv")
    local_kpv_cfg = weighting_config_for_model(cfg, "local_kpv")
    dfpv_cfg = weighting_config_for_model(cfg, "weighted_dfpv")
    local_dfpv_cfg = weighting_config_for_model(cfg, "local_dfpv")
    nmmr_cfg = weighting_config_for_model(cfg, "weighted_nmmr")
    naive_cfg = weighting_config_for_model(cfg, "naive_oosr")
    oracle_naive_cfg = weighting_config_for_model(cfg, "oracle_naive_oosr")
    assert np.isclose(kpv_cfg.tau, 0.37)
    assert np.isclose(kpv_cfg.lam, 4.2)
    assert kpv_cfg.n_rounds == 6
    assert np.isclose(pmmr_cfg.tau, 0.37)
    assert np.isclose(pmmr_cfg.lam, 4.2)
    assert pmmr_cfg.n_rounds == 6
    assert np.isclose(local_pmmr_cfg.tau, 0.37)
    assert np.isclose(local_pmmr_cfg.lam, 4.2)
    assert local_pmmr_cfg.n_rounds == 6
    assert local_pmmr_cfg.weight_mode == "local_only"
    assert np.isclose(dfpv_cfg.tau, 0.37)
    assert np.isclose(dfpv_cfg.lam, 4.2)
    assert dfpv_cfg.n_rounds == 6
    assert np.isclose(local_kpv_cfg.tau, 0.37)
    assert np.isclose(local_kpv_cfg.lam, 4.2)
    assert local_kpv_cfg.n_rounds == 6
    assert local_kpv_cfg.weight_mode == "local_only"
    assert np.isclose(local_dfpv_cfg.tau, 0.37)
    assert np.isclose(local_dfpv_cfg.lam, 4.2)
    assert local_dfpv_cfg.n_rounds == 6
    assert local_dfpv_cfg.weight_mode == "local_only"
    assert np.isclose(nmmr_cfg.tau, 0.37)
    assert np.isclose(nmmr_cfg.lam, 4.2)
    assert nmmr_cfg.n_rounds == 6
    assert np.isclose(naive_cfg.tau, 0.37)
    assert np.isclose(naive_cfg.lam, 4.2)
    assert naive_cfg.n_rounds == 6
    assert np.isclose(oracle_naive_cfg.tau, 0.37)
    assert np.isclose(oracle_naive_cfg.lam, 4.2)
    assert oracle_naive_cfg.n_rounds == 6

    cfg.weighting_pmmr.tau = 0.91
    cfg.weighting_pmmr.lam = 1.3
    cfg.weighting_pmmr.n_rounds = 4
    cfg.weighting_local_pmmr.tau = 0.63
    cfg.weighting_local_pmmr.n_rounds = 2
    cfg.weighting_nmmr_u.tau = 0.22
    cfg.weighting_nmmr_u.lam = 5.5
    cfg.weighting_nmmr_u.n_rounds = 9
    cfg.nmmr.loss_name = "U_statistic"
    cfg.weighting_kpv.tau = 0.41
    cfg.weighting_kpv.lam = 2.7
    cfg.weighting_kpv.n_rounds = 3
    cfg.weighting_local_kpv.tau = 0.29
    cfg.weighting_local_kpv.n_rounds = 4
    cfg.weighting_dfpv.tau = 0.73
    cfg.weighting_dfpv.lam = 6.1
    cfg.weighting_dfpv.n_rounds = 8
    cfg.weighting_local_dfpv.tau = 0.57
    cfg.weighting_local_dfpv.n_rounds = 5
    cfg.weighting_naive_oosr.tau = 0.19
    cfg.weighting_naive_oosr.lam = 1.8
    cfg.weighting_naive_oosr.n_rounds = 7
    cfg.weighting_oracle_naive_oosr.tau = 0.27
    cfg.weighting_oracle_naive_oosr.lam = 3.4
    cfg.weighting_oracle_naive_oosr.n_rounds = 1

    kpv_cfg = weighting_config_for_model(cfg, "weighted_kpv")
    local_kpv_cfg = weighting_config_for_model(cfg, "local_kpv")
    dfpv_cfg = weighting_config_for_model(cfg, "weighted_dfpv")
    local_dfpv_cfg = weighting_config_for_model(cfg, "local_dfpv")
    pmmr_cfg = weighting_config_for_model(cfg, "weighted_pmmr")
    local_pmmr_cfg = weighting_config_for_model(cfg, "local_pmmr")
    nmmr_u_cfg = weighting_config_for_model(cfg, "weighted_nmmr")
    naive_cfg = weighting_config_for_model(cfg, "naive_oosr")
    oracle_naive_cfg = weighting_config_for_model(cfg, "oracle_naive_oosr")
    assert np.isclose(kpv_cfg.tau, 0.41)
    assert np.isclose(kpv_cfg.lam, 2.7)
    assert kpv_cfg.n_rounds == 3
    assert np.isclose(pmmr_cfg.tau, 0.91)
    assert np.isclose(pmmr_cfg.lam, 1.3)
    assert pmmr_cfg.n_rounds == 4
    assert np.isclose(local_pmmr_cfg.tau, 0.63)
    assert np.isclose(local_pmmr_cfg.lam, 4.2)
    assert local_pmmr_cfg.n_rounds == 2
    assert local_pmmr_cfg.weight_mode == "local_only"
    assert np.isclose(dfpv_cfg.tau, 0.73)
    assert np.isclose(dfpv_cfg.lam, 6.1)
    assert dfpv_cfg.n_rounds == 8
    assert np.isclose(local_kpv_cfg.tau, 0.29)
    assert np.isclose(local_kpv_cfg.lam, 4.2)
    assert local_kpv_cfg.n_rounds == 4
    assert local_kpv_cfg.weight_mode == "local_only"
    assert np.isclose(local_dfpv_cfg.tau, 0.57)
    assert np.isclose(local_dfpv_cfg.lam, 4.2)
    assert local_dfpv_cfg.n_rounds == 5
    assert local_dfpv_cfg.weight_mode == "local_only"
    assert np.isclose(nmmr_u_cfg.tau, 0.22)
    assert np.isclose(nmmr_u_cfg.lam, 5.5)
    assert nmmr_u_cfg.n_rounds == 9
    assert np.isclose(naive_cfg.tau, 0.19)
    assert np.isclose(naive_cfg.lam, 1.8)
    assert naive_cfg.n_rounds == 7
    assert np.isclose(oracle_naive_cfg.tau, 0.27)
    assert np.isclose(oracle_naive_cfg.lam, 3.4)
    assert oracle_naive_cfg.n_rounds == 1

    cfg = ExperimentConfig()
    cfg.weighting.tau = 0.51
    cfg.weighting.lam = 7.0
    cfg.weighting.n_rounds = 5
    cfg.weighting_pmmr.n_rounds = 11
    pmmr_cfg = weighting_config_for_model(cfg, "weighted_pmmr")
    assert np.isclose(pmmr_cfg.tau, 0.51)
    assert np.isclose(pmmr_cfg.lam, 7.0)
    assert pmmr_cfg.n_rounds == 11


