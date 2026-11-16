from __future__ import annotations

import copy

import numpy as np

from src.data.simulator import Wu2024Simulator
from src.models.dfpv import BridgeTrainData as DFPVTrainData
from src.models.dfpv import DFPVBridge
from src.models.dr_kernel_proxy import BridgeTrainData as DRKernelTrainData
from src.models.dr_kernel_proxy import DRKPVBridge, DRPMMRConditionalBridge, DRPMMRMarginalBridge
from src.models.kpv import BridgeTrainData as KPVTrainData
from src.models.kpv import KPVBridge
from src.models.nmmr import BridgeTrainData as NMMRTrainData
from src.models.nmmr import NMMRBridge
from src.models.pmmr import BridgeTrainData as PMMRTrainData
from src.models.pmmr import PMMRBridge
from src.models.s_learner import SLearner


def _bridge_data(sample):
    return PMMRTrainData(
        treatment=sample.treatment,
        treatment_proxy=sample.treatment_proxy,
        outcome_proxy=sample.outcome_proxy,
        covariates=sample.covariates,
        outcome=sample.outcome,
    )


def _bridge_data_nmmr(sample):
    return NMMRTrainData(
        treatment=sample.treatment,
        treatment_proxy=sample.treatment_proxy,
        outcome_proxy=sample.outcome_proxy,
        covariates=sample.covariates,
        outcome=sample.outcome,
    )


def _bridge_data_kpv(sample):
    return KPVTrainData(
        treatment=sample.treatment,
        treatment_proxy=sample.treatment_proxy,
        outcome_proxy=sample.outcome_proxy,
        covariates=sample.covariates,
        outcome=sample.outcome,
    )


def _bridge_data_dfpv(sample):
    return DFPVTrainData(
        treatment=sample.treatment,
        treatment_proxy=sample.treatment_proxy,
        outcome_proxy=sample.outcome_proxy,
        covariates=sample.covariates,
        outcome=sample.outcome,
    )


def _bridge_data_dr(sample):
    return DRKernelTrainData(
        treatment=sample.treatment,
        treatment_proxy=sample.treatment_proxy,
        outcome_proxy=sample.outcome_proxy,
        covariates=sample.covariates,
        outcome=sample.outcome,
    )


def _assert_crossfit_metadata(model, n_samples: int, grid_size: int, n_folds: int) -> None:
    assert model.mean_pseudooutcome_metadata_["crossfit"] is True
    assert model.mean_pseudooutcome_metadata_["n_rows"] == n_samples * grid_size
    assert model.mean_pseudooutcome_metadata_["n_original"] == n_samples
    assert model.mean_pseudooutcome_metadata_["treatment_grid_size"] == grid_size
    assert len(model.mean_crossfit_folds_) == n_folds
    assert model.mean_regressor_group_ids_.shape == (n_samples * grid_size,)
    holdout = np.concatenate([fold["holdout_idx"] for fold in model.mean_crossfit_folds_])
    assert np.array_equal(np.sort(holdout), np.arange(n_samples))
    for fold in model.mean_crossfit_folds_:
        assert np.intersect1d(fold["train_idx"], fold["holdout_idx"]).size == 0


def test_pmmr_bridge_fit_and_predict(tiny_config):
    simulator = Wu2024Simulator(tiny_config.data)
    train_sample, _ = simulator.sample_train_test(tiny_config.seed)
    model = PMMRBridge(tiny_config.pmmr, tiny_config.mean, tiny_config.w_flow, simulator.support)
    sample_weight = np.linspace(0.5, 1.5, train_sample.treatment.shape[0])
    model.fit(_bridge_data(train_sample), sample_weight=sample_weight, mean_method="pseudo_outcomes")

    pred = model.mean_predict(train_sample.treatment[:8], train_sample.covariates[:8])
    assert pred.shape == (8, 1)
    assert np.all(np.isfinite(pred))
    _assert_crossfit_metadata(
        model,
        n_samples=train_sample.treatment.shape[0],
        grid_size=tiny_config.mean.treatment_grid_size,
        n_folds=tiny_config.mean.crossfit_folds,
    )
    first_fold = model.mean_crossfit_folds_[0]
    assert np.isclose(first_fold["train_weight_sum"], np.sum(sample_weight[first_fold["train_idx"]]))
    assert np.isclose(first_fold["holdout_weight_sum"], np.sum(sample_weight[first_fold["holdout_idx"]]))


def test_kpv_bridge_fit_and_predict(tiny_config):
    simulator = Wu2024Simulator(tiny_config.data)
    train_sample, _ = simulator.sample_train_test(tiny_config.seed)
    model = KPVBridge(tiny_config.kpv, tiny_config.mean_kpv, tiny_config.w_flow, simulator.support)
    model.fit(_bridge_data_kpv(train_sample), mean_method="pseudo_outcomes")

    pred = model.mean_predict(train_sample.treatment[:8], train_sample.covariates[:8])
    bridge = model.bridge_predict(
        train_sample.treatment[:8],
        train_sample.outcome_proxy[:8],
        train_sample.covariates[:8],
    )
    assert pred.shape == (8, 1)
    assert bridge.shape == (8, 1)
    assert np.all(np.isfinite(pred))
    assert np.all(np.isfinite(bridge))
    assert model.alpha is not None
    assert model.stage1_indices_ is not None
    assert model.stage2_indices_ is not None
    assert model.alpha.shape == (model.stage1_indices_.shape[0], model.stage2_indices_.shape[0])
    assert model.lambda1_ is not None
    assert model.lambda2_ is not None
    assert len(model.lambda1_search_) == tiny_config.kpv.n_lambda1_search
    assert len(model.lambda2_search_) == tiny_config.kpv.n_lambda2_search
    _assert_crossfit_metadata(
        model,
        n_samples=train_sample.treatment.shape[0],
        grid_size=tiny_config.mean_kpv.treatment_grid_size,
        n_folds=tiny_config.mean_kpv.crossfit_folds,
    )


def test_dfpv_bridge_fit_and_predict(tiny_config):
    simulator = Wu2024Simulator(tiny_config.data)
    train_sample, _ = simulator.sample_train_test(tiny_config.seed)
    model = DFPVBridge(tiny_config.dfpv, tiny_config.mean_dfpv, tiny_config.w_flow, simulator.support)
    model.fit(_bridge_data_dfpv(train_sample), mean_method="pseudo_outcomes")

    pred = model.mean_predict(train_sample.treatment[:8], train_sample.covariates[:8])
    bridge = model.bridge_predict(
        train_sample.treatment[:8],
        train_sample.outcome_proxy[:8],
        train_sample.covariates[:8],
    )
    assert pred.shape == (8, 1)
    assert bridge.shape == (8, 1)
    assert np.all(np.isfinite(pred))
    assert np.all(np.isfinite(bridge))


def test_dr_kpv_fit_and_predict(tiny_config):
    simulator = Wu2024Simulator(tiny_config.data)
    train_sample, _ = simulator.sample_train_test(tiny_config.seed)
    model = DRKPVBridge(tiny_config.dr_kpv, tiny_config.mean_kpv, simulator.support)
    model.fit(_bridge_data_dr(train_sample), seed=tiny_config.seed)

    pred = model.mean_predict(train_sample.treatment[:8], train_sample.covariates[:8])
    bridge = model.bridge_predict(
        train_sample.treatment[:8],
        train_sample.outcome_proxy[:8],
        train_sample.covariates[:8],
    )
    assert pred.shape == (8, 1)
    assert bridge.shape == (8, 1)
    assert np.all(np.isfinite(pred))
    assert np.all(np.isfinite(bridge))
    assert model.fit_indices_.shape[0] <= tiny_config.dr_kpv.max_train_size
    assert model.outcome_bridge.lambda1 == tiny_config.dr_kpv.kpv_lambda1
    assert model.outcome_bridge.lambda2 == tiny_config.dr_kpv.kpv_lambda2
    assert model.treatment_bridge.lambda_ == tiny_config.dr_kpv.kap_lambda
    assert model.lambda_dr_ == tiny_config.dr_kpv.lambda_dr
    grid = simulator.policy_grid[:4]
    grid_pred = model.mean_predict_grid(grid[:, None], train_sample.covariates[:5])
    expected = np.vstack(
        [
            model.mean_predict(np.full((5, 1), a_val, dtype=float), train_sample.covariates[:5]).reshape(-1)
            for a_val in grid
        ]
    )
    assert grid_pred.shape == (4, 5)
    assert np.allclose(grid_pred, expected)


def test_dr_pmmr_marginal_fit_and_predict(tiny_config):
    simulator = Wu2024Simulator(tiny_config.data)
    train_sample, _ = simulator.sample_train_test(tiny_config.seed)
    model = DRPMMRMarginalBridge(tiny_config.dr_pmmr, tiny_config.mean_pmmr, simulator.support)
    model.fit(_bridge_data_dr(train_sample), seed=tiny_config.seed)

    pred = model.mean_predict(train_sample.treatment[:6], train_sample.covariates[:6])
    assert pred.shape == (6, 1)
    assert np.all(np.isfinite(pred))
    assert model.fit_indices_.shape[0] <= tiny_config.dr_pmmr.max_train_size
    grid = simulator.policy_grid[:4]
    grid_pred = model.mean_predict_grid(grid[:, None], train_sample.covariates[:5])
    expected = np.vstack(
        [
            model.mean_predict(np.full((5, 1), a_val, dtype=float), train_sample.covariates[:5]).reshape(-1)
            for a_val in grid
        ]
    )
    assert grid_pred.shape == (4, 5)
    assert np.allclose(grid_pred, expected)


def test_dr_pmmr_conditional_grid_predict_matches_rowwise_calls(tiny_config):
    simulator = Wu2024Simulator(tiny_config.data)
    train_sample, _ = simulator.sample_train_test(tiny_config.seed)
    model = DRPMMRConditionalBridge(tiny_config.dr_pmmr_conditional, tiny_config.mean_pmmr, simulator.support)
    model.fit(_bridge_data_dr(train_sample), seed=tiny_config.seed)

    grid = simulator.policy_grid[:4]
    grid_pred = model.mean_predict_grid(grid[:, None], train_sample.covariates[:5])
    expected = np.vstack(
        [
            model.mean_predict(np.full((5, 1), a_val, dtype=float), train_sample.covariates[:5]).reshape(-1)
            for a_val in grid
        ]
    )
    assert grid_pred.shape == (4, 5)
    assert np.allclose(grid_pred, expected)


def test_nmmr_bridge_fit_and_predict(tiny_config):
    simulator = Wu2024Simulator(tiny_config.data)
    train_sample, _ = simulator.sample_train_test(tiny_config.seed)
    model = NMMRBridge(tiny_config.nmmr, tiny_config.mean, tiny_config.w_flow, simulator.support)
    model.fit(_bridge_data_nmmr(train_sample), mean_method="pseudo_outcomes")

    pred = model.mean_predict(train_sample.treatment[:8], train_sample.covariates[:8])
    assert pred.shape == (8, 1)
    assert np.all(np.isfinite(pred))
    _assert_crossfit_metadata(
        model,
        n_samples=train_sample.treatment.shape[0],
        grid_size=tiny_config.mean.treatment_grid_size,
        n_folds=tiny_config.mean.crossfit_folds,
    )


def test_nmmr_bridge_fit_with_product_kernel_and_deactivated_rolloff(tiny_config):
    simulator = Wu2024Simulator(tiny_config.data)
    train_sample, _ = simulator.sample_train_test(tiny_config.seed)
    tiny_config.nmmr.product_kernel = True
    tiny_config.nmmr.deactivate_rolloff = True
    model = NMMRBridge(tiny_config.nmmr, tiny_config.mean, tiny_config.w_flow, simulator.support)
    model.fit(_bridge_data_nmmr(train_sample), mean_method="pseudo_outcomes")

    pred = model.mean_predict(train_sample.treatment[:8], train_sample.covariates[:8])
    bridge = model.bridge_predict(
        train_sample.treatment[:8],
        train_sample.outcome_proxy[:8],
        train_sample.covariates[:8],
    )
    assert pred.shape == (8, 1)
    assert bridge.shape == (8, 1)
    assert np.all(np.isfinite(pred))
    assert np.all(np.isfinite(bridge))


def test_nmmr_quadratic_treatment_adds_bridge_input_feature(tiny_config):
    simulator = Wu2024Simulator(tiny_config.data)
    train_sample, _ = simulator.sample_train_test(tiny_config.seed)
    tiny_config.nmmr.quadratic_treatment = True
    model = NMMRBridge(tiny_config.nmmr, tiny_config.mean, tiny_config.w_flow, simulator.support)
    model.fit(_bridge_data_nmmr(train_sample), mean_method="pseudo_outcomes")

    expected_dim = (
        train_sample.treatment.shape[1]
        + train_sample.treatment.shape[1]
        + train_sample.outcome_proxy.shape[1]
        + train_sample.covariates.shape[1]
    )
    assert model.model is not None
    assert model.model.layers[0].in_features == expected_dim
    bridge = model.bridge_predict(
        train_sample.treatment[:8],
        train_sample.outcome_proxy[:8],
        train_sample.covariates[:8],
    )
    assert bridge.shape == (8, 1)
    assert np.all(np.isfinite(bridge))


def test_pmmr_mean_regressor_can_use_only_factual_treatments(tiny_config):
    simulator = Wu2024Simulator(tiny_config.data)
    train_sample, _ = simulator.sample_train_test(tiny_config.seed)
    tiny_config.mean.use_treatment_grid = False
    model = PMMRBridge(tiny_config.pmmr, tiny_config.mean, tiny_config.w_flow, simulator.support)
    model.fit(_bridge_data(train_sample), mean_method="pseudo_outcomes")

    assert model.mean_pseudooutcome_metadata_["uses_treatment_grid"] is False
    assert model.mean_pseudooutcome_metadata_["treatment_grid_size"] == 0
    assert model.mean_pseudooutcome_metadata_["n_rows"] == train_sample.treatment.shape[0]
    pred = model.mean_predict(train_sample.treatment[:8], train_sample.covariates[:8])
    assert pred.shape == (8, 1)
    assert np.all(np.isfinite(pred))


def test_nmmr_mean_regressor_can_use_only_factual_treatments(tiny_config):
    simulator = Wu2024Simulator(tiny_config.data)
    train_sample, _ = simulator.sample_train_test(tiny_config.seed)
    tiny_config.mean.use_treatment_grid = False
    model = NMMRBridge(tiny_config.nmmr, tiny_config.mean, tiny_config.w_flow, simulator.support)
    model.fit(_bridge_data_nmmr(train_sample), mean_method="pseudo_outcomes")

    assert model.mean_pseudooutcome_metadata_["uses_treatment_grid"] is False
    assert model.mean_pseudooutcome_metadata_["treatment_grid_size"] == 0
    assert model.mean_pseudooutcome_metadata_["n_rows"] == train_sample.treatment.shape[0]
    pred = model.mean_predict(train_sample.treatment[:8], train_sample.covariates[:8])
    assert pred.shape == (8, 1)
    assert np.all(np.isfinite(pred))


def test_kpv_mean_regressor_can_use_only_factual_treatments(tiny_config):
    simulator = Wu2024Simulator(tiny_config.data)
    train_sample, _ = simulator.sample_train_test(tiny_config.seed)
    tiny_config.mean_kpv.use_treatment_grid = False
    model = KPVBridge(tiny_config.kpv, tiny_config.mean_kpv, tiny_config.w_flow, simulator.support)
    model.fit(_bridge_data_kpv(train_sample), mean_method="pseudo_outcomes")

    assert model.mean_pseudooutcome_metadata_["uses_treatment_grid"] is False
    assert model.mean_pseudooutcome_metadata_["treatment_grid_size"] == 0
    assert model.mean_pseudooutcome_metadata_["n_rows"] == train_sample.treatment.shape[0]
    pred = model.mean_predict(train_sample.treatment[:8], train_sample.covariates[:8])
    assert pred.shape == (8, 1)
    assert np.all(np.isfinite(pred))


def test_dfpv_mean_regressor_can_use_only_factual_treatments(tiny_config):
    simulator = Wu2024Simulator(tiny_config.data)
    train_sample, _ = simulator.sample_train_test(tiny_config.seed)
    tiny_config.mean_dfpv.use_treatment_grid = False
    model = DFPVBridge(tiny_config.dfpv, tiny_config.mean_dfpv, tiny_config.w_flow, simulator.support)
    model.fit(_bridge_data_dfpv(train_sample), mean_method="pseudo_outcomes")

    assert model.mean_pseudooutcome_metadata_["uses_treatment_grid"] is False
    assert model.mean_pseudooutcome_metadata_["treatment_grid_size"] == 0
    assert model.mean_pseudooutcome_metadata_["n_rows"] == train_sample.treatment.shape[0]
    pred = model.mean_predict(train_sample.treatment[:8], train_sample.covariates[:8])
    assert pred.shape == (8, 1)
    assert np.all(np.isfinite(pred))


def test_nmmr_product_kernel_group_scales_rescale_bandwidth_groups(tiny_config):
    simulator = Wu2024Simulator(tiny_config.data)
    train_sample, _ = simulator.sample_train_test(tiny_config.seed)
    base_cfg = copy.deepcopy(tiny_config.nmmr)
    base_cfg.product_kernel = True
    scaled_cfg = copy.deepcopy(base_cfg)
    scaled_cfg.a_scale = 2.0
    scaled_cfg.z_scale = 0.5
    scaled_cfg.x_scale = 1.5

    base_model = NMMRBridge(base_cfg, tiny_config.mean, tiny_config.w_flow, simulator.support)
    scaled_model = NMMRBridge(scaled_cfg, tiny_config.mean, tiny_config.w_flow, simulator.support)
    base_model._fit_bridge(_bridge_data_nmmr(train_sample))
    scaled_model._fit_bridge(_bridge_data_nmmr(train_sample))

    base_bw = np.asarray(base_model.azx_bandwidth, dtype=float)
    scaled_bw = np.asarray(scaled_model.azx_bandwidth, dtype=float)
    a_slice = scaled_model.moment_group_slices_["A"]
    z_slice = scaled_model.moment_group_slices_["Z"]
    x_slice = scaled_model.moment_group_slices_["X"]
    assert np.allclose(scaled_bw[a_slice], base_bw[a_slice] * 2.0)
    assert np.allclose(scaled_bw[z_slice], base_bw[z_slice] * 0.5)
    assert np.allclose(scaled_bw[x_slice], base_bw[x_slice] * 1.5)


def test_s_learner_fit_and_predict(tiny_config):
    simulator = Wu2024Simulator(tiny_config.data)
    train_sample, _ = simulator.sample_train_test(tiny_config.seed)
    model = SLearner(tiny_config.s_learner, tiny_config.mean, simulator.support)
    model.fit(
        train_sample.treatment,
        train_sample.treatment_proxy,
        train_sample.outcome_proxy,
        train_sample.covariates,
        train_sample.outcome,
    )

    pred = model.direct_predict(
        train_sample.treatment[:8],
        train_sample.treatment_proxy[:8],
        train_sample.outcome_proxy[:8],
        train_sample.covariates[:8],
    )
    assert pred.shape == (8, 1)
    assert np.all(np.isfinite(pred))
    assert model.mean_regressor is None


def test_s_learner_direct_mode_without_proxies_ignores_z_w(tiny_config):
    simulator = Wu2024Simulator(tiny_config.data)
    train_sample, _ = simulator.sample_train_test(tiny_config.seed)
    model = SLearner(tiny_config.s_learner, tiny_config.mean, simulator.support)
    model.fit(
        train_sample.treatment,
        train_sample.treatment_proxy,
        train_sample.outcome_proxy,
        train_sample.covariates,
        train_sample.outcome,
        include_proxies=False,
    )

    treatment = train_sample.treatment[:8]
    covariates = train_sample.covariates[:8]
    z_ref = train_sample.treatment_proxy[:8]
    w_ref = train_sample.outcome_proxy[:8]
    z_alt = np.flipud(z_ref.copy())
    w_alt = np.flipud(w_ref.copy())

    pred_ref = model.direct_predict(treatment, z_ref, w_ref, covariates)
    pred_alt = model.direct_predict(treatment, z_alt, w_alt, covariates)
    assert np.allclose(pred_ref, pred_alt)

