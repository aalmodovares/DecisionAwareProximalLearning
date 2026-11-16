from __future__ import annotations

import numpy as np

from src.config import KernelDiagnosticsConfig
from src.data.simulator import Wu2024Simulator
from src.models.kernel_diagnostics import summarize_kernel
from src.models.kernel_utils import rbf_kernel
from src.models.kpv import BridgeTrainData as KPVTrainData
from src.models.kpv import KPVBridge
from src.models.nmmr import BridgeTrainData as NMMRTrainData
from src.models.nmmr import NMMRBridge
from src.models.pmmr import BridgeTrainData as PMMRTrainData
from src.models.pmmr import PMMRBridge


def _metadata() -> dict[str, object]:
    return {
        "model": "test",
        "stage": "test/bridge",
        "round": "",
        "estimator": "test",
        "kernel": "K",
        "weighted": False,
        "bandwidth_multiplier": 1.0,
        "bandwidth_min": 1.0,
        "bandwidth_mean": 1.0,
        "bandwidth_max": 1.0,
    }


def _pmmr_data(sample) -> PMMRTrainData:
    return PMMRTrainData(
        treatment=sample.treatment,
        treatment_proxy=sample.treatment_proxy,
        outcome_proxy=sample.outcome_proxy,
        covariates=sample.covariates,
        outcome=sample.outcome,
    )


def _nmmr_data(sample) -> NMMRTrainData:
    return NMMRTrainData(
        treatment=sample.treatment,
        treatment_proxy=sample.treatment_proxy,
        outcome_proxy=sample.outcome_proxy,
        covariates=sample.covariates,
        outcome=sample.outcome,
    )


def _kpv_data(sample) -> KPVTrainData:
    return KPVTrainData(
        treatment=sample.treatment,
        treatment_proxy=sample.treatment_proxy,
        outcome_proxy=sample.outcome_proxy,
        covariates=sample.covariates,
        outcome=sample.outcome,
    )


def test_kernel_summary_flags_too_diagonal_and_too_constant():
    cfg = KernelDiagnosticsConfig(max_points=50)
    x = np.linspace(0.0, 1.0, 30)[:, None]

    diagonal_result = summarize_kernel(
        rbf_kernel(x, x, bandwidth=1e-4),
        metadata=_metadata(),
        config=cfg,
    )
    assert diagonal_result.rows[0]["status"] == "TOO_DIAGONAL"

    constant_result = summarize_kernel(
        rbf_kernel(x, x, bandwidth=1e4),
        metadata=_metadata(),
        config=cfg,
    )
    assert constant_result.rows[0]["status"] == "TOO_CONSTANT"


def test_pmmr_kernel_diagnostics_returns_awx_and_azx_rows(tiny_config):
    tiny_config.kernel_diagnostics.enabled = True
    tiny_config.kernel_diagnostics.max_points = 40
    simulator = Wu2024Simulator(tiny_config.data)
    train_sample, _ = simulator.sample_train_test(tiny_config.seed)
    model = PMMRBridge(tiny_config.pmmr, tiny_config.mean, tiny_config.w_flow, simulator.support)
    sample_weight = np.linspace(0.5, 1.5, train_sample.treatment.shape[0])
    model._fit_bridge(_pmmr_data(train_sample), sample_weight=sample_weight)

    result = model.kernel_diagnostics(
        config=tiny_config.kernel_diagnostics,
        model_name="PMMR",
        sample_weight=sample_weight,
    )
    kernels = {row["kernel"] for row in result.rows}
    assert {"K_AWX", "K_AZX"} <= kernels
    assert {False, True} <= {row["weighted"] for row in result.rows}


def test_nmmr_kernel_diagnostics_returns_azx_rows(tiny_config):
    tiny_config.kernel_diagnostics.enabled = True
    tiny_config.kernel_diagnostics.max_points = 40
    tiny_config.nmmr.epochs = 2
    tiny_config.nmmr.min_epochs = 1
    tiny_config.nmmr.early_stop_patience = 1
    simulator = Wu2024Simulator(tiny_config.data)
    train_sample, _ = simulator.sample_train_test(tiny_config.seed)
    model = NMMRBridge(tiny_config.nmmr, tiny_config.mean, tiny_config.w_flow, simulator.support)
    model._fit_bridge(_nmmr_data(train_sample))

    result = model.kernel_diagnostics(config=tiny_config.kernel_diagnostics, model_name="NMMR")
    assert {row["kernel"] for row in result.rows} == {"K_AZX"}
    assert all(np.isfinite(float(row["offdiag_mean"])) for row in result.rows)


def test_kpv_kernel_diagnostics_returns_stage_rows(tiny_config):
    tiny_config.kernel_diagnostics.enabled = True
    tiny_config.kernel_diagnostics.max_points = 40
    simulator = Wu2024Simulator(tiny_config.data)
    train_sample, _ = simulator.sample_train_test(tiny_config.seed)
    model = KPVBridge(tiny_config.kpv, tiny_config.mean_kpv, tiny_config.w_flow, simulator.support)
    model._fit_bridge(_kpv_data(train_sample))

    result = model.kernel_diagnostics(config=tiny_config.kernel_diagnostics, model_name="KPV")
    assert {row["kernel"] for row in result.rows} == {"K_AZX_stage1", "K_W_stage1", "K_AX_stage2"}
    assert all(np.isfinite(float(row["offdiag_mean"])) for row in result.rows)


def test_nmmr_product_kernel_diagnostics_returns_product_rows(tiny_config):
    tiny_config.kernel_diagnostics.enabled = True
    tiny_config.kernel_diagnostics.max_points = 40
    tiny_config.nmmr.product_kernel = True
    tiny_config.nmmr.epochs = 2
    tiny_config.nmmr.min_epochs = 1
    tiny_config.nmmr.early_stop_patience = 1
    simulator = Wu2024Simulator(tiny_config.data)
    train_sample, _ = simulator.sample_train_test(tiny_config.seed)
    model = NMMRBridge(tiny_config.nmmr, tiny_config.mean, tiny_config.w_flow, simulator.support)
    model._fit_bridge(_nmmr_data(train_sample))

    result = model.kernel_diagnostics(config=tiny_config.kernel_diagnostics, model_name="NMMR")
    kernels = {row["kernel"] for row in result.rows}
    assert {"K_AZX_product", "K_A", "K_Z", "K_X"} <= kernels
    components = {row.get("component") for row in result.rows}
    assert {"AZX", "A", "Z", "X"} <= components
    assert all(np.isfinite(float(row["offdiag_mean"])) for row in result.rows)
