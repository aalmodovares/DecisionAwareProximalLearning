from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from src.experiment import run_experiment


def _decaflow_available() -> bool:
    try:
        from src.models.decaflow import _import_decaflow_components

        _import_decaflow_components()
    except ImportError:
        return False
    return True


ALL_EXPERIMENT_MODELS = [
    "pmmr",
    "local_pmmr",
    "kpv",
    "local_kpv",
    "weighted_kpv",
    "dfpv",
    "local_dfpv",
    "weighted_dfpv",
    "dr_kpv",
    "decaflow",
    "nmmr",
    "nmmr_u",
    "nmmr_v",
    "weighted_pmmr",
    "weighted_nmmr",
    "weighted_nmmr_u",
    "weighted_nmmr_v",
    "s_learner",
    "oracle_s_learner",
    "naive_oosr",
    "oracle_naive_oosr",
]


def test_tiny_end_to_end_run_produces_all_outputs(tiny_config):
    decaflow_available = _decaflow_available()
    selected_models = ALL_EXPERIMENT_MODELS if decaflow_available else [model for model in ALL_EXPERIMENT_MODELS if model != "decaflow"]
    result = run_experiment(tiny_config, selected_models=selected_models)
    methods = {row["method"] for row in result["metrics"]}
    expected = {
        "PMMR",
        "Local PMMR",
        "KPV",
        "Local KPV",
        "Weighted KPV",
        "DFPV",
        "Local DFPV",
        "Weighted DFPV",
        "DR-KPV",
        "DeCaFlow",
        "NMMR",
        "NMMR-U",
        "NMMR-V",
        "Weighted PMMR",
        "Weighted NMMR",
        "Weighted NMMR-U",
        "Weighted NMMR-V",
        "S-Learner",
        "Oracle S-Learner",
        "Naive OOSR",
        "Oracle Naive OOSR",
    }
    if not decaflow_available:
        expected.remove("DeCaFlow")
    assert methods == expected

    output_dir = tiny_config.output_dir
    expected_files = [
        "metrics.csv",
        "artifacts.json",
        "training_histories.json",
        "local_pmmr_history.json",
        "weighted_pmmr_history.json",
        "weighted_nmmr_history.json",
        "local_kpv_history.json",
        "weighted_kpv_history.json",
        "local_dfpv_history.json",
        "weighted_dfpv_history.json",
        "naive_oosr_history.json",
        "local_pmmr_diagnostics.npz",
        "weighted_pmmr_diagnostics.npz",
        "weighted_nmmr_diagnostics.npz",
        "local_kpv_diagnostics.npz",
        "weighted_kpv_diagnostics.npz",
        "local_dfpv_diagnostics.npz",
        "weighted_dfpv_diagnostics.npz",
        "naive_oosr_diagnostics.npz",
        "train_sample.npz",
        "test_sample.npz",
        "policies.npz",
        "mean_potential_outcome_curves.npz",
        "mean_potential_outcome_curves_direct.npz",
    ]
    for name in expected_files:
        assert (output_dir / name).exists(), name

    figures_dir = output_dir / "figures"
    expected_figures = [
        "train_dataset_diagnostics.png",
        "train_population_ate_curve.png",
        "train_conditional_distributions.png",
        "train_pairwise_scatters.png",
        "method_metrics.png",
        "policy_scatter.png",
        "policy_vs_x0.png",
        "policy_error_by_region.png",
        "mean_potential_outcome_curves.png",
        "mean_curve_direct.png",
        "mean_potential_outcome_curves_direct.png",
        "local_pmmr_region_effect.png",
        "weighted_pmmr_region_effect.png",
        "weighted_nmmr_region_effect.png",
        "local_kpv_region_effect.png",
        "weighted_kpv_region_effect.png",
        "local_dfpv_region_effect.png",
        "weighted_dfpv_region_effect.png",
        "naive_oosr_region_effect.png",
    ]
    for name in expected_figures:
        assert (figures_dir / name).exists(), name

    patient_dir = figures_dir / "patient_surfaces"
    assert patient_dir.exists()
    patient_files = sorted(patient_dir.glob("patient_*.png"))
    assert len(patient_files) >= 10
    assert (patient_dir / "patient_surface_gallery.png").exists()

    with (output_dir / "metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_rows = 21 if decaflow_available else 20
    assert len(rows) == expected_rows

    policies = np.load(output_dir / "policies.npz")
    assert "optimal_treatment" in policies.files
    assert "KPV" in policies.files
    assert "Local_KPV" in policies.files
    assert "Weighted_KPV" in policies.files
    assert "DFPV" in policies.files
    assert "Local_DFPV" in policies.files
    assert "Weighted_DFPV" in policies.files
    assert "DR_KPV" in policies.files
    if decaflow_available:
        assert "DeCaFlow" in policies.files
    assert "Local_PMMR" in policies.files
    assert "Weighted_PMMR" in policies.files
    assert "Weighted_NMMR" in policies.files
    assert "Oracle_S_Learner" in policies.files
    assert "Oracle_Naive_OOSR" in policies.files

    assert "figure_files" in result["artifacts"]
    assert len(result["artifacts"]["figure_files"]) >= len(expected_figures) + 11

def test_local_kpv_end_to_end_run_produces_outputs(tiny_config):
    result = run_experiment(tiny_config, selected_models=["local_kpv"])
    methods = {row["method"] for row in result["metrics"]}
    assert methods == {"Local KPV"}

    output_dir = tiny_config.output_dir
    assert (output_dir / "metrics.csv").exists()
    assert (output_dir / "training_histories.json").exists()
    assert (output_dir / "local_kpv_history.json").exists()
    assert (output_dir / "local_kpv_diagnostics.npz").exists()
    assert (output_dir / "mean_potential_outcome_curves.npz").exists()
    assert (output_dir / "mean_potential_outcome_curves_direct.npz").exists()

    policies = np.load(output_dir / "policies.npz")
    assert "Local_KPV" in policies.files
    curves = np.load(output_dir / "mean_potential_outcome_curves.npz")
    assert "Local_KPV" in curves.files

def test_kpv_end_to_end_run_produces_outputs(tiny_config):
    result = run_experiment(tiny_config, selected_models=["kpv"])
    methods = {row["method"] for row in result["metrics"]}
    assert methods == {"KPV"}

    output_dir = tiny_config.output_dir
    assert (output_dir / "metrics.csv").exists()
    assert (output_dir / "training_histories.json").exists()
    assert (output_dir / "mean_potential_outcome_curves.npz").exists()
    assert (output_dir / "mean_potential_outcome_curves_direct.npz").exists()

    policies = np.load(output_dir / "policies.npz")
    assert "KPV" in policies.files
    curves = np.load(output_dir / "mean_potential_outcome_curves.npz")
    assert "KPV" in curves.files
    with (output_dir / "training_histories.json").open(encoding="utf-8") as handle:
        histories = json.load(handle)
    assert "kpv_losses" in histories


def test_dfpv_end_to_end_run_produces_outputs(tiny_config):
    result = run_experiment(tiny_config, selected_models=["dfpv"])
    methods = {row["method"] for row in result["metrics"]}
    assert methods == {"DFPV"}

    output_dir = tiny_config.output_dir
    assert (output_dir / "metrics.csv").exists()
    assert (output_dir / "training_histories.json").exists()
    assert (output_dir / "mean_potential_outcome_curves.npz").exists()
    assert (output_dir / "mean_potential_outcome_curves_direct.npz").exists()

    policies = np.load(output_dir / "policies.npz")
    assert "DFPV" in policies.files
    curves = np.load(output_dir / "mean_potential_outcome_curves.npz")
    assert "DFPV" in curves.files
    with (output_dir / "training_histories.json").open(encoding="utf-8") as handle:
        histories = json.load(handle)
    assert "dfpv_losses" in histories


def test_dr_pmmr_conditional_end_to_end_run_produces_outputs(tiny_config):
    result = run_experiment(tiny_config, selected_models=["dr_pmmr_conditional"])
    methods = {row["method"] for row in result["metrics"]}
    assert methods == {"DR-PMMR conditional"}

    output_dir = tiny_config.output_dir
    assert (output_dir / "metrics.csv").exists()
    assert (output_dir / "training_histories.json").exists()
    assert (output_dir / "mean_potential_outcome_curves.npz").exists()
    assert (output_dir / "mean_potential_outcome_curves_direct.npz").exists()

    policies = np.load(output_dir / "policies.npz")
    assert "DR_PMMR_Conditional" in policies.files
    curves = np.load(output_dir / "mean_potential_outcome_curves.npz")
    assert "DR_PMMR_Conditional" in curves.files
    with (output_dir / "training_histories.json").open(encoding="utf-8") as handle:
        histories = json.load(handle)
    assert "dr_pmmr_conditional_losses" in histories


@pytest.mark.skipif(not _decaflow_available(), reason="Conditional DeCaFlow optional dependencies are unavailable.")
def test_decaflow_end_to_end_run_produces_outputs(tiny_config):
    result = run_experiment(tiny_config, selected_models=["decaflow"])
    methods = {row["method"] for row in result["metrics"]}
    assert methods == {"DeCaFlow"}

    output_dir = tiny_config.output_dir
    assert (output_dir / "metrics.csv").exists()
    assert (output_dir / "training_histories.json").exists()
    assert (output_dir / "mean_potential_outcome_curves.npz").exists()

    policies = np.load(output_dir / "policies.npz")
    assert "DeCaFlow" in policies.files
    curves = np.load(output_dir / "mean_potential_outcome_curves.npz")
    assert "DeCaFlow" in curves.files
    with (output_dir / "training_histories.json").open(encoding="utf-8") as handle:
        histories = json.load(handle)
    assert "decaflow_losses" in histories


def test_nmmr_u_direct_curve_is_saved_under_configured_output_dir(tiny_config):
    result = run_experiment(tiny_config, selected_models=["nmmr_u"])
    figures_dir = tiny_config.output_dir / "figures"
    assert (figures_dir / "mean_curve_direct.png").exists()
    assert (figures_dir / "mean_potential_outcome_curves_direct.png").exists()
    assert (tiny_config.output_dir / "mean_curve_direct.csv").exists()
    assert (tiny_config.output_dir / "mean_curve_direct.npz").exists()
    assert "figure_files" in result["artifacts"]
    assert any(str(path).endswith("mean_curve_direct.png") for path in result["artifacts"]["figure_files"])


def test_nmmr_u_no_treatment_grid_uses_fixed_evaluation_grid(tiny_config):
    tiny_config.mean_nmmr_u.use_treatment_grid = False
    result = run_experiment(tiny_config, selected_models=["nmmr_u"])
    curves = np.load(tiny_config.output_dir / "mean_potential_outcome_curves.npz")
    assert curves["treatment_grid"].shape[0] == tiny_config.mean_nmmr_u.evaluation_grid_size
    assert result["artifacts"]["config"]["data"]["policy_grid_size"] == tiny_config.mean_nmmr_u.evaluation_grid_size


def test_oracle_models_force_two_stage_even_when_disabled(tiny_config):
    tiny_config.use_mean_slearner = False
    tiny_config.use_mean_oosr = False
    result = run_experiment(tiny_config, selected_models=["oracle_s_learner", "oracle_naive_oosr"])
    rows = {row["method"]: row for row in result["metrics"]}

    oracle_s = rows["Oracle S-Learner"]
    oracle_oosr = rows["Oracle Naive OOSR"]
    assert "factual_mean_rmse" in oracle_s
    assert "factual_mean_rmse" in oracle_oosr


