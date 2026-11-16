from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def tiny_config(tmp_path):
    from src.config import ExperimentConfig

    cfg = ExperimentConfig()
    cfg.output_dir = tmp_path / "outputs"
    cfg.seed = 13
    cfg.data.n_train = 120
    cfg.data.n_test = 180
    cfg.data.policy_grid_size = 31
    cfg.mean.treatment_grid_size = 11
    cfg.mean.crossfit_folds = 2
    cfg.mean_kpv.treatment_grid_size = 11
    cfg.mean_kpv.crossfit_folds = 2
    cfg.mean_kpv.epochs = 5
    cfg.mean_kpv.batch_size = 64
    cfg.mean_dfpv.treatment_grid_size = 51
    cfg.mean_dfpv.crossfit_folds = 2
    cfg.mean_dfpv.epochs = 5
    cfg.mean_dfpv.batch_size = 64
    cfg.propensity_flow.epochs = 3
    cfg.propensity_flow.batch_size = 64
    cfg.w_flow.epochs = 3
    cfg.w_flow.batch_size = 64
    cfg.mean.epochs = 5
    cfg.mean.batch_size = 64
    cfg.nmmr.epochs = 8
    cfg.nmmr.batch_size = 64
    cfg.kpv.max_train_size = 100
    cfg.kpv.n_lambda1_search = 3
    cfg.kpv.n_lambda2_search = 3
    cfg.kpv.prediction_batch_size = 64
    cfg.dfpv.epochs = 6
    cfg.dfpv.min_epochs = 2
    cfg.dfpv.early_stop_patience = 2
    cfg.dfpv.hidden_dims = (8,)
    cfg.dfpv.feature_dim = 4
    cfg.dfpv.max_train_size = 80
    cfg.dfpv.prediction_batch_size = 64
    cfg.dr_kpv.max_train_size = 60
    cfg.dr_kpv.prediction_batch_size = 32
    cfg.dr_kpv.kpv_lambda1 = 0.01
    cfg.dr_kpv.kpv_lambda2 = 0.01
    cfg.dr_kpv.kap_lambda = 0.01
    cfg.dr_kpv.kap_eta = 0.01
    cfg.dr_kpv.kap_lambda2 = 0.01
    cfg.dr_kpv.kap_zeta = 0.01
    cfg.dr_kpv.lambda_dr = 0.01
    cfg.dr_pmmr.max_train_size = 50
    cfg.dr_pmmr.prediction_batch_size = 32
    cfg.dr_pmmr.pmmr_lambda = 0.01
    cfg.dr_pmmr.kap_lambda = 0.01
    cfg.dr_pmmr.kap_eta = 0.01
    cfg.dr_pmmr.kap_lambda2 = 0.01
    cfg.dr_pmmr.kap_zeta = 0.01
    cfg.dr_pmmr.lambda_dr = 0.01
    cfg.dr_pmmr_conditional.max_train_size = 50
    cfg.dr_pmmr_conditional.prediction_batch_size = 32
    cfg.dr_pmmr_conditional.pmmr_lambda = 0.01
    cfg.dr_pmmr_conditional.kap_lambda = 0.01
    cfg.dr_pmmr_conditional.kap_eta = 0.01
    cfg.dr_pmmr_conditional.kap_lambda2 = 0.01
    cfg.dr_pmmr_conditional.kap_zeta = 0.01
    cfg.dr_pmmr_conditional.lambda_dr = 0.01
    cfg.mean_pmmr.treatment_grid_size = 11
    cfg.mean_pmmr.crossfit_folds = 2
    cfg.mean_pmmr.epochs = 5
    cfg.mean_pmmr.batch_size = 64
    cfg.s_learner.epochs = 8
    cfg.s_learner.batch_size = 64
    cfg.decaflow.latent_dim = 2
    cfg.decaflow.encoder_hidden_features = (8,)
    cfg.decaflow.decoder_hidden_features = (8,)
    cfg.decaflow.conditioner_hidden_features = (4,)
    cfg.decaflow.epochs = 3
    cfg.decaflow.min_epochs = 1
    cfg.decaflow.early_stop_patience = 1
    cfg.decaflow.batch_size = 32
    cfg.decaflow.prediction_batch_size = 32
    cfg.decaflow.interventional_mc_samples = 4
    cfg.weighting.n_rounds = 2
    cfg.weighting.early_stop_patience = 1
    return cfg
