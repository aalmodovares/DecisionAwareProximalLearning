from __future__ import annotations

import numpy as np
import pytest
import torch

from src.config import DeCaFlowConfig
from src.models.decaflow import ConditionalDeCaFlow


pytest.importorskip("lightning")
pytest.importorskip("causalflows")


def _toy_proximal_sample(n: int, seed: int = 0) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 1)).astype(np.float32)
    u = ( 0.4 * rng.normal(size=(n, 1))).astype(np.float32)
    z = (0.8 * u + 0.4 * x + 0.2 * rng.normal(size=(n, 1))).astype(np.float32)
    w = (0.7 * u + 0.5 * x + 0.2 * rng.normal(size=(n, 1))).astype(np.float32)
    a = (0.6 * u + 0.5 * z + 0.3 * x + 0.15 * rng.normal(size=(n, 1))).astype(np.float32)
    y = (1.1 * a + 0.7 * w + 0.5 * u + 0.4 * x + 0.1 * rng.normal(size=(n, 1))).astype(np.float32)
    return a, z, w, x, y


def test_conditional_decaflow_adapter_builds_expected_adjacency_and_output_shapes():
    train = _toy_proximal_sample(10, seed=1)
    val = _toy_proximal_sample(6, seed=2)

    adapter = ConditionalDeCaFlow(
        DeCaFlowConfig(
            flow_type="nsf",
            latent_dim=1,
            encoder_hidden_features=(8,),
            decoder_hidden_features=(8,),
            conditioner_hidden_features=(4,),
            activation="relu",
            regularize=True,
            warmup=1,
            lr=1e-3,
            epochs=2,
            min_epochs=1,
            early_stop_patience=1,
            batch_size=4,
            prediction_batch_size=4,
            interventional_mc_samples=3,
        )
    )

    spec = adapter._observed_spec(train[1], train[2], train[3])
    adjacency = adapter._build_adjacency(spec)

    expected_adjacency = torch.tensor(
        [
            # U, Z, W, A, Y, X
            [1, 0, 0, 0, 0, 1], #U
            [1, 1, 0, 0, 0, 1], #Z
            [1, 0, 1, 0, 0, 1], #W
            [1, 1, 0, 1, 0, 1], #A
            [1, 0, 1, 1, 1, 1], #Y
            [0, 0, 0, 0, 0, 1], #X
        ],
        dtype=torch.bool,
    )

    print("\nDeCaFlow adapter full adjacency:\n", adjacency.to(dtype=torch.int32))
    assert torch.equal(adjacency, expected_adjacency)

    adapter.fit(
        train[0],
        train[1],
        train[2],
        train[3],
        train[4],
        val_treatment=val[0],
        val_treatment_proxy=val[1],
        val_outcome_proxy=val[2],
        val_covariates=val[3],
        val_outcome=val[4],
        verbose=0,
        name="DeCaFlowAdapterTest",
        seed=7,
    )

    assert adapter.full_adjacency_ is not None
    assert adapter.encoder_adjacency_ is not None
    assert adapter.decoder_adjacency_ is not None
    assert adapter.prior_adjacency_ is not None
    assert torch.equal(adapter.full_adjacency_, expected_adjacency)
    print("Encoder adjacency:\n", adapter.encoder_adjacency_.to(dtype=torch.int32))
    print("Decoder adjacency:\n", adapter.decoder_adjacency_.to(dtype=torch.int32))
    print("Prior adjacency:\n", adapter.prior_adjacency_.to(dtype=torch.int32))
    assert adapter.encoder_adjacency_.shape == (1, 6)
    assert adapter.decoder_adjacency_.shape == (4, 6)
    assert adapter.prior_adjacency_ is not None
    assert adapter.prior_adjacency_.shape == (1, 2)

    factual_pred = adapter.mean_predict(val[0], val[3])
    assert factual_pred.shape == (val[0].shape[0], 1)
    assert np.isfinite(factual_pred).all()

    treatment_grid = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    surface = adapter.mean_predict_grid(treatment_grid, val[3])
    assert surface.shape == (treatment_grid.shape[0], val[3].shape[0])
    assert np.isfinite(surface).all()

    surface_via_alias = adapter.predict_surface(treatment_grid, val[3])
    assert surface_via_alias.shape == surface.shape
    assert np.isfinite(surface_via_alias).all()

    obs_metrics = adapter.observational_metrics(val[0], val[1], val[2], val[3], val[4])
    val_metrics = adapter.validation_observational_metrics()
    for metrics in (obs_metrics, val_metrics):
        assert set(metrics) == {"observational_loss", "observational_log_prob", "observational_kl"}
        assert np.isfinite(metrics["observational_loss"])
        assert np.isfinite(metrics["observational_log_prob"])

    assert len(adapter.training_history) == 1
    history = adapter.training_history[0]
    assert "train_losses" in history
    assert "val_losses" in history
    assert len(history["train_losses"]) == len(history["val_losses"])
    assert int(history["best_epoch"]) <= int(history["stopped_epoch"])


def test_conditional_decaflow_adapter_uses_external_adjacency_override():
    train = _toy_proximal_sample(5, seed=3)
    custom_adjacency = (
        (1, 0, 0, 0, 0, 1),
        (1, 1, 0, 0, 0, 0),
        (1, 0, 1, 0, 0, 1),
        (1, 1, 0, 1, 0, 0),
        (1, 0, 1, 1, 1, 1),
        (0, 0, 0, 0, 0, 1),
    )

    adapter = ConditionalDeCaFlow(
        DeCaFlowConfig(
            flow_type="nsf",
            prior_type="maf",
            latent_dim=1,
            adjacency_override=custom_adjacency,
            encoder_hidden_features=(8,),
            decoder_hidden_features=(8,),
            conditioner_hidden_features=(4,),
            epochs=1,
            min_epochs=1,
            early_stop_patience=1,
            batch_size=4,
            prediction_batch_size=4,
            interventional_mc_samples=2,
        )
    )

    spec = adapter._observed_spec(train[1], train[2], train[3])
    adjacency = adapter._build_adjacency(spec)
    expected = torch.tensor(custom_adjacency, dtype=torch.bool)

    print("\nExternal override adjacency:\n", adjacency.to(dtype=torch.int32))
    assert torch.equal(adjacency, expected)


def test_conditional_decaflow_adapter_skips_prior_when_override_disconnects_covariates():
    train = _toy_proximal_sample(5, seed=4)
    custom_adjacency = (
        (1, 0, 0, 0, 0, 0),
        (1, 1, 0, 0, 0, 0),
        (1, 0, 1, 0, 0, 0),
        (1, 1, 0, 1, 0, 1),
        (1, 0, 1, 1, 1, 1),
        (0, 0, 0, 0, 0, 1),
    )

    adapter = ConditionalDeCaFlow(
        DeCaFlowConfig(
            flow_type="nsf",
            latent_dim=1,
            adjacency_override=custom_adjacency,
            encoder_hidden_features=(8,),
            decoder_hidden_features=(8,),
            conditioner_hidden_features=(4,),
            epochs=1,
            min_epochs=1,
            early_stop_patience=1,
            batch_size=4,
            prediction_batch_size=4,
            interventional_mc_samples=2,
        )
    )

    spec = adapter._observed_spec(train[1], train[2], train[3])
    _, _, _, model = adapter._build_model(spec)

    assert adapter.prior_adjacency_ is None
    assert model.prior_conditioner is None
