from __future__ import annotations

import numpy as np

from src.config import FlowConfig
from src.models.flows import ConditionalFlowDensity


def test_conditional_flow_density_fit_pdf_and_sample():
    rng = np.random.default_rng(13)
    n = 80
    context = rng.normal(size=(n, 3))
    target = (0.5 + 0.15 * context[:, :1] + 0.05 * rng.normal(size=(n, 1))).clip(0.0, 1.0)

    config = FlowConfig(epochs=2, batch_size=32, hidden_features=(16, 16), transforms=2, bins=8)
    flow = ConditionalFlowDensity(features=1, context_dim=3, config=config, support=(0.0, 1.0))
    flow.fit(target, context)

    pdf = flow.pdf(target[:10], context[:10])
    samples = flow.sample(context[:5], n_samples=4)

    assert pdf.shape == (10,)
    assert np.all(np.isfinite(pdf))
    assert np.all(pdf > 0)
    assert samples.shape == (4, 5, 1)
    assert np.all(np.isfinite(samples))

