from __future__ import annotations

import numpy as np

from src.utils import kfold_indices


def test_kfold_indices_hold_out_each_sample_once():
    folds = kfold_indices(17, 5, seed=123, shuffle=True)

    holdout = np.concatenate([fold[1] for fold in folds])
    assert np.array_equal(np.sort(holdout), np.arange(17))
    for train_idx, holdout_idx in folds:
        assert np.intersect1d(train_idx, holdout_idx).size == 0
        assert train_idx.size + holdout_idx.size == 17


def test_kfold_indices_single_fold_keeps_legacy_full_sample_behavior():
    folds = kfold_indices(6, 1, seed=123, shuffle=True)

    assert len(folds) == 1
    train_idx, holdout_idx = folds[0]
    assert np.array_equal(train_idx, np.arange(6))
    assert np.array_equal(holdout_idx, np.arange(6))
